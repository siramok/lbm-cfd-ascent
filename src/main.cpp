#include <iomanip>
#include <iostream>
#include <vector>

#include "lbmd2q9_mpi.hpp"
#include <conduit_blueprint.hpp>

#ifdef ASCENT_ENABLED
#include "insitu_ascent.hpp"
#endif

using namespace conduit;

// Probably a better way to handle this than using global variables,
// but it works to demonstrate the idea

// Simulation
LbmD2Q9 *lbm;
uint32_t dim_x;
uint32_t dim_y;
int time_steps;
int time_steps_prev;

// Physical properties
double physical_density;
double physical_speed;
double physical_length;
double physical_viscosity;
double physical_time;
double physical_freq;
double reynolds_number;

// Convert physical properties into simulation properties
double simulation_dx;
double simulation_dt;
double simulation_speed;
double simulation_viscosity;

// Control flow
int rank;
int num_ranks;
bool early_shutdown = false;
bool recalculate = false;
bool all_stable = true;

// Run simulation
void runLbmCfdSimulation(uint32_t dim_x, uint32_t dim_y);

// Automatically-invoked callbacks
bool isStable();
bool isUnstable();
void setStability(Node &params, Node &output);
void computeVorticity(Node &params, Node &output);
void createCheckpoint(Node &params, Node &output);
void shutdown(Node &params, Node &output);

// Manually-invoked callbacks
void restorePreviousState(Node &params, Node &output);
void getSimInfo(Node &params, Node &output);
void setNumTimesteps(Node &params, Node &output);

int main(int argc, char **argv)
{
    int rc;
    rc = MPI_Init(NULL, NULL);
    rc |= MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    rc |= MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
    if (rc != 0)
    {
        std::cerr << "Error initializing MPI" << std::endl;
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
    }

#ifdef ASCENT_ENABLED
    MPI_Comm comm = MPI_COMM_WORLD;
    insitu::setup(comm);

    insitu::register_callback("isStable", isStable);
    insitu::register_callback("isUnstable", isUnstable);
    insitu::register_callback("setStability", setStability);
    insitu::register_callback("computeVorticity", computeVorticity);
    insitu::register_callback("createCheckpoint", createCheckpoint);
    insitu::register_callback("shutdown", shutdown);

    insitu::register_callback("restorePreviousState", restorePreviousState);
    insitu::register_callback("getSimInfo", getSimInfo);
    insitu::register_callback("setNumTimesteps", setNumTimesteps);
#endif

    // Controls output image size
    dim_x = 200;
    dim_y = 80;

    // Stable value
    //time_steps = 3200;

    // Unstable value
    time_steps = 320;

    runLbmCfdSimulation(dim_x, dim_y);

#ifdef ASCENT_ENABLED
    insitu::finalize();
#endif

    MPI_Finalize();
}

void runLbmCfdSimulation(uint32_t dim_x, uint32_t dim_y)
{
    // Simulate corn syrup at 25 C in a 2 m pipe, moving 0.75 m/s for 8 sec
    physical_density = 1380.0;   // kg/m^3
    physical_speed = 0.75;       // m/s
    physical_length = 2.0;       // m
    physical_viscosity = 1.3806; // Pa s
    physical_time = 8.0;         // s
    physical_freq = 0.04;        // s
    reynolds_number = (physical_density * physical_speed * physical_length) / physical_viscosity;

    // Convert physical properties into simulation properties
    simulation_dx = physical_length / (double)dim_y;
    simulation_dt = physical_time / (double)time_steps;
    simulation_speed = simulation_dt / simulation_dx * physical_speed;
    simulation_viscosity = simulation_dt / (simulation_dx * simulation_dx * reynolds_number);

    // Although the pointer is a global variable, we don't initialize it until here
    lbm = new LbmD2Q9(dim_x, dim_y, rank, num_ranks);

    // Output simulation initial conditions
    if (rank == 0)
    {
        std::cout << "\nLBM-CFD> Simulation running" << std::endl;
        std::cout << std::fixed << std::setprecision(6) << "LBM-CFD> speed: " << simulation_speed << ", viscosity: " << simulation_viscosity << ", reynolds: " << reynolds_number << std::endl;
        std::cout << "Press enter to begin..." << std::endl;

        // Wait for the user to hit enter before starting, lets me prepare the demo
        std::cin.get();
    }

    // Initialize simulation
    std::vector<Barrier *> barriers;
    barriers.push_back(new BarrierVertical(8 * dim_y / 27 + 1, 12 * dim_y / 27 - 1, dim_x / 8));
    barriers.push_back(new BarrierVertical(8 * dim_y / 27 + 1, 12 * dim_y / 27 - 1, dim_x / 8 + 1));
    barriers.push_back(new BarrierVertical(13 * dim_y / 27 + 1, 17 * dim_y / 27 - 1, dim_x / 8));
    barriers.push_back(new BarrierVertical(13 * dim_y / 27 + 1, 17 * dim_y / 27 - 1, dim_x / 8 + 1));
    lbm->initBarrier(barriers);
    lbm->initFluid(simulation_speed);

    // Sync all processes
    MPI_Barrier(MPI_COMM_WORLD);

    // Restore variables
    int t_prev = 1;
    int output_count_prev = 1;
    lbm->createCheckpoint();

    // Run simulation
    int output_count = 1;
    double next_output_time = 0.0;
    for (int t = 1; t <= time_steps; t++)
    {
        // output data at frequency equivalent to `physical_freq` time
        if ((t * simulation_dt) >= next_output_time)
        {
            if (rank == 0)
            {
                std::cout << std::fixed << std::setprecision(3) << "LBM-CFD> time: " << (double)t * simulation_dt << " / " << physical_time << " , time step: " << t << " / " << time_steps << std::endl;
            }

#ifdef ASCENT_ENABLED
            insitu::update(lbm, simulation_dt, t);
            MPI_Barrier(MPI_COMM_WORLD);

            if (isStable() && t % 100 == 0)
            {
                lbm->createCheckpoint();
                t_prev = t;
                output_count_prev = output_count;
            }
            else if (recalculate)
            {
                double simulation_speed_prev = simulation_speed;

                t = ((double)t / (double)time_steps_prev) * time_steps;
                output_count = output_count_prev;
                next_output_time = 0.0;
                simulation_dt = physical_time / (double)time_steps;
                simulation_speed = simulation_dt / simulation_dx * physical_speed;
                simulation_viscosity = simulation_dt / (simulation_dx * simulation_dx * reynolds_number);

                double speed_scale = simulation_speed / simulation_speed_prev;
                lbm->scaleFluidVelocity(speed_scale);
                recalculate = false;
            }
            else if (early_shutdown)
            {
                if (rank == 0)
                {
                    std::cout << "\nLBM-CFD> Shutdown callback has been invoked" << std::endl;
                }
                return;
            }
            else
            {
                output_count++;
                next_output_time = output_count * physical_freq;
            }
#else
            Node params;
            Node output;
            setStability(params, output);
            output_count++;
            next_output_time = output_count * physical_freq;
#endif
        }

        // Perform one iteration of the simulation
        lbm->collide(simulation_viscosity);
        lbm->stream();
        lbm->bounceBackStream();
    }
}

//-----------------------------------------------------------------------------
// -- begin automatic callbacks --
//-----------------------------------------------------------------------------
bool isStable()
{
    return all_stable;
}

bool isUnstable()
{
    return !all_stable;
}

void setStability(Node &params, Node &output)
{
    bool stable = lbm->checkStability();
    MPI_Allreduce(&stable, &all_stable, 1, MPI_UNSIGNED_CHAR, MPI_MAX, MPI_COMM_WORLD);
    if (!all_stable && rank == 0)
    {
        std::cerr << "LBM-CFD> Warning: simulation has become unstable (more time steps needed)" << std::endl;
    }
}

void computeVorticity(Node &params, Node &output)
{
    lbm->computeVorticity();
}

void createCheckpoint(Node &params, Node &output)
{
    lbm->createCheckpoint();
    if (rank == 0)
    {
        std::cout << "LBM-CFD> Simulation checkpointed" << std::endl;
    }
}

void shutdown(Node &params, Node &output)
{
    early_shutdown = true;
    lbm->createCheckpoint();
    MPI_Barrier(MPI_COMM_WORLD);
}
//-----------------------------------------------------------------------------
// -- end automatic callbacks --
//-----------------------------------------------------------------------------

//-----------------------------------------------------------------------------
// -- begin manual callbacks --
//-----------------------------------------------------------------------------
void restorePreviousState(Node &params, Node &output)
{
    lbm->restorePreviousState();
    recalculate = true;
    output["test"] = "Hello back from C++";
}

void getSimInfo(Node &params, Node &output)
{
    output["timesteps"] = time_steps;
    output["dim_x"] = dim_x;
    output["dim_y"] = dim_y;
    output["control_flow/stable"] = all_stable;
    output["control_flow/needs_recalculation"] = recalculate;
    output["physical/density"] = physical_density;
    output["physical/speed"] = physical_speed;
    output["physical/length"] = physical_length;
    output["physical/viscosity"] = physical_viscosity;
    output["physical/time"] = physical_time;
    output["physical/freq"] = physical_freq;
    output["physical/reynolds_number"] = reynolds_number;
    output["simulation/dx"] = simulation_dx;
    output["simulation/dt"] = simulation_dt;
    output["simulation/speed"] = simulation_speed;
    output["simulation/viscosity"] = simulation_viscosity;
}

void setNumTimesteps(Node &params, Node &output)
{
    // If a new timestep value was passed in, set it
    if (params.has_path("timesteps"))
    {
        time_steps_prev = time_steps;
        time_steps = params["timesteps"].as_int64();
        if (rank == 0)
        {
            std::cout << "LBM-CFD> Timesteps changed from " << time_steps_prev << " to " << time_steps << std::endl;
        }
        recalculate = true;
    }
}
//-----------------------------------------------------------------------------
// -- end manual callbacks --
//-----------------------------------------------------------------------------
