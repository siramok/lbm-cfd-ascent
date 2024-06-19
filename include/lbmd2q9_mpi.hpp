#ifndef _LBMD2Q9_MPI_HPP_
#define _LBMD2Q9_MPI_HPP_

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include <mpi.h>

// Helper class for creating barriers
class Barrier
{
public:
    enum Type
    {
        HORIZONTAL,
        VERTICAL
    };

protected:
    Type type;
    int x1;
    int x2;
    int y1;
    int y2;

public:
    Type getType() { return type; }
    int getX1() { return x1; }
    int getX2() { return x2; }
    int getY1() { return y1; }
    int getY2() { return y2; }
};

class BarrierHorizontal : public Barrier
{
public:
    BarrierHorizontal(int x_start, int x_end, int y)
    {
        type = Barrier::HORIZONTAL;
        x1 = x_start;
        x2 = x_end;
        y1 = y;
        y2 = y;
    }
    ~BarrierHorizontal() {}
};

class BarrierVertical : public Barrier
{
public:
    BarrierVertical(int y_start, int y_end, int x)
    {
        type = Barrier::VERTICAL;
        x1 = x;
        x2 = x;
        y1 = y_start;
        y2 = y_end;
    }
    ~BarrierVertical() {}
};

// Lattice-Boltzman Methods CFD simulation
class LbmD2Q9
{
private:
    enum Neighbor
    {
        NeighborN,
        NeighborE,
        NeighborS,
        NeighborW,
        NeighborNE,
        NeighborNW,
        NeighborSE,
        NeighborSW
    };
    enum Column
    {
        LeftBoundaryCol,
        LeftCol,
        RightCol,
        RightBoundaryCol
    };

    int rank;
    int num_ranks;
    uint32_t total_x;
    uint32_t total_y;
    uint32_t dim_x;
    uint32_t dim_y;
    uint32_t start_x;
    uint32_t start_y;
    uint32_t num_x;
    uint32_t num_y;
    int offset_x;
    int offset_y;
    uint32_t size;
    double *f_0;
    double *f_N;
    double *f_E;
    double *f_S;
    double *f_W;
    double *f_NE;
    double *f_NW;
    double *f_SE;
    double *f_SW;
    double *density;
    double *velocity_x;
    double *velocity_y;
    double *vorticity;
    double *speed;
    bool *barrier;
    double *f_0_prev;
    double *f_N_prev;
    double *f_E_prev;
    double *f_S_prev;
    double *f_W_prev;
    double *f_NE_prev;
    double *f_NW_prev;
    double *f_SE_prev;
    double *f_SW_prev;
    double *density_prev;
    double *velocity_x_prev;
    double *velocity_y_prev;
    double *vorticity_prev;
    double *speed_prev;
    bool *barrier_prev;
    int neighbors[8];
    MPI_Datatype columns_2d[4];

    void setEquilibrium(int x, int y, double new_velocity_x, double new_velocity_y, double new_density);
    void getClosestFactors2(int value, int *factor_1, int *factor_2);
    void exchangeBoundaries();

public:
    LbmD2Q9(uint32_t width, uint32_t height, int task_id, int num_tasks);
    ~LbmD2Q9();

    void initBarrier(std::vector<Barrier *> barriers);
    void initFluid(double speed);
    void scaleFluidVelocity(double scale);
    void collide(double viscosity);
    void stream();
    void bounceBackStream();
    bool checkStability();
    void computeSpeed();
    void computeVorticity();
    bool *getBarrier();
    double *getDensity();
    double *getVelocityX();
    double *getVelocityY();
    double *getVorticity();
    double *getSpeed();
    int getDimX();
    int getDimY();
    int getOffsetX();
    int getOffsetY();
    void createCheckpoint();
    void restorePreviousState();
};

// constructor
LbmD2Q9::LbmD2Q9(uint32_t width, uint32_t height, int task_id, int num_tasks)
{
    rank = task_id;
    num_ranks = num_tasks;

    // split up problem space
    int n_x, n_y, col, row, chunk_w, chunk_h, extra_w, extra_h;
    int neighbor_cols, neighbor_rows;
    getClosestFactors2(num_ranks, &n_x, &n_y);
    chunk_w = width / n_x;
    chunk_h = height / n_y;
    extra_w = width % n_x;
    extra_h = height % n_y;
    col = rank % n_x;
    row = rank / n_x;
    num_x = chunk_w + ((col < extra_w) ? 1 : 0);
    num_y = chunk_h + ((row < extra_h) ? 1 : 0);
    offset_x = col * chunk_w + std::min(col, extra_w);
    offset_y = row * chunk_h + std::min(row, extra_h);
    neighbor_cols = (num_ranks == 1) ? 0 : ((col == 0 || col == n_x - 1) ? 1 : 2);
    neighbor_rows = (num_ranks == 1) ? 0 : ((row == 0 || row == n_y - 1) ? 1 : 2);
    start_x = (col == 0) ? 0 : 1;
    start_y = (row == 0) ? 0 : 1;
    neighbors[NeighborN] = (row == n_y - 1) ? -1 : rank + n_x;
    neighbors[NeighborE] = (col == n_x - 1) ? -1 : rank + 1;
    neighbors[NeighborS] = (row == 0) ? -1 : rank - n_x;
    neighbors[NeighborW] = (col == 0) ? -1 : rank - 1;
    neighbors[NeighborNE] = (row == n_y - 1 || col == n_x - 1) ? -1 : rank + n_x + 1;
    neighbors[NeighborNW] = (row == n_y - 1 || col == 0) ? -1 : rank + n_x - 1;
    neighbors[NeighborSE] = (row == 0 || col == n_x - 1) ? -1 : rank - n_x + 1;
    neighbors[NeighborSW] = (row == 0 || col == 0) ? -1 : rank - n_x - 1;
    // create data types for exchanging data with neighbors
    int block_width, block_height, array[2], subsize[2], offsets[2];
    block_width = num_x + neighbor_cols;
    block_height = num_y + neighbor_rows;
    array[0] = block_height;
    array[1] = block_width;
    subsize[0] = num_y;
    subsize[1] = 1;
    offsets[0] = start_y;
    offsets[1] = 0;
    MPI_Type_create_subarray(2, array, subsize, offsets, MPI_ORDER_C, MPI_DOUBLE, &columns_2d[LeftBoundaryCol]);
    MPI_Type_commit(&columns_2d[LeftBoundaryCol]); // left boundary column
    offsets[1] = start_x;
    MPI_Type_create_subarray(2, array, subsize, offsets, MPI_ORDER_C, MPI_DOUBLE, &columns_2d[LeftCol]);
    MPI_Type_commit(&columns_2d[LeftCol]); // left column
    offsets[1] = start_x + num_x - 1;
    MPI_Type_create_subarray(2, array, subsize, offsets, MPI_ORDER_C, MPI_DOUBLE, &columns_2d[RightCol]);
    MPI_Type_commit(&columns_2d[RightCol]); // right column
    offsets[1] = block_width - 1;
    MPI_Type_create_subarray(2, array, subsize, offsets, MPI_ORDER_C, MPI_DOUBLE, &columns_2d[RightBoundaryCol]);
    MPI_Type_commit(&columns_2d[RightBoundaryCol]); // right boundary column

    // set up sub grid for simulation
    total_x = width;
    total_y = height;
    dim_x = block_width;
    dim_y = block_height;

    size = dim_x * dim_y;

    // allocate all double arrays at once
    double *dbl_arrays = new double[14 * size];

    // set array pointers
    f_0 = dbl_arrays;
    f_N = dbl_arrays + (size);
    f_E = dbl_arrays + (2 * size);
    f_S = dbl_arrays + (3 * size);
    f_W = dbl_arrays + (4 * size);
    f_NE = dbl_arrays + (5 * size);
    f_NW = dbl_arrays + (6 * size);
    f_SE = dbl_arrays + (7 * size);
    f_SW = dbl_arrays + (8 * size);
    density = dbl_arrays + (9 * size);
    velocity_x = dbl_arrays + (10 * size);
    velocity_y = dbl_arrays + (11 * size);
    vorticity = dbl_arrays + (12 * size);
    speed = dbl_arrays + (13 * size);

    // allocate boolean array
    barrier = new bool[size];

    // previous of everything
    double *dbl_arrays_prev = new double[14 * size];

    // set array pointers
    f_0_prev = dbl_arrays_prev;
    f_N_prev = dbl_arrays_prev + (size);
    f_E_prev = dbl_arrays_prev + (2 * size);
    f_S_prev = dbl_arrays_prev + (3 * size);
    f_W_prev = dbl_arrays_prev + (4 * size);
    f_NE_prev = dbl_arrays_prev + (5 * size);
    f_NW_prev = dbl_arrays_prev + (6 * size);
    f_SE_prev = dbl_arrays_prev + (7 * size);
    f_SW_prev = dbl_arrays_prev + (8 * size);
    density_prev = dbl_arrays_prev + (9 * size);
    velocity_x_prev = dbl_arrays_prev + (10 * size);
    velocity_y_prev = dbl_arrays_prev + (11 * size);
    vorticity_prev = dbl_arrays_prev + (12 * size);
    speed_prev = dbl_arrays_prev + (13 * size);

    // allocate boolean array
    barrier_prev = new bool[size];
}

// destructor
LbmD2Q9::~LbmD2Q9()
{
    delete[] f_0;
    delete[] barrier;
    delete[] f_0_prev;
    delete[] barrier_prev;
}

// initialize barrier based on selected type
void LbmD2Q9::initBarrier(std::vector<Barrier *> barriers)
{
    // clear barrier to all `false`
    memset(barrier, 0, dim_x * dim_y);

    // set barrier to `true` where horizontal or vertical barriers exist
    int sx = (offset_x == 0) ? 0 : offset_x - 1;
    int sy = (offset_y == 0) ? 0 : offset_y - 1;
    int i, j;
    for (i = 0; i < barriers.size(); i++)
    {
        if (barriers[i]->getType() == Barrier::Type::HORIZONTAL)
        {
            int y = barriers[i]->getY1() - sy;
            if (y >= 0 && y < dim_y)
            {
                for (j = barriers[i]->getX1(); j <= barriers[i]->getX2(); j++)
                {
                    int x = j - sx;
                    if (x >= 0 && x < dim_x)
                    {
                        barrier[y * dim_x + x] = true;
                    }
                }
            }
        }
        else
        { // Barrier::VERTICAL
            int x = barriers[i]->getX1() - sx;
            if (x >= 0 && x < dim_x)
            {
                for (j = barriers[i]->getY1(); j <= barriers[i]->getY2(); j++)
                {
                    int y = j - sy;
                    if (y >= 0 && y < dim_y)
                    {
                        barrier[y * dim_x + x] = true;
                    }
                }
            }
        }
    }
}

// initialize fluid
void LbmD2Q9::initFluid(double speed)
{
    int i, j, row;
    for (j = 0; j < dim_y; j++)
    {
        row = j * dim_x;
        for (i = 0; i < dim_x; i++)
        {
            setEquilibrium(i, j, speed, 0.0, 1.0);
            vorticity[row + i] = 0.0;
        }
    }
}

void LbmD2Q9::scaleFluidVelocity(double scale)
{
    for (int i = 0; i < dim_x; i++)
    {
        for (int j = 0; j < dim_y; j++)
        {
            int idx = j * dim_x + i;
            velocity_x[idx] *= scale;
            velocity_y[idx] *= scale;

            double one_ninth = 1.0 / 9.0;
            double four_ninths = 4.0 / 9.0;
            double one_thirtysixth = 1.0 / 36.0;

            double new_density = density[idx];
            double new_velocity_x = velocity_x[idx];
            double new_velocity_y = velocity_y[idx];

            double velocity_3x = 3.0 * new_velocity_x;
            double velocity_3y = 3.0 * new_velocity_y;
            double velocity_x2 = new_velocity_x * new_velocity_x;
            double velocity_y2 = new_velocity_y * new_velocity_y;
            double velocity_2xy = 2.0 * new_velocity_x * new_velocity_y;
            double vecocity_2 = velocity_x2 + velocity_y2;
            double vecocity_2_15 = 1.5 * vecocity_2;
            f_0[idx] = four_ninths * new_density * (1.0 - vecocity_2_15);
            f_E[idx] = one_ninth * new_density * (1.0 + velocity_3x + 4.5 * velocity_x2 - vecocity_2_15);
            f_W[idx] = one_ninth * new_density * (1.0 - velocity_3x + 4.5 * velocity_x2 - vecocity_2_15);
            f_N[idx] = one_ninth * new_density * (1.0 + velocity_3y + 4.5 * velocity_y2 - vecocity_2_15);
            f_S[idx] = one_ninth * new_density * (1.0 - velocity_3y + 4.5 * velocity_y2 - vecocity_2_15);
            f_NE[idx] = one_thirtysixth * new_density * (1.0 + velocity_3x + velocity_3y + 4.5 * (vecocity_2 + velocity_2xy) - vecocity_2_15);
            f_SE[idx] = one_thirtysixth * new_density * (1.0 + velocity_3x - velocity_3y + 4.5 * (vecocity_2 - velocity_2xy) - vecocity_2_15);
            f_NW[idx] = one_thirtysixth * new_density * (1.0 - velocity_3x + velocity_3y + 4.5 * (vecocity_2 - velocity_2xy) - vecocity_2_15);
            f_SW[idx] = one_thirtysixth * new_density * (1.0 - velocity_3x - velocity_3y + 4.5 * (vecocity_2 + velocity_2xy) - vecocity_2_15);
        }
    }
}

// particle collision
void LbmD2Q9::collide(double viscosity)
{
    int i, j, row, idx;
    double omega = 1.0 / (3.0 * viscosity + 0.5); // reciprocal of relaxation time
    for (j = 1; j < dim_y - 1; j++)
    {
        row = j * dim_x;
        for (i = 1; i < dim_x - 1; i++)
        {
            idx = row + i;
            density[idx] = f_0[idx] + f_N[idx] + f_S[idx] + f_E[idx] + f_W[idx] + f_NW[idx] + f_NE[idx] + f_SW[idx] + f_SE[idx];
            velocity_x[idx] = (f_E[idx] + f_NE[idx] + f_SE[idx] - f_W[idx] - f_NW[idx] - f_SW[idx]) / density[idx];
            velocity_y[idx] = (f_N[idx] + f_NE[idx] + f_NW[idx] - f_S[idx] - f_SE[idx] - f_SW[idx]) / density[idx];
            double one_ninth_density = (1.0 / 9.0) * density[idx];
            double four_ninths_density = (4.0 / 9.0) * density[idx];
            double one_thirtysixth_density = (1.0 / 36.0) * density[idx];
            double velocity_3x = 3.0 * velocity_x[idx];
            double velocity_3y = 3.0 * velocity_y[idx];
            double velocity_x2 = velocity_x[idx] * velocity_x[idx];
            double velocity_y2 = velocity_y[idx] * velocity_y[idx];
            double velocity_2xy = 2.0 * velocity_x[idx] * velocity_y[idx];
            double vecocity_2 = velocity_x2 + velocity_y2;
            double vecocity_2_15 = 1.5 * vecocity_2;
            f_0[idx] += omega * (four_ninths_density * (1 - vecocity_2_15) - f_0[idx]);
            f_E[idx] += omega * (one_ninth_density * (1 + velocity_3x + 4.5 * velocity_x2 - vecocity_2_15) - f_E[idx]);
            f_W[idx] += omega * (one_ninth_density * (1 - velocity_3x + 4.5 * velocity_x2 - vecocity_2_15) - f_W[idx]);
            f_N[idx] += omega * (one_ninth_density * (1 + velocity_3y + 4.5 * velocity_y2 - vecocity_2_15) - f_N[idx]);
            f_S[idx] += omega * (one_ninth_density * (1 - velocity_3y + 4.5 * velocity_y2 - vecocity_2_15) - f_S[idx]);
            f_NE[idx] += omega * (one_thirtysixth_density * (1 + velocity_3x + velocity_3y + 4.5 * (vecocity_2 + velocity_2xy) - vecocity_2_15) - f_NE[idx]);
            f_SE[idx] += omega * (one_thirtysixth_density * (1 + velocity_3x - velocity_3y + 4.5 * (vecocity_2 - velocity_2xy) - vecocity_2_15) - f_SE[idx]);
            f_NW[idx] += omega * (one_thirtysixth_density * (1 - velocity_3x + velocity_3y + 4.5 * (vecocity_2 - velocity_2xy) - vecocity_2_15) - f_NW[idx]);
            f_SW[idx] += omega * (one_thirtysixth_density * (1 - velocity_3x - velocity_3y + 4.5 * (vecocity_2 + velocity_2xy) - vecocity_2_15) - f_SW[idx]);
        }
    }

    exchangeBoundaries();
}

// particle streaming
void LbmD2Q9::stream()
{
    int i, j, row, rowp, rown, idx;
    for (j = dim_y - 2; j > 0; j--) // first start in NW corner...
    {
        row = j * dim_x;
        rowp = (j - 1) * dim_x;
        for (i = 1; i < dim_x - 1; i++)
        {
            f_N[row + i] = f_N[rowp + i];
            f_NW[row + i] = f_NW[rowp + i + 1];
        }
    }
    for (j = dim_y - 2; j > 0; j--) // then start in NE corner...
    {
        row = j * dim_x;
        rowp = (j - 1) * dim_x;
        for (i = dim_x - 2; i > 0; i--)
        {
            f_E[row + i] = f_E[row + i - 1];
            f_NE[row + i] = f_NE[rowp + i - 1];
        }
    }
    for (j = 1; j < dim_y - 1; j++) // then start in SE corner...
    {
        row = j * dim_x;
        rown = (j + 1) * dim_x;
        for (i = dim_x - 2; i > 0; i--)
        {
            f_S[row + i] = f_S[rown + i];
            f_SE[row + i] = f_SE[rown + i - 1];
        }
    }
    for (j = 1; j < dim_y - 1; j++) // then start in the SW corner...
    {
        row = j * dim_x;
        rown = (j + 1) * dim_x;
        for (i = 1; i < dim_x - 1; i++)
        {
            f_W[row + i] = f_W[row + i + 1];
            f_SW[row + i] = f_SW[rown + i + 1];
        }
    }

    exchangeBoundaries();
}

// particle streaming bouncing back off of barriers
void LbmD2Q9::bounceBackStream()
{
    int i, j, row, rowp, rown, idx;
    for (j = 1; j < dim_y - 1; j++) // handle bounce-back from barriers
    {
        row = j * dim_x;
        rowp = (j - 1) * dim_x;
        rown = (j + 1) * dim_x;
        for (i = 1; i < dim_x - 1; i++)
        {
            idx = row + i;
            if (barrier[row + i - 1])
            {
                f_E[idx] = f_W[row + i - 1];
            }
            if (barrier[row + i + 1])
            {
                f_W[idx] = f_E[row + i + 1];
            }
            if (barrier[rowp + i])
            {
                f_N[idx] = f_S[rowp + i];
            }
            if (barrier[rown + i])
            {
                f_S[idx] = f_N[rown + i];
            }
            if (barrier[rowp + i - 1])
            {
                f_NE[idx] = f_SW[rowp + i - 1];
            }
            if (barrier[rowp + i + 1])
            {
                f_NW[idx] = f_SE[rowp + i + 1];
            }
            if (barrier[rown + i - 1])
            {
                f_SE[idx] = f_NW[rown + i - 1];
            }
            if (barrier[rown + i + 1])
            {
                f_SW[idx] = f_NE[rown + i + 1];
            }
        }
    }
}

// check if simulation has become unstable (if so, more time steps are required)
bool LbmD2Q9::checkStability()
{
    bool stable = true;
    for (int i = 0; i < dim_x; i++)
    {
        for (int j = 0; j < dim_y; j++)
        {
            if (density[i + j * dim_x] <= 0)
            {
                stable = false;
            }
        }
    }
    return stable;
}

// compute speed (magnitude of velocity vector)
void LbmD2Q9::computeSpeed()
{
    int i, j, row;
    for (j = 1; j < dim_y - 1; j++)
    {
        row = j * dim_x;
        for (i = 1; i < dim_x - 1; i++)
        {
            speed[row + i] = sqrt(velocity_x[row + i] * velocity_x[row + i] + velocity_y[row + i] * velocity_y[row + i]);
        }
    }
}

// compute vorticity (rotational velocity)
void LbmD2Q9::computeVorticity()
{
    int i, j, row, rowp, rown;
    for (j = 1; j < dim_y - 1; j++)
    {
        row = j * dim_x;
        rowp = (j - 1) * dim_x;
        rown = (j + 1) * dim_x;
        for (i = 1; i < dim_x - 1; i++)
        {
            vorticity[row + i] = velocity_y[row + i + 1] - velocity_y[row + i - 1] - velocity_x[rown + i] + velocity_x[rowp + i];
        }
    }
}

// get barrier array
bool *LbmD2Q9::getBarrier()
{
    return barrier;
}

// get density array
double *LbmD2Q9::getDensity()
{
    return density;
}

// get density array
double *LbmD2Q9::getVelocityX()
{
    return velocity_x;
}

// get density array
double *LbmD2Q9::getVelocityY()
{
    return velocity_y;
}

// get vorticity array
double *LbmD2Q9::getVorticity()
{
    return vorticity;
}

// get speed array
double *LbmD2Q9::getSpeed()
{
    return speed;
}

// private - set fluid equalibrium
void LbmD2Q9::setEquilibrium(int x, int y, double new_velocity_x, double new_velocity_y, double new_density)
{
    int idx = y * dim_x + x;

    double one_ninth = 1.0 / 9.0;
    double four_ninths = 4.0 / 9.0;
    double one_thirtysixth = 1.0 / 36.0;

    double velocity_3x = 3.0 * new_velocity_x;
    double velocity_3y = 3.0 * new_velocity_y;
    double velocity_x2 = new_velocity_x * new_velocity_x;
    double velocity_y2 = new_velocity_y * new_velocity_y;
    double velocity_2xy = 2.0 * new_velocity_x * new_velocity_y;
    double vecocity_2 = velocity_x2 + velocity_y2;
    double vecocity_2_15 = 1.5 * vecocity_2;
    f_0[idx] = four_ninths * new_density * (1.0 - vecocity_2_15);
    f_E[idx] = one_ninth * new_density * (1.0 + velocity_3x + 4.5 * velocity_x2 - vecocity_2_15);
    f_W[idx] = one_ninth * new_density * (1.0 - velocity_3x + 4.5 * velocity_x2 - vecocity_2_15);
    f_N[idx] = one_ninth * new_density * (1.0 + velocity_3y + 4.5 * velocity_y2 - vecocity_2_15);
    f_S[idx] = one_ninth * new_density * (1.0 - velocity_3y + 4.5 * velocity_y2 - vecocity_2_15);
    f_NE[idx] = one_thirtysixth * new_density * (1.0 + velocity_3x + velocity_3y + 4.5 * (vecocity_2 + velocity_2xy) - vecocity_2_15);
    f_SE[idx] = one_thirtysixth * new_density * (1.0 + velocity_3x - velocity_3y + 4.5 * (vecocity_2 - velocity_2xy) - vecocity_2_15);
    f_NW[idx] = one_thirtysixth * new_density * (1.0 - velocity_3x + velocity_3y + 4.5 * (vecocity_2 - velocity_2xy) - vecocity_2_15);
    f_SW[idx] = one_thirtysixth * new_density * (1.0 - velocity_3x - velocity_3y + 4.5 * (vecocity_2 + velocity_2xy) - vecocity_2_15);
    density[idx] = new_density;
    velocity_x[idx] = new_velocity_x;
    velocity_y[idx] = new_velocity_y;
}

// private - get 2 factors of a given number that are closest to each other
void LbmD2Q9::getClosestFactors2(int value, int *factor_1, int *factor_2)
{
    int test_num = (int)sqrt(value);
    while (value % test_num != 0)
    {
        test_num--;
    }
    *factor_2 = test_num;
    *factor_1 = value / test_num;
}

// private - exchange boundary information between MPI ranks
void LbmD2Q9::exchangeBoundaries()
{
    MPI_Status status;
    int nx = dim_x;
    int ny = dim_y;
    int sx = start_x;
    int sy = start_y;
    int cx = num_x;
    int cy = num_y;

    if (neighbors[NeighborN] >= 0)
    {
        MPI_Sendrecv(&(f_0[(ny - 2) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(f_0[(ny - 1) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_N[(ny - 2) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(f_N[(ny - 1) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_E[(ny - 2) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(f_E[(ny - 1) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_S[(ny - 2) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(f_S[(ny - 1) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_W[(ny - 2) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(f_W[(ny - 1) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_NE[(ny - 2) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(f_NE[(ny - 1) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_NW[(ny - 2) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(f_NW[(ny - 1) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_SE[(ny - 2) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(f_SE[(ny - 1) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_SW[(ny - 2) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(f_SW[(ny - 1) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(density[(ny - 2) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(density[(ny - 1) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(velocity_x[(ny - 2) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(velocity_x[(ny - 1) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(velocity_y[(ny - 2) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborS, &(velocity_y[(ny - 1) * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborN], NeighborN, MPI_COMM_WORLD, &status);
    }
    if (neighbors[NeighborE] >= 0)
    {
        MPI_Sendrecv(f_0, 1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, f_0, 1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_N, 1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, f_N, 1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_E, 1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, f_E, 1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_S, 1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, f_S, 1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_W, 1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, f_W, 1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_NE, 1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, f_NE, 1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_NW, 1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, f_NW, 1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_SE, 1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, f_SE, 1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_SW, 1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, f_SW, 1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(density, 1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, density, 1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(velocity_x, 1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, velocity_x, 1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(velocity_y, 1, columns_2d[RightCol], neighbors[NeighborE], NeighborW, velocity_y, 1, columns_2d[RightBoundaryCol], neighbors[NeighborE], NeighborE, MPI_COMM_WORLD, &status);
    }
    if (neighbors[NeighborS] >= 0)
    {
        MPI_Sendrecv(&(f_0[sy * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(f_0[sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_N[sy * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(f_N[sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_E[sy * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(f_E[sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_S[sy * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(f_S[sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_W[sy * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(f_W[sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_NE[sy * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(f_NE[sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_NW[sy * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(f_NW[sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_SE[sy * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(f_SE[sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_SW[sy * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(f_SW[sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(density[sy * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(density[sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(velocity_x[sy * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(velocity_x[sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(velocity_y[sy * nx + sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborN, &(velocity_y[sx]), cx, MPI_DOUBLE, neighbors[NeighborS], NeighborS, MPI_COMM_WORLD, &status);
    }
    if (neighbors[NeighborW] >= 0)
    {
        MPI_Sendrecv(f_0, 1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, f_0, 1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_N, 1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, f_N, 1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_E, 1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, f_E, 1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_S, 1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, f_S, 1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_W, 1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, f_W, 1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_NE, 1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, f_NE, 1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_NW, 1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, f_NW, 1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_SE, 1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, f_SE, 1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(f_SW, 1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, f_SW, 1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(density, 1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, density, 1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(velocity_x, 1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, velocity_x, 1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(velocity_y, 1, columns_2d[LeftCol], neighbors[NeighborW], NeighborE, velocity_y, 1, columns_2d[LeftBoundaryCol], neighbors[NeighborW], NeighborW, MPI_COMM_WORLD, &status);
    }
    if (neighbors[NeighborNE] >= 0)
    {
        MPI_Sendrecv(&(f_0[(ny - 2) * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(f_0[(ny - 1) * nx + nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_N[(ny - 2) * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(f_N[(ny - 1) * nx + nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_E[(ny - 2) * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(f_E[(ny - 1) * nx + nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_S[(ny - 2) * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(f_S[(ny - 1) * nx + nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_W[(ny - 2) * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(f_W[(ny - 1) * nx + nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_NE[(ny - 2) * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(f_NE[(ny - 1) * nx + nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_NW[(ny - 2) * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(f_NW[(ny - 1) * nx + nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_SE[(ny - 2) * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(f_SE[(ny - 1) * nx + nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_SW[(ny - 2) * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(f_SW[(ny - 1) * nx + nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(density[(ny - 2) * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(density[(ny - 1) * nx + nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(velocity_x[(ny - 2) * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(velocity_x[(ny - 1) * nx + nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(velocity_y[(ny - 2) * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborSW, &(velocity_y[(ny - 1) * nx + nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborNE], NeighborNE, MPI_COMM_WORLD, &status);
    }
    if (neighbors[NeighborNW] >= 0)
    {
        MPI_Sendrecv(&(f_0[(ny - 2) * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(f_0[(ny - 1) * nx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_N[(ny - 2) * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(f_N[(ny - 1) * nx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_E[(ny - 2) * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(f_E[(ny - 1) * nx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_S[(ny - 2) * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(f_S[(ny - 1) * nx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_W[(ny - 2) * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(f_W[(ny - 1) * nx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_NE[(ny - 2) * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(f_NE[(ny - 1) * nx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_NW[(ny - 2) * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(f_NW[(ny - 1) * nx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_SE[(ny - 2) * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(f_SE[(ny - 1) * nx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_SW[(ny - 2) * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(f_SW[(ny - 1) * nx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(density[(ny - 2) * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(density[(ny - 1) * nx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(velocity_x[(ny - 2) * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(velocity_x[(ny - 1) * nx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(velocity_y[(ny - 2) * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborSE, &(velocity_y[(ny - 1) * nx]), 1, MPI_DOUBLE, neighbors[NeighborNW], NeighborNW, MPI_COMM_WORLD, &status);
    }
    if (neighbors[NeighborSE] >= 0)
    {
        MPI_Sendrecv(&(f_0[sy * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(f_0[nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_N[sy * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(f_N[nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_E[sy * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(f_E[nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_S[sy * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(f_S[nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_W[sy * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(f_W[nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_NE[sy * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(f_NE[nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_NW[sy * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(f_NW[nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_SE[sy * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(f_SE[nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_SW[sy * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(f_SW[nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(density[sy * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(density[nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(velocity_x[sy * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(velocity_x[nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(velocity_y[sy * nx + nx - 2]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborNW, &(velocity_y[nx - 1]), 1, MPI_DOUBLE, neighbors[NeighborSE], NeighborSE, MPI_COMM_WORLD, &status);
    }
    if (neighbors[NeighborSW] >= 0)
    {
        MPI_Sendrecv(&(f_0[sy * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(f_0[0]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_N[sy * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(f_N[0]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_E[sy * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(f_E[0]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_S[sy * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(f_S[0]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_W[sy * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(f_W[0]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_NE[sy * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(f_NE[0]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_NW[sy * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(f_NW[0]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_SE[sy * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(f_SE[0]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(f_SW[sy * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(f_SW[0]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(density[sy * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(density[0]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(velocity_x[sy * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(velocity_x[0]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
        MPI_Sendrecv(&(velocity_y[sy * nx + sx]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborNE, &(velocity_y[0]), 1, MPI_DOUBLE, neighbors[NeighborSW], NeighborSW, MPI_COMM_WORLD, &status);
    }
}

int LbmD2Q9::getDimX()
{
    return dim_x;
}

int LbmD2Q9::getDimY()
{
    return dim_y;
}

int LbmD2Q9::getOffsetX()
{
    return offset_x;
}

int LbmD2Q9::getOffsetY()
{
    return offset_y;
}

void LbmD2Q9::createCheckpoint()
{
    std::memcpy(f_0_prev, f_0, 14 * size * sizeof(double));
    std::memcpy(barrier_prev, barrier, size);
}

void LbmD2Q9::restorePreviousState()
{
    std::memcpy(f_0, f_0_prev, 14 * size * sizeof(double));
    std::memcpy(barrier, barrier_prev, size * sizeof(bool));
}

#endif // _LBMD2Q9_MPI_HPP_
