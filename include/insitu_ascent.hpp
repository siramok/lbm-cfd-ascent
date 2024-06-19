#ifndef _INSITU_ASCENT_HPP_
#define _INSITU_ASCENT_HPP_

#include "lbmd2q9_mpi.hpp"

#include <ascent.hpp>
#include <conduit_blueprint.hpp>

using namespace conduit;

namespace insitu
{
    static ascent::Ascent mAscent;

    void setup(MPI_Comm comm_in)
    {
        MPI_Comm comm;
        MPI_Comm_dup(comm_in, &comm);
        Node ascent_opts;

        ascent_opts["mpi_comm"] = MPI_Comm_c2f(comm);
        mAscent.open(ascent_opts);
    }

    void register_callback(std::string callback_name, void (*callback_function)(Node &, Node &))
    {
        ascent::register_callback(callback_name, callback_function);
    }

    void register_callback(std::string callback_name, bool (*callback_function)(void))
    {
        ascent::register_callback(callback_name, callback_function);
    }

    void update(LbmD2Q9 *lbm, double time, int step)
    {
        int dim_x = lbm->getDimX();
        int dim_y = lbm->getDimY();
        int dim_size = dim_x * dim_y;

        Node data;
        data["state/cycle"] = step;
        data["state/time"] = time;

        data["coordsets/coords/type"] = "uniform";
        data["coordsets/coords/dims/i"] = dim_x;
        data["coordsets/coords/dims/j"] = dim_y;

        data["coordsets/coords/origin/x"] = lbm->getOffsetX();
        data["coordsets/coords/origin/y"] = lbm->getOffsetY();
        data["coordsets/coords/spacing/dx"] = 1;
        data["coordsets/coords/spacing/dy"] = 1;

        data["topologies/topo/type"] = "uniform";
        data["topologies/topo/coordset"] = "coords";

        data["fields/density/association"] = "vertex";
        data["fields/density/topology"] = "topo";
        data["fields/density/values"].set_external(lbm->getDensity(), dim_size);

        data["fields/vorticity/association"] = "vertex";
        data["fields/vorticity/topology"] = "topo";
        data["fields/vorticity/values"].set_external(lbm->getVorticity(), dim_size);

        data["fields/velocity_x/association"] = "vertex";
        data["fields/velocity_x/topology"] = "topo";
        data["fields/velocity_x/values"].set_external(lbm->getVelocityX(), dim_size);

        data["fields/velocity_y/association"] = "vertex";
        data["fields/velocity_y/topology"] = "topo";
        data["fields/velocity_y/values"].set_external(lbm->getVelocityY(), dim_size);

        mAscent.publish(data);

        Node actions;
        mAscent.execute(actions);
    }

    void finalize()
    {
        mAscent.close();
    }
}

#endif // _INSITU_ASCENT_HPP_
