//#include "sd3d.hpp"
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include "define.hpp"
#include "global.hpp"
//MPI global variables
int cpu_rank;
int cpu_size;
MPI_Comm Comm;
int cpu_x=1;
int cpu_y=1;
int cpu_z=1;
int rank_x;
int rank_y;
int rank_z;
int x_i=0;
int y_i=0;
int z_i=0;

int Master=1;

int N_comms;

//Runtime dimensionality (see define.hpp/global.hpp)
int NGH_rt[3] = {NGH, NGH, NGH};
int nGH_rt[3] = {nGH, nGH, nGH};
RunConfig cfg;

void set_runtime_dimensionality(bool ax, bool ay, bool az){
    cfg.active[_x_] = ax;
    cfg.active[_y_] = ay;
    cfg.active[_z_] = az;
    cfg.ndim = int(ax) + int(ay) + int(az);
    NGH_rt[_x_] = ax ? NGH : 0;
    NGH_rt[_y_] = ay ? NGH : 0;
    NGH_rt[_z_] = az ? NGH : 0;
    nGH_rt[_x_] = ax ? nGH : 0;
    nGH_rt[_y_] = ay ? nGH : 0;
    nGH_rt[_z_] = az ? nGH : 0;
}