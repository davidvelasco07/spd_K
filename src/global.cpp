//#include "sd3d.hpp"
#include <Kokkos_Core.hpp>
#include <mpi.h>
#include "define.hpp"
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