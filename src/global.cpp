//#include "sd3d.hpp"
#include <cstdlib>
#include <iostream>
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

Matrix amr_P, amr_R;
Matrix amr_RS_sp[2], amr_RS_cv[2];
Matrix amr_RF;
Matrix amr_P_fp, amr_RF_fp;
Vector amr_x_fp;

int ssp_rk_coefficients(int order, double* a){
    a[0]=0; a[1]=0; a[2]=0;
    switch(order){
        case 1:                                  return 1; //forward Euler
        case 2: a[1]=0.5;                        return 2; //SSPRK(2,2) (Heun)
        case 3: a[1]=0.75; a[2]=1.0/3.0;         return 3; //SSPRK(3,3)
        default: return 0;
    }
}

void set_runtime_dimensionality(bool ax, bool ay, bool az, int p){
    cfg.active[_x_] = ax;
    cfg.active[_y_] = ay;
    cfg.active[_z_] = az;
    cfg.ndim = int(ax) + int(ay) + int(az);
    //FV halo width, in sub-grid cells. This is the ONE place it is decided:
    //everything that sizes an FV array (dimension::fv_ncells / idL) or fills a
    //ghost layer (the exchange, the physical BCs) reads nGH_rt, so the array
    //and the exchange can no longer disagree. Before this, structs.hpp sized
    //the sub-grid with the compile-time nGH while the exchange filled nGH_rt,
    //and widening the halo wrote neighbour data over interior cells.
    //
    //The MOOD cascade re-runs detection once per revision, so the width the
    //cascade needs may exceed the single-pass stencil (athenak sizes ghosts
    //from the revision count for the same reason). SPD_FV_GHOST overrides it
    //so the width can be measured before it is wired to fallback/max_revs.
    int g = nGH;
    if(const char* e = getenv("SPD_FV_GHOST")){
        const int v = atoi(e);
        if(v >= nGH) g = v;
    }
    nGH_rt[_x_] = ax ? g : 0;
    nGH_rt[_y_] = ay ? g : 0;
    nGH_rt[_z_] = az ? g : 0;
    //SD halo width, in elements. An FV ghost cell is a solution point of an SD
    //ghost element (dimension::idL = NGH*n_sp - g), so the NGH*n_sp ghost points
    //per side only cover the g FV ghosts while g <= NGH*n_sp. Past that idL goes
    //negative and the outermost FV ghosts have no SD source -- which is not
    //fatal: nothing copies SD into an FV ghost (fv_update_solution writes active
    //cells only, the exchange fills the rest), and structs.hpp extends the ghost
    //*coordinates* by whole elements. p = 0 (job/scheme=muscl) already runs that
    //way at the default width, so the SD halo stays at NGH and that lane is
    //untouched. Reported below rather than silently accepted.
    NGH_rt[_x_] = ax ? NGH : 0;
    NGH_rt[_y_] = ay ? NGH : 0;
    NGH_rt[_z_] = az ? NGH : 0;
    if(Master && g > NGH*(p+1))
        std::cout<<"NOTE: FV halo "<<g<<" exceeds the SD ghost supply "<<NGH*(p+1)
                 <<" (p="<<p<<"); the outermost FV ghost coordinates are extrapolated"
                 <<std::endl;
}