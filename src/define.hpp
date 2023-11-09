//#define MPI

#define INDUCTION
//#define HYDRO

#define FV

#define PI 3.141592653589793
#define CFL 0.8
#define min_c2 1E-14
#define gmma 1.4

//#define SD
#define NGH 1
#define X 1
#define Y 1
#define Z 1

#define DIM (X+Y+Z)

#define NVAR (2+DIM+1)

#define _x_ 0
#define _y_ 1
#define _z_ 2

#define _vx_ X
#define _vy_ (Y+_vx_)
#define _vz_ (Z+_vy_)
#define _p_  (1+_vz_)
#define _e_  _p_

enum {_periodic_, _gradfree_};

#define _BCx_ _periodic_
#define _BCy_ _periodic_
#define _BCz_ _periodic_

#if DIM>=2
#define _2D_
#endif
#if DIM==3
#define _3D_
#endif
#define LLF

#ifdef X
#define NGHx NGH
#else
#define NGHx 0
#endif
#ifdef Y
#define NGHy NGH
#else
#define NGHy 0
#endif
#ifdef Z
#define NGHz NGH
#else
#define NGHz 0
#endif

#ifdef KOKKOS_ENABLE_CUDA
#define MemSpace Kokkos::CudaUVMSpace
#define Layout Kokkos::LayoutRight
#else
#define MemSpace Kokkos::HostSpace
#define Layout Kokkos::LayoutRight
#endif

using ExecSpace = MemSpace::execution_space;

typedef Kokkos::View<double*,MemSpace>  Vector;
typedef Kokkos::View<double**,MemSpace>  Matrix;
typedef Kokkos::View<double********,Layout,MemSpace>  SD_Vector;
typedef Kokkos::View<double*****,Layout,MemSpace>  FV_Vector;

typedef Matrix::HostMirror Matrix_h;
typedef Vector::HostMirror Vector_h;
typedef SD_Vector::HostMirror SD_Vector_h;
typedef FV_Vector::HostMirror FV_Vector_h;

#define SD_min_cells( call) \
double min_value=1; \
Kokkos::parallel_reduce("min_cells",Kokkos::MDRangePolicy<Kokkos::Rank<6>>({1,1,1,0,0,0},{Nz-1,Ny-1,Nx-1,pz,py,px}), KOKKOS_LAMBDA (int k, int j, int i, int kk, int jj, int ii, double& reduce) { \
    call; \
}, Kokkos::Min<double>(min_value)); Kokkos::fence();

#define SD_for_cells( call) \
Kokkos::parallel_for("for_cells",Kokkos::MDRangePolicy<Kokkos::Rank<6>>({0,0,0,0,0,0},{Nz,Ny,Nx,pz,py,px}), KOKKOS_LAMBDA (int k, int j, int i, int kk, int jj, int ii) { \
    call; \
}); Kokkos::fence();

#define SD_for_all( call) \
SD_for_cells( \
    for(int t_id=0; t_id<nader; t_id++){ \
        for(int var=0; var<nvar; var++){ \
        call; \
        } \
    } \
);

#define SD_for_ader( call) \
SD_for_cells( \
    for(int t_id=0; t_id<nader; t_id++){ \
        call; \
    } \
);

#define SD_for_variables( call) \
SD_for_cells( \
    for(int var=0; var<nvar; var++){ \
        call; \
    } \
);
