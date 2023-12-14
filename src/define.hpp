//#define MPI

#define INDUCTION
//#define HYDRO

#define FV
//#define SD

//#define OHMIC_DIFFUSION

#define PI 3.141592653589793
#define CFL 0.8
#define min_c2 1E-14
#define gmma 1.4
#define LENGHT 1.0//2*PI/0.78
#define rho_min 1E-10
#define rho_max 1E10
#define p_min 1E-10
#define p_max 1E10

#define NGH 1
#define nGH 2
#define X 1
#define Y 1
#define Z 1

#define DIM (X+Y+Z)

#define NVAR (2+DIM+1)

#define _x_ 0
#define _y_ 1
#define _z_ 2

#define _d_ 0
#define _vx_ X
#define _vy_ (Y+_vx_)
#define _vz_ (Z+_vy_)
#define _p_  (1+_vz_)
#define _e_  _p_

enum {_E_,_b1_,_b2_,_v1_,_v2_,_Ed_,_b1d_,_b2d_};
enum {_periodic_, _gradfree_};
enum {_center_,_face_};

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
#define nGHx nGH
#else
#define NGHx 0
#define nGHx 0
#endif
#ifdef Y
#define NGHy NGH
#define nGHy nGH
#else
#define NGHy 0
#define nGHy 0
#endif
#ifdef Z
#define NGHz NGH
#define nGHz nGH
#else
#define NGHz 0
#define nGHz 0
#endif

#define I (ii+nGHx+(i-NGHx)*qx)
#define J (jj+nGHy+(j-NGHy)*qy)
#define K (kk+nGHz+(k-NGHz)*qz)

#define INDICES t_id,var,Nid[_z_],Nid[_y_],Nid[_x_],nid[_z_],nid[_y_],nid[_x_]
#define INDICES_L t_id,var,NidL[_z_],NidL[_y_],NidL[_x_],nidL[_z_],nidL[_y_],nidL[_x_]
#define INDICES_R t_id,var,NidR[_z_],NidR[_y_],NidR[_x_],nidR[_z_],nidR[_y_],nidR[_x_]

#define NODE nid[_z_],nid[_y_],nid[_x_]

#define FV_INDICES var,Nid[_z_],Nid[_y_],Nid[_x_]

#ifdef KOKKOS_ENABLE_CUDA
#define MemSpace Kokkos::CudaUVMSpace
#define Layout Kokkos::LayoutLeft
#else
#define MemSpace Kokkos::HostSpace
#define Layout Kokkos::LayoutRight
#endif

using ExecSpace = MemSpace::execution_space;

typedef Kokkos::View<double*,MemSpace>  Vector;
typedef Kokkos::View<double**,MemSpace>  Matrix;
typedef Kokkos::View<double********,Layout,MemSpace>  SD_Vector;
typedef Kokkos::View<double****,Layout,MemSpace> FV_Vector;

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


#define SD_for_active_cells( call) \
Kokkos::parallel_for("for_active_cells",Kokkos::MDRangePolicy<Kokkos::Rank<6>>({NGHz,NGHy,NGHx,0,0,0},{Nz-NGHz,Ny-NGHy,Nx-NGHx,pz,py,px}), KOKKOS_LAMBDA (int k, int j, int i, int kk, int jj, int ii) { \
call; \
}); Kokkos::fence();

#define SD_for_active_cells_all( call) \
SD_for_active_cells( \
    for(int t_id=0; t_id<nader; t_id++){ \
        for(int var=0; var<nvar; var++){ \
        call; \
        } \
    } \
);

#define FV_for_cells( call) \
Kokkos::parallel_for("for_cells",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{Nz,Ny,Nx}), KOKKOS_LAMBDA (int k, int j, int i) { \
    call; \
}); Kokkos::fence();

#define FV_for_cells_NGH( call) \
Kokkos::parallel_for("for_cells",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({NGHz,NGHy,NGHx},{Nz-NGHz,Ny-NGHy,Nx-NGHx}), KOKKOS_LAMBDA (int k, int j, int i) { \
    call; \
}); Kokkos::fence();

#define FV_for_cells_2NGH( call) \
Kokkos::parallel_for("for_active_cells",Kokkos::MDRangePolicy<Kokkos::Rank<3>>({2*NGHz,2*NGHy,2*NGHx},{Nz-2*NGHz,Ny-2*NGHy,Nx-2*NGHx}), KOKKOS_LAMBDA (int k, int j, int i) { \
    call; \
}); Kokkos::fence();

#define FV_for_cells_all( call) \
FV_for_cells( \
    for(int var=0; var<nvar; var++){ \
        call; \
    } \
);

#define FV_for_cells_all_NGH( call) \
FV_for_cells_NGH( \
    for(int var=0; var<nvar; var++){ \
        call; \
    } \
);

#define FV_for_cells_all_2NGH( call) \
FV_for_cells_2NGH( \
    for(int var=0; var<nvar; var++){ \
        call; \
    } \
);
