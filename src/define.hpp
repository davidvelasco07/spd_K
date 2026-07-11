//#define MPI

//The system (hydro/induction), dimensionality, FV fallback, boundary
//types, problem and physics parameters are all runtime choices made from
//the input file (see main.cpp and the RunConfig struct in global.hpp).

//#define OHMIC_DIFFUSION

//Use the reference (p+1)^DIM tensor-product transforms instead of the
//directional sweeps; only for validation/regression comparisons
//#define REF_TRANSFORMS

#define PI 3.141592653589793
#define min_c2 1E-14
#define LENGHT 1.0//2*PI/0.78
#define rho_min 1E-10
#define rho_max 1E10
#define p_min 1E-10
#define p_max 1E10

#define NGH 1
#define nGH 2

//All three directions are always compiled; dimensionality is a runtime
//choice made from the input file (inactive directions have one point,
//zero ghosts, and are skipped by runtime checks on cfg.active[]).
#define X 1
#define Y 1
#define Z 1

#define DIM (X+Y+Z)

//rho, vx, vy, vz, e + one bookkeeping slot (FV trouble flag aggregate).
//All velocity components are carried regardless of the runtime
//dimensionality, so the variable indices below are fixed.
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
enum {_integrator_ader_, _integrator_rk_};
enum {_ic_sine_wave_, _ic_sedov_, _ic_spherical_blast_};
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

//Per-direction ghost counts, set at startup from the runtime
//dimensionality (NGH/nGH for active directions, 0 for inactive ones).
//These are HOST-side globals: use them freely when computing loop bounds,
//but inside device lambdas capture them through GHOST_LOCALS below.
extern int NGH_rt[3];
extern int nGH_rt[3];
#define NGHx NGH_rt[_x_]
#define NGHy NGH_rt[_y_]
#define NGHz NGH_rt[_z_]
#define nGHx nGH_rt[_x_]
#define nGHy nGH_rt[_y_]
#define nGHz nGH_rt[_z_]

//Declare ghost-count locals for capture into device lambdas (globals are
//not accessible from device code)
#define GHOST_LOCALS int ghx=NGHx, ghy=NGHy, ghz=NGHz, sghx=nGHx, sghy=nGHy, sghz=nGHz

//Element/point -> global FV cell index maps; require GHOST_LOCALS in scope
#define I (ii+sghx+(i-ghx)*qx)
#define J (jj+sghy+(j-ghy)*qy)
#define K (kk+sghz+(k-ghz)*qz)

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

typedef Matrix::host_mirror_type Matrix_h;
typedef Vector::host_mirror_type Vector_h;
typedef SD_Vector::host_mirror_type SD_Vector_h;
typedef FV_Vector::host_mirror_type FV_Vector_h;

//Loop helpers: thin wrappers over Kokkos::parallel_for/parallel_reduce
//taking a KOKKOS_LAMBDA. Kernels launched on the same execution space
//instance execute in order, so no fence is attached here; call
//Kokkos::fence() explicitly before the host reads device data
//(outputs, MPI staging).

//Element loop over all elements and their solution/flux points.
//Lambda signature: (int k, int j, int i, int kk, int jj, int ii)
template <class Functor>
void sd_for_cells(int Nz, int Ny, int Nx, int nz, int ny, int nx,
                  const Functor& f, const char* label="sd_for_cells"){
    Kokkos::parallel_for(label,
        Kokkos::MDRangePolicy<Kokkos::Rank<6>>({0,0,0,0,0,0},{Nz,Ny,Nx,nz,ny,nx}), f);
}

//Same as sd_for_cells but skipping ghost elements
template <class Functor>
void sd_for_active_cells(int Nz, int Ny, int Nx, int nz, int ny, int nx,
                         const Functor& f, const char* label="sd_for_active_cells"){
    Kokkos::parallel_for(label,
        Kokkos::MDRangePolicy<Kokkos::Rank<6>>({NGHz,NGHy,NGHx,0,0,0},{Nz-NGHz,Ny-NGHy,Nx-NGHx,nz,ny,nx}), f);
}

//Min-reduction over interior elements; blocks until the result is ready.
//Lambda signature: (int k, int j, int i, int kk, int jj, int ii, double& reduce)
template <class Functor>
double sd_min_cells(int Nz, int Ny, int Nx, int nz, int ny, int nx,
                    const Functor& f){
    double min_value=1;
    Kokkos::parallel_reduce("sd_min_cells",
        Kokkos::MDRangePolicy<Kokkos::Rank<6>>({NGHz,NGHy,NGHx,0,0,0},{Nz-NGHz,Ny-NGHy,Nx-NGHx,nz,ny,nx}),
        f, Kokkos::Min<double>(min_value));
    return min_value;
}

//Cell loops for the FV representation.
//Lambda signature: (int k, int j, int i)
template <class Functor>
void fv_for_cells(int Nz, int Ny, int Nx,
                  const Functor& f, const char* label="fv_for_cells"){
    Kokkos::parallel_for(label,
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{Nz,Ny,Nx}), f);
}

template <class Functor>
void fv_for_cells_ngh(int Nz, int Ny, int Nx,
                      const Functor& f, const char* label="fv_for_cells_ngh"){
    Kokkos::parallel_for(label,
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({NGHz,NGHy,NGHx},{Nz-NGHz,Ny-NGHy,Nx-NGHx}), f);
}

template <class Functor>
void fv_for_cells_2ngh(int Nz, int Ny, int Nx,
                       const Functor& f, const char* label="fv_for_cells_2ngh"){
    Kokkos::parallel_for(label,
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({2*NGHz,2*NGHy,2*NGHx},{Nz-2*NGHz,Ny-2*NGHy,Nx-2*NGHx}), f);
}

//Cell loop for face-writing kernels: each cell writes its LEFT face, so to
//cover every face of every active cell (including the domain-boundary face
//on the right) the range extends one cell past the active region in each
//active direction. For periodic conservation the two boundary faces of a
//direction must then receive bitwise-identical values, which requires the
//input arrays (including trouble flags) to hold proper periodic ghosts.
template <class Functor>
void fv_for_faces(int Nz, int Ny, int Nx,
                  const Functor& f, const char* label="fv_for_faces"){
    Kokkos::parallel_for(label,
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({nGHz,nGHy,nGHx},
            {Nz-nGHz+(nGHz>0),Ny-nGHy+(nGHy>0),Nx-nGHx+(nGHx>0)}), f);
}
