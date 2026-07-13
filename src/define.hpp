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
enum {_periodic_, _gradfree_, _reflective_};
enum {_integrator_ader_, _integrator_rk_};
enum {_ic_sine_wave_, _ic_sedov_, _ic_spherical_blast_, _ic_square_,
      _ic_sod_, _ic_shu_osher_, _ic_kelvin_helmholtz_, _ic_implosion_,
      _ic_rti_, _ic_user_};
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

//Bulk solution arrays live in device memory (CudaSpace): all per-step
//kernels touch only these, so no host<->device traffic occurs during the
//evolution loop. Small setup arrays (transform matrices, quadrature nodes,
//face coordinates) are filled by host code once at startup and only read by
//kernels afterwards, so they use managed memory (SetupSpace) for simplicity.
//
//LayoutRight everywhere: with the index order (t,var,k,j,i,kk,jj,ii) the
//point index ii is stride-1, so a warp of consecutive flat indices reads
//contiguous memory, and var sits at a large stride outside the coalesced
//pattern. Measured on A100 (tests/bench_layout.cpp): 1120 GB/s for the
//production access pattern vs 457 GB/s with LayoutLeft. Files written on
//GPU and CPU now share the same (C-order) on-disk layout.
#ifdef KOKKOS_ENABLE_CUDA
#define MemSpace Kokkos::CudaSpace
#define SetupSpace Kokkos::CudaUVMSpace
#else
#define MemSpace Kokkos::HostSpace
#define SetupSpace Kokkos::HostSpace
#endif
#define Layout Kokkos::LayoutRight

using ExecSpace = MemSpace::execution_space;

typedef Kokkos::View<double*,SetupSpace>  Vector;
typedef Kokkos::View<double**,SetupSpace>  Matrix;
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
//
//All bulk loops are flattened RangePolicy kernels: the 1d thread index is
//decomposed so that its fastest-varying component is the smallest-stride
//parallel index of the views (LayoutRight: the point index ii), keeping
//warp accesses contiguous on GPU and inner loops cache-friendly on host.

//Unsigned 32-bit index arithmetic: 64-bit integer division is emulated on
//GPUs and would dominate light kernels. All loops here fit comfortably in
//32 bits (checked with an assert in the helpers below).
using flat_range = Kokkos::RangePolicy<Kokkos::IndexType<unsigned>>;

//(element,point) index decomposition; extents Mz/My/Mx exclude ghosts,
//oz/oy/ox are the ghost offsets added back to the element indices
KOKKOS_INLINE_FUNCTION
void flat_index6(unsigned idx,
                 unsigned Mz, unsigned My, unsigned Mx,
                 unsigned nz, unsigned ny, unsigned nx,
                 int oz, int oy, int ox,
                 int& k, int& j, int& i, int& kk, int& jj, int& ii){
    ii = int(idx % nx);      idx /= nx;
    jj = int(idx % ny);      idx /= ny;
    kk = int(idx % nz);      idx /= nz;
    i  = int(idx % Mx) + ox; idx /= Mx;
    j  = int(idx % My) + oy;
    k  = int(idx / My) + oz;
}

KOKKOS_INLINE_FUNCTION
void flat_index3(unsigned idx,
                 unsigned Mz, unsigned My, unsigned Mx,
                 int oz, int oy, int ox,
                 int& k, int& j, int& i){
    i  = int(idx % Mx) + ox; idx /= Mx;
    j  = int(idx % My) + oy;
    k  = int(idx / My) + oz;
}

//Wrapper functors (instead of nested extended lambdas, which nvcc rejects)
template <class Functor>
struct Flat6 {
    Functor f;
    unsigned Mz, My, Mx, nz, ny, nx;
    int oz, oy, ox;
    KOKKOS_INLINE_FUNCTION
    void operator()(const unsigned idx) const {
        int k, j, i, kk, jj, ii;
        flat_index6(idx, Mz, My, Mx, nz, ny, nx, oz, oy, ox, k, j, i, kk, jj, ii);
        f(k, j, i, kk, jj, ii);
    }
    KOKKOS_INLINE_FUNCTION
    void operator()(const unsigned idx, double& r) const {
        int k, j, i, kk, jj, ii;
        flat_index6(idx, Mz, My, Mx, nz, ny, nx, oz, oy, ox, k, j, i, kk, jj, ii);
        f(k, j, i, kk, jj, ii, r);
    }
};

template <class Functor>
struct Flat3 {
    Functor f;
    unsigned Mz, My, Mx;
    int oz, oy, ox;
    KOKKOS_INLINE_FUNCTION
    void operator()(const unsigned idx) const {
        int k, j, i;
        flat_index3(idx, Mz, My, Mx, oz, oy, ox, k, j, i);
        f(k, j, i);
    }
    KOKKOS_INLINE_FUNCTION
    void operator()(const unsigned idx, double& r) const {
        int k, j, i;
        flat_index3(idx, Mz, My, Mx, oz, oy, ox, k, j, i);
        f(k, j, i, r);
    }
};

inline unsigned flat_total(int64_t total){
    assert(total < int64_t(1) << 32);
    return unsigned(total);
}

template <class Functor>
Flat6<Functor> make_flat6(const Functor& f, int Mz, int My, int Mx,
                          int nz, int ny, int nx, int oz, int oy, int ox){
    return Flat6<Functor>{f,unsigned(Mz),unsigned(My),unsigned(Mx),
                          unsigned(nz),unsigned(ny),unsigned(nx),oz,oy,ox};
}

template <class Functor>
Flat3<Functor> make_flat3(const Functor& f, int Mz, int My, int Mx,
                          int oz, int oy, int ox){
    return Flat3<Functor>{f,unsigned(Mz),unsigned(My),unsigned(Mx),oz,oy,ox};
}

//Element loop over all elements and their solution/flux points.
//Lambda signature: (int k, int j, int i, int kk, int jj, int ii)
template <class Functor>
void sd_for_cells(int Nz, int Ny, int Nx, int nz, int ny, int nx,
                  const Functor& f, const char* label="sd_for_cells"){
    int64_t total = (int64_t)Nz*Ny*Nx*nz*ny*nx;
    if(total <= 0) return;
    Kokkos::parallel_for(label, flat_range(0,flat_total(total)),
        make_flat6(f,Nz,Ny,Nx,nz,ny,nx,0,0,0));
}

//Same as sd_for_cells but skipping ghost elements
template <class Functor>
void sd_for_active_cells(int Nz, int Ny, int Nx, int nz, int ny, int nx,
                         const Functor& f, const char* label="sd_for_active_cells"){
    int Mz=Nz-2*NGHz, My=Ny-2*NGHy, Mx=Nx-2*NGHx;
    int64_t total = (int64_t)Mz*My*Mx*nz*ny*nx;
    if(total <= 0) return;
    Kokkos::parallel_for(label, flat_range(0,flat_total(total)),
        make_flat6(f,Mz,My,Mx,nz,ny,nx,NGHz,NGHy,NGHx));
}

//Min-reduction over interior elements; blocks until the result is ready.
//Lambda signature: (int k, int j, int i, int kk, int jj, int ii, double& reduce)
template <class Functor>
double sd_min_cells(int Nz, int Ny, int Nx, int nz, int ny, int nx,
                    const Functor& f){
    int Mz=Nz-2*NGHz, My=Ny-2*NGHy, Mx=Nx-2*NGHx;
    int64_t total = (int64_t)Mz*My*Mx*nz*ny*nx;
    double min_value=1;
    if(total <= 0) return min_value;
    Kokkos::parallel_reduce("sd_min_cells", flat_range(0,flat_total(total)),
        make_flat6(f,Mz,My,Mx,nz,ny,nx,NGHz,NGHy,NGHx),
        Kokkos::Min<double>(min_value));
    return min_value;
}

//Sum-reduction over interior elements and their points.
//Lambda signature: (int k, int j, int i, int kk, int jj, int ii, double& reduce)
template <class Functor>
double sd_sum_active_cells(int Nz, int Ny, int Nx, int nz, int ny, int nx,
                           const Functor& f){
    int Mz=Nz-2*NGHz, My=Ny-2*NGHy, Mx=Nx-2*NGHx;
    int64_t total = (int64_t)Mz*My*Mx*nz*ny*nx;
    double sum=0;
    if(total <= 0) return sum;
    Kokkos::parallel_reduce("sd_sum_active_cells", flat_range(0,flat_total(total)),
        make_flat6(f,Mz,My,Mx,nz,ny,nx,NGHz,NGHy,NGHx), sum);
    return sum;
}

//Cell loops for the FV representation.
//Lambda signature: (int k, int j, int i)
template <class Functor>
void fv_for_cells(int Nz, int Ny, int Nx,
                  const Functor& f, const char* label="fv_for_cells"){
    int64_t total = (int64_t)Nz*Ny*Nx;
    if(total <= 0) return;
    Kokkos::parallel_for(label, flat_range(0,flat_total(total)),
        make_flat3(f,Nz,Ny,Nx,0,0,0));
}

template <class Functor>
void fv_for_cells_ngh(int Nz, int Ny, int Nx,
                      const Functor& f, const char* label="fv_for_cells_ngh"){
    int Mz=Nz-2*NGHz, My=Ny-2*NGHy, Mx=Nx-2*NGHx;
    int64_t total = (int64_t)Mz*My*Mx;
    if(total <= 0) return;
    Kokkos::parallel_for(label, flat_range(0,flat_total(total)),
        make_flat3(f,Mz,My,Mx,NGHz,NGHy,NGHx));
}

template <class Functor>
void fv_for_cells_2ngh(int Nz, int Ny, int Nx,
                       const Functor& f, const char* label="fv_for_cells_2ngh"){
    int Mz=Nz-4*NGHz, My=Ny-4*NGHy, Mx=Nx-4*NGHx;
    int64_t total = (int64_t)Mz*My*Mx;
    if(total <= 0) return;
    Kokkos::parallel_for(label, flat_range(0,flat_total(total)),
        make_flat3(f,Mz,My,Mx,2*NGHz,2*NGHy,2*NGHx));
}

//Sum-reduction over FV cells inside the nGH ghost frame.
//Lambda signature: (int k, int j, int i, double& reduce)
template <class Functor>
double fv_sum_cells_ngh2(int Nz, int Ny, int Nx, const Functor& f){
    int Mz=Nz-2*nGHz, My=Ny-2*nGHy, Mx=Nx-2*nGHx;
    int64_t total = (int64_t)Mz*My*Mx;
    double sum=0;
    if(total <= 0) return sum;
    Kokkos::parallel_reduce("fv_sum_cells_ngh2", flat_range(0,flat_total(total)),
        make_flat3(f,Mz,My,Mx,nGHz,nGHy,nGHx), sum);
    return sum;
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
    int Mz=Nz-2*nGHz+(nGHz>0), My=Ny-2*nGHy+(nGHy>0), Mx=Nx-2*nGHx+(nGHx>0);
    int64_t total = (int64_t)Mz*My*Mx;
    if(total <= 0) return;
    Kokkos::parallel_for(label, flat_range(0,flat_total(total)),
        make_flat3(f,Mz,My,Mx,nGHz,nGHy,nGHx));
}
