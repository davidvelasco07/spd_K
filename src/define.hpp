#ifndef DEFINE_HPP_
#define DEFINE_HPP_

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

//Default ghost widths: NGH SD elements and nGH FV sub-grid cells per side.
//These are the DEFAULTS ONLY -- set_runtime_dimensionality turns them into the
//per-direction runtime widths NGH_rt/nGH_rt below, and everything that sizes or
//indexes a halo must read those. Using NGH/nGH directly reintroduces the second
//source of truth that let the exchange and the arrays disagree.
#define NGH 1
#define nGH 2

//The two cells adjacent to a face. A reconstruction stencil size, unrelated to
//any ghost width; see compute_fluxes/level_flux in hydro.cpp.
#define FACE_CELLS 2

//All three directions are always compiled; dimensionality is a runtime
//choice made from the input file (inactive directions have one point,
//zero ghosts, and are skipped by runtime checks on cfg.active[]).
#define X 1
#define Y 1
#define Z 1

#define DIM (X+Y+Z)

//rho, vx, vy, vz, e. All velocity components are carried regardless of the
//runtime dimensionality, so the variable indices below are fixed.
//
//The FV trouble-flag aggregate used to ride along here as a sixth slot, which
//put a non-physics component in every SD, ADER, flux and boundary array: a
//sixth of all that traffic carried no physics, and because the SD path never
//writes it, on GPU it held NaN. It now lives in its own per-cell array
//(`cascade`), which is also the shape the MHD module's MOOD cascade index
//already had.
#define NVAR (2+DIM)

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
//_gradfree_ is zeroth-order OUTFLOW: the ghost copies the adjacent interior
//value, so no gradient is imposed and waves leave freely.
//_inflow_ is a PRESCRIBED state on the LOW side of the direction, and outflow
//everywhere else -- both on the high side and, on the low side, at any point the
//problem does not inject through. That second part is not a detail: a jet nozzle
//occupies a fraction of its boundary, and clamping the REST of that face to a
//fixed state walls in the cocoon backflow instead of letting it drain. The
//prescribed state is precomputed once into Boundaries::InflowL, with a NEGATIVE
//density marking "not an inflow point here, fall back to outflow" -- physical
//densities are positive, so the sentinel is unambiguous and needs no mask array.
//_gradfree_ is the plain zeroth-order copy: the ghost takes the adjacent
//interior value verbatim. It permits spurious RE-ENTRY -- nothing stops the
//copied state having a normal velocity pointing back into the domain, so an
//inward pressure gradient sucks material in through what is nominally an exit.
//Measured on the Mach-800 jet: with an open base, the mean v_y outside the
//nozzle on the bottom row reached 727 towards the domain, against an injected
//800, which is not backflow draining but the boundary feeding the cocoon.
//_outflow_ is that copy plus a no-reentry clamp: if the ghost's normal momentum
//points inward it is zeroed, so waves leave and nothing enters.
//_inflow_ prescribes a state on the LOW side where the problem injects and is
//_outflow_ everywhere else -- high side, and the parts of the low side outside
//the nozzle. Prescribed values live in Boundaries::InflowL; a NEGATIVE density
//marks "not an inflow point here", and the array is initialised to that
//sentinel so an inflow boundary nobody fills degrades to outflow.
enum {_periodic_, _gradfree_, _reflective_, _inflow_, _outflow_};
enum {_integrator_ader_, _integrator_rk_};
//MHD face Riemann solver. Appended, never reordered: llf=0 and hlld=1 are the
//values every existing input and golden was generated under.
//hll is the two-wave solver of MDZ21 eq. 28, and is the base solver whose
//fan supplies the UCT-HLL emf coefficients (mhd_uct_hll_coeffs).
enum {_rsolver_llf_, _rsolver_hlld_, _rsolver_hll_};
//How the edge/corner electromotive force is built (mhd/emf).
//  2sweep -- two sequential 1-D edge Riemann sweeps (SD) + the four-state LLF
//            bound at demoted corners. The DEFAULT: it is what every MHD
//            golden in the tree encodes.
//  uct    -- upwind constrained transport (Mignone & Del Zanna 2021 eq. 33),
//            with the a/d coefficients taken from the face solver's own fan,
//            so mhd/rsolver picks the UCT flavour: hll -> UCT-HLL,
//            hlld -> UCT-HLLD. Not available under llf, which has no fan.
enum {_emf_2sweep_, _emf_uct_};
//MUSCL/FV slope limiter (fallback/limiter). minmod is the default and is what
//every existing golden encodes; see src/muscl.hpp.
enum {_lim_minmod_, _lim_vanleer_, _lim_moncen_};
enum {_nad_b_mag_, _nad_b_comps_};          // MHD NAD on |B| or (Bx,By,Bz)
enum {_nad_v_off_, _nad_v_mag_, _nad_v_comps_}; // MHD NAD velocity: off / |v| / comps
enum {_nad_scale_relative_, _nad_scale_delta_, _nad_scale_grange_, _nad_scale_gcfl_};
//How the multi-dimensional CFL bound is formed. SUM is
//  dt = cfl * min_d(dx_d) / sum_d(|v_d| + c_fast)
//and MIN is the standard unsplit form
//  dt = cfl * min_d( dx_d / (|v_d| + c_fast) ).
//In 2D with comparable wave speeds the sum form is ~2x smaller for the same
//nominal cfl -- measured 1.794x on the figure-22 KH lane at both 128^2 and
//2048^2 -- so `cfl` means different things under the two. SUM is the default
//because every golden in the tree was generated with it; Athena++/AthenaK use
//MIN, which is what makes their `cfl_number` directly comparable.
enum {_cfl_sum_, _cfl_min_};
enum {_ic_sine_wave_, _ic_sedov_, _ic_spherical_blast_, _ic_square_,
      _ic_sod_, _ic_shu_osher_, _ic_kelvin_helmholtz_, _ic_implosion_,
      _ic_rti_, _ic_user_, _ic_orszag_tang_, _ic_field_loop_,
      _ic_mhd_vortex_, _ic_mhd_blast_, _ic_mhd_jet_, _ic_current_sheet_, _ic_kh_mdz_, _ic_kh_rr22_};
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
//Pack-wide variants: the leading axis carries the block (boff = b*n_ader).
#define INDICES_L_B boff+t_id,var,NidL[_z_],NidL[_y_],NidL[_x_],nidL[_z_],nidL[_y_],nidL[_x_]
#define INDICES_R_B boff+t_id,var,NidR[_z_],NidR[_y_],NidR[_x_],nidR[_z_],nidR[_y_],nidR[_x_]

#define NODE nid[_z_],nid[_y_],nid[_x_]

#define FV_INDICES var,Nid[_z_],Nid[_y_],Nid[_x_]

//Batched (packed multi-meshblock) counterparts. A pack folds the block
//index into the leading axis, so the only difference from the per-block
//macros above is the boff term; declare it with BOFF(nader) at the top of
//a batched lambda. With nb = 1 and boff = 0 these are the macros above,
//which is why batching does not change the arithmetic.
#define BOFF(nader) const int boff = b*(nader)
#define B_INDICES (boff+t_id),var,Nid[_z_],Nid[_y_],Nid[_x_],nid[_z_],nid[_y_],nid[_x_]
#define B_INDICES_L (boff+t_id),var,NidL[_z_],NidL[_y_],NidL[_x_],nidL[_z_],nidL[_y_],nidL[_x_]
#define B_INDICES_R (boff+t_id),var,NidR[_z_],NidR[_y_],NidR[_x_],nidR[_z_],nidR[_y_],nidR[_x_]
#define B_FV_INDICES (boff+var),Nid[_z_],Nid[_y_],Nid[_x_]

//Bulk solution arrays live in device memory (CudaSpace): all per-step
//kernels touch only these, so no host<->device traffic occurs during the
//evolution loop. Small setup arrays (transform matrices, quadrature nodes,
//face coordinates) are allocated in the same device space and filled once
//at startup via host mirrors + deep_copy (never CudaUVMSpace).
//
//LayoutRight everywhere: with the index order (t,var,k,j,i,kk,jj,ii) the
//point index ii is stride-1, so a warp of consecutive flat indices reads
//contiguous memory, and var sits at a large stride outside the coalesced
//pattern. Measured on A100 (tests/bench_layout.cpp): 1120 GB/s for the
//production access pattern vs 457 GB/s with LayoutLeft. Files written on
//GPU and CPU now share the same (C-order) on-disk layout.
#ifdef KOKKOS_ENABLE_CUDA
#define MemSpace Kokkos::CudaSpace
#define SetupSpace MemSpace
#else
#define MemSpace Kokkos::HostSpace
#define SetupSpace MemSpace
#endif
#define Layout Kokkos::LayoutRight

using ExecSpace = MemSpace::execution_space;

typedef Kokkos::View<double*,SetupSpace>  Vector;
//Layout is EXPLICIT here, not defaulted. A View without a layout takes the
//memory space's preferred one -- LayoutRight on the host, LayoutLeft on
//CudaSpace -- so a packed Matrix silently changed shape between backends.
//level_fluxes_b hands each block its own coordinates as a raw row pointer
//(cxm.data() + b*extent(1)), which is only a row while rows are contiguous:
//on GPU that arithmetic walked the same column of successive blocks instead,
//fed non-monotonic coordinates into the MUSCL slopes, and NaN'd every level-1
//flux. Keeping Matrix LayoutRight also matches the bulk arrays below and makes
//host and device agree, which is what the CPU-vs-GPU comparisons assume.
typedef Kokkos::View<double**,Layout,SetupSpace>  Matrix;
typedef Kokkos::View<double********,Layout,MemSpace>  SD_Vector;
typedef Kokkos::View<double****,Layout,MemSpace> FV_Vector;

//Neighbour tables consumed by the batched ghost exchanges: one entry per
//block (or per block-face pair), so the block loop becomes a kernel axis.
typedef Kokkos::View<int*,SetupSpace>  IntVector;
typedef IntVector::host_mirror_type IntVector_h;

typedef Matrix::host_mirror_type Matrix_h;
typedef Vector::host_mirror_type Vector_h;
typedef SD_Vector::host_mirror_type SD_Vector_h;
typedef FV_Vector::host_mirror_type FV_Vector_h;

// Host mirror of a setup view for one-time initialization (device = SetupSpace).
template<typename View>
inline typename View::host_mirror_type setup_mirror(const View& dev){
    return Kokkos::create_mirror_view(dev);
}

template<typename View>
inline void setup_push(View& dev, const typename View::host_mirror_type& host){
    Kokkos::deep_copy(dev, host);
}

template<typename View>
inline void setup_pull(const View& dev, typename View::host_mirror_type& host){
    Kokkos::deep_copy(host, dev);
}

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

//Batched (multi-meshblock) decompositions. The block index is the
//slowest-varying component, so consecutive threads still walk contiguous
//memory inside one block: a single launch covers every block of a pack
//without changing the per-block access pattern.
KOKKOS_INLINE_FUNCTION
void flat_index7(unsigned idx,
                 unsigned Mz, unsigned My, unsigned Mx,
                 unsigned nz, unsigned ny, unsigned nx,
                 int oz, int oy, int ox,
                 int& b, int& k, int& j, int& i, int& kk, int& jj, int& ii){
    ii = int(idx % nx);      idx /= nx;
    jj = int(idx % ny);      idx /= ny;
    kk = int(idx % nz);      idx /= nz;
    i  = int(idx % Mx) + ox; idx /= Mx;
    j  = int(idx % My) + oy; idx /= My;
    k  = int(idx % Mz) + oz; idx /= Mz;
    b  = int(idx);
}

KOKKOS_INLINE_FUNCTION
void flat_index4(unsigned idx,
                 unsigned Mz, unsigned My, unsigned Mx,
                 int oz, int oy, int ox,
                 int& b, int& k, int& j, int& i){
    i  = int(idx % Mx) + ox; idx /= Mx;
    j  = int(idx % My) + oy; idx /= My;
    k  = int(idx % Mz) + oz; idx /= Mz;
    b  = int(idx);
}

//Box decompositions: like flat_index3/6 but with an explicit [lo,hi) per
//element axis, for the coarse-fine operators that work on a sub-box of a
//block rather than the whole grid or the whole interior.
KOKKOS_INLINE_FUNCTION
void flat_index2(unsigned idx, unsigned My, unsigned Mx,
                 int oy, int ox, int& j, int& i){
    i = int(idx % Mx) + ox; idx /= Mx;
    j = int(idx) + oy;
}

//3 element axes + 1 point axis (the fastest-varying one).
KOKKOS_INLINE_FUNCTION
void flat_index3p1(unsigned idx, unsigned Mz, unsigned My, unsigned Mx,
                   unsigned n1, int oz, int oy, int ox,
                   int& k, int& j, int& i, int& a){
    a = int(idx % n1);       idx /= n1;
    i = int(idx % Mx) + ox;  idx /= Mx;
    j = int(idx % My) + oy;  idx /= My;
    k = int(idx) + oz;
}

//3 element axes + 2 point axes; which two the caller means is its own business
//(the CT operators sweep yz, zx and xy faces).
KOKKOS_INLINE_FUNCTION
void flat_index3p2(unsigned idx, unsigned Mz, unsigned My, unsigned Mx,
                   unsigned n1, unsigned n2, int oz, int oy, int ox,
                   int& k, int& j, int& i, int& a, int& b){
    b = int(idx % n2);       idx /= n2;
    a = int(idx % n1);       idx /= n1;
    i = int(idx % Mx) + ox;  idx /= Mx;
    j = int(idx % My) + oy;  idx /= My;
    k = int(idx) + oz;
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

template <class Functor>
struct Flat7 {
    Functor f;
    unsigned Mz, My, Mx, nz, ny, nx;
    int oz, oy, ox;
    KOKKOS_INLINE_FUNCTION
    void operator()(const unsigned idx) const {
        int b, k, j, i, kk, jj, ii;
        flat_index7(idx, Mz, My, Mx, nz, ny, nx, oz, oy, ox, b, k, j, i, kk, jj, ii);
        f(b, k, j, i, kk, jj, ii);
    }
    KOKKOS_INLINE_FUNCTION
    void operator()(const unsigned idx, double& r) const {
        int b, k, j, i, kk, jj, ii;
        flat_index7(idx, Mz, My, Mx, nz, ny, nx, oz, oy, ox, b, k, j, i, kk, jj, ii);
        f(b, k, j, i, kk, jj, ii, r);
    }
};

template <class Functor>
struct Flat4 {
    Functor f;
    unsigned Mz, My, Mx;
    int oz, oy, ox;
    KOKKOS_INLINE_FUNCTION
    void operator()(const unsigned idx) const {
        int b, k, j, i;
        flat_index4(idx, Mz, My, Mx, oz, oy, ox, b, k, j, i);
        f(b, k, j, i);
    }
    KOKKOS_INLINE_FUNCTION
    void operator()(const unsigned idx, double& r) const {
        int b, k, j, i;
        flat_index4(idx, Mz, My, Mx, oz, oy, ox, b, k, j, i);
        f(b, k, j, i, r);
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

template <class Functor>
Flat7<Functor> make_flat7(const Functor& f, int Mz, int My, int Mx,
                          int nz, int ny, int nx, int oz, int oy, int ox){
    return Flat7<Functor>{f,unsigned(Mz),unsigned(My),unsigned(Mx),
                          unsigned(nz),unsigned(ny),unsigned(nx),oz,oy,ox};
}

template <class Functor>
Flat4<Functor> make_flat4(const Functor& f, int Mz, int My, int Mx,
                          int oz, int oy, int ox){
    return Flat4<Functor>{f,unsigned(Mz),unsigned(My),unsigned(Mx),oz,oy,ox};
}

template <class Functor>
struct Flat2 {
    Functor f;
    unsigned My, Mx;
    int oy, ox;
    KOKKOS_INLINE_FUNCTION
    void operator()(const unsigned idx) const {
        int j, i;
        flat_index2(idx, My, Mx, oy, ox, j, i);
        f(j, i);
    }
};

template <class Functor>
struct Flat3p1 {
    Functor f;
    unsigned Mz, My, Mx, n1;
    int oz, oy, ox;
    KOKKOS_INLINE_FUNCTION
    void operator()(const unsigned idx) const {
        int k, j, i, a;
        flat_index3p1(idx, Mz, My, Mx, n1, oz, oy, ox, k, j, i, a);
        f(k, j, i, a);
    }
};

template <class Functor>
struct Flat3p2 {
    Functor f;
    unsigned Mz, My, Mx, n1, n2;
    int oz, oy, ox;
    KOKKOS_INLINE_FUNCTION
    void operator()(const unsigned idx) const {
        int k, j, i, a, b;
        flat_index3p2(idx, Mz, My, Mx, n1, n2, oz, oy, ox, k, j, i, a, b);
        f(k, j, i, a, b);
    }
};

template <class Functor>
Flat2<Functor> make_flat2(const Functor& f, int My, int Mx, int oy, int ox){
    return Flat2<Functor>{f,unsigned(My),unsigned(Mx),oy,ox};
}

template <class Functor>
Flat3p1<Functor> make_flat3p1(const Functor& f, int Mz, int My, int Mx,
                              int n1, int oz, int oy, int ox){
    return Flat3p1<Functor>{f,unsigned(Mz),unsigned(My),unsigned(Mx),
                            unsigned(n1),oz,oy,ox};
}

template <class Functor>
Flat3p2<Functor> make_flat3p2(const Functor& f, int Mz, int My, int Mx,
                              int n1, int n2, int oz, int oy, int ox){
    return Flat3p2<Functor>{f,unsigned(Mz),unsigned(My),unsigned(Mx),
                            unsigned(n1),unsigned(n2),oz,oy,ox};
}


//Host-side element loop (startup IC projection only; avoids device quadrature
//kernels reading setup views on some CUDA builds).
template <class Functor>
void sd_for_cells_host(int Nz, int Ny, int Nx, int nz, int ny, int nx,
                       const Functor& f){
    for(int k=0;k<Nz;k++)
    for(int j=0;j<Ny;j++)
    for(int i=0;i<Nx;i++)
    for(int kk=0;kk<nz;kk++)
    for(int jj=0;jj<ny;jj++)
    for(int ii=0;ii<nx;ii++)
        f(k,j,i,kk,jj,ii);
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

//Max-reduction over interior elements; blocks until the result is ready.
//Lambda signature: (int k, int j, int i, int kk, int jj, int ii, double& reduce)
template <class Functor>
double sd_max_cells(int Nz, int Ny, int Nx, int nz, int ny, int nx,
                    const Functor& f){
    int Mz=Nz-2*NGHz, My=Ny-2*NGHy, Mx=Nx-2*NGHx;
    int64_t total = (int64_t)Mz*My*Mx*nz*ny*nx;
    double max_value=0;
    if(total <= 0) return max_value;
    Kokkos::parallel_reduce("sd_max_cells", flat_range(0,flat_total(total)),
        make_flat6(f,Mz,My,Mx,nz,ny,nx,NGHz,NGHy,NGHx),
        Kokkos::Max<double>(max_value));
    return max_value;
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

//Loops over an explicit [lo,hi) element box. The coarse-fine transfer
//operators touch only the part of a block that a neighbour covers, so they
//need a box rather than "all" or "interior".
//Lambda signature: (int j, int i)
template <class Functor>
void for_box2(int y0, int y1, int x0, int x1,
              const Functor& f, const char* label="for_box2"){
    int My=y1-y0, Mx=x1-x0;
    int64_t total = (int64_t)My*Mx;
    if(total <= 0) return;
    Kokkos::parallel_for(label, flat_range(0,flat_total(total)),
        make_flat2(f,My,Mx,y0,x0));
}

//Lambda signature: (int k, int j, int i)
template <class Functor>
void for_box3(int z0, int z1, int y0, int y1, int x0, int x1,
              const Functor& f, const char* label="for_box3"){
    int Mz=z1-z0, My=y1-y0, Mx=x1-x0;
    int64_t total = (int64_t)Mz*My*Mx;
    if(total <= 0) return;
    Kokkos::parallel_for(label, flat_range(0,flat_total(total)),
        make_flat3(f,Mz,My,Mx,z0,y0,x0));
}

//Lambda signature: (int k, int j, int i, int a)
template <class Functor>
void for_box3p1(int z0, int z1, int y0, int y1, int x0, int x1, int n1,
                const Functor& f, const char* label="for_box3p1"){
    int Mz=z1-z0, My=y1-y0, Mx=x1-x0;
    int64_t total = (int64_t)Mz*My*Mx*n1;
    if(total <= 0) return;
    Kokkos::parallel_for(label, flat_range(0,flat_total(total)),
        make_flat3p1(f,Mz,My,Mx,n1,z0,y0,x0));
}

//Lambda signature: (int k, int j, int i, int a, int b)
template <class Functor>
void for_box3p2(int z0, int z1, int y0, int y1, int x0, int x1, int n1, int n2,
                const Functor& f, const char* label="for_box3p2"){
    int Mz=z1-z0, My=y1-y0, Mx=x1-x0;
    int64_t total = (int64_t)Mz*My*Mx*n1*n2;
    if(total <= 0) return;
    Kokkos::parallel_for(label, flat_range(0,flat_total(total)),
        make_flat3p2(f,Mz,My,Mx,n1,n2,z0,y0,x0));
}

//Element box with every solution/flux point of each element.
//Lambda signature: (int k, int j, int i, int kk, int jj, int ii)
template <class Functor>
void sd_for_box_cells(int z0, int z1, int y0, int y1, int x0, int x1,
                      int nz, int ny, int nx,
                      const Functor& f, const char* label="sd_for_box_cells"){
    int Mz=z1-z0, My=y1-y0, Mx=x1-x0;
    int64_t total = (int64_t)Mz*My*Mx*nz*ny*nx;
    if(total <= 0) return;
    Kokkos::parallel_for(label, flat_range(0,flat_total(total)),
        make_flat6(f,Mz,My,Mx,nz,ny,nx,z0,y0,x0));
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

//The two detection loops keep a STENCIL MARGIN, not a ghost width: _ngh leaves
//room for the one-cell reads of NAD/PAD/the blending ring, _2ngh for the
//two-cell reads of smooth extrema detection. The margin is a property of the
//stencil and must not follow the halo width -- a wider halo only means these
//run over more ghost layers, whose results the active-only update_cascade and
//the following exchange discard. (They were written as NGH and 2*NGH, which
//read as ghost widths and happened to equal 1 and 2.)
#define FV_STENCIL_MARGIN 1
#define FV_SED_MARGIN     2

template <class Functor>
void fv_for_cells_ngh(int Nz, int Ny, int Nx,
                      const Functor& f, const char* label="fv_for_cells_ngh"){
    const int mz=FV_STENCIL_MARGIN*(nGHz>0), my=FV_STENCIL_MARGIN*(nGHy>0),
              mx=FV_STENCIL_MARGIN*(nGHx>0);
    int Mz=Nz-2*mz, My=Ny-2*my, Mx=Nx-2*mx;
    int64_t total = (int64_t)Mz*My*Mx;
    if(total <= 0) return;
    Kokkos::parallel_for(label, flat_range(0,flat_total(total)),
        make_flat3(f,Mz,My,Mx,mz,my,mx));
}

template <class Functor>
void fv_for_cells_2ngh(int Nz, int Ny, int Nx,
                       const Functor& f, const char* label="fv_for_cells_2ngh"){
    const int mz=FV_SED_MARGIN*(nGHz>0), my=FV_SED_MARGIN*(nGHy>0),
              mx=FV_SED_MARGIN*(nGHx>0);
    int Mz=Nz-2*mz, My=Ny-2*my, Mx=Nx-2*mx;
    int64_t total = (int64_t)Mz*My*Mx;
    if(total <= 0) return;
    Kokkos::parallel_for(label, flat_range(0,flat_total(total)),
        make_flat3(f,Mz,My,Mx,mz,my,mx));
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

/////////////////////////////////////////////////////////////////////
// Batched variants: one launch spanning nb meshblocks
//
// Every meshblock carries the same element count regardless of its
// refinement level (only h and the coordinates differ), so a pack of
// blocks is rectangular and a single flattened range covers all of them.
// The lambda takes a leading block index b; per-block arrays are reached
// through the pack's folded leading extent (b*n_ader + t_id).
/////////////////////////////////////////////////////////////////////

//Lambda signature: (int b, int k, int j, int i, int kk, int jj, int ii)
template <class Functor>
void sd_for_cells_b(int nb, int Nz, int Ny, int Nx, int nz, int ny, int nx,
                    const Functor& f, const char* label="sd_for_cells_b"){
    int64_t total = (int64_t)nb*Nz*Ny*Nx*nz*ny*nx;
    if(total <= 0) return;
    Kokkos::parallel_for(label, flat_range(0,flat_total(total)),
        make_flat7(f,Nz,Ny,Nx,nz,ny,nx,0,0,0));
}

template <class Functor>
void sd_for_active_cells_b(int nb, int Nz, int Ny, int Nx, int nz, int ny, int nx,
                           const Functor& f, const char* label="sd_for_active_cells_b"){
    int Mz=Nz-2*NGHz, My=Ny-2*NGHy, Mx=Nx-2*NGHx;
    int64_t total = (int64_t)nb*Mz*My*Mx*nz*ny*nx;
    if(total <= 0) return;
    Kokkos::parallel_for(label, flat_range(0,flat_total(total)),
        make_flat7(f,Mz,My,Mx,nz,ny,nx,NGHz,NGHy,NGHx));
}

//Lambda signature: (int b, int k, int j, int i, int kk, int jj, int ii, double& reduce)
template <class Functor>
double sd_min_cells_b(int nb, int Nz, int Ny, int Nx, int nz, int ny, int nx,
                      const Functor& f){
    int Mz=Nz-2*NGHz, My=Ny-2*NGHy, Mx=Nx-2*NGHx;
    int64_t total = (int64_t)nb*Mz*My*Mx*nz*ny*nx;
    double min_value=1;
    if(total <= 0) return min_value;
    Kokkos::parallel_reduce("sd_min_cells_b", flat_range(0,flat_total(total)),
        make_flat7(f,Mz,My,Mx,nz,ny,nx,NGHz,NGHy,NGHx),
        Kokkos::Min<double>(min_value));
    return min_value;
}

template <class Functor>
double sd_max_cells_b(int nb, int Nz, int Ny, int Nx, int nz, int ny, int nx,
                      const Functor& f){
    int Mz=Nz-2*NGHz, My=Ny-2*NGHy, Mx=Nx-2*NGHx;
    int64_t total = (int64_t)nb*Mz*My*Mx*nz*ny*nx;
    double max_value=0;
    if(total <= 0) return max_value;
    Kokkos::parallel_reduce("sd_max_cells_b", flat_range(0,flat_total(total)),
        make_flat7(f,Mz,My,Mx,nz,ny,nx,NGHz,NGHy,NGHx),
        Kokkos::Max<double>(max_value));
    return max_value;
}

template <class Functor>
double sd_sum_active_cells_b(int nb, int Nz, int Ny, int Nx, int nz, int ny, int nx,
                             const Functor& f){
    int Mz=Nz-2*NGHz, My=Ny-2*NGHy, Mx=Nx-2*NGHx;
    int64_t total = (int64_t)nb*Mz*My*Mx*nz*ny*nx;
    double sum=0;
    if(total <= 0) return sum;
    Kokkos::parallel_reduce("sd_sum_active_cells_b", flat_range(0,flat_total(total)),
        make_flat7(f,Mz,My,Mx,nz,ny,nx,NGHz,NGHy,NGHx), sum);
    return sum;
}

//Lambda signature: (int b, int k, int j, int i)
template <class Functor>
void fv_for_cells_b(int nb, int Nz, int Ny, int Nx,
                    const Functor& f, const char* label="fv_for_cells_b"){
    int64_t total = (int64_t)nb*Nz*Ny*Nx;
    if(total <= 0) return;
    Kokkos::parallel_for(label, flat_range(0,flat_total(total)),
        make_flat4(f,Nz,Ny,Nx,0,0,0));
}

//Stencil margins, as in the unbatched pair above.
template <class Functor>
void fv_for_cells_ngh_b(int nb, int Nz, int Ny, int Nx,
                        const Functor& f, const char* label="fv_for_cells_ngh_b"){
    const int mz=FV_STENCIL_MARGIN*(nGHz>0), my=FV_STENCIL_MARGIN*(nGHy>0),
              mx=FV_STENCIL_MARGIN*(nGHx>0);
    int Mz=Nz-2*mz, My=Ny-2*my, Mx=Nx-2*mx;
    int64_t total = (int64_t)nb*Mz*My*Mx;
    if(total <= 0) return;
    Kokkos::parallel_for(label, flat_range(0,flat_total(total)),
        make_flat4(f,Mz,My,Mx,mz,my,mx));
}

template <class Functor>
void fv_for_cells_2ngh_b(int nb, int Nz, int Ny, int Nx,
                         const Functor& f, const char* label="fv_for_cells_2ngh_b"){
    const int mz=FV_SED_MARGIN*(nGHz>0), my=FV_SED_MARGIN*(nGHy>0),
              mx=FV_SED_MARGIN*(nGHx>0);
    int Mz=Nz-2*mz, My=Ny-2*my, Mx=Nx-2*mx;
    int64_t total = (int64_t)nb*Mz*My*Mx;
    if(total <= 0) return;
    Kokkos::parallel_for(label, flat_range(0,flat_total(total)),
        make_flat4(f,Mz,My,Mx,mz,my,mx));
}

//Lambda signature: (int b, int k, int j, int i, double& reduce)
template <class Functor>
double fv_sum_cells_ngh2_b(int nb, int Nz, int Ny, int Nx, const Functor& f){
    int Mz=Nz-2*nGHz, My=Ny-2*nGHy, Mx=Nx-2*nGHx;
    int64_t total = (int64_t)nb*Mz*My*Mx;
    double sum=0;
    if(total <= 0) return sum;
    Kokkos::parallel_reduce("fv_sum_cells_ngh2_b", flat_range(0,flat_total(total)),
        make_flat4(f,Mz,My,Mx,nGHz,nGHy,nGHx), sum);
    return sum;
}

template <class Functor>
void fv_for_faces_b(int nb, int Nz, int Ny, int Nx,
                    const Functor& f, const char* label="fv_for_faces_b"){
    int Mz=Nz-2*nGHz+(nGHz>0), My=Ny-2*nGHy+(nGHy>0), Mx=Nx-2*nGHx+(nGHx>0);
    int64_t total = (int64_t)nb*Mz*My*Mx;
    if(total <= 0) return;
    Kokkos::parallel_for(label, flat_range(0,flat_total(total)),
        make_flat4(f,Mz,My,Mx,nGHz,nGHy,nGHx));
}

#endif
