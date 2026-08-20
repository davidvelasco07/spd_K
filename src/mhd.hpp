#ifndef MHD_HPP_
#define MHD_HPP_
//========================================================================================
// spd_K ideal-MHD module
//
// Coupled fluid + constrained-transport spectral-difference scheme, ported from the
// Python reference (spd/MHD, spd/induction/induction_sd_scheme.py). The 8-variable
// cell-centered state (rho, vx, vy, vz, P/E, Bx, By, Bz) is advanced with high-order SD
// fluxes and an MHD Riemann solver (LLF or Miyoshi–Kusano HLLD; see cfg.rsolver /
// mhd/rsolver), while the divergence-free magnetic field is carried on cell faces and
// evolved by CT from edge EMFs. The edge EMF couples the face B (interpolated to edges)
// with the fluid velocity/state (interpolated from the cell-centered primitives to the
// same edge points) through a matching electric-field Riemann solver.
//
// SSP-RK time integration only in this first port (ADER MHD deferred). The module
// registers its tasks into the Driver's stage lists (see driver.hpp).
//========================================================================================

using namespace std;

// Fixed 8-variable MHD layout (matches the Python spd ordering
// [rho, vx, vy, vz, P/E, Bx, By, Bz]); all velocity and field components are
// always carried, independent of the runtime dimensionality.
#define NMHD 8
#define NUCT 5   // face UCT coeffs: aL, dL, dR, vt1, vt2 (Mignone & Del Zanna)
#define _mrho_ 0
#define _mvx_  1
#define _mvy_  2
#define _mvz_  3
#define _mprs_ 4
#define _mbx_  5
#define _mby_  6
#define _mbz_  7

// Number of components in the edge Riemann state (E, B1, B2, v1, v2, B3, rho, p)
#define NEMHD 8

//----------------------------------------------------------------------------------------
// MHD equation kernels (mhd.cpp)
extern void mhd_compute_conservatives(SD_Solution W, SD_Solution U);
extern void mhd_compute_primitives(SD_Solution U, SD_Solution W);
extern void mhd_compute_primitives(FV_Solution U, FV_Solution W);
extern void mhd_floor_cv(SD_Solution U_cv, FV_Solution B_cv);
extern void mhd_compute_fluxes(SD_Solution W, SD_Solution F, int dim);
extern void mhd_riemann_solver(SD_Solution U, SD_Solution F, int dim,
                               SD_Solution Bn={}, SD_Solution UCT={});
extern void mhd_face_B_to_fp(SD_Solution U_fp, SD_Solution B_fp, int dim);
extern void mhd_uct_edge_E(SD_Solution E, SD_Solution UCT1, SD_Solution UCT2,
                           Matrix sp_to_fp, int edim);
extern double mhd_compute_dt(SD_Solution W, double dx, double dy, double dz);

// CT coupling kernels (mhd.cpp): edge EMF from fluid velocity + face B
extern void mhd_compute_E(SD_Solution E, SD_Solution W_sp,
                          SD_Solution B1, SD_Solution B2, SD_Solution Bcc,
                          Matrix sp_to_fp, int dim);
extern void mhd_E_riemann_solver(SD_Solution E, int dim, int v_index);
extern void mhd_B_to_U(SD_Solution U, SD_Solution Bx, SD_Solution By, SD_Solution Bz,
                       SD_Solution Tx, SD_Solution Ty, SD_Solution Tz, Matrix fp_to_sp);
extern void mhd_B_to_U_b(SD_Solution U, SD_Solution Bx, SD_Solution By, SD_Solution Bz,
                         Matrix fp_to_sp);
extern void mhd_compute_B_sp_from_fp(SD_Solution Bcc, SD_Solution Bx, SD_Solution By,
                                     SD_Solution Bz, Matrix fp_to_sp);

// Initial conditions (mhd.cpp)
extern void mhd_Initialize(SD_Solution W, Matrix faces_x, Matrix faces_y, Matrix faces_z,
                           Vector x_sp, Vector w_sp);
extern void mhd_Initialize_A(SD_Solution A, Matrix Xs, Matrix Ys, Matrix Zs, int dim);
extern void mhd_jet_inflow_apply(SD_Solution W, SD_Solution U,
                                 Matrix faces_x, Matrix faces_y,
                                 Vector x_sp, Vector w_sp);

// Diagnostics (mhd.cpp)
extern double mhd_max_divB(SD_Solution Bx, SD_Solution By, SD_Solution Bz,
                           Matrix dfp_to_sp, double dx, double dy, double dz);

// MOOD trouble detection (mhd.cpp): NAD on control-volume averages (rho, gas p,
// and B as components or |B| per cfg.mood_nad_b) + magnetic PAD. Tolerance scale
// is cfg.mood_nad_scale (relative|delta|grange|gcfl).
extern int  mhd_detection_vars(FV_Solution U, FV_Solution det);
extern void mhd_nad_compute_gscales(FV_Solution det_old, FV_Solution W, int nvar,
                                    double* gscale, double dt,
                                    double dx, double dy, double dz,
                                    bool apply_cfl=true);
extern void mhd_nad_partial_gscales(FV_Solution det_old, FV_Solution W, int nvar,
                                    double* gmin, double* gmax, double& vmax);
extern void mhd_nad_finalize_gscales(const double* gmin, const double* gmax, double vmax,
                                     int nvar, double* gscale, double dt, double dxmin,
                                     bool apply_cfl=true);
extern void mhd_NAD(FV_Solution det_new, FV_Solution det_old, FV_Solution troubles,
                    double tol, int nvar, const double* gscale);
extern void mhd_PAD(FV_Solution U, FV_Solution troubles);
extern void mhd_detect_troubles(FV_Solution U_new, FV_Solution U_old,
                                FV_Solution det_new, FV_Solution det_old,
                                FV_Solution troubles, bool PAD);

// Low-order FV operators + MOOD cascade assembly (mhd.cpp)
extern void mhd_face_B_to_fv(SD_Solution B, FV_Solution Bfv, int dim);
extern void mhd_fv_fluxes(FV_Solution W, FV_Solution F, FV_Solution Bn_f, FV_Solution UCT,
                          Vector x_c, Vector x_f, Vector y_c, Vector y_f, Vector z_c, Vector z_f,
                          int dim, bool muscl);
extern void mhd_four_state_E(FV_Solution E, FV_Solution W,
                             Vector x_c, Vector x_f, Vector y_c, Vector y_f, Vector z_c, Vector z_f,
                             int dim, bool muscl);
extern void mhd_uct_corner_E(FV_Solution E, FV_Solution Bx, FV_Solution By, FV_Solution Bz,
                             FV_Solution UCTx, FV_Solution UCTy, FV_Solution UCTz,
                             int Nz, int Ny, int Nx,
                             Vector x_c, Vector x_f, Vector y_c, Vector y_f, Vector z_c, Vector z_f,
                             int dim, bool muscl);
extern void mhd_assign_edge_E(FV_Solution E0, FV_Solution E1, FV_Solution E2,
                              FV_Solution cascade, int dim);
extern void mhd_set_candidate_B(FV_Solution U_new, FV_Solution B_cand);
extern void mhd_set_candidate_B_b(FV_Solution U_new, FV_Solution B_cand);
extern void mhd_floor_cv_b(SD_Solution U_cv, FV_Solution B_cv);
extern void mhd_assign_edge_E_b(FV_Solution E0, FV_Solution E1, FV_Solution E2,
                                FV_Solution cascade, int dim);
extern void mhd_compute_primitives_b(FV_Solution U, FV_Solution W);
extern int  mhd_detection_vars_b(FV_Solution U, FV_Solution det);
extern void mhd_face_B_to_fv_b(SD_Solution B, FV_Solution Bfv, int dim);
extern void mhd_face_B_to_fp_b(SD_Solution U_fp, SD_Solution B_fp, int dim);
extern void mhd_compute_fluxes_b(SD_Solution U, SD_Solution F, int dim);
extern void mhd_riemann_solver_b(SD_Solution U, SD_Solution F, int dim,
                                 SD_Solution Bn={}, SD_Solution UCT={});
extern void mhd_compute_E_b(SD_Solution E, SD_Solution W_sp, SD_Solution B1,
                            SD_Solution B2, SD_Solution Bcc, Matrix sp_to_fp, int dim);
extern void mhd_E_riemann_solver_b(SD_Solution E, int dim, int v_index);
extern void mhd_fv_fluxes_b(FV_Solution W, FV_Solution F, FV_Solution Bn_f, FV_Solution UCT,
                            Matrix x_c, Matrix x_f, Matrix y_c, Matrix y_f,
                            Matrix z_c, Matrix z_f, int dim, bool muscl);
extern void mhd_four_state_E_b(FV_Solution E, FV_Solution W,
                               Matrix x_c, Matrix x_f, Matrix y_c, Matrix y_f,
                               Matrix z_c, Matrix z_f, int dim, bool muscl);

//========================================================================================
//! \struct MHD_ader
//  \brief coupled fluid + constrained-transport SD module (SSP-RK). Supports 3D and
//         true 2D (x-y plane): in 2D the CT machinery degenerates to the Ez edge family
//         (Bx = -dEz/dy, By = +dEz/dx on faces) and Bz evolves as a cell-centered
//         conserved variable, exactly as in the Python spd reference. Registers its
//         tasks into the Driver's stage lists.
//========================================================================================
struct MHD_ader : public PhysicsModule {
    int n_ader;   // 1 (RK only)
    int nvar;     // 8

    CommHelper comm_;
    dimension Xdim_;
    dimension Ydim_;
    dimension Zdim_;

    Vector wt;    // RK stage weight ({1})

    Matrix sp_to_fp;
    Matrix fp_to_sp;
    Matrix dfp_to_sp;
    Matrix sp_to_cv;
    Matrix cv_to_sp;
    Matrix fp_to_cv;

    Vector xx, wx;   // GL quadrature on [0,1] for IC / jet inflow projection

    // Fluid state (8-var, cell-centered)
    SD_Solution U_sp;
    SD_Solution W_sp;
    SD_Solution W_cv;
    SD_Solution U_cv;
    SD_Solution T_sweep;
    SD_Solution U0_sp;
    SD_Solution U_ader_sp;
    SD_Solution U_ader_fp_x, F_ader_fp_x;
    SD_Solution U_ader_fp_y, F_ader_fp_y;
    SD_Solution U_ader_fp_z, F_ader_fp_z;
    SD_Solution UCT_fp_x, UCT_fp_y, UCT_fp_z;  // face HLLD UCT coeffs (NUCT)
    Boundaries BC_fp_x, BC_fp_y, BC_fp_z;

    // Face-staggered magnetic field + CT scratch
    SD_Solution Bx_fp_x, By_fp_y, Bz_fp_z;
    SD_Solution B0x_fp_x, B0y_fp_y, B0z_fp_z;
    SD_Solution Tx_, Ty_, Tz_;   // scratch for B_to_U projection (unused placeholder)
    SD_Solution B2_cv;           // (Bx,By,Bz,|B|^2) cell averages for output

    // Vector-potential edge init
    SD_Solution Ax_ep_yz, Ay_ep_zx, Az_ep_xy;

    // Edge electric-field state (8-var: E,B1,B2,v1,v2,B3,rho,p)
    SD_Solution Ex_ep_yz, Ey_ep_zx, Ez_ep_xy;
    Boundaries BC_Ey_ep_x, BC_Ez_ep_x;
    Boundaries BC_Ex_ep_y, BC_Ez_ep_y;
    Boundaries BC_Ey_ep_z, BC_Ex_ep_z;

    //================================================================
    // MOOD cascade fallback (allocated only when cfg.fallback is set)
    //================================================================
    // Fluid FV state (8-var cell averages, ghosted for reconstruction)
    FV_Solution U_old_fv, U_new_fv, W_fv;
    FV_Solution det_old, det_new;        // NAD vars: rho,p,+B(+v) per mood_nad_*
    FV_Solution troubles;                // per-cell MOOD flag
    FV_Solution cascade;                 // ghosted per-cell cascade index
    FV_Boundaries BCu_x, BCu_y, BCu_z;   // 8-var halo (U_old_fv / W_fv)
    FV_Boundaries BCs_x, BCs_y, BCs_z;   // 1-var halo (troubles / cascade)
    // Transverse halos for face-B and UCT coeffs (edge recon is cell-centered in
    // the transverse directions). Named BCb_<comp>_<dir> / BCu_uct_<comp>_<dir>.
    FV_Boundaries BCb_x_y, BCb_x_z, BCb_y_x, BCb_y_z, BCb_z_x, BCb_z_y;
    FV_Boundaries BCu_x_y, BCu_x_z, BCu_y_x, BCu_y_z, BCu_z_x, BCu_z_y;
    // Per-level face fluxes (level 0 doubles as the assembled flux: the cascade
    // assembly overwrites demoted faces in place). face_integral scratch reuses
    // U_ader_fp_* (dead after the Riemann solve).
    FV_Solution F0_x,F1_x,F2_x, F0_y,F1_y,F2_y, F0_z,F1_z,F2_z;
    // Per-level UCT face coefficients (aL,dL,dR,vt1,vt2) for HLLD corner EMF.
    FV_Solution UCT1_x,UCT2_x, UCT1_y,UCT2_y, UCT1_z,UCT2_z;
    // Face-staggered B: FV-face working copy (SD layout) + FV cell/face arrays
    SD_Solution Bxf, Byf, Bzf;           // FV-face representation of the face field
    SD_Solution TB_x, TB_y, TB_z;        // scratch for sp<->cv face transforms
    FV_Solution Bx_old,By_old,Bz_old, Bx_new,By_new,Bz_new;
    FV_Solution B_old_cv, B_new_cv;      // (Bx,By,Bz,|B|^2) cell averages
    // Per-level edge E (FV edge lattice, single-var; level 0 doubles as the
    // assembled E, overwritten in place at demoted edges)
    FV_Solution E0x,E1x,E2x, E0y,E1y,E2y, E0z,E1z,E2z;

    bool standalone_ = true;

    //Non-null when this block's arrays are slices of a mesh-wide pack, so
    //mesh-level kernels can span every block in one launch (see BlockPack).
    BlockPack* pack_ = nullptr;
    int pib_ = 0;

    void alloc(SD_Solution& s, const char* name, int nader, int nv,
               dimension Zd, dimension Yd, dimension Xd, bool z, bool y, bool x){
        if(pack_) s.init_packed(*pack_, pib_, name, nader, nv, Zd, Yd, Xd, z, y, x);
        else      s.init(name, nader, nv, Zd, Yd, Xd, z, y, x);
    }

    void alloc(FV_Solution& s, const char* name, int nv,
               dimension Zd, dimension Yd, dimension Xd, bool z, bool y, bool x){
        if(pack_) s.init_packed(*pack_, pib_, name, nv, Zd, Yd, Xd, z, y, x);
        else      s.init(name, nv, Zd, Yd, Xd, z, y, x);
    }

    MHD_ader(
        CommHelper comm,
        int p,
        dimension X_dim,
        dimension Y_dim,
        dimension Z_dim,
        double* x,
        double* w,
        double* x_sp,
        double* x_fp,
        bool standalone=true, //false when driven as one block of a mesh
        BlockPack* pack=nullptr,
        int pack_ib=0,
        //See the same two parameters on Hydro_ader: one per-run operator set to
        //alias instead of rebuilding, and "this block is being rebuilt by a
        //regrid, so its initial conditions are about to be overwritten".
        const SDOperators* ops=nullptr,
        bool run_ic=true
    ) : comm_(comm), Xdim_(X_dim), Ydim_(Y_dim), Zdim_(Z_dim), standalone_(standalone),
        pack_(pack), pib_(pack_ib) {
        //Constrained transport needs at least the x-y plane: 3D runs evolve all
        //three face fields from three edge-EMF families; 2D (x-y, z inactive)
        //degenerates to the Ez family only, with Bz a cell-centered conserved
        //variable (matching the Python spd 2D MHD path).
        if(!(cfg.active[_x_] && cfg.active[_y_])){
            if(Master) cout<<"ERROR: the MHD solver requires the x and y directions "
                             "to be active (2D runs must use the x-y plane; CT is "
                             "degenerate in 1D)"<<endl;
            exit(1);
        }
        if(cfg.integrator != _integrator_rk_){
            if(Master) cout<<"ERROR: the MHD solver currently supports only SSP-RK "
                             "(time/integrator=rk1|rk2|rk3)"<<endl;
            exit(1);
        }
        nvar = NMHD;
        n_ader = 1;
        n_output = 0;
        n_step = 0;
        t = 0;

        //Alias the one per-run operator set (build_sd_operators) instead of
        //rebuilding six matrices and their host mirrors per block per adapt.
        //MHD is RK-only, so wt stays the single unit stage weight it always was
        //rather than being taken from the operator set.
        SDOperators local_ops;
        if(!ops){ build_sd_operators(local_ops, p, x_sp, x_fp); ops = &local_ops; }
        Kokkos::resize(wt,1);
        Kokkos::deep_copy(wt,1.0);
        sp_to_fp  = ops->sp_to_fp;
        fp_to_sp  = ops->fp_to_sp;
        dfp_to_sp = ops->dfp_to_sp;
        sp_to_cv  = ops->sp_to_cv;
        cv_to_sp  = ops->cv_to_sp;
        fp_to_cv  = ops->fp_to_cv;

        //Only mhd_Initialize below consumes these; they are the same GL rule for
        //every block, so take them from the operator set.
        Vector xx = ops->xx, wx = ops->wx;

        alloc(U_sp, "U_sp",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        alloc(W_sp, "W_sp",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        alloc(W_cv, "W_cv",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        alloc(U_cv, "U_cv",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        alloc(T_sweep, "T_sweep",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        alloc(U0_sp, "U0_sp",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);

        alloc(U_ader_sp, "U_ader_sp",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        alloc(U_ader_fp_x, "U_ader_fp_x",1,nvar,Z_dim,Y_dim,X_dim,0,0,1);
        alloc(F_ader_fp_x, "F_ader_fp_x",1,nvar,Z_dim,Y_dim,X_dim,0,0,1);
        BC_fp_x.init(X_dim,cfg.bc[_x_],1,nvar,Z_dim.N_total,Y_dim.N_total,1,Z_dim.n_sp,Y_dim.n_sp,1);
        alloc(U_ader_fp_y, "U_ader_fp_y",1,nvar,Z_dim,Y_dim,X_dim,0,1,0);
        alloc(F_ader_fp_y, "F_ader_fp_y",1,nvar,Z_dim,Y_dim,X_dim,0,1,0);
        BC_fp_y.init(Y_dim,cfg.bc[_y_],1,nvar,Z_dim.N_total,1,X_dim.N_total,Z_dim.n_sp,1,X_dim.n_sp);
        alloc(U_ader_fp_z, "U_ader_fp_z",1,nvar,Z_dim,Y_dim,X_dim,1,0,0);
        alloc(F_ader_fp_z, "F_ader_fp_z",1,nvar,Z_dim,Y_dim,X_dim,1,0,0);
        BC_fp_z.init(Z_dim,cfg.bc[_z_],1,nvar,1,Y_dim.N_total,X_dim.N_total,1,Y_dim.n_sp,X_dim.n_sp);
        UCT_fp_x.init("UCT_fp_x",1,NUCT,Z_dim,Y_dim,X_dim,0,0,1);
        UCT_fp_y.init("UCT_fp_y",1,NUCT,Z_dim,Y_dim,X_dim,0,1,0);
        UCT_fp_z.init("UCT_fp_z",1,NUCT,Z_dim,Y_dim,X_dim,1,0,0);

        alloc(Bx_fp_x, "Bx_fp_x",1,1,Z_dim,Y_dim,X_dim,0,0,1);
        alloc(By_fp_y, "By_fp_y",1,1,Z_dim,Y_dim,X_dim,0,1,0);
        alloc(Bz_fp_z, "Bz_fp_z",1,1,Z_dim,Y_dim,X_dim,1,0,0);
        alloc(B0x_fp_x, "B0x_fp_x",1,1,Z_dim,Y_dim,X_dim,0,0,1);
        alloc(B0y_fp_y, "B0y_fp_y",1,1,Z_dim,Y_dim,X_dim,0,1,0);
        alloc(B0z_fp_z, "B0z_fp_z",1,1,Z_dim,Y_dim,X_dim,1,0,0);
        alloc(B2_cv, "B2_cv",1,DIM+1,Z_dim,Y_dim,X_dim,0,0,0);

        alloc(Ax_ep_yz, "Ax_ep_yz",1,1,Z_dim,Y_dim,X_dim,1,1,0);
        alloc(Ay_ep_zx, "Ay_ep_zx",1,1,Z_dim,Y_dim,X_dim,1,0,1);
        alloc(Az_ep_xy, "Az_ep_xy",1,1,Z_dim,Y_dim,X_dim,0,1,1);

        alloc(Ex_ep_yz, "Ex_ep_yz",1,NEMHD,Z_dim,Y_dim,X_dim,1,1,0);
        alloc(Ey_ep_zx, "Ey_ep_zx",1,NEMHD,Z_dim,Y_dim,X_dim,1,0,1);
        alloc(Ez_ep_xy, "Ez_ep_xy",1,NEMHD,Z_dim,Y_dim,X_dim,0,1,1);

        BC_Ey_ep_x.init(X_dim,cfg.bc[_x_],1,NEMHD,Z_dim.N_total,Y_dim.N_total,1,Z_dim.n_fp,Y_dim.n_sp,1);
        BC_Ez_ep_x.init(X_dim,cfg.bc[_x_],1,NEMHD,Z_dim.N_total,Y_dim.N_total,1,Z_dim.n_sp,Y_dim.n_fp,1);
        BC_Ex_ep_y.init(Y_dim,cfg.bc[_y_],1,NEMHD,Z_dim.N_total,1,X_dim.N_total,Z_dim.n_fp,1,X_dim.n_sp);
        BC_Ez_ep_y.init(Y_dim,cfg.bc[_y_],1,NEMHD,Z_dim.N_total,1,X_dim.N_total,Z_dim.n_sp,1,X_dim.n_fp);
        BC_Ex_ep_z.init(Z_dim,cfg.bc[_z_],1,NEMHD,1,Y_dim.N_total,X_dim.N_total,1,Y_dim.n_fp,X_dim.n_sp);
        BC_Ey_ep_z.init(Z_dim,cfg.bc[_z_],1,NEMHD,1,Y_dim.N_total,X_dim.N_total,1,Y_dim.n_sp,X_dim.n_fp);

        if(cfg.fallback){
            alloc(U_old_fv, "U_old_fv",NMHD,Z_dim,Y_dim,X_dim,0,0,0);
            alloc(U_new_fv, "U_new_fv",NMHD,Z_dim,Y_dim,X_dim,0,0,0);
            alloc(W_fv, "W_fv",NMHD,Z_dim,Y_dim,X_dim,0,0,0);
            //8 slots: mhd_detection_vars writes rho, P, then B as |B| (1 slot) or
            //components (3, the mood_nad_b=comps default), then optionally v as
            //|v| (1) or components (3). Max 8 -- and the DEFAULT needs 5, so the
            //old 4-slot sizing overran on the first fallback step.
            alloc(det_old, "det_old",8,Z_dim,Y_dim,X_dim,0,0,0);
            alloc(det_new, "det_new",8,Z_dim,Y_dim,X_dim,0,0,0);
            alloc(troubles, "troubles",1,Z_dim,Y_dim,X_dim,0,0,0);
            alloc(cascade, "cascade",1,Z_dim,Y_dim,X_dim,0,0,0);
            BCu_x.init(X_dim,cfg.bc[_x_],NMHD,Z_dim.fv_ncells,Y_dim.fv_ncells,nGHx);
            BCu_y.init(Y_dim,cfg.bc[_y_],NMHD,Z_dim.fv_ncells,nGHy,X_dim.fv_ncells);
            BCu_z.init(Z_dim,cfg.bc[_z_],NMHD,nGHz,Y_dim.fv_ncells,X_dim.fv_ncells);
            BCs_x.init(X_dim,cfg.bc[_x_],1,Z_dim.fv_ncells,Y_dim.fv_ncells,nGHx);
            BCs_y.init(Y_dim,cfg.bc[_y_],1,Z_dim.fv_ncells,nGHy,X_dim.fv_ncells);
            BCs_z.init(Z_dim,cfg.bc[_z_],1,nGHz,Y_dim.fv_ncells,X_dim.fv_ncells);

            alloc(F0_x, "F0_x",NMHD,Z_dim,Y_dim,X_dim,0,0,1); alloc(F1_x, "F1_x",NMHD,Z_dim,Y_dim,X_dim,0,0,1);
            alloc(F2_x, "F2_x",NMHD,Z_dim,Y_dim,X_dim,0,0,1);
            alloc(F0_y, "F0_y",NMHD,Z_dim,Y_dim,X_dim,0,1,0); alloc(F1_y, "F1_y",NMHD,Z_dim,Y_dim,X_dim,0,1,0);
            alloc(F2_y, "F2_y",NMHD,Z_dim,Y_dim,X_dim,0,1,0);
            alloc(F0_z, "F0_z",NMHD,Z_dim,Y_dim,X_dim,1,0,0); alloc(F1_z, "F1_z",NMHD,Z_dim,Y_dim,X_dim,1,0,0);
            alloc(F2_z, "F2_z",NMHD,Z_dim,Y_dim,X_dim,1,0,0);
            //UCT face coefficients (aL,dL,dR,vt1,vt2) per cascade level for the MDZ
            //corner EMF. Allocated now; wired up with the emf knob.
            alloc(UCT1_x, "UCT1_x",NUCT,Z_dim,Y_dim,X_dim,0,0,1); alloc(UCT2_x, "UCT2_x",NUCT,Z_dim,Y_dim,X_dim,0,0,1);
            alloc(UCT1_y, "UCT1_y",NUCT,Z_dim,Y_dim,X_dim,0,1,0); alloc(UCT2_y, "UCT2_y",NUCT,Z_dim,Y_dim,X_dim,0,1,0);
            alloc(UCT1_z, "UCT1_z",NUCT,Z_dim,Y_dim,X_dim,1,0,0); alloc(UCT2_z, "UCT2_z",NUCT,Z_dim,Y_dim,X_dim,1,0,0);
            alloc(Bxf, "Bxf",1,1,Z_dim,Y_dim,X_dim,0,0,1);
            alloc(Byf, "Byf",1,1,Z_dim,Y_dim,X_dim,0,1,0);
            alloc(Bzf, "Bzf",1,1,Z_dim,Y_dim,X_dim,1,0,0);
            alloc(TB_x, "TB_x",1,1,Z_dim,Y_dim,X_dim,0,0,1);
            alloc(TB_y, "TB_y",1,1,Z_dim,Y_dim,X_dim,0,1,0);
            alloc(TB_z, "TB_z",1,1,Z_dim,Y_dim,X_dim,1,0,0);
            alloc(Bx_old, "Bx_old",1,Z_dim,Y_dim,X_dim,0,0,1); alloc(Bx_new, "Bx_new",1,Z_dim,Y_dim,X_dim,0,0,1);
            alloc(By_old, "By_old",1,Z_dim,Y_dim,X_dim,0,1,0); alloc(By_new, "By_new",1,Z_dim,Y_dim,X_dim,0,1,0);
            alloc(Bz_old, "Bz_old",1,Z_dim,Y_dim,X_dim,1,0,0); alloc(Bz_new, "Bz_new",1,Z_dim,Y_dim,X_dim,1,0,0);
            //Transverse halos for face B and the UCT coefficients (the edge
            //reconstruction is cell-centred transversally). FV_Boundaries are not
            //pack-allocated, so these keep .init().
            BCb_x_y.init(Y_dim,cfg.bc[_y_],1,Bx_old.Nz,nGHy,Bx_old.Nx);
            BCb_x_z.init(Z_dim,cfg.bc[_z_],1,nGHz,Bx_old.Ny,Bx_old.Nx);
            BCb_y_x.init(X_dim,cfg.bc[_x_],1,By_old.Nz,By_old.Ny,nGHx);
            BCb_y_z.init(Z_dim,cfg.bc[_z_],1,nGHz,By_old.Ny,By_old.Nx);
            BCb_z_x.init(X_dim,cfg.bc[_x_],1,Bz_old.Nz,Bz_old.Ny,nGHx);
            BCb_z_y.init(Y_dim,cfg.bc[_y_],1,Bz_old.Nz,nGHy,Bz_old.Nx);
            BCu_x_y.init(Y_dim,cfg.bc[_y_],NUCT,UCT1_x.Nz,nGHy,UCT1_x.Nx);
            BCu_x_z.init(Z_dim,cfg.bc[_z_],NUCT,nGHz,UCT1_x.Ny,UCT1_x.Nx);
            BCu_y_x.init(X_dim,cfg.bc[_x_],NUCT,UCT1_y.Nz,UCT1_y.Ny,nGHx);
            BCu_y_z.init(Z_dim,cfg.bc[_z_],NUCT,nGHz,UCT1_y.Ny,UCT1_y.Nx);
            BCu_z_x.init(X_dim,cfg.bc[_x_],NUCT,UCT1_z.Nz,UCT1_z.Ny,nGHx);
            BCu_z_y.init(Y_dim,cfg.bc[_y_],NUCT,UCT1_z.Nz,nGHy,UCT1_z.Nx);
            alloc(B_old_cv, "B_old_cv",4,Z_dim,Y_dim,X_dim,0,0,0);
            alloc(B_new_cv, "B_new_cv",4,Z_dim,Y_dim,X_dim,0,0,0);

            alloc(E0x, "E0x",1,Z_dim,Y_dim,X_dim,1,1,0);
            alloc(E1x, "E1x",1,Z_dim,Y_dim,X_dim,1,1,0); alloc(E2x, "E2x",1,Z_dim,Y_dim,X_dim,1,1,0);
            alloc(E0y, "E0y",1,Z_dim,Y_dim,X_dim,1,0,1);
            alloc(E1y, "E1y",1,Z_dim,Y_dim,X_dim,1,0,1); alloc(E2y, "E2y",1,Z_dim,Y_dim,X_dim,1,0,1);
            alloc(E0z, "E0z",1,Z_dim,Y_dim,X_dim,0,1,1);
            alloc(E1z, "E1z",1,Z_dim,Y_dim,X_dim,0,1,1); alloc(E2z, "E2z",1,Z_dim,Y_dim,X_dim,0,1,1);
        }

        ////////////////////////
        // Initial conditions
        ////////////////////////
        //Skipped for a block a regrid is rebuilding -- see Hydro_ader. The face
        //field is transferred by install_snap / prolongate_block_face_B, which
        //is exactly why it must NOT be re-inited from the vector potential here.
        if(!run_ic){ Dt = 0.0; return; }
        // Fluid primitives (control-volume averages) -> W_cv -> W_sp. In 2D the
        // Bz row keeps the primitive IC value (cell-centered variable); in 3D
        // all B rows are overwritten from the CT face field below.
        mhd_Initialize(W_cv,X_dim.sd_faces,Y_dim.sd_faces,Z_dim.sd_faces,xx,wx);
        transform_cv_to_sp(W_cv,W_sp);
        // Divergence-free B from the vector potential (B = curl A); the
        // inactive-direction terms are skipped inside rotational_a_to_b, and in
        // 2D only Az contributes (Bx = dAz/dy, By = -dAz/dx).
        mhd_Initialize_A(Ax_ep_yz,X_dim.sd_centers,Y_dim.sd_faces  ,Z_dim.sd_faces  ,_x_);
        mhd_Initialize_A(Ay_ep_zx,X_dim.sd_faces  ,Y_dim.sd_centers,Z_dim.sd_faces  ,_y_);
        mhd_Initialize_A(Az_ep_xy,X_dim.sd_faces  ,Y_dim.sd_faces  ,Z_dim.sd_centers,_z_);
        rotational_a_to_b(Ay_ep_zx,Az_ep_xy,Bx_fp_x,dfp_to_sp,Y_dim.h,Z_dim.h,_x_);
        rotational_a_to_b(Az_ep_xy,Ax_ep_yz,By_fp_y,dfp_to_sp,Z_dim.h,X_dim.h,_y_);
        if(cfg.active[_z_])
            rotational_a_to_b(Ax_ep_yz,Ay_ep_zx,Bz_fp_z,dfp_to_sp,X_dim.h,Y_dim.h,_z_);
        // Project the CT face field onto the cell-centered B rows of the PRIMITIVE
        // state, then build the conservatives so the total energy includes the
        // magnetic energy of the actual (divergence-free) B field.
        mhd_B_to_U(W_sp,Bx_fp_x,By_fp_y,Bz_fp_z,Tx_,Ty_,Tz_,fp_to_sp);
        mhd_compute_conservatives(W_sp,U_sp);
        cons_to_prim_cv();

        Dt = mhd_compute_dt(W_cv,X_dim.h,Y_dim.h,Z_dim.h);
        if(standalone_){
            if(Master) cout<<"dx = "<<X_dim.h<<" dt = "<<Dt<<endl;
            if(cfg.outputs) Write_outputs();
        }
    }

    /////////////////////////////////////////////////////////////////////
    // Tasklist interface
    /////////////////////////////////////////////////////////////////////
    void AssembleTasks(Driver* d) override {
        TaskID none(0);
        auto bti = d->tl_map["before_timeintegrator"];
        auto stg = d->tl_map["stagen"];
        auto ati = d->tl_map["after_timeintegrator"];
        bti->AddTask(&MHD_ader::TaskSaveState, this, none);
        TaskID copy = stg->AddTask(&MHD_ader::TaskCopyCons, this, none);
        TaskID adv  = stg->AddTask(&MHD_ader::TaskAdvance,  this, copy);
        TaskID comb = stg->AddTask(&MHD_ader::TaskCombine,  this, adv);
        stg->AddTask(&MHD_ader::TaskBtoU, this, comb);
        ati->AddTask(&MHD_ader::TaskConsToPrim, this, none);
    }

    TaskStatus TaskSaveState(Driver* d, int stage){
        Kokkos::deep_copy(U0_sp.Vector,U_sp.Vector);
        Kokkos::deep_copy(B0x_fp_x.Vector,Bx_fp_x.Vector);
        Kokkos::deep_copy(B0y_fp_y.Vector,By_fp_y.Vector);
        Kokkos::deep_copy(B0z_fp_z.Vector,Bz_fp_z.Vector);
        return TaskStatus::complete;
    }

    TaskStatus TaskCopyCons(Driver* d, int stage){
        Kokkos::deep_copy(U_ader_sp.Vector,U_sp.Vector);
        //Primitives at solution points feed the edge-EMF velocity interpolation
        mhd_compute_primitives(U_sp,W_sp);
        return TaskStatus::complete;
    }

    void apply_jet_inflow(){
        if(cfg.problem!=_ic_mhd_jet_) return;
        mhd_jet_inflow_apply(W_sp,U_sp,Xdim_.sd_faces,Ydim_.sd_faces,xx,wx);
        Kokkos::deep_copy(U_ader_sp.Vector,U_sp.Vector);
    }

    TaskStatus TaskAdvance(Driver* d, int stage){
        apply_jet_inflow();
        Solve_faces(comm_);
        Solve_E(comm_);
        if(cfg.fallback){
            //MOOD cascade: reproduces the high-order update where no cell is
            //flagged, and locally demotes troubled cells' faces and edges to
            //MUSCL / first-order (single-valued -> conservation and divB=0 hold)
            MOOD_update(comm_);
        } else {
            Update_CT();
        }
        apply_jet_inflow();
        return TaskStatus::complete;
    }

    TaskStatus TaskCombine(Driver* d, int stage){
        if(d->rk_a[stage-1]>0){
            combine_solution(U_sp,U0_sp,d->rk_a[stage-1]);
            combine_solution(Bx_fp_x,B0x_fp_x,d->rk_a[stage-1]);
            combine_solution(By_fp_y,B0y_fp_y,d->rk_a[stage-1]);
            combine_solution(Bz_fp_z,B0z_fp_z,d->rk_a[stage-1]);
        }
        return TaskStatus::complete;
    }

    TaskStatus TaskBtoU(Driver* d, int stage){
        mhd_B_to_U(U_sp,Bx_fp_x,By_fp_y,Bz_fp_z,Tx_,Ty_,Tz_,fp_to_sp);
        return TaskStatus::complete;
    }

    TaskStatus TaskConsToPrim(Driver* d, int stage){
        mhd_compute_primitives(U_sp,W_sp);
        cons_to_prim_cv();
        return TaskStatus::complete;
    }

    double ComputeDt() override {
        return mhd_compute_dt(W_cv,Xdim_.h,Ydim_.h,Zdim_.h);
    }

    void WriteOutputs() override { Write_outputs(); }

    /////////////////////////////////////////////////////////////////////
    // Spatial operators
    /////////////////////////////////////////////////////////////////////
    //Split like Hydro_ader so a mesh driver can substitute block-to-block
    //exchanges for the single-block halo (standalone_=false skips BC).
    void Fluxes_pre(){
        bool az=cfg.active[_z_];
        transform_a_to_b_1d(U_ader_sp,U_ader_fp_x,sp_to_fp,_x_);
        transform_a_to_b_1d(U_ader_sp,U_ader_fp_y,sp_to_fp,_y_);
        if(az) transform_a_to_b_1d(U_ader_sp,U_ader_fp_z,sp_to_fp,_z_);
        //Take the normal B at the flux points from the single-valued CT face
        //field instead of the transform-reconstructed value. Gated on the
        //solver to match docs/mhd.md ("SD face HLLD overwrites the normal B
        //with the CT face field, never reconstructed"): unconditionally it also
        //changes the llf path, which moved the mhd_field_loop_* goldens by
        //3.7e-02 (fluid only -- B stays at round-off, since B is CT-evolved and
        //only the FLUID fluxes see the normal component). Arguably it is the
        //better treatment for llf too -- a face's normal field IS single-valued
        //under CT, and AthenaK always takes it from there -- but enabling it for
        //llf is a deliberate default change that needs the goldens regenerated,
        //not a merge side effect.
        if(cfg.rsolver != _rsolver_llf_){
            mhd_face_B_to_fp(U_ader_fp_x,Bx_fp_x,_x_);
            mhd_face_B_to_fp(U_ader_fp_y,By_fp_y,_y_);
            if(az) mhd_face_B_to_fp(U_ader_fp_z,Bz_fp_z,_z_);
        }
        mhd_compute_fluxes(U_ader_fp_x,F_ader_fp_x,_x_);
        mhd_compute_fluxes(U_ader_fp_y,F_ader_fp_y,_y_);
        if(az) mhd_compute_fluxes(U_ader_fp_z,F_ader_fp_z,_z_);
    }

    void apply_fp_boundaries(CommHelper comm){
        bool az=cfg.active[_z_];
        boundaries(comm,BC_fp_x,U_ader_fp_x);
        boundaries(comm,BC_fp_y,U_ader_fp_y);
        if(az) boundaries(comm,BC_fp_z,U_ader_fp_z);
    }

    void Riemann_Solver(){
        bool az=cfg.active[_z_];
        mhd_riemann_solver(U_ader_fp_x,F_ader_fp_x,_x_);
        mhd_riemann_solver(U_ader_fp_y,F_ader_fp_y,_y_);
        if(az) mhd_riemann_solver(U_ader_fp_z,F_ader_fp_z,_z_);
    }

    void Solve_faces(CommHelper comm){
        Fluxes_pre();
        if(standalone_) apply_fp_boundaries(comm);
        Riemann_Solver();
    }

    void Compute_E(){
        bool az=cfg.active[_z_];
        //Edge EMF from the face B field + fluid velocity (from W_sp). In 2D
        //only the Ez family exists (edges reduce to x-y corner points); the
        //Ex/Ey families would need the z-staggered field.
        mhd_compute_E(Ez_ep_xy,W_sp,Bx_fp_x,By_fp_y,U_sp,sp_to_fp,_z_);
        if(az){
            mhd_compute_E(Ey_ep_zx,W_sp,Bz_fp_z,Bx_fp_x,U_sp,sp_to_fp,_y_);
            mhd_compute_E(Ex_ep_yz,W_sp,By_fp_y,Bz_fp_z,U_sp,sp_to_fp,_x_);
        }
    }

    void E_Riemann_Solver(){
        bool az=cfg.active[_z_];
        //Edge Riemann (LLF-E); v_index 3/4 per the induction/Python convention
        if(az) mhd_E_riemann_solver(Ey_ep_zx,_x_,4);
        mhd_E_riemann_solver(Ez_ep_xy,_x_,3);
        mhd_E_riemann_solver(Ez_ep_xy,_y_,4);
        if(az){
            mhd_E_riemann_solver(Ex_ep_yz,_y_,3);
            mhd_E_riemann_solver(Ex_ep_yz,_z_,4);
            mhd_E_riemann_solver(Ey_ep_zx,_z_,3);
        }
    }

    void Solve_E(CommHelper comm){
        Compute_E();
        if(standalone_) apply_E_boundaries(comm);
        E_Riemann_Solver();
    }

    void apply_E_boundaries(CommHelper comm){
        bool az=cfg.active[_z_];
        if(az) boundaries(comm,BC_Ey_ep_x,Ey_ep_zx);
        boundaries(comm,BC_Ez_ep_x,Ez_ep_xy);
        if(az) boundaries(comm,BC_Ex_ep_y,Ex_ep_yz);
        boundaries(comm,BC_Ez_ep_y,Ez_ep_xy);
        if(az){
            boundaries(comm,BC_Ex_ep_z,Ex_ep_yz);
            boundaries(comm,BC_Ey_ep_z,Ey_ep_zx);
        }
    }

    //Fluid + constrained-transport face-B update (no MOOD).
    void Update_CT(){
        update_solution(U_sp,U_ader_sp,F_ader_fp_x,F_ader_fp_y,F_ader_fp_z,
                        dfp_to_sp,wt,Xdim_.h,Ydim_.h,Zdim_.h,dt);
        update_B_solution(Bx_fp_x,Ey_ep_zx,Ez_ep_xy,dfp_to_sp,wt,Ydim_.h,Zdim_.h,dt,_x_);
        update_B_solution(By_fp_y,Ez_ep_xy,Ex_ep_yz,dfp_to_sp,wt,Zdim_.h,Xdim_.h,dt,_y_);
        if(cfg.active[_z_])
            update_B_solution(Bz_fp_z,Ex_ep_yz,Ey_ep_zx,dfp_to_sp,wt,Xdim_.h,Ydim_.h,dt,_z_);
    }

    //W_cv is the primitive state of the conserved CELL AVERAGE:
    //primitives(sp_to_cv(U_sp)), not sp_to_cv(primitives(U_sp)). The conserved
    //average is what the scheme actually carries and what has to be admissible,
    //so the primitives are taken from it; averaging the primitives instead
    //applies a nonlinear map before the average and is a different quantity
    //(density agrees -- it is linear and shared by both sets -- which is why
    //the mass checks never saw the difference).
    //
    //NOTE: primitives(average) and average(primitives) differ at O(h^2)
    //because the map is nonlinear, so this conversion is second order however
    //large p is. If that is ever shown to cap the achievable order, add the
    //PLUTO-style fourth-order correction (a Laplacian term in the conversion)
    //rather than going back to averaging the primitives.
    //--------------------------------------------------------------------------
    //SYSTEM HOOKS -- the counterparts of Hydro_ader's. See the comment there:
    //Mesh calls these by one name for both systems so the step-level tasks carry
    //no `if constexpr (is_hydro)`. Add a hook, not a branch.
    //--------------------------------------------------------------------------

    static void primitives_b(SD_Solution U, SD_Solution W){ mhd_compute_primitives_b(U,W); }

    //MHD allocates U_ader_sp with n_ader = 1 (see the alloc below: it is RK-only),
    //so broadcasting the state into the ADER stages is a straight copy.
    //Hydro_ader::copy_ader is the general n_ader form; both are reached as
    //Block::copy_ader.
    static void copy_ader(SD_Solution U, SD_Solution U_ader){
        Kokkos::deep_copy(U_ader.Vector, U.Vector);
    }

    //nu is hydro's viscosity; MHD has no viscous dt term. Mesh::ComputeDt owns
    //the MPI reduction for both systems.
    static double dt_b(SD_Solution W, Vector hx, Vector hy, Vector hz, double nu){
        (void)nu;
        return mhd_compute_dt_b(W,hx,hy,hz);
    }

    //The RK state is the conserved volume state AND the staggered face field --
    //miss the latter and the CT half of an RK stage is not restored.
    static constexpr std::array<std::pair<const char*,const char*>,4> rk_state(){
        return {{{"U0_sp","U_sp"},
                 {"B0x_fp_x","Bx_fp_x"},
                 {"B0y_fp_y","By_fp_y"},
                 {"B0z_fp_z","Bz_fp_z"}}};
    }

    //compute_E reads W_sp inside the stage, so the primitives must be current at
    //the top of one.
    static constexpr bool prim_at_stage_start = true;

    void cons_to_prim_cv(){
        transform_sp_to_cv(U_sp, U_cv);
        mhd_compute_primitives(U_cv, W_cv);
    }

    void transform_cv_to_sp(SD_Solution U_cv_, SD_Solution U_sp_){
        transform_a_to_b(U_cv_,U_sp_,T_sweep,cv_to_sp);
    }
    void transform_sp_to_cv(SD_Solution U_sp_, SD_Solution U_cv_){
        transform_a_to_b(U_sp_,U_cv_,T_sweep,sp_to_cv);
    }

    /////////////////////////////////////////////////////////////////////
    // MOOD cascade fallback
    //
    // The high-order (level 0) fluxes/edge-E are the SD Riemann results
    // integrated onto the sub-cell FV lattice (face_integral/edge_integral);
    // they reproduce the SD update to round-off when no cell is flagged.
    // Levels 1 (MUSCL) and 2 (first order) are built on the ghosted FV
    // primitive field W_fv. Each revision assembles a single-valued flux per
    // face and E per edge from the pooled cascade index, forms the candidate
    // fluid + CT cell averages, detects troubles (NAD on rho/p/B + magnetic PAD),
    // and demotes still-troubled cells. Because the
    // assembled flux (edge E) is single-valued, conservation (divB=0) is
    // preserved at every level.
    /////////////////////////////////////////////////////////////////////
    void mood_halo_U(CommHelper comm, FV_Solution U){
        if(!standalone_) return; //mesh fills ghosts via block_boundary_fv
        boundaries(comm,BCu_x,U,_center_,0);
        boundaries(comm,BCu_y,U,_center_,0);
        if(cfg.active[_z_]) boundaries(comm,BCu_z,U,_center_,0);
    }
    void mood_halo_scalar(CommHelper comm, FV_Solution S){
        if(!standalone_) return;
        boundaries(comm,BCs_x,S,_center_,0);
        boundaries(comm,BCs_y,S,_center_,0);
        if(cfg.active[_z_]) boundaries(comm,BCs_z,S,_center_,0);
    }
    // Transverse-only ghosts of the staggered face field / UCT coeffs for UCT corner E.
    void mood_halo_face_B(CommHelper comm){
        bool az=cfg.active[_z_];
        boundaries(comm,BCb_y_x,By_old,_center_,0);
        if(az) boundaries(comm,BCb_z_x,Bz_old,_center_,0);
        boundaries(comm,BCb_x_y,Bx_old,_center_,0);
        if(az) boundaries(comm,BCb_z_y,Bz_old,_center_,0);
        if(az){
            boundaries(comm,BCb_x_z,Bx_old,_center_,0);
            boundaries(comm,BCb_y_z,By_old,_center_,0);
        }
    }
    void mood_halo_uct(CommHelper comm, FV_Solution UCTx, FV_Solution UCTy, FV_Solution UCTz){
        bool az=cfg.active[_z_];
        boundaries(comm,BCu_y_x,UCTy,_center_,0);
        if(az) boundaries(comm,BCu_z_x,UCTz,_center_,0);
        boundaries(comm,BCu_x_y,UCTx,_center_,0);
        if(az) boundaries(comm,BCu_z_y,UCTz,_center_,0);
        if(az){
            boundaries(comm,BCu_x_z,UCTx,_center_,0);
            boundaries(comm,BCu_y_z,UCTy,_center_,0);
        }
    }
    // FV-face working copies (SD layout) of the current stage face field.
    void mood_reset_face_B(){
        transform_a_to_b_2d(Bx_fp_x,Bxf,TB_x,sp_to_cv,_x_);
        transform_a_to_b_2d(By_fp_y,Byf,TB_y,sp_to_cv,_y_);
        if(cfg.active[_z_]) transform_a_to_b_2d(Bz_fp_z,Bzf,TB_z,sp_to_cv,_z_);
    }
    // Single-valued flux/edge-E from the cascade index, assembled IN PLACE into
    // the level-0 arrays (valid because the cascade never decreases). In 2D the
    // z flux sweep and the Ex/Ey edge families do not exist.
    void mood_assemble(){
        bool az=cfg.active[_z_];
        assign_face_flux(F0_x,F1_x,F2_x,cascade,_x_);
        assign_face_flux(F0_y,F1_y,F2_y,cascade,_y_);
        if(az) assign_face_flux(F0_z,F1_z,F2_z,cascade,_z_);
        if(az){
            mhd_assign_edge_E(E0x,E1x,E2x,cascade,_x_);
            mhd_assign_edge_E(E0y,E1y,E2y,cascade,_y_);
        }
        mhd_assign_edge_E(E0z,E1z,E2z,cascade,_z_);
    }
    // Candidate CT face-B update from the assembled edge E: resets the FV-face
    // copy from the stage face field, applies the curl (writes the copy in
    // place), and forms the cell-averaged candidate field B_new_cv. The invalid
    // E-family terms are skipped inside fv_update_B_solution (2D: Ez only).
    void mood_ct_update(){
        mood_reset_face_B();
        fv_update_B_solution(Bx_new,Bx_old,Bxf,E0y,E0z,Ydim_.fv_faces,Zdim_.fv_faces,wt,dt,0,_x_,1);
        fv_update_B_solution(By_new,By_old,Byf,E0z,E0x,Zdim_.fv_faces,Xdim_.fv_faces,wt,dt,0,_y_,1);
        if(cfg.active[_z_])
            fv_update_B_solution(Bz_new,Bz_old,Bzf,E0x,E0y,Xdim_.fv_faces,Ydim_.fv_faces,wt,dt,0,_z_,1);
        compute_B_cv_from_cf(B_new_cv,Bxf,Byf,Bzf,fp_to_cv);
    }
    void mood_fluid_update(bool commit){
        fv_update_solution(U_new_fv,U_old_fv,U_cv,
                           F0_x,Xdim_.fv_faces,F0_y,Ydim_.fv_faces,F0_z,Zdim_.fv_faces,
                           wt,0,dt,commit);
    }

    void MOOD_update(CommHelper comm){
        mood_begin();
        mood_halo_U(comm,U_old_fv);
        mood_after_U_halo();
        for(int rev=0; rev<cfg.max_revs; rev++){
            int demoted = mood_revision();
            #ifdef MPI
            int g; MPI_Allreduce(&demoted,&g,1,MPI_INT,MPI_SUM,Comm); demoted=g;
            #endif
            if(demoted==0) break;
            mood_halo_scalar(comm,cascade);   // neighbour demotions feed the pooling
        }
        mood_commit();
    }

    //Mesh-callable MOOD phases (halos filled by the mesh between calls).
    void mood_begin(){
        transform_sp_to_cv(U_sp,U_cv);
        mood_reset_face_B();
        compute_B_cv_from_cf(B_old_cv,Bxf,Byf,Bzf,fp_to_cv);
        bool az=cfg.active[_z_];
        face_integral(F_ader_fp_x,F0_x,U_ader_fp_x,sp_to_cv,0,_x_);
        face_integral(F_ader_fp_y,F0_y,U_ader_fp_y,sp_to_cv,0,_y_);
        if(az) face_integral(F_ader_fp_z,F0_z,U_ader_fp_z,sp_to_cv,0,_z_);
        if(az){
            edge_integral(Ex_ep_yz,E0x,sp_to_cv,0,_x_);
            edge_integral(Ey_ep_zx,E0y,sp_to_cv,0,_y_);
        }
        edge_integral(Ez_ep_xy,E0z,sp_to_cv,0,_z_);
        fv_update_solution(U_new_fv,U_old_fv,U_cv,
                           F0_x,Xdim_.fv_faces,F0_y,Ydim_.fv_faces,F0_z,Zdim_.fv_faces,
                           wt,0,dt,0);
        mhd_set_candidate_B(U_old_fv,B_old_cv);
    }

    //True when the EMF is built by UCT. Wired to the emf knob; false for now, so
    //the merged tree reproduces HEAD's four-state LLF corner EMF exactly.
    bool use_uct = false;
    int n_det = 3;              //live detection variables (mhd_detection_vars)
    double nad_gscale[8] = {};  //frozen global NAD scales for grange/gcfl

    void mood_after_U_halo(){
        mhd_compute_primitives(U_old_fv,W_fv);
        //FV-face copy of the stage face field, consumed by the low-order sweep
        //below (mhd_fv_fluxes, under hlld) and by mhd_uct_corner_E. It MUST be
        //filled here: mood_ct_update refills it as a side effect of
        //fv_update_B_solution, but that runs AFTER the sweep, so without this the
        //first stage reads zeros and every later stage reads the previous one's
        //field. Bxf/Byf/Bzf hold the current stage field (mood_begin ran
        //mood_reset_face_B). Write-only under llf, which reconstructs b1.
        mhd_face_B_to_fv(Bxf,Bx_old,_x_);
        mhd_face_B_to_fv(Byf,By_old,_y_);
        if(cfg.active[_z_]) mhd_face_B_to_fv(Bzf,Bz_old,_z_);
        //Detection-variable count and the frozen global NAD scales, computed once
        //per stage from the haloed OLD state. Members because HEAD splits this
        //across mood_after_U_halo() and mood_revision().
        n_det = mhd_detection_vars(U_old_fv,det_old);
        //Standalone: one block IS the domain, so reduce it here. Under a mesh the
        //scales must span every block (Mesh::mhd_reduce_nad_gscales), or grange/
        //gcfl give each block its own NAD band and multiblock stops agreeing with
        //single-block -- measured on OT before this split.
        if(standalone_)
            mhd_nad_compute_gscales(det_old,W_fv,n_det,nad_gscale,dt,
                                    Xdim_.h,Ydim_.h,Zdim_.h);
        for(int dim=0; dim<3; dim++){
            if(cfg.active[dim]){
                FV_Solution &F1=(dim==_x_?F1_x:(dim==_y_?F1_y:F1_z));
                FV_Solution &F2=(dim==_x_?F2_x:(dim==_y_?F2_y:F2_z));
                FV_Solution &Bn=(dim==_x_?Bx_old:(dim==_y_?By_old:Bz_old));
                FV_Solution &U1=(dim==_x_?UCT1_x:(dim==_y_?UCT1_y:UCT1_z));
                FV_Solution &U2=(dim==_x_?UCT2_x:(dim==_y_?UCT2_y:UCT2_z));
                mhd_fv_fluxes(W_fv,F1,Bn,U1,Xdim_.fv_centers,Xdim_.fv_faces,Ydim_.fv_centers,
                              Ydim_.fv_faces,Zdim_.fv_centers,Zdim_.fv_faces,dim,true);
                mhd_fv_fluxes(W_fv,F2,Bn,U2,Xdim_.fv_centers,Xdim_.fv_faces,Ydim_.fv_centers,
                              Ydim_.fv_faces,Zdim_.fv_centers,Zdim_.fv_faces,dim,false);
            }
            int d1=(dim==_z_?_x_:(dim==_y_?_z_:_y_));
            int d2=(dim==_z_?_y_:(dim==_y_?_x_:_z_));
            if(cfg.active[d1] && cfg.active[d2]){
                FV_Solution &E1=(dim==_x_?E1x:(dim==_y_?E1y:E1z));
                FV_Solution &E2=(dim==_x_?E2x:(dim==_y_?E2y:E2z));
                if(use_uct){
                    //Halo UCT after all face sweeps for this level pair; do once
                    //outside the dim loop below after both MUSCL and FO fills.
                }else{
                    mhd_four_state_E(E1,W_fv,Xdim_.fv_centers,Xdim_.fv_faces,Ydim_.fv_centers,
                                     Ydim_.fv_faces,Zdim_.fv_centers,Zdim_.fv_faces,dim,true);
                    mhd_four_state_E(E2,W_fv,Xdim_.fv_centers,Xdim_.fv_faces,Ydim_.fv_centers,
                                     Ydim_.fv_faces,Zdim_.fv_centers,Zdim_.fv_faces,dim,false);
                }
            }
        }
        if(use_uct){
            //UCT coeffs need transverse ghosts before the corner composition.
            mood_halo_uct(comm_,UCT1_x,UCT1_y,UCT1_z);
            mood_halo_uct(comm_,UCT2_x,UCT2_y,UCT2_z);
            for(int dim=0; dim<3; dim++){
                int d1=(dim==_z_?_x_:(dim==_y_?_z_:_y_));
                int d2=(dim==_z_?_y_:(dim==_y_?_x_:_z_));
                if(!(cfg.active[d1] && cfg.active[d2])) continue;
                FV_Solution &E1=(dim==_x_?E1x:(dim==_y_?E1y:E1z));
                FV_Solution &E2=(dim==_x_?E2x:(dim==_y_?E2y:E2z));
                mhd_uct_corner_E(E1,Bx_old,By_old,Bz_old,UCT1_x,UCT1_y,UCT1_z,
                                 W_fv.Nz,W_fv.Ny,W_fv.Nx,
                                 Xdim_.fv_centers,Xdim_.fv_faces,Ydim_.fv_centers,
                                 Ydim_.fv_faces,Zdim_.fv_centers,Zdim_.fv_faces,dim,true);
                mhd_uct_corner_E(E2,Bx_old,By_old,Bz_old,UCT2_x,UCT2_y,UCT2_z,
                                 W_fv.Nz,W_fv.Ny,W_fv.Nx,
                                 Xdim_.fv_centers,Xdim_.fv_faces,Ydim_.fv_centers,
                                 Ydim_.fv_faces,Zdim_.fv_centers,Zdim_.fv_faces,dim,false);
            }
        }
        //deep_copy covers the ghosts too, so a forced level needs no halo.
        Kokkos::deep_copy(cascade.Vector,
                          cfg.mood_force_level>=0 ? (double)cfg.mood_force_level : 0.0);
    }

    //Mesh-driven NAD scales: accumulate this block's partial into the caller's
    //running min/max, then take back the single combined result.
    void mood_partial_gscales(double* gmin, double* gmax, double& vmax){
        mhd_nad_partial_gscales(det_old,W_fv,n_det,gmin,gmax,vmax);
    }
    void mood_set_gscales(const double* gs){
        for(int v=0; v<8; v++) nad_gscale[v]=gs[v];
    }

    //Candidate + detection from an ALREADY-ASSEMBLED flux/edge-E set. Split out
    //of mood_revision so a mesh can make the assembled flux single-valued at
    //block interfaces (Mesh::enforce_fv_flux_consistency) BEFORE the candidate
    //is built: the DMP test runs on that candidate, so a still-double-valued
    //flux biases the very decision driving the cascade.
    //Under mhd/mood_force_level every cell is pinned at that level, so there is
    //nothing to detect and nothing to demote: return 0 and let the caller fall
    //straight through to the commit (diagnostic lane -- pure MUSCL or pure
    //first-order CT on the subcell mesh).
    int mood_detect(){
        if(cfg.mood_force_level>=0) return 0;
        mood_fluid_update(false);
        mood_ct_update();
        mhd_set_candidate_B(U_new_fv,B_new_cv);
        mhd_detection_vars(U_new_fv,det_new);
        mhd_NAD(det_new,det_old,troubles,cfg.nad_tolerance,n_det,nad_gscale);
        mhd_PAD(U_new_fv,troubles);
        return update_cascade(troubles,cascade,2);
    }

    //One cascade revision; returns demoted count (caller may MPI-reduce).
    //Standalone form: one block IS the domain, so nothing to reconcile.
    int mood_revision(){
        if(cfg.mood_force_level>=0) return 0;
        mood_assemble();
        return mood_detect();
    }

    //Commit from an already-assembled (and, under a mesh, already corrected)
    //flux/edge-E set.
    void mood_commit_assembled(){
        bool az=cfg.active[_z_];
        mood_fluid_update(true);
        mood_ct_update();
        if(cfg.floor_cons) mhd_floor_cv(U_cv,B_new_cv);
        transform_cv_to_sp(U_cv,U_sp);
        transform_a_to_b_2d(Bxf,Bx_fp_x,TB_x,cv_to_sp,_x_);
        transform_a_to_b_2d(Byf,By_fp_y,TB_y,cv_to_sp,_y_);
        if(az) transform_a_to_b_2d(Bzf,Bz_fp_z,TB_z,cv_to_sp,_z_);
    }

    void mood_commit(){
        mood_assemble();
        mood_commit_assembled();
    }

    void Write_outputs(){
        double divB = mhd_max_divB(Bx_fp_x,By_fp_y,Bz_fp_z,dfp_to_sp,Xdim_.h,Ydim_.h,Zdim_.h);
        if(Master) cout<<endl<<"OUTPUT "<<n_output<<"  max|divB| = "<<divB<<endl;
        compute_B2_cv(B2_cv,Bx_fp_x,By_fp_y,Bz_fp_z,fp_to_cv,sp_to_cv);
        Write(W_cv,n_output);
        if(cfg.fallback) Write(cascade,n_output);
        Write(B2_cv,n_output++);
    }
};

#endif  // MHD_HPP_
