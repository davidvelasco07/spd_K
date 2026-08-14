#ifndef MHD_HPP_
#define MHD_HPP_
//========================================================================================
// spd_K ideal-MHD module
//
// Coupled fluid + constrained-transport spectral-difference scheme, ported from the
// Python reference (spd/MHD, spd/induction/induction_sd_scheme.py). The 8-variable
// cell-centered state (rho, vx, vy, vz, P/E, Bx, By, Bz) is advanced with high-order SD
// fluxes and an MHD LLF Riemann solver, while the divergence-free magnetic field is
// carried on cell faces and evolved by CT from edge EMFs. The edge EMF couples the
// face B (interpolated to edges) with the fluid velocity/state (interpolated from the
// cell-centered primitives to the same edge points) through an LLF electric-field
// Riemann solver based on the fast magnetosonic speed.
//
// SSP-RK time integration only in this first port (ADER MHD deferred). The module
// registers its tasks into the Driver's stage lists (see driver.hpp).
//========================================================================================

using namespace std;

// Fixed 8-variable MHD layout (matches the Python spd ordering
// [rho, vx, vy, vz, P/E, Bx, By, Bz]); all velocity and field components are
// always carried, independent of the runtime dimensionality.
#define NMHD 8
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
extern void mhd_riemann_solver(SD_Solution U, SD_Solution F, int dim);
extern double mhd_compute_dt(SD_Solution W, double dx, double dy, double dz);

// CT coupling kernels (mhd.cpp): edge EMF from fluid velocity + face B
extern void mhd_compute_E(SD_Solution E, SD_Solution W_sp,
                          SD_Solution B1, SD_Solution B2, SD_Solution Bcc,
                          Matrix sp_to_fp, int dim);
extern void mhd_E_riemann_solver(SD_Solution E, int dim, int v_index);
extern void mhd_B_to_U(SD_Solution U, SD_Solution Bx, SD_Solution By, SD_Solution Bz,
                       SD_Solution Tx, SD_Solution Ty, SD_Solution Tz, Matrix fp_to_sp);
extern void mhd_compute_B_sp_from_fp(SD_Solution Bcc, SD_Solution Bx, SD_Solution By,
                                     SD_Solution Bz, Matrix fp_to_sp);

// Initial conditions (mhd.cpp)
extern void mhd_Initialize(SD_Solution W, Matrix faces_x, Matrix faces_y, Matrix faces_z,
                           Vector x_sp, Vector w_sp);
extern void mhd_Initialize_A(SD_Solution A, Matrix Xs, Matrix Ys, Matrix Zs, int dim);

// Diagnostics (mhd.cpp)
extern double mhd_max_divB(SD_Solution Bx, SD_Solution By, SD_Solution Bz,
                           Matrix dfp_to_sp, double dx, double dy, double dz);

// MOOD trouble detection (mhd.cpp): |B|-based NAD on control-volume averages + PAD
extern void mhd_detection_vars(FV_Solution U, FV_Solution det);
extern void mhd_NAD(FV_Solution det_new, FV_Solution det_old, FV_Solution troubles, double tol);
extern void mhd_PAD(FV_Solution U, FV_Solution troubles);
extern void mhd_detect_troubles(FV_Solution U_new, FV_Solution U_old,
                                FV_Solution det_new, FV_Solution det_old,
                                FV_Solution troubles, bool PAD);

// Low-order FV operators + MOOD cascade assembly (mhd.cpp)
extern void mhd_fv_fluxes(FV_Solution W, FV_Solution F,
                          Vector x_c, Vector x_f, Vector y_c, Vector y_f, Vector z_c, Vector z_f,
                          int dim, bool muscl);
extern void mhd_four_state_E(FV_Solution E, FV_Solution W,
                             Vector x_c, Vector x_f, Vector y_c, Vector y_f, Vector z_c, Vector z_f,
                             int dim, bool muscl);
extern void mhd_assign_edge_E(FV_Solution E0, FV_Solution E1, FV_Solution E2,
                              FV_Solution cascade, int dim);
extern void mhd_set_candidate_B(FV_Solution U_new, FV_Solution B_cand);

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
    FV_Solution det_old, det_new;        // (rho, p, |B|, aggregate)
    FV_Solution troubles;                // per-cell MOOD flag
    FV_Solution cascade;                 // ghosted per-cell cascade index
    FV_Boundaries BCu_x, BCu_y, BCu_z;   // 8-var halo (U_old_fv / W_fv)
    FV_Boundaries BCs_x, BCs_y, BCs_z;   // 1-var halo (troubles / cascade)
    // Per-level face fluxes (level 0 doubles as the assembled flux: the cascade
    // assembly overwrites demoted faces in place). face_integral scratch reuses
    // U_ader_fp_* (dead after the Riemann solve).
    FV_Solution F0_x,F1_x,F2_x, F0_y,F1_y,F2_y, F0_z,F1_z,F2_z;
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
        int pack_ib=0
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

        Kokkos::resize(wt,1);
        Kokkos::deep_copy(wt,1.0);

        Kokkos::resize(sp_to_fp,p+2,p+1);
        Kokkos::resize(fp_to_sp,p+1,p+2);
        Kokkos::resize(dfp_to_sp,p+1,p+2);
        Kokkos::resize(sp_to_cv,p+1,p+1);
        Kokkos::resize(cv_to_sp,p+1,p+1);
        Kokkos::resize(fp_to_cv,p+1,p+2);
        //Setup matrices live in SetupSpace, which is CudaSpace on GPU: these
        //builders are host loops, so they must fill a mirror that is then
        //pushed (.cursor/rules/kokkos-no-uvm.mdc). Writing the device views
        //directly aborted every MHD run on CUDA with an inaccessible
        //memory-space error, which is why MHD had never run on the A100s.
        {
            Matrix_h sp_to_fp_h = setup_mirror(sp_to_fp);
            Matrix_h fp_to_sp_h = setup_mirror(fp_to_sp);
            Matrix_h dfp_to_sp_h = setup_mirror(dfp_to_sp);
            Matrix_h sp_to_cv_h = setup_mirror(sp_to_cv);
            Matrix_h cv_to_sp_h = setup_mirror(cv_to_sp);
            Matrix_h fp_to_cv_h = setup_mirror(fp_to_cv);
            lagrange_matrix(sp_to_fp_h, x_sp, x_fp, p+1, p+2);
            lagrange_matrix(fp_to_sp_h, x_fp, x_sp, p+2, p+1);
            lagrange_prime_matrix(dfp_to_sp_h, x_fp, x_sp, p+2, p+1);
            integral_matrix(sp_to_cv_h, x_fp, x_sp, p+1, p+1);
            integral_matrix(fp_to_cv_h, x_fp, x_fp, p+1, p+2);
            inverse(sp_to_cv_h, cv_to_sp_h, p+1);
            setup_push(sp_to_fp, sp_to_fp_h);
            setup_push(fp_to_sp, fp_to_sp_h);
            setup_push(dfp_to_sp, dfp_to_sp_h);
            setup_push(sp_to_cv, sp_to_cv_h);
            setup_push(cv_to_sp, cv_to_sp_h);
            setup_push(fp_to_cv, fp_to_cv_h);
        }

        Vector xx, wx;
        Kokkos::resize(xx,p+1);
        Kokkos::resize(wx,p+1);
        {
            Vector_h xx_h = setup_mirror(xx);
            Vector_h wx_h = setup_mirror(wx);
            gauss_legendre(0.0, 1.0, p+1, xx_h.data(), wx_h.data());
            setup_push(xx, xx_h);
            setup_push(wx, wx_h);
        }

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
            alloc(det_old, "det_old",4,Z_dim,Y_dim,X_dim,0,0,0);
            alloc(det_new, "det_new",4,Z_dim,Y_dim,X_dim,0,0,0);
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
            alloc(Bxf, "Bxf",1,1,Z_dim,Y_dim,X_dim,0,0,1);
            alloc(Byf, "Byf",1,1,Z_dim,Y_dim,X_dim,0,1,0);
            alloc(Bzf, "Bzf",1,1,Z_dim,Y_dim,X_dim,1,0,0);
            alloc(TB_x, "TB_x",1,1,Z_dim,Y_dim,X_dim,0,0,1);
            alloc(TB_y, "TB_y",1,1,Z_dim,Y_dim,X_dim,0,1,0);
            alloc(TB_z, "TB_z",1,1,Z_dim,Y_dim,X_dim,1,0,0);
            alloc(Bx_old, "Bx_old",1,Z_dim,Y_dim,X_dim,0,0,1); alloc(Bx_new, "Bx_new",1,Z_dim,Y_dim,X_dim,0,0,1);
            alloc(By_old, "By_old",1,Z_dim,Y_dim,X_dim,0,1,0); alloc(By_new, "By_new",1,Z_dim,Y_dim,X_dim,0,1,0);
            alloc(Bz_old, "Bz_old",1,Z_dim,Y_dim,X_dim,1,0,0); alloc(Bz_new, "Bz_new",1,Z_dim,Y_dim,X_dim,1,0,0);
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
        transform_sp_to_cv(W_sp,W_cv);

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

    TaskStatus TaskAdvance(Driver* d, int stage){
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
        transform_sp_to_cv(W_sp,W_cv);
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
    // fluid + CT cell averages, detects troubles (|B| NAD on control-volume
    // averages + magnetic PAD), and demotes still-troubled cells. Because the
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

    void mood_after_U_halo(){
        mhd_compute_primitives(U_old_fv,W_fv);
        mhd_detection_vars(U_old_fv,det_old);
        for(int dim=0; dim<3; dim++){
            if(cfg.active[dim]){
                FV_Solution &F1=(dim==_x_?F1_x:(dim==_y_?F1_y:F1_z));
                FV_Solution &F2=(dim==_x_?F2_x:(dim==_y_?F2_y:F2_z));
                mhd_fv_fluxes(W_fv,F1,Xdim_.fv_centers,Xdim_.fv_faces,Ydim_.fv_centers,
                              Ydim_.fv_faces,Zdim_.fv_centers,Zdim_.fv_faces,dim,true);
                mhd_fv_fluxes(W_fv,F2,Xdim_.fv_centers,Xdim_.fv_faces,Ydim_.fv_centers,
                              Ydim_.fv_faces,Zdim_.fv_centers,Zdim_.fv_faces,dim,false);
            }
            int d1=(dim==_z_?_x_:(dim==_y_?_z_:_y_));
            int d2=(dim==_z_?_y_:(dim==_y_?_x_:_z_));
            if(cfg.active[d1] && cfg.active[d2]){
                FV_Solution &E1=(dim==_x_?E1x:(dim==_y_?E1y:E1z));
                FV_Solution &E2=(dim==_x_?E2x:(dim==_y_?E2y:E2z));
                mhd_four_state_E(E1,W_fv,Xdim_.fv_centers,Xdim_.fv_faces,Ydim_.fv_centers,
                                 Ydim_.fv_faces,Zdim_.fv_centers,Zdim_.fv_faces,dim,true);
                mhd_four_state_E(E2,W_fv,Xdim_.fv_centers,Xdim_.fv_faces,Ydim_.fv_centers,
                                 Ydim_.fv_faces,Zdim_.fv_centers,Zdim_.fv_faces,dim,false);
            }
        }
        Kokkos::deep_copy(cascade.Vector,0.0);
    }

    //One cascade revision; returns demoted count (caller may MPI-reduce).
    int mood_revision(){
        mood_assemble();
        mood_fluid_update(false);
        mood_ct_update();
        mhd_set_candidate_B(U_new_fv,B_new_cv);
        mhd_detection_vars(U_new_fv,det_new);
        mhd_NAD(det_new,det_old,troubles,cfg.nad_tolerance);
        mhd_PAD(U_new_fv,troubles);
        return update_cascade(troubles,cascade,2);
    }

    void mood_commit(){
        bool az=cfg.active[_z_];
        mood_assemble();
        mood_fluid_update(true);
        mood_ct_update();
        if(cfg.floor_cons) mhd_floor_cv(U_cv,B_new_cv);
        transform_cv_to_sp(U_cv,U_sp);
        transform_a_to_b_2d(Bxf,Bx_fp_x,TB_x,cv_to_sp,_x_);
        transform_a_to_b_2d(Byf,By_fp_y,TB_y,cv_to_sp,_y_);
        if(az) transform_a_to_b_2d(Bzf,Bz_fp_z,TB_z,cv_to_sp,_z_);
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
