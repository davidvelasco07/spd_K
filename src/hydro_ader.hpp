using namespace std;

struct Hydro_ader : public PhysicsModule{
    int n_ader;
    int nvar;
    int n_stages;     //RK stages (1 for ADER); also needed by Hydro_mesh
    double rk_a[3];   //per-stage convex weights of the SSP combination
    double nu;
    double beta;
    bool viscosity;   //viscous terms active (enabled at runtime when nu>0)

    //Stored so the task methods (which receive only Driver*) can reach the
    //comm handle and per-direction geometry.
    CommHelper comm_;
    dimension Xdim_;
    dimension Ydim_;
    dimension Zdim_;

    Vector xt;   //temporal nodes/weights: GL (p+1) for ADER, {1} for RK stages
    Vector wt;
    Vector xx;   //spatial GL quadrature (control-volume averages of the ICs)
    Vector wx;

    Matrix sp_to_fp;
    Matrix fp_to_sp;
    Matrix dfp_to_sp;
    Matrix sp_to_cv;
    Matrix cv_to_sp;
    Matrix fp_to_cv;
    Matrix ader;
    Matrix invader;

    SD_Solution W_sp;
    SD_Solution U_sp;
    SD_Solution W_cv;
    SD_Solution U_cv;
    SD_Solution T_sweep; //scratch for directional-sweep transforms
    SD_Solution U0_sp;   //step-start state for the SSP-RK convex combination

    SD_Solution U_ader_sp;
    //Per-direction arrays are always declared; inactive directions hold
    //size-1 views and are skipped at runtime through cfg.active[]
    SD_Solution U_ader_fp_x;
    SD_Solution F_ader_fp_x;
    SD_Solution dUx_sp;
    Boundaries BC_fp_x;
    SD_Solution U_ader_fp_y;
    SD_Solution F_ader_fp_y;
    SD_Solution dUy_sp;
    Boundaries BC_fp_y;
    SD_Solution U_ader_fp_z;
    SD_Solution F_ader_fp_z;
    SD_Solution dUz_sp;
    Boundaries BC_fp_z;

    //FV update + fallback scheme (allocated only when cfg.fallback is set)
    FV_Solution U_new;
    FV_Solution W_new;
    FV_Solution U_old;
    FV_Solution W_old;
    FV_Solution troubles;  //per-variable trouble flags, block-local
    FV_Solution cascade;   //per-cell fallback level, the only flag data exchanged
    FV_Solution theta;     //fractional blend factor per cell
    FV_Solution theta_tmp;
    FV_Solution F_x;
    FV_Solution alpha_x;
    FV_Boundaries BC_x;
    SD_Solution T_fp_x; //scratch for face_integral sweeps
    FV_Solution F_y;
    FV_Solution alpha_y;
    FV_Boundaries BC_y;
    SD_Solution T_fp_y;
    FV_Solution F_z;
    FV_Solution alpha_z;
    FV_Boundaries BC_z;
    SD_Solution T_fp_z;

    //Non-null when this block's arrays are slices of a mesh-wide pack
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

    Hydro_ader(
        CommHelper comm,
        int p,
        dimension X_dim,
        dimension Y_dim,
        dimension Z_dim,
        double* x,
        double* w,
        double* x_sp,
        double* x_fp,
        double _nu,
        double _beta,
        bool standalone=true, //false when driven as one block of a mesh
        //When a pack is supplied this block's arrays are slices of it (see
        //BlockPack), so mesh-level kernels can span every block in one launch.
        BlockPack* pack=nullptr,
        int pack_ib=0
    ) : comm_(comm), Xdim_(X_dim), Ydim_(Y_dim), Zdim_(Z_dim),
        pack_(pack), pib_(pack_ib) {
        //Number of variables: rho, vx, vy, vz, e + FV bookkeeping slot
        nvar = NVAR;
        n_output = 0;
        n_step = 0;
        t=0;
        nu = _nu;
        beta = _beta;
        viscosity = nu > 0.0;
        Kokkos::resize(xx,p+1);
        Kokkos::resize(wx,p+1);
        {
            Vector_h xx_h = setup_mirror(xx);
            Vector_h wx_h = setup_mirror(wx);
            gauss_legendre(0.0, 1.0, p+1, xx_h.data(), wx_h.data());
            setup_push(xx, xx_h);
            setup_push(wx, wx_h);
        }

        //ADER carries p+1 temporal quadrature slices at the same GL nodes as
        //the spatial quadrature; an SSP-RK stage is a single slice advanced
        //by a full forward-Euler step (weight 1) and combined convexly with U0
        if(cfg.integrator==_integrator_rk_){
            n_ader = 1;
            n_stages = ssp_rk_coefficients(cfg.rk_order,rk_a);
            Kokkos::resize(xt,1);
            Kokkos::resize(wt,1);
            Kokkos::deep_copy(xt,0.0);
            Kokkos::deep_copy(wt,1.0);
        }
        else{
            n_ader = p+1;
            n_stages = 1;
            xt = xx;
            wt = wx;
        }

        //////////////
        //Matrices to perform tensorial transformations
        //////////////
        Kokkos::resize(sp_to_fp,p+2,p+1);
        Kokkos::resize(fp_to_sp,p+1,p+2);
        Kokkos::resize(dfp_to_sp,p+1,p+2);
        Kokkos::resize(sp_to_cv,p+1,p+1);
        Kokkos::resize(cv_to_sp,p+1,p+1);
        Kokkos::resize(fp_to_cv,p+1,p+2);

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
            //The ADER (temporal) matrices need the p+1 GL nodes; RK never uses them
            if(cfg.integrator==_integrator_ader_){
                Kokkos::resize(ader,p+1,p+1);
                Kokkos::resize(invader,p+1,p+1);
                Matrix_h ader_h = setup_mirror(ader);
                Matrix_h invader_h = setup_mirror(invader);
                Vector_h xt_h = setup_mirror(xt);
                Vector_h wt_h = setup_mirror(wt);
                setup_pull(xt, xt_h);
                setup_pull(wt, wt_h);
                ader_matrix(ader_h, xt_h, wt_h, p+1);
                inverse(ader_h, invader_h, p+1);
                setup_push(ader, ader_h);
                setup_push(invader, invader_h);
            }
        }

        alloc(W_sp,"W_sp",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        alloc(U_sp,"U_sp",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        alloc(W_cv,"W_cv",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        alloc(U_cv,"U_cv",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        alloc(T_sweep,"T_sweep",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);

        alloc(U_ader_sp,"U_ader_sp",n_ader,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        if(cfg.integrator==_integrator_rk_)
            alloc(U0_sp,"U0_sp",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);

        alloc(U_ader_fp_x,"U_ader_fp_x",n_ader,nvar,Z_dim,Y_dim,X_dim,0,0,cfg.active[_x_]);
        alloc(F_ader_fp_x,"F_ader_fp_x",n_ader,nvar,Z_dim,Y_dim,X_dim,0,0,cfg.active[_x_]);
        alloc(dUx_sp,"dUx_sp",n_ader,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        BC_fp_x.init(X_dim,cfg.bc[_x_],n_ader,nvar,Z_dim.N_total,Y_dim.N_total,1,Z_dim.n_sp,Y_dim.n_sp,1);
        alloc(U_ader_fp_y,"U_ader_fp_y",n_ader,nvar,Z_dim,Y_dim,X_dim,0,cfg.active[_y_],0);
        alloc(F_ader_fp_y,"F_ader_fp_y",n_ader,nvar,Z_dim,Y_dim,X_dim,0,cfg.active[_y_],0);
        alloc(dUy_sp,"dUy_sp",n_ader,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        BC_fp_y.init(Y_dim,cfg.bc[_y_],n_ader,nvar,Z_dim.N_total,1,X_dim.N_total,Z_dim.n_sp,1,X_dim.n_sp);
        alloc(U_ader_fp_z,"U_ader_fp_z",n_ader,nvar,Z_dim,Y_dim,X_dim,cfg.active[_z_],0,0);
        alloc(F_ader_fp_z,"F_ader_fp_z",n_ader,nvar,Z_dim,Y_dim,X_dim,cfg.active[_z_],0,0);
        alloc(dUz_sp,"dUz_sp",n_ader,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        BC_fp_z.init(Z_dim,cfg.bc[_z_],n_ader,nvar,1,Y_dim.N_total,X_dim.N_total,1,Y_dim.n_sp,X_dim.n_sp);

        if(cfg.fallback){
            alloc(U_new,"U_new",nvar,Z_dim,Y_dim,X_dim,0,0,0);
            alloc(W_new,"W_new",nvar,Z_dim,Y_dim,X_dim,0,0,0);
            alloc(U_old,"U_old",nvar,Z_dim,Y_dim,X_dim,0,0,0);
            alloc(W_old,"W_old",nvar,Z_dim,Y_dim,X_dim,0,0,0);
            alloc(troubles,"troubles",nvar,Z_dim,Y_dim,X_dim,0,0,0);
            alloc(cascade,"cascade",1,Z_dim,Y_dim,X_dim,0,0,0);
            alloc(theta,"theta",1,Z_dim,Y_dim,X_dim,0,0,0);
            alloc(theta_tmp,"theta_tmp",1,Z_dim,Y_dim,X_dim,0,0,0);
            alloc(F_x,"F_x",nvar,Z_dim,Y_dim,X_dim,0,0,cfg.active[_x_]);
            alloc(alpha_x,"alpha_x",nvar,Z_dim,Y_dim,X_dim,0,0,0);
            BC_x.init(X_dim,cfg.bc[_x_],nvar,Z_dim.fv_ncells,Y_dim.fv_ncells,nGHx);
            alloc(T_fp_x,"T_fp_x",1,nvar,Z_dim,Y_dim,X_dim,0,0,cfg.active[_x_]);
            alloc(F_y,"F_y",nvar,Z_dim,Y_dim,X_dim,0,cfg.active[_y_],0);
            alloc(alpha_y,"alpha_y",nvar,Z_dim,Y_dim,X_dim,0,0,0);
            BC_y.init(Y_dim,cfg.bc[_y_],nvar,Z_dim.fv_ncells,nGHy,X_dim.fv_ncells);
            alloc(T_fp_y,"T_fp_y",1,nvar,Z_dim,Y_dim,X_dim,0,cfg.active[_y_],0);
            alloc(F_z,"F_z",nvar,Z_dim,Y_dim,X_dim,cfg.active[_z_],0,0);
            alloc(alpha_z,"alpha_z",nvar,Z_dim,Y_dim,X_dim,0,0,0);
            BC_z.init(Z_dim,cfg.bc[_z_],nvar,nGHz,Y_dim.fv_ncells,X_dim.fv_ncells);
            alloc(T_fp_z,"T_fp_z",1,nvar,Z_dim,Y_dim,X_dim,cfg.active[_z_],0,0);
        }

        ////////////////////////
        //Initial Conditions:
        ////////////////////////
        Initialize(W_cv,X_dim.sd_faces,Y_dim.sd_faces,Z_dim.sd_faces,xx,wx);
        #ifdef PERTURB_IC
        //Diagnostic: 1-ulp perturbation to measure the solver's intrinsic
        //round-off amplification, independent of any code change
        {
            SD_Solution W = W_cv;
            int Nx=W.Nx, Ny=W.Ny, Nz=W.Nz, px=W.nx, py=W.ny, pz=W.nz;
            int nvar=W.n_var;
            sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
                for(int var=0; var<nvar; var++)
                    W.Vector(0,var,k,j,i,kk,jj,ii) *= (1.0 + 1e-15);
            });
        }
        #endif
        transform_cv_to_sp(W_cv,W_sp);
        compute_conservatives(W_sp,U_sp);

        Dt = compute_dt(W_cv,X_dim.h,Y_dim.h,Z_dim.h,nu);
        if(standalone){
            if(Master)
                cout<<"dx = "<<X_dim.h<<" dt = "<<Dt<<endl;
            if(cfg.outputs)
                Write_outputs();
        }
    }

    /////////////////////////////////////////////////////////////////////
    // Tasklist interface (PhysicsModule): the monolithic ADER_step/RK_step
    // are broken into task methods registered into the driver's phases. The
    // per-stage chain is CopyCons -> Advance -> Combine; Advance folds the
    // ADER Picard predictor (degenerating to a single flux solve under RK).
    /////////////////////////////////////////////////////////////////////
    void AssembleTasks(Driver* d) override {
        TaskID none(0);
        auto bti = d->tl_map["before_timeintegrator"];
        auto stg = d->tl_map["stagen"];
        auto ati = d->tl_map["after_timeintegrator"];
        //SSP-RK saves the step-start state once per cycle for the convex combine
        if(cfg.integrator==_integrator_rk_)
            bti->AddTask(&Hydro_ader::TaskSaveState, this, none);
        TaskID copy = stg->AddTask(&Hydro_ader::TaskCopyCons, this, none);
        TaskID adv  = stg->AddTask(&Hydro_ader::TaskAdvance,  this, copy);
        stg->AddTask(&Hydro_ader::TaskCombine, this, adv);
        //cons->prim + control-volume averages (consumed by outputs and the CFL)
        ati->AddTask(&Hydro_ader::TaskConsToPrim, this, none);
    }

    TaskStatus TaskSaveState(Driver* d, int stage){
        Kokkos::deep_copy(U0_sp.Vector,U_sp.Vector);
        return TaskStatus::complete;
    }

    TaskStatus TaskCopyCons(Driver* d, int stage){
        copy_ader(U_sp,U_ader_sp);
        return TaskStatus::complete;
    }

    TaskStatus TaskAdvance(Driver* d, int stage){
        //ADER: Picard predictor over the p+1 temporal slices (the final slice
        //leaves the fluxes ready for the corrector). RK: n_ader==1, a single
        //forward-Euler flux solve.
        for(int ader=0; ader<n_ader; ader++){
            Solve_fluxes(comm_,Xdim_,Ydim_,Zdim_);
            if(ader<n_ader-1)
                Update_prediction(Xdim_.h,Ydim_.h,Zdim_.h);
        }
        if(cfg.fallback)
            FV_Update_solution(comm_,Xdim_,Ydim_,Zdim_);
        else
            Update_solution(Xdim_.h,Ydim_.h,Zdim_.h);
        return TaskStatus::complete;
    }

    TaskStatus TaskCombine(Driver* d, int stage){
        //SSP convex combination U <- a*U0 + (1-a)*U (RK only; stage is 1-based)
        if(cfg.integrator==_integrator_rk_ && d->rk_a[stage-1]>0)
            combine_solution(U_sp,U0_sp,d->rk_a[stage-1]);
        return TaskStatus::complete;
    }

    TaskStatus TaskConsToPrim(Driver* d, int stage){
        compute_primitives(U_sp,W_sp);
        transform_sp_to_cv(W_sp,W_cv);
        return TaskStatus::complete;
    }

    double ComputeDt() override {
        return compute_dt(W_cv,Xdim_.h,Ydim_.h,Zdim_.h,nu);
    }

    void WriteOutputs() override { Write_outputs(); }

    void time_evolution(
        CommHelper comm,
        double t_end,
        double dt_output,
        dimension X_dim,
        dimension Y_dim,
        dimension Z_dim){

        dt=Dt;
        double t_output=dt_output;

        while(t<t_end){
            if(cfg.integrator==_integrator_rk_)
                RK_step(comm,X_dim,Y_dim,Z_dim);
            else
                ADER_step(comm,X_dim,Y_dim,Z_dim);
            compute_primitives(U_sp,W_sp);
            transform_sp_to_cv(W_sp,W_cv);
            t+=dt;
            n_step++;
            dt=compute_dt(W_cv,X_dim.h,Y_dim.h,Z_dim.h,nu);

            //A diverged state gives a NaN dt, which makes t NaN and turns
            //t<t_end false, so the run would exit reporting success while
            //dumping garbage. Fail loudly instead.
            if(!std::isfinite(dt)){
                if(Master)
                    cout<<endl<<"ERROR: non-finite dt at step "<<n_step
                        <<" (t = "<<t<<"), solution has diverged"<<endl;
                Kokkos::finalize();
                exit(1);
            }

            //Outputs
            if(Master) cout<<".";
            if(t>=t_output){
                t_output=t+dt_output;
                Write_outputs();
            }
            if(t+dt>t_output){
                dt=t_output-t;
            }
        }
        cout<<endl;
    }

    //One evaluation of the spatial operator on the n_ader slices of the
    //U_ader_* arrays, split into phases around the ghost exchange so a
    //multi-block driver can substitute block-to-block exchanges for the
    //single-block halo exchange.
    void Fluxes_pre(){
        Interpolate_to_fp();
        Compute_Fluxes();
    }

    void Solve_fluxes(CommHelper comm, dimension X_dim, dimension Y_dim, dimension Z_dim){
        Fluxes_pre();
        apply_boundaries(comm);
        Riemann_Solver();

        if(viscosity){
            Viscosity(X_dim.h,Y_dim.h,Z_dim.h);
            apply_boundaries(comm);
            Rusanov_Solver();
        }
    }

    void ADER_step(CommHelper comm, dimension X_dim, dimension Y_dim, dimension Z_dim){
        ////Initialize ADER time slices
        copy_ader(U_sp,U_ader_sp);

        //Picard iteration
        for(int ader=0;ader<n_ader;ader++){
            Solve_fluxes(comm,X_dim,Y_dim,Z_dim);
            if(ader<n_ader-1)
                Update_prediction(X_dim.h,Y_dim.h,Z_dim.h);
        }
        if(cfg.fallback)
            FV_Update_solution(comm,X_dim,Y_dim,Z_dim);
        else
            Update_solution(X_dim.h,Y_dim.h,Z_dim.h);
    }

    //SSP-RK step (Shu-Osher form): each stage is a full forward-Euler step
    //(reusing the ADER machinery with a single time slice, so the per-stage
    //fallback detection/blending applies unchanged) followed by the convex
    //combination U <- a*U0 + (1-a)*U. Convexity preserves the admissibility
    //enforced per stage, and every contribution stays in flux form, so the
    //blended update remains exactly conservative.
    void RK_step(CommHelper comm, dimension X_dim, dimension Y_dim, dimension Z_dim){
        Kokkos::deep_copy(U0_sp.Vector,U_sp.Vector);
        for(int s=0;s<n_stages;s++){
            copy_ader(U_sp,U_ader_sp);
            Solve_fluxes(comm,X_dim,Y_dim,Z_dim);
            if(cfg.fallback)
                FV_Update_solution(comm,X_dim,Y_dim,Z_dim);
            else
                Update_solution(X_dim.h,Y_dim.h,Z_dim.h);
            if(rk_a[s]>0)
                combine_solution(U_sp,U0_sp,rk_a[s]);
        }
    }

    void transform_cv_to_sp(SD_Solution U_cv, SD_Solution U_sp){
        #ifdef REF_TRANSFORMS
        transform_a_to_b_ref(U_cv, U_sp, cv_to_sp, cv_to_sp, cv_to_sp);
        #else
        transform_a_to_b(U_cv, U_sp, T_sweep, cv_to_sp);
        #endif
    }

    void transform_sp_to_cv(SD_Solution U_sp, SD_Solution U_cv){
        #ifdef REF_TRANSFORMS
        transform_a_to_b_ref(U_sp, U_cv, sp_to_cv, sp_to_cv, sp_to_cv);
        #else
        transform_a_to_b(U_sp, U_cv, T_sweep, sp_to_cv);
        #endif
    }

    void Interpolate_to_fp(){
        // Interpolate to flux points
        if(cfg.active[_x_])
            transform_a_to_b_1d(U_ader_sp,U_ader_fp_x,sp_to_fp,_x_);
        if(cfg.active[_y_])
            transform_a_to_b_1d(U_ader_sp,U_ader_fp_y,sp_to_fp,_y_);
        if(cfg.active[_z_])
            transform_a_to_b_1d(U_ader_sp,U_ader_fp_z,sp_to_fp,_z_);
    }

    void Compute_Fluxes(){
        //Compute fluxes
        if(cfg.active[_x_])
            compute_fluxes(U_ader_fp_x,F_ader_fp_x,_vx_,_vy_,_vz_);
        if(cfg.active[_y_])
            compute_fluxes(U_ader_fp_y,F_ader_fp_y,_vy_,_vz_,_vx_);
        if(cfg.active[_z_])
            compute_fluxes(U_ader_fp_z,F_ader_fp_z,_vz_,_vx_,_vy_);
    }

    void apply_boundaries(CommHelper comm){
        //Communications are done sequentially in different directions
        //to ensure that corners are properly communicated
        if(cfg.active[_x_])
            boundaries(comm,BC_fp_x,U_ader_fp_x);
        if(cfg.active[_y_])
            boundaries(comm,BC_fp_y,U_ader_fp_y);
        if(cfg.active[_z_])
            boundaries(comm,BC_fp_z,U_ader_fp_z);
    }

    void Riemann_Solver(){
        if(cfg.active[_x_])
            sd_riemann_solver(U_ader_fp_x,F_ader_fp_x,_vx_,_vy_,_vz_,_x_,viscosity);
        if(cfg.active[_y_])
            sd_riemann_solver(U_ader_fp_y,F_ader_fp_y,_vy_,_vz_,_vx_,_y_,viscosity);
        if(cfg.active[_z_])
            sd_riemann_solver(U_ader_fp_z,F_ader_fp_z,_vz_,_vx_,_vy_,_z_,viscosity);
    }

    void Viscosity(double dx, double dy, double dz){
        if(cfg.active[_x_])
            compute_gradient(U_ader_fp_x, dUx_sp, dx, dfp_to_sp, _x_);
        if(cfg.active[_y_])
            compute_gradient(U_ader_fp_y, dUy_sp, dy, dfp_to_sp, _y_);
        if(cfg.active[_z_])
            compute_gradient(U_ader_fp_z, dUz_sp, dz, dfp_to_sp, _z_);
        if(cfg.active[_x_])
            compute_viscous_flux(U_ader_fp_x,dUx_sp,_vx_,dUy_sp,_vy_,dUz_sp,_vz_,sp_to_fp,nu,beta,_x_);
        if(cfg.active[_y_])
            compute_viscous_flux(U_ader_fp_y,dUy_sp,_vy_,dUz_sp,_vz_,dUx_sp,_vx_,sp_to_fp,nu,beta,_y_);
        if(cfg.active[_z_])
            compute_viscous_flux(U_ader_fp_z,dUz_sp,_vz_,dUx_sp,_vx_,dUy_sp,_vy_,sp_to_fp,nu,beta,_z_);
    }

    void Rusanov_Solver(){
        if(cfg.active[_x_])
            sd_rusanov_solver(U_ader_fp_x,F_ader_fp_x,_x_);
        if(cfg.active[_y_])
            sd_rusanov_solver(U_ader_fp_y,F_ader_fp_y,_y_);
        if(cfg.active[_z_])
            sd_rusanov_solver(U_ader_fp_z,F_ader_fp_z,_z_);
    }

    void Update_prediction(const double dx,const double dy,const double dz){
        update_prediction(U_sp,U_ader_sp,
            F_ader_fp_x,F_ader_fp_y,F_ader_fp_z,
            dfp_to_sp,invader,wt,dx,dy,dz,dt);
    }

    void Update_solution(double dx, double dy, double dz){
        update_solution(U_sp,U_ader_sp,
            F_ader_fp_x,F_ader_fp_y,F_ader_fp_z,
            dfp_to_sp,wt,dx,dy,dz,dt);
    }

    void Write_outputs(){
        if(Master)
            cout<<endl<<"OUTPUT "<<n_output<<endl;
        if(Master){
            std::ofstream f(output_folder()+"mass.txt",
                            n_output==0 ? std::ios::trunc : std::ios::app);
            f<<std::setprecision(17)<<t<<" "
             <<fv_mass(W_cv,Xdim_,Ydim_,Zdim_)<<endl;
        }
        if(cfg.fallback){
            Write(troubles,n_output);
            Write(cascade,n_output);
        }
        Write(F_ader_fp_x,n_output);
        Write(W_cv,n_output++);
    }

    void copy_ader(SD_Solution U, SD_Solution U_ader){
        ////indices: t,nvar,N,n
        int Nx = U.Nx;
        int Ny = U.Ny;
        int Nz = U.Nz;
        int px = U.nx;
        int py = U.ny;
        int pz = U.nz;
        int nvar = U.n_var;
        int nader = U_ader.n_ader;
        int nb = U.nb;
        sd_for_cells_b(nb,Nz,Ny,Nx,pz,py,px,
            KOKKOS_LAMBDA(int b, int k, int j, int i, int kk, int jj, int ii){
            BOFF(nader);
            for(int t_id=0; t_id<nader; t_id++){
            for(int var=0; var<nvar; var++){
            U_ader.Vector(boff+t_id,var,k,j,i,kk,jj,ii) = U.Vector(b,var,k,j,i,kk,jj,ii);
            }}
        }, "copy_ader");
    }

    void Integrate_fluxes(int ader){
        #ifdef REF_TRANSFORMS
        if(cfg.active[_x_])
            face_integral_ref(F_ader_fp_x, F_x, sp_to_cv, ader, _x_);
        if(cfg.active[_y_])
            face_integral_ref(F_ader_fp_y, F_y, sp_to_cv, ader, _y_);
        if(cfg.active[_z_])
            face_integral_ref(F_ader_fp_z, F_z, sp_to_cv, ader, _z_);
        #else
        if(cfg.active[_x_])
            face_integral(F_ader_fp_x, F_x, T_fp_x, sp_to_cv, ader, _x_);
        if(cfg.active[_y_])
            face_integral(F_ader_fp_y, F_y, T_fp_y, sp_to_cv, ader, _y_);
        if(cfg.active[_z_])
            face_integral(F_ader_fp_z, F_z, T_fp_z, sp_to_cv, ader, _z_);
        #endif
    }

    #ifdef DEBUG_MASS
    double fv_mass_cells(FV_Solution U, dimension X_dim, dimension Y_dim, dimension Z_dim){
        //Total mass of the density over active FV cells
        int Nx=U.Nx, Ny=U.Ny, Nz=U.Nz;
        Vector fx = X_dim.fv_faces;
        Vector fy = Y_dim.fv_faces;
        Vector fz = Z_dim.fv_faces;
        bool ay=cfg.active[_y_], az=cfg.active[_z_];
        double mass = fv_sum_cells_ngh2(Nz,Ny,Nx,
            KOKKOS_LAMBDA(int k,int j,int i,double& sum){
                double V = fx(i+1)-fx(i);
                if(ay) V *= fy(j+1)-fy(j);
                if(az) V *= fz(k+1)-fz(k);
                sum += U.Vector(0,k,j,i)*V;
            });
        return mass;
    }
    #endif

    double fv_mass(SD_Solution U, dimension X_dim, dimension Y_dim, dimension Z_dim){
        //Total mass of the cv-average density over active cells
        int Nx=U.Nx, Ny=U.Ny, Nz=U.Nz, px=U.nx, py=U.ny, pz=U.nz;
        int qx=px, qy=py, qz=pz;
        Vector fx = X_dim.fv_faces;
        Vector fy = Y_dim.fv_faces;
        Vector fz = Z_dim.fv_faces;
        bool ay=cfg.active[_y_], az=cfg.active[_z_];
        GHOST_LOCALS;
        double mass = sd_sum_active_cells(Nz,Ny,Nx,pz,py,px,
            KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii,double& sum){
                double V = fx(I+1)-fx(I);
                if(ay) V *= fy(J+1)-fy(J);
                if(az) V *= fz(K+1)-fz(K);
                sum += U.Vector(0,0,k,j,i,kk,jj,ii)*V;
            });
        return mass;
    }

    //FV update phases: the per-node body is split at every ghost exchange
    //(U_old/U_new, troubles, theta) so a multi-block driver can run each
    //phase over all blocks and substitute block-to-block exchanges. The
    //single-block wrapper below preserves the exact original sequence.

    void FV_begin(){
        transform_sp_to_cv(U_sp,U_cv);
    }

    //Tentative high-order update: SD face fluxes -> FV faces -> candidate
    void FV_flux_update(int ader, dimension X_dim, dimension Y_dim, dimension Z_dim){
        Integrate_fluxes(ader);
        fv_update_solution(U_new,U_old,U_cv,
            F_x,X_dim.fv_faces,
            F_y,Y_dim.fv_faces,
            F_z,Z_dim.fv_faces,
            wt,ader,dt,0);
    }

    //Requires ghosted U_old/U_new
    void FV_detect(dimension X_dim, dimension Y_dim, dimension Z_dim){
        //W_old feeds the MUSCL reconstruction, so it is needed either way.
        compute_primitives(U_old,W_old);
        //Pure MUSCL blends in the fallback everywhere, so the flags are never
        //read and the whole detection stencil can be skipped.
        if(cfg.muscl_only) return;
        compute_primitives(U_new,W_new);
        //Following the reference implementation, only density and
        //pressure enter the NAD/SED checks (uniform or zero fields,
        //like transverse velocities, have no meaningful relative band)
        detect_troubles(W_new,W_old,troubles,cascade,
            alpha_x,alpha_y,alpha_z,
            X_dim,Y_dim,Z_dim,1,(1<<_d_)|(1<<_p_));
    }

    //Fractional blend factor: spread the trouble flags to the neighborhood
    //(0.75/0.5/0.375 weights + 0.25 ring), or use the raw flags when
    //blending is disabled. Requires ghosted trouble flags.
    void FV_theta(){
        //Fills the ghosts too, so the usual theta exchange is a no-op here.
        if(cfg.muscl_only){
            Kokkos::deep_copy(theta.Vector, 1.0);
            return;
        }
        if(cfg.blending){
            apply_blending(cascade,theta_tmp);
            blending_ring(theta_tmp,theta);
        }
        else
            theta_from_cascade(cascade,theta);
    }

    //Blend MUSCL fluxes into the troubled faces and redo the update.
    //Requires ghosted theta (identical blended fluxes on both sides of
    //every face, so the correction stays exactly conservative).
    void FV_blend(int ader, dimension X_dim, dimension Y_dim, dimension Z_dim){
        fallback_fluxes(W_old,theta,
            X_dim.fv_centers,X_dim.fv_faces,F_x,
            Y_dim.fv_centers,Y_dim.fv_faces,F_y,
            Z_dim.fv_centers,Z_dim.fv_faces,F_z,
            ader,wt,dt);
    }

    void FV_commit(int ader, dimension X_dim, dimension Y_dim, dimension Z_dim){
        fv_update_solution(U_new,U_old,U_cv,
            F_x,X_dim.fv_faces,
            F_y,Y_dim.fv_faces,
            F_z,Z_dim.fv_faces,
            wt,ader,dt,1);
    }

    //Split so a multi-block driver can reconcile the blended fluxes across
    //coarse-fine faces before they are committed to the solution.
    void FV_apply(int ader, dimension X_dim, dimension Y_dim, dimension Z_dim){
        FV_blend(ader,X_dim,Y_dim,Z_dim);
        FV_commit(ader,X_dim,Y_dim,Z_dim);
    }

    void FV_end(){
        transform_cv_to_sp(U_cv,U_sp);
    }

    void FV_Update_solution(CommHelper comm, dimension X_dim,dimension Y_dim,dimension Z_dim){
        FV_begin();
        #ifdef DEBUG_MASS
        printf("step %d mass in : %.15e\n", n_step, fv_mass(U_cv,X_dim,Y_dim,Z_dim));
        #endif
        for(int ader=0;ader<n_ader;ader++){
            FV_flux_update(ader,X_dim,Y_dim,Z_dim);
            #ifdef DEBUG_MASS
            printf("  ader %d U_new after SD-flux update : %.15e\n", ader, fv_mass_cells(U_new,X_dim,Y_dim,Z_dim));
            #endif
            apply_fv_boundaries(comm,U_old);
            apply_fv_boundaries(comm,U_new);
            FV_detect(X_dim,Y_dim,Z_dim);
            //Ghost levels must be periodic images so that the blending
            //stencils near the domain boundary see the same data as their
            //periodic partners. Only the pooled level is read downstream, so
            //this is a one-component halo rather than the whole flag array.
            if(!cfg.muscl_only) apply_fv_boundaries(comm,cascade);
            FV_theta();
            //Ghost thetas must also be exact periodic images so the two
            //domain boundary faces of each direction receive identical
            //blended fluxes (exact conservation)
            if(!cfg.muscl_only) apply_fv_boundaries(comm,theta);
            #ifdef DEBUG_MASS
            if(t==0 && ader==0) Write(F_x,899);
            #endif
            FV_apply(ader,X_dim,Y_dim,Z_dim);
            #ifdef DEBUG_MASS
            printf("  ader %d U_new after fallback update: %.15e\n", ader, fv_mass_cells(U_new,X_dim,Y_dim,Z_dim));
            if(t==0 && ader==0){
                Write(F_x,900);
                Write(F_y,901);
                Write(F_z,902);
                Write(troubles,903);
                Write(U_old,904);
                Write(U_new,905);
            }
            printf("step %d ader %d mass: %.15e\n", n_step, ader, fv_mass(U_cv,X_dim,Y_dim,Z_dim));
            #endif
        }
        FV_end();
        #ifdef DEBUG_MASS
        transform_sp_to_cv(U_sp,U_cv);
        printf("step %d roundtrip : %.15e\n", n_step, fv_mass(U_cv,X_dim,Y_dim,Z_dim));
        #endif
    }

    void apply_fv_boundaries(CommHelper comm, FV_Solution U){
        //Communications are done sequentially in different directions
        //to ensure that corners are properly communicated
        if(cfg.active[_x_])
            boundaries(comm,BC_x,U,_center_,0);
        if(cfg.active[_y_])
            boundaries(comm,BC_y,U,_center_,0);
        if(cfg.active[_z_])
            boundaries(comm,BC_z,U,_center_,0);
    }
};
