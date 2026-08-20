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
    FV_Solution flagged;   //per-cell pooled trouble flag, the only flag data exchanged
    FV_Solution theta;     //fractional blend factor per cell
    FV_Solution theta_tmp;
    //MOOD cascade (cfg.mood_cascade only): monotonic per-cell level plus the
    //per-level face fluxes it selects between. Level 0 is F_x/F_y/F_z, which
    //the assembly overwrites in place.
    FV_Solution cascade;
    FV_Solution F1_x, F1_y, F1_z;
    FV_Solution F2_x, F2_y, F2_z;
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
        int pack_ib=0,
        //One per-run operator set to alias instead of rebuilding (SDOperators).
        //Null means "build a private one", which keeps the standalone path and
        //the unit tests working unchanged.
        const SDOperators* ops=nullptr,
        //A block rebuilt by a regrid has its whole state overwritten from the
        //snapshot immediately afterwards (transfer_from_snapshot either finds an
        //ancestor / children or hard-exits), so running the initial conditions
        //there is pure waste -- a full IC kernel per block per adapt.
        bool run_ic=true
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
        //Alias the one per-run operator set; build a private one only when no
        //mesh supplied it (standalone / unit tests). Either way the operators
        //come from build_sd_operators, which is the single source of truth.
        SDOperators local_ops;
        if(!ops){ build_sd_operators(local_ops, p, x_sp, x_fp); ops = &local_ops; }
        n_ader   = ops->n_ader;
        n_stages = ops->n_stages;
        for(int i=0;i<3;i++) rk_a[i] = ops->rk_a[i];
        xx = ops->xx;   wx = ops->wx;
        xt = ops->xt;   wt = ops->wt;
        sp_to_fp  = ops->sp_to_fp;
        fp_to_sp  = ops->fp_to_sp;
        dfp_to_sp = ops->dfp_to_sp;
        sp_to_cv  = ops->sp_to_cv;
        cv_to_sp  = ops->cv_to_sp;
        fp_to_cv  = ops->fp_to_cv;
        ader      = ops->ader;
        invader   = ops->invader;

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
            alloc(flagged,"flagged",1,Z_dim,Y_dim,X_dim,0,0,0);
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
            if(cfg.mood_cascade){
                alloc(cascade,"cascade",1,Z_dim,Y_dim,X_dim,0,0,0);
                alloc(F1_x,"F1_x",nvar,Z_dim,Y_dim,X_dim,0,0,cfg.active[_x_]);
                alloc(F2_x,"F2_x",nvar,Z_dim,Y_dim,X_dim,0,0,cfg.active[_x_]);
                alloc(F1_y,"F1_y",nvar,Z_dim,Y_dim,X_dim,0,cfg.active[_y_],0);
                alloc(F2_y,"F2_y",nvar,Z_dim,Y_dim,X_dim,0,cfg.active[_y_],0);
                alloc(F1_z,"F1_z",nvar,Z_dim,Y_dim,X_dim,cfg.active[_z_],0,0);
                alloc(F2_z,"F2_z",nvar,Z_dim,Y_dim,X_dim,cfg.active[_z_],0,0);
            }
        }

        ////////////////////////
        //Initial Conditions:
        ////////////////////////
        //Skipped for a block the regrid is rebuilding: transfer_from_snapshot
        //overwrites U_sp/W_sp for every surviving block (a block with neither a
        //snapshot ancestor nor a full set of children is a hard error there),
        //adapt() calls finish_block_ic() to re-derive the primitives, and it
        //ends with recompute_dt(), so nothing below survives the regrid.
        if(!run_ic){ Dt = 0.0; return; }
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
        compute_primitives(U_sp,W_sp);   //W_sp still feeds the AMR criteria
        cons_to_prim_cv();
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
            cons_to_prim_cv();
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
    void cons_to_prim_cv(){
        transform_sp_to_cv(U_sp, U_cv);
        compute_primitives(U_cv, W_cv);
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
            Write(flagged,n_output);
            if(cfg.mood_cascade) Write(cascade,n_output);
        }
        Write(F_ader_fp_x,n_output);
        Write(W_cv,n_output++);
    }

    //--------------------------------------------------------------------------
    //SYSTEM HOOKS. Mesh calls these by one name for both systems, so a
    //step-level task has no `if constexpr (is_hydro)` in it. That fork is how
    //every batching regression got in: the hydro side was updated, the MHD side
    //sat in the `else` of the same function, and nothing marked it as stale.
    //Add a hook here and in MHD_ader rather than a branch in Mesh.
    //--------------------------------------------------------------------------

    //Primitives over a whole pack.
    static void primitives_b(SD_Solution U, SD_Solution W){ compute_primitives(U,W); }

    //CFL dt over a whole pack. No MPI reduction here -- Mesh::ComputeDt owns
    //that for both systems.
    static double dt_b(SD_Solution W, Vector hx, Vector hy, Vector hz, double nu){
        return compute_dt_b(W,hx,hy,hz,nu);
    }

    //State the RK integrator saves and recombines, as (saved, live) pack-array
    //names. ONE list drives both TaskSaveState and TaskCombine, so the two
    //cannot disagree about what the state is -- which for MHD they nearly did,
    //each carrying its own four-line copy of the face-field list.
    static constexpr std::array<std::pair<const char*,const char*>,1> rk_state(){
        return {{{"U0_sp","U_sp"}}};
    }

    //Does the flux/EMF path read W_sp WITHIN a stage? MHD's compute_E does;
    //hydro builds its fluxes from U directly. A scheme property, not duplication.
    static constexpr bool prim_at_stage_start = false;

    //STATIC: it reads only its arguments' extents, so it is a system hook like
    //the ones above and Mesh can call it as Block::copy_ader for either system.
    static void copy_ader(SD_Solution U, SD_Solution U_ader){
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
        if(cfg.fv_only) return;
        compute_primitives(U_new,W_new);
        //Following the reference implementation, only density and
        //pressure enter the NAD/SED checks (uniform or zero fields,
        //like transverse velocities, have no meaningful relative band)
        detect_troubles(W_new,W_old,troubles,flagged,
            alpha_x,alpha_y,alpha_z,
            X_dim,Y_dim,Z_dim,1,(1<<_d_)|(1<<_p_));
    }

    //Fractional blend factor: spread the trouble flags to the neighborhood
    //(0.75/0.5/0.375 weights + 0.25 ring), or use the raw flags when
    //blending is disabled. Requires ghosted trouble flags.
    void FV_theta(){
        //Fills the ghosts too, so the usual theta exchange is a no-op here.
        if(cfg.fv_only){
            Kokkos::deep_copy(theta.Vector, 1.0);
            return;
        }
        if(cfg.blending){
            apply_blending(flagged,theta_tmp);
            blending_ring(theta_tmp,theta);
        }
        else
            theta_from_flagged(flagged,theta);
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

    //================================================================
    // MOOD cascade fallback (cfg.mood_cascade)
    //================================================================
    // Instead of blending the MUSCL flux into troubled faces by a fractional
    // weight, every cell carries a level and each face takes the flux of the
    // more demoted of its two cells. The levels only rise, so the assembly
    // overwrites the level-0 array in place and every face stays
    // single-valued -- the property the conservative update needs.
    //
    // The two lower-level flux sets are built once per ader step from the
    // halo'd old state; a revision then only re-assembles and re-tests.

    //Candidate update: same as FV_commit but leaves U_cv alone.
    void FV_candidate(int ader, dimension X_dim, dimension Y_dim, dimension Z_dim){
        fv_update_solution(U_new,U_old,U_cv,
            F_x,X_dim.fv_faces,
            F_y,Y_dim.fv_faces,
            F_z,Z_dim.fv_faces,
            wt,ader,dt,0);
    }

    //Requires ghosted U_old (W_old is derived from it).
    void FV_cascade_levels(int ader, dimension X_dim, dimension Y_dim, dimension Z_dim){
        level_fluxes(W_old,
            X_dim.fv_centers,X_dim.fv_faces,F1_x,
            Y_dim.fv_centers,Y_dim.fv_faces,F1_y,
            Z_dim.fv_centers,Z_dim.fv_faces,F1_z,
            ader,wt,dt,true);
        level_fluxes(W_old,
            X_dim.fv_centers,X_dim.fv_faces,F2_x,
            Y_dim.fv_centers,Y_dim.fv_faces,F2_y,
            Z_dim.fv_centers,Z_dim.fv_faces,F2_z,
            ader,wt,dt,false);
        Kokkos::deep_copy(cascade.Vector,0.0);
    }

    void FV_cascade_assemble(){
        assign_face_flux(F_x,F1_x,F2_x,cascade,_x_);
        if(cfg.active[_y_]) assign_face_flux(F_y,F1_y,F2_y,cascade,_y_);
        if(cfg.active[_z_]) assign_face_flux(F_z,F1_z,F2_z,cascade,_z_);
    }

    //A revision is split at the ghost exchange it needs, like the rest of the
    //FV phases: assemble and form the candidate, then (after the candidate has
    //been halo'd) re-detect and demote. The halo in the middle is not optional
    //-- SED reads W_new across a two-cell stencil, so a revision that reused
    //the previous candidate's ghosts would limit against stale data. MHD's
    //MOOD needs only one U halo for the whole loop because it runs no SED.
    void FV_cascade_candidate(int ader, dimension X_dim, dimension Y_dim, dimension Z_dim){
        FV_cascade_assemble();
        FV_candidate(ader,X_dim,Y_dim,Z_dim);
    }

    //Requires ghosted U_new. Returns the number of cells demoted, so the
    //caller can stop once a sweep changes nothing.
    int FV_cascade_detect(dimension X_dim, dimension Y_dim, dimension Z_dim){
        compute_primitives(U_new,W_new);
        detect_troubles(W_new,W_old,troubles,flagged,
            alpha_x,alpha_y,alpha_z,
            X_dim,Y_dim,Z_dim,1,(1<<_d_)|(1<<_p_));
        return update_cascade(flagged,cascade,2);
    }

    void FV_Update_solution_cascade(CommHelper comm, dimension X_dim, dimension Y_dim, dimension Z_dim){
        FV_begin();
        for(int ader=0;ader<n_ader;ader++){
            FV_flux_update(ader,X_dim,Y_dim,Z_dim);
            apply_fv_boundaries(comm,U_old);
            compute_primitives(U_old,W_old);
            FV_cascade_levels(ader,X_dim,Y_dim,Z_dim);
            for(int rev=0; rev<cfg.max_revs; rev++){
                FV_cascade_candidate(ader,X_dim,Y_dim,Z_dim);
                apply_fv_boundaries(comm,U_new);
                if(FV_cascade_detect(X_dim,Y_dim,Z_dim)==0) break;
                //A demotion has to be visible from the other side of every
                //face it touches, or the two sides would assemble different
                //fluxes and the update would stop conserving.
                apply_fv_boundaries(comm,cascade);
            }
            FV_cascade_assemble();
            FV_commit(ader,X_dim,Y_dim,Z_dim);
        }
        FV_end();
    }

    void FV_Update_solution(CommHelper comm, dimension X_dim,dimension Y_dim,dimension Z_dim){
        if(cfg.mood_cascade){
            FV_Update_solution_cascade(comm,X_dim,Y_dim,Z_dim);
            return;
        }
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
            if(!cfg.fv_only) apply_fv_boundaries(comm,flagged);
            FV_theta();
            //Ghost thetas must also be exact periodic images so the two
            //domain boundary faces of each direction receive identical
            //blended fluxes (exact conservation)
            if(!cfg.fv_only) apply_fv_boundaries(comm,theta);
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
