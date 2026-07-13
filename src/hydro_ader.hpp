using namespace std;

struct Hydro_ader{
    int n_output;
    int n_step;
    int n_ader;
    int nvar;
    int n_stages;     //RK stages (1 for ADER)
    double rk_a[3];   //per-stage convex weights of the SSP combination

    double t;
    double dt;
    double Dt;
    double nu;
    double beta;
    bool viscosity;   //viscous terms active (enabled at runtime when nu>0)

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
    FV_Solution troubles;
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
        double _beta
    ){
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
        gauss_legendre(0.0, 1.0, p+1, xx.data(), wx.data());

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

        lagrange_matrix(sp_to_fp, x_sp, x_fp, p+1, p+2);
        lagrange_matrix(fp_to_sp, x_fp, x_sp, p+2, p+1);
        lagrange_prime_matrix(dfp_to_sp, x_fp, x_sp, p+2, p+1);
        integral_matrix(sp_to_cv, x_fp, x_sp, p+1, p+1);
        integral_matrix(fp_to_cv, x_fp, x_fp, p+1, p+2);
        inverse(sp_to_cv, cv_to_sp, p+1);
        //The ADER (temporal) matrices need the p+1 GL nodes; RK never uses them
        if(cfg.integrator==_integrator_ader_){
            Kokkos::resize(ader,p+1,p+1);
            Kokkos::resize(invader,p+1,p+1);
            ader_matrix(ader, xt, wt, p+1);
            inverse(ader, invader, p+1);
        }

        W_sp.init("W_sp",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        U_sp.init("U_sp",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        W_cv.init("W_cv",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        U_cv.init("U_cv",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        T_sweep.init("T_sweep",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);

        U_ader_sp.init("U_ader_sp",n_ader,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        if(cfg.integrator==_integrator_rk_)
            U0_sp.init("U0_sp",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);

        U_ader_fp_x.init("U_ader_fp_x",n_ader,nvar,Z_dim,Y_dim,X_dim,0,0,cfg.active[_x_]);
        F_ader_fp_x.init("F_ader_fp_x",n_ader,nvar,Z_dim,Y_dim,X_dim,0,0,cfg.active[_x_]);
        dUx_sp.init("dUx_sp",n_ader,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        BC_fp_x.init(X_dim,cfg.bc[_x_],n_ader,nvar,Z_dim.N_total,Y_dim.N_total,1,Z_dim.n_sp,Y_dim.n_sp,1);
        U_ader_fp_y.init("U_ader_fp_y",n_ader,nvar,Z_dim,Y_dim,X_dim,0,cfg.active[_y_],0);
        F_ader_fp_y.init("F_ader_fp_y",n_ader,nvar,Z_dim,Y_dim,X_dim,0,cfg.active[_y_],0);
        dUy_sp.init("dUy_sp",n_ader,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        BC_fp_y.init(Y_dim,cfg.bc[_y_],n_ader,nvar,Z_dim.N_total,1,X_dim.N_total,Z_dim.n_sp,1,X_dim.n_sp);
        U_ader_fp_z.init("U_ader_fp_z",n_ader,nvar,Z_dim,Y_dim,X_dim,cfg.active[_z_],0,0);
        F_ader_fp_z.init("F_ader_fp_z",n_ader,nvar,Z_dim,Y_dim,X_dim,cfg.active[_z_],0,0);
        dUz_sp.init("dUz_sp",n_ader,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        BC_fp_z.init(Z_dim,cfg.bc[_z_],n_ader,nvar,1,Y_dim.N_total,X_dim.N_total,1,Y_dim.n_sp,X_dim.n_sp);

        if(cfg.fallback){
            U_new.init("U_new",nvar,Z_dim,Y_dim,X_dim,0,0,0);
            W_new.init("W_new",nvar,Z_dim,Y_dim,X_dim,0,0,0);
            U_old.init("U_old",nvar,Z_dim,Y_dim,X_dim,0,0,0);
            W_old.init("W_old",nvar,Z_dim,Y_dim,X_dim,0,0,0);
            troubles.init("troubles",nvar,Z_dim,Y_dim,X_dim,0,0,0);
            theta.init("theta",1,Z_dim,Y_dim,X_dim,0,0,0);
            theta_tmp.init("theta_tmp",1,Z_dim,Y_dim,X_dim,0,0,0);
            F_x.init("F_x",nvar,Z_dim,Y_dim,X_dim,0,0,cfg.active[_x_]);
            alpha_x.init("alpha_x",nvar,Z_dim,Y_dim,X_dim,0,0,0);
            BC_x.init(X_dim,cfg.bc[_x_],nvar,Z_dim.fv_ncells,Y_dim.fv_ncells,nGHx);
            T_fp_x.init("T_fp_x",1,nvar,Z_dim,Y_dim,X_dim,0,0,cfg.active[_x_]);
            F_y.init("F_y",nvar,Z_dim,Y_dim,X_dim,0,cfg.active[_y_],0);
            alpha_y.init("alpha_y",nvar,Z_dim,Y_dim,X_dim,0,0,0);
            BC_y.init(Y_dim,cfg.bc[_y_],nvar,Z_dim.fv_ncells,nGHy,X_dim.fv_ncells);
            T_fp_y.init("T_fp_y",1,nvar,Z_dim,Y_dim,X_dim,0,cfg.active[_y_],0);
            F_z.init("F_z",nvar,Z_dim,Y_dim,X_dim,cfg.active[_z_],0,0);
            alpha_z.init("alpha_z",nvar,Z_dim,Y_dim,X_dim,0,0,0);
            BC_z.init(Z_dim,cfg.bc[_z_],nvar,nGHz,Y_dim.fv_ncells,X_dim.fv_ncells);
            T_fp_z.init("T_fp_z",1,nvar,Z_dim,Y_dim,X_dim,cfg.active[_z_],0,0);
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
        if(Master)
            cout<<"dx = "<<X_dim.h<<" dt = "<<Dt<<endl;
        if(cfg.outputs)
            Write_outputs();
    }

    void time_evolution(
        CommHelper comm,
        double t_end,
        double dt_output,
        dimension X_dim,
        dimension Y_dim,
        dimension Z_dim){

        dt=Dt;
        double t_output=dt_output;

        //Time only the evolution loop: IC, setup and any host<->device
        //transfers before/after the run are excluded. Time spent writing
        //outputs (device->host copy + disk) is measured and subtracted.
        Kokkos::fence();
        Kokkos::Timer timer;
        double t_io = 0;
        int step0 = n_step;

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

            //Outputs
            if(Master) cout<<".";
            if(cfg.outputs){
                if(t>=t_output){
                    t_output=t+dt_output;
                    Kokkos::fence();
                    Kokkos::Timer io_timer;
                    Write_outputs();
                    t_io += io_timer.seconds();
                }
                if(t+dt>t_output){
                    dt=t_output-t;
                }
            }
        }
        Kokkos::fence();
        double t_evol = timer.seconds() - t_io;
        cout<<endl;
        if(Master){
            long long n_cells = (long long)(X_dim.N*X_dim.n_sp)
                              * (Y_dim.N*Y_dim.n_sp)
                              * (Z_dim.N*Z_dim.n_sp);
            int steps = n_step - step0;
            cout<<"evolution: "<<steps<<" steps, "<<t_evol<<" s"
                <<" ("<<(steps>0 ? t_evol/steps*1e3 : 0)<<" ms/step), "
                <<(t_evol>0 ? n_cells*(double)steps/t_evol : 0)
                <<" zone-cycles/s"<<endl;
        }
    }

    //One evaluation of the spatial operator on the n_ader slices of the
    //U_ader_* arrays: interpolation to flux points, physical fluxes, halo
    //exchange and Riemann solve (plus viscous terms when enabled)
    void Solve_fluxes(CommHelper comm, dimension X_dim, dimension Y_dim, dimension Z_dim){
        Interpolate_to_fp();
        Compute_Fluxes();
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
        // Interpolate to flux points, one coalesced 1d sweep per direction
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
        if(cfg.fallback)
            Write(troubles,n_output);
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
        sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
            for(int t_id=0; t_id<nader; t_id++){
            for(int var=0; var<nvar; var++){
            U_ader.Vector(t_id,var,k,j,i,kk,jj,ii) = U.Vector(0,var,k,j,i,kk,jj,ii);
            }}
        });
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
        return fv_sum_cells_ngh2(Nz,Ny,Nx,
            KOKKOS_LAMBDA(int k,int j,int i,double& sum){
                double V = fx(i+1)-fx(i);
                if(ay) V *= fy(j+1)-fy(j);
                if(az) V *= fz(k+1)-fz(k);
                sum += U.Vector(0,k,j,i)*V;
            });
    }

    double fv_mass(SD_Solution U, dimension X_dim, dimension Y_dim, dimension Z_dim){
        //Total mass of the cv-average density over active cells
        int Nx=U.Nx, Ny=U.Ny, Nz=U.Nz, px=U.nx, py=U.ny, pz=U.nz;
        int qx=px, qy=py, qz=pz;
        Vector fx = X_dim.fv_faces;
        Vector fy = Y_dim.fv_faces;
        Vector fz = Z_dim.fv_faces;
        bool ay=cfg.active[_y_], az=cfg.active[_z_];
        GHOST_LOCALS;
        return sd_sum_active_cells(Nz,Ny,Nx,pz,py,px,
            KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii,double& sum){
                double V = fx(I+1)-fx(I);
                if(ay) V *= fy(J+1)-fy(J);
                if(az) V *= fz(K+1)-fz(K);
                sum += U.Vector(0,0,k,j,i,kk,jj,ii)*V;
            });
    }
    #endif

    void FV_Update_solution(CommHelper comm, dimension X_dim,dimension Y_dim,dimension Z_dim){
        transform_sp_to_cv(U_sp,U_cv);
        #ifdef DEBUG_MASS
        printf("step %d mass in : %.15e\n", n_step, fv_mass(U_cv,X_dim,Y_dim,Z_dim));
        #endif
        for(int ader=0;ader<n_ader;ader++){
            Integrate_fluxes(ader);
            fv_update_solution(U_new,U_old,U_cv,
                F_x,X_dim.fv_faces,
                F_y,Y_dim.fv_faces,
                F_z,Z_dim.fv_faces,
                wt,ader,dt,0);
            #ifdef DEBUG_MASS
            printf("  ader %d U_new after SD-flux update : %.15e\n", ader, fv_mass_cells(U_new,X_dim,Y_dim,Z_dim));
            #endif
            apply_fv_boundaries(comm,U_old);
            apply_fv_boundaries(comm,U_new);
            compute_primitives(U_new,W_new);
            compute_primitives(U_old,W_old);
            //Following the reference implementation, only density and
            //pressure enter the NAD/SED checks (uniform or zero fields,
            //like transverse velocities, have no meaningful relative band)
            detect_troubles(W_new,W_old,troubles,
                alpha_x,alpha_y,alpha_z,
                X_dim,Y_dim,Z_dim,1,(1<<_d_)|(1<<_p_));
            //Ghost flags must be periodic images so that the blending
            //stencils near the domain boundary see the same data as their
            //periodic partners
            apply_fv_boundaries(comm,troubles);
            //Fractional blend factor: spread the trouble flags to the
            //neighborhood (0.75/0.5/0.375 weights + 0.25 ring), or use the
            //raw flags when blending is disabled
            if(cfg.blending){
                apply_blending(troubles,theta_tmp);
                blending_ring(theta_tmp,theta);
            }
            else
                theta_from_troubles(troubles,theta);
            //Ghost thetas must also be exact periodic images so the two
            //domain boundary faces of each direction receive identical
            //blended fluxes (exact conservation)
            apply_fv_boundaries(comm,theta);
            #ifdef DEBUG_MASS
            if(t==0 && ader==0) Write(F_x,899);
            #endif
            fallback_fluxes(W_old,theta,
                X_dim.fv_centers,X_dim.fv_faces,F_x,
                Y_dim.fv_centers,Y_dim.fv_faces,F_y,
                Z_dim.fv_centers,Z_dim.fv_faces,F_z,
                ader,wt,dt);
            fv_update_solution(U_new,U_old,U_cv,
                F_x,X_dim.fv_faces,
                F_y,Y_dim.fv_faces,
                F_z,Z_dim.fv_faces,
                wt,ader,dt,1);
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
            #endif
            #ifdef DEBUG_MASS
            printf("step %d ader %d mass: %.15e\n", n_step, ader, fv_mass(U_cv,X_dim,Y_dim,Z_dim));
            #endif
        }
        transform_cv_to_sp(U_cv,U_sp);
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
