using namespace std;

struct Hydro_ader{
    int n_output;
    int n_step;
    int n_ader;
    int nvar;

    double t;
    double dt;
    double Dt;
    double nu;
    double beta;

    Vector xt;
    Vector wt;

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

    SD_Solution U_ader_sp;
    #if X
    SD_Solution U_ader_fp_x;
    SD_Solution F_ader_fp_x;
    SD_Solution dUx_sp;
    Boundaries BC_fp_x;
    #endif
    #if Y
    SD_Solution U_ader_fp_y;
    SD_Solution F_ader_fp_y;
    SD_Solution dUy_sp;
    Boundaries BC_fp_y;
    #endif
    #if Z
    SD_Solution U_ader_fp_z;
    SD_Solution F_ader_fp_z;
    SD_Solution dUz_sp;
    Boundaries BC_fp_z;
    #endif    

    #ifdef FV
    FV_Solution U_new;
    FV_Solution W_new;
    FV_Solution U_old;
    FV_Solution W_old;
    FV_Solution troubles;
    #if X
    FV_Solution F_x;
    FV_Solution alpha_x;
    FV_Boundaries BC_x;
    #endif
    #if Y
    FV_Solution F_y;
    FV_Solution alpha_y;
    FV_Boundaries BC_y;
    #endif
    #if Z
    FV_Solution F_z;
    FV_Solution alpha_z;
    FV_Boundaries BC_z;
    #endif    
    #endif

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
        //Number of variables, tipically (rho,vx,vy,vz,e)
        nvar = NVAR;
        n_output = 0;
        n_ader = p+1;
        t=0;
        nu = _nu;
        beta = _beta;
        Kokkos::resize(xt,p+1);
        Kokkos::resize(wt,p+1);
        gauss_legendre(0.0, 1.0, p+1, xt.data(), wt.data());

        //////////////
        //Matrices to perform tensorial transformations
        //////////////
        Kokkos::resize(sp_to_fp,p+2,p+1);
        Kokkos::resize(fp_to_sp,p+1,p+2);
        Kokkos::resize(dfp_to_sp,p+1,p+2);
        Kokkos::resize(sp_to_cv,p+1,p+1);
        Kokkos::resize(cv_to_sp,p+1,p+1);
        Kokkos::resize(fp_to_cv,p+1,p+2);
        Kokkos::resize(ader,p+1,p+1);
        Kokkos::resize(invader,p+1,p+1);

        lagrange_matrix(sp_to_fp, x_sp, x_fp, p+1, p+2);
        lagrange_matrix(fp_to_sp, x_fp, x_sp, p+2, p+1);
        lagrange_prime_matrix(dfp_to_sp, x_fp, x_sp, p+2, p+1);
        integral_matrix(sp_to_cv, x_fp, x_sp, p+1, p+1);
        integral_matrix(fp_to_cv, x_fp, x_fp, p+1, p+2);
        inverse(sp_to_cv, cv_to_sp, p+1);
        ader_matrix(ader, xt, wt, p+1);
        inverse(ader, invader, p+1);

        W_sp.init("W_sp",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        U_sp.init("U_sp",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        W_cv.init("W_cv",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        U_cv.init("U_cv",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);

        U_ader_sp.init("U_ader_sp",n_ader,nvar,Z_dim,Y_dim,X_dim,0,0,0);
   
        #if X
        U_ader_fp_x.init("U_ader_fp_x",n_ader,nvar,Z_dim,Y_dim,X_dim,0,0,1);
        F_ader_fp_x.init("F_ader_fp_x",n_ader,nvar,Z_dim,Y_dim,X_dim,0,0,1);
        dUx_sp.init("dUx_sp",n_ader,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        BC_fp_x.init(X_dim,_BCx_,n_ader,nvar,Z_dim.N_total,Y_dim.N_total,1,Z_dim.n_sp,Y_dim.n_sp,1);
        #endif
        #if Y
        U_ader_fp_y.init("U_ader_fp_y",n_ader,nvar,Z_dim,Y_dim,X_dim,0,1,0);
        F_ader_fp_y.init("F_ader_fp_y",n_ader,nvar,Z_dim,Y_dim,X_dim,0,1,0);
        dUy_sp.init("dUy_sp",n_ader,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        BC_fp_y.init(Y_dim,_BCy_,n_ader,nvar,Z_dim.N_total,1,X_dim.N_total,Z_dim.n_sp,1,X_dim.n_sp);
        #endif
        #if Z
        U_ader_fp_z.init("U_ader_fp_z",n_ader,nvar,Z_dim,Y_dim,X_dim,1,0,0);
        F_ader_fp_z.init("F_ader_fp_z",n_ader,nvar,Z_dim,Y_dim,X_dim,1,0,0);
        dUz_sp.init("dUz_sp",n_ader,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        BC_fp_z.init(Z_dim,_BCz_,n_ader,nvar,1,Y_dim.N_total,X_dim.N_total,1,Y_dim.n_sp,X_dim.n_sp);
        #endif

        #ifdef FV
        U_new.init("U_new",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        W_new.init("W_new",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        U_old.init("U_old",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        W_old.init("W_old",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        troubles.init("troubles",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        #if X
        F_x.init("F_x",n_ader,nvar,Z_dim,Y_dim,X_dim,0,0,1);
        alpha_x.init("alpha_x",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        BC_x.init(X_dim,_BCx_,nvar,Z_dim.fv_cells,Y_dim.fv_cells,nGHx);
        #endif
        #if Y
        F_y.init("F_y",n_ader,nvar,Z_dim,Y_dim,X_dim,0,1,0);
        alpha_y.init("alpha_y",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        BC_y.init(Y_dim,_BCy_,nvar,Z_dim.fv_cells,nGHy,X_dim.fv_cells);
        #endif
        F_z.init("F_z",n_ader,nvar,Z_dim,Y_dim,X_dim,1,0,0);
        alpha_z.init("alpha_z",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        BC_z.init(Z_dim,_BCz_,nvar,nGHz,Y_dim.fv_cells,X_dim.fv_cells);
        #endif

        ////////////////////////
        //Initial Conditions:
        ////////////////////////
        Initialize(W_cv,X_dim.sd_faces,Y_dim.sd_faces,Z_dim.sd_faces,xt,wt);
        transform_cv_to_sp(W_cv,W_sp);
        compute_conservatives(W_sp,U_sp);

        Dt = compute_dt(W_cv,X_dim.h,Y_dim.h,Z_dim.h);
        if(Master)
            cout<<"dx = "<<X_dim.h<<" dt = "<<Dt<<endl;
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

        while(t<t_end){
            ////Initialize ADER time slices
            copy_ader(U_sp,U_ader_sp);
            
            //Picard iteration
            for(int ader=0;ader<n_ader;ader++){ 
                Interpolate_to_fp();
                Compute_Fluxes();
                Boundaries(comm);
                Riemann_Solver();

                #ifdef VISCOSITY
                Viscosity(X_dim.h,Y_dim.h,Z_dim.h);
                Boundaries(comm);
                Rusanov_Solver();
                #endif

                if(ader<n_ader-1)
                    Update_prediction(X_dim.h,Y_dim.h,Z_dim.h);
            }
            #ifdef FV
            FV_Update_solution(comm,X_dim,Y_dim,Z_dim);
            #else
            Update_solution(X_dim.h,Y_dim.h,Z_dim.h);
            #endif
            compute_primitives(U_sp,W_sp);
            transform_sp_to_cv(W_sp,W_cv);
            t+=dt;
            n_step++;
            dt=compute_dt(W_cv,X_dim.h,Y_dim.h,Z_dim.h);
        
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

    void transform_cv_to_sp(SD_Solution U_cv, SD_Solution U_sp){
        transform_a_to_b(U_cv,
        U_sp
        #if X
        ,cv_to_sp
        #endif
        #if Y
        ,cv_to_sp
        #endif
        #if Z
        ,cv_to_sp
        #endif
        ); 
    }

    void transform_sp_to_cv(SD_Solution U_sp, SD_Solution U_cv){
        transform_a_to_b(U_sp,
        U_cv
        #if X
        ,sp_to_cv
        #endif
        #if Y
        ,sp_to_cv
        #endif
        #if Z
        ,sp_to_cv
        #endif
        ); 
    }
    
    void Interpolate_to_fp(){
        // Interpolate to flux points
        #if X
        transform_a_to_b_1d(U_ader_sp,U_ader_fp_x,sp_to_fp,_x_);
        #endif
        #if Y
        transform_a_to_b_1d(U_ader_sp,U_ader_fp_y,sp_to_fp,_y_);
        #endif
        #if Z
        transform_a_to_b_1d(U_ader_sp,U_ader_fp_z,sp_to_fp,_z_);
        #endif
    }

    void Compute_Fluxes(){
        //Compute fluxes
        #if X
        compute_fluxes(
            U_ader_fp_x,
            F_ader_fp_x,
            _vx_
            #if Y
            ,_vy_
            #endif
            #if Z
            ,_vz_
            #endif
            );
        #endif
        #if Y
        compute_fluxes(
            U_ader_fp_y,
            F_ader_fp_y,
            _vy_
            #if Z
            ,_vz_
            #endif
            #if X
            ,_vx_
            #endif
            );
        #endif
        #if Z
        compute_fluxes(
            U_ader_fp_z,
            F_ader_fp_z,
            _vz_
            #if X
            ,_vx_
            #endif
            #if Y
            ,_vy_
            #endif
            );
        #endif
    }
    
    void Boundaries(CommHelper comm){
        //Communications are done sequentially in different directions
        //to ensure that corners zare properly communicated
        #if X
        boundaries(comm,BC_fp_x,U_ader_fp_x);
        #endif
        #if Y
        boundaries(comm,BC_fp_y,U_ader_fp_y);
        #endif
        #if Z
        boundaries(comm,BC_fp_z,U_ader_fp_z);
        #endif
    }

    void Riemann_Solver(){
        #if X
        sd_riemann_solver(
            U_ader_fp_x,
            F_ader_fp_x,
            _vx_
            #if Y
            ,_vy_
            #endif
            #if Z
            ,_vz_
            #endif
            ,_x_);
        #endif
        #if Y
        sd_riemann_solver(
            U_ader_fp_y,
            F_ader_fp_y,
            _vy_
            #if Z
            ,_vz_
            #endif
            #if X
            ,_vx_
            #endif
            ,_y_);
        #endif
        #if Z
        sd_riemann_solver(
            U_ader_fp_z,
            F_ader_fp_z,
            _vz_
            #if X
            ,_vx_
            #endif
            #if Y
            ,_vy_
            #endif
            ,_z_);
        #endif
    }

    void Viscosity(double dx, double dy, double dz){
        #if X
        compute_gradient(U_ader_fp_x, dUx_sp, dx, dfp_to_sp, _x_);
        #endif
        #if Y
        compute_gradient(U_ader_fp_y, dUy_sp, dy, dfp_to_sp, _y_);
        #endif
        #if Z
        compute_gradient(U_ader_fp_z, dUz_sp, dz, dfp_to_sp, _z_);
        #endif
        #if X
        compute_viscous_flux(
            U_ader_fp_x,
            dUx_sp,
            _vx_,
            #if Y
            dUy_sp,
            _vy_,
            #endif
            #if Z
            dUz_sp,
            _vz_,
            #endif
            sp_to_fp,
            nu,
            beta,
            _x_);
        #endif
        #if Y
        compute_viscous_flux(
            U_ader_fp_y,
            dUy_sp,
            _vy_,
            #if Z
            dUz_sp,
            _vz_,
            #endif
            #if X
            dUx_sp,
            _vx_,
            #endif
            sp_to_fp,
            nu,
            beta,
            _y_);
        #endif
        #if Z
        compute_viscous_flux(
            U_ader_fp_z,
            dUz_sp,
            _vz_,
            #if X
            dUx_sp,
            _vx_,
            #endif
            #if Y
            dUy_sp,
            _vy_,
            #endif
            sp_to_fp,
            nu,
            beta,
            _y_);
        #endif
    }

    void Rusanov_Solver(){
        #if X
        sd_rusanov_solver(U_ader_fp_x,F_ader_fp_x,_x_);
        #endif
        #if Y
        sd_rusanov_solver(U_ader_fp_y,F_ader_fp_y,_y_);
        #endif
        #if Z
        sd_rusanov_solver(U_ader_fp_z,F_ader_fp_z,_z_);
        #endif
    }

    void Update_prediction(const double dx,const double dy,const double dz){
        update_prediction(U_sp,
            U_ader_sp,
            F_ader_fp_x,
            #if Y
            F_ader_fp_y,
            #endif
            #if Z
            F_ader_fp_z,
            #endif
            dfp_to_sp,
            invader,
            wt,
            dx,
            #if Y
            dy,
            #endif
            #if Z
            dz,
            #endif
            dt);
    }

    void Update_solution(double dx, double dy, double dz){
        update_solution(U_sp,
            F_ader_fp_x,
            #if Y
            F_ader_fp_y,
            #endif
            #if Z
            F_ader_fp_z,
            #endif
            dfp_to_sp,
            wt,
            dx,
            #if Y
            dy,
            #endif
            #if Z
            dz,
            #endif
            dt);
    }

    void Write_outputs(){
        if(Master)
            cout<<endl<<"OUTPUT "<<n_output<<endl;
        #ifdef FV
        Write(troubles,n_output);
        #endif
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
        SD_for_all(
            U_ader.Vector(t_id,var,k,j,i,kk,jj,ii) = U.Vector(0,var,k,j,i,kk,jj,ii);
        );
    }

    #ifdef FV
    void Integrate_fluxes(){
        #if X
        face_integral(F_ader_fp_x, F_x, sp_to_cv, _x_);
        #endif
        #if Y
        face_integral(F_ader_fp_y, F_y, sp_to_cv, _y_);
        #endif
        #if Z
        face_integral(F_ader_fp_z, F_z, sp_to_cv, _z_);
        #endif
    }

    void FV_Update_solution(CommHelper comm, dimension X_dim,dimension Y_dim,dimension Z_dim){
        Integrate_fluxes();
        transform_sp_to_cv(U_sp,U_cv);
        for(int ader=0;ader<n_ader;ader++){
            fv_update_solution(
                U_new,
                U_old,
                U_cv,
                #if X
                F_x,
                X_dim.fv_faces,
                #endif
                #if Y
                F_y,
                Y_dim.fv_faces,
                #endif
                #if Z
                F_z,
                Z_dim.fv_faces,
                #endif
                wt,
                ader,
                dt,
                0);
            FV_Boundaries(comm,U_old);
            FV_Boundaries(comm,U_new);
            compute_primitives(U_new,W_new);
            compute_primitives(U_old,W_old);
            detect_troubles(
                W_new,
                W_old,
                troubles,
                #if X
                alpha_x,
                #endif
                #if Y
                alpha_y,
                #endif
                #if Z
                alpha_z,
                #endif
                X_dim,
                Y_dim,
                Z_dim
                );
            fallback_fluxes(
                W_old,
                troubles,
                X_dim.fv_centers,
                X_dim.fv_faces,
                F_x,
                #if Y
                Y_dim.fv_centers,
                Y_dim.fv_faces,
                F_y,
                #endif
                #if Z
                Z_dim.fv_centers,
                Z_dim.fv_faces,
                F_z,
                #endif
                ader,
                wt,
                dt);
            fv_update_solution(
                U_new,
                U_old,
                U_cv,
                #if X
                F_x,
                X_dim.fv_faces,
                #endif
                #if Y
                F_y,
                Y_dim.fv_faces,
                #endif
                #if Z
                F_z,
                Z_dim.fv_faces,
                #endif
                wt,
                ader,
                dt,
                1);
        }
        transform_cv_to_sp(U_cv,U_sp);
    }

    void FV_Boundaries(CommHelper comm, FV_Solution U){
        //Communications are done sequentially in different directions
        //to ensure that corners zare properly communicated
        #if X
        boundaries(comm,BC_x,U);
        #endif
        #if Y
        boundaries(comm,BC_y,U);
        #endif
        #if Z
        boundaries(comm,BC_z,U);
        #endif
    }

    #endif
};
