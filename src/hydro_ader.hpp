using namespace std;

struct Hydro_ader{
    int n_output;
    int n_step;
    int n_ader;
    int nvar;

    double t;
    double dt;
    double Dt;

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
    SD_Solution U_cv;
    SD_Solution W_cv;

    SD_Solution U_ader_sp;
    #if X
    SD_Solution U_ader_fp_x;
    SD_Solution F_ader_fp_x;
    Boundaries BC_fp_x;
    #endif
    #if Y
    SD_Solution U_ader_fp_y;
    SD_Solution F_ader_fp_y;
    Boundaries BC_fp_y;
    #endif
    #if Z
    SD_Solution U_ader_fp_z;
    SD_Solution F_ader_fp_z;
    Boundaries BC_fp_z;
    #endif    

    #ifdef FV
    SD_Solution U_new;
    #if X
    FV_Solution F_x;
    #endif
    #if Y
    FV_Solution F_y;
    #endif
    #if Z
    FV_Solution F_z;
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
        double* x_fp
    ){
        //Number of variables, tipically (rho,vx,vy,vz,e,Vrho)
        nvar = 3 + DIM;
        n_output = 0;
        n_ader = p+1;
        t=0;
        p=p;
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

        U_ader_sp.init("U_ader_sp",n_ader,nvar,Z_dim,Y_dim,X_dim,0,0,0);
   
        #if X
        U_ader_fp_x.init("U_ader_fp_x",n_ader,nvar,Z_dim,Y_dim,X_dim,0,0,1);
        F_ader_fp_x.init("F_ader_fp_x",n_ader,nvar,Z_dim,Y_dim,X_dim,0,0,1);
        BC_fp_x.init(X_dim,_BCx_,n_ader,nvar,Z_dim.N_total,Y_dim.N_total,1,Z_dim.n_sp,Y_dim.n_sp,1);
        #endif
        #if Y
        U_ader_fp_y.init("U_ader_fp_y",n_ader,nvar,Z_dim,Y_dim,X_dim,0,1,0);
        F_ader_fp_y.init("U_ader_fp_y",n_ader,nvar,Z_dim,Y_dim,X_dim,0,1,0);
        BC_fp_y.init(Y_dim,_BCy_,n_ader,nvar,Z_dim.N_total,1,X_dim.N_total,Z_dim.n_sp,1,X_dim.n_sp);
        #endif
        #if Z
        U_ader_fp_z.init("U_ader_fp_z",n_ader,nvar,Z_dim,Y_dim,X_dim,1,0,0);
        F_ader_fp_z.init("U_ader_fp_z",n_ader,nvar,Z_dim,Y_dim,X_dim,1,0,0);
        BC_fp_z.init(Z_dim,_BCz_,n_ader,nvar,1,Y_dim.N_total,X_dim.N_total,1,Y_dim.n_sp,X_dim.n_sp);
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
                Compute_Fluxes();
                
                Boundaries(comm);

                Riemann_Solver();

            if(ader<n_ader-1){
                Update_prediction(X_dim.h,Y_dim.h,Z_dim.h);
            }
        }
        #ifdef FV
        Integrate_fluxes();
        transform_sp_to_cv(U_sp,U_cv);
        for(int ader=0;ader<n_ader;ader++)
            fv_update_solution(X_dim.sd_faces,Y_dim.sd_faces,Z_dim.sd_faces,ader);
        transform_cv_to_sp(U_cv,U_sp);
        #else
        Update_solution(X_dim.h,Y_dim.h,Z_dim.h);
        #endif
        compute_primitives(U_sp,W_sp);
        transform_sp_to_cv(W_sp,W_cv);
        t+=dt;
        n_step++;
        dt=compute_dt(W_cv,X_dim.h,Y_dim.h,Z_dim.h);
        //Outputs
        if(Master){
            cout<<".";
        }
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
        compute_fluxes(U_ader_fp_x,F_ader_fp_x,_vx_,_vy_,_vz_);
        #endif
        #if Y
        compute_fluxes(U_ader_fp_y,F_ader_fp_y,_vy_,_vx_,_vz_);
        #endif
        #if Z
        compute_fluxes(U_ader_fp_z,F_ader_fp_z,_vz_,_vx_,_vy_);
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
        sd_riemann_solver(U_ader_fp_x,F_ader_fp_x,_vx_,_vy_,_vz_,0);
        #endif
        #if Y
        sd_riemann_solver(U_ader_fp_y,F_ader_fp_y,_vy_,_vx_,_vz_,1);
        #endif
        #if Z
        sd_riemann_solver(U_ader_fp_z,F_ader_fp_z,_vz_,_vx_,_vy_,2);
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
        face_integral(F_ader_fp_x, F_x, sp_to_cv, 0);
        #endif
        #if Y
        face_integral(F_ader_fp_y, F_y, sp_to_cv, 1);
        #endif
        #if Z
        face_integral(F_ader_fp_z, F_z, sp_to_cv, 2);
        #endif
    }

    void fv_update_solution(
        #if X
        Matrix faces_x,
        #endif
        #if Y
        Matrix faces_y,
        #endif
        #if Z
        Matrix faces_z,
        #endif
        int t_id
    ){
        //FV_for_active_cells(
        //    double h;
        //    #ifdef X
        //    h  = faces_x(i+1)-faces_x(i);
        //    F += (F_x.Vector(t_id,var,k,j,i+1)-F_x.Vector(t_id,var,k,j,i))/h;
        //    #endif
        //    #ifdef Y
        //    h  = faces_y(j+1)-faces_y(j);
        //    F += (F_y.Vector(t_id,var,k,j+1,i)-F_y.Vector(t_id,var,k,j,i))/h;
        //    #endif
        //    #ifdef Z
        //    h  = faces_z(k+1)-faces_z(k);
        //    F += (F_z.Vector(t_id,var,k+1,j,i)-F_z.Vector(t_id,var,k,j,i))/h;
        //    #endif
        //    U_new.Vector(var,k,j,i) = U_cv.Vector(var,k,j,i) - wt(t_id)*dt*F;
        //);
    }
    #endif
};
