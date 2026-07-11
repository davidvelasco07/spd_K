using namespace std;

struct Induction_ader{

    int n_output;
    int n_step;
    int n_ader;
    int nvar;

    double t;
    double dt;
    double Dt;
    double nu;

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

    SD_Solution Bx_fp_x;
    SD_Solution By_fp_y;
    SD_Solution Bz_fp_z;
    SD_Solution B2_cv;

    SD_Solution Ax_ep_yz;
    SD_Solution Ay_ep_zx;
    SD_Solution Az_ep_xy;

    SD_Solution V_cv;
    SD_Solution V_sp;

    SD_Solution Bx_ader_fp_x;
    SD_Solution By_ader_fp_y;
    SD_Solution Bz_ader_fp_z;

    SD_Solution Ex_ader_ep_yz;
    SD_Solution Ey_ader_ep_zx;
    SD_Solution Ez_ader_ep_xy;

    SD_Solution D1_yz;
    SD_Solution D2_yz;
    SD_Solution D1_zx;
    SD_Solution D2_zx;
    SD_Solution D1_xy;
    SD_Solution D2_xy;

    Boundaries BC_Ey_ep_x;
    Boundaries BC_Ez_ep_x;
    Boundaries BC_Ex_ep_y;
    Boundaries BC_Ez_ep_y;
    Boundaries BC_Ey_ep_z;
    Boundaries BC_Ex_ep_z;

    //FV update + fallback (allocated only when cfg.fallback is set)
    SD_Solution Bx;
    SD_Solution By;
    SD_Solution Bz;
    SD_Solution T_fp_x; //scratch for directional-sweep transforms
    SD_Solution T_fp_y;
    SD_Solution T_fp_z;
    FV_Solution Bx_old;
    FV_Solution By_old;
    FV_Solution Bz_old;
    FV_Solution B_old;
    FV_Solution Bx_new;
    FV_Solution By_new;
    FV_Solution Bz_new;
    FV_Solution B_new;
    FV_Solution Ex;
    FV_Solution Ey;
    FV_Solution Ez;
    FV_Boundaries BC_x;
    FV_Boundaries BC_y;
    FV_Boundaries BC_z;
    FV_Solution troubles;
    FV_Solution alpha_x;
    FV_Solution alpha_y;
    FV_Solution alpha_z;

    Induction_ader(
        CommHelper comm,
        int p,
        dimension X_dim,
        dimension Y_dim,
        dimension Z_dim,
        double* x,
        double* w,
        double* x_sp,
        double* x_fp,
        double _nu
    ){
        if(cfg.ndim != 3){
            if(Master)
                cout<<"ERROR: the induction solver currently requires 3 active dimensions"<<endl;
            exit(1);
        }
        //E3,B1,B2,v1,v2,VxB3,B1,B2
        nvar = 1 + 2 + 2 + 1 + 2;
        n_output = 0;
        n_step = 0;
        n_ader = p+1;
        t=0;
        p=p;
        nu = _nu;
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

        Bx_fp_x.init("Bx_fp_x",1,1,Z_dim,Y_dim,X_dim,0,0,1);
        By_fp_y.init("By_fp_y",1,1,Z_dim,Y_dim,X_dim,0,1,0);
        Bz_fp_z.init("Bz_fp_z",1,1,Z_dim,Y_dim,X_dim,1,0,0);

        B2_cv.init("B2_cv",1,DIM+1,Z_dim,Y_dim,X_dim,0,0,0);

        Ax_ep_yz.init("Ax_ep_yz",1,1,Z_dim,Y_dim,X_dim,1,1,0);
        Ay_ep_zx.init("Ay_ep_zx",1,1,Z_dim,Y_dim,X_dim,1,0,1);
        Az_ep_xy.init("Az_ep_xy",1,1,Z_dim,Y_dim,X_dim,0,1,1);

        V_cv.init("V_cv",1,DIM,Z_dim,Y_dim,X_dim,0,0,0);
        V_sp.init("V_sp",1,DIM,Z_dim,Y_dim,X_dim,0,0,0);

        ////////////////////////
        //Initial Conditions:
        ////////////////////////
        Initialize_ep(Ax_ep_yz,X_dim.sd_centers,Y_dim.sd_faces  ,Z_dim.sd_faces  ,_x_);
        Initialize_ep(Ay_ep_zx,X_dim.sd_faces  ,Y_dim.sd_centers,Z_dim.sd_faces  ,_y_);
        Initialize_ep(Az_ep_xy,X_dim.sd_faces  ,Y_dim.sd_faces  ,Z_dim.sd_centers,_z_);
    
        //Initialize(V_cv,X_dim.sd_faces,Y_dim.sd_faces,Z_dim.sd_faces,xt,wt);
        //transform_a_to_b(V_cv,V_sp,cv_to_sp,cv_to_sp,cv_to_sp);
        set_value(V_sp,1);

        //B=VxA
        rotational_a_to_b(Ay_ep_zx,Az_ep_xy,Bx_fp_x,dfp_to_sp,Y_dim.h,Z_dim.h,_x_);
        rotational_a_to_b(Az_ep_xy,Ax_ep_yz,By_fp_y,dfp_to_sp,Z_dim.h,X_dim.h,_y_);
        rotational_a_to_b(Ax_ep_yz,Ay_ep_zx,Bz_fp_z,dfp_to_sp,X_dim.h,Y_dim.h,_z_);

        Bx_ader_fp_x.init("Bx_ader_fp_x",n_ader,1,Z_dim,Y_dim,X_dim,0,0,1);
        By_ader_fp_y.init("By_ader_fp_y",n_ader,1,Z_dim,Y_dim,X_dim,0,1,0);
        Bz_ader_fp_z.init("Bz_ader_fp_z",n_ader,1,Z_dim,Y_dim,X_dim,1,0,0);

        //variables: E3,B1,B2,v1,v2
        //D = V x B 
        //Ex,By,Bz,vy,vz,Dx -> f,f,s
        //Ey,Bx,Bz,vx,vz,Dy -> f,s,f
        //Ez,Bx,By,vx,vy,Dz -> s,f,f
        Ex_ader_ep_yz.init("Ex_ader_ep_yz",n_ader,nvar,Z_dim,Y_dim,X_dim,1,1,0);
        Ey_ader_ep_zx.init("Ey_ader_ep_zx",n_ader,nvar,Z_dim,Y_dim,X_dim,1,0,1);
        Ez_ader_ep_xy.init("Ez_ader_ep_xy",n_ader,nvar,Z_dim,Y_dim,X_dim,0,1,1);

        #ifdef OHMIC_DIFFUSION
        D1_yz.init("D1_yz",n_ader,1,Z_dim,Y_dim,X_dim,1,0,0);
        D2_yz.init("D2_yz",n_ader,1,Z_dim,Y_dim,X_dim,0,1,0);
        D1_zx.init("D1_zx",n_ader,1,Z_dim,Y_dim,X_dim,0,0,1);
        D2_zx.init("D2_zx",n_ader,1,Z_dim,Y_dim,X_dim,1,0,0);
        D1_xy.init("D1_xy",n_ader,1,Z_dim,Y_dim,X_dim,0,1,0);
        D2_xy.init("D2_xy",n_ader,1,Z_dim,Y_dim,X_dim,0,0,1);
        #endif

        //X-interfaces
        BC_Ey_ep_x.init(X_dim,cfg.bc[_x_],n_ader,nvar,Z_dim.N_total,Y_dim.N_total,1,Z_dim.n_fp,Y_dim.n_sp,1);
        BC_Ez_ep_x.init(X_dim,cfg.bc[_x_],n_ader,nvar,Z_dim.N_total,Y_dim.N_total,1,Z_dim.n_sp,Y_dim.n_fp,1);
        //Y-interfaces
        BC_Ex_ep_y.init(Y_dim,cfg.bc[_y_],n_ader,nvar,Z_dim.N_total,1,X_dim.N_total,Z_dim.n_fp,1,X_dim.n_sp);
        BC_Ez_ep_y.init(Y_dim,cfg.bc[_y_],n_ader,nvar,Z_dim.N_total,1,X_dim.N_total,Z_dim.n_sp,1,X_dim.n_fp);
        //Z-interfaces
        BC_Ex_ep_z.init(Z_dim,cfg.bc[_z_],n_ader,nvar,1,Y_dim.N_total,X_dim.N_total,1,Y_dim.n_fp,X_dim.n_sp);
        BC_Ey_ep_z.init(Z_dim,cfg.bc[_z_],n_ader,nvar,1,Y_dim.N_total,X_dim.N_total,1,Y_dim.n_sp,X_dim.n_fp);

        if(cfg.fallback){
        Bx.init("Bx",1,1,Z_dim,Y_dim,X_dim,0,0,1);
        By.init("By",1,1,Z_dim,Y_dim,X_dim,0,1,0);
        Bz.init("Bz",1,1,Z_dim,Y_dim,X_dim,1,0,0);
        T_fp_x.init("T_fp_x",1,1,Z_dim,Y_dim,X_dim,0,0,1);
        T_fp_y.init("T_fp_y",1,1,Z_dim,Y_dim,X_dim,0,1,0);
        T_fp_z.init("T_fp_z",1,1,Z_dim,Y_dim,X_dim,1,0,0);
        Bx_old.init("Bx_old",1,Z_dim,Y_dim,X_dim,0,0,1);
        By_old.init("By_old",1,Z_dim,Y_dim,X_dim,0,1,0);
        Bz_old.init("Bz_old",1,Z_dim,Y_dim,X_dim,1,0,0);
        B_old.init("B_old",4,Z_dim,Y_dim,X_dim,0,0,0);
        Bx_new.init("Bx_new",1,Z_dim,Y_dim,X_dim,0,0,1);
        By_new.init("By_new",1,Z_dim,Y_dim,X_dim,0,1,0);
        Bz_new.init("Bz_new",1,Z_dim,Y_dim,X_dim,1,0,0);
        B_new.init("B_new",4,Z_dim,Y_dim,X_dim,0,0,0);
        troubles.init("troubles",4,Z_dim,Y_dim,X_dim,0,0,0);
        alpha_x.init("alpha_x",4,Z_dim,Y_dim,X_dim,0,0,0);
        alpha_y.init("alpha_y",4,Z_dim,Y_dim,X_dim,0,0,0);
        alpha_z.init("alpha_z",4,Z_dim,Y_dim,X_dim,0,0,0);
        Ex.init("Ex",1,Z_dim,Y_dim,X_dim,1,1,0);
        Ey.init("Ey",1,Z_dim,Y_dim,X_dim,1,0,1);
        Ez.init("Ez",1,Z_dim,Y_dim,X_dim,0,1,1);
        BC_x.init(X_dim,cfg.bc[_x_],4,Z_dim.fv_nfaces,Y_dim.fv_nfaces,nGHx);
        BC_y.init(Y_dim,cfg.bc[_y_],4,Z_dim.fv_nfaces,nGHy,X_dim.fv_nfaces);
        BC_z.init(Z_dim,cfg.bc[_z_],4,nGHz,Y_dim.fv_nfaces,X_dim.fv_nfaces);
        }

        dt = Induction_compute_dt(
            B2_cv,
            X_dim.h,
            Y_dim.h,
            Z_dim.h,
            X_dim.sd_centers,
            Y_dim.sd_centers,
            Z_dim.sd_centers
            );
        if(nu>0.0)
            Dt = 0.5*cfg.cfl*(X_dim.h*X_dim.h/nu)/(cfg.ndim)/(cfg.ndim)/(p+1)/(p+1);
        else Dt=1E6;
        if(Master)
            cout<<"dx = "<<X_dim.h<<" dt_nu = "<<Dt<<" dt_v = "<<dt<<endl;
        Dt = min(dt,Dt);
        cout<<Dt<<endl;
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
            copy_ader(Bx_fp_x,Bx_ader_fp_x);
            copy_ader(By_fp_y,By_ader_fp_y);
            copy_ader(Bz_fp_z,Bz_ader_fp_z);
            
            //Picard iteration
            for(int ader=0;ader<n_ader;ader++){ 
                //Compute electric field at edge points
                ElectricField(X_dim,Y_dim,Z_dim);
                //Communicate boundary conditions 
                Boundaries(comm);
                //Solve the Riemann problem for E3,B1,B2,v1,v2
                Riemann_Solver();
                #ifdef OHMIC_DIFFUSION
                Ohmic_Diffusion(X_dim.h,Y_dim.h,Z_dim.h);
                Boundaries(comm);
                //Solve the Riemann problem for (VxB)^f
                Ohmic_Riemann_Solver();
                #endif

            if(ader<n_ader-1)
                Update_prediction(X_dim.h,Y_dim.h,Z_dim.h);
            }
            if(cfg.fallback)
                FV_Update_solution(comm,X_dim,Y_dim,Z_dim);
            else
                Update_solution(X_dim.h,Y_dim.h,Z_dim.h);
            t+=dt;
            n_step++;
            dt=Dt;
            //Outputs
            if(Master)cout<<".";
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

    void ElectricField(dimension X_dim, dimension Y_dim, dimension Z_dim){
        //Ez,Bx,By,vx,vy -> s,f,f: Ez = vxBy - vyBx
        compute_E(
            Ez_ader_ep_xy,
            V_sp,
            Bx_ader_fp_x,
            By_ader_fp_y,
            sp_to_fp,
            X_dim.sd_faces,
            Y_dim.sd_faces,
            Z_dim.sd_centers,
            _z_);
        //Ey,Bx,Bz,vx,vz -> f,s,f: Ey = vzBx - vxBz
        compute_E(
            Ey_ader_ep_zx,
            V_sp,
            Bz_ader_fp_z,
            Bx_ader_fp_x,
            sp_to_fp,
            X_dim.sd_faces,
            Y_dim.sd_centers,
            Z_dim.sd_faces,
            _y_);
        //Ex,By,Bz,vy,Bz -> f,f,s: Ex = vyBz - vzBy
        compute_E(
            Ex_ader_ep_yz,
            V_sp,
            By_ader_fp_y,
            Bz_ader_fp_z,
            sp_to_fp,
            X_dim.sd_centers,
            Y_dim.sd_faces,
            Z_dim.sd_faces,
            _x_);
    }

    void Boundaries(CommHelper comm){
        // Remember: The arrays for Ex,Ey,Ez at edge-points (ep)
        // contain the variables E3,B1,B2,v1,v2 
        // that need to be communicated to solve
        // the Riemann problem at both interfaces
        // X-direction: Ey,Ez
        boundaries(comm,BC_Ey_ep_x,Ey_ader_ep_zx);
        boundaries(comm,BC_Ez_ep_x,Ez_ader_ep_xy);
        // Y-direction: Ex,Ez
        boundaries(comm,BC_Ex_ep_y,Ex_ader_ep_yz);
        boundaries(comm,BC_Ez_ep_y,Ez_ader_ep_xy);
        // Z-direction: Ex,Ey
        boundaries(comm,BC_Ex_ep_z,Ex_ader_ep_yz);
        boundaries(comm,BC_Ey_ep_z,Ey_ader_ep_zx);
    }
    void Riemann_Solver(){
        //E3,B1,B2,v1,v2: E3 = v1B2 - v2B1
        // X-direction
        E_riemann_solver(Ey_ader_ep_zx,_x_,4);//Ey = vzBx - vxBz: vx=v2
        E_riemann_solver(Ez_ader_ep_xy,_x_,3);//Ez = vxBy - vyBx: vx=v1
        //// Y-direction
        E_riemann_solver(Ez_ader_ep_xy,_y_,4);//Ez = vxBy - vyBx: vy=v2
        E_riemann_solver(Ex_ader_ep_yz,_y_,3);//Ex = vyBz - vzBy: vy=v1
        ////// Z-direction
        E_riemann_solver(Ex_ader_ep_yz,_z_,4);//Ex = vyBz - vzBy: vz=v2
        E_riemann_solver(Ey_ader_ep_zx,_z_,3);//Ey = vzBx - vxBz: vz=v1
    }
    #ifdef OHMIC_DIFFUSION
    void Ohmic_Riemann_Solver(){
        //E3,B1,B2,v1,v2: E3 = v1B2 - v2B1
        // X-direction
        E_Ohmic_riemann_solver(Ey_ader_ep_zx,_x_,4);//Ey = vzBx - vxBz: vx=v2
        E_Ohmic_riemann_solver(Ez_ader_ep_xy,_x_,3);//Ez = vxBy - vyBx: vx=v1
        //// Y-direction
        E_Ohmic_riemann_solver(Ez_ader_ep_xy,_y_,4);//Ez = vxBy - vyBx: vy=v2
        E_Ohmic_riemann_solver(Ex_ader_ep_yz,_y_,3);//Ex = vyBz - vzBy: vy=v1
        ////// Z-direction
        E_Ohmic_riemann_solver(Ex_ader_ep_yz,_z_,4);//Ex = vyBz - vzBy: vz=v2
        E_Ohmic_riemann_solver(Ey_ader_ep_zx,_z_,3);//Ey = vzBx - vxBz: vz=v1
    }
    void Ohmic_Diffusion(double dx, double dy, double dz){
        // At this stage B1,B2 are uniquiley defined at each interface.
        // Therefore, we can compute VxB from flux points to solution points.
        // Then interpolate back to flux points (VxB)^s -> (VxB)^f
        compute_rotational_B(Ex_ader_ep_yz, D1_yz, D2_yz, dy, dz, dfp_to_sp, sp_to_fp, nu,_x_);
        compute_rotational_B(Ey_ader_ep_zx, D1_zx, D2_zx, dz, dx, dfp_to_sp, sp_to_fp, nu,_y_);
        compute_rotational_B(Ez_ader_ep_xy, D1_xy, D2_xy, dx, dy, dfp_to_sp, sp_to_fp, nu,_z_);
    }
    #endif
    void Update_prediction(double dx, double dy, double dz){
        //Compute prediction
        //dBxdt = dEydz - dEzdy
        //dBydt = dEzdx - dExdz
        //dBzdt = dExdy - dEydx
        update_B_prediction(Bx_fp_x,Bx_ader_fp_x,Ey_ader_ep_zx,Ez_ader_ep_xy,dfp_to_sp,invader,wt,dy,dz,dt,_x_);
        update_B_prediction(By_fp_y,By_ader_fp_y,Ez_ader_ep_xy,Ex_ader_ep_yz,dfp_to_sp,invader,wt,dz,dx,dt,_y_);
        update_B_prediction(Bz_fp_z,Bz_ader_fp_z,Ex_ader_ep_yz,Ey_ader_ep_zx,dfp_to_sp,invader,wt,dx,dy,dt,_z_);
    }

    void Update_solution(double dx, double dy, double dz){
        update_B_solution(Bx_fp_x,Ey_ader_ep_zx,Ez_ader_ep_xy,dfp_to_sp,wt,dy,dz,dt,_x_);
        update_B_solution(By_fp_y,Ez_ader_ep_xy,Ex_ader_ep_yz,dfp_to_sp,wt,dz,dx,dt,_y_);
        update_B_solution(Bz_fp_z,Ex_ader_ep_yz,Ey_ader_ep_zx,dfp_to_sp,wt,dx,dy,dt,_z_);
    }

    void Write_outputs(){
        if(Master)
            cout<<endl<<"OUTPUT "<<n_output<<endl;
        compute_B2_cv(B2_cv,Bx_fp_x,By_fp_y,Bz_fp_z,fp_to_cv,sp_to_cv);
        if(cfg.fallback)
            Write(troubles,n_output);
        Write(B2_cv,n_output++);
    }

    void set_value(SD_Solution U,double value){
        int Nx = U.Nx;
        int Ny = U.Ny;
        int Nz = U.Nz;
        int px = U.nx;
        int py = U.ny;
        int pz = U.nz;
        int nader = U.n_ader;
        int nvar = U.n_var;
        sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
            for(int t_id=0; t_id<nader; t_id++){
            for(int var=0; var<nvar; var++){
            if(var==0)
                U.Vector(t_id,var,k,j,i,kk,jj,ii) = value;
            else
                U.Vector(t_id,var,k,j,i,kk,jj,ii) = 0;
            }}
        });
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

    void FV_Update_solution(CommHelper comm, dimension X_dim,dimension Y_dim,dimension Z_dim){
        //B(sp)->B(cv)
        Transform_to_cv();
        for(int ader=0;ader<n_ader;ader++){
            Integrate_edges(ader);
            FV_update_solution(ader, X_dim.fv_faces, Y_dim.fv_faces, Z_dim.fv_faces, 1);
            FV_Boundaries(comm);
            detect_troubles(
                B_new,
                B_old,
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
                Z_dim,
                0,
                //all three B components enter the NAD/SED checks
                0b0111
                ); 
        //    //fallback_fluxes();
        //    //FV_Update_solution();
        }
        //B(cv)->B(sp)
        Transform_to_sp();
    }

    void FV_update_solution(int ader, Vector xs, Vector ys, Vector zs, bool update){
        fv_update_B_solution(Bx_new, Bx_old, Bx, Ey, Ez, ys, zs, wt, dt, ader, _x_, update);
        fv_update_B_solution(By_new, By_old, By, Ez, Ex, zs, xs, wt, dt, ader, _y_, update);
        fv_update_B_solution(Bz_new, Bz_old, Bz, Ex, Ey, xs, ys, wt, dt, ader, _z_, update);
        compute_B_cv_from_cf(B_new, Bx, By, Bz, fp_to_cv);
    }

    void Transform_to_cv(){
        //Computes control-face averages,
        //integrating over the perpendicular dimensions
        #ifdef REF_TRANSFORMS
        transform_a_to_b_2d_ref(Bx_fp_x,Bx,sp_to_cv,_x_);
        transform_a_to_b_2d_ref(By_fp_y,By,sp_to_cv,_y_);
        transform_a_to_b_2d_ref(Bz_fp_z,Bz,sp_to_cv,_z_);
        #else
        transform_a_to_b_2d(Bx_fp_x,Bx,T_fp_x,sp_to_cv,_x_);
        transform_a_to_b_2d(By_fp_y,By,T_fp_y,sp_to_cv,_y_);
        transform_a_to_b_2d(Bz_fp_z,Bz,T_fp_z,sp_to_cv,_z_);
        #endif
        //We also need contro-volume averages for trouble dection
        compute_B_cv_from_cf(B_old,Bx,By,Bz,fp_to_cv);
    }

    void Transform_to_sp(){
        //Computes the solution at faces from the
        //control-face average
        #ifdef REF_TRANSFORMS
        transform_a_to_b_2d_ref(Bx,Bx_fp_x,cv_to_sp,_x_);
        transform_a_to_b_2d_ref(By,By_fp_y,cv_to_sp,_y_);
        transform_a_to_b_2d_ref(Bz,Bz_fp_z,cv_to_sp,_z_);
        #else
        transform_a_to_b_2d(Bx,Bx_fp_x,T_fp_x,cv_to_sp,_x_);
        transform_a_to_b_2d(By,By_fp_y,T_fp_y,cv_to_sp,_y_);
        transform_a_to_b_2d(Bz,Bz_fp_z,T_fp_z,cv_to_sp,_z_);
        #endif
    }

    void Integrate_edges(int ader){
        #if X
        edge_integral(Ex_ader_ep_yz, Ex, sp_to_cv, ader, _x_);
        #endif
        #if Y
        edge_integral(Ey_ader_ep_zx, Ey, sp_to_cv, ader, _y_);
        #endif
        #if Z
        edge_integral(Ez_ader_ep_xy, Ez, sp_to_cv, ader, _z_);
        #endif
    }

    void FV_Boundaries(CommHelper comm){
        //Communications are done sequentially in different directions
        //to ensure that corners zare properly communicated
        //B_old contains the control volume averages for Bx,By,Bz
        //needed to compute the trouble detection.
        //B_new doesn't need to be communicated as each domain only
        //needs to detect troubles in their active region
        //detected troubles are then shared. 
        #if X
        boundaries(comm,BC_x,B_old,_center_,0);
        boundaries(comm,BC_x,Bx_old,_face_,_x_);
        boundaries(comm,BC_x,By_old,_face_,_y_);
        boundaries(comm,BC_x,Bz_old,_face_,_z_);
        #endif
        #if Y
        boundaries(comm,BC_y,B_old,_center_,0);
        boundaries(comm,BC_y,Bx_old,_face_,_x_);
        boundaries(comm,BC_y,By_old,_face_,_y_);
        boundaries(comm,BC_y,Bz_old,_face_,_z_);
        #endif
        #if Z
        boundaries(comm,BC_z,B_old,_center_,0);
        boundaries(comm,BC_z,Bx_old,_face_,_x_);
        boundaries(comm,BC_z,By_old,_face_,_y_);
        boundaries(comm,BC_z,Bz_old,_face_,_z_);
        #endif
    }
};
