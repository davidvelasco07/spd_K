using namespace std;

struct Hydro_muscl{
    int n_output;
    int n_step;
    int nvar;

    double t;
    double dt;
    double Dt;

    FV_Solution U_old;
    FV_Solution U_new;
    FV_Solution W_cv;

    FV_Solution UL;
    FV_Solution UR;
    FV_Solution dUdt;      

    #if X
    FV_Solution F_x;
    FV_Solution dUdx;
    FV_Boundaries BC_x;
    #endif
    #if Y
    FV_Solution F_y;
    FV_Solution dUdy;
    FV_Boundaries BC_y;
    #endif
    #if Z
    FV_Solution F_z;
    FV_Solution dUdz;
    FV_Boundaries BC_z;
    #endif    

    Hydro_muscl(
        CommHelper comm,
        FV_dimension X_dim,
        FV_dimension Y_dim,
        FV_dimension Z_dim,
    ){
        //Number of variables, tipically (rho,vx,vy,vz,e,Vrho)
        nvar = 3 + DIM;
        n_output = 0;
        t=0;

        U_old.init("U_old",nvar,Z_dim,Y_dim,X_dim,0,0,0);
        U_new.init("U_new",nvar,Z_dim,Y_dim,X_dim,0,0,0);
        W_cv.init ( "W_cv",nvar,Z_dim,Y_dim,X_dim,0,0,0);
        dUdt.init ( "dUdt",nvar,Z_dim,Y_dim,X_dim,0,0,0);

        UL.init ( "UL",nvar,Z_dim,Y_dim,X_dim,1,1,1);
        UR.init ( "UR",nvar,Z_dim,Y_dim,X_dim,1,1,1);        
   
        #if X
        F_x.init ( "F_x",nvar,Z_dim,Y_dim,X_dim,0,0,1);
        dUdx.init("dUdx",nvar,Z_dim,Y_dim,X_dim,0,0,1);
        BC_x.init(nvar,Z_dim.N_total,Y_dim.N_total,NGH);
        #endif
        #if Y
        F_y.init ( "F_y",nvar,Z_dim,Y_dim,X_dim,0,1,0);
        dUdy.init("dUdy",nvar,Z_dim,Y_dim,X_dim,0,1,0);
        BC_y.init(nvar,Z_dim.N_total,NGH,X_dim.N_total);
        #endif
        #if Z
        F_z.init ( "F_z",nvar,Z_dim,Y_dim,X_dim,1,0,0);
        dUdz.init("dUdz",nvar,Z_dim,Y_dim,X_dim,1,0,0);
        BC_z.init(nvar,NGH,Y_dim.N_total,X_dim.N_total);
        #endif
    }

    void initial_conditions(){
        Initialize(W_cv,X_dim.centers,Y_dim.centers,Z_dim.centers);
        compute_conservatives(W_cv,U_old);
        Dt = compute_dt(W_cv,X_dim.h,Y_dim.h,Z_dim.h);
        if(Master)
            cout<<"dx = "<<X_dim.h<<" dt = "<<Dt<<endl;
        Write_outputs();
    }

    void Initialize(FV_Solution W, Vector Xs, Vector Ys, Vector Zs){
        int Nx = W.Nx;
        int Ny = W.Ny;
        int Nz = W.Nz;
        int px = W.nx;
        int py = W.ny;
        int pz = W.nz;
        int nvar = W.n_var;
        FV_for_variables(
            W.Vector(var,k,j,i) = square(dim,Xs(i),Ys(j),Zs(k));
        );
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
            MUSCL_HANCOCK();
            Boundaries(comm);
            Riemann_Solver();    
            Update_solution();
            t+=dt;
            n_step++;
            dt=compute_dt(W_cv,X_dim.h,Y_dim.h,Z_dim.h);
            //Outputs
            if(Master)cout<<".";
            if(t>=t_output){
                t_output=t+dt_output;
                Write_outputs();
            }
            if(t+dt>t_output)dt=t_output-t;
            
        }
        cout<<endl;
    }

    void MUSCL_HANCOCK(){
        #ifdef X
        derivative(W_cv,U_L,_x_);
        limit_slope(W_cv,U_L,dUdx,_x_);
        #endif
        #ifdef Y
        derivative(W_cv,U_L,_y_);
        limit_slope(W_cv,U_L,dUdy,_y_);
        #endif
        #ifdef Z
        derivative(W_cv,U_L,_z_);
        limit_slope(W_cv,U_L,dUdz,_z_);
        #endif
        //compute_prediction(dUdt,W_cv,dUdx,dUdy,dUdz);
    }
    
    void Boundaries(CommHelper comm){
        //Communications are done sequentially in different directions
        //to ensure that corners zare properly communicated
        #if X
        boundaries_x(comm,BC_x,U_old);
        #endif
        #if Y
        boundaries_y(comm,BC_y,U_old);
        #endif
        #if Z
        boundaries_z(comm,BC_y,U_old);
        #endif
    }

    void Riemann_Solver(){
        #ifdef X
        interpolate(W_cv, dUdx, dUdt, U_L, U_R, _x_);
        riemann_llf(F_x,U_L,U_R,_vx_,_vy_,_vz_);
        #endif
        #ifdef Y
        interpolate(W_cv, dUdy, dUdt, U_L, U_R, _y_);
        riemann_llf(F_y,U_L,U_R,_vy_,_vx_,_vz_);
        #endif
        #ifdef Z
        interpolate(W_cv, dUdz, dUdt, U_L, U_R, _z_);
        riemann_llf(F_z,U_L,U_R,_vz_,_vx_,_vy_);
        #endif
    }

    void Update_Solution(){
        fv_update_solution(
            U_new,
            U_old,
            #if X
            F_x,
            faces_x,
            #endif
            #if Y
            F_y,
            faces_y,
            #endif
            #if Z
            F_z,
            faces_z,
            #endif
            dt);
    }

    void Write_outputs(){
        if(Master)
            cout<<endl<<"OUTPUT "<<n_output<<endl;
        Write(W_cv,n_output++);
    }
};
