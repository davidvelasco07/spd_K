#include "spd_k.hpp"

KOKKOS_INLINE_FUNCTION
double max3(double a, double b, double c){
    return max(max(a,b),c);
}
KOKKOS_INLINE_FUNCTION
double min3(double a, double b, double c){
    return min(min(a,b),c);
}

KOKKOS_INLINE_FUNCTION
double Alpha(double dv, double dUm, double dU, double dUp){
    double vL,vR;
    double alpha_m,alpha_p,alphaL,alphaR;
    //vL = dU(i-1)-dU(i)
    vL = dU-dUm;
    //alphaL = min(1,-max(vL,0)/dv),1,min(1,-min(vL,0)/dv) for dv<0,dv=0,dv>0
    alpha_p = min(1.0,max(vL,0.0)/dv);
    alpha_m = min(1.0,min(vL,0.0)/dv);
    alphaL = (dv  <  0.0 ? alpha_m : alpha_p);
    alphaL = (dv  == 0.0 ? 1.0 : alphaL);
    //vR = dU(i+1)-dU(i)
    vR = dUp-dU;
    //alphaR = min(1,max(vR,0)/dv),1,min(1,min(vR,0)/dv) for dv>0,dv=0,dv<0
    alpha_p = min(1.0,max(vR,0.0)/dv);
    alpha_m = min(1.0,min(vR,0.0)/dv);
    alphaR = (dv  <  0.0 ? alpha_m : alpha_p);
    alphaR = (dv  == 0.0 ? 1.0 : alphaR);
    return min(alphaL,alphaR);
}

//limit_mask: bit set per variable index included in the NAD/SED checks
//(the reference implementation limits only density and pressure for hydro)
void NAD(FV_Solution U_new, FV_Solution U, FV_Solution troubles, double tolerance, int limit_mask){
    int Nx = U.Nx;
    int Ny = U.Ny;
    int Nz = U.Nz;
    int nvar = U.n_var-1;
    bool ax = cfg.active[_x_];
    bool ay = cfg.active[_y_];
    bool az = cfg.active[_z_];
    bool moore = cfg.nad_moore;
    bool delta_mode = cfg.nad_delta;
    fv_for_cells_ngh(Nz,Ny,Nx, KOKKOS_LAMBDA(int k, int j, int i){
        for(int var=0; var<nvar; var++){
        if(!((limit_mask>>var)&1)) continue;
        double maximum;
        double minimum;
        double u_L;
        double u_R;
        double u_new;
        u_new = U_new.Vector(var,k,j,i);
        maximum = U.Vector(var,k,j,i);
        minimum = U.Vector(var,k,j,i);
        if(moore){
            //DMP bounds over the Moore (box) neighborhood, diagonals included
            for(int dk=-(int)az; dk<=(int)az; dk++)
            for(int dj=-(int)ay; dj<=(int)ay; dj++)
            for(int di=-(int)ax; di<=(int)ax; di++){
                double u = U.Vector(var,k+dk,j+dj,i+di);
                maximum = max(maximum,u);
                minimum = min(minimum,u);
            }
        }
        else{
            //von Neumann (face) neighborhood
            if(ax){
                u_L = U.Vector(var,k,j,i-1);
                u_R = U.Vector(var,k,j,i+1);
                maximum = max3(u_L,maximum,u_R);
                minimum = min3(u_L,minimum,u_R);
            }
            if(ay){
                u_L = U.Vector(var,k,j-1,i);
                u_R = U.Vector(var,k,j+1,i);
                maximum = max3(u_L,maximum,u_R);
                minimum = min3(u_L,minimum,u_R);
            }
            if(az){
                u_L = U.Vector(var,k-1,j,i);
                u_R = U.Vector(var,k+1,j,i);
                maximum = max3(u_L,maximum,u_R);
                minimum = min3(u_L,minimum,u_R);
            }
        }
        if(delta_mode){
            //band scaled by the local solution range
            double eps = tolerance*(maximum-minimum);
            minimum -= eps;
            maximum += eps;
        }
        else{
            minimum -= abs(minimum)*tolerance;
            maximum += abs(maximum)*tolerance;
        }
        if( u_new > maximum || u_new < minimum)
            troubles.Vector(var,k,j,i) = 1;
        else
            troubles.Vector(var,k,j,i) = 0;
        
        }
    });
}


void smooth_extrema(FV_Solution U, FV_Solution alpha, Vector centers, Vector faces, int dim, int limit_mask){
    int Nx = U.Nx;
    int Ny = U.Ny;
    int Nz = U.Nz;
    int nvar = U.n_var-1;
    fv_for_cells_2ngh(Nz,Ny,Nx, KOKKOS_LAMBDA(int k, int j, int i){
        for(int var=0; var<nvar; var++){
        if(!((limit_mask>>var)&1)) continue;
        //double nid[3];
        double u = U.Vector(var,k,j,i);
        double du;
        double dup;
        double dum;  
        double d2u;
        double dv;
        int l = dim==_x_ ? i : (dim==_y_ ?  j : k);
        double h  = (centers(l+1)-centers(l-1));
        double hp = (centers(l+2)-centers(l  ));
        double hm = (centers(l  )-centers(l-2));
        //First derivative 
        if(dim==0){
            du  = (U.Vector(var,k,j,i+1)-U.Vector(var,k,j,i-1))/h ;
            dup = (U.Vector(var,k,j,i+2)-u                    )/hp;
            dum = (u                    -U.Vector(var,k,j,i-2))/hm;
        }
        else if(dim==1){
            du  = (U.Vector(var,k,j+1,i)-U.Vector(var,k,j-1,i))/h ;
            dup = (U.Vector(var,k,j+2,i)-u                    )/hp;
            dum = (u                    -U.Vector(var,k,j-2,i))/hm;
        }
        else if(dim==2){
            du  = (U.Vector(var,k+1,j,i)-U.Vector(var,k-1,j,i))/h ;
            dup = (U.Vector(var,k+2,j,i)-u                    )/hp;
            dum = (u                    -U.Vector(var,k-2,j,i))/hm;
        }
        //Second derivative
        d2u = (dup-dum)/(centers(l+1)-centers(l-1));
        dv  = 0.5*d2u*(faces(l+1)-faces(l));
        alpha.Vector(var,k,j,i) = Alpha(dv,dum,du,dup);
        }
    });
}

//use_sed=false skips the smooth-extrema relaxation (flags pass through)
//and only builds the aggregate slot
void relax_NAD(FV_Solution troubles, FV_Solution alpha_x, FV_Solution alpha_y, FV_Solution alpha_z, int limit_mask, bool use_sed){
    int Nx = troubles.Nx;
    int Ny = troubles.Ny;
    int Nz = troubles.Nz;
    int nvar = troubles.n_var-1;
    bool ax = cfg.active[_x_];
    bool ay = cfg.active[_y_];
    bool az = cfg.active[_z_];
    (void)ax;
    fv_for_cells_ngh(Nz,Ny,Nx, KOKKOS_LAMBDA(int k, int j, int i){
        double trouble;
        for(int var=0; var<nvar; var++){
        if(!((limit_mask>>var)&1)) continue;
        double alpha=1;
        if(ax)
            alpha = min3(alpha_x.Vector(var,k,j,i-1),alpha_x.Vector(var,k,j,i),alpha_x.Vector(var,k,j,i+1));
        if(ay)
            alpha = min(alpha,min3(alpha_y.Vector(var,k,j-1,i),alpha_y.Vector(var,k,j,i),alpha_y.Vector(var,k,j+1,i)));
        if(az)
            alpha = min(alpha,min3(alpha_z.Vector(var,k-1,j,i),alpha_z.Vector(var,k,j,i),alpha_z.Vector(var,k+1,j,i)));
        //alpha==1 -> smooth extrema
        trouble = troubles.Vector(var,k,j,i);
        if(use_sed)
            troubles.Vector(var,k,j,i) = alpha<1 ? trouble : 0;
        }
        trouble=0;
        for(int var=0; var<nvar; var++)
            if((limit_mask>>var)&1)
                trouble = max(trouble,troubles.Vector(var,k,j,i));
        troubles.Vector(nvar,k,j,i) = trouble;
    });
}

//Fractional blending weights following the reference implementation:
//a troubled cell contributes 1 to its own theta, 0.75 to face neighbors,
//0.5 to edge-diagonal neighbors and 0.375 to corner neighbors. A final
//pass adds a 0.25 ring around every positive theta. The blended flux at a
//face is then theta_face*F_MUSCL + (1-theta_face)*F_SD with
//theta_face = max of the two adjacent cells (see fallback compute_fluxes).
void apply_blending(FV_Solution troubles, FV_Solution theta){
    int Nx = troubles.Nx;
    int Ny = troubles.Ny;
    int Nz = troubles.Nz;
    int nvar = troubles.n_var-1; //aggregate slot
    bool ax = cfg.active[_x_];
    bool ay = cfg.active[_y_];
    bool az = cfg.active[_z_];
    fv_for_cells_ngh(Nz,Ny,Nx, KOKKOS_LAMBDA(int k, int j, int i){
        const double w[4] = {1.0, 0.75, 0.5, 0.375};
        double th = 0;
        for(int dk=-(int)az; dk<=(int)az; dk++)
        for(int dj=-(int)ay; dj<=(int)ay; dj++)
        for(int di=-(int)ax; di<=(int)ax; di++){
            int n0 = (di!=0) + (dj!=0) + (dk!=0);
            th = max(th, w[n0]*troubles.Vector(nvar,k+dk,j+dj,i+di));
        }
        theta.Vector(0,k,j,i) = th;
    });
}

//theta = raw trouble aggregate (used when blending is disabled)
void theta_from_troubles(FV_Solution troubles, FV_Solution theta){
    int Nx = troubles.Nx;
    int Ny = troubles.Ny;
    int Nz = troubles.Nz;
    int nvar = troubles.n_var-1;
    fv_for_cells(Nz,Ny,Nx, KOKKOS_LAMBDA(int k, int j, int i){
        theta.Vector(0,k,j,i) = troubles.Vector(nvar,k,j,i);
    });
}

//Adds the 0.25 ring: theta_out = max(theta_in, 0.25*(any box neighbor of
//theta_in positive)). Reads only theta_in, so the result matches the
//reference's single-dilation semantics and is order-independent.
void blending_ring(FV_Solution theta_in, FV_Solution theta_out){
    int Nx = theta_in.Nx;
    int Ny = theta_in.Ny;
    int Nz = theta_in.Nz;
    bool ax = cfg.active[_x_];
    bool ay = cfg.active[_y_];
    bool az = cfg.active[_z_];
    fv_for_cells_ngh(Nz,Ny,Nx, KOKKOS_LAMBDA(int k, int j, int i){
        double th = theta_in.Vector(0,k,j,i);
        double ring = 0;
        for(int dk=-(int)az; dk<=(int)az; dk++)
        for(int dj=-(int)ay; dj<=(int)ay; dj++)
        for(int di=-(int)ax; di<=(int)ax; di++)
            if(theta_in.Vector(0,k+dk,j+dj,i+di) > 0) ring = 0.25;
        theta_out.Vector(0,k,j,i) = max(th, ring);
    });
}

void PAD_criteria(FV_Solution W, FV_Solution troubles){
    int Nx = W.Nx;
    int Ny = W.Ny;
    int Nz = W.Nz;
    int nvar = W.n_var-1;
    fv_for_cells_ngh(Nz,Ny,Nx, KOKKOS_LAMBDA(int k, int j, int i){
        double density;
        double pressure;
        density  = W.Vector(_d_,k,j,i);
        pressure = W.Vector(_p_,k,j,i);
        if(density<rho_min || density>rho_max)
            troubles.Vector(nvar,k,j,i) = 1;
        if(pressure<p_min || pressure>p_max)
            troubles.Vector(nvar,k,j,i) = 1;
    });
}

void detect_troubles(
    FV_Solution W_new,
    FV_Solution W_old,
    FV_Solution troubles,
    FV_Solution alpha_x,
    FV_Solution alpha_y,
    FV_Solution alpha_z,
    dimension X_dim,
    dimension Y_dim,
    dimension Z_dim,
    bool PAD,
    int limit_mask
    ){
    double tolerance = cfg.nad_tolerance;
    //Following the reference, smooth extrema detection only applies for p>1
    //(the SED stencil needs a genuinely high-order candidate solution)
    bool use_sed = cfg.sed && X_dim.p > 1;
    NAD(W_new, W_old, troubles, tolerance, limit_mask);
    if(use_sed){
        if(cfg.active[_x_])
            smooth_extrema(W_new, alpha_x, X_dim.fv_centers, X_dim.fv_faces, _x_, limit_mask);
        if(cfg.active[_y_])
            smooth_extrema(W_new, alpha_y, Y_dim.fv_centers, Y_dim.fv_faces, _y_, limit_mask);
        if(cfg.active[_z_])
            smooth_extrema(W_new, alpha_z, Z_dim.fv_centers, Z_dim.fv_faces, _z_, limit_mask);
    }
    relax_NAD(troubles, alpha_x, alpha_y, alpha_z, limit_mask, use_sed);
    if(PAD)
        PAD_criteria(W_new, troubles);
}