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

void NAD(FV_Solution U_new, FV_Solution U, FV_Solution troubles, double tolerance){
    int Nx = U.Nx;
    int Ny = U.Ny;
    int Nz = U.Nz;
    int nvar = U.n_var-1;
    FV_for_cells_all_NGH(
        double maximum;
        double minimum;
        double u_L;
        double u_R;
        double u_new;
        u_new = U_new.Vector(var,k,j,i);
        maximum = U.Vector(var,k,j,i);
        minimum = U.Vector(var,k,j,i);
        #if X
        u_L = U.Vector(var,k,j,i-1);
        u_R = U.Vector(var,k,j,i+1);
        maximum = max3(u_L,maximum,u_R);
        minimum = min3(u_L,minimum,u_R);
        #endif
        #ifdef Y
        u_L = U.Vector(var,k,j-1,i);
        u_R = U.Vector(var,k,j+1,i);
        maximum = max3(u_L,maximum,u_R);
        minimum = min3(u_L,minimum,u_R);
        #endif
        #ifdef Z
        u_L = U.Vector(var,k-1,j,i);
        u_R = U.Vector(var,k+1,j,i);
        maximum = max3(u_L,maximum,u_R);
        minimum = min3(u_L,minimum,u_R);
        #endif
        minimum -= abs(minimum)*tolerance;
        maximum += abs(maximum)*tolerance;
        if( u_new > maximum || u_new < minimum)
            troubles.Vector(var,k,j,i) = 1;
        else
            troubles.Vector(var,k,j,i) = 0;
        
    );
}


void smooth_extrema(FV_Solution U, FV_Solution alpha, Vector centers, Vector faces, int dim){
    int Nx = U.Nx;
    int Ny = U.Ny;
    int Nz = U.Nz;
    int nvar = U.n_var-1;
    FV_for_cells_all_2NGH(
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
    );
}

void relax_NAD(FV_Solution troubles, FV_Solution alpha_x, FV_Solution alpha_y, FV_Solution alpha_z){
    int Nx = troubles.Nx;
    int Ny = troubles.Ny;
    int Nz = troubles.Nz;
    int nvar = troubles.n_var-1;
    FV_for_cells_NGH(
        double trouble;
        for(int var=0; var<nvar; var++){
        double alpha=1;
        #ifdef X
        alpha = min3(alpha_x.Vector(var,k,j,i-1),alpha_x.Vector(var,k,j,i),alpha_x.Vector(var,k,j,i+1));
        #endif
        #ifdef Y
        alpha = min(alpha,min3(alpha_y.Vector(var,k,j-1,i),alpha_y.Vector(var,k,j,i),alpha_y.Vector(var,k,j+1,i)));
        #endif
        #ifdef Z
        alpha = min(alpha,min3(alpha_z.Vector(var,k-1,j,i),alpha_z.Vector(var,k,j,i),alpha_z.Vector(var,k+1,j,i)));
        #endif
        //alpha==1 -> smooth extrema
        trouble = troubles.Vector(var,k,j,i);
        troubles.Vector(var,k,j,i) = alpha<1 ? trouble : 0; 
        }
        trouble=0;
        for(int var=0; var<nvar; var++)
            trouble = max(trouble,troubles.Vector(var,k,j,i));
        troubles.Vector(nvar,k,j,i) = trouble;
    );
}

void convex_combo(FV_Solution troubles){
    int Nx = troubles.Nx;
    int Ny = troubles.Ny;
    int Nz = troubles.Nz;
    int nvar = troubles.n_var-1;
    FV_for_cells_2NGH(
        double trouble;
        trouble = troubles.Vector(nvar,k,j,i);
        trouble = max(trouble,.75*troubles.Vector(nvar,k,j,i+1));
        trouble = max(trouble,.75*troubles.Vector(nvar,k,j,i-1));
        #if Y
        trouble = max(trouble,.75*troubles.Vector(nvar,k,j+1,i));
        trouble = max(trouble,.75*troubles.Vector(nvar,k,j-1,i));
        #endif
        #if Z
        trouble = max(trouble,.75*troubles.Vector(nvar,k+1,j,i));
        trouble = max(trouble,.75*troubles.Vector(nvar,k-1,j,i));
        #endif
    );
}

void PAD_criteria(FV_Solution W, FV_Solution troubles){
    int Nx = W.Nx;
    int Ny = W.Ny;
    int Nz = W.Nz;
    int nvar = W.n_var-1;
    FV_for_cells_NGH(
        double density;
        double pressure;
        density  = W.Vector(_d_,k,j,i);
        pressure = W.Vector(_p_,k,j,i);
        if(density<rho_min || density>rho_max)
            troubles.Vector(nvar,k,j,i) = 1;
        if(pressure<p_min || pressure>p_max)
            troubles.Vector(nvar,k,j,i) = 1;
    );
}

void detect_troubles(
    FV_Solution W_new,
    FV_Solution W_old,
    FV_Solution troubles,
    #if X
    FV_Solution alpha_x,
    #endif
    #if Y
    FV_Solution alpha_y,
    #endif
    #if Z
    FV_Solution alpha_z,
    #endif
    dimension X_dim,
    dimension Y_dim,
    dimension Z_dim
    ){
    double tolerance = 1E-6;
    NAD(W_new, W_old, troubles, tolerance);
    smooth_extrema(W_new, alpha_x, X_dim.fv_centers, X_dim.fv_faces, _x_);
    smooth_extrema(W_new, alpha_y, Y_dim.fv_centers, Y_dim.fv_faces, _y_);
    smooth_extrema(W_new, alpha_z, Z_dim.fv_centers, Z_dim.fv_faces, _z_);
    relax_NAD(troubles, alpha_x, alpha_y, alpha_z);
    PAD_criteria(W_new, troubles);
}