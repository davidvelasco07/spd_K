#include "spd_k.hpp"

#define NODE nid[_z_],nid[_y_],nid[_x_]
#define INDICES   t_id,var,Nid[_z_],Nid[_y_] ,Nid[_x_],nid[_z_],nid[_y_],nid[_x_]
#define INDICES_L t_id,var,NidL[_z_],NidL[_y_],NidL[_x_],nidL[_z_],nidL[_y_],nidL[_x_]
#define INDICES_R t_id,var,NidR[_z_],NidR[_y_],NidR[_x_],nidR[_z_],nidR[_y_],nidR[_x_]

KOKKOS_INLINE_FUNCTION
void indices(int* N_id, int* n_id, int k, int j, int i, int kk, int jj, int ii, int l, int ll, int dim){
    //Returns the indeces according to the dimension
    N_id[_x_] = (dim == _x_ ? l  : i );
    n_id[_x_] = (dim == _x_ ? ll : ii);
    N_id[_y_] = (dim == _y_ ? l  : j );
    n_id[_y_] = (dim == _y_ ? ll : jj);
    N_id[_z_] = (dim == _z_ ? l  : k );
    n_id[_z_] = (dim == _z_ ? ll : kk);
}

KOKKOS_INLINE_FUNCTION
void indices_n(int* n_id, int k, int j, int i, int l, int dim){
    //Returns the indeces according to the dimension
    n_id[_x_] = dim == _x_ ? l : i;
    n_id[_y_] = dim == _y_ ? l : j;
    n_id[_z_] = dim == _z_ ? l : k;
}

KOKKOS_INLINE_FUNCTION
double sound_speed(double rho, double p){
    double c2 = gmma*p/rho;
    c2 = max(c2,min_c2);
    return sqrt(c2);
}

KOKKOS_INLINE_FUNCTION
void conservatives(double* w, double* u){
    double E_kin=0;
    u[0] = w[0];
    #if X
    u[_vx_] = w[0]*w[_vx_];
    E_kin += w[_vx_]*u[_vx_];
    #endif
    #if Y
    u[_vy_] = w[0]*w[_vy_];
    E_kin += w[_vy_]*u[_vy_];
    #endif
    #if Z
    u[_vz_] = w[0]*w[_vz_];
    E_kin += w[_vz_]*u[_vz_];
    #endif
    u[_e_] = w[_p_]/(gmma-1.)+0.5*E_kin;
}

void compute_conservatives(
    SD_Solution W,
    SD_Solution U
    ){
    int Nx = W.Nx;
    int Ny = W.Ny;
    int Nz = W.Nz;
    int px = W.nx;
    int py = W.ny;
    int pz = W.nz;
    int nader = W.n_ader;
    int nvar = W.n_var;
    SD_for_ader(
        double u[NVAR];
        double w[NVAR];
        int var;
        for(var=0; var<nvar; var++)
            w[var] = W.Vector(t_id,var,k,j,i,kk,jj,ii);
        conservatives(w,u);
        for(var=0; var<nvar; var++)
            U.Vector(t_id,var,k,j,i,kk,jj,ii) = u[var];
    );
}

void compute_conservatives(
    FV_Solution W,
    FV_Solution U
    ){
    int Nx = W.Nx;
    int Ny = W.Ny;
    int Nz = W.Nz;
    int nvar = W.n_var;
    FV_for_cells(
        double u[NVAR];
        double w[NVAR];
        int var;
        for(var=0; var<nvar; var++)
            w[var] = W.Vector(0,var,k,j,i);
        conservatives(w,u);
        for(var=0; var<nvar; var++)
            U.Vector(0,var,k,j,i) = u[var];
    );
}

KOKKOS_INLINE_FUNCTION
void primitives(double* u, double* w){
    double E_kin=0;
    w[0] = u[0];
    #if X
    w[_vx_] = u[_vx_]/u[0];
    E_kin += w[_vx_]*u[_vx_];
    #endif
    #if Y
    w[_vy_] = u[_vy_]/u[0];
    E_kin += w[_vy_]*u[_vy_];
    #endif
    #if Z
    w[_vz_] = u[_vz_]/u[0];
    E_kin += w[_vz_]*u[_vz_];
    #endif
    w[_p_] = (u[_e_]-0.5*E_kin)*(gmma-1.);
}

void compute_primitives(
    SD_Solution U, 
    SD_Solution W
    ){
    int Nx = W.Nx;
    int Ny = W.Ny;
    int Nz = W.Nz;
    int px = W.nx;
    int py = W.ny;
    int pz = W.nz;
    int nader = W.n_ader;
    int nvar = W.n_var;
    SD_for_ader(
        int var;
        double u[NVAR];
        double w[NVAR];
        for(var=0; var<nvar; var++)
            u[var] = U.Vector(t_id,var,k,j,i,kk,jj,ii);
        primitives(u,w);
        for(var=0; var<nvar; var++)
            W.Vector(t_id,var,k,j,i,kk,jj,ii) = w[var];
    );
}

void compute_primitives(
    FV_Solution U,
    FV_Solution W
    ){
    int Nx = U.Nx;
    int Ny = U.Ny;
    int Nz = U.Nz;
    int nvar = U.n_var;
    FV_for_cells(
        double u[10];
        double w[10];
        int var;
        for(var=0; var<nvar; var++)
            u[var] = U.Vector(0,var,k,j,i);
        primitives(u,w);
        for(var=0; var<nvar; var++)
            W.Vector(0,var,k,j,i) = w[var];
    );
}

KOKKOS_INLINE_FUNCTION
void fluxes(double* u, double* w, double* f, int _v1_, int _v2_, int _v3_){
    f[   0] = u[   0]*w[_v1_];
    f[_v1_] = u[_v1_]*w[_v1_] + w[_p_];
    #ifdef _2D_
    f[_v2_] = u[_v2_]*w[_v1_];
    #endif
    #ifdef _3D_
    f[_v3_] = u[_v3_]*w[_v1_];
    #endif
    f[_e_] = (u[_e_]+w[_p_])*w[_v1_];
}

void compute_fluxes(
    SD_Solution U,
    SD_Solution F, 
    int _v1_, 
    int _v2_, 
    int _v3_){
    int Nx = U.Nx;
    int Ny = U.Ny;
    int Nz = U.Nz;
    int px = U.nx;
    int py = U.ny;
    int pz = U.nz;
    int nader = U.n_ader;
    int nvar = U.n_var;
    SD_for_ader(
        int var;
        double u[NVAR];
        double w[NVAR];
        double f[NVAR];
        for(var=0; var<nvar; var++)
            u[var] = U.Vector(t_id,var,k,j,i,kk,jj,ii);
        primitives(u,w);
        fluxes(u,w,f,_v1_,_v2_,_v3_);
        for(var=0; var<nvar; var++)
            F.Vector(t_id,var,k,j,i,kk,jj,ii) = f[var];
    );
}

double compute_dt(
    SD_Solution W,
    double dx, 
    double dy, 
    double dz
    ){
    int Nx = W.Nx;
    int Ny = W.Ny;
    int Nz = W.Nz;
    int px = W.nx;
    int py = W.ny;
    int pz = W.nz;
    SD_min_cells(
        double c_s;
        double c_max=0;
        double dx_min=1;
        double dt_min=1;
        c_s = sound_speed(W.Vector(0,0,k,j,i,kk,jj,ii),W.Vector(0,_p_,k,j,i,kk,jj,ii));
        #if X
        c_max += (abs(W.Vector(0,_vx_,k,j,i,kk,jj,ii)) + c_s);
        dx_min = min(dx_min,dx);
        #endif
        #if Y
        c_max += (abs(W.Vector(0,_vy_,k,j,i,kk,jj,ii)) + c_s);
        dx_min = min(dx_min,dy);
        #endif
        #if Z
        c_max += (abs(W.Vector(0,_vz_,k,j,i,kk,jj,ii)) + c_s);
        dx_min = min(dx_min,dz);
        #endif
        dt_min = CFL*dx_min/c_max/px;
        reduce = reduce < dt_min ? reduce : dt_min; 
    );
    #ifdef MPI
    return MPI_Allreduce(&min_value, &dt, 1, MPI_DOUBLE, MPI_MIN, Comm);
    #else
    return min_value;;
    #endif
}

KOKKOS_INLINE_FUNCTION
void riemann_wind(double *F, double *U_L, double *U_R, double wind){
    for(int var=0; var<NVAR; var++)
        F[var] = 0.5*(U_R[var]+U_L[var])+wind*(U_R[var]-U_L[var]);
}

KOKKOS_INLINE_FUNCTION
void riemann_llf(double *F, double *U_L, double *U_R, int _v1_, int _v2_, int _v3_){
    double c_L,c_R,c_max;
    double W_L[NVAR];
    double W_R[NVAR];
    double F_L[NVAR];
    double F_R[NVAR];
    primitives(U_L, W_L);
    primitives(U_R, W_R);  
    fluxes(U_L,W_L,F_L,_v1_,_v2_,_v3_);
    fluxes(U_R,W_R,F_R,_v1_,_v2_,_v3_);
    c_L = sound_speed(W_L[0],W_L[_p_])+abs(W_L[_v1_]);
    c_R = sound_speed(W_R[0],W_R[_p_])+abs(W_R[_v1_]);
    c_max = max(c_L,c_R);
    for(int var=0; var<NVAR; var++)
        F[var] = 0.5*(F_R[var]+F_L[var])-0.5*c_max*(U_R[var]-U_L[var]);
}

KOKKOS_INLINE_FUNCTION
void riemann_hllc(double *F, double *U_L, double *U_R, int _v1_, int _v2_, int _v3_){
    // F, U_L, U_R are arrays of size (nvar,faces)
    // Input: U_L, U_R
    // Output: F
    // Create local arrays of size nvar to solve Riemann problem at every given face.
    double W_L[NVAR];
    double W_R[NVAR];
    double c_L,c_R,c_max;
    double v_L,v_R,s_L,s_R,rc_L,rc_R;
    double v_star,p_star;
    double r_starL,r_starR,e_starL,e_starR;
    double r_gdv,v_gdv,p_gdv,e_gdv;
    primitives(U_L, W_L);
    primitives(U_R, W_R);

    c_L = sound_speed(W_L[_d_],W_L[_p_])+abs(W_L[_v1_]);
    c_R = sound_speed(W_R[_d_],W_R[_p_])+abs(W_R[_v1_]);
    c_max = max(c_L,c_R);

    v_L = W_L[_v1_];
    v_R = W_R[_v1_];

    //Compute HLL wave speed
    s_L = min(v_L,v_R)-c_max;
    s_R = max(v_L,v_R)+c_max;
    //Compute lagrangian sound speed
    rc_L = W_L[0]*(v_L-s_L);
    rc_R = W_R[0]*(s_R-v_R);
    //Compute acoustic star state
    v_star = (rc_R*v_R + rc_L*v_L + (W_L[_p_]-W_R[_p_]))/(rc_R+rc_L);
    p_star = (rc_R*W_L[_p_] + rc_L*W_R[_p_] + rc_L*rc_R*(v_L-v_R))/(rc_R+rc_L);
    //Left star region variables
    r_starL = W_L[_d_]*(s_L-v_L)/(s_L-v_star);
    e_starL = ((s_L-v_L)*U_L[_e_]-W_L[_p_]*v_L+p_star*v_star)/(s_L-v_star);
    //Right star region variables
    r_starR = W_R[0]*(s_R-v_R)/(s_R-v_star);
    e_starR = ((s_R-v_R)*U_R[_e_]-W_R[_p_]*v_R+p_star*v_star)/(s_R-v_star);
    //If   s_L>0 -> U_gdv = U_L
    //elif v*>0  -> U_gdv = U*_L
    //eilf s_R>0 -> U_gdv = U*_R
    //else       -> U_gnv = U_R
    if(s_L>0){
        r_gdv = W_L[_p_];
        v_gdv = W_L[_v1_];
        p_gdv = W_L[_p_];
        e_gdv = U_L[_e_];
    }
    else if(v_star>0){
        r_gdv = r_starL;
        v_gdv = v_star;
        p_gdv = p_star;
        e_gdv = e_starL;
    }
    else if(s_R>0){
        r_gdv = r_starR;
        v_gdv = v_star;
        p_gdv = p_star;
        e_gdv = e_starR;
    }
    else{
        r_gdv = W_R[_d_];
        v_gdv = W_R[_v1_];
        p_gdv = W_R[_p_];
        e_gdv = U_R[_e_];
    }

    F[_d_]  = r_gdv*v_gdv;
    F[_v1_] = r_gdv*v_gdv*v_gdv + p_gdv;
    #ifdef _2D_
    F[_v2_] = r_gdv*v_gdv*(v_star>0 ? W_L[_v2_] : W_R[_v2_]); 
    #endif
    #ifdef _3D_
    F[_v3_] = r_gdv*v_gdv*(v_star>0 ? W_L[_v3_] : W_R[_v3_]);
    #endif
    F[_p_]  = v_gdv*(e_gdv + p_gdv);
}

void sd_riemann_solver(SD_Solution U, SD_Solution F, int v1, int v2, int v3, int dim){
    //Store fluxes at the interface between elements to then solve the unique flux
    int Nx = U.Nx - (dim==_x_); 
    int Ny = U.Ny - (dim==_y_);
    int Nz = U.Nz - (dim==_z_);
    int px = dim==_x_ ? 1 : U.nx; 
    int py = dim==_y_ ? 1 : U.ny;
    int pz = dim==_z_ ? 1 : U.nz;
    int n = dim==_x_ ? U.nx : (dim==_y_ ? U.ny : U.nz);
    int nader = U.n_ader;
    int nvar = U.n_var;
    //cout<<dim<<": "<<Nx<<","<<Ny<<","<<Nz<<","<<px<<","<<py<<","<<pz<<" "<<n<<" "<<v1<<v2<<v3<<endl;
    SD_for_cells(
        int var;
        double u_L[NVAR];
        double u_R[NVAR];
        double f[NVAR];
        int NidL[3];
        int nidL[3];
        int NidR[3];
        int nidR[3];
        indices(NidL,nidL,k,j,i,kk,jj,ii,(dim==_x_ ? i : (dim==_y_ ? j : k))  ,n-1,dim);
        indices(NidR,nidR,k,j,i,kk,jj,ii,(dim==_x_ ? i : (dim==_y_ ? j : k))+1,  0,dim);
        for(int t_id=0; t_id<nader; t_id++){
            for(var=0;var<nvar;var++){
                u_L[var] = U.Vector(INDICES_L);
                u_R[var] = U.Vector(INDICES_R);
            }
            riemann_llf(f,u_L,u_R,v1,v2,v3);
            for(var=0;var<nvar;var++){
                F.Vector(INDICES_L) = f[var];
                F.Vector(INDICES_R) = f[var];
            }
            ////For diffusive terms
            riemann_wind(f,u_L,u_R,0.5);
            for(var=0;var<nvar;var++){
                U.Vector(INDICES_L) = f[var];
                U.Vector(INDICES_R) = f[var];
            }
        }
    );
}

void sd_rusanov_solver(SD_Solution U, SD_Solution F, int dim){
    //Store fluxes at the interface between elements to then solve the unique flux
    int Nx = U.Nx - (dim==_x_); 
    int Ny = U.Ny - (dim==_y_);
    int Nz = U.Nz - (dim==_z_);
    int px = dim==_x_ ? 1 : U.nx; 
    int py = dim==_y_ ? 1 : U.ny;
    int pz = dim==_z_ ? 1 : U.nz;
    int n = dim==_x_ ? U.nx : (dim==_y_ ? U.ny : U.nz);
    int nader = U.n_ader;
    int nvar = U.n_var;
    SD_for_cells(
        int var;
        double u_L[NVAR];
        double u_R[NVAR];
        double f[NVAR];
        int NidL[3];
        int nidL[3];
        int NidR[3];
        int nidR[3];
        indices(NidL,nidL,k,j,i,kk,jj,ii,(dim==_x_ ? i : (dim==_y_ ? j : k))  ,n-1,dim);
        indices(NidR,nidR,k,j,i,kk,jj,ii,(dim==_x_ ? i : (dim==_y_ ? j : k))+1,  0,dim);
        for(int t_id=0; t_id<nader; t_id++){
            for(var=0;var<nvar;var++){
                u_L[var] = U.Vector(INDICES_R);
                u_R[var] = U.Vector(INDICES_L);
            }
            riemann_wind(f,u_L,u_R,-0.5);
            for(var=0;var<nvar;var++){
                F.Vector(INDICES_L) += f[var];
                F.Vector(INDICES_R) += f[var];
            }
        }
    );
}

void compute_gradient(
    SD_Solution U,
    SD_Solution dU,
    double h,
    Matrix dfp_to_sp,
    int dim){
    int Nx = dU.Nx;
    int Ny = dU.Ny;
    int Nz = dU.Nz;
    int px = dU.nx;
    int py = dU.ny;
    int pz = dU.nz;
    //(solution points)
    int q = dim==_x_ ? px:(dim==_y_ ? py : pz); 
    int nader = dU.n_ader;
    int nvar = dU.n_var;
    //Compute dU:
    SD_for_active_cells_all(
        double du=0;
        //Loop over flux points:
        for(int ll=0; ll<q+1; ll++){
            if(dim==_x_)
                du += U.Vector(t_id,var,k,j,i,kk,jj,ll)*dfp_to_sp(ii,ll)/h;
            else if(dim==_y_)
                du += U.Vector(t_id,var,k,j,i,kk,ll,ii)*dfp_to_sp(jj,ll)/h;
            else if(dim==_z_)
                du += U.Vector(t_id,var,k,j,i,ll,jj,ii)*dfp_to_sp(kk,ll)/h;
        }
        dU.Vector(t_id,var,k,j,i,kk,jj,ii) = du;
    );
}

KOKKOS_INLINE_FUNCTION
void viscous_fluxes(
    double* f,
    double* w,
    double* du1,
    int _v1_,
    #ifdef _2D_
    double* du2,
    int _v2_,
    #endif
    #ifdef _3D_
    double* du3,
    int _v3_,
    #endif
    double nu,
    double beta){
    f[_d_]  = 0;
    f[_e_]  = 0;
    f[_v1_] = 2*nu*du1[_v1_] + beta*du1[_v1_];
    #ifdef _2D_
    f[_v1_] += beta*du2[_v2_];
    f[_v2_] = nu*(du1[_v2_]+du2[_v1_]);
    f[_e_]  += w[_v2_]*f[_v2_];
    #endif
    #ifdef _3D_
    f[_v1_] += beta*du3[_v3_];
    f[_v3_] = nu*(du1[_v3_]+du3[_v1_]);
    f[_e_]  += w[_v3_]*f[_v3_];
    #endif
    f[_e_]  += w[_v1_]*f[_v1_];
}

void compute_viscous_flux(
    SD_Solution U,
    SD_Solution dU1,
    int _v1_,
    #ifdef _2D_
    SD_Solution dU2,
    int _v2_,
    #endif
     #ifdef _3D_
    SD_Solution dU3,
    int _v3_,
    #endif
    Matrix sp_to_fp,
    double nu,
    double beta,
    int dim){
    int Nx = U.Nx;
    int Ny = U.Ny;
    int Nz = U.Nz;
    int px = U.nx;
    int py = U.ny;
    int pz = U.nz;
    //flux points
    int q = dim==_x_ ? px : (dim==_y_ ? py : pz);
    int nader = U.n_ader;
    int nvar  = U.n_var;
    //Se interpolate dU back to flux points
    SD_for_ader(
        double f[NVAR];
        double u[NVAR];
        double w[NVAR];
        double du1[NVAR];
        double du2[NVAR];
        double du3[NVAR];
        int nid[3];
        int id;
        id = (dim==_x_ ? ii : (dim==_y_ ? jj : kk));
        for(int var=0; var<nvar; var++){
            du1[var] = 0;
            du2[var] = 0;
            du3[var] = 0;
            u[var] = U.Vector(t_id,var,k,j,i,kk,jj,ii);
            //Loop over solution points
            for(int ll=0; ll<q-1; ll++){
                indices_n(nid,kk,jj,ii,ll,dim);
                du1[var] += dU1.Vector(t_id,var,k,j,i,NODE)*sp_to_fp(id,ll);
                #ifdef _2D_
                du2[var] += dU2.Vector(t_id,var,k,j,i,NODE)*sp_to_fp(id,ll);
                #endif
                #ifdef _3D_
                du3[var] += dU3.Vector(t_id,var,k,j,i,NODE)*sp_to_fp(id,ll);
                #endif
            }
        }
        primitives(u,w);
        viscous_fluxes(
            f,
            w,
            du1,
            _v1_,
            #ifdef _2D_
            du2,
            _v2_,
            #endif
            #ifdef _3D_
            du3,
            _v3_,
            #endif
            nu,beta);
        for(int var=0; var<nvar; var++)
            U.Vector(t_id,var,k,j,i,kk,jj,ii) = f[var];
    );
}

//////////////////
/// MUSCL-HANCOCK
//////////////////

KOKKOS_INLINE_FUNCTION
double minmod(double S_L, double S_R, double x_L, double x_R){
    double ratio,slope;
    //Compute ratio between slopes SlopeR/SlopeL
    ratio = S_R/S_L;
    //limit the ratio to in (0,1)
    ratio = max(0.0,min(ratio,1.0));
    //mult_p_ly by SlopeL to get the limited slope at the cell center
    slope = ratio*S_L;
    //Compute SlopeC*dx/2                             
    return 0.5*slope*(x_R-x_L);
}

KOKKOS_INLINE_FUNCTION
void corrector(double* W, double* dWt, double* dWx, double* dWy, double* dWz){
    dWt[_d_] = 0;
    dWt[_p_] = 0;
    
    dWt[_d_] -= W[_vx_]*dWx[_d_] + W[_d_]*dWx[_vx_];
    dWt[_vx_] = -W[_vx_]*dWx[_vx_] - dWx[_p_]/W[_d_];
    dWt[_p_] -= W[_vx_]*dWx[_p_] + dWx[_vx_]*gmma*W[_p_];
    #if Y
    dWt[_vx_] -= W[_vy_]*dWy[_vx_];
    #endif
    #ifdef Z
    dWt[_vx_] -= W[_vz_]*dWz[_vx_];
    #endif

    #ifdef Y
    dWt[_d_] -= W[_vy_]*dWx[_d_] + W[_d_]*dWy[_vy_];
    dWt[_vy_] = -W[_vy_]*dWy[_vy_] - dWy[_p_]/W[_d_];
    dWt[_p_] -= W[_vy_]*dWy[_p_] + dWy[_vy_]*gmma*W[_p_];
    dWt[_vy_] -= W[_vx_]*dWx[_vy_];
    #ifdef Z
    dWt[_vy_] -= W[_vz_]*dWz[_vy_];
    #endif
    #endif
                         
    #ifdef Z
    dWt[_d_] -= W[_vz_]*dWz[_d_] + W[_d_]*dWz[_vz_];
    dWt[_vz_] = -W[_vz_]*dWz[_vz_] - dWz[_p_]/W[_d_];
    dWt[_p_] -= W[_vz_]*dWz[_p_] + dWz[_vz_]*gmma*W[_p_];
    dWt[_vz_] -= W[_vx_]*dWx[_vz_];
    dWt[_vz_] -= W[_vy_]*dWy[_vz_];
    #endif
}

KOKKOS_INLINE_FUNCTION
void slopes(
    FV_Vector W,
    Vector x_c,
    Vector x_f,
    #if Y
    Vector y_c,
    Vector y_f,
    #endif
    #if Z
    Vector z_c,
    Vector z_f,
    #endif
    int k,
    int j,
    int i,
    double WL[3][NVAR],
    double WR[3][NVAR],
    double dt){
    
    double wL;
    double wR;
    double h;
    double w[NVAR];
    double dwx[NVAR];
    double dwy[NVAR];
    double dwz[NVAR];
    double dwt[NVAR];
    for(int var=0; var<NVAR; var++){
        w[var] = W(0,var,k,j,i);
        wL=W(0,var,k,j,i-1);
        wR=W(0,var,k,j,i+1);
        h = x_c(i+1)-x_c(i);
        dwx[var] = minmod((wR - w[var])/h,(w[var] - wL)/h,x_f(i),x_f(i+1));
        #if Y
        wL=W(0,var,k,j-1,i);
        wR=W(0,var,k,j+1,i);
        h = y_c(j+1)-y_c(j);
        dwy[var] = minmod((wR - w[var])/h,(w[var] - wL)/h,y_f(j),y_f(j+1));
        #endif
        #if Z
        wL=W(0,var,k-1,j,i);
        wR=W(0,var,k+1,j,i);
        h = z_c(k+1)-z_c(k);
        dwz[var] = minmod((wR - w[var])/h,(w[var] - wL)/h,z_f(k),z_f(k+1));
        #endif
    }
    corrector(w,dwt,dwz,dwy,dwx);
    for(int var=0; var<NVAR; var++){
        h = x_f(i+1)-x_f(i);
        WR[_x_][var] = w[var] - dwx[var] + dwt[var]*dt/h;
        WL[_x_][var] = w[var] + dwx[var] + dwt[var]*dt/h;
        #if Y
        h = y_f(j+1)-y_f(j);
        WR[_y_][var] = w[var] - dwy[var] + dwt[var]*dt/h;
        WL[_y_][var] = w[var] + dwy[var] + dwt[var]*dt/h;
        #endif
        #if Z
        h = z_f(k+1)-z_f(k);
        WR[_z_][var] = w[var] - dwz[var] + dwt[var]*dt/h;
        WL[_z_][var] = w[var] + dwz[var] + dwt[var]*dt/h;
        #endif
    } 
}

KOKKOS_INLINE_FUNCTION
void compute_fluxes(
    FV_Vector W,
    FV_Vector F,
    FV_Vector troubles,
    Vector x_c,
    Vector x_f,
    #if Y
    Vector y_c,
    Vector y_f,
    #endif
    #if Z
    Vector z_c,
    Vector z_f,
    #endif
    int k,
    int j,
    int i,
    double dt,
    int ader,
    int dim
    ){
    int nvar = NVAR-1;
    double wL[3][3][NVAR];
    double wR[3][3][NVAR];
    double uL[NVAR];
    double uR[NVAR];
    double fL[NVAR];
    //double fR[NVAR];
    double f;
    double tr;
    double trL;
    double trR;
    int v1 = dim==_x_ ? _vx_ : (dim==_y_ ? _vy_ : _vz_);
    int v2 = dim==_x_ ? _vy_ : (dim==_y_ ? _vz_ : _vx_);
    int v3 = dim==_x_ ? _vz_ : (dim==_y_ ? _vx_ : _vy_);
    for(int l=-NGH; l<NGH; l++)
        slopes(
            W,
            x_c,
            x_f,
            y_c,
            y_f,
            z_c,
            z_f,
            k + (dim==_z_ ? l:0),
            j + (dim==_y_ ? l:0),
            i + (dim==_x_ ? l:0),
            (wL[l+NGH]),
            (wR[l+NGH]),
            dt);
    //Now we have the reconstructed values at both faces
    //We can then solve the Riemann problem
    //Left Boundary
    conservatives(wL[0][dim],uL);
    conservatives(wR[1][dim],uR);
    riemann_hllc(fL,uL,uR,v1,v2,v3);
    //Right Boundary
    //riemann_llf(fR,uL[1][dim],uR[2][dim],v1,v2,v3);
    for(int var=0; var<NVAR; var++){
        f  = F(ader,var,k,j,i);
        tr  = troubles(0,nvar,k,j,i);
        trL = troubles(0,nvar,k-(dim==_z_ ? 1:0),j-(dim==_y_ ? 1:0),i-(dim==_x_ ? 1:0));
        trR = troubles(0,nvar,k+(dim==_z_ ? 1:0),j+(dim==_y_ ? 1:0),i+(dim==_x_ ? 1:0));
        tr = max(tr,max(trL,trR));
        f  = (1-tr)*f + tr*fL[var];
        F(ader,var,k,j,i) = f;
    }
}

void fallback_fluxes(
    FV_Solution U,
    FV_Solution troubles,
    Vector x_c,
    Vector x_f,
    FV_Solution F_x,
    #if Y
    Vector y_c,
    Vector y_f,
    FV_Solution F_y,
    #endif
    #if Z
    Vector z_c,
    Vector z_f,
    FV_Solution F_z,
    #endif
    int ader,
    Vector w,
    double dt
    ){
    int Nx = U.Nx;
    int Ny = U.Ny;
    int Nz = U.Nz;
    FV_for_cells_2NGH(
        compute_fluxes(
            U.Vector,
            F_x.Vector,
            troubles.Vector,
            x_c,
            x_f,
            #if Y
            y_c,
            y_f,
            #endif
            #if Z
            z_c,
            z_f,
            #endif
            k,
            j,
            i,
            w[ader]*dt,
            ader,
            _x_);
        #if Y
        compute_fluxes(
            U.Vector,
            F_y.Vector,
            troubles.Vector,
            x_c,
            x_f,
            y_c,
            y_f,
            #if Z
            z_c,
            z_f,
            #endif
            k,
            j,
            i,
            w[ader]*dt,
            ader,
            _y_);
        #endif
        #if Z
        compute_fluxes(
            U.Vector,
            F_z.Vector,
            troubles.Vector,
            x_c,
            x_f,
            y_c,
            y_f,
            z_c,
            z_f,
            k,
            j,
            i,
            w[ader]*dt,
            ader,
            _z_);
        #endif
    );
}