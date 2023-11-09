#include "spd_k.hpp"

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
        double u[10];
        double w[10];
        int var;
        for(var=0; var<nvar; var++){
            w[var] = W.Vector(t_id,var,k,j,i,kk,jj,ii);
        }
        conservatives(w,u);
        for(var=0; var<nvar; var++){
            U.Vector(t_id,var,k,j,i,kk,jj,ii) = u[var];
        }
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
        double u[10];
        double w[10];
        for(var=0; var<nvar; var++){
            u[var] = U.Vector(t_id,var,k,j,i,kk,jj,ii);
        }
        primitives(u,w);
        for(var=0; var<nvar; var++){
            W.Vector(t_id,var,k,j,i,kk,jj,ii) = w[var];
        }
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
        double u[10];
        double w[10];
        double f[10];
        for(var=0; var<nvar; var++){
            u[var] = U.Vector(t_id,var,k,j,i,kk,jj,ii);
        }
        primitives(u,w);
        fluxes(u,w,f,_v1_,_v2_,_v3_);
        for(var=0; var<nvar; var++){
            F.Vector(t_id,var,k,j,i,kk,jj,ii) = f[var];
        }
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

template <typename T>
void sd_riemann_solver(T U, T F, int v1, int v2, int v3, int dim){
    //Store fluxes at the interface between elements to then solve the unique flux
    int Nx = U.Nx-dim==0; 
    int Ny = U.Ny-dim==1;
    int Nz = U.Nz-dim==2;
    int px = dim==0 ? 1 : U.nx; 
    int py = dim==1 ? 1 : U.ny;
    int pz = dim==2 ? 1 : U.nz;
    int n = dim==0 ? U.nx : (dim==1 ? U.ny : U.nz);
    int nader = U.n_ader;
    int nvar = U.n_var;
    SD_for_cells(
        int var;
        double u_L[10];
        double u_R[10];
        double f[10];
        for(int t_id=0; t_id<nader; t_id++){
        for(var=0;var<nvar;var++){
            if(dim==0){
                u_L[var] = U.Vector(t_id,var,k,j,i  ,kk,jj,n-1);
                u_R[var] = U.Vector(t_id,var,k,j,i+1,kk,jj,  0);
            }
            else if(dim==1){
                u_L[var] = U.Vector(t_id,var,k,j  ,i,kk,n-1,ii);
                u_R[var] = U.Vector(t_id,var,k,j+1,i,kk,  0,ii);
            }
            else if(dim==2){
                u_L[var] = U.Vector(t_id,var,k  ,j,i,n-1,jj,ii);
                u_R[var] = U.Vector(t_id,var,k+1,j,i,  0,jj,ii);
            }
        }
            riemann_llf(f,u_L,u_R,v1,v2,v3);
        for(var=0;var<nvar;var++){
            if(dim==0){
                F.Vector(t_id,var,k,j,i  ,kk,jj,n-1) = f[var];
                F.Vector(t_id,var,k,j,i+1,kk,jj,  0) = f[var];
            }
            else if(dim==1){
                F.Vector(t_id,var,k,j  ,i,kk,n-1,ii) = f[var];
                F.Vector(t_id,var,k,j+1,i,kk,  0,ii) = f[var];
            }
            else if(dim==2){
                F.Vector(t_id,var,k  ,j,i,n-1,jj,ii) = f[var];
                F.Vector(t_id,var,k+1,j,i,  0,jj,ii) = f[var];
            }
        }
        }
    );
}
