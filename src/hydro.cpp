#include "spd_k.hpp"

KOKKOS_INLINE_FUNCTION
int choose(int dim, int i, int j, int k){
    return (dim==_x_ ? i : (dim==_y_ ?  j : k));
}

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
double sound_speed(double rho, double p, double gm){
    double c2 = gm*p/rho;
    c2 = max(c2,min_c2);
    return sqrt(c2);
}

KOKKOS_INLINE_FUNCTION
void conservatives(double* w, double* u, double gm){
    double E_kin=0;
    u[0] = w[0];
    u[_vx_] = w[0]*w[_vx_];
    E_kin += w[_vx_]*u[_vx_];
    u[_vy_] = w[0]*w[_vy_];
    E_kin += w[_vy_]*u[_vy_];
    u[_vz_] = w[0]*w[_vz_];
    E_kin += w[_vz_]*u[_vz_];
    u[_e_] = w[_p_]/(gm-1.)+0.5*E_kin;
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
    double gm = cfg.gamma;
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        for(int t_id=0; t_id<nader; t_id++){
        double u[NVAR]={0};
        double w[NVAR];
        int var;
        for(var=0; var<nvar; var++)
            w[var] = W.Vector(t_id,var,k,j,i,kk,jj,ii);
        conservatives(w,u,gm);
        for(var=0; var<nvar; var++)
            U.Vector(t_id,var,k,j,i,kk,jj,ii) = u[var];
        }
    });
}

void compute_conservatives(
    FV_Solution W,
    FV_Solution U
    ){
    int Nx = W.Nx;
    int Ny = W.Ny;
    int Nz = W.Nz;
    int nvar = W.n_var;
    double gm = cfg.gamma;
    fv_for_cells(Nz,Ny,Nx, KOKKOS_LAMBDA(int k, int j, int i){
        double u[NVAR]={0};
        double w[NVAR];
        int var;
        for(var=0; var<nvar; var++)
            w[var] = W.Vector(var,k,j,i);
        conservatives(w,u,gm);
        for(var=0; var<nvar; var++)
            U.Vector(var,k,j,i) = u[var];
    });
}

KOKKOS_INLINE_FUNCTION
void primitives(double* u, double* w, double gm){
    double E_kin=0;
    w[0] = u[0];
    w[_vx_] = u[_vx_]/u[0];
    E_kin += w[_vx_]*u[_vx_];
    w[_vy_] = u[_vy_]/u[0];
    E_kin += w[_vy_]*u[_vy_];
    w[_vz_] = u[_vz_]/u[0];
    E_kin += w[_vz_]*u[_vz_];
    w[_p_] = (u[_e_]-0.5*E_kin)*(gm-1.);
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
    double gm = cfg.gamma;
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        for(int t_id=0; t_id<nader; t_id++){
        int var;
        double u[NVAR];
        double w[NVAR]={0};
        for(var=0; var<NVAR; var++)
            u[var] = U.Vector(t_id,var,k,j,i,kk,jj,ii);
        primitives(u,w,gm);
        for(var=0; var<NVAR; var++)
            W.Vector(t_id,var,k,j,i,kk,jj,ii) = w[var];
        }
    });
}

void compute_primitives(
    FV_Solution U,
    FV_Solution W
    ){
    int Nx = U.Nx;
    int Ny = U.Ny;
    int Nz = U.Nz;
    int nvar = U.n_var;
    double gm = cfg.gamma;
    fv_for_cells(Nz,Ny,Nx, KOKKOS_LAMBDA(int k, int j, int i){
        double u[10]={0};
        double w[10]={0};
        int var;
        for(var=0; var<nvar; var++)
            u[var] = U.Vector(var,k,j,i);
        primitives(u,w,gm);
        for(var=0; var<nvar; var++)
            W.Vector(var,k,j,i) = w[var];
    });
}

KOKKOS_INLINE_FUNCTION
void fluxes(double* u, double* w, double* f, int _v1_, int _v2_, int _v3_){
    f[   0] = u[   0]*w[_v1_];
    f[_v1_] = u[_v1_]*w[_v1_] + w[_p_];
    f[_v2_] = u[_v2_]*w[_v1_];
    f[_v3_] = u[_v3_]*w[_v1_];
    f[_e_] = (u[_e_]+w[_p_])*w[_v1_];
}

//Velocity indices as template parameters: with compile-time V1/V2/V3 the
//var loops fully unroll and the u/w/f locals stay in registers instead of
//spilling to local memory (runtime indices force stack-backed arrays)
template<int V1, int V2, int V3>
void compute_fluxes_t(
    SD_Solution U,
    SD_Solution F){
    int Nx = U.Nx;
    int Ny = U.Ny;
    int Nz = U.Nz;
    int px = U.nx;
    int py = U.ny;
    int pz = U.nz;
    int nader = U.n_ader;
    double gm = cfg.gamma;
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        for(int t_id=0; t_id<nader; t_id++){
        int var;
        double u[NVAR];
        double w[NVAR]={0};
        double f[NVAR]={0};
        for(var=0; var<NVAR; var++)
            u[var] = U.Vector(t_id,var,k,j,i,kk,jj,ii);
        primitives(u,w,gm);
        fluxes(u,w,f,V1,V2,V3);
        for(var=0; var<NVAR; var++)
            F.Vector(t_id,var,k,j,i,kk,jj,ii) = f[var];
        }
    });
}

void compute_fluxes(
    SD_Solution U,
    SD_Solution F,
    int _v1_,
    int _v2_,
    int _v3_){
    //Dispatch to the compile-time instantiation (only three cyclic
    //combinations are ever used, one per direction)
    if(_v1_==_vx_)      compute_fluxes_t<_vx_,_vy_,_vz_>(U,F);
    else if(_v1_==_vy_) compute_fluxes_t<_vy_,_vz_,_vx_>(U,F);
    else                compute_fluxes_t<_vz_,_vx_,_vy_>(U,F);
}

double compute_dt(
    SD_Solution W,
    double dx, 
    double dy, 
    double dz,
    double nu
    ){
    int Nx = W.Nx;
    int Ny = W.Ny;
    int Nz = W.Nz;
    int px = W.nx;
    int py = W.ny;
    int pz = W.nz;
    double gm = cfg.gamma;
    double cfl = cfg.cfl;
    bool ax = cfg.active[_x_];
    bool ay = cfg.active[_y_];
    bool az = cfg.active[_z_];
    double min_value = sd_min_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii, double& reduce){
        double c_s;
        double c_max=0;
        double dx_min=1;
        double dt_min=1;
        c_s = sound_speed(W.Vector(0,0,k,j,i,kk,jj,ii),W.Vector(0,_p_,k,j,i,kk,jj,ii),gm);
        if(ax){
            c_max += (abs(W.Vector(0,_vx_,k,j,i,kk,jj,ii)) + c_s);
            dx_min = min(dx_min,dx);
        }
        if(ay){
            c_max += (abs(W.Vector(0,_vy_,k,j,i,kk,jj,ii)) + c_s);
            dx_min = min(dx_min,dy);
        }
        if(az){
            c_max += (abs(W.Vector(0,_vz_,k,j,i,kk,jj,ii)) + c_s);
            dx_min = min(dx_min,dz);
        }
        dt_min = cfl*dx_min/c_max/px;
        //Explicit viscous (diffusion-number) limit on the sub-cell grid,
        //matching the spd reference (sd_scheme.compute_dt): with sub-cell
        //spacing h = dx/(p+1) the diffusion limit is 0.25*h^2/nu. Only active
        //when viscosity is requested at runtime (hydro/nu>0); an inviscid run
        //keeps a pure CFL timestep.
        if(nu>0.0){
            double dx_sub = dx_min/px;
            double dt_visc = 0.25*cfl*dx_sub*dx_sub/nu;
            dt_min = dt_min < dt_visc ? dt_min : dt_visc;
        }
        reduce = reduce < dt_min ? reduce : dt_min;
    });
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
void riemann_llf(double *F, double *U_L, double *U_R, int _v1_, int _v2_, int _v3_, double gm){
    double c_L,c_R,c_max;
    //Zero-initialized: the flux functions only fill the physical variables,
    //any extra bookkeeping slot (NVAR includes the FV trouble flag) would
    //otherwise carry uninitialized stack values into the flux arrays
    double W_L[NVAR]={0};
    double W_R[NVAR]={0};
    double F_L[NVAR]={0};
    double F_R[NVAR]={0};
    primitives(U_L, W_L, gm);
    primitives(U_R, W_R, gm);  
    fluxes(U_L,W_L,F_L,_v1_,_v2_,_v3_);
    fluxes(U_R,W_R,F_R,_v1_,_v2_,_v3_);
    c_L = sound_speed(W_L[0],W_L[_p_],gm)+abs(W_L[_v1_]);
    c_R = sound_speed(W_R[0],W_R[_p_],gm)+abs(W_R[_v1_]);
    c_max = max(c_L,c_R);
    for(int var=0; var<NVAR; var++)
        F[var] = 0.5*(F_R[var]+F_L[var])-0.5*c_max*(U_R[var]-U_L[var]);
}

KOKKOS_INLINE_FUNCTION
void riemann_hllc(double *F, double *U_L, double *U_R, int _v1_, int _v2_, int _v3_, double gm){
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
    primitives(U_L, W_L, gm);
    primitives(U_R, W_R, gm);

    c_L = sound_speed(W_L[_d_],W_L[_p_],gm)+abs(W_L[_v1_]);
    c_R = sound_speed(W_R[_d_],W_R[_p_],gm)+abs(W_R[_v1_]);
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
        r_gdv = W_L[_d_];
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
    F[_v2_] = r_gdv*v_gdv*(v_star>0 ? W_L[_v2_] : W_R[_v2_]);
    F[_v3_] = r_gdv*v_gdv*(v_star>0 ? W_L[_v3_] : W_R[_v3_]);
    F[_p_]  = v_gdv*(e_gdv + p_gdv);
}

//Direction and velocity indices as template parameters (see
//compute_fluxes_t): compile-time indexing keeps the u_L/u_R/f locals in
//registers instead of local memory
template<int D, int V1, int V2, int V3>
void sd_riemann_solver_t(SD_Solution U, SD_Solution F, bool viscous){
    //Store fluxes at the interface between elements to then solve the unique flux
    int Nx = U.Nx - (D==_x_);
    int Ny = U.Ny - (D==_y_);
    int Nz = U.Nz - (D==_z_);
    int px = D==_x_ ? 1 : U.nx;
    int py = D==_y_ ? 1 : U.ny;
    int pz = D==_z_ ? 1 : U.nz;
    int n = choose(D, U.nx, U.ny, U.nz);
    int nader = U.n_ader;
    double gm = cfg.gamma;
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        int var;
        double u_L[NVAR];
        double u_R[NVAR];
        double f[NVAR]={0};
        int NidL[3];
        int nidL[3];
        int NidR[3];
        int nidR[3];
        int l = choose(D,i,j,k);
        indices(NidL,nidL,k,j,i,kk,jj,ii,l  ,n-1,D);
        indices(NidR,nidR,k,j,i,kk,jj,ii,l+1,  0,D);
        for(int t_id=0; t_id<nader; t_id++){
            for(var=0;var<NVAR;var++){
                u_L[var] = U.Vector(INDICES_L);
                u_R[var] = U.Vector(INDICES_R);
            }
            riemann_llf(f,u_L,u_R,V1,V2,V3,gm);
            for(var=0;var<NVAR;var++){
                F.Vector(INDICES_L) = f[var];
                F.Vector(INDICES_R) = f[var];
            }
            if(viscous){
                //Central interface state, consumed only by the diffusive terms
                riemann_wind(f,u_L,u_R,0.5);
                for(var=0;var<NVAR;var++){
                    U.Vector(INDICES_L) = f[var];
                    U.Vector(INDICES_R) = f[var];
                }
            }
        }
    });
}

void sd_riemann_solver(SD_Solution U, SD_Solution F, int v1, int v2, int v3, int dim, bool viscous){
    if(dim==_x_)      sd_riemann_solver_t<_x_,_vx_,_vy_,_vz_>(U,F,viscous);
    else if(dim==_y_) sd_riemann_solver_t<_y_,_vy_,_vz_,_vx_>(U,F,viscous);
    else              sd_riemann_solver_t<_z_,_vz_,_vx_,_vy_>(U,F,viscous);
}

void sd_rusanov_solver(SD_Solution U, SD_Solution F, int dim){
    //Store fluxes at the interface between elements to then solve the unique flux
    int Nx = U.Nx - (dim==_x_); 
    int Ny = U.Ny - (dim==_y_);
    int Nz = U.Nz - (dim==_z_);
    int px = dim==_x_ ? 1 : U.nx; 
    int py = dim==_y_ ? 1 : U.ny;
    int pz = dim==_z_ ? 1 : U.nz;
    int n = choose(dim,U.nx,U.ny,U.nz);
    int nader = U.n_ader;
    int nvar = U.n_var;
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        int var;
        double u_L[NVAR];
        double u_R[NVAR];
        double f[NVAR];
        int NidL[3];
        int nidL[3];
        int NidR[3];
        int nidR[3];
        int l = choose(dim,i,j,k);
        indices(NidL,nidL,k,j,i,kk,jj,ii,l  ,n-1,dim);
        indices(NidR,nidR,k,j,i,kk,jj,ii,l+1,  0,dim);
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
    });
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
    int q = choose(dim,px,py,pz);; 
    int nader = dU.n_ader;
    int nvar = dU.n_var;
    //Compute dU:
    sd_for_active_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        for(int t_id=0; t_id<nader; t_id++){
        for(int var=0; var<nvar; var++){
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
        }}
    });
}

KOKKOS_INLINE_FUNCTION
void viscous_fluxes(
    double* f,
    double* w,
    double* du1,
    int _v1_,
    double* du2,
    int _v2_,
    double* du3,
    int _v3_,
    double nu,
    double beta){
    //Terms belonging to inactive dimensions vanish identically because the
    //corresponding velocity components and gradients are zero
    f[_d_]  = 0;
    f[_e_]  = 0;
    f[_v1_] = 2*nu*du1[_v1_] + beta*du1[_v1_];
    f[_v1_] += beta*du2[_v2_];
    f[_v2_] = nu*(du1[_v2_]+du2[_v1_]);
    f[_e_]  += w[_v2_]*f[_v2_];
    f[_v1_] += beta*du3[_v3_];
    f[_v3_] = nu*(du1[_v3_]+du3[_v1_]);
    f[_e_]  += w[_v3_]*f[_v3_];
    f[_e_]  += w[_v1_]*f[_v1_];
}

void compute_viscous_flux(
    SD_Solution U,
    SD_Solution dU1,
    int _v1_,
    SD_Solution dU2,
    int _v2_,
    SD_Solution dU3,
    int _v3_,
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
    int q = choose(dim,px,py,pz);
    int nader = U.n_ader;
    int nvar  = U.n_var;
    double gm = cfg.gamma;
    //Se interpolate dU back to flux points
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        for(int t_id=0; t_id<nader; t_id++){
        double f[NVAR]={0};
        double u[NVAR];
        double w[NVAR]={0};
        double du1[NVAR];
        double du2[NVAR];
        double du3[NVAR];
        int nid[3];
        int id;
        id = choose(dim,ii,jj,kk);
        for(int var=0; var<nvar; var++){
            du1[var] = 0;
            du2[var] = 0;
            du3[var] = 0;
            u[var] = U.Vector(t_id,var,k,j,i,kk,jj,ii);
            //Loop over solution points
            for(int ll=0; ll<q-1; ll++){
                indices_n(nid,kk,jj,ii,ll,dim);
                du1[var] += dU1.Vector(t_id,var,k,j,i,NODE)*sp_to_fp(id,ll);
                du2[var] += dU2.Vector(t_id,var,k,j,i,NODE)*sp_to_fp(id,ll);
                du3[var] += dU3.Vector(t_id,var,k,j,i,NODE)*sp_to_fp(id,ll);
            }
        }
        primitives(u,w,gm);
        viscous_fluxes(f,w,du1,_v1_,du2,_v2_,du3,_v3_,nu,beta);
        for(int var=0; var<nvar; var++)
            U.Vector(t_id,var,k,j,i,kk,jj,ii) = f[var];
        }
    });
}

//////////////////
/// MUSCL-HANCOCK
//////////////////

//KOKKOS_INLINE_FUNCTION
//double minmod(double S_L, double S_R, double x_L, double x_R){
//    double ratio,slope;
//    //Compute ratio between slopes SlopeR/SlopeL
//    ratio = S_R/S_L;
//    //limit the ratio to in (0,1)
//    ratio = max(0.0,min(ratio,1.0));
//    //mult_p_ly by SlopeL to get the limited slope at the cell center
//    slope = ratio*S_L;
//    //Compute SlopeC*dx/2                             
//    return 0.5*slope*(x_R-x_L);
//}

KOKKOS_INLINE_FUNCTION
void corrector(double* W, double* dWt, double* dWx, double* dWy, double* dWz, double gm){
    //Terms belonging to inactive dimensions vanish identically because the
    //corresponding velocity components and slopes are zero
    dWt[_d_] = 0;
    dWt[_p_] = 0;

    dWt[_d_] -= W[_vx_]*dWx[_d_] + W[_d_]*dWx[_vx_];
    dWt[_vx_] = -W[_vx_]*dWx[_vx_] - dWx[_p_]/W[_d_];
    dWt[_p_] -= W[_vx_]*dWx[_p_] + dWx[_vx_]*gm*W[_p_];
    dWt[_vx_] -= W[_vy_]*dWy[_vx_];
    dWt[_vx_] -= W[_vz_]*dWz[_vx_];

    dWt[_d_] -= W[_vy_]*dWy[_d_] + W[_d_]*dWy[_vy_];
    dWt[_vy_] = -W[_vy_]*dWy[_vy_] - dWy[_p_]/W[_d_];
    dWt[_p_] -= W[_vy_]*dWy[_p_] + dWy[_vy_]*gm*W[_p_];
    dWt[_vy_] -= W[_vx_]*dWx[_vy_];
    dWt[_vy_] -= W[_vz_]*dWz[_vy_];

    dWt[_d_] -= W[_vz_]*dWz[_d_] + W[_d_]*dWz[_vz_];
    dWt[_vz_] = -W[_vz_]*dWz[_vz_] - dWz[_p_]/W[_d_];
    dWt[_p_] -= W[_vz_]*dWz[_p_] + dWz[_vz_]*gm*W[_p_];
    dWt[_vz_] -= W[_vx_]*dWx[_vz_];
    dWt[_vz_] -= W[_vy_]*dWy[_vz_];
}

//MUSCL-Hancock reconstruction of one cell, projected only onto the two
//faces of direction D (the slopes and the Hancock time corrector still use
//all active directions). D is a compile-time parameter so all local arrays
//stay in registers; the arithmetic is identical to reconstructing all
//directions and discarding the unused ones.
template<int D>
KOKKOS_INLINE_FUNCTION
void slopes_d(
    FV_Vector W,
    Vector x_c,
    Vector x_f,
    Vector y_c,
    Vector y_f,
    Vector z_c,
    Vector z_f,
    int k,
    int j,
    int i,
    double* WL,
    double* WR,
    double dt,
    bool ay,
    bool az,
    double gm){

    double wL;
    double wR;
    double h;
    double w[NVAR];
    double dwx[NVAR];
    double dwy[NVAR];
    double dwz[NVAR];
    double dwt[NVAR];
    for(int var=0; var<NVAR; var++){
        w[var] = W(var,k,j,i);
        wL=W(var,k,j,i-1);
        wR=W(var,k,j,i+1);
        //Each one-sided difference uses its own center spacing: the FV
        //sub-grid (Gauss points) is non-uniform, and a shared h breaks the
        //mirror symmetry at reflective walls (spurious wall mass flux)
        dwx[var] = minmod((wR - w[var])/(x_c(i+1)-x_c(i)),
                          (w[var] - wL)/(x_c(i)-x_c(i-1)),x_f(i),x_f(i+1));
        dwy[var] = 0;
        dwz[var] = 0;
        if(ay){
            wL=W(var,k,j-1,i);
            wR=W(var,k,j+1,i);
            dwy[var] = minmod((wR - w[var])/(y_c(j+1)-y_c(j)),
                              (w[var] - wL)/(y_c(j)-y_c(j-1)),y_f(j),y_f(j+1));
        }
        if(az){
            wL=W(var,k-1,j,i);
            wR=W(var,k+1,j,i);
            dwz[var] = minmod((wR - w[var])/(z_c(k+1)-z_c(k)),
                              (w[var] - wL)/(z_c(k)-z_c(k-1)),z_f(k),z_f(k+1));
        }
    }
    corrector(w,dwt,dwx,dwy,dwz,gm);
    for(int var=0; var<NVAR; var++){
        h = (D==_x_ ? x_f(i+1)-x_f(i) : (D==_y_ ? y_f(j+1)-y_f(j) : z_f(k+1)-z_f(k)));
        double dw = (D==_x_ ? dwx[var] : (D==_y_ ? dwy[var] : dwz[var]));
        WR[var] = w[var] - dw + dwt[var]*dt/h;
        WL[var] = w[var] + dw + dwt[var]*dt/h;
    }
}

template<int D>
KOKKOS_INLINE_FUNCTION
void compute_fluxes(
    FV_Vector W,
    FV_Vector F,
    FV_Vector theta,
    Vector x_c,
    Vector x_f,
    Vector y_c,
    Vector y_f,
    Vector z_c,
    Vector z_f,
    int k,
    int j,
    int i,
    double dt,
    int ader,
    bool ay,
    bool az,
    double gm
    ){
    //wL/wR hold only the D-direction faces of the two cells adjacent to the
    //face (l = -1, 0); with D compile-time everything stays in registers
    double wL[2*NGH][NVAR];
    double wR[2*NGH][NVAR];
    double uL[NVAR];
    double uR[NVAR];
    double fL[NVAR];
    double f;
    double th;
    int v1 = choose(D,_vx_,_vy_,_vz_);
    int v2 = choose(D,_vy_,_vz_,_vx_);
    int v3 = choose(D,_vz_,_vx_,_vy_);
    for(int l=-NGH; l<NGH; l++)
        slopes_d<D>(
            W,
            x_c,
            x_f,
            y_c,
            y_f,
            z_c,
            z_f,
            k + (D==_z_ ? l:0),
            j + (D==_y_ ? l:0),
            i + (D==_x_ ? l:0),
            (wL[l+NGH]),
            (wR[l+NGH]),
            dt,
            ay,
            az,
            gm);
    //Now we have the reconstructed values at both faces
    //We can then solve the Riemann problem
    //Left Boundary
    conservatives(wL[0],uL,gm);
    conservatives(wR[1],uR,gm);
    riemann_hllc(fL,uL,uR,v1,v2,v3,gm);
    //Face blend factor: max of the thetas of the two adjacent cells
    //(reference: affected_faces). Convex blend of the primary and the
    //fallback flux; the same value is seen from both sides of the face,
    //which preserves exact conservation.
    th = max(theta(0,k,j,i),
             theta(0,k-(D==_z_ ? 1:0),j-(D==_y_ ? 1:0),i-(D==_x_ ? 1:0)));
    for(int var=0; var<NVAR; var++){
        f  = F(var,k,j,i);
        f  = f + th*(fL[var]-f);
        F(var,k,j,i) = f;
    }
}

void fallback_fluxes(
    FV_Solution U,
    FV_Solution theta,
    Vector x_c,
    Vector x_f,
    FV_Solution F_x,
    Vector y_c,
    Vector y_f,
    FV_Solution F_y,
    Vector z_c,
    Vector z_f,
    FV_Solution F_z,
    int ader,
    Vector w,
    double dt
    ){
    int Nx = U.Nx;
    int Ny = U.Ny;
    int Nz = U.Nz;
    double gm = cfg.gamma;
    bool ay = cfg.active[_y_];
    bool az = cfg.active[_z_];
    fv_for_faces(Nz,Ny,Nx, KOKKOS_LAMBDA(int k, int j, int i){
        compute_fluxes<_x_>(U.Vector,F_x.Vector,theta.Vector,
            x_c,x_f,y_c,y_f,z_c,z_f,
            k,j,i,w[ader]*dt,ader,ay,az,gm);
        if(ay)
            compute_fluxes<_y_>(U.Vector,F_y.Vector,theta.Vector,
                x_c,x_f,y_c,y_f,z_c,z_f,
                k,j,i,w[ader]*dt,ader,ay,az,gm);
        if(az)
            compute_fluxes<_z_>(U.Vector,F_z.Vector,theta.Vector,
                x_c,x_f,y_c,y_f,z_c,z_f,
                k,j,i,w[ader]*dt,ader,ay,az,gm);
    });
}