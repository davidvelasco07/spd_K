#include "spd_k.hpp"

//========================================================================================
// Ideal-MHD equation kernels + constrained-transport coupling.
//
// Ported from the Python reference (spd/MHD/mhd.py and
// spd/induction/induction_sd_scheme.py). The 8-variable cell-centered state is
//   [rho, vx, vy, vz, P/E, Bx, By, Bz]   (see mhd.hpp for the index macros)
// and the divergence-free field also lives on the cell faces (Bx_fp_x, ...).
//========================================================================================

KOKKOS_INLINE_FUNCTION
int mhd_choose(int dim, int i, int j, int k){
    return (dim==_x_ ? i : (dim==_y_ ? j : k));
}

KOKKOS_INLINE_FUNCTION
void mhd_indices(int* N_id, int* n_id, int k, int j, int i, int kk, int jj, int ii, int l, int ll, int dim){
    N_id[_x_] = dim == _x_ ? l  : i;
    n_id[_x_] = dim == _x_ ? ll : ii;
    N_id[_y_] = dim == _y_ ? l  : j;
    n_id[_y_] = dim == _y_ ? ll : jj;
    N_id[_z_] = dim == _z_ ? l  : k;
    n_id[_z_] = dim == _z_ ? ll : kk;
}

KOKKOS_INLINE_FUNCTION
int mhd_indices_n(int* n_id, int k, int j, int i, int l, int dim){
    n_id[_x_] = dim == _x_ ? l : i;
    n_id[_y_] = dim == _y_ ? l : j;
    n_id[_z_] = dim == _z_ ? l : k;
    return mhd_choose(dim,i,j,k);
}

//----------------------------------------------------------------------------------------
// Thermodynamics / wave speeds
//----------------------------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION
double mhd_cs2(double p, double rho, double gm){
    double c2 = gm*p/rho;
    return c2 > min_c2 ? c2 : min_c2;
}

// Fast magnetosonic speed (RAMSES-style floors, cf. mhd.compute_fast_vel).
KOKKOS_INLINE_FUNCTION
double mhd_fast_vel(double p, double rho, double Bn, double Bt1, double Bt2, double gm){
    rho = rho > rho_min ? rho : rho_min;
    double B2 = Bn*Bn + Bt1*Bt1 + Bt2*Bt2;
    double c2 = mhd_cs2(p,rho,gm);
    double d2 = 0.5*(B2/rho + c2);
    double disc = d2*d2 - c2*Bn*Bn/rho;
    disc = disc > 0 ? disc : 0;
    return sqrt(d2 + sqrt(disc));
}

//----------------------------------------------------------------------------------------
// Primitive <-> conservative
//----------------------------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION
void mhd_conservatives(double* w, double* u, double gm){
    u[_mrho_] = w[_mrho_];
    u[_mvx_]  = w[_mrho_]*w[_mvx_];
    u[_mvy_]  = w[_mrho_]*w[_mvy_];
    u[_mvz_]  = w[_mrho_]*w[_mvz_];
    u[_mbx_]  = w[_mbx_];
    u[_mby_]  = w[_mby_];
    u[_mbz_]  = w[_mbz_];
    double Ekin = 0.5*w[_mrho_]*(w[_mvx_]*w[_mvx_]+w[_mvy_]*w[_mvy_]+w[_mvz_]*w[_mvz_]);
    double Emag = 0.5*(w[_mbx_]*w[_mbx_]+w[_mby_]*w[_mby_]+w[_mbz_]*w[_mbz_]);
    u[_mprs_] = w[_mprs_]/(gm-1.) + Ekin + Emag;
}

// Cons-to-prim with primitive-view floors (density at dfl, pressure at pfl).
// The conserved state u is NOT touched (RAMSES ctoprim semantics; matches the
// Python spd implementation bit for bit when dfl/pfl take the RAMSES
// defaults). Returns true when a floor fired so callers that own the evolved
// state can optionally repair it (AthenaK floor semantics, see
// mhd_floor_cons).
KOKKOS_INLINE_FUNCTION
bool mhd_primitives(double* u, double* w, double gm,
                    double dfl = rho_min, double pfl = -1.0){
    double rho = u[_mrho_];
    w[_mrho_] = rho;
    w[_mvx_]  = u[_mvx_]/rho;
    w[_mvy_]  = u[_mvy_]/rho;
    w[_mvz_]  = u[_mvz_]/rho;
    w[_mbx_]  = u[_mbx_];
    w[_mby_]  = u[_mby_];
    w[_mbz_]  = u[_mbz_];
    double Ekin = 0.5*rho*(w[_mvx_]*w[_mvx_]+w[_mvy_]*w[_mvy_]+w[_mvz_]*w[_mvz_]);
    double Emag = 0.5*(w[_mbx_]*w[_mbx_]+w[_mby_]*w[_mby_]+w[_mbz_]*w[_mbz_]);
    w[_mprs_] = (u[_mprs_] - Ekin - Emag)*(gm-1.);
    if(pfl < 0) pfl = dfl*min_c2/gm;   // RAMSES smallp default
    bool fired = false;
    if(w[_mrho_] < dfl){ w[_mrho_] = dfl; fired = true; }
    if(w[_mprs_] < pfl){ w[_mprs_] = pfl; fired = true; }
    return fired;
}

// AthenaK floor semantics: rebuild the conserved state from the floored
// primitives (momenta and B are kept; density and total energy are repaired).
// A cell whose total energy implies negative internal energy is amputated
// once instead of carrying the energy debt forever: with primitive-only
// floors the debt persists in u_E, the cell keeps zero effective pressure
// gradient, and degenerate low-beta regions grow while the fast speed
// collapses the time step. Only meaningful with a physically sensible
// pressure floor (hydro/pfloor): repairing to the RAMSES smallp (~1e-24)
// creates zero-pressure cells that are themselves unstable.
KOKKOS_INLINE_FUNCTION
void mhd_floor_cons(double* w, double* u, double gm){
    u[_mrho_] = w[_mrho_];
    double Ekin = 0.5*w[_mrho_]*(w[_mvx_]*w[_mvx_]+w[_mvy_]*w[_mvy_]+w[_mvz_]*w[_mvz_]);
    double Emag = 0.5*(w[_mbx_]*w[_mbx_]+w[_mby_]*w[_mby_]+w[_mbz_]*w[_mbz_]);
    u[_mprs_] = w[_mprs_]/(gm-1.) + Ekin + Emag;
}

void mhd_compute_conservatives(SD_Solution W, SD_Solution U){
    int Nx=W.Nx, Ny=W.Ny, Nz=W.Nz, px=W.nx, py=W.ny, pz=W.nz;
    int nader=W.n_ader;
    double gm=cfg.gamma;
#ifdef KOKKOS_ENABLE_CUDA
    SD_Vector_h Wh = Kokkos::create_mirror_view(W.Vector);
    SD_Vector_h Uh = Kokkos::create_mirror_view(U.Vector);
    Kokkos::deep_copy(Wh, W.Vector);
    sd_for_cells_host(Nz,Ny,Nx,pz,py,px, [&](int k,int j,int i,int kk,int jj,int ii){
        for(int t_id=0;t_id<nader;t_id++){
            double w[NMHD], u[NMHD];
            for(int var=0;var<NMHD;var++) w[var]=Wh(t_id,var,k,j,i,kk,jj,ii);
            mhd_conservatives(w,u,gm);
            for(int var=0;var<NMHD;var++) Uh(t_id,var,k,j,i,kk,jj,ii)=u[var];
        }
    });
    Kokkos::deep_copy(U.Vector, Uh);
#else
    SD_Vector Vw = W.Vector;
    SD_Vector Vu = U.Vector;
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii){
        for(int t_id=0;t_id<nader;t_id++){
            double w[NMHD], u[NMHD];
            for(int var=0;var<NMHD;var++) w[var]=Vw(t_id,var,k,j,i,kk,jj,ii);
            mhd_conservatives(w,u,gm);
            for(int var=0;var<NMHD;var++) Vu(t_id,var,k,j,i,kk,jj,ii)=u[var];
        }
    });
#endif
}

void mhd_compute_primitives(SD_Solution U, SD_Solution W){
    int Nx=W.Nx, Ny=W.Ny, Nz=W.Nz, px=W.nx, py=W.ny, pz=W.nz;
    int nader=W.n_ader;
    double gm=cfg.gamma;
    double dfl=cfg.dfloor, pfl=cfg.pfloor;
#ifdef KOKKOS_ENABLE_CUDA
    SD_Vector_h Uh = Kokkos::create_mirror_view(U.Vector);
    SD_Vector_h Wh = Kokkos::create_mirror_view(W.Vector);
    Kokkos::deep_copy(Uh, U.Vector);
    sd_for_cells_host(Nz,Ny,Nx,pz,py,px, [&](int k,int j,int i,int kk,int jj,int ii){
        for(int t_id=0;t_id<nader;t_id++){
            double u[NMHD], w[NMHD];
            for(int var=0;var<NMHD;var++) u[var]=Uh(t_id,var,k,j,i,kk,jj,ii);
            mhd_primitives(u,w,gm,dfl,pfl);
            for(int var=0;var<NMHD;var++) Wh(t_id,var,k,j,i,kk,jj,ii)=w[var];
        }
    });
    Kokkos::deep_copy(W.Vector, Wh);
#else
    SD_Vector Vu = U.Vector;
    SD_Vector Vw = W.Vector;
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii){
        for(int t_id=0;t_id<nader;t_id++){
            double u[NMHD], w[NMHD];
            for(int var=0;var<NMHD;var++) u[var]=Vu(t_id,var,k,j,i,kk,jj,ii);
            mhd_primitives(u,w,gm,dfl,pfl);
            for(int var=0;var<NMHD;var++) Vw(t_id,var,k,j,i,kk,jj,ii)=w[var];
        }
    });
#endif
}

void mhd_compute_primitives(FV_Solution U, FV_Solution W){
    int Nx=U.Nx, Ny=U.Ny, Nz=U.Nz;
    double gm=cfg.gamma;
    double dfl=cfg.dfloor, pfl=cfg.pfloor;
    fv_for_cells(Nz,Ny,Nx, KOKKOS_LAMBDA(int k,int j,int i){
        double u[NMHD], w[NMHD];
        for(int var=0;var<NMHD;var++) u[var]=U.Vector(var,k,j,i);
        mhd_primitives(u,w,gm,dfl,pfl);
        for(int var=0;var<NMHD;var++) W.Vector(var,k,j,i)=w[var];
    });
}

// AthenaK floor semantics at the correct granularity for an SD scheme: repair
// the committed CELL AVERAGES (sub-cell control volumes) of the MOOD update.
// The gas pressure is measured against the candidate CT field average (B_ct
// rows 0..2 of B_cv), which is what B_to_U will project onto the state; when
// it falls below hydro/pfloor the cell's total energy is rebuilt from the
// floored pressure + kinetic + CT magnetic energy. Density is floored at
// hydro/dfloor (momenta kept). Total energy conservation is sacrificed in the
// repaired cells -- the standard trade-off of AthenaK-style floors.
void mhd_floor_cv(SD_Solution U_cv, FV_Solution B_cv){
    double gm=cfg.gamma, dfl=cfg.dfloor, pfl=cfg.pfloor;
    if(pfl < 0) pfl = dfl*min_c2/gm;
    int Nx=U_cv.Nx, Ny=U_cv.Ny, Nz=U_cv.Nz, px=U_cv.nx, py=U_cv.ny, pz=U_cv.nz;
    int qx=px, qy=py, qz=pz;
    bool az=cfg.active[_z_];
    GHOST_LOCALS;
    sd_for_active_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii){
        double rho = U_cv.Vector(0,_mrho_,k,j,i,kk,jj,ii);
        if(rho < dfl){ rho = dfl; U_cv.Vector(0,_mrho_,k,j,i,kk,jj,ii) = rho; }
        double mx=U_cv.Vector(0,_mvx_,k,j,i,kk,jj,ii);
        double my=U_cv.Vector(0,_mvy_,k,j,i,kk,jj,ii);
        double mz=U_cv.Vector(0,_mvz_,k,j,i,kk,jj,ii);
        double Bx=B_cv.Vector(0,K,J,I), By=B_cv.Vector(1,K,J,I);
        //2D: Bz is a cell-centered conserved variable, not a CT row
        double Bz=az ? B_cv.Vector(2,K,J,I) : U_cv.Vector(0,_mbz_,k,j,i,kk,jj,ii);
        double Ekin = 0.5*(mx*mx+my*my+mz*mz)/rho;
        double Emag = 0.5*(Bx*Bx+By*By+Bz*Bz);
        double p = (U_cv.Vector(0,_mprs_,k,j,i,kk,jj,ii) - Ekin - Emag)*(gm-1.);
        if(p < pfl)
            U_cv.Vector(0,_mprs_,k,j,i,kk,jj,ii) = pfl/(gm-1.) + Ekin + Emag;
    });
}

//----------------------------------------------------------------------------------------
// Physical fluxes (normal direction with velocity index v1, field index b1)
//----------------------------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION
void mhd_fluxes(double* w, double* f, int v1, int v2, int v3, int b1, int b2, int b3, double gm){
    double rho = w[_mrho_];
    double vn  = w[v1];
    double Bn  = w[b1];
    double B2  = w[_mbx_]*w[_mbx_]+w[_mby_]*w[_mby_]+w[_mbz_]*w[_mbz_];
    double v2sq= w[_mvx_]*w[_mvx_]+w[_mvy_]*w[_mvy_]+w[_mvz_]*w[_mvz_];
    double vdotB = w[_mvx_]*w[_mbx_]+w[_mvy_]*w[_mby_]+w[_mvz_]*w[_mbz_];
    double pT  = w[_mprs_] + 0.5*B2;
    double E   = w[_mprs_]/(gm-1.) + 0.5*rho*v2sq + 0.5*B2;
    double m   = rho*vn;
    f[_mrho_] = m;
    f[v1] = m*vn + pT - Bn*Bn;
    f[v2] = m*w[v2] - Bn*w[b2];
    f[v3] = m*w[v3] - Bn*w[b3];
    f[_mprs_] = vn*(E + pT) - Bn*vdotB;
    f[b1] = 0.0;
    f[b2] = vn*w[b2] - w[v2]*Bn;
    f[b3] = vn*w[b3] - w[v3]*Bn;
}

template<int V1,int V2,int V3,int B1,int B2,int B3>
void mhd_compute_fluxes_t(SD_Solution U, SD_Solution F){
    int Nx=U.Nx, Ny=U.Ny, Nz=U.Nz, px=U.nx, py=U.ny, pz=U.nz;
    int nader=U.n_ader;
    double gm=cfg.gamma;
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii){
        for(int t_id=0;t_id<nader;t_id++){
            double u[NMHD], w[NMHD], f[NMHD];
            for(int var=0;var<NMHD;var++) u[var]=U.Vector(t_id,var,k,j,i,kk,jj,ii);
            mhd_primitives(u,w,gm);
            mhd_fluxes(w,f,V1,V2,V3,B1,B2,B3,gm);
            for(int var=0;var<NMHD;var++) F.Vector(t_id,var,k,j,i,kk,jj,ii)=f[var];
        }
    });
}

// U holds conservatives at flux points; F is filled with the conservative fluxes.
void mhd_compute_fluxes(SD_Solution U, SD_Solution F, int dim){
    if(dim==_x_)      mhd_compute_fluxes_t<_mvx_,_mvy_,_mvz_,_mbx_,_mby_,_mbz_>(U,F);
    else if(dim==_y_) mhd_compute_fluxes_t<_mvy_,_mvz_,_mvx_,_mby_,_mbz_,_mbx_>(U,F);
    else              mhd_compute_fluxes_t<_mvz_,_mvx_,_mvy_,_mbz_,_mbx_,_mby_>(U,F);
}

//----------------------------------------------------------------------------------------
// Face Riemann solvers (U carries conservative state at the flux points)
//
// LLF and Miyoshi–Kusano HLLD (ported from spd/MHD/riemann_solver.py). The outer
// wave-speed estimate is eq. (67): S_L/R = min/max(u_L,u_R) ∓ max(c_L,c_R). The normal
// field inside the fan is the average of the interface traces; the low-order FV path
// hands in identical traces taken from the CT face field, so there it is exactly the
// staggered value (see mhd_fv_fluxes_t). The SD path still averages its two traces.
//----------------------------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION
double mhd_sgn(double x){ return (x>0.0) - (x<0.0); }

KOKKOS_INLINE_FUNCTION
void mhd_riemann_llf(double* f, double* uL, double* uR,
                     int v1, int v2, int v3, int b1, int b2, int b3, double gm){
    double wL[NMHD], wR[NMHD], fL[NMHD], fR[NMHD];
    mhd_primitives(uL,wL,gm);
    mhd_primitives(uR,wR,gm);
    mhd_fluxes(wL,fL,v1,v2,v3,b1,b2,b3,gm);
    mhd_fluxes(wR,fR,v1,v2,v3,b1,b2,b3,gm);
    double cL = fabs(wL[v1]) + mhd_fast_vel(wL[_mprs_],wL[_mrho_],wL[b1],wL[b2],wL[b3],gm);
    double cR = fabs(wR[v1]) + mhd_fast_vel(wR[_mprs_],wR[_mrho_],wR[b1],wR[b2],wR[b3],gm);
    double cmax = cL>cR ? cL : cR;
    for(int var=0;var<NMHD;var++)
        f[var] = 0.5*(fR[var]+fL[var]) - 0.5*cmax*(uR[var]-uL[var]);
}

KOKKOS_INLINE_FUNCTION
void mhd_hlld_star_factors(double rho, double u, double S, double S_M, double Bn,
                           double& fac_v, double& fac_b){
    double dSu = S - u;
    double denom = rho*dSu*(S - S_M) - Bn*Bn;
    double scale = rho*dSu*dSu + Bn*Bn;
    if(fabs(denom) > 1e-12*scale){
        fac_v = Bn*(S_M - u)/denom;
        fac_b = (rho*dSu*dSu - Bn*Bn)/denom;
    }else{
        fac_v = 0.0;
        fac_b = 1.0;
    }
}

// Build U* on one side of the contact (Miyoshi & Kusano eqs. 43-48).
KOKKOS_INLINE_FUNCTION
void mhd_hlld_star_state(const double* W, const double* U,
                         int v1, int v2, int v3, int b1, int b2, int b3,
                         double rho, double u, double pT, double rho_s,
                         double S, double S_M, double pT_s, double Bn,
                         double* U_s, double& vt1_s, double& vt2_s,
                         double& Bt1_s, double& Bt2_s, double& vB_s){
    double fac_v, fac_b;
    mhd_hlld_star_factors(rho,u,S,S_M,Bn,fac_v,fac_b);
    for(int var=0;var<NMHD;var++) U_s[var]=U[var];
    vt1_s = W[v2] - fac_v*W[b2];
    vt2_s = W[v3] - fac_v*W[b3];
    Bt1_s = fac_b*W[b2];
    Bt2_s = fac_b*W[b3];
    U_s[_mrho_] = rho_s;
    U_s[v1] = rho_s*S_M;
    U_s[v2] = rho_s*vt1_s;
    U_s[v3] = rho_s*vt2_s;
    U_s[b1] = Bn;
    U_s[b2] = Bt1_s;
    U_s[b3] = Bt2_s;
    double vB = u*Bn + W[v2]*W[b2] + W[v3]*W[b3];
    vB_s = S_M*Bn + vt1_s*Bt1_s + vt2_s*Bt2_s;
    U_s[_mprs_] = ((S - u)*U[_mprs_] - pT*u + pT_s*S_M + Bn*(vB - vB_s))/(S - S_M);
}

// UCT-HLL face coefficients (Mignone & Del Zanna 2021): aL, dL=dR, vt1, vt2.
KOKKOS_INLINE_FUNCTION
void mhd_uct_hll_coeffs(double S_L, double S_R,
                        double vt1_L, double vt1_R, double vt2_L, double vt2_R,
                        double* uct){
    double alpha_r = S_R>0.0 ? S_R : 0.0;
    double alpha_l = S_L<0.0 ? -S_L : 0.0;
    double isum = 1.0/(alpha_r + alpha_l + 1e-100);
    uct[0] = alpha_r*isum;
    double dval = alpha_r*alpha_l*isum;
    uct[1] = dval;
    uct[2] = dval;
    uct[3] = (alpha_r*vt1_L + alpha_l*vt1_R)*isum;
    uct[4] = (alpha_r*vt2_L + alpha_l*vt2_R)*isum;
}

// UCT-HLLD face coefficients from the five-wave fan (MDZ21 Eq. 44-46).
KOKKOS_INLINE_FUNCTION
void mhd_uct_hlld_coeffs(double S_L, double S_Ls, double S_M, double S_Rs, double S_R,
                         double u_L, double u_R,
                         double vt1_L, double vt1_R, double vt2_L, double vt2_R,
                         double* uct){
    // Transverse velocities: same HLL weighting as UCT-HLL (MDZ Eq. 29).
    double alpha_r = S_R>0.0 ? S_R : 0.0;
    double alpha_l = S_L<0.0 ? -S_L : 0.0;
    double isum = 1.0/(alpha_r + alpha_l + 1e-100);
    uct[3] = (alpha_r*vt1_L + alpha_l*vt1_R)*isum;
    uct[4] = (alpha_r*vt2_L + alpha_l*vt2_R)*isum;

    double eps = 1e-9;
    double fan = fabs(S_R - S_L);
    double nust = 0.0;
    if(fabs(S_Rs - S_Ls) > eps*fan)
        nust = (fabs(S_Rs) - fabs(S_Ls))/(S_Rs - S_Ls);

    double denom_L = S_Ls + S_L - 2.0*S_M;
    double chitL = (fabs(denom_L) > eps*fan)
        ? (u_L - S_M)*(S_L - S_M)/denom_L
        : 0.5*(u_L - S_M);
    double nuL = (fabs(S_Ls - S_L) > 1e-30)
        ? (fabs(S_Ls) - fabs(S_L))/(S_Ls - S_L)
        : (S_L >= 0.0 ? 1.0 : -1.0);

    double denom_R = S_Rs + S_R - 2.0*S_M;
    double chitR = (fabs(denom_R) > eps*fan)
        ? (u_R - S_M)*(S_R - S_M)/denom_R
        : 0.5*(u_R - S_M);
    double nuR = (fabs(S_Rs - S_R) > 1e-30)
        ? (fabs(S_Rs) - fabs(S_R))/(S_Rs - S_R)
        : (S_R >= 0.0 ? 1.0 : -1.0);

    uct[0] = 0.5*(1.0 + nust);
    uct[1] = 0.5*((nuL - nust)*chitL + fabs(S_Ls) - nust*S_Ls);
    uct[2] = 0.5*((nuR - nust)*chitR + fabs(S_Rs) - nust*S_Rs);
}

// If uct != nullptr, also writes NUCT face coefficients for the UCT corner EMF.
KOKKOS_INLINE_FUNCTION
void mhd_riemann_hlld(double* f, double* uL, double* uR,
                      int v1, int v2, int v3, int b1, int b2, int b3, double gm,
                      double* uct=nullptr){
    double wL[NMHD], wR[NMHD], fL[NMHD], fR[NMHD];
    mhd_primitives(uL,wL,gm);
    mhd_primitives(uR,wR,gm);
    mhd_fluxes(wL,fL,v1,v2,v3,b1,b2,b3,gm);
    mhd_fluxes(wR,fR,v1,v2,v3,b1,b2,b3,gm);

    double rho_L=wL[_mrho_], rho_R=wR[_mrho_];
    double u_L=wL[v1], u_R=wR[v1];
    double p_L=wL[_mprs_], p_R=wR[_mprs_];
    double Bn = 0.5*(wL[b1]+wR[b1]);

    double c_L = mhd_fast_vel(p_L,rho_L,wL[b1],wL[b2],wL[b3],gm);
    double c_R = mhd_fast_vel(p_R,rho_R,wR[b1],wR[b2],wR[b3],gm);
    double c_max = c_L>c_R ? c_L : c_R;
    double S_L = (u_L<u_R ? u_L : u_R) - c_max;
    double S_R = (u_L>u_R ? u_L : u_R) + c_max;

    double B2_L = wL[b1]*wL[b1]+wL[b2]*wL[b2]+wL[b3]*wL[b3];
    double B2_R = wR[b1]*wR[b1]+wR[b2]*wR[b2]+wR[b3]*wR[b3];
    double pT_L = p_L + 0.5*B2_L;
    double pT_R = p_R + 0.5*B2_R;

    double dSu_L = S_L - u_L;
    double dSu_R = S_R - u_R;
    double denom = dSu_R*rho_R - dSu_L*rho_L;
    double S_M = (dSu_R*rho_R*u_R - dSu_L*rho_L*u_L - pT_R + pT_L)/denom;
    double pT_s = (dSu_R*rho_R*pT_L - dSu_L*rho_L*pT_R
                   + rho_L*rho_R*dSu_R*dSu_L*(u_R - u_L))/denom;
    double rho_sL = rho_L*dSu_L/(S_L - S_M);
    double rho_sR = rho_R*dSu_R/(S_R - S_M);
    double S_Ls = S_M - fabs(Bn)/sqrt(rho_sL);
    double S_Rs = S_M + fabs(Bn)/sqrt(rho_sR);

    // Miyoshi & Kusano derive the star states assuming S_L <= S_Ls <= S_M <= S_Rs <= S_R.
    // The ordering can break when an intermediate density collapses (S_L -> S_M drives
    // rho_s -> 0, so the Alfven speeds run off outside the outer fan). Sampling then picks
    // a state outside its own region of validity, so drop to HLL for that face instead.
    bool ordered = isfinite(S_M) && isfinite(pT_s) && rho_sL > 0.0 && rho_sR > 0.0
                   && S_L <= S_Ls && S_Ls <= S_M && S_M <= S_Rs && S_Rs <= S_R;
    if(!ordered){
        if(S_L >= 0.0)      for(int var=0;var<NMHD;var++) f[var]=fL[var];
        else if(S_R <= 0.0) for(int var=0;var<NMHD;var++) f[var]=fR[var];
        else                for(int var=0;var<NMHD;var++)
            f[var] = (S_R*fL[var] - S_L*fR[var] + S_L*S_R*(uR[var]-uL[var]))/(S_R - S_L);
        if(uct) mhd_uct_hll_coeffs(S_L,S_R,wL[v2],wR[v2],wL[v3],wR[v3],uct);
        return;
    }

    if(uct) mhd_uct_hlld_coeffs(S_L,S_Ls,S_M,S_Rs,S_R,u_L,u_R,
                                wL[v2],wR[v2],wL[v3],wR[v3],uct);

    double U_sL[NMHD], U_sR[NMHD], U_ssL[NMHD], U_ssR[NMHD];
    double vt1_sL, vt2_sL, Bt1_sL, Bt2_sL, vB_sL;
    double vt1_sR, vt2_sR, Bt1_sR, Bt2_sR, vB_sR;
    mhd_hlld_star_state(wL,uL,v1,v2,v3,b1,b2,b3,rho_L,u_L,pT_L,rho_sL,
                        S_L,S_M,pT_s,Bn, U_sL,vt1_sL,vt2_sL,Bt1_sL,Bt2_sL,vB_sL);
    mhd_hlld_star_state(wR,uR,v1,v2,v3,b1,b2,b3,rho_R,u_R,pT_R,rho_sR,
                        S_R,S_M,pT_s,Bn, U_sR,vt1_sR,vt2_sR,Bt1_sR,Bt2_sR,vB_sR);

    double sgn = mhd_sgn(Bn);
    double sr_L = sqrt(rho_sL), sr_R = sqrt(rho_sR);
    double sr_den = sr_L + sr_R;
    double vt1_ss = (sr_L*vt1_sL + sr_R*vt1_sR + sgn*(Bt1_sR - Bt1_sL))/sr_den;
    double vt2_ss = (sr_L*vt2_sL + sr_R*vt2_sR + sgn*(Bt2_sR - Bt2_sL))/sr_den;
    double Bt1_ss = (sr_L*Bt1_sR + sr_R*Bt1_sL + sgn*sr_L*sr_R*(vt1_sR - vt1_sL))/sr_den;
    double Bt2_ss = (sr_L*Bt2_sR + sr_R*Bt2_sL + sgn*sr_L*sr_R*(vt2_sR - vt2_sL))/sr_den;
    double vB_ss = S_M*Bn + vt1_ss*Bt1_ss + vt2_ss*Bt2_ss;

    for(int var=0;var<NMHD;var++){ U_ssL[var]=U_sL[var]; U_ssR[var]=U_sR[var]; }
    U_ssL[v2]=rho_sL*vt1_ss; U_ssL[v3]=rho_sL*vt2_ss;
    U_ssL[b2]=Bt1_ss;        U_ssL[b3]=Bt2_ss;
    U_ssL[_mprs_] = U_sL[_mprs_] - sr_L*sgn*(vB_sL - vB_ss);
    U_ssR[v2]=rho_sR*vt1_ss; U_ssR[v3]=rho_sR*vt2_ss;
    U_ssR[b2]=Bt1_ss;        U_ssR[b3]=Bt2_ss;
    U_ssR[_mprs_] = U_sR[_mprs_] + sr_R*sgn*(vB_sR - vB_ss);

    double FsL[NMHD], FssL[NMHD], FsR[NMHD], FssR[NMHD];
    for(int var=0;var<NMHD;var++){
        FsL[var]  = fL[var] + S_L*(U_sL[var]  - uL[var]);
        FssL[var] = FsL[var] + S_Ls*(U_ssL[var] - U_sL[var]);
        FsR[var]  = fR[var] + S_R*(U_sR[var]  - uR[var]);
        FssR[var] = FsR[var] + S_Rs*(U_ssR[var] - U_sR[var]);
    }

    if(S_M >= 0.0){
        if(S_L > 0.0)            for(int var=0;var<NMHD;var++) f[var]=fL[var];
        else if(S_Ls >= 0.0)     for(int var=0;var<NMHD;var++) f[var]=FsL[var];
        else                     for(int var=0;var<NMHD;var++) f[var]=FssL[var];
    }else{
        if(S_R < 0.0)            for(int var=0;var<NMHD;var++) f[var]=fR[var];
        else if(S_Rs <= 0.0)     for(int var=0;var<NMHD;var++) f[var]=FsR[var];
        else                     for(int var=0;var<NMHD;var++) f[var]=FssR[var];
    }
}

KOKKOS_INLINE_FUNCTION
void mhd_riemann(double* f, double* uL, double* uR,
                 int v1, int v2, int v3, int b1, int b2, int b3, double gm, int rsolver,
                 double* uct=nullptr){
    if(rsolver==_rsolver_hlld_)
        mhd_riemann_hlld(f,uL,uR,v1,v2,v3,b1,b2,b3,gm,uct);
    else
        mhd_riemann_llf(f,uL,uR,v1,v2,v3,b1,b2,b3,gm);
}

template<int D,int V1,int V2,int V3,int B1,int B2,int B3>
void mhd_riemann_solver_t(SD_Solution U, SD_Solution F, SD_Solution Bn, SD_Solution UCT){
    int Nx=U.Nx-(D==_x_), Ny=U.Ny-(D==_y_), Nz=U.Nz-(D==_z_);
    int px=D==_x_?1:U.nx, py=D==_y_?1:U.ny, pz=D==_z_?1:U.nz;
    int n=mhd_choose(D,U.nx,U.ny,U.nz);
    int nader=U.n_ader;
    double gm=cfg.gamma;
    int rsolver=cfg.rsolver;
    bool want_uct = (rsolver==_rsolver_hlld_ && UCT.n_var>=NUCT);
    bool use_bn = (Bn.n_var>=1);
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii){
        double uL[NMHD], uR[NMHD], f[NMHD], uct[NUCT];
        int NidL[3],nidL[3],NidR[3],nidR[3];
        int l=mhd_choose(D,i,j,k);
        mhd_indices(NidL,nidL,k,j,i,kk,jj,ii,l  ,n-1,D);
        mhd_indices(NidR,nidR,k,j,i,kk,jj,ii,l+1,0  ,D);
        for(int t_id=0;t_id<nader;t_id++){
            for(int var=0;var<NMHD;var++){
                uL[var]=U.Vector(INDICES_L);
                uR[var]=U.Vector(INDICES_R);
            }
            if(use_bn){
                double bnL = Bn.Vector(0,0,NidL[_z_],NidL[_y_],NidL[_x_],nidL[_z_],nidL[_y_],nidL[_x_]);
                double bnR = Bn.Vector(0,0,NidR[_z_],NidR[_y_],NidR[_x_],nidR[_z_],nidR[_y_],nidR[_x_]);
                double bn = 0.5*(bnL+bnR);
                uL[B1]=uR[B1]=bn;
            }
            mhd_riemann(f,uL,uR,V1,V2,V3,B1,B2,B3,gm,rsolver, want_uct?uct:nullptr);
            for(int var=0;var<NMHD;var++){
                F.Vector(INDICES_L)=f[var];
                F.Vector(INDICES_R)=f[var];
            }
            if(want_uct){
                for(int uv=0;uv<NUCT;uv++){
                    UCT.Vector(t_id,uv,NidL[_z_],NidL[_y_],NidL[_x_],nidL[_z_],nidL[_y_],nidL[_x_])=uct[uv];
                    UCT.Vector(t_id,uv,NidR[_z_],NidR[_y_],NidR[_x_],nidR[_z_],nidR[_y_],nidR[_x_])=uct[uv];
                }
            }
        }
    });
}

void mhd_riemann_solver(SD_Solution U, SD_Solution F, int dim,
                        SD_Solution Bn, SD_Solution UCT){
    if(dim==_x_)      mhd_riemann_solver_t<_x_,_mvx_,_mvy_,_mvz_,_mbx_,_mby_,_mbz_>(U,F,Bn,UCT);
    else if(dim==_y_) mhd_riemann_solver_t<_y_,_mvy_,_mvz_,_mvx_,_mby_,_mbz_,_mbx_>(U,F,Bn,UCT);
    else              mhd_riemann_solver_t<_z_,_mvz_,_mvx_,_mvy_,_mbz_,_mbx_,_mby_>(U,F,Bn,UCT);
}

// Overwrite the face-normal B row of the fp state with the CT face field (never
// reconstructed), matching the FV mhd_fv_fluxes Bn_f requirement for HLLD.
void mhd_face_B_to_fp(SD_Solution U_fp, SD_Solution B_fp, int dim){
    int brow = (dim==_x_?_mbx_:(dim==_y_?_mby_:_mbz_));
    int Nx=U_fp.Nx, Ny=U_fp.Ny, Nz=U_fp.Nz, px=U_fp.nx, py=U_fp.ny, pz=U_fp.nz;
#ifdef KOKKOS_ENABLE_CUDA
    SD_Vector_h Ufh = Kokkos::create_mirror_view(U_fp.Vector);
    SD_Vector_h Bfh = Kokkos::create_mirror_view(B_fp.Vector);
    Kokkos::deep_copy(Ufh, U_fp.Vector);
    Kokkos::deep_copy(Bfh, B_fp.Vector);
    sd_for_cells_host(Nz,Ny,Nx,pz,py,px, [&](int k,int j,int i,int kk,int jj,int ii){
        Ufh(0,brow,k,j,i,kk,jj,ii) = Bfh(0,0,k,j,i,kk,jj,ii);
    });
    Kokkos::deep_copy(U_fp.Vector, Ufh);
#else
    SD_Vector Vuf = U_fp.Vector;
    SD_Vector Vbf = B_fp.Vector;
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii){
        Vuf(0,brow,k,j,i,kk,jj,ii) = Vbf(0,0,k,j,i,kk,jj,ii);
    });
#endif
}

//----------------------------------------------------------------------------------------
// CFL condition on the fast magnetosonic speed (summed over active dimensions)
//----------------------------------------------------------------------------------------

double mhd_compute_dt(SD_Solution W, double dx, double dy, double dz){
    int Nx=W.Nx, Ny=W.Ny, Nz=W.Nz, px=W.nx, py=W.ny, pz=W.nz;
    double gm=cfg.gamma, cfl=cfg.cfl;
    bool ax=cfg.active[_x_], ay=cfg.active[_y_], az=cfg.active[_z_];
#ifdef KOKKOS_ENABLE_CUDA
    SD_Vector_h Wh = Kokkos::create_mirror_view(W.Vector);
    Kokkos::deep_copy(Wh, W.Vector);
    double min_value = 1;
    sd_for_cells_host(Nz,Ny,Nx,pz,py,px, [&](int k,int j,int i,int kk,int jj,int ii){
        double rho=Wh(0,_mrho_,k,j,i,kk,jj,ii);
        double p  =Wh(0,_mprs_,k,j,i,kk,jj,ii);
        double Bx =Wh(0,_mbx_,k,j,i,kk,jj,ii);
        double By =Wh(0,_mby_,k,j,i,kk,jj,ii);
        double Bz =Wh(0,_mbz_,k,j,i,kk,jj,ii);
        double c_max=0, dx_min=1;
        if(ax){ c_max += fabs(Wh(0,_mvx_,k,j,i,kk,jj,ii)) + mhd_fast_vel(p,rho,Bx,By,Bz,gm); dx_min=min(dx_min,dx); }
        if(ay){ c_max += fabs(Wh(0,_mvy_,k,j,i,kk,jj,ii)) + mhd_fast_vel(p,rho,By,Bz,Bx,gm); dx_min=min(dx_min,dy); }
        if(az){ c_max += fabs(Wh(0,_mvz_,k,j,i,kk,jj,ii)) + mhd_fast_vel(p,rho,Bz,Bx,By,gm); dx_min=min(dx_min,dz); }
        if(c_max > 0){
            double dt_min = cfl*dx_min/c_max/px;
            if(dt_min < min_value) min_value = dt_min;
        }
    });
#else
    SD_Vector Vw = W.Vector;
    double min_value = sd_min_cells(Nz,Ny,Nx,pz,py,px,
        KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii,double& reduce){
            double rho=Vw(0,_mrho_,k,j,i,kk,jj,ii);
            double p  =Vw(0,_mprs_,k,j,i,kk,jj,ii);
            double Bx =Vw(0,_mbx_,k,j,i,kk,jj,ii);
            double By =Vw(0,_mby_,k,j,i,kk,jj,ii);
            double Bz =Vw(0,_mbz_,k,j,i,kk,jj,ii);
            double c_max=0, dx_min=1;
            if(ax){ c_max += fabs(Vw(0,_mvx_,k,j,i,kk,jj,ii)) + mhd_fast_vel(p,rho,Bx,By,Bz,gm); dx_min=min(dx_min,dx); }
            if(ay){ c_max += fabs(Vw(0,_mvy_,k,j,i,kk,jj,ii)) + mhd_fast_vel(p,rho,By,Bz,Bx,gm); dx_min=min(dx_min,dy); }
            if(az){ c_max += fabs(Vw(0,_mvz_,k,j,i,kk,jj,ii)) + mhd_fast_vel(p,rho,Bz,Bx,By,gm); dx_min=min(dx_min,dz); }
            if(c_max > 0){
                double dt_min = cfl*dx_min/c_max/px;
                reduce = reduce < dt_min ? reduce : dt_min;
            }
        });
#endif
    #ifdef MPI
    double g;
    MPI_Allreduce(&min_value,&g,1,MPI_DOUBLE,MPI_MIN,Comm);
    return g;
    #else
    return min_value;
    #endif
}

//----------------------------------------------------------------------------------------
// Constrained transport: edge EMF from fluid velocity + face B
//
// Builds the 8-component edge Riemann state [E, B1, B2, v1, v2, B3, rho, p] at the edge
// points for the given edge direction. B1,B2 are the two transverse face fields
// interpolated onto the edge (single sweep each, exactly as induction::compute_E);
// v1,v2,B3,rho,p are fluid quantities interpolated from the cell-centered primitives
// W_sp onto the same edge points (double sweep across the two transverse directions).
//----------------------------------------------------------------------------------------

// Interpolate cell-centered primitive var `pvar` from solution points to the edge
// point of edge-direction `dim` (double interpolation across the two transverse dims).
KOKKOS_INLINE_FUNCTION
double interp_cc_to_edge(const SD_Solution& W, int pvar,
                         int k,int j,int i,int kk,int jj,int ii,
                         int dim, const Matrix& sp_to_fp, int q){
    // dim==_z_: edge at (x_fp,y_fp,z_sp) -> interp over x (a) and y (b)
    // dim==_y_: edge at (x_fp,y_sp,z_fp) -> interp over x (a) and z (b)
    // dim==_x_: edge at (x_sp,y_fp,z_fp) -> interp over y (a) and z (b)
    double v=0;
    for(int b=0;b<q;b++) for(int a=0;a<q;a++){
        double s;
        if(dim==_z_) s = W.Vector(0,pvar,k,j,i,kk,b,a)*sp_to_fp(jj,b)*sp_to_fp(ii,a);
        else if(dim==_y_) s = W.Vector(0,pvar,k,j,i,b,jj,a)*sp_to_fp(kk,b)*sp_to_fp(ii,a);
        else s = W.Vector(0,pvar,k,j,i,b,a,ii)*sp_to_fp(kk,b)*sp_to_fp(jj,a);
        v += s;
    }
    return v;
}

// E: edge array (8-var), staggered at the edge points of `dim`.
// B1,B2: transverse face fields (single-var). Bcc unused (kept for signature symmetry).
// W_sp: cell-centered primitives (8-var).
void mhd_compute_E(SD_Solution E, SD_Solution W_sp,
                   SD_Solution B1, SD_Solution B2, SD_Solution Bcc,
                   Matrix sp_to_fp, int dim){
    int Nx=E.Nx, Ny=E.Ny, Nz=E.Nz, px=E.nx, py=E.ny, pz=E.nz;
    // The B1/B2 interpolations run along the transverse (active) directions,
    // whose solution-point count is that of x (always active); using the
    // edge-direction extent would degenerate to 1 in 2D.
    int q=W_sp.nx;
    // Velocity / field component indices for this edge direction:
    //   Ez: v1=vx, v2=vy, B3=Bz ; Ey: v1=vz, v2=vx, B3=By ; Ex: v1=vy, v2=vz, B3=Bx
    int v1_var = mhd_choose(dim,_mvy_,_mvz_,_mvx_);
    int v2_var = mhd_choose(dim,_mvz_,_mvx_,_mvy_);
    int b3_var = mhd_choose(dim,_mbx_,_mby_,_mbz_);
    int qsp = W_sp.nx; // solution points per element (all active dims equal)
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii){
        double b1=0,b2=0;
        for(int ll=0;ll<q;ll++){
            if(dim==_z_){
                b1 += B1.Vector(0,0,k,j,i,kk,ll,ii)*sp_to_fp(jj,ll); //Bx along y
                b2 += B2.Vector(0,0,k,j,i,kk,jj,ll)*sp_to_fp(ii,ll); //By along x
            } else if(dim==_y_){
                b1 += B1.Vector(0,0,k,j,i,kk,jj,ll)*sp_to_fp(ii,ll); //Bz along x
                b2 += B2.Vector(0,0,k,j,i,ll,jj,ii)*sp_to_fp(kk,ll); //Bx along z
            } else {
                b1 += B1.Vector(0,0,k,j,i,ll,jj,ii)*sp_to_fp(kk,ll); //By along z
                b2 += B2.Vector(0,0,k,j,i,kk,ll,ii)*sp_to_fp(jj,ll); //Bz along y
            }
        }
        double v1  = interp_cc_to_edge(W_sp,v1_var,k,j,i,kk,jj,ii,dim,sp_to_fp,qsp);
        double v2  = interp_cc_to_edge(W_sp,v2_var,k,j,i,kk,jj,ii,dim,sp_to_fp,qsp);
        double B3  = interp_cc_to_edge(W_sp,b3_var,k,j,i,kk,jj,ii,dim,sp_to_fp,qsp);
        double rho = interp_cc_to_edge(W_sp,_mrho_,k,j,i,kk,jj,ii,dim,sp_to_fp,qsp);
        double prs = interp_cc_to_edge(W_sp,_mprs_,k,j,i,kk,jj,ii,dim,sp_to_fp,qsp);
        E.Vector(0,0,k,j,i,kk,jj,ii) = v1*b2 - v2*b1;
        E.Vector(0,1,k,j,i,kk,jj,ii) = b1;
        E.Vector(0,2,k,j,i,kk,jj,ii) = b2;
        E.Vector(0,3,k,j,i,kk,jj,ii) = v1;
        E.Vector(0,4,k,j,i,kk,jj,ii) = v2;
        E.Vector(0,5,k,j,i,kk,jj,ii) = B3;
        E.Vector(0,6,k,j,i,kk,jj,ii) = rho;
        E.Vector(0,7,k,j,i,kk,jj,ii) = prs;
    });
}

// Edge electric-field Riemann solvers (8-var edge state
// [E, B1, B2, v1, v2, B3, rho, p], with E = v1*B2 - v2*B1).
//
// v_index selects the sweep (matching Python mhd_sd_scheme / hlld_E):
//   3: normal = dim1; continuous Bn = B1, discontinuous Bt = B2, induction flux +E
//   4: normal = dim2; continuous Bn = B2, discontinuous Bt = B1, induction flux -E
// Writes a single-valued interface state to both sides so the next sweep sees
// a consistent edge state (as in the Python broadcast).

KOKKOS_INLINE_FUNCTION
void mhd_E_riemann_llf(double* es, const double* eL, const double* eR,
                       int v_index, double gm){
    double cL = mhd_fast_vel(eL[7],eL[6],eL[1],eL[2],eL[5],gm);
    double cR = mhd_fast_vel(eR[7],eR[6],eR[1],eR[2],eR[5],gm);
    double vmax = max(fabs(eL[v_index]),fabs(eR[v_index]));
    double Ss = vmax + max(cL,cR);
    double diss = (v_index==3) ? -0.5*Ss*(eR[2]-eL[2]) : 0.5*Ss*(eR[1]-eL[1]);
    for(int var=0;var<NEMHD;var++) es[var]=0.5*(eR[var]+eL[var]) + diss;
}

KOKKOS_INLINE_FUNCTION
double mhd_hlld_select(double S_L, double S_Ls, double S_M, double S_Rs, double S_R,
                       double qL, double qsL, double qssL,
                       double qssR, double qsR, double qR){
    if(S_M >= 0.0){
        if(S_L > 0.0) return qL;
        if(S_Ls >= 0.0) return qsL;
        return qssL;
    }
    if(S_R < 0.0) return qR;
    if(S_Rs <= 0.0) return qsR;
    return qssR;
}

// One-dimensional HLLD sweep of the edge E Riemann problem (spd hlld_E).
KOKKOS_INLINE_FUNCTION
void mhd_E_riemann_hlld(double* Es, const double* eL, const double* eR,
                        int vel, double gm){
    int _u_, _vt_, _bn_, _bt_;
    double e_sign;
    if(vel==3){ _u_=3; _vt_=4; _bn_=1; _bt_=2; e_sign= 1.0; }
    else      { _u_=4; _vt_=3; _bn_=2; _bt_=1; e_sign=-1.0; }

    double rho_L=eL[6], rho_R=eR[6];
    double p_L=eL[7],   p_R=eR[7];
    double u_L=eL[_u_], u_R=eR[_u_];
    double vt_L=eL[_vt_], vt_R=eR[_vt_];
    double Bt_L=eL[_bt_], Bt_R=eR[_bt_];
    double B3_L=eL[5],   B3_R=eR[5];
    double Bn = 0.5*(eL[_bn_]+eR[_bn_]);

    double c_L = mhd_fast_vel(p_L,rho_L,eL[_bn_],Bt_L,B3_L,gm);
    double c_R = mhd_fast_vel(p_R,rho_R,eR[_bn_],Bt_R,B3_R,gm);
    double c_max = c_L>c_R ? c_L : c_R;
    double S_L = (u_L<u_R ? u_L : u_R) - c_max;
    double S_R = (u_L>u_R ? u_L : u_R) + c_max;

    double pT_L = p_L + 0.5*(eL[_bn_]*eL[_bn_] + Bt_L*Bt_L + B3_L*B3_L);
    double pT_R = p_R + 0.5*(eR[_bn_]*eR[_bn_] + Bt_R*Bt_R + B3_R*B3_R);

    double dSu_L = S_L - u_L;
    double dSu_R = S_R - u_R;
    double denom = dSu_R*rho_R - dSu_L*rho_L;
    double S_M = (dSu_R*rho_R*u_R - dSu_L*rho_L*u_L - pT_R + pT_L)/denom;
    double rho_sL = rho_L*dSu_L/(S_L - S_M);
    double rho_sR = rho_R*dSu_R/(S_R - S_M);
    double S_Ls = S_M - fabs(Bn)/sqrt(rho_sL);
    double S_Rs = S_M + fabs(Bn)/sqrt(rho_sR);

    // Same fan-ordering requirement as the face solver, and it bites harder here: the
    // resolved state feeds the second sweep as its rho and p, so a collapsing rho_s makes
    // S_Ls run off, G_ss with it, and the edge E (hence the CT field) blows up. Degenerate
    // fans fall back to a Rusanov edge value on a plain averaged state.
    if(!(isfinite(S_M) && rho_sL > 0.0 && rho_sR > 0.0
         && S_L <= S_Ls && S_Ls <= S_M && S_M <= S_Rs && S_Rs <= S_R)){
        double Ss = max(fabs(u_L),fabs(u_R)) + c_max;
        for(int var=0;var<NEMHD;var++) Es[var]=0.5*(eL[var]+eR[var]);
        Es[0]    = 0.5*(eL[0]+eR[0]) - e_sign*0.5*Ss*(Bt_R - Bt_L);
        Es[_bn_] = Bn;
        return;
    }

    double fac_vL, fac_bL, fac_vR, fac_bR;
    mhd_hlld_star_factors(rho_L,u_L,S_L,S_M,Bn,fac_vL,fac_bL);
    mhd_hlld_star_factors(rho_R,u_R,S_R,S_M,Bn,fac_vR,fac_bR);
    double vt_sL = vt_L - fac_vL*Bt_L;
    double vt_sR = vt_R - fac_vR*Bt_R;
    double Bt_sL = fac_bL*Bt_L;
    double Bt_sR = fac_bR*Bt_R;
    double B3_sL = fac_bL*B3_L;
    double B3_sR = fac_bR*B3_R;

    double sgn = mhd_sgn(Bn);
    double sr_L = sqrt(rho_sL), sr_R = sqrt(rho_sR);
    double sr_den = sr_L + sr_R;
    double vt_ss = (sr_L*vt_sL + sr_R*vt_sR + sgn*(Bt_sR - Bt_sL))/sr_den;
    double Bt_ss = (sr_L*Bt_sR + sr_R*Bt_sL + sgn*sr_L*sr_R*(vt_sR - vt_sL))/sr_den;
    double B3_ss = (sr_L*B3_sR + sr_R*B3_sL)/sr_den;

    double G_L   = e_sign*eL[0];
    double G_R   = e_sign*eR[0];
    double G_sL  = G_L + S_L*(Bt_sL - Bt_L);
    double G_sR  = G_R + S_R*(Bt_sR - Bt_R);
    double G_ssL = G_sL + S_Ls*(Bt_ss - Bt_sL);
    double G_ssR = G_sR + S_Rs*(Bt_ss - Bt_sR);

    Es[0]    = e_sign*mhd_hlld_select(S_L,S_Ls,S_M,S_Rs,S_R, G_L,G_sL,G_ssL,G_ssR,G_sR,G_R);
    Es[_bn_] = Bn;
    Es[_bt_] = mhd_hlld_select(S_L,S_Ls,S_M,S_Rs,S_R, Bt_L,Bt_sL,Bt_ss,Bt_ss,Bt_sR,Bt_R);
    Es[_u_]  = mhd_hlld_select(S_L,S_Ls,S_M,S_Rs,S_R, u_L,S_M,S_M,S_M,S_M,u_R);
    Es[_vt_] = mhd_hlld_select(S_L,S_Ls,S_M,S_Rs,S_R, vt_L,vt_sL,vt_ss,vt_ss,vt_sR,vt_R);
    Es[5]    = mhd_hlld_select(S_L,S_Ls,S_M,S_Rs,S_R, B3_L,B3_sL,B3_ss,B3_ss,B3_sR,B3_R);
    Es[6]    = mhd_hlld_select(S_L,S_Ls,S_M,S_Rs,S_R, rho_L,rho_sL,rho_sL,rho_sR,rho_sR,rho_R);
    Es[7]    = (S_M >= 0.0) ? p_L : p_R;
}

KOKKOS_INLINE_FUNCTION
void mhd_E_riemann(double* es, const double* eL, const double* eR,
                   int v_index, double gm, int rsolver){
    if(rsolver==_rsolver_hlld_)
        mhd_E_riemann_hlld(es,eL,eR,v_index,gm);
    else
        mhd_E_riemann_llf(es,eL,eR,v_index,gm);
}

void mhd_E_riemann_solver(SD_Solution E, int dim, int v_index){
    int Nx=E.Nx-(dim==_x_), Ny=E.Ny-(dim==_y_), Nz=E.Nz-(dim==_z_);
    int px=dim==_x_?1:E.nx, py=dim==_y_?1:E.ny, pz=dim==_z_?1:E.nz;
    int n=mhd_choose(dim,E.nx,E.ny,E.nz);
    double gm=cfg.gamma;
    int rsolver=cfg.rsolver;
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii){
        int NidL[3],nidL[3],NidR[3],nidR[3];
        int l=mhd_choose(dim,i,j,k);
        mhd_indices(NidL,nidL,k,j,i,kk,jj,ii,l  ,n-1,dim);
        mhd_indices(NidR,nidR,k,j,i,kk,jj,ii,l+1,0  ,dim);
        int t_id=0; (void)t_id;
        double eL[NEMHD], eR[NEMHD], es[NEMHD];
        for(int var=0;var<NEMHD;var++){ eL[var]=E.Vector(INDICES_L); eR[var]=E.Vector(INDICES_R); }
        mhd_E_riemann(es,eL,eR,v_index,gm,rsolver);
        for(int var=0;var<NEMHD;var++){ E.Vector(INDICES_L)=es[var]; E.Vector(INDICES_R)=es[var]; }
    });
}

//----------------------------------------------------------------------------------------
// SD UCT edge EMF (MDZ21 / Berta+2024 Eq.39, spd E = v×B convention).
// One formula covers interior (v×B), mid-face 1D IFs, and 4-element 2D corners:
// discontinuous axes take L/R from neighbour edge states + face HLLD (a,d); continuous
// axes collapse to the local mhd_compute_E state with a=1/2, d=0.
//----------------------------------------------------------------------------------------

KOKKOS_INLINE_FUNCTION
double mhd_sd_clamp_a(double a){
    return a<0.0 ? 0.0 : (a>1.0 ? 1.0 : a);
}

KOKKOS_INLINE_FUNCTION
double mhd_sd_clamp_d(double d){
    return d>0.0 ? d : 0.0;
}

KOKKOS_INLINE_FUNCTION
double mhd_uct_formula(double aW, double aE, double aS, double aN,
                        double v1W, double v1E, double v2S, double v2N,
                        double B2W, double B2E, double B1S, double B1N,
                        double dW, double dE, double dS, double dN){
    return (aW*v1W*B2W + aE*v1E*B2E)
         - (aS*v2S*B1S + aN*v2N*B1N)
         - (dE*B2E - dW*B2W)
         + (dN*B1N - dS*B1S);
}

// Interpolate UCT face coefficient `uvar` on the face normal to `dim1` at interface
// index id1_if. When `dim2` is sp on that face, read at sp index id2; when fp on the
// edge, sum sp nodes along dim2 with sp_to_fp(id2,·).
KOKKOS_INLINE_FUNCTION
double sd_interp_uct_face(const SD_Solution& UCT, int uvar,
                          int k,int j,int i,int kk,int jj,int ii,
                          int dim1, int dim2, int id1_if, int id2,
                          const Matrix& sp_to_fp, int q){
    if(dim1==_x_){
        if(dim2==_y_){
            double v=0;
            for(int ll=0;ll<q;ll++)
                v += UCT.Vector(0,uvar,k,j,i,kk,ll,id1_if)*sp_to_fp(id2,ll);
            return v;
        }
        return UCT.Vector(0,uvar,k,j,i,id2,jj,id1_if);
    }
    if(dim1==_y_){
        if(dim2==_x_){
            double v=0;
            for(int ll=0;ll<q;ll++)
                v += UCT.Vector(0,uvar,k,j,i,kk,id1_if,ll)*sp_to_fp(id2,ll);
            return v;
        }
        return UCT.Vector(0,uvar,k,j,i,kk,id1_if,ii);
    }
    if(dim2==_x_){
        double v=0;
        for(int ll=0;ll<q;ll++)
            v += UCT.Vector(0,uvar,k,j,i,id1_if,jj,ll)*sp_to_fp(id2,ll);
        return v;
    }
    double v=0;
    for(int ll=0;ll<q;ll++)
        v += UCT.Vector(0,uvar,k,j,i,id1_if,ll,ii)*sp_to_fp(id2,ll);
    return v;
}

KOKKOS_INLINE_FUNCTION
double sd_E_comp(const SD_Solution& E, int var,
                 int k,int j,int i,int kk,int jj,int ii){
    return E.Vector(0,var,k,j,i,kk,jj,ii);
}

template<int EDIM>
void mhd_uct_edge_E_t(SD_Solution E, SD_Solution UCT1, SD_Solution UCT2, Matrix sp_to_fp){
    const int dim1 = (EDIM==_z_?_x_:(EDIM==_y_?_z_:_y_));
    const int dim2 = (EDIM==_z_?_y_:(EDIM==_y_?_x_:_z_));
    int Nx=E.Nx, Ny=E.Ny, Nz=E.Nz, px=E.nx, py=E.ny, pz=E.nz;
    int n1 = mhd_choose(dim1,px,py,pz);
    int n2 = mhd_choose(dim2,px,py,pz);
    int q1 = n1 - 1; // sp nodes along dim1 on UCT1 face
    int q2 = n2 - 1; // sp nodes along dim2 on UCT2 face

    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii){
        int id1 = mhd_choose(dim1,ii,jj,kk);
        int id2 = mhd_choose(dim2,ii,jj,kk);
        bool on_IF1 = (id1==0 || id1==n1-1);
        bool on_IF2 = (id2==0 || id2==n2-1);

        double v1_loc = sd_E_comp(E,3,k,j,i,kk,jj,ii);
        double v2_loc = sd_E_comp(E,4,k,j,i,kk,jj,ii);
        double B1_loc = sd_E_comp(E,1,k,j,i,kk,jj,ii);
        double B2_loc = sd_E_comp(E,2,k,j,i,kk,jj,ii);

        double v1W,v1E,v2S,v2N,B2W,B2E,B1S,B1N;
        double aW,aE,aS,aN,dW,dE,dS,dN;

        // --- dim1 (e.g. x for Ez): v1, B2 weighted by UCT1 ---
        if(on_IF1){
            int kW=k,jW=j,iW=i,kkW=kk,jjW=jj,iiW=ii;
            int kE=k,jE=j,iE=i,kkE=kk,jjE=jj,iiE=ii;
            if(id1==0){
                if(dim1==_x_){ iW=i-1; iiW=n1-1; iiE=0; }
                else if(dim1==_y_){ jW=j-1; jjW=n1-1; jjE=0; }
                else { kW=k-1; kkW=n1-1; kkE=0; }
            }else{
                if(dim1==_x_){ iE=i+1; iiW=n1-1; iiE=0; }
                else if(dim1==_y_){ jE=j+1; jjW=n1-1; jjE=0; }
                else { kE=k+1; kkW=n1-1; kkE=0; }
            }
            v1W = sd_E_comp(E,3,kW,jW,iW,kkW,jjW,iiW);
            v1E = sd_E_comp(E,3,kE,jE,iE,kkE,jjE,iiE);
            B2W = sd_E_comp(E,2,kW,jW,iW,kkW,jjW,iiW);
            B2E = sd_E_comp(E,2,kE,jE,iE,kkE,jjE,iiE);
            int id1_if = (id1==0 ? 0 : n1-1);
            aW = mhd_sd_clamp_a(sd_interp_uct_face(UCT1,0,k,j,i,kk,jj,ii,dim1,dim2,id1_if,id2,sp_to_fp,q1));
            aE = 1.0 - aW;
            dW = mhd_sd_clamp_d(sd_interp_uct_face(UCT1,1,k,j,i,kk,jj,ii,dim1,dim2,id1_if,id2,sp_to_fp,q1));
            dE = mhd_sd_clamp_d(sd_interp_uct_face(UCT1,2,k,j,i,kk,jj,ii,dim1,dim2,id1_if,id2,sp_to_fp,q1));
        }else{
            v1W=v1E=v1_loc; B2W=B2E=B2_loc; aW=aE=0.5; dW=dE=0.0;
        }

        // --- dim2 (e.g. y for Ez): v2, B1 weighted by UCT2 ---
        if(on_IF2){
            int kS=k,jS=j,iS=i,kkS=kk,jjS=jj,iiS=ii;
            int kN=k,jN=j,iN=i,kkN=kk,jjN=jj,iiN=ii;
            if(id2==0){
                if(dim2==_x_){ iS=i-1; iiS=n2-1; iiN=0; }
                else if(dim2==_y_){ jS=j-1; jjS=n2-1; jjN=0; }
                else { kS=k-1; kkS=n2-1; kkN=0; }
            }else{
                if(dim2==_x_){ iN=i+1; iiS=n2-1; iiN=0; }
                else if(dim2==_y_){ jN=j+1; jjS=n2-1; jjN=0; }
                else { kN=k+1; kkS=n2-1; kkN=0; }
            }
            v2S = sd_E_comp(E,4,kS,jS,iS,kkS,jjS,iiS);
            v2N = sd_E_comp(E,4,kN,jN,iN,kkN,jjN,iiN);
            B1S = sd_E_comp(E,1,kS,jS,iS,kkS,jjS,iiS);
            B1N = sd_E_comp(E,1,kN,jN,iN,kkN,jjN,iiN);
            int id2_if = (id2==0 ? 0 : n2-1);
            aS = mhd_sd_clamp_a(sd_interp_uct_face(UCT2,0,k,j,i,kk,jj,ii,dim2,dim1,id2_if,id1,sp_to_fp,q2));
            aN = 1.0 - aS;
            dS = mhd_sd_clamp_d(sd_interp_uct_face(UCT2,1,k,j,i,kk,jj,ii,dim2,dim1,id2_if,id1,sp_to_fp,q2));
            dN = mhd_sd_clamp_d(sd_interp_uct_face(UCT2,2,k,j,i,kk,jj,ii,dim2,dim1,id2_if,id1,sp_to_fp,q2));
        }else{
            v2S=v2N=v2_loc; B1S=B1N=B1_loc; aS=aN=0.5; dS=dN=0.0;
        }

        E.Vector(0,0,k,j,i,kk,jj,ii) = mhd_uct_formula(
            aW,aE,aS,aN, v1W,v1E,v2S,v2N, B2W,B2E,B1S,B1N, dW,dE,dS,dN);
    });
}

void mhd_uct_edge_E(SD_Solution E, SD_Solution UCT1, SD_Solution UCT2,
                    Matrix sp_to_fp, int edim){
    if(edim==_x_)      mhd_uct_edge_E_t<_x_>(E,UCT1,UCT2,sp_to_fp);
    else if(edim==_y_) mhd_uct_edge_E_t<_y_>(E,UCT1,UCT2,sp_to_fp);
    else               mhd_uct_edge_E_t<_z_>(E,UCT1,UCT2,sp_to_fp);
}

//----------------------------------------------------------------------------------------
// B_to_U: project the (divergence-free) face-staggered field onto the cell-centered B
// rows of the conservative state. For each direction, interpolate the face field
// (fp along that dim) to solution points and write into the matching B row of U.
//----------------------------------------------------------------------------------------

// Interpolate a single-var face field (fp along `dim`) to solution points, writing
// into var-row `brow` of the (8-var) conservative array U.
static void project_face_to_row(SD_Solution U, int brow, SD_Solution B, Matrix fp_to_sp, int dim){
    int Nx=U.Nx, Ny=U.Ny, Nz=U.Nz, px=U.nx, py=U.ny, pz=U.nz;
    int q=mhd_choose(dim,B.nx,B.ny,B.nz);
#ifdef KOKKOS_ENABLE_CUDA
    Matrix_h Fh = setup_mirror(fp_to_sp);
    setup_pull(fp_to_sp, Fh);
    SD_Vector_h Uh = Kokkos::create_mirror_view(U.Vector);
    SD_Vector_h Bh = Kokkos::create_mirror_view(B.Vector);
    //Pull U before touching it. This kernel writes ONE row (brow) but the
    //deep_copy below pushes the WHOLE mirror back, so without this every other
    //variable is overwritten with whatever the fresh mirror held. The #else
    //path writes U in place and is unaffected, which is why the suite has
    //always been green on CPU while MHD produced a zero/NaN state from t=0 on
    //CUDA: W_sp lost rho/v/p here, then cons_to_prim_cv divided by rho = 0.
    Kokkos::deep_copy(Uh, U.Vector);
    Kokkos::deep_copy(Bh, B.Vector);
    sd_for_cells_host(Nz,Ny,Nx,pz,py,px, [&](int k,int j,int i,int kk,int jj,int ii){
        int nid[3];
        double u=0;
        int id=mhd_choose(dim,ii,jj,kk);
        for(int ll=0;ll<q;ll++){
            mhd_indices_n(nid,kk,jj,ii,ll,dim);
            u += Bh(0,0,k,j,i,NODE)*Fh(id,ll);
        }
        Uh(0,brow,k,j,i,kk,jj,ii)=u;
    });
    Kokkos::deep_copy(U.Vector, Uh);
#else
    SD_Vector Vu = U.Vector;
    SD_Vector Vb = B.Vector;
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii){
        int nid[3];
        double u=0;
        int id=mhd_choose(dim,ii,jj,kk);
        for(int ll=0;ll<q;ll++){
            mhd_indices_n(nid,kk,jj,ii,ll,dim);
            u += Vb(0,0,k,j,i,NODE)*fp_to_sp(id,ll);
        }
        Vu(0,brow,k,j,i,kk,jj,ii)=u;
    });
#endif
}

void mhd_B_to_U(SD_Solution U, SD_Solution Bx, SD_Solution By, SD_Solution Bz,
                SD_Solution Tx, SD_Solution Ty, SD_Solution Tz, Matrix fp_to_sp){
    (void)Tx;(void)Ty;(void)Tz;
    if(cfg.active[_x_]) project_face_to_row(U,_mbx_,Bx,fp_to_sp,_x_);
    if(cfg.active[_y_]) project_face_to_row(U,_mby_,By,fp_to_sp,_y_);
    if(cfg.active[_z_]) project_face_to_row(U,_mbz_,Bz,fp_to_sp,_z_);
}

void mhd_compute_B_sp_from_fp(SD_Solution Bcc, SD_Solution Bx, SD_Solution By,
                              SD_Solution Bz, Matrix fp_to_sp){
    (void)Bcc;(void)Bx;(void)By;(void)Bz;(void)fp_to_sp;
}

//----------------------------------------------------------------------------------------
// MHD MOOD trouble detection (control-volume averages, |B| in the NAD)
//
// Detection runs on cell-averaged quantities. The candidate conserved state U (FV
// layout, cell averages) has its B rows already replaced by the cell average of the
// candidate constrained-transport field, so the NAD/PAD tests see the *true* new B.
// The NAD variables are the density, the gas pressure and the field magnitude |B|
// (per the AthenaK finding that |B| is more robust than the individual components);
// the PAD bounds the density and the gas pressure (magnetic energy subtracted).
//----------------------------------------------------------------------------------------

// Extract NAD detection variables from a cell-averaged conservative MHD state.
// Layout (prefix always present):
//   row 0: rho, row 1: gas p
// then B per cfg.mood_nad_b:
//   comps (default): Bx, By, Bz
//   mag:             |B|
// then velocity per cfg.mood_nad_v:
//   comps (default): vx, vy, vz
//   mag:             |v|
//   off:             (nothing)
// Returns the number of active detection variables.
int mhd_detection_vars(FV_Solution U, FV_Solution det){
    int Nx=U.Nx, Ny=U.Ny, Nz=U.Nz;
    double gm=cfg.gamma;
    int bmode=cfg.mood_nad_b, vmode=cfg.mood_nad_v;
    fv_for_cells(Nz,Ny,Nx, KOKKOS_LAMBDA(int k,int j,int i){
        double u[NMHD], w[NMHD];
        for(int var=0;var<NMHD;var++) u[var]=U.Vector(var,k,j,i);
        mhd_primitives(u,w,gm);
        det.Vector(0,k,j,i)=w[_mrho_];
        det.Vector(1,k,j,i)=w[_mprs_];
        int r=2;
        if(bmode==_nad_b_mag_){
            double B2=w[_mbx_]*w[_mbx_]+w[_mby_]*w[_mby_]+w[_mbz_]*w[_mbz_];
            det.Vector(r++,k,j,i)=sqrt(B2);
        }else{
            det.Vector(r++,k,j,i)=w[_mbx_];
            det.Vector(r++,k,j,i)=w[_mby_];
            det.Vector(r++,k,j,i)=w[_mbz_];
        }
        if(vmode==_nad_v_mag_){
            double v2=w[_mvx_]*w[_mvx_]+w[_mvy_]*w[_mvy_]+w[_mvz_]*w[_mvz_];
            det.Vector(r++,k,j,i)=sqrt(v2);
        }else if(vmode==_nad_v_comps_){
            det.Vector(r++,k,j,i)=w[_mvx_];
            det.Vector(r++,k,j,i)=w[_mvy_];
            det.Vector(r++,k,j,i)=w[_mvz_];
        }
    });
    int nvar=2;
    nvar += (bmode==_nad_b_mag_) ? 1 : 3;
    if(vmode==_nad_v_mag_) nvar += 1;
    else if(vmode==_nad_v_comps_) nvar += 3;
    return nvar;
}

// Domain-wide NAD tolerance scales for grange/gcfl (AthenaK mood_nad_scale).
// For each detection variable, gscale[var] = max-min over active cells of det_old.
// With gcfl the ranges are further multiplied by min(1, dt·vmax/dxmin), vmax being
// the domain max of max(|vx|,|vy|,|vz|) from the stage-input primitives W.
// gscale must hold at least nvar entries; relative/delta modes leave it unused.
// Per-BLOCK partial reduction for the global NAD scales: min/max of each
// detection variable over the block's active cells, plus max|v| for the gcfl
// softening. No MPI and no CFL factor here -- the caller combines across every
// block it owns first (Mesh::mhd_reduce_nad_gscales), because a domain range
// reduced per block is not the same number as one reduced over the domain, and
// under AMR it is not even the same physical region. See mhd_nad_finalize_gscales.
void mhd_nad_partial_gscales(FV_Solution det_old, FV_Solution W, int nvar,
                             double* gmin, double* gmax, double& vmax){
    int Nx=det_old.Nx, Ny=det_old.Ny, Nz=det_old.Nz;
    int scale=cfg.mood_nad_scale;
    if(scale!=_nad_scale_grange_ && scale!=_nad_scale_gcfl_) return;

    // Active-cell frame (NGH ghosts), matching AthenaK's active-only reduce.
    unsigned Mz=(unsigned)(Nz-2*NGHz), My=(unsigned)(Ny-2*NGHy), Mx=(unsigned)(Nx-2*NGHx);
    int64_t total = (int64_t)Mz*My*Mx;
    if(total<=0) return;
    int oz=NGHz, oy=NGHy, ox=NGHx;

    for(int var=0;var<nvar;var++){
        double mn=1e300, mx=-1e300;
        auto detV=det_old.Vector;
        Kokkos::parallel_reduce("mhd_nad_grange", flat_range(0,flat_total(total)),
            KOKKOS_LAMBDA(const unsigned idx, double& rmin, double& rmax){
                int k,j,i;
                flat_index3(idx,Mz,My,Mx,oz,oy,ox,k,j,i);
                double u=detV(var,k,j,i);
                rmin = rmin<u ? rmin : u;
                rmax = rmax>u ? rmax : u;
            }, Kokkos::Min<double>(mn), Kokkos::Max<double>(mx));
        if(mn<gmin[var]) gmin[var]=mn;
        if(mx>gmax[var]) gmax[var]=mx;
    }

    if(scale==_nad_scale_gcfl_){
        double vm=0.0;
        auto WV=W.Vector;
        Kokkos::parallel_reduce("mhd_nad_vmax", flat_range(0,flat_total(total)),
            KOKKOS_LAMBDA(const unsigned idx, double& rmax){
                int k,j,i;
                flat_index3(idx,Mz,My,Mx,oz,oy,ox,k,j,i);
                double vx=fabs(WV(_mvx_,k,j,i));
                double vy=fabs(WV(_mvy_,k,j,i));
                double vz=fabs(WV(_mvz_,k,j,i));
                double v = vx>vy ? vx : vy;
                v = v>vz ? v : vz;
                rmax = rmax>v ? rmax : v;
            }, Kokkos::Max<double>(vm));
        if(vm>vmax) vmax=vm;
    }
}

// Turn combined (already cross-block, already cross-rank) partials into the NAD
// band scales. dxmin must be the GLOBAL minimum cell size: under AMR the gcfl
// softening dt*vmax/dxmin would otherwise differ level by level, so a fine block
// and a coarse block covering the same flow would get different bands.
void mhd_nad_finalize_gscales(const double* gmin, const double* gmax, double vmax,
                              int nvar, double* gscale, double dt, double dxmin,
                              bool apply_cfl){
    int scale=cfg.mood_nad_scale;
    for(int var=0;var<nvar;var++) gscale[var]=0.0;
    if(scale!=_nad_scale_grange_ && scale!=_nad_scale_gcfl_) return;
    for(int var=0;var<nvar;var++)
        gscale[var] = gmax[var]>gmin[var] ? gmax[var]-gmin[var] : 0.0;
    if(scale==_nad_scale_gcfl_ && apply_cfl){
        double cfl_adv = (dxmin>0.0 && isfinite(dxmin)) ? dt*vmax/dxmin : 0.0;
        double fac = cfl_adv<1.0 ? cfl_adv : 1.0;
        for(int var=0;var<nvar;var++) gscale[var]*=fac;
    }
}

// Single-block convenience wrapper: partial + rank reduce + finalize. Used by the
// standalone (non-mesh) MOOD path, where one block IS the domain.
void mhd_nad_compute_gscales(FV_Solution det_old, FV_Solution W, int nvar,
                             double* gscale, double dt,
                             double dx, double dy, double dz,
                             bool apply_cfl){
    double gmin[8], gmax[8], vmax=0.0;
    for(int v=0;v<8;v++){ gmin[v]=1e300; gmax[v]=-1e300; }
    mhd_nad_partial_gscales(det_old,W,nvar,gmin,gmax,vmax);
    #ifdef MPI
    double b[8];
    MPI_Allreduce(gmin,b,8,MPI_DOUBLE,MPI_MIN,Comm); for(int v=0;v<8;v++) gmin[v]=b[v];
    MPI_Allreduce(gmax,b,8,MPI_DOUBLE,MPI_MAX,Comm); for(int v=0;v<8;v++) gmax[v]=b[v];
    double vg; MPI_Allreduce(&vmax,&vg,1,MPI_DOUBLE,MPI_MAX,Comm); vmax=vg;
    #endif
    double dxmin=1e300;
    if(cfg.active[_x_]) dxmin=dxmin<dx ? dxmin : dx;
    if(cfg.active[_y_]) dxmin=dxmin<dy ? dxmin : dy;
    if(cfg.active[_z_]) dxmin=dxmin<dz ? dxmin : dz;
    mhd_nad_finalize_gscales(gmin,gmax,vmax,nvar,gscale,dt,dxmin,apply_cfl);
}

// Discrete-maximum-principle NAD on the detection variables. A cell is flagged
// when the candidate leaves the neighbourhood band of the old state for any
// variable. Band half-width depends on cfg.mood_nad_scale:
//   relative: max(rtol*|bound|, atol) per side (spd legacy)
//   delta:    max(rtol·local_range, atol)
//   grange/gcfl: max(rtol·gscale[var], atol), plus eps0*|bound| floor
//                (AthenaK; gscale from mhd_nad_compute_gscales)
void mhd_NAD(FV_Solution det_new, FV_Solution det_old, FV_Solution troubles,
             double tol, int nvar, const double* gscale){
    int Nx=det_old.Nx, Ny=det_old.Ny, Nz=det_old.Nz;
    bool ax=cfg.active[_x_], ay=cfg.active[_y_], az=cfg.active[_z_];
    bool moore=cfg.nad_moore;
    int scale=cfg.mood_nad_scale;
    double atol=cfg.nad_atol, eps0=cfg.nad_eps0;
    // Capture gscale into a fixed array for the device lambda (nvar ≤ 8).
    double gs[8];
    for(int v=0;v<8;v++) gs[v]=(gscale && v<nvar) ? gscale[v] : 0.0;

    fv_for_cells_ngh(Nz,Ny,Nx, KOKKOS_LAMBDA(int k,int j,int i){
        double trouble=0;
        for(int var=0;var<nvar;var++){
            double u_new=det_new.Vector(var,k,j,i);
            double mx=det_old.Vector(var,k,j,i);
            double mn=mx;
            if(moore){
                for(int dk=-(int)az;dk<=(int)az;dk++)
                for(int dj=-(int)ay;dj<=(int)ay;dj++)
                for(int di=-(int)ax;di<=(int)ax;di++){
                    double u=det_old.Vector(var,k+dk,j+dj,i+di);
                    mx=max(mx,u); mn=min(mn,u);
                }
            } else {
                if(ax){ double l=det_old.Vector(var,k,j,i-1),r=det_old.Vector(var,k,j,i+1); mx=max(mx,max(l,r)); mn=min(mn,min(l,r)); }
                if(ay){ double l=det_old.Vector(var,k,j-1,i),r=det_old.Vector(var,k,j+1,i); mx=max(mx,max(l,r)); mn=min(mn,min(l,r)); }
                if(az){ double l=det_old.Vector(var,k-1,j,i),r=det_old.Vector(var,k+1,j,i); mx=max(mx,max(l,r)); mn=min(mn,min(l,r)); }
            }
            double eps_m, eps_p;
            if(scale==_nad_scale_grange_ || scale==_nad_scale_gcfl_){
                double eps=max(tol*gs[var], atol);
                eps_m=max(eps, eps0*fabs(mn));
                eps_p=max(eps, eps0*fabs(mx));
            }else if(scale==_nad_scale_delta_){
                double eps=max(tol*(mx-mn), atol);
                eps_m=eps_p=eps;
            }else{
                eps_m=max(fabs(mn)*tol, atol);
                eps_p=max(fabs(mx)*tol, atol);
            }
            mn-=eps_m; mx+=eps_p;
            if(!isfinite(u_new) || u_new>mx || u_new<mn) trouble=1;
        }
        troubles.Vector(0,k,j,i)=trouble;
    });
}

// Physical-admissibility detection: floor/ceiling on density and on the gas pressure
// (total energy minus kinetic and magnetic energy), plus an isfinite check on the
// conserved state (AthenaK-style NaN demotion). Flags the aggregate slot. The floors
// are runtime parameters (fallback/min_rho, fallback/min_P): raising them above the
// ctoprim floors makes the detection catch degenerating low-beta cells before the
// primitive floors have to carry them. The raw (unfloored) internal energy is tested,
// so states the ctoprim floor would mask are still flagged.
void mhd_PAD(FV_Solution U, FV_Solution troubles){
    int Nx=U.Nx, Ny=U.Ny, Nz=U.Nz;
    double gm=cfg.gamma, mrho=cfg.pad_min_rho, mP=cfg.pad_min_P;
    fv_for_cells_ngh(Nz,Ny,Nx, KOKKOS_LAMBDA(int k,int j,int i){
        bool bad=false;
        for(int var=0;var<NMHD;var++){
            if(!isfinite(U.Vector(var,k,j,i))){ bad=true; break; }
        }
        double rho=U.Vector(_mrho_,k,j,i);
        double Ekin=0.5*(U.Vector(_mvx_,k,j,i)*U.Vector(_mvx_,k,j,i)
                        +U.Vector(_mvy_,k,j,i)*U.Vector(_mvy_,k,j,i)
                        +U.Vector(_mvz_,k,j,i)*U.Vector(_mvz_,k,j,i))/rho;
        double Emag=0.5*(U.Vector(_mbx_,k,j,i)*U.Vector(_mbx_,k,j,i)
                        +U.Vector(_mby_,k,j,i)*U.Vector(_mby_,k,j,i)
                        +U.Vector(_mbz_,k,j,i)*U.Vector(_mbz_,k,j,i));
        double p=(U.Vector(_mprs_,k,j,i)-Ekin-Emag)*(gm-1.);
        if(bad || !isfinite(p) || !isfinite(Ekin) || !isfinite(Emag))
            troubles.Vector(0,k,j,i)=1;
        if(rho<mrho || rho>rho_max) troubles.Vector(0,k,j,i)=1;
        if(p<mP || p>p_max)         troubles.Vector(0,k,j,i)=1;
    });
}

// Full MHD detection: build detection variables from the candidate/old cell-averaged
// conserved states, run NAD (B components or |B|), then the magnetic PAD. `troubles`
// (row 0) holds the per-cell flag consumed by the MOOD cascade (face/edge mask pooling).
void mhd_detect_troubles(FV_Solution U_new, FV_Solution U_old,
                         FV_Solution det_new, FV_Solution det_old,
                         FV_Solution troubles, bool PAD){
    int nvar=mhd_detection_vars(U_new,det_new);
    mhd_detection_vars(U_old,det_old);
    double gscale[8]={};
    // No stage dt available: grange only (skip CFL softening).
    mhd_nad_compute_gscales(det_old,U_old,nvar,gscale,0.0,1.0,1.0,1.0,false);
    mhd_NAD(det_new,det_old,troubles,cfg.nad_tolerance,nvar,gscale);
    if(PAD) mhd_PAD(U_new,troubles);
}

//========================================================================================
// Low-order finite-volume operators for the MOOD cascade
//
// These run in the ghosted FV cell layout on the primitive field W (8-var, cell
// averages, B rows = cell average of the face field). Both the low-order fluxes and the
// four-state edge E are built from the *same* W, exactly as the Python reference, so the
// candidate state is self-consistent. All assemblies are single-valued per face / per
// edge, which keeps the constrained-transport update divergence-free.
//========================================================================================

// Half-slope (minmod) of primitive `var` along `dim` at FV cell (k,j,i). minmod already
// returns 0.5*slope*(x_R-x_L), i.e. the half-increment from the cell centre to a face.
KOKKOS_INLINE_FUNCTION
double mhd_fv_dslope(FV_Vector W, int var, int k, int j, int i, int dim,
                     Vector x_c, Vector x_f, Vector y_c, Vector y_f, Vector z_c, Vector z_f, int lim){
    double w=W(var,k,j,i), wm, wp;
    if(dim==_x_){ wm=W(var,k,j,i-1); wp=W(var,k,j,i+1);
        return limited_slope((wp-w)/(x_c(i+1)-x_c(i)),(w-wm)/(x_c(i)-x_c(i-1)),
                             x_c(i+1)-x_c(i),x_c(i)-x_c(i-1),x_f(i),x_f(i+1),lim); }
    if(dim==_y_){ wm=W(var,k,j-1,i); wp=W(var,k,j+1,i);
        return limited_slope((wp-w)/(y_c(j+1)-y_c(j)),(w-wm)/(y_c(j)-y_c(j-1)),
                             y_c(j+1)-y_c(j),y_c(j)-y_c(j-1),y_f(j),y_f(j+1),lim); }
    wm=W(var,k-1,j,i); wp=W(var,k+1,j,i);
    return limited_slope((wp-w)/(z_c(k+1)-z_c(k)),(w-wm)/(z_c(k)-z_c(k-1)),
                         z_c(k+1)-z_c(k),z_c(k)-z_c(k-1),z_f(k),z_f(k+1),lim);
}

// Copy the staggered face field (SD layout, `dim`-normal) onto the FV face lattice, which
// is the same index space the low-order face fluxes are written on. Mirrors the index map
// used by fv_update_B_solution.
void mhd_face_B_to_fv(SD_Solution B, FV_Solution Bfv, int dim){
    int Nx=B.Nx, Ny=B.Ny, Nz=B.Nz;
    int px=B.nx, py=B.ny, pz=B.nz;
    int qx=px-(dim==_x_), qy=py-(dim==_y_), qz=pz-(dim==_z_);
    int Ni=Bfv.Nx, Nj=Bfv.Ny, Nk=Bfv.Nz;
    GHOST_LOCALS;
    sd_for_active_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii){
        if(K < Nk && J < Nj && I < Ni)
            Bfv.Vector(0,K,J,I) = B.Vector(0,0,k,j,i,kk,jj,ii);
    });
}

// MHD flux at every face of direction `dim` from the ghosted FV primitives W.
// muscl=true reconstructs the two face states with limited half-slopes; muscl=false uses
// the donor-cell (first-order) values. Writes into the FV face-flux array F (8-var).
// Uses cfg.rsolver (LLF or HLLD), matching the SD face Riemann path.
//
// Bn_f carries the constrained-transport face field normal to `dim` on the same lattice.
// The normal component is never reconstructed: both traces are overwritten with the
// single-valued, divergence-free face value (as AthenaK feeds b0.x1f into its solvers and
// RAMSES keeps the longitudinal field single-valued). HLLD is built on a continuous Bn --
// its wave fan, star states and momentum/induction rows all key off it -- so a per-cell
// reconstruction that disagrees with the field CT is actually evolving acts like a
// monopole source, one-sided because HLLD upwinds.
// UCT may be a null/empty view (n_var==0): then only F is written. With HLLD, UCT receives
// the five MDZ face coefficients used by mhd_uct_corner_E.
template<int D>
void mhd_fv_fluxes_t(FV_Solution W, FV_Solution F, FV_Solution Bn_f, FV_Solution UCT,
                     Vector x_c, Vector x_f, Vector y_c, Vector y_f, Vector z_c, Vector z_f,
                     bool muscl){
    const int lim = cfg.limiter;   //device cannot read cfg; capture then thread
    // Drive the face loop from the CELL-array extent (as hydro::fallback_fluxes does):
    // fv_for_faces then covers exactly the active faces, and the +-2 cell reconstruction
    // stays inside the (haloed) 2-ghost frame of W.
    int Nx=W.Nx, Ny=W.Ny, Nz=W.Nz;
    double gm=cfg.gamma;
    int rsolver=cfg.rsolver;
    bool want_uct = (rsolver==_rsolver_hlld_ && UCT.n_var>=NUCT);
    // Take the normal B from the single-valued CT face field rather than
    // reconstructing it. Gated on the solver for the same reason as the SD-side
    // mhd_face_B_to_fp: under llf the amr line reconstructed b1 like any other
    // primitive, and switching that is a deliberate default change needing the
    // MHD goldens regenerated, not a merge side effect. Under hlld this is the
    // documented behaviour and the UCT corner composition depends on it.
    const bool take_bn = (rsolver!=_rsolver_llf_);
    const int v1 = (D==_x_?_mvx_:(D==_y_?_mvy_:_mvz_));
    const int v2 = (D==_x_?_mvy_:(D==_y_?_mvz_:_mvx_));
    const int v3 = (D==_x_?_mvz_:(D==_y_?_mvx_:_mvy_));
    const int b1 = (D==_x_?_mbx_:(D==_y_?_mby_:_mbz_));
    const int b2 = (D==_x_?_mby_:(D==_y_?_mbz_:_mbx_));
    const int b3 = (D==_x_?_mbz_:(D==_y_?_mbx_:_mby_));
    fv_for_faces(Nz,Ny,Nx, KOKKOS_LAMBDA(int k,int j,int i){
        int kL=k-(D==_z_), jL=j-(D==_y_), iL=i-(D==_x_);
        double wL[NMHD], wR[NMHD], uL[NMHD], uR[NMHD], f[NMHD], uct[NUCT];
        for(int var=0;var<NMHD;var++){
            double dL = muscl ? mhd_fv_dslope(W.Vector,var,kL,jL,iL,D,x_c,x_f,y_c,y_f,z_c,z_f,lim) : 0.0;
            double dR = muscl ? mhd_fv_dslope(W.Vector,var,k ,j ,i ,D,x_c,x_f,y_c,y_f,z_c,z_f,lim) : 0.0;
            wL[var]=W.Vector(var,kL,jL,iL)+dL;   // right face of the left cell
            wR[var]=W.Vector(var,k ,j ,i )-dR;   // left  face of the right cell
        }
        if(take_bn) wL[b1]=wR[b1]=Bn_f.Vector(0,k,j,i);  // CT face field, never reconstructed
        mhd_conservatives(wL,uL,gm);
        mhd_conservatives(wR,uR,gm);
        mhd_riemann(f,uL,uR,v1,v2,v3,b1,b2,b3,gm,rsolver, want_uct?uct:nullptr);
        for(int var=0;var<NMHD;var++) F.Vector(var,k,j,i)=f[var];
        if(want_uct) for(int var=0;var<NUCT;var++) UCT.Vector(var,k,j,i)=uct[var];
    });
}

void mhd_fv_fluxes(FV_Solution W, FV_Solution F, FV_Solution Bn_f, FV_Solution UCT,
                   Vector x_c, Vector x_f, Vector y_c, Vector y_f, Vector z_c, Vector z_f,
                   int dim, bool muscl){
    if(dim==_x_)      mhd_fv_fluxes_t<_x_>(W,F,Bn_f,UCT,x_c,x_f,y_c,y_f,z_c,z_f,muscl);
    else if(dim==_y_) mhd_fv_fluxes_t<_y_>(W,F,Bn_f,UCT,x_c,x_f,y_c,y_f,z_c,z_f,muscl);
    else              mhd_fv_fluxes_t<_z_>(W,F,Bn_f,UCT,x_c,x_f,y_c,y_f,z_c,z_f,muscl);
}

//----------------------------------------------------------------------------------------
// Four-state corner electric field (mhd_fv_scheme.four_state_E), single-valued on the
// edge lattice. E-family `dim` has transverse directions (dim1,dim2); the four corner
// states are the (o1,o2) in {-1,0}^2 neighbour cells reconstructed towards the shared
// corner. E = v1*B2 - v2*B1 with v1,B1 along dim1 and v2,B2 along dim2. Writes the FV
// edge array E (1-var). muscl=false gives the first-order (donor-cell) corners.
//
// LLF: four-state average with fast-magnetosonic dissipation on the B jumps.
// HLLD (cfg.rsolver): two successive 1D HLLD sweeps (vel=3 then vel=4) on the
// 8-component edge state, mirroring spd.MHD.mhd_fv_scheme.four_state_E.
//----------------------------------------------------------------------------------------

// Reconstruct one primitive toward a corner from cell (ck,cj,ci).
KOKKOS_INLINE_FUNCTION
double mhd_fv_corner_val(FV_Solution W, int var, int ck, int cj, int ci,
                         int dim1, int dim2, double sgn1, double sgn2, bool muscl,
                         Vector x_c, Vector x_f, Vector y_c, Vector y_f,
                         Vector z_c, Vector z_f, int lim){
    double val = W.Vector(var,ck,cj,ci);
    if(muscl){
        val += sgn1*mhd_fv_dslope(W.Vector,var,ck,cj,ci,dim1,x_c,x_f,y_c,y_f,z_c,z_f,lim);
        val += sgn2*mhd_fv_dslope(W.Vector,var,ck,cj,ci,dim2,x_c,x_f,y_c,y_f,z_c,z_f,lim);
    }
    return val;
}

// Edge interface states of a face-lattice field between transverse indices lo and hi
// (hi = lo+1). muscl uses limited half-slopes; otherwise donor-cell. Returns (qL,qR).
KOKKOS_INLINE_FUNCTION
void mhd_edge_recon(FV_Solution Q, int var, int k, int j, int i,
                    int dim_t, int lo, bool muscl,
                    Vector x_c, Vector x_f, Vector y_c, Vector y_f, Vector z_c, Vector z_f,
                    double& qL, double& qR, int lim){
    int k0=k, j0=j, i0=i, k1=k, j1=j, i1=i;
    if(dim_t==_x_){ i0=lo; i1=lo+1; }
    else if(dim_t==_y_){ j0=lo; j1=lo+1; }
    else { k0=lo; k1=lo+1; }
    qL = Q.Vector(var,k0,j0,i0);
    qR = Q.Vector(var,k1,j1,i1);
    if(muscl){
        qL += mhd_fv_dslope(Q.Vector,var,k0,j0,i0,dim_t,x_c,x_f,y_c,y_f,z_c,z_f,lim);
        qR -= mhd_fv_dslope(Q.Vector,var,k1,j1,i1,dim_t,x_c,x_f,y_c,y_f,z_c,z_f,lim);
    }
}

KOKKOS_INLINE_FUNCTION
double mhd_edge_interp_a(FV_Solution Q, int var, int k, int j, int i,
                         int dim_t, int lo, bool muscl,
                         Vector x_c, Vector x_f, Vector y_c, Vector y_f, Vector z_c, Vector z_f, int lim){
    double qL, qR;
    mhd_edge_recon(Q,var,k,j,i,dim_t,lo,muscl,x_c,x_f,y_c,y_f,z_c,z_f,qL,qR,lim);
    double a = 0.5*(qL+qR);
    return a<0.0 ? 0.0 : (a>1.0 ? 1.0 : a);
}

KOKKOS_INLINE_FUNCTION
double mhd_edge_interp_d(FV_Solution Q, int var, int k, int j, int i,
                         int dim_t, int lo, bool muscl,
                         Vector x_c, Vector x_f, Vector y_c, Vector y_f, Vector z_c, Vector z_f, int lim){
    double qL, qR;
    mhd_edge_recon(Q,var,k,j,i,dim_t,lo,muscl,x_c,x_f,y_c,y_f,z_c,z_f,qL,qR,lim);
    double d = 0.5*(qL+qR);
    return d>0.0 ? d : 0.0;
}

// AthenaK UCT corner EMF (Berta+2024 Eq.39 / MDZ21 Eq.33), sign-flipped to the spd
// convention E = v×B (AthenaK stores E = -v×B). For E-family D with transverse dims
// (dim1,dim2): Bn1/UCT1 on dim1-faces, Bn2/UCT2 on dim2-faces. The a,d weights are
// crossed: a,d from dim1-faces multiply (v1,B2) from dim2-faces, and vice versa.
//   E = (aW v1W B2W + aE v1E B2E) - (aS v2S B1S + aN v2N B1N)
//     - (dE B2E - dW B2W) + (dN B1N - dS B1S)
// Nx,Ny,Nz are the CELL extents (same as mhd_four_state_E / mhd_fv_fluxes).
template<int D>
void mhd_uct_corner_E_t(FV_Solution E, FV_Solution Bn1, FV_Solution Bn2,
                        FV_Solution UCT1, FV_Solution UCT2,
                        int Nz, int Ny, int Nx,
                        Vector x_c, Vector x_f, Vector y_c, Vector y_f, Vector z_c, Vector z_f,
                        bool muscl){
    const int lim = cfg.limiter;   //device cannot read cfg; capture then thread
    const int dim1 = (D==_z_?_x_:(D==_y_?_z_:_y_));
    const int dim2 = (D==_z_?_y_:(D==_y_?_x_:_z_));
    // Face Riemann stores uct[3]=first transverse vel, uct[4]=second. For each face
    // normal the first transverse is the next cyclic direction, so:
    //   dim1-face contributes vel along dim2 as uct[3]
    //   dim2-face contributes vel along dim1 as uct[4]
    // (Ez: x-face vy=uct[3], y-face vx=uct[4]; Ex/Ey follow by the same cyclic rule.)
    const int ivt_from_1 = 3;
    const int ivt_from_2 = 4;

    fv_for_faces(Nz,Ny,Nx, KOKKOS_LAMBDA(int k,int j,int i){
        // Edge (k,j,i) at dim1-face f1 and dim2-face f2 (AthenaK i-1/2, j-1/2).
        // AthenaK crosses families: a,d from dim1-faces weight (v1,B2) from dim2-faces,
        // and a,d from dim2-faces weight (v2,B1) from dim1-faces.
        int f1 = (dim1==_x_?i:(dim1==_y_?j:k));
        int f2 = (dim2==_x_?i:(dim2==_y_?j:k));
        int lo1 = f1-1, lo2 = f2-1;

        int k1=k, j1=j, i1=i;
        if(dim1==_x_) i1=f1; else if(dim1==_y_) j1=f1; else k1=f1;
        int k2=k, j2=j, i2=i;
        if(dim2==_x_) i2=f2; else if(dim2==_y_) j2=f2; else k2=f2;

        // a,d for the (v1,B2) / W-E terms: from dim1-faces, along dim2
        double aW = mhd_edge_interp_a(UCT1,0,k1,j1,i1,dim2,lo2,muscl,x_c,x_f,y_c,y_f,z_c,z_f,lim);
        double aE = 1.0 - aW;
        double dW = mhd_edge_interp_d(UCT1,1,k1,j1,i1,dim2,lo2,muscl,x_c,x_f,y_c,y_f,z_c,z_f,lim);
        double dE = mhd_edge_interp_d(UCT1,2,k1,j1,i1,dim2,lo2,muscl,x_c,x_f,y_c,y_f,z_c,z_f,lim);
        // v1,B2 themselves: from dim2-faces, along dim1
        double v1W, v1E, B2W, B2E;
        mhd_edge_recon(UCT2,ivt_from_2,k2,j2,i2,dim1,lo1,muscl,x_c,x_f,y_c,y_f,z_c,z_f,v1W,v1E,lim);
        mhd_edge_recon(Bn2,0,k2,j2,i2,dim1,lo1,muscl,x_c,x_f,y_c,y_f,z_c,z_f,B2W,B2E,lim);

        // a,d for the (v2,B1) / S-N terms: from dim2-faces, along dim1
        double aS = mhd_edge_interp_a(UCT2,0,k2,j2,i2,dim1,lo1,muscl,x_c,x_f,y_c,y_f,z_c,z_f,lim);
        double aN = 1.0 - aS;
        double dS = mhd_edge_interp_d(UCT2,1,k2,j2,i2,dim1,lo1,muscl,x_c,x_f,y_c,y_f,z_c,z_f,lim);
        double dN = mhd_edge_interp_d(UCT2,2,k2,j2,i2,dim1,lo1,muscl,x_c,x_f,y_c,y_f,z_c,z_f,lim);
        // v2,B1 themselves: from dim1-faces, along dim2
        double v2S, v2N, B1S, B1N;
        mhd_edge_recon(UCT1,ivt_from_1,k1,j1,i1,dim2,lo2,muscl,x_c,x_f,y_c,y_f,z_c,z_f,v2S,v2N,lim);
        mhd_edge_recon(Bn1,0,k1,j1,i1,dim2,lo2,muscl,x_c,x_f,y_c,y_f,z_c,z_f,B1S,B1N,lim);

        E.Vector(0,k,j,i) = (aW*v1W*B2W + aE*v1E*B2E)
                          - (aS*v2S*B1S + aN*v2N*B1N)
                          - (dE*B2E - dW*B2W)
                          + (dN*B1N - dS*B1S);
    });
}

void mhd_uct_corner_E(FV_Solution E, FV_Solution Bx, FV_Solution By, FV_Solution Bz,
                      FV_Solution UCTx, FV_Solution UCTy, FV_Solution UCTz,
                      int Nz, int Ny, int Nx,
                      Vector x_c, Vector x_f, Vector y_c, Vector y_f, Vector z_c, Vector z_f,
                      int dim, bool muscl){
    if(dim==_x_)
        mhd_uct_corner_E_t<_x_>(E,By,Bz,UCTy,UCTz,Nz,Ny,Nx,x_c,x_f,y_c,y_f,z_c,z_f,muscl);
    else if(dim==_y_)
        mhd_uct_corner_E_t<_y_>(E,Bz,Bx,UCTz,UCTx,Nz,Ny,Nx,x_c,x_f,y_c,y_f,z_c,z_f,muscl);
    else
        mhd_uct_corner_E_t<_z_>(E,Bx,By,UCTx,UCTy,Nz,Ny,Nx,x_c,x_f,y_c,y_f,z_c,z_f,muscl);
}

// LLF four-state corner E (used when rsolver=llf).
template<int D>
void mhd_four_state_E_t(FV_Solution E, FV_Solution W,
                        Vector x_c, Vector x_f, Vector y_c, Vector y_f, Vector z_c, Vector z_f,
                        bool muscl){
    const int lim = cfg.limiter;   //device cannot read cfg; capture then thread
    // Drive from the CELL-array extent so the transverse +-2 reconstruction stays inside
    // W's haloed 2-ghost frame (see mhd_fv_fluxes_t).
    int Nx=W.Nx, Ny=W.Ny, Nz=W.Nz;
    double gm=cfg.gamma;
    // Transverse directions and matching velocity/field indices (E = v1 B2 - v2 B1).
    const int dim1 = (D==_z_?_x_:(D==_y_?_z_:_y_));
    const int dim2 = (D==_z_?_y_:(D==_y_?_x_:_z_));
    const int v1v = (D==_z_?_mvx_:(D==_y_?_mvz_:_mvy_));
    const int v2v = (D==_z_?_mvy_:(D==_y_?_mvx_:_mvz_));
    const int b1v = (D==_z_?_mbx_:(D==_y_?_mbz_:_mby_));
    const int b2v = (D==_z_?_mby_:(D==_y_?_mbx_:_mbz_));
    const int b3v = (D==_z_?_mbz_:(D==_y_?_mby_:_mbx_));
    // Iterate the active edge lattice (fv_for_faces range): the MUSCL corner
    // reconstruction reaches +-2 cells transversally, all within the 2 ghost layers.
    fv_for_faces(Nz,Ny,Nx, KOKKOS_LAMBDA(int k,int j,int i){
        double Esum=0, dB2_1=0, dB1_2=0, Sp1=0, Sp2=0;
        for(int o1=-1;o1<=0;o1++) for(int o2=-1;o2<=0;o2++){
            int ck=k, cj=j, ci=i;
            if(dim1==_x_) ci+=o1; else if(dim1==_y_) cj+=o1; else ck+=o1;
            if(dim2==_x_) ci+=o2; else if(dim2==_y_) cj+=o2; else ck+=o2;
            double sgn1 = (o1==-1)? 1.0 : -1.0;
            double sgn2 = (o2==-1)? 1.0 : -1.0;
            double rho = mhd_fv_corner_val(W,_mrho_,ck,cj,ci,dim1,dim2,sgn1,sgn2,muscl,
                                           x_c,x_f,y_c,y_f,z_c,z_f,lim);
            double p   = mhd_fv_corner_val(W,_mprs_,ck,cj,ci,dim1,dim2,sgn1,sgn2,muscl,
                                           x_c,x_f,y_c,y_f,z_c,z_f,lim);
            if(rho<=rho_min) rho=W.Vector(_mrho_,ck,cj,ci);
            if(p<=p_min)     p  =W.Vector(_mprs_,ck,cj,ci);
            double V1 = mhd_fv_corner_val(W,v1v,ck,cj,ci,dim1,dim2,sgn1,sgn2,muscl,
                                          x_c,x_f,y_c,y_f,z_c,z_f,lim);
            double V2 = mhd_fv_corner_val(W,v2v,ck,cj,ci,dim1,dim2,sgn1,sgn2,muscl,
                                          x_c,x_f,y_c,y_f,z_c,z_f,lim);
            double B1 = mhd_fv_corner_val(W,b1v,ck,cj,ci,dim1,dim2,sgn1,sgn2,muscl,
                                          x_c,x_f,y_c,y_f,z_c,z_f,lim);
            double B2 = mhd_fv_corner_val(W,b2v,ck,cj,ci,dim1,dim2,sgn1,sgn2,muscl,
                                          x_c,x_f,y_c,y_f,z_c,z_f,lim);
            double B3 = mhd_fv_corner_val(W,b3v,ck,cj,ci,dim1,dim2,sgn1,sgn2,muscl,
                                          x_c,x_f,y_c,y_f,z_c,z_f,lim);
            Esum += V1*B2 - V2*B1;
            double c = sqrt((gm*p + B1*B1 + B2*B2 + B3*B3)/rho);
            Sp1 = max(Sp1, fabs(V1)+c);
            Sp2 = max(Sp2, fabs(V2)+c);
            double js1 = (o1==0)? 1.0 : -1.0;
            double js2 = (o2==0)? 1.0 : -1.0;
            dB2_1 += 0.5*js1*B2;
            dB1_2 += 0.5*js2*B1;
        }
        E.Vector(0,k,j,i) = 0.25*Esum - 0.5*Sp1*dB2_1 + 0.5*Sp2*dB1_2;
    });
}

void mhd_four_state_E(FV_Solution E, FV_Solution W,
                      Vector x_c, Vector x_f, Vector y_c, Vector y_f, Vector z_c, Vector z_f,
                      int dim, bool muscl){
    if(dim==_x_)      mhd_four_state_E_t<_x_>(E,W,x_c,x_f,y_c,y_f,z_c,z_f,muscl);
    else if(dim==_y_) mhd_four_state_E_t<_y_>(E,W,x_c,x_f,y_c,y_f,z_c,z_f,muscl);
    else              mhd_four_state_E_t<_z_>(E,W,x_c,x_f,y_c,y_f,z_c,z_f,muscl);
}

//----------------------------------------------------------------------------------------
// MOOD cascade assembly: per-face flux and per-edge E selected from the level arrays
// (0 = high order, 1 = MUSCL, 2 = first order) by the pooled cascade index.
//   face level = max cascade index of the two adjacent cells
//   edge level = max cascade index of the (up to four) cells sharing the edge
//----------------------------------------------------------------------------------------
// Assembles IN PLACE into the level-0 array F0 (memory: no separate assembled copy).
// This is safe across revision sweeps because the cascade never decreases: a face whose
// F0 slot was overwritten has pooled level >= 1 and is never read at level 0 again.
// Edge E level pooling: the edge of family `dim` at (K,J,I) is shared by the cells offset
// by (o1,o2) in {-1,0} along the two transverse directions. Assembles IN PLACE into the
// level-0 array E0 (see mhd_assign_face_flux: valid because the cascade never decreases,
// and the single-valued overwrite keeps the CT curl divergence-free).
void mhd_assign_edge_E(FV_Solution E0, FV_Solution E1, FV_Solution E2,
                       FV_Solution cascade, int dim){
    int Nx=cascade.Nx, Ny=cascade.Ny, Nz=cascade.Nz;
    const int dim1 = (dim==_z_?_x_:(dim==_y_?_z_:_y_));
    const int dim2 = (dim==_z_?_y_:(dim==_y_?_x_:_z_));
    fv_for_faces(Nz,Ny,Nx, KOKKOS_LAMBDA(int k,int j,int i){
        double c=0;
        for(int o1=-1;o1<=0;o1++) for(int o2=-1;o2<=0;o2++){
            int ck=k, cj=j, ci=i;
            if(dim1==_x_) ci+=o1; else if(dim1==_y_) cj+=o1; else ck+=o1;
            if(dim2==_x_) ci+=o2; else if(dim2==_y_) cj+=o2; else ck+=o2;
            c=max(c,cascade.Vector(0,ck,cj,ci));
        }
        if(c>=1)
            E0.Vector(0,k,j,i) = c>=2 ? E2.Vector(0,k,j,i) : E1.Vector(0,k,j,i);
    });
}

// Replace the cell-averaged B rows of the candidate conserved FV state U_new with the
// candidate constrained-transport cell-averaged field, so the detection tests the true
// new B (NAD on B components or |B|, PAD pressure with magnetic energy). Only the
// CT-evolved (active direction) components are replaced: in 2D the Bz row is a plain
// cell-centered conserved variable already updated by the fluid fluxes (Python: only
// sim.dims rows).
void mhd_set_candidate_B(FV_Solution U_new, FV_Solution B_cand){
    int Nx=U_new.Nx, Ny=U_new.Ny, Nz=U_new.Nz;
    bool az=cfg.active[_z_];
    fv_for_cells(Nz,Ny,Nx, KOKKOS_LAMBDA(int k,int j,int i){
        U_new.Vector(_mbx_,k,j,i)=B_cand.Vector(0,k,j,i);
        U_new.Vector(_mby_,k,j,i)=B_cand.Vector(1,k,j,i);
        if(az) U_new.Vector(_mbz_,k,j,i)=B_cand.Vector(2,k,j,i);
    });
}

// Demote still-troubled, revisable cells one cascade level; returns the number demoted.
// Runs over the interior FV cells (the cascade ghosts are refreshed by a halo exchange).
//----------------------------------------------------------------------------------------
// CT-consistent divergence of the face-staggered field, evaluated at solution points:
//   div B = dBx_fp/dx + dBy_fp/dy + dBz_fp/dz
// using the same dfp_to_sp derivative that drives update_B_solution. For a
// divergence-free CT scheme this must stay at round-off for all time. Returns
// max|div B| over the active (non-ghost) solution points.
//----------------------------------------------------------------------------------------
double mhd_max_divB(SD_Solution Bx, SD_Solution By, SD_Solution Bz,
                    Matrix dfp_to_sp, double dx, double dy, double dz){
    int Nx=Bx.Nx, Ny=Bx.Ny, Nz=Bx.Nz, px=By.nx, py=Bx.ny, pz=Bx.nz;
    int qx=Bx.nx, qy=By.ny, qz=Bz.nz;   // p+2 flux points along the staggered dim
    bool ax=cfg.active[_x_], ay=cfg.active[_y_], az=cfg.active[_z_];
    double res = sd_max_cells(Nz,Ny,Nx,pz,py,px,
        KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii,double& reduce){
            double d=0;
            if(ax){ double s=0; for(int ll=0;ll<qx;ll++) s+=Bx.Vector(0,0,k,j,i,kk,jj,ll)*dfp_to_sp(ii,ll); d+=s/dx; }
            if(ay){ double s=0; for(int ll=0;ll<qy;ll++) s+=By.Vector(0,0,k,j,i,kk,ll,ii)*dfp_to_sp(jj,ll); d+=s/dy; }
            if(az){ double s=0; for(int ll=0;ll<qz;ll++) s+=Bz.Vector(0,0,k,j,i,ll,jj,ii)*dfp_to_sp(kk,ll); d+=s/dz; }
            d=fabs(d);
            reduce = reduce > d ? reduce : d;
        });
    #ifdef MPI
    double g; MPI_Allreduce(&res,&g,1,MPI_DOUBLE,MPI_MAX,Comm); return g;
    #else
    return res;
    #endif
}

//----------------------------------------------------------------------------------------
// Initial conditions
//----------------------------------------------------------------------------------------

// Primitive MHD state (rho, vx, vy, vz, P, Bx, By, Bz). B is set from the
// vector potential separately (this only seeds the fluid rows + a placeholder B
// that is immediately overwritten by B_to_U). Orszag-Tang vortex (gamma=5/3).
// Balsara (2004) / Leidi et al. (2022) magnetized vortex. When pp.sigma>0 use the
// Leidi kernel exp((1-r²)/2) on [-5,5]²; otherwise the Balsara-2004 kernel exp(1-r²)
// with uniform background flow (v1,v2). pp.amp = vortex strength, pp.p1 = |B| scale,
// pp.v3 = uniform Bz for the 3D extension.
KOKKOS_INLINE_FUNCTION
double mhd_ic_vortex(int var, double x, double y, double z, ProblemParams pp){
    double xr = x - pp.cx;
    double yr = y - pp.cy;
    double r2 = xr*xr + yr*yr;
    bool leidi = pp.sigma > 0.0;
    double ker = leidi ? exp(0.5*(1.0 - r2)) : exp(1.0 - r2);
    double V = pp.amp;
    double B = pp.p1;
    if(var==_mrho_) return 1.0;
    if(var==_mvx_)  return pp.v1 - V*ker*yr;
    if(var==_mvy_)  return pp.v2 + V*ker*xr;
    if(var==_mvz_)  return 0.0;
    if(var==_mprs_) {
        if(leidi)
            return pp.p0 + 0.5*B*B*(1.0 - r2) - 0.5*V*V*exp(1.0 - r2);
        return pp.p0 - 0.5*V*V*ker*ker*r2;
    }
    if(var==_mbx_) return -B*ker*yr;
    if(var==_mby_) return  B*ker*xr;
    if(var==_mbz_) return  pp.v3;
    return 0.0;
}

// Wu & Shu (2018) strongly magnetized MHD blast (Balsara et al. 2025 §8.1).
// Uniform rho=1, v=0, p=p0 except r<radius where p=p1, Bx=amp.
// Smooth-interface Kelvin-Helmholtz for MHD: Stone et al. (2020) figure 22,
// which is figure 21's hydro setup plus a uniform horizontal field Bx = 0.1
// (here pp.amp, matching mhd_ic_blast/mhd_ic_jet). Reference time t = 1.5.
//
// The profile MUST stay identical to the hydro kelvin_helmholtz() in
// initial_conditions.cpp -- the point of the test is that the two differ only
// by B. That includes the correction of the paper's eq. 26 typo: it writes the
// tanh argument as |y - 0.25|, which gives ONE interface, a density contrast of
// 1.5 and a velocity jump of 0.5, contradicting its own text and leaving the
// state non-periodic in y. The intended argument is (|y| - 0.25).
KOKKOS_INLINE_FUNCTION
double mhd_ic_kelvin_helmholtz(int var, double x, double y, ProblemParams pp){
    const double Lsh = 0.01;   //shear layer thickness
    const double amp = 0.01;   //velocity perturbation amplitude
    const double sig = 0.2;    //thickness of the perturbed layer
    double dy = fabs(y - 0.5*LENGHT) - 0.25;
    double s  = tanh(dy/Lsh);
    if(var==_mrho_) return 1.5 - 0.5*s;
    if(var==_mvx_)  return 0.5*s;
    if(var==_mvy_)  return amp*cos(4*PI*x)*exp(-(dy*dy)/(sig*sig));
    if(var==_mprs_) return 2.5;
    if(var==_mbx_)  return pp.amp;
    return 0.0;
}

KOKKOS_INLINE_FUNCTION
double mhd_ic_blast(int var, double x, double y, double z, ProblemParams pp){
    double xr = x - pp.cx;
    double yr = y - pp.cy;
    double r = sqrt(xr*xr + yr*yr);
    if(var==_mrho_) return pp.d0;
    if(var==_mprs_) return r < pp.radius ? pp.p1 : pp.p0;
    if(var==_mbx_)  return pp.amp;
    return 0.0;
}

// Wu & Shu (2018) Mach-800 magnetized jet ambient (+ nozzle IC at t=0).
// Ambient: rho=d0, p=p0, v=0, B=(0,amp,0). Nozzle (|x-cx|<radius): rho=d1, vy=v2.
KOKKOS_INLINE_FUNCTION
double mhd_ic_jet(int var, double x, double y, double z, ProblemParams pp){
    bool nozzle = fabs(x - pp.cx) < pp.radius;
    if(var==_mrho_) return nozzle ? pp.d1 : pp.d0;
    if(var==_mvy_)  return nozzle ? pp.v2 : 0.0;
    if(var==_mprs_) return pp.p0;
    if(var==_mby_)  return pp.amp;
    return 0.0;
}

KOKKOS_INLINE_FUNCTION
double mhd_ic_jet_inflow(int var, double x, double y, ProblemParams pp){
    return mhd_ic_jet(var, x, y, 0.0, pp);
}

KOKKOS_INLINE_FUNCTION
double mhd_ic_orszag_tang(int var, double x, double y, double z){
    const double rho0 = 25.0/(36.0*PI);
    const double p0   = 5.0/(12.0*PI);
    if(var==_mrho_) return rho0;
    if(var==_mvx_)  return -sin(2*PI*y);
    if(var==_mvy_)  return  sin(2*PI*x);
    if(var==_mprs_) return p0;
    return 0.0;
}

// Field-loop advection (Gardiner & Stone): weak magnetic loop advected
// diagonally by a uniform flow; B stays a passive loop and div(B)=0 must hold.
// Quasi-2D setup on the z-invariant slab: rho=1, p=1, v=(2,1,0).
KOKKOS_INLINE_FUNCTION
double mhd_ic_field_loop(int var, double x, double y, double z){
    if(var==_mrho_) return 1.0;
    if(var==_mvx_)  return 2.0;
    if(var==_mvy_)  return 1.0;
    if(var==_mprs_) return 1.0;
    return 0.0;
}

KOKKOS_INLINE_FUNCTION
double mhd_ic_primitive(int problem, int var, double x, double y, double z, ProblemParams pp){
    if(problem==_ic_field_loop_) return mhd_ic_field_loop(var,x,y,z);
    if(problem==_ic_mhd_vortex_) return mhd_ic_vortex(var,x,y,z,pp);
    if(problem==_ic_mhd_blast_)   return mhd_ic_blast(var,x,y,z,pp);
    if(problem==_ic_mhd_jet_)     return mhd_ic_jet(var,x,y,z,pp);
    if(problem==_ic_kelvin_helmholtz_) return mhd_ic_kelvin_helmholtz(var,x,y,pp);
    return mhd_ic_orszag_tang(var,x,y,z);
}

// Vector potential component (edge-point init for CT).
// Orszag-Tang: only Az != 0,
//   Az = B0/(2pi) cos(2pi y) + B0/(4pi) cos(4pi x),  B0 = 1/sqrt(4pi)
// gives Bx = dAz/dy = -B0 sin(2pi y), By = -dAz/dx = B0 sin(4pi x).
// Field loop: Az = A0 (R - r) inside r < R (loop centred at (0.5,0.5)).
KOKKOS_INLINE_FUNCTION
double mhd_ic_vector_potential(int problem, int dim, double x, double y, double z, ProblemParams pp){
    if(problem==_ic_field_loop_){
        const double A0=1e-3, R=0.3;
        if(dim==_z_){
            double r = sqrt((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5));
            return r<R ? A0*(R-r) : 0.0;
        }
        return 0.0;
    }
    if(problem==_ic_mhd_vortex_){
        double xr = x - pp.cx;
        double yr = y - pp.cy;
        double r2 = xr*xr + yr*yr;
        if(dim==_z_){
            if(pp.sigma > 0.0)
                return pp.p1*exp(0.5*(1.0 - r2));
            return 0.5*pp.p1*exp(1.0 - r2);
        }
        if(dim==_x_) return -0.5*pp.v3*yr;
        if(dim==_y_) return  0.5*pp.v3*xr;
        return 0.0;
    }
    if(problem==_ic_mhd_blast_){
        if(dim==_z_) return pp.amp*y;
        return 0.0;
    }
    if(problem==_ic_mhd_jet_){
        if(dim==_z_) return -pp.amp*x;
        return 0.0;
    }
    //KH (fig 22): uniform Bx = pp.amp. Bx = dAz/dy - dAy/dz, so Az = amp*y.
    if(problem==_ic_kelvin_helmholtz_){
        if(dim==_z_) return pp.amp*y;
        return 0.0;
    }
    const double B0 = 1.0/sqrt(4.0*PI);
    if(dim==_z_)
        return B0/(2*PI)*cos(2*PI*y) + B0/(4*PI)*cos(4*PI*x);
    return 0.0;
}

void mhd_Initialize(SD_Solution W, Matrix faces_x, Matrix faces_y, Matrix faces_z,
                    Vector x_sp, Vector w_sp){
    int Nx=W.Nx, Ny=W.Ny, Nz=W.Nz, px=W.nx, py=W.ny, pz=W.nz;
    int problem=cfg.problem;
    bool ay=cfg.active[_y_], az=cfg.active[_z_];
    ProblemParams pp = cfg.pp;
#ifdef KOKKOS_ENABLE_CUDA
    Matrix_h fx = setup_mirror(faces_x); setup_pull(faces_x, fx);
    Matrix_h fy = setup_mirror(faces_y); setup_pull(faces_y, fy);
    Matrix_h fz = setup_mirror(faces_z); setup_pull(faces_z, fz);
    Vector_h xs = setup_mirror(x_sp); setup_pull(x_sp, xs);
    Vector_h ws = setup_mirror(w_sp); setup_pull(w_sp, ws);
    SD_Vector_h Wh = Kokkos::create_mirror_view(W.Vector);
    sd_for_cells_host(Nz,Ny,Nx,pz,py,px, [&](int k,int j,int i,int kk,int jj,int ii){
        for(int var=0;var<NMHD;var++){
            double value=0, x, y=0, z=0;
            for(int nn=0;nn<pz;nn++){
                if(az) z = fz(k,kk) + xs(nn)*(fz(k,kk+1)-fz(k,kk));
                for(int mm=0;mm<py;mm++){
                    if(ay) y = fy(j,jj) + xs(mm)*(fy(j,jj+1)-fy(j,jj));
                    for(int ll=0;ll<px;ll++){
                        x = fx(i,ii) + xs(ll)*(fx(i,ii+1)-fx(i,ii));
                        double s = mhd_ic_primitive(problem,var,x,y,z,pp)*ws(ll);
                        if(ay) s*=ws(mm);
                        if(az) s*=ws(nn);
                        value+=s;
                    }
                }
            }
            Wh(0,var,k,j,i,kk,jj,ii)=value;
        }
    });
    Kokkos::deep_copy(W.Vector, Wh);
#else
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii){
        for(int var=0;var<NMHD;var++){
            double value=0, x, y=0, z=0;
            for(int nn=0;nn<pz;nn++){
                if(az) z = faces_z(k,kk) + x_sp(nn)*(faces_z(k,kk+1)-faces_z(k,kk));
                for(int mm=0;mm<py;mm++){
                    if(ay) y = faces_y(j,jj) + x_sp(mm)*(faces_y(j,jj+1)-faces_y(j,jj));
                    for(int ll=0;ll<px;ll++){
                        x = faces_x(i,ii) + x_sp(ll)*(faces_x(i,ii+1)-faces_x(i,ii));
                        double s = mhd_ic_primitive(problem,var,x,y,z,pp)*w_sp(ll);
                        if(ay) s*=w_sp(mm);
                        if(az) s*=w_sp(nn);
                        value+=s;
                    }
                }
            }
            W.Vector(0,var,k,j,i,kk,jj,ii)=value;
        }
    });
#endif
}

void mhd_Initialize_A(SD_Solution A, Matrix Xs, Matrix Ys, Matrix Zs, int dim){
    int Nx=A.Nx, Ny=A.Ny, Nz=A.Nz, px=A.nx, py=A.ny, pz=A.nz;
    int problem=cfg.problem;
    ProblemParams pp = cfg.pp;
#ifdef KOKKOS_ENABLE_CUDA
    Matrix_h Xh = setup_mirror(Xs); setup_pull(Xs, Xh);
    Matrix_h Yh = setup_mirror(Ys); setup_pull(Ys, Yh);
    Matrix_h Zh = setup_mirror(Zs); setup_pull(Zs, Zh);
    SD_Vector_h Ah = Kokkos::create_mirror_view(A.Vector);
    sd_for_cells_host(Nz,Ny,Nx,pz,py,px, [&](int k,int j,int i,int kk,int jj,int ii){
        double x=Xh(i,ii), y=Yh(j,jj), z=Zh(k,kk);
        Ah(0,0,k,j,i,kk,jj,ii)=mhd_ic_vector_potential(problem,dim,x,y,z,pp);
    });
    Kokkos::deep_copy(A.Vector, Ah);
#else
    SD_Vector Va = A.Vector;
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii){
        double x=Xs(i,ii), y=Ys(j,jj), z=Zs(k,kk);
        Va(0,0,k,j,i,kk,jj,ii)=mhd_ic_vector_potential(problem,dim,x,y,z,pp);
    });
#endif
}

// Overwrite y-min ghost element with jet/nozzle primitives, then conservatives.
void mhd_jet_inflow_apply(SD_Solution W, SD_Solution U,
                            Matrix faces_x, Matrix faces_y,
                            Vector x_sp, Vector w_sp){
    int Nx=W.Nx, Ny=W.Ny, Nz=W.Nz, px=W.nx, py=W.ny, pz=W.nz;
    ProblemParams pp = cfg.pp;
    double gm = cfg.gamma;
#ifdef KOKKOS_ENABLE_CUDA
    Matrix_h fx = setup_mirror(faces_x); setup_pull(faces_x, fx);
    Matrix_h fy = setup_mirror(faces_y); setup_pull(faces_y, fy);
    Vector_h xs = setup_mirror(x_sp); setup_pull(x_sp, xs);
    Vector_h ws = setup_mirror(w_sp); setup_pull(w_sp, ws);
    SD_Vector_h Wh = Kokkos::create_mirror_view(W.Vector);
    SD_Vector_h Uh = Kokkos::create_mirror_view(U.Vector);
    Kokkos::deep_copy(Wh, W.Vector);
    Kokkos::deep_copy(Uh, U.Vector);
    sd_for_cells_host(Nz,Ny,Nx,pz,py,px, [&](int k,int j,int i,int kk,int jj,int ii){
        if(j > 1) return;
        double w[NMHD], u[NMHD];
        for(int var=0;var<NMHD;var++) w[var]=0;
        for(int nn=0;nn<pz;nn++){
            for(int mm=0;mm<py;mm++){
                double y = fy(j,jj) + xs(mm)*(fy(j,jj+1)-fy(j,jj));
                for(int ll=0;ll<px;ll++){
                    double x = fx(i,ii) + xs(ll)*(fx(i,ii+1)-fx(i,ii));
                    double wt = ws(ll);
                    if(py>1) wt*=ws(mm);
                    if(pz>1) wt*=ws(nn);
                    for(int var=0;var<NMHD;var++)
                        w[var]+=mhd_ic_jet_inflow(var,x,y,pp)*wt;
                }
            }
        }
        mhd_conservatives(w,u,gm);
        for(int var=0;var<NMHD;var++){
            Wh(0,var,k,j,i,kk,jj,ii)=w[var];
            Uh(0,var,k,j,i,kk,jj,ii)=u[var];
        }
    });
    Kokkos::deep_copy(W.Vector, Wh);
    Kokkos::deep_copy(U.Vector, Uh);
#else
    SD_Vector Vw = W.Vector;
    SD_Vector Vu = U.Vector;
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii){
        if(j > 1) return;
        double w[NMHD], u[NMHD];
        for(int var=0;var<NMHD;var++) w[var]=0;
        for(int nn=0;nn<pz;nn++){
            for(int mm=0;mm<py;mm++){
                double y = faces_y(j,jj) + x_sp(mm)*(faces_y(j,jj+1)-faces_y(j,jj));
                for(int ll=0;ll<px;ll++){
                    double x = faces_x(i,ii) + x_sp(ll)*(faces_x(i,ii+1)-faces_x(i,ii));
                    double wt = w_sp(ll);
                    if(py>1) wt*=w_sp(mm);
                    if(pz>1) wt*=w_sp(nn);
                    for(int var=0;var<NMHD;var++)
                        w[var]+=mhd_ic_jet_inflow(var,x,y,pp)*wt;
                }
            }
        }
        mhd_conservatives(w,u,gm);
        for(int var=0;var<NMHD;var++){
            Vw(0,var,k,j,i,kk,jj,ii)=w[var];
            Vu(0,var,k,j,i,kk,jj,ii)=u[var];
        }
    });
#endif
}
