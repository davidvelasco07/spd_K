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
    //The CUDA branch here was a full HOST ROUND TRIP: pull the whole 8-variable
    //SD array to a mirror, compute on the host, push it back -- every stage, on
    //every block. The #else branch below did the identical work as a device
    //kernel, so CUDA was paying hundreds of MB of transfer per stage to run the
    //same arithmetic more slowly. Measured on a 1024^2 single block, 200 steps:
    //MHD cost 64.5 s against the hydro PLM lane's 1.28 s -- 50x, for a scheme
    //that should cost about 3x. Same defect class as project_face_to_row
    //(01a2913, f089a35) and mhd_face_B_to_fp (ef3d49a); this makes six.
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
}

void mhd_compute_primitives(SD_Solution U, SD_Solution W){
    int Nx=W.Nx, Ny=W.Ny, Nz=W.Nz, px=W.nx, py=W.ny, pz=W.nz;
    int nader=W.n_ader;
    double gm=cfg.gamma;
    double dfl=cfg.dfloor, pfl=cfg.pfloor;
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
}

//Batched over the block axis: one launch for the whole pack instead of one per
//block. The per-block loop this replaces was 1.903 s of rk/copy_cons and 0.898 s
//of rk/cons_to_prim on a 352-leaf mesh over 109 steps, against 0.005 s and
//0.006 s for the already-batched hydro equivalents -- 380x and 150x on identical
//work. SD packs fold the block into the leading (n_ader) axis, not the variable
//axis, hence boff = b*nader.
void mhd_compute_primitives_b(SD_Solution U, SD_Solution W){
    int Nx=W.Nx, Ny=W.Ny, Nz=W.Nz, px=W.nx, py=W.ny, pz=W.nz, nb=W.nb;
    int nader=W.n_ader;
    double gm=cfg.gamma;
    double dfl=cfg.dfloor, pfl=cfg.pfloor;
    SD_Vector Vu = U.Vector;
    SD_Vector Vw = W.Vector;
    sd_for_cells_b(nb,Nz,Ny,Nx,pz,py,px,
        KOKKOS_LAMBDA(int b,int k,int j,int i,int kk,int jj,int ii){
        const int boff = b*nader;
        for(int t_id=0;t_id<nader;t_id++){
            double u[NMHD], w[NMHD];
            for(int var=0;var<NMHD;var++) u[var]=Vu(boff+t_id,var,k,j,i,kk,jj,ii);
            mhd_primitives(u,w,gm,dfl,pfl);
            for(int var=0;var<NMHD;var++) Vw(boff+t_id,var,k,j,i,kk,jj,ii)=w[var];
        }
    }, "mhd_compute_primitives_b");
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

// Pack-wide form of the FV mhd_compute_primitives.
void mhd_compute_primitives_b(FV_Solution U, FV_Solution W){
    int nb=U.nb, Nx=U.Nx, Ny=U.Ny, Nz=U.Nz;
    int nvu=U.n_var, nvw=W.n_var;
    double gm=cfg.gamma;
    double dfl=cfg.dfloor, pfl=cfg.pfloor;
    fv_for_cells_b(nb,Nz,Ny,Nx, KOKKOS_LAMBDA(int b,int k,int j,int i){
        const int uo=b*nvu, wo=b*nvw;
        double u[NMHD], w[NMHD];
        for(int var=0;var<NMHD;var++) u[var]=U.Vector(uo+var,k,j,i);
        mhd_primitives(u,w,gm,dfl,pfl);
        for(int var=0;var<NMHD;var++) W.Vector(wo+var,k,j,i)=w[var];
    }, "mhd_compute_primitives_b");
}

// Pack-wide form of mhd_detection_vars. The returned count is a pure function of
// cfg, identical for every block, so it needs no reduction.
int mhd_detection_vars_b(FV_Solution U, FV_Solution det){
    int nb=U.nb, Nx=U.Nx, Ny=U.Ny, Nz=U.Nz;
    int nvu=U.n_var, nvd=det.n_var;
    double gm=cfg.gamma;
    int bmode=cfg.mood_nad_b, vmode=cfg.mood_nad_v;
    fv_for_cells_b(nb,Nz,Ny,Nx, KOKKOS_LAMBDA(int b,int k,int j,int i){
        const int uo=b*nvu, doff=b*nvd;
        double u[NMHD], w[NMHD];
        for(int var=0;var<NMHD;var++) u[var]=U.Vector(uo+var,k,j,i);
        mhd_primitives(u,w,gm);
        det.Vector(doff+0,k,j,i)=w[_mrho_];
        det.Vector(doff+1,k,j,i)=w[_mprs_];
        int r=2;
        if(bmode==_nad_b_mag_){
            double B2=w[_mbx_]*w[_mbx_]+w[_mby_]*w[_mby_]+w[_mbz_]*w[_mbz_];
            det.Vector(doff+(r++),k,j,i)=sqrt(B2);
        }else{
            det.Vector(doff+(r++),k,j,i)=w[_mbx_];
            det.Vector(doff+(r++),k,j,i)=w[_mby_];
            det.Vector(doff+(r++),k,j,i)=w[_mbz_];
        }
        if(vmode==_nad_v_mag_){
            double v2=w[_mvx_]*w[_mvx_]+w[_mvy_]*w[_mvy_]+w[_mvz_]*w[_mvz_];
            det.Vector(doff+(r++),k,j,i)=sqrt(v2);
        }else if(vmode==_nad_v_comps_){
            det.Vector(doff+(r++),k,j,i)=w[_mvx_];
            det.Vector(doff+(r++),k,j,i)=w[_mvy_];
            det.Vector(doff+(r++),k,j,i)=w[_mvz_];
        }
    }, "mhd_detection_vars_b");
    int nvar=2;
    nvar += (bmode==_nad_b_mag_) ? 1 : 3;
    if(vmode==_nad_v_mag_) nvar += 1;
    else if(vmode==_nad_v_comps_) nvar += 3;
    return nvar;
}

// Pack-wide form of mhd_face_B_to_fv.
void mhd_face_B_to_fv_b(SD_Solution B, FV_Solution Bfv, int dim){
    int nb=B.nb;
    int Nx=B.Nx, Ny=B.Ny, Nz=B.Nz;
    int px=B.nx, py=B.ny, pz=B.nz;
    int qx=px-(dim==_x_), qy=py-(dim==_y_), qz=pz-(dim==_z_);
    int Ni=Bfv.Nx, Nj=Bfv.Ny, Nk=Bfv.Nz;
    int na=B.n_ader, nvf=Bfv.n_var;
    GHOST_LOCALS;
    sd_for_active_cells_b(nb,Nz,Ny,Nx,pz,py,px,
        KOKKOS_LAMBDA(int b,int k,int j,int i,int kk,int jj,int ii){
        BOFF(na);
        if(K < Nk && J < Nj && I < Ni)
            Bfv.Vector(b*nvf+0,K,J,I) = B.Vector(boff+0,0,k,j,i,kk,jj,ii);
    }, "mhd_face_B_to_fv_b");
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

// SPD_EMF_TRACE=1 names the EMF path each call actually takes. This exists because
// the UCT kernels sat in the tree with ZERO call sites for a release while
// docs/mhd.md described them as live: mhd_uct_edge_E was never called (every face
// solve passed a default-empty UCT view, so want_uct was false) and
// mhd_uct_corner_E was gated on a `use_uct` member hardwired to false. Both looked
// wired up on a read. Counting calls is the only thing that settles it, and the
// trace prints for the legacy paths too, so a silent path is real silence rather
// than a dead printf. getenv is cached: these are host functions on the per-stage
// path (the codebase idiom, cf. faceB_prolong_const in amr.cpp).
inline bool emf_trace(){
    static const bool v = getenv("SPD_EMF_TRACE") != nullptr;
    return v;
}
#define EMF_TRACE(name) do{ if(emf_trace()) printf("[emf] %s\n", name); }while(0)

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

// The two-wave HLL average, given the states, their fluxes and the outer fan.
// Factored out so mhd_riemann_hll and HLLD's degenerate-fan fallback are the SAME
// arithmetic rather than two copies that can drift (CLAUDE.md rule 4). It takes
// the pieces already computed instead of recomputing them, so substituting it
// into the HLLD fallback is bit-identical to the inline form it replaced.
KOKKOS_INLINE_FUNCTION
void mhd_hll_flux(double* f, const double* uL, const double* uR,
                  const double* fL, const double* fR, double S_L, double S_R){
    if(S_L >= 0.0)      for(int var=0;var<NMHD;var++) f[var]=fL[var];
    else if(S_R <= 0.0) for(int var=0;var<NMHD;var++) f[var]=fR[var];
    else                for(int var=0;var<NMHD;var++)
        f[var] = (S_R*fL[var] - S_L*fR[var] + S_L*S_R*(uR[var]-uL[var]))/(S_R - S_L);
}

// HLL (MDZ21 eq. 28): the two-wave solver, and the base scheme whose fan supplies
// the UCT-HLL emf coefficients. Same outer wave-speed estimate as HLLD (eq. 67),
// so the two solvers see an identical fan and differ only in what they do inside
// it. If uct != nullptr, also writes the NUCT face coefficients.
KOKKOS_INLINE_FUNCTION
void mhd_riemann_hll(double* f, double* uL, double* uR,
                     int v1, int v2, int v3, int b1, int b2, int b3, double gm,
                     double* uct=nullptr){
    double wL[NMHD], wR[NMHD], fL[NMHD], fR[NMHD];
    mhd_primitives(uL,wL,gm);
    mhd_primitives(uR,wR,gm);
    mhd_fluxes(wL,fL,v1,v2,v3,b1,b2,b3,gm);
    mhd_fluxes(wR,fR,v1,v2,v3,b1,b2,b3,gm);

    double c_L = mhd_fast_vel(wL[_mprs_],wL[_mrho_],wL[b1],wL[b2],wL[b3],gm);
    double c_R = mhd_fast_vel(wR[_mprs_],wR[_mrho_],wR[b1],wR[b2],wR[b3],gm);
    double c_max = c_L>c_R ? c_L : c_R;
    double u_L=wL[v1], u_R=wR[v1];
    double S_L = (u_L<u_R ? u_L : u_R) - c_max;
    double S_R = (u_L>u_R ? u_L : u_R) + c_max;

    mhd_hll_flux(f,uL,uR,fL,fR,S_L,S_R);
    if(uct) mhd_uct_hll_coeffs(S_L,S_R,wL[v2],wR[v2],wL[v3],wR[v3],uct);
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
        mhd_hll_flux(f,uL,uR,fL,fR,S_L,S_R);
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
    else if(rsolver==_rsolver_hll_)
        mhd_riemann_hll(f,uL,uR,v1,v2,v3,b1,b2,b3,gm,uct);
    else
        mhd_riemann_llf(f,uL,uR,v1,v2,v3,b1,b2,b3,gm);
}

// True when this face solver exposes a wave fan the UCT coefficients can be read
// off. llf has none (it is a single-speed bound), so mhd/emf=uct is refused for
// it in main.cpp rather than silently falling back to the two-sweep edge.
KOKKOS_INLINE_FUNCTION
bool mhd_rsolver_has_fan(int rsolver){
    return rsolver==_rsolver_hlld_ || rsolver==_rsolver_hll_;
}

template<int D,int V1,int V2,int V3,int B1,int B2,int B3>
void mhd_riemann_solver_t(SD_Solution U, SD_Solution F, SD_Solution Bn, SD_Solution UCT){
    int Nx=U.Nx-(D==_x_), Ny=U.Ny-(D==_y_), Nz=U.Nz-(D==_z_);
    int px=D==_x_?1:U.nx, py=D==_y_?1:U.ny, pz=D==_z_?1:U.nz;
    int n=mhd_choose(D,U.nx,U.ny,U.nz);
    int nader=U.n_ader;
    double gm=cfg.gamma;
    int rsolver=cfg.rsolver;
    bool want_uct = (mhd_rsolver_has_fan(rsolver) && UCT.n_var>=NUCT);
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

//Pack-wide form of mhd_riemann_solver.
template<int D,int V1,int V2,int V3,int B1,int B2,int B3>
void mhd_riemann_solver_t_b(SD_Solution U, SD_Solution F, SD_Solution Bn, SD_Solution UCT){
    int nb=U.nb;
    int Nx=U.Nx-(D==_x_), Ny=U.Ny-(D==_y_), Nz=U.Nz-(D==_z_);
    int px=D==_x_?1:U.nx, py=D==_y_?1:U.ny, pz=D==_z_?1:U.nz;
    int n=mhd_choose(D,U.nx,U.ny,U.nz);
    int nader=U.n_ader;
    int nab=Bn.n_var>=1 ? Bn.n_ader : 1;
    double gm=cfg.gamma;
    int rsolver=cfg.rsolver;
    bool want_uct = (mhd_rsolver_has_fan(rsolver) && UCT.n_var>=NUCT);
    int nau=want_uct ? UCT.n_ader : 1;
    bool use_bn = (Bn.n_var>=1);
    sd_for_cells_b(nb,Nz,Ny,Nx,pz,py,px,
        KOKKOS_LAMBDA(int b,int k,int j,int i,int kk,int jj,int ii){
        BOFF(nader);
        double uL[NMHD], uR[NMHD], f[NMHD], uct[NUCT];
        int NidL[3],nidL[3],NidR[3],nidR[3];
        int l=mhd_choose(D,i,j,k);
        mhd_indices(NidL,nidL,k,j,i,kk,jj,ii,l  ,n-1,D);
        mhd_indices(NidR,nidR,k,j,i,kk,jj,ii,l+1,0  ,D);
        for(int t_id=0;t_id<nader;t_id++){
            for(int var=0;var<NMHD;var++){
                uL[var]=U.Vector(INDICES_L_B);
                uR[var]=U.Vector(INDICES_R_B);
            }
            if(use_bn){
                double bnL = Bn.Vector(b*nab+0,0,NidL[_z_],NidL[_y_],NidL[_x_],
                                       nidL[_z_],nidL[_y_],nidL[_x_]);
                double bnR = Bn.Vector(b*nab+0,0,NidR[_z_],NidR[_y_],NidR[_x_],
                                       nidR[_z_],nidR[_y_],nidR[_x_]);
                double bn = 0.5*(bnL+bnR);
                uL[B1]=uR[B1]=bn;
            }
            mhd_riemann(f,uL,uR,V1,V2,V3,B1,B2,B3,gm,rsolver, want_uct?uct:nullptr);
            for(int var=0;var<NMHD;var++){
                F.Vector(INDICES_L_B)=f[var];
                F.Vector(INDICES_R_B)=f[var];
            }
            if(want_uct){
                for(int uv=0;uv<NUCT;uv++){
                    UCT.Vector(b*nau+t_id,uv,NidL[_z_],NidL[_y_],NidL[_x_],
                               nidL[_z_],nidL[_y_],nidL[_x_])=uct[uv];
                    UCT.Vector(b*nau+t_id,uv,NidR[_z_],NidR[_y_],NidR[_x_],
                               nidR[_z_],nidR[_y_],nidR[_x_])=uct[uv];
                }
            }
        }
    }, "mhd_riemann_solver_b");
}

void mhd_riemann_solver_b(SD_Solution U, SD_Solution F, int dim,
                          SD_Solution Bn, SD_Solution UCT){
    if(dim==_x_)      mhd_riemann_solver_t_b<_x_,_mvx_,_mvy_,_mvz_,_mbx_,_mby_,_mbz_>(U,F,Bn,UCT);
    else if(dim==_y_) mhd_riemann_solver_t_b<_y_,_mvy_,_mvz_,_mvx_,_mby_,_mbz_,_mbx_>(U,F,Bn,UCT);
    else              mhd_riemann_solver_t_b<_z_,_mvz_,_mvx_,_mvy_,_mbz_,_mbx_,_mby_>(U,F,Bn,UCT);
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
    //This was a HOST round trip under CUDA -- two mirrors and two deep_copies per
    //block per call for a plain elementwise copy, and it arrived that way with the
    //UCT/HLLD merge (0973f7f) with no stated reason while the CPU branch did the
    //identical work as a device kernel. It only runs when rsolver != llf, i.e. in
    //exactly the HLLD lane figure 22 wants. One kernel, both backends.
    SD_Vector Vuf = U_fp.Vector;
    SD_Vector Vbf = B_fp.Vector;
    //Energy consistency, bit 4 of mhd/energy_fix. This state is CONSERVATIVE and
    //the Riemann solver recovers p from it as (g-1)(E - Ekin - B^2/2), so
    //replacing the normal B without moving E hands the solver a pressure that
    //is off by (B_CT^2 - B_rec^2)/2. A code that reconstructed PRIMITIVES would
    //not have this problem -- pressure is carried directly -- which is why the
    //issue is specific to this discretisation rather than to CT.
    bool fix=(cfg.mhd_energy_fix & 4);
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii){
        double bo = Vuf(0,brow,k,j,i,kk,jj,ii);
        double bn = Vbf(0,0,k,j,i,kk,jj,ii);
        Vuf(0,brow,k,j,i,kk,jj,ii) = bn;
        if(fix) Vuf(0,_mprs_,k,j,i,kk,jj,ii) += 0.5*(bn*bn - bo*bo);
    });
}

//Pack-wide form of mhd_face_B_to_fp.
void mhd_face_B_to_fp_b(SD_Solution U_fp, SD_Solution B_fp, int dim){
    int brow = (dim==_x_?_mbx_:(dim==_y_?_mby_:_mbz_));
    int nb=U_fp.nb;
    int Nx=U_fp.Nx, Ny=U_fp.Ny, Nz=U_fp.Nz, px=U_fp.nx, py=U_fp.ny, pz=U_fp.nz;
    int nau=U_fp.n_ader, nab=B_fp.n_ader;
    SD_Vector Vuf = U_fp.Vector;
    SD_Vector Vbf = B_fp.Vector;
    bool fix=(cfg.mhd_energy_fix & 4);   //see mhd_face_B_to_fp
    sd_for_cells_b(nb,Nz,Ny,Nx,pz,py,px,
        KOKKOS_LAMBDA(int b,int k,int j,int i,int kk,int jj,int ii){
        double bo = Vuf(b*nau+0,brow,k,j,i,kk,jj,ii);
        double bn = Vbf(b*nab+0,0,k,j,i,kk,jj,ii);
        Vuf(b*nau+0,brow,k,j,i,kk,jj,ii) = bn;
        if(fix) Vuf(b*nau+0,_mprs_,k,j,i,kk,jj,ii) += 0.5*(bn*bn - bo*bo);
    }, "mhd_face_B_to_fp_b");
}

//Pack-wide form of mhd_compute_fluxes.
template<int V1,int V2,int V3,int B1,int B2,int B3>
void mhd_compute_fluxes_t_b(SD_Solution U, SD_Solution F){
    int nb=U.nb;
    int Nx=U.Nx, Ny=U.Ny, Nz=U.Nz, px=U.nx, py=U.ny, pz=U.nz;
    int nader=U.n_ader;
    double gm=cfg.gamma;
    sd_for_cells_b(nb,Nz,Ny,Nx,pz,py,px,
        KOKKOS_LAMBDA(int b,int k,int j,int i,int kk,int jj,int ii){
        BOFF(nader);
        for(int t_id=0;t_id<nader;t_id++){
            double u[NMHD], w[NMHD], f[NMHD];
            for(int var=0;var<NMHD;var++) u[var]=U.Vector(boff+t_id,var,k,j,i,kk,jj,ii);
            mhd_primitives(u,w,gm);
            mhd_fluxes(w,f,V1,V2,V3,B1,B2,B3,gm);
            for(int var=0;var<NMHD;var++) F.Vector(boff+t_id,var,k,j,i,kk,jj,ii)=f[var];
        }
    }, "mhd_compute_fluxes_b");
}

void mhd_compute_fluxes_b(SD_Solution U, SD_Solution F, int dim){
    if(dim==_x_)      mhd_compute_fluxes_t_b<_mvx_,_mvy_,_mvz_,_mbx_,_mby_,_mbz_>(U,F);
    else if(dim==_y_) mhd_compute_fluxes_t_b<_mvy_,_mvz_,_mvx_,_mby_,_mbz_,_mbx_>(U,F);
    else              mhd_compute_fluxes_t_b<_mvz_,_mvx_,_mvy_,_mbz_,_mbx_,_mby_>(U,F);
}

//----------------------------------------------------------------------------------------
// CFL condition on the fast magnetosonic speed (summed over active dimensions)
//----------------------------------------------------------------------------------------

double mhd_compute_dt(SD_Solution W, double dx, double dy, double dz){
    int Nx=W.Nx, Ny=W.Ny, Nz=W.Nz, px=W.nx, py=W.ny, pz=W.nz;
    double gm=cfg.gamma, cfl=cfg.cfl;
    const bool cfl_min = (cfg.cfl_type == _cfl_min_);
    bool ax=cfg.active[_x_], ay=cfg.active[_y_], az=cfg.active[_z_];
    //Same host round trip as the conversions above, and worse: the reduction ran
    //as a SERIAL host loop over every cell, every step. The #else branch is a
    //Kokkos min-reduction.
    SD_Vector Vw = W.Vector;
    double min_value = sd_min_cells(Nz,Ny,Nx,pz,py,px,
        KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii,double& reduce){
            double rho=Vw(0,_mrho_,k,j,i,kk,jj,ii);
            double p  =Vw(0,_mprs_,k,j,i,kk,jj,ii);
            double Bx =Vw(0,_mbx_,k,j,i,kk,jj,ii);
            double By =Vw(0,_mby_,k,j,i,kk,jj,ii);
            double Bz =Vw(0,_mbz_,k,j,i,kk,jj,ii);
            //Both CFL forms, selected by time/cfl_type; see the enum in define.hpp.
            double c_max=0, dx_min=1, inv_dt=0;
            if(ax){ double a=fabs(Vw(0,_mvx_,k,j,i,kk,jj,ii))+mhd_fast_vel(p,rho,Bx,By,Bz,gm); c_max+=a; dx_min=min(dx_min,dx); inv_dt=max(inv_dt,a/dx); }
            if(ay){ double a=fabs(Vw(0,_mvy_,k,j,i,kk,jj,ii))+mhd_fast_vel(p,rho,By,Bz,Bx,gm); c_max+=a; dx_min=min(dx_min,dy); inv_dt=max(inv_dt,a/dy); }
            if(az){ double a=fabs(Vw(0,_mvz_,k,j,i,kk,jj,ii))+mhd_fast_vel(p,rho,Bz,Bx,By,gm); c_max+=a; dx_min=min(dx_min,dz); inv_dt=max(inv_dt,a/dz); }
            if(c_max > 0){
                double dt_min = cfl_min ? cfl/inv_dt/px : cfl*dx_min/c_max/px;
                reduce = reduce < dt_min ? reduce : dt_min;
            }
        });
    #ifdef MPI
    double g;
    MPI_Allreduce(&min_value,&g,1,MPI_DOUBLE,MPI_MIN,Comm);
    return g;
    #else
    return min_value;
    #endif
}

//Batched dt: the block is an index in the reduction, not a host loop around one
//launch plus one MPI_Allreduce PER BLOCK. Same statement hydro's compute_dt_b
//carries; MHD kept the loop, at 0.864 s against hydro's 0.004 s (216x) on a
//352-leaf mesh over 109 steps. Per-block h comes in by block index, exactly as
//compute_dt_b takes hx/hy/hz.
double mhd_compute_dt_b(SD_Solution W, Vector hx, Vector hy, Vector hz){
    int Nx=W.Nx, Ny=W.Ny, Nz=W.Nz, px=W.nx, py=W.ny, pz=W.nz, nb=W.nb;
    int nader=W.n_ader;
    double gm=cfg.gamma, cfl=cfg.cfl;
    const bool cfl_min = (cfg.cfl_type == _cfl_min_);
    bool ax=cfg.active[_x_], ay=cfg.active[_y_], az=cfg.active[_z_];
    SD_Vector Vw = W.Vector;
    double min_value = sd_min_cells_b(nb,Nz,Ny,Nx,pz,py,px,
        KOKKOS_LAMBDA(int b,int k,int j,int i,int kk,int jj,int ii,double& reduce){
            const int boff = b*nader;
            const double dx=hx(b), dy=hy(b), dz=hz(b);
            double rho=Vw(boff,_mrho_,k,j,i,kk,jj,ii);
            double p  =Vw(boff,_mprs_,k,j,i,kk,jj,ii);
            double Bx =Vw(boff,_mbx_,k,j,i,kk,jj,ii);
            double By =Vw(boff,_mby_,k,j,i,kk,jj,ii);
            double Bz =Vw(boff,_mbz_,k,j,i,kk,jj,ii);
            //Both CFL forms, selected by time/cfl_type; see the enum in define.hpp.
            double c_max=0, dx_min=1, inv_dt=0;
            if(ax){ double a=fabs(Vw(boff,_mvx_,k,j,i,kk,jj,ii))+mhd_fast_vel(p,rho,Bx,By,Bz,gm); c_max+=a; dx_min=min(dx_min,dx); inv_dt=max(inv_dt,a/dx); }
            if(ay){ double a=fabs(Vw(boff,_mvy_,k,j,i,kk,jj,ii))+mhd_fast_vel(p,rho,By,Bz,Bx,gm); c_max+=a; dx_min=min(dx_min,dy); inv_dt=max(inv_dt,a/dy); }
            if(az){ double a=fabs(Vw(boff,_mvz_,k,j,i,kk,jj,ii))+mhd_fast_vel(p,rho,Bz,Bx,By,gm); c_max+=a; dx_min=min(dx_min,dz); inv_dt=max(inv_dt,a/dz); }
            if(c_max > 0){
                double dt_min = cfl_min ? cfl/inv_dt/px : cfl*dx_min/c_max/px;
                reduce = reduce < dt_min ? reduce : dt_min;
            }
        });
    //NO MPI reduction here, deliberately: Mesh::ComputeDt reduces once for both
    //systems. Hydro's compute_dt_b has never reduced internally, and a shared
    //name for the two must not hide an asymmetry in what it means.
    return min_value;
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
//Pack-wide twin of interp_cc_to_edge. `boff` offsets the leading (block x ader)
//axis; the local loop counters are renamed off `a`/`b` because `b` is the block.
KOKKOS_INLINE_FUNCTION
double interp_cc_to_edge_b(const SD_Solution& W, int boff, int pvar,
                           int k,int j,int i,int kk,int jj,int ii,
                           int dim, const Matrix& sp_to_fp, int q){
    double v=0;
    for(int q2=0;q2<q;q2++) for(int q1=0;q1<q;q1++){
        double s;
        if(dim==_z_)      s = W.Vector(boff,pvar,k,j,i,kk,q2,q1)*sp_to_fp(jj,q2)*sp_to_fp(ii,q1);
        else if(dim==_y_) s = W.Vector(boff,pvar,k,j,i,q2,jj,q1)*sp_to_fp(kk,q2)*sp_to_fp(ii,q1);
        else              s = W.Vector(boff,pvar,k,j,i,q2,q1,ii)*sp_to_fp(kk,q2)*sp_to_fp(jj,q1);
        v += s;
    }
    return v;
}

//Pack-wide form of mhd_compute_E.
void mhd_compute_E_b(SD_Solution E, SD_Solution W_sp,
                     SD_Solution B1, SD_Solution B2, SD_Solution Bcc,
                     Matrix sp_to_fp, int dim){
    int nb=E.nb;
    int Nx=E.Nx, Ny=E.Ny, Nz=E.Nz, px=E.nx, py=E.ny, pz=E.nz;
    int q=W_sp.nx;
    int v1_var = mhd_choose(dim,_mvy_,_mvz_,_mvx_);
    int v2_var = mhd_choose(dim,_mvz_,_mvx_,_mvy_);
    int b3_var = mhd_choose(dim,_mbx_,_mby_,_mbz_);
    int qsp = W_sp.nx;
    int nae=E.n_ader, naw=W_sp.n_ader, na1=B1.n_ader, na2=B2.n_ader;
    (void)Bcc;
    sd_for_cells_b(nb,Nz,Ny,Nx,pz,py,px,
        KOKKOS_LAMBDA(int b,int k,int j,int i,int kk,int jj,int ii){
        const int eo=b*nae, wo=b*naw, o1=b*na1, o2=b*na2;
        double b1=0,b2=0;
        for(int ll=0;ll<q;ll++){
            if(dim==_z_){
                b1 += B1.Vector(o1,0,k,j,i,kk,ll,ii)*sp_to_fp(jj,ll);
                b2 += B2.Vector(o2,0,k,j,i,kk,jj,ll)*sp_to_fp(ii,ll);
            } else if(dim==_y_){
                b1 += B1.Vector(o1,0,k,j,i,kk,jj,ll)*sp_to_fp(ii,ll);
                b2 += B2.Vector(o2,0,k,j,i,ll,jj,ii)*sp_to_fp(kk,ll);
            } else {
                b1 += B1.Vector(o1,0,k,j,i,ll,jj,ii)*sp_to_fp(kk,ll);
                b2 += B2.Vector(o2,0,k,j,i,kk,ll,ii)*sp_to_fp(jj,ll);
            }
        }
        double v1  = interp_cc_to_edge_b(W_sp,wo,v1_var,k,j,i,kk,jj,ii,dim,sp_to_fp,qsp);
        double v2  = interp_cc_to_edge_b(W_sp,wo,v2_var,k,j,i,kk,jj,ii,dim,sp_to_fp,qsp);
        double B3  = interp_cc_to_edge_b(W_sp,wo,b3_var,k,j,i,kk,jj,ii,dim,sp_to_fp,qsp);
        double rho = interp_cc_to_edge_b(W_sp,wo,_mrho_,k,j,i,kk,jj,ii,dim,sp_to_fp,qsp);
        double prs = interp_cc_to_edge_b(W_sp,wo,_mprs_,k,j,i,kk,jj,ii,dim,sp_to_fp,qsp);
        E.Vector(eo,0,k,j,i,kk,jj,ii) = v1*b2 - v2*b1;
        E.Vector(eo,1,k,j,i,kk,jj,ii) = b1;
        E.Vector(eo,2,k,j,i,kk,jj,ii) = b2;
        E.Vector(eo,3,k,j,i,kk,jj,ii) = v1;
        E.Vector(eo,4,k,j,i,kk,jj,ii) = v2;
        E.Vector(eo,5,k,j,i,kk,jj,ii) = B3;
        E.Vector(eo,6,k,j,i,kk,jj,ii) = rho;
        E.Vector(eo,7,k,j,i,kk,jj,ii) = prs;
    }, "mhd_compute_E_b");
}

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

//Pack-wide form of mhd_E_riemann_solver.
void mhd_E_riemann_solver_b(SD_Solution E, int dim, int v_index){
    EMF_TRACE("E_riemann_solver_b (SD, 2sweep, batched)");
    int nb=E.nb;
    int Nx=E.Nx-(dim==_x_), Ny=E.Ny-(dim==_y_), Nz=E.Nz-(dim==_z_);
    int px=dim==_x_?1:E.nx, py=dim==_y_?1:E.ny, pz=dim==_z_?1:E.nz;
    int n=mhd_choose(dim,E.nx,E.ny,E.nz);
    int nader=E.n_ader;
    double gm=cfg.gamma;
    int rsolver=cfg.rsolver;
    sd_for_cells_b(nb,Nz,Ny,Nx,pz,py,px,
        KOKKOS_LAMBDA(int b,int k,int j,int i,int kk,int jj,int ii){
        BOFF(nader);
        int NidL[3],nidL[3],NidR[3],nidR[3];
        int l=mhd_choose(dim,i,j,k);
        mhd_indices(NidL,nidL,k,j,i,kk,jj,ii,l  ,n-1,dim);
        mhd_indices(NidR,nidR,k,j,i,kk,jj,ii,l+1,0  ,dim);
        int t_id=0;
        double eL[NEMHD], eR[NEMHD], es[NEMHD];
        for(int var=0;var<NEMHD;var++){ eL[var]=E.Vector(INDICES_L_B); eR[var]=E.Vector(INDICES_R_B); }
        mhd_E_riemann(es,eL,eR,v_index,gm,rsolver);
        for(int var=0;var<NEMHD;var++){ E.Vector(INDICES_L_B)=es[var]; E.Vector(INDICES_R_B)=es[var]; }
    }, "mhd_E_riemann_solver_b");
}

void mhd_E_riemann_solver(SD_Solution E, int dim, int v_index){
    EMF_TRACE("E_riemann_solver (SD, 2sweep)");
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
//
// `aoff` is the leading-axis offset: 0 for a block's own view, b*n_ader for a pack
// view (CLAUDE.md rule 1 -- SD packs fold the block into the leading axis). It is a
// parameter rather than two copies of the function so the per-block and batched
// edge kernels are the SAME arithmetic by construction.
KOKKOS_INLINE_FUNCTION
double sd_interp_uct_face(const SD_Solution& UCT, int aoff, int uvar,
                          int k,int j,int i,int kk,int jj,int ii,
                          int dim1, int dim2, int id1_if, int id2,
                          const Matrix& sp_to_fp, int q){
    if(dim1==_x_){
        if(dim2==_y_){
            double v=0;
            for(int ll=0;ll<q;ll++)
                v += UCT.Vector(aoff,uvar,k,j,i,kk,ll,id1_if)*sp_to_fp(id2,ll);
            return v;
        }
        return UCT.Vector(aoff,uvar,k,j,i,id2,jj,id1_if);
    }
    if(dim1==_y_){
        if(dim2==_x_){
            double v=0;
            for(int ll=0;ll<q;ll++)
                v += UCT.Vector(aoff,uvar,k,j,i,kk,id1_if,ll)*sp_to_fp(id2,ll);
            return v;
        }
        return UCT.Vector(aoff,uvar,k,j,i,kk,id1_if,ii);
    }
    if(dim2==_x_){
        double v=0;
        for(int ll=0;ll<q;ll++)
            v += UCT.Vector(aoff,uvar,k,j,i,id1_if,jj,ll)*sp_to_fp(id2,ll);
        return v;
    }
    double v=0;
    for(int ll=0;ll<q;ll++)
        v += UCT.Vector(aoff,uvar,k,j,i,id1_if,ll,ii)*sp_to_fp(id2,ll);
    return v;
}

KOKKOS_INLINE_FUNCTION
double sd_E_comp(const SD_Solution& E, int aoff, int var,
                 int k,int j,int i,int kk,int jj,int ii){
    return E.Vector(aoff,var,k,j,i,kk,jj,ii);
}


// One edge point of the UCT composition (MDZ21 eq. 33). Shared verbatim by the
// per-block and pack kernels below -- the only difference between them is the
// launch shape and the leading-axis offset `aoff`, so there is no second copy of
// this arithmetic to drift (CLAUDE.md rule 4).
KOKKOS_INLINE_FUNCTION
double mhd_uct_edge_point(const SD_Solution& E, const SD_Solution& UCT1,
                          const SD_Solution& UCT2, const Matrix& sp_to_fp,
                          int aoff, int dim1, int dim2, int n1, int n2,
                          int q1, int q2, int Nx, int Ny, int Nz,
                          int k,int j,int i,int kk,int jj,int ii){
        //An element interface only has two sides if the neighbour element
        //exists. On the OUTERMOST ghost ring it does not, and reading it
        //indexes element -1 / Nx. Treat that point as CONTINUOUS instead --
        //the same branch a non-interface point takes. This has to be a guard
        //rather than a smaller loop: edge_integral ranges over N+1 elements, so
        //the last edge of the active region lives in the first GHOST element's
        //storage (CLAUDE.md rule 6), and skipping ghosts leaves it holding the
        //un-composed value. Measured: that left the assembled E0z non-periodic
        //at 1.6e-03 and the cascade lane drifting 1.6e-08 in mass, with every
        //per-level EMF already clean.
        const int NE1 = (dim1==_x_?Nx:(dim1==_y_?Ny:Nz));
        const int NE2 = (dim2==_x_?Nx:(dim2==_y_?Ny:Nz));
        const int e1  = (dim1==_x_?i:(dim1==_y_?j:k));
        const int e2  = (dim2==_x_?i:(dim2==_y_?j:k));
        int id1 = mhd_choose(dim1,ii,jj,kk);
        int id2 = mhd_choose(dim2,ii,jj,kk);
        bool on_IF1 = (id1==0 || id1==n1-1)
                      && (id1==0 ? e1-1 >= 0 : e1+1 < NE1);
        bool on_IF2 = (id2==0 || id2==n2-1)
                      && (id2==0 ? e2-1 >= 0 : e2+1 < NE2);

        double v1_loc = sd_E_comp(E,aoff,3,k,j,i,kk,jj,ii);
        double v2_loc = sd_E_comp(E,aoff,4,k,j,i,kk,jj,ii);
        double B1_loc = sd_E_comp(E,aoff,1,k,j,i,kk,jj,ii);
        double B2_loc = sd_E_comp(E,aoff,2,k,j,i,kk,jj,ii);

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
            v1W = sd_E_comp(E,aoff,3,kW,jW,iW,kkW,jjW,iiW);
            v1E = sd_E_comp(E,aoff,3,kE,jE,iE,kkE,jjE,iiE);
            B2W = sd_E_comp(E,aoff,2,kW,jW,iW,kkW,jjW,iiW);
            B2E = sd_E_comp(E,aoff,2,kE,jE,iE,kkE,jjE,iiE);
            int id1_if = (id1==0 ? 0 : n1-1);
            aW = mhd_sd_clamp_a(sd_interp_uct_face(UCT1,aoff,0,k,j,i,kk,jj,ii,dim1,dim2,id1_if,id2,sp_to_fp,q1));
            aE = 1.0 - aW;
            dW = mhd_sd_clamp_d(sd_interp_uct_face(UCT1,aoff,1,k,j,i,kk,jj,ii,dim1,dim2,id1_if,id2,sp_to_fp,q1));
            dE = mhd_sd_clamp_d(sd_interp_uct_face(UCT1,aoff,2,k,j,i,kk,jj,ii,dim1,dim2,id1_if,id2,sp_to_fp,q1));
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
            v2S = sd_E_comp(E,aoff,4,kS,jS,iS,kkS,jjS,iiS);
            v2N = sd_E_comp(E,aoff,4,kN,jN,iN,kkN,jjN,iiN);
            B1S = sd_E_comp(E,aoff,1,kS,jS,iS,kkS,jjS,iiS);
            B1N = sd_E_comp(E,aoff,1,kN,jN,iN,kkN,jjN,iiN);
            int id2_if = (id2==0 ? 0 : n2-1);
            aS = mhd_sd_clamp_a(sd_interp_uct_face(UCT2,aoff,0,k,j,i,kk,jj,ii,dim2,dim1,id2_if,id1,sp_to_fp,q2));
            aN = 1.0 - aS;
            dS = mhd_sd_clamp_d(sd_interp_uct_face(UCT2,aoff,1,k,j,i,kk,jj,ii,dim2,dim1,id2_if,id1,sp_to_fp,q2));
            dN = mhd_sd_clamp_d(sd_interp_uct_face(UCT2,aoff,2,k,j,i,kk,jj,ii,dim2,dim1,id2_if,id1,sp_to_fp,q2));
        }else{
            v2S=v2N=v2_loc; B1S=B1N=B1_loc; aS=aN=0.5; dS=dN=0.0;
        }

        return mhd_uct_formula(
            aW,aE,aS,aN, v1W,v1E,v2S,v2N, B2W,B2E,B1S,B1N, dW,dE,dS,dN);
}

// The (dim1, dim2) transverse pair for an edge running along EDIM, and the point
// counts on each. Shared by both launch shapes so they cannot disagree.
#define UCT_EDGE_GEOM(EDIM)                                                   \
    const int dim1 = (EDIM==_z_?_x_:(EDIM==_y_?_z_:_y_));                     \
    const int dim2 = (EDIM==_z_?_y_:(EDIM==_y_?_x_:_z_));                     \
    const int n1 = mhd_choose(dim1,E.nx,E.ny,E.nz);                           \
    const int n2 = mhd_choose(dim2,E.nx,E.ny,E.nz);                           \
    const int q1 = n1 - 1;  /* sp nodes along dim1 on the UCT1 face */        \
    const int q2 = n2 - 1   /* sp nodes along dim2 on the UCT2 face */

//Runs over EVERY element, ghosts included, with the missing-neighbour guard in
//mhd_uct_edge_point handling the outermost ring. The two-sweep edge solver
//writes both sides of every interface for the same reason: edge_integral needs
//the first ghost element's edge to carry a composed value.
template<int EDIM>
void mhd_uct_edge_E_t(SD_Solution E, SD_Solution UCT1, SD_Solution UCT2, Matrix sp_to_fp){
    UCT_EDGE_GEOM(EDIM);
    int Nx=E.Nx, Ny=E.Ny, Nz=E.Nz, px=E.nx, py=E.ny, pz=E.nz;
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii){
        E.Vector(0,0,k,j,i,kk,jj,ii) = mhd_uct_edge_point(
            E,UCT1,UCT2,sp_to_fp,0,dim1,dim2,n1,n2,q1,q2,Nx,Ny,Nz,k,j,i,kk,jj,ii);
    });
}

//Pack-wide twin. The block index is a kernel axis, not a host loop (CLAUDE.md
//rule 1): one launch spans every block. SD packs fold the block into the leading
//axis as b*n_ader, which is what `boff` carries into the shared point kernel.
template<int EDIM>
void mhd_uct_edge_E_t_b(SD_Solution E, SD_Solution UCT1, SD_Solution UCT2, Matrix sp_to_fp){
    UCT_EDGE_GEOM(EDIM);
    int nb=E.nb;
    int Nx=E.Nx, Ny=E.Ny, Nz=E.Nz, px=E.nx, py=E.ny, pz=E.nz;
    int nader=E.n_ader;
    sd_for_cells_b(nb,Nz,Ny,Nx,pz,py,px,
        KOKKOS_LAMBDA(int b,int k,int j,int i,int kk,int jj,int ii){
        BOFF(nader);
        E.Vector(boff,0,k,j,i,kk,jj,ii) = mhd_uct_edge_point(
            E,UCT1,UCT2,sp_to_fp,boff,dim1,dim2,n1,n2,q1,q2,Nx,Ny,Nz,k,j,i,kk,jj,ii);
    }, "mhd_uct_edge_E_b");
}

// Zero the wall-TANGENTIAL electric field on a reflecting boundary.
//
// A perfectly conducting wall has E_t = 0, and under CT that is exactly the
// condition that keeps the NORMAL face field pinned: dB_n/dt is the tangential
// curl of E, so a tangential E that vanishes on the wall leaves B_n at whatever
// the initial condition set -- zero, given the reflective state parity
// (boundary.cpp flips row 5+dim). Without it the wall slowly grows a normal
// field and the "equilibrium" current-sheet test stops being one.
//
// `wall` is the direction normal to the wall; the wall edge points are the
// first/last flux point of the first/last ACTIVE element along it.
//
// KNOWN GAP: this pins the SD edge arrays only. The MOOD cascade assembles its
// EMF on the FV lattice (E0z from E1z/E2z), and those are NOT wall-pinned, so a
// cell demoted while sitting ON a reflecting wall would update the wall's normal
// face field. The equilibrium gate does not catch it -- nothing demotes on a
// stationary solution -- so this is latent rather than measured. It needs the
// same treatment on E0*/E1*/E2* before a wall test that actually shocks at the
// boundary (the paper's KH develops rolls mid-domain, not at y = +-1).
void mhd_zero_wall_emf(SD_Solution E, int wall){
    if(cfg.bc[wall] != _reflective_) return;
    const int Nx=E.Nx, Ny=E.Ny, Nz=E.Nz;
    const int px=E.nx, py=E.ny, pz=E.nz;
    const int NG = (wall==_x_?NGHx:(wall==_y_?NGHy:NGHz));
    const int NE = (wall==_x_?Nx:(wall==_y_?Ny:Nz));
    const int np = (wall==_x_?px:(wall==_y_?py:pz));
    //Low wall: element NG, flux point 0. High wall: element NE-NG-1, point np-1.
    const int eLo=NG, pLo=0, eHi=NE-NG-1, pHi=np-1;
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii){
        const int e = (wall==_x_?i:(wall==_y_?j:k));
        const int q = (wall==_x_?ii:(wall==_y_?jj:kk));
        if((e==eLo && q==pLo) || (e==eHi && q==pHi))
            E.Vector(0,0,k,j,i,kk,jj,ii) = 0.0;
    }, "mhd_zero_wall_emf");
}

void mhd_uct_edge_E(SD_Solution E, SD_Solution UCT1, SD_Solution UCT2,
                    Matrix sp_to_fp, int edim){
    EMF_TRACE("uct_edge_E (SD, UCT)");
    if(edim==_x_)      mhd_uct_edge_E_t<_x_>(E,UCT1,UCT2,sp_to_fp);
    else if(edim==_y_) mhd_uct_edge_E_t<_y_>(E,UCT1,UCT2,sp_to_fp);
    else               mhd_uct_edge_E_t<_z_>(E,UCT1,UCT2,sp_to_fp);
}

void mhd_uct_edge_E_b(SD_Solution E, SD_Solution UCT1, SD_Solution UCT2,
                      Matrix sp_to_fp, int edim){
    EMF_TRACE("uct_edge_E_b (SD, UCT, batched)");
    if(edim==_x_)      mhd_uct_edge_E_t_b<_x_>(E,UCT1,UCT2,sp_to_fp);
    else if(edim==_y_) mhd_uct_edge_E_t_b<_y_>(E,UCT1,UCT2,sp_to_fp);
    else               mhd_uct_edge_E_t_b<_z_>(E,UCT1,UCT2,sp_to_fp);
}

//----------------------------------------------------------------------------------------
// B_to_U: project the (divergence-free) face-staggered field onto the cell-centered B
// rows of the conservative state. For each direction, interpolate the face field
// (fp along that dim) to solution points and write into the matching B row of U.
//----------------------------------------------------------------------------------------

// Interpolate a single-var face field (fp along `dim`) to solution points, writing
// into var-row `brow` of the (8-var) conservative array U.
//Project the staggered face field onto one conserved row of the fp state.
//
//This used to take a HOST round trip under CUDA: pull all of U and all of B to a
//mirror, do the work on the host, push the whole U mirror back -- per block, per
//direction, per call. 01a2913 fixed the correctness half of that (the push had no
//matching pull, so every other variable came back as whatever the fresh mirror
//held, and MHD produced a zero/NaN state from t=0 on CUDA while the CPU suite
//stayed green). The cost half remained, and it was the single largest phase of
//the mixed-level MHD advance: 33.7% of the fenced time on a 352-leaf AMR mesh.
//
//The device kernel writes ONLY row brow, in place, so it cannot disturb the other
//variables -- which is what made the mirror dangerous in the first place. One
//kernel, both backends, no mirrors.
static void project_face_to_row(SD_Solution U, int brow, SD_Solution B, Matrix fp_to_sp, int dim){
    int Nx=U.Nx, Ny=U.Ny, Nz=U.Nz, px=U.nx, py=U.ny, pz=U.nz;
    int q=mhd_choose(dim,B.nx,B.ny,B.nz);
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
}

//Pack-wide form: one launch over every block.
static void project_face_to_row_b(SD_Solution U, int brow, SD_Solution B,
                                  Matrix fp_to_sp, int dim){
    int nb=U.nb;
    int Nx=U.Nx, Ny=U.Ny, Nz=U.Nz, px=U.nx, py=U.ny, pz=U.nz;
    int q=mhd_choose(dim,B.nx,B.ny,B.nz);
    int nau=U.n_ader, nab=B.n_ader;
    SD_Vector Vu = U.Vector;
    SD_Vector Vb = B.Vector;
    sd_for_cells_b(nb,Nz,Ny,Nx,pz,py,px,
        KOKKOS_LAMBDA(int b,int k,int j,int i,int kk,int jj,int ii){
        int nid[3];
        double u=0;
        int id=mhd_choose(dim,ii,jj,kk);
        for(int ll=0;ll<q;ll++){
            mhd_indices_n(nid,kk,jj,ii,ll,dim);
            u += Vb(b*nab+0,0,k,j,i,NODE)*fp_to_sp(id,ll);
        }
        Vu(b*nau+0,brow,k,j,i,kk,jj,ii)=u;
    }, "project_face_to_row_b");
}

// Shift the magnetic term of the total energy by `sign` * B^2/2, using U's OWN
// B rows. Called with +1 before B_to_U overwrites those rows and -1 after, which
// leaves the thermal + kinetic energy exactly invariant across the replacement:
//
//     E <- E - Bc^2/2 (old rows) ... rows replaced ... E <- E + Bf^2/2 (new rows)
//
// i.e. E <- E - (Bc^2 - Bf^2)/2, which is MDZ21 section 6.4's prescription
// verbatim. WHY IT IS NEEDED: E is advanced by the Godunov step using the
// cell-centred B, then mhd_B_to_U REPLACES those rows with the projection of the
// staggered CT field without touching E -- so p = (gm-1)(E - Ekin - B^2/2) is
// then evaluated with a B that E was never built from. At beta ~ 1 the mismatch
// is irrelevant; at the 3D blast's beta ~ 2.5e-4 the thermal energy is 0.06% of
// the magnetic one and the leftover is larger than the entire pressure.
// MEASURED on the 192^3 blast WITHOUT this correction: |Bc^2-Bf^2|/2 has median
// 1.5e-03 in cells that hit the pressure floor against 3.8e-08 in cells that do
// not, and 15% of the domain ends up floored at the RAMSES smallp. The paper
// says as much: "no scheme preserves energy positivity without energy
// correction for this test, not even with a minmod limiter".
//
// Off by default (mhd/energy_fix): switching it on changes every MHD result, so
// it is opt-in and the decks that want it say so.
static void mhd_shift_Emag(SD_Solution U, double sign){
    int Nx=U.Nx, Ny=U.Ny, Nz=U.Nz, px=U.nx, py=U.ny, pz=U.nz;
    SD_Vector Vu = U.Vector;
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii){
        double bx=Vu(0,_mbx_,k,j,i,kk,jj,ii);
        double by=Vu(0,_mby_,k,j,i,kk,jj,ii);
        double bz=Vu(0,_mbz_,k,j,i,kk,jj,ii);
        Vu(0,_mprs_,k,j,i,kk,jj,ii) += sign*0.5*(bx*bx+by*by+bz*bz);
    });
}

//Pack-wide form: one launch over every block (CLAUDE.md rule 1).
static void mhd_shift_Emag_b(SD_Solution U, double sign){
    int nb=U.nb, Nx=U.Nx, Ny=U.Ny, Nz=U.Nz, px=U.nx, py=U.ny, pz=U.nz;
    int nau=U.n_ader;
    SD_Vector Vu = U.Vector;
    sd_for_cells_b(nb,Nz,Ny,Nx,pz,py,px,
        KOKKOS_LAMBDA(int b,int k,int j,int i,int kk,int jj,int ii){
        double bx=Vu(b*nau+0,_mbx_,k,j,i,kk,jj,ii);
        double by=Vu(b*nau+0,_mby_,k,j,i,kk,jj,ii);
        double bz=Vu(b*nau+0,_mbz_,k,j,i,kk,jj,ii);
        Vu(b*nau+0,_mprs_,k,j,i,kk,jj,ii) += sign*0.5*(bx*bx+by*by+bz*bz);
    });
}

void mhd_B_to_U_b(SD_Solution U, SD_Solution Bx, SD_Solution By, SD_Solution Bz,
                  Matrix fp_to_sp, bool cons){
    //`cons` says U really is the CONSERVATIVE state. The energy correction is
    //only meaningful there: the same function is called on the PRIMITIVE W_sp
    //at initialisation, where row _mprs_ holds the PRESSURE, and shifting it
    //corrupts the state rather than repairing it.
    if(cons && (cfg.mhd_energy_fix & 1)) mhd_shift_Emag_b(U,-1.0);
    if(cfg.active[_x_]) project_face_to_row_b(U,_mbx_,Bx,fp_to_sp,_x_);
    if(cfg.active[_y_]) project_face_to_row_b(U,_mby_,By,fp_to_sp,_y_);
    if(cfg.active[_z_]) project_face_to_row_b(U,_mbz_,Bz,fp_to_sp,_z_);
    if(cons && (cfg.mhd_energy_fix & 1)) mhd_shift_Emag_b(U,+1.0);
}

void mhd_B_to_U(SD_Solution U, SD_Solution Bx, SD_Solution By, SD_Solution Bz,
                SD_Solution Tx, SD_Solution Ty, SD_Solution Tz, Matrix fp_to_sp,
                bool cons){
    (void)Tx;(void)Ty;(void)Tz;
    if(cons && (cfg.mhd_energy_fix & 1)) mhd_shift_Emag(U,-1.0);
    if(cfg.active[_x_]) project_face_to_row(U,_mbx_,Bx,fp_to_sp,_x_);
    if(cfg.active[_y_]) project_face_to_row(U,_mby_,By,fp_to_sp,_y_);
    if(cfg.active[_z_]) project_face_to_row(U,_mbz_,Bz,fp_to_sp,_z_);
    if(cons && (cfg.mhd_energy_fix & 1)) mhd_shift_Emag(U,+1.0);
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
    //2 marks a PHYSICAL failure so update_cascade can let it (and only it)
    //reach first order; 1 keeps the historical, indistinguishable flag.
    const double padmark = cfg.mood_pad_first_order ? 2.0 : 1.0;
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
            troubles.Vector(0,k,j,i)=padmark;
        if(rho<mrho || rho>rho_max) troubles.Vector(0,k,j,i)=padmark;
        if(p<mP || p>p_max)         troubles.Vector(0,k,j,i)=padmark;
    });
}

//Batched NAD/PAD over the block axis. Same arithmetic and same loop order as the
//pair above, so the batched cascade is bit-identical to the per-block one; the FV
//pack folds the block into the leading axis as b*n_var.
//
//`gscale` is ONE array for the whole mesh, not per block:
//Mesh::mhd_reduce_nad_gscales combines the partials across blocks (and MPI) and
//pushes the same result into every block (mood_set_gscales), and the non-global
//scales (`relative`, `delta`) do not read it at all.
void mhd_NAD_b(FV_Solution det_new, FV_Solution det_old, FV_Solution troubles,
               double tol, int nvar, const double* gscale){
    int Nx=det_old.Nx, Ny=det_old.Ny, Nz=det_old.Nz, nb=det_old.nb;
    const int dnv=det_old.n_var, tnv=troubles.n_var;
    bool ax=cfg.active[_x_], ay=cfg.active[_y_], az=cfg.active[_z_];
    bool moore=cfg.nad_moore;
    int scale=cfg.mood_nad_scale;
    double atol=cfg.nad_atol, eps0=cfg.nad_eps0;
    double gs[8];
    for(int v=0;v<8;v++) gs[v]=(gscale && v<nvar) ? gscale[v] : 0.0;

    fv_for_cells_ngh_b(nb,Nz,Ny,Nx, KOKKOS_LAMBDA(int b,int k,int j,int i){
        const int dof=b*dnv, tof=b*tnv;
        double trouble=0;
        for(int var=0;var<nvar;var++){
            double u_new=det_new.Vector(dof+var,k,j,i);
            double mx=det_old.Vector(dof+var,k,j,i);
            double mn=mx;
            if(moore){
                for(int dk=-(int)az;dk<=(int)az;dk++)
                for(int dj=-(int)ay;dj<=(int)ay;dj++)
                for(int di=-(int)ax;di<=(int)ax;di++){
                    double u=det_old.Vector(dof+var,k+dk,j+dj,i+di);
                    mx=max(mx,u); mn=min(mn,u);
                }
            } else {
                if(ax){ double l=det_old.Vector(dof+var,k,j,i-1),r=det_old.Vector(dof+var,k,j,i+1); mx=max(mx,max(l,r)); mn=min(mn,min(l,r)); }
                if(ay){ double l=det_old.Vector(dof+var,k,j-1,i),r=det_old.Vector(dof+var,k,j+1,i); mx=max(mx,max(l,r)); mn=min(mn,min(l,r)); }
                if(az){ double l=det_old.Vector(dof+var,k-1,j,i),r=det_old.Vector(dof+var,k+1,j,i); mx=max(mx,max(l,r)); mn=min(mn,min(l,r)); }
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
        troubles.Vector(tof+0,k,j,i)=trouble;
    }, "mhd_NAD_b");
}

void mhd_PAD_b(FV_Solution U, FV_Solution troubles){
    //2 marks a PHYSICAL failure so update_cascade can let it (and only it)
    //reach first order; 1 keeps the historical, indistinguishable flag.
    const double padmark = cfg.mood_pad_first_order ? 2.0 : 1.0;
    int Nx=U.Nx, Ny=U.Ny, Nz=U.Nz, nb=U.nb;
    const int unv=U.n_var, tnv=troubles.n_var;
    double gm=cfg.gamma, mrho=cfg.pad_min_rho, mP=cfg.pad_min_P;
    fv_for_cells_ngh_b(nb,Nz,Ny,Nx, KOKKOS_LAMBDA(int b,int k,int j,int i){
        const int uof=b*unv, tof=b*tnv;
        bool bad=false;
        for(int var=0;var<NMHD;var++){
            if(!isfinite(U.Vector(uof+var,k,j,i))){ bad=true; break; }
        }
        double rho=U.Vector(uof+_mrho_,k,j,i);
        double Ekin=0.5*(U.Vector(uof+_mvx_,k,j,i)*U.Vector(uof+_mvx_,k,j,i)
                        +U.Vector(uof+_mvy_,k,j,i)*U.Vector(uof+_mvy_,k,j,i)
                        +U.Vector(uof+_mvz_,k,j,i)*U.Vector(uof+_mvz_,k,j,i))/rho;
        double Emag=0.5*(U.Vector(uof+_mbx_,k,j,i)*U.Vector(uof+_mbx_,k,j,i)
                        +U.Vector(uof+_mby_,k,j,i)*U.Vector(uof+_mby_,k,j,i)
                        +U.Vector(uof+_mbz_,k,j,i)*U.Vector(uof+_mbz_,k,j,i));
        double p=(U.Vector(uof+_mprs_,k,j,i)-Ekin-Emag)*(gm-1.);
        if(bad || !isfinite(p) || !isfinite(Ekin) || !isfinite(Emag))
            troubles.Vector(tof+0,k,j,i)=padmark;
        if(rho<mrho || rho>rho_max) troubles.Vector(tof+0,k,j,i)=padmark;
        if(p<mP || p>p_max)         troubles.Vector(tof+0,k,j,i)=padmark;
    }, "mhd_PAD_b");
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

// Hancock half-step prediction for ideal MHD in primitive variables. Ported from
// the Python spd reference (spd/finite_volume/muscl.py: compute_prediction_mhd),
// which follows RAMSES trace2d/trace3d (mhd/umuscl.f90). Per sweep direction n
// with velocity v_n and normal field B_n, t being the two transverse components:
//
//   drho/dt -= v_n drho + rho dv_n
//   dp/dt   -= v_n dp   + gamma p dv_n
//   dv_n/dt -= v_n dv_n + (dp + sum_t B_t dB_t)/rho
//   dv_t/dt -= v_n dv_t - B_n dB_t/rho             (magnetic tension)
//   dB_t/dt -= v_n dB_t + B_t dv_n - B_n dv_t      (induction)
//
// The B_n dB_n magnetic-pressure/tension pair cancels in the NORMAL momentum
// equation, B_n has no source from its own sweep (it is constant in that 1D
// subsystem, and under CT it is the single-valued face value anyway), and the
// v(div B) terms are dropped as in RAMSES.
//
// dW holds true GRADIENTS (per unit length), not the half-increments the hydro
// corrector takes. That is deliberate: the FV sub-grid is non-uniform and the
// three directions do not share a cell size, so folding h into the slope would
// let a transverse term be divided by the sweep direction's h.
//
// The transverse loop runs over both other components even when a direction is
// INACTIVE -- in 2D, d(vz)/dx and d(Bz)/dx are real gradients along an active
// axis. Only the outer sweep loop is gated on activity.
KOKKOS_INLINE_FUNCTION
void mhd_corrector(const double* W, double* dWt, const double dW[3][NMHD],
                   bool ay, bool az, double gm){
    for(int var=0; var<NMHD; var++) dWt[var]=0.0;
    const double rho = W[_mrho_];
    const int vel[3] = {_mvx_,_mvy_,_mvz_};
    const int bcomp[3] = {_mbx_,_mby_,_mbz_};
    const bool act[3] = {true, ay, az};
    for(int d=0; d<3; d++){
        if(!act[d]) continue;
        const int vn = vel[d], bn = bcomp[d];
        const double* g = dW[d];
        dWt[_mrho_] -= W[vn]*g[_mrho_] + rho*g[vn];
        dWt[_mprs_] -= W[vn]*g[_mprs_] + gm*W[_mprs_]*g[vn];
        double dptot = g[_mprs_];
        for(int t=0; t<3; t++) if(t!=d) dptot += W[bcomp[t]]*g[bcomp[t]];
        dWt[vn] -= W[vn]*g[vn] + dptot/rho;
        for(int t=0; t<3; t++){
            if(t==d) continue;
            const int vt = vel[t], bt = bcomp[t];
            dWt[vt] -= W[vn]*g[vt] - W[bn]*g[bt]/rho;
            dWt[bt] -= W[vn]*g[bt] + W[bt]*g[vn] - W[bn]*g[vt];
        }
    }
}

// One cell's Hancock-predicted primitive state: W + (dt/2) dW/dt.
// mhd_fv_dslope returns the limited HALF-INCREMENT (0.5*slope*h), so the
// gradient is that times 2/h.
KOKKOS_INLINE_FUNCTION
void mhd_fv_predict(FV_Vector W, int k, int j, int i, double sdt, double gm,
                    Vector x_c, Vector x_f, Vector y_c, Vector y_f,
                    Vector z_c, Vector z_f, bool ay, bool az, int lim,
                    double* w_out){
    double g[3][NMHD];
    for(int d=0; d<3; d++) for(int v=0; v<NMHD; v++) g[d][v]=0.0;
    const double hx = x_f(i+1)-x_f(i);
    const double hy = ay ? (y_f(j+1)-y_f(j)) : 1.0;
    const double hz = az ? (z_f(k+1)-z_f(k)) : 1.0;
    for(int v=0; v<NMHD; v++){
        g[_x_][v] = mhd_fv_dslope(W,v,k,j,i,_x_,x_c,x_f,y_c,y_f,z_c,z_f,lim)*2.0/hx;
        if(ay) g[_y_][v] = mhd_fv_dslope(W,v,k,j,i,_y_,x_c,x_f,y_c,y_f,z_c,z_f,lim)*2.0/hy;
        if(az) g[_z_][v] = mhd_fv_dslope(W,v,k,j,i,_z_,x_c,x_f,y_c,y_f,z_c,z_f,lim)*2.0/hz;
    }
    double w[NMHD], dWt[NMHD];
    for(int v=0; v<NMHD; v++) w[v]=W(v,k,j,i);
    mhd_corrector(w,dWt,g,ay,az,gm);
    for(int v=0; v<NMHD; v++) w_out[v] = w[v] + 0.5*sdt*dWt[v];
}

// Pack-wide twin of mhd_fv_dslope: identical arithmetic, geometry read from the
// per-block coordinate packs and the variable axis offset by the block.
KOKKOS_INLINE_FUNCTION
double mhd_fv_dslope_b(FV_Vector W, int voff, int var, int k, int j, int i, int dim,
                       Matrix x_c, Matrix x_f, Matrix y_c, Matrix y_f,
                       Matrix z_c, Matrix z_f, int b, int lim){
    double w=W(voff+var,k,j,i), wm, wp;
    if(dim==_x_){ wm=W(voff+var,k,j,i-1); wp=W(voff+var,k,j,i+1);
        return limited_slope((wp-w)/(x_c(b,i+1)-x_c(b,i)),(w-wm)/(x_c(b,i)-x_c(b,i-1)),
                             x_c(b,i+1)-x_c(b,i),x_c(b,i)-x_c(b,i-1),
                             x_f(b,i),x_f(b,i+1),lim); }
    if(dim==_y_){ wm=W(voff+var,k,j-1,i); wp=W(voff+var,k,j+1,i);
        return limited_slope((wp-w)/(y_c(b,j+1)-y_c(b,j)),(w-wm)/(y_c(b,j)-y_c(b,j-1)),
                             y_c(b,j+1)-y_c(b,j),y_c(b,j)-y_c(b,j-1),
                             y_f(b,j),y_f(b,j+1),lim); }
    wm=W(voff+var,k-1,j,i); wp=W(voff+var,k+1,j,i);
    return limited_slope((wp-w)/(z_c(b,k+1)-z_c(b,k)),(w-wm)/(z_c(b,k)-z_c(b,k-1)),
                         z_c(b,k+1)-z_c(b,k),z_c(b,k)-z_c(b,k-1),
                         z_f(b,k),z_f(b,k+1),lim);
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
                     bool muscl, Vector w_rk, int ader, double dt){
    const int lim = cfg.limiter;   //device cannot read cfg; capture then thread
    //MUSCL-Hancock: half-step the CELL state before reconstructing, exactly as
    //the Python reference does (predict M, then faces = M_pred +- S with the
    //UNPREDICTED slopes). Only meaningful at the MUSCL level -- donor cell has
    //no slopes to correct with -- and only under job/scheme=vl2.
    const bool pred = muscl && cfg.fv_predictor;
    const bool ay_ = cfg.active[_y_], az_ = cfg.active[_z_];
    // Drive the face loop from the CELL-array extent (as hydro::fallback_fluxes does):
    // fv_for_faces then covers exactly the active faces, and the +-2 cell reconstruction
    // stays inside the (haloed) 2-ghost frame of W.
    int Nx=W.Nx, Ny=W.Ny, Nz=W.Nz;
    double gm=cfg.gamma;
    int rsolver=cfg.rsolver;
    bool want_uct = (mhd_rsolver_has_fan(rsolver) && UCT.n_var>=NUCT);
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
    Vector wv = w_rk;
    fv_for_faces(Nz,Ny,Nx, KOKKOS_LAMBDA(int k,int j,int i){
        int kL=k-(D==_z_), jL=j-(D==_y_), iL=i-(D==_x_);
        double wL[NMHD], wR[NMHD], uL[NMHD], uR[NMHD], f[NMHD], uct[NUCT];
        double cL[NMHD], cR[NMHD];
        if(pred){
            const double sdt = wv(ader)*dt;
            mhd_fv_predict(W.Vector,kL,jL,iL,sdt,gm,x_c,x_f,y_c,y_f,z_c,z_f,ay_,az_,lim,cL);
            mhd_fv_predict(W.Vector,k ,j ,i ,sdt,gm,x_c,x_f,y_c,y_f,z_c,z_f,ay_,az_,lim,cR);
        }
        for(int var=0;var<NMHD;var++){
            double dL = muscl ? mhd_fv_dslope(W.Vector,var,kL,jL,iL,D,x_c,x_f,y_c,y_f,z_c,z_f,lim) : 0.0;
            double dR = muscl ? mhd_fv_dslope(W.Vector,var,k ,j ,i ,D,x_c,x_f,y_c,y_f,z_c,z_f,lim) : 0.0;
            //Base state: the Hancock-predicted cell value under vl2, the plain
            //cell average otherwise. The SLOPES stay unpredicted either way.
            const double bL = pred ? cL[var] : W.Vector(var,kL,jL,iL);
            const double bR = pred ? cR[var] : W.Vector(var,k ,j ,i );
            wL[var]=bL+dL;   // right face of the left cell
            wR[var]=bR-dR;   // left  face of the right cell
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
                   int dim, bool muscl, Vector w_rk, int ader, double dt){
    if(dim==_x_)      mhd_fv_fluxes_t<_x_>(W,F,Bn_f,UCT,x_c,x_f,y_c,y_f,z_c,z_f,muscl,w_rk,ader,dt);
    else if(dim==_y_) mhd_fv_fluxes_t<_y_>(W,F,Bn_f,UCT,x_c,x_f,y_c,y_f,z_c,z_f,muscl,w_rk,ader,dt);
    else              mhd_fv_fluxes_t<_z_>(W,F,Bn_f,UCT,x_c,x_f,y_c,y_f,z_c,z_f,muscl,w_rk,ader,dt);
}

//Pack-wide twin of mhd_fv_predict. Only the gradient GATHER differs (packed
//geometry, block-offset variable axis); the source terms themselves stay in the
//single shared mhd_corrector, so the two paths cannot drift on the physics.
KOKKOS_INLINE_FUNCTION
void mhd_fv_predict_b(FV_Vector W, int wo, int k, int j, int i, double sdt, double gm,
                      Matrix x_c, Matrix x_f, Matrix y_c, Matrix y_f,
                      Matrix z_c, Matrix z_f, int b, bool ay, bool az, int lim,
                      double* w_out){
    double g[3][NMHD];
    for(int d=0; d<3; d++) for(int v=0; v<NMHD; v++) g[d][v]=0.0;
    const double hx = x_f(b,i+1)-x_f(b,i);
    const double hy = ay ? (y_f(b,j+1)-y_f(b,j)) : 1.0;
    const double hz = az ? (z_f(b,k+1)-z_f(b,k)) : 1.0;
    for(int v=0; v<NMHD; v++){
        g[_x_][v] = mhd_fv_dslope_b(W,wo,v,k,j,i,_x_,x_c,x_f,y_c,y_f,z_c,z_f,b,lim)*2.0/hx;
        if(ay) g[_y_][v] = mhd_fv_dslope_b(W,wo,v,k,j,i,_y_,x_c,x_f,y_c,y_f,z_c,z_f,b,lim)*2.0/hy;
        if(az) g[_z_][v] = mhd_fv_dslope_b(W,wo,v,k,j,i,_z_,x_c,x_f,y_c,y_f,z_c,z_f,b,lim)*2.0/hz;
    }
    double w[NMHD], dWt[NMHD];
    for(int v=0; v<NMHD; v++) w[v]=W(wo+v,k,j,i);
    mhd_corrector(w,dWt,g,ay,az,gm);
    for(int v=0; v<NMHD; v++) w_out[v] = w[v] + 0.5*sdt*dWt[v];
}

//Pack-wide twin of mhd_fv_fluxes_t. The face body is the same arithmetic; only the
//variable axis (offset by the block) and the geometry (per-block coordinate packs)
//differ. SPD_NO_MHD_BATCH=1 runs the per-block original, and the two must agree
//bitwise -- that is the check that keeps this copy honest.
template<int D>
void mhd_fv_fluxes_t_b(FV_Solution W, FV_Solution F, FV_Solution Bn_f, FV_Solution UCT,
                       Matrix x_c, Matrix x_f, Matrix y_c, Matrix y_f,
                       Matrix z_c, Matrix z_f, bool muscl, Vector w_rk, int ader, double dt){
    const int lim = cfg.limiter;
    const bool pred = muscl && cfg.fv_predictor;
    const bool ay_ = cfg.active[_y_], az_ = cfg.active[_z_];
    int nb=W.nb, Nx=W.Nx, Ny=W.Ny, Nz=W.Nz;
    double gm=cfg.gamma;
    int rsolver=cfg.rsolver;
    bool want_uct = (mhd_rsolver_has_fan(rsolver) && UCT.n_var>=NUCT);
    const bool take_bn = (rsolver!=_rsolver_llf_);
    const int nvw=W.n_var, nvf=F.n_var, nvb=Bn_f.n_var;
    const int nvu=want_uct?UCT.n_var:1;
    const int v1 = (D==_x_?_mvx_:(D==_y_?_mvy_:_mvz_));
    const int v2 = (D==_x_?_mvy_:(D==_y_?_mvz_:_mvx_));
    const int v3 = (D==_x_?_mvz_:(D==_y_?_mvx_:_mvy_));
    const int b1 = (D==_x_?_mbx_:(D==_y_?_mby_:_mbz_));
    const int b2 = (D==_x_?_mby_:(D==_y_?_mbz_:_mbx_));
    const int b3 = (D==_x_?_mbz_:(D==_y_?_mbx_:_mby_));
    Vector wv = w_rk;
    fv_for_faces_b(nb,Nz,Ny,Nx, KOKKOS_LAMBDA(int b,int k,int j,int i){
        const int wo=b*nvw, fo=b*nvf, bo=b*nvb, uo=b*nvu;
        int kL=k-(D==_z_), jL=j-(D==_y_), iL=i-(D==_x_);
        double wL[NMHD], wR[NMHD], uL[NMHD], uR[NMHD], f[NMHD], uct[NUCT];
        double cL[NMHD], cR[NMHD];
        if(pred){
            const double sdt = wv(ader)*dt;
            mhd_fv_predict_b(W.Vector,wo,kL,jL,iL,sdt,gm,x_c,x_f,y_c,y_f,z_c,z_f,b,ay_,az_,lim,cL);
            mhd_fv_predict_b(W.Vector,wo,k ,j ,i ,sdt,gm,x_c,x_f,y_c,y_f,z_c,z_f,b,ay_,az_,lim,cR);
        }
        for(int var=0;var<NMHD;var++){
            double dL = muscl ? mhd_fv_dslope_b(W.Vector,wo,var,kL,jL,iL,D,
                                                x_c,x_f,y_c,y_f,z_c,z_f,b,lim) : 0.0;
            double dR = muscl ? mhd_fv_dslope_b(W.Vector,wo,var,k ,j ,i ,D,
                                                x_c,x_f,y_c,y_f,z_c,z_f,b,lim) : 0.0;
            const double bL = pred ? cL[var] : W.Vector(wo+var,kL,jL,iL);
            const double bR = pred ? cR[var] : W.Vector(wo+var,k ,j ,i );
            wL[var]=bL+dL;
            wR[var]=bR-dR;
        }
        if(take_bn) wL[b1]=wR[b1]=Bn_f.Vector(bo+0,k,j,i);
        mhd_conservatives(wL,uL,gm);
        mhd_conservatives(wR,uR,gm);
        mhd_riemann(f,uL,uR,v1,v2,v3,b1,b2,b3,gm,rsolver, want_uct?uct:nullptr);
        for(int var=0;var<NMHD;var++) F.Vector(fo+var,k,j,i)=f[var];
        if(want_uct) for(int var=0;var<NUCT;var++) UCT.Vector(uo+var,k,j,i)=uct[var];
    }, "mhd_fv_fluxes_b");
}

void mhd_fv_fluxes_b(FV_Solution W, FV_Solution F, FV_Solution Bn_f, FV_Solution UCT,
                     Matrix x_c, Matrix x_f, Matrix y_c, Matrix y_f,
                     Matrix z_c, Matrix z_f, int dim, bool muscl,
                     Vector w_rk, int ader, double dt){
    if(dim==_x_)      mhd_fv_fluxes_t_b<_x_>(W,F,Bn_f,UCT,x_c,x_f,y_c,y_f,z_c,z_f,muscl,w_rk,ader,dt);
    else if(dim==_y_) mhd_fv_fluxes_t_b<_y_>(W,F,Bn_f,UCT,x_c,x_f,y_c,y_f,z_c,z_f,muscl,w_rk,ader,dt);
    else              mhd_fv_fluxes_t_b<_z_>(W,F,Bn_f,UCT,x_c,x_f,y_c,y_f,z_c,z_f,muscl,w_rk,ader,dt);
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

// Pack-wide twin of mhd_fv_corner_val.
KOKKOS_INLINE_FUNCTION
double mhd_fv_corner_val_b(FV_Solution W, int voff, int var, int ck, int cj, int ci,
                           int dim1, int dim2, double sgn1, double sgn2, bool muscl,
                           Matrix x_c, Matrix x_f, Matrix y_c, Matrix y_f,
                           Matrix z_c, Matrix z_f, int b, int lim){
    double val = W.Vector(voff+var,ck,cj,ci);
    if(muscl){
        val += sgn1*mhd_fv_dslope_b(W.Vector,voff,var,ck,cj,ci,dim1,
                                    x_c,x_f,y_c,y_f,z_c,z_f,b,lim);
        val += sgn2*mhd_fv_dslope_b(W.Vector,voff,var,ck,cj,ci,dim2,
                                    x_c,x_f,y_c,y_f,z_c,z_f,b,lim);
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
    EMF_TRACE("uct_corner_E (FV, UCT)");
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
    EMF_TRACE("four_state_E (FV, LLF bound)");
    if(dim==_x_)      mhd_four_state_E_t<_x_>(E,W,x_c,x_f,y_c,y_f,z_c,z_f,muscl);
    else if(dim==_y_) mhd_four_state_E_t<_y_>(E,W,x_c,x_f,y_c,y_f,z_c,z_f,muscl);
    else              mhd_four_state_E_t<_z_>(E,W,x_c,x_f,y_c,y_f,z_c,z_f,muscl);
}

//Pack-wide twin of mhd_four_state_E_t. Same arithmetic as the per-block version;
//see mhd_fv_fluxes_t_b on why the copy is safe to keep.
template<int D>
void mhd_four_state_E_t_b(FV_Solution E, FV_Solution W,
                          Matrix x_c, Matrix x_f, Matrix y_c, Matrix y_f,
                          Matrix z_c, Matrix z_f, bool muscl){
    const int lim = cfg.limiter;
    int nb=W.nb, Nx=W.Nx, Ny=W.Ny, Nz=W.Nz;
    double gm=cfg.gamma;
    const int nvw=W.n_var, nve=E.n_var;
    const int dim1 = (D==_z_?_x_:(D==_y_?_z_:_y_));
    const int dim2 = (D==_z_?_y_:(D==_y_?_x_:_z_));
    const int v1v = (D==_z_?_mvx_:(D==_y_?_mvz_:_mvy_));
    const int v2v = (D==_z_?_mvy_:(D==_y_?_mvx_:_mvz_));
    const int b1v = (D==_z_?_mbx_:(D==_y_?_mbz_:_mby_));
    const int b2v = (D==_z_?_mby_:(D==_y_?_mbx_:_mbz_));
    const int b3v = (D==_z_?_mbz_:(D==_y_?_mby_:_mbx_));
    fv_for_faces_b(nb,Nz,Ny,Nx, KOKKOS_LAMBDA(int b,int k,int j,int i){
        const int wo=b*nvw, eo=b*nve;
        double Esum=0, dB2_1=0, dB1_2=0, Sp1=0, Sp2=0;
        for(int o1=-1;o1<=0;o1++) for(int o2=-1;o2<=0;o2++){
            int ck=k, cj=j, ci=i;
            if(dim1==_x_) ci+=o1; else if(dim1==_y_) cj+=o1; else ck+=o1;
            if(dim2==_x_) ci+=o2; else if(dim2==_y_) cj+=o2; else ck+=o2;
            double sgn1 = (o1==-1)? 1.0 : -1.0;
            double sgn2 = (o2==-1)? 1.0 : -1.0;
            double rho = mhd_fv_corner_val_b(W,wo,_mrho_,ck,cj,ci,dim1,dim2,sgn1,sgn2,muscl,
                                             x_c,x_f,y_c,y_f,z_c,z_f,b,lim);
            double p   = mhd_fv_corner_val_b(W,wo,_mprs_,ck,cj,ci,dim1,dim2,sgn1,sgn2,muscl,
                                             x_c,x_f,y_c,y_f,z_c,z_f,b,lim);
            if(rho<=rho_min) rho=W.Vector(wo+_mrho_,ck,cj,ci);
            if(p<=p_min)     p  =W.Vector(wo+_mprs_,ck,cj,ci);
            double V1 = mhd_fv_corner_val_b(W,wo,v1v,ck,cj,ci,dim1,dim2,sgn1,sgn2,muscl,
                                            x_c,x_f,y_c,y_f,z_c,z_f,b,lim);
            double V2 = mhd_fv_corner_val_b(W,wo,v2v,ck,cj,ci,dim1,dim2,sgn1,sgn2,muscl,
                                            x_c,x_f,y_c,y_f,z_c,z_f,b,lim);
            double B1 = mhd_fv_corner_val_b(W,wo,b1v,ck,cj,ci,dim1,dim2,sgn1,sgn2,muscl,
                                            x_c,x_f,y_c,y_f,z_c,z_f,b,lim);
            double B2 = mhd_fv_corner_val_b(W,wo,b2v,ck,cj,ci,dim1,dim2,sgn1,sgn2,muscl,
                                            x_c,x_f,y_c,y_f,z_c,z_f,b,lim);
            double B3 = mhd_fv_corner_val_b(W,wo,b3v,ck,cj,ci,dim1,dim2,sgn1,sgn2,muscl,
                                            x_c,x_f,y_c,y_f,z_c,z_f,b,lim);
            Esum += V1*B2 - V2*B1;
            double c = sqrt((gm*p + B1*B1 + B2*B2 + B3*B3)/rho);
            Sp1 = max(Sp1, fabs(V1)+c);
            Sp2 = max(Sp2, fabs(V2)+c);
            double js1 = (o1==0)? 1.0 : -1.0;
            double js2 = (o2==0)? 1.0 : -1.0;
            dB2_1 += 0.5*js1*B2;
            dB1_2 += 0.5*js2*B1;
        }
        E.Vector(eo+0,k,j,i) = 0.25*Esum - 0.5*Sp1*dB2_1 + 0.5*Sp2*dB1_2;
    }, "mhd_four_state_E_b");
}

void mhd_four_state_E_b(FV_Solution E, FV_Solution W,
                        Matrix x_c, Matrix x_f, Matrix y_c, Matrix y_f,
                        Matrix z_c, Matrix z_f, int dim, bool muscl){
    EMF_TRACE("four_state_E_b (FV, LLF bound, batched)");
    if(dim==_x_)      mhd_four_state_E_t_b<_x_>(E,W,x_c,x_f,y_c,y_f,z_c,z_f,muscl);
    else if(dim==_y_) mhd_four_state_E_t_b<_y_>(E,W,x_c,x_f,y_c,y_f,z_c,z_f,muscl);
    else              mhd_four_state_E_t_b<_z_>(E,W,x_c,x_f,y_c,y_f,z_c,z_f,muscl);
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
//
// ENERGY CONSISTENCY (mhd/energy_fix). This is the FV twin of the swap that
// mhd_B_to_U performs on the SD state, and it has the same consequence: the
// energy row was built with the Godunov-updated B and is left untouched while
// the B rows are replaced, so the PAD's gas pressure -- E minus kinetic minus
// magnetic -- is evaluated with a B that E was never formed from. At beta ~ 1
// that is noise; at the 3D blast's beta ~ 2.5e-4 the thermal energy is 0.06% of
// the magnetic one and the residual swamps it. Shifting E by the same amount
// keeps thermal + kinetic invariant across the swap (MDZ21 section 6.4).
//
// The shift uses exactly the rows that are REPLACED: in 2D the Bz row is a
// plain conserved variable the fluid fluxes already advanced, so it is not
// swapped and must not enter the correction. Accumulating old and new over the
// same component set makes that automatic.
void mhd_set_candidate_B(FV_Solution U_new, FV_Solution B_cand){
    int Nx=U_new.Nx, Ny=U_new.Ny, Nz=U_new.Nz;
    bool az=cfg.active[_z_];
    bool fix=(cfg.mhd_energy_fix & 2);
    fv_for_cells(Nz,Ny,Nx, KOKKOS_LAMBDA(int k,int j,int i){
        double eo=0, en=0;
        if(fix){
            double bx=U_new.Vector(_mbx_,k,j,i), by=U_new.Vector(_mby_,k,j,i);
            eo = bx*bx + by*by;
            en = B_cand.Vector(0,k,j,i)*B_cand.Vector(0,k,j,i)
               + B_cand.Vector(1,k,j,i)*B_cand.Vector(1,k,j,i);
            if(az){
                double bz=U_new.Vector(_mbz_,k,j,i);
                eo += bz*bz;
                en += B_cand.Vector(2,k,j,i)*B_cand.Vector(2,k,j,i);
            }
        }
        U_new.Vector(_mbx_,k,j,i)=B_cand.Vector(0,k,j,i);
        U_new.Vector(_mby_,k,j,i)=B_cand.Vector(1,k,j,i);
        if(az) U_new.Vector(_mbz_,k,j,i)=B_cand.Vector(2,k,j,i);
        if(fix) U_new.Vector(_mprs_,k,j,i) += 0.5*(en - eo);
    });
}

// Pack-wide form of mhd_assign_edge_E.
void mhd_assign_edge_E_b(FV_Solution E0, FV_Solution E1, FV_Solution E2,
                         FV_Solution cascade, int dim){
    int Nx=cascade.Nx, Ny=cascade.Ny, Nz=cascade.Nz, nb=cascade.nb;
    const int nve = E0.n_var, cnv = cascade.n_var;
    const int dim1 = (dim==_z_?_x_:(dim==_y_?_z_:_y_));
    const int dim2 = (dim==_z_?_y_:(dim==_y_?_x_:_z_));
    fv_for_faces_b(nb,Nz,Ny,Nx, KOKKOS_LAMBDA(int b,int k,int j,int i){
        const int eo = b*nve, co = b*cnv;
        double c=0;
        for(int o1=-1;o1<=0;o1++) for(int o2=-1;o2<=0;o2++){
            int ck=k, cj=j, ci=i;
            if(dim1==_x_) ci+=o1; else if(dim1==_y_) cj+=o1; else ck+=o1;
            if(dim2==_x_) ci+=o2; else if(dim2==_y_) cj+=o2; else ck+=o2;
            c=max(c,cascade.Vector(co+0,ck,cj,ci));
        }
        if(c>=1)
            E0.Vector(eo+0,k,j,i) = c>=2 ? E2.Vector(eo+0,k,j,i)
                                         : E1.Vector(eo+0,k,j,i);
    }, "mhd_assign_edge_E_b");
}

// Pack-wide form of mhd_floor_cv.
void mhd_floor_cv_b(SD_Solution U_cv, FV_Solution B_cv){
    double gm=cfg.gamma, dfl=cfg.dfloor, pfl=cfg.pfloor;
    if(pfl < 0) pfl = dfl*min_c2/gm;
    int nb=U_cv.nb;
    int Nx=U_cv.Nx, Ny=U_cv.Ny, Nz=U_cv.Nz, px=U_cv.nx, py=U_cv.ny, pz=U_cv.nz;
    int qx=px, qy=py, qz=pz;
    int na=U_cv.n_ader, nvb=B_cv.n_var;
    bool az=cfg.active[_z_];
    GHOST_LOCALS;
    sd_for_active_cells_b(nb,Nz,Ny,Nx,pz,py,px,
        KOKKOS_LAMBDA(int b,int k,int j,int i,int kk,int jj,int ii){
        BOFF(na);
        const int bo = b*nvb;
        double rho = U_cv.Vector(boff+0,_mrho_,k,j,i,kk,jj,ii);
        if(rho < dfl){ rho = dfl; U_cv.Vector(boff+0,_mrho_,k,j,i,kk,jj,ii) = rho; }
        double mx=U_cv.Vector(boff+0,_mvx_,k,j,i,kk,jj,ii);
        double my=U_cv.Vector(boff+0,_mvy_,k,j,i,kk,jj,ii);
        double mz=U_cv.Vector(boff+0,_mvz_,k,j,i,kk,jj,ii);
        double Bx=B_cv.Vector(bo+0,K,J,I), By=B_cv.Vector(bo+1,K,J,I);
        double Bz=az ? B_cv.Vector(bo+2,K,J,I)
                     : U_cv.Vector(boff+0,_mbz_,k,j,i,kk,jj,ii);
        double Ekin = 0.5*(mx*mx+my*my+mz*mz)/rho;
        double Emag = 0.5*(Bx*Bx+By*By+Bz*Bz);
        double p = (U_cv.Vector(boff+0,_mprs_,k,j,i,kk,jj,ii) - Ekin - Emag)*(gm-1.);
        if(p < pfl)
            U_cv.Vector(boff+0,_mprs_,k,j,i,kk,jj,ii) = pfl/(gm-1.) + Ekin + Emag;
    }, "mhd_floor_cv_b");
}

// Pack-wide form of mhd_set_candidate_B.
void mhd_set_candidate_B_b(FV_Solution U_new, FV_Solution B_cand){
    int nb = U_new.nb;
    int Nx=U_new.Nx, Ny=U_new.Ny, Nz=U_new.Nz;
    int nvu = U_new.n_var, nvb = B_cand.n_var;
    bool az=cfg.active[_z_];
    bool fix=(cfg.mhd_energy_fix & 2);  //see mhd_set_candidate_B for what this is for
    fv_for_cells_b(nb,Nz,Ny,Nx, KOKKOS_LAMBDA(int b,int k,int j,int i){
        const int uo = b*nvu, bo = b*nvb;
        double eo=0, en=0;
        if(fix){
            double bx=U_new.Vector(uo+_mbx_,k,j,i), by=U_new.Vector(uo+_mby_,k,j,i);
            eo = bx*bx + by*by;
            en = B_cand.Vector(bo+0,k,j,i)*B_cand.Vector(bo+0,k,j,i)
               + B_cand.Vector(bo+1,k,j,i)*B_cand.Vector(bo+1,k,j,i);
            if(az){
                double bz=U_new.Vector(uo+_mbz_,k,j,i);
                eo += bz*bz;
                en += B_cand.Vector(bo+2,k,j,i)*B_cand.Vector(bo+2,k,j,i);
            }
        }
        U_new.Vector(uo+_mbx_,k,j,i)=B_cand.Vector(bo+0,k,j,i);
        U_new.Vector(uo+_mby_,k,j,i)=B_cand.Vector(bo+1,k,j,i);
        if(az) U_new.Vector(uo+_mbz_,k,j,i)=B_cand.Vector(bo+2,k,j,i);
        if(fix) U_new.Vector(uo+_mprs_,k,j,i) += 0.5*(en - eo);
    }, "mhd_set_candidate_B_b");
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

// Blast wave in a strongly magnetized ambient: Wu & Shu (2018) in 2D and
// Balsara & Spicer (1999) in 3D, the latter being Mignone & Del Zanna 2021
// section 6.4. Uniform rho = d0, v = 0, p = p0 outside a sphere of radius
// `radius` and p = p1 inside, threaded by a uniform field at an angle
// (MDZ21 eq. 61):
//     B = B0 (sin(th) cos(ph), sin(th) sin(ph), cos(th)),   B0 = pp.amp
// The direction is given as a VECTOR (pp.v1, pp.v2, pp.v3), normalised here,
// not as the angles (th, ph). That is deliberate: no double is an exact
// arccos of zero, so cos(0.5*PI) = 6.12e-17 and an "axis-aligned" default
// expressed as th = pi/2 gives Bz = 6.12e-14 instead of 0 -- measured, it moved
// the inputs/balsara decks off their pre-change dumps. A direction vector has
// exact axis alignment, and the default (1,0,0) reproduces the old B = (amp,0,0)
// BIT-IDENTICALLY (checked on blast_smoke_p3_fb: 9/9 dumps).
// MDZ21's th = pi/2, ph = pi/4 is the direction (1,1,0).
//
// The radius picks up z only when the z direction is ACTIVE. In a 2D run z is
// left at 0 by the quadrature loops in mhd_initial_conditions while pp.cz is
// half the box length, so an unguarded 3D radius would silently add cz^2 to
// every cell and move the existing 2D decks.
//
// MDZ21 uses rho=1, p0=0.1, p1=1e3, radius=0.1, B0=100/sqrt(4 pi) and the
// direction (1,1,0) on [-1/2,1/2]^3, giving beta ~ 2.5e-4 in the ambient
// medium -- a genuinely hard test: the paper notes no scheme keeps the pressure
// positive on it without an energy correction.
KOKKOS_INLINE_FUNCTION
double mhd_ic_blast(int var, double x, double y, double z, bool az, ProblemParams pp){
    double xr = x - pp.cx;
    double yr = y - pp.cy;
    double zr = az ? z - pp.cz : 0.0;
    double r = sqrt(xr*xr + yr*yr + zr*zr);
    if(var==_mrho_) return pp.d0;
    if(var==_mprs_) return r < pp.radius ? pp.p1 : pp.p0;
    //B is set from the vector potential (mhd_ic_vector_potential) so that the
    //staggered field is divergence-free by construction; these cell-centred
    //rows only seed W and must agree with it.
    double bn = sqrt(pp.v1*pp.v1 + pp.v2*pp.v2 + pp.v3*pp.v3);
    if(bn == 0.0) bn = 1.0;
    if(var==_mbx_)  return pp.amp*pp.v1/bn;
    if(var==_mby_)  return pp.amp*pp.v2/bn;
    if(var==_mbz_)  return pp.amp*pp.v3/bn;
    return 0.0;
}

// Wu & Shu (2018) Mach-800 magnetized jet ambient (+ nozzle IC at t=0).
// Ambient: rho=d0, p=p0, v=0, B=(0,amp,0). Nozzle (|x-cx|<radius): rho=d1, vy=v2.
//
// pp.sigma is the nozzle EDGE WIDTH. sigma = 0 is the paper's sharp top hat and
// is what this IC always did; sigma > 0 replaces the jump by
//     s(x) = [1 - tanh((|x - cx| - radius)/sigma)] / 2
// with rho = d0 + (d1-d0) s and vy = v2 s.
//
// WHY THE SMOOTH OPTION EXISTS. A top hat is a discontinuity, and the IC is laid
// down by evaluating this function at SOLUTION POINTS -- a degree-p polynomial
// per element. Where an element straddles the jump the polynomial overshoots
// (Gibbs), and at Mach 800 that overshoot is fatal before a single step is
// taken. MEASURED on the pure-hydro jet, reading p and v_y straight out of the
// t = 0 dump:
//
//   nx1   dx      nozzle edge on an element boundary?   p_min      max|v_y|
//   20    0.05    yes                                   1.000      800.0
//   25    0.04    no                                    7.14e-25   805.7
//   50    0.02    no                                    7.14e-25   854.5
//   100   0.01    yes                                   1.000      800.0
//
// i.e. unless dx happens to divide the nozzle half-width, rho undershoots while
// rho*v stays large, the recovered p = (g-1)(E - rho v^2/2) goes NEGATIVE and is
// floored -- at t = 0, with no evolution at all. That is why the failure was
// indifferent to the Riemann solver, the emf, the CFL, the energy fix and even
// to switching the MOOD fallback off: none of them can repair a broken IC.
//
// Smoothing changes the problem slightly and the paper's nozzle is sharp, so
// sigma stays 0 by DEFAULT and the existing inputs/balsara/jet_*.athinput decks
// are bit-identical across this change.
// The nozzle state at transverse position x: rho and vy blend from ambient to
// jet across the nozzle edge. pp.sigma = 0 is the paper's sharp top hat (the
// original expression verbatim -- `d0 + (d1-d0)*1.0` is not bit-identical to
// `d1` for arbitrary values); sigma > 0 smooths it over that width.
KOKKOS_INLINE_FUNCTION
double mhd_jet_nozzle(int var, double x, ProblemParams pp){
    if(var==_mprs_) return pp.p0;
    if(var==_mby_)  return pp.amp;
    if(pp.sigma <= 0.0){
        bool nozzle = fabs(x - pp.cx) < pp.radius;
        if(var==_mrho_) return nozzle ? pp.d1 : pp.d0;
        if(var==_mvy_)  return nozzle ? pp.v2 : 0.0;
        return 0.0;
    }
    double s = 0.5*(1.0 - tanh((fabs(x - pp.cx) - pp.radius)/pp.sigma));
    if(var==_mrho_) return pp.d0 + (pp.d1 - pp.d0)*s;
    if(var==_mvy_)  return pp.v2*s;
    return 0.0;
}

// INITIAL CONDITION: quiescent ambient EVERYWHERE. rho = d0, p = p0, v = 0,
// B = (0, amp, 0). The jet enters only through the lower boundary
// (mhd_ic_jet_inflow), which is the standard setup for this test.
//
// This used to return the NOZZLE state instead -- and mhd_jet_nozzle has no y
// dependence, so the beam spanned the entire domain height. A uniform beam in
// pressure equilibrium with its surroundings is an EXACT steady solution, so the
// run did nothing: measured at 400x600 DoF to the paper's t = 0.002, 2080 steps
// and max|rho - rho(0)| = 1.4e-10, with |B|^2 uniform to 1e-6. It looked like a
// working jet only because it never developed a bow shock, a cocoon, or anything
// else. Nothing referenced the old behaviour -- there is no jet configuration in
// the test suite and no golden.
KOKKOS_INLINE_FUNCTION
double mhd_ic_jet(int var, double x, double y, double z, ProblemParams pp){
    (void)x; (void)y; (void)z;
    if(var==_mrho_) return pp.d0;
    if(var==_mprs_) return pp.p0;
    if(var==_mby_)  return pp.amp;
    return 0.0;
}

// BOUNDARY: the nozzle, imposed on the y-min ghost row every stage.
KOKKOS_INLINE_FUNCTION
double mhd_ic_jet_inflow(int var, double x, double y, ProblemParams pp){
    (void)y;
    return mhd_jet_nozzle(var, x, pp);
}

// Magnetized current sheet, Mignone & Del Zanna 2021 section 6.2.
//
// A Harris sheet, B(y) = B0 tanh(y/a) x^, held in equilibrium by a thermal
// pressure gradient that counteracts the Lorentz force:
//     p(y) = (B0^2/2)(beta + 1) - Bx(y)^2/2
// so p + B^2/2 is uniform. With beta = 2 p_inf / B0^2 = 10 and a = 0.04.
// Domain x in [-1,1], y in [-1/2,1/2], periodic in x, REFLECTING at y = +-1/2.
// spd_K's mesh starts at the origin, so the box is [0,2] x [0,1] and the sheet
// sits at y = cy (pp.cy, defaulting to the box midpoint).
//
// pp.amp = B0, pp.p0 = beta, pp.sigma = a (the sheet half-width),
// pp.d0 = rho, pp.p1 = epsilon (the perturbation amplitude).
//
// WHY THIS TEST: with no physical resistivity the UNPERTURBED sheet is an exact
// stationary solution of ideal MHD, so any evolution is the scheme's own
// numerical resistivity. That makes it both the paper's dissipation measure and
// a gate that cannot be fudged -- see mhd_current_sheet_equilibrium_2d.
// Magnetized Kelvin-Helmholtz, Mignone & Del Zanna 2021 section 6.5. This is a
// DIFFERENT problem from mhd_ic_kelvin_helmholtz above, which is Stone+2020
// figure 22.
//
//   vx = (M/2) tanh(y/a),  a = 0.01,  M = 1 (the sonic Mach number)
//   rho = 1,  p = 1/Gamma  (so c_s = 1 and velocities are in units of it)
//   B = B0 x^,  B0 = vA sqrt(rho)  with vA = 1/2
//   vy = eps M exp(-(y/20a)^2)   seeding the instability
//
// Paper domain x in [0,1], y in [-1,1]; on spd_K's origin-anchored mesh that is
// [0,1] x [0,2] with the shear layer at y = cy. Periodic in x, REFLECTING at
// y = 0 and y = 2. The field is flow-aligned, so magnetic tension stabilises the
// instability and the configuration is only weakly unstable -- which is what
// makes it a sharp test of a scheme's dissipation.
//
// THE SEED IS NOT THE PAPER'S. The paper draws eps per zone from a uniform
// random distribution. A per-QUADRATURE-POINT random value would not survive
// this code's initialisation: Initialize integrates the IC over each cell, so
// white noise inside a cell averages back to ~0 and seeds nothing. Rather than
// fake a cell index from a physical coordinate, this uses a deterministic sum of
// modes with fixed irrational phases -- broadband, so the fastest-growing mode
// that fits the box still emerges and the measured growth rate is comparable,
// but reproducible run to run and resolution to resolution. pp.p1 = eps.
KOKKOS_INLINE_FUNCTION
double mhd_ic_kh_mdz(int var, double x, double y, ProblemParams pp){
    const double M = pp.v1, a = pp.sigma, B0 = pp.amp, eps = pp.p1;
    const double yr = y - pp.cy;
    const double Lx = 2.0*pp.cx;
    if(var==_mrho_) return pp.d0;
    if(var==_mvx_)  return 0.5*M*tanh(yr/a);
    if(var==_mvy_){
        //Broadband seed: modes 1..8 across the box with fixed incommensurate
        //phases, normalised so the peak amplitude is eps*M as in the paper.
        double sum = 0.0, norm = 0.0;
        for(int n=1; n<=8; n++){
            const double ph = 2.0*PI*fmod(n*0.7548776662466927, 1.0); //phi = frac(n/phi_golden)
            sum  += sin(2.0*PI*n*x/Lx + ph)/n;
            norm += 1.0/n;
        }
        const double env = exp(-(yr/(20.0*a))*(yr/(20.0*a)));
        return eps*M*env*sum/norm;
    }
    if(var==_mprs_) return pp.p0;
    if(var==_mbx_)  return B0;   //overwritten by the vector-potential init
    return 0.0;
}

// Magnetized Kelvin-Helmholtz of Rueda-Ramirez, Hindenlang, Chan & Gassner 2022
// (arXiv:2203.06062) section 5.2, who take it from Mignone et al. It is the same
// family as mhd_ic_kh_mdz above but far better posed for a high-order code:
//
//   rho = 1,  p = 1/gamma  (so c_s = 1),  gamma = 5/3
//   v1 = (M/2) tanh(y/y0),        M = 1,  y0 = 1/20
//   v2 = v2_0 sin(2 pi x) exp(-y^2/sigma^2),   v2_0 = 0.01, sigma = 0.1
//   B  = (ca cos theta, 0, ca sin theta),      ca = 0.1, theta = pi/3
//
// Three things make this the better KH test here:
//  - The perturbation is a SINGLE DETERMINISTIC MODE, so there is no random
//    seed to reproduce and no caveat about cell-integrated white noise (see
//    mhd_ic_kh_mdz, which needs one).
//  - y0 = 0.05 is five times the MDZ21 shear width, so it is actually resolved:
//    at 128x256 DoF it spans ~6 cells rather than ~1.3.
//  - The reference runs it at polynomial degree N = 3 and N = 7, which are
//    exactly this code's SDFB4 and SDFB8 lanes.
//
// B has a TOROIDAL component B3 = ca sin theta. In true 2D that row is a plain
// cell-centred conserved variable (B_to_U does not overwrite it), which is what
// makes the Lorentz force act on the z-momentum -- the "three-dimensional
// effects in a pseudo-2D example" the paper notes.
//
// pp: amp = ca, v1 = M, v2 = v2_0, p1 = theta, sigma = y0, radius = perturbation
// width, d0 = rho, p0 = p. Domain x in [0,1], y in [-1,1] maps to spd_K's
// origin-anchored [0,1] x [0,2] with the shear layer at y = pp.cy.
KOKKOS_INLINE_FUNCTION
double mhd_ic_kh_rr22(int var, double x, double y, ProblemParams pp){
    const double M=pp.v1, y0=pp.sigma, ca=pp.amp, th=pp.p1;
    const double v20=pp.v2, sg=pp.radius;
    const double yr = y - pp.cy;
    const double Lx = 2.0*pp.cx;
    if(var==_mrho_) return pp.d0;
    if(var==_mvx_)  return 0.5*M*tanh(yr/y0);
    if(var==_mvy_)  return v20*sin(2.0*PI*x/Lx)*exp(-(yr*yr)/(sg*sg));
    if(var==_mvz_)  return 0.0;
    if(var==_mprs_) return pp.p0;
    if(var==_mbx_)  return ca*cos(th);   //overwritten from the vector potential
    if(var==_mbz_)  return ca*sin(th);   //toroidal; cell-centred in true 2D
    return 0.0;
}

KOKKOS_INLINE_FUNCTION
double mhd_ic_current_sheet(int var, double x, double y, ProblemParams pp){
    const double B0 = pp.amp, beta = pp.p0, a = pp.sigma;
    const double yr = y - pp.cy;
    const double Bx = B0*tanh(yr/a);
    if(var==_mrho_) return pp.d0;
    if(var==_mprs_) return 0.5*B0*B0*(beta + 1.0) - 0.5*Bx*Bx;
    if(var==_mbx_)  return Bx;   //overwritten by the vector-potential init
    (void)x;
    return 0.0;                  //v = 0, By = Bz = 0
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

// Field-loop advection (Gardiner & Stone; MDZ21 section 6.1): a weak magnetic
// loop advected diagonally by a uniform flow. B stays a passive loop and
// div(B)=0 must hold. Quasi-2D on the z-invariant slab.
//
// Parameterised so the paper's 2:1 rectangle is reachable: it uses
// x in [-1,1], y in [-1/2,1/2] with the loop at the origin, which on spd_K's
// origin-anchored mesh is [0,2] x [0,1] with the loop at (1, 0.5) -- i.e. the
// box midpoint, which is what pp.cx/pp.cy default to. The defaults below
// (v=(2,1), rho=p=1, A0=1e-3, R=0.3) reproduce the previous hardcoded values
// exactly, so the four mhd_field_loop_* goldens are untouched.
KOKKOS_INLINE_FUNCTION
double mhd_ic_field_loop(int var, double x, double y, double z, ProblemParams pp){
    (void)x; (void)y; (void)z;
    if(var==_mrho_) return pp.d0;
    if(var==_mvx_)  return pp.v1;
    if(var==_mvy_)  return pp.v2;
    if(var==_mvz_)  return pp.v3;
    if(var==_mprs_) return pp.p0;
    return 0.0;
}

KOKKOS_INLINE_FUNCTION
double mhd_ic_primitive(int problem, int var, double x, double y, double z,
                        bool az, ProblemParams pp){
    if(problem==_ic_field_loop_) return mhd_ic_field_loop(var,x,y,z,pp);
    if(problem==_ic_mhd_vortex_) return mhd_ic_vortex(var,x,y,z,pp);
    if(problem==_ic_mhd_blast_)   return mhd_ic_blast(var,x,y,z,az,pp);
    if(problem==_ic_mhd_jet_)     return mhd_ic_jet(var,x,y,z,pp);
    if(problem==_ic_kelvin_helmholtz_) return mhd_ic_kelvin_helmholtz(var,x,y,pp);
    if(problem==_ic_current_sheet_) return mhd_ic_current_sheet(var,x,y,pp);
    if(problem==_ic_kh_mdz_) return mhd_ic_kh_mdz(var,x,y,pp);
    if(problem==_ic_kh_rr22_) return mhd_ic_kh_rr22(var,x,y,pp);
    return mhd_ic_orszag_tang(var,x,y,z);
}

// Vector potential component (edge-point init for CT).
// Orszag-Tang: only Az != 0,
//   Az = B0/(2pi) cos(2pi y) + B0/(4pi) cos(4pi x),  B0 = 1/sqrt(4pi)
// gives Bx = dAz/dy = -B0 sin(2pi y), By = -dAz/dx = B0 sin(4pi x).
// Field loop: Az = A0 (R - r) inside r < R (loop centred at (0.5,0.5)).
KOKKOS_INLINE_FUNCTION
double mhd_ic_vector_potential(int problem, int dim, double x, double y, double z, ProblemParams pp){
    //Az = A0 (R - r) inside r < R, centred on (pp.cx, pp.cy) -- the box
    //midpoint by default, which is where both the unit-box and the paper's
    //2:1-box versions put the loop.
    if(problem==_ic_field_loop_){
        const double A0=pp.amp, R=pp.radius;
        if(dim==_z_){
            double r = sqrt((x-pp.cx)*(x-pp.cx) + (y-pp.cy)*(y-pp.cy));
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
        //MDZ21 eq. 61 as a vector potential, so the staggered field is
        //divergence-free by construction. For a uniform B = (bx,by,bz),
        //  Ax = 0,  Ay = bz x,  Az = bx y - by x
        //gives Bx = dAz/dy, By = -dAz/dx, Bz = dAy/dx exactly.
        //The default direction (1,0,0) collapses to Az = amp*y, which is the
        //expression this branch had before the direction was a parameter.
        double bn = sqrt(pp.v1*pp.v1 + pp.v2*pp.v2 + pp.v3*pp.v3);
        if(bn == 0.0) bn = 1.0;
        if(dim==_y_) return pp.amp*(pp.v3/bn)*x;
        if(dim==_z_) return pp.amp*((pp.v1/bn)*y - (pp.v2/bn)*x);
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
    //Current sheet. Bx = dAz/dy, so the Harris profile B0 tanh(y/a) comes from
    //Az = B0 a ln(cosh(y/a)). The paper perturbs with
    //     dAz = eps B0 cos(ky y / 2) cos(kx x),   kx = 2pi/Lx, ky = 2pi/Ly
    //and differentiates THAT rather than setting dB directly, which is how the
    //perturbed state stays divergence-free to machine precision (eq. 59 and the
    //sentence after it). pp.p1 = eps.
    //Poloidal part of the RR22 field is uniform Bx = ca cos(theta), so
    //Az = ca cos(theta) y. The toroidal B3 is NOT set from A: in true 2D it is a
    //cell-centred conserved row seeded by the primitive IC above.
    if(problem==_ic_kh_rr22_){
        if(dim==_z_) return pp.amp*cos(pp.p1)*y;
        return 0.0;
    }
    //Uniform flow-aligned field: Bx = dAz/dy, so Az = B0 y.
    if(problem==_ic_kh_mdz_){
        if(dim==_z_) return pp.amp*y;
        return 0.0;
    }
    if(problem==_ic_current_sheet_){
        if(dim!=_z_) return 0.0;
        const double B0=pp.amp, a=pp.sigma, eps=pp.p1;
        const double yr = y - pp.cy;
        const double Lx = 2.0*pp.cx, Ly = 2.0*pp.cy;
        const double kx = 2.0*PI/Lx, ky = 2.0*PI/Ly;
        return B0*a*log(cosh(yr/a)) + eps*B0*cos(0.5*ky*yr)*cos(kx*(x-pp.cx));
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
                        double s = mhd_ic_primitive(problem,var,x,y,z,az,pp)*ws(ll);
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
                        double s = mhd_ic_primitive(problem,var,x,y,z,az,pp)*w_sp(ll);
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

// Fill a boundary's prescribed-inflow state ONCE, at setup. The state is
// time-independent, so precomputing it keeps boundary.cpp free of any problem
// knowledge: the BC branch just copies InflowL where its density row is >= 0 and
// falls back to outflow where it is negative (see define.hpp).
//
// The nozzle occupies |x - cx| < radius of the y-min face; everywhere else on
// that face gets the sentinel, i.e. OUTFLOW. That distinction is the whole point.
// Clamping the entire face to a fixed ambient state -- which is what a naive
// "apply the nozzle profile along the boundary" does, since mhd_jet_nozzle
// returns ambient outside the nozzle -- pins v = 0 there and walls in the cocoon
// backflow instead of letting it drain, which shows up as a visibly wrong jet
// base.
//
// SD form: the y-face array's transverse index (i,ii) sits at solution points, so
// its position is Xdim.sd_centers(i,ii).
void mhd_jet_fill_inflow_sd(Boundaries& BC, Matrix x_centers){
    if(BC.InflowL.size()==0) return;
    int Nx=BC.Nx, Ny=BC.Ny, Nz=BC.Nz, px=BC.nx, py=BC.ny, pz=BC.nz;
    int nader=BC.nader, nvar=BC.nvar;
    ProblemParams pp = cfg.pp;
    double gm = cfg.gamma;
    SD_Vector IN = BC.InflowL;
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii){
        double x = x_centers(i,ii);
        bool inject = fabs(x - pp.cx) < pp.radius + (pp.sigma>0.0 ? 4.0*pp.sigma : 0.0);
        double w[NMHD], u[NMHD];
        if(inject){
            for(int var=0;var<NMHD;var++) w[var]=mhd_jet_nozzle(var,x,pp);
            mhd_conservatives(w,u,gm);
        }
        for(int t=0;t<nader;t++)
        for(int var=0;var<nvar;var++)
            IN(t,var,k,j,i,kk,jj,ii) = inject ? (var<NMHD ? u[var] : 0.0) : -1.0;
    });
}

// FV form: the transverse position of cell i is the midpoint of its faces.
void mhd_jet_fill_inflow_fv(FV_Boundaries& BC, Vector fx){
    if(BC.InflowL.size()==0) return;
    int Nx=BC.Nx, Ny=BC.Ny, Nz=BC.Nz, nvar=BC.nvar;
    ProblemParams pp = cfg.pp;
    double gm = cfg.gamma;
    FV_Vector IN = BC.InflowL;
    fv_for_cells(Nz,Ny,Nx, KOKKOS_LAMBDA(int k,int j,int i){
        double x = 0.5*(fx(i)+fx(i+1));
        bool inject = fabs(x - pp.cx) < pp.radius + (pp.sigma>0.0 ? 4.0*pp.sigma : 0.0);
        double w[NMHD], u[NMHD];
        if(inject){
            for(int var=0;var<NMHD;var++) w[var]=mhd_jet_nozzle(var,x,pp);
            mhd_conservatives(w,u,gm);
        }
        for(int var=0;var<nvar;var++)
            IN(var,k,j,i) = inject ? (var<NMHD ? u[var] : 0.0) : -1.0;
    });
}

// Overwrite the y-min GHOST element with jet/nozzle primitives, then conservatives.
//
// The guard is `j > 0`, i.e. ghost row j = 0 ONLY. It used to be `j > 1`, which
// with NGH = 1 also pinned j = 1 -- the first PHYSICAL element row -- re-imposing
// the nozzle state on live interior cells at every stage. That contradicted this
// function's own comment and it is why the Mach-800 jet had never run: the test
// went non-finite within ~50 steps at every resolution, solver, emf, CFL, with
// the energy correction on and off, and with the MOOD fallback disabled entirely.
// MEASURED with the one-character fix, PLM+RK2 at the paper's 400x600 DoF and its
// full t = 0.002: 2080 steps, zero non-finite values, sharp nozzle.
//
// Clamping a physical row is wrong even when it does not blow up: it holds the
// solution at the inflow state where the flow should be free to respond, so the
// working surface never forms.
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
        if(j > 0) return;
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
        if(j > 0) return;
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
