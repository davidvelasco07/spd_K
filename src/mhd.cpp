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
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii){
        for(int t_id=0;t_id<nader;t_id++){
            double w[NMHD], u[NMHD];
            for(int var=0;var<NMHD;var++) w[var]=W.Vector(t_id,var,k,j,i,kk,jj,ii);
            mhd_conservatives(w,u,gm);
            for(int var=0;var<NMHD;var++) U.Vector(t_id,var,k,j,i,kk,jj,ii)=u[var];
        }
    });
}

void mhd_compute_primitives(SD_Solution U, SD_Solution W){
    int Nx=W.Nx, Ny=W.Ny, Nz=W.Nz, px=W.nx, py=W.ny, pz=W.nz;
    int nader=W.n_ader;
    double gm=cfg.gamma;
    double dfl=cfg.dfloor, pfl=cfg.pfloor;
    //NOTE: no conserved-state write-back here. Pointwise repair at solution
    //points injects non-smooth perturbations into the element polynomial and
    //destabilizes the SD scheme (verified on OT N=128: NaN by t=0.75). The
    //AthenaK-style repair acts on cell averages instead (mhd_floor_cv).
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii){
        for(int t_id=0;t_id<nader;t_id++){
            double u[NMHD], w[NMHD];
            for(int var=0;var<NMHD;var++) u[var]=U.Vector(t_id,var,k,j,i,kk,jj,ii);
            mhd_primitives(u,w,gm,dfl,pfl);
            for(int var=0;var<NMHD;var++) W.Vector(t_id,var,k,j,i,kk,jj,ii)=w[var];
        }
    });
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
// LLF Riemann solver (U carries conservative state at the flux points)
//----------------------------------------------------------------------------------------

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

template<int D,int V1,int V2,int V3,int B1,int B2,int B3>
void mhd_riemann_solver_t(SD_Solution U, SD_Solution F){
    int Nx=U.Nx-(D==_x_), Ny=U.Ny-(D==_y_), Nz=U.Nz-(D==_z_);
    int px=D==_x_?1:U.nx, py=D==_y_?1:U.ny, pz=D==_z_?1:U.nz;
    int n=mhd_choose(D,U.nx,U.ny,U.nz);
    int nader=U.n_ader;
    double gm=cfg.gamma;
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii){
        double uL[NMHD], uR[NMHD], f[NMHD];
        int NidL[3],nidL[3],NidR[3],nidR[3];
        int l=mhd_choose(D,i,j,k);
        mhd_indices(NidL,nidL,k,j,i,kk,jj,ii,l  ,n-1,D);
        mhd_indices(NidR,nidR,k,j,i,kk,jj,ii,l+1,0  ,D);
        for(int t_id=0;t_id<nader;t_id++){
            for(int var=0;var<NMHD;var++){
                uL[var]=U.Vector(INDICES_L);
                uR[var]=U.Vector(INDICES_R);
            }
            mhd_riemann_llf(f,uL,uR,V1,V2,V3,B1,B2,B3,gm);
            for(int var=0;var<NMHD;var++){
                F.Vector(INDICES_L)=f[var];
                F.Vector(INDICES_R)=f[var];
            }
        }
    });
}

void mhd_riemann_solver(SD_Solution U, SD_Solution F, int dim){
    if(dim==_x_)      mhd_riemann_solver_t<_x_,_mvx_,_mvy_,_mvz_,_mbx_,_mby_,_mbz_>(U,F);
    else if(dim==_y_) mhd_riemann_solver_t<_y_,_mvy_,_mvz_,_mvx_,_mby_,_mbz_,_mbx_>(U,F);
    else              mhd_riemann_solver_t<_z_,_mvz_,_mvx_,_mvy_,_mbz_,_mbx_,_mby_>(U,F);
}

//----------------------------------------------------------------------------------------
// CFL condition on the fast magnetosonic speed (summed over active dimensions)
//----------------------------------------------------------------------------------------

double mhd_compute_dt(SD_Solution W, double dx, double dy, double dz){
    int Nx=W.Nx, Ny=W.Ny, Nz=W.Nz, px=W.nx, py=W.ny, pz=W.nz;
    double gm=cfg.gamma, cfl=cfg.cfl;
    bool ax=cfg.active[_x_], ay=cfg.active[_y_], az=cfg.active[_z_];
    double min_value = sd_min_cells(Nz,Ny,Nx,pz,py,px,
        KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii,double& reduce){
            double rho=W.Vector(0,_mrho_,k,j,i,kk,jj,ii);
            double p  =W.Vector(0,_mprs_,k,j,i,kk,jj,ii);
            double Bx =W.Vector(0,_mbx_,k,j,i,kk,jj,ii);
            double By =W.Vector(0,_mby_,k,j,i,kk,jj,ii);
            double Bz =W.Vector(0,_mbz_,k,j,i,kk,jj,ii);
            double c_max=0, dx_min=1;
            if(ax){ c_max += fabs(W.Vector(0,_mvx_,k,j,i,kk,jj,ii)) + mhd_fast_vel(p,rho,Bx,By,Bz,gm); dx_min=min(dx_min,dx); }
            if(ay){ c_max += fabs(W.Vector(0,_mvy_,k,j,i,kk,jj,ii)) + mhd_fast_vel(p,rho,By,Bz,Bx,gm); dx_min=min(dx_min,dy); }
            if(az){ c_max += fabs(W.Vector(0,_mvz_,k,j,i,kk,jj,ii)) + mhd_fast_vel(p,rho,Bz,Bx,By,gm); dx_min=min(dx_min,dz); }
            double dt_min = cfl*dx_min/c_max/px;
            reduce = reduce < dt_min ? reduce : dt_min;
        });
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

// LLF electric-field Riemann solver (edge state, 8-var). v_index is 3 for the
// dim1 sweep (dissipate on B2, index 2) or 4 for the dim2 sweep (dissipate on B1,
// index 1), matching mhd_sd_scheme.llf_E. Writes the resolved E-field (component 0)
// to both sides; the transverse components are set to the dissipated average so a
// subsequent sweep sees a single-valued interface (as in the Python broadcast).
void mhd_E_riemann_solver(SD_Solution E, int dim, int v_index){
    int Nx=E.Nx-(dim==_x_), Ny=E.Ny-(dim==_y_), Nz=E.Nz-(dim==_z_);
    int px=dim==_x_?1:E.nx, py=dim==_y_?1:E.ny, pz=dim==_z_?1:E.nz;
    int n=mhd_choose(dim,E.nx,E.ny,E.nz);
    double gm=cfg.gamma;
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii){
        int NidL[3],nidL[3],NidR[3],nidR[3];
        int l=mhd_choose(dim,i,j,k);
        mhd_indices(NidL,nidL,k,j,i,kk,jj,ii,l  ,n-1,dim);
        mhd_indices(NidR,nidR,k,j,i,kk,jj,ii,l+1,0  ,dim);
        int t_id=0; (void)t_id;
        double eL[NEMHD], eR[NEMHD], es[NEMHD];
        int var;
        for(var=0;var<NEMHD;var++){ eL[var]=E.Vector(INDICES_L); eR[var]=E.Vector(INDICES_R); }
        double cL = mhd_fast_vel(eL[7],eL[6],eL[1],eL[2],eL[5],gm);
        double cR = mhd_fast_vel(eR[7],eR[6],eR[1],eR[2],eR[5],gm);
        double vmax = max(fabs(eL[v_index]),fabs(eR[v_index]));
        double Ss = vmax + max(cL,cR);
        double diss;
        if(v_index==3) diss = -0.5*Ss*(eR[2]-eL[2]);
        else           diss =  0.5*Ss*(eR[1]-eL[1]);
        for(var=0;var<NEMHD;var++) es[var]=0.5*(eR[var]+eL[var]) + diss;
        for(var=0;var<NEMHD;var++){ E.Vector(INDICES_L)=es[var]; E.Vector(INDICES_R)=es[var]; }
    });
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
    int q=mhd_choose(dim,B.nx,B.ny,B.nz); // p+2 flux points along dim
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii){
        int nid[3];
        double u=0;
        int id=mhd_choose(dim,ii,jj,kk);
        for(int ll=0;ll<q;ll++){
            mhd_indices_n(nid,kk,jj,ii,ll,dim);
            u += B.Vector(0,0,k,j,i,NODE)*fp_to_sp(id,ll);
        }
        U.Vector(0,brow,k,j,i,kk,jj,ii)=u;
    });
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

// Extract the three NAD detection variables (rho, gas p, |B|) from a cell-averaged
// conservative MHD state into rows [0,1,2] of `det` (row 3 is the aggregate slot).
void mhd_detection_vars(FV_Solution U, FV_Solution det){
    int Nx=U.Nx, Ny=U.Ny, Nz=U.Nz;
    double gm=cfg.gamma;
    fv_for_cells(Nz,Ny,Nx, KOKKOS_LAMBDA(int k,int j,int i){
        double u[NMHD], w[NMHD];
        for(int var=0;var<NMHD;var++) u[var]=U.Vector(var,k,j,i);
        mhd_primitives(u,w,gm);
        double B2=w[_mbx_]*w[_mbx_]+w[_mby_]*w[_mby_]+w[_mbz_]*w[_mbz_];
        det.Vector(0,k,j,i)=w[_mrho_];
        det.Vector(1,k,j,i)=w[_mprs_];
        det.Vector(2,k,j,i)=sqrt(B2);
    });
}

// Discrete-maximum-principle NAD on the detection variables (rho, p, |B|). A cell is
// flagged (aggregate row 3 set to 1) when the candidate leaves the neighbourhood band
// of the old state for any of the three variables. Mirrors trouble_detection::NAD but
// over the MHD detection variables and writing a single aggregate flag per cell.
void mhd_NAD(FV_Solution det_new, FV_Solution det_old, FV_Solution troubles, double tol){
    int Nx=det_old.Nx, Ny=det_old.Ny, Nz=det_old.Nz;
    bool ax=cfg.active[_x_], ay=cfg.active[_y_], az=cfg.active[_z_];
    bool moore=cfg.nad_moore, delta=cfg.nad_delta;
    fv_for_cells_ngh(Nz,Ny,Nx, KOKKOS_LAMBDA(int k,int j,int i){
        double trouble=0;
        for(int var=0;var<3;var++){
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
            if(delta){ double eps=tol*(mx-mn); mn-=eps; mx+=eps; }
            else { mn-=fabs(mn)*tol; mx+=fabs(mx)*tol; }
            if(u_new>mx || u_new<mn) trouble=1;
        }
        troubles.Vector(0,k,j,i)=trouble;
    });
}

// Physical-admissibility detection: floor/ceiling on density and on the gas pressure
// (total energy minus kinetic and magnetic energy). Flags the aggregate slot. The
// floors are runtime parameters (fallback/min_rho, fallback/min_P): raising them above
// the ctoprim floors makes the detection catch degenerating low-beta cells before the
// primitive floors have to carry them. The raw (unfloored) internal energy is tested,
// so states the ctoprim floor would mask are still flagged.
void mhd_PAD(FV_Solution U, FV_Solution troubles){
    int Nx=U.Nx, Ny=U.Ny, Nz=U.Nz;
    double gm=cfg.gamma, mrho=cfg.pad_min_rho, mP=cfg.pad_min_P;
    fv_for_cells_ngh(Nz,Ny,Nx, KOKKOS_LAMBDA(int k,int j,int i){
        double rho=U.Vector(_mrho_,k,j,i);
        double Ekin=0.5*(U.Vector(_mvx_,k,j,i)*U.Vector(_mvx_,k,j,i)
                        +U.Vector(_mvy_,k,j,i)*U.Vector(_mvy_,k,j,i)
                        +U.Vector(_mvz_,k,j,i)*U.Vector(_mvz_,k,j,i))/rho;
        double Emag=0.5*(U.Vector(_mbx_,k,j,i)*U.Vector(_mbx_,k,j,i)
                        +U.Vector(_mby_,k,j,i)*U.Vector(_mby_,k,j,i)
                        +U.Vector(_mbz_,k,j,i)*U.Vector(_mbz_,k,j,i));
        double p=(U.Vector(_mprs_,k,j,i)-Ekin-Emag)*(gm-1.);
        if(rho<mrho || rho>rho_max) troubles.Vector(0,k,j,i)=1;
        if(p<mP || p>p_max)         troubles.Vector(0,k,j,i)=1;
    });
}

// Full MHD detection: build detection variables from the candidate/old cell-averaged
// conserved states, run the |B| NAD, then the magnetic PAD. `troubles` (row 0) holds
// the per-cell flag consumed by the MOOD cascade (face/edge mask pooling).
void mhd_detect_troubles(FV_Solution U_new, FV_Solution U_old,
                         FV_Solution det_new, FV_Solution det_old,
                         FV_Solution troubles, bool PAD){
    mhd_detection_vars(U_new,det_new);
    mhd_detection_vars(U_old,det_old);
    mhd_NAD(det_new,det_old,troubles,cfg.nad_tolerance);
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
                     Vector x_c, Vector x_f, Vector y_c, Vector y_f, Vector z_c, Vector z_f){
    double w=W(var,k,j,i), wm, wp;
    if(dim==_x_){ wm=W(var,k,j,i-1); wp=W(var,k,j,i+1);
        return minmod((wp-w)/(x_c(i+1)-x_c(i)),(w-wm)/(x_c(i)-x_c(i-1)),x_f(i),x_f(i+1)); }
    if(dim==_y_){ wm=W(var,k,j-1,i); wp=W(var,k,j+1,i);
        return minmod((wp-w)/(y_c(j+1)-y_c(j)),(w-wm)/(y_c(j)-y_c(j-1)),y_f(j),y_f(j+1)); }
    wm=W(var,k-1,j,i); wp=W(var,k+1,j,i);
    return minmod((wp-w)/(z_c(k+1)-z_c(k)),(w-wm)/(z_c(k)-z_c(k-1)),z_f(k),z_f(k+1));
}

// LLF MHD flux at every face of direction `dim` from the ghosted FV primitives W.
// muscl=true reconstructs the two face states with limited half-slopes; muscl=false uses
// the donor-cell (first-order) values. Writes into the FV face-flux array F (8-var).
template<int D>
void mhd_fv_fluxes_t(FV_Solution W, FV_Solution F,
                     Vector x_c, Vector x_f, Vector y_c, Vector y_f, Vector z_c, Vector z_f,
                     bool muscl){
    // Drive the face loop from the CELL-array extent (as hydro::fallback_fluxes does):
    // fv_for_faces then covers exactly the active faces, and the +-2 cell reconstruction
    // stays inside the (haloed) 2-ghost frame of W.
    int Nx=W.Nx, Ny=W.Ny, Nz=W.Nz;
    double gm=cfg.gamma;
    const int v1 = (D==_x_?_mvx_:(D==_y_?_mvy_:_mvz_));
    const int v2 = (D==_x_?_mvy_:(D==_y_?_mvz_:_mvx_));
    const int v3 = (D==_x_?_mvz_:(D==_y_?_mvx_:_mvy_));
    const int b1 = (D==_x_?_mbx_:(D==_y_?_mby_:_mbz_));
    const int b2 = (D==_x_?_mby_:(D==_y_?_mbz_:_mbx_));
    const int b3 = (D==_x_?_mbz_:(D==_y_?_mbx_:_mby_));
    fv_for_faces(Nz,Ny,Nx, KOKKOS_LAMBDA(int k,int j,int i){
        int kL=k-(D==_z_), jL=j-(D==_y_), iL=i-(D==_x_);
        double wL[NMHD], wR[NMHD], uL[NMHD], uR[NMHD], f[NMHD];
        for(int var=0;var<NMHD;var++){
            double dL = muscl ? mhd_fv_dslope(W.Vector,var,kL,jL,iL,D,x_c,x_f,y_c,y_f,z_c,z_f) : 0.0;
            double dR = muscl ? mhd_fv_dslope(W.Vector,var,k ,j ,i ,D,x_c,x_f,y_c,y_f,z_c,z_f) : 0.0;
            wL[var]=W.Vector(var,kL,jL,iL)+dL;   // right face of the left cell
            wR[var]=W.Vector(var,k ,j ,i )-dR;   // left  face of the right cell
        }
        mhd_conservatives(wL,uL,gm);
        mhd_conservatives(wR,uR,gm);
        mhd_riemann_llf(f,uL,uR,v1,v2,v3,b1,b2,b3,gm);
        for(int var=0;var<NMHD;var++) F.Vector(var,k,j,i)=f[var];
    });
}

void mhd_fv_fluxes(FV_Solution W, FV_Solution F,
                   Vector x_c, Vector x_f, Vector y_c, Vector y_f, Vector z_c, Vector z_f,
                   int dim, bool muscl){
    if(dim==_x_)      mhd_fv_fluxes_t<_x_>(W,F,x_c,x_f,y_c,y_f,z_c,z_f,muscl);
    else if(dim==_y_) mhd_fv_fluxes_t<_y_>(W,F,x_c,x_f,y_c,y_f,z_c,z_f,muscl);
    else              mhd_fv_fluxes_t<_z_>(W,F,x_c,x_f,y_c,y_f,z_c,z_f,muscl);
}

//----------------------------------------------------------------------------------------
// Four-state LLF corner electric field (mhd_fv_scheme.four_state_E), single-valued on the
// edge lattice. E-family `dim` has transverse directions (dim1,dim2); the four corner
// states are the (o1,o2) in {-1,0}^2 neighbour cells reconstructed towards the shared
// corner. E = v1*B2 - v2*B1 with v1,B1 along dim1 and v2,B2 along dim2. Writes the FV
// edge array E (1-var). muscl=false gives the first-order (donor-cell) corners.
//----------------------------------------------------------------------------------------
template<int D>
void mhd_four_state_E_t(FV_Solution E, FV_Solution W,
                        Vector x_c, Vector x_f, Vector y_c, Vector y_f, Vector z_c, Vector z_f,
                        bool muscl){
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
            auto g = [&](int var)->double{
                double val=W.Vector(var,ck,cj,ci);
                if(muscl){
                    val += sgn1*mhd_fv_dslope(W.Vector,var,ck,cj,ci,dim1,x_c,x_f,y_c,y_f,z_c,z_f);
                    val += sgn2*mhd_fv_dslope(W.Vector,var,ck,cj,ci,dim2,x_c,x_f,y_c,y_f,z_c,z_f);
                }
                return val;
            };
            double rho=g(_mrho_), p=g(_mprs_);
            if(rho<=rho_min) rho=W.Vector(_mrho_,ck,cj,ci);
            if(p<=p_min)     p  =W.Vector(_mprs_,ck,cj,ci);
            double V1=g(v1v), V2=g(v2v), B1=g(b1v), B2=g(b2v), B3=g(b3v);
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
void mhd_assign_face_flux(FV_Solution F0, FV_Solution F1, FV_Solution F2,
                          FV_Solution cascade, int dim){
    int Nx=cascade.Nx, Ny=cascade.Ny, Nz=cascade.Nz, nvar=F0.n_var;
    fv_for_faces(Nz,Ny,Nx, KOKKOS_LAMBDA(int k,int j,int i){
        int kL=k-(dim==_z_), jL=j-(dim==_y_), iL=i-(dim==_x_);
        double c = max(cascade.Vector(0,k,j,i), cascade.Vector(0,kL,jL,iL));
        if(c>=1){
            for(int var=0;var<nvar;var++)
                F0.Vector(var,k,j,i) = c>=2 ? F2.Vector(var,k,j,i)
                                            : F1.Vector(var,k,j,i);
        }
    });
}

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
// new B (NAD on |B|, PAD pressure with magnetic energy). Only the CT-evolved (active
// direction) components are replaced: in 2D the Bz row is a plain cell-centered
// conserved variable already updated by the fluid fluxes (Python: only sim.dims rows).
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
int mhd_update_cascade(FV_Solution troubles, FV_Solution cascade, int n_cascade){
    int Nx=troubles.Nx, Ny=troubles.Ny, Nz=troubles.Nz;
    double demoted = fv_sum_cells_ngh2(Nz,Ny,Nx,
        KOKKOS_LAMBDA(int k,int j,int i,double& s){
            double tr=troubles.Vector(0,k,j,i);
            double c =cascade.Vector(0,k,j,i);
            if(tr>0 && c<n_cascade){ cascade.Vector(0,k,j,i)=c+1; s+=1; }
        });
    return (int)demoted;
}

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
double mhd_ic_primitive(int problem, int var, double x, double y, double z){
    if(problem==_ic_field_loop_) return mhd_ic_field_loop(var,x,y,z);
    return mhd_ic_orszag_tang(var,x,y,z);
}

// Vector potential component (edge-point init for CT).
// Orszag-Tang: only Az != 0,
//   Az = B0/(2pi) cos(2pi y) + B0/(4pi) cos(4pi x),  B0 = 1/sqrt(4pi)
// gives Bx = dAz/dy = -B0 sin(2pi y), By = -dAz/dx = B0 sin(4pi x).
// Field loop: Az = A0 (R - r) inside r < R (loop centred at (0.5,0.5)).
KOKKOS_INLINE_FUNCTION
double mhd_ic_vector_potential(int problem, int dim, double x, double y, double z){
    if(problem==_ic_field_loop_){
        const double A0=1e-3, R=0.3;
        if(dim==_z_){
            double r = sqrt((x-0.5)*(x-0.5) + (y-0.5)*(y-0.5));
            return r<R ? A0*(R-r) : 0.0;
        }
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
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii){
        for(int var=0;var<NMHD;var++){
            double value=0, x, y=0, z=0;
            for(int nn=0;nn<pz;nn++){
                if(az) z = faces_z(k,kk) + x_sp(nn)*(faces_z(k,kk+1)-faces_z(k,kk));
                for(int mm=0;mm<py;mm++){
                    if(ay) y = faces_y(j,jj) + x_sp(mm)*(faces_y(j,jj+1)-faces_y(j,jj));
                    for(int ll=0;ll<px;ll++){
                        x = faces_x(i,ii) + x_sp(ll)*(faces_x(i,ii+1)-faces_x(i,ii));
                        double s = mhd_ic_primitive(problem,var,x,y,z)*w_sp(ll);
                        if(ay) s*=w_sp(mm);
                        if(az) s*=w_sp(nn);
                        value+=s;
                    }
                }
            }
            W.Vector(0,var,k,j,i,kk,jj,ii)=value;
        }
    });
}

void mhd_Initialize_A(SD_Solution A, Matrix Xs, Matrix Ys, Matrix Zs, int dim){
    int Nx=A.Nx, Ny=A.Ny, Nz=A.Nz, px=A.nx, py=A.ny, pz=A.nz;
    int problem=cfg.problem;
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k,int j,int i,int kk,int jj,int ii){
        double x=Xs(i,ii), y=Ys(j,jj), z=Zs(k,kk);
        A.Vector(0,0,k,j,i,kk,jj,ii)=mhd_ic_vector_potential(problem,dim,x,y,z);
    });
}
