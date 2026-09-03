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

//======================================================================
// Detection, per cell.
//
// Each criterion's body lives in an inline function taking a block offset
// into the leading axis, so the per-block driver (offset 0, its own array)
// and the batched one (offset b*nvar into the pack) run the identical
// arithmetic. Splitting them into two copies of the physics is how the two
// would silently drift apart.
//======================================================================

KOKKOS_INLINE_FUNCTION
void nad_cell(FV_Vector U_new, FV_Vector U, FV_Vector troubles, int off,
              int k, int j, int i, int nvar, int limit_mask, double tolerance,
              bool ax, bool ay, bool az, bool moore, bool delta_mode){
        for(int var=off; var<off+nvar; var++){
        if(!((limit_mask>>(var-off))&1)) continue;
        double maximum;
        double minimum;
        double u_L;
        double u_R;
        double u_new;
        u_new = U_new(var,k,j,i);
        maximum = U(var,k,j,i);
        minimum = U(var,k,j,i);
        if(moore){
            //DMP bounds over the Moore (box) neighborhood, diagonals included
            for(int dk=-(int)az; dk<=(int)az; dk++)
            for(int dj=-(int)ay; dj<=(int)ay; dj++)
            for(int di=-(int)ax; di<=(int)ax; di++){
                double u = U(var,k+dk,j+dj,i+di);
                maximum = max(maximum,u);
                minimum = min(minimum,u);
            }
        }
        else{
            //von Neumann (face) neighborhood
            if(ax){
                u_L = U(var,k,j,i-1);
                u_R = U(var,k,j,i+1);
                maximum = max3(u_L,maximum,u_R);
                minimum = min3(u_L,minimum,u_R);
            }
            if(ay){
                u_L = U(var,k,j-1,i);
                u_R = U(var,k,j+1,i);
                maximum = max3(u_L,maximum,u_R);
                minimum = min3(u_L,minimum,u_R);
            }
            if(az){
                u_L = U(var,k-1,j,i);
                u_R = U(var,k+1,j,i);
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
            troubles(var,k,j,i) = 1;
        else
            troubles(var,k,j,i) = 0;
        }
}

//limit_mask: bit set per variable index included in the NAD/SED checks
//(the reference implementation limits only density and pressure for hydro)
void NAD(FV_Solution U_new, FV_Solution U, FV_Solution troubles, double tolerance, int limit_mask){
    int Nx=U.Nx, Ny=U.Ny, Nz=U.Nz;
    //Every component of a solution array is a physical variable now, and the
    //mask decides which of them are limited.
    int nvar = U.n_var;
    bool ax=cfg.active[_x_], ay=cfg.active[_y_], az=cfg.active[_z_];
    bool moore=cfg.nad_moore, delta_mode=cfg.nad_delta;
    FV_Vector un=U_new.Vector, u=U.Vector, tr=troubles.Vector;
    fv_for_cells_ngh(Nz,Ny,Nx, KOKKOS_LAMBDA(int k, int j, int i){
        nad_cell(un,u,tr,0,k,j,i,nvar,limit_mask,tolerance,ax,ay,az,moore,delta_mode);
    });
}

//Same over a whole pack: one launch for every block.
void NAD_b(FV_Solution U_new, FV_Solution U, FV_Solution troubles, double tolerance, int limit_mask){
    int Nx=U.Nx, Ny=U.Ny, Nz=U.Nz, nb=U.nb;
    int nvar = U.n_var;
    bool ax=cfg.active[_x_], ay=cfg.active[_y_], az=cfg.active[_z_];
    bool moore=cfg.nad_moore, delta_mode=cfg.nad_delta;
    FV_Vector un=U_new.Vector, u=U.Vector, tr=troubles.Vector;
    fv_for_cells_ngh_b(nb,Nz,Ny,Nx, KOKKOS_LAMBDA(int b,int k,int j,int i){
        nad_cell(un,u,tr,b*nvar,k,j,i,nvar,limit_mask,tolerance,ax,ay,az,moore,delta_mode);
    });
}


//The geometry comes in as the four spacings the stencil needs, because the
//per-block driver reads them from a Vector and the batched one from its row of
//a packed Matrix.
KOKKOS_INLINE_FUNCTION
void sed_cell(FV_Vector U, FV_Vector alpha, int off, int k, int j, int i,
              int nvar, int limit_mask, int dim,
              double h, double hp, double hm, double dface){
        for(int var=off; var<off+nvar; var++){
        if(!((limit_mask>>(var-off))&1)) continue;
        double u = U(var,k,j,i);
        double du;
        double dup;
        double dum;
        double d2u;
        double dv;
        //First derivative
        if(dim==0){
            du  = (U(var,k,j,i+1)-U(var,k,j,i-1))/h ;
            dup = (U(var,k,j,i+2)-u             )/hp;
            dum = (u             -U(var,k,j,i-2))/hm;
        }
        else if(dim==1){
            du  = (U(var,k,j+1,i)-U(var,k,j-1,i))/h ;
            dup = (U(var,k,j+2,i)-u             )/hp;
            dum = (u             -U(var,k,j-2,i))/hm;
        }
        else{
            du  = (U(var,k+1,j,i)-U(var,k-1,j,i))/h ;
            dup = (U(var,k+2,j,i)-u             )/hp;
            dum = (u             -U(var,k-2,j,i))/hm;
        }
        //Second derivative
        d2u = (dup-dum)/h;
        dv  = 0.5*d2u*dface;
        alpha(var,k,j,i) = Alpha(dv,dum,du,dup);
        }
}

void smooth_extrema(FV_Solution U, FV_Solution alpha, Vector centers, Vector faces, int dim, int limit_mask){
    int Nx=U.Nx, Ny=U.Ny, Nz=U.Nz;
    int nvar = U.n_var;
    FV_Vector u=U.Vector, al=alpha.Vector;
    fv_for_cells_2ngh(Nz,Ny,Nx, KOKKOS_LAMBDA(int k, int j, int i){
        int l = dim==_x_ ? i : (dim==_y_ ?  j : k);
        sed_cell(u,al,0,k,j,i,nvar,limit_mask,dim,
                 centers(l+1)-centers(l-1),
                 centers(l+2)-centers(l  ),
                 centers(l  )-centers(l-2),
                 faces(l+1)-faces(l));
    });
}

//Same over a whole pack; cmat/fmat carry one row of coordinates per block.
void smooth_extrema_b(FV_Solution U, FV_Solution alpha, Matrix cmat, Matrix fmat,
                      int dim, int limit_mask){
    int Nx=U.Nx, Ny=U.Ny, Nz=U.Nz, nb=U.nb;
    int nvar = U.n_var;
    FV_Vector u=U.Vector, al=alpha.Vector;
    fv_for_cells_2ngh_b(nb,Nz,Ny,Nx, KOKKOS_LAMBDA(int b,int k,int j,int i){
        int l = dim==_x_ ? i : (dim==_y_ ?  j : k);
        sed_cell(u,al,b*nvar,k,j,i,nvar,limit_mask,dim,
                 cmat(b,l+1)-cmat(b,l-1),
                 cmat(b,l+2)-cmat(b,l  ),
                 cmat(b,l  )-cmat(b,l-2),
                 fmat(b,l+1)-fmat(b,l));
    });
}

//use_sed=false skips the smooth-extrema relaxation (flags pass through)
//and only pools the per-variable flags into the cascade level.
//
//`cascade` is the per-cell fallback level: 0 where the high-order candidate
//was accepted, 1 where any limited variable was flagged. It is the only part
//of the detection that anything downstream reads, and the only part that
//needs a halo.
KOKKOS_INLINE_FUNCTION
void relax_cell(FV_Vector troubles, FV_Vector flagged, FV_Vector alpha_x,
                FV_Vector alpha_y, FV_Vector alpha_z, int off, int foff,
                int k, int j, int i, int nvar, int limit_mask, bool use_sed,
                bool ax, bool ay, bool az){
        double trouble;
        for(int var=off; var<off+nvar; var++){
        if(!((limit_mask>>(var-off))&1)) continue;
        double alpha=1;
        if(ax)
            alpha = min3(alpha_x(var,k,j,i-1),alpha_x(var,k,j,i),alpha_x(var,k,j,i+1));
        if(ay)
            alpha = min(alpha,min3(alpha_y(var,k,j-1,i),alpha_y(var,k,j,i),alpha_y(var,k,j+1,i)));
        if(az)
            alpha = min(alpha,min3(alpha_z(var,k-1,j,i),alpha_z(var,k,j,i),alpha_z(var,k+1,j,i)));
        //alpha==1 -> smooth extrema
        trouble = troubles(var,k,j,i);
        if(use_sed)
            troubles(var,k,j,i) = alpha<1 ? trouble : 0;
        }
        trouble=0;
        for(int var=off; var<off+nvar; var++)
            if((limit_mask>>(var-off))&1)
                trouble = max(trouble,troubles(var,k,j,i));
        flagged(foff,k,j,i) = trouble;
}

void relax_NAD(FV_Solution troubles, FV_Solution flagged, FV_Solution alpha_x, FV_Solution alpha_y, FV_Solution alpha_z, int limit_mask, bool use_sed){
    int Nx=troubles.Nx, Ny=troubles.Ny, Nz=troubles.Nz;
    int nvar = troubles.n_var;
    bool ax=cfg.active[_x_], ay=cfg.active[_y_], az=cfg.active[_z_];
    FV_Vector tr=troubles.Vector, fl=flagged.Vector;
    FV_Vector axv=alpha_x.Vector, ayv=alpha_y.Vector, azv=alpha_z.Vector;
    fv_for_cells_ngh(Nz,Ny,Nx, KOKKOS_LAMBDA(int k, int j, int i){
        relax_cell(tr,fl,axv,ayv,azv,0,0,k,j,i,nvar,limit_mask,use_sed,ax,ay,az);
    });
}

//Same over a whole pack. `flagged` is one component per block, so its offset
//is b rather than b*nvar.
void relax_NAD_b(FV_Solution troubles, FV_Solution flagged, FV_Solution alpha_x, FV_Solution alpha_y, FV_Solution alpha_z, int limit_mask, bool use_sed){
    int Nx=troubles.Nx, Ny=troubles.Ny, Nz=troubles.Nz, nb=troubles.nb;
    int nvar = troubles.n_var;
    int fnv = flagged.n_var;
    bool ax=cfg.active[_x_], ay=cfg.active[_y_], az=cfg.active[_z_];
    FV_Vector tr=troubles.Vector, fl=flagged.Vector;
    FV_Vector axv=alpha_x.Vector, ayv=alpha_y.Vector, azv=alpha_z.Vector;
    fv_for_cells_ngh_b(nb,Nz,Ny,Nx, KOKKOS_LAMBDA(int b,int k,int j,int i){
        relax_cell(tr,fl,axv,ayv,azv,b*nvar,b*fnv,k,j,i,nvar,limit_mask,use_sed,ax,ay,az);
    });
}

//Fractional blending weights following the reference implementation:
//a troubled cell contributes 1 to its own theta, 0.75 to face neighbors,
//0.5 to edge-diagonal neighbors and 0.375 to corner neighbors. A final
//pass adds a 0.25 ring around every positive theta. The blended flux at a
//face is then theta_face*F_MUSCL + (1-theta_face)*F_SD with
//theta_face = max of the two adjacent cells (see fallback compute_fluxes).
KOKKOS_INLINE_FUNCTION
void blend_cell(FV_Vector flagged, FV_Vector theta, int foff, int toff,
                int k, int j, int i, bool ax, bool ay, bool az){
        const double w[4] = {1.0, 0.75, 0.5, 0.375};
        double th = 0;
        for(int dk=-(int)az; dk<=(int)az; dk++)
        for(int dj=-(int)ay; dj<=(int)ay; dj++)
        for(int di=-(int)ax; di<=(int)ax; di++){
            int n0 = (di!=0) + (dj!=0) + (dk!=0);
            th = max(th, w[n0]*flagged(foff,k+dk,j+dj,i+di));
        }
        theta(toff,k,j,i) = th;
}

void apply_blending(FV_Solution flagged, FV_Solution theta){
    int Nx=flagged.Nx, Ny=flagged.Ny, Nz=flagged.Nz;
    bool ax=cfg.active[_x_], ay=cfg.active[_y_], az=cfg.active[_z_];
    FV_Vector fl=flagged.Vector, th=theta.Vector;
    fv_for_cells_ngh(Nz,Ny,Nx, KOKKOS_LAMBDA(int k, int j, int i){
        blend_cell(fl,th,0,0,k,j,i,ax,ay,az);
    });
}

void apply_blending_b(FV_Solution flagged, FV_Solution theta){
    int Nx=flagged.Nx, Ny=flagged.Ny, Nz=flagged.Nz, nb=flagged.nb;
    int fnv=flagged.n_var, tnv=theta.n_var;
    bool ax=cfg.active[_x_], ay=cfg.active[_y_], az=cfg.active[_z_];
    FV_Vector fl=flagged.Vector, th=theta.Vector;
    fv_for_cells_ngh_b(nb,Nz,Ny,Nx, KOKKOS_LAMBDA(int b,int k,int j,int i){
        blend_cell(fl,th,b*fnv,b*tnv,k,j,i,ax,ay,az);
    });
}

//theta = raw pooled flag (used when blending is disabled)
void theta_from_flagged(FV_Solution flagged, FV_Solution theta){
    int Nx=flagged.Nx, Ny=flagged.Ny, Nz=flagged.Nz;
    FV_Vector fl=flagged.Vector, th=theta.Vector;
    fv_for_cells(Nz,Ny,Nx, KOKKOS_LAMBDA(int k, int j, int i){
        th(0,k,j,i) = fl(0,k,j,i);
    });
}

void theta_from_flagged_b(FV_Solution flagged, FV_Solution theta){
    int Nx=flagged.Nx, Ny=flagged.Ny, Nz=flagged.Nz, nb=flagged.nb;
    int fnv=flagged.n_var, tnv=theta.n_var;
    FV_Vector fl=flagged.Vector, th=theta.Vector;
    fv_for_cells_b(nb,Nz,Ny,Nx, KOKKOS_LAMBDA(int b,int k,int j,int i){
        th(b*tnv,k,j,i) = fl(b*fnv,k,j,i);
    });
}

//======================================================================
// MOOD cascade
//
// The alternative to blending: instead of mixing the high-order and
// fallback fluxes by a fractional weight, every cell carries an integer
// level and each face takes the flux of the more demoted of the two cells
// it separates. A cell's level only ever rises, so the assembly can
// overwrite the level-0 array in place and a face's value is single-valued
// from both sides -- which is what keeps the update conservative (and, for
// MHD, the CT curl divergence-free).
//
// These two are system-neutral: they read the component count from the
// array and know nothing about which variables it holds. Hydro and MHD
// share them; MHD's edge-E assembly stays in mhd.cpp because the edge
// lattice is specific to constrained transport.
//======================================================================

//Demote every flagged cell by one level, up to n_cascade. Returns the number
//demoted, so the caller can stop revising once a sweep changes nothing.
// A cell already at the BOTTOM tier has nothing left to demote to. spd_K has
// always declined to increment it (c < n_cascade), but it still left the trouble
// FLAG set, so a saturated cell kept advertising itself as troubled to everything
// downstream that reads `flagged`. AthenaK drops such a cell out of detection
// entirely (hydro_mood.cpp / mhd_mood.cpp: `if (lv >= n_fb) { fofc_ = false;
// return; }`). cfg.mood_tier_exclude reproduces that: clear the flag once the
// cell is pinned at the floor of the cascade.
//
// MEASURED: in spd_K this is a NO-OP. Current sheet, UCT-HLLD, tol 1e-5,
// 128x64 DoF, with and without it -- E_B(30) 0.9527 both ways at p=3 and 0.4651
// both ways at p=7, and the level-1/level-2 fractions agree to every digit.
// The reason is structural: downstream assembly keys off the CASCADE LEVEL
// (mhd_assign_face_flux / mhd_assign_edge_E pool `cascade`), not off the trouble
// flag, so clearing the flag changes nothing. In AthenaK the same guard DOES
// matter because its `fofc_` flag is what drives the face revision.
// Kept, switchable, because it makes the two codes structurally comparable and
// because it would start mattering the moment `flagged` gained a consumer.
int update_cascade(FV_Solution flagged, FV_Solution cascade, int n_cascade){
    int Nx=flagged.Nx, Ny=flagged.Ny, Nz=flagged.Nz;
    FV_Vector fl=flagged.Vector, ca=cascade.Vector;
    bool excl=cfg.mood_tier_exclude;
    bool pad1st=cfg.mood_pad_first_order;
    double demoted = fv_sum_cells_ngh2(Nz,Ny,Nx,
        KOKKOS_LAMBDA(int k,int j,int i,double& s){
            double tr=fl(0,k,j,i);
            double c =ca(0,k,j,i);
            //A NAD flag (1) may only reach MUSCL; first order needs a PAD
            //failure (2). Without pad_1st the cap is n_cascade for both.
            int cap = (pad1st && tr<2 && n_cascade>1) ? 1 : n_cascade;
            if(tr>0 && c<cap){ ca(0,k,j,i)=c+1; s+=1; }
            else if(excl && c>=cap) fl(0,k,j,i)=0;
        });
    return (int)demoted;
}

//Same over a whole pack, returning the total demoted across every block, so
//the revision loop still stops on one global count.
int update_cascade_b(FV_Solution flagged, FV_Solution cascade, int n_cascade){
    int Nx=flagged.Nx, Ny=flagged.Ny, Nz=flagged.Nz, nb=flagged.nb;
    int fnv=flagged.n_var, cnv=cascade.n_var;
    FV_Vector fl=flagged.Vector, ca=cascade.Vector;
    bool excl=cfg.mood_tier_exclude;
    bool pad1st=cfg.mood_pad_first_order;
    double demoted = fv_sum_cells_ngh2_b(nb,Nz,Ny,Nx,
        KOKKOS_LAMBDA(int b,int k,int j,int i,double& s){
            double tr=fl(b*fnv,k,j,i);
            double c =ca(b*cnv,k,j,i);
            int cap = (pad1st && tr<2 && n_cascade>1) ? 1 : n_cascade;
            if(tr>0 && c<cap){ ca(b*cnv,k,j,i)=c+1; s+=1; }
            else if(excl && c>=cap) fl(b*fnv,k,j,i)=0;   //see update_cascade
        });
    return (int)demoted;
}

//Pool the level over the two cells adjacent to each face and take that
//level's flux, assembled in place into F0.
KOKKOS_INLINE_FUNCTION
void assign_face_cell(FV_Vector F0, FV_Vector F1, FV_Vector F2, FV_Vector cascade,
                      int off, int coff, int k, int j, int i, int nvar, int dim){
        int kL=k-(dim==_z_), jL=j-(dim==_y_), iL=i-(dim==_x_);
        double c = max(cascade(coff,k,j,i), cascade(coff,kL,jL,iL));
        if(c>=1){
            for(int var=off;var<off+nvar;var++)
                F0(var,k,j,i) = c>=2 ? F2(var,k,j,i) : F1(var,k,j,i);
        }
}

void assign_face_flux(FV_Solution F0, FV_Solution F1, FV_Solution F2,
                      FV_Solution cascade, int dim){
    int Nx=cascade.Nx, Ny=cascade.Ny, Nz=cascade.Nz, nvar=F0.n_var;
    FV_Vector f0=F0.Vector, f1=F1.Vector, f2=F2.Vector, ca=cascade.Vector;
    fv_for_faces(Nz,Ny,Nx, KOKKOS_LAMBDA(int k,int j,int i){
        assign_face_cell(f0,f1,f2,ca,0,0,k,j,i,nvar,dim);
    });
}

//Same over a whole pack. This one sits inside the revision loop, so it is the
//per-block launch that repeats most: three directions times max_revs times the
//block count, every stage.
void assign_face_flux_b(FV_Solution F0, FV_Solution F1, FV_Solution F2,
                        FV_Solution cascade, int dim){
    int Nx=cascade.Nx, Ny=cascade.Ny, Nz=cascade.Nz, nb=cascade.nb;
    int nvar=F0.n_var, cnv=cascade.n_var;
    FV_Vector f0=F0.Vector, f1=F1.Vector, f2=F2.Vector, ca=cascade.Vector;
    fv_for_faces_b(nb,Nz,Ny,Nx, KOKKOS_LAMBDA(int b,int k,int j,int i){
        assign_face_cell(f0,f1,f2,ca,b*nvar,b*cnv,k,j,i,nvar,dim);
    });
}

//Adds the 0.25 ring: theta_out = max(theta_in, 0.25*(any box neighbor of
//theta_in positive)). Reads only theta_in, so the result matches the
//reference's single-dilation semantics and is order-independent.
KOKKOS_INLINE_FUNCTION
void ring_cell(FV_Vector ti, FV_Vector to, int off, int k, int j, int i,
               bool ax, bool ay, bool az){
        double th = ti(off,k,j,i);
        double ring = 0;
        for(int dk=-(int)az; dk<=(int)az; dk++)
        for(int dj=-(int)ay; dj<=(int)ay; dj++)
        for(int di=-(int)ax; di<=(int)ax; di++)
            if(ti(off,k+dk,j+dj,i+di) > 0) ring = 0.25;
        to(off,k,j,i) = max(th, ring);
}

void blending_ring(FV_Solution theta_in, FV_Solution theta_out){
    int Nx=theta_in.Nx, Ny=theta_in.Ny, Nz=theta_in.Nz;
    bool ax=cfg.active[_x_], ay=cfg.active[_y_], az=cfg.active[_z_];
    FV_Vector ti=theta_in.Vector, to=theta_out.Vector;
    fv_for_cells_ngh(Nz,Ny,Nx, KOKKOS_LAMBDA(int k, int j, int i){
        ring_cell(ti,to,0,k,j,i,ax,ay,az);
    });
}

void blending_ring_b(FV_Solution theta_in, FV_Solution theta_out){
    int Nx=theta_in.Nx, Ny=theta_in.Ny, Nz=theta_in.Nz, nb=theta_in.nb;
    int nv=theta_in.n_var;
    bool ax=cfg.active[_x_], ay=cfg.active[_y_], az=cfg.active[_z_];
    FV_Vector ti=theta_in.Vector, to=theta_out.Vector;
    fv_for_cells_ngh_b(nb,Nz,Ny,Nx, KOKKOS_LAMBDA(int b,int k,int j,int i){
        ring_cell(ti,to,b*nv,k,j,i,ax,ay,az);
    });
}

KOKKOS_INLINE_FUNCTION
void pad_cell(FV_Vector W, FV_Vector flagged, int off, int foff,
              int k, int j, int i){
        const double density  = W(off+_d_,k,j,i);
        const double pressure = W(off+_p_,k,j,i);
        if(density<rho_min || density>rho_max)  flagged(foff,k,j,i) = 1;
        if(pressure<p_min  || pressure>p_max)   flagged(foff,k,j,i) = 1;
}

void PAD_criteria(FV_Solution W, FV_Solution flagged){
    int Nx=W.Nx, Ny=W.Ny, Nz=W.Nz;
    FV_Vector w=W.Vector, fl=flagged.Vector;
    fv_for_cells_ngh(Nz,Ny,Nx, KOKKOS_LAMBDA(int k, int j, int i){
        pad_cell(w,fl,0,0,k,j,i);
    });
}

void PAD_criteria_b(FV_Solution W, FV_Solution flagged){
    int Nx=W.Nx, Ny=W.Ny, Nz=W.Nz, nb=W.nb;
    int nvar=W.n_var, fnv=flagged.n_var;
    FV_Vector w=W.Vector, fl=flagged.Vector;
    fv_for_cells_ngh_b(nb,Nz,Ny,Nx, KOKKOS_LAMBDA(int b,int k,int j,int i){
        pad_cell(w,fl,b*nvar,b*fnv,k,j,i);
    });
}

void detect_troubles(
    FV_Solution W_new,
    FV_Solution W_old,
    FV_Solution troubles,
    FV_Solution flagged,
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
    relax_NAD(troubles, flagged, alpha_x, alpha_y, alpha_z, limit_mask, use_sed);
    if(PAD)
        PAD_criteria(W_new, flagged);
}

//Whole-pack detection: one launch per criterion instead of one per block.
//
//This is what makes AMR affordable with the MOOD cascade. The cascade calls
//detection once per revision (up to cfg.max_revs per stage), so a per-block
//detection loop gets multiplied by the revision count on top of the block
//count -- with two refinement levels that was minutes per step. The geometry
//arrives as packed Matrices, one row per block, since blocks at different
//levels have different cell spacings.
void detect_troubles_b(
    FV_Solution W_new,
    FV_Solution W_old,
    FV_Solution troubles,
    FV_Solution flagged,
    FV_Solution alpha_x,
    FV_Solution alpha_y,
    FV_Solution alpha_z,
    Matrix cx, Matrix fx,
    Matrix cy, Matrix fy,
    Matrix cz, Matrix fz,
    int p,
    bool PAD,
    int limit_mask
    ){
    double tolerance = cfg.nad_tolerance;
    bool use_sed = cfg.sed && p > 1;
    NAD_b(W_new, W_old, troubles, tolerance, limit_mask);
    if(use_sed){
        if(cfg.active[_x_]) smooth_extrema_b(W_new, alpha_x, cx, fx, _x_, limit_mask);
        if(cfg.active[_y_]) smooth_extrema_b(W_new, alpha_y, cy, fy, _y_, limit_mask);
        if(cfg.active[_z_]) smooth_extrema_b(W_new, alpha_z, cz, fz, _z_, limit_mask);
    }
    relax_NAD_b(troubles, flagged, alpha_x, alpha_y, alpha_z, limit_mask, use_sed);
    if(PAD)
        PAD_criteria_b(W_new, flagged);
}