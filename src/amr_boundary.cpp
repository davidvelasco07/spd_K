#include "spd_k.hpp"
#include "forest.hpp"

KOKKOS_INLINE_FUNCTION
void amr_indices(int* N_id, int* n_id, int k, int j, int i, int kk, int jj, int ii,
                 int l, int ll, int dim){
    N_id[_x_] = dim == _x_ ? l  : i;
    n_id[_x_] = dim == _x_ ? ll : ii;
    N_id[_y_] = dim == _y_ ? l  : j;
    n_id[_y_] = dim == _y_ ? ll : jj;
    N_id[_z_] = dim == _z_ ? l  : k;
    n_id[_z_] = dim == _z_ ? ll : kk;
}

KOKKOS_INLINE_FUNCTION
void fv_indices(int* N_id, int k, int j, int i, int l, int dim){
    N_id[_x_] = dim == _x_ ? l  : i;
    N_id[_y_] = dim == _y_ ? l  : j;
    N_id[_z_] = dim == _z_ ? l  : k;
}

static SD_Solution make_scratch_like(const SD_Solution& ref, const char* name){
    SD_Solution s;
    s.n_ader = ref.n_ader;
    s.n_var = ref.n_var;
    s.Nx = ref.Nx; s.Ny = ref.Ny; s.Nz = ref.Nz;
    s.nx = ref.nx; s.ny = ref.ny; s.nz = ref.nz;
    s.label = name;
    Kokkos::resize(s.Vector, ref.n_ader, ref.n_var, ref.Nz, ref.Ny, ref.Nx,
                   ref.nz, ref.ny, ref.nx);
    return s;
}

static SD_Solution& block_fp(Hydro_ader& blk, int dim){
    return dim==_x_ ? blk.U_ader_fp_x
         : dim==_y_ ? blk.U_ader_fp_y
                    : blk.U_ader_fp_z;
}

static SD_Solution& block_Ffp(Hydro_ader& blk, int dim){
    return dim==_x_ ? blk.F_ader_fp_x
         : dim==_y_ ? blk.F_ader_fp_y
                    : blk.F_ader_fp_z;
}

static void copy_face_to_ghost(SD_Solution U, SD_Solution& src, int dim, int side){
    int N = (dim==_x_ ? U.Nx : (dim==_y_ ? U.Ny : U.Nz));
    int n = (dim==_x_ ? U.nx : (dim==_y_ ? U.ny : U.nz));
    int nader=U.n_ader, nvar=U.n_var;
    int Nx=U.Nx, Ny=U.Ny, Nz=U.Nz;
    int px=U.nx, py=U.ny, pz=U.nz;
    int Ns = (dim==_x_ ? src.Nx : (dim==_y_ ? src.Ny : src.Nz));
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        for(int t_id=0; t_id<nader; t_id++){
        for(int var=0; var<nvar; var++){
        int Nid[3], nid[3];
        double v;
        if(side==0){
            v = src.value(t_id,var,k,j,i,kk,jj,ii,Ns-2,n-1,dim);
            amr_indices(Nid,nid,k,j,i,kk,jj,ii,0,n-1,dim);
        } else {
            v = src.value(t_id,var,k,j,i,kk,jj,ii,1,0,dim);
            amr_indices(Nid,nid,k,j,i,kk,jj,ii,N-1,0,dim);
        }
        U.Vector(INDICES) = v;
        }}
    });
}

static void apply_domain_bc_fp(SD_Solution U, int dim, int side){
    if(cfg.bc[dim] != _gradfree_) return;
    int N = (dim==_x_ ? U.Nx : (dim==_y_ ? U.Ny : U.Nz));
    int n = (dim==_x_ ? U.nx : (dim==_y_ ? U.ny : U.nz));
    int nader=U.n_ader, nvar=U.n_var;
    int Nx=U.Nx, Ny=U.Ny, Nz=U.Nz;
    int px=U.nx, py=U.ny, pz=U.nz;
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        for(int t_id=0; t_id<nader; t_id++){
        for(int var=0; var<nvar; var++){
        int Nid[3], nid[3];
        double v;
        if(side==0){
            v = U.value(t_id,var,k,j,i,kk,jj,ii,1,0,dim);
            amr_indices(Nid,nid,k,j,i,kk,jj,ii,0,n-1,dim);
        } else {
            v = U.value(t_id,var,k,j,i,kk,jj,ii,N-2,n-1,dim);
            amr_indices(Nid,nid,k,j,i,kk,jj,ii,N-1,0,dim);
        }
        U.Vector(INDICES) = v;
        }}
    });
}

//Overwrite the shared interface flux on a block's boundary face. The Riemann
//solver stores each common flux twice -- once on the ghost side of the
//interface and once on the interior side -- and the update reads the interior
//copy, so a correction that only wrote the ghost would be a no-op.
static void set_interface_flux(SD_Solution U, SD_Solution& src, int dim, int side){
    int N = (dim==_x_ ? U.Nx : (dim==_y_ ? U.Ny : U.Nz));
    int n = (dim==_x_ ? U.nx : (dim==_y_ ? U.ny : U.nz));
    int nader=U.n_ader, nvar=U.n_var;
    int Nx=U.Nx, Ny=U.Ny, Nz=U.Nz;
    int px=U.nx, py=U.ny, pz=U.nz;
    int Ns = (dim==_x_ ? src.Nx : (dim==_y_ ? src.Ny : src.Nz));
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        for(int t_id=0; t_id<nader; t_id++){
        for(int var=0; var<nvar; var++){
        int Nid[3], nid[3];
        double v;
        if(side==0){
            v = src.value(t_id,var,k,j,i,kk,jj,ii,Ns-2,n-1,dim);
            amr_indices(Nid,nid,k,j,i,kk,jj,ii,0,n-1,dim);
            U.Vector(INDICES) = v;
            amr_indices(Nid,nid,k,j,i,kk,jj,ii,1,0,dim);
            U.Vector(INDICES) = v;
        } else {
            v = src.value(t_id,var,k,j,i,kk,jj,ii,1,0,dim);
            amr_indices(Nid,nid,k,j,i,kk,jj,ii,N-1,0,dim);
            U.Vector(INDICES) = v;
            amr_indices(Nid,nid,k,j,i,kk,jj,ii,N-2,n-1,dim);
            U.Vector(INDICES) = v;
        }
        }}
    });
}

void forest_exchange_fp(BlockForest& forest, std::vector<Hydro_ader>& blocks, int dim){
    if(!cfg.active[dim]) return;
    int nb = forest.Nblocks();

    for(int side=0; side<2; side++){
        const auto& sj = forest.same_jb[dim][side];
        if(!sj.empty()){
            for(int ib=0; ib<nb; ib++)
                copy_face_to_ghost(block_fp(blocks[ib], dim),
                                   block_fp(blocks[sj[ib]], dim), dim, side);
            continue;
        }
        const FaceGroups& g = forest.face_groups[dim][side];
        for(size_t k=0; k<g.same_ib.size(); k++)
            copy_face_to_ghost(block_fp(blocks[g.same_ib[k]], dim),
                               block_fp(blocks[g.same_jb[k]], dim), dim, side);
        for(int ib : g.bc_ib)
            apply_domain_bc_fp(block_fp(blocks[ib], dim), dim, side);
        for(size_t k=0; k<g.co_ib.size(); k++){
            SD_Solution& fine = block_fp(blocks[g.co_ib[k]], dim);
            SD_Solution& coarse = block_fp(blocks[g.co_jb[k]], dim);
            SD_Solution ghost = make_scratch_like(fine, "ghost");
            prolongate_face_coarser(coarse, ghost, amr_P, dim, g.co_sub[k]);
            copy_face_to_ghost(fine, ghost, dim, side);
        }
        //The fine neighbours already carry their facing trace at the same
        //`dim` element/point that copy_face_to_ghost will read out of `ghost`,
        //so they feed the restriction directly -- no staging copy.
        for(size_t k=0; k<g.fi_ib.size(); k++){
            int ib = g.fi_ib[k];
            SD_Solution& coarse = block_fp(blocks[ib], dim);
            SD_Solution ghost = make_scratch_like(coarse, "ghost");
            const SD_Solution* traces[8];
            int ns = (int)g.fi_jb[k].size();
            for(int s=0; s<ns; s++)
                traces[s] = &block_fp(blocks[g.fi_jb[k][s]], dim);
            restrict_face_overlap_sp(traces, ns, ghost, amr_RF, dim);
            copy_face_to_ghost(coarse, ghost, dim, side);
        }
    }
}

void correct_coarse_fine_flux(BlockForest& forest, std::vector<Hydro_ader>& blocks, int dim){
    if(forest.max_level()==0 || !cfg.active[dim]) return;
    for(int side=0; side<2; side++){
        const FaceGroups& g = forest.face_groups[dim][side];
        for(size_t k=0; k<g.fi_ib.size(); k++){
            int ib = g.fi_ib[k];
            SD_Solution& coarse = block_Ffp(blocks[ib], dim);
            SD_Solution ghost = make_scratch_like(coarse, "ghost");
            const SD_Solution* traces[8];
            int ns = (int)g.fi_jb[k].size();
            for(int s=0; s<ns; s++)
                traces[s] = &block_Ffp(blocks[g.fi_jb[k][s]], dim);
            restrict_face_overlap_sp(traces, ns, ghost, amr_RF, dim);
            set_interface_flux(coarse, ghost, dim, side);
        }
    }
}

static FV_Solution& block_Ffv(Hydro_ader& blk, int dim){
    return dim==_x_ ? blk.F_x
         : dim==_y_ ? blk.F_y
                    : blk.F_z;
}

//Restrict one fine neighbour's boundary-face fluxes onto the part of the
//coarse boundary face it covers. `c*` select which half of the coarse face
//this neighbour occupies along each transverse direction, `Nc*` are the
//coarse block's interior cell counts and `n*` its cells per element.
//
//The FV cells are flux-point spaced, so two fine cells do not simply halve a
//coarse one: amr_RS_cv holds, per child, the fraction of each coarse cell
//covered by each fine cell. Those weights are what make this exact -- summed
//over a coarse cell they give its width back, so flux times area balances.
void restrict_face_fv_sub(FV_Solution C, FV_Solution F, int dim,
                          int cface, int fface, int cx, int cy, int cz,
                          int Ncx, int Ncy, int Ncz, int nx, int ny, int nz){
    bool tx = cfg.active[_x_] && dim!=_x_;
    bool ty = cfg.active[_y_] && dim!=_y_;
    bool tz = cfg.active[_z_] && dim!=_z_;
    Matrix R0 = amr_RS_cv[0], R1 = amr_RS_cv[1];
    int nvar = C.n_var;
    GHOST_LOCALS;
    int x0,x1,y0,y1,z0,z1;
    if(dim==_x_){ x0=cface; x1=cface+1; }
    else if(tx){ x0=sghx+cx*(Ncx/2); x1=x0+Ncx/2; }
    else { x0=0; x1=1; }
    if(dim==_y_){ y0=cface; y1=cface+1; }
    else if(ty){ y0=sghy+cy*(Ncy/2); y1=y0+Ncy/2; }
    else { y0=0; y1=1; }
    if(dim==_z_){ z0=cface; z1=cface+1; }
    else if(tz){ z0=sghz+cz*(Ncz/2); z1=z0+Ncz/2; }
    else { z0=0; z1=1; }
    Kokkos::parallel_for("restrict_face_fv_sub",
        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({z0,y0,x0},{z1,y1,x1}),
        KOKKOS_LAMBDA(int k, int j, int i){
        //Coarse cell -> its element and the cell within it, which selects
        //the row of the overlap weights.
        int ex=0,jx=0,ey=0,jy=0,ez=0,jz=0;
        if(tx){ int r=i-sghx-cx*(Ncx/2); ex=r/nx; jx=r%nx; }
        if(ty){ int r=j-sghy-cy*(Ncy/2); ey=r/ny; jy=r%ny; }
        if(tz){ int r=k-sghz-cz*(Ncz/2); ez=r/nz; jz=r%nz; }
        for(int var=0; var<nvar; var++){
            double u=0;
            for(int sz=0; sz<(tz?2:1); sz++)
            for(int iz=0; iz<(tz?nz:1); iz++)
            for(int sy=0; sy<(ty?2:1); sy++)
            for(int iy=0; iy<(ty?ny:1); iy++)
            for(int sx=0; sx<(tx?2:1); sx++)
            for(int ix=0; ix<(tx?nx:1); ix++){
                double w=1.0;
                int fi = (dim==_x_ ? fface : i);
                int fj = (dim==_y_ ? fface : j);
                int fk = (dim==_z_ ? fface : k);
                if(tx){ fi = sghx+(2*ex+sx)*nx+ix; w *= (sx==0?R0:R1)(jx,ix); }
                if(ty){ fj = sghy+(2*ey+sy)*ny+iy; w *= (sy==0?R0:R1)(jy,iy); }
                if(tz){ fk = sghz+(2*ez+sz)*nz+iz; w *= (sz==0?R0:R1)(jz,iz); }
                u += w * F.Vector(var,fk,fj,fi);
            }
            C.Vector(var,k,j,i) = u;
        }
    });
}

//At a coarse-fine face the coarse block's FV flux has to equal the
//overlap-weighted average of the fine fluxes covering it, or the coarse cell
//fails to lose exactly what the fine cells gain. The high-order fluxes were
//already reconciled in correct_coarse_fine_flux, but the fallback blends
//MUSCL fluxes in afterwards and the two sides blend differently, so the
//balance has to be restored on the final flux.
void correct_coarse_fine_fv_flux(BlockForest& forest, std::vector<Hydro_ader>& blocks, int dim){
    if(forest.max_level()==0 || !cfg.active[dim]) return;
    GHOST_LOCALS;
    for(int side=0; side<2; side++){
        const FaceGroups& g = forest.face_groups[dim][side];
        for(size_t q=0; q<g.fi_ib.size(); q++){
            int ib = g.fi_ib[q];
            FV_Solution C = block_Ffv(blocks[ib], dim);
            SD_Solution S = blocks[ib].W_cv;
            int nx=S.nx, ny=S.ny, nz=S.nz;
            int Ncx=(S.Nx-2*NGHx)*nx, Ncy=(S.Ny-2*NGHy)*ny, Ncz=(S.Nz-2*NGHz)*nz;
            //Own boundary face along the normal, and the fine neighbour's
            //face that meets it (the opposite end of its block).
            int lo = (dim==_x_?sghx:dim==_y_?sghy:sghz);
            int hi = lo + (dim==_x_?Ncx:dim==_y_?Ncy:Ncz);
            int cface = (side==0 ? lo : hi);
            int fface = (side==0 ? hi : lo);
            int ns = (int)g.fi_jb[q].size();
            for(int sub=0; sub<ns; sub++){
                int cx=0, cy=0, cz=0;
                int bit=0;
                for(int d=0; d<3; d++){
                    if(d==dim || !cfg.active[d]) continue;
                    int v = (sub>>bit)&1;
                    if(d==_x_) cx=v; else if(d==_y_) cy=v; else cz=v;
                    bit++;
                }
                restrict_face_fv_sub(C, block_Ffv(blocks[g.fi_jb[q][sub]], dim),
                                     dim, cface, fface, cx, cy, cz,
                                     Ncx, Ncy, Ncz, nx, ny, nz);
            }
        }
    }
}

static void fv_copy_slab(FV_Solution U, FV_Solution src, int dim, int side, int ngh){
    int N = (dim==_x_ ? U.Nx : (dim==_y_ ? U.Ny : U.Nz));
    int Ns = (dim==_x_ ? src.Nx : (dim==_y_ ? src.Ny : src.Nz));
    int nvar = U.n_var;
    int Nx = (dim==_x_ ? ngh : U.Nx);
    int Ny = (dim==_y_ ? ngh : U.Ny);
    int Nz = (dim==_z_ ? ngh : U.Nz);
    fv_for_cells(Nz,Ny,Nx, KOKKOS_LAMBDA(int k, int j, int i){
        for(int var=0; var<nvar; var++){
        int Nid[3];
        double v;
        int l = (dim==_x_ ? i : (dim==_y_ ? j : k));
        if(side==0){
            fv_indices(Nid,k,j,i,Ns-2*ngh+l,dim);
            v = src.Vector(FV_INDICES);
            fv_indices(Nid,k,j,i,l,dim);
        } else {
            fv_indices(Nid,k,j,i,ngh+l,dim);
            v = src.Vector(FV_INDICES);
            fv_indices(Nid,k,j,i,N-ngh+l,dim);
        }
        U.Vector(FV_INDICES) = v;
        }
    });
}

static void fv_inject_coarser(FV_Solution U, FV_Solution coarse, int dim, int side, int /*sub*/){
    int ngh = nGH_rt[dim];
    int half = std::max(1, ngh/2);
    int N = (dim==_x_ ? U.Nx : (dim==_y_ ? U.Ny : U.Nz));
    int Nc = (dim==_x_ ? coarse.Nx : (dim==_y_ ? coarse.Ny : coarse.Nz));
    int nvar = U.n_var;
    int Nx = (dim==_x_ ? ngh : U.Nx);
    int Ny = (dim==_y_ ? ngh : U.Ny);
    int Nz = (dim==_z_ ? ngh : U.Nz);
    //This fills the transverse ghost corners too, by prolongating this one
    //coarse neighbour across them. Those cells lie outside the block, so the
    //value is only a fallback: forest_exchange_fv_same overwrites it wherever
    //a same-level neighbour owns the corner.
    fv_for_cells(Nz,Ny,Nx, KOKKOS_LAMBDA(int k, int j, int i){
        for(int var=0; var<nvar; var++){
        int Nid[3], Nidc[3];
        int l = (dim==_x_ ? i : (dim==_y_ ? j : k));
        int cl = (side==0 ? Nc-2*ngh-half+l/2 : ngh+l/2);
        fv_indices(Nidc,k,j,i,cl,dim);
        double v = coarse.Vector(var,Nidc[_z_],Nidc[_y_],Nidc[_x_]);
        fv_indices(Nid,k,j,i,(side==0?l:N-ngh+l),dim);
        U.Vector(FV_INDICES) = v;
        }
    });
}

static void fv_restrict_finer(FV_Solution U, FV_Solution& f0, FV_Solution& f1,
                              FV_Solution& f2, FV_Solution& f3, int nf, int dim, int side){
    int ngh = nGH_rt[dim];
    int nvar = U.n_var;
    int Nx = (dim==_x_ ? ngh : U.Nx);
    int Ny = (dim==_y_ ? ngh : U.Ny);
    int Nz = (dim==_z_ ? ngh : U.Nz);
    fv_for_cells(Nz,Ny,Nx, KOKKOS_LAMBDA(int k, int j, int i){
        for(int var=0; var<nvar; var++){
        double sum=0; int cnt=0;
        if(nf>0){ int Nid[3]; int l=(dim==_x_?i:(dim==_y_?j:k)); int fl=(side==0?f0.Nx-3*ngh+l:ngh+l);
            fv_indices(Nid,k,j,i,fl,dim); sum+=f0.Vector(FV_INDICES); cnt++; }
        if(nf>1){ int Nid[3]; int l=(dim==_x_?i:(dim==_y_?j:k)); int fl=(side==0?f1.Nx-3*ngh+l:ngh+l);
            fv_indices(Nid,k,j,i,fl,dim); sum+=f1.Vector(FV_INDICES); cnt++; }
        if(nf>2){ int Nid[3]; int l=(dim==_x_?i:(dim==_y_?j:k)); int fl=(side==0?f2.Nx-3*ngh+l:ngh+l);
            fv_indices(Nid,k,j,i,fl,dim); sum+=f2.Vector(FV_INDICES); cnt++; }
        if(nf>3){ int Nid[3]; int l=(dim==_x_?i:(dim==_y_?j:k)); int fl=(side==0?f3.Nx-3*ngh+l:ngh+l);
            fv_indices(Nid,k,j,i,fl,dim); sum+=f3.Vector(FV_INDICES); cnt++; }
        int Nid[3];
        if(dim==_x_) fv_indices(Nid,k,j,i,(side==0?i:U.Nx-ngh+i),dim);
        else if(dim==_y_) fv_indices(Nid,k,j,i,(side==0?j:U.Ny-ngh+j),dim);
        else fv_indices(Nid,k,j,i,(side==0?k:U.Nz-ngh+k),dim);
        U.Vector(FV_INDICES) = sum/max(cnt,1);
        }
    });
}

//Same-level copies only. Run after every direction has been exchanged, this
//fills each transverse ghost corner from the neighbour that actually owns it,
//which the coarse-fine operators deliberately skip.
void forest_exchange_fv_same(BlockForest& forest, std::vector<Hydro_ader>& blocks,
                             FV_Solution Hydro_ader::*member, int dim){
    if(!cfg.active[dim]) return;
    int nb = forest.Nblocks();
    for(int side=0; side<2; side++){
        const auto& sj = forest.same_jb[dim][side];
        if(!sj.empty()){
            for(int ib=0; ib<nb; ib++)
                fv_copy_slab(blocks[ib].*member, blocks[sj[ib]].*member,
                             dim, side, nGH_rt[dim]);
            continue;
        }
        const FaceGroups& g = forest.face_groups[dim][side];
        for(size_t k=0; k<g.same_ib.size(); k++)
            fv_copy_slab(blocks[g.same_ib[k]].*member, blocks[g.same_jb[k]].*member,
                         dim, side, nGH_rt[dim]);
    }
}

void forest_exchange_fv(BlockForest& forest, std::vector<Hydro_ader>& blocks,
                        FV_Solution Hydro_ader::*member, int dim){
    if(!cfg.active[dim]) return;
    int nb = forest.Nblocks();
    for(int side=0; side<2; side++){
        const auto& sj = forest.same_jb[dim][side];
        if(!sj.empty()){
            for(int ib=0; ib<nb; ib++)
                fv_copy_slab(blocks[ib].*member, blocks[sj[ib]].*member, dim, side, nGH_rt[dim]);
            continue;
        }
        const FaceGroups& g = forest.face_groups[dim][side];
        for(size_t k=0; k<g.same_ib.size(); k++)
            fv_copy_slab(blocks[g.same_ib[k]].*member, blocks[g.same_jb[k]].*member,
                         dim, side, nGH_rt[dim]);
        for(size_t k=0; k<g.co_ib.size(); k++)
            fv_inject_coarser(blocks[g.co_ib[k]].*member, blocks[g.co_jb[k]].*member,
                              dim, side, g.co_sub[k]);
        for(size_t k=0; k<g.fi_ib.size(); k++){
            FV_Solution &U = blocks[g.fi_ib[k]].*member;
            FV_Solution &f0 = blocks[g.fi_jb[k][0]].*member;
            FV_Solution f1=f0,f2=f0,f3=f0;
            int nf = (int)g.fi_jb[k].size();
            if(nf>1) f1 = blocks[g.fi_jb[k][1]].*member;
            if(nf>2) f2 = blocks[g.fi_jb[k][2]].*member;
            if(nf>3) f3 = blocks[g.fi_jb[k][3]].*member;
            fv_restrict_finer(U, f0, f1, f2, f3, nf, dim, side);
        }
    }
}
