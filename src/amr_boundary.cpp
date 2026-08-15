#include "spd_k.hpp"
#include "forest.hpp"
#include <type_traits>

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

//Unpack a sub-face index into one transverse half per active direction. Device
//copy of fv_sub_bits, which reads cfg on the host: the activity flags have to
//ride into the kernel as plain ints.
KOKKOS_INLINE_FUNCTION
void fv_sub_bits_d(int sub, int dim, int actx, int acty, int actz,
                   int& bx, int& by, int& bz){
    bx = by = bz = 0;
    int bit = 0;
    for(int d=0; d<3; d++){
        int act = (d==_x_?actx:(d==_y_?acty:actz));
        if(d==dim || !act) continue;
        int v = (sub>>bit)&1;
        if(d==_x_) bx=v; else if(d==_y_) by=v; else bz=v;
        bit++;
    }
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

//Pick SP or FP transfer matrices from the transverse point count of U.
static Matrix prolong_mat_for(const SD_Solution& U, int dim){
    int nt = (dim==_x_ ? U.ny : (dim==_y_ ? U.nx : U.nx));
    //Inactive transverse dims report n=1; prefer SP matrices then.
    if(nt == (int)amr_P_fp.extent(1)) return amr_P_fp;
    return amr_P;
}
static Matrix restrict_mat_for(const SD_Solution& U, int dim){
    int nt = (dim==_x_ ? U.ny : (dim==_y_ ? U.nx : U.nx));
    if(nt == (int)amr_RF_fp.extent(0)) return amr_RF_fp;
    return amr_RF;
}

template<typename Block>
static SD_Solution& block_fp(Block& blk, int dim){
    return dim==_x_ ? blk.U_ader_fp_x
         : dim==_y_ ? blk.U_ader_fp_y
                    : blk.U_ader_fp_z;
}

template<typename Block>
static SD_Solution& block_Ffp(Block& blk, int dim){
    return dim==_x_ ? blk.F_ader_fp_x
         : dim==_y_ ? blk.F_ader_fp_y
                    : blk.F_ader_fp_z;
}

template<typename Block>
static FV_Solution& block_Ffv(Block& blk, int dim);

template<>
FV_Solution& block_Ffv<Hydro_ader>(Hydro_ader& blk, int dim){
    return dim==_x_ ? blk.F_x
         : dim==_y_ ? blk.F_y
                    : blk.F_z;
}

template<>
FV_Solution& block_Ffv<MHD_ader>(MHD_ader& blk, int dim){
    return dim==_x_ ? blk.F0_x
         : dim==_y_ ? blk.F0_y
                    : blk.F0_z;
}

//src is taken by value on purpose. nvcc's extended lambdas cannot capture a
//reference: `[=]` on a reference parameter leaves a host pointer in the
//closure, which is fine when host and device are the same space and reads
//garbage on CUDA. SD_Solution holds Kokkos Views, so a copy is a shallow
//handle and writes still land in the same allocation.
static void copy_face_to_ghost(SD_Solution U, SD_Solution src, int dim, int side){
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

static void mirror_face_to_ghost(SD_Solution U, int dim, int side){
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

void apply_domain_bc_fp(SD_Solution U, int dim, int side){
    if(cfg.bc[dim] != _gradfree_) return;
    mirror_face_to_ghost(U, dim, side);
}

//FV counterpart: fill a block's ghost slab at a physical domain boundary.
//The gathers only ever write ghosts that have a neighbour, so without this
//the ghost slab of a boundary block keeps whatever was last in it.
//
//Gradfree only, matching apply_domain_bc_fp. Reflective is NOT handled here
//and must not be: the mesh path has no reflective support anywhere (the
//uniform neighbour tables turn it into a periodic wrap), so a silent mirror
//here would paper over half of a wrong answer. main.cpp rejects a
//multiblock run with any other boundary type.
//
//Semantics match the single-block path in boundary.cpp: ghost cell l takes
//the interior cell nGH+l on the low side, N-2*nGH+l on the high side.
void apply_domain_bc_fv(FV_Solution U, int dim, int side, int ngh){
    if(cfg.bc[dim] != _gradfree_) return;
    const int N = (dim==_x_ ? U.Nx : (dim==_y_ ? U.Ny : U.Nz));
    const int nvar = U.n_var;
    const int Nx = (dim==_x_ ? ngh : U.Nx);
    const int Ny = (dim==_y_ ? ngh : U.Ny);
    const int Nz = (dim==_z_ ? ngh : U.Nz);
    fv_for_cells(Nz,Ny,Nx, KOKKOS_LAMBDA(int k, int j, int i){
        const int l  = (dim==_x_ ? i : (dim==_y_ ? j : k));
        const int sl = (side==0 ? ngh+l : N-2*ngh+l);  //nearest interior cell
        const int dl = (side==0 ? l     : N-ngh+l);    //my ghost cell
        int Nsrc[3], Ndst[3];
        fv_indices(Nsrc,k,j,i,sl,dim);
        fv_indices(Ndst,k,j,i,dl,dim);
        for(int var=0; var<nvar; var++)
            U.Vector(var,Ndst[_z_],Ndst[_y_],Ndst[_x_]) =
            U.Vector(var,Nsrc[_z_],Nsrc[_y_],Nsrc[_x_]);
    }, "apply_domain_bc_fv");
}

//Overwrite the shared interface flux on a block's boundary face. The Riemann
//solver stores each common flux twice -- once on the ghost side of the
//interface and once on the interior side -- and the update reads the interior
//copy, so a correction that only wrote the ghost would be a no-op.
//By value, for the reason given on copy_face_to_ghost.
static void set_interface_flux(SD_Solution U, SD_Solution src, int dim, int side){
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

template<typename Block>
void forest_exchange_fp(BlockForest& forest, std::vector<Block>& blocks, int dim){
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

//Generic SD-field forest exchange (face B, edge EMF, etc.).
//cf_prolong=true: coarse↔fine use transverse P/R (face-B / fluid-like).
//cf_prolong=false: coarse↔fine mirror own interior (edge EMF on the fp
//lattice — Lagrange fp P is ill-conditioned for long SMR runs).
template<typename Block>
void forest_exchange_sd(BlockForest& forest, std::vector<Block>& blocks,
                        SD_Solution Block::*member, int dim, bool cf_prolong){
    if(!cfg.active[dim]) return;
    int nb = forest.Nblocks();
    for(int side=0; side<2; side++){
        const auto& sj = forest.same_jb[dim][side];
        if(!sj.empty()){
            for(int ib=0; ib<nb; ib++)
                copy_face_to_ghost(blocks[ib].*member, blocks[sj[ib]].*member,
                                   dim, side);
            continue;
        }
        const FaceGroups& g = forest.face_groups[dim][side];
        for(size_t k=0; k<g.same_ib.size(); k++)
            copy_face_to_ghost(blocks[g.same_ib[k]].*member,
                               blocks[g.same_jb[k]].*member, dim, side);
        for(int ib : g.bc_ib)
            apply_domain_bc_fp(blocks[ib].*member, dim, side);
        if(!cf_prolong){
            for(size_t k=0; k<g.co_ib.size(); k++)
                mirror_face_to_ghost(blocks[g.co_ib[k]].*member, dim, side);
            for(size_t k=0; k<g.fi_ib.size(); k++)
                mirror_face_to_ghost(blocks[g.fi_ib[k]].*member, dim, side);
            continue;
        }
        for(size_t k=0; k<g.co_ib.size(); k++){
            SD_Solution& fine = blocks[g.co_ib[k]].*member;
            SD_Solution& coarse = blocks[g.co_jb[k]].*member;
            SD_Solution ghost = make_scratch_like(fine, "ghostE");
            prolongate_face_coarser(coarse, ghost, prolong_mat_for(fine, dim),
                                    dim, g.co_sub[k]);
            copy_face_to_ghost(fine, ghost, dim, side);
        }
        for(size_t k=0; k<g.fi_ib.size(); k++){
            int ib = g.fi_ib[k];
            SD_Solution& coarse = blocks[ib].*member;
            SD_Solution ghost = make_scratch_like(coarse, "ghostE");
            const SD_Solution* traces[8];
            int ns = (int)g.fi_jb[k].size();
            for(int s=0; s<ns; s++)
                traces[s] = &(blocks[g.fi_jb[k][s]].*member);
            restrict_face_overlap_sp(traces, ns, ghost,
                                     restrict_mat_for(coarse, dim), dim);
            copy_face_to_ghost(coarse, ghost, dim, side);
        }
    }
}

//Shared face-B identity across levels: left/coarse-fine interface gets a
//single value. Same-level: left neighbour wins. Fine next to coarse: prolongate
//the coarse face onto the fine interface. Coarse next to fine: restrict the
//covering fine faces onto the coarse interface (both sides of the face).
template<typename Block>
void forest_sync_face_B(BlockForest& forest, std::vector<Block>& blocks,
                        SD_Solution Block::*member, int dim){
    if(!cfg.active[dim]) return;
    int nb = forest.Nblocks();
    for(int side=0; side<2; side++){
        const auto& sj = forest.same_jb[dim][side];
        if(!sj.empty()){
            for(int ib=0; ib<nb; ib++){
                int L = (side==0 ? sj[ib] : ib);
                if(side==0)
                    sync_shared_face_sd(blocks[ib].*member, blocks[L].*member,
                                        blocks[ib].*member, _periodic_, _periodic_, dim);
            }
            continue;
        }
        const FaceGroups& g = forest.face_groups[dim][side];
        for(size_t k=0; k<g.same_ib.size(); k++){
            if(side==0)
                sync_shared_face_sd(blocks[g.same_ib[k]].*member,
                                    blocks[g.same_jb[k]].*member,
                                    blocks[g.same_ib[k]].*member,
                                    _periodic_, _periodic_, dim);
        }
        for(size_t k=0; k<g.co_ib.size(); k++){
            SD_Solution& fine = blocks[g.co_ib[k]].*member;
            SD_Solution& coarse = blocks[g.co_jb[k]].*member;
            SD_Solution ghost = make_scratch_like(fine, "syncB");
            prolongate_face_coarser(coarse, ghost, prolong_mat_for(fine, dim),
                                    dim, g.co_sub[k]);
            set_interface_flux(fine, ghost, dim, side);
        }
        for(size_t k=0; k<g.fi_ib.size(); k++){
            SD_Solution& coarse = blocks[g.fi_ib[k]].*member;
            SD_Solution ghost = make_scratch_like(coarse, "syncB");
            const SD_Solution* traces[8];
            int ns = (int)g.fi_jb[k].size();
            for(int s=0; s<ns; s++)
                traces[s] = &(blocks[g.fi_jb[k][s]].*member);
            restrict_face_overlap_sp(traces, ns, ghost, restrict_mat_for(coarse, dim), dim);
            set_interface_flux(coarse, ghost, dim, side);
        }
    }
}

template<typename Block>
void correct_coarse_fine_flux(BlockForest& forest, std::vector<Block>& blocks, int dim){
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

//At a coarse-fine face the coarse block's edge EMF must equal the line-integral
//average of the overlapping fine EMFs, or the CT update of face B fails to
//telescope across the interface (AthenaK flux_correct_fc).
template<typename Block>
void correct_coarse_fine_emf(BlockForest& forest, std::vector<Block>& blocks, int dim){
    if(forest.max_level()==0 || !cfg.active[dim]) return;
    if constexpr (std::is_same_v<Block, MHD_ader>){
        auto correct_one = [&](SD_Solution MHD_ader::*member){
            for(int side=0; side<2; side++){
                const FaceGroups& g = forest.face_groups[dim][side];
                for(size_t k=0; k<g.fi_ib.size(); k++){
                    SD_Solution& coarse = blocks[g.fi_ib[k]].*member;
                    int ns = (int)g.fi_jb[k].size();
                    //Build a coarse-shaped buffer holding only the interface
                    //face: transverse restrict of each fine neighbour's facing
                    //trace, then set_interface_flux.
                    SD_Solution ghost = make_scratch_like(coarse, "emf");
                    Kokkos::deep_copy(ghost.Vector, coarse.Vector);
                    const SD_Solution* traces[8];
                    for(int s=0; s<ns; s++)
                        traces[s] = &(blocks[g.fi_jb[k][s]].*member);
                    //Use SP restrict when transverse count matches amr_RF,
                    //else the constant-preserving fp restrict.
                    restrict_face_overlap_sp(traces, ns, ghost,
                                             restrict_mat_for(coarse, dim), dim);
                    set_interface_flux(coarse, ghost, dim, side);
                }
            }
        };
        //Only the EMF itself (used by CT) must match; correcting all NEMHD
        //channels is fine and matches the fluid flux-correction pattern.
        if(dim==_x_ || dim==_y_)
            correct_one(&MHD_ader::Ez_ep_xy);
        if(cfg.active[_z_]){
            if(dim==_x_ || dim==_z_) correct_one(&MHD_ader::Ey_ep_zx);
            if(dim==_y_ || dim==_z_) correct_one(&MHD_ader::Ex_ep_yz);
        }
    }
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
    for_box3(z0,z1,y0,y1,x0,x1, KOKKOS_LAMBDA(int k, int j, int i){
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
    }, "restrict_face_fv_sub");
}

//One same-level interface, projected onto its shared average. The two copies
//live at my_face of U and nb_face of N; both are read before either is written,
//so passing the same block twice (a periodic wrap onto itself) is safe.
static void symmetrize_face_fv_pair(FV_Solution U, FV_Solution N, int dim,
                                    int my_face, int nb_face,
                                    int Ncx, int Ncy, int Ncz){
    const bool tx = cfg.active[_x_] && dim!=_x_;
    const bool ty = cfg.active[_y_] && dim!=_y_;
    const bool tz = cfg.active[_z_] && dim!=_z_;
    const int nvar = U.n_var;
    GHOST_LOCALS;
    //Transverse extent is the active cells -- the ones whose divergence reads
    //this face. The normal axis is collapsed to one; fv_indices puts each
    //side's own face index back in.
    const int x0 = tx ? sghx : 0, x1 = tx ? sghx+Ncx : 1;
    const int y0 = ty ? sghy : 0, y1 = ty ? sghy+Ncy : 1;
    const int z0 = tz ? sghz : 0, z1 = tz ? sghz+Ncz : 1;
    FV_Vector u = U.Vector, n = N.Vector;
    for_box3(z0,z1,y0,y1,x0,x1, KOKKOS_LAMBDA(int k, int j, int i){
        int Nm[3], Nn[3];
        fv_indices(Nm,k,j,i,my_face,dim);
        fv_indices(Nn,k,j,i,nb_face,dim);
        for(int var=0; var<nvar; var++){
            const double avg = 0.5*(u(var,Nm[_z_],Nm[_y_],Nm[_x_])
                                  + n(var,Nn[_z_],Nn[_y_],Nn[_x_]));
            u(var,Nm[_z_],Nm[_y_],Nm[_x_]) = avg;
            n(var,Nn[_z_],Nn[_y_],Nn[_x_]) = avg;
        }
    }, "symmetrize_face_fv_pair");
}

//A same-level block interface is stored twice: as the high face of the lower
//block and as the low face of the upper one. The two copies agree only while
//both sides pick the same flux, and the fallback blend (theta varies across an
//interface) and the MOOD cascade are per-cell decisions -- so the face can end
//up double-valued, and whatever the copies disagree by is exactly what leaks
//between the two blocks. Project each pair onto its average, which is the
//unique flux both sides then see. This is spd's symmetrize_same_level_fv_flux.
template<typename Block>
void symmetrize_same_level_fv_flux(BlockForest& forest, std::vector<Block>& blocks, int dim){
    if(!cfg.active[dim]) return;
    GHOST_LOCALS;
    //Every same-level interface appears exactly once as some block's LOW face
    //(side 0) with a SAME neighbour -- periodic wraps included -- so one pass
    //over that group covers each face once and needs no seen-set.
    const FaceGroups& g = forest.face_groups[dim][0];
    for(size_t q=0; q<g.same_ib.size(); q++){
        const int ib = g.same_ib[q], jb = g.same_jb[q];
        SD_Solution S = blocks[ib].W_cv;
        const int Ncx=(S.Nx-2*NGHx)*S.nx, Ncy=(S.Ny-2*NGHy)*S.ny, Ncz=(S.Nz-2*NGHz)*S.nz;
        const int lo = (dim==_x_?sghx:dim==_y_?sghy:sghz);
        const int hi = lo + (dim==_x_?Ncx:dim==_y_?Ncy:Ncz);
        symmetrize_face_fv_pair(block_Ffv<Block>(blocks[ib], dim),
                                block_Ffv<Block>(blocks[jb], dim),
                                dim, lo, hi, Ncx, Ncy, Ncz);
    }
}

//At a coarse-fine face the coarse block's FV flux has to equal the
//overlap-weighted average of the fine fluxes covering it, or the coarse cell
//fails to lose exactly what the fine cells gain. The high-order fluxes were
//already reconciled in correct_coarse_fine_flux, but the fallback blends
//MUSCL fluxes in afterwards and the two sides blend differently, so the
//balance has to be restored on the final flux.
template<typename Block>
void correct_coarse_fine_fv_flux(BlockForest& forest, std::vector<Block>& blocks, int dim){
    if(forest.max_level()==0 || !cfg.active[dim]) return;
    GHOST_LOCALS;
    for(int side=0; side<2; side++){
        const FaceGroups& g = forest.face_groups[dim][side];
        for(size_t q=0; q<g.fi_ib.size(); q++){
            int ib = g.fi_ib[q];
            FV_Solution C = block_Ffv<Block>(blocks[ib], dim);
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
                restrict_face_fv_sub(C, block_Ffv<Block>(blocks[g.fi_jb[q][sub]], dim),
                                     dim, cface, fface, cx, cy, cz,
                                     Ncx, Ncy, Ncz, nx, ny, nz);
            }
        }
    }
}

//Same correction as correct_coarse_fine_fv_flux, run off the fine->coarse
//transaction table instead of a host loop over (coarse block, side, sub): the
//table already carries exactly one entry per (coarse receiver, fine neighbour,
//sub-face), which is the unit of work here. Coarse and fine fluxes live in the
//same pack, so both are offsets into one array.
//
//One kernel per (dim, side), constant in the block count.
void correct_cf_fv_flux_b(FV_Solution C, IntVector recv, IntVector send,
                          IntVector subv, int ntr, int dim, int side,
                          int cface, int fface,
                          int Ncx, int Ncy, int Ncz, int nx, int ny, int nz){
    if(ntr <= 0) return;
    const int actx=cfg.active[_x_], acty=cfg.active[_y_], actz=cfg.active[_z_];
    const bool tx = actx && dim!=_x_, ty = acty && dim!=_y_, tz = actz && dim!=_z_;
    Matrix R0 = amr_RS_cv[0], R1 = amr_RS_cv[1];
    const int nvar = C.n_var;
    GHOST_LOCALS;
    //One coarse face cell along the normal; half the coarse face in each
    //transverse direction, which is the part this fine neighbour covers.
    const int Qx = (dim==_x_) ? 1 : (tx ? Ncx/2 : 1);
    const int Qy = (dim==_y_) ? 1 : (ty ? Ncy/2 : 1);
    const int Qz = (dim==_z_) ? 1 : (tz ? Ncz/2 : 1);
    fv_for_cells_b(ntr,Qz,Qy,Qx, KOKKOS_LAMBDA(int b,int kq,int jq,int iq){
        int cx,cy,cz; fv_sub_bits_d(subv(b),dim,actx,acty,actz,cx,cy,cz);
        const int i = (dim==_x_) ? cface : (tx ? sghx + cx*(Ncx/2) + iq : 0);
        const int j = (dim==_y_) ? cface : (ty ? sghy + cy*(Ncy/2) + jq : 0);
        const int k = (dim==_z_) ? cface : (tz ? sghz + cz*(Ncz/2) + kq : 0);
        const int rb = recv(b)*nvar, sb = send(b)*nvar;
        //Coarse cell -> its element and the cell within it, which selects the
        //row of the overlap weights.
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
                u += w * C.Vector(sb+var,fk,fj,fi);
            }
            C.Vector(rb+var,k,j,i) = u;
        }
    }, "correct_cf_fv_flux_b");
}

//Same symmetrization as symmetrize_same_level_fv_flux, run off the same-level
//transaction table instead of a host loop over interfaces: the table already
//carries exactly one entry per (upper block, lower neighbour), which is the
//unit of work here. Both sides live in one pack, so the pair is two offsets
//into the same array.
//
//One kernel per dim, constant in the block count.
void symmetrize_same_level_fv_flux_b(FV_Solution C, IntVector recv, IntVector send,
                                     int ntr, int dim, int lo, int hi,
                                     int Ncx, int Ncy, int Ncz){
    if(ntr <= 0) return;
    const int nvar = C.n_var;
    GHOST_LOCALS;
    const bool tx = cfg.active[_x_] && dim!=_x_;
    const bool ty = cfg.active[_y_] && dim!=_y_;
    const bool tz = cfg.active[_z_] && dim!=_z_;
    const int Qx = tx ? Ncx : 1, Qy = ty ? Ncy : 1, Qz = tz ? Ncz : 1;
    fv_for_cells_b(ntr,Qz,Qy,Qx, KOKKOS_LAMBDA(int b,int kq,int jq,int iq){
        const int i = tx ? sghx+iq : 0;
        const int j = ty ? sghy+jq : 0;
        const int k = tz ? sghz+kq : 0;
        //recv owns the LOW face of the interface; send is its low-side
        //neighbour, which holds the matching HIGH face.
        int Nl[3], Nh[3];
        fv_indices(Nl,k,j,i,lo,dim);
        fv_indices(Nh,k,j,i,hi,dim);
        const int rb = recv(b)*nvar, sb = send(b)*nvar;
        for(int var=0; var<nvar; var++){
            const double avg = 0.5*(C.Vector(rb+var,Nl[_z_],Nl[_y_],Nl[_x_])
                                  + C.Vector(sb+var,Nh[_z_],Nh[_y_],Nh[_x_]));
            C.Vector(rb+var,Nl[_z_],Nl[_y_],Nl[_x_]) = avg;
            C.Vector(sb+var,Nh[_z_],Nh[_y_],Nh[_x_]) = avg;
        }
    }, "symmetrize_same_level_fv_flux_b");
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

//Floor division by two. The built-in truncates toward zero, which is off by
//one for the negative offsets that transverse ghost cells produce.
KOKKOS_INLINE_FUNCTION int fv_fdiv2(int a){ return a>=0 ? a/2 : -((-a+1)/2); }

KOKKOS_INLINE_FUNCTION int fv_clamp(int a, int hi){
    return a<0 ? 0 : (a>hi ? hi : a);
}

//Unpack the sub-face index into one transverse half per active direction, in
//the ascending-dimension order the forest packs them (forest.cpp fills
//`row[e.sub]`, and correct_coarse_fine_fv_flux unpacks the same way).
static void fv_sub_bits(int sub, int dim, int& bx, int& by, int& bz){
    bx = by = bz = 0;
    int bit = 0;
    for(int d=0; d<3; d++){
        if(d==dim || !cfg.active[d]) continue;
        int v = (sub>>bit)&1;
        if(d==_x_) bx=v; else if(d==_y_) by=v; else bz=v;
        bit++;
    }
}

//Coarse -> fine ghost injection.
//
//The fine block's cells are half as wide as its coarse neighbour's and it
//covers only one half of that neighbour in each transverse direction, so both
//indices have to be mapped: two fine cells share one coarse cell along the
//normal, and the transverse offset depends on which half (`sub`) this block
//occupies. Reading the coarse neighbour at the fine block's own index -- as
//this did before -- samples a cell up to half a block away.
//
//The transverse ghost corners are filled from this one neighbour as a
//fallback; forest_exchange_fv_same overwrites every corner a same-level
//neighbour actually owns.
static void fv_inject_coarser(FV_Solution U, FV_Solution coarse, int dim, int side, int sub){
    int ngh = nGH_rt[dim];
    int N = (dim==_x_ ? U.Nx : (dim==_y_ ? U.Ny : U.Nz));
    int Nc = (dim==_x_ ? coarse.Nx : (dim==_y_ ? coarse.Ny : coarse.Nz));
    int nvar = U.n_var;
    int Nx = (dim==_x_ ? ngh : U.Nx);
    int Ny = (dim==_y_ ? ngh : U.Ny);
    int Nz = (dim==_z_ ? ngh : U.Nz);
    int bx, by, bz;
    fv_sub_bits(sub, dim, bx, by, bz);
    int gx = nGH_rt[_x_], gy = nGH_rt[_y_], gz = nGH_rt[_z_];
    int ax = U.Nx - 2*gx, ay = U.Ny - 2*gy, az = U.Nz - 2*gz;
    int cx = coarse.Nx-1, cy = coarse.Ny-1, cz = coarse.Nz-1;
    fv_for_cells(Nz,Ny,Nx, KOKKOS_LAMBDA(int k, int j, int i){
        //Transverse: fine active offset a maps to coarse active offset
        //(b*Na + a)/2, b being the half of the coarse block this one covers.
        int ci = (dim==_x_) ? i : fv_clamp(gx + fv_fdiv2(bx*ax + (i-gx)), cx);
        int cj = (dim==_y_) ? j : fv_clamp(gy + fv_fdiv2(by*ay + (j-gy)), cy);
        int ck = (dim==_z_) ? k : fv_clamp(gz + fv_fdiv2(bz*az + (k-gz)), cz);
        int l = (dim==_x_ ? i : (dim==_y_ ? j : k));
        //Normal: the coarse cell holding this ghost, counted off the interface
        //(two fine cells deep per coarse cell).
        int cl = (side==0) ? (Nc-ngh) - ((ngh-l)+1)/2
                           : (ngh-1) + ((l+2)/2);
        for(int var=0; var<nvar; var++){
        int Nid[3], Nidc[3];
        fv_indices(Nidc,ck,cj,ci,cl,dim);
        double v = coarse.Vector(var,Nidc[_z_],Nidc[_y_],Nidc[_x_]);
        fv_indices(Nid,k,j,i,(side==0?l:N-ngh+l),dim);
        U.Vector(FV_INDICES) = v;
        }
    });
}

//Fine -> coarse ghost fill.
//
//One coarse ghost cell covers two fine cells along the normal and two more
//across each transverse direction, all inside the single fine neighbour that
//owns that transverse half. So the value is a volume average over 2^ndim fine
//cells -- not, as this did before, an average of every fine neighbour sampled
//at the coarse block's own index, which reads each of them at the wrong place.
//
//`take_max` swaps the average for a maximum, which is what the MOOD cascade
//index needs: a demotion on either side of a level jump must be seen by both.
static void fv_from_finer(FV_Solution U, FV_Solution f0, FV_Solution f1,
                          FV_Solution f2, FV_Solution f3, int nf, int dim,
                          int side, bool take_max){
    int ngh = nGH_rt[dim];
    int nvar = U.n_var;
    int Nx = (dim==_x_ ? ngh : U.Nx);
    int Ny = (dim==_y_ ? ngh : U.Ny);
    int Nz = (dim==_z_ ? ngh : U.Nz);
    int gx = nGH_rt[_x_], gy = nGH_rt[_y_], gz = nGH_rt[_z_];
    int ax = U.Nx - 2*gx, ay = U.Ny - 2*gy, az = U.Nz - 2*gz;
    //Fine blocks carry the same cell counts as the coarse one.
    int hx = U.Nx-1, hy = U.Ny-1, hz = U.Nz-1;
    int Nf = (dim==_x_ ? U.Nx : (dim==_y_ ? U.Ny : U.Nz));
    //cfg is a host global, so the activity flags have to ride into the kernel
    //as plain locals.
    int actx = cfg.active[_x_], acty = cfg.active[_y_], actz = cfg.active[_z_];
    //Every direction contributes two fine cells except the inactive ones.
    int ox = actx ? 2 : 1;
    int oy = acty ? 2 : 1;
    int oz = actz ? 2 : 1;
    fv_for_cells(Nz,Ny,Nx, KOKKOS_LAMBDA(int k, int j, int i){
        //Transverse: which half owns this cell, and where it lands inside it.
        int bx=0, by=0, bz=0;
        int fx=i, fy=j, fz=k;
        if(dim!=_x_ && actx){ int a=i-gx; bx = (2*a>=ax); fx = gx + 2*a - bx*ax; }
        if(dim!=_y_ && acty){ int a=j-gy; by = (2*a>=ay); fy = gy + 2*a - by*ay; }
        if(dim!=_z_ && actz){ int a=k-gz; bz = (2*a>=az); fz = gz + 2*a - bz*az; }
        int sub=0, bit=0;
        for(int d=0; d<3; d++){
            int act = (d==_x_?actx:(d==_y_?acty:actz));
            if(d==dim || !act) continue;
            sub |= (d==_x_?bx:(d==_y_?by:bz))<<bit;
            bit++;
        }
        if(sub >= nf) sub = 0;
        //Normal: the two fine cells this coarse ghost spans.
        int l = (dim==_x_ ? i : (dim==_y_ ? j : k));
        int nbase = (side==0) ? (Nf-ngh) - 2*(ngh-l) : ngh + 2*l;
        if(dim==_x_)      fx = nbase;
        else if(dim==_y_) fy = nbase;
        else              fz = nbase;
        for(int var=0; var<nvar; var++){
        double acc = take_max ? -1e300 : 0.0;
        int cnt = 0;
        for(int dz=0; dz<oz; dz++)
        for(int dy=0; dy<oy; dy++)
        for(int dx=0; dx<ox; dx++){
            int px = fv_clamp(fx+dx, hx);
            int py = fv_clamp(fy+dy, hy);
            int pz = fv_clamp(fz+dz, hz);
            double v;
            if     (sub==0) v = f0.Vector(var,pz,py,px);
            else if(sub==1) v = f1.Vector(var,pz,py,px);
            else if(sub==2) v = f2.Vector(var,pz,py,px);
            else            v = f3.Vector(var,pz,py,px);
            if(take_max) acc = max(acc, v); else acc += v;
            cnt++;
        }
        int Nid[3];
        fv_indices(Nid,k,j,i,(side==0?l:Nf-ngh+l),dim);
        U.Vector(FV_INDICES) = take_max ? acc : acc/max(cnt,1);
        }
    });
}

static void fv_restrict_finer(FV_Solution U, FV_Solution& f0, FV_Solution& f1,
                              FV_Solution& f2, FV_Solution& f3, int nf, int dim, int side){
    fv_from_finer(U, f0, f1, f2, f3, nf, dim, side, false);
}

//Same-level copies only. Run after every direction has been exchanged, this
//fills each transverse ghost corner from the neighbour that actually owns it,
//which the coarse-fine operators deliberately skip.
template<typename Block>
void forest_exchange_fv_same(BlockForest& forest, std::vector<Block>& blocks,
                             FV_Solution Block::*member, int dim){
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

template<typename Block>
void forest_exchange_fv(BlockForest& forest, std::vector<Block>& blocks,
                        FV_Solution Block::*member, int dim){
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
        //Physical boundaries, as gather_all_fp does for the flux points.
        for(int ib : g.bc_ib)
            apply_domain_bc_fv(blocks[ib].*member, dim, side, nGH_rt[dim]);
    }
}

//Like forest_exchange_fv but coarse-fine takes the max (MOOD cascade index:
//a demotion on either side of a level jump must be visible to both).
static void fv_max_finer(FV_Solution U, FV_Solution& f0, FV_Solution& f1,
                         FV_Solution& f2, FV_Solution& f3, int nf, int dim, int side){
    fv_from_finer(U, f0, f1, f2, f3, nf, dim, side, true);
}

template<typename Block>
void forest_exchange_fv_max(BlockForest& forest, std::vector<Block>& blocks,
                            FV_Solution Block::*member, int dim){
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
            fv_max_finer(U, f0, f1, f2, f3, nf, dim, side);
        }
    }
}

template void forest_exchange_fp<Hydro_ader>(BlockForest&, std::vector<Hydro_ader>&, int);
template void forest_exchange_fp<MHD_ader>(BlockForest&, std::vector<MHD_ader>&, int);
template void forest_exchange_sd<Hydro_ader>(BlockForest&, std::vector<Hydro_ader>&,
                                             SD_Solution Hydro_ader::*, int, bool);
template void forest_exchange_sd<MHD_ader>(BlockForest&, std::vector<MHD_ader>&,
                                           SD_Solution MHD_ader::*, int, bool);
template void forest_sync_face_B<MHD_ader>(BlockForest&, std::vector<MHD_ader>&,
                                           SD_Solution MHD_ader::*, int);
template void correct_coarse_fine_flux<Hydro_ader>(BlockForest&, std::vector<Hydro_ader>&, int);
template void correct_coarse_fine_flux<MHD_ader>(BlockForest&, std::vector<MHD_ader>&, int);
template void correct_coarse_fine_emf<Hydro_ader>(BlockForest&, std::vector<Hydro_ader>&, int);
template void correct_coarse_fine_emf<MHD_ader>(BlockForest&, std::vector<MHD_ader>&, int);
template void symmetrize_same_level_fv_flux<Hydro_ader>(BlockForest&, std::vector<Hydro_ader>&, int);
template void symmetrize_same_level_fv_flux<MHD_ader>(BlockForest&, std::vector<MHD_ader>&, int);
template void correct_coarse_fine_fv_flux<Hydro_ader>(BlockForest&, std::vector<Hydro_ader>&, int);
template void correct_coarse_fine_fv_flux<MHD_ader>(BlockForest&, std::vector<MHD_ader>&, int);
template void forest_exchange_fv<Hydro_ader>(BlockForest&, std::vector<Hydro_ader>&,
                                             FV_Solution Hydro_ader::*, int);
template void forest_exchange_fv<MHD_ader>(BlockForest&, std::vector<MHD_ader>&,
                                           FV_Solution MHD_ader::*, int);
template void forest_exchange_fv_same<Hydro_ader>(BlockForest&, std::vector<Hydro_ader>&,
                                                  FV_Solution Hydro_ader::*, int);
template void forest_exchange_fv_same<MHD_ader>(BlockForest&, std::vector<MHD_ader>&,
                                                FV_Solution MHD_ader::*, int);
template void forest_exchange_fv_max<Hydro_ader>(BlockForest&, std::vector<Hydro_ader>&,
                                                 FV_Solution Hydro_ader::*, int);
template void forest_exchange_fv_max<MHD_ader>(BlockForest&, std::vector<MHD_ader>&,
                                               FV_Solution MHD_ader::*, int);

//======================================================================
// Receiver-driven flux-point ghost gather (the replacement exchange).
//
// One kernel per (dim, side), ranging over (transaction, face cell) with the
// transaction index as the leading kernel axis -- the block is just another
// index, so the launch count is constant in the block count and the only
// price of more meshblocks is the surface-to-volume ratio.
//
// The loop is driven by the *receiver's* ghost cells, so every ghost is
// written exactly once by construction. That is the property the previous
// push-style exchanges lacked, and the reason their corner handling needed a
// second same-level pass to paint over a "fallback" value.
//
// A transaction names the receiving block and its source. Mixed levels will
// add a relation code and a sub-face index here, changing only which operator
// the kernel applies (copy / prolongate / overlap-restrict), not the loop
// structure; same-level is the degenerate case where the operator is a copy.
//======================================================================
void gather_fp_same(SD_Solution U, IntVector recv, IntVector send,
                    int ntr, int dim, int side){
    if(ntr <= 0) return;
    int N = (dim==_x_ ? U.Nx : (dim==_y_ ? U.Ny : U.Nz));
    int n = (dim==_x_ ? U.nx : (dim==_y_ ? U.ny : U.nz));
    int nader = U.n_ader, nvar = U.n_var;
    //Collapse the normal axis: exactly one thread per destination cell.
    int Nx = (dim==_x_ ? 1 : U.Nx);
    int Ny = (dim==_y_ ? 1 : U.Ny);
    int Nz = (dim==_z_ ? 1 : U.Nz);
    int px = (dim==_x_ ? 1 : U.nx);
    int py = (dim==_y_ ? 1 : U.ny);
    int pz = (dim==_z_ ? 1 : U.nz);
    //Destination is my ghost element; source is the neighbour's last (side 0)
    //or first (side 1) active element, at the flux point on the shared face.
    const int de = (side==0 ? 0   : N-1);
    const int dp = (side==0 ? n-1 : 0  );
    const int se = (side==0 ? N-2 : 1  );
    const int sp = (side==0 ? n-1 : 0  );
    sd_for_cells_b(ntr,Nz,Ny,Nx,pz,py,px,
        KOKKOS_LAMBDA(int b,int k,int j,int i,int kk,int jj,int ii){
        const int rb = recv(b)*nader;
        const int sb = send(b)*nader;
        int Nid[3], nid[3];
        for(int t_id=0; t_id<nader; t_id++)
        for(int var=0; var<nvar; var++){
            amr_indices(Nid,nid,k,j,i,kk,jj,ii,se,sp,dim);
            double v = U.Vector(sb+t_id,var,Nid[_z_],Nid[_y_],Nid[_x_],
                                            nid[_z_],nid[_y_],nid[_x_]);
            amr_indices(Nid,nid,k,j,i,kk,jj,ii,de,dp,dim);
            U.Vector(rb+t_id,var,Nid[_z_],Nid[_y_],Nid[_x_],
                                 nid[_z_],nid[_y_],nid[_x_]) = v;
        }
    }, "gather_fp_same");
}

//Unpack a sub-face index into one transverse half per active direction.
KOKKOS_INLINE_FUNCTION
void sub_halves(int sub, int dim, int actx, int acty, int actz,
                int& cx, int& cy, int& cz){
    cx=0; cy=0; cz=0;
    int bit=0;
    for(int d=0; d<3; d++){
        int act = (d==_x_?actx:(d==_y_?acty:actz));
        if(d==dim || !act) continue;
        int v = (sub>>bit)&1;
        if(d==_x_) cx=v; else if(d==_y_) cy=v; else cz=v;
        bit++;
    }
}

//COARSER: the receiver is fine and reads a coarse neighbour. Its whole ghost
//face is the transverse prolongation of the sub-face of the coarse trace that
//it covers -- the same mapping prolongate_face_coarser uses, with the normal
//index pinned to my ghost slot and the neighbour's facing active slot.
void gather_fp_coarser(SD_Solution U, IntVector recv, IntVector send, IntVector subv,
                       int ntr, int dim, int side, Matrix P){
    if(ntr <= 0) return;
    int N = (dim==_x_ ? U.Nx : (dim==_y_ ? U.Ny : U.Nz));
    int n = (dim==_x_ ? U.nx : (dim==_y_ ? U.ny : U.nz));
    int nader=U.n_ader, nvar=U.n_var;
    int Nx=U.Nx, Ny=U.Ny, Nz=U.Nz, nx=U.nx, ny=U.ny, nz=U.nz;
    int actx=cfg.active[_x_], acty=cfg.active[_y_], actz=cfg.active[_z_];
    bool tx = actx && dim!=_x_, ty = acty && dim!=_y_, tz = actz && dim!=_z_;
    int NBx=Nx-2*NGHx, NBy=Ny-2*NGHy, NBz=Nz-2*NGHz;
    const int de = (side==0 ? 0 : N-1), dp = (side==0 ? n-1 : 0);
    const int se = (side==0 ? N-2 : 1), sp = (side==0 ? n-1 : 0);
    //Transverse span of the receiver's face; the normal axis is collapsed.
    int bz0 = tz?NGHz:0, bz1 = tz?Nz-NGHz:1;
    int by0 = ty?NGHy:0, by1 = ty?Ny-NGHy:1;
    int bx0 = tx?NGHx:0, bx1 = tx?Nx-NGHx:1;
    int pnz = tz?nz:1, pny = ty?ny:1, pnx = tx?nx:1;
    GHOST_LOCALS;
    sd_for_cells_b(ntr, bz1-bz0, by1-by0, bx1-bx0, pnz, pny, pnx,
        KOKKOS_LAMBDA(int b,int kk_,int jj_,int ii_,int kp,int jp,int ip){
        int k = kk_+bz0, j = jj_+by0, i = ii_+bx0;
        int cxh,cyh,czh; sub_halves(subv(b),dim,actx,acty,actz,cxh,cyh,czh);
        int gx = tx ? cxh*NBx + (i-ghx) : 0;
        int gy = ty ? cyh*NBy + (j-ghy) : 0;
        int gz = tz ? czh*NBz + (k-ghz) : 0;
        int cex = tx ? ghx+gx/2 : 0, sx = tx ? gx%2 : 0;
        int cey = ty ? ghy+gy/2 : 0, sy = ty ? gy%2 : 0;
        int cez = tz ? ghz+gz/2 : 0, sz = tz ? gz%2 : 0;
        const int rb = recv(b)*nader, sb = send(b)*nader;
        int Nid[3], nid[3];
        for(int t_id=0; t_id<nader; t_id++)
        for(int var=0; var<nvar; var++){
            double u=0;
            for(int nn=0; nn<(tz?nz:1); nn++)
            for(int mm=0; mm<(ty?ny:1); mm++)
            for(int ll=0; ll<(tx?nx:1); ll++){
                //source: neighbour's facing active element, coarse transverse cell
                amr_indices(Nid,nid, tz?cez:k, ty?cey:j, tx?cex:i,
                                     tz?nn:kp, ty?mm:jp, tx?ll:ip, se, sp, dim);
                double s = U.Vector(sb+t_id,var,Nid[_z_],Nid[_y_],Nid[_x_],
                                                nid[_z_],nid[_y_],nid[_x_]);
                if(tx) s *= P(sx*nx+ip,ll);
                if(ty) s *= P(sy*ny+jp,mm);
                if(tz) s *= P(sz*nz+kp,nn);
                u += s;
            }
            amr_indices(Nid,nid,k,j,i,kp,jp,ip,de,dp,dim);
            U.Vector(rb+t_id,var,Nid[_z_],Nid[_y_],Nid[_x_],
                                 nid[_z_],nid[_y_],nid[_x_]) = u;
        }
    }, "gather_fp_coarser");
}

//FINER: the receiver is coarse and each fine neighbour supplies one quadrant
//of its ghost face, overlap-restricted (amr_RF). One transaction per fine
//neighbour, and the quadrants partition the face, so every ghost cell is
//still written exactly once.
void gather_fp_finer(SD_Solution U, IntVector recv, IntVector send, IntVector subv,
                     int ntr, int dim, int side, Matrix R){
    if(ntr <= 0) return;
    int N = (dim==_x_ ? U.Nx : (dim==_y_ ? U.Ny : U.Nz));
    int n = (dim==_x_ ? U.nx : (dim==_y_ ? U.ny : U.nz));
    int nader=U.n_ader, nvar=U.n_var;
    int Nx=U.Nx, Ny=U.Ny, Nz=U.Nz, nx=U.nx, ny=U.ny, nz=U.nz;
    int actx=cfg.active[_x_], acty=cfg.active[_y_], actz=cfg.active[_z_];
    bool tx = actx && dim!=_x_, ty = acty && dim!=_y_, tz = actz && dim!=_z_;
    int NBx=Nx-2*NGHx, NBy=Ny-2*NGHy, NBz=Nz-2*NGHz;
    const int de = (side==0 ? 0 : N-1), dp = (side==0 ? n-1 : 0);
    const int se = (side==0 ? N-2 : 1), sp = (side==0 ? n-1 : 0);
    //Each transaction covers half the face per transverse direction.
    int hz = tz?NBz/2:1, hy = ty?NBy/2:1, hx = tx?NBx/2:1;
    int pnz = tz?nz:1, pny = ty?ny:1, pnx = tx?nx:1;
    GHOST_LOCALS;
    sd_for_cells_b(ntr, hz, hy, hx, pnz, pny, pnx,
        KOKKOS_LAMBDA(int b,int kk_,int jj_,int ii_,int kp,int jp,int ip){
        int cxh,cyh,czh; sub_halves(subv(b),dim,actx,acty,actz,cxh,cyh,czh);
        int i = tx ? ghx + cxh*(NBx/2) + ii_ : 0;
        int j = ty ? ghy + cyh*(NBy/2) + jj_ : 0;
        int k = tz ? ghz + czh*(NBz/2) + kk_ : 0;
        int fx = tx ? ghx+2*(i-ghx)-cxh*NBx : 0;
        int fy = ty ? ghy+2*(j-ghy)-cyh*NBy : 0;
        int fz = tz ? ghz+2*(k-ghz)-czh*NBz : 0;
        const int rb = recv(b)*nader, sb = send(b)*nader;
        int Nid[3], nid[3];
        for(int t_id=0; t_id<nader; t_id++)
        for(int var=0; var<nvar; var++){
            double u=0;
            for(int nn=0; nn<(tz?2*nz:1); nn++)
            for(int mm=0; mm<(ty?2*ny:1); mm++)
            for(int ll=0; ll<(tx?2*nx:1); ll++){
                amr_indices(Nid,nid, tz?fz+nn/nz:k, ty?fy+mm/ny:j, tx?fx+ll/nx:i,
                                     tz?nn%nz:kp,   ty?mm%ny:jp,   tx?ll%nx:ip,
                                     se, sp, dim);
                double s = U.Vector(sb+t_id,var,Nid[_z_],Nid[_y_],Nid[_x_],
                                                nid[_z_],nid[_y_],nid[_x_]);
                if(tx) s *= R(ip,ll);
                if(ty) s *= R(jp,mm);
                if(tz) s *= R(kp,nn);
                u += s;
            }
            amr_indices(Nid,nid,k,j,i,kp,jp,ip,de,dp,dim);
            U.Vector(rb+t_id,var,Nid[_z_],Nid[_y_],Nid[_x_],
                                 nid[_z_],nid[_y_],nid[_x_]) = u;
        }
    }, "gather_fp_finer");
}

//The earlier "not correct yet" note blamed this kernel; it was wrong. The
//normal-direction pinning here matches set_interface_flux for both sides and
//the transverse mapping matches restrict_face_overlap_sp. The defect was in the
//caller, which chose amr_RF_fp over amr_RF whenever the transverse point counts
//happened to coincide -- the reference always restricts with amr_RF. With that
//fixed the two paths are bit-identical.
//
//Conservative flux correction at a coarse-fine face, off the fine->coarse
//table: the coarse block's interface flux becomes the overlap-weighted average
//of the fine fluxes covering it, or the coarse cell fails to lose exactly what
//the fine cells gain.
//
//This replaces make_scratch_like + restrict_face_overlap_sp +
//set_interface_flux per (coarse block, side, sub). Folding the three into one
//kernel matters for more than the launch count: the scratch buffer was a fresh
//allocation inside the correction loop, and an allocation fences the device.
//
//The Riemann solver stores each common flux twice, on the ghost side of the
//interface and on the interior side, and the update reads the interior copy --
//so both slots are written, exactly as set_interface_flux did.
void correct_cf_flux_b(SD_Solution U, IntVector recv, IntVector send, IntVector subv,
                       int ntr, int dim, int side, Matrix R){
    if(ntr <= 0) return;
    const int N = (dim==_x_ ? U.Nx : (dim==_y_ ? U.Ny : U.Nz));
    const int n = (dim==_x_ ? U.nx : (dim==_y_ ? U.ny : U.nz));
    const int nader=U.n_ader, nvar=U.n_var;
    const int nx=U.nx, ny=U.ny, nz=U.nz;
    const int actx=cfg.active[_x_], acty=cfg.active[_y_], actz=cfg.active[_z_];
    const bool tx = actx && dim!=_x_, ty = acty && dim!=_y_, tz = actz && dim!=_z_;
    const int NBx=U.Nx-2*NGHx, NBy=U.Ny-2*NGHy, NBz=U.Nz-2*NGHz;
    //Source: the fine neighbour's facing interface. Destinations: my own two
    //copies of that interface flux.
    const int se = (side==0 ? N-2 : 1  ), sp = (side==0 ? n-1 : 0  );
    const int d1 = (side==0 ? 0   : N-1), p1 = (side==0 ? n-1 : 0  );
    const int d2 = (side==0 ? 1   : N-2), p2 = (side==0 ? 0   : n-1);
    //Each transaction covers the half of the coarse face its neighbour spans.
    const int Hx = tx ? NBx/2 : 1, Hy = ty ? NBy/2 : 1, Hz = tz ? NBz/2 : 1;
    const int Px = tx ? nx : (dim==_x_ ? 1 : nx);
    const int Py = ty ? ny : (dim==_y_ ? 1 : ny);
    const int Pz = tz ? nz : (dim==_z_ ? 1 : nz);
    GHOST_LOCALS;
    sd_for_cells_b(ntr,Hz,Hy,Hx,Pz,Py,Px,
        KOKKOS_LAMBDA(int b,int kq,int jq,int iq,int kk,int jj,int ii){
        int cx,cy,cz; fv_sub_bits_d(subv(b),dim,actx,acty,actz,cx,cy,cz);
        const int i = tx ? ghx + cx*(NBx/2) + iq : (dim==_x_ ? 0 : ghx+iq);
        const int j = ty ? ghy + cy*(NBy/2) + jq : (dim==_y_ ? 0 : ghy+jq);
        const int k = tz ? ghz + cz*(NBz/2) + kq : (dim==_z_ ? 0 : ghz+kq);
        //Transversally each coarse element gathers the two fine elements that
        //overlap it, from the neighbour covering that half.
        const int fx = tx ? ghx+2*(i-ghx)-cx*NBx : i;
        const int fy = ty ? ghy+2*(j-ghy)-cy*NBy : j;
        const int fz = tz ? ghz+2*(k-ghz)-cz*NBz : k;
        const int rb = recv(b)*nader, sb = send(b)*nader;
        int Nid[3], nid[3];
        for(int t_id=0; t_id<nader; t_id++)
        for(int var=0; var<nvar; var++){
            double u=0;
            for(int nn=0; nn<(tz ? 2*nz:1); nn++)
            for(int mm=0; mm<(ty ? 2*ny:1); mm++)
            for(int ll=0; ll<(tx ? 2*nx:1); ll++){
                amr_indices(Nid,nid, tz?fz+nn/nz:k, ty?fy+mm/ny:j, tx?fx+ll/nx:i,
                                     tz?nn%nz:kk,   ty?mm%ny:jj,   tx?ll%nx:ii,
                                     se, sp, dim);
                double s = U.Vector(sb+t_id,var,Nid[_z_],Nid[_y_],Nid[_x_],
                                                nid[_z_],nid[_y_],nid[_x_]);
                if(tx) s *= R(ii,ll);
                if(ty) s *= R(jj,mm);
                if(tz) s *= R(kk,nn);
                u += s;
            }
            amr_indices(Nid,nid,k,j,i,kk,jj,ii,d1,p1,dim);
            U.Vector(rb+t_id,var,Nid[_z_],Nid[_y_],Nid[_x_],
                                 nid[_z_],nid[_y_],nid[_x_]) = u;
            amr_indices(Nid,nid,k,j,i,kk,jj,ii,d2,p2,dim);
            U.Vector(rb+t_id,var,Nid[_z_],Nid[_y_],Nid[_x_],
                                 nid[_z_],nid[_y_],nid[_x_]) = u;
        }
    }, "correct_cf_flux_b");
}

//======================================================================
// FV ghost fill, same transaction tables as the flux-point gathers above.
//
// The FV arrays are (nvar, Nz, Ny, Nx) with no ADER or sub-point axis, and
// the halo is a slab `ngh` cells deep rather than a single face layer, so
// the kernel ranges over (transaction, slab cell) and the block folds into
// the leading axis as b*nvar -- the FV pack's counterpart of the SD pack's
// b*nader.
//
// Same-level is a straight copy: my ghost slab is the neighbour's last
// (side 0) or first (side 1) `ngh` active cells. Receiver-driven, so every
// ghost cell is written exactly once.
//======================================================================
void gather_fv_same(FV_Solution U, IntVector recv, IntVector send,
                    int ntr, int dim, int side, int ngh){
    if(ntr <= 0) return;
    const int N = (dim==_x_ ? U.Nx : (dim==_y_ ? U.Ny : U.Nz));
    const int nvar = U.n_var;
    //Collapse the normal axis to the slab depth: one thread per ghost cell.
    const int Nx = (dim==_x_ ? ngh : U.Nx);
    const int Ny = (dim==_y_ ? ngh : U.Ny);
    const int Nz = (dim==_z_ ? ngh : U.Nz);
    fv_for_cells_b(ntr,Nz,Ny,Nx, KOKKOS_LAMBDA(int b,int k,int j,int i){
        const int rb = recv(b)*nvar;
        const int sb = send(b)*nvar;
        const int l = (dim==_x_ ? i : (dim==_y_ ? j : k));
        const int sl = (side==0 ? N-2*ngh+l : ngh+l);   //source: active cell
        const int dl = (side==0 ? l         : N-ngh+l); //dest: my ghost cell
        int Nsrc[3], Ndst[3];
        fv_indices(Nsrc,k,j,i,sl,dim);
        fv_indices(Ndst,k,j,i,dl,dim);
        for(int var=0; var<nvar; var++)
            U.Vector(rb+var,Ndst[_z_],Ndst[_y_],Ndst[_x_]) =
            U.Vector(sb+var,Nsrc[_z_],Nsrc[_y_],Nsrc[_x_]);
    }, "gather_fv_same");
}

//COARSER: the receiver is fine and reads a coarse neighbour. Same mapping as
//fv_inject_coarser -- two fine cells share one coarse cell along the normal,
//and the transverse offset depends on which half of the neighbour this block
//covers -- with the block folded into the leading axis. One transaction per
//receiving face, so the whole ghost slab including the transverse corners is
//written exactly once (those corners are a fallback that the same-level pass
//overwrites wherever a same-level neighbour owns them).
void gather_fv_coarser(FV_Solution U, IntVector recv, IntVector send, IntVector subv,
                       int ntr, int dim, int side, int ngh){
    if(ntr <= 0) return;
    const int N = (dim==_x_ ? U.Nx : (dim==_y_ ? U.Ny : U.Nz));
    const int nvar = U.n_var;
    const int Nx = (dim==_x_ ? ngh : U.Nx);
    const int Ny = (dim==_y_ ? ngh : U.Ny);
    const int Nz = (dim==_z_ ? ngh : U.Nz);
    const int gx = nGH_rt[_x_], gy = nGH_rt[_y_], gz = nGH_rt[_z_];
    const int ax = U.Nx-2*gx, ay = U.Ny-2*gy, az = U.Nz-2*gz;
    //Blocks in a pack share extents, so the coarse neighbour's are these.
    const int cxm = U.Nx-1, cym = U.Ny-1, czm = U.Nz-1;
    const int actx=cfg.active[_x_], acty=cfg.active[_y_], actz=cfg.active[_z_];
    fv_for_cells_b(ntr,Nz,Ny,Nx, KOKKOS_LAMBDA(int b,int k,int j,int i){
        int bx,by,bz; fv_sub_bits_d(subv(b),dim,actx,acty,actz,bx,by,bz);
        //Transverse: fine active offset a maps to coarse active offset
        //(b*Na + a)/2, b being the half of the coarse block this one covers.
        const int ci = (dim==_x_) ? i : fv_clamp(gx + fv_fdiv2(bx*ax + (i-gx)), cxm);
        const int cj = (dim==_y_) ? j : fv_clamp(gy + fv_fdiv2(by*ay + (j-gy)), cym);
        const int ck = (dim==_z_) ? k : fv_clamp(gz + fv_fdiv2(bz*az + (k-gz)), czm);
        const int l  = (dim==_x_ ? i : (dim==_y_ ? j : k));
        //Normal: the coarse cell holding this ghost, counted off the interface.
        const int cl = (side==0) ? (N-ngh) - ((ngh-l)+1)/2
                                 : (ngh-1) + ((l+2)/2);
        const int rb = recv(b)*nvar, sb = send(b)*nvar;
        int Nid[3], Nidc[3];
        fv_indices(Nidc,ck,cj,ci,cl,dim);
        fv_indices(Nid,k,j,i,(side==0?l:N-ngh+l),dim);
        for(int var=0; var<nvar; var++)
            U.Vector(rb+var,Nid[_z_],Nid[_y_],Nid[_x_]) =
            U.Vector(sb+var,Nidc[_z_],Nidc[_y_],Nidc[_x_]);
    }, "gather_fv_coarser");
}

//FINER: the receiver is coarse and each fine neighbour supplies one quadrant of
//its ghost slab, volume-averaged over the 2^ndim fine cells each coarse ghost
//covers. One transaction per fine neighbour, and the quadrants partition the
//active transverse range, so every ghost there is written exactly once.
//
//take_max swaps the average for a maximum, which is what the MOOD cascade index
//needs: a demotion on either side of a level jump must be seen by both.
//
//The quadrants extend over the transverse ghost cells on their own side, so
//they still partition the whole slab and every ghost -- corners included -- is
//written exactly once. That has to match how the per-block operator assigns
//them: its `b = (2*(i-gx) >= Na)` test puts the low-side ghosts in half 0 and
//the high-side ghosts in half 1. Leaving the corners out instead is not
//harmless, because apply_blending reads a full box neighbourhood including the
//diagonals: at two refinement levels that produced O(0.1) corner errors and the
//run diverged.
void gather_fv_finer(FV_Solution U, IntVector recv, IntVector send, IntVector subv,
                     int ntr, int dim, int side, int ngh, bool take_max){
    if(ntr <= 0) return;
    const int nvar = U.n_var;
    const int gx = nGH_rt[_x_], gy = nGH_rt[_y_], gz = nGH_rt[_z_];
    const int ax = U.Nx-2*gx, ay = U.Ny-2*gy, az = U.Nz-2*gz;
    const int hxm = U.Nx-1, hym = U.Ny-1, hzm = U.Nz-1;
    const int Nf = (dim==_x_ ? U.Nx : (dim==_y_ ? U.Ny : U.Nz));
    const int actx=cfg.active[_x_], acty=cfg.active[_y_], actz=cfg.active[_z_];
    const bool tx = actx && dim!=_x_, ty = acty && dim!=_y_, tz = actz && dim!=_z_;
    //Half the full transverse extent, which is gx + ax/2 on either side.
    const int Qx = (dim==_x_) ? ngh : (tx ? gx + ax/2 : 1);
    const int Qy = (dim==_y_) ? ngh : (ty ? gy + ay/2 : 1);
    const int Qz = (dim==_z_) ? ngh : (tz ? gz + az/2 : 1);
    //Every direction contributes two fine cells except the inactive ones.
    const int ox = actx ? 2 : 1, oy = acty ? 2 : 1, oz = actz ? 2 : 1;
    fv_for_cells_b(ntr,Qz,Qy,Qx, KOKKOS_LAMBDA(int b,int kq,int jq,int iq){
        int bx,by,bz; fv_sub_bits_d(subv(b),dim,actx,acty,actz,bx,by,bz);
        //Receiver cell: half 0 owns [0, gx+ax/2), half 1 owns [gx+ax/2, Nx).
        const int i = (dim==_x_) ? iq : (tx ? bx*(gx+ax/2) + iq : iq);
        const int j = (dim==_y_) ? jq : (ty ? by*(gy+ay/2) + jq : jq);
        const int k = (dim==_z_) ? kq : (tz ? bz*(gz+az/2) + kq : kq);
        //Fine cell base, same mapping as the per-block operator: the active
        //offset doubles inside the half that owns it. Negative offsets (low
        //ghosts) are intended; fv_clamp below keeps the reads in range.
        int fx = tx ? gx + 2*(i-gx) - bx*ax : i;
        int fy = ty ? gy + 2*(j-gy) - by*ay : j;
        int fz = tz ? gz + 2*(k-gz) - bz*az : k;
        const int l = (dim==_x_ ? i : (dim==_y_ ? j : k));
        const int nbase = (side==0) ? (Nf-ngh) - 2*(ngh-l) : ngh + 2*l;
        if(dim==_x_)      fx = nbase;
        else if(dim==_y_) fy = nbase;
        else              fz = nbase;
        const int rb = recv(b)*nvar, sb = send(b)*nvar;
        int Nid[3];
        fv_indices(Nid,k,j,i,(side==0?l:Nf-ngh+l),dim);
        for(int var=0; var<nvar; var++){
            double acc = take_max ? -1e300 : 0.0;
            int cnt = 0;
            for(int dz=0; dz<oz; dz++)
            for(int dy=0; dy<oy; dy++)
            for(int dx=0; dx<ox; dx++){
                const int px = fv_clamp(fx+dx, hxm);
                const int py = fv_clamp(fy+dy, hym);
                const int pz = fv_clamp(fz+dz, hzm);
                const double v = U.Vector(sb+var,pz,py,px);
                if(take_max) acc = max(acc, v); else acc += v;
                cnt++;
            }
            U.Vector(rb+var,Nid[_z_],Nid[_y_],Nid[_x_]) =
                take_max ? acc : acc/max(cnt,1);
        }
    }, "gather_fv_finer");
}
