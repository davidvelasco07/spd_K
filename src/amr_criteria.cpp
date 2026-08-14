#include "spd_k.hpp"
#include "amr_criteria.hpp"
#include <set>
#include <type_traits>

double lohner_score(SD_Solution W, int var){
    W.copy();
    double g2 = 0.0;
    int t = 0;
    #ifdef KOKKOS_ENABLE_CUDA
    auto& A = W.Vector_h;
    #else
    auto& A = W.Vector;
    #endif
    for(int dim=0; dim<3; dim++){
        if(!cfg.active[dim]) continue;
        if((dim==_x_ ? W.Nx : (dim==_y_ ? W.Ny : W.Nz)) < 3) continue;
        for(int k=NGHz; k<W.Nz-NGHz; k++)
        for(int j=NGHy; j<W.Ny-NGHy; j++)
        for(int i=NGHx; i<W.Nx-NGHx; i++)
        for(int kk=0; kk<W.nz; kk++)
        for(int jj=0; jj<W.ny; jj++)
        for(int ii=0; ii<W.nx; ii++){
            double v0,v1,v2;
            if(dim==_x_){
                if(i-1<NGHx || i+1>=W.Nx-NGHx) continue;
                v0=A(t,var,k,j,i-1,kk,jj,ii);
                v1=A(t,var,k,j,i,kk,jj,ii);
                v2=A(t,var,k,j,i+1,kk,jj,ii);
            } else if(dim==_y_){
                if(j-1<NGHy || j+1>=W.Ny-NGHy) continue;
                v0=A(t,var,k,j-1,i,kk,jj,ii);
                v1=A(t,var,k,j,i,kk,jj,ii);
                v2=A(t,var,k,j+1,i,kk,jj,ii);
            } else {
                if(k-1<NGHz || k+1>=W.Nz-NGHz) continue;
                v0=A(t,var,k-1,j,i,kk,jj,ii);
                v1=A(t,var,k,j,i,kk,jj,ii);
                v2=A(t,var,k+1,j,i,kk,jj,ii);
            }
            g2 = std::max(g2, std::abs(v0 - 2.0*v1 + v2));
        }
    }
    double den = 0.0;
    int cnt = 0;
    for(int k=NGHz; k<W.Nz-NGHz; k++)
    for(int j=NGHy; j<W.Ny-NGHy; j++)
    for(int i=NGHx; i<W.Nx-NGHx; i++)
    for(int kk=0; kk<W.nz; kk++)
    for(int jj=0; jj<W.ny; jj++)
    for(int ii=0; ii<W.nx; ii++){
        den += std::abs(A(t,var,k,j,i,kk,jj,ii));
        cnt++;
    }
    den = den/std::max(cnt,1) + 1e-12;
    return g2/den;
}

double pressure_gradient_score(SD_Solution W){
    W.copy();
    int t = 0, var = _p_;
    #ifdef KOKKOS_ENABLE_CUDA
    auto& A = W.Vector_h;
    #else
    auto& A = W.Vector;
    #endif
    double pmax = -1e300, pmin = 1e300;
    for(int k=NGHz; k<W.Nz-NGHz; k++)
    for(int j=NGHy; j<W.Ny-NGHy; j++)
    for(int i=NGHx; i<W.Nx-NGHx; i++)
    for(int kk=0; kk<W.nz; kk++)
    for(int jj=0; jj<W.ny; jj++)
    for(int ii=0; ii<W.nx; ii++){
        double p = A(t,var,k,j,i,kk,jj,ii);
        pmax = std::max(pmax, p);
        pmin = std::min(pmin, p);
    }
    return (pmax-pmin)/std::max(pmax, 1e-12);
}

//Velocity-shear indicator, Athena++ (Stone et al. 2020) eq. 27:
//    g = h * max(d vy/dx, d vx/dy)
//A centred difference between neighbouring solution points already carries the
//factor h, so g is just the largest velocity change across one cell anywhere in
//the block and no grid spacing is needed. Element and point indices are
//flattened into one running index per direction so the stencil crosses element
//boundaries; the spacing is only piecewise uniform (flux-point spacing within
//an element), which is fine for a refinement indicator.
double shear_score(SD_Solution W){
    W.copy();
    #ifdef KOKKOS_ENABLE_CUDA
    auto& A = W.Vector_h;
    #else
    auto& A = W.Vector;
    #endif
    int Gx = (W.Nx-2*NGHx)*W.nx;
    int Gy = (W.Ny-2*NGHy)*W.ny;
    int Gz = (W.Nz-2*NGHz)*W.nz;
    auto at = [&](int var, int gz, int gy, int gx){
        return A(0,var, NGHz+gz/W.nz, NGHy+gy/W.ny, NGHx+gx/W.nx,
                        gz%W.nz,      gy%W.ny,      gx%W.nx);
    };
    double g = 0.0;
    if(cfg.active[_x_] && Gx >= 3){
        for(int gz=0; gz<Gz; gz++)
        for(int gy=0; gy<Gy; gy++)
        for(int gx=1; gx<Gx-1; gx++)
            g = std::max(g, 0.5*std::abs(at(_vy_,gz,gy,gx+1) - at(_vy_,gz,gy,gx-1)));
    }
    if(cfg.active[_y_] && Gy >= 3){
        for(int gz=0; gz<Gz; gz++)
        for(int gy=1; gy<Gy-1; gy++)
        for(int gx=0; gx<Gx; gx++)
            g = std::max(g, 0.5*std::abs(at(_vx_,gz,gy+1,gx) - at(_vx_,gz,gy-1,gx)));
    }
    return g;
}

//Element means of B^2 = Bx^2+By^2+Bz^2 over the block. B^2 rather than |B|
//because |B| has a cusp wherever the field passes through zero -- on the
//Orszag-Tang lattice of nulls, say -- and a curvature indicator would then keep
//refining those points at every resolution.
static std::vector<double> bfield_element_means(SD_Solution W){
    W.copy();
    int t = 0;
    #ifdef KOKKOS_ENABLE_CUDA
    auto& A = W.Vector_h;
    #else
    auto& A = W.Vector;
    #endif
    std::vector<double> m((size_t)W.Nz*W.Ny*W.Nx, 0.0);
    double inv = 1.0/(W.nz*W.ny*W.nx);
    for(int k=0; k<W.Nz; k++)
    for(int j=0; j<W.Ny; j++)
    for(int i=0; i<W.Nx; i++){
        double s = 0.0;
        for(int kk=0; kk<W.nz; kk++)
        for(int jj=0; jj<W.ny; jj++)
        for(int ii=0; ii<W.nx; ii++){
            double bx = A(t,_mbx_,k,j,i,kk,jj,ii);
            double by = A(t,_mby_,k,j,i,kk,jj,ii);
            double bz = A(t,_mbz_,k,j,i,kk,jj,ii);
            s += bx*bx + by*by + bz*bz;
        }
        m[((size_t)k*W.Ny + j)*W.Nx + i] = s*inv;
    }
    return m;
}

double bfield_mean(SD_Solution W){
    auto m = bfield_element_means(W);
    double s = 0.0;
    int cnt = 0;
    for(int k=NGHz; k<W.Nz-NGHz; k++)
    for(int j=NGHy; j<W.Ny-NGHy; j++)
    for(int i=NGHx; i<W.Nx-NGHx; i++){
        s += m[((size_t)k*W.Ny + j)*W.Nx + i];
        cnt++;
    }
    return s/std::max(cnt,1);
}

//Löhner indicator on B^2, in its filtered form
//
//    L = |v2 - 2 v1 + v0| / (|v2 - v1| + |v1 - v0| + eps (|v2| + 2|v1| + |v0|))
//
//scored as the largest L over the block. It tracks Orszag-Tang structure from
//t = 0, where density and pressure are uniform and a hydro indicator sees
//nothing. Being dimensionless, L falls off once a feature is resolved, so
//refinement stops instead of walking to a uniformly refined mesh.
//
//Two details are specific to this discretisation. The stencil runs over element
//*means*, not solution points: the SD representation is discontinuous across
//elements, so a point-to-point stencil measures the inter-element jump, which
//stays O(1) at every resolution and saturates the indicator everywhere. And the
//filter term carries a floor built from a forest-wide field strength bref,
//without which a field-free block (the exterior of a field loop) compares
//round-off against round-off and scores as though it were a discontinuity.
//
//Only stencils that stay inside the block are used, so the score never depends
//on how recently the ghost elements were exchanged.
double bfield_lohner_score(SD_Solution W, double bref){
    const double eps = 0.01;
    auto m = bfield_element_means(W);
    auto at = [&](int k, int j, int i){
        return m[((size_t)k*W.Ny + j)*W.Nx + i];
    };
    double g2 = 0.0;
    for(int dim=0; dim<3; dim++){
        if(!cfg.active[dim]) continue;
        int lo = (dim==_x_ ? NGHx+1 : (dim==_y_ ? NGHy+1 : NGHz+1));
        int hi = (dim==_x_ ? W.Nx-NGHx-1 : (dim==_y_ ? W.Ny-NGHy-1 : W.Nz-NGHz-1));
        if(hi <= lo) continue;
        for(int k=NGHz; k<W.Nz-NGHz; k++)
        for(int j=NGHy; j<W.Ny-NGHy; j++)
        for(int i=NGHx; i<W.Nx-NGHx; i++){
            int c = (dim==_x_ ? i : (dim==_y_ ? j : k));
            if(c < lo || c >= hi) continue;
            double v0,v1,v2;
            if(dim==_x_){        v0=at(k,j,i-1); v1=at(k,j,i); v2=at(k,j,i+1); }
            else if(dim==_y_){   v0=at(k,j-1,i); v1=at(k,j,i); v2=at(k,j+1,i); }
            else {               v0=at(k-1,j,i); v1=at(k,j,i); v2=at(k+1,j,i); }
            double den = std::abs(v2-v1) + std::abs(v1-v0)
                       + eps*(std::abs(v2) + 2.0*std::abs(v1) + std::abs(v0))
                       + 4.0*eps*bref;
            g2 = std::max(g2, std::abs(v0 - 2.0*v1 + v2)/den);
        }
    }
    return g2;
}

template<typename Block>
static double trouble_fraction_impl(Block& blk){
    if(!cfg.fallback) return 0.0;
    blk.troubles.copy();
    #ifdef KOKKOS_ENABLE_CUDA
    auto& T = blk.troubles.Vector_h;
    #else
    auto& T = blk.troubles.Vector;
    #endif
    double s = 0.0;
    int cnt = 0;
    for(int k=nGHz; k<blk.troubles.Nz-nGHz; k++)
    for(int j=nGHy; j<blk.troubles.Ny-nGHy; j++)
    for(int i=nGHx; i<blk.troubles.Nx-nGHx; i++){
        s += T(0,k,j,i);
        cnt++;
    }
    return s/std::max(cnt,1);
}

double trouble_fraction(Hydro_ader& blk){ return trouble_fraction_impl(blk); }
double trouble_fraction(MHD_ader& blk){ return trouble_fraction_impl(blk); }

template<typename Block>
static bool refine_flag(int criterion, Block& blk){
    if constexpr (std::is_same_v<Block, Hydro_ader>)
        compute_primitives(blk.U_sp, blk.W_sp);
    else
        mhd_compute_primitives(blk.U_sp, blk.W_sp);
    switch(criterion){
        case 1: return pressure_gradient_score(blk.W_sp) > 0.03;
        case 2: return trouble_fraction(blk) > 0.01;
        case 3: return shear_score(blk.W_sp) > cfg.amr_refine_threshold;
        //criterion 4 on MHD is handled in tag_blocks_impl, which needs every
        //block's score at once to set the field scale and to derefine
        default: return lohner_score(blk.W_sp, _d_) > 0.5;
    }
}

template<typename Block>
static bool derefine_flag(int criterion, const std::vector<Block*>& sibs){
    switch(criterion){
        case 1:{
            double dP = 0.0;
            for(auto* b : sibs) dP = std::max(dP, pressure_gradient_score(b->W_sp));
            return dP < 0.015*0.5;
        }
        case 2:{
            double f = 0.0;
            for(auto* b : sibs) f = std::max(f, trouble_fraction(*b));
            return f < 0.001;
        }
        case 3:{
            double g = 0.0;
            for(auto* b : sibs) g = std::max(g, shear_score(b->W_sp));
            return g < cfg.amr_derefine_threshold;
        }
        default:{
            double s = 0.0;
            for(auto* b : sibs) s = std::max(s, lohner_score(b->W_sp, _d_));
            return s < 0.05*0.25;
        }
    }
}

template<typename Block>
static void tag_blocks_impl(BlockForest& forest, std::vector<Block>& blocks,
                            std::vector<int>& to_refine,
                            std::vector<std::vector<int>>& to_derefine,
                            int max_level, int criterion){
    to_refine.clear();
    to_derefine.clear();
    //Scores of the |B| criterion, which needs every block's score in one place
    //so the derefine pass can reuse it. Empty for every other criterion, which
    //keeps its per-block refine_flag/derefine_flag test.
    std::vector<double> bscore;
    double bpeak = 0.0;
    if constexpr (std::is_same_v<Block, MHD_ader>){
        if(criterion==4){
            int nb = forest.Nblocks();
            bscore.assign(nb, 0.0);
            double bref = 0.0;
            for(int ib=0; ib<nb; ib++){
                mhd_compute_primitives(blocks[ib].U_sp, blocks[ib].W_sp);
                bref = std::max(bref, bfield_mean(blocks[ib].W_sp));
            }
            for(int ib=0; ib<nb; ib++){
                bscore[ib] = bfield_lohner_score(blocks[ib].W_sp, bref);
                bpeak = std::max(bpeak, bscore[ib]);
            }
            double vmax = 0.0;
            for(int ib=0; ib<nb; ib++){
                if(max_level>=0 && forest.blocks[ib].level >= max_level) continue;
                vmax = std::max(vmax, bscore[ib]);
            }
            for(int ib=0; ib<nb; ib++){
                if(max_level>=0 && forest.blocks[ib].level >= max_level) continue;
                if(bscore[ib] > cfg.amr_bfield_threshold &&
                   bscore[ib] >= cfg.amr_refine_frac*vmax - 1e-15)
                    to_refine.push_back(ib);
            }
            //Tuning a criterion is mostly a matter of seeing how separated the
            //scores are, which is invisible from the block counts alone.
            if(getenv("SPD_AMR_DEBUG")){
                std::vector<double> s = bscore;
                std::sort(s.begin(), s.end());
                std::cout<<"\n[amr] nb="<<nb<<" vmax="<<vmax
                         <<" med="<<s[nb/2]<<" p90="<<s[(9*nb)/10]
                         <<" min="<<s[0]<<" tagged="<<to_refine.size()<<std::endl;
            }
        }
    }
    if(bscore.empty()){
        for(int ib=0; ib<forest.Nblocks(); ib++){
            if(max_level>=0 && forest.blocks[ib].level >= max_level) continue;
            if(refine_flag(criterion, blocks[ib]))
                to_refine.push_back(ib);
        }
    }
    int n_sib = 1;
    for(int d=0; d<3; d++) if(forest.active[d]) n_sib *= 2;
    std::map<std::tuple<int,int,int,int>, std::vector<int>> groups;
    for(int ib=0; ib<forest.Nblocks(); ib++){
        const MeshBlock& b = forest.blocks[ib];
        if(b.level==0) continue;
        auto key = std::make_tuple(b.level, b.logical[0]/2, b.logical[1]/2, b.logical[2]/2);
        groups[key].push_back(ib);
    }
    std::set<int> refining(to_refine.begin(), to_refine.end());
    for(auto& kv : groups){
        if((int)kv.second.size() != n_sib) continue;
        //A group whose score straddles the refine threshold and the derefine cut
        //can be tagged both ways; refinement wins, since the group would be gone
        //by the time the derefine pass ran.
        bool clash = false;
        for(int ib : kv.second) if(refining.count(ib)) clash = true;
        if(clash) continue;
        if(!bscore.empty()){
            double s = 0.0;
            for(int ib : kv.second) s = std::max(s, bscore[ib]);
            //A fixed cut alone ratchets the mesh: the indicator saturates on
            //whatever grid-scale noise the scheme carries, so a block that once
            //refined keeps scoring above any absolute cut and the mesh walks to
            //uniform refinement. Releasing anything well below the current peak
            //score instead makes the fine region track the strongest feature and
            //bounds the block count. Off by default (derefine_frac = 0).
            if(s < 0.5*cfg.amr_bfield_threshold ||
               s < cfg.amr_derefine_frac*bpeak)
                to_derefine.push_back(kv.second);
            continue;
        }
        std::vector<Block*> sibs;
        for(int ib : kv.second) sibs.push_back(&blocks[ib]);
        if(derefine_flag(criterion, sibs))
            to_derefine.push_back(kv.second);
    }
}

void tag_blocks(BlockForest& forest, std::vector<Hydro_ader>& blocks,
                std::vector<int>& to_refine,
                std::vector<std::vector<int>>& to_derefine,
                int max_level, int criterion){
    tag_blocks_impl(forest, blocks, to_refine, to_derefine, max_level, criterion);
}

void tag_blocks(BlockForest& forest, std::vector<MHD_ader>& blocks,
                std::vector<int>& to_refine,
                std::vector<std::vector<int>>& to_derefine,
                int max_level, int criterion){
    tag_blocks_impl(forest, blocks, to_refine, to_derefine, max_level, criterion);
}
