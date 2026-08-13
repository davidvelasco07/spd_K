#include "spd_k.hpp"
#include "amr_criteria.hpp"

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

double trouble_fraction(Hydro_ader& blk){
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

static bool refine_flag(int criterion, Hydro_ader& blk){
    compute_primitives(blk.U_sp, blk.W_sp);
    switch(criterion){
        case 1: return pressure_gradient_score(blk.W_sp) > 0.03;
        case 2: return trouble_fraction(blk) > 0.01;
        //Thresholds are the paper's: refine above 0.01, derefine below 0.005.
        case 3: return shear_score(blk.W_sp) > 0.01;
        default: return lohner_score(blk.W_sp, _d_) > 0.5;
    }
}

static bool derefine_flag(int criterion, const std::vector<Hydro_ader*>& sibs){
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
            return g < 0.005;
        }
        default:{
            double s = 0.0;
            for(auto* b : sibs) s = std::max(s, lohner_score(b->W_sp, _d_));
            return s < 0.05*0.25;
        }
    }
}

void tag_blocks(BlockForest& forest, std::vector<Hydro_ader>& blocks,
                std::vector<int>& to_refine,
                std::vector<std::vector<int>>& to_derefine,
                int max_level, int criterion){
    to_refine.clear();
    to_derefine.clear();
    for(int ib=0; ib<forest.Nblocks(); ib++){
        if(max_level>=0 && forest.blocks[ib].level >= max_level) continue;
        if(refine_flag(criterion, blocks[ib]))
            to_refine.push_back(ib);
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
    for(auto& kv : groups){
        if((int)kv.second.size() != n_sib) continue;
        std::vector<Hydro_ader*> sibs;
        for(int ib : kv.second) sibs.push_back(&blocks[ib]);
        if(derefine_flag(criterion, sibs))
            to_derefine.push_back(kv.second);
    }
}
