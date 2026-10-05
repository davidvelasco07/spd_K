#include "spd_k.hpp"
#include "amr_criteria.hpp"
#include <set>
#include <type_traits>

//SPD_LOHNER_INTERIOR=1: the interior-only Lohner score (commit 6339d79 and
//before), kept as the A/B reference. It skips the block-edge elements, so a
//block of two elements per side scores exactly 0 and is never tagged.
bool lohner_edge_on(){
    static const bool v = getenv("SPD_LOHNER_INTERIOR") == nullptr;
    return v;
}

//The Lohner score is an undivided second difference between neighbouring
//ELEMENTS at the same sub-point (CLAUDE.md 7a4). An interior element has both
//neighbours inside the block. A block-edge element has one of them across the
//face, where only one point of the ghost element is valid: the point next to
//the shared face, the one layer the SD field exchange writes
//(copy_face_to_ghost / gather_fp_same / apply_domain_bc_fp). So the edge element
//gets exactly one stencil, at its face-adjacent sub-point: s = n-1 across the
//low face, s = 0 across the high face. A one-element block has no stencil.
//side: -1 interior, 0 across the low face, 1 across the high face.
KOKKOS_INLINE_FUNCTION bool lohner_stencil(bool lo, bool hi, int s, int n, int& side){
    side = -1;
    if(lo && hi) return false;
    if(lo){ if(s != n-1) return false; side = 0; }
    else if(hi){ if(s != 0) return false; side = 1; }
    return true;
}
//a: distance from the edge sub-point to the ghost point, in units of this
//block's element width (lohner_edge_ratios). At the same level and at a wall it
//is 1 and this is the plain second difference. At a level jump the ghost holds
//the neighbour's own face-adjacent point -- prolongated or restricted ALONG the
//face, not across it -- so it sits at the neighbour's spacing, and the plain
//difference would read a smooth gradient as curvature (a first-derivative term
//of 0.07h at p=3, 0.5h at p=0). The divided form with the true spacing, scaled
//back to the element width, is exact zero on a linear profile.
KOKKOS_INLINE_FUNCTION double lohner_d2(double v0, double v1, double v2, int side, double a){
    if(side < 0 || a == 1.0) return fabs(v0 - 2.0*v1 + v2);
    if(side == 0) return fabs(2.0*((v2 - v1) - (v1 - v0)/a)/(1.0 + a));
    return fabs(2.0*((v2 - v1)/a - (v1 - v0))/(1.0 + a));
}

double lohner_score(SD_Solution W, int var, const double* edge){
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
            int side;
            if(dim==_x_){
                if(!lohner_stencil(i-1<NGHx, i+1>=W.Nx-NGHx, ii, W.nx, side)) continue;
            } else if(dim==_y_){
                if(!lohner_stencil(j-1<NGHy, j+1>=W.Ny-NGHy, jj, W.ny, side)) continue;
            } else {
                if(!lohner_stencil(k-1<NGHz, k+1>=W.Nz-NGHz, kk, W.nz, side)) continue;
            }
            double a = 1.0;
            if(side >= 0){ a = edge ? edge[2*dim+side] : 0.0; if(a <= 0.0) continue; }
            if(dim==_x_){
                v0=A(t,var,k,j,i-1,kk,jj,ii);
                v1=A(t,var,k,j,i,kk,jj,ii);
                v2=A(t,var,k,j,i+1,kk,jj,ii);
            } else if(dim==_y_){
                v0=A(t,var,k,j-1,i,kk,jj,ii);
                v1=A(t,var,k,j,i,kk,jj,ii);
                v2=A(t,var,k,j+1,i,kk,jj,ii);
            } else {
                v0=A(t,var,k-1,j,i,kk,jj,ii);
                v1=A(t,var,k,j,i,kk,jj,ii);
                v2=A(t,var,k+1,j,i,kk,jj,ii);
            }
            g2 = std::max(g2, lohner_d2(v0, v1, v2, side, a));
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
    //A passive-scalar row (amr/lohner_vars = scalar) is scored by the second difference itself: a concentration has a
    //unit scale, and dividing by a block mean that tends to zero would refine on the round-off tail of the scalar.
    if(var >= NVAR) return g2;
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

//All blocks' refinement scores in ONE launch, one thread per block.
//
//What this removes is not arithmetic, it is a host round trip. Every score
//function above opens with W.copy() -- a synchronous device->host copy of that
//block's whole primitive array -- and then loops on the CPU, and a regrid pays
//it once per block for the refine test plus once per sibling for the derefine
//test: ~1000 copies at 352 leaves. Fenced, `amr/tag` measured 0.207 s of
//`amr/adapt`'s 0.207 s -- the ENTIRE regrid cost -- which was 21% of the fenced
//mixed-level MHD advance and 92.8% of hydro's whole fenced total.
//
//ONE THREAD PER BLOCK, not one per cell, and that is deliberate. Each thread
//walks the host loop's exact nested order, so the order-dependent sum in
//lohner's denominator comes out BIT-IDENTICAL rather than merely close. A
//cell-parallel reduction would change the last bits of `den`, which is precisely
//how this could go wrong invisibly: a block whose score sits on the threshold
//flips its tag, the mesh diverges from the reference, and every dump after it
//differs for a reason that looks like a bug in the transfer. The parallelism
//that matters here is the block count anyway -- this runs once per regrid.
//
//`which` mirrors cfg.amr_criterion: 1 pressure gradient, 3 shear, anything else
//the density Lohner indicator. Criterion 2 (trouble fraction, an FV array) and 4
//(the |B| Lohner, which needs a forest-wide field scale first) keep their
//per-block paths; see tag_blocks_impl.
//
//NGHx/NGHy/NGHz expand to NGH_rt, a HOST global: nvcc rejects them in device
//code, and the host build compiles them happily, so this only fails under CUDA.
//They are captured as ngx/ngy/ngz -- not gx/gy/gz, which are the shear
//indicator's own flattened cell indices below (naming them alike shadowed the
//loop variables, and the A/B caught it as three lanes going non-identical).
void block_scores_b(SD_Solution W, int which, int var, Vector out, Vector edge){
    const int nb = W.nb;
    if(nb <= 0) return;
    const int Nx=W.Nx, Ny=W.Ny, Nz=W.Nz, px=W.nx, py=W.ny, pz=W.nz;
    const int nader=W.n_ader;
    const int ngx=NGHx, ngy=NGHy, ngz=NGHz;
    const int actx=cfg.active[_x_], acty=cfg.active[_y_], actz=cfg.active[_z_];
    const int vp=_p_, vvx=_vx_, vvy=_vy_;
    auto A = W.Vector;
    Kokkos::parallel_for("block_scores_b", flat_range(0,flat_total(nb)),
        KOKKOS_LAMBDA(const unsigned bb){
        const int boff = (int)bb*nader;
        double res = 0.0;
        if(which==1){
            //pressure_gradient_score
            double pmax=-1e300, pmin=1e300;
            for(int k=ngz;k<Nz-ngz;k++)
            for(int j=ngy;j<Ny-ngy;j++)
            for(int i=ngx;i<Nx-ngx;i++)
            for(int kk=0;kk<pz;kk++)
            for(int jj=0;jj<py;jj++)
            for(int ii=0;ii<px;ii++){
                const double q = A(boff,vp,k,j,i,kk,jj,ii);
                pmax = q>pmax ? q : pmax;
                pmin = q<pmin ? q : pmin;
            }
            res = (pmax-pmin)/(pmax>1e-12 ? pmax : 1e-12);
        } else if(which==3){
            //shear_score, on the flattened (element,point) index per direction
            const int Gx=(Nx-2*ngx)*px, Gy=(Ny-2*ngy)*py, Gz=(Nz-2*ngz)*pz;
            double g = 0.0;
            if(actx && Gx>=3){
                for(int gz=0;gz<Gz;gz++)
                for(int gy=0;gy<Gy;gy++)
                for(int gx=1;gx<Gx-1;gx++){
                    const double vp1 = A(boff,vvy,ngz+gz/pz,ngy+gy/py,ngx+(gx+1)/px,
                                              gz%pz,gy%py,(gx+1)%px);
                    const double vm1 = A(boff,vvy,ngz+gz/pz,ngy+gy/py,ngx+(gx-1)/px,
                                              gz%pz,gy%py,(gx-1)%px);
                    const double d = 0.5*fabs(vp1-vm1);
                    g = d>g ? d : g;
                }
            }
            if(acty && Gy>=3){
                for(int gz=0;gz<Gz;gz++)
                for(int gy=1;gy<Gy-1;gy++)
                for(int gx=0;gx<Gx;gx++){
                    const double vp1 = A(boff,vvx,ngz+gz/pz,ngy+(gy+1)/py,ngx+gx/px,
                                              gz%pz,(gy+1)%py,gx%px);
                    const double vm1 = A(boff,vvx,ngz+gz/pz,ngy+(gy-1)/py,ngx+gx/px,
                                              gz%pz,(gy-1)%py,gx%px);
                    const double d = 0.5*fabs(vp1-vm1);
                    g = d>g ? d : g;
                }
            }
            res = g;
        } else {
            //lohner_score(var)
            double g2 = 0.0;
            for(int dim=0; dim<3; dim++){
                if(!(dim==_x_ ? actx : (dim==_y_ ? acty : actz))) continue;
                if((dim==_x_ ? Nx : (dim==_y_ ? Ny : Nz)) < 3) continue;
                for(int k=ngz;k<Nz-ngz;k++)
                for(int j=ngy;j<Ny-ngy;j++)
                for(int i=ngx;i<Nx-ngx;i++)
                for(int kk=0;kk<pz;kk++)
                for(int jj=0;jj<py;jj++)
                for(int ii=0;ii<px;ii++){
                    double v0,v1,v2;
                    int side;
                    if(dim==_x_){
                        if(!lohner_stencil(i-1<ngx, i+1>=Nx-ngx, ii, px, side)) continue;
                    } else if(dim==_y_){
                        if(!lohner_stencil(j-1<ngy, j+1>=Ny-ngy, jj, py, side)) continue;
                    } else {
                        if(!lohner_stencil(k-1<ngz, k+1>=Nz-ngz, kk, pz, side)) continue;
                    }
                    double a = 1.0;
                    if(side >= 0){ a = edge(6*(int)bb + 2*dim + side); if(a <= 0.0) continue; }
                    if(dim==_x_){
                        v0=A(boff,var,k,j,i-1,kk,jj,ii);
                        v1=A(boff,var,k,j,i,kk,jj,ii);
                        v2=A(boff,var,k,j,i+1,kk,jj,ii);
                    } else if(dim==_y_){
                        v0=A(boff,var,k,j-1,i,kk,jj,ii);
                        v1=A(boff,var,k,j,i,kk,jj,ii);
                        v2=A(boff,var,k,j+1,i,kk,jj,ii);
                    } else {
                        v0=A(boff,var,k-1,j,i,kk,jj,ii);
                        v1=A(boff,var,k,j,i,kk,jj,ii);
                        v2=A(boff,var,k+1,j,i,kk,jj,ii);
                    }
                    const double d = lohner_d2(v0, v1, v2, side, a);
                    g2 = d>g2 ? d : g2;
                }
            }
            //The denominator is a SUM, so it is accumulated in the host loop's
            //order inside this one thread. That is the whole reason this kernel
            //is not cell-parallel.
            double den = 0.0;
            int cnt = 0;
            for(int k=ngz;k<Nz-ngz;k++)
            for(int j=ngy;j<Ny-ngy;j++)
            for(int i=ngx;i<Nx-ngx;i++)
            for(int kk=0;kk<pz;kk++)
            for(int jj=0;jj<py;jj++)
            for(int ii=0;ii<px;ii++){
                den += fabs(A(boff,var,k,j,i,kk,jj,ii));
                cnt++;
            }
            den = den/(cnt>1 ? cnt : 1) + 1e-12;
            res = var >= NVAR ? g2 : g2/den;   //a scalar row is not normalised, see lohner_score
        }
        out(bb) = res;
    });
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
        //Count troubled CELLS, not the sum of flag values: the flag carries a
        //severity (1 = NAD, 2 = PAD) when mhd/mood_pad_first_order is on, and a
        //sum would then weight PAD cells double in a quantity called a fraction.
        s += (T(0,k,j,i) > 0) ? 1.0 : 0.0;
        cnt++;
    }
    return s/std::max(cnt,1);
}

double trouble_fraction(Hydro_ader& blk){ return trouble_fraction_impl(blk); }
double trouble_fraction(MHD_ader& blk){ return trouble_fraction_impl(blk); }

//The score -> tag decision, in one place. The batched and per-block paths
//differ only in HOW the score is computed, never in where the cut sits, so the
//thresholds live here and both callers use them: a threshold duplicated between
//the two would make an A/B mismatch look like an arithmetic difference.
//Every threshold criterion reads amr/refine_threshold and
//amr/derefine_threshold; main.cpp fills in the criterion's own pair when the
//deck sets neither (pressure 0.03/0.0075, lohner 0.5/0.0125, shear 0.1/0.05).
//Pressure and lohner used to hard-code those numbers here and silently ignore
//the input -- three different refine_threshold values gave md5-identical block
//maps on the Sod lane, which is how it was found.
static bool refine_from_score(int, double s){
    return s > cfg.amr_refine_threshold;
}
static bool derefine_from_score(int, double s){
    return s < cfg.amr_derefine_threshold;
}
//One block's score, on the host, with the per-block device->host copy each of
//these functions opens with. This is the reference path (SPD_NO_SCORE_BATCH=1).
static double block_score_host(int criterion, SD_Solution W, const double* edge){
    switch(criterion){
        case 1:  return pressure_gradient_score(W);
        case 3:  return shear_score(W);
        default: {
            //amr/lohner_vars: the larger of the per-variable scores
            double s = 0.0;
            if(cfg.amr_lohner_vars & 1) s = std::max(s, lohner_score(W, _d_, edge));
            if(cfg.amr_lohner_vars & 2) s = std::max(s, lohner_score(W, _p_, edge));
            if(cfg.amr_lohner_vars & 4) s = std::max(s, lohner_score(W, NVAR, edge));   //first passive scalar
            return s;
        }
    }
}

template<typename Block>
static void block_primitives(Block& blk){
    if constexpr (std::is_same_v<Block, Hydro_ader>)
        compute_primitives(blk.U_sp, blk.W_sp);
    else
        mhd_compute_primitives(blk.U_sp, blk.W_sp);
}

template<typename Block>
static bool refine_flag(int criterion, Block& blk, const double* edge){
    block_primitives(blk);
    //criterion 4 on MHD is handled in tag_blocks_impl, which needs every
    //block's score at once to set the field scale and to derefine
    if(criterion==2) return trouble_fraction(blk) > 0.01;
    return refine_from_score(criterion, block_score_host(criterion, blk.W_sp, edge));
}

template<typename Block>
static bool derefine_flag(int criterion, const std::vector<Block*>& sibs,
                          const std::vector<const double*>& edges){
    if(criterion==2){
        double f = 0.0;
        for(auto* b : sibs) f = std::max(f, trouble_fraction(*b));
        return f < 0.001;
    }
    double s = 0.0;
    for(size_t q=0; q<sibs.size(); q++)
        s = std::max(s, block_score_host(criterion, sibs[q]->W_sp, edges[q]));
    return derefine_from_score(criterion, s);
}

//Per block and face, index 6*ib + 2*dim + side: the distance, in units of the
//block's element width, from the edge element's face-adjacent sub-point to the
//ghost point across that face (see lohner_d2). The ghost point is the
//neighbour's own face-adjacent point. Same level: x_sp[0] + (1 - x_sp[p]) = 1
//for symmetric nodes. Physical boundary: the mirrored or copied interior point,
//1. Coarser neighbour: its point sits twice as far from the face; finer: half
//as far. 0 means no stencil across that face.
static std::vector<double> lohner_edge_ratios(const BlockForest& forest){
    const int nb = forest.Nblocks();
    const bool on = lohner_edge_on() && !(nb <= 1 && forest.max_level() == 0);
    //(Exchange_sd_field fills nothing on a single unrefined block.)
    std::vector<double> r(6*nb, on ? 1.0 : 0.0);
    if(!on) return r;
    const double xl = amr_sp_last, xf = amr_sp_first;
    const double co[2] = {2.0 - xl, 1.0 + xf};
    const double fi[2] = {0.5*(1.0 + xl), 1.0 - 0.5*xf};
    for(int dim=0; dim<3; dim++){
        if(!forest.active[dim]) continue;
        for(int side=0; side<2; side++){
            if(!forest.same_jb[dim][side].empty()) continue;   //one level on this face
            const FaceGroups& g = forest.face_groups[dim][side];
            for(int ib : g.co_ib) r[6*ib+2*dim+side] = co[side];
            for(int ib : g.fi_ib) r[6*ib+2*dim+side] = fi[side];
        }
    }
    return r;
}

//SPD_NO_SCORE_BATCH=1 restores the per-block score path, which is the A/B
//reference the batched one has to reproduce bitwise.
static bool no_score_batch(){
    static const bool v = getenv("SPD_NO_SCORE_BATCH") != nullptr;
    return v;
}

template<typename Block>
static void tag_blocks_impl(BlockForest& forest, std::vector<Block>& blocks,
                            SD_Solution U_pack, SD_Solution W_pack,
                            std::vector<int>& to_refine,
                            std::vector<std::vector<int>>& to_derefine,
                            int max_level, int criterion){
    to_refine.clear();
    to_derefine.clear();
    //Every block's score in two launches (primitives, then scores) instead of a
    //per-block kernel plus a per-block device->host copy inside
    //refine_flag/derefine_flag. Criterion 2 reads an FV array and criterion 4
    //needs a forest-wide field scale first, so both keep the per-block path.
    std::vector<double> score;
    const std::vector<double> edge = lohner_edge_ratios(forest);
    if(!no_score_batch() && criterion!=2 && !(std::is_same_v<Block,MHD_ader> && criterion==4)
       && W_pack.Vector.size()>0 && W_pack.nb == forest.Nblocks()){
        const int nb = forest.Nblocks();
        Block::primitives_b(U_pack, W_pack);
        Vector sc("block_scores", nb);
        Vector ev("lohner_edge", edge.size());
        {
            auto evh = Kokkos::create_mirror_view(ev);
            for(size_t q=0; q<edge.size(); q++) evh(q) = edge[q];
            Kokkos::deep_copy(ev, evh);
        }
        //The Lohner criterion scores each variable of amr/lohner_vars in its own
        //launch and keeps the larger; density alone is one launch, as before.
        const int lvars = criterion==0 ? cfg.amr_lohner_vars : 1;
        const int vars[3] = {_d_, _p_, NVAR};   //bit 4: the first passive scalar (hydro only, main.cpp)
        score.assign(nb, 0.0);
        for(int q=0; q<3; q++){
            if(!(lvars & (1<<q))) continue;
            block_scores_b(W_pack, criterion, vars[q], sc, ev);
            auto h = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), sc);
            for(int ib=0; ib<nb; ib++) score[ib] = std::max(score[ib], h(ib));
        }
    }
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
        //The host reference (SPD_NO_SCORE_BATCH=1) scores blocks it never tests
        //for refinement: the derefine pass scores sibling groups AT max_level,
        //which refine_flag skips. Their W_sp then keeps the primitives of the
        //last stage, including a ghost point layer that the pre-tag exchange has
        //since rewritten in U_sp -- the Lohner edge stencil read it and this path
        //parted from the batched one on the p=0 implosion lane (2 of 8 files).
        //The interior values are current either way, so the interior-only score
        //is unaffected. Reference path at regrid cadence: a per-block loop is
        //allowed here (rule 1).
        if(score.empty())
            for(int ib=0; ib<forest.Nblocks(); ib++) block_primitives(blocks[ib]);
        for(int ib=0; ib<forest.Nblocks(); ib++){
            if(max_level>=0 && forest.blocks[ib].level >= max_level) continue;
            const bool tag = score.empty()
                           ? refine_flag(criterion, blocks[ib], &edge[6*ib])
                           : refine_from_score(criterion, score[ib]);
            if(tag) to_refine.push_back(ib);
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
        if(!score.empty()){
            double s = 0.0;
            for(int ib : kv.second) s = std::max(s, score[ib]);
            if(derefine_from_score(criterion, s)) to_derefine.push_back(kv.second);
            continue;
        }
        std::vector<Block*> sibs;
        std::vector<const double*> sib_edges;
        for(int ib : kv.second){ sibs.push_back(&blocks[ib]); sib_edges.push_back(&edge[6*ib]); }
        if(derefine_flag(criterion, sibs, sib_edges))
            to_derefine.push_back(kv.second);
    }
}

void tag_blocks(BlockForest& forest, std::vector<Hydro_ader>& blocks,
                SD_Solution U_pack, SD_Solution W_pack,
                std::vector<int>& to_refine,
                std::vector<std::vector<int>>& to_derefine,
                int max_level, int criterion){
    tag_blocks_impl(forest, blocks, U_pack, W_pack, to_refine, to_derefine,
                    max_level, criterion);
}

void tag_blocks(BlockForest& forest, std::vector<MHD_ader>& blocks,
                SD_Solution U_pack, SD_Solution W_pack,
                std::vector<int>& to_refine,
                std::vector<std::vector<int>>& to_derefine,
                int max_level, int criterion){
    tag_blocks_impl(forest, blocks, U_pack, W_pack, to_refine, to_derefine,
                    max_level, criterion);
}
