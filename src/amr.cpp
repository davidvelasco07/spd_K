#include "spd_k.hpp"

//Prolongation and restriction between adjacent refinement levels, ported
//from the spd reference (spd/amr/transfer.py on the amr-modular branch).
//
//Block-based: prolongate_block takes one coarse block's active region and
//produces one of the 2^ndim fine children; restrict_block is the adjoint,
//writing the child's subregion of the coarse block. Child bits are
//(cx,cy,cz) in {0,1} (0 for inactive dimensions).
//
//The Lagrange matrices use the fine points
//    x_fine = concat(x_sp/2, x_sp/2 + 1/2)   (size 2(p+1))
//so restrict_block(prolongate_block(W)) recovers the coarse data to
//round-off for any polynomial data of degree <= p (all SD solutions).
//
//Both kernels require an even number of elements per block per active
//dimension, so that the two fine elements overlapping a coarse element
//always belong to the same child (the usual block-AMR constraint).

//P: (2(p+1), p+1) coarse solution points -> fine solution points
//R: (p+1, 2(p+1)) fine solution points -> coarse solution points
void transfer_matrices(Matrix P, Matrix R, double* x_sp, int p){
    int n = p+1;
    double* x_fine = malloc_host<double>(2*n);
    for(int i=0;i<n;i++){
        x_fine[i]   = 0.5*x_sp[i];
        x_fine[n+i] = 0.5*x_sp[i]+0.5;
    }
    Matrix_h P_h = Kokkos::create_mirror_view(P);
    Matrix_h R_h = Kokkos::create_mirror_view(R);
    lagrange_matrix(P_h, x_sp, x_fine, n, 2*n);
    lagrange_matrix(R_h, x_fine, x_sp, 2*n, n);
    Kokkos::deep_copy(P, P_h);
    Kokkos::deep_copy(R, R_h);
    free(x_fine);
}

//Copy the active region of block array B into the global array G at the
//element offset (ox,oy,oz) (block position in active elements). Used to
//stitch per-block data into a single rank-wide array for output.
void gather_block(SD_Solution B, SD_Solution G, int ox, int oy, int oz){
    int Nx=B.Nx-2*NGHx, Ny=B.Ny-2*NGHy, Nz=B.Nz-2*NGHz;
    int nx=B.nx, ny=B.ny, nz=B.nz;
    int nader=B.n_ader, nvar=B.n_var;
    GHOST_LOCALS;
    sd_for_box_cells(0,Nz,0,Ny,0,Nx,nz,ny,nx, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        for(int t_id=0; t_id<nader; t_id++){
        for(int var=0; var<nvar; var++){
        G.Vector(t_id,var,ghz+oz+k,ghy+oy+j,ghx+ox+i,kk,jj,ii) =
            B.Vector(t_id,var,ghz+k,ghy+j,ghx+i,kk,jj,ii);
        }}
    }, "gather_block");
}

//Fill the active region of fine child (cx,cy,cz) from the coarse block.
//Fine element f (block-local) covers the coarse element (c*NB+f)/2, taking
//its left/right half according to sub=(c*NB+f)%2; point values interpolate
//with rows [sub*n, (sub+1)*n) of P. Inactive dimensions pass through.
void prolongate_block(SD_Solution C, SD_Solution F, Matrix P, int cx, int cy, int cz){
    int Nx=F.Nx, Ny=F.Ny, Nz=F.Nz;
    int nx=F.nx, ny=F.ny, nz=F.nz;
    int nader=F.n_ader, nvar=F.n_var;
    bool ax=cfg.active[_x_], ay=cfg.active[_y_], az=cfg.active[_z_];
    int NBx=Nx-2*NGHx, NBy=Ny-2*NGHy, NBz=Nz-2*NGHz;
    GHOST_LOCALS;
    sd_for_box_cells(NGHz,Nz-NGHz,NGHy,Ny-NGHy,NGHx,Nx-NGHx,nz,ny,nx, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        //global fine element index and its coarse parent/sub-half per dim
        int gx = cx*NBx + (i-ghx);
        int gy = cy*NBy + (j-ghy);
        int gz = cz*NBz + (k-ghz);
        int cex = ax ? ghx+gx/2 : i;
        int cey = ay ? ghy+gy/2 : j;
        int cez = az ? ghz+gz/2 : k;
        int sx = ax ? gx%2 : 0;
        int sy = ay ? gy%2 : 0;
        int sz = az ? gz%2 : 0;
        for(int t_id=0; t_id<nader; t_id++){
        for(int var=0; var<nvar; var++){
            double u=0;
            for(int nn=0; nn<(az ? nz:1); nn++){
            for(int mm=0; mm<(ay ? ny:1); mm++){
            for(int ll=0; ll<(ax ? nx:1); ll++){
                double s = C.Vector(t_id,var,cez,cey,cex,
                                    az ? nn:kk, ay ? mm:jj, ax ? ll:ii);
                if(ax) s *= P(sx*nx+ii,ll);
                if(ay) s *= P(sy*ny+jj,mm);
                if(az) s *= P(sz*nz+kk,nn);
                u += s;
            }}}
            F.Vector(t_id,var,k,j,i,kk,jj,ii) = u;
        }}
    }, "prolongate_block");
}

//Admissibility of one conserved control-volume state, phrased exactly as
//pad_cell/PAD_criteria: density and pressure inside the shared PAD bounds.
//Comparisons are written so a NaN fails rather than passes.
KOKKOS_INLINE_FUNCTION
bool pad_admissible(const double* u, double gm){
    const double rho = u[_d_];
    if(!(rho >= rho_min) || !(rho <= rho_max)) return false;
    const double ekin = 0.5*(u[_vx_]*u[_vx_] + u[_vy_]*u[_vy_]
                           + u[_vz_]*u[_vz_]) / rho;
    const double pres = (u[_e_] - ekin)*(gm - 1.0);
    return (pres >= p_min) && (pres <= p_max);
}

//prolongate_block is an unlimited Lagrange interpolation, and it is applied to
//the CONSERVED variables independently. Across a discontinuity that manufactures
//new extrema: rho, rho*v and E each stay within their own interpolation error,
//but the pressure they imply, (E - |rho v|^2/2rho)(gamma-1), is a nonlinear
//combination and can come out negative even though every coarse value it came
//from was admissible. Measured on the sharp 10->0.1 blast, one regrid took the
//minimum pressure from +0.0999 to -1.067; nothing downstream repairs it,
//because the fallback limits the *update*, not the state it is handed.
//
//So limit the prolongation the way the cascade limits a flux: keep the
//high-order result wherever it is admissible, and drop an element that is not
//to its own mean.
//
//Both the mean and the admissibility test are taken on the CV (cell-average)
//representation, weighted by the fv_faces sub-cell widths -- exactly the
//quadrature fv_mass uses -- so the element's conserved content is unchanged
//and the transfer stays exactly conservative. Getting this wrong is silent:
//weighting the solution-point values instead (sp_to_cv is not the identity)
//drifts mass ~1e-5 per limited regrid, and the Gauss-Legendre wx vector is the
//ADER *temporal* rule while the spatial points are Chebyshev, which drifts too.
//The kernel works purely on the CV array; the caller maps the result back to
//the solution points. Only a *constant* element is the same in both
//representations -- a theta-blend is not -- so writing blended CV values
//straight into U_sp desynchronises them and drifts mass ~1e-5.
//
//Returns the number of elements limited.
int limit_prolongation(SD_Solution C,
                       Vector fx, Vector fy, Vector fz, double gm,
                       int* n_unfixable){
    const int Nx=C.Nx, Ny=C.Ny, Nz=C.Nz;
    const int nx=C.nx, ny=C.ny, nz=C.nz;
    const int nader=C.n_ader, nvar=C.n_var;
    const int qx=nx, qy=ny, qz=nz;
    const bool ay=cfg.active[_y_], az=cfg.active[_z_];
    const int Mz=Nz-2*NGHz, My=Ny-2*NGHy, Mx=Nx-2*NGHx;
    const int64_t total = (int64_t)nader*Mz*My*Mx;
    if(total <= 0) return 0;
    GHOST_LOCALS;
    SD_Vector V = C.Vector;      //cell averages: mean, test and conservation
    int n_lim = 0, n_bad_mean = 0;
    Kokkos::parallel_reduce("limit_prolongation",
        flat_range(0, flat_total(total)),
        KOKKOS_LAMBDA(const unsigned idx, int& lim, int& badmean){
            unsigned r = idx;
            //ghx/ghy/ghz are the GHOST_LOCALS copies: NGHx & co. are runtime
            //host globals and are not addressable from device code.
            const int i = ghx + (int)(r % Mx); r /= Mx;
            const int j = ghy + (int)(r % My); r /= My;
            const int k = ghz + (int)(r % Mz); r /= Mz;
            const int t = (int)r;
            double mean[NVAR];
            for(int var=0; var<nvar; var++) mean[var] = 0.0;
            double vol = 0.0;
            bool bad = false;
            for(int kk=0; kk<nz; kk++)
            for(int jj=0; jj<ny; jj++)
            for(int ii=0; ii<nx; ii++){
                double wq = fx(I+1)-fx(I);
                if(ay) wq *= fy(J+1)-fy(J);
                if(az) wq *= fz(K+1)-fz(K);
                vol += wq;
                double u[NVAR];
                for(int var=0; var<nvar; var++){
                    u[var] = V(t,var,k,j,i,kk,jj,ii);
                    mean[var] += wq*u[var];
                }
                //Same admissibility rule as pad_cell (PAD_criteria): the
                //bounds are the shared rho_min/rho_max/p_min/p_max, not a
                //hand-rolled positivity test, so the transfer and the MOOD
                //detector agree on what "admissible" means. Per control
                //volume, which is the granularity the cascade limits at.
                if(!pad_admissible(u, gm)) bad = true;
            }
            if(!bad || !(vol > 0.0)) return;
            for(int var=0; var<nvar; var++) mean[var] /= vol;
            //The element mean is a convex combination of this element's CVs,
            //so it is admissible whenever enough of them are. It is NOT
            //guaranteed: the donor value that always is, is the parent
            //element's mean, but that unit is 2^ndim fine elements wide.
            //Count the cases so the need for that escalation is measured
            //rather than assumed.
            if(!pad_admissible(mean, gm)) badmean++;
            //Shrink the element toward its own mean by the least amount that
            //makes every control volume admissible (Zhang & Shu). Detection is
            //per control volume -- theta is set by the worst one -- but the
            //correction redistributes instead of overwriting, so the element
            //mean, and with it the conserved content, is untouched. An
            //admissible CV is barely moved; only a blanket theta=0 would be the
            //full collapse. Overwriting the bad CVs alone is NOT an option: it
            //changes their integral with nothing to compensate, and costs
            //1.2e-4 of the total mass on this problem.
            double theta = 1.0;
            for(int kk=0; kk<nz; kk++)
            for(int jj=0; jj<ny; jj++)
            for(int ii=0; ii<nx; ii++){
                double u2[NVAR], ut[NVAR];
                for(int var=0; var<nvar; var++) u2[var]=V(t,var,k,j,i,kk,jj,ii);
                if(pad_admissible(u2, gm)) continue;
                //Largest admissible t in [0,1]; t=0 is the mean, which is
                //admissible whenever this element can be fixed at all.
                double lo=0.0, hi=1.0;
                for(int it=0; it<40; it++){
                    const double mid = 0.5*(lo+hi);
                    for(int var=0; var<nvar; var++)
                        ut[var] = mean[var] + mid*(u2[var]-mean[var]);
                    if(pad_admissible(ut, gm)) lo = mid; else hi = mid;
                }
                if(lo < theta) theta = lo;
            }
            for(int kk=0; kk<nz; kk++)
            for(int jj=0; jj<ny; jj++)
            for(int ii=0; ii<nx; ii++)
                for(int var=0; var<nvar; var++)
                    V(t,var,k,j,i,kk,jj,ii) =
                        mean[var] + theta*(V(t,var,k,j,i,kk,jj,ii)-mean[var]);
            lim++;
        }, Kokkos::Sum<int>(n_lim), Kokkos::Sum<int>(n_bad_mean));
    if(n_unfixable) *n_unfixable += n_bad_mean;
    return n_lim;
}

//Adjoint of prolongate_block: write the coarse subregion covered by fine
//child (cx,cy,cz). Coarse element c (block-local, within the child's half)
//gathers its two overlapping fine elements 2c+sub through columns
//[sub*n, (sub+1)*n) of R. Requires even NB per active dimension.
void restrict_block(SD_Solution F, SD_Solution C, Matrix R, int cx, int cy, int cz){
    int Nx=C.Nx, Ny=C.Ny, Nz=C.Nz;
    int nx=C.nx, ny=C.ny, nz=C.nz;
    int nader=C.n_ader, nvar=C.n_var;
    bool ax=cfg.active[_x_], ay=cfg.active[_y_], az=cfg.active[_z_];
    int NBx=Nx-2*NGHx, NBy=Ny-2*NGHy, NBz=Nz-2*NGHz;
    //coarse element range covered by this child
    int ix0 = NGHx + (ax ? cx*NBx/2 : 0);
    int iy0 = NGHy + (ay ? cy*NBy/2 : 0);
    int iz0 = NGHz + (az ? cz*NBz/2 : 0);
    int ix1 = ax ? ix0+NBx/2 : Nx-NGHx;
    int iy1 = ay ? iy0+NBy/2 : Ny-NGHy;
    int iz1 = az ? iz0+NBz/2 : Nz-NGHz;
    GHOST_LOCALS;
    sd_for_box_cells(iz0,iz1,iy0,iy1,ix0,ix1,nz,ny,nx, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        //first of the two fine elements (block-local, +ghost offset)
        int fx = ax ? ghx+2*(i-ghx)-cx*NBx : i;
        int fy = ay ? ghy+2*(j-ghy)-cy*NBy : j;
        int fz = az ? ghz+2*(k-ghz)-cz*NBz : k;
        for(int t_id=0; t_id<nader; t_id++){
        for(int var=0; var<nvar; var++){
            double u=0;
            for(int nn=0; nn<(az ? 2*nz:1); nn++){
            for(int mm=0; mm<(ay ? 2*ny:1); mm++){
            for(int ll=0; ll<(ax ? 2*nx:1); ll++){
                double s = F.Vector(t_id,var,
                                    az ? fz+nn/nz : k,
                                    ay ? fy+mm/ny : j,
                                    ax ? fx+ll/nx : i,
                                    az ? nn%nz : kk,
                                    ay ? mm%ny : jj,
                                    ax ? ll%nx : ii);
                if(ax) s *= R(ii,ll);
                if(ay) s *= R(jj,mm);
                if(az) s *= R(kk,nn);
                u += s;
            }}}
            C.Vector(t_id,var,k,j,i,kk,jj,ii) = u;
        }}
    }, "restrict_block");
}

//Side-aware overlap restriction matrices (spd/amr/transfer.py).
void build_overlap_restrict_matrices(Matrix R_sp[2], Matrix R_cv[2],
                                     double* x_sp, double* x_fp, int p){
    int n = p+1;
    int m = (p+2)/2;
    double* x_c = malloc_host<double>(2*n+1);
    for(int i=0;i<n;i++) x_c[i] = 2.0*x_fp[i];
    int nc = n;
    if(p%2==0){
        x_c[m] = 1.0;
        nc = n+1;
    }
    double* x_1 = malloc_host<double>(m+1);
    double* x_2 = malloc_host<double>(nc-m);
    for(int i=0;i<m+1;i++) x_1[i] = x_c[i];
    for(int i=0;i<nc-m;i++) x_2[i] = x_c[m+i]-1.0;

    for(int side=0; side<2; side++){
        double* edges = (side==0 ? x_1 : x_2);
        int nseg = (side==0 ? m+1 : nc-m);
        Matrix L("L_side", nseg, n);
        Matrix_h L_h = Kokkos::create_mirror_view(L);
        integral_matrix(L_h, edges, x_sp, nseg, n);
        Kokkos::deep_copy(L, L_h);

        Matrix_h R_h("R_sp", n, n);
        for(int j=0;j<n;j++)
        for(int i=0;i<n;i++) R_h(j,i) = 0.0;
        for(int k=0;k<nseg;k++){
            double a0 = edges[k], a1 = edges[k+1];
            double mid = 0.5*(a0+a1);
            int row = 0;
            while(row < n-1 && x_fp[row+1] <= mid) row++;
            if(row > n-1) row = n-1;
            double lens = a1-a0;
            double w = lens/(x_fp[row+1]-x_fp[row]);
            for(int i=0;i<n;i++)
                R_h(row,i) += w * L_h(k,i);
        }
        R_sp[side] = Matrix("R_sp_side", n, n);
        Kokkos::deep_copy(R_sp[side], R_h);

        Matrix_h Rcv_h("R_cv", n, n);
        for(int j=0;j<n;j++)
        for(int i=0;i<n;i++) Rcv_h(j,i) = 0.0;
        double fine_edges[32];
        for(int i=0;i<=n;i++) fine_edges[i] = 0.5*(x_fp[i]+double(side));
        for(int j=0;j<n;j++){
            double a0=x_fp[j], a1=x_fp[j+1], cw=a1-a0;
            for(int i=0;i<n;i++){
                double b0=fine_edges[i], b1=fine_edges[i+1];
                double overlap = std::max(0.0, std::min(a1,b1)-std::max(a0,b0));
                if(overlap>0.0) Rcv_h(j,i) += overlap/cw;
            }
        }
        R_cv[side] = Matrix("R_cv_side", n, n);
        Kokkos::deep_copy(R_cv[side], Rcv_h);
    }
    free(x_1); free(x_2); free(x_c);
}

//Apply M along the point axis `tdim`: C[..,q,..] = sum_l M(q,l) F[..,l,..],
//with all other indices carried through untouched. `tdim` is the axis being
//interpolated, not the face normal; callers sweep it over the directions
//transverse to the face. F and C must be distinct views (every thread reads
//the whole `tdim` row, so writing in place would race).
static void apply_mat_transverse(SD_Solution F, SD_Solution C, Matrix M, int tdim){
    int nader=F.n_ader, nvar=F.n_var;
    int Nx=F.Nx, Ny=F.Ny, Nz=F.Nz;
    int nx=F.nx, ny=F.ny, nz=F.nz;
    int np = (tdim==_x_ ? nx : (tdim==_y_ ? ny : nz));
    sd_for_cells(Nz,Ny,Nx,nz,ny,nx, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        int q = (tdim==_x_ ? ii : (tdim==_y_ ? jj : kk));
        for(int t_id=0; t_id<nader; t_id++){
        for(int var=0; var<nvar; var++){
            double u=0;
            for(int ll=0; ll<np; ll++){
                double s;
                if(tdim==_x_)      s = F.Vector(t_id,var,k,j,i,kk,jj,ll);
                else if(tdim==_y_) s = F.Vector(t_id,var,k,j,i,kk,ll,ii);
                else               s = F.Vector(t_id,var,k,j,i,ll,jj,ii);
                u += M(q,ll)*s;
            }
            C.Vector(t_id,var,k,j,i,kk,jj,ii) = u;
        }}
    });
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

//coarse_face by value: nvcc extended lambdas cannot capture references, and
//a Kokkos View copy still writes through to the same allocation.
void restrict_face_overlap_sp(const SD_Solution** fine_faces, int n_sub,
                              SD_Solution coarse_face, Matrix R, int dim){
    if(cfg.ndim==1 || n_sub<=1){
        Kokkos::deep_copy(coarse_face.Vector, fine_faces[0]->Vector);
        return;
    }
    //Adjoint of prolongate_face_coarser: identity along the face normal, and
    //transversally each coarse element gathers the two fine elements that
    //overlap it (columns [sub*n,(sub+1)*n) of R) from whichever fine
    //neighbour covers that half of the coarse block. The element ranges
    //partition the coarse interior, so every entry is written exactly once.
    int Nx=coarse_face.Nx, Ny=coarse_face.Ny, Nz=coarse_face.Nz;
    int nx=coarse_face.nx, ny=coarse_face.ny, nz=coarse_face.nz;
    int nader=coarse_face.n_ader, nvar=coarse_face.n_var;
    bool tx = cfg.active[_x_] && dim!=_x_;
    bool ty = cfg.active[_y_] && dim!=_y_;
    bool tz = cfg.active[_z_] && dim!=_z_;
    int NBx=Nx-2*NGHx, NBy=Ny-2*NGHy, NBz=Nz-2*NGHz;
    GHOST_LOCALS;
    for(int sub=0; sub<n_sub; sub++){
        int cx=0, cy=0, cz=0;
        {
            int bit=0;
            for(int d=0; d<3; d++){
                if(d==dim || !cfg.active[d]) continue;
                int v = (sub>>bit)&1;
                if(d==_x_) cx=v; else if(d==_y_) cy=v; else cz=v;
                bit++;
            }
        }
        int ix0 = NGHx + (tx ? cx*NBx/2 : 0), ix1 = tx ? ix0+NBx/2 : Nx-NGHx;
        int iy0 = NGHy + (ty ? cy*NBy/2 : 0), iy1 = ty ? iy0+NBy/2 : Ny-NGHy;
        int iz0 = NGHz + (tz ? cz*NBz/2 : 0), iz1 = tz ? iz0+NBz/2 : Nz-NGHz;
        SD_Solution Fs = *fine_faces[sub];
        sd_for_box_cells(iz0,iz1,iy0,iy1,ix0,ix1,nz,ny,nx, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
            int fx = tx ? ghx+2*(i-ghx)-cx*NBx : i;
            int fy = ty ? ghy+2*(j-ghy)-cy*NBy : j;
            int fz = tz ? ghz+2*(k-ghz)-cz*NBz : k;
            for(int t_id=0; t_id<nader; t_id++){
            for(int var=0; var<nvar; var++){
                double u=0;
                for(int nn=0; nn<(tz ? 2*nz:1); nn++){
                for(int mm=0; mm<(ty ? 2*ny:1); mm++){
                for(int ll=0; ll<(tx ? 2*nx:1); ll++){
                    double s = Fs.Vector(t_id,var,
                                         tz ? fz+nn/nz : k,
                                         ty ? fy+mm/ny : j,
                                         tx ? fx+ll/nx : i,
                                         tz ? nn%nz : kk,
                                         ty ? mm%ny : jj,
                                         tx ? ll%nx : ii);
                    if(tx) s *= R(ii,ll);
                    if(ty) s *= R(jj,mm);
                    if(tz) s *= R(kk,nn);
                    u += s;
                }}}
                coarse_face.Vector(t_id,var,k,j,i,kk,jj,ii) = u;
            }}
        }, "restrict_face_overlap_sp");
    }
}

void prolongate_face_coarser(SD_Solution coarse_face, SD_Solution fine_face,
                             Matrix P, int dim, int sub){
    if(cfg.ndim==1){
        Kokkos::deep_copy(fine_face.Vector, coarse_face.Vector);
        return;
    }
    //Transverse directions are refined, the face normal is not: at the
    //interface both blocks resolve `dim` identically, so that index passes
    //through. Transversally this is prolongate_block's mapping -- bit b of
    //`sub` picks which half of the coarse block the fine neighbour covers,
    //and the element parity within it picks the half of P.
    //By value: a reference captured into a device lambda leaves a host pointer
    //in the closure, which CUDA rejects outright. Views copy shallowly, so
    //writes through C still land in the caller's allocation.
    SD_Solution C = coarse_face;
    SD_Solution F = fine_face;
    int Nx=F.Nx, Ny=F.Ny, Nz=F.Nz;
    int nx=F.nx, ny=F.ny, nz=F.nz;
    int nader=F.n_ader, nvar=F.n_var;
    bool tx = cfg.active[_x_] && dim!=_x_;
    bool ty = cfg.active[_y_] && dim!=_y_;
    bool tz = cfg.active[_z_] && dim!=_z_;
    int NBx=Nx-2*NGHx, NBy=Ny-2*NGHy, NBz=Nz-2*NGHz;
    int cx=0, cy=0, cz=0;
    {
        int bit=0;
        for(int d=0; d<3; d++){
            if(d==dim || !cfg.active[d]) continue;
            int v = (sub>>bit)&1;
            if(d==_x_) cx=v; else if(d==_y_) cy=v; else cz=v;
            bit++;
        }
    }
    GHOST_LOCALS;
    sd_for_box_cells(NGHz,Nz-NGHz,NGHy,Ny-NGHy,NGHx,Nx-NGHx,nz,ny,nx, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        int gx = tx ? cx*NBx + (i-ghx) : 0;
        int gy = ty ? cy*NBy + (j-ghy) : 0;
        int gz = tz ? cz*NBz + (k-ghz) : 0;
        int cex = tx ? ghx+gx/2 : i;
        int cey = ty ? ghy+gy/2 : j;
        int cez = tz ? ghz+gz/2 : k;
        int sx = tx ? gx%2 : 0;
        int sy = ty ? gy%2 : 0;
        int sz = tz ? gz%2 : 0;
        for(int t_id=0; t_id<nader; t_id++){
        for(int var=0; var<nvar; var++){
            double u=0;
            for(int nn=0; nn<(tz ? nz:1); nn++){
            for(int mm=0; mm<(ty ? ny:1); mm++){
            for(int ll=0; ll<(tx ? nx:1); ll++){
                double s = C.Vector(t_id,var,cez,cey,cex,
                                    tz ? nn:kk, ty ? mm:jj, tx ? ll:ii);
                if(tx) s *= P(sx*nx+ii,ll);
                if(ty) s *= P(sy*ny+jj,mm);
                if(tz) s *= P(sz*nz+kk,nn);
                u += s;
            }}}
            F.Vector(t_id,var,k,j,i,kk,jj,ii) = u;
        }}
    }, "prolongate_face_coarser");
}

//Restrict one face-normal B component from fine child (cx,cy,cz) onto the
//covered coarse subregion. Transverse directions use the conservative L2
//matrix R (= amr_RF); along the face normal, element 2:1 with point-index
//injection from the fine half that owns each coarse flux point (AthenaK
//RestrictFC). When all children write their halves this preserves magnetic
//flux through the coarse faces.
void restrict_block_face_B(SD_Solution F, SD_Solution C, Matrix R,
                           int face_dim, int cx, int cy, int cz){
    int Nx=C.Nx, Ny=C.Ny, Nz=C.Nz;
    int nx=C.nx, ny=C.ny, nz=C.nz;
    int nader=C.n_ader, nvar=C.n_var;
    bool ax=cfg.active[_x_], ay=cfg.active[_y_], az=cfg.active[_z_];
    bool tx = ax && face_dim!=_x_;
    bool ty = ay && face_dim!=_y_;
    bool tz = az && face_dim!=_z_;
    int NBx=Nx-2*NGHx, NBy=Ny-2*NGHy, NBz=Nz-2*NGHz;
    int ix0 = NGHx + (ax ? cx*NBx/2 : 0);
    int iy0 = NGHy + (ay ? cy*NBy/2 : 0);
    int iz0 = NGHz + (az ? cz*NBz/2 : 0);
    int ix1 = ax ? ix0+NBx/2 : Nx-NGHx;
    int iy1 = ay ? iy0+NBy/2 : Ny-NGHy;
    int iz1 = az ? iz0+NBz/2 : Nz-NGHz;
    GHOST_LOCALS;
    sd_for_box_cells(iz0,iz1,iy0,iy1,ix0,ix1,nz,ny,nx, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        int fx = ax ? ghx+2*(i-ghx)-cx*NBx : i;
        int fy = ay ? ghy+2*(j-ghy)-cy*NBy : j;
        int fz = az ? ghz+2*(k-ghz)-cz*NBz : k;
        //Normal-direction fine element: left half of the flux-point index
        //range maps to fine sub 0, right half to sub 1 (AthenaK injection).
        int nsub = 0;
        if(face_dim==_x_ && ax) nsub = (ii*2 >= nx) ? 1 : 0;
        if(face_dim==_y_ && ay) nsub = (jj*2 >= ny) ? 1 : 0;
        if(face_dim==_z_ && az) nsub = (kk*2 >= nz) ? 1 : 0;
        for(int t_id=0; t_id<nader; t_id++){
        for(int var=0; var<nvar; var++){
            double u=0;
            for(int nn=0; nn<(tz ? 2*nz:1); nn++){
            for(int mm=0; mm<(ty ? 2*ny:1); mm++){
            for(int ll=0; ll<(tx ? 2*nx:1); ll++){
                int fkk = tz ? nn%nz : kk;
                int fjj = ty ? mm%ny : jj;
                int fii = tx ? ll%nx : ii;
                int fek = tz ? fz+nn/nz : k;
                int fej = ty ? fy+mm/ny : j;
                int fei = tx ? fx+ll/nx : i;
                if(face_dim==_x_ && ax){ fei = fx+nsub; fii = ii; }
                if(face_dim==_y_ && ay){ fej = fy+nsub; fjj = jj; }
                if(face_dim==_z_ && az){ fek = fz+nsub; fkk = kk; }
                double s = F.Vector(t_id,var,fek,fej,fei,fkk,fjj,fii);
                if(tx) s *= R(ii,ll);
                if(ty) s *= R(jj,mm);
                if(tz) s *= R(kk,nn);
                u += s;
            }}}
            C.Vector(t_id,var,k,j,i,kk,jj,ii) = u;
        }}
    }, "restrict_block_face_B");
}

//2D Toth–Roe interior faces inside each 2×2 of fine elements covering one
//coarse element (AthenaK ProlongFCInternal). Fields are transverse face
//averages (sp_to_cv already applied). Interior faces are written as
//face-constants so the integral form of divB=0 holds on each fine cell.
static void fill_interior_face_B_2d(SD_Solution Bx, SD_Solution By,
                                    int cx, int cy, int NBx, int NBy){
    int nx=Bx.nx, ny=By.ny;
    int nty=Bx.ny, ntx=By.nx;
    int Nz=Bx.Nz;
    GHOST_LOCALS;
    int ix0 = NGHx, ix1 = Bx.Nx-NGHx;
    int iy0 = NGHy, iy1 = By.Ny-NGHy;
    for_box2(iy0,iy1,ix0,ix1, KOKKOS_LAMBDA(int j, int i){
        int gx = (i-ghx) + cx*NBx;
        int gy = (j-ghy) + cy*NBy;
        if((gx%2)!=0 || (gy%2)!=0) return;
        int i0=i, j0=j, i1=i+1, j1=j+1;
        if(i1>=ix1 || j1>=iy1) return;
        int k0 = (Nz>1 ? ghz : 0);
        double Bx_L_S=0, Bx_L_N=0, Bx_R_S=0, Bx_R_N=0;
        for(int jj=0;jj<nty;jj++){
            Bx_L_S += Bx.Vector(0,0,k0,j0,i0,0,jj,0);
            Bx_L_N += Bx.Vector(0,0,k0,j1,i0,0,jj,0);
            Bx_R_S += Bx.Vector(0,0,k0,j0,i1,0,jj,nx-1);
            Bx_R_N += Bx.Vector(0,0,k0,j1,i1,0,jj,nx-1);
        }
        Bx_L_S/=nty; Bx_L_N/=nty; Bx_R_S/=nty; Bx_R_N/=nty;
        double By_B_W=0, By_B_E=0, By_T_W=0, By_T_E=0;
        for(int ii=0;ii<ntx;ii++){
            By_B_W += By.Vector(0,0,k0,j0,i0,0,0,ii);
            By_B_E += By.Vector(0,0,k0,j0,i1,0,0,ii);
            By_T_W += By.Vector(0,0,k0,j1,i0,0,ny-1,ii);
            By_T_E += By.Vector(0,0,k0,j1,i1,0,ny-1,ii);
        }
        By_B_W/=ntx; By_B_E/=ntx; By_T_W/=ntx; By_T_E/=ntx;
        double tmp1 = 0.25*((By_T_E - By_B_E) - (By_T_W - By_B_W));
        double tmp2 = 0.25*((Bx_L_S - Bx_R_S) - (Bx_L_N - Bx_R_N));
        double Bx_M_S = 0.5*(Bx_L_S + Bx_R_S) + tmp1;
        double Bx_M_N = 0.5*(Bx_L_N + Bx_R_N) + tmp1;
        double By_M_W = 0.5*(By_B_W + By_T_W) + tmp2;
        double By_M_E = 0.5*(By_B_E + By_T_E) + tmp2;
        for(int jj=0;jj<nty;jj++){
            Bx.Vector(0,0,k0,j0,i0,0,jj,nx-1) = Bx_M_S;
            Bx.Vector(0,0,k0,j0,i1,0,jj,0)    = Bx_M_S;
            Bx.Vector(0,0,k0,j1,i0,0,jj,nx-1) = Bx_M_N;
            Bx.Vector(0,0,k0,j1,i1,0,jj,0)    = Bx_M_N;
        }
        for(int ii=0;ii<ntx;ii++){
            By.Vector(0,0,k0,j0,i0,0,ny-1,ii) = By_M_W;
            By.Vector(0,0,k0,j1,i0,0,0,ii)    = By_M_W;
            By.Vector(0,0,k0,j0,i1,0,ny-1,ii) = By_M_E;
            By.Vector(0,0,k0,j1,i1,0,0,ii)    = By_M_E;
        }
    }, "fill_interior_face_B_2d");
}

//Inject the coarse face average (constant) onto each fine face that lies on a
//coarse element face. Full-face averages preserve the coarse FV divergence so
//Toth–Roe then yields exact fine divB=0 (unlike half-face Lagrange + flatten).
//The average is the flux-point-spacing weighted sum of the control volumes, not
//a plain mean: the control volumes tile the face unevenly, and only the true
//integral makes the coarse element's face balance equal its (zero) divergence.
static void prolongate_shared_face_B_const(SD_Solution C, SD_Solution F,
                                          int face_dim, int cx, int cy, int cz){
    int Nx=F.Nx, Ny=F.Ny, Nz=F.Nz;
    int nx=F.nx, ny=F.ny, nz=F.nz;
    bool ax=cfg.active[_x_], ay=cfg.active[_y_], az=cfg.active[_z_];
    int NBx=Nx-2*NGHx, NBy=Ny-2*NGHy, NBz=Nz-2*NGHz;
    Vector xfp = amr_x_fp;
    GHOST_LOCALS;
    (void)cz;
    if(face_dim==_x_){
        int nty=F.ny, ntyC=C.ny;
        for_box3p1(NGHz,Nz-NGHz,NGHy,Ny-NGHy,NGHx,Nx-NGHx,nz, KOKKOS_LAMBDA(int k, int j, int i, int kk){
                int gx = ax ? cx*NBx + (i-ghx) : 0;
                int gy = ay ? cy*NBy + (j-ghy) : 0;
                int gz = az ? cz*NBz + (k-ghz) : 0;
                int cex = ax ? ghx+gx/2 : i;
                int cey = ay ? ghy+gy/2 : j;
                int cez = az ? ghz+gz/2 : k;
                for(int side=0; side<2; side++){
                    int ii = side ? nx-1 : 0;
                    bool on_coarse = ax ? (side ? ((gx%2)==1) : ((gx%2)==0)) : true;
                    if(!on_coarse) continue;
                    int cii = side ? C.nx-1 : 0;
                    double a=0;
                    for(int jj=0; jj<ntyC; jj++)
                        a += (xfp(jj+1)-xfp(jj))
                           * C.Vector(0,0,cez,cey,cex,kk,jj,cii);
                    for(int jj=0; jj<nty; jj++)
                        F.Vector(0,0,k,j,i,kk,jj,ii) = a;
                }
            }, "prolong_shared_Bx_const");
    } else if(face_dim==_y_){
        int ntx=F.nx, ntxC=C.nx;
        for_box3p1(NGHz,Nz-NGHz,NGHy,Ny-NGHy,NGHx,Nx-NGHx,nz, KOKKOS_LAMBDA(int k, int j, int i, int kk){
                int gx = ax ? cx*NBx + (i-ghx) : 0;
                int gy = ay ? cy*NBy + (j-ghy) : 0;
                int gz = az ? cz*NBz + (k-ghz) : 0;
                int cex = ax ? ghx+gx/2 : i;
                int cey = ay ? ghy+gy/2 : j;
                int cez = az ? ghz+gz/2 : k;
                for(int side=0; side<2; side++){
                    int jj = side ? ny-1 : 0;
                    bool on_coarse = ay ? (side ? ((gy%2)==1) : ((gy%2)==0)) : true;
                    if(!on_coarse) continue;
                    int cjj = side ? C.ny-1 : 0;
                    double a=0;
                    for(int ii=0; ii<ntxC; ii++)
                        a += (xfp(ii+1)-xfp(ii))
                           * C.Vector(0,0,cez,cey,cex,kk,cjj,ii);
                    for(int ii=0; ii<ntx; ii++)
                        F.Vector(0,0,k,j,i,kk,jj,ii) = a;
                }
            }, "prolong_shared_By_const");
    }
}

//After element-face FPs are set (shared prolongate + Toth–Roe), fill the
//interior normal-direction flux points by linear interpolation between the
//two element faces in the reference coordinate ξ∈[0,1]. Face fluxes are
//unchanged; for face-constant data the discrete dfp_to_sp divergence matches
//the FV face divergence (zero after Toth–Roe).
static void fill_interior_normal_fps(SD_Solution B, int face_dim){
    int Nx=B.Nx, Ny=B.Ny, Nz=B.Nz;
    int nx=B.nx, ny=B.ny, nz=B.nz;
    Vector xfp = amr_x_fp;
    int nfp = (int)xfp.extent(0);
    GHOST_LOCALS;
    if(face_dim==_x_){
        if(nx < 3 || nfp < nx) return;
        for_box3p2(NGHz,Nz-NGHz,NGHy,Ny-NGHy,NGHx,Nx-NGHx,nz,ny, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj){
                double BL = B.Vector(0,0,k,j,i,kk,jj,0);
                double BR = B.Vector(0,0,k,j,i,kk,jj,nx-1);
                for(int ii=1; ii<nx-1; ii++){
                    double xi = xfp(ii);
                    B.Vector(0,0,k,j,i,kk,jj,ii) = (1.0-xi)*BL + xi*BR;
                }
            }, "fill_interior_normal_fps_x");
    } else if(face_dim==_y_){
        if(ny < 3 || nfp < ny) return;
        for_box3p2(NGHz,Nz-NGHz,NGHy,Ny-NGHy,NGHx,Nx-NGHx,nz,nx, KOKKOS_LAMBDA(int k, int j, int i, int kk, int ii){
                double BB = B.Vector(0,0,k,j,i,kk,0,ii);
                double BT = B.Vector(0,0,k,j,i,kk,ny-1,ii);
                for(int jj=1; jj<ny-1; jj++){
                    double xi = xfp(jj);
                    B.Vector(0,0,k,j,i,kk,jj,ii) = (1.0-xi)*BB + xi*BT;
                }
            }, "fill_interior_normal_fps_y");
    } else {
        if(nz < 3 || nfp < nz) return;
        for_box3p2(NGHz,Nz-NGHz,NGHy,Ny-NGHy,NGHx,Nx-NGHx,ny,nx, KOKKOS_LAMBDA(int k, int j, int i, int jj, int ii){
                double B0 = B.Vector(0,0,k,j,i,0,jj,ii);
                double B1 = B.Vector(0,0,k,j,i,nz-1,jj,ii);
                for(int kk=1; kk<nz-1; kk++){
                    double xi = xfp(kk);
                    B.Vector(0,0,k,j,i,kk,jj,ii) = (1.0-xi)*B0 + xi*B1;
                }
            }, "fill_interior_normal_fps_z");
    }
}

//Restrict the coarse faces of one fine child onto its quadrant of the coarse
//block, as face constants. A coarse face average is the mean of the two fine
//face averages covering it; the faces interior to the coarse element are
//dropped. Summing the four fine elements' face balances cancels those interior
//faces, so the coarse element inherits a zero balance exactly -- no projection
//needed. Each fine face average is the flux-point-spacing weighted sum of its
//control volumes, which is the true integral and hence equals the fine
//element's divergence via the divergence theorem.
static void restrict_shared_face_B_const(SD_Solution Fa, SD_Solution Ca,
                                         int face_dim, int cx, int cy){
    int Nx=Ca.Nx, Ny=Ca.Ny, Nz=Ca.Nz;
    int nx=Ca.nx, ny=Ca.ny, nz=Ca.nz;
    bool ax=cfg.active[_x_], ay=cfg.active[_y_], az=cfg.active[_z_];
    int NBx=Nx-2*NGHx, NBy=Ny-2*NGHy, NBz=Nz-2*NGHz;
    int ix0 = NGHx + (ax ? cx*NBx/2 : 0);
    int iy0 = NGHy + (ay ? cy*NBy/2 : 0);
    int ix1 = ax ? ix0+NBx/2 : Nx-NGHx;
    int iy1 = ay ? iy0+NBy/2 : Ny-NGHy;
    int iz1 = az ? NGHz+NBz/2 : Nz-NGHz;
    Vector xfp = amr_x_fp;
    GHOST_LOCALS;
    if(face_dim==_x_){
        int nty=Ca.ny;
        int nsy = ay ? 2 : 1;
        for_box3p1(NGHz,iz1,iy0,iy1,ix0,ix1,nz, KOKKOS_LAMBDA(int k, int j, int i, int kk){
                int fx = ax ? ghx+2*(i-ghx)-cx*NBx : i;
                int fy = ay ? ghy+2*(j-ghy)-cy*NBy : j;
                double L=0, R=0;
                for(int sub=0; sub<nsy; sub++)
                for(int jj=0; jj<nty; jj++){
                    double w = (xfp(jj+1)-xfp(jj))/nsy;
                    L += w*Fa.Vector(0,0,k,fy+sub,fx,       kk,jj,0);
                    R += w*Fa.Vector(0,0,k,fy+sub,fx+(ax?1:0),kk,jj,nx-1);
                }
                for(int jj=0; jj<nty; jj++){
                    Ca.Vector(0,0,k,j,i,kk,jj,0)    = L;
                    Ca.Vector(0,0,k,j,i,kk,jj,nx-1) = R;
                }
            }, "restrict_shared_Bx_const");
    } else if(face_dim==_y_){
        int ntx=Ca.nx;
        int nsx = ax ? 2 : 1;
        for_box3p1(NGHz,iz1,iy0,iy1,ix0,ix1,nz, KOKKOS_LAMBDA(int k, int j, int i, int kk){
                int fx = ax ? ghx+2*(i-ghx)-cx*NBx : i;
                int fy = ay ? ghy+2*(j-ghy)-cy*NBy : j;
                double B=0, T=0;
                for(int sub=0; sub<nsx; sub++)
                for(int ii=0; ii<ntx; ii++){
                    double w = (xfp(ii+1)-xfp(ii))/nsx;
                    B += w*Fa.Vector(0,0,k,fy,       fx+sub,kk,0,ii);
                    T += w*Fa.Vector(0,0,k,fy+(ay?1:0),fx+sub,kk,ny-1,ii);
                }
                for(int ii=0; ii<ntx; ii++){
                    Ca.Vector(0,0,k,j,i,kk,0,ii)    = B;
                    Ca.Vector(0,0,k,j,i,kk,ny-1,ii) = T;
                }
            }, "restrict_shared_By_const");
    }
}

//Zero FV divergence on every element by adjusting the +x face, keeping other
//faces fixed. Makes face-constant (+ linear-normal) data exactly SD-div-free
//before cv_to_sp. Shared faces may disagree until Sync_face_B; call after
//all children are prolongated and again after Sync if needed.
void project_face_B_divfree_2d(SD_Solution Bx, SD_Solution By){
    int Nx=Bx.Nx, Ny=Bx.Ny, Nz=Bx.Nz;
    int nx=Bx.nx, ny=By.ny, nty=Bx.ny, ntx=By.nx, nz=Bx.nz;
    Vector xfp = amr_x_fp;
    GHOST_LOCALS;
    for_box3(NGHz,Nz-NGHz,NGHy,Ny-NGHy,NGHx,Nx-NGHx, KOKKOS_LAMBDA(int k, int j, int i){
            for(int kk=0; kk<nz; kk++){
                double bxL=0, bxR=0, byB=0, byT=0;
                for(int jj=0; jj<nty; jj++){
                    bxL += Bx.Vector(0,0,k,j,i,kk,jj,0);
                    bxR += Bx.Vector(0,0,k,j,i,kk,jj,nx-1);
                }
                for(int ii=0; ii<ntx; ii++){
                    byB += By.Vector(0,0,k,j,i,kk,0,ii);
                    byT += By.Vector(0,0,k,j,i,kk,ny-1,ii);
                }
                bxL/=nty; bxR/=nty; byB/=ntx; byT/=ntx;
                double d = (bxR - bxL) + (byT - byB);
                double bxR_new = bxR - d;
                for(int jj=0; jj<nty; jj++){
                    Bx.Vector(0,0,k,j,i,kk,jj,nx-1) = bxR_new;
                    double BL = Bx.Vector(0,0,k,j,i,kk,jj,0);
                    for(int ii=1; ii<nx-1; ii++){
                        double xi = xfp(ii);
                        Bx.Vector(0,0,k,j,i,kk,jj,ii) = (1.0-xi)*BL + xi*bxR_new;
                    }
                }
            }
        }, "project_face_B_divfree_2d");
}

void prolongate_block_face_B(SD_Solution BxC, SD_Solution ByC, SD_Solution BzC,
                             SD_Solution BxF, SD_Solution ByF, SD_Solution BzF,
                             Matrix P, Matrix sp_to_cv, Matrix cv_to_sp,
                             int cx, int cy, int cz){
    if(cfg.active[_z_]){
        if(Master)
            std::cout<<"ERROR: prolongate_block_face_B 3D not implemented "
                       <<"(Stage 4 supports true-2D MHD SMR only)"<<std::endl;
        exit(1);
    }
    (void)cz;
    SD_Solution BxCa = make_scratch_like(BxC, "BxCa");
    SD_Solution ByCa = make_scratch_like(ByC, "ByCa");
    SD_Solution BxFa = make_scratch_like(BxF, "BxFa");
    SD_Solution ByFa = make_scratch_like(ByF, "ByFa");
    SD_Solution Tx = make_scratch_like(BxC, "TxB");
    SD_Solution Ty = make_scratch_like(ByC, "TyB");
    transform_a_to_b_2d(BxC, BxCa, Tx, sp_to_cv, _x_);
    transform_a_to_b_2d(ByC, ByCa, Ty, sp_to_cv, _y_);
    Kokkos::deep_copy(BxFa.Vector, 0.0);
    Kokkos::deep_copy(ByFa.Vector, 0.0);
    prolongate_shared_face_B_const(BxCa, BxFa, _x_, cx, cy, 0);
    prolongate_shared_face_B_const(ByCa, ByFa, _y_, cx, cy, 0);
    int NBx=BxF.Nx-2*NGHx, NBy=ByF.Ny-2*NGHy;
    fill_interior_face_B_2d(BxFa, ByFa, cx, cy, NBx, NBy);
    project_face_B_divfree_2d(BxFa, ByFa);
    fill_interior_normal_fps(BxFa, _x_);
    fill_interior_normal_fps(ByFa, _y_);
    SD_Solution TxF = make_scratch_like(BxF, "TxF");
    SD_Solution TyF = make_scratch_like(ByF, "TyF");
    transform_a_to_b_2d(BxFa, BxF, TxF, cv_to_sp, _x_);
    transform_a_to_b_2d(ByFa, ByF, TyF, cv_to_sp, _y_);
    prolongate_block(BzC, BzF, P, cx, cy, 0);
}

//Derefinement counterpart of prolongate_block_face_B. Works in the same
//transverse-face-average representation so the coarse block ends up
//face-constant with a linear normal profile, the one form in which the SD
//divergence and the FV face balance coincide. Called once per child; the
//coarse block is round-tripped through the cv representation each time, which
//is exact and leaves the quadrants already written by earlier children alone.
void restrict_block_face_B_2d(SD_Solution BxF, SD_Solution ByF, SD_Solution BzF,
                              SD_Solution BxC, SD_Solution ByC, SD_Solution BzC,
                              Matrix R, Matrix sp_to_cv, Matrix cv_to_sp,
                              int cx, int cy, int cz){
    if(cfg.active[_z_]){
        if(Master)
            std::cout<<"ERROR: restrict_block_face_B_2d 3D not implemented "
                       <<"(Stage 4 supports true-2D MHD SMR only)"<<std::endl;
        exit(1);
    }
    SD_Solution BxFa = make_scratch_like(BxF, "BxFa_r");
    SD_Solution ByFa = make_scratch_like(ByF, "ByFa_r");
    SD_Solution BxCa = make_scratch_like(BxC, "BxCa_r");
    SD_Solution ByCa = make_scratch_like(ByC, "ByCa_r");
    SD_Solution TxF = make_scratch_like(BxF, "TxF_r");
    SD_Solution TyF = make_scratch_like(ByF, "TyF_r");
    SD_Solution TxC = make_scratch_like(BxC, "TxC_r");
    SD_Solution TyC = make_scratch_like(ByC, "TyC_r");
    transform_a_to_b_2d(BxF, BxFa, TxF, sp_to_cv, _x_);
    transform_a_to_b_2d(ByF, ByFa, TyF, sp_to_cv, _y_);
    transform_a_to_b_2d(BxC, BxCa, TxC, sp_to_cv, _x_);
    transform_a_to_b_2d(ByC, ByCa, TyC, sp_to_cv, _y_);
    restrict_shared_face_B_const(BxFa, BxCa, _x_, cx, cy);
    restrict_shared_face_B_const(ByFa, ByCa, _y_, cx, cy);
    fill_interior_normal_fps(BxCa, _x_);
    fill_interior_normal_fps(ByCa, _y_);
    transform_a_to_b_2d(BxCa, BxC, TxC, cv_to_sp, _x_);
    transform_a_to_b_2d(ByCa, ByC, TyC, cv_to_sp, _y_);
    restrict_block_face_B(BzF, BzC, R, _z_, cx, cy, cz);
}


void init_amr_transfer_matrices(double* x_sp, double* x_fp, int p){
    int n = p+1;
    int m = p+2;
    amr_x_fp = Vector("amr_x_fp", m);
    {
        Vector_h xf = Kokkos::create_mirror_view(amr_x_fp);
        for(int i=0;i<m;i++) xf(i) = x_fp[i];
        Kokkos::deep_copy(amr_x_fp, xf);
    }
    amr_P = Matrix("amr_P", 2*n, n);
    amr_R = Matrix("amr_R", n, 2*n);
    transfer_matrices(amr_P, amr_R, x_sp, p);
    build_overlap_restrict_matrices(amr_RS_sp, amr_RS_cv, x_sp, x_fp, p);

    //amr_R is exact on degree-p data but does not preserve integrals, so
    //restricting with it leaks mass at every regrid and every coarse-fine
    //interface. Build the L2 projection instead: each coarse cell averages the
    //fine element's actual polynomial over their true overlap. That conserves
    //(the coarse cells tile the element, so the pieces sum to the whole) and
    //stays exact. Averaging precomputed fine cell averages would conserve too
    //but drop to first order, because the fine cells -- flux-point spaced and
    //halved -- straddle the coarse cell boundaries.
    //Built entirely on host mirrors: integral_matrix and inverse are host
    //loops, so handing them device views trips the CUDA memory-space check.
    //Only the final amr_RF needs to live on the device.
    Matrix_h s2c("amr_sp_to_cv", n, n), c2s("amr_cv_to_sp", n, n);
    integral_matrix(s2c, x_fp, x_sp, n, n);
    inverse(s2c, c2s, n);

    Matrix_h rf_cv("rf_cv", n, 2*n);
    for(int j=0; j<n; j++)
    for(int b=0; b<2*n; b++) rf_cv(j,b) = 0.0;
    for(int j=0; j<n; j++){
        double c0=x_fp[j], c1=x_fp[j+1], cw=c1-c0;
        for(int s=0; s<2; s++){
            //fine element s occupies [s/2,(s+1)/2) of the coarse element
            double a0 = std::max(c0, 0.5*s), a1 = std::min(c1, 0.5*(s+1));
            if(a1 <= a0) continue;
            double seg[2] = {2.0*a0-s, 2.0*a1-s};  //fine-element local coords
            Matrix_h Lh("Lseg", 1, n);
            integral_matrix(Lh, seg, x_sp, 1, n);
            for(int b=0; b<n; b++)
                rf_cv(j, s*n+b) += ((a1-a0)/cw) * Lh(0,b);
        }
    }
    amr_RF = Matrix("amr_RF", n, 2*n);
    Matrix_h rf = Kokkos::create_mirror_view(amr_RF);
    for(int a=0; a<n; a++)
    for(int b=0; b<2*n; b++){
        double v=0;
        for(int u=0; u<n; u++) v += c2s(a,u)*rf_cv(u,b);
        rf(a,b) = v;
    }
    Kokkos::deep_copy(amr_RF, rf);

    //Flux-point lattice operators for edge-EMF coarse/fine exchange. The
    //concatenated fine nodes skip the duplicate midpoint (right end of the
    //left half == left end of the right half) so the Vandermonde stays
    //well-conditioned; each half still has m-1 unique interiors + the shared
    //midpoint once, totaling 2m-1 fine nodes. We keep the (2m) x m P shape by
    //repeating the midpoint row, and build R as an L2 overlap restrict onto
    //the coarse fp nodes (same construction as amr_RF).
    double* x_fine_fp = malloc_host<double>(2*m);
    for(int i=0;i<m;i++){
        x_fine_fp[i]   = 0.5*x_fp[i];
        x_fine_fp[m+i] = 0.5*x_fp[i]+0.5;
    }
    amr_P_fp = Matrix("amr_P_fp", 2*m, m);
    Matrix_h Pfp_h = Kokkos::create_mirror_view(amr_P_fp);
    lagrange_matrix(Pfp_h, x_fp, x_fine_fp, m, 2*m);
    Kokkos::deep_copy(amr_P_fp, Pfp_h);

    Matrix_h s2c_f("fp_to_seg", m, m), c2s_f("seg_to_fp", m, m);
    //Integrate the fp Lagrange basis over the m segments [x_fp[j], x_fp[j+1]).
    //x_fp has m = p+2 points so there are m-1 intervals; pad the last row.
    //Reuse the same overlap recipe as amr_RF but with x_fp as the node set.
    integral_matrix(s2c_f, x_fp, x_fp, m, m); // best-effort; see overlap below
    //Build overlap RF directly: each coarse segment averages fine fp samples
    //over the physical overlap of the two fine halves.
    Matrix_h rf_fp("rf_fp", m, 2*m);
    for(int j=0;j<m;j++)
    for(int b=0;b<2*m;b++) rf_fp(j,b)=0.0;
    //Simple, stable restrict: each coarse fp node is the average of the two
    //fine-half interpolants at that node (rows of the Lagrange R from
    //unique-ish fine nodes). Equivalent to 0.5*(P_left^+ + P_right^+)
    //evaluated back — implement as equal-weight gathering of the matching
    //fine indices after mapping through the half.
    //
    //Practical Stage-4 choice that preserves constants: coarse[j] =
    //0.5*(fine_left[j] + fine_right[j]). Encoded as R(j, j)=R(j, m+j)=0.5.
    for(int j=0;j<m;j++){
        rf_fp(j, j)   = 0.5;
        rf_fp(j, m+j) = 0.5;
    }
    amr_RF_fp = Matrix("amr_RF_fp", m, 2*m);
    Kokkos::deep_copy(amr_RF_fp, rf_fp);
    free(x_fine_fp);
    (void)s2c_f;(void)c2s_f;
}
