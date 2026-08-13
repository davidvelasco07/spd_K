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
    Kokkos::parallel_for("gather_block",
        Kokkos::MDRangePolicy<Kokkos::Rank<6>>({0,0,0,0,0,0},{Nz,Ny,Nx,nz,ny,nx}),
        KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        for(int t_id=0; t_id<nader; t_id++){
        for(int var=0; var<nvar; var++){
        G.Vector(t_id,var,ghz+oz+k,ghy+oy+j,ghx+ox+i,kk,jj,ii) =
            B.Vector(t_id,var,ghz+k,ghy+j,ghx+i,kk,jj,ii);
        }}
    });
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
    Kokkos::parallel_for("prolongate_block",
        Kokkos::MDRangePolicy<Kokkos::Rank<6>>({NGHz,NGHy,NGHx,0,0,0},{Nz-NGHz,Ny-NGHy,Nx-NGHx,nz,ny,nx}),
        KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
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
    });
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
    Kokkos::parallel_for("restrict_block",
        Kokkos::MDRangePolicy<Kokkos::Rank<6>>({iz0,iy0,ix0,0,0,0},{iz1,iy1,ix1,nz,ny,nx}),
        KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
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
    });
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

void restrict_face_overlap_sp(const SD_Solution** fine_faces, int n_sub,
                              SD_Solution& coarse_face, Matrix R, int dim){
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
        Kokkos::parallel_for("restrict_face_overlap_sp",
            Kokkos::MDRangePolicy<Kokkos::Rank<6>>({iz0,iy0,ix0,0,0,0},
                                                   {iz1,iy1,ix1,nz,ny,nx}),
            KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
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
        });
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
    SD_Solution& C = coarse_face;
    SD_Solution& F = fine_face;
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
    Kokkos::parallel_for("prolongate_face_coarser",
        Kokkos::MDRangePolicy<Kokkos::Rank<6>>({NGHz,NGHy,NGHx,0,0,0},
                                               {Nz-NGHz,Ny-NGHy,Nx-NGHx,nz,ny,nx}),
        KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
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
    });
}

void init_amr_transfer_matrices(double* x_sp, double* x_fp, int p){
    int n = p+1;
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
}
