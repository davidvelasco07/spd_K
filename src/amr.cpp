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
    lagrange_matrix(P, x_sp, x_fine, n, 2*n);
    lagrange_matrix(R, x_fine, x_sp, 2*n, n);
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
