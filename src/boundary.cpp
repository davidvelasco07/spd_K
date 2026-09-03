#include "spd_k.hpp"

void exec_comm(Boundaries BC, CommHelper comm, int L, int R){
    #ifdef MPI
    Kokkos::fence(); //buffers must be packed before MPI reads them
    int nreq=0;
    MPI_Request mpi_requests_recv[2];
    MPI_Request mpi_requests_send[2];
    if(L!=-1){
        BC.isend_irecv_L(comm,L,&mpi_requests_send[nreq],&mpi_requests_recv[nreq]);
        nreq++;
    }
    if(R!=-1){
        BC.isend_irecv_R(comm,R,&mpi_requests_send[nreq],&mpi_requests_recv[nreq]);
        nreq++;
    }
    if(nreq>0){
        MPI_Waitall(nreq,mpi_requests_send,MPI_STATUSES_IGNORE);
        MPI_Waitall(nreq,mpi_requests_recv,MPI_STATUSES_IGNORE);
    }
    if(L!=-1)BC.copy_L();
    if(R!=-1)BC.copy_R();
    #endif
}

KOKKOS_INLINE_FUNCTION
double value(SD_Solution U, int t_id, int var, int k, int j, int i, int kk, int jj, int ii, int l, int ll, int dim){
    if(dim==0)
        return U.Vector(t_id,var,k,j,l,kk,jj,ll);
    else if(dim==1)
        return U.Vector(t_id,var,k,l,i,kk,ll,ii);
    else if(dim==2)
        return U.Vector(t_id,var,l,j,i,ll,jj,ii);
    else return 0;
}

KOKKOS_INLINE_FUNCTION
void indices(int* N_id, int* n_id, int k, int j, int i, int kk, int jj, int ii, int l, int ll, int dim){
    //Returns the indeces according to the dimension
    N_id[_x_] = dim == _x_ ? l  : i;
    n_id[_x_] = dim == _x_ ? ll : ii;
    N_id[_y_] = dim == _y_ ? l  : j;
    n_id[_y_] = dim == _y_ ? ll : jj;
    N_id[_z_] = dim == _z_ ? l  : k;
    n_id[_z_] = dim == _z_ ? ll : kk;
}

//SPD_JET_BASE_WALL=1 restores the reflecting-wall fallback outside the nozzle
//for an _inflow_ face. It is a REFERENCE, kept so the choice of default can be
//re-measured in one binary; it is broken (see the _inflow_ branch) and must not
//be a production setting.
static inline bool jet_base_wall(){
    static const bool v = getenv("SPD_JET_BASE_WALL") != nullptr;
    return v;
}

void boundaries(
    CommHelper comm,
    Boundaries BC,
    SD_Solution U
    ){
    int Nx = BC.Nx;
    int Ny = BC.Ny;
    int Nz = BC.Nz;
    int px = BC.nx;
    int py = BC.ny;
    int pz = BC.nz;
    int nader = BC.nader;
    int nvar  = BC.nvar;
    const bool mhd = BC.mhd;
    int type = BC.type;
    const bool wall_off_nozzle = jet_base_wall();
    int N = BC.N;
    int n = BC.n;
    int dim = BC.dim;
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        for(int t_id=0; t_id<nader; t_id++){
        for(int var=0; var<nvar; var++){
        int Nid[3];
        int nid[3];
        if(type == _periodic_){ 
            indices(Nid,nid,k,j,i,kk,jj,ii,N-2,n-1,dim);
            BC.BoundaryL(t_id,var,k,j,i,kk,jj,ii) = U.Vector(INDICES);
            indices(Nid,nid,k,j,i,kk,jj,ii,  1,  0,dim);
            BC.BoundaryR(t_id,var,k,j,i,kk,jj,ii) = U.Vector(INDICES);
        }
        else if(type == _gradfree_){
            indices(Nid,nid,k,j,i,kk,jj,ii,  1,  0,dim);
            BC.BoundaryL(t_id,var,k,j,i,kk,jj,ii) = U.Vector(INDICES);
            indices(Nid,nid,k,j,i,kk,jj,ii,N-2,n-1,dim);
            BC.BoundaryR(t_id,var,k,j,i,kk,jj,ii) = U.Vector(INDICES);
        }
        else if(type == _outflow_){
            //Plain zeroth-order copy, same as _gradfree_. A no-reentry clamp
            //(zeroing the ghost's normal momentum when it points inward) was
            //tried here and MEASURED to be far worse: it puts a velocity
            //discontinuity at the boundary, which sets up a strong Riemann
            //problem every step. On the Mach-800 jet at 400x600 DoF it collapsed
            //dt from 1.32e-06 to ~3.2e-09 -- 400x, for MUSCL as well as SDFB --
            //and no lane finished. Preventing re-entry needs a characteristic
            //treatment, not a clamp on one row.
            indices(Nid,nid,k,j,i,kk,jj,ii,  1,  0,dim);
            BC.BoundaryL(t_id,var,k,j,i,kk,jj,ii) = U.Vector(INDICES);
            indices(Nid,nid,k,j,i,kk,jj,ii,N-2,n-1,dim);
            BC.BoundaryR(t_id,var,k,j,i,kk,jj,ii) = U.Vector(INDICES);
        }
        else if(type == _inflow_){
            //LOW side: the prescribed state where the problem injects, OUTFLOW
            //everywhere else on that face. The sentinel is the density row of
            //InflowL: negative means "not an inflow point". Clamping the whole
            //face instead of just the nozzle walls in the cocoon backflow.
            //HIGH side is always outflow.
            if(BC.InflowL(t_id,0,k,j,i,kk,jj,ii) >= 0.0){
                BC.BoundaryL(t_id,var,k,j,i,kk,jj,ii) =
                    BC.InflowL(t_id,var,k,j,i,kk,jj,ii);
            } else if(wall_off_nozzle){
                //REFERENCE ONLY, and MEASURED BROKEN -- see the outflow branch
                //below. Kept behind SPD_JET_BASE_WALL=1 as the A/B against
                //which the default was chosen.
                double sgn = (var == 1+dim || (mhd && var == 5+dim)) ? -1.0 : 1.0;
                indices(Nid,nid,k,j,i,kk,jj,ii,  1,  0,dim);
                BC.BoundaryL(t_id,var,k,j,i,kk,jj,ii) = sgn*U.Vector(INDICES);
            } else {
                //Outside the nozzle the low face is OUTFLOW: the plain
                //zeroth-order copy, exactly as define.hpp documents.
                //
                //It was briefly a reflecting WALL, on the reasoning that a jet
                //emerges from a nozzle in a solid surface. That is wrong here,
                //and not marginally: _reflective_ flips the NORMAL magnetic row
                //(5+dim), which is only the conducting-wall condition when
                //B.n = 0 on that wall. This jet's field is NORMAL to its base
                //with |B_y| = 141.42, so the flip puts a jump of 2*141.42 in the
                //normal field straight into the y-face Riemann problem -- a
                //div-B violation by construction, and HLLD is not well posed on
                //one.
                //
                //MEASURED on the AMBIENT CONTROL (the same deck with
                //problem/d1=d0 and problem/v2=0, so the quiescent ambient is an
                //EXACT stationary solution and any motion is boundary error),
                //100x150 DoF, PLM+RK2:
                //   wall outside the nozzle : dt COLLAPSED to 9.8e-07 of dt0 at
                //                             step 1404; max|v| 9.5e+03
                //   outflow outside nozzle  : 384 steps, max|v| = 2.8e-10
                //   whole base clamped      : 384 steps, max|v| = 1.5e-03
                //The wall's failure is an exponential instability seeded at
                //round-off: max|v| grows ~10x per 1.2e-4 of time from 2e-10,
                //always on the BOTTOM ROW and in the wall region (x ~ 0.9, far
                //from the nozzle at x ~ 0.5), with tangential B_x generated in
                //lockstep. Round-off in, 1e4 out.
                //
                //Outflow is also what the problem SPECIFIES: Wu & Shu (2018),
                //whom Balsara et al. 2025 section 8.2 defers to for the setup,
                //prescribe the fixed jet state on {y = 0, |x| < 0.05} "while
                //the other boundary conditions are outflow".
                //
                //The earlier verdict that a plain outflow base "sucks material
                //in" (mean v_y ~ 727 inward outside the nozzle) was REAL -- it
                //reproduces here at 435 -- but its cause is not the fluid
                //boundary. It was the inlet's normal FIELD being advanced from
                //an extrapolated EMF, which drove |B|^2 at the base to 2.17e+06
                //and over-pressured it. Pinning the inlet EMF
                //(mhd_pin_bc_emf_fv) takes that same number to 0.98 with the
                //base still a plain outflow. Fix the field condition, not the
                //fluid one.
                indices(Nid,nid,k,j,i,kk,jj,ii,  1,  0,dim);
                BC.BoundaryL(t_id,var,k,j,i,kk,jj,ii) = U.Vector(INDICES);
            }
            indices(Nid,nid,k,j,i,kk,jj,ii,N-2,n-1,dim);
            BC.BoundaryR(t_id,var,k,j,i,kk,jj,ii) = U.Vector(INDICES);
        }
        else if(type == _reflective_){
            //Mirror state at the wall: the ghost interface point carries the
            //interior interface value with the normal velocity (momentum)
            //component sign-flipped, so the Riemann problem at the wall sees
            //(U, mirror(U)) and returns zero mass/energy flux.
            //For MHD the NORMAL magnetic row flips too (rows 5..7 are Bx,By,Bz,
            //so the normal one is 5+dim): that is what makes B.n = 0 at a
            //perfectly conducting wall by antisymmetry, while the tangential
            //components stay mirrored. Both of the MDZ21 wall tests need this
            //parity -- the current sheet has B = B0 tanh(y/a) x^ with walls at
            //y = +-1/2, so the tangential field must survive the wall and the
            //normal field must vanish on it.
            double sgn = (var == 1+dim || (mhd && var == 5+dim)) ? -1.0 : 1.0;
            indices(Nid,nid,k,j,i,kk,jj,ii,  1,  0,dim);
            BC.BoundaryL(t_id,var,k,j,i,kk,jj,ii) = sgn*U.Vector(INDICES);
            indices(Nid,nid,k,j,i,kk,jj,ii,N-2,n-1,dim);
            BC.BoundaryR(t_id,var,k,j,i,kk,jj,ii) = sgn*U.Vector(INDICES);
        }
        #ifdef MPI
        BC.BufferL(t_id,var,k,j,i,kk,jj,ii) = value(U,t_id,var,k,j,i,kk,jj,ii,  1,  0,dim);
        BC.BufferR(t_id,var,k,j,i,kk,jj,ii) = value(U,t_id,var,k,j,i,kk,jj,ii,N-2,n-1,dim);
        #endif
        }}
    });
    #ifdef MPI
    exec_comm(BC, comm, comm.left, comm.right);
    #endif
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        for(int t_id=0; t_id<nader; t_id++){
        for(int var=0; var<nvar; var++){
        int Nid[3];
        int nid[3];
        indices(Nid,nid,k,j,i,kk,jj,ii,  0,n-1,dim);
        U.Vector(INDICES) = BC.BoundaryL(t_id,var,k,j,i,kk,jj,ii);
        indices(Nid,nid,k,j,i,kk,jj,ii,N-1,  0,dim);
        U.Vector(INDICES) = BC.BoundaryR(t_id,var,k,j,i,kk,jj,ii);
        }}
    });
}

//Fill the direction-dim interface ghost planes of U from the neighbor
//blocks UL (left) and UR (right), with the same semantics as the periodic
//exchange above: the ghost element's interface flux point receives the
//neighbor's last active interface value. A side of type _gradfree_ ignores
//the neighbor and copies the block's own first/last active value (domain
//edge); with periodic BCs the wrap is resolved by the caller's neighbor
//lookup, so both sides are plain neighbor copies.
void block_boundary_sd(
    SD_Solution U,
    SD_Solution UL,
    SD_Solution UR,
    int typeL,
    int typeR,
    int dim){
    int Nx = dim==_x_ ? 1 : U.Nx;
    int Ny = dim==_y_ ? 1 : U.Ny;
    int Nz = dim==_z_ ? 1 : U.Nz;
    int px = dim==_x_ ? 1 : U.nx;
    int py = dim==_y_ ? 1 : U.ny;
    int pz = dim==_z_ ? 1 : U.nz;
    int N = (dim==_x_ ? U.Nx : (dim==_y_ ? U.Ny : U.Nz));
    int n = (dim==_x_ ? U.nx : (dim==_y_ ? U.ny : U.nz));
    int nader = U.n_ader;
    int nvar  = U.n_var;
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        for(int t_id=0; t_id<nader; t_id++){
        for(int var=0; var<nvar; var++){
        int Nid[3];
        int nid[3];
        double v;
        //left ghost (element 0, interface point n-1)
        if(typeL == _gradfree_)
            v = value(U ,t_id,var,k,j,i,kk,jj,ii,  1,  0,dim);
        else
            v = value(UL,t_id,var,k,j,i,kk,jj,ii,N-2,n-1,dim);
        indices(Nid,nid,k,j,i,kk,jj,ii,  0,n-1,dim);
        U.Vector(INDICES) = v;
        //right ghost (element N-1, interface point 0)
        if(typeR == _gradfree_)
            v = value(U ,t_id,var,k,j,i,kk,jj,ii,N-2,n-1,dim);
        else
            v = value(UR,t_id,var,k,j,i,kk,jj,ii,  1,  0,dim);
        indices(Nid,nid,k,j,i,kk,jj,ii,N-1,  0,dim);
        U.Vector(INDICES) = v;
        }}
    });
}

//Face-staggered fields store the shared block-interface face in BOTH blocks'
//active region (left's last-active right face == right's first-active left
//face). Each block overwrites its left active face from the left neighbour so
//a single left-to-right sweep makes the two copies bit-identical after
//independent CT updates. Right-boundary / gradfree ends are left alone.
void sync_shared_face_sd(
    SD_Solution U,
    SD_Solution UL,
    SD_Solution UR,
    int typeL,
    int typeR,
    int dim){
    (void)UR; (void)typeR;
    if(typeL == _gradfree_) return;
    int Nx = dim==_x_ ? 1 : U.Nx;
    int Ny = dim==_y_ ? 1 : U.Ny;
    int Nz = dim==_z_ ? 1 : U.Nz;
    int px = dim==_x_ ? 1 : U.nx;
    int py = dim==_y_ ? 1 : U.ny;
    int pz = dim==_z_ ? 1 : U.nz;
    int N = (dim==_x_ ? U.Nx : (dim==_y_ ? U.Ny : U.Nz));
    int n = (dim==_x_ ? U.nx : (dim==_y_ ? U.ny : U.nz));
    int nader = U.n_ader;
    int nvar  = U.n_var;
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        for(int t_id=0; t_id<nader; t_id++){
        for(int var=0; var<nvar; var++){
        int Nid[3];
        int nid[3];
        double v = value(UL,t_id,var,k,j,i,kk,jj,ii,N-2,n-1,dim);
        indices(Nid,nid,k,j,i,kk,jj,ii,1,0,dim);
        U.Vector(INDICES) = v;
        }}
    });
}

//The same shared-face identity as sync_shared_face_sd above, over the whole pack:
//one launch per direction instead of one per BLOCK per direction.
//
//This was the last per-block host loop in the tree, and it was invisible to every
//AMR profile because Mesh only calls it when max_level == 0. On a UNIFORM
//multiblock mesh it dominated everything: fenced at 2048^2 in blocks of 16^2 it
//was 2.465 s of a 3.166 s total (77.8%) against 0.044 s in blocks of 128^2 -- 56x
//for the same total cells, from ~32k launches per stage at 16384 blocks.
//
//Both sides live in one pack, so the neighbour is a leading-axis offset rather
//than a second array, and the per-block side type is read on the device from the
//same typL table block_boundary_fv_b uses -- which is how the reference's
//`if(typeL == _gradfree_) return;` survives batching: at a gradfree boundary the
//block keeps its own face.
void sync_shared_face_sd_b(SD_Solution U, IntVector nbrL, IntVector typL, int dim){
    const int nb = U.nb;
    if(nb <= 0) return;
    const int Nx = dim==_x_ ? 1 : U.Nx;
    const int Ny = dim==_y_ ? 1 : U.Ny;
    const int Nz = dim==_z_ ? 1 : U.Nz;
    const int px = dim==_x_ ? 1 : U.nx;
    const int py = dim==_y_ ? 1 : U.ny;
    const int pz = dim==_z_ ? 1 : U.nz;
    const int N = (dim==_x_ ? U.Nx : (dim==_y_ ? U.Ny : U.Nz));
    const int n = (dim==_x_ ? U.nx : (dim==_y_ ? U.ny : U.nz));
    const int nader = U.n_ader, nvar = U.n_var;
    //Negative control for the side-type skip, same idea as SPD_BREAK_SDGATHER:
    //SPD_BREAK_SYNCB=1 copies across a gradfree boundary too. If a gradfree lane
    //is bit-identical WITH this set, the A/B never exercised the early return and
    //says nothing about it.
    static const bool ignore_type = getenv("SPD_BREAK_SYNCB") != nullptr;
    const int honour_type = ignore_type ? 0 : 1;
    sd_for_cells_b(nb,Nz,Ny,Nx,pz,py,px,
        KOKKOS_LAMBDA(int b, int k, int j, int i, int kk, int jj, int ii){
        if(honour_type && typL(b) == _gradfree_) return;
        const int boff  = b*nader;
        const int boffL = nbrL(b)*nader;
        int Nid[3], nid[3];
        for(int t_id=0; t_id<nader; t_id++)
        for(int var=0; var<nvar; var++){
            //Source: the left neighbour's last active element, last point.
            indices(Nid,nid,k,j,i,kk,jj,ii,N-2,n-1,dim);
            const double v = U.Vector(boffL+t_id,var,Nid[_z_],Nid[_y_],Nid[_x_],
                                                     nid[_z_],nid[_y_],nid[_x_]);
            //Destination: my first active element, first point.
            indices(Nid,nid,k,j,i,kk,jj,ii,1,0,dim);
            U.Vector(boff+t_id,var,Nid[_z_],Nid[_y_],Nid[_x_],
                                   nid[_z_],nid[_y_],nid[_x_]) = v;
        }
    }, "sync_shared_face_sd_b");
}

KOKKOS_INLINE_FUNCTION
void fv_indices(int* N_id, int k, int j, int i, int l, int dim){
    //Returns the indeces according to the dimension
    N_id[_x_] = dim == _x_ ? l  : i;
    N_id[_y_] = dim == _y_ ? l  : j;
    N_id[_z_] = dim == _z_ ? l  : k;
}

//Fill the direction-dim nGH ghost-cell layers of U from the neighbor blocks
//UL/UR (cell-centered alignment). Same layer semantics as the single-block
//exchange below: ghost layer l gets the neighbor's active layer nGH+l (from
//the facing side), or the block's own facing active layers for _gradfree_.
//Directions must be exchanged sequentially over all blocks (x, then y, then
//z) so corner ghosts propagate, exactly like the single-block path.
void block_boundary_fv(
    FV_Solution U,
    FV_Solution UL,
    FV_Solution UR,
    int typeL,
    int typeR,
    int dim){
    int Nx = dim==_x_ ? nGHx : U.Nx;
    int Ny = dim==_y_ ? nGHy : U.Ny;
    int Nz = dim==_z_ ? nGHz : U.Nz;
    int N  = (dim==_x_ ? U.Nx : (dim==_y_ ? U.Ny : U.Nz));
    int ngh = nGH_rt[dim];
    int nvar = U.n_var;
    fv_for_cells(Nz,Ny,Nx, KOKKOS_LAMBDA(int k, int j, int i){
        for(int var=0; var<nvar; var++){
        int Nid[3];
        double v;
        int l = (dim==_x_ ? i : (dim==_y_ ? j : k));
        //left ghost layer l
        if(typeL == _gradfree_){
            fv_indices(Nid,k,j,i,ngh+l,dim);
            v = U.Vector(FV_INDICES);
        }
        else{
            fv_indices(Nid,k,j,i,N-2*ngh+l,dim);
            v = UL.Vector(FV_INDICES);
        }
        fv_indices(Nid,k,j,i,l,dim);
        U.Vector(FV_INDICES) = v;
        //right ghost layer l
        if(typeR == _gradfree_){
            fv_indices(Nid,k,j,i,N-2*ngh+l,dim);
            v = U.Vector(FV_INDICES);
        }
        else{
            fv_indices(Nid,k,j,i,ngh+l,dim);
            v = UR.Vector(FV_INDICES);
        }
        fv_indices(Nid,k,j,i,N-ngh+l,dim);
        U.Vector(FV_INDICES) = v;
        }
    });
}

void block_boundary_fv_b(
    FV_Solution U,
    IntVector nbrL,
    IntVector nbrR,
    IntVector typL,
    IntVector typR,
    int dim){
    int nb = U.nb;
    int Nx = dim==_x_ ? nGHx : U.Nx;
    int Ny = dim==_y_ ? nGHy : U.Ny;
    int Nz = dim==_z_ ? nGHz : U.Nz;
    int N  = (dim==_x_ ? U.Nx : (dim==_y_ ? U.Ny : U.Nz));
    int ngh = nGH_rt[dim];
    int nvar = U.n_var;
    fv_for_cells_b(nb,Nz,Ny,Nx, KOKKOS_LAMBDA(int b, int k, int j, int i){
        const int boff  = b*nvar;
        const int boffL = nbrL(b)*nvar;
        const int boffR = nbrR(b)*nvar;
        const int tL = typL(b), tR = typR(b);
        for(int var=0; var<nvar; var++){
        int Nid[3];
        double v;
        int l = (dim==_x_ ? i : (dim==_y_ ? j : k));
        if(tL == _gradfree_){
            fv_indices(Nid,k,j,i,ngh+l,dim);
            v = U.Vector(boff+var,Nid[_z_],Nid[_y_],Nid[_x_]);
        }
        else{
            fv_indices(Nid,k,j,i,N-2*ngh+l,dim);
            v = U.Vector(boffL+var,Nid[_z_],Nid[_y_],Nid[_x_]);
        }
        fv_indices(Nid,k,j,i,l,dim);
        U.Vector(boff+var,Nid[_z_],Nid[_y_],Nid[_x_]) = v;
        if(tR == _gradfree_){
            fv_indices(Nid,k,j,i,N-2*ngh+l,dim);
            v = U.Vector(boff+var,Nid[_z_],Nid[_y_],Nid[_x_]);
        }
        else{
            fv_indices(Nid,k,j,i,ngh+l,dim);
            v = U.Vector(boffR+var,Nid[_z_],Nid[_y_],Nid[_x_]);
        }
        fv_indices(Nid,k,j,i,N-ngh+l,dim);
        U.Vector(boff+var,Nid[_z_],Nid[_y_],Nid[_x_]) = v;
        }
    }, "block_boundary_fv");
}

void boundaries(
    CommHelper comm,
    FV_Boundaries BC,
    FV_Solution U,
    int alignment,
    int a_dim
    ){
    //Alignment allows to reuse this function for staggered fields,
    //and to also reuse the same buffer, in a given direction, for
    //fields with different staggering (like Bx, By and Bz)  
    int dim = BC.dim;
    int shift = alignment*(a_dim==dim);
    int Nx = BC.Nx-(a_dim==_x_)*(alignment-shift);
    int Ny = BC.Ny-(a_dim==_y_)*(alignment-shift);
    int Nz = BC.Nz-(a_dim==_z_)*(alignment-shift);
    int type = BC.type;
    const bool wall_off_nozzle = jet_base_wall();
    int N = BC.N;
    //Runtime halo width: nGH_rt is the one source of truth for how many ghost
    //layers exist, and BC.Nx/Ny/Nz were sized from it. Reading the compile-time
    //nGH here would source the wrong interior layers as soon as the two differ.
    int ngh = nGH_rt[dim];

    int nvar  = U.n_var;
    fv_for_cells(Nz,Ny,Nx, KOKKOS_LAMBDA(int k, int j, int i){
        for(int var=0; var<nvar; var++){
        int Nid[3];
        int l;
        l = dim==_x_ ? i : (dim==_y_ ? j : k);
        if(type == _periodic_){
            fv_indices(Nid,k,j,i,N-2*ngh+l-shift,dim);
            BC.BoundaryL(var,k,j,i) = U.Vector(FV_INDICES);
            fv_indices(Nid,k,j,i,    ngh+l+shift,dim);
            BC.BoundaryR(var,k,j,i) = U.Vector(FV_INDICES);
        }
        else if(type == _gradfree_){
            fv_indices(Nid,k,j,i,    ngh+l+shift,dim);
            BC.BoundaryL(var,k,j,i) = U.Vector(FV_INDICES);
            fv_indices(Nid,k,j,i,N-2*ngh+l-shift,dim);
            BC.BoundaryR(var,k,j,i) = U.Vector(FV_INDICES);
        }
        else if(type == _outflow_){
            //See the SD branch: plain copy; the no-reentry clamp collapsed dt.
            fv_indices(Nid,k,j,i,    ngh+l+shift,dim);
            BC.BoundaryL(var,k,j,i) = U.Vector(FV_INDICES);
            fv_indices(Nid,k,j,i,N-2*ngh+l-shift,dim);
            BC.BoundaryR(var,k,j,i) = U.Vector(FV_INDICES);
        }
        else if(type == _inflow_){
            //See the SD branch above: prescribed where the problem injects,
            //outflow elsewhere on the low side, outflow on the high side.
            if(BC.InflowL(0,k,j,i) >= 0.0){
                BC.BoundaryL(var,k,j,i) = BC.InflowL(var,k,j,i);
            } else if(wall_off_nozzle){
                //Reference only, measured broken; see the SD branch.
                double sgn = (var == 1+dim) ? -1.0 : 1.0;
                fv_indices(Nid,k,j,i,2*ngh-1-l,dim);
                BC.BoundaryL(var,k,j,i) = sgn*U.Vector(FV_INDICES);
            } else {
                //Outflow outside the nozzle; see the SD branch for why a
                //reflecting wall is wrong when B.n != 0 on that wall, and for
                //the ambient-control numbers that decided it.
                fv_indices(Nid,k,j,i,    ngh+l+shift,dim);
                BC.BoundaryL(var,k,j,i) = U.Vector(FV_INDICES);
            }
            fv_indices(Nid,k,j,i,N-2*ngh+l-shift,dim);
            BC.BoundaryR(var,k,j,i) = U.Vector(FV_INDICES);
        }
        else if(type == _reflective_){
            //Mirror the first/last ngh interior cells across the wall with
            //the normal velocity (momentum) sign-flipped. Only used for
            //cell-centered fields (shift = 0); flag arrays (nvar = 1) are
            //mirrored without any sign change.
            double sgn = (var == 1+dim) ? -1.0 : 1.0;
            fv_indices(Nid,k,j,i,2*ngh-1-l,dim);
            BC.BoundaryL(var,k,j,i) = sgn*U.Vector(FV_INDICES);
            fv_indices(Nid,k,j,i,N-ngh-1-l,dim);
            BC.BoundaryR(var,k,j,i) = sgn*U.Vector(FV_INDICES);
        }
        #ifdef MPI
        BC.BufferL(t_id,var,k,j,i) = value(U,t_id,var,k,j,i,kk,jj,ii,  1,  0,dim);
        BC.BufferR(t_id,var,k,j,i) = value(U,t_id,var,k,j,i,kk,jj,ii,N-2,n-1,dim);
        #endif
        }
    });
    #ifdef MPI
    exec_comm(BC, comm, comm.left, comm.right);
    #endif
    fv_for_cells(Nz,Ny,Nx, KOKKOS_LAMBDA(int k, int j, int i){
        for(int var=0; var<nvar; var++){
        int Nid[3];
        int l;
        l = dim==_x_ ? i : (dim==_y_ ? j : k);
        fv_indices(Nid,k,j,i,      l,dim);
        U.Vector(FV_INDICES) = BC.BoundaryL(var,k,j,i);
        fv_indices(Nid,k,j,i,N-ngh+l,dim);
        U.Vector(FV_INDICES) = BC.BoundaryR(var,k,j,i);
        }
    });
}