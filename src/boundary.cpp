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
    int type = BC.type;
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

KOKKOS_INLINE_FUNCTION
void fv_indices(int* N_id, int k, int j, int i, int l, int dim){
    //Returns the indeces according to the dimension
    N_id[_x_] = dim == _x_ ? l  : i;
    N_id[_y_] = dim == _y_ ? l  : j;
    N_id[_z_] = dim == _z_ ? l  : k;
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
    int N = BC.N;
    
    int nvar  = U.n_var;
    fv_for_cells(Nz,Ny,Nx, KOKKOS_LAMBDA(int k, int j, int i){
        for(int var=0; var<nvar; var++){
        int Nid[3];
        int l;
        l = dim==_x_ ? i : (dim==_y_ ? j : k);
        if(type == _periodic_){ 
            fv_indices(Nid,k,j,i,N-2*nGH+l-shift,dim);
            BC.BoundaryL(var,k,j,i) = U.Vector(FV_INDICES);
            fv_indices(Nid,k,j,i,    nGH+l+shift,dim);
            BC.BoundaryR(var,k,j,i) = U.Vector(FV_INDICES);
        }
        else if(type == _gradfree_){
            fv_indices(Nid,k,j,i,    nGH+l+shift,dim);
            BC.BoundaryL(var,k,j,i) = U.Vector(FV_INDICES);
            fv_indices(Nid,k,j,i,N-2*nGH+l-shift,dim);
            BC.BoundaryR(var,k,j,i) = U.Vector(FV_INDICES);
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
        fv_indices(Nid,k,j,i,N-nGH+l,dim);
        U.Vector(FV_INDICES) = BC.BoundaryR(var,k,j,i);
        }
    });
}