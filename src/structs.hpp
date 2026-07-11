using namespace std;

template <typename T>
T* malloc_host(size_t N, T value=T()) {
    T* ptr = (T*)(malloc(N*sizeof(T)));
    std::fill(ptr, ptr+N, value);
    return ptr;
}

template <typename T>
void Write_arrays(T *array, int N, std::string name){
    std::ofstream output;
    output.open(name);
    output.write((char*)array, N*sizeof(T));
    output.close();
}

//This might end up being another header file
class dimension{
    public:
        int N_global=1;
        int N=1;  //number of active elements
        int p=0;  //degree of spatial approximation
        int N_total=1; //total number of elements
        int n_cells=1; //total number of cells
        int n_faces=1; //total number of faces
        int n_sp=1; //number of solution points
        int n_fp=1; //number of flux points
        int dim; //dimension index (x=0, y=1 or z=2)
        int fv_ncells=1;
        int fv_nfaces=1;
        int idL=0; //defaults valid for inactive dimensions
        int idR=1;
        double L=1.0; //Box lenght
        double h=1.0; //element size
        Matrix sd_faces;
        Matrix sd_centers;
        Vector fv_faces;
        Vector fv_centers;
        dimension(int _dim, int N_elements_global, int N_elements, int degree, int start, double box_lenght, double *x_fp, bool active){
            dim = _dim,
            N_global = N_elements_global;
            p = degree;
            L = box_lenght;
            h = L/N_global;
            N = N_elements;
            if(active){
                n_sp = p+1;
                n_fp = p+2;
                N_total = (N+2*NGH);
                n_cells = N_total*n_sp;
                n_faces = n_cells+1;
                fv_ncells = N*n_sp+2*nGH;
                fv_nfaces = fv_ncells+1;
                idL = n_sp-nGH;
                idR = fv_ncells+idL;
            }
            Kokkos::resize(sd_faces, N_total,n_fp);
            Kokkos::resize(sd_centers, N_total,n_sp);
            Kokkos::resize(fv_faces  , fv_nfaces);
            Kokkos::resize(fv_centers, fv_ncells);

            for(int j=0;j<N_total;j++){
                for(int i=0;i<n_fp;i++){
                    sd_faces(j,i)= (start+j-NGH + x_fp[i])*h;
                    if((i+j*n_sp)>=idL && (i+j*n_sp)<idR)
                        fv_faces(i-idL+j*n_sp) = sd_faces(j,i);
                }
                for(int i=0;i<n_sp;i++){
                    sd_centers(j,i)= 0.5*(sd_faces(j,i+1)+sd_faces(j,i));
                    if((i+j*n_sp)>=idL && (i+j*n_sp)<idR)
                        fv_centers(i-idL+j*n_sp) = sd_centers(j,i);
                }
            }
        }	
};

class SD_Solution{
    public:
    SD_Vector Vector;
    #ifdef KOKKOS_ENABLE_CUDA
    SD_Vector_h Vector_h = Kokkos::create_mirror_view(Vector);
    #endif
    int Nx;
    int Ny;
    int Nz;
    int nx;
    int ny;
    int nz;
    int n_ader=1;
    int n_var;
    string label;
    SD_Solution() = default;
    SD_Solution(string name,
        int nader,
        int nvar,
        dimension Zdim,
        dimension Ydim,
        dimension Xdim,
        bool z,
        bool y,
        bool x
        ){init(name,nader,nvar,Zdim,Ydim,Xdim,z,y,x);}
    void init(string name,
        int nader,
        int nvar,
        dimension Zdim,
        dimension Ydim,
        dimension Xdim,
        bool z,
        bool y,
        bool x
        ){
        n_var=nvar;
        n_ader=nader;
        Nx=Xdim.N_total;
        Ny=Ydim.N_total;
        Nz=Zdim.N_total;
        nx = ( x  ?  Xdim.n_fp : Xdim.n_sp);
        ny = ( y  ?  Ydim.n_fp : Ydim.n_sp);
        nz = ( z  ?  Zdim.n_fp : Zdim.n_sp);

        Kokkos::resize(Vector,n_ader,nvar,Nz,Ny,Nx,nz,ny,nx);
        #ifdef KOKKOS_ENABLE_CUDA
        Kokkos::resize(Vector_h,n_ader,nvar,Nz,Ny,Nx,nz,ny,nx);
        #endif
        //cout<<name<<":"<<n_ader<<","<<nvar<<","<<Nz<<","<<Ny<<","<<Nx<<","<<nz<<","<<ny<<","<<nx<<endl;
        label=name;
    }

    void copy(){
        #ifdef KOKKOS_ENABLE_CUDA
        Kokkos::deep_copy (Vector_h, Vector);
        #endif
    }

    KOKKOS_INLINE_FUNCTION
    double value(int t_id, int var, int i, int j, int k, int ii, int jj, int kk, int l, int ll, int dim){
        if(dim==0)
            return Vector(t_id,var,k,j,l,kk,jj,ll);
        else if(dim==1)
            return Vector(t_id,var,k,l,i,kk,ll,ii);
        else if(dim==2)
            return Vector(t_id,var,l,j,i,ll,jj,ii);
        else
            return 0;
    }

    //KOKKOS_INLINE_FUNCTION
    //tuple<int,int,int,int,int,int,int,int> indeces(int t_id, int var, int i, int j, int k, int ii, int jj, int kk, int l, int ll, int dim){
    //    if(dim==0)
    //        return make_tuple(t_id,var,k,j,l,kk,jj,ll);
    //    else if(dim==1)
    //        return make_tuple(t_id,var,k,l,i,kk,ll,ii);
    //    else if(dim==2)
    //        return make_tuple(t_id,var,l,j,i,ll,jj,ii);
    //}
};

struct CommHelper {
  #ifdef MPI
  MPI_Comm comm;
  #endif
  // Num MPI ranks in each dimension
  int nx, ny, nz;

  // My rank
  int me;

  // My pos in proc grid
  int x, y, z;

  // Neighbor Ranks
  int up,down,left,right,front,back;
    
  CommHelper(
    #ifdef MPI
    MPI_Comm comm_
    #endif
    ){
    #ifdef MPI
    comm = comm_;
    int nranks;
    MPI_Comm_size(comm, &nranks);
    MPI_Comm_rank(comm, &me);
    #else
    int nranks=1;
    me =0;
    #endif
    nx = std::pow(1.0*nranks,1.0/3.0);
    while(nranks%nx != 0) nx++;

    ny = std::sqrt(1.0*(nranks/nx));
    while((nranks/nx)%ny != 0) ny++;
    
    nz = nranks/nx/ny;
    x = me%nx;
    y = (me/nx)%ny;
    z = (me/nx/ny);

    left  = nx==1 ? -1 : (x==0    ? nx-1:me-1);
    right = nx==1 ? -1 : (x==nx-1 ? 0:me+1);
    down  = ny==1 ? -1 : (y==0    ? ny*nx-nx:me-nx);
    up    = ny==1 ? -1 : (y==ny-1 ? 0:me+nx);
    front = nz==1 ? -1 : (z==0    ? nx*ny*nz-nx*ny:me-nx*ny);
    back  = nz==1 ? -1 : (z==nz-1 ? 0:me+nx*ny);

    printf("NumRanks: %i Me: %i Grid: %i %i %i MyPos: %i %i %i\n",nranks,me,nx,ny,nz,x,y,z);
    printf("Me: %i MyNeighs: %i %i %i %i %i %i\n",me,left,right,down,up,front,back);
  }
  #ifdef MPI
  template<class ViewType>
  void isend_irecv(int partner, ViewType send_buffer, ViewType recv_buffer, MPI_Request* request_send, MPI_Request* request_recv) {
    MPI_Irecv(recv_buffer.data(), recv_buffer.size(), MPI_DOUBLE, partner, 1, comm, request_recv);
    MPI_Isend(send_buffer.data(), send_buffer.size(), MPI_DOUBLE, partner, 1, comm, request_send);
  }
  #endif
};

struct Boundaries {
    int Nx;
    int Ny;
    int Nz;
    int nx;
    int ny;
    int nz;
    int nader;
    int nvar;
    int type;
    int N;
    int n;
    int dim;
    SD_Vector BoundaryL;
    SD_Vector BoundaryR;
    #ifdef MPI
    SD_Vector BufferL;
    SD_Vector BufferR;
    #ifdef KOKKOS_ENABLE_CUDA
    SD_Vector::host_mirror_type BufferL_h = Kokkos::create_mirror_view(BufferL);
    SD_Vector::host_mirror_type BufferR_h = Kokkos::create_mirror_view(BufferR);
    SD_Vector::host_mirror_type BoundaryL_h = Kokkos::create_mirror_view(BoundaryL);
    SD_Vector::host_mirror_type BoundaryR_h = Kokkos::create_mirror_view(BoundaryR);
    #endif
    #endif
    Boundaries() = default;
    Boundaries(dimension Dim, int _type, int _nader, int _nvar, int _Nz, int _Ny, int _Nx, int _nz, int _ny, int _nx) {
        init(Dim, _type, _nader, _nvar, _Nz, _Ny, _Nx, _nz, _ny, _nx);
    }
    void init(dimension Dim, int _type, int _nader, int _nvar, int _Nz, int _Ny, int _Nx, int _nz, int _ny, int _nx) {
        type  = _type;
        nader = _nader;
        nvar  = _nvar;
        Nz    = _Nz;
        Ny    = _Ny;
        Nx    = _Nx;
        nz    = _nz;
        ny    = _ny;
        nx    = _nx;
        n     = Dim.n_fp;
        N     = Dim.N_total;
        dim   = Dim.dim;
        Kokkos::resize(BoundaryL,nader,nvar,Nz,Ny,Nx,nz,ny,nx);
        Kokkos::resize(BoundaryR,nader,nvar,Nz,Ny,Nx,nz,ny,nx);
        #ifdef MPI
        Kokkos::resize(BufferL,nader,nvar,Nz,Ny,Nx,nz,ny,nx);
        Kokkos::resize(BufferR,nader,nvar,Nz,Ny,Nx,nz,ny,nx);
        #ifdef KOKKOS_ENABLE_CUDA
        Kokkos::resize(BufferL_h,nader,nvar,Nz,Ny,Nx,nz,ny,nx);
        Kokkos::resize(BufferR_h,nader,nvar,Nz,Ny,Nx,nz,ny,nx);
        Kokkos::resize(BoundaryL_h,nader,nvar,Nz,Ny,Nx,nz,ny,nx);
        Kokkos::resize(BoundaryR_h,nader,nvar,Nz,Ny,Nx,nz,ny,nx);
        #endif
        #endif
    }

    #ifdef MPI
    void isend_irecv_L(CommHelper comm, int partner, MPI_Request* request_send, MPI_Request* request_recv) {
        #ifdef KOKKOS_ENABLE_CUDA
        Kokkos::deep_copy (BufferL_h, BufferL);
        comm.isend_irecv(partner,BufferL_h,BoundaryL_h, request_send, request_recv);
        #else
        comm.isend_irecv(partner,BufferL,BoundaryL, request_send, request_recv);
        #endif
    }

    void isend_irecv_R(CommHelper comm, int partner, MPI_Request* request_send, MPI_Request* request_recv) {
        #ifdef KOKKOS_ENABLE_CUDA
        Kokkos::deep_copy (BufferR_h, BufferR);
        comm.isend_irecv(partner,BufferR_h,BoundaryR_h, request_send, request_recv);
        #else
        comm.isend_irecv(partner,BufferR,BoundaryR, request_send, request_recv);
        #endif
    }
    #endif
};