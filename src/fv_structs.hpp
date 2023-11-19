using namespace std;

class FV_dimension{
    public:
        int N_global=1;
        int N=1;  //number of active elements
        int p=0;  //degree of spatial approximation
        int N_total=1; //total number of elements
        int n_cells=1; //number of cells
        int n_faces=2; //number of faces
        double L=1.0; //Box lenght
        double h=1.0; //element size
        Vector faces;
        Vector centers;
        FV_dimension(
            int N_cells_global,
            int N_cells,
            int start,
            double box_lenght,
            bool active){
            N_global = N_cells_global;
            L = box_lenght;
            h = L/N_global;
            N = N_cells;
            if(active){
                N_total = (N+2*NGH);
                n_cells = N_total;
                n_faces = n_cells+1;
            }
            Kokkos::resize(faces, n_faces);
            Kokkos::resize(centers, n_cells);
            for(int i=0;i<n_faces;i++)
                faces(i)= (start+i-NGH)*h;
            
            for(int i=0;i<n_cells;i++)
                centers(i)= 0.5*(faces(i+1)+faces(i)); 
  
        }	
};

class FV_Solution{
    public:
    FV_Vector Vector;
    #ifdef KOKKOS_ENABLE_CUDA
    FV_Vector_h Vector_h = Kokkos::create_mirror_view(Vector);
    #endif
    int Nx;
    int Ny;
    int Nz;
    int n_ader;
    int n_var;
    int iL;
    int iR;
    int jL;
    int jR;
    int kL;
    int kR;
    string label;
    FV_Solution() = default;
    FV_Solution(string name,
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
        n_ader=nader;
        n_var=nvar;
        Nx=Xdim.fv_cells+x;
        Ny=Ydim.fv_cells+y;
        Nz=Zdim.fv_cells+z;
        iL = Xdim.idL;
        iR = Xdim.idR;
        jL = Ydim.idL;
        jR = Ydim.idR;
        kL = Zdim.idL;
        kR = Zdim.idR;

        Kokkos::resize(Vector,nader,nvar,Nz,Ny,Nx);
        #ifdef KOKKOS_ENABLE_CUDA
        Kokkos::resize(Vector_h,nader,nvar,Nz,Ny,Nx);
        #endif
        label=name;
    }

    void copy(){
        #ifdef KOKKOS_ENABLE_CUDA
        Kokkos::deep_copy (Vector_h, Vector);
        #endif
    }
};

struct FV_Boundaries {
    int Nx;
    int Ny;
    int Nz;
    int nvar;
    int type;
    int N;
    int dim;
    FV_Vector BoundaryL;
    FV_Vector BoundaryR;
    #ifdef MPI
    FV_Vector BufferL;
    FV_Vector BufferR;
    #ifdef KOKKOS_ENABLE_CUDA
    FV_Vector::HostMirror BufferL_h = Kokkos::create_mirror_view(BufferL);
    FV_Vector::HostMirror BufferR_h = Kokkos::create_mirror_view(BufferR);
    FV_Vector::HostMirror BoundaryL_h = Kokkos::create_mirror_view(BoundaryL);
    FV_Vector::HostMirror BoundaryR_h = Kokkos::create_mirror_view(BoundaryR);
    #endif
    #endif
    FV_Boundaries() = default;
    FV_Boundaries(dimension Dim, int _type, int _nvar, int _Nz, int _Ny, int _Nx) {
        init(Dim, _type, _nvar, _Nz, _Ny, _Nx);
    }
    void init(dimension Dim, int _type, int _nvar, int _Nz, int _Ny, int _Nx) {
        type  = _type;
        nvar  = _nvar;
        Nz    = _Nz;
        Ny    = _Ny;
        Nx    = _Nx;
        N     = Dim.fv_cells;
        dim   = Dim.dim;
        Kokkos::resize(BoundaryL,1,nvar,Nz,Ny,Nx);
        Kokkos::resize(BoundaryR,1,nvar,Nz,Ny,Nx);
        #ifdef MPI
        Kokkos::resize(BufferL,1,nvar,Nz,Ny,Nx);
        Kokkos::resize(BufferR,1,nvar,Nz,Ny,Nx);
        #ifdef KOKKOS_ENABLE_CUDA
        Kokkos::resize(BufferL_h,1,nvar,Nz,Ny,Nx);
        Kokkos::resize(BufferR_h,1,nvar,Nz,Ny,Nx);
        Kokkos::resize(BoundaryL_h,1,nvar,Nz,Ny,Nx);
        Kokkos::resize(BoundaryR_h,1,nvar,Nz,Ny,Nx);
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