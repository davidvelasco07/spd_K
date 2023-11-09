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
    int n_var;
    string label;
    FV_Solution() = default;
    FV_Solution(string name,
        int nvar,
        dimension Zdim,
        dimension Ydim,
        dimension Xdim,
        bool z,
        bool y,
        bool x
        ){init(name,nvar,Zdim,Ydim,Xdim,z,y,x);}
    void init(string name,
        int nvar,
        dimension Zdim,
        dimension Ydim,
        dimension Xdim,
        bool z,
        bool y,
        bool x
        ){
        n_var=nvar;
        Nx=Xdim.N_total+x;
        Ny=Ydim.N_total+y;
        Nz=Zdim.N_total+z;

        Kokkos::resize(Vector,nvar,Nz,Ny,Nx);
        #ifdef KOKKOS_ENABLE_CUDA
        Kokkos::resize(Vector_h,nvar,Nz,Ny,Nx);
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
    FV_Boundaries(int nvar, int Nz, int Ny, int Nx) {
        init(nvar, Nz, Ny, Nx);
    }
    void init(int nvar, int Nz, int Ny, int Nx) {
        Kokkos::resize(BoundaryL,nvar,Nz,Ny,Nx);
        Kokkos::resize(BoundaryR,nvar,Nz,Ny,Nx);
        #ifdef MPI
        Kokkos::resize(BufferL,nvar,Nz,Ny,Nx);
        Kokkos::resize(BufferR,nvar,Nz,Ny,Nx);
        #ifdef KOKKOS_ENABLE_CUDA
        Kokkos::resize(BufferL_h,nvar,Nz,Ny,Nx);
        Kokkos::resize(BufferR_h,nvar,Nz,Ny,Nx);
        Kokkos::resize(BoundaryL_h,nvar,Nz,Ny,Nx);
        Kokkos::resize(BoundaryR_h,nvar,Nz,Ny,Nx);
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