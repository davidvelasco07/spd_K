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
            }else{
                n_cells = 1;
                n_faces = 2;
            }
            int nf = n_faces;
            int nc = n_cells;
            if(!active){
                if(nf < 2) nf = 2;
                if(nc < 2) nc = 2;
            }
            faces = Vector("faces", nf);
            centers = Vector("centers", nc);
            Vector_h faces_h = setup_mirror(faces);
            Vector_h centers_h = setup_mirror(centers);
            for(int i=0;i<n_faces;i++)
                faces_h(i)= (start+i-NGH)*h;
            
            for(int i=0;i<n_cells;i++)
                centers_h(i)= 0.5*(faces_h(i+1)+faces_h(i));
            setup_push(faces, faces_h);
            setup_push(centers, centers_h);
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
    int iL;
    int iR;
    int jL;
    int jR;
    int kL;
    int kR;
    char label[64];
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
        Nx=Xdim.fv_ncells+x;
        Ny=Ydim.fv_ncells+y;
        Nz=Zdim.fv_ncells+z;
        iL = Xdim.idL;
        iR = Xdim.idR;
        jL = Ydim.idL;
        jR = Ydim.idR;
        kL = Zdim.idL;
        kR = Zdim.idR;

        Kokkos::resize(Vector,nvar,Nz,Ny,Nx);
        snprintf(label, sizeof(label), "%s", name.c_str());
    }

    //Lazy host mirror: allocated on first copy() (see SD_Solution::copy)
    void copy(){
        #ifdef KOKKOS_ENABLE_CUDA
        if(Vector_h.size() != Vector.size())
            Vector_h = Kokkos::create_mirror_view(Vector);
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
    FV_Vector::host_mirror_type BufferL_h = Kokkos::create_mirror_view(BufferL);
    FV_Vector::host_mirror_type BufferR_h = Kokkos::create_mirror_view(BufferR);
    FV_Vector::host_mirror_type BoundaryL_h = Kokkos::create_mirror_view(BoundaryL);
    FV_Vector::host_mirror_type BoundaryR_h = Kokkos::create_mirror_view(BoundaryR);
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
        N     = Dim.fv_ncells;
        dim   = Dim.dim;
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