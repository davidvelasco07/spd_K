#ifndef STRUCTS_HPP_
#define STRUCTS_HPP_

#include <map>
#include <string>

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
        dimension() = default;
        dimension(int _dim, int N_elements_global, int N_elements, int degree, int start, double box_lenght, double *x_fp, bool active){
            dim = _dim,
            N_global = N_elements_global;
            p = degree;
            L = box_lenght;
            h = L/N_global;
            N = N_elements;
            //Ghost widths come from the runtime globals, never from the
            //compile-time NGH/nGH: the exchange and the physical BCs fill
            //NGH_rt/nGH_rt layers, and sizing the arrays from anything else
            //lets a widened halo write over interior cells.
            const int NG = NGH_rt[dim];
            const int g  = nGH_rt[dim];
            if(active){
                n_sp = p+1;
                n_fp = p+2;
                N_total = (N+2*NG);
                n_cells = N_total*n_sp;
                n_faces = n_cells+1;
                fv_ncells = N*n_sp+2*g;
                fv_nfaces = fv_ncells+1;
                //FV cell c is SD solution point c+idL, so the first active FV
                //cell (index g) lands on the first active SD point (NG*n_sp).
                idL = NG*n_sp-g;
                idR = fv_ncells+idL;
            }else{
                n_sp = 1;
                n_fp = 2;
            }
            int sd_nrows = N_total, sd_ncols = n_fp, sd_spcols = n_sp;
            int fv_nf = fv_nfaces, fv_nc = fv_ncells;
            if(!active){
                if(sd_nrows < 2) sd_nrows = 2;
                if(sd_ncols < 2) sd_ncols = 2;
                if(sd_spcols < 2) sd_spcols = 2;
                if(fv_nf < 2) fv_nf = 2;
                if(fv_nc < 2) fv_nc = 2;
            }
            sd_faces = Matrix("sd_faces", sd_nrows, sd_ncols);
            sd_centers = Matrix("sd_centers", sd_nrows, sd_spcols);
            fv_faces = Vector("fv_faces", fv_nf);
            fv_centers = Vector("fv_centers", fv_nc);
            Matrix_h sd_faces_h = setup_mirror(sd_faces);
            Matrix_h sd_centers_h = setup_mirror(sd_centers);
            Vector_h fv_faces_h = setup_mirror(fv_faces);
            Vector_h fv_centers_h = setup_mirror(fv_centers);

            std::vector<char> got_f(fv_nf,0), got_c(fv_nc,0);
            for(int j=0;j<N_total;j++){
                for(int i=0;i<n_fp;i++){
                    sd_faces_h(j,i)= (start+j-NG + x_fp[i])*h;
                    if((i+j*n_sp)>=idL && (i+j*n_sp)<idR){
                        fv_faces_h(i-idL+j*n_sp) = sd_faces_h(j,i);
                        got_f[i-idL+j*n_sp] = 1;
                    }
                }
                for(int i=0;i<n_sp;i++){
                    sd_centers_h(j,i)= 0.5*(sd_faces_h(j,i+1)+sd_faces_h(j,i));
                    if((i+j*n_sp)>=idL && (i+j*n_sp)<idR){
                        fv_centers_h(i-idL+j*n_sp) = sd_centers_h(j,i);
                        got_c[i-idL+j*n_sp] = 1;
                    }
                }
            }
            //The FV sub-grid keeps g ghost cells per side, but the SD grid only
            //supplies NG ghost elements of n_sp points each. For g <= NG*n_sp
            //every FV ghost has an SD source; past that idL < 0 and the
            //outermost ghost coordinates have none and would stay zero. Two
            //ways in: p = 0 (job/scheme=vl2|plm) has n_sp = 1 < g at the default
            //width, and widening the halo (SPD_FV_GHOST) does it at any p.
            //slopes_d divides by the centre spacing there, so the boundary
            //ghost slope -- and with it the flux on the two domain boundary
            //faces -- comes out wrong and asymmetric, which breaks
            //conservation. The sub-grid repeats every n_sp cells with element
            //size h, so extend it by whole elements. Only the coordinates need
            //this: no ghost cell takes its *data* from SD (fv_update_solution
            //writes active cells only; the exchange fills the ghosts).
            if(active){
                auto extend = [&](Vector_h v, std::vector<char>& got, int n){
                    for(int d=n-1; d>=0; d--)
                        if(!got[d] && d+n_sp < n){ v(d) = v(d+n_sp) - h; got[d] = 1; }
                    for(int d=0; d<n; d++)
                        if(!got[d] && d-n_sp >= 0){ v(d) = v(d-n_sp) + h; got[d] = 1; }
                };
                extend(fv_faces_h, got_f, fv_nf);
                extend(fv_centers_h, got_c, fv_nc);
            }
            setup_push(sd_faces, sd_faces_h);
            setup_push(sd_centers, sd_centers_h);
            setup_push(fv_faces, fv_faces_h);
            setup_push(fv_centers, fv_centers_h);
        }	
};

//The spectral-difference operators are functions of the degree p and the
//reference node sets alone, so every block in a run needs the SAME ones.
//Building them per block cost 12 device Views + 12 host mirrors plus the whole
//host polynomial build (lagrange / lagrange_prime / integral / inverse /
//ader_matrix) for each of nblocks -- and build_block_solvers() redid all of it
//from scratch on every adapt, which is a large part of why a regrid was
//allocation-bound rather than physics-bound. Build one set per run and alias it
//into every block: these arrays are read-only after construction (the only
//writes are inside the builders themselves), so a Matrix copy is a refcount
//bump, not a copy of the data.
struct SDOperators {
    Vector xt, wt;   //temporal nodes/weights: GL (p+1) for ADER, {0}/{1} for RK
    Vector xx, wx;   //spatial GL quadrature (control-volume averages of the ICs)
    Matrix sp_to_fp, fp_to_sp, dfp_to_sp;
    Matrix sp_to_cv, cv_to_sp, fp_to_cv;
    Matrix ader, invader;   //ADER only; left empty under RK
    int n_ader = 1;
    int n_stages = 1;
    double rk_a[3] = {0.0, 0.0, 0.0};
    bool built = false;
};

//Packed multi-meshblock storage (the AthenaK layout).
//
//Every meshblock of a mesh carries the same element count whatever its
//refinement level, so the per-block arrays of a given name are rectangular
//and can live in ONE allocation indexed by a leading block axis. A block's
//own array is then a range subview of that allocation: still LayoutRight,
//still contiguous, and assignable to the plain per-block view type, so all
//per-block code (ghost exchange, AMR transfer, output) is unchanged while
//batched kernels can span every block in a single launch.
//
//Kokkos caps views at rank 8 and the SD arrays already use all 8, so the
//block axis is folded into the leading extent instead of added: an SD pack
//has leading extent nb*n_ader and is indexed [b*n_ader + t_id], an FV pack
//has leading extent nb*n_var and is indexed [b*n_var + var]. The INDICES
//macros add that offset, so kernel bodies are otherwise untouched.
struct PackMeta {
    int nb=1, nader=1, nvar=1;
    int Nz=1, Ny=1, Nx=1, nz=1, ny=1, nx=1;
    int iL=0, iR=1, jL=0, jR=1, kL=0, kR=1;  //FV sub-grid bounds
};

struct BlockPack {
    int nb = 1;            //number of meshblocks in the pack
    bool active = false;   //false => callers allocate per-block as before
    std::map<std::string, SD_Vector> sd;
    std::map<std::string, FV_Vector> fv;
    std::map<std::string, PackMeta>  meta;

    void reset(int n){ nb=n; active=true; sd.clear(); fv.clear(); meta.clear(); }

    //Allocate the pack on first request, then hand out block ib's slice.
    SD_Vector sd_slice(const std::string& name, int ib, int nader, int nvar,
                       int Nz, int Ny, int Nx, int nz, int ny, int nx){
        auto it = sd.find(name);
        if(it == sd.end()){
            it = sd.emplace(name,
                SD_Vector(name, nb*nader, nvar, Nz, Ny, Nx, nz, ny, nx)).first;
            meta[name] = PackMeta{nb,nader,nvar,Nz,Ny,Nx,nz,ny,nx};
        }
        return Kokkos::subview(it->second,
            Kokkos::make_pair(ib*nader,(ib+1)*nader),
            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL, Kokkos::ALL,
            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
    }

    FV_Vector fv_slice(const std::string& name, int ib, int nvar,
                       int Nz, int Ny, int Nx){
        auto it = fv.find(name);
        if(it == fv.end()){
            it = fv.emplace(name,
                FV_Vector(name, nb*nvar, Nz, Ny, Nx)).first;
            meta[name] = PackMeta{nb,1,nvar,Nz,Ny,Nx,1,1,1};
        }
        return Kokkos::subview(it->second,
            Kokkos::make_pair(ib*nvar,(ib+1)*nvar),
            Kokkos::ALL, Kokkos::ALL, Kokkos::ALL);
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
    int nb=1;   //meshblocks spanned: 1 for a block's own view, nblocks for a pack
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
    void set_extents(int nader, int nvar,
        dimension Zdim, dimension Ydim, dimension Xdim,
        bool z, bool y, bool x
        ){
        n_var=nvar;
        n_ader=nader;
        Nx=Xdim.N_total;
        Ny=Ydim.N_total;
        Nz=Zdim.N_total;
        nx = ( x  ?  Xdim.n_fp : Xdim.n_sp);
        ny = ( y  ?  Ydim.n_fp : Ydim.n_sp);
        nz = ( z  ?  Zdim.n_fp : Zdim.n_sp);
    }

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
        set_extents(nader,nvar,Zdim,Ydim,Xdim,z,y,x);
        Kokkos::resize(Vector,n_ader,nvar,Nz,Ny,Nx,nz,ny,nx);
        label=name;
    }

    //Same shape as init(), but the storage is block ib's slice of a shared
    //pack rather than a private allocation. Bitwise-identical arithmetic:
    //only where the data lives changes.
    void init_packed(BlockPack& pk, int ib, string name,
        int nader, int nvar,
        dimension Zdim, dimension Ydim, dimension Xdim,
        bool z, bool y, bool x
        ){
        set_extents(nader,nvar,Zdim,Ydim,Xdim,z,y,x);
        Vector = pk.sd_slice(name, ib, n_ader, n_var, Nz, Ny, Nx, nz, ny, nx);
        label = name;
    }

    //The host mirror is allocated lazily on the first copy(): only arrays that
    //are actually written to disk pay the host-memory cost of a mirror
    void copy(){
        #ifdef KOKKOS_ENABLE_CUDA
        if(Vector_h.size() != Vector.size())
            Vector_h = Kokkos::create_mirror_view(Vector);
        Kokkos::deep_copy (Vector_h, Vector);
        #endif
    }

    //Slowest-to-fastest (k,j,i / kk,jj,ii) argument order, matching the free
    //value() in boundary.cpp and the sd_for_cells lambda signature. The body
    //places k in the z slot, so the transposed (i,j,k) order silently indexes
    //z with an x index -- out of bounds whenever z is inactive.
    KOKKOS_INLINE_FUNCTION
    double value(int t_id, int var, int k, int j, int i, int kk, int jj, int ii, int l, int ll, int dim) const{
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

//Descriptor for a whole pack, shaped like a per-block SD_Solution but with
//nb set to the block count and Vector spanning every block. Batched kernels
//take one of these and index the leading axis as b*n_ader + t_id.
inline SD_Solution sd_pack_view(BlockPack& pk, const std::string& name){
    SD_Solution s;
    auto it = pk.sd.find(name);
    if(it == pk.sd.end()) return s;
    const PackMeta& m = pk.meta.at(name);
    s.Vector = it->second;
    s.nb = m.nb; s.n_ader = m.nader; s.n_var = m.nvar;
    s.Nz = m.Nz; s.Ny = m.Ny; s.Nx = m.Nx;
    s.nz = m.nz; s.ny = m.ny; s.nx = m.nx;
    s.label = name;
    return s;
}

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

#endif