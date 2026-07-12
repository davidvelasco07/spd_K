//Layout micro-benchmark: measures the achieved memory bandwidth of the two
//dominant kernel patterns (a pure stream and a 1d interpolation sweep) for
//several stride orderings of the 8d solution arrays, using LayoutStride so
//no kernel code changes are needed. The thread-index decomposition follows
//the same ordering as the layout (fastest thread index = stride-1
//dimension), as the production loop helpers do.
//
//Usage: bench_layout [N_elements] [p]   (defaults: 64 3, i.e. 256^3 cells)
#include <Kokkos_Core.hpp>
#include <cstdio>
#include <string>

#ifdef KOKKOS_ENABLE_CUDA
#define MemSpace Kokkos::CudaSpace
#else
#define MemSpace Kokkos::HostSpace
#endif

using SView = Kokkos::View<double********, Kokkos::LayoutStride, MemSpace>;
using Perm = Kokkos::Array<int,8>;
using Ext  = Kokkos::Array<int,8>;

SView alloc_view(const char* name, const Ext& ext, const Perm& perm){
    //perm[r] = dimension holding the r-th smallest stride
    int64_t stride[8];
    int64_t cur = 1;
    for(int r=0; r<8; r++){
        stride[perm[r]] = cur;
        cur *= ext[perm[r]];
    }
    Kokkos::LayoutStride ls(ext[0],stride[0], ext[1],stride[1],
                            ext[2],stride[2], ext[3],stride[3],
                            ext[4],stride[4], ext[5],stride[5],
                            ext[6],stride[6], ext[7],stride[7]);
    return SView(Kokkos::view_alloc(std::string(name), Kokkos::WithoutInitializing), ls);
}

//Decompose a flat thread index following the layout ordering
KOKKOS_INLINE_FUNCTION
void decompose(unsigned idx, const Ext& ext, const Perm& perm, int* id){
    for(int r=0; r<8; r++){
        int d = perm[r];
        id[d] = int(idx % unsigned(ext[d]));
        idx  /= unsigned(ext[d]);
    }
}

//Same, but only over the six spatial dims (t and var handled by the caller),
//still following the layout ordering. Mirrors the production loop helpers,
//which parallelize space and loop t/var inside the thread.
KOKKOS_INLINE_FUNCTION
void decompose_space(unsigned idx, const Ext& ext, const Perm& perm, int* id){
    for(int r=0; r<8; r++){
        int d = perm[r];
        if(d<2) continue;
        id[d] = int(idx % unsigned(ext[d]));
        idx  /= unsigned(ext[d]);
    }
}

double time_kernel(int reps, const std::function<void()>& body){
    body(); body();                    //warmup
    Kokkos::fence();
    Kokkos::Timer t;
    for(int r=0; r<reps; r++) body();
    Kokkos::fence();
    return t.seconds()/reps;
}

int main(int argc, char** argv){
    Kokkos::initialize(argc, argv);
    {
        int N = argc>1 ? std::atoi(argv[1]) : 64;
        int p = argc>2 ? std::atoi(argv[2]) : 3;
        int Ne = N+2;                  //with ghost elements
        int ns = p+1, nf = p+2;
        int nvar = 6;

        Ext ext_sp = {1,nvar,Ne,Ne,Ne,ns,ns,ns};
        Ext ext_fp = {1,nvar,Ne,Ne,Ne,ns,ns,nf};

        //Interpolation matrix (managed memory: tiny, read-only)
        Kokkos::View<double**,
            #ifdef KOKKOS_ENABLE_CUDA
            Kokkos::CudaUVMSpace
            #else
            Kokkos::HostSpace
            #endif
            > M("M",nf,ns);
        for(int a=0; a<nf; a++)
            for(int b=0; b<ns; b++)
                M(a,b) = 0.25*(a+1)/(b+1);

        struct Cand { const char* name; Perm perm; };
        //dims: 0=t 1=var 2=k 3=j 4=i 5=kk 6=jj 7=ii  (perm = fastest first)
        Cand cands[] = {
            {"A current (t,var,k,j,i fast)", {0,1,2,3,4,5,6,7}},
            {"B right   (ii,jj,kk,i,j,k)  ", {7,6,5,4,3,2,1,0}},
            {"C x-contig (ii,i | jj,j |..)", {7,4,6,3,5,2,1,0}},
            {"D var,ii,jj,kk then i,j,k   ", {1,7,6,5,4,3,2,0}},
            {"E ii,i,var then jj,j,kk,k   ", {7,4,1,6,3,5,2,0}},
            {"F t,var,ii,jj,kk then i,j,k ", {0,1,7,6,5,4,3,2}},
            {"G t,var,kk,jj,ii then k,j,i ", {0,1,5,6,7,2,3,4}},
        };

        int64_t n_sp=1, n_fp=1;
        for(int d=0; d<8; d++){ n_sp*=ext_sp[d]; n_fp*=ext_fp[d]; }
        double gb_stream = 2.0*n_sp*8/1e9;          //read + write
        double gb_sweep  = (n_sp+n_fp)*8/1e9;       //read sp + write fp

        //Spin the GPU clocks up from idle before timing anything, otherwise
        //the first candidates run at reduced frequency
        {
            Kokkos::View<double*,MemSpace> W("warmup",n_sp);
            for(int r=0; r<40; r++)
                Kokkos::parallel_for("warmup",Kokkos::RangePolicy<>(0,n_sp),
                    KOKKOS_LAMBDA(int64_t idx){ W(idx) = 1.0001*W(idx)+1e-9; });
            Kokkos::fence();
        }

        printf("N=%d p=%d: sp %.2f GB, fp %.2f GB\n",
               N,p,n_sp*8/1e9,n_fp*8/1e9);
        printf("%-32s %12s %12s %12s\n",
               "layout","stream GB/s","varloop GB/s","sweep GB/s");

        for(const auto& c : cands){
            SView A = alloc_view("A",ext_sp,c.perm);
            SView B = alloc_view("B",ext_sp,c.perm);
            SView F = alloc_view("F",ext_fp,c.perm);
            Kokkos::deep_copy(A,1.0);

            Ext es=ext_sp, ef=ext_fp; Perm pm=c.perm;
            int q=ns;

            int nv = nvar;
            int64_t n_space = n_sp/nvar;
            //Production-style stream: parallelize space only, loop var
            //inside the thread (as the sd_for_cells helpers do)
            auto stream_varloop = [&](){
                Kokkos::parallel_for("stream_varloop",
                    Kokkos::RangePolicy<Kokkos::IndexType<unsigned>>(0,unsigned(n_space)),
                    KOKKOS_LAMBDA(unsigned idx){
                        int id[8]; id[0]=0;
                        decompose_space(idx,es,pm,id);
                        for(int v=0; v<nv; v++)
                            B(0,v,id[2],id[3],id[4],id[5],id[6],id[7]) =
                                1.0001*A(0,v,id[2],id[3],id[4],id[5],id[6],id[7]);
                    });
            };
            //Pure stream: B = 1.0001*A
            auto stream = [&](){
                Kokkos::parallel_for("stream",
                    Kokkos::RangePolicy<Kokkos::IndexType<unsigned>>(0,unsigned(n_sp)),
                    KOKKOS_LAMBDA(unsigned idx){
                        int id[8];
                        decompose(idx,es,pm,id);
                        B(id[0],id[1],id[2],id[3],id[4],id[5],id[6],id[7]) =
                            1.0001*A(id[0],id[1],id[2],id[3],id[4],id[5],id[6],id[7]);
                    });
            };
            //1d interpolation sweep along x: F(..,f) = sum_l M(f,l) A(..,l)
            auto sweep = [&](){
                Kokkos::parallel_for("sweep",
                    Kokkos::RangePolicy<Kokkos::IndexType<unsigned>>(0,unsigned(n_fp)),
                    KOKKOS_LAMBDA(unsigned idx){
                        int id[8];
                        decompose(idx,ef,pm,id);
                        double u=0;
                        for(int l=0; l<q; l++)
                            u += M(id[7],l)*A(id[0],id[1],id[2],id[3],id[4],id[5],id[6],l);
                        F(id[0],id[1],id[2],id[3],id[4],id[5],id[6],id[7]) = u;
                    });
            };

            double ts = time_kernel(10,stream);
            double tv = time_kernel(10,stream_varloop);
            double tw = time_kernel(10,sweep);
            printf("%-32s %12.0f %12.0f %12.0f\n",
                   c.name,gb_stream/ts,gb_stream/tv,gb_sweep/tw);
        }
    }
    Kokkos::finalize();
    return 0;
}
