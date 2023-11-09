#include "spd_k.hpp"

void Write(SD_Solution U, int n){
  int p = U.nx-1;
  int N = U.Nx-2*NGH;
  string filename =  "../outputs/"+U.label+"_N"+to_string(N)+"p"+to_string(p)+"_"+to_string(n)+"_"+to_string(cpu_rank)+".dat";
  //cout<<N<<","<<p<<endl;
  #ifdef KOKKOS_ENABLE_CUDA
  U.copy();
  Write_arrays(U.Vector_h.data(), U.Vector_h.size(), filename);
  #else
  Write_arrays(U.Vector.data(), U.Vector.size(), filename);
  #endif
  Write_arrays(U.Vector.data(), U.Vector.size(), filename);
}