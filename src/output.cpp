#include "spd_k.hpp"
#include <filesystem>

string output_folder(){
  const char* env = getenv("SPD_OUTPUT_DIR");
  string folder = env ? string(env) : string("output");
  std::filesystem::create_directories(folder);
  return folder+"/";
}

void Write(SD_Solution U, int n){
  Kokkos::fence(); //device work must finish before the host reads the data
  int p = U.nx-1;
  int N = U.Nx-2*NGH;
  string filename =  output_folder()+U.label+"_N"+to_string(N)+"p"+to_string(p)+"_"+to_string(n)+"_"+to_string(cpu_rank)+".dat";
  //cout<<N<<","<<p<<endl;
  #ifdef KOKKOS_ENABLE_CUDA
  U.copy();
  Write_arrays(U.Vector_h.data(), U.Vector_h.size(), filename);
  #else
  Write_arrays(U.Vector.data(), U.Vector.size(), filename);
  #endif
}

void Write(FV_Solution U, int n){
  Kokkos::fence(); //device work must finish before the host reads the data
  int N = U.Nx;
  string filename =  output_folder()+U.label+"_N"+to_string(N)+"_"+to_string(n)+"_"+to_string(cpu_rank)+".dat";
  //cout<<N<<","<<p<<endl;
  #ifdef KOKKOS_ENABLE_CUDA
  U.copy();
  Write_arrays(U.Vector_h.data(), U.Vector_h.size(), filename);
  #else
  Write_arrays(U.Vector.data(), U.Vector.size(), filename);
  #endif
}

void Write_edges(dimension Dim, string name){
  int p = Dim.p;
  int N = Dim.N;
  string foldername =  output_folder();
  string filename = name+"_N"+to_string(N)+"p"+to_string(p)+"_"+to_string(cpu_rank)+".dat";
  Write_arrays(Dim.fv_faces.data(), Dim.fv_faces.size(), foldername+filename);
}

void Write_dimensions(dimension X_dim, dimension Y_dim, dimension Z_dim){
   Write_edges(X_dim,"X");
   Write_edges(Y_dim,"Y");
   Write_edges(Z_dim,"Z");
}