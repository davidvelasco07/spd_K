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
  int N = U.Nx-2*NGHx;
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
  #ifdef KOKKOS_ENABLE_CUDA
  Vector_h fv_faces_h = setup_mirror(Dim.fv_faces);
  setup_pull(Dim.fv_faces, fv_faces_h);
  Write_arrays(fv_faces_h.data(), fv_faces_h.size(), foldername+filename);
  #else
  Write_arrays(Dim.fv_faces.data(), Dim.fv_faces.size(), foldername+filename);
  #endif
}

void Write_dimensions(dimension X_dim, dimension Y_dim, dimension Z_dim){
   Write_edges(X_dim,"X");
   Write_edges(Y_dim,"Y");
   Write_edges(Z_dim,"Z");
}

void Write_amr_blocks(const BlockForest& forest, int n_output,
                      int NBx, int NBy, int NBz){
  if(!Master) return;
  int M = forest.max_level();
  int sc = 1 << M;
  int Nfx = forest.N_base[0] * NBx * sc;
  int Nfy = forest.active[1] ? forest.N_base[1] * NBy * sc : 1;
  int Nfz = forest.active[2] ? forest.N_base[2] * NBz * sc : 1;
  string path = output_folder() + "amr_blocks_" + to_string(n_output) + ".txt";
  std::ofstream out(path);
  out << "# nblocks max_level ndim NBx NBy NBz Nfx Nfy Nfz\n";
  out << forest.Nblocks() << " " << M << " " << forest.ndim << " "
      << NBx << " " << NBy << " " << NBz << " "
      << Nfx << " " << Nfy << " " << Nfz << "\n";
  out << "# ib level lx ly lz x0 x1 y0 y1 z0 z1\n";
  for(int ib=0; ib<forest.Nblocks(); ib++){
    const MeshBlock& b = forest.blocks[ib];
    out << ib << " " << b.level << " "
        << b.logical[0] << " " << b.logical[1] << " " << b.logical[2] << " "
        << b.lim[0][0] << " " << b.lim[0][1] << " "
        << b.lim[1][0] << " " << b.lim[1][1] << " "
        << b.lim[2][0] << " " << b.lim[2][1] << "\n";
  }
}