#include "spd_k.hpp"

int main(int argc, char** argv){
    #ifdef MPI
    MPI_Init(&argc,&argv);
    #endif
    Kokkos::initialize( argc, argv );
    {
        
        #ifdef MPI
        // Communicator
        Comm = MPI_COMM_WORLD;
        CommHelper comm(Comm);
        #else
        CommHelper comm;
        #endif
        cpu_rank = comm.me;
        Master = cpu_rank==0;

        size_t p  = 1;   //Order of space approximation
        size_t N  = 64;
        size_t NX = N;   //Number of elements in X-direction
        size_t NY = N;   //Number of elements in Y-direction
        size_t NZ = N;   //Number of elements in Z-direction

        size_t Nx = NX/comm.nx;
        size_t Ny = NY/comm.ny;
        size_t Nz = NZ/comm.nz;
        
        if(Master)
            cout<< "p = "<<p<<", N = ("<<Nx<<","<<Ny<<","<<Nz<<")"<<endl;
        
        double *x = malloc_host<double>(p);
        double *w = malloc_host<double>(p);
        gauss_legendre(0.0, 1.0, p, x, w);

        double *x_sp = malloc_host<double>(p+1);   
        double *x_fp = malloc_host<double>(p+2);

        flux_points(x_fp,x,p);
        solution_points(x_sp,p);

        double L=LENGHT;
        cout<<_x_<<_y_<<_z_<<" "<<L<<endl;
        dimension X_dim(_x_,NX,Nx,X*p,comm.x*Nx,L,x_fp,X);
        dimension Y_dim(_y_,NY,Ny,Y*p,comm.y*Ny,L,x_fp,Y);
        dimension Z_dim(_z_,NZ,Nz,Z*p,comm.z*Nz,L,x_fp,Z);
        Kokkos::Timer timer;
        
        #ifdef INDUCTION
        Induction_ader system(comm,p,X_dim,Y_dim,Z_dim,x,w,x_sp,x_fp,0.0025);
        system.time_evolution(comm,20*PI,2*PI,X_dim,Y_dim,Z_dim);
        #endif
        #ifdef HYDRO
        Hydro_ader system(comm,p,X_dim,Y_dim,Z_dim,x,w,x_sp,x_fp);
        #endif
        cout<<"time taken: "<<timer.seconds()<<endl;
    }
    Kokkos::finalize();
    #ifdef MPI
    MPI_Finalize();
    #endif
}