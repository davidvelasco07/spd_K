#include "spd_k.hpp"
#include "parameter_input.hpp"
#include <fstream>

int bc_id(const string &name){
    if(name == "periodic") return _periodic_;
    if(name == "gradfree") return _gradfree_;
    cout<<"ERROR: unknown boundary type '"<<name<<"'"<<endl;
    exit(1);
}

int problem_id(const string &name){
    if(name == "sine_wave")       return _ic_sine_wave_;
    if(name == "sedov")           return _ic_sedov_;
    if(name == "spherical_blast") return _ic_spherical_blast_;
    cout<<"ERROR: unknown problem '"<<name<<"'"<<endl;
    exit(1);
}

//"ader" or "rkN" (N = 1, 2, 3); sets cfg.integrator and cfg.rk_order
void select_integrator(const string &name){
    if(name == "ader"){
        cfg.integrator = _integrator_ader_;
        return;
    }
    if(name.rfind("rk",0) == 0 && name.size() == 3){
        int order = name[2]-'0';
        double a[3];
        if(ssp_rk_coefficients(order,a) > 0){
            cfg.integrator = _integrator_rk_;
            cfg.rk_order = order;
            return;
        }
    }
    if(Master) cout<<"ERROR: unknown integrator '"<<name<<"' (ader, rk1, rk2, rk3)"<<endl;
    exit(1);
}

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

        ////////////////////////
        //Runtime parameters: -i <file> plus block/name=value overrides
        ////////////////////////
        ParameterInput pin;
        {
            string input_file;
            std::vector<string> overrides;
            for(int n=1; n<argc; n++){
                string arg = argv[n];
                if(arg=="-i" && n+1<argc)
                    input_file = argv[++n];
                else if(arg.find('/')!=string::npos && arg.find('=')!=string::npos)
                    overrides.push_back(arg);
                else{
                    if(Master) cout<<"ERROR: unrecognized argument '"<<arg<<"'"<<endl;
                    exit(1);
                }
            }
            if(!input_file.empty())
                pin.LoadFromFile(input_file);
            for(const auto &o : overrides)
                pin.ParseOverride(o);
        }

        int p  = pin.GetOrAddInteger("mesh","p",3);
        int NX = pin.GetOrAddInteger("mesh","nx1",8);
        int NY = pin.GetOrAddInteger("mesh","nx2",8);
        int NZ = pin.GetOrAddInteger("mesh","nx3",8);
        double boxlen_x = pin.GetOrAddReal("mesh","x1len",1.0);
        double boxlen_y = pin.GetOrAddReal("mesh","x2len",1.0);
        double boxlen_z = pin.GetOrAddReal("mesh","x3len",1.0);

        //Dimensionality is chosen at runtime: a direction with a single
        //element is inactive (no ghosts, no sweeps, no fluxes)
        bool ax = NX>1;
        bool ay = NY>1;
        bool az = NZ>1;
        if(!ax){
            if(Master) cout<<"ERROR: the x-direction must be active (nx1>1)"<<endl;
            exit(1);
        }
        set_runtime_dimensionality(ax,ay,az);

        cfg.cfl      = pin.GetOrAddReal("time","cfl",0.8);
        cfg.gamma    = pin.GetOrAddReal("hydro","gamma",1.4);
        cfg.fallback = pin.GetOrAddBoolean("job","fallback",true);
        cfg.nad_tolerance = pin.GetOrAddReal("fallback","tolerance",1e-5);
        cfg.nad_delta = pin.GetOrAddString("fallback","NAD","relative")=="delta";
        cfg.nad_moore = pin.GetOrAddString("fallback","NAD_neighbors","2nd")=="2nd";
        cfg.sed       = pin.GetOrAddBoolean("fallback","SED",true);
        cfg.blending  = pin.GetOrAddBoolean("fallback","blending",true);
        cfg.problem  = problem_id(pin.GetOrAddString("problem","problem","sine_wave"));
        cfg.bc[_x_]  = bc_id(pin.GetOrAddString("mesh","x1_bc","periodic"));
        cfg.bc[_y_]  = bc_id(pin.GetOrAddString("mesh","x2_bc","periodic"));
        cfg.bc[_z_]  = bc_id(pin.GetOrAddString("mesh","x3_bc","periodic"));

        double tlim      = pin.GetOrAddReal("time","tlim",0.1);
        double dt_output = pin.GetOrAddReal("output","dt",tlim);
        select_integrator(pin.GetOrAddString("time","integrator","ader"));
        string system_name = pin.GetOrAddString("job","system","hydro");

        //Number of elements on this rank
        int Nx = ax ? NX/comm.nx : 1;
        int Ny = ay ? NY/comm.ny : 1;
        int Nz = az ? NZ/comm.nz : 1;

        if(Master){
            cout<<"system = "<<system_name<<", ndim = "<<cfg.ndim
                <<", p = "<<p<<", N = ("<<Nx<<","<<Ny<<","<<Nz<<")"
                <<", integrator = "
                <<(cfg.integrator==_integrator_ader_ ? "ader" : "rk"+to_string(cfg.rk_order))
                <<endl;
            //Echo the effective parameters for provenance
            std::ofstream dump(output_folder()+"parameters.txt");
            pin.Dump(dump);
        }

        double *x = malloc_host<double>(p);
        double *w = malloc_host<double>(p);
        gauss_legendre(0.0, 1.0, p, x, w);

        double *x_sp = malloc_host<double>(p+1);
        double *x_fp = malloc_host<double>(p+2);

        flux_points(x_fp,x,p);
        solution_points(x_sp,p);

        dimension X_dim(_x_,NX,Nx,ax ? p:0,comm.x*Nx,boxlen_x,x_fp,ax);
        dimension Y_dim(_y_,NY,Ny,ay ? p:0,comm.y*Ny,boxlen_y,x_fp,ay);
        dimension Z_dim(_z_,NZ,Nz,az ? p:0,comm.z*Nz,boxlen_z,x_fp,az);
        Write_dimensions(X_dim,Y_dim,Z_dim);
        Kokkos::Timer timer;

        if(system_name == "induction"){
            double eta = pin.GetOrAddReal("induction","nu",0.0025);
            Induction_ader system(comm,p,X_dim,Y_dim,Z_dim,x,w,x_sp,x_fp,eta);
            system.time_evolution(comm,tlim,dt_output,X_dim,Y_dim,Z_dim);
        }
        else if(system_name == "hydro"){
            double nu   = pin.GetOrAddReal("hydro","nu",0.00001);
            double beta = pin.GetOrAddReal("hydro","beta",-2./3*pin.GetReal("hydro","nu"));
            Hydro_ader system(comm,p,X_dim,Y_dim,Z_dim,x,w,x_sp,x_fp,nu,beta);
            system.time_evolution(comm,tlim,dt_output,X_dim,Y_dim,Z_dim);
        }
        else{
            if(Master) cout<<"ERROR: unknown system '"<<system_name<<"'"<<endl;
            exit(1);
        }
        Kokkos::fence();
        cout<<"time taken: "<<timer.seconds()<<endl;
    }
    Kokkos::finalize();
    #ifdef MPI
    MPI_Finalize();
    #endif
}
