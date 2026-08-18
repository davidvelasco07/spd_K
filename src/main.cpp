#include "spd_k.hpp"
#include "parameter_input.hpp"
#include "forest.hpp"
#include <fstream>
#include <regex>

int bc_id(const string &name){
    if(name == "periodic")   return _periodic_;
    if(name == "gradfree")   return _gradfree_;
    if(name == "reflective") return _reflective_;
    cout<<"ERROR: unknown boundary type '"<<name<<"'"<<endl;
    exit(1);
}

int problem_id(const string &name){
    if(name == "sine_wave")        return _ic_sine_wave_;
    if(name == "sedov")            return _ic_sedov_;
    if(name == "spherical_blast")  return _ic_spherical_blast_;
    if(name == "square")           return _ic_square_;
    if(name == "sod_shock_tube")   return _ic_sod_;
    if(name == "shu_osher")        return _ic_shu_osher_;
    if(name == "kelvin_helmholtz") return _ic_kelvin_helmholtz_;
    if(name == "implosion")        return _ic_implosion_;
    if(name == "rti")              return _ic_rti_;
    if(name == "orszag_tang")      return _ic_orszag_tang_;
    if(name == "field_loop")       return _ic_field_loop_;
    if(name == "user")             return _ic_user_;
    cout<<"ERROR: unknown problem '"<<name<<"'"<<endl;
    exit(1);
}

//Per-problem defaults for the <problem> block; any field can be overridden
//from the input file (e.g. problem/amp=0.2). Fields a problem does not use
//are ignored by its initial_condition() branch.
void problem_defaults(int problem, ProblemParams &pp){
    switch(problem){
        case _ic_sine_wave_:        //amp, v1..v3 (advection velocity), p0
            pp = {0.125, 1.0,1.0,0.0, 1.0,0.0, 1.0,0.0, 0.0,0.0, 0};
            break;
        case _ic_square_:           //d0 (background), d1 (top-hat), v1..v3, p0
            pp = {0.0, 1.0,1.0,0.0, 1.0,2.0, 1.0,0.0, 0.25,0.0, 0};
            break;
        case _ic_sod_:              //d0/p0 left, d1/p1 right, radius = interface
            pp = {0.0, 0.0,0.0,0.0, 1.0,0.125, 1.0,0.1, 0.5,0.0, 0};
            break;
        case _ic_shu_osher_:        //standard values are hardcoded; amp = sine amplitude
            pp = {0.2, 0.0,0.0,0.0, 1.0,0.0, 1.0,0.0, 0.0,0.0, 0};
            break;
        case _ic_kelvin_helmholtz_: //d0/d1 layers, v1 shear, amp+sigma perturbation, p0
            pp = {0.1, 0.5,0.0,0.0, 1.0,2.0, 2.5,0.0, 0.0,0.05/sqrt(2.0), 0};
            break;
        case _ic_implosion_:        //d0/p0 outside, d1/p1 corner, radius = diagonal
            pp = {0.0, 0.0,0.0,0.0, 1.0,0.125, 1.0,0.14, 0.15,0.0, 0};
            break;
        case _ic_sedov_:            //radius of the energy deposit
            pp = {0.0, 0.0,0.0,0.0, 1.0,0.0, 0.0,0.0, 0.1,0.0, 0};
            break;
        case _ic_spherical_blast_:  //d0, p0 inside, p1 outside, radius
            pp = {0.0, 0.0,0.0,0.0, 1.0,0.0, 10.0,0.1, 0.1,0.0, 0};
            break;
        case _ic_rti_:              //d0 heavy (lower), d1 light (upper), p0 ref,
                                    //radius = interface yc, amp = perturbation
            pp = {0.025, 0.0,0.0,0.0, 2.0,1.0, 1.0,0.0, 0.5,0.0, 0};
            break;
        case _ic_user_:             //free-form: defaults for the sample Gaussian pulse
            pp = {0.5, 1.0,0.0,0.0, 1.0,0.0, 1.0,0.0, 0.5,0.1, 0};
            break;
    }
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
        set_runtime_dimensionality(ax,ay,az,p);

        //SD at p=3 is unstable at cfl=0.8: dt is exactly constant until
        //t~0.1 and then falls six orders of magnitude. Measured stable at
        //0.4 and below (see the dt-collapse guard in driver.hpp). Higher p
        //has a tighter limit still and has not been measured.
        cfg.cfl      = pin.GetOrAddReal("time","cfl",0.4);
        cfg.nlim     = pin.GetOrAddInteger("time","nlim",-1);
        cfg.gamma    = pin.GetOrAddReal("hydro","gamma",1.4);
        //Constant gravitational acceleration (source term); default 0 leaves
        //the homogeneous Euler equations untouched. Set e.g. hydro/g2 for a
        //vertical field (Rayleigh-Taylor).
        cfg.g[_x_]   = pin.GetOrAddReal("hydro","g1",0.0);
        cfg.g[_y_]   = pin.GetOrAddReal("hydro","g2",0.0);
        cfg.g[_z_]   = pin.GetOrAddReal("hydro","g3",0.0);
        cfg.fallback = pin.GetOrAddBoolean("job","fallback",true);
        cfg.nad_tolerance = pin.GetOrAddReal("fallback","tolerance",1e-5);
        cfg.nad_delta = pin.GetOrAddString("fallback","NAD","relative")=="delta";
        cfg.nad_moore = pin.GetOrAddString("fallback","NAD_neighbors","2nd")=="2nd";
        cfg.sed       = pin.GetOrAddBoolean("fallback","SED",true);
        cfg.blending  = pin.GetOrAddBoolean("fallback","blending",true);
        //fallback/style = blend (fractional theta, the Python-reference scheme)
        //or cascade (discrete MOOD levels, as the MHD module runs).
        string fbstyle = pin.GetOrAddString("fallback","style","blend");
        if(fbstyle=="cascade")     cfg.mood_cascade = true;
        else if(fbstyle!="blend"){
            if(Master) cout<<"ERROR: unknown fallback/style '"<<fbstyle
                           <<"' (expected blend or cascade)"<<endl;
            exit(1);
        }
        //job/scheme picks the discretisation on the flux-point subgrid:
        //  sd   spectral difference with the FV fallback (the primary scheme)
        //  vl2  MUSCL-Hancock -- limited slopes + Hancock half-step predictor,
        //       second order in time on its own, so pair it with time/integrator=rk1
        //  plm  plain PLM -- limited slopes only, no predictor; the time accuracy
        //       is the outer integrator's, so pair it with rk2/rk3
        //vl2 and plm pin the fallback blend to 1 on every face, giving a
        //low-order reference lane at matched DoF. (vl2 names the scheme after
        //Athena++'s integrator for comparison; MUSCL-Hancock predicts the FACE
        //STATES locally with one Riemann solve, where VL2 does a conservative
        //half-step with donor-cell fluxes and solves twice. Same order, and the
        //closest lane spd_K has, but not the same algorithm.)
        string scheme = pin.GetOrAddString("job","scheme","sd");
        if(scheme=="vl2" || scheme=="plm"){
            cfg.fv_only = true;
            cfg.fallback = true;
            cfg.fv_predictor = (scheme=="vl2");
        } else if(scheme!="sd"){
            cout<<"ERROR: unknown scheme '"<<scheme<<"' (expected sd, vl2 or plm)"<<endl;
            exit(1);
        }
        cfg.max_revs  = pin.GetOrAddInteger("fallback","max_revs",3);
        cfg.pad_min_rho = pin.GetOrAddReal("fallback","min_rho",1e-10);
        cfg.pad_min_P   = pin.GetOrAddReal("fallback","min_P",1e-10);
        cfg.floor_cons  = pin.GetOrAddString("hydro","floors","ramses")=="athenak";
        cfg.dfloor      = pin.GetOrAddReal("hydro","dfloor",1e-10);
        cfg.pfloor      = pin.GetOrAddReal("hydro","pfloor",-1.0);
        cfg.problem  = problem_id(pin.GetOrAddString("problem","problem","sine_wave"));
        problem_defaults(cfg.problem, cfg.pp);
        cfg.pp.amp    = pin.GetOrAddReal("problem","amp",cfg.pp.amp);
        cfg.pp.v1     = pin.GetOrAddReal("problem","v1",cfg.pp.v1);
        cfg.pp.v2     = pin.GetOrAddReal("problem","v2",cfg.pp.v2);
        cfg.pp.v3     = pin.GetOrAddReal("problem","v3",cfg.pp.v3);
        cfg.pp.d0     = pin.GetOrAddReal("problem","d0",cfg.pp.d0);
        cfg.pp.d1     = pin.GetOrAddReal("problem","d1",cfg.pp.d1);
        cfg.pp.p0     = pin.GetOrAddReal("problem","p0",cfg.pp.p0);
        cfg.pp.p1     = pin.GetOrAddReal("problem","p1",cfg.pp.p1);
        cfg.pp.radius = pin.GetOrAddReal("problem","radius",cfg.pp.radius);
        cfg.pp.sigma  = pin.GetOrAddReal("problem","sigma",cfg.pp.sigma);
        cfg.pp.dir    = pin.GetOrAddInteger("problem","dir",cfg.pp.dir);
        //Domain center in physical coordinates, so ICs (e.g. spherical_blast)
        //stay centered in rectangular boxes (x[ilj]len != 1).
        cfg.pp.cx     = 0.5*boxlen_x;
        cfg.pp.cy     = 0.5*boxlen_y;
        cfg.pp.cz     = 0.5*boxlen_z;
        cfg.bc[_x_]  = bc_id(pin.GetOrAddString("mesh","x1_bc","periodic"));
        cfg.bc[_y_]  = bc_id(pin.GetOrAddString("mesh","x2_bc","periodic"));
        cfg.bc[_z_]  = bc_id(pin.GetOrAddString("mesh","x3_bc","periodic"));

        cfg.adapt_interval = pin.GetOrAddInteger("amr","adapt_interval",0);
        cfg.amr_max_level  = pin.GetOrAddInteger("amr","max_level",0);
        cfg.amr_prolong_dmp = pin.GetOrAddBoolean("amr","prolong_dmp",false);
        string crit = pin.GetOrAddString("amr","criterion","lohner");
        if(crit=="pressure") cfg.amr_criterion = 1;
        else if(crit=="trouble") cfg.amr_criterion = 2;
        else if(crit=="shear") cfg.amr_criterion = 3;
        else if(crit=="bfield") cfg.amr_criterion = 4;
        else cfg.amr_criterion = 0;
        cfg.amr_refine_threshold =
            pin.GetOrAddReal("amr","refine_threshold",cfg.amr_refine_threshold);
        cfg.amr_derefine_threshold =
            pin.GetOrAddReal("amr","derefine_threshold",cfg.amr_derefine_threshold);
        cfg.amr_bfield_threshold =
            pin.GetOrAddReal("amr","bfield_threshold",cfg.amr_bfield_threshold);
        //Drive the initial refinement to convergence before the first step
        //(Athena++ Mesh::Initialize). OFF by default: it is only well posed for a
        //THRESHOLD criterion. A ranking criterion (amr/refine_frac) always has a
        //top-scoring block, so iterating it refines the peak, then the next peak,
        //and converges only when everything sits at max_level -- on the symmetric
        //Orszag-Tang that turned a 22-block mixed mesh into 64 blocks all at
        //level 1. Enable it for threshold criteria (shear, pressure, lohner).
        cfg.amr_initial_refine =
            pin.GetOrAddBoolean("amr","initial_refine",false);
        cfg.amr_refine_frac =
            pin.GetOrAddReal("amr","refine_frac",cfg.amr_refine_frac);
        cfg.amr_derefine_frac =
            pin.GetOrAddReal("amr","derefine_frac",cfg.amr_derefine_frac);

        double tlim      = pin.GetOrAddReal("time","tlim",0.1);
        //Outputs are opt-in (athenak-style): files are only written when the
        //input file (or an override) provides <output> dt with a positive
        //value. Perf runs can disable them with output/dt=-1.
        cfg.outputs = pin.DoesParameterExist("output","dt")
                      && pin.GetReal("output","dt") > 0.0;
        double dt_output = cfg.outputs ? pin.GetReal("output","dt") : tlim;
        select_integrator(pin.GetOrAddString("time","integrator","ader"));

        //The FV fallback has to match the temporal treatment of the scheme it
        //falls back FROM, or the time integration is applied twice. ADER hands
        //each stage a time-accurate state, so the fallback flux needs its own
        //Hancock half-step to be centred with it (vl2). An RK stage instead
        //takes its time accuracy from the outer integrator, so predicting again
        //inside the flux double-counts it -- plain PLM is the consistent
        //choice there. The standalone lanes name the scheme outright and keep
        //what they were given; only job/scheme=sd (SDFB) is derived.
        if(scheme=="sd")
            cfg.fv_predictor = (cfg.integrator == _integrator_ader_);
        //Explicit override, for either lane.
        cfg.fv_predictor = pin.GetOrAddBoolean("fallback","predictor",
                                               cfg.fv_predictor);
        string system_name = pin.GetOrAddString("job","system","hydro");

        //Number of elements on this rank
        int Nx = ax ? NX/comm.nx : 1;
        int Ny = ay ? NY/comm.ny : 1;
        int Nz = az ? NZ/comm.nz : 1;

        //Meshblock decomposition (elements per block; default = one block
        //covering the whole rank domain, i.e. the single-block solver)
        int NBx = pin.GetOrAddInteger("meshblock","nx1",Nx);
        int NBy = pin.GetOrAddInteger("meshblock","nx2",Ny);
        int NBz = pin.GetOrAddInteger("meshblock","nx3",Nz);
        if(!ay) NBy = 1;
        if(!az) NBz = 1;
        int nbx = Nx/NBx, nby = Ny/NBy, nbz = Nz/NBz;
        bool multiblock = (NBx!=Nx)||(NBy!=Ny)||(NBz!=Nz);
        bool use_mesh = multiblock || cfg.amr_max_level>0;

        std::vector<RefinementRegion> refinements;
        std::regex ref_re("^refinement\\d+$");
        for(const string& bname : pin.BlockNames()){
            if(!std::regex_match(bname, ref_re)) continue;
            RefinementRegion r;
            r.level = pin.GetOrAddInteger(bname,"level",1);
            r.xmin = pin.GetOrAddReal(bname,"x1min",0.0);
            r.xmax = pin.GetOrAddReal(bname,"x1max",1.0);
            r.ymin = pin.GetOrAddReal(bname,"x2min",0.0);
            r.ymax = pin.GetOrAddReal(bname,"x2max",1.0);
            r.zmin = pin.GetOrAddReal(bname,"x3min",0.0);
            r.zmax = pin.GetOrAddReal(bname,"x3max",1.0);
            refinements.push_back(r);
        }

        if(use_mesh){
            if(NBx<1 || Nx%NBx || NBy<1 || Ny%NBy || NBz<1 || Nz%NBz){
                if(Master) cout<<"ERROR: meshblock size ("<<NBx<<","<<NBy<<","<<NBz
                               <<") must divide the rank domain ("<<Nx<<","<<Ny<<","<<Nz<<")"<<endl;
                exit(1);
            }
            if(comm.nx*comm.ny*comm.nz>1){
                if(Master) cout<<"ERROR: meshblocks/AMR are not yet supported with MPI"<<endl;
                exit(1);
            }
            //The mesh path implements periodic and gradfree domain boundaries
            //only. Anything else is not merely unhandled, it is silently
            //wrong: neighbors_uniform labels a non-gradfree boundary block
            //_periodic_ and wraps its neighbour index around the domain, so a
            //reflective wall is exchanged with the opposite side. That still
            //conserves mass, so it passes the usual checks while returning a
            //completely different solution (measured: 56% relative error on a
            //4-block implosion, and the SD-only variant diverges outright).
            //Reflective walls remain available in single-block runs, which use
            //the boundary.cpp path instead.
            for(int d=0; d<3; d++){
                if(!cfg.active[d]) continue;
                if(cfg.bc[d]==_periodic_ || cfg.bc[d]==_gradfree_) continue;
                if(Master)
                    cout<<"ERROR: meshblocks/AMR support only periodic and gradfree "
                        <<"boundaries, but x"<<(d+1)<<"_bc is neither. Reflective "
                        <<"boundaries are implemented for single-block runs only "
                        <<"(remove the <meshblock> block, or use periodic/gradfree)."<<endl;
                exit(1);
            }
            //Mixed-level AMR needs the MOOD cascade, not the fractional blend.
            //A blended flux is a weighted mix of two fluxes, and the weight is
            //a per-cell theta: the coarse and fine sides of a level jump blend
            //differently, so the shared face is not single-valued and the
            //correction has to patch the imbalance back afterwards. The
            //cascade instead *selects* one flux per face from the pooled level,
            //which is single-valued from both sides by construction. MHD always
            //runs its own cascade, so this is a hydro-only requirement.
            //job/scheme=vl2|plm is exempt: they pin theta to 1 on every face, so
            //every face takes the MUSCL flux and is single-valued after all.
            //It is the low-order reference lane, not a blend.
            if(system_name=="hydro" && cfg.amr_max_level>0 && cfg.fallback
               && !cfg.mood_cascade && !cfg.fv_only){
                if(Master)
                    cout<<"ERROR: mixed-level AMR with the FV fallback requires "
                        <<"fallback/style=cascade; the fractional blend is not "
                        <<"single-valued across a coarse-fine face"<<endl;
                exit(1);
            }
            //Meshblocks are allowed for hydro and MHD (uniform multiblock).
            //Mixed-level AMR for MHD is rejected inside Mesh<MHD_ader>.
            if(cfg.amr_max_level>0 && cfg.integrator==_integrator_ader_){
                if(Master) cout<<"ERROR: ADER is not supported with mixed-level AMR (use rk2/rk3)"<<endl;
                exit(1);
            }
        }

        if(Master){
            cout<<"system = "<<system_name<<", ndim = "<<cfg.ndim
                <<", p = "<<p<<", N = ("<<Nx<<","<<Ny<<","<<Nz<<")"
                <<", integrator = "
                <<(cfg.integrator==_integrator_ader_ ? "ader" : "rk"+to_string(cfg.rk_order))
                <<", outputs = "<<(cfg.outputs ? "on" : "off")
                <<endl;
            if(cfg.outputs || use_mesh){
                //Echo the effective parameters for provenance
                std::ofstream dump(output_folder()+"parameters.txt");
                pin.Dump(dump);
                dump<<"<build>"<<endl;
                dump<<"layout = "
                    <<(std::is_same<Layout,Kokkos::LayoutLeft>::value
                       ? "LayoutLeft" : "LayoutRight")<<endl;
                dump<<endl;
            }
        }

        double *x = malloc_host<double>(p);
        double *w = malloc_host<double>(p);
        gauss_legendre(0.0, 1.0, p, x, w);

        double *x_sp = malloc_host<double>(p+1);
        double *x_fp = malloc_host<double>(p+2);

        flux_points(x_fp,x,p);
        solution_points(x_sp,p);
        if(use_mesh)
            init_amr_transfer_matrices(x_sp, x_fp, p);

        dimension X_dim(_x_,NX,Nx,ax ? p:0,comm.x*Nx,boxlen_x,x_fp,ax);
        dimension Y_dim(_y_,NY,Ny,ay ? p:0,comm.y*Ny,boxlen_y,x_fp,ay);
        dimension Z_dim(_z_,NZ,Nz,az ? p:0,comm.z*Nz,boxlen_z,x_fp,az);
        if(cfg.outputs)
            Write_dimensions(X_dim,Y_dim,Z_dim);
        Kokkos::Timer timer;

        if(system_name == "induction"){
            double eta = pin.GetOrAddReal("induction","nu",0.0025);
            Induction_ader system(comm,p,X_dim,Y_dim,Z_dim,x,w,x_sp,x_fp,eta);
            Driver driver(&system);
            driver.Execute(tlim,dt_output);
        }
        else if(system_name == "hydro"){
            //Viscosity is opt-in at runtime (athenak-style): set hydro/nu>0 in
            //the input file to switch on the viscous terms; the default nu=0
            //runs the inviscid Euler equations.
            double nu   = pin.GetOrAddReal("hydro","nu",0.0);
            double beta = pin.GetOrAddReal("hydro","beta",-2./3*nu);
            if(Master && nu>0.0)
                cout<<"viscosity on: nu = "<<nu<<", beta = "<<beta<<endl;
            if(Master && (cfg.g[_x_]||cfg.g[_y_]||cfg.g[_z_]))
                cout<<"gravity on: g = ("<<cfg.g[_x_]<<", "<<cfg.g[_y_]
                    <<", "<<cfg.g[_z_]<<")"<<endl;
            if(use_mesh){
                double lim[3][2] = {
                    {0.0, boxlen_x},
                    {0.0, boxlen_y},
                    {0.0, boxlen_z}
                };
                int bc3[3] = {cfg.bc[_x_], cfg.bc[_y_], cfg.bc[_z_]};
                BlockForest forest = BlockForest::uniform_grid(
                    cfg.ndim, ax, ay, az, NBx, NBy, NBz, nbx, nby, nbz, lim, bc3);
                if(!refinements.empty()){
                    for(auto& r : refinements)
                        r.level = min(r.level, cfg.amr_max_level);
                    forest.refine_to_levels(refinements);
                    forest.enforce_2to1_balance();
                }
                Mesh<Hydro_ader> mesh(std::move(forest), comm, p, X_dim, Y_dim, Z_dim,
                                      NBx, NBy, NBz, x, w, x_sp, x_fp, nu, beta);
                Driver driver(&mesh);
                driver.Execute(tlim, dt_output);
            }
            else{
                Hydro_ader system(comm,p,X_dim,Y_dim,Z_dim,x,w,x_sp,x_fp,nu,beta);
                Driver driver(&system);
                driver.Execute(tlim,dt_output);
            }
        }
        else if(system_name == "mhd"){
            if(use_mesh){
                if(cfg.amr_max_level>0 && az){
                    if(Master) cout<<"ERROR: mixed-level AMR for 3D MHD is not implemented yet"
                                     <<" (Stage 4 supports true-2D static refinement)"<<endl;
                    exit(1);
                }
                double lim[3][2] = {
                    {0.0, boxlen_x},
                    {0.0, boxlen_y},
                    {0.0, boxlen_z}
                };
                int bc3[3] = {cfg.bc[_x_], cfg.bc[_y_], cfg.bc[_z_]};
                BlockForest forest = BlockForest::uniform_grid(
                    cfg.ndim, ax, ay, az, NBx, NBy, NBz, nbx, nby, nbz, lim, bc3);
                if(!refinements.empty()){
                    for(auto& r : refinements)
                        r.level = min(r.level, cfg.amr_max_level);
                    forest.refine_to_levels(refinements);
                    forest.enforce_2to1_balance();
                }
                Mesh<MHD_ader> mesh(std::move(forest), comm, p, X_dim, Y_dim, Z_dim,
                                    NBx, NBy, NBz, x, w, x_sp, x_fp);
                Driver driver(&mesh);
                driver.Execute(tlim, dt_output);
            }
            else{
                MHD_ader system(comm,p,X_dim,Y_dim,Z_dim,x,w,x_sp,x_fp);
                Driver driver(&system);
                driver.Execute(tlim,dt_output);
            }
        }
        else{
            if(Master) cout<<"ERROR: unknown system '"<<system_name<<"'"<<endl;
            exit(1);
        }
        Kokkos::fence();
        cout<<"time taken: "<<timer.seconds()<<endl;
        //Release AMR setup matrices before Kokkos::finalize (globals outlive main)
        amr_P = Matrix();
        amr_R = Matrix();
        amr_RS_sp[0] = amr_RS_sp[1] = Matrix();
        amr_RS_cv[0] = amr_RS_cv[1] = Matrix();
        amr_RF = Matrix();
        amr_P_fp = Matrix();
        amr_RF_fp = Matrix();
        amr_x_fp = Vector();
    }
    Kokkos::finalize();
    #ifdef MPI
    MPI_Finalize();
    #endif
}
