#include "spd_k.hpp"
#include "parameter_input.hpp"
#include "forest.hpp"
#include <fstream>
#include <regex>

int bc_id(const string &name){
    if(name == "periodic")   return _periodic_;
    if(name == "gradfree")   return _gradfree_;
    if(name == "reflective") return _reflective_;
    if(name == "outflow")    return _outflow_;    //gradfree + no re-entry
    if(name == "inflow")     return _inflow_;
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
    if(name == "mhd_vortex")       return _ic_mhd_vortex_;
    if(name == "mhd_blast")        return _ic_mhd_blast_;
    if(name == "mhd_jet")          return _ic_mhd_jet_;
    if(name == "ha_jet")           return _ic_ha_jet_;
    if(name == "current_sheet")    return _ic_current_sheet_;
    if(name == "kh_mdz")           return _ic_kh_mdz_;
    if(name == "kh_rr22")          return _ic_kh_rr22_;
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
        case _ic_mhd_vortex_:      //amp=V, v1/v2 background, p0 base P, p1=|B|, sigma>0 -> Leidi kernel
            pp = {1.0, 1.0,1.0,0.0, 1.0,0.0, 1.0,1.0, 0.0,0.0, 0};
            break;
        case _ic_mhd_blast_:        //d0=rho, p0/p1 ambient/overpressure, amp=B0,
                                    //v1/v2/v3 = field DIRECTION (normalised in
                                    //the IC), radius = overpressured sphere.
            //(1,0,0) is B = (amp,0,0), the orientation this IC had before the
            //direction was a parameter, so inputs/balsara/* are bit-identical.
            //MDZ21's theta=pi/2, phi=pi/4 is the direction (1,1,0).
            pp = {1000.0, 1.0,0.0,0.0, 1.0,0.0, 0.1,10000.0, 0.1,0.0, 0};
            break;
        case _ic_field_loop_:       //amp=A0, radius=R, v1..v3 advection, d0=rho, p0=P
            //Exactly the values that used to be hardcoded in mhd_ic_field_loop
            //and mhd_ic_vector_potential, so the existing goldens do not move.
            pp = {1e-3, 2.0,1.0,0.0, 1.0,0.0, 1.0,0.0, 0.3,0.0, 0};
            break;
        case _ic_kh_rr22_:          //amp=ca, v1=M, v2=v2_0, p1=theta, sigma=y0,
                                    //radius=perturbation width, d0=rho, p0=p
            //RR22 sec 5.2: ca=0.1, M=1, v2_0=0.01, theta=pi/3, y0=1/20,
            //sigma=0.1, rho=1, p=1/gamma=0.6 for gamma=5/3.
            pp = {0.1, 1.0,0.01,0.0, 1.0,0.0, 0.6,1.0471975511965976, 0.1,0.05, 0};
            break;
        case _ic_kh_mdz_:           //amp=B0, v1=M, sigma=a, d0=rho, p0=P, p1=eps
            //B0 = vA sqrt(rho) = 0.5, M = 1, a = 0.01, p = 1/Gamma = 0.6 for
            //Gamma = 5/3 (set p0 explicitly if gamma differs), eps = 1e-5.
            pp = {0.5, 1.0,0.0,0.0, 1.0,0.0, 0.6,1e-5, 0.0,0.01, 0};
            break;
        case _ic_current_sheet_:    //amp=B0, p0=beta, sigma=a, d0=rho, p1=eps
            pp = {1.0, 0.0,0.0,0.0, 1.0,0.0, 10.0,1e-3, 0.0,0.04, 0};
            break;
        case _ic_mhd_jet_:          //d0 ambient rho, d1 jet rho, p0, v2 jet vy, amp=By, radius nozzle
            pp = {141.421356, 0.0,800.0,0.0, 0.14,1.4, 1.0,0.0, 0.05,0.0, 0};
            pp.cx = 0.5;  //nozzle centred at x=0 on [-0.5,0.5] when x1len=1
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
        //0.4 and below (see the dt-collapse guard in driver.hpp).
        //THE LIMIT DEPENDS ON time/cfl_type. MEASURED 2026-09-01 on the
        //unperturbed Harris sheet -- an EXACT equilibrium, so any motion at all
        //is the scheme going unstable, which catches slow growth that a
        //"did dt collapse?" probe misses entirely (64 DoF, t=0.5, round-off
        //~6e-13):
        //           cfl:   0.50      0.40      0.30      0.25      0.20
        //   p=3 min      2.6e-01   1.3e-02   6.3e-13   6.3e-13   6.3e-13
        //   p=3 sum      6.3e-13   6.3e-13   6.3e-13   6.3e-13   6.3e-13
        //   p=7 min      collapse  collapse  collapse  1.3e-01   1.9e-14
        //   p=7 sum      1.3e-01   3.0e-14   2.5e-14   3.0e-14   4.0e-14
        //So: p=3 needs <= 0.30 under min and holds 0.5 under sum; p=7 needs
        //<= 0.20 under min and <= 0.40 under sum. An earlier Orszag-Tang probe
        //reported "p=7 stable at 0.30" under min -- WRONG, because it only
        //looked for dt collapse and the run was quietly unstable well before
        //that. Probe stability on a problem whose exact answer you know.
        cfg.cfl      = pin.GetOrAddReal("time","cfl",0.4);
        //time/cfl_type = sum | min. SUM is the default because it is what every
        //golden in the tree was generated with; MIN is the standard unsplit
        //multi-dimensional form that Athena++/AthenaK use, under which the same
        //nominal cfl gives a ~1.79x larger dt in 2D (measured on the figure-22
        //KH lane at 128^2 and 2048^2 alike). See the enum in define.hpp.
        {
            std::string ct = pin.GetOrAddString("time","cfl_type","sum");
            if(ct == "sum")      cfg.cfl_type = _cfl_sum_;
            else if(ct == "min") cfg.cfl_type = _cfl_min_;
            else {
                if(Master) cout<<"ERROR: unknown time/cfl_type '"<<ct
                                <<"' (sum, min)"<<endl;
                exit(1);
            }
        }
        cfg.nlim     = pin.GetOrAddInteger("time","nlim",-1);
        //`nlim = 0` reads as "zero steps" and is NOT: the driver's cap is
        //`cfg.nlim > 0`, so 0 falls through as UNLIMITED. That trap, combined
        //with a small output/dt, is how two runs here wrote ~740 GB each and
        //filled the filesystem twice. -1 is the documented way to say unlimited.
        if(cfg.nlim == 0){
            if(Master) cout<<"ERROR: time/nlim = 0 is ambiguous. Use -1 for an "
                             "unlimited step count, or a positive cap."<<endl;
            exit(1);
        }
        cfg.gamma    = pin.GetOrAddReal("hydro","gamma",1.4);
        //Constant gravitational acceleration (source term); default 0 leaves
        //the homogeneous Euler equations untouched. Set e.g. hydro/g2 for a
        //vertical field (Rayleigh-Taylor).
        cfg.g[_x_]   = pin.GetOrAddReal("hydro","g1",0.0);
        cfg.g[_y_]   = pin.GetOrAddReal("hydro","g2",0.0);
        cfg.g[_z_]   = pin.GetOrAddReal("hydro","g3",0.0);
        cfg.fallback = pin.GetOrAddBoolean("job","fallback",true);
        cfg.nad_tolerance = pin.GetOrAddReal("fallback","tolerance",1e-5);
        //MUSCL/FV slope limiter, shared by the hydro fallback, the vl2/plm lane
        //and the MHD MOOD low-order levels. minmod is what every golden encodes.
        string slim = pin.GetOrAddString("fallback","limiter","minmod");
        if(slim=="minmod")       cfg.limiter = _lim_minmod_;
        else if(slim=="vanleer") cfg.limiter = _lim_vanleer_;
        else if(slim=="moncen")  cfg.limiter = _lim_moncen_;
        else{
            if(Master) cout<<"ERROR: fallback/limiter = '"<<slim
                           <<"' not implemented (minmod|vanleer|moncen)"<<endl;
            exit(1);
        }
        cfg.nad_atol = pin.GetOrAddReal("fallback","atol",0.0);
        cfg.nad_eps0 = pin.GetOrAddReal("fallback","eps0",1e-12);
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
        //Domain center in physical coordinates (defaults to box midpoint).
        cfg.pp.cx     = pin.GetOrAddReal("problem","cx",0.5*boxlen_x);
        cfg.pp.cy     = pin.GetOrAddReal("problem","cy",0.5*boxlen_y);
        cfg.pp.cz     = pin.GetOrAddReal("problem","cz",0.5*boxlen_z);
        //Prescribed-inflow face: what it does OUTSIDE the nozzle, and the
        //signal speed it contributes to the timestep. Both are no-ops unless a
        //problem actually prescribes an inflow. See Config::inflow_outside and
        //Config::inflow_rho.
        {
            string jo = pin.GetOrAddString("problem","inflow_outside","reservoir");
            if(jo=="outflow")        cfg.inflow_outside = _jo_outflow_;
            else if(jo=="ambient")   cfg.inflow_outside = _jo_ambient_;
            else if(jo=="reservoir") cfg.inflow_outside = _jo_reservoir_;
            else {
                cout<<"ERROR: unknown problem/inflow_outside '"<<jo
                    <<"' (expected outflow, ambient or reservoir)"<<endl;
                exit(1);
            }
        }
        if(cfg.problem == _ic_ha_jet_){
            //The beam enters only through the boundary, so a dt taken from the
            //quiescent interior does not see it: measured 9.99e-04 against a
            //tlim of 1e-03, i.e. the whole run in one step.
            cfg.inflow_rho = cfg.pp.d1;
            cfg.inflow_p   = cfg.pp.p0;
            cfg.inflow_vx  = cfg.pp.v1;
        }
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
        //MHD Riemann solver (faces + edge EMF). Default llf; hlld matches the
        //Python spd Miyoshi–Kusano HLLD / dimension-by-dimension UCT path.
        {
            string rs = pin.GetOrAddString("mhd","rsolver","llf");
            if(rs=="llf")       cfg.rsolver = _rsolver_llf_;
            else if(rs=="hlld") cfg.rsolver = _rsolver_hlld_;
            else if(rs=="hll")  cfg.rsolver = _rsolver_hll_;
            else{
                if(Master) cout<<"ERROR: mhd/rsolver = '"<<rs
                               <<"' not implemented (llf|hll|hlld)"<<endl;
                exit(1);
            }
            //Electromotive force construction. Default 2sweep, which is what
            //every MHD golden in the tree was generated with; uct selects the
            //Mignone & Del Zanna 2021 upwind CT composition (eq. 33), taking its
            //a/d coefficients from the face solver's own wave fan -- so the UCT
            //flavour follows mhd/rsolver rather than being named separately.
            //Refused under llf: a single-speed bound has no fan to read them off,
            //and silently falling back to 2sweep would report a scheme that did
            //not run (the failure mode that left UCT dead in the tree for a
            //release -- both UCT kernels had zero call sites while docs/mhd.md
            //described them as live).
            {
                string em = pin.GetOrAddString("mhd","emf","2sweep");
                if(em=="2sweep")   cfg.emf = _emf_2sweep_;
                else if(em=="uct") cfg.emf = _emf_uct_;
                else{
                    if(Master) cout<<"ERROR: mhd/emf = '"<<em
                                   <<"' not implemented (2sweep|uct)"<<endl;
                    exit(1);
                }
                if(cfg.emf==_emf_uct_ && cfg.rsolver==_rsolver_llf_){
                    if(Master) cout<<"ERROR: mhd/emf=uct needs a Riemann solver "
                        "with a wave fan to take the UCT coefficients from. Use "
                        "mhd/rsolver=hll (UCT-HLL) or mhd/rsolver=hlld "
                        "(UCT-HLLD)."<<endl;
                    exit(1);
                }
            }
            //MDZ21 6.4's energy correction: when B_to_U replaces U's B rows with
            //the projection of the staggered CT field, shift E by the same amount
            //so the thermal energy is invariant. OFF by default because it moves
            //every MHD result; strongly magnetized decks (the 3D blast, beta ~
            //2.5e-4) need it or 15% of the domain hits the pressure floor.
            {
                //Accept a bitmask (0/1/2/4/7) or the older true/false spelling,
                //where true means "every site".
                string ef = pin.GetOrAddString("mhd","energy_fix","0");
                if(ef=="true")       cfg.mhd_energy_fix = 7;
                else if(ef=="false") cfg.mhd_energy_fix = 0;
                else {
                    try { cfg.mhd_energy_fix = std::stoi(ef); }
                    catch(...) {
                        if(Master) cout<<"ERROR: mhd/energy_fix = '"<<ef
                            <<"' is not an integer bitmask (0|1|2|4|7) or true/false"<<endl;
                        exit(1);
                    }
                }
            }
            //NAD on the candidate CT field: magnitude (default) or components.
            //comps is what AthenaK / Python spd use and is the better detector --
            //|B|-only misses Alfvénic / transverse oscillations -- but flipping
            //the default moves the MHD goldens, so it waits until the global NAD
            //scale is decomposition-invariant and they can be regenerated once.
            //NOTE the code default and the parser default USED TO DISAGREE:
            //global.hpp documents `comps` while this line parsed "mag", and the
            //parser wins -- so every run so far has detected on |B| alone.
            //`comps` is available and is AthenaK's default, but it stays
            //non-default here because it is measurably worse on the one
            //benchmark with an exactly known answer. MDZ21 current sheet,
            //128x64 DoF, UCT-HLLD, tol 1e-5, where the correct E_B(30) is 1:
            //     mood_nad_b=mag    p=3  0.9527    p=7  0.4651
            //     mood_nad_b=comps  p=3  0.3998    p=7  0.4784
            //Three components fire more often than |B|, and on a smooth
            //equilibrium the extra firing is entirely spurious: p=3 loses a
            //factor of 2.4. |B| really is blind to a magnitude-preserving
            //rotation, so `comps` is the better DETECTOR in principle; what
            //makes it expensive is the CASCADE's response to the extra flags.
            //Deepest cascade tier. 2 (default) is the historical behaviour and
            //admits first order; 1 stops the cascade at MUSCL, so the fallback
            //can never be MORE dissipative than the scheme it falls back to.
            cfg.mood_max_level = pin.GetOrAddInteger("mhd","mood_max_level",2);
            if(cfg.mood_max_level < 0 || cfg.mood_max_level > 2){
                if(Master) cout<<"ERROR: mhd/mood_max_level must be 0, 1 or 2"<<endl;
                exit(1);
            }
            //Only a PAD failure may take a cell to first order; a NAD flag alone
            //stops at MUSCL. Off by default (bit-identical); see mhd_PAD.
            cfg.mood_pad_first_order = pin.GetOrAddBoolean("mhd","mood_pad_first_order",false);
            cfg.mood_tier_exclude = pin.GetOrAddBoolean("mhd","mood_tier_exclude",true);
            string nadb = pin.GetOrAddString("mhd","mood_nad_b","mag");
            if(nadb=="mag")         cfg.mood_nad_b = _nad_b_mag_;
            else if(nadb=="comps")  cfg.mood_nad_b = _nad_b_comps_;
            else{
                if(Master) cout<<"ERROR: mhd/mood_nad_b = '"<<nadb
                               <<"' not implemented (mag|comps)"<<endl;
                exit(1);
            }
            //Velocity in NAD (mirrors AthenaK mood_nad_v). Default comps: needed
            //with HLLD+FB to catch ringing that leaves |B| smooth.
            string nadv = pin.GetOrAddString("mhd","mood_nad_v","off");
            if(nadv=="off")         cfg.mood_nad_v = _nad_v_off_;
            else if(nadv=="mag")    cfg.mood_nad_v = _nad_v_mag_;
            else if(nadv=="comps")  cfg.mood_nad_v = _nad_v_comps_;
            else{
                if(Master) cout<<"ERROR: mhd/mood_nad_v = '"<<nadv
                               <<"' not implemented (off|mag|comps)"<<endl;
                exit(1);
            }
            //NAD tolerance scale (AthenaK mood_nad_scale). Default relative: a
            //purely LOCAL band. grange/gcfl take a domain-range reduction, and
            //mhd_nad_compute_gscales is currently called per block, so under them
            //each block gets its own band and multiblock stops agreeing with
            //single-block (measured: OT true2d 1.205e-01 vs true2d_mb 9.700e-02
            //under gcfl; identical 1.227e-01 under relative). Default moves to
            //gcfl once that reduction is hoisted to mesh level.
            string nadsc = pin.GetOrAddString("mhd","mood_nad_scale","relative");
            if(nadsc=="relative")     cfg.mood_nad_scale = _nad_scale_relative_;
            else if(nadsc=="delta")   cfg.mood_nad_scale = _nad_scale_delta_;
            else if(nadsc=="grange")  cfg.mood_nad_scale = _nad_scale_grange_;
            else if(nadsc=="gcfl")    cfg.mood_nad_scale = _nad_scale_gcfl_;
            else{
                if(Master) cout<<"ERROR: mhd/mood_nad_scale = '"<<nadsc
                               <<"' not implemented (relative|delta|grange|gcfl)"<<endl;
                exit(1);
            }
            //Keep legacy fallback/NAD=delta in sync when the user only set that.
            if(cfg.mood_nad_scale==_nad_scale_delta_) cfg.nad_delta = true;
            if(cfg.mood_nad_scale==_nad_scale_relative_) cfg.nad_delta = false;
            //Force a fixed cascade level (diagnostic: pure MUSCL / FO CT update).
            //Ask BEFORE GetOrAdd: GetOrAdd inserts the key with its default, so
            //DoesParameterExist afterwards is always true.
            const bool force_level_given =
                pin.DoesParameterExist("mhd","mood_force_level");
            cfg.mood_force_level = pin.GetOrAddInteger("mhd","mood_force_level",-1);
            if(cfg.mood_force_level<-1 || cfg.mood_force_level>2){
                if(Master) cout<<"ERROR: mhd/mood_force_level = "<<cfg.mood_force_level
                               <<" (expected -1|0|1|2)"<<endl;
                exit(1);
            }
            //job/scheme for MHD. It was a silent no-op: cfg.fv_only is read only
            //by hydro_ader/mesh and cfg.fv_predictor only by hydro.cpp, so
            //job/scheme=vl2 and job/scheme=plm under system=mhd both produced
            //output bit-identical to job/scheme=sd -- measured, md5-identical
            //with detection live and with it pinned. A low-order MHD lane is
            //reachable only through the cascade level, so map plm onto it and
            //refuse vl2 rather than let either be reported as a scheme it is not.
            //job/scheme for MHD. Both low-order lanes pin every cell at the
            //cascade's MUSCL level, which IS limited-slope reconstruction on the
            //sub-cell mesh; they differ in cfg.fv_predictor:
            //  plm  slopes only -- time accuracy comes from the outer
            //       integrator, so pair it with rk2/rk3
            //  vl2  slopes plus the MHD Hancock half-step (mhd_fv_predict,
            //       ported from the Python spd reference / RAMSES trace3d), so
            //       it is second order in time on its own -- pair it with rk1
            //vl2 used to be REFUSED here because mhd_fv_fluxes_t had no
            //predictor and would have run plain PLM under that name. It has one
            //now -- but MEASURED, it does not buy second order in time.
            //
            //Orszag-Tang 64^2 DoF at t=0.05 (smooth), MUSCL level, mesh FIXED
            //and only dt refined so the spatial error cancels:
            //    plm + rk1        L1 1.10e-03 at cfl 0.4, order 1.06 - 1.22
            //    vl2 + rk1        L1 2.85e-04 at cfl 0.4, order 1.03 - 1.22
            //    plm + rk2        L1 2.00e-05 at cfl 0.4, order 1.99 - 2.07
            //The control confirms the method resolves second order, so vl2's
            //first order is real: a LOCAL Hancock predictor time-centres the
            //fluid fluxes (hence the 4x smaller error) but cannot time-centre
            //the CT FACE FIELD, whose half-step needs the edge EMF and is
            //therefore non-local. That is why Athena++/AthenaK use a two-stage
            //VL2 (a conservative donor-cell half-step INCLUDING a CT update)
            //for MHD rather than MUSCL-Hancock.
            //
            //Note MDZ21's own 2nd-order base scheme is SSP-RK2 with piecewise
            //linear reconstruction -- that is job/scheme=plm + time/integrator=rk2,
            //which is second order here. vl2 is an extra lane, not the paper's.
            if(system_name=="mhd" && scheme!="sd"){
                if(!force_level_given)
                    cfg.mood_force_level = 1;
                else if(cfg.mood_force_level != 1 && Master)
                    cout<<"NOTE: job/scheme="<<scheme<<" asks for the MUSCL level but "
                          "mhd/mood_force_level="<<cfg.mood_force_level
                        <<" was set explicitly; the explicit value wins"<<endl;
                if(scheme=="vl2" && cfg.integrator==_integrator_rk_
                   && cfg.rk_order>1 && Master)
                    cout<<"NOTE: job/scheme=vl2 is MUSCL-Hancock, already second "
                          "order in time; pairing it with rk"<<cfg.rk_order
                        <<" applies the time integration twice. time/integrator=rk1 "
                          "is the matching choice."<<endl;
            }
        }

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

        //UCT at DEMOTED corners is standalone-only for now. The SD edge UCT is
        //batched and runs fine under a Mesh (mhd_uct_edge_E_b), but the FV corner
        //composition needs the five face coefficients haloed across block
        //boundaries first, and the only halo for them (MHD_ader::mood_halo_uct)
        //goes through per-block FV_Boundaries that the Mesh lane never
        //initialises -- so the Mesh path has no UCT corners at all and would
        //silently assemble them with the four-state LLF bound instead.
        //Refuse rather than report a run as UCT when half its faces are not:
        //that exact failure -- a scheme named in the input and absent from the
        //build -- is what left both UCT kernels with zero call sites while
        //docs/mhd.md described them as live.
        //MEASURED (Orszag-Tang 16^2, t=0.03, pure SD): single-block and a 2x2
        //decomposition agree to 0.000e+00 under emf=2sweep -- exact bit
        //identity is the invariant every other path in this tree satisfies --
        //but differ by 2.3e-04 (UCT-HLLD) and 4.3e-06 (UCT-HLL) under emf=uct.
        //The face coefficients are only ever written at interfaces the Riemann
        //solve covers, so a ghost element's OUTER face keeps whatever it was
        //initialised with; the edge composition reads it, and which elements are
        //ghosts depends on the decomposition. The batched and per-block UCT
        //kernels agree bit-for-bit with each other, so this is not the pack --
        //it is a missing halo of UCT_fp_*, and fixing it needs an exchange that
        //writes a point layer the existing fp gather does not.
        //Refused rather than warned: a multiblock UCT number would be a function
        //of the block layout, and nothing downstream would say so.
        if(system_name=="mhd" && cfg.emf==_emf_uct_ && use_mesh){
            if(Master) cout<<"ERROR: mhd/emf=uct is single-block only for now. "
                "The UCT face coefficients are not haloed, so a ghost element's "
                "outer face is unwritten and the edge composition reads it: the "
                "solution then depends on the block decomposition (measured "
                "2.3e-04 on Orszag-Tang at 16^2 by t=0.03, against exactly 0 for "
                "mhd/emf=2sweep). Drop meshblock/nx* and amr/max_level for a "
                "single-block run, or use mhd/emf=2sweep."<<endl;
            exit(1);
        }

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
                <<(cfg.integrator==_integrator_ader_ ? "ader" : "rk"+to_string(cfg.rk_order));
            if(system_name=="mhd"){
                cout<<", rsolver = "
                    <<(cfg.rsolver==_rsolver_hlld_ ? "hlld"
                       : (cfg.rsolver==_rsolver_hll_ ? "hll" : "llf"))
                    <<", emf = "
                    <<(cfg.emf==_emf_uct_ ? "uct" : "2sweep")
                    <<", mood_nad_b = "
                    <<(cfg.mood_nad_b==_nad_b_mag_ ? "mag" : "comps")
                    <<", mood_nad_v = "
                    <<(cfg.mood_nad_v==_nad_v_off_ ? "off"
                       : (cfg.mood_nad_v==_nad_v_mag_ ? "mag" : "comps"))
                    <<", mood_nad_scale = "
                    <<(cfg.mood_nad_scale==_nad_scale_relative_ ? "relative"
                       : (cfg.mood_nad_scale==_nad_scale_delta_ ? "delta"
                          : (cfg.mood_nad_scale==_nad_scale_grange_ ? "grange"
                             : "gcfl")));
                if(cfg.mood_force_level>=0)
                    cout<<", mood_force_level = "<<cfg.mood_force_level;
            }
            cout<<", outputs = "<<(cfg.outputs ? "on" : "off")
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
