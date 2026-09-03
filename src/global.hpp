//MPI global variables
extern int cpu_rank;
extern int cpu_size;
extern MPI_Comm Comm;
extern int cpu_x;
extern int cpu_y;
extern int cpu_z;
extern int rank_x;
extern int rank_y;
extern int rank_z;
extern int x_i;
extern int y_i;
extern int z_i;

extern int Master;
extern int N_comms;

//Runtime parameters of the initial condition, parsed from the <problem>
//block. POD, passed by value into the Initialize kernel so device code can
//read it directly. Each problem uses the subset it needs and ignores the
//rest (see problem_defaults in main.cpp for the per-problem defaults).
struct ProblemParams {
    double amp = 0.125;   //perturbation amplitude
    double v1 = 1.0;      //background/primary velocity
    double v2 = 1.0;      //secondary velocity (shear/transverse)
    double v3 = 0.0;      //tertiary velocity
    double d0 = 1.0;      //primary density (left/inner)
    double d1 = 0.125;    //secondary density (right/outer)
    double p0 = 1.0;      //primary pressure (left/inner)
    double p1 = 0.1;      //secondary pressure (right/outer)
    double radius = 0.1;  //feature radius / interface position
    double sigma = 0.05;  //smoothing/perturbation width
    int dir = 0;          //direction of 1d profiles (0=x, 1=y, 2=z)
    double cx = 0.5;      //domain center x (set from the box length at runtime)
    double cy = 0.5;      //domain center y
    double cz = 0.5;      //domain center z
};

//Runtime configuration chosen from the input file. POD so its fields can
//be copied into locals and captured by device lambdas.
struct RunConfig {
    int  ndim = 3;
    bool active[3] = {true, true, true}; //x, y, z
    int  problem = 0;                    //initial-condition id
    bool fallback = true;                //FV update + fallback scheme
    double gamma = 1.4;
    double cfl = 0.8;
    int  cfl_type = _cfl_sum_;           //time/cfl_type: sum (default) | min
    double g[3] = {0.0, 0.0, 0.0};       //constant gravitational acceleration (source term)
    double nad_tolerance = 1e-5;         //NAD band width (rtol)
    double nad_atol = 0.0;               //absolute floor on the NAD band (AthenaK mood_atol)
    double nad_eps0 = 1e-12;             //relative floor eps0*|bound| (AthenaK mood_eps0)
    bool nad_delta = false;              //legacy local-range band (fallback/NAD=delta)
    int mood_nad_scale = _nad_scale_gcfl_; //MHD NAD tolerance scale (AthenaK mood_nad_scale):
                                         //relative | delta | grange | gcfl (default)
    int mood_force_level = -1;           //MOOD diagnostic: -1 = normal detect/demote;
                                         //0/1/2 = force that cascade level everywhere
                                         //(skip detection). 1 = MUSCL, 2 = first order.
    bool nad_moore = true;               //DMP bounds over the Moore (box) neighborhood
    bool sed = true;                     //smooth extrema detection (only applied for p>1)
    bool blending = true;                //fractional theta blending of fallback fluxes
    bool fv_only = false;                //take every face from the FV flux (theta=1):
                                         //the low-order lane, job/scheme=vl2|plm
    bool fv_predictor = true;            //FV reconstruction: true = MUSCL-Hancock
                                         //(job/scheme=vl2), false = plain PLM
                                         //(job/scheme=plm), whose time accuracy
                                         //comes from the outer integrator
    bool mood_cascade = false;           //fallback style: false = fractional theta blend
                                         //(matches the Python reference), true = discrete
                                         //MOOD cascade levels as the MHD module uses.
                                         //The blend is the default so reference parity
                                         //and the existing goldens are unaffected.
    int max_revs = 3;                    //cap on MOOD detection/revision sweeps; the loop
                                         //exits early once no revisable troubled cell
                                         //remains, so this is a safety cap, not a cost.
                                         //Truncating it commits unverified candidates
                                         //(they lean on the ctoprim floors).
    double pad_min_rho = 1e-10;          //PAD floors (runtime-tunable detection
    double pad_min_P   = 1e-10;          //strictness, cf. fallback min_rho/min_P)
    bool floor_cons = false;             //ctoprim floor semantics: false = RAMSES
                                         //(primitive view only; matches Python spd),
                                         //true = AthenaK (repair the conserved state)
    double dfloor = 1e-10;               //ctoprim density floor
    double pfloor = -1.0;                //ctoprim pressure floor; <0 derives the
                                         //RAMSES smallp from dfloor
    int bc[3] = {0, 0, 0};               //boundary type per direction
    int integrator = _integrator_ader_;  //time integrator (ADER or SSP-RK)
    int rk_order = 3;                    //SSP-RK order (1, 2 or 3)
    int nlim = -1;                       //step cap (-1 = unlimited). Bounds a
                                         //throughput measurement by steps
                                         //rather than by an end time.

    int rsolver = _rsolver_llf_;         //MHD Riemann solver (faces + edges)
    int emf = _emf_2sweep_;
    int mhd_energy_fix = 0;              //MDZ21 6.4 energy correction, as a BITMASK
                                         //over the three places a state's B rows
                                         //are replaced by the CT field without the
                                         //energy following (so p = (g-1)(E - Ekin
                                         //- B^2/2) is then read off a B that E was
                                         //never built from):
                                         //  1  SD  mhd_B_to_U -- the END OF A
                                         //        STAGE, where the Godunov B is
                                         //        replaced wholesale by the CT
                                         //        field. This is the one MDZ21
                                         //        prescribes and the only one
                                         //        that should normally be on.
                                         //  2  FV  mhd_set_candidate_B (the PAD
                                         //        pressure is computed here)
                                         //  4  fp  mhd_face_B_to_fp (the state
                                         //        the Riemann solver sees)
                                         //2 and 4 sit MID-UPDATE rather than at a
                                         //stage boundary, where the cell-centred
                                         //B is a working value and not yet the
                                         //garbage the stage-end swap discards.
                                         //They exist to be measured, not used.
                                         //0 = off (default; the correction moves
                                         //every MHD result, so decks opt in).
                                         //A bit per site so a measurement can be
                                         //attributed to one of them (rule 2).              //MHD electromotive force (mhd/emf):
                                         //2sweep = two 1-D edge Riemann sweeps at
                                         //SD edges + the four-state LLF bound at
                                         //demoted corners (the default; every MHD
                                         //golden encodes it), uct = upwind
                                         //constrained transport (MDZ21 eq. 33),
                                         //whose flavour follows rsolver
                                         //(hll -> UCT-HLL, hlld -> UCT-HLLD).
    int limiter = _lim_minmod_;          //MUSCL/FV slope limiter (fallback/limiter)
    int mood_nad_b = _nad_b_comps_;      //MHD NAD B mode: comps (default) or mag
                                         //(|B|-only is blind to Alfvénic / transverse
                                         //structure; matches AthenaK mood_nad_b=comps
                                         //and Python spd limiting_variables)
    bool mood_pad_first_order = false;   //when true, only a PAD failure (negative
                                         //density/pressure, non-finite) may demote
                                         //a cell to FIRST ORDER; a NAD flag alone
                                         //stops at MUSCL. First order is the
                                         //positivity last resort, so it answers to
                                         //physics rather than to ringing.
    int mood_max_level = 2;              //deepest cascade tier the detector may
                                         //demote to: 0 = high order only,
                                         //1 = stop at MUSCL, 2 = allow first
                                         //order (the historical behaviour).
                                         //Setting 1 makes the fallback bound the
                                         //solution from above by MUSCL, which is
                                         //what a cascade is supposed to guarantee
                                         //and what spd_K currently does NOT do.
    bool mood_tier_exclude = true;       //drop cells pinned at the bottom cascade
                                         //tier out of detection, as AthenaK does
                                         //(mhd/mood_tier_exclude=false restores
                                         //the previous always-flag behaviour).
    int mood_nad_v = _nad_v_off_;        //MHD NAD velocity: off (default) / mag / comps.
                                         //AthenaK HLLD+FB ringing-stable configs use comps
                                         //together with mood_nad_b=comps (and a global NAD
                                         //scale); keep off unless needed — relative NAD on
                                         //near-zero velocity components over-triggers.
    bool outputs = false;                //file outputs (opt-in via <output> block)
    ProblemParams pp;                    //initial-condition parameters
    //Prescribed-inflow signal speed, for the TIMESTEP.
    //
    //A jet that enters only through a boundary is invisible to a dt computed
    //from the interior: at t = 0 the domain is quiescent, so the CFL condition
    //sees only the ambient sound speed. On the Ha et al. jet that gave
    //dt = 9.99e-04 against a tlim of 1e-03 -- ONE step for the whole run --
    //while the state about to enter carries v_x = 800. The boundary is part of
    //the problem, so its signal speed has to bound the step.
    //
    //Held as the prescribed PRIMITIVE state (it is time-independent) and folded
    //into ComputeDt with the same CFL form the interior uses, so cfl_type is
    //honoured. inflow_rho <= 0 means "no prescribed inflow", which is the
    //default and makes this a no-op for every existing problem.
    //What the prescribed-inflow face does OUTSIDE the nozzle. The two papers
    //that specify this jet disagree, so it is a parameter rather than a guess:
    //  _jo_outflow_   sentinel -> plain outflow copy (the MHD jet's contract,
    //                 and the closest thing to RR23's characteristics-based BC)
    //  _jo_ambient_   clamp to the ambient state (rho = d0, v = 0)
    //  _jo_reservoir_ clamp to the JET density at rest (rho = d1, v = 0) --
    //                 Fu 2019 (CPC 244, 117) section 4.3.4 states this
    //                 explicitly: "(rho,u,p) = (5,0,0,0.4127) otherwise", i.e.
    //                 rho = 5 on the whole left face, not the ambient 0.5.
    int inflow_outside = 0;
    double inflow_rho = 0.0, inflow_p = 0.0;
    double inflow_vx = 0.0, inflow_vy = 0.0, inflow_vz = 0.0;
    int adapt_interval = 0;              //0 = no dynamic AMR; else adapt every N steps
    int amr_max_level = 0;               //maximum refinement level
    int amr_criterion = 0;               //0=Lohner, 1=pressure, 2=trouble, 3=shear, 4=bfield
    //Bound the p>=1 Lagrange prolongation by a discrete maximum principle taken
    //from the coarse neighbourhood, on top of the PAD limiter. Off keeps the
    //unlimited interpolation, which rings at sharp interfaces; on can clip
    //legitimate sub-element structure, so it is a switch, not a constant.
    bool amr_prolong_dmp = false;
    //Refine/derefine cuts for the shear criterion. shear_score is an undivided
    //velocity difference between neighbouring cells, so it carries a factor of
    //the cell size: halving the root resolution roughly doubles the score, and
    //a threshold calibrated at one resolution must be rescaled for another.
    //Too small a value tags the whole domain once vorticity has spread (0.01
    //leaves a Kelvin-Helmholtz mesh 97% refined at the finest level by t=0.9).
    double amr_refine_threshold = 0.1;
    double amr_derefine_threshold = 0.05;
    //Cut on the filtered |B| Löhner indicator, which is dimensionless and O(1)
    //at an under-resolved feature.
    double amr_bfield_threshold = 0.8;
    //Fraction of the peak score a block must reach to be tagged. 1 refines only
    //the peak block(s) and lets 2:1 balance grow a single compact patch; 0 tags
    //every block over the threshold.
    bool amr_initial_refine = false;  //iterate refinement at t=0 (Athena++ style)
    double amr_refine_frac = 0.0;
    //Fraction of the peak score below which a sibling group is released, so the
    //fine region follows the feature instead of accumulating. 0 disables it and
    //leaves only the absolute cut at half the refine threshold.
    double amr_derefine_frac = 0.0;
};
extern RunConfig cfg;

//AMR prolongation / overlap-restriction matrices (built once at startup)
extern Matrix amr_P;
extern Matrix amr_R;
extern Matrix amr_RS_sp[2];
extern Matrix amr_RS_cv[2];
//Conservative face-flux restriction (n x 2n), exact on the face integral
extern Matrix amr_RF;
//Same operators on the flux-point lattice (edge EMF / face-normal B along
//the staggered dim). Built from x_fp; size (p+2) instead of (p+1).
extern Matrix amr_P_fp;
extern Matrix amr_RF_fp;
//Flux-point nodes in [0,1] (length p+2); used to fill interior normal FPs
//after face-only Toth–Roe prolongation of face B.
extern Vector amr_x_fp;
void init_amr_transfer_matrices(double* x_sp, double* x_fp, int p);

//Sets cfg.active/ndim and the per-direction ghost widths NGH_rt/nGH_rt. Needs
//the polynomial degree because the FV halo is drawn from the SD ghost elements
//(n_sp = p+1 sub-cells each), so p decides how wide an FV halo the SD grid can
//source. Must run before any dimension/Block is constructed -- they size their
//arrays from these globals.
void set_runtime_dimensionality(bool ax, bool ay, bool az, int p);

//SSP (Shu-Osher) Runge-Kutta: every stage is a forward-Euler step with the
//full dt followed by the convex combination U <- a*U0 + (1-a)*U with the
//step-start state U0. Fills a[] and returns the number of stages.
int ssp_rk_coefficients(int order, double* a);