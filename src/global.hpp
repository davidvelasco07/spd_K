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
    double g[3] = {0.0, 0.0, 0.0};       //constant gravitational acceleration (source term)
    double nad_tolerance = 1e-5;         //NAD band width
    bool nad_delta = false;              //band scaled by local range instead of |W|
    bool nad_moore = true;               //DMP bounds over the Moore (box) neighborhood
    bool sed = true;                     //smooth extrema detection (only applied for p>1)
    bool blending = true;                //fractional theta blending of fallback fluxes
    bool muscl_only = false;             //take every face from the MUSCL fallback (theta=1)
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
    bool outputs = false;                //file outputs (opt-in via <output> block)
    ProblemParams pp;                    //initial-condition parameters
    int adapt_interval = 0;              //0 = no dynamic AMR; else adapt every N steps
    int amr_max_level = 0;               //maximum refinement level
    int amr_criterion = 0;               //0=Lohner, 1=pressure, 2=trouble, 3=shear, 4=bfield
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

void set_runtime_dimensionality(bool ax, bool ay, bool az);

//SSP (Shu-Osher) Runge-Kutta: every stage is a forward-Euler step with the
//full dt followed by the convex combination U <- a*U0 + (1-a)*U with the
//step-start state U0. Fills a[] and returns the number of stages.
int ssp_rk_coefficients(int order, double* a);