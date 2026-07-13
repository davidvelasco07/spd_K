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
    int bc[3] = {0, 0, 0};               //boundary type per direction
    int integrator = _integrator_ader_;  //time integrator (ADER or SSP-RK)
    int rk_order = 3;                    //SSP-RK order (1, 2 or 3)
    bool outputs = false;                //file outputs (opt-in via <output> block)
    ProblemParams pp;                    //initial-condition parameters
};
extern RunConfig cfg;

void set_runtime_dimensionality(bool ax, bool ay, bool az);

//SSP (Shu-Osher) Runge-Kutta: every stage is a forward-Euler step with the
//full dt followed by the convex combination U <- a*U0 + (1-a)*U with the
//step-start state U0. Fills a[] and returns the number of stages.
int ssp_rk_coefficients(int order, double* a);