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

//Runtime configuration chosen from the input file. POD so its fields can
//be copied into locals and captured by device lambdas.
struct RunConfig {
    int  ndim = 3;
    bool active[3] = {true, true, true}; //x, y, z
    int  problem = 0;                    //initial-condition id
    bool fallback = true;                //FV update + fallback scheme
    double gamma = 1.4;
    double cfl = 0.8;
    double nad_tolerance = 1e-5;         //NAD band width
    bool nad_delta = false;              //band scaled by local range instead of |W|
    bool nad_moore = true;               //DMP bounds over the Moore (box) neighborhood
    bool sed = true;                     //smooth extrema detection (only applied for p>1)
    bool blending = true;                //fractional theta blending of fallback fluxes
    int bc[3] = {0, 0, 0};               //boundary type per direction
};
extern RunConfig cfg;

void set_runtime_dimensionality(bool ax, bool ay, bool az);