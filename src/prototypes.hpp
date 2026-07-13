//Polynomials
extern double lagrange(double*,double,int,int);
extern double lagrange_prime(double*,double,int,int);
extern void lagrange_matrix(Matrix,double*,double*,int,int);
extern void lagrange_prime_matrix(Matrix,double*,double*,int,int);
extern void gauss_legendre(double, double, int, double*, double*);
extern void flux_points(double*, double*, int);
extern void solution_points(double *, int);
extern void ader_matrix(Matrix, Vector, Vector, int);
extern void integral_matrix(Matrix, double*, double*, int, int);
extern void inverse(Matrix, Matrix, int);

//Transforms
extern void transform_a_to_b_ref(SD_Solution, SD_Solution, Matrix, Matrix, Matrix);
extern void transform_a_to_b(SD_Solution, SD_Solution, SD_Solution, Matrix);
extern void transform_a_to_b_1d(SD_Solution, SD_Solution, Matrix, int);
extern void transform_a_to_b_1d_slice(SD_Solution, SD_Solution, Matrix, int, int);
extern void transform_a_to_b_2d_ref(SD_Solution, SD_Solution, Matrix, int);
extern void transform_a_to_b_2d(SD_Solution, SD_Solution, SD_Solution, Matrix, int);
extern void combine_solution(SD_Solution, SD_Solution, double);

extern void update_prediction(SD_Solution, SD_Solution, SD_Solution, SD_Solution, SD_Solution, Matrix, Matrix, Vector, double, double, double, double);
extern void update_solution(SD_Solution, SD_Solution, SD_Solution, SD_Solution, SD_Solution, Matrix, Vector, double, double, double, double);
extern void update_B_prediction(SD_Solution,SD_Solution,SD_Solution,SD_Solution,Matrix,Matrix,Vector,double,double,double,int);
extern void update_B_solution(SD_Solution,SD_Solution,SD_Solution,Matrix,Vector,double,double,double,int);


//Initial Conditions
extern void Initialize(SD_Solution,Matrix,Matrix,Matrix,Vector,Vector);
extern void Initialize_ep(SD_Solution,Matrix,Matrix,Matrix,int);

//Hydro
extern void compute_conservatives(SD_Solution, SD_Solution);
extern void compute_primitives(SD_Solution, SD_Solution);
extern void compute_fluxes(SD_Solution, SD_Solution, int, int, int);
extern double compute_dt(SD_Solution, double, double, double, double nu=0.0);
extern void compute_gradient(SD_Solution, SD_Solution, double, Matrix, int);
extern void compute_viscous_flux(SD_Solution, SD_Solution, int, SD_Solution, int, SD_Solution, int, Matrix, double, double, int);


extern void compute_conservatives(FV_Solution, FV_Solution);
extern void compute_primitives(FV_Solution, FV_Solution);


//Induction
extern void rotational_a_to_b(SD_Solution, SD_Solution, SD_Solution, Matrix, double, double, int);
extern void compute_E(SD_Solution, SD_Solution, SD_Solution, SD_Solution, Matrix, Matrix, Matrix, Matrix, int);
extern void compute_B2_cv(SD_Solution, SD_Solution, SD_Solution, SD_Solution, Matrix, Matrix);
extern void compute_B_cv_from_cf(FV_Solution, SD_Solution, SD_Solution, SD_Solution, Matrix);
extern void compute_rotational_B(SD_Solution, SD_Solution, SD_Solution, double, double, Matrix, Matrix, double, int);
extern double Induction_compute_dt(SD_Solution, double, double, double, Matrix, Matrix, Matrix);

//Riemann Solvers
extern void sd_riemann_solver(SD_Solution, SD_Solution, int, int, int, int, bool);
extern void sd_rusanov_solver(SD_Solution, SD_Solution, int dim);
extern void E_riemann_solver(SD_Solution, int, int);
extern void E_Ohmic_riemann_solver(SD_Solution, int, int);

//Boundary Conditions
extern void boundaries(CommHelper, Boundaries, SD_Solution);
extern void boundaries(CommHelper, FV_Boundaries, FV_Solution, int, int);

//Finite Volume
extern void face_integral_ref(SD_Solution, FV_Solution, Matrix, int, int);
extern void face_integral(SD_Solution, FV_Solution, SD_Solution, Matrix, int, int);
extern void edge_integral(SD_Solution, FV_Solution, Matrix, int, int);
extern void fv_update_solution(
    FV_Solution, FV_Solution, SD_Solution,
    FV_Solution, Vector, FV_Solution, Vector, FV_Solution, Vector,
    Vector, int, double, bool);

//Trouble detection
extern void detect_troubles(
    FV_Solution, FV_Solution, FV_Solution,
    FV_Solution, FV_Solution, FV_Solution,
    dimension, dimension, dimension, bool, int);
extern void apply_blending(FV_Solution, FV_Solution);
extern void blending_ring(FV_Solution, FV_Solution);
extern void theta_from_troubles(FV_Solution, FV_Solution);

void fallback_fluxes(
    FV_Solution, FV_Solution,
    Vector, Vector, FV_Solution,
    Vector, Vector, FV_Solution,
    Vector, Vector, FV_Solution,
    int, Vector, double);

void fv_update_B_solution(
    FV_Solution,
    FV_Solution,
    SD_Solution,
    FV_Solution,
    FV_Solution,
    Vector,
    Vector,
    Vector,
    double,
    int,
    int,
    bool);

//Output
extern string output_folder();
extern void Write(SD_Solution, int);
extern void Write(FV_Solution, int);
extern void Write_dimensions(dimension, dimension, dimension);