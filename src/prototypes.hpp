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
#if DIM==3
extern void transform_a_to_b(SD_Solution, SD_Solution, Matrix, Matrix, Matrix);
#elif DIM==2
extern void transform_a_to_b(SD_Solution, SD_Solution, Matrix, Matrix);
#else
extern void transform_a_to_b(SD_Solution, SD_Solution, Matrix);
#endif
extern void transform_a_to_b_1d(SD_Solution, SD_Solution, Matrix, int);

#ifdef _3D_
extern void update_prediction(SD_Solution, SD_Solution, SD_Solution, SD_Solution, SD_Solution, Matrix, Matrix, Vector, double, double, double, double);
extern void update_solution(SD_Solution, SD_Solution, SD_Solution, SD_Solution, Matrix, Vector, double, double, double, double);
#else
extern void update_prediction(SD_Solution, SD_Solution, SD_Solution, SD_Solution, Matrix, Matrix, Vector, double, double, double, double);
extern void update_solution(SD_Solution, SD_Solution, SD_Solution, Matrix, Vector, double, double, double, double);
#endif
extern void update_B_prediction(SD_Solution,SD_Solution,SD_Solution,SD_Solution,Matrix,Matrix,Vector,double,double,double,int);
extern void update_B_solution(SD_Solution,SD_Solution,SD_Solution,Matrix,Vector,double,double,double,int);


//Initial Conditions
#ifdef _3D_
extern void Initialize(SD_Solution,Matrix,Matrix,Matrix,Vector,Vector);
#else
extern void Initialize(SD_Solution,Matrix,Matrix,Vector,Vector);
#endif
extern void Initialize_ep(SD_Solution,Matrix,Matrix,Matrix,int);

//Hydro
extern void compute_conservatives(SD_Solution, SD_Solution);
extern void compute_primitives(SD_Solution, SD_Solution);
extern void compute_fluxes(SD_Solution, SD_Solution, int, int, int);
extern double compute_dt(SD_Solution, double, double, double);

//Induction
extern void rotational_a_to_b(SD_Solution, SD_Solution, SD_Solution, Matrix, double, double, int);
extern void compute_E(SD_Solution, SD_Solution, SD_Solution, SD_Solution, Matrix, Matrix, Matrix, Matrix, int);
extern void compute_B2_cv(SD_Solution, SD_Solution, SD_Solution, SD_Solution, Matrix, Matrix);
extern void compute_rotational_B(SD_Solution, SD_Solution, SD_Solution, double, double, Matrix, Matrix, double, int);
extern double Induction_compute_dt(SD_Solution, double, double, double, Matrix, Matrix, Matrix);

//Riemann Solvers
extern void sd_riemann_solver(SD_Solution, SD_Solution, int, int, int, int);
extern void E_riemann_solver(SD_Solution, int, int);

//Boundary Conditions
extern void boundaries(CommHelper, Boundaries, SD_Solution);
extern void boundaries_x(CommHelper, Boundaries, SD_Solution);
extern void boundaries_y(CommHelper, Boundaries, SD_Solution);
extern void boundaries_z(CommHelper, Boundaries, SD_Solution);

//Finite Volume
extern void face_integral(SD_Solution, FV_Solution, Matrix, int);
//Output
extern void Write(SD_Solution, int);