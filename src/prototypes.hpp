//Polynomials
extern double lagrange(double*,double,int,int);
extern double lagrange_prime(double*,double,int,int);
extern void lagrange_matrix(Matrix,double*,double*,int,int);
extern void lagrange_matrix(Matrix_h,double*,double*,int,int);
extern void lagrange_prime_matrix(Matrix,double*,double*,int,int);
extern void lagrange_prime_matrix(Matrix_h,double*,double*,int,int);
extern void gauss_legendre(double, double, int, double*, double*);
extern void flux_points(double*, double*, int);
extern void solution_points(double *, int);
extern void ader_matrix(Matrix, Vector, Vector, int);
extern void ader_matrix(Matrix_h, Vector_h, Vector_h, int);
extern void integral_matrix(Matrix, double*, double*, int, int);
extern void integral_matrix(Matrix_h, double*, double*, int, int);
extern void inverse(Matrix, Matrix, int);
extern void inverse(Matrix_h, Matrix_h, int);

//Transforms
extern void transform_a_to_b_ref(SD_Solution, SD_Solution, Matrix, Matrix, Matrix);
extern void transform_a_to_b(SD_Solution, SD_Solution, SD_Solution, Matrix);
extern void transform_a_to_b_1d(SD_Solution, SD_Solution, Matrix, int);
extern void transform_a_to_b_1d_slice(SD_Solution, SD_Solution, Matrix, int, int);
extern void transform_a_to_b_2d_ref(SD_Solution, SD_Solution, Matrix, int);
extern void transform_a_to_b_2d(SD_Solution, SD_Solution, SD_Solution, Matrix, int);
extern void combine_solution(SD_Solution, SD_Solution, double);

//AMR transfer operators (prolongation/restriction between levels)
extern void transfer_matrices(Matrix, Matrix, double*, int);
extern void build_overlap_restrict_matrices(Matrix R_sp[2], Matrix R_cv[2],
                                            double* x_sp, double* x_fp, int p);
extern void prolongate_block(SD_Solution, SD_Solution, Matrix, int, int, int);
extern void restrict_block(SD_Solution, SD_Solution, Matrix, int, int, int);
extern void restrict_face_overlap_sp(const SD_Solution** fine_faces, int n_sub,
                                     SD_Solution coarse_face, Matrix R, int dim);
extern void prolongate_face_coarser(SD_Solution coarse_face, SD_Solution fine_face,
                                    Matrix P, int dim, int sub);
//Face-staggered B (CT): restrict one normal-component field from a fine child
//onto the covered coarse subregion (transverse amr_RF, inject along face normal).
extern void restrict_block_face_B(SD_Solution fine, SD_Solution coarse, Matrix R,
                                  int face_dim, int cx, int cy, int cz);
//Restrict face B from one fine child onto its quadrant of the coarse block,
//preserving discrete divB=0 (2D). Coarse faces are the averages of the fine
//faces they cover, so the fine faces interior to a coarse element cancel.
extern void restrict_block_face_B_2d(SD_Solution BxF, SD_Solution ByF, SD_Solution BzF,
                                     SD_Solution BxC, SD_Solution ByC, SD_Solution BzC,
                                     Matrix R, Matrix sp_to_cv, Matrix cv_to_sp,
                                     int cx, int cy, int cz);
//Prolongate face B from coarse parent to fine child. Shared faces: transverse
//amr_P. Interior faces (2D): Toth–Roe on face averages so discrete divB=0.
//3D mixed-level errors out (Stage 4 is 2D-first).
extern void prolongate_block_face_B(SD_Solution BxC, SD_Solution ByC, SD_Solution BzC,
                                    SD_Solution BxF, SD_Solution ByF, SD_Solution BzF,
                                    Matrix P, Matrix sp_to_cv, Matrix cv_to_sp,
                                    int cx, int cy, int cz);
//Zero per-element FV/SD divergence of face-constant B (2D) by adjusting +x faces.
extern void project_face_B_divfree_2d(SD_Solution Bx, SD_Solution By);
extern void gather_block(SD_Solution, SD_Solution, int, int, int);
extern void init_amr_transfer_matrices(double* x_sp, double* x_fp, int p);

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
//Block-to-block ghost exchange (multi-block, single rank)
extern void block_boundary_sd(SD_Solution, SD_Solution, SD_Solution, int, int, int);
extern void block_boundary_fv(FV_Solution, FV_Solution, FV_Solution, int, int, int);
//Whole-pack ghost exchange: neighbour indices and side types come in as
//per-block tables, so one launch replaces the per-block loop.
extern void apply_domain_bc_fp(SD_Solution, int, int);
extern void apply_domain_bc_fv(FV_Solution, int, int, int);
extern void gather_fp_same(SD_Solution, IntVector, IntVector, int, int, int);
extern void gather_fv_same(FV_Solution, IntVector, IntVector, int, int, int, int);
extern void gather_fv_coarser(FV_Solution, IntVector, IntVector, IntVector,
                              int, int, int, int);
extern void gather_fv_finer(FV_Solution, IntVector, IntVector, IntVector,
                            int, int, int, int, bool);
//Conservative flux correction off the same fine->coarse table.
extern void correct_cf_fv_flux_b(FV_Solution, IntVector, IntVector, IntVector,
                                 int, int, int, int, int,
                                 int, int, int, int, int, int);
extern void gather_fp_coarser(SD_Solution, IntVector, IntVector, IntVector,
                              int, int, int, Matrix);
extern void gather_fp_finer(SD_Solution, IntVector, IntVector, IntVector,
                            int, int, int, Matrix);
extern void block_boundary_sd_b(SD_Solution, IntVector, IntVector,
                                IntVector, IntVector, int);
extern void block_boundary_fv_b(FV_Solution, IntVector, IntVector,
                                IntVector, IntVector, int);
//Make the shared interface face of a face-staggered field identical on both
//sides of a same-level block boundary (left's last-active right face is the
//canonical value). No-op for _gradfree_ ends.
extern void sync_shared_face_sd(SD_Solution U, SD_Solution UL, SD_Solution UR,
                                int typeL, int typeR, int dim);

//Finite Volume
extern void face_integral_ref(SD_Solution, FV_Solution, Matrix, int, int);
extern void face_integral(SD_Solution, FV_Solution, SD_Solution, Matrix, int, int);
extern void edge_integral(SD_Solution, FV_Solution, Matrix, int, int);
extern void fv_update_solution(
    FV_Solution, FV_Solution, SD_Solution,
    FV_Solution, Vector, FV_Solution, Vector, FV_Solution, Vector,
    Vector, int, double, bool);

//Batched (whole-pack) counterparts; geometry arrives as one row per block
extern void transform_a_to_b_1d_slice_b(SD_Solution, SD_Solution, Matrix, int, int);
extern void face_integral_ref_b(SD_Solution, FV_Solution, Matrix, int, int);
extern void face_integral_b(SD_Solution, FV_Solution, SD_Solution, Matrix, int, int);
extern void fv_update_solution_b(
    FV_Solution, FV_Solution, SD_Solution,
    FV_Solution, Matrix, FV_Solution, Matrix, FV_Solution, Matrix,
    Vector, int, double, bool);

//Trouble detection. detect_troubles takes (W_new, W_old, troubles, cascade,
//alpha_x, alpha_y, alpha_z, ...): the per-variable flags stay block-local in
//`troubles`, and only the pooled per-cell level in `cascade` is read
//downstream, so only `cascade` needs a halo.
extern void detect_troubles(
    FV_Solution, FV_Solution, FV_Solution, FV_Solution,
    FV_Solution, FV_Solution, FV_Solution,
    dimension, dimension, dimension, bool, int);
//Whole-pack detection: same criteria, one launch each rather than one per
//block. Geometry comes in as packed Matrices (a row per block) because blocks
//at different refinement levels have different spacings.
extern void detect_troubles_b(
    FV_Solution, FV_Solution, FV_Solution, FV_Solution,
    FV_Solution, FV_Solution, FV_Solution,
    Matrix, Matrix, Matrix, Matrix, Matrix, Matrix,
    int, bool, int);
extern int  update_cascade_b(FV_Solution, FV_Solution, int);
extern void apply_blending(FV_Solution, FV_Solution);
extern void blending_ring(FV_Solution, FV_Solution);
extern void theta_from_flagged(FV_Solution, FV_Solution);
//Whole-pack twins of the theta/assemble kernels.
extern void apply_blending_b(FV_Solution, FV_Solution);
extern void blending_ring_b(FV_Solution, FV_Solution);
extern void theta_from_flagged_b(FV_Solution, FV_Solution);
extern void assign_face_flux_b(FV_Solution, FV_Solution, FV_Solution,
                               FV_Solution, int);
//MOOD cascade, shared by hydro and MHD (see trouble_detection.cpp).
extern int  update_cascade(FV_Solution flagged, FV_Solution cascade, int n_cascade);
extern void assign_face_flux(FV_Solution F0, FV_Solution F1, FV_Solution F2,
                             FV_Solution cascade, int dim);

void fallback_fluxes(
    FV_Solution, FV_Solution,
    Vector, Vector, FV_Solution,
    Vector, Vector, FV_Solution,
    Vector, Vector, FV_Solution,
    int, Vector, double);

//One MOOD cascade level's unblended face fluxes (muscl = limited slopes,
//else donor cell). Same argument order as fallback_fluxes minus theta.
void level_fluxes(
    FV_Solution,
    Vector, Vector, FV_Solution,
    Vector, Vector, FV_Solution,
    Vector, Vector, FV_Solution,
    int, Vector, double, bool);

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
class BlockForest;
//Leaf-block metadata for AMR visualization (text): one file per output index.
extern void Write_amr_blocks(const BlockForest& forest, int n_output,
                             int NBx, int NBy, int NBz);