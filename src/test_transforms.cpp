//Unit checks for the directional-sweep transforms against the reference
//tensor-product implementations. Returns nonzero on failure.
#include "spd_k.hpp"

//Deterministic smooth-ish filler so runs are comparable across machines
double test_value(int t, int var, int k, int j, int i, int kk, int jj, int ii){
    return sin(1.7*i + 2.3*j + 3.1*k + 0.7*ii + 1.3*jj + 0.5*kk + 0.9*var + 0.3*t)
         + 0.1*cos(2.9*i - 1.1*jj + 0.8*var);
}

void fill(SD_Solution U){
    for(int t=0; t<U.n_ader; t++)
    for(int var=0; var<U.n_var; var++)
    for(int k=0; k<U.Nz; k++)
    for(int j=0; j<U.Ny; j++)
    for(int i=0; i<U.Nx; i++)
    for(int kk=0; kk<U.nz; kk++)
    for(int jj=0; jj<U.ny; jj++)
    for(int ii=0; ii<U.nx; ii++)
        U.Vector(t,var,k,j,i,kk,jj,ii) = test_value(t,var,k,j,i,kk,jj,ii);
}

double max_diff(SD_Solution A, SD_Solution B){
    double d=0;
    for(int t=0; t<A.n_ader; t++)
    for(int var=0; var<A.n_var; var++)
    for(int k=0; k<A.Nz; k++)
    for(int j=0; j<A.Ny; j++)
    for(int i=0; i<A.Nx; i++)
    for(int kk=0; kk<A.nz; kk++)
    for(int jj=0; jj<A.ny; jj++)
    for(int ii=0; ii<A.nx; ii++)
        d = max(d, abs(A.Vector(t,var,k,j,i,kk,jj,ii)-B.Vector(t,var,k,j,i,kk,jj,ii)));
    return d;
}

double max_diff_fv(FV_Solution A, FV_Solution B){
    double d=0;
    for(int var=0; var<A.n_var; var++)
    for(int k=0; k<A.Nz; k++)
    for(int j=0; j<A.Ny; j++)
    for(int i=0; i<A.Nx; i++)
        d = max(d, abs(A.Vector(var,k,j,i)-B.Vector(var,k,j,i)));
    return d;
}

int check(string name, double diff, double tol){
    bool ok = diff < tol;
    cout<<(ok ? "PASS " : "FAIL ")<<name<<": max diff = "<<diff<<endl;
    return ok ? 0 : 1;
}

int main(int argc, char** argv){
    Kokkos::initialize(argc, argv);
    int failures = 0;
    {
        CommHelper comm;
        cpu_rank = comm.me;
        Master = cpu_rank==0;
        set_runtime_dimensionality(true,true,true);

        int p = 3;
        int N = 4;

        double *x = malloc_host<double>(p);
        double *w = malloc_host<double>(p);
        gauss_legendre(0.0, 1.0, p, x, w);
        double *x_sp = malloc_host<double>(p+1);
        double *x_fp = malloc_host<double>(p+2);
        flux_points(x_fp,x,p);
        solution_points(x_sp,p);

        dimension X_dim(_x_,N,N,X*p,0,1.0,x_fp,X);
        dimension Y_dim(_y_,N,N,Y*p,0,1.0,x_fp,Y);
        dimension Z_dim(_z_,N,N,Z*p,0,1.0,x_fp,Z);

        Matrix sp_to_cv("sp_to_cv",p+1,p+1);
        Matrix cv_to_sp("cv_to_sp",p+1,p+1);
        integral_matrix(sp_to_cv, x_fp, x_sp, p+1, p+1);
        inverse(sp_to_cv, cv_to_sp, p+1);

        int nvar = NVAR;
        SD_Solution U_a  ("U_a"  ,1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        SD_Solution U_ref("U_ref",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        SD_Solution U_new("U_new",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
        SD_Solution T    ("T"    ,1,nvar,Z_dim,Y_dim,X_dim,0,0,0);

        fill(U_a);
        Kokkos::fence();

        //Full transform: reference (p+1)^DIM contraction vs directional sweeps
        transform_a_to_b_ref(U_a, U_ref, sp_to_cv, sp_to_cv, sp_to_cv);
        transform_a_to_b(U_a, U_new, T, sp_to_cv);
        Kokkos::fence();
        failures += check("transform_a_to_b (sp_to_cv)", max_diff(U_ref,U_new), 1e-13);

        transform_a_to_b_ref(U_a, U_ref, cv_to_sp, cv_to_sp, cv_to_sp);
        transform_a_to_b(U_a, U_new, T, cv_to_sp);
        Kokkos::fence();
        failures += check("transform_a_to_b (cv_to_sp)", max_diff(U_ref,U_new), 1e-13);

        //Round trip should recover the input
        transform_a_to_b(U_new, U_ref, T, sp_to_cv);
        Kokkos::fence();
        failures += check("round trip cv->sp->cv", max_diff(U_a,U_ref), 1e-12);

        //Team (single-kernel, scratch-staged) version: same sweep arithmetic,
        //so it must agree with the kernel-per-sweep version bitwise
        transform_a_to_b(U_a, U_ref, T, sp_to_cv);
        transform_a_to_b_team(U_a, U_new, sp_to_cv);
        Kokkos::fence();
        failures += check("transform_a_to_b_team vs sweeps (bitwise)", max_diff(U_ref,U_new), 1e-300);
        transform_a_to_b_team(U_a, U_new, cv_to_sp);
        transform_a_to_b(U_a, U_ref, T, cv_to_sp);
        Kokkos::fence();
        failures += check("transform_a_to_b_team (cv_to_sp, bitwise)", max_diff(U_ref,U_new), 1e-300);

        //Transverse (2d) transform on face-staggered fields, one per direction
        for(int dim=0; dim<3; dim++){
            SD_Solution S_a  ("S_a"  ,1,nvar,Z_dim,Y_dim,X_dim,dim==_z_,dim==_y_,dim==_x_);
            SD_Solution S_ref("S_ref",1,nvar,Z_dim,Y_dim,X_dim,dim==_z_,dim==_y_,dim==_x_);
            SD_Solution S_new("S_new",1,nvar,Z_dim,Y_dim,X_dim,dim==_z_,dim==_y_,dim==_x_);
            SD_Solution S_T  ("S_T"  ,1,nvar,Z_dim,Y_dim,X_dim,dim==_z_,dim==_y_,dim==_x_);
            fill(S_a);
            Kokkos::fence();
            transform_a_to_b_2d_ref(S_a, S_ref, sp_to_cv, dim);
            transform_a_to_b_2d(S_a, S_new, S_T, sp_to_cv, dim);
            Kokkos::fence();
            failures += check("transform_a_to_b_2d dim="+to_string(dim), max_diff(S_ref,S_new), 1e-13);
        }

        //Face integral of an ADER flux slice, one per direction
        int nader = 3;
        for(int dim=0; dim<3; dim++){
            SD_Solution F_fp("F_fp",nader,nvar,Z_dim,Y_dim,X_dim,dim==_z_,dim==_y_,dim==_x_);
            SD_Solution F_T ("F_T" ,1    ,nvar,Z_dim,Y_dim,X_dim,dim==_z_,dim==_y_,dim==_x_);
            FV_Solution F_ref("F_ref",nvar,Z_dim,Y_dim,X_dim,dim==_z_,dim==_y_,dim==_x_);
            FV_Solution F_new("F_new",nvar,Z_dim,Y_dim,X_dim,dim==_z_,dim==_y_,dim==_x_);
            fill(F_fp);
            Kokkos::fence();
            for(int t_id=0; t_id<nader; t_id++){
                face_integral_ref(F_fp, F_ref, sp_to_cv, t_id, dim);
                face_integral(F_fp, F_new, F_T, sp_to_cv, t_id, dim);
                Kokkos::fence();
                failures += check("face_integral dim="+to_string(dim)+" t="+to_string(t_id),
                                  max_diff_fv(F_ref,F_new), 1e-13);
            }
        }
    }
    Kokkos::finalize();
    if(failures)
        cout<<failures<<" TEST(S) FAILED"<<endl;
    else
        cout<<"ALL TESTS PASSED"<<endl;
    return failures;
}
