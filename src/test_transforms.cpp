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

        //AMR transfer operators: prolongation is exact for (elementwise)
        //degree-p polynomial data, and restriction inverts it
        {
            Matrix P("P",2*(p+1),p+1);
            Matrix R("R",p+1,2*(p+1));
            transfer_matrices(P,R,x_sp,p);

            SD_Solution C ("C" ,1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
            SD_Solution C2("C2",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
            SD_Solution F ("F" ,1,nvar,Z_dim,Y_dim,X_dim,0,0,0);

            //Nodal values of a global degree-p polynomial on the block
            //domain [0,1]^3 (h = 1/N per dim)
            auto poly = [](int var, double x, double y, double z){
                return (1.0+var) + x*x*x - 2.0*y*y + 0.5*z + x*y - z*z*x;
            };
            double h = 1.0/N;
            for(int var=0; var<nvar; var++)
            for(int k=0; k<C.Nz; k++)
            for(int j=0; j<C.Ny; j++)
            for(int i=0; i<C.Nx; i++)
            for(int kk=0; kk<C.nz; kk++)
            for(int jj=0; jj<C.ny; jj++)
            for(int ii=0; ii<C.nx; ii++)
                C.Vector(0,var,k,j,i,kk,jj,ii) =
                    poly(var,(i-NGHx+x_sp[ii])*h,(j-NGHy+x_sp[jj])*h,(k-NGHz+x_sp[kk])*h);
            //ghosts are untouched by the child loop below; copy them so the
            //identity check can compare full arrays
            Kokkos::deep_copy(C2.Vector,C.Vector);

            double dmax=0;
            for(int c=0; c<8; c++){
                int cx=c&1, cy=(c>>1)&1, cz=(c>>2)&1;
                prolongate_block(C,F,P,cx,cy,cz);
                Kokkos::fence();
                //child (cx,cy,cz) covers fine elements [c*N, c*N+N) of the
                //2N-element fine grid with spacing h/2
                for(int var=0; var<nvar; var++)
                for(int k=NGHz; k<F.Nz-NGHz; k++)
                for(int j=NGHy; j<F.Ny-NGHy; j++)
                for(int i=NGHx; i<F.Nx-NGHx; i++)
                for(int kk=0; kk<F.nz; kk++)
                for(int jj=0; jj<F.ny; jj++)
                for(int ii=0; ii<F.nx; ii++){
                    double x = (cx*N+i-NGHx+x_sp[ii])*0.5*h;
                    double y = (cy*N+j-NGHy+x_sp[jj])*0.5*h;
                    double z = (cz*N+k-NGHz+x_sp[kk])*0.5*h;
                    dmax = max(dmax, abs(F.Vector(0,var,k,j,i,kk,jj,ii)-poly(var,x,y,z)));
                }
                restrict_block(F,C2,R,cx,cy,cz);
                Kokkos::fence();
            }
            failures += check("prolongate_block exact (degree-p data)", dmax, 1e-12);
            //the degree-(2p+1) restriction basis amplifies round-off by ~1e2
            failures += check("restrict(prolongate) identity", max_diff(C,C2), 1e-11);
        }

        //AMR face transfer across a coarse-fine interface. The trace a coarse
        //block hands to a fine neighbour is identity along the face normal
        //(both blocks resolve that direction identically at the interface) and
        //refined in the two transverse directions, where the fine block spans
        //only half the coarse block's extent. Restriction must invert it.
        {
            init_amr_transfer_matrices(x_sp, x_fp, p);
            auto poly = [](int var, double x, double y, double z){
                return (1.0+var) + x*x*x - 2.0*y*y + 0.5*z + x*y - z*z*x;
            };
            double h = 1.0/N;
            //which half of the coarse block `sub` selects along transverse axis d
            auto sub_half = [&](int sub, int dim, int d){
                int bit=0;
                for(int dd=0; dd<3; dd++){
                    if(dd==dim || !cfg.active[dd]) continue;
                    if(dd==d) return (sub>>bit)&1;
                    bit++;
                }
                return 0;
            };
            int n_sub = 4;  //3d: two transverse directions
            double dpro=0, dres=0;
            for(int dim=0; dim<3; dim++){
                SD_Solution C("Cface",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
                for(int var=0; var<nvar; var++)
                for(int k=0;k<C.Nz;k++) for(int j=0;j<C.Ny;j++) for(int i=0;i<C.Nx;i++)
                for(int kk=0;kk<C.nz;kk++) for(int jj=0;jj<C.ny;jj++) for(int ii=0;ii<C.nx;ii++)
                    C.Vector(0,var,k,j,i,kk,jj,ii) =
                        poly(var,(i-NGHx+x_sp[ii])*h,(j-NGHy+x_sp[jj])*h,(k-NGHz+x_sp[kk])*h);

                SD_Solution sub_f[4];
                const SD_Solution* ptrs[4];
                for(int s=0; s<n_sub; s++){
                    sub_f[s] = SD_Solution("Fface",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
                    prolongate_face_coarser(C, sub_f[s], amr_P, dim, s);
                    Kokkos::fence();
                    ptrs[s] = &sub_f[s];
                    int sx=sub_half(s,dim,_x_), sy=sub_half(s,dim,_y_), sz=sub_half(s,dim,_z_);
                    for(int var=0; var<nvar; var++)
                    for(int k=NGHz;k<C.Nz-NGHz;k++) for(int j=NGHy;j<C.Ny-NGHy;j++)
                    for(int i=NGHx;i<C.Nx-NGHx;i++)
                    for(int kk=0;kk<C.nz;kk++) for(int jj=0;jj<C.ny;jj++)
                    for(int ii=0;ii<C.nx;ii++){
                        double x = dim==_x_ ? (i-NGHx+x_sp[ii])*h
                                            : (sx*N+i-NGHx+x_sp[ii])*0.5*h;
                        double y = dim==_y_ ? (j-NGHy+x_sp[jj])*h
                                            : (sy*N+j-NGHy+x_sp[jj])*0.5*h;
                        double z = dim==_z_ ? (k-NGHz+x_sp[kk])*h
                                            : (sz*N+k-NGHz+x_sp[kk])*0.5*h;
                        dpro = max(dpro, abs(sub_f[s].Vector(0,var,k,j,i,kk,jj,ii)
                                             - poly(var,x,y,z)));
                    }
                }
                SD_Solution back("back",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
                restrict_face_overlap_sp(ptrs, n_sub, back, amr_RF, dim);
                Kokkos::fence();
                for(int var=0; var<nvar; var++)
                for(int k=NGHz;k<C.Nz-NGHz;k++) for(int j=NGHy;j<C.Ny-NGHy;j++)
                for(int i=NGHx;i<C.Nx-NGHx;i++)
                for(int kk=0;kk<C.nz;kk++) for(int jj=0;jj<C.ny;jj++)
                for(int ii=0;ii<C.nx;ii++)
                    dres = max(dres, abs(back.Vector(0,var,k,j,i,kk,jj,ii)
                                         - C.Vector(0,var,k,j,i,kk,jj,ii)));
            }
            failures += check("prolongate_face_coarser exact (degree-p)", dpro, 1e-12);
            failures += check("restrict_face(prolongate_face) identity", dres, 1e-11);
        }

        //Conservation. Accuracy and conservation are independent properties:
        //an operator can be exact on degree-p data and still lose mass on
        //general data. Every regrid and every coarse-fine interface applies
        //these, so a defect here leaks continuously rather than showing up as
        //a wrong answer at one point.
        {
            //quadrature weights of the solution points over one element:
            //int f = h * sum_i q_i f(x_sp_i), from the cell widths and sp->cv
            double q[16];
            for(int i=0;i<=p;i++){
                q[i]=0;
                for(int j=0;j<=p;j++) q[i] += (x_fp[j+1]-x_fp[j])*sp_to_cv(j,i);
            }
            double h = 1.0/N;
            //deliberately not a polynomial: conservation must not depend on
            //the data lying in the space the operators are exact for
            auto rough = [](int var, double x, double y, double z){
                return 1.0 + 0.3*sin(7.0*x) + 0.2*cos(5.0*y+1.0)
                           + 0.1*z*z*z + 0.05*var + 0.15*sin(11.0*x*y);
            };
            auto fill = [&](SD_Solution S, double ox, double oy, double oz, double sc){
                for(int var=0; var<nvar; var++)
                for(int k=0;k<S.Nz;k++) for(int j=0;j<S.Ny;j++) for(int i=0;i<S.Nx;i++)
                for(int kk=0;kk<S.nz;kk++) for(int jj=0;jj<S.ny;jj++)
                for(int ii=0;ii<S.nx;ii++)
                    S.Vector(0,var,k,j,i,kk,jj,ii) =
                        rough(var, ox+(i-NGHx+x_sp[ii])*sc,
                                   oy+(j-NGHy+x_sp[jj])*sc,
                                   oz+(k-NGHz+x_sp[kk])*sc);
            };
            auto integral = [&](SD_Solution S, double sc){
                double t=0;
                for(int var=0; var<nvar; var++)
                for(int k=NGHz;k<S.Nz-NGHz;k++) for(int j=NGHy;j<S.Ny-NGHy;j++)
                for(int i=NGHx;i<S.Nx-NGHx;i++)
                for(int kk=0;kk<S.nz;kk++) for(int jj=0;jj<S.ny;jj++)
                for(int ii=0;ii<S.nx;ii++)
                    t += q[ii]*q[jj]*q[kk]*S.Vector(0,var,k,j,i,kk,jj,ii);
                return t*sc*sc*sc;
            };

            SD_Solution C("Ccons",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
            SD_Solution F("Fcons",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
            SD_Solution C2("C2cons",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
            fill(C, 0,0,0, h);
            double Ic = integral(C, h), Ip = 0;
            for(int c=0;c<8;c++){
                prolongate_block(C,F,amr_P,c&1,(c>>1)&1,(c>>2)&1);
                Kokkos::fence();
                Ip += integral(F, 0.5*h);
            }
            failures += check("prolongate_block conserves integral",
                              abs(Ip-Ic)/abs(Ic), 1e-13);

            //restriction of arbitrary (non-polynomial) fine data
            Kokkos::deep_copy(C2.Vector, 0.0);
            double If = 0;
            for(int c=0;c<8;c++){
                int cx=c&1, cy=(c>>1)&1, cz=(c>>2)&1;
                fill(F, cx*0.5, cy*0.5, cz*0.5, 0.5*h);
                If += integral(F, 0.5*h);
                restrict_block(F,C2,amr_RF,cx,cy,cz);
                Kokkos::fence();
            }
            failures += check("restrict_block conserves integral",
                              abs(integral(C2,h)-If)/abs(If), 1e-13);

            //Face-flux restriction: what the coarse cell receives across an
            //interface must equal what its fine neighbours emitted, or the
            //interface leaks every step.
            auto face_int = [&](SD_Solution S, int dim, int de, int dp, double sc){
                double t=0;
                for(int var=0; var<nvar; var++)
                for(int k=NGHz;k<S.Nz-NGHz;k++) for(int j=NGHy;j<S.Ny-NGHy;j++)
                for(int i=NGHx;i<S.Nx-NGHx;i++)
                for(int kk=0;kk<S.nz;kk++) for(int jj=0;jj<S.ny;jj++)
                for(int ii=0;ii<S.nx;ii++){
                    int e  = (dim==_x_? i : (dim==_y_? j : k));
                    int pt = (dim==_x_? ii: (dim==_y_? jj: kk));
                    if(e!=de || pt!=dp) continue;
                    double w=1;
                    if(dim!=_x_) w*=q[ii];
                    if(dim!=_y_) w*=q[jj];
                    if(dim!=_z_) w*=q[kk];
                    t += w*S.Vector(0,var,k,j,i,kk,jj,ii);
                }
                return t*sc*sc;
            };
            double dflux=0;
            for(int dim=0; dim<3; dim++){
                int de = (dim==_x_? C.Nx-NGHx-1 : (dim==_y_? C.Ny-NGHy-1 : C.Nz-NGHz-1));
                int dp = (dim==_x_? C.nx-1 : (dim==_y_? C.ny-1 : C.nz-1));
                SD_Solution fs[4];
                const SD_Solution* pf[4];
                double emitted=0;
                for(int s=0;s<4;s++){
                    fs[s] = SD_Solution("ffl",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
                    fill(fs[s], 0.13*s, 0.29*s, 0.07*s, 0.5*h);
                    pf[s] = &fs[s];
                    emitted += face_int(fs[s], dim, de, dp, 0.5*h);
                }
                SD_Solution cf("cfl",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
                restrict_face_overlap_sp(pf, 4, cf, amr_RF, dim);
                Kokkos::fence();
                double received = face_int(cf, dim, de, dp, h);
                dflux = max(dflux, abs(received-emitted)/max(abs(emitted),1e-300));
            }
            failures += check("face flux restriction conserves", dflux, 1e-13);

            //Same property for the finite-volume face fluxes the fallback
            //writes. Fine fluxes vary sharply on purpose: weights that merely
            //sum to one still conserve a constant flux, so a smooth test
            //would pass even with the overlap weights wrong.
            SD_Solution shape("shape",1,nvar,Z_dim,Y_dim,X_dim,0,0,0);
            int cnx=shape.nx, cny=shape.ny, cnz=shape.nz;
            int Ncx=(shape.Nx-2*NGHx)*cnx;
            int Ncy=(shape.Ny-2*NGHy)*cny;
            int Ncz=(shape.Nz-2*NGHz)*cnz;
            GHOST_LOCALS;
            //FV arrays hold one value per cell, so they integrate against the
            //cell widths, not the solution-point quadrature weights in q.
            double dq[16];
            for(int i=0;i<cnx;i++) dq[i] = x_fp[i+1]-x_fp[i];
            auto cellw = [&](int c, int gh, int n, bool active){
                return active ? dq[(c-gh)%n] : 1.0;
            };
            auto fv_face_int = [&](FV_Solution S, int dim, int face, double sc){
                bool ax=cfg.active[_x_], ay=cfg.active[_y_], az=cfg.active[_z_];
                int x0 = dim==_x_? face : (ax? sghx : 0);
                int x1 = dim==_x_? face+1 : (ax? sghx+Ncx : 1);
                int y0 = dim==_y_? face : (ay? sghy : 0);
                int y1 = dim==_y_? face+1 : (ay? sghy+Ncy : 1);
                int z0 = dim==_z_? face : (az? sghz : 0);
                int z1 = dim==_z_? face+1 : (az? sghz+Ncz : 1);
                double t=0;
                for(int var=0; var<nvar; var++)
                for(int k=z0;k<z1;k++) for(int j=y0;j<y1;j++) for(int i=x0;i<x1;i++){
                    double w=1;
                    if(dim!=_x_) w *= cellw(i,sghx,cnx,ax)*sc;
                    if(dim!=_y_) w *= cellw(j,sghy,cny,ay)*sc;
                    if(dim!=_z_) w *= cellw(k,sghz,cnz,az)*sc;
                    t += w*S.Vector(var,k,j,i);
                }
                return t;
            };
            double dfv=0;
            for(int dim=0; dim<3; dim++){
                if(!cfg.active[dim]) continue;
                int n_sub = 1;
                for(int d=0; d<3; d++) if(d!=dim && cfg.active[d]) n_sub *= 2;
                int lo = (dim==_x_?sghx:dim==_y_?sghy:sghz);
                int hi = lo + (dim==_x_?Ncx:dim==_y_?Ncy:Ncz);
                FV_Solution CF("CFfv",nvar,Z_dim,Y_dim,X_dim,
                               dim==_z_,dim==_y_,dim==_x_);
                Kokkos::deep_copy(CF.Vector, 0.0);
                double emitted=0;
                for(int sub=0; sub<n_sub; sub++){
                    int cx=0, cy=0, cz=0, bit=0;
                    for(int d=0; d<3; d++){
                        if(d==dim || !cfg.active[d]) continue;
                        int v=(sub>>bit)&1;
                        if(d==_x_) cx=v; else if(d==_y_) cy=v; else cz=v;
                        bit++;
                    }
                    FV_Solution FF("FFfv",nvar,Z_dim,Y_dim,X_dim,
                                   dim==_z_,dim==_y_,dim==_x_);
                    int Fx=FF.Nx, Fy=FF.Ny, Fz=FF.Nz, fnv=nvar;
                    Kokkos::parallel_for("fill_fv",
                        Kokkos::MDRangePolicy<Kokkos::Rank<3>>({0,0,0},{Fz,Fy,Fx}),
                        KOKKOS_LAMBDA(int k,int j,int i){
                        for(int var=0; var<fnv; var++)
                            FF.Vector(var,k,j,i) =
                                sin(2.7*i+0.3*var) + 1.7*cos(1.9*j) + 0.6*k + 0.11*var;
                    });
                    Kokkos::fence();
                    emitted += fv_face_int(FF, dim, hi, 0.5);
                    restrict_face_fv_sub(CF, FF, dim, lo, hi, cx, cy, cz,
                                         Ncx, Ncy, Ncz, cnx, cny, cnz);
                    Kokkos::fence();
                }
                double received = fv_face_int(CF, dim, lo, 1.0);
                dfv = max(dfv, abs(received-emitted)/max(abs(emitted),1e-300));
            }
            failures += check("fv face flux restriction conserves", dfv, 1e-13);
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

        //BlockForest: uniform 2x2 neighbors are SAME; refine gives FINER/COARSER
        {
            double lim[3][2] = {{0,1},{0,1},{0,1}};
            int bc[3] = {_periodic_, _periodic_, _periodic_};
            BlockForest f = BlockForest::uniform_grid(2, true, true, false,
                4, 4, 1, 2, 2, 1, lim, bc);
            failures += check("forest uniform Nblocks", f.Nblocks()==4 ? 0.0 : 1.0, 0.5);
            bool all_same = true;
            for(int ib=0; ib<f.Nblocks(); ib++)
            for(int d=0; d<2; d++)
            for(int s=0; s<2; s++){
                const auto& e = f.blocks[ib].neighbors[d][s];
                if(e.size()!=1 || e[0].rel!=NEIGH_SAME) all_same = false;
            }
            failures += check("forest uniform SAME neighbors", all_same ? 0.0 : 1.0, 0.5);

            f.refine_blocks({0});
            f.rebuild_neighbors();
            //2d: one block becomes four, so 4 - 1 + 4 = 7
            failures += check("forest refine Nblocks", f.Nblocks()==7 ? 0.0 : 1.0, 0.5);
            bool has_cf = false, has_fc = false;
            for(int ib=0; ib<f.Nblocks(); ib++)
            for(int d=0; d<2; d++)
            for(int s=0; s<2; s++)
            for(const auto& e : f.blocks[ib].neighbors[d][s]){
                if(e.rel==NEIGH_FINER) has_fc = true;
                if(e.rel==NEIGH_COARSER) has_cf = true;
            }
            failures += check("forest refine FINER/COARSER", (has_fc&&has_cf)?0.0:1.0, 0.5);

            //Centre patch on a 4x4 base grid: a coarse rim must survive, else
            //the "refined" forest is just a uniform grid one level up.
            BlockForest f2 = BlockForest::uniform_grid(2, true, true, false,
                4, 4, 1, 4, 4, 1, lim, bc);
            RefinementRegion r; r.level=1;
            r.xmin=0.375; r.xmax=0.625; r.ymin=0.375; r.ymax=0.625;
            f2.refine_to_levels({r});
            f2.enforce_2to1_balance();
            int n_coarse=0, n_fine=0;
            for(int ib=0; ib<f2.Nblocks(); ib++)
                (f2.blocks[ib].level==0 ? n_coarse : n_fine)++;
            failures += check("forest centre patch stays mixed",
                              (n_coarse>0 && n_fine>0) ? 0.0 : 1.0, 0.5);
            //2:1 invariant: face neighbours differ by at most one level
            bool balanced = true;
            for(int ib=0; ib<f2.Nblocks(); ib++)
            for(int d=0; d<2; d++)
            for(int s=0; s<2; s++)
            for(const auto& e : f2.blocks[ib].neighbors[d][s]){
                if(e.jb < 0) continue;
                if(abs(f2.blocks[e.jb].level - f2.blocks[ib].level) > 1)
                    balanced = false;
            }
            failures += check("forest 2:1 balance invariant",
                              balanced ? 0.0 : 1.0, 0.5);
        }
    }
    amr_P = Matrix();
    amr_R = Matrix();
    amr_RS_sp[0] = amr_RS_sp[1] = Matrix();
    amr_RS_cv[0] = amr_RS_cv[1] = Matrix();
    amr_RF = Matrix();
    Kokkos::finalize();
    if(failures)
        cout<<failures<<" TEST(S) FAILED"<<endl;
    else
        cout<<"ALL TESTS PASSED"<<endl;
    return failures;
}
