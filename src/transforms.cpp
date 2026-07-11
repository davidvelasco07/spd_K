#include "spd_k.hpp"

KOKKOS_INLINE_FUNCTION
int choose(int dim, int i, int j, int k){
    return (dim==_x_ ? i : (dim==_y_ ?  j : k));
}

KOKKOS_INLINE_FUNCTION
void indices(int* N_id, int* n_id, int k, int j, int i, int kk, int jj, int ii, int l, int ll, int dim){
    //Returns the indeces according to the dimension
    N_id[_x_] = dim == _x_ ? l  : i;
    n_id[_x_] = dim == _x_ ? ll : ii;
    N_id[_y_] = dim == _y_ ? l  : j;
    n_id[_y_] = dim == _y_ ? ll : jj;
    N_id[_z_] = dim == _z_ ? l  : k;
    n_id[_z_] = dim == _z_ ? ll : kk;
}

KOKKOS_INLINE_FUNCTION
int indices_n(int* n_id, int k, int j, int i, int l, int dim){
    //Returns the indeces according to the dimension
    n_id[_x_] = dim == _x_ ? l  : i;
    n_id[_y_] = dim == _y_ ? l  : j;
    n_id[_z_] = dim == _z_ ? l  : k;
    return choose(dim,i,j,k);
}

//Reference implementation of the full tensor-product transform.
//Cost per point is (p+1)^DIM; kept only to validate the sweep version.
void transform_a_to_b_ref(
    SD_Solution U_a,
    SD_Solution U_b,
    Matrix a_to_b_z,
    Matrix a_to_b_y,
    Matrix a_to_b_x
    ){
    int Nx = U_b.Nx;
    int Ny = U_b.Ny;
    int Nz = U_b.Nz;
    int px = U_b.nx;
    int py = U_b.ny;
    int pz = U_b.nz;
    int qx = U_a.nx;
    int qy = U_a.ny;
    int qz = U_a.nz;
    int nader = U_a.n_ader;
    int nvar = U_a.n_var;
    bool ax = cfg.active[_x_];
    bool ay = cfg.active[_y_];
    bool az = cfg.active[_z_];
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        for(int t_id=0; t_id<nader; t_id++){
        for(int var=0; var<nvar; var++){
        double result=0;
        double s;
        for(int nn=0; nn<qz; nn++){
            for(int mm=0; mm<qy; mm++){
                for(int ll=0; ll<qx; ll++){
                    s = U_a.Vector(t_id,var,k,j,i,nn,mm,ll);
                    if(ax) s *= a_to_b_x(ii,ll);
                    if(ay) s *= a_to_b_y(jj,mm);
                    if(az) s *= a_to_b_z(kk,nn);
                    result += s;
                }
            }
        }
        U_b.Vector(t_id,var,k,j,i,kk,jj,ii) = result;
        }}
    });
}

//Full tensor-product transform factorized into one 1d sweep per active
//dimension: cost per point is DIM*(p+1) instead of (p+1)^DIM.
//The same matrix is applied along every direction (cv<->sp use square
//matrices, so all point counts of U_a, U_b and the scratch T must match).
//Sweeps are separate kernels; they run in order on the same execution
//space instance, so no synchronization beyond the kernel boundary is needed.
void transform_a_to_b(
    SD_Solution U_a,
    SD_Solution U_b,
    SD_Solution T,
    Matrix a_to_b
    ){
    int dims[3];
    int nd=0;
    if(cfg.active[_x_]) dims[nd++]=_x_;
    if(cfg.active[_y_]) dims[nd++]=_y_;
    if(cfg.active[_z_]) dims[nd++]=_z_;
    //Ping-pong between U_b and T so that the last sweep writes U_b
    SD_Solution src = U_a;
    for(int s=0; s<nd; s++){
        SD_Solution dst = ((nd-1-s)%2==0) ? U_b : T;
        transform_a_to_b_1d(src, dst, a_to_b, dims[s]);
        src = dst;
    }
}

//Stage-2 GPU version of the full tensor-product transform: a single kernel
//with one team per (element, var, t_id). The element's point values are
//staged in team scratch memory, then swept once per active dimension with a
//team barrier between sweeps, and written back once. Compared with the
//kernel-per-sweep version this avoids two full round trips of the array
//through global memory plus the extra kernel launches. Arithmetic is the
//same sequence of 1d contractions, so results are bit-identical.
//On host backends TeamPolicy resolves to team_size 1 and the barriers are
//no-ops, so the same code runs everywhere.
//Requires U_a and U_b to have identical point counts (square matrices).
void transform_a_to_b_team(
    SD_Solution U_a,
    SD_Solution U_b,
    Matrix a_to_b
    ){
    int Nx = U_b.Nx;
    int Ny = U_b.Ny;
    int Nz = U_b.Nz;
    int nx = U_b.nx;
    int ny = U_b.ny;
    int nz = U_b.nz;
    int n_pts = nx*ny*nz;
    int nvar = U_a.n_var;
    int nader = U_a.n_ader;
    int dims[3];
    int nd=0;
    if(cfg.active[_x_]) dims[nd++]=_x_;
    if(cfg.active[_y_]) dims[nd++]=_y_;
    if(cfg.active[_z_]) dims[nd++]=_z_;
    int d0 = nd>0 ? dims[0] : 0;
    int d1 = nd>1 ? dims[1] : 0;
    int d2 = nd>2 ? dims[2] : 0;
    int league = Nz*Ny*Nx*nvar*nader;
    using team_policy = Kokkos::TeamPolicy<>;
    using member_t = team_policy::member_type;
    size_t scratch_bytes = 2*size_t(n_pts)*sizeof(double);
    Kokkos::parallel_for("transform_a_to_b_team",
        team_policy(league, Kokkos::AUTO)
            .set_scratch_size(0, Kokkos::PerTeam(scratch_bytes)),
        KOKKOS_LAMBDA(const member_t& team){
            int r = team.league_rank();
            int t_id = r % nader; r /= nader;
            int var  = r % nvar;  r /= nvar;
            int i    = r % Nx;    r /= Nx;
            int j    = r % Ny;
            int k    = r / Ny;
            double* buf0 = (double*)team.team_scratch(0).get_shmem(n_pts*sizeof(double));
            double* buf1 = (double*)team.team_scratch(0).get_shmem(n_pts*sizeof(double));
            //Stage the element in scratch
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team,n_pts),[&](int idx){
                int ii = idx % nx;
                int jj = (idx/nx) % ny;
                int kk = idx/(nx*ny);
                buf0[idx] = U_a.Vector(t_id,var,k,j,i,kk,jj,ii);
            });
            team.team_barrier();
            double* in = buf0;
            double* out = buf1;
            for(int s=0; s<nd; s++){
                int dim = (s==0 ? d0 : (s==1 ? d1 : d2));
                Kokkos::parallel_for(Kokkos::TeamThreadRange(team,n_pts),[&](int idx){
                    int ii = idx % nx;
                    int jj = (idx/nx) % ny;
                    int kk = idx/(nx*ny);
                    int id = (dim==_x_ ? ii : (dim==_y_ ? jj : kk));
                    int q  = (dim==_x_ ? nx : (dim==_y_ ? ny : nz));
                    double u=0;
                    for(int ll=0; ll<q; ll++){
                        int i2 = (dim==_x_ ? ll : ii);
                        int j2 = (dim==_y_ ? ll : jj);
                        int k2 = (dim==_z_ ? ll : kk);
                        u += in[i2 + nx*(j2 + ny*k2)]*a_to_b(id,ll);
                    }
                    out[idx] = u;
                });
                team.team_barrier();
                double* tmp = in; in = out; out = tmp;
            }
            //Single write back to global memory
            Kokkos::parallel_for(Kokkos::TeamThreadRange(team,n_pts),[&](int idx){
                int ii = idx % nx;
                int jj = (idx/nx) % ny;
                int kk = idx/(nx*ny);
                U_b.Vector(t_id,var,k,j,i,kk,jj,ii) = in[idx];
            });
        });
}

void transform_a_to_b_1d(
    SD_Solution U_a,
    SD_Solution U_b, 
    Matrix a_to_b, 
    int dim){
    int Nx = U_b.Nx;
    int Ny = U_b.Ny;
    int Nz = U_b.Nz;
    int px = U_b.nx;
    int py = U_b.ny;
    int pz = U_b.nz;
    int q = choose(dim, U_a.nx, U_a.ny, U_a.nz);
    int nader = U_a.n_ader;
    int nvar = U_a.n_var;
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        for(int t_id=0; t_id<nader; t_id++){
        for(int var=0; var<nvar; var++){
        int nid[3];
        double u=0;
        int id = choose(dim,ii,jj,kk);
        for(int ll=0; ll<q; ll++){
            indices_n(nid,kk,jj,ii,ll,dim);
            u += U_a.Vector(t_id,var,k,j,i,NODE)*a_to_b(id,ll);
        }
        U_b.Vector(t_id,var,k,j,i,kk,jj,ii) = u;
        }}
    });
}

//Single-direction sweep of one time slice: reads slice t_src of U_a and
//writes slice 0 of U_b. Used by transforms that operate per ADER slice.
void transform_a_to_b_1d_slice(
    SD_Solution U_a,
    SD_Solution U_b,
    Matrix a_to_b,
    int dim,
    int t_src){
    int Nx = U_b.Nx;
    int Ny = U_b.Ny;
    int Nz = U_b.Nz;
    int px = U_b.nx;
    int py = U_b.ny;
    int pz = U_b.nz;
    int q = choose(dim, U_a.nx, U_a.ny, U_a.nz);
    int nvar = U_a.n_var;
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        for(int var=0; var<nvar; var++){
        int nid[3];
        double u=0;
        int id = choose(dim,ii,jj,kk);
        for(int ll=0; ll<q; ll++){
            indices_n(nid,kk,jj,ii,ll,dim);
            u += U_a.Vector(t_src,var,k,j,i,NODE)*a_to_b(id,ll);
        }
        U_b.Vector(0,var,k,j,i,kk,jj,ii) = u;
        }
    });
}

//Reference implementation of the transverse (2d) transform.
//Cost per point is (p+1)^2; kept only to validate the sweep version.
void transform_a_to_b_2d_ref(
    SD_Solution U_a,
    SD_Solution U_b,
    Matrix a_to_b,
    int dim){
    int Nx = U_b.Nx;
    int Ny = U_b.Ny;
    int Nz = U_b.Nz;
    int px = U_b.nx;
    int py = U_b.ny;
    int pz = U_b.nz;
    int nader = U_a.n_ader;
    int nvar = U_a.n_var;
    int dim1 = choose(dim , _y_, _z_, _x_);
    int dim2 = choose(dim1, _y_, _z_, _x_);
    int q1 = choose(dim1, px, py, pz);
    int q2 = choose(dim2, px, py, pz);
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        for(int t_id=0; t_id<nader; t_id++){
        for(int var=0; var<nvar; var++){
        double s=0;
        double u=0;
        int nid[3];
        int id1 = choose(dim1,ii,jj,kk);
        int id2 = choose(dim2,ii,jj,kk);
        for(int nn=0; nn<q2; nn++){
            indices_n(nid,kk,jj,ii,nn,dim2);
            for(int ll=0; ll<q1; ll++){
                indices_n(nid,NODE,ll,dim1);
                s = U_a.Vector(t_id,var,k,j,i,NODE);
                s *= a_to_b(id1,ll);
                s *= a_to_b(id2,nn);
                u += s;
            }
        }
        U_b.Vector(t_id,var,k,j,i,kk,jj,ii) = u;
        }}
    });
}

//Transverse transform factorized into one sweep per active transverse
//dimension (2*(p+1) per point instead of (p+1)^2). Inactive dimensions are
//skipped, which is the mathematically consistent behavior (the reference
//version applies a spurious matrix factor along inactive dimensions in
//non-3D builds). U_a, U_b and the scratch T must have identical shapes.
void transform_a_to_b_2d(
    SD_Solution U_a,
    SD_Solution U_b,
    SD_Solution T,
    Matrix a_to_b,
    int dim){
    int dim1 = choose(dim , _y_, _z_, _x_);
    int dim2 = choose(dim1, _y_, _z_, _x_);
    int dims[2];
    int nd=0;
    if(cfg.active[dim1]) dims[nd++]=dim1;
    if(cfg.active[dim2]) dims[nd++]=dim2;
    if(nd==0){
        Kokkos::deep_copy(U_b.Vector, U_a.Vector);
        return;
    }
    //Ping-pong between U_b and T so that the last sweep writes U_b
    SD_Solution src = U_a;
    for(int s=0; s<nd; s++){
        SD_Solution dst = ((nd-1-s)%2==0) ? U_b : T;
        transform_a_to_b_1d(src, dst, a_to_b, dims[s]);
        src = dst;
    }
}

void update_prediction(
    SD_Solution U,
    SD_Solution U_ader,
    SD_Solution F_x,
    SD_Solution F_y,
    SD_Solution F_z,
    Matrix dfp_to_sp,
    Matrix invader,
    Vector w,
    double dx,
    double dy,
    double dz,
    double dt){

    int Nx = U.Nx;
    int Ny = U.Ny;
    int Nz = U.Nz;
    int px = U.nx;
    int py = U.ny;
    int pz = U.nz;
    int q  = F_x.nx;
    int nader = F_x.n_ader;
    int nvar = U.n_var;
    bool ax = cfg.active[_x_];
    bool ay = cfg.active[_y_];
    bool az = cfg.active[_z_];
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        for(int var=0; var<nvar; var++){
        double dudt[10];
        int t_id;
        int ll;
        double u_old;
        double du;
        u_old =  U.Vector(0,var,k,j,i,kk,jj,ii);
        for(t_id =0; t_id<nader; t_id++){
            dudt[t_id] = 0;
            for(ll=0; ll<q; ll++){
                if(ax) dudt[t_id] += F_x.Vector(t_id,var,k,j,i,kk,jj,ll)*dfp_to_sp(ii,ll)/dx;
                if(ay) dudt[t_id] += F_y.Vector(t_id,var,k,j,i,kk,ll,ii)*dfp_to_sp(jj,ll)/dy;
                if(az) dudt[t_id] += F_z.Vector(t_id,var,k,j,i,ll,jj,ii)*dfp_to_sp(kk,ll)/dz;
            }
        }
        for(t_id=0; t_id<nader; t_id++){
            du=0;
            for(ll=0; ll<nader; ll++){
                du += dudt[ll]*invader(t_id,ll)*w(ll)*dt;
            }
            U_ader.Vector(t_id,var,k,j,i,kk,jj,ii) = u_old - du;
        }
        }
    });
}

void update_solution(
    SD_Solution U,
    SD_Solution F_x,
    SD_Solution F_y,
    SD_Solution F_z,
    Matrix da_to_b,
    Vector w,
    double dx,
    double dy,
    double dz,
    double dt){

    int Nx = U.Nx;
    int Ny = U.Ny;
    int Nz = U.Nz;
    int px = U.nx;
    int py = U.ny;
    int pz = U.nz;
    int q  = F_x.nx;
    int nader = F_x.n_ader;
    int nvar = U.n_var;
    bool ax = cfg.active[_x_];
    bool ay = cfg.active[_y_];
    bool az = cfg.active[_z_];
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        for(int var=0; var<nvar; var++){
        double dudt;
        double du;
        du=0;
        for(int t_id =0; t_id<nader; t_id++){
            dudt = 0;
            for(int ll=0; ll<q; ll++){
                if(ax) dudt += F_x.Vector(t_id,var,k,j,i,kk,jj,ll)*da_to_b(ii,ll)/dx;
                if(ay) dudt += F_y.Vector(t_id,var,k,j,i,kk,ll,ii)*da_to_b(jj,ll)/dy;
                if(az) dudt += F_z.Vector(t_id,var,k,j,i,ll,jj,ii)*da_to_b(kk,ll)/dz;
            }
            du += dudt*w(t_id)*dt;
        }
        U.Vector(0,var,k,j,i,kk,jj,ii) -= du;
        }
    });
}

//Reference implementation of the transverse face integration.
//Cost per point is (p+1)^2 in 3D; kept only to validate the sweep version.
void face_integral_ref(
    SD_Solution F_fp,
    FV_Solution F,
    Matrix sp_to_cv,
    int t_id,
    int dim){
    int Nx = F_fp.Nx+(dim==0);
    int Ny = F_fp.Ny+(dim==1);
    int Nz = F_fp.Nz+(dim==2);
    int px = F_fp.nx-(dim==0);
    int py = F_fp.ny-(dim==1);
    int pz = F_fp.nz-(dim==2);
    //Needed to compute K,J,I
    int qx = px;
    int qy = py;
    int qz = pz;
    int dim1 = choose(dim , _y_, _z_, _x_);
    int dim2 = choose(dim1, _y_, _z_, _x_);
    int q1   = choose(dim1 , px, py, pz);
    int q2   = choose(dim2 , px, py, pz);
    int nvar  = F_fp.n_var;
    int Ni = F.Nx;
    int Nj = F.Ny;
    int Nk = F.Nz;
    bool a1 = cfg.active[dim1];
    bool a2 = cfg.active[dim2];
    GHOST_LOCALS;
    sd_for_active_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        double s=0;
        double f;
        int nid[3];
        int id1 = choose(dim1, ii, jj, kk);
        int id2 = choose(dim2, ii, jj, kk);
        for(int var=0; var<nvar; var++){
            f=0;
            for(int nn=0; nn<q2; nn++){
                indices_n(nid,kk,jj,ii,nn,dim2);
                for(int ll=0; ll<q1; ll++){
                    indices_n(nid,NODE,ll,dim1);
                    s = F_fp.Vector(t_id,var,k,j,i,NODE);
                    if(a1) s *= sp_to_cv(id1,ll);
                    if(a2) s *= sp_to_cv(id2,nn);
                    f += s;
                }
            }
            if(K < Nk && J < Nj && I < Ni)
                F.Vector(var,K,J,I) = f;
        }
    });
}

//Face integration factorized into directional sweeps. In 3D the two
//transverse contractions become: (1) a 1d sweep of the t_id slice along
//dim1 into slice 0 of the scratch T (same shape as F_fp, n_ader=1), and
//(2) a contraction along dim2 that scatters to the FV face array. In 1D/2D
//there is at most one transverse contraction, so the reference kernel is
//already a single sweep and is used directly.
void face_integral(
    SD_Solution F_fp,
    FV_Solution F,
    SD_Solution T,
    Matrix sp_to_cv,
    int t_id,
    int dim){
    int dim1 = choose(dim , _y_, _z_, _x_);
    int dim2 = choose(dim1, _y_, _z_, _x_);
    //With fewer than two active transverse dimensions there is at most one
    //contraction, so the reference kernel is already a single sweep
    if(!cfg.active[dim1] || !cfg.active[dim2]){
        face_integral_ref(F_fp, F, sp_to_cv, t_id, dim);
        return;
    }
    transform_a_to_b_1d_slice(F_fp, T, sp_to_cv, dim1, t_id);
    int Nx = F_fp.Nx+(dim==0);
    int Ny = F_fp.Ny+(dim==1);
    int Nz = F_fp.Nz+(dim==2);
    int px = F_fp.nx-(dim==0);
    int py = F_fp.ny-(dim==1);
    int pz = F_fp.nz-(dim==2);
    //Needed to compute K,J,I
    int qx = px;
    int qy = py;
    int qz = pz;
    int q2   = choose(dim2 , px, py, pz);
    int nvar  = F_fp.n_var;
    int Ni = F.Nx;
    int Nj = F.Ny;
    int Nk = F.Nz;
    GHOST_LOCALS;
    sd_for_active_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        double f;
        int nid[3];
        int id2 = choose(dim2, ii, jj, kk);
        for(int var=0; var<nvar; var++){
            f=0;
            for(int nn=0; nn<q2; nn++){
                indices_n(nid,kk,jj,ii,nn,dim2);
                f += T.Vector(0,var,k,j,i,NODE)*sp_to_cv(id2,nn);
            }
            if(K < Nk && J < Nj && I < Ni)
                F.Vector(var,K,J,I) = f;
        }
    });
}

void fv_update_solution(
    FV_Solution U_new,
    FV_Solution U_old,
    SD_Solution U_cv,
    FV_Solution F_x,
    Vector faces_x,
    FV_Solution F_y,
    Vector faces_y,
    FV_Solution F_z,
    Vector faces_z,
    Vector w,
    int t_id,
    double dt,
    bool update
){
    int Nx = U_cv.Nx;
    int Ny = U_cv.Ny;
    int Nz = U_cv.Nz;
    int px = U_cv.nx;
    int py = U_cv.ny;
    int pz = U_cv.nz;
    //Needed to compute K,J,I
    int qx = px;
    int qy = py;
    int qz = pz;
    int nvar = U_cv.n_var;
    bool ax = cfg.active[_x_];
    bool ay = cfg.active[_y_];
    bool az = cfg.active[_z_];
    GHOST_LOCALS;
    sd_for_active_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        for(int var=0; var<nvar; var++){
        double h;
        double F;
        double u_new;
        F=0;
        U_old.Vector(var,K,J,I) = U_cv.Vector(0,var,k,j,i,kk,jj,ii);
        if(ax){
            h = faces_x(I+1)-faces_x(I);
            F += (F_x.Vector(var,K,J,I+1)-F_x.Vector(var,K,J,I))/h;
        }
        if(ay){
            h = faces_y(J+1)-faces_y(J);
            F += (F_y.Vector(var,K,J+1,I)-F_y.Vector(var,K,J,I))/h;
        }
        if(az){
            h = faces_z(K+1)-faces_z(K);
            F += (F_z.Vector(var,K+1,J,I)-F_z.Vector(var,K,J,I))/h;
        }
        u_new = U_old.Vector(var,K,J,I) - w(t_id)*dt*F;
        U_new.Vector(var,K,J,I) = u_new;
        if(update)
            U_cv.Vector(0,var,k,j,i,kk,jj,ii) = u_new;
        }
    });
}

void FV_to_SD(
    FV_Solution U_fv,
    SD_Solution U_sd){
    int Nx = U_sd.Nx;
    int Ny = U_sd.Ny;
    int Nz = U_sd.Nz;
    int px = U_sd.nx;
    int py = U_sd.ny;
    int pz = U_sd.nz;
    int qx = px;
    int qy = py;
    int qz = pz;
    int nvar = U_sd.n_var;
    GHOST_LOCALS;
    sd_for_active_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        for(int var=0; var<nvar; var++)
        U_sd.Vector(0,var,k,j,i,kk,jj,ii) = U_fv.Vector(var,K,J,I);
    });
}