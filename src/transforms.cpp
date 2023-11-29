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

void transform_a_to_b(
    SD_Solution U_a,
    SD_Solution U_b,
    #if Z
    Matrix a_to_b_z,
    #endif
    #if Y
    Matrix a_to_b_y,
    #endif
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
    SD_for_all(
        double result=0;
        double s;
        for(int nn=0; nn<qz; nn++){
            for(int mm=0; mm<qy; mm++){
                for(int ll=0; ll<qx; ll++){
                    s = U_a.Vector(t_id,var,k,j,i,nn,mm,ll);
                    #if X==1
                    s *= a_to_b_x(ii,ll);
                    #endif
                    #if Y==1
                    s *= a_to_b_y(jj,mm);
                    #endif
                    #if Z==1
                    s *= a_to_b_z(kk,nn);
                    #endif
                    result += s;
                }
            }
        }
        U_b.Vector(t_id,var,k,j,i,kk,jj,ii) = result;
    );
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
    SD_for_all(
        int nid[3];
        double u=0;
        int id = choose(dim,ii,jj,kk);
        for(int ll=0; ll<q; ll++){
            indices_n(nid,kk,jj,ii,ll,dim);
            u += U_a.Vector(t_id,var,k,j,i,NODE)*a_to_b(id,ll);
        }
        U_b.Vector(t_id,var,k,j,i,kk,jj,ii) = u;
    );
}

void transform_a_to_b_2d(
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
    SD_for_all(
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
    );
}

void update_prediction(
    SD_Solution U,
    SD_Solution U_ader,
    #if X
    SD_Solution F_x,
    #endif
    #if Y
    SD_Solution F_y,
    #endif
    #if Z
    SD_Solution F_z,
    #endif
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
    SD_for_variables(
        double dudt[10];
        int t_id;
        int ll;
        double u_old;
        double du;
        u_old =  U.Vector(0,var,k,j,i,kk,jj,ii);
        for(t_id =0; t_id<nader; t_id++){
            dudt[t_id] = 0;
            for(ll=0; ll<q; ll++){
                #if X
                dudt[t_id] += F_x.Vector(t_id,var,k,j,i,kk,jj,ll)*dfp_to_sp(ii,ll)/dx;
                #endif
                #if Y
                dudt[t_id] += F_y.Vector(t_id,var,k,j,i,kk,ll,ii)*dfp_to_sp(jj,ll)/dy;
                #endif
                #if Z
                dudt[t_id] += F_z.Vector(t_id,var,k,j,i,ll,jj,ii)*dfp_to_sp(kk,ll)/dz;
                #endif
            }
        }
        for(t_id=0; t_id<nader; t_id++){
            du=0;
            for(ll=0; ll<nader; ll++){
                du += dudt[ll]*invader(t_id,ll)*w(ll)*dt;
            }
            U_ader.Vector(t_id,var,k,j,i,kk,jj,ii) = u_old - du;
        }
    );
}

void update_solution(
    SD_Solution U,
    #if X
    SD_Solution F_x,
    #endif
    #if Y
    SD_Solution F_y,
    #endif
    #if Z
    SD_Solution F_z,
    #endif
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
    SD_for_variables(
        double dudt;
        double du;
        du=0;
        for(int t_id =0; t_id<nader; t_id++){
            dudt = 0;
            for(int ll=0; ll<q; ll++){
                #if X
                dudt += F_x.Vector(t_id,var,k,j,i,kk,jj,ll)*da_to_b(ii,ll)/dx;
                #endif
                #if Y
                dudt += F_y.Vector(t_id,var,k,j,i,kk,ll,ii)*da_to_b(jj,ll)/dy;
                #endif
                #if Z
                dudt += F_z.Vector(t_id,var,k,j,i,ll,jj,ii)*da_to_b(kk,ll)/dz;
                #endif
            }
            du += dudt*w(t_id)*dt;
        }    
        U.Vector(0,var,k,j,i,kk,jj,ii) -= du;    
    );
}

void face_integral(
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
    //cout<<dim<<dim1<<dim2<<" "<<px<<" "<<py<<" "<<pz<<": "<<q1<<" "<<q2<<endl;
    SD_for_active_cells(
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
                    #ifdef _2D_
                    s *= sp_to_cv(id1,ll);
                    #endif
                    #ifdef _3D_
                    s *= sp_to_cv(id2,nn);
                    #endif
                    f += s;
                }
            }
            if(K < Nk && J < Nj && I < Ni)
                F.Vector(var,K,J,I) = f;
        }
    );
}

void fv_update_solution(
    FV_Solution U_new,
    FV_Solution U_old,
    SD_Solution U_cv,
    #if X
    FV_Solution F_x,
    Vector faces_x,
    #endif
    #if Y
    FV_Solution F_y,
    Vector faces_y,
    #endif
    #if Z
    FV_Solution F_z,
    Vector faces_z,
    #endif
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
    SD_for_active_cells(
        for(int var=0; var<nvar; var++){
        double h;
        double F;
        double u_new;
        F=0;
        U_old.Vector(var,K,J,I) = U_cv.Vector(0,var,k,j,i,kk,jj,ii);
        #if X
        h = faces_x(I+1)-faces_x(I);
        F += (F_x.Vector(var,K,J,I+1)-F_x.Vector(var,K,J,I))/h;
        #endif
        #if Y
        h = faces_y(J+1)-faces_y(J);
        F += (F_y.Vector(var,K,J+1,I)-F_y.Vector(var,K,J,I))/h;
        #endif
        #if Z
        h = faces_z(K+1)-faces_z(K);
        F += (F_z.Vector(var,K+1,J,I)-F_z.Vector(var,K,J,I))/h;
        #endif
        u_new = U_old.Vector(var,K,J,I) - w(t_id)*dt*F;
        U_new.Vector(var,K,J,I) = u_new;
        if(update)
            U_cv.Vector(0,var,k,j,i,kk,jj,ii) = u_new;
        }
    );
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
    SD_for_active_cells(
        for(int var=0; var<nvar; var++)
        U_sd.Vector(0,var,k,j,i,kk,jj,ii) = U_fv.Vector(var,K,J,I);
    );
}