#include "spd_k.hpp"

#define eta 0.0001

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
    int q = dim==0 ? px : (dim==1 ? py : pz);
    int nader = U_a.n_ader;
    int nvar = U_a.n_var;
    SD_for_all(
        double result=0;
        for(int ll=0; ll<q; ll++){
            if(dim==2)
                result += U_a.Vector(t_id,var,k,j,i,ll,jj,ii)*a_to_b(kk,ll);
            else if(dim==1)
                result += U_a.Vector(t_id,var,k,j,i,kk,ll,ii)*a_to_b(jj,ll);
            else if(dim==0)
                result += U_a.Vector(t_id,var,k,j,i,kk,jj,ll)*a_to_b(ii,ll);
        }
        U_b.Vector(t_id,var,k,j,i,kk,jj,ii) = result;
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
    int qx = F_x.nx;
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
            for(ll=0; ll<qx; ll++){
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
    int qx = F_x.nx;
    int nader = F_x.n_ader;
    int nvar = U.n_var;
    SD_for_variables(
        double dudt;
        double du;
        du=0;
        for(int t_id =0; t_id<nader; t_id++){
            dudt = 0;
            for(int ll=0; ll<qx; ll++){
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
    int dim){

    int Nx = F_fp.Nx;
    int Ny = F_fp.Ny;
    int Nz = F_fp.Nz;
    int px = F_fp.nx-(dim==0);
    int py = F_fp.ny-(dim==1);
    int pz = F_fp.nz-(dim==2);
    int q1 = dim==0 ? py : (dim==1 ? px : px);
    int q2 = dim==0 ? pz : (dim==1 ? pz : py);
    int nader = F_fp.n_ader;
    int nvar = F_fp.n_var;
    SD_for_all(
        double s=0;
        double f=0;
        for(int nn=0; nn<q2; nn++){
            for(int ll=0; ll<q1; ll++){
                #if Z
                if(dim==2){
                    s = F_fp.Vector(t_id,var,k,j,i,kk,nn,ll);
                    #if X
                    s *= sp_to_cv(ii,ll);
                    #endif
                    #if Y
                    s *= sp_to_cv(jj,nn);
                    #endif
                    f+=s;
                }
                #endif
                #if Y
                else if(dim==1){
                    s = F_fp.Vector(t_id,var,k,j,i,nn,jj,ll);
                    #if X
                    s *= sp_to_cv(ii,ll);
                    #endif
                    #if Z
                    s *= sp_to_cv(kk,nn);
                    #endif
                    f+=s;
                }
                #endif
                #if X
                else if(dim==0){
                    s = F_fp.Vector(t_id,var,k,j,i,nn,ll,ii);
                    #if Y
                    s *= sp_to_cv(jj,ll);
                    #endif
                    #if Z
                    s *= sp_to_cv(kk,nn);
                    #endif
                    f+=s;
                }
                #endif
            }
        }
        F.Vector(t_id,var,(k+1)*kk,(j+1)*jj,(i+1)*ii) = f;
    );
}