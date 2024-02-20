#include "spd_k.hpp"

KOKKOS_INLINE_FUNCTION
int choose(int dim, int i, int j, int k){
    return (dim==_x_ ? i : (dim==_y_ ?  j : k));
}

KOKKOS_INLINE_FUNCTION
void indices(int* N_id, int* n_id, int k, int j, int i, int kk, int jj, int ii, int l, int ll, int dim){
    //Returns the indeces according to the dimension
    N_id[_x_] = (dim == _x_ ? l  : i );
    n_id[_x_] = (dim == _x_ ? ll : ii);
    N_id[_y_] = (dim == _y_ ? l  : j );
    n_id[_y_] = (dim == _y_ ? ll : jj);
    N_id[_z_] = (dim == _z_ ? l  : k );
    n_id[_z_] = (dim == _z_ ? ll : kk);
}

KOKKOS_INLINE_FUNCTION
int indices_n(int* n_id, int k, int j, int i, int l, int dim){
    //Returns the indeces according to the dimension
    n_id[_x_] = dim == _x_ ? l  : i;
    n_id[_y_] = dim == _y_ ? l  : j;
    n_id[_z_] = dim == _z_ ? l  : k;
    return choose(dim,i,j,k);
}

KOKKOS_INLINE_FUNCTION
double velocity(int var, double x, double y, double z){
    if(var<2)return 1;
    else return 0;
    double L = LENGHT;
    double r1=1.;
    x = x-0.5*L;
    y = y-0.5*L;
    double r; 
    r = sqrt(x*x+y*y);
    double theta;
    theta = atan2(x,y);
    double omega;
    omega = r<r1 ? 0.609711 : 0.0;
    if(var==2)
        return r<r1 ? 0.792624 : 0.0;
    else if(var==1)
        return -sin(theta)*r*omega;
    else if(var==0)
        return cos(theta)*r*omega;
    else
        return 0;
}

double Induction_compute_dt(
    SD_Solution B2,
    double dx,
    double dy,
    double dz,
    Matrix x,
    Matrix y,
    Matrix z
    ){
    int Nx = B2.Nx;
    int Ny = B2.Ny;
    int Nz = B2.Nz;
    int px = B2.nx;
    int py = B2.ny;
    int pz = B2.nz;
    SD_min_cells(
        double v_max=0;
        double dx_min=1;
        double dt_min=1;
        #if X
        v_max += abs(velocity(_x_,x(i,ii),y(j,jj),z(k,kk)));
        dx_min = min(dx_min,dx);
        #endif
        #if Y
        v_max += abs(velocity(_y_,x(i,ii),y(j,jj),z(k,kk)));
        dx_min = min(dx_min,dy);
        #endif
        #if Z
        v_max += abs(velocity(_z_,x(i,ii),y(j,jj),z(k,kk)));
        dx_min = min(dx_min,dz);
        #endif
        dt_min = CFL*dx_min/v_max/px;
        reduce = reduce < dt_min ? reduce : dt_min; 
        );
    #ifdef MPI
    return MPI_Allreduce(&min_value, &dt, 1, MPI_DOUBLE, MPI_MIN, Comm);
    #else
    return min_value;
    #endif
}

void rotational_a_to_b(
    SD_Solution A1, 
    SD_Solution A2, 
    SD_Solution B, 
    Matrix da_to_b, 
    double d1, 
    double d2, 
    int dim){
    int Nx = B.Nx;
    int Ny = B.Ny;
    int Nz = B.Nz;
    int px = B.nx;
    int py = B.ny;
    int pz = B.nz;
    int q = choose(dim, px, py, pz);
    int nader = B.n_ader;
    SD_for_ader(
        double d1a2=0;
        double d2a1=0;
        for(int ll=0; ll<q; ll++){
            if(dim==0){
                //Bx = dyAz - dzAy
                d1a2 += A2.Vector(t_id,0,k,j,i,kk,ll,ii)*da_to_b(jj,ll);
                d2a1 += A1.Vector(t_id,0,k,j,i,ll,jj,ii)*da_to_b(kk,ll);
            }
            else if(dim==1){
                //By = dzAx - dxAz
                d1a2 += A2.Vector(t_id,0,k,j,i,ll,jj,ii)*da_to_b(kk,ll);
                d2a1 += A1.Vector(t_id,0,k,j,i,kk,jj,ll)*da_to_b(ii,ll);
            }
            else if(dim==2){
                //Bz = dxAy - dyAx
                d1a2 += A2.Vector(t_id,0,k,j,i,kk,jj,ll)*da_to_b(ii,ll);
                d2a1 += A1.Vector(t_id,0,k,j,i,kk,ll,ii)*da_to_b(jj,ll);
            }
            else{
                ///////Error
            }
        }
        B.Vector(t_id,0,k,j,i,kk,jj,ii) = d1a2/d1 - d2a1/d2;
    );
}

void compute_E(
    SD_Solution E,
    SD_Solution V,
    SD_Solution B1,
    SD_Solution B2,
    Matrix sp_to_fp,
    Matrix x,
    Matrix y,
    Matrix z,
    int dim){

    int Nx = E.Nx;
    int Ny = E.Ny;
    int Nz = E.Nz;
    int px = E.nx;
    int py = E.ny;
    int pz = E.nz;
    int q = choose(dim, px, py, pz);
    int nader = E.n_ader;
    SD_for_ader(
        int dim1 = choose(dim ,_y_, _z_,_x_);
        int dim2 = choose(dim1,_y_, _z_,_x_);
        double b1=0;
        double b2=0;
        double v1=0;
        double v2=0;
        for(int ll=0; ll<q; ll++){
            if(dim==2){
                //Ez = vxBy - vyBx
                b1 += B1.Vector(t_id,0,k,j,i,kk,ll,ii)*sp_to_fp(jj,ll);//Bx
                b2 += B2.Vector(t_id,0,k,j,i,kk,jj,ll)*sp_to_fp(ii,ll);//By
                //for(int mm=0; mm<q; mm++){
                //    v1 += V.Vector(0,_x_,k,j,i,kk,ll,mm)*sp_to_fp(jj,ll)sp_to_fp(ii,mm);
                //    v2 += V.Vector(0,_y_,k,j,i,kk,mm,ll)*sp_to_fp(ii,ll)*sp_to_fp(jj,mm);
                //}
            }
            else if(dim==1){
                //Ey = vzBx - vxBz
                b1 += B1.Vector(t_id,0,k,j,i,kk,jj,ll)*sp_to_fp(ii,ll);//Bz
                b2 += B2.Vector(t_id,0,k,j,i,ll,jj,ii)*sp_to_fp(kk,ll);//Bx
                //for(int mm=0; mm<q; mm++){
                //    v1 += V.Vector(0,_z_,k,j,i,mm,jj,ll)*sp_to_fp(ii,ll)*sp_to_fp(kk,mm);
                //    v2 += V.Vector(0,_x_,k,j,i,ll,jj,mm)*sp_to_fp(kk,ll)*sp_to_fp(ii,mm);
                //}
            }
            else if(dim==0){
                //Ex = vyBz - vzBy
                b1 += B1.Vector(t_id,0,k,j,i,ll,jj,ii)*sp_to_fp(kk,ll);//By
                b2 += B2.Vector(t_id,0,k,j,i,kk,ll,ii)*sp_to_fp(jj,ll);//Bz
                //for(int mm=0; mm<q; mm++){
                //    v1 += V.Vector(0,_y_,k,j,i,ll,mm,ii)*sp_to_fp(kk,ll)*sp_to_fp(jj,mm);
                //    v2 += V.Vector(0,_z_,k,j,i,mm,ll,ii)*sp_to_fp(jj,ll)*sp_to_fp(kk,mm);
                //}
            }
        }
        v1 = velocity(dim1,x(i,ii),y(j,jj),z(k,kk));
        v2 = velocity(dim2,x(i,ii),y(j,jj),z(k,kk));
        E.Vector(t_id,0,k,j,i,kk,jj,ii) = v1*b2-v2*b1;
        E.Vector(t_id,1,k,j,i,kk,jj,ii) = b1;
        E.Vector(t_id,2,k,j,i,kk,jj,ii) = b2;
        E.Vector(t_id,3,k,j,i,kk,jj,ii) = v1;
        E.Vector(t_id,4,k,j,i,kk,jj,ii) = v2;
        E.Vector(t_id,5,k,j,i,kk,jj,ii) = 0;
        E.Vector(t_id,6,k,j,i,kk,jj,ii) = b1;
        E.Vector(t_id,7,k,j,i,kk,jj,ii) = b2;

    );
}

void compute_rotational_B(
    SD_Solution E,
    SD_Solution D1,
    SD_Solution D2,
    double d1,
    double d2,
    Matrix dfp_to_sp,
    Matrix sp_to_fp,
    double nu,
    int dim){
    //D3 = d1B2 - d2B1
    //E.Vector -> (E3,B1,B2,v1,v2,E3d,B1d,B2d)
    int Nx = E.Nx;
    int Ny = E.Ny;
    int Nz = E.Nz;
    int px = D1.nx;
    int py = D1.ny;
    int pz = D1.nz;
    int q = choose(dim, px, py, pz); //p+1 (solution points)
    int nader = E.n_ader;
    //Compute d1B2:
    SD_for_ader(
        double d1b2=0;
        //Loop over flux points:
        for(int ll=0; ll<q+1; ll++){
            if(dim==2)
                //Dz = dxBy
                d1b2 += E.Vector(t_id,_b2d_,k,j,i,kk,jj,ll)*dfp_to_sp(ii,ll)/d1;
            else if(dim==1)
                //Dy = dzBx
                d1b2 += E.Vector(t_id,_b2d_,k,j,i,ll,jj,ii)*dfp_to_sp(kk,ll)/d1;
            else if(dim==0)
                //Dx = dyBz
                d1b2 += E.Vector(t_id,_b2d_,k,j,i,kk,ll,ii)*dfp_to_sp(jj,ll)/d1;
        }
        D1.Vector(t_id,0,k,j,i,kk,jj,ii) = d1b2;
    );
    px = D2.nx;
    py = D2.ny;
    pz = D2.nz;
    //Compute d2B1:
    SD_for_ader(
        double d2b1=0;
        //Loop over flux points:
        for(int ll=0; ll<q+1; ll++){
            if(dim==2)
                //Dz = dyBx
                d2b1 += E.Vector(t_id,_b1d_,k,j,i,kk,ll,ii)*dfp_to_sp(jj,ll)/d2;
            else if(dim==1)
                //Dy = dxBz
                d2b1 += E.Vector(t_id,_b1d_,k,j,i,kk,jj,ll)*dfp_to_sp(ii,ll)/d2;
            else if(dim==0)
                //Dx = dzBy
                d2b1 += E.Vector(t_id,_b1d_,k,j,i,ll,jj,ii)*dfp_to_sp(kk,ll)/d2;
        }
        D2.Vector(t_id,0,k,j,i,kk,jj,ii) = d2b1;
    );
    //Output: D at solution points
    //Now we interpolate D back to flux points
    px = E.nx;
    py = E.ny;
    pz = E.nz;
    SD_for_ader(
        double d1=0;
        double d2=0;
        //Loop over solution points
        for(int ll=0; ll<q; ll++){
            if(dim==2){
                //Dz = dxBy - dyBx
                d2 += D2.Vector(t_id,0,k,j,i,kk,ll,ii)*sp_to_fp(jj,ll);
                d1 += D1.Vector(t_id,0,k,j,i,kk,jj,ll)*sp_to_fp(ii,ll);
            }
            else if(dim==1){
                //Dy = dzBx - dxBz
                d2 += D2.Vector(t_id,0,k,j,i,kk,jj,ll)*sp_to_fp(ii,ll);
                d1 += D1.Vector(t_id,0,k,j,i,ll,jj,ii)*sp_to_fp(kk,ll);
            }
            else if(dim==0){
                //Dx = dyBz - dzBy
                d2 += D2.Vector(t_id,0,k,j,i,ll,jj,ii)*sp_to_fp(kk,ll);
                d1 += D1.Vector(t_id,0,k,j,i,kk,ll,ii)*sp_to_fp(jj,ll);
            }
        }
        E.Vector(t_id,_E_,k,j,i,kk,jj,ii) -= nu*(d1-d2);
    );
}

void update_B_prediction(
    SD_Solution B,
    SD_Solution B_ader,
    SD_Solution E_1,
    SD_Solution E_2,
    Matrix dfp_to_sp,
    Matrix invader,
    Vector w,
    double d1,
    double d2,
    double dt,
    int dim){
    int Nx = B.Nx;
    int Ny = B.Ny;
    int Nz = B.Nz;
    int px = B.nx;
    int py = B.ny;
    int pz = B.nz;
    int nader = B_ader.n_ader;
    int q = choose(dim, px, py, pz);
    SD_for_active_cells(
        double dBdt[10];
        int t_id;
        int ll;
        double B_old;
        double dB;
        double E;
        B_old =  B.Vector(0,0,k,j,i,kk,jj,ii);
        for(t_id =0; t_id<nader; t_id++){
            dBdt[t_id] = 0;
            for(ll=0; ll<q; ll++){
                if(dim==0){
                    //dBxdt = dEydz - dEzdy
                    E = E_1.Vector(t_id,0,k,j,i,ll,jj,ii);
                    dBdt[t_id] += E*dfp_to_sp(kk,ll)/d2;
                    E = E_2.Vector(t_id,0,k,j,i,kk,ll,ii);
                    dBdt[t_id] -= E*dfp_to_sp(jj,ll)/d1;
                }
                else if(dim==1){
                    //dBydt = dEzdx - dExdz
                    E = E_1.Vector(t_id,0,k,j,i,kk,jj,ll);
                    dBdt[t_id] += E*dfp_to_sp(ii,ll)/d2;
                    E = E_2.Vector(t_id,0,k,j,i,ll,jj,ii);
                    dBdt[t_id] -= E*dfp_to_sp(kk,ll)/d1; 
                }
                else{
                    //dBzdt = dExdy - dEydx
                    E = E_1.Vector(t_id,0,k,j,i,kk,ll,ii);
                    dBdt[t_id] += E*dfp_to_sp(jj,ll)/d2;
                    E = E_2.Vector(t_id,0,k,j,i,kk,jj,ll);
                    dBdt[t_id] -= E*dfp_to_sp(ii,ll)/d1;
                }
            }
        }
        for(t_id=0; t_id<nader; t_id++){
            dB=0;
            for(ll=0; ll<nader; ll++){
                dB += dBdt[ll]*invader(t_id,ll)*w(ll)*dt;
            }
            B_ader.Vector(t_id,0,k,j,i,kk,jj,ii) = B_old - dB;
        }
    );
}

void update_B_solution(
    SD_Solution B,
    SD_Solution E_1,
    SD_Solution E_2,
    Matrix dfp_to_sp,
    Vector w,
    double d1,
    double d2,
    double dt,
    int dim){

    int Nx = B.Nx;
    int Ny = B.Ny;
    int Nz = B.Nz;
    int px = B.nx;
    int py = B.ny;
    int pz = B.nz;
    int nader = E_1.n_ader;
    int q = choose(dim, px, py, pz);
    //cout<<dim<<" "<<q<<endl;
    SD_for_active_cells(
        double dBdt;
        double dB=0;
        double E;
        for(int t_id =0; t_id<nader; t_id++){
            dBdt = 0;
            for(int ll=0; ll<q; ll++){
                if(dim==0){
                    //dBxdt = dEydz - dEzdy
                    E = E_1.Vector(t_id,0,k,j,i,ll,jj,ii);
                    dBdt += E*dfp_to_sp(kk,ll)/d2;
                    E = E_2.Vector(t_id,0,k,j,i,kk,ll,ii);
                    dBdt -= E*dfp_to_sp(jj,ll)/d1;
                }
                else if(dim==1){
                    //dBydt = dEzdx - dExdz
                    E = E_1.Vector(t_id,0,k,j,i,kk,jj,ll);
                    dBdt += E*dfp_to_sp(ii,ll)/d2;
                    E = E_2.Vector(t_id,0,k,j,i,ll,jj,ii);
                    dBdt -= E*dfp_to_sp(kk,ll)/d1; 
                }
                else{
                    //dBzdt = dExdy - dEydx
                    E = E_1.Vector(t_id,0,k,j,i,kk,ll,ii);
                    dBdt += E*dfp_to_sp(jj,ll)/d2;
                    E = E_2.Vector(t_id,0,k,j,i,kk,jj,ll);
                    dBdt -= E*dfp_to_sp(ii,ll)/d1;
                }
            }
            dB += dBdt*w(t_id)*dt;
        }
        B.Vector(0,0,k,j,i,kk,jj,ii) -= dB;
    );
}

void compute_B2_cv(
    SD_Solution B2,
    SD_Solution B_x,
    SD_Solution B_y,
    SD_Solution B_z,
    Matrix fp_to_cv,
    Matrix sp_to_cv
    ){
    int Nx = B2.Nx;
    int Ny = B2.Ny;
    int Nz = B2.Nz;
    int px = B2.nx;
    int py = B2.ny;
    int pz = B2.nz;
    int q = B_x.nx;
    //cout<<q<<":"<<Nx<<","<<Ny<<","<<Nz<<","<<px<<","<<py<<","<<pz<<endl;
    SD_for_cells(
        int ll;
        int mm;
        int nn;
        double bx=0;
        double by=0;
        double bz=0;
        for(nn=0; nn<pz; nn++)
            for(mm=0; mm<py; mm++)
                for(ll=0; ll<q; ll++)
                    bx += B_x.Vector(0,0,k,j,i,nn,mm,ll)*sp_to_cv(kk,nn)*sp_to_cv(jj,mm)*fp_to_cv(ii,ll);

        for(nn=0; nn<pz; nn++)
            for(mm=0; mm<q; mm++)
                for(ll=0; ll<px; ll++)
                    by += B_y.Vector(0,0,k,j,i,nn,mm,ll)*sp_to_cv(kk,nn)*fp_to_cv(jj,mm)*sp_to_cv(ii,ll);     

        for(nn=0; nn<q; nn++)
            for(mm=0; mm<py; mm++)
                for(ll=0; ll<px; ll++)
                    bz += B_z.Vector(0,0,k,j,i,nn,mm,ll)*fp_to_cv(kk,nn)*sp_to_cv(jj,mm)*sp_to_cv(ii,ll);
        
        B2.Vector(0,0,k,j,i,kk,jj,ii) = bx;
        B2.Vector(0,1,k,j,i,kk,jj,ii) = by;
        B2.Vector(0,2,k,j,i,kk,jj,ii) = bz;
        B2.Vector(0,3,k,j,i,kk,jj,ii) = bx*bx + by*by + bz*bz;
    );
}

void compute_B_cv_from_cf(
    FV_Solution B,
    SD_Solution B_x,
    SD_Solution B_y,
    SD_Solution B_z,
    Matrix fp_to_cv
    ){
    int Nx = B_x.Nx;
    int Ny = B_x.Ny;
    int Nz = B_x.Nz;
    int px = B_x.nx-1;
    int py = B_x.ny;
    int pz = B_x.nz;
    int q = px+1;
    int qx = px;
    int qy = py;
    int qz = pz;
    SD_for_active_cells(
        int ll;
        double bx=0;
        double by=0;
        double bz=0;
        for(ll=0; ll<q; ll++){
            bx += B_x.Vector(0,0,k,j,i,kk,jj,ll)*fp_to_cv(ii,ll);
            by += B_y.Vector(0,0,k,j,i,kk,ll,ii)*fp_to_cv(jj,ll); 
            bz += B_z.Vector(0,0,k,j,i,ll,jj,ii)*fp_to_cv(kk,ll);  
        }        
        B.Vector(0,K,J,I) = bx;
        B.Vector(1,K,J,I) = by;
        B.Vector(2,K,J,I) = bz;
        B.Vector(3,K,J,I) = bx*bx + by*by + bz*bz;
    );
}

KOKKOS_INLINE_FUNCTION
double upwind(double left, double right, double vel){
    if(vel==0) return 0.5*(left+right);
    return (vel > 0 ? left:right);
}

#define VEL 0
void E_riemann_solver(
    SD_Solution E,
    int dim,
    int _v1_){
    //Store fluxes at the interface between elements to then solve the unique flux
    int Nx = E.Nx-(dim==0); //Active faces
    int Ny = E.Ny-(dim==1);
    int Nz = E.Nz-(dim==2);
    int px = dim==0 ? 1:E.nx; 
    int py = dim==1 ? 1:E.ny; 
    int pz = dim==2 ? 1:E.nz; 
    int n = choose(dim, E.nx, E.ny, E.nz);
    int nader = E.n_ader;
    int nvar = E.n_var;
    SD_for_cells(
        double v;
        double v_L;
        double v_R;
        double e_L;
        double e_R;
        double e;
        int var;
        int NidL[3];
        int nidL[3];
        int NidR[3];
        int nidR[3];
        int l = choose(dim,i,j,k);
        indices(NidL,nidL,k,j,i,kk,jj,ii,l  ,n-1,dim);
        indices(NidR,nidR,k,j,i,kk,jj,ii,l+1,  0,dim);
        for(int t_id=0; t_id<nader; t_id++){
            var=_v1_;
            v_L = E.Vector(INDICES_L);
            v_R = E.Vector(INDICES_R);
            v = max(v_L,v_R);
            for(var=0; var<nvar-3; var++){
                e_L = E.Vector(INDICES_L);
                e_R = E.Vector(INDICES_R);
                e = upwind(e_L,e_R,v);
                E.Vector(INDICES_L) = e;
                E.Vector(INDICES_R) = e;
            }
            //Rusanov solver for diffusive terms
            for(var=nvar-2; var<nvar; var++){
                e_L = E.Vector(INDICES_L);
                e_R = E.Vector(INDICES_R);
                e = upwind(e_L,e_R,VEL);
                E.Vector(INDICES_L) = e;
                E.Vector(INDICES_R) = e;
            }
        }
    );
}

void E_Ohmic_riemann_solver(
    SD_Solution E,
    int dim,
    int _v1_){
    //Store fluxes at the interface between elements to then solve the unique flux
    int Nx = E.Nx-(dim==0); //Active faces
    int Ny = E.Ny-(dim==1);
    int Nz = E.Nz-(dim==2);
    int px = dim==0 ? 1:E.nx; 
    int py = dim==1 ? 1:E.ny; 
    int pz = dim==2 ? 1:E.nz; 
    int n = choose(dim, E.nx, E.ny, E.nz);
    int nader = E.n_ader;
    SD_for_cells(
        double e_L;
        double e_R;
        double e;
        int var=0;
        int NidL[3];
        int nidL[3];
        int NidR[3];
        int nidR[3];
        int l = choose(dim,i,j,k);
        indices(NidL,nidL,k,j,i,kk,jj,ii,l  ,n-1,dim);
        indices(NidR,nidR,k,j,i,kk,jj,ii,l+1,  0,dim);
        //Rusanov solver for diffusive terms
        for(int t_id=0; t_id<nader; t_id++){
            e_L = E.Vector(INDICES_L);
            e_R = E.Vector(INDICES_R);
            e = upwind(e_L,e_R,-VEL);
            E.Vector(INDICES_L) = e;
            E.Vector(INDICES_R) = e;
        }
    );
}

//////////////
/// FV
//////////////

void edge_integral(
    SD_Solution E_ep,
    FV_Solution E,
    Matrix sp_to_cv,
    int t_id,
    int dim){
    //Electric field is located at faces
    // in their respective perpendicular
    // dimension. We need then to include 
    // the right ghost element in order
    // to include the last active faces in
    // those dimensions.
    int Nx = E_ep.Nx+(dim==_x_ ? 0:1);
    int Ny = E_ep.Ny+(dim==_y_ ? 0:1);
    int Nz = E_ep.Nz+(dim==_z_ ? 0:1);
    // For those perpendicular dimensions
    // we skip the last faces of each element,
    // which are identical at this point to
    // the first faces of the adjacent element
    int px = E_ep.nx-(dim==_x_ ? 0:1);
    int py = E_ep.ny-(dim==_y_ ? 0:1);
    int pz = E_ep.nz-(dim==_z_ ? 0:1);
    //Needed to compute K,J,I
    int qx = px;
    int qy = py;
    int qz = pz;
    int q  = choose(dim, px, py, pz);
    int Ni = E.Nx;
    int Nj = E.Ny;
    int Nk = E.Nz;
    //cout<<dim<<" "<<Nx<<" "<<Ny<<" "<<Nz<<" "<<px<<py<<pz<<endl;
    SD_for_active_cells(
        double s;
        double e=0;
        int nid[3];
        int id = choose(dim, ii, jj, kk);
        //Loop over solution points along the edge   
        for(int ll=0; ll<q; ll++){
            indices_n(nid,kk,jj,ii,ll,dim);
            s = E_ep.Vector(t_id,0,k,j,i,NODE);
            #ifdef _2D_
            s *= sp_to_cv(id,ll);
            #endif
            e += s;
        }
        //We iterate over N+1 elements
        //E is smaller than E_ep for p>1
        //This ensures no overflow when computing the last element
        if(K < Nk && J < Nj && I < Ni)
            E.Vector(0,K,J,I) = e;
    );
}

void fv_update_B_solution(
    FV_Solution B_new,
    FV_Solution B_old,
    SD_Solution B,
    FV_Solution E_1,
    FV_Solution E_2,
    Vector faces_1,
    Vector faces_2,
    Vector w,
    double dt,
    int t_id,
    int dim,
    bool update){

    int Nx = B.Nx;
    int Ny = B.Ny;
    int Nz = B.Nz;
    int px = B.nx;
    int py = B.ny;
    int pz = B.nz;
    //Needed to compute K,J,I
    int qx = px-(dim==_x_);
    int qy = py-(dim==_y_);
    int qz = pz-(dim==_z_);
    int Ni = B_new.Nx;
    int Nj = B_new.Ny;
    int Nk = B_new.Nz;
    int dim1 = choose(dim ,_y_,_z_,_x_);
    int dim2 = choose(dim1,_y_,_z_,_x_);
    //cout<<dim<<" "<<q<<endl;
    SD_for_active_cells(
        double dBdt;
        double b_new;
        double b_old;
        double ep;
        double h;
        int id;
        dBdt = 0;
        b_old = B.Vector(0,0,k,j,i,kk,jj,ii);
        if(K < Nk && J < Nj && I < Ni){
            //This is going to be used to limit the solution
            B_old.Vector(0,K,J,I) = b_old;
            //dBxdt = dEydz - dEzdy
            //dBydt = dEzdx - dExdz
            //dBzdt = dExdy - dEydx
            ep = E_1.Vector(0,K+(dim2==_z_),J+(dim2==_y_),I+(dim2==_x_));
            id = choose(dim2, I, J, K);
            h = (faces_2(id+1)-faces_2(id));
            dBdt = (ep-E_1.Vector(0,K,J,I))/h;

            ep = E_2.Vector(0,K+(dim1==_z_),J+(dim1==_y_),I+(dim1==_x_));
            id = choose(dim1, I, J, K);
            h = (faces_1(id+1)-faces_1(id));
            dBdt-= (ep-E_2.Vector(0,K,J,I))/h;

            b_new = b_old - dBdt*w(t_id)*dt;
            //Need if condition to only allow left threads to write at 
            //element interfaces
            B_new.Vector(0,K,J,I) = b_new;
            if(update)
                B.Vector(0,0,k,j,i,kk,jj,ii) = b_new;
        }
    );
}

//////////////////
/// MUSCL-HANCOCK
//////////////////

KOKKOS_INLINE_FUNCTION
void slopes(
    FV_Vector W,
    Vector x_c,
    Vector x_f,
    #if Y
    Vector y_c,
    Vector y_f,
    #endif
    #if Z
    Vector z_c,
    Vector z_f,
    #endif
    int k,
    int j,
    int i,
    double WL[3][NVAR],
    double WR[3][NVAR],
    double dt){
    
    double wL;
    double wR;
    double h;
    double w[NVAR];
    double dwx[NVAR];
    double dwy[NVAR];
    double dwz[NVAR];
    double dwt[NVAR];
    for(int var=0; var<NVAR; var++){
        w[var] = W(var,k,j,i);
        wL=W(var,k,j,i-1);
        wR=W(var,k,j,i+1);
        h = x_c(i+1)-x_c(i);
        dwx[var] = minmod((wR - w[var])/h,(w[var] - wL)/h,x_f(i),x_f(i+1));
        #if Y
        wL=W(var,k,j-1,i);
        wR=W(var,k,j+1,i);
        h = y_c(j+1)-y_c(j);
        dwy[var] = minmod((wR - w[var])/h,(w[var] - wL)/h,y_f(j),y_f(j+1));
        #endif
        #if Z
        wL=W(var,k-1,j,i);
        wR=W(var,k+1,j,i);
        h = z_c(k+1)-z_c(k);
        dwz[var] = minmod((wR - w[var])/h,(w[var] - wL)/h,z_f(k),z_f(k+1));
        #endif
    }
    //corrector(w,dwt,dwz,dwy,dwx);
    for(int var=0; var<NVAR; var++){
        h = x_f(i+1)-x_f(i);
        WR[_x_][var] = w[var] - dwx[var] + dwt[var]*dt/h;
        WL[_x_][var] = w[var] + dwx[var] + dwt[var]*dt/h;
        #if Y
        h = y_f(j+1)-y_f(j);
        WR[_y_][var] = w[var] - dwy[var] + dwt[var]*dt/h;
        WL[_y_][var] = w[var] + dwy[var] + dwt[var]*dt/h;
        #endif
        #if Z
        h = z_f(k+1)-z_f(k);
        WR[_z_][var] = w[var] - dwz[var] + dwt[var]*dt/h;
        WL[_z_][var] = w[var] + dwz[var] + dwt[var]*dt/h;
        #endif
    } 
}