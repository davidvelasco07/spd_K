#include "spd_k.hpp"

KOKKOS_INLINE_FUNCTION
double velocity(int var, double x, double y, double z){
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
    int q = dim==0 ? px:( dim==1 ? py : pz);
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
    int q = dim==0 ? px:( dim==1 ? py : pz);
    int nader = E.n_ader;
    SD_for_ader(
        int dim1 = dim==0 ? _y_ : (dim==1 ? _z_ : _x_);
        int dim2 = dim==0 ? _z_ : (dim==1 ? _x_ : _y_);
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
    //E.Vector -> (E3,B1,B2,v1,v2)
    int Nx = E.Nx;
    int Ny = E.Ny;
    int Nz = E.Nz;
    int px = D1.nx;
    int py = D1.ny;
    int pz = D1.nz;
    int q = dim==0 ? px:( dim==1 ? py : pz); //p+1 (solution points)
    int nader = E.n_ader;
    //Compute d1B2:
    SD_for_ader(
        double d1b2=0;
        //Loop over flux points:
        for(int ll=0; ll<q+1; ll++){
            if(dim==2)
                //Dz = dxBy
                d1b2 += E.Vector(t_id,7,k,j,i,kk,jj,ll)*dfp_to_sp(ii,ll)/d1;
            else if(dim==1)
                //Dy = dzBx
                d1b2 += E.Vector(t_id,7,k,j,i,ll,jj,ii)*dfp_to_sp(kk,ll)/d1;
            else if(dim==0)
                //Dx = dyBz
                d1b2 += E.Vector(t_id,7,k,j,i,kk,ll,ii)*dfp_to_sp(jj,ll)/d1;
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
                d2b1 += E.Vector(t_id,6,k,j,i,kk,ll,ii)*dfp_to_sp(jj,ll)/d2;
            else if(dim==1)
                //Dy = dxBz
                d2b1 += E.Vector(t_id,6,k,j,i,kk,jj,ll)*dfp_to_sp(ii,ll)/d2;
            else if(dim==0)
                //Dx = dzBy
                d2b1 += E.Vector(t_id,6,k,j,i,ll,jj,ii)*dfp_to_sp(kk,ll)/d2;
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
        E.Vector(t_id,5,k,j,i,kk,jj,ii) = nu*(d1-d2);
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
    int q = dim==0 ? px:( dim==1 ? py : pz);
    SD_for_cells(
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
                    E = E_1.Vector(t_id,0,k,j,i,ll,jj,ii)-E_1.Vector(t_id,5,k,j,i,ll,jj,ii);
                    dBdt[t_id] += E*dfp_to_sp(kk,ll)/d2;
                    E = E_2.Vector(t_id,0,k,j,i,kk,ll,ii)-E_2.Vector(t_id,5,k,j,i,kk,ll,ii);
                    dBdt[t_id] -= E*dfp_to_sp(jj,ll)/d1;
                }
                else if(dim==1){
                    //dBydt = dEzdx - dExdz
                    E = E_1.Vector(t_id,0,k,j,i,kk,jj,ll)-E_1.Vector(t_id,5,k,j,i,kk,jj,ll);
                    dBdt[t_id] += E*dfp_to_sp(ii,ll)/d2;
                    E = E_2.Vector(t_id,0,k,j,i,ll,jj,ii)-E_2.Vector(t_id,5,k,j,i,ll,jj,ii);
                    dBdt[t_id] -= E*dfp_to_sp(kk,ll)/d1; 
                }
                else{
                    //dBzdt = dExdy - dEydx
                    E = E_1.Vector(t_id,0,k,j,i,kk,ll,ii)-E_1.Vector(t_id,5,k,j,i,kk,ll,ii);
                    dBdt[t_id] += E*dfp_to_sp(jj,ll)/d2;
                    E = E_2.Vector(t_id,0,k,j,i,kk,jj,ll)-E_2.Vector(t_id,5,k,j,i,kk,jj,ll);
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
    int q = dim==0 ? px:( dim==1 ? py : pz);
    //cout<<dim<<" "<<q<<endl;
    SD_for_cells(
        double dBdt;
        double dB=0;
        double E;
        for(int t_id =0; t_id<nader; t_id++){
            dBdt = 0;
            for(int ll=0; ll<q; ll++){
                if(dim==0){
                    //dBxdt = dEydz - dEzdy
                    E = E_1.Vector(t_id,0,k,j,i,ll,jj,ii)-E_1.Vector(t_id,5,k,j,i,ll,jj,ii);
                    dBdt += E*dfp_to_sp(kk,ll)/d2;
                    E = E_2.Vector(t_id,0,k,j,i,kk,ll,ii)-E_2.Vector(t_id,5,k,j,i,kk,ll,ii);
                    dBdt -= E*dfp_to_sp(jj,ll)/d1;
                }
                else if(dim==1){
                    //dBydt = dEzdx - dExdz
                    E = E_1.Vector(t_id,0,k,j,i,kk,jj,ll)-E_1.Vector(t_id,5,k,j,i,kk,jj,ll);
                    dBdt += E*dfp_to_sp(ii,ll)/d2;
                    E = E_2.Vector(t_id,0,k,j,i,ll,jj,ii)-E_2.Vector(t_id,5,k,j,i,ll,jj,ii);
                    dBdt -= E*dfp_to_sp(kk,ll)/d1; 
                }
                else{
                    //dBzdt = dExdy - dEydx
                    E = E_1.Vector(t_id,0,k,j,i,kk,ll,ii)-E_1.Vector(t_id,5,k,j,i,kk,ll,ii);
                    dBdt += E*dfp_to_sp(jj,ll)/d2;
                    E = E_2.Vector(t_id,0,k,j,i,kk,jj,ll)-E_2.Vector(t_id,5,k,j,i,kk,jj,ll);
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

KOKKOS_INLINE_FUNCTION
double upwind(double left, double right, double vel){
    if(vel==0) return 0.5*(left+right);
    return (vel > 0 ? left:right);
}

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
    int n = dim==0 ? E.nx:(dim==1 ? E.ny:E.nz);
    int nader = E.n_ader;
    int nvar = E.n_var-3;
    SD_for_cells(
        double v;
        double v_L;
        double v_R;
        double e_L;
        double e_R;
        double e;
        double sigma=0.5;
        for(int t_id=0; t_id<nader; t_id++){
            if(dim==0){
                v_L = E.Vector(t_id,_v1_,k,j,i  ,kk,jj,n-1);
                v_R = E.Vector(t_id,_v1_,k,j,i+1,kk,jj,  0);
                v = max(v_L,v_R);
                for(int var=0; var<nvar; var++){
                    e_L = E.Vector(t_id,var,k,j,i  ,kk,jj,n-1);
                    e_R = E.Vector(t_id,var,k,j,i+1,kk,jj,  0);
                    e = upwind(e_L,e_R,v);
                    E.Vector(t_id,var,k,j,i  ,kk,jj,n-1) = e;
                    E.Vector(t_id,var,k,j,i+1,kk,jj,  0) = e;
                }
                for(int var=nvar+1; var<nvar+3; var++){
                    e_L = E.Vector(t_id,var,k,j,i  ,kk,jj,n-1);
                    e_R = E.Vector(t_id,var,k,j,i+1,kk,jj,  0);
                    e = 0.5*(e_L+e_R) - sigma*(e_L-e_R);
                    E.Vector(t_id,var,k,j,i  ,kk,jj,n-1) = e;
                    E.Vector(t_id,var,k,j,i+1,kk,jj,  0) = e;
                }
            }
            else if(dim==1){
                v_L = E.Vector(t_id,_v1_,k,j  ,i,kk,n-1,ii);
                v_R = E.Vector(t_id,_v1_,k,j+1,i,kk,  0,ii);
                v = max(v_L,v_R);
                for(int var=0; var<nvar; var++){
                    e_L = E.Vector(t_id,var,k,j  ,i,kk,n-1,ii);
                    e_R = E.Vector(t_id,var,k,j+1,i,kk,  0,ii);
                    e = upwind(e_L,e_R,v);
                    E.Vector(t_id,var,k,j  ,i,kk,n-1,ii) = e;
                    E.Vector(t_id,var,k,j+1,i,kk,  0,ii) = e;
                }
                for(int var=nvar+1; var<nvar+3; var++){
                    e_L = E.Vector(t_id,var,k,j  ,i,kk,n-1,ii);
                    e_R = E.Vector(t_id,var,k,j+1,i,kk,  0,ii);
                    e = 0.5*(e_L+e_R) - sigma*(e_L-e_R);
                    E.Vector(t_id,var,k,j  ,i,kk,n-1,ii) = e;
                    E.Vector(t_id,var,k,j+1,i,kk,  0,ii) = e;
                }
            }
            else if(dim==2){
                v_L = E.Vector(t_id,_v1_,k  ,j,i,n-1,jj,ii);
                v_R = E.Vector(t_id,_v1_,k+1,j,i,  0,jj,ii);
                v = max(v_L,v_R);
                for(int var=0; var<nvar; var++){
                    e_L = E.Vector(t_id,var,k  ,j,i,n-1,jj,ii);
                    e_R = E.Vector(t_id,var,k+1,j,i,  0,jj,ii);
                    e = upwind(e_L,e_R,v);
                    E.Vector(t_id,var,k  ,j,i,n-1,jj,ii) = e;
                    E.Vector(t_id,var,k+1,j,i,  0,jj,ii) = e;
                }
                for(int var=nvar+1; var<nvar+3; var++){
                    e_L = E.Vector(t_id,var,k  ,j,i,n-1,jj,ii);
                    e_R = E.Vector(t_id,var,k+1,j,i,  0,jj,ii);
                    e = 0.5*(e_L+e_R) - sigma*(e_L-e_R);
                    E.Vector(t_id,var,k  ,j,i,n-1,jj,ii) = e;
                    E.Vector(t_id,var,k+1,j,i,  0,jj,ii) = e;
                }
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
    int n = dim==0 ? E.nx:(dim==1 ? E.ny:E.nz);
    int nader = E.n_ader;
    int nvar = E.n_var;
    SD_for_cells(
        double e_L;
        double e_R;
        double e;
        double sigma=0.5;
        for(int t_id=0; t_id<nader; t_id++){
            if(dim==0){
                e_L = E.Vector(t_id,5,k,j,i  ,kk,jj,n-1);
                e_R = E.Vector(t_id,5,k,j,i+1,kk,jj,  0);
                e = 0.5*(e_L+e_R) + sigma*(e_L-e_R);
                E.Vector(t_id,5,k,j,i  ,kk,jj,n-1) = e;
                E.Vector(t_id,5,k,j,i+1,kk,jj,  0) = e;
            }
            else if(dim==1){
                e_L = E.Vector(t_id,5,k,j  ,i,kk,n-1,ii);
                e_R = E.Vector(t_id,5,k,j+1,i,kk,  0,ii);
                e = 0.5*(e_L+e_R) + sigma*(e_L-e_R);
                E.Vector(t_id,5,k,j  ,i,kk,n-1,ii) = e;
                E.Vector(t_id,5,k,j+1,i,kk,  0,ii) = e;
            }
            else if(dim==2){
                e_L = E.Vector(t_id,5,k  ,j,i,n-1,jj,ii);
                e_R = E.Vector(t_id,5,k+1,j,i,  0,jj,ii);
                e = 0.5*(e_L+e_R) + sigma*(e_L-e_R);
                E.Vector(t_id,5,k  ,j,i,n-1,jj,ii) = e;
                E.Vector(t_id,5,k+1,j,i,  0,jj,ii) = e;
            }
        }
    );
}