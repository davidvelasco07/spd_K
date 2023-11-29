#include "spd_k.hpp"

KOKKOS_INLINE_FUNCTION
double square_signal(int var, double x, double y, double z){
    if(var==0){
        #ifndef _3D_
        if( abs(x-0.5)<0.25 && abs(y-0.5)<0.25)
        #else
        if( abs(x-0.5)<0.25 && abs(y-0.5)<0.25 && abs(z-0.5)<0.25)
        #endif
            return 2;
        else
            return 1;
    }
    else if(var==_vz_)
        return 0;
    else
        return 1;
}

KOKKOS_INLINE_FUNCTION
double sine_wave(int var, double x, double y, double z){
    if(var==0)
       return 1.0+0.125*(sin(2*PI*(x+y)));
    else if(var==_vz_)
        return 0;
    else
        return 1;
}

KOKKOS_INLINE_FUNCTION
double sedov_blast(int var, double x, double y, double z){
    //Note: set gmma=5./3.
    double r,R=0.1;
    #ifndef _3D_
    r=sqrt(pow(x-0.5,2) + pow(y-0.5,2));
    #else
    r=sqrt(pow(x-0.5,2.0) + pow(y-0.5,2.0) + pow(z-0.5,2.0));
    #endif
    if(var==0) return 1.0;
    else if(var==_p_){
        if(r<R)
            return 1E-6+(gmma-1);
        else
            return 1E-6;
    }
    else
        return 0;
}

KOKKOS_INLINE_FUNCTION
double spherical_blast(int var, double x, double y, double z){
    //Note: set gmma=5./3.
    double r,R=0.1;
    double xc=0.5*LENGHT;
    double yc=0.5*LENGHT;
    double zc=0.5*LENGHT;
    #ifndef _3D_
    r=sqrt(pow(x-xc,2) + pow(y-yc,2));
    #else
    r=sqrt(pow(x-xc,2) + pow(y-yc,2) + pow(z-zc,2));
    #endif
    if(var==0){
        return 1;
    }
    else if(var==_p_){
        if(r<R)
            return 10;
        else
            return 0.1;
    }
    else
        return 0;
}

void Initialize(
    SD_Solution U,
    Matrix faces_x,
    Matrix faces_y,
    Matrix faces_z,
    Vector x_sp,
    Vector w_sp){

    int Nx = U.Nx;
    int Ny = U.Ny;
    int Nz = U.Nz;
    int px = U.nx;
    int py = U.ny;
    int pz = U.nz;
    int nader = U.n_ader;
    int nvar = U.n_var;
    SD_for_all(
        double value=0;
        double s;
        double x;
        double y=0;
        double z=0;
        for(int nn=0; nn<pz; nn++){
            #if Z
            z = faces_z(k,kk) + x_sp(nn)*(faces_z(k,kk+1)-faces_z(k,kk));
            #endif
            for(int mm=0; mm<py; mm++){
                y = faces_y(j,jj) + x_sp(mm)*(faces_y(j,jj+1)-faces_y(j,jj));
                for(int ll=0; ll<px; ll++){
                    x = faces_x(i,ii) + x_sp(ll)*(faces_x(i,ii+1)-faces_x(i,ii));
                    s = sine_wave(var,x,y,z);
                    #if X
                    s*=w_sp(ll);
                    #endif
                    #if Y
                    s*=w_sp(mm);
                    #endif
                    #if Z
                    s*=w_sp(nn);
                    #endif
                    value+=s;
                }
            }
        }
        U.Vector(t_id,var,k,j,i,kk,jj,ii) = value;
    );
}

////////////////
// INDUCTION
////////////////

KOKKOS_INLINE_FUNCTION
double magnetic_loop(int var, double x, double y, double z){
    x = x-0.5;
    y = y-0.5;
    double r = sqrt(x*x + y*y);
    double A0 = 0.001;
    double R = 0.3;
    if(var==2){
       if(r<=R)
            return A0*(R-r);
        else
            return 0;
    }
    else
        return 0;
}

KOKKOS_INLINE_FUNCTION
double rotating_loop(int var, double x, double y, double z){
    x = x-0.75;
    y = y-0.5;
    double r = sqrt(x*x + y*y);
    double A0 = 0.001;
    double R = 0.1;
    if(var==2){
       if(r<=R)
            return A0*(R-r);
        else
            return 0;
    }
    else
        return 0;
}

KOKKOS_INLINE_FUNCTION
double delta_function(int var, double x, double y, double z, double dx){
    x=x-0.5;
    if(var==2){
        if( x < -0.9*dx)
            return 0;
        else if ( x <= 0.9*dx)
            return -(x+.9*dx);
        else
            return -1.8*dx;
    }
    else
        return 0;
}

KOKKOS_INLINE_FUNCTION
double delta_function_2d(int var, double x, double y, double z, double dx){
    //double ;
    x=x-0.5;
    z=z-0.5;
    //double r = sqrt(x*x + z*z);
    if(var==2){
        if( x < -0.9*dx)
            return 0;
        else if ( x <= 0.9*dx)
            return -(x+.9*dx);
        else
            return -1.8*dx;
    }
    if(var==0){
        if( z < -0.9*dx)
            return 0;
        else if ( z <= 0.9*dx)
            return (z+.9*dx);
        else
            return 1.8*dx;
    }
    else
        return 0;
}

KOKKOS_INLINE_FUNCTION
double gaussian(int var, double x, double y, double z){
    //double ;
    x=x-0.5;
    if(var==2){
        return 0.5*erfc(sqrt(2.0)*x/0.25)-1;
    }
    else
        return 0;
}

KOKKOS_INLINE_FUNCTION
double ponomarenko(int var, double x, double y, double z){
    double L = LENGHT;
    double k = 2*PI/L;
    x = x-0.5*L;
    y = y-0.5*L;
    double r = sqrt(x*x+y*y);
    double theta = atan2(y,x);
    double P = r*(2-r);
    //Bx = dyAz - dzAy = P*cos(theta)
    //By = dzAx - dxAz =-(P+r*Pp)*sin(theta)
    //Bz = dxAy - dyAx = P*cos(theta)
    //return 1e-3;
    double m=1;
    double Ar = max(P,0.0)*cos(m*theta)*sin(z*k);
    if(var==2){//Az
        return 0;
    }
    else if(var==1){//Ay
        return Ar*sin(theta);
    }
    else if(var==0){//Ax
        return Ar*cos(theta);
    }
    else
        return 0;
}

void Initialize_ep(
    SD_Solution A,
    Matrix Xs,
    Matrix Ys,
    Matrix Zs,
    int dim){

    int Nx = A.Nx;
    int Ny = A.Ny;
    int Nz = A.Nz;
    int px = A.nx;
    int py = A.ny;
    int pz = A.nz;
    SD_for_cells(
        double x;
        double y;
        double z;
        z = Zs(k,kk);
        y = Ys(j,jj);
        x = Xs(i,ii);
        A.Vector(0,0,k,j,i,kk,jj,ii) = magnetic_loop(dim,x,y,z);
    );
}


