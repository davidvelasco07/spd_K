#include "spd_k.hpp"
#include "user_ic.hpp"

//All initial conditions return PRIMITIVE variables: var 0 = density,
//_vx_/_vy_/_vz_ = velocities, _p_ = pressure. Runtime parameters come from
//the <problem> input block through ProblemParams (see problem_defaults in
//main.cpp for the per-problem meaning of each field).

KOKKOS_INLINE_FUNCTION
double square_signal(int var, double x, double y, double z,
                     bool ay, bool az, ProblemParams pp){
    //Advected top-hat: density pp.d1 inside a box of half-width pp.radius,
    //pp.d0 outside; uniform velocity (pp.v1,pp.v2,pp.v3), pressure pp.p0
    if(var==0){
        bool in = abs(x-0.5)<pp.radius;
        if(ay) in = in && abs(y-0.5)<pp.radius;
        if(az) in = in && abs(z-0.5)<pp.radius;
        return in ? pp.d1 : pp.d0;
    }
    else if(var==_vx_) return pp.v1;
    else if(var==_vy_) return pp.v2;
    else if(var==_vz_) return pp.v3;
    else if(var==_p_)  return pp.p0;
    else return 0;
}

KOKKOS_INLINE_FUNCTION
double sine_wave(int var, double x, double y, double z, ProblemParams pp){
    if(var==0)
       return pp.d0+pp.amp*(sin(2*PI*(x+y)));
    else if(var==_vx_) return pp.v1;
    else if(var==_vy_) return pp.v2;
    else if(var==_vz_) return pp.v3;
    else if(var==_p_)  return pp.p0;
    else return 0;
}

KOKKOS_INLINE_FUNCTION
double sedov_blast(int var, double x, double y, double z, double gm,
                   bool az, ProblemParams pp){
    //Note: set gamma=5./3.
    double r,R=pp.radius;
    r=sqrt(pow(x-0.5,2.0) + pow(y-0.5,2.0) + (az ? pow(z-0.5,2.0) : 0.0));
    if(var==0) return pp.d0;
    else if(var==_p_){
        if(r<R)
            return 1E-6+(gm-1);
        else
            return 1E-6;
    }
    else
        return 0;
}

KOKKOS_INLINE_FUNCTION
double spherical_blast(int var, double x, double y, double z,
                       bool az, ProblemParams pp){
    //Note: set gamma=5./3.
    double r,R=pp.radius;
    double xc=0.5*LENGHT;
    double yc=0.5*LENGHT;
    double zc=0.5*LENGHT;
    r=sqrt(pow(x-xc,2) + pow(y-yc,2) + (az ? pow(z-zc,2) : 0.0));
    if(var==0){
        return pp.d0;
    }
    else if(var==_p_){
        if(r<R)
            return pp.p0;
        else
            return pp.p1;
    }
    else
        return 0;
}

KOKKOS_INLINE_FUNCTION
double sod_shock_tube(int var, double x, double y, double z, ProblemParams pp){
    //Classic Sod tube along direction pp.dir with the interface at
    //pp.radius: (d0,p0) on the left, (d1,p1) on the right, gas at rest.
    //Use gradfree boundaries in the tube direction.
    double s = (pp.dir==_x_ ? x : (pp.dir==_y_ ? y : z));
    bool left = s < pp.radius;
    if(var==0)        return left ? pp.d0 : pp.d1;
    else if(var==_p_) return left ? pp.p0 : pp.p1;
    else return 0;
}

KOKKOS_INLINE_FUNCTION
double shu_osher(int var, double x, double y, double z, ProblemParams pp){
    //Shu & Osher (1989): Mach-3 shock running into a sinusoidal density
    //field. Defined on x in [-1,1]: use x1len=2 (the box coordinate is
    //shifted by -1 here). pp.amp is the sine amplitude. gamma = 1.4,
    //tlim = 0.47, gradfree boundaries in x.
    double xs = x - 1.0;
    bool left = xs < -0.8;
    if(var==0)         return left ? 3.857143 : 1.0+pp.amp*sin(5*PI*xs);
    else if(var==_vx_) return left ? 2.629369 : 0.0;
    else if(var==_p_)  return left ? 10.33333 : 1.0;
    else return 0;
}

KOKKOS_INLINE_FUNCTION
double kelvin_helmholtz(int var, double x, double y, double z, ProblemParams pp){
    //Double shear layer in y with a sinusoidal vy perturbation: density
    //pp.d1 and velocity +pp.v1 in the central band (0.25 < y < 0.75),
    //pp.d0 and -pp.v1 outside; pressure pp.p0. Periodic boundaries.
    bool inner = (y>0.25 && y<0.75);
    if(var==0)         return inner ? pp.d1 : pp.d0;
    else if(var==_vx_) return inner ? pp.v1 : -pp.v1;
    else if(var==_vy_)
        return pp.amp*sin(4*PI*x)
               *(exp(-pow(y-0.25,2)/(2*pp.sigma*pp.sigma))
                +exp(-pow(y-0.75,2)/(2*pp.sigma*pp.sigma)));
    else if(var==_p_)  return pp.p0;
    else return 0;
}

KOKKOS_INLINE_FUNCTION
double implosion(int var, double x, double y, double z, ProblemParams pp){
    //Liska & Wendroff (2003) sec 4.7 corner implosion: a low-density,
    //low-pressure triangle below the diagonal x+y = pp.radius in a gas at
    //rest. Domain [0,0.3]^2 (x1len=x2len=0.3), reflective walls, gamma=1.4.
    //The x<->y symmetry of the late-time jet is the hard part of this test.
    bool inside = (x + y) < pp.radius;
    if(var==0)        return inside ? pp.d1 : pp.d0;
    else if(var==_p_) return inside ? pp.p1 : pp.p0;
    else return 0;
}

KOKKOS_INLINE_FUNCTION
double initial_condition(int problem, int var, double x, double y, double z,
                         double gm, bool ay, bool az, ProblemParams pp){
    switch(problem){
        case _ic_sine_wave_:        return sine_wave(var,x,y,z,pp);
        case _ic_sedov_:            return sedov_blast(var,x,y,z,gm,az,pp);
        case _ic_spherical_blast_:  return spherical_blast(var,x,y,z,az,pp);
        case _ic_square_:           return square_signal(var,x,y,z,ay,az,pp);
        case _ic_sod_:              return sod_shock_tube(var,x,y,z,pp);
        case _ic_shu_osher_:        return shu_osher(var,x,y,z,pp);
        case _ic_kelvin_helmholtz_: return kelvin_helmholtz(var,x,y,z,pp);
        case _ic_implosion_:        return implosion(var,x,y,z,pp);
        case _ic_user_:             return user_ic(var,x,y,z,gm,ay,az,pp);
        default:                    return 0;
    }
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
    int problem = cfg.problem;
    double gm = cfg.gamma;
    bool ay = cfg.active[_y_];
    bool az = cfg.active[_z_];
    ProblemParams pp = cfg.pp;
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        for(int t_id=0; t_id<nader; t_id++){
        for(int var=0; var<nvar; var++){
        double value=0;
        double s;
        double x;
        double y=0;
        double z=0;
        for(int nn=0; nn<pz; nn++){
            if(az)
                z = faces_z(k,kk) + x_sp(nn)*(faces_z(k,kk+1)-faces_z(k,kk));
            for(int mm=0; mm<py; mm++){
                if(ay)
                    y = faces_y(j,jj) + x_sp(mm)*(faces_y(j,jj+1)-faces_y(j,jj));
                for(int ll=0; ll<px; ll++){
                    x = faces_x(i,ii) + x_sp(ll)*(faces_x(i,ii+1)-faces_x(i,ii));
                    s = initial_condition(problem,var,x,y,z,gm,ay,az,pp);
                    s*=w_sp(ll);
                    if(ay) s*=w_sp(mm);
                    if(az) s*=w_sp(nn);
                    value+=s;
                }
            }
        }
        U.Vector(t_id,var,k,j,i,kk,jj,ii) = value;
        }}
    });
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
    sd_for_cells(Nz,Ny,Nx,pz,py,px, KOKKOS_LAMBDA(int k, int j, int i, int kk, int jj, int ii){
        double x;
        double y;
        double z;
        z = Zs(k,kk);
        y = Ys(j,jj);
        x = Xs(i,ii);
        A.Vector(0,0,k,j,i,kk,jj,ii) = magnetic_loop(dim,x,y,z);
    });
}


