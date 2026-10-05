#include "spd_k.hpp"
#include "user_ic.hpp"
#include "dmr.hpp"

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
    //problem/p0 > 0 is the TOTAL energy E deposited in the radius (per unit
    //length in 2D), so the similarity solution's scale is (E t^2/rho)^(1/4)
    //in 2D and (E t^2/rho)^(1/5) in 3D; problem/p1 > 0 is the ambient
    //pressure. With both unset the deposit is the historical p = gamma-1
    //inside R over an ambient 1e-6, i.e. E = (gamma-1)/(gamma-1) * vol(R) =
    //pi R^2 in 2D, which the existing decks and goldens rely on.
    double r,R=pp.radius;
    r=sqrt(pow(x-0.5,2.0) + pow(y-0.5,2.0) + (az ? pow(z-0.5,2.0) : 0.0));
    const double pamb = pp.p1 > 0.0 ? pp.p1 : 1E-6;
    const double vol  = az ? 4.0*M_PI*R*R*R/3.0 : M_PI*R*R;
    const double pin  = pp.p0 > 0.0 ? pamb + (gm-1)*pp.p0/vol : 1E-6+(gm-1);
    if(var==0) return pp.d0;
    else if(var==_p_){
        if(r<R)
            return pin;
        else
            return pamb;
    }
    else
        return 0;
}

KOKKOS_INLINE_FUNCTION
double spherical_blast(int var, double x, double y, double z,
                       bool az, ProblemParams pp){
    //Note: set gamma=5./3.
    double r,R=pp.radius;
    double xc=pp.cx;
    double yc=pp.cy;
    double zc=pp.cz;
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
double woodward_colella(int var, double x, double y, double z, ProblemParams pp){
    //Woodward & Colella (1984) interacting blast waves on [0,1] along pp.dir:
    //rho = 1 at rest, p = 1000 for s < 0.1, 0.01 for 0.1 <= s < 0.9 and 100
    //beyond. Reflecting walls, gamma = 1.4, compare at t = 0.038 (Paper I
    //sec 4.2.2). The states are the standard ones and are hardcoded, as for
    //shu_osher.
    double s = (pp.dir==_x_ ? x : (pp.dir==_y_ ? y : z));
    if(var==0)        return 1.0;
    else if(var==_p_) return s < 0.1 ? 1000.0 : (s < 0.9 ? 0.01 : 100.0);
    else return 0;
}

KOKKOS_INLINE_FUNCTION
double dmr(int var, double x, double y){
    //Double Mach reflection at t = 0: post-shock state behind the line
    //x = 1/6 + y/tan(60 deg), undisturbed gas ahead (dmr.hpp).
    return dmr_prim(dmr_behind(x,y,0.0), var);
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
double rti(int var, double x, double y, double gm, double gy, ProblemParams pp){
    //Rayleigh-Taylor instability (single mode), ported from the spd
    //reference (initial_conditions_2d.RTI). Heavy fluid pp.d0 rests below
    //light fluid pp.d1 across the interface y = pp.radius, in hydrostatic
    //balance under the constant vertical acceleration gy (= cfg.g[y]); with
    //heavy fluid on the low side and gy>0 (buoyancy pointing up) the layer is
    //RT-unstable. A single-mode vertical velocity perturbation seeds it.
    double yc     = pp.radius;
    double rho_hi = pp.d0;   //heavy (lower) layer
    double rho_lo = pp.d1;   //light (upper) layer
    double P0     = pp.p0;   //reference pressure at the interface
    bool   top    = y > yc;
    if(var==_d_)
        return top ? rho_lo : rho_hi;
    else if(var==_vy_){
        //perturbation scaled by the local sound speed (spd's dv), single
        //mode cos(8 pi x): one wavelength across the x in [0,0.25] box
        double dv = sqrt(gm*(rho_hi*yc + 1.0)/rho_hi);
        return -pp.amp*dv*cos(8.0*PI*x);
    }
    else if(var==_p_){
        //hydrostatic: dP/dy = rho*gy, continuous across the interface
        return top ? (P0 + rho_lo*gy*y + (rho_hi-rho_lo)*gy*yc)
                   : (P0 + rho_hi*gy*y);
    }
    else
        return 0;
}

//Smooth-interface Kelvin-Helmholtz, Athena++ (Stone et al. 2020) eq. 26.
//
//The paper writes the tanh argument as |y - 0.25|/L, but that puts a single
//interface at y=0.25 and gives a density contrast of 1.5 with a velocity jump
//of 0.5 -- contradicting its own text ("a density contrast of two and a
//velocity jump of one") and leaving the state non-periodic across y. The
//intended argument is (|y| - 0.25)/L: interfaces at y = +-0.25, rho in
//[1,2], vx in [-0.5,0.5]. The perturbation wavelength is 0.5, so each
//interface carries two wavelengths across the unit box.
//
//Paper coordinates are [-0.5,0.5]^2; the code box is [0,LENGHT]^2, so y is
//shifted. The x shift is a whole number of periods of cos(4 pi x) and drops out.
KOKKOS_INLINE_FUNCTION
double kelvin_helmholtz(int var, double x, double y){
    const double Lsh = 0.01;   //shear layer thickness
    const double amp = 0.01;   //perturbation amplitude
    const double sig = 0.2;    //thickness of the perturbed layer
    double dy = fabs(y - 0.5*LENGHT) - 0.25;
    double s  = tanh(dy/Lsh);
    if(var==_d_)  return 1.5 - 0.5*s;
    if(var==_vx_) return 0.5*s;
    if(var==_vy_) return amp*cos(4*PI*x)*exp(-(dy*dy)/(sig*sig));
    if(var==_p_)  return 2.5;
    return 0;
}

//========================================================================================
// Ha et al. high-Mach astrophysical jet -- the PURE HYDRO stress test.
//
// Ha, Gardner, Gelb & Shu, J. Sci. Comput. 24, 29 (2005), as run in
// Rueda-Ramirez, Bolm, Kuzmin & Gassner 2023 (arXiv:2303.00374) section 3.5.
//
//   domain  [-0.5,0.5]^2, gamma = 5/3, NO magnetic field
//   ambient rho = 0.5, p = 0.4127, v = 0
//   inlet   on the LEFT face, |y| < 0.05: rho = 5, p = 0.4127, v_x = 800
//   Mach 2156.91 w.r.t. the JET sound speed, 682.08 w.r.t. the AMBIENT
//   periodic top/bottom; inflow/outflow left/right; t = 1e-3
//
// spd_K is origin-anchored, so the box is [0,1]^2 and the nozzle is centred on
// pp.cy (the box midpoint) rather than on y = 0.
//
// WHY THIS TEST, next to the Balsara 8.2 MHD jet: it is the same class of
// problem -- a hypersonic beam into a quiescent medium -- but UNMAGNETIZED, and
// RR23 publishes the actual numbers (their table 5: rho and p ranges at
// t = 1e-3) rather than only a figure. The Balsara density panel cannot be read
// quantitatively: through its own printed colourbar the undisturbed ambient
// comes out at rho = 0.010 where the stated IC fixes 0.14. So this problem is
// the one that can actually PASS or FAIL against a paper, and it separates
// "does spd_K do hypersonic jets" from "does spd_K do beta = 1e-4".
//
// Read table 5 with care before treating any single number as a target: across
// limiter and CFL, within ONE code at ONE resolution, their rho_min spans 19x
// (7.11e-04 to 1.33e-02) and they explain the mechanism -- less dissipation
// gives a lower rho_min, which raises the sound speed, which cuts dt, which
// cuts dissipation again. The cocoon minimum is the most scheme-sensitive
// number in the problem. rho_max is far steadier: 23.85-40.67, i.e. 4.8-8.1x
// the injected 5, against a gamma=5/3 single-shock bound of 4 -- the excess is
// stacked compressions at the working surface.
//
// pp.d0 ambient rho, pp.d1 jet rho, pp.p0 pressure (both), pp.v1 jet v_x,
// pp.radius nozzle half-width, pp.cy nozzle centre.

//Quiescent ambient EVERYWHERE. The beam enters only through the left boundary,
//exactly as in mhd_ic_jet: a uniform beam laid into the domain would be an
//exact steady solution and the run would do nothing.
KOKKOS_INLINE_FUNCTION
double ha_jet(int var, double x, double y, double z, ProblemParams pp){
    (void)x; (void)y; (void)z;
    if(var==_d_) return pp.d0;
    if(var==_p_) return pp.p0;
    return 0.0;
}

//Shock-cloud interaction, the adiabatic set-up of Pittard & Parkin (2016, MNRAS 457, 4470; 3D) after Klein, McKee &
//Colella (1994): a planar shock of Mach number pp.v1 runs along +x through an ambient medium at rest (pp.d0, pp.p0)
//into a cloud in pressure equilibrium with it. The cloud is soft-edged, eq. 18-19 of Pittard et al. (2009, MNRAS
//394, 1351):
//    rho(r) = rho_amb [psi + (1 - psi) eta],   eta = (1/2)[1 + (alpha-1)/(alpha+1)],
//    alpha  = exp{min[20, p1 ((r/r_c)^2 - 1)]},
//with psi set so that the central density is pp.d1 (the contrast chi = d1/d0) and p1 = pp.p1 the steepness of the
//edge (10 in both papers: an edge about a tenth of the radius wide). The cloud centre is (pp.cx, pp.cy, pp.cz), its
//radius pp.radius; the shock starts pp.amp radii upstream of the centre, with the Rankine-Hugoniot state behind it.
//In 2D the cloud is a cylinder, in 1D a slab. Boundaries: gradfree on every face (the upstream face then keeps
//feeding the post-shock state).
KOKKOS_INLINE_FUNCTION
double shock_cloud_density(double x, double y, double z, bool ay, bool az, ProblemParams pp){
    double q = (x-pp.cx)*(x-pp.cx);
    if(ay) q += (y-pp.cy)*(y-pp.cy);
    if(az) q += (z-pp.cz)*(z-pp.cz);
    q /= pp.radius*pp.radius;                                    //(r/r_c)^2
    const double a0   = exp(-pp.p1);
    const double eta0 = 0.5*(1.0 + (a0-1.0)/(a0+1.0));            //eta at the centre
    const double psi  = (pp.d1/pp.d0 - eta0)/(1.0 - eta0);
    const double al   = exp(fmin(20.0, pp.p1*(q-1.0)));
    const double eta  = 0.5*(1.0 + (al-1.0)/(al+1.0));
    return pp.d0*(psi + (1.0-psi)*eta);
}

KOKKOS_INLINE_FUNCTION
double shock_cloud(int var, double x, double y, double z, double gm, bool ay, bool az, ProblemParams pp){
    if(x < pp.cx - pp.amp*pp.radius){
        //behind the shock: Rankine-Hugoniot for Mach M into (d0, p0) at rest
        const double M2 = pp.v1*pp.v1;
        const double rho2 = pp.d0*(gm+1.0)*M2/((gm-1.0)*M2 + 2.0);
        if(var==_d_)  return rho2;
        if(var==_vx_) return pp.v1*sqrt(gm*pp.p0/pp.d0)*(1.0 - pp.d0/rho2);
        if(var==_p_)  return pp.p0*(2.0*gm*M2 - (gm-1.0))/(gm+1.0);
        return 0.0;
    }
    if(var==_d_) return shock_cloud_density(x,y,z,ay,az,pp);
    if(var==_p_) return pp.p0;
    return 0.0;
}

//The advected scalar of Pittard & Parkin (2016) that marks cloud material: kappa = rho/(chi rho_amb) within two cloud
//radii of the centre and zero beyond, so it is 1 at the centre and 1/chi at the edge of its support.
KOKKOS_INLINE_FUNCTION
double shock_cloud_kappa(double x, double y, double z, bool ay, bool az, ProblemParams pp){
    double r2 = (x-pp.cx)*(x-pp.cx);
    if(ay) r2 += (y-pp.cy)*(y-pp.cy);
    if(az) r2 += (z-pp.cz)*(z-pp.cz);
    if(r2 >= 4.0*pp.radius*pp.radius || x < pp.cx - pp.amp*pp.radius) return 0.0;
    return shock_cloud_density(x,y,z,ay,az,pp)/pp.d1;
}

KOKKOS_INLINE_FUNCTION
double euler_ic(int problem, int var, double x, double y, double z,
                double gm, double gy, bool ay, bool az, ProblemParams pp){
    switch(problem){
        case _ic_sine_wave_:        return sine_wave(var,x,y,z,pp);
        case _ic_sedov_:            return sedov_blast(var,x,y,z,gm,az,pp);
        case _ic_spherical_blast_:  return spherical_blast(var,x,y,z,az,pp);
        case _ic_square_:           return square_signal(var,x,y,z,ay,az,pp);
        case _ic_sod_:              return sod_shock_tube(var,x,y,z,pp);
        case _ic_shu_osher_:        return shu_osher(var,x,y,z,pp);
        //Athena++ smooth-interface KH (Stone et al. 2020); see overload without pp
        case _ic_kelvin_helmholtz_: return kelvin_helmholtz(var,x,y);
        case _ic_implosion_:        return implosion(var,x,y,z,pp);
        case _ic_woodward_colella_: return woodward_colella(var,x,y,z,pp);
        case _ic_dmr_:              return dmr(var,x,y);
        case _ic_rti_:              return rti(var,x,y,gm,gy,pp);
        case _ic_ha_jet_:           return ha_jet(var,x,y,z,pp);
        case _ic_shock_cloud_:      return shock_cloud(var,x,y,z,gm,ay,az,pp);
        case _ic_user_:             return user_ic(var,x,y,z,gm,ay,az,pp);
        default:                    return 0;
    }
}

//Initial CONCENTRATION of passive scalar n (row NVAR+n, define.hpp), chosen by problem/scalar (_sic_*). A problem
//that marks its own material (a cloud, a jet) should define the scalar here under its problem id instead.
KOKKOS_INLINE_FUNCTION
double scalar_ic(int sic, int n, int problem, double x, double y, double z,
                 double gm, double gy, bool ay, bool az, ProblemParams pp){
    if(problem == _ic_shock_cloud_ && n == 0) return shock_cloud_kappa(x,y,z,ay,az,pp);   //its own marker
    switch(sic){
        case _sic_uniform_: return 1.0;
        case _sic_sine_:    return 0.5 + 0.25*sin(2*PI*(x + y + z + 0.25*n));
        case _sic_blob_: {
            double r2 = (x-pp.cx)*(x-pp.cx);
            if(ay) r2 += (y-pp.cy)*(y-pp.cy);
            if(az) r2 += (z-pp.cz)*(z-pp.cz);
            return r2 < pp.radius*pp.radius ? 1.0 : 0.0;
        }
        case _sic_density_: return euler_ic(problem,_d_,x,y,z,gm,gy,ay,az,pp)/pp.d0;
        default:            return 0.0;
    }
}

KOKKOS_INLINE_FUNCTION
double initial_condition(int problem, int var, double x, double y, double z,
                         double gm, double gy, bool ay, bool az, ProblemParams pp, int sic){
    if(var < NVAR) return euler_ic(problem,var,x,y,z,gm,gy,ay,az,pp);
    return scalar_ic(sic,var-NVAR,problem,x,y,z,gm,gy,ay,az,pp);
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
    double gy = cfg.g[_y_];
    bool ay = cfg.active[_y_];
    bool az = cfg.active[_z_];
    ProblemParams pp = cfg.pp;
    const int sic = cfg.scalar_ic;
    //One device kernel on every backend (CLAUDE.md rule 5). The CUDA build used
    //to project on ONE host core instead -- nvar x (p+1)^ndim IC evaluations per
    //DoF -- which held a 3D initial-refine pass on the host for >15 min with the
    //GPU idle.
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
                    s = initial_condition(problem,var,x,y,z,gm,gy,ay,az,pp,sic);
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


