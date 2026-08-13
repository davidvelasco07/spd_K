#pragma once

//User-defined initial condition, selected at runtime with problem=user.
//
//Edit this function and rebuild. It must return PRIMITIVE variables:
//  var 0    -> density
//  var _vx_ -> x-velocity      (_vy_, _vz_ likewise)
//  var _p_  -> pressure
//for the point (x,y,z). Inactive directions have y=0 / z=0; ay/az say
//whether y/z are active. The solver integrates the returned point values
//with Gauss-Legendre quadrature, so discontinuous profiles are fine.
//
//All <problem> block parameters are available through pp (amp, v1..v3,
//d0, d1, p0, p1, radius, sigma, dir), so simple variations of the IC can
//be explored from the input file without recompiling, e.g.:
//    ./spd_K -i inputs/user.athinput problem/amp=0.3 problem/sigma=0.2
//
//The default below is a Gaussian pulse in the transverse velocity (v_y) that
//varies only in x, carried by a uniform flow v1. With uniform density and
//pressure this is an exact shear layer: it neither compresses nor generates
//sound, so it evolves by pure advection-diffusion,
//    d(v_y)/dt + v1 d(v_y)/dx = nu d^2(v_y)/dx^2 .
//With nu=0 (the default) the pulse simply advects and returns to its start at
//t = L/v1. Enabling viscosity at runtime (hydro/nu>0 in the input file) makes
//it spread exactly as sigma^2(t) = sigma0^2 + 2*nu*t with peak amp*sigma0/
//sigma(t), which makes it a clean verification of the viscous operator.

KOKKOS_INLINE_FUNCTION
double user_ic(int var, double x, double y, double z, double gm,
               bool ay, bool az, ProblemParams pp){
    double dx = x - pp.radius;
    double pulse = pp.amp*exp(-dx*dx/(2*pp.sigma*pp.sigma));
    if(var==0)         return pp.d0;                 //uniform density
    else if(var==_vx_) return pp.v1;                 //uniform advection in x
    else if(var==_vy_) return pulse;                 //transverse velocity shear pulse
    else if(var==_vz_) return 0.0;
    else if(var==_p_)  return pp.p0;
    else return 0;
}
