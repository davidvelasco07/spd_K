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
//The default below is a Gaussian density pulse advected diagonally.

KOKKOS_INLINE_FUNCTION
double user_ic(int var, double x, double y, double z, double gm,
               bool ay, bool az, ProblemParams pp){
    if(var==0){
        double r2 = pow(x-pp.radius,2);
        if(ay) r2 += pow(y-pp.radius,2);
        if(az) r2 += pow(z-pp.radius,2);
        return pp.d0 + pp.amp*exp(-r2/(2*pp.sigma*pp.sigma));
    }
    else if(var==_vx_) return pp.v1;
    else if(var==_vy_) return ay ? pp.v1 : 0.0;
    else if(var==_vz_) return az ? pp.v1 : 0.0;
    else if(var==_p_)  return pp.p0;
    else return 0;
}
