#pragma once
//Double Mach reflection (Woodward & Colella 1984): a Mach-10 shock in gamma=1.4
//air hits a reflecting wall that begins at x = 1/6, the shock inclined at 60
//degrees to the wall. Shared by the initial condition (t = 0) and the
//`doublemach` boundary type, which prescribes the post-shock state on the left
//boundary and on the bottom wall for x < 1/6, a reflecting wall on the rest of
//the bottom, the exact moving shock on the top boundary, and outflow on the
//right. Domain [0,4] x [0,1], compare at t = 0.2 (Paper I sec 4.3.4; Athena++
//sec 3.4.2). Constants and states match spd/initial_conditions_2d.py.
#include "define.hpp"
#define DMR_XC     (1.0/6.0)
#define DMR_TAN60  1.7320508075688772
#define DMR_SIN60  0.8660254037844386
#define DMR_SPEED  10.0

//x-position of the shock front at height y and time t.
KOKKOS_INLINE_FUNCTION double dmr_shock_x(double y, double t){
    return DMR_XC + y/DMR_TAN60 + DMR_SPEED*t/DMR_SIN60;
}
KOKKOS_INLINE_FUNCTION bool dmr_behind(double x, double y, double t){
    return x < dmr_shock_x(y, t);
}
//Primitive state: post-shock (rho 8, v = 8.25 at -30 degrees, p 116.5) or the
//undisturbed gas (rho 1.4 at rest, p 1).
KOKKOS_INLINE_FUNCTION double dmr_prim(bool post, int var){
    if(var==0)         return post ? 8.0 : 1.4;
    else if(var==_vx_) return post ? 8.25*DMR_SIN60 : 0.0;   //8.25 cos 30 = 7.1447
    else if(var==_vy_) return post ? -8.25*0.5 : 0.0;        //-8.25 sin 30 = -4.125
    else if(var==_p_)  return post ? 116.5 : 1.0;
    else return 0.0;
}
//The same state as the conserved vector the ghost arrays carry.
KOKKOS_INLINE_FUNCTION double dmr_cons(bool post, int var, double gm){
    const double rho = dmr_prim(post,0), vx = dmr_prim(post,_vx_), vy = dmr_prim(post,_vy_);
    const double p = dmr_prim(post,_p_);
    if(var==0)         return rho;
    else if(var==_vx_) return rho*vx;
    else if(var==_vy_) return rho*vy;
    else if(var==_vz_) return 0.0;
    else if(var==_e_)  return p/(gm-1.0) + 0.5*rho*(vx*vx+vy*vy);
    else return 0.0;
}
