#pragma once
//The post-shock state of the shock-cloud problem (problem = shock_cloud; initial_conditions.cpp has the set-up):
//Rankine-Hugoniot for a shock of Mach number pp.v1 running into (pp.d0, pp.p0) at rest. One function feeds the
//initial condition behind the shock AND the prescribed-inflow boundary (mesh/x1_bc = inflow) on the low x face, so
//the two cannot drift apart. The inflow matters: with the zero-gradient face the 3D pilot's upstream flow ran away
//(rho 28 and p 13 000 against the post-shock 3.9 and 125 by 9 crushing times -- the copy fed its own noise back in,
//and the time step went with it). A supersonic inflow has every characteristic entering, so the state is simply
//prescribed. The passive scalars are zero behind the shock (nothing marked enters).
//Include after spd_k.hpp (define.hpp and global.hpp have no guards of their own).

KOKKOS_INLINE_FUNCTION
void shock_cloud_postshock(ProblemParams pp, double gm, double& rho2, double& vx2, double& p2){
    const double M2 = pp.v1*pp.v1;
    rho2 = pp.d0*(gm+1.0)*M2/((gm-1.0)*M2 + 2.0);
    vx2  = pp.v1*sqrt(gm*pp.p0/pp.d0)*(1.0 - pp.d0/rho2);
    p2   = pp.p0*(2.0*gm*M2 - (gm-1.0))/(gm+1.0);
}

//Conservative row `var` of the post-shock state (the arithmetic of conservatives() in hydro.cpp for a state with
//x velocity only); rows beyond the energy -- the passive scalars -- are zero.
KOKKOS_INLINE_FUNCTION
double shock_cloud_inflow_cons(int var, ProblemParams pp, double gm){
    double rho, vx, p;
    shock_cloud_postshock(pp, gm, rho, vx, p);
    if(var == _d_)  return rho;
    if(var == _vx_) return rho*vx;
    if(var == _e_)  return p/(gm-1.)+0.5*(vx*(rho*vx));
    return 0.0;
}
