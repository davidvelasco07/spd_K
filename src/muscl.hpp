//======================================================================
// Slope limiters for the MUSCL / FV reconstruction.
//
// All three take the two ONE-SIDED SLOPES (already divided by their own
// centre spacing -- the FV sub-grid is non-uniform, so a shared h breaks
// mirror symmetry at reflective walls) and return the HALF-INCREMENT used
// to reconstruct to a face:  W_face = W_cell +- limited_slope(...).
//
//   S_p, h_p : forward  (plus-side)  slope and its centre spacing
//   S_m, h_m : backward (minus-side) slope and its centre spacing
//   x_L, x_R : this cell's own faces; h_M = x_R - x_L is the cell size
//
// minmod is the reference implementation and the default: every existing
// golden encodes it. vanleer and moncen are ported from the references
// noted on each.
//======================================================================

//Ratio form, kept bit-for-bit as it was before the other limiters existed:
//r = S_m/S_p clipped to [0,1], times S_p. S_p == 0 gives inf/nan, which
//compares false in both clamps and falls through to 0*S_p = 0.
KOKKOS_INLINE_FUNCTION
double minmod(double S_L, double S_R, double x_L, double x_R){
    double ratio,slope;
    //Compute ratio between slopes SlopeR/SlopeL
    ratio = S_R/S_L;
    //limit the ratio to in (0,1)
    ratio = max(0.0,min(ratio,1.0));
    //mult_p_ly by SlopeL to get the limited slope at the cell center
    slope = ratio*S_L;
    //Compute SlopeC*dx/2                             
    return 0.5*slope*(x_R-x_L);
}

//van Leer's harmonic mean (FARGO3D / standard MUSCL form): 2 S_m S_p /
//(S_m + S_p) where the two one-sided slopes agree in sign, 0 otherwise.
//Smoother than minmod near extrema -- it approaches the central slope where
//the data are smooth instead of always taking the smaller side -- while
//still vanishing at a sign change, so it stays TVD.
KOKKOS_INLINE_FUNCTION
double vanleer_slope(double S_p, double S_m, double x_L, double x_R){
    double prod = S_m*S_p;
    double slope = (prod > 0.0) ? 2.0*prod/(S_m + S_p) : 0.0;
    return 0.5*slope*(x_R-x_L);
}

//Monotonized central (moncen), ported from Python spd
//(spd/finite_volume/muscl.py Slope_limiter.moncen). The geometry-weighted
//central slope, capped by twice each one-sided slope measured over the cell
//size, and zeroed at a sign change. Least diffusive of the three.
KOKKOS_INLINE_FUNCTION
double moncen_slope(double S_p, double S_m, double h_p, double h_m,
                    double x_L, double x_R){
    if(S_m*S_p < 0.0) return 0.0;
    double h_M = x_R - x_L;
    double dU_C = (h_m*S_m + h_p*S_p)/(h_m + h_p);
    double a = fabs(2.0*S_m*h_m/h_M);
    double b = fabs(2.0*S_p*h_p/h_M);
    double s = a < b ? a : b;
    double c = fabs(dU_C);
    s = s < c ? s : c;
    s = dU_C >= 0.0 ? s : -s;
    return 0.5*s*h_M;
}

//Runtime dispatch. `lim` is cfg.limiter, captured into a local before the
//kernel (device code cannot read the host cfg) and threaded down.
KOKKOS_INLINE_FUNCTION
double limited_slope(double S_p, double S_m, double h_p, double h_m,
                     double x_L, double x_R, int lim){
    if(lim==_lim_vanleer_) return vanleer_slope(S_p,S_m,x_L,x_R);
    if(lim==_lim_moncen_)  return moncen_slope(S_p,S_m,h_p,h_m,x_L,x_R);
    return minmod(S_p,S_m,x_L,x_R);
}
