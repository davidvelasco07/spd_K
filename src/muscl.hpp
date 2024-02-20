
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