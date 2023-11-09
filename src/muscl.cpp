#include "spd_k.hpp"

KOKKOS_INLINE_FUNCTION
double minmod(double S_L, double S_R, double x_L, double x_R){
    double ratio,slope;
    //Compute ratio between slopes SlopeR/SlopeL
    ratio = S_R/S_L;
    //limit the ratio to in (0,1)
    ratio = max(0.0,min(ratio,1.0));
    //multiply by SlopeL to get the limited slope at the cell center
    slope = ratio*S_L;
    //Compute SlopeC*dx/2                             
    return 0.5*slope*(x_R-x_L);
}

template <typedef T>
void derivative(
    T U,
    T dU,
    Vector centers,
    int dim){
    FV_for_cells(
        double h;
        double du;
        if(dim==0){
            h = centers(i+1)-centers(i);
            du = (U.Vector(var,k,j,i+1) - U.Vector(var,k,j,i))/h;
        }
        else if(dim==1){
            h = centers(j+1)-centers(j);
            du = (U.Vector(var,k,j+1,i) - U.Vector(var,k,j,i))/h;
        }
        else if(dim==2){
            h = centers(k+1)-centers(k);
            du = (U.Vector(var,k+1,j,i) - U.Vector(var,k,j,i))/h;
        }
        dU.Vector(var,k,j,i) = du;
    );
}

template <typedef T>
void limit_slope(
    T dUc,
    T dU,
    Vector faces,
    int dim){
    FV_for_cells(
        if(dim==0)
            dUc.Vector(var,k,j,i) = minmod(dU.Vector(var,k,j,i),dU.Vector(var,k,j,i+1),faces(i),faces(i+1));
        else if(dim==1)
            dUc.Vector(var,k,j,i) = minmod(dU.Vector(var,k,j,i),dU.Vector(var,k,j+1,j),faces(j),faces(j+1));
        else if(dim==2)
            dUc.Vector(var,k,j,i) = minmod(dU.Vector(var,k,j,i),dU.Vector(var,k+1,j,i),faces(k),faces(k+1));
    );
}

template <typedef T>   
void interpolate(
    T U,
    T dU,
    T dUt,
    T U_L,
    T U_R,
    Vector faces,
    int dim){
    FV_for_cells(
        double h;
        double uR;
        double uL;
        if(dim==0)      h = faces(i+1)-faces(i);
        else if(dim==1) h = faces(j+1)-faces(j);
        else if(dim==2) h = faces(k+1)-faces(k);
        uR = U.Vector(var,k,j,i) - dU.Vector(var,k,j,i) + dUt.Vector(var,k,j,i)*dt/h;
        uL = U.Vector(var,k,j,i) + dU.Vector(var,k,j,i) + dUt.Vector(var,k,j,i)*dt/h;
        if(dim==0){
            U_R.Vector(var,k,j,i  ) = uR;
            U_L.Vector(var,k,j,i+1) = uL;
        }
        else if(dim==1){
            U_R.Vector(var,k,j  ,i) = uR;
            U_L.Vector(var,k,j+1,i) = uL;
        }
        else if(dim==2){
            U_R.Vector(var,k  ,j,i) = uR;
            U_L.Vector(var,k+1,j,i) = uL;
        }
    );
}

template <typedef T>
void fv_update_solution(
    T U_new,
    T U_old,
    #if X
    T F_x,
    Vector faces_x,
    #endif
    #if Y
    T F_y,
    Vector faces_y,
    #endif
    #if Z
    T F_z,
    Vector faces_z,
    #endif
    double dt
    ){
    FV_for_active_cells(
        double h;
        #ifdef X
        h  = faces_x(i+1)-faces_x(i);
        F += (F_x.Vector(var,k,j,i+1)-F_x.Vector(var,k,j,i))/h;
        #endif
        #ifdef Y
        h  = faces_y(j+1)-faces_y(j);
        F += (F_y.Vector(var,k,j+1,i)-F_y.Vector(var,k,j,i))/h;
        #endif
        #ifdef Z
        h  = faces_z(k+1)-faces_z(k);
        F += (F_z.Vector(var,k+1,j,i)-F_z.Vector(var,k,j,i))/h;
        #endif
        U_new.Vector(var,k,j,i) = U_old.Vector(var,k,j,i) - dt*F;
    );
}


