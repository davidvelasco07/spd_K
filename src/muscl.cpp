#include "spd_k.hpp"

//void limit_slope(
//    FV_Solution dUc,
//    FV_Solution dU,
//    Vector faces,
//    int dim){
//    FV_for_cells(
//        if(dim==0)
//            dUc.Vector(0,var,k,j,i) = minmod(dU.Vector(0,var,k,j,i),dU.Vector(0,var,k,j,i+1),faces(i),faces(i+1));
//        else if(dim==1)
//            dUc.Vector(0,var,k,j,i) = minmod(dU.Vector(0,var,k,j,i),dU.Vector(0,var,k,j+1,j),faces(j),faces(j+1));
//        else if(dim==2)
//            dUc.Vector(0,var,k,j,i) = minmod(dU.Vector(0,var,k,j,i),dU.Vector(0,var,k+1,j,i),faces(k),faces(k+1));
//    );
//}
//
//void interpolate(
//    FV_Solution U,
//    FV_Solution dU,
//    FV_Solution dUt,
//    FV_Solution U_L,
//    FV_Solution U_R,
//    Vector faces,
//    int dim){
//    FV_for_cells(
//        double h;
//        double uR;
//        double uL;
//        if(dim==0)      h = faces(i+1)-faces(i);
//        else if(dim==1) h = faces(j+1)-faces(j);
//        else if(dim==2) h = faces(k+1)-faces(k);
//        uR = U.Vector(0,var,k,j,i) - dU.Vector(0,var,k,j,i) + dUt.Vector(0,var,k,j,i)*dt/h;
//        uL = U.Vector(0,var,k,j,i) + dU.Vector(0,var,k,j,i) + dUt.Vector(0,var,k,j,i)*dt/h;
//        if(dim==0){
//            U_R.Vector(0,var,k,j,i  ) = uR;
//            U_L.Vector(0,var,k,j,i+1) = uL;
//        }
//        else if(dim==1){
//            U_R.Vector(0,var,k,j  ,i) = uR;
//            U_L.Vector(0,var,k,j+1,i) = uL;
//        }
//        else if(dim==2){
//            U_R.Vector(0,var,k  ,j,i) = uR;
//            U_L.Vector(0,var,k+1,j,i) = uL;
//        }
//    );
//}


