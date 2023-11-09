#include "spd_k.hpp"

double lagrange(double *x, double y, int i, int n){
    int j;
    double lag=1.;
    for(j=0;j<n;j++){
        if(j!=i)
            lag *= (y-x[j])/(x[i]-x[j]);
    }
    return lag;
}

void lagrange_matrix(Matrix a_to_b, double *x_a, double *x_b, int n_a, int n_b){
  for(int j=0;j<n_b;j++){
        for(int i=0;i<n_a;i++){
            a_to_b(j,i)=lagrange(x_a,x_b[j],i,n_a);
        }
    }
}

double lagrange_prime(double *x, double y, int i, int n){
    int j,k;
    double lag,lagp=0;
    for(k=0;k<n;k++){
      if(k!=i){
        lag=1;
        for(j=0;j<n;j++){
            if(j!=i && j!=k){
              lag *= (y-x[j])/(x[i]-x[j]);
            }
        }
        lagp+=lag/(x[i]-x[k]);
      }
    }
    return lagp;
}

void lagrange_prime_matrix(Matrix da_to_b, double *x_a, double *x_b, int n_a, int n_b){
  for(int j=0;j<n_b;j++){
        for(int i=0;i<n_a;i++){
            da_to_b(j,i)=lagrange_prime(x_a,x_b[j],i,n_a);
        }
    }
}

void gauss_legendre(double xi, double xf, int n, double *x, double *w){
    int i,j,m;
    double p1, p2, p3, pp, xl, xm, z, z1;
    double eps=3E-14;
    m = (n+1)/2;
    xm = 0.5*(xf+xi);
    xl = 0.5*(xf-xi);
    if(n>0){
      for(i=0;i<m;i++){
          z = cos(3.141592654*(i+0.75)/(n+0.5));
          z1 = 0.0;
          while(abs(z-z1) > eps){
              p1 = 1.0;
              p2 = 0.0;
              for(j=0;j<n;j++){
                  p3 = p2;
                  p2 = p1;
                  p1 = ((2.0*j+1)*z*p2-j*p3)/(j+1);
              }
              pp = n*(z*p1-p2)/(z*z-1.0);
              z1 = z;
              z = z1 - p1/pp;   
          }
          x[i] = xm - xl*z;
          x[n-i-1] = xm + xl*z;
          w[i] = (2.0*xl)/((1.0-z*z)*pp*pp);
          w[n-i-1] = w[i];
      }
    }
    else{
      x[0]=0.5;
      w[0]=1;
    }
}

void flux_points(double *x_fp, double *x, int n){
  x_fp[0]=0.0;
  for(int i=1;i<n+1;i++){
    x_fp[i] = x[i-1];
  }
  x_fp[n+1]=1.0;
}

void solution_points(double *x_sp, int n){
  for(int i=0;i<=n;i++){
    x_sp[i] = 0.5*(1.0 - cos((2*i + 1)/(2.0*(n+1))*PI));   
  }
}

void ader_matrix(Matrix ader, Vector x_t, Vector w_t, int n){
    for(int j=0;j<n;j++){
        for(int i=0;i<n;i++){
          ader(j,i)=lagrange(x_t.data(),1,i,n)*lagrange(x_t.data(),1,j,n)-lagrange_prime(x_t.data(),x_t(i),j,n)*w_t(i);
        }
    }
}

void integral_matrix(Matrix sp_to_cv, double *x_fp, double *x_sp, int n_cv, int n_sp){
    double integral;
    int p = n_sp-1;
    double *x = malloc_host<double>(p);
    double *w = malloc_host<double>(p);
    gauss_legendre(0.0, 1.0, p, x, w);
    for(int k=0;k<n_cv;k++){
        if(p>0)
            gauss_legendre(x_fp[k], x_fp[k+1], p, x, w);
        for(int j=0;j<n_sp;j++){
            if(p>0){
                integral=0.0;
                for(int i=0;i<p;i++){
                    integral+=lagrange(x_sp,x[i],j,p+1)*w[i];
                }
            }
            else
                integral = 1.0;
            sp_to_cv(k,j)=integral/(x_fp[k+1]-x_fp[k]);
        }
    }
}

void inverse(Matrix A, Matrix C, int n){
  double coeff;
  double *b = malloc_host<double>(n);
  double *d = malloc_host<double>(n);
  double *x = malloc_host<double>(n);

  double *B = malloc_host<double>(n*n);
  double *L = malloc_host<double>(n*n);
  double *U = malloc_host<double>(n*n);
  int i,j,k;
  for(j=0; j<n; j++){
    b[j]=0.0;
    for(i=0; i<n; i++){
      B[i+j*n]=A(j,i);
      U[i+j*n]=0;
      L[i+j*n]=0;
    }
  }
  //Step 1: forward elimination
  for(k=0; k<(n-1); k++){
    for(i=k+1; i<n; i++){
      coeff = B[k+i*n]/B[k+k*n];
      L[k+i*n] = coeff;
      for(j=k+1; j<n; j++){
        B[j+i*n] -= coeff*B[j+k*n];
      }
    }
  }
  //Step 2: prepare L and U matrices 
  //L matrix is a matrix of the elimination coefficient
  // + the diagonal elements are 1.0
  for(i=0;i<n;i++){
    L[i+i*n]=1.0;
  }
  //U matrix is the upper triangular part of A
  for(j=0; j<n; j++){
    for(i=0; i<=j; i++){
      U[j+i*n] = B[j+i*n];
    }
  }
  //Step 3: compute columns of the inverse matrix C
  for(k=0;k<n;k++){
    b[k]=1.0;
    d[0]=b[0];
    //Step 3a: Solve Ld=b using the forward substitution
    for(i=1;i<n;i++){
      d[i]=b[i];
      for(j=0;j<i;j++){
        d[i] -= L[j+i*n]*d[j];
      }
    }
    //Step 3b: Solve Ux=d using the back substitution
    x[n-1]=d[n-1]/U[n*n-1];
    for(i=n-2;i>=0;i--){
      x[i]=d[i];
      for(j=n-1;j>=i+1;j--){
        x[i]=x[i]-U[j+i*n]*x[j];
      }
      x[i] = x[i]/U[i+i*n];
    }
    //Step 3c: fill the solutions x(n) into column k of C
    for(i=0;i<n;i++){
      C(i,k) = x[i];
    }
    b[k]=0.0;
  }
  free(B);
  free(L);
  free(U);
  free(x);
  free(b);
  free(d);
}
