#include <math.h>
#define JMAX 20 /*Maximum number of iterations in qsimp*/
#define FUNC(x,y,z) ((*func)(x,y,z))

double trapzd(double (*func)(double,double,double),double c1,double c2, double a, double b,int n){
  /* LUGER: From Numerical Recipes in C, Second Edition, page 137
  *  Trapezoidal rule, used by qsimp below to integrate the function func
  */
  
  double x,tnm,sum,del;
  static double s;
  int it,j;
  
  if (n==1){
    return (s=0.5*(b-a)*(FUNC(a,c1,c2)+FUNC(b,c1,c2)));
  } else {
    for (it=1,j=1;j<n-1;j++) it <<= 1;
    tnm=it;
    del=(b-a)/tnm;
    x=a+0.5*del;
    for (sum=0.0,j=1;j<=it;j++,x+=del) sum += FUNC(x,c1,c2);
    s=0.5*(s+(b-a)*sum/tnm);
    return s;
  }
}

double qsimp(double (*func)(double,double,double),double c1,double c2,double a, double b, double eps){
  /* LUGER: From Numerical Recipes in C, Second Edition, page 139
  *  Integrates the function func between a and b using Simpson's rule. Arguments
  *  c1 and c2 are arbitrary constants passed to the function being integrated.
  *  The routine terminates once the difference between successive solutions is less than eps.
  */

  double trapzd(double (*func)(double,double,double),double c1,double c2, double a, double b, int n);
  int j;
  double s,st,ost,os;
  
  ost=os=-1.0e30;
  for (j=1;j<=JMAX;j++) {
    st=trapzd(func,c1,c2,a,b,j);
    s=(4.0*st-ost)/3.0;
    if (fabs(s-os) < eps*fabs(os)) return s;
    os=s;
    ost=st;
  }
  
  // We failed!
  return NAN;
  
}