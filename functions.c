#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "chydro.h"

void dbl_free(double **ptr, int nsize){
  /* 
      Called by python to free a 2D double array
  */ 
  
  int i;
  for (i=0;i<nsize;i++) free(ptr[i]);
  free(ptr);
} 

double fdMDot(double *daR, SYSTEM *SYS, PLANET *EARTH, double U[3][SYS->iNGrid]){
  // Calculate density and velocity at sonic point
  double dR, dRho, dV, dCs;
  long int i;
  
  // TODO: use trapz
  for (i=2;i<SYS->iNGrid-2;i++){
    dV = U[1][i]/U[0][i];
    dCs = sqrt(EARTH->dGamma*(EARTH->dGamma-1)*(U[2][i]/U[0][i] - 0.5*dV*dV));
    if (dV >= dCs){
      dRho = U[0][i]/(daR[i]*daR[i]);
      return 4*PI*pow(daR[i]*EARTH->dR0,2)*(dRho*EARTH->dN0*MH)*(dCs*sqrt(KBOLTZ*EARTH->dT0/MH));
    }
  }
  
  // Not supersonic
  return 0;
}

double fdRXUV(double *daR, SYSTEM *SYS, PLANET *EARTH, double U[3][SYS->iNGrid]){
  // Find R corresponding to XUV expectation value
  // TODO: use trapz  
  long int i;
  double dTau;
  double daRho[SYS->iNGrid], daRhoInt[SYS->iNGrid];
  daRhoInt[SYS->iNGrid-1] = 0;
  for (i=SYS->iNGrid-2;i>=0;i--){
    daRho[i] = U[0][i]/(daR[i]*daR[i]);  
    daRhoInt[i] = daRhoInt[i+1] + daRho[i]*(daR[i+1]-daR[i]);
    dTau = EARTH->dQA*daRhoInt[i];
    if (dTau>=1) return daR[i]*EARTH->dR0;
  }
  return 0;
}

double fdLiang(double x){
  // Flux limiter, Liang+ (2006)
  double foo;
  if (2*x < 1)
    foo = 2*x;
  else
    foo = 1;
  if (foo > 0)
    return foo;
  else
    return 0;
}

double fdMinMod(double x){
  // Minmod flux limiter
  double foo;
  if (x < 1)
    foo = x;
  else
    foo = 1;
  if (foo > 0)
    return foo;
  else
    return 0;
}

double fdG(double x, double Cr){
  // Liang+ (2006)
  double C;
  if (Cr<=0.5)
    C = Cr*(1-Cr);
  else
    C = 0.25;
  return 0.5*C*(1-fdMinMod(x));
}

double fdInitR(double i, SYSTEM *SYS){
  return pow(SYS->dRMax,pow(((double)i/(double)(SYS->iNGrid-1)),SYS->dGridPower));
}

//double fdInitRho(double dR, double dBeta){
  //return exp(dBeta*(-1+1/dR));
double fdInitRho(double dT0, double dMass, double dR0, double dR, double dN0) {
  double sound_speed_sq = KBOLTZ*dT0/MH;
  return MH*dN0*exp((-BIGG*dMass/sound_speed_sq)*(1.0/dR0-1.0/dR));
}

// can have an initial velocity
double fdInitV(double dR){
  return 0.01*(dR-1);
}

// starts with the same temperature?
double fdInitT(double dR){
  return 1.;
}

void fvLaxFriedrichs(double dDt, double *daR, SYSTEM *SYS, PLANET *EARTH, 
                     double U_CURR[3][SYS->iNGrid], double G_CURR[3][SYS->iNGrid], 
                     double Q_CURR[3][SYS->iNGrid], double U_NEXT[3][SYS->iNGrid]){
  // First order Lax-Friedrichs with numerical diffusion
  long int i;
  int k;
  double dDeltaR, dV, dT, dLambda;
  double dGNumericPlus, dGNumericMinus;
  
  for(i=2;i<SYS->iNGrid-2;i++){
    dDeltaR = (daR[i]-daR[i-1]);
    dV = U_CURR[1][i]/U_CURR[0][i];
    dT = (EARTH->dGamma-1)*(U_CURR[2][i]/U_CURR[0][i] - 0.5*dV*dV);
    dLambda = fabs(dV) + sqrt(EARTH->dGamma*dT);
    for(k=0;k<3;k++){
      dGNumericPlus = 0.5*(G_CURR[k][i+1]+G_CURR[k][i]) - 0.5*dLambda*(U_CURR[k][i+1]-U_CURR[k][i]);
      dGNumericMinus = 0.5*(G_CURR[k][i]+G_CURR[k][i-1]) - 0.5*dLambda*(U_CURR[k][i]-U_CURR[k][i-1]);        
      U_NEXT[k][i] = U_CURR[k][i] - dDt*((dGNumericPlus-dGNumericMinus)/dDeltaR - Q_CURR[k][i]);
    }
  }
}

double fdDeltaT(double *daR, SYSTEM *SYS, PLANET *EARTH, double U[3][SYS->iNGrid]){
  double dV, dT, foo, max;
  long int i;
  // Find delta t (Courant)
  for(i=0;i<SYS->iNGrid;i++){
    dV = U[1][i]/U[0][i];
    dT = (EARTH->dGamma-1)*(U[2][i]/U[0][i] - 0.5*dV*dV);
    foo = sqrt(EARTH->dGamma*dT) + fabs(dV);
    if (foo>max) max = foo;
  }
  return SYS->dEps*(daR[1]-daR[0])/max;
}

double fdDelta(SYSTEM *SYS, double U_CURR[3][SYS->iNGrid], double U_NEXT[3][SYS->iNGrid]){
  // Calculate change in U over one time step; use this to check for convergence
  double foo = 0;
  long int i;
  int k;
  for(k=0;k<3;k++){
    // Not including ghost cells
    for(i=2;i<SYS->iNGrid-2;i++){
      foo += pow((U_NEXT[k][i]-U_CURR[k][i])/U_CURR[k][i],2);
    }
  }
  return sqrt(foo/(3.*SYS->iNGrid));
}

void fvUpdateState(double *daR, double *daQ, SYSTEM *SYS, PLANET *EARTH, double U[3][SYS->iNGrid], double G[3][SYS->iNGrid], double Q[3][SYS->iNGrid]){
  // Calculate G and Q based on U
  double dV, dT, dRho;
  long int i;
  
  for(i=2;i<SYS->iNGrid-2;i++){
    dV = U[1][i]/U[0][i];
    dRho = U[0][i]/(daR[i]*daR[i]);
    dT = (EARTH->dGamma-1)*(U[2][i]/U[0][i] - 0.5*dV*dV);
    G[0][i] = U[1][i];
    G[1][i] = U[0][i]*(dV*dV+dT);
    G[2][i] = G[0][i]*(0.5*dV*dV + EARTH->dGamma/(EARTH->dGamma-1)*dT);
    Q[0][i] = 0;
    Q[1][i] = -dRho*(EARTH->dBeta-2*dT*daR[i]);  
    Q[2][i] = -dRho*dV*EARTH->dBeta + daQ[i]*daR[i]*daR[i];
  }
}

void fvFindHeatingRate(double *daR, double *daQ, SYSTEM *SYS, PLANET *EARTH, double U[3][SYS->iNGrid]){
  // Murray-Clay et al 2009 (4) and (5)  
  long int i;
  double dTau;
  double daRho[SYS->iNGrid], daRhoInt[SYS->iNGrid];
  daRhoInt[SYS->iNGrid-1] = 0;
  for (i=SYS->iNGrid-2;i>=0;i--){
    daRho[i] = U[0][i]/(daR[i]*daR[i]);  
    daRhoInt[i] = daRhoInt[i+1] + daRho[i]*(daR[i+1]-daR[i]);
    dTau = EARTH->dQA*daRhoInt[i];
    daQ[i] = EARTH->dQB*exp(-dTau)*daRho[i];
  }
  daQ[SYS->iNGrid-1] = 0;
}

void fvSetLims(double *daR, double *daQ, SYSTEM *SYS, PLANET *EARTH, double U[3][SYS->iNGrid], double G[3][SYS->iNGrid], double Q[3][SYS->iNGrid]){
  // Set ghost cell values
  double dRhoLower,dRhoUpper,dVLower,dVUpper,dTLower,dTUpper;
  long int i;
  
  // Boundary conditions
  dRhoLower = 1;
  dRhoUpper = U[0][SYS->iNGrid-3]/(daR[SYS->iNGrid-3]*daR[SYS->iNGrid-3]);
  dVLower = (U[1][2]/U[0][2]);
  dVUpper = U[1][SYS->iNGrid-3]/U[0][SYS->iNGrid-3];
  dTLower = 1;
  dTUpper = (EARTH->dGamma-1)*(U[2][SYS->iNGrid-3]/U[0][SYS->iNGrid-3] - 0.5*dVUpper*dVUpper);
  
  // Lower ghost cells
  for (i=0;i<2;i++){
    U[0][i] = dRhoLower*daR[i]*daR[i];
    U[1][i] = U[0][i]*dVLower;
    U[2][i] = U[0][i]*(0.5*dVLower*dVLower+dTLower/(EARTH->dGamma-1));
    G[0][i] = U[1][i];
    G[1][i] = U[0][i]*(dVLower*dVLower + dTLower);
    G[2][i] = G[0][i]*(0.5*dVLower*dVLower+EARTH->dGamma*dTLower/(EARTH->dGamma-1));
    Q[0][i] = 0;
    Q[1][i] = -dRhoLower*(EARTH->dBeta-2*dTLower*daR[i]);
    Q[2][i] = -dRhoLower*dVLower*EARTH->dBeta + daQ[i]*daR[i]*daR[i];  
  }
  
  // Upper ghost cells
  for (i=SYS->iNGrid-2;i<SYS->iNGrid;i++){
    U[0][i] = dRhoUpper*daR[i]*daR[i];
    U[1][i] = U[0][i]*dVUpper;
    U[2][i] = U[0][i]*(0.5*dVUpper*dVUpper+dTUpper/(EARTH->dGamma-1));
    G[0][i] = U[1][i];
    G[1][i] = U[0][i]*(dVUpper*dVUpper + dTUpper);
    G[2][i] = G[0][i]*(0.5*dVUpper*dVUpper+EARTH->dGamma*dTUpper/(EARTH->dGamma-1));
    Q[0][i] = 0;
    Q[1][i] = -dRhoUpper*(EARTH->dBeta-2*dTUpper*daR[i]);
    Q[2][i] = -dRhoUpper*dVUpper*EARTH->dBeta + daQ[i]*daR[i]*daR[i];  
  }
}

void fvOutput(OUTPUT *OUT, double *daR, double *daQ, SYSTEM *SYS, PLANET *EARTH, double U[3][SYS->iNGrid], double G[3][SYS->iNGrid], double Q[3][SYS->iNGrid]){

  long int i;
  double dV, dT, dRho;
  double dLambda;
  
  for(i=0;i<SYS->iNGrid;i++){
    
    // Log the primary variables
    OUT->dR[OUT->iOutputNum][i] = daR[i];
    OUT->dU0[OUT->iOutputNum][i] = U[0][i];
    OUT->dU1[OUT->iOutputNum][i] = U[1][i];
    OUT->dU2[OUT->iOutputNum][i] = U[2][i];
    OUT->dG0[OUT->iOutputNum][i] = G[0][i];
    OUT->dG1[OUT->iOutputNum][i] = G[1][i];
    OUT->dG2[OUT->iOutputNum][i] = G[2][i];
    OUT->dQ0[OUT->iOutputNum][i] = daQ[i];
    OUT->dQ1[OUT->iOutputNum][i] = Q[1][i];
    OUT->dQ2[OUT->iOutputNum][i] = Q[2][i];

    // Compute the velocity, temperature, and density
    dV = U[1][i]/U[0][i]; 
    dT = (EARTH->dGamma-1)*(U[2][i]/U[0][i] - 0.5*dV*dV);
    dRho = U[0][i]/(daR[i]*daR[i]);
    OUT->dT[OUT->iOutputNum][i] = dT;
    OUT->dV[OUT->iOutputNum][i] = dV;
    OUT->dRho[OUT->iOutputNum][i] = dRho;
    
    // Normalized numerical viscosity
    if (SYS->iIntegrator == INT_LAXFRIED) {
      dLambda = fabs(dV) + sqrt(EARTH->dGamma*dT); 
      if ((i > 0) && (i < SYS->iNGrid - 1)){
        OUT->dMVisc[OUT->iOutputNum][i] = 0.5 * dLambda * (U[0][i+1] - U[0][i-1]) / U[0][i];
        OUT->dPVisc[OUT->iOutputNum][i] = 0.5 * dLambda * (U[1][i+1] - U[1][i-1]) / U[1][i];
        OUT->dEVisc[OUT->iOutputNum][i] = 0.5 * dLambda * (U[2][i+1] - U[2][i-1]) / U[2][i];
      } else {
        OUT->dMVisc[OUT->iOutputNum][i] = 0.;
        OUT->dPVisc[OUT->iOutputNum][i] = 0.;
        OUT->dEVisc[OUT->iOutputNum][i] = 0.;
      }
    } else {
      OUT->dMVisc[OUT->iOutputNum][i] = 0.;
      OUT->dPVisc[OUT->iOutputNum][i] = 0.;
      OUT->dEVisc[OUT->iOutputNum][i] = 0.;
    }
  }
  
  OUT->iOutputNum++;

}