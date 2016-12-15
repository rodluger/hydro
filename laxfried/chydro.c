#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "chydro.h"

// IDEAS:
// 1. Find the heating rate, then re-update the initial state?
// 2. Use trapz to integrate and find heating rate
// 3. Erkaev+2013 states that sigxuv = 5x higher

int fiHydro(SYSTEM SYS, PLANET EARTH, OUTPUT *OUT){
  
  // Calculate some things
  EARTH.dBeta = (BIGG*MH*EARTH.dMass/(EARTH.dR0*KBOLTZ*EARTH.dT0));
  EARTH.dQA = EARTH.dSigXUV*EARTH.dN0*EARTH.dR0;
  EARTH.dQB = (EARTH.dEpsXUV*EARTH.dFXUV*EARTH.dSigXUV*EARTH.dR0/(MH*pow(KBOLTZ*EARTH.dT0/MH,1.5)));

  // State variables
  double U_CURR[3][SYS.iNGrid], G_CURR[3][SYS.iNGrid], Q_CURR[3][SYS.iNGrid];
  double U_HALF[3][SYS.iNGrid], G_HALF[3][SYS.iNGrid], Q_HALF[3][SYS.iNGrid];
  double U_NEXT[3][SYS.iNGrid], G_NEXT[3][SYS.iNGrid], Q_NEXT[3][SYS.iNGrid];
  double daRho[SYS.iNGrid], daR[SYS.iNGrid], daV[SYS.iNGrid], daT[SYS.iNGrid], daQ[SYS.iNGrid];
  double dDt, dDelta;
      
  // Other variables
  long int i;
  long int ti;
  int k;
  double dProg;
  double dTime = 0;
  int bConverged = 0;
  
  // Initialize state variables
  for(i=0;i<SYS.iNGrid;i++){
    daR[i] = fdInitR(i,&SYS);
    daRho[i] = fdInitRho(daR[i],EARTH.dBeta);
    daV[i] = fdInitV(daR[i]);
    daT[i] = fdInitT(daR[i]);
    daQ[i] = 0;
    U_CURR[0][i] = daRho[i]*daR[i]*daR[i];
    U_CURR[1][i] = daRho[i]*daR[i]*daR[i]*daV[i];
    U_CURR[2][i] = daRho[i]*daR[i]*daR[i]*(0.5*daV[i]*daV[i]+daT[i]/(EARTH.dGamma-1));
    G_CURR[0][i] = U_CURR[1][i];
    G_CURR[1][i] = U_CURR[0][i]*(daV[i]*daV[i]+daT[i]);
    G_CURR[2][i] = U_CURR[1][i]*(0.5*daV[i]*daV[i] + EARTH.dGamma/(EARTH.dGamma-1)*daT[i]);
    Q_CURR[0][i] = 0;
    Q_CURR[1][i] = -daRho[i]*(EARTH.dBeta-2*daT[i]*daR[i]);  
    Q_CURR[2][i] = -daRho[i]*daV[i]*EARTH.dBeta + daQ[i]*daR[i]*daR[i];
  }
  fvFindHeatingRate(daR, daQ, &SYS, &EARTH, U_CURR);
  dDt = SYS.dEps*(daR[1]-daR[0])/(sqrt(EARTH.dGamma) + fabs(daV[SYS.iNGrid-1]));
  fvSetLims(daR, daQ, &SYS, &EARTH, U_CURR, G_CURR, Q_CURR);
  
  // Initialize output arrays
  OUT->iSize = (int) ((1 + ceil((double) SYS.iNRuns / (double) SYS.iOutputTime)));
  OUT->iOutputNum = 0;
  OUT->dR = malloc(OUT->iSize * sizeof(double*));
  OUT->dV = malloc(OUT->iSize * sizeof(double*));
  OUT->dT = malloc(OUT->iSize * sizeof(double*));
  OUT->dRho = malloc(OUT->iSize * sizeof(double*));
  OUT->dU0 = malloc(OUT->iSize * sizeof(double*));
  OUT->dU1 = malloc(OUT->iSize * sizeof(double*));
  OUT->dU2 = malloc(OUT->iSize * sizeof(double*));
  OUT->dG0 = malloc(OUT->iSize * sizeof(double*));
  OUT->dG1 = malloc(OUT->iSize * sizeof(double*));
  OUT->dG2 = malloc(OUT->iSize * sizeof(double*));
  OUT->dQ0 = malloc(OUT->iSize * sizeof(double*));
  OUT->dQ1 = malloc(OUT->iSize * sizeof(double*));
  OUT->dQ2 = malloc(OUT->iSize * sizeof(double*));
  OUT->dMVisc = malloc(OUT->iSize * sizeof(double*));
  OUT->dPVisc = malloc(OUT->iSize * sizeof(double*));
  OUT->dEVisc = malloc(OUT->iSize * sizeof(double*));
  for (i=0;i<OUT->iSize;i++) {
    OUT->dR[i] = malloc(SYS.iNGrid * sizeof(double));
    OUT->dV[i] = malloc(SYS.iNGrid * sizeof(double));
    OUT->dT[i] = malloc(SYS.iNGrid * sizeof(double));
    OUT->dRho[i] = malloc(SYS.iNGrid * sizeof(double));
    OUT->dU0[i] = malloc(SYS.iNGrid * sizeof(double));
    OUT->dU1[i] = malloc(SYS.iNGrid * sizeof(double));
    OUT->dU2[i] = malloc(SYS.iNGrid * sizeof(double));
    OUT->dG0[i] = malloc(SYS.iNGrid * sizeof(double));
    OUT->dG1[i] = malloc(SYS.iNGrid * sizeof(double));
    OUT->dG2[i] = malloc(SYS.iNGrid * sizeof(double));
    OUT->dQ0[i] = malloc(SYS.iNGrid * sizeof(double));
    OUT->dQ1[i] = malloc(SYS.iNGrid * sizeof(double));
    OUT->dQ2[i] = malloc(SYS.iNGrid * sizeof(double));
    OUT->dMVisc[i] = malloc(SYS.iNGrid * sizeof(double));
    OUT->dPVisc[i] = malloc(SYS.iNGrid * sizeof(double));
    OUT->dEVisc[i] = malloc(SYS.iNGrid * sizeof(double));
  }
  fvOutput(OUT, daR, daQ, &SYS, &EARTH, U_CURR, G_CURR, Q_CURR);
  
  // Begin main loop
  for(ti=1;ti<SYS.iNRuns;ti++){
    
    // Take one step
    if (SYS.iIntegrator == INT_LAXFRIED) {
      fvLaxFriedrichs(dDt, daR, &SYS, &EARTH, U_CURR, G_CURR, Q_CURR, U_NEXT);
    } else {
      fprintf(stderr,"Invalid integrator selected.\n");
      exit(1);
    }
    
    // Check for convergence every 100 runs
    if (ti % (int)((double)SYS.iNRuns / 100.) == 0){
      dDelta = fdDelta(&SYS, U_CURR, U_NEXT);
    }
    
    // Update U
    for(i=2;i<SYS.iNGrid-2;i++){
      for(k=0;k<3;k++)
        U_CURR[k][i] = U_NEXT[k][i];
    }
    
    // Update everything else
    fvFindHeatingRate(daR, daQ, &SYS, &EARTH, U_CURR);
    fvUpdateState(daR, daQ, &SYS, &EARTH, U_CURR, G_CURR, Q_CURR);
    fvSetLims(daR, daQ, &SYS, &EARTH, U_CURR, G_CURR, Q_CURR);
    dDt = fdDeltaT(daR, &SYS, &EARTH, U_CURR);
    dTime += dDt;
    
    // Progress
    if (ti % (int)((double)SYS.iNRuns/100.) == 0){
      dProg = 100*(double)ti/SYS.iNRuns;
      fprintf(stderr,"Completed %.0f%%...    Delta = %.6e  \r", dProg, dDelta);
      if (dDelta < SYS.dTol) {
        bConverged = 1;
        break;
      }
    }
    
    // Output?
    if (ti % SYS.iOutputTime == 0){
      fvOutput(OUT, daR, daQ, &SYS, &EARTH, U_CURR, G_CURR, Q_CURR);
    }
  }
  
  // Output
  fvOutput(OUT, daR, daQ, &SYS, &EARTH, U_CURR, G_CURR, Q_CURR);
  
  // Check for convergence and calculate escape rate
  if (bConverged) {
    fprintf(stderr,"Converged!                              \n");
    OUT->dMDot = fdMDot(daR, &SYS, &EARTH, U_CURR);
    OUT->dRXUV = fdRXUV(daR, &SYS, &EARTH, U_CURR);
  }
  else {
    fprintf(stderr,"Run did not converge.                   \n");
    OUT->dMDot = 0;
    OUT->dRXUV = 0;
  }
  
  return bConverged;
}