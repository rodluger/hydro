#define INT_LAXFRIED  0
#define PI            3.1415926535
#define MH            1.673e-24
#define KBOLTZ        1.381e-16
#define BIGG          6.672e-8
#define MEARTH        5.9742e27
#define REARTH        6.3781e8
#define XUVEARTH      4.64

typedef struct {
  double dMass;
  double dR0;
  double dT0;
  double dGamma;
  double dBeta;
  double dEpsXUV;
  double dSigXUV;
  double dFXUV;
  double dN0;
  double dQA;
  double dQB;
} PLANET;

typedef struct {
  int iNGrid;
  int iNRuns;
  int iIntegrator;
  int iOutputTime;
  double dRMax;
  double dEps;
  double dTol;
  double dGridPower;
} SYSTEM;

typedef struct {
  int iOutputNum;
  int iSize;
  double **dR;
  double **dV;
  double **dT;
  double **dRho;
  double **dU0;
  double **dU1;
  double **dU2;
  double **dG0;
  double **dG1;
  double **dG2;
  double **dQ0;
  double **dQ1;
  double **dQ2;
  double **dMVisc;
  double **dPVisc;
  double **dEVisc;
  double dMDot;
  double dRXUV;
} OUTPUT;

int fiHydro(SYSTEM SYS, PLANET EARTH, OUTPUT *OUT);
double fdMDot(double *daR, SYSTEM *SYS, PLANET *EARTH, double U[3][SYS->iNGrid]);
double fdRXUV(double *daR, SYSTEM *SYS, PLANET *EARTH, double U[3][SYS->iNGrid]);
double fdLiang(double x);
double fdMinMod(double x);
double fdG(double x, double Cr);
double fdInitR(double i, SYSTEM *SYS);
double fdInitRho(double dR, double dBeta);
double fdInitV(double dR);
double fdInitT(double dR);
void fvLaxFriedrichs(double dDt, double *daR, SYSTEM *SYS, PLANET *EARTH, double U_CURR[3][SYS->iNGrid], double G_CURR[3][SYS->iNGrid], double Q_CURR[3][SYS->iNGrid], double U_NEXT[3][SYS->iNGrid]);
double fdDeltaT(double *daR, SYSTEM *SYS, PLANET *EARTH, double U[3][SYS->iNGrid]);
double fdDelta(SYSTEM *SYS, double U_CURR[3][SYS->iNGrid], double U_NEXT[3][SYS->iNGrid]);
void fvUpdateState(double *daR, double *daQ, SYSTEM *SYS, PLANET *EARTH, double U[3][SYS->iNGrid], double G[3][SYS->iNGrid], double Q[3][SYS->iNGrid]);
void fvFindHeatingRate(double *daR, double *daQ, SYSTEM *SYS, PLANET *EARTH, double U[3][SYS->iNGrid]);
void fvSetLims(double *daR, double *daQ, SYSTEM *SYS, PLANET *EARTH, double U[3][SYS->iNGrid], double G[3][SYS->iNGrid], double Q[3][SYS->iNGrid]);
void fvOutput(OUTPUT *OUT, double *daR, double *daQ, SYSTEM *SYS, PLANET *EARTH, double U[3][SYS->iNGrid], double G[3][SYS->iNGrid], double Q[3][SYS->iNGrid]);
void dbl_free(double **ptr, int nsize);