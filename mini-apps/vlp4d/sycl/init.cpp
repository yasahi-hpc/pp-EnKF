#include "init.hpp"

// Prototypes
void importFromFile(const char* f, Config& conf);
void print(Config& conf);
void initcase(Config& conf, RealView4D& fn);
void testcaseCYL02(Config& conf, RealView4D& fn);
void testcaseCYL05(Config& conf, RealView4D& fn);
void testcaseSLD10(Config& conf, RealView4D& fn);
void testcaseTSI20(Config& conf, RealView4D& fn);

void importFromFile(const char* f, Config& conf) {
  char idcase[8], tmp;
  Physics& phys = conf.phys_;
  Domain& dom = conf.dom_;
  FILE* stream = fopen(f, "r");

  if(stream == (FILE*)nullptr) {
    printf("import: error file not found\n");
    abort();
  }

  /*Physical parameters */
  phys.eps0_ = 1.; /* permittivity of free space */
  phys.echarge_ = 1.; /* charge of particle */
  phys.mass_ = 1; /* mass of particle */
  phys.omega02_ = 0; /* coefficient of applied field */
  phys.vbeam_ = 0.; /* beam velocity */
  phys.S_ = 1; /* lattice period */
  phys.psi_ = 0;

  for(int i = 0; i < DIMENSION; i++) {
    do
      tmp = fgetc(stream);
    while(tmp != ':');
    int nx_tmp;
    fscanf(stream, " %d\n", &nx_tmp);
    dom.nxmax_[i] = nx_tmp;
  }

  for(int i = 0; i < DIMENSION; i++) {
    do
      tmp = fgetc(stream);
    while(tmp != ':');
    fscanf(stream, " %le\n", &(dom.minPhy_[i]));
    do
      tmp = fgetc(stream);
    while(tmp != ':');
    fscanf(stream, " %le\n", &(dom.maxPhy_[i]));
  }

  do
    tmp = fgetc(stream);
  while(tmp != ':');
  fgets(idcase, 7, stream);

  do
    tmp = fgetc(stream);
  while(tmp != ':');
  fscanf(stream, " %le\n", &(dom.dt_));

  do
    tmp = fgetc(stream);
  while(tmp != ':');
  fscanf(stream, " %d\n", &(dom.nbiter_));
  
  do
    tmp = fgetc(stream);
  while(tmp != ':');
  fscanf(stream, " %d\n", &(dom.ifreq_));
   
  do
    tmp = fgetc(stream);
  while(tmp != ':');
  fscanf(stream, " %d\n", &(dom.fxvx_));

  dom.idcase_ = atoi(idcase);
  
  for(int i = 0; i < DIMENSION; i++) {
    dom.dx_[i] = (dom.maxPhy_[i] - dom.minPhy_[i]) / dom.nxmax_[i];
  }
  
  fclose(stream);
};

void print(Config& conf) {
  Domain dom = conf.dom_;

  printf("** Definition of mesh\n");
  printf("Number of points in  x with the coarse mesh : %d\n", dom.nxmax_[0]);
  printf("Number of points in  y with the coarse mesh : %d\n", dom.nxmax_[1]);
  printf("Number of points in Vx with the coarse mesh : %d\n", dom.nxmax_[2]);
  printf("Number of points in Vy with the coarse mesh : %d\n", dom.nxmax_[3]);
  
  printf("\n** Defintion of the geometry of the domain\n");
  printf("Minimal value of Ex : %lf\n", dom.minPhy_[0]);
  printf("Maximal value of Ex : %lf\n", dom.maxPhy_[0]);
  printf("Minimal value of Ey : %lf\n", dom.minPhy_[1]);
  printf("Maximal value of Ey : %lf\n", dom.maxPhy_[1]);
  
  printf("\nMinimal value of Vx : %lf\n", dom.minPhy_[2]);
  printf("Maximal value of Vx   : %lf\n", dom.maxPhy_[2]);
  printf("Minimal value of Vy   : %lf\n", dom.minPhy_[3]);
  printf("Maximal value of Vy   : %lf\n", dom.maxPhy_[3]);
  
  printf("\n** Considered test cases");
  printf("\n-10- Landau Damping");
  printf("\n-11- Landau Damping 2");
  printf("\n-20- Two beam instability");
  printf("\nNumber of the chosen test case : %d\n", dom.idcase_);
  
  printf("\n** Iterations in time and diagnostics\n");
  printf("Time step : %lf\n", dom.dt_);
  printf("Number of total iterations : %d\n", dom.nbiter_);
  printf("Frequency of diagnostics : %d\n", dom.ifreq_);
  
  printf("Diagnostics of fxvx : %d\n", dom.fxvx_);
}

void initcase(Config& conf, RealView4D &fn) {
  switch(conf.dom_.idcase_)
  {
    case 2:
      testcaseCYL02(conf, fn);
      break; // CYL02 ;
    case 5:
      testcaseCYL05(conf, fn);
      break; // CYL05 ;
    case 10:
      testcaseSLD10(conf, fn);
      break; // SLD10 ;
    case 20:
      testcaseTSI20(conf, fn);
      break; // TSI20 ;
    default:
      printf("Unknown test case !\n");
      abort();
      break;
  }
}

void testcaseCYL02(Config& conf, RealView4D &fn) {
  Domain dom = conf.dom_;
  auto [s_nxmax, s_nymax, s_nvxmax, s_nvymax] = dom.nxmax_;
  const double AMPLI = 4;
  const double PERIOD = 0.5 * M_PI;
  const double cc = 0.50 * (6. / 16.);
  const double rc = 0.50 * (4. / 16.);

  for(int ivy = 0; ivy < s_nvymax; ivy++) {
    for(int ivx = 0; ivx < s_nvxmax; ivx++) {
      double vx = dom.minPhy_[2] + ivx * dom.dx_[2];
      for(int iy = 0; iy < s_nymax; iy++) {
        for(int ix = 0; ix < s_nxmax; ix++) {
          double x = dom.minPhy_[0] + ix * dom.dx_[0];
          double xx = x;
          double vv = vx;
                                                           
          double hv = 0.0;
          double hx = 0.0;
                                                           
          if((vv <= cc + rc) && (vv >= cc - rc)) {
            hv = cos(PERIOD * ((vv - cc) / rc));
          }
          else if((vv <= -cc + rc) && (vv >= -cc - rc)) {
            hv = -cos(PERIOD * ((vv + cc) / rc));
          }

          if((xx <= cc + rc) && (xx >= cc - rc)) {
            hx = cos(PERIOD * ((xx - cc) / rc));
          }
          else if((xx <= -cc + rc) && (xx >= -cc - rc)) {
            hx = -cos(PERIOD * ((xx + cc) / rc));
          }
                                                           
          fn(ix, iy, ivx, ivy) = (AMPLI * hx * hv);
        }
      }
    }
  }
}

void testcaseCYL05(Config& conf, RealView4D &fn) {
  Domain dom = conf.dom_;
  auto [s_nxmax, s_nymax, s_nvxmax, s_nvymax] = dom.nxmax_;
  const double AMPLI = 4;
  const double PERIOD = 0.5 * M_PI;
  const double cc = 0.50 * (6. / 16.);
  const double rc = 0.50 * (4. / 16.);

  for(int ivy = 0; ivy < s_nvymax; ivy++) {
    for(int ivx = 0; ivx < s_nvxmax; ivx++) {
      double vy = dom.minPhy_[3] + ivy * dom.dx_[3];
      for(int iy = 0; iy < s_nymax; iy++) {
        double y = dom.minPhy_[1] + iy * dom.dx_[1];
        for(int ix = 0; ix < s_nxmax; ix++) {
          double xx = y;
          double vv = vy;
                                                           
          double hv = 0.0;
          double hx = 0.0;
                                                           
          if((vv <= cc + rc) && (vv >= cc - rc)) {
            hv = cos(PERIOD * ((vv - cc) / rc));
          }
          else if((vv <= -cc + rc) && (vv >= -cc - rc)) {
            hv = -cos(PERIOD * ((vv + cc) / rc));
          }

          if((xx <= cc + rc) && (xx >= cc - rc)) {
            hx = cos(PERIOD * ((xx - cc) / rc));
          }
          else if((xx <= -cc + rc) && (xx >= -cc - rc)) {
            hx = -cos(PERIOD * ((xx + cc) / rc));
          }
                                                           
          fn(ix, iy, ivx, ivy) = (AMPLI * hx * hv);
        }
      }
    }
  }
}

void testcaseSLD10(Config& conf, RealView4D &fn) {
  Domain dom = conf.dom_;
  auto [s_nxmax, s_nymax, s_nvxmax, s_nvymax] = dom.nxmax_;

  for(int ivy = 0; ivy < s_nvymax; ivy++) {
    for(int ivx = 0; ivx < s_nvxmax; ivx++) {
      double vy = dom.minPhy_[3] + ivy * dom.dx_[3];
      double vx = dom.minPhy_[2] + ivx * dom.dx_[2];
      for(int iy = 0; iy < s_nymax; iy++) {
        double y = dom.minPhy_[1] + iy * dom.dx_[1];
        for(int ix = 0; ix < s_nxmax; ix++) {
          double x = dom.minPhy_[0] + ix * dom.dx_[0];

          double sum = (vx * vx + vy * vy);
          fn(ix, iy, ivx, ivy) = (1. / (2 * M_PI)) * exp(-0.5 * (sum)) * (1 + 0.05 * (cos(0.5 * x) * cos(0.5 * y)));
        }
      }
    }
  }
}

void testcaseTSI20(Config& conf, RealView4D &fn) {
  Domain dom = conf.dom_;
  auto [s_nxmax, s_nymax, s_nvxmax, s_nvymax] = dom.nxmax_;

  double xi = 0.90;
  double alpha = 0.05;
  /*  eps=0.15;  */
  /* x et y sont definis sur [0,2*pi] */

  for(int ivy = 0; ivy < s_nvymax; ivy++) {
    for(int ivx = 0; ivx < s_nvxmax; ivx++) {
      double vy = dom.minPhy_[3] + ivy * dom.dx_[3];
      double vx = dom.minPhy_[2] + ivx * dom.dx_[2];
      for(int iy = 0; iy < s_nymax; iy++) {
        //double y = dom->minPhy_[1] + iy * dom->dx_[1];
        for(int ix = 0; ix < s_nxmax; ix++) {
          double x = dom.minPhy_[0] + ix * dom.dx_[0];

          double sum = (vx * vx + vy * vy);
          fn(ix, iy, ivx, ivy) = (1 + alpha * (cos(0.5 * x))) * 1 / (2 * M_PI) * ((2 - 2 * xi) / (3 - 2 * xi)) * (1 + .5 * vx * vx / (1 - xi)) * exp(-.5 * sum);
        }
      }
    }
  }
}

void init(const char* file,
          sycl::queue& q,
          Config& conf,
          RealView4D& fn,
          RealView4D& fnp1,
          std::unique_ptr<Efield>& ef,
          std::unique_ptr<Diags>& dg) {
  importFromFile(file, conf);
  print(conf);

  // Allocate 4D data structures
  auto [nx, ny, nvx, nvy] = conf.dom_.nxmax_;
  fn   = RealView4D(q, "fn", nx, ny, nvx, nvy);
  fnp1 = RealView4D(q, "fnp1", nx, ny, nvx, nvy);

  ef = std::make_unique<Efield>(q, conf);

  // allocate and initialize diagnostics data structures
  dg = std::make_unique<Diags>(q, conf);

  initcase(conf, fn);
  fn.updateDevice();
  fnp1.updateDevice();
}

void finalize(Config& conf, std::unique_ptr<Diags>& dg) {
  // Store diagnostics
  dg->save(conf);
}