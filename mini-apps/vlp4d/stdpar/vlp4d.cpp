/*
 * @brief The vlp4d code solves Vlasov-Poisson equations in 4D (2d space, 2d velocity). 
 *        From the numerical point of view, vlp4d is based on a semi-lagrangian scheme. 
 *        Vlasov solver is typically based on a directional Strang splitting. 
 *        The Poisson equation is treated with 2D Fourier transforms. 
 *        For the sake of simplicity, all directions are, for the moment, handled with periodic boundary conditions.
 *        The Vlasov solver is based on advection's operators:
 *
 *        1D advection along x (Dt/2)
 *        1D advection along y (Dt/2)
 *        Poisson solver -> compute electric fields Ex and E
 *        1D advection along vx (Dt)
 *        1D advection along vy (Dt)
 *        1D advection along x (Dt/2)
 *        1D advection along y (Dt/2)
 *
 *        Interpolation operator within advection is Lagrange polynomial of order 5, 7 depending on a compilation flag (order 5 by default).
 *
 *  @author
 *  @url    https://gitlab.maisondelasimulation.fr/GyselaX/vlp4d/tree/master
 */

#include "../config.hpp"
#include "../timer.hpp"
#include "types.hpp"
#include "efield.hpp"
#include "field.hpp"
#include "diags.hpp"
#include "init.hpp"
#include "timestep.hpp"
#include <cstdio>

int main (int argc, char* argv[]) {
  Config conf;
  RealView4D fn, fnp1;

  Efield *ef = nullptr;
  Diags *dg = nullptr;

  std::vector<Timer*> timers;
  defineTimers(timers);

  // Initialization
  if(argc == 2) {
    // A file is given in parameter
    printf("reading input file %s\n", argv[1]);
    init(argv[1], conf, fn, fnp1, &ef, &dg);
  }
  else {
    printf("argc != 2, reading 'data.dat' by default\n");
    init("data.dat", conf, fn, fnp1, &ef, &dg);
  }
  int iter = 0;

  timers[Total]->begin();
  field_rho(conf, fn, ef);
  field_poisson(conf, ef);
  dg->compute(conf, ef, iter);
  if(conf.dom_.fxvx_) Advection::print_fxvx(conf, fn, iter);

  while(iter < conf.dom_.nbiter_) {
    timers[MainLoop]->begin();
    printf("iter %d\n", iter);

    iter++;
    onetimestep(conf, fn, fnp1, ef, dg, timers, iter);
    timers[MainLoop]->end();
  }
  timers[Total]->end();

  finalize(conf, &ef, &dg);

  printTimers(timers);
  freeTimers(timers);

  return 0;
}
