#ifndef TIMESTEP_HPP
#define TIMESTEP_HPP

#include "../timer.hpp"
#include "efield.hpp"
#include "diags.hpp"
#include "types.hpp"
#include "advection.hpp"
#include "field.hpp"

void onetimestep(sycl::queue& q, Config& conf, RealView4D& fn, RealView4D& fnp1, std::unique_ptr<Efield>& ef, std::unique_ptr<Diags>& dg, std::vector<Timer*>& timers, int iter);

void onetimestep(sycl::queue& q, Config& conf, RealView4D& fn, RealView4D& fnp1, std::unique_ptr<Efield>& ef, std::unique_ptr<Diags>& dg, std::vector<Timer*>& timers, int iter) {
  const Domain dom = conf.dom_;

  timers[Advec1D_x]->begin();
  Advection::advect_1D_x(q, conf, fn, fnp1, 0.5 * dom.dt_);
  timers[Advec1D_x]->end();

  timers[Advec1D_y]->begin();
  Advection::advect_1D_y(q, conf, fnp1, fn, 0.5 * dom.dt_);
  timers[Advec1D_y]->end();

  timers[Field]->begin();
  field_rho(q, conf, fn, ef);
  timers[Field]->end();

  timers[Fourier]->begin();
  field_poisson(q, conf, ef);
  timers[Fourier]->end();

  timers[Diag]->begin();
  dg->compute(q, conf, ef, iter);
  timers[Diag]->end();

  timers[Advec1D_vx]->begin();
  Advection::advect_1D_vx(q, conf, fn, fnp1, ef, dom.dt_);
  timers[Advec1D_vx]->end();

  timers[Advec1D_vy]->begin();
  Advection::advect_1D_vy(q, conf, fnp1, fn, ef, dom.dt_);
  timers[Advec1D_vy]->end();

  timers[Advec1D_x]->begin();
  Advection::advect_1D_x(q, conf, fn, fnp1, 0.5 * dom.dt_);
  timers[Advec1D_x]->end();

  timers[Advec1D_y]->begin();
  Advection::advect_1D_y(q, conf, fnp1, fn, 0.5 * dom.dt_);
  timers[Advec1D_y]->end();

  if(dom.fxvx_) {
    if(iter % dom.ifreq_ == 0) {
      Advection::print_fxvx(conf, fn, iter);
    }
  }
}

#endif
