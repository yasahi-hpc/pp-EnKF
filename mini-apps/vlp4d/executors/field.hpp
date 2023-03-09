#ifndef __FIELD_HPP__
#define __FIELD_HPP__

#include "../config.hpp"
#include "types.hpp"
#include "efield.hpp"
#include "diags.hpp"
#include <stdexec/execution.hpp>
#include "exec/on.hpp"

template <class Scheduler>
void field_rho(const Config& conf, Scheduler&& scheduler, const RealView4D& fn, Efield* ef) {
  const Domain dom = conf.dom_;
  auto [nx, ny, nvx, nvy] = dom.nxmax_;
  float64 dvx = dom.dx_[2], dvy = dom.dx_[3];
  
  const auto _fn = fn.mdspan();
  auto rho = ef->rho();
  
  auto integral = [=] MDSPAN_FORCE_INLINE_FUNCTION (const std::size_t idx) {
    const int ix = idx % nx;
    const int iy = idx / nx;
    float64 sum = 0.0;
    for(int ivy=0; ivy<nvy; ivy++) {
      for(int ivx=0; ivx<nvx; ivx++) {
        sum += _fn(ix, iy, ivx, ivy);
      }
    }
    rho(ix, iy) = sum * dvx * dvy;
  };

  auto integral_task = stdexec::just()
    | exec::on( scheduler, stdexec::bulk(nx*ny, integral) );
  stdexec::sync_wait( std::move(integral_task) );
}

template <class Scheduler>
void field_poisson(const Config& conf, Scheduler&& scheduler, Efield* ef) {
  const Domain dom = conf.dom_;
  auto [nx, ny, nvx, nvy] = dom.nxmax_;
  float64 dx = dom.dx_[0], dy = dom.dx_[1];
  float64 minPhyx = dom.minPhy_[0], minPhyy = dom.minPhy_[1];
 
  auto rho = ef->rho();
  auto ex  = ef->ex();
  auto ey  = ef->ey();
 
  switch(dom.idcase_)
  {
    case 2:
    {
      auto poisson_task = stdexec::just()
        | exec::on( scheduler, 
            stdexec::bulk(nx*ny,
              [=] MDSPAN_FORCE_INLINE_FUNCTION (const std::size_t idx) {
                const int ix = idx % nx;
                const int iy = idx / nx;
                ex(ix, iy) = -(minPhyx + ix * dx);
                ey(ix, iy) = 0.;
              }));
      stdexec::sync_wait( std::move(poisson_task) );
      break;
    }
    case 6:
    {
      auto poisson_task = stdexec::just()
        | exec::on( scheduler, 
            stdexec::bulk(nx*ny,
              [=] MDSPAN_FORCE_INLINE_FUNCTION (const std::size_t idx) {
                const int ix = idx % nx;
                const int iy = idx / nx;
                ey(ix, iy) = -(minPhyy + iy * dy);
                ex(ix, iy) = 0.;
              }));
      stdexec::sync_wait( std::move(poisson_task) );
      break;
    }
    case 10:
    case 20:
    {
      auto poisson_task = stdexec::just()
        | exec::on( scheduler, 
            stdexec::bulk(nx*ny,
              [=] MDSPAN_FORCE_INLINE_FUNCTION (const std::size_t idx) {
                const int ix = idx % nx;
                const int iy = idx / nx;
                rho(ix, iy) -= 1.;
              }));
      stdexec::sync_wait( std::move(poisson_task) );
      ef->solve_poisson_fftw(scheduler);
      break;
    }
    default:
    {
      ef->solve_poisson_fftw(scheduler);
      break;
    }
  }
}

#endif
