#include "field.hpp"
#include <executors/Parallel_For.hpp>

void lu_solve_poisson(const Config& conf, Efield* ef);

void field_rho(const Config& conf, const RealView4D &fn, Efield* ef) {
  const Domain dom = conf.dom_;
  auto [_nx, _ny, _nvx, _nvy] = dom.nxmax_;
  const int nx  = static_cast<int>(_nx);
  const int ny  = static_cast<int>(_ny);
  const int nvx = static_cast<int>(_nvx);
  const int nvy = static_cast<int>(_nvy);
  float64 dvx = dom.dx_[2], dvy = dom.dx_[3];

  const auto _fn = fn.mdspan();
  auto rho = ef->rho();

  auto integral = [=] MDSPAN_FORCE_INLINE_FUNCTION (const int ix, const int iy) {
    float64 sum = 0.0;
    for(int ivy=0; ivy<nvy; ivy++) {
      for(int ivx=0; ivx<nvx; ivx++) {
        sum += _fn(ix, iy, ivx, ivy);
      }
    }
    rho(ix, iy) = sum * dvx * dvy;
  };

  Iterate_policy<2> policy2d({0, 0}, {nx, ny});
  Impl::for_each(policy2d, integral);
};

void field_poisson(const Config& conf, Efield* ef) {
  const Domain dom = conf.dom_;
  auto [_nx, _ny, _nvx, _nvy] = dom.nxmax_;
  const int nx  = static_cast<int>(_nx);
  const int ny  = static_cast<int>(_ny);
  float64 dx = dom.dx_[0], dy = dom.dx_[1];
  float64 minPhyx = dom.minPhy_[0], minPhyy = dom.minPhy_[1];

  auto rho = ef->rho();
  auto ex  = ef->ex();
  auto ey  = ef->ey();

  Iterate_policy<2> policy2d({0, 0}, {nx, ny});

  switch(dom.idcase_)
  {
    case 2:
        Impl::for_each(policy2d,
          [=] MDSPAN_FORCE_INLINE_FUNCTION (const int ix, const int iy){
            ex(ix, iy) = -(minPhyx + ix * dx);
            ey(ix, iy) = 0.;
          });
        break;
    case 6:
        Impl::for_each(policy2d,
          [=] MDSPAN_FORCE_INLINE_FUNCTION (const int ix, const int iy){
            ey(ix, iy) = -(minPhyy + iy * dy);
            ex(ix, iy) = 0.;
          });
        break;
    case 10:
    case 20:
        Impl::for_each(policy2d,
          [=] MDSPAN_FORCE_INLINE_FUNCTION (const int ix, const int iy){
            rho(ix, iy) -= 1.;
          });
        lu_solve_poisson(conf, ef);
        break;
    default:
        lu_solve_poisson(conf, ef);
        break;
  }
};

void lu_solve_poisson(const Config& conf, Efield* ef) {
  ef->solve_poisson_fftw();
};
