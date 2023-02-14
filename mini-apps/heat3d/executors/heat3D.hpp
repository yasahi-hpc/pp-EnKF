#ifndef __HEAT3D_HPP__
#define __HEAT3D_HPP__

#include <iostream>
#include <stdexec/execution.hpp>
#include <thrust/execution_policy.h>
#include "exec/on.hpp"
#include "../types.hpp"
#include "../config.hpp"
#include "grid.hpp"
#include "variable.hpp"

using counting_iterator = thrust::counting_iterator<size_type>;

template <typename RealType>
struct init_functor{
private:
  using RealView1D = View1D<RealType>;
  using RealView3D = View3D<RealType>;

  Config conf_;
  RealView1D x_, y_, z_;
  RealView3D u_, un_;

public:
  init_functor(const Config& conf,
               const RealView1D& x,
               const RealView1D& y,
               const RealView1D& z,
               RealView3D& u,
               RealView3D& un)
    : conf_(conf), x_(x), y_(y), z_(z), u_(u), un_(un) {}

  MDSPAN_FORCE_INLINE_FUNCTION
  void operator()(const std::size_t idx) const {
    const std::size_t ix  = idx % conf_.nx_;
    const std::size_t iyz = idx / conf_.nx_;
    const std::size_t iy  = iyz % conf_.ny_;
    const std::size_t iz  = iyz / conf_.ny_;

    u_(ix, iy, iz) = conf_.umax_ * cos(  x_(ix) / conf_.Lx_ * 2.0 * M_PI
                                       + y_(iy) / conf_.Ly_ * 2.0 * M_PI
                                       + z_(iz) / conf_.Lz_ * 2.0 * M_PI
                                      );
    un_(ix, iy, iz) = 0.0;
  }
};

template <typename RealType>
struct heat3d_functor{
private:
  using RealView3D = View3D<RealType>;

  Config conf_;
  double coef_;
  RealView3D u_, un_;

public:
  heat3d_functor(const Config& conf,
                 RealView3D& u,
                 RealView3D& un)
    : conf_(conf), u_(u), un_(un) {
    coef_ = conf_.Kappa_ * conf_.dt_ / (conf_.dx_*conf_.dx_);
  }

  MDSPAN_FORCE_INLINE_FUNCTION
  void operator()(const std::size_t idx) const {
    const std::size_t ix  = idx % conf_.nx_;
    const std::size_t iyz = idx / conf_.nx_;
    const std::size_t iy  = iyz % conf_.ny_;
    const std::size_t iz  = iyz / conf_.ny_;

    const int ixp1 = (ix + conf_.nx_ + 1) % conf_.nx_;
    const int ixm1 = (ix + conf_.nx_ - 1) % conf_.nx_;
    const int iyp1 = (iy + conf_.ny_ + 1) % conf_.ny_;
    const int iym1 = (iy + conf_.ny_ - 1) % conf_.ny_;
    const int izp1 = (iz + conf_.nz_ + 1) % conf_.nz_;
    const int izm1 = (iz + conf_.nz_ - 1) % conf_.nz_;

    un_(ix, iy, iz) = u_(ix, iy, iz)
                    + coef_ * ( u_(ixp1, iy, iz) + u_(ixm1, iy, iz)
                              + u_(ix, iyp1, iz) + u_(ix, iym1, iz)
                              + u_(ix, iy, izp1) + u_(ix, iy, izm1)
                              - 6. * u_(ix, iy, iz) );
  }
};

template <typename RealType>
struct analytical_solution_functor{
private:
  using RealView1D = View1D<RealType>;
  using RealView3D = View3D<RealType>;

  Config conf_;
  double time_;
  RealView1D x_, y_, z_;
  RealView3D un_;

public:
  analytical_solution_functor(const Config& conf,
                              const double time,
                              const RealView1D& x,
                              const RealView1D& y,
                              const RealView1D& z,
                              RealView3D& un)
    : conf_(conf), time_(time), x_(x), y_(y), z_(z), un_(un) {}

  MDSPAN_FORCE_INLINE_FUNCTION
  void operator()(const std::size_t idx) const {
    const std::size_t ix  = idx % conf_.nx_;
    const std::size_t iyz = idx / conf_.nx_;
    const std::size_t iy  = iyz % conf_.ny_;
    const std::size_t iz  = iyz / conf_.ny_;

    const auto u_init = conf_.umax_ * cos(  x_(ix) / conf_.Lx_ * 2.0 * M_PI
                                          + y_(iy) / conf_.Ly_ * 2.0 * M_PI
                                          + z_(iz) / conf_.Lz_ * 2.0 * M_PI
                                         );
    un_(ix, iy, iz) = u_init
                    * exp(-conf_.Kappa_ * (pow((2.0*M_PI/conf_.Lx_), 2) + pow(2.0*M_PI/conf_.Ly_, 2) + pow(2.0*M_PI/conf_.Lz_, 2)) * time_);
  }
};

template <class Scheduler, typename RealType>
void initialize(const Config& conf,
                const Grid<RealType>& grid,
                Scheduler&& scheduler,
                Variable<RealType>& variables) {
  const std::size_t n = conf.nx_ * conf.ny_ * conf.nz_;
  const auto x = grid.x();
  const auto y = grid.y();
  const auto z = grid.z();

  auto u = variables.u();
  auto un = variables.un();

  auto initializer = stdexec::just()
                   | exec::on( scheduler, 
                               stdexec::bulk(n, init_functor(conf, x, y, z, u, un)) ); 
  stdexec::sync_wait( std::move(initializer) );
}

template <class Scheduler, typename RealType>
void solve(const Config& conf,
           Scheduler&& scheduler,
           Variable<RealType>& variables) {
  const std::size_t n = conf.nx_ * conf.ny_ * conf.nz_;

  auto u = variables.u();
  auto un = variables.un();

  for(std::size_t i=0; i<conf.nbiter_; i++) {
    auto step = stdexec::just()
              | exec::on( scheduler, 
                          stdexec::bulk(n, heat3d_functor(conf, u, un)) )
              | stdexec::then( [&]{ std::swap(u, un); } );
    stdexec::sync_wait( std::move(step) );
  }
}

template <class Scheduler, typename RealType>
void finalize(const Config& conf,
              const Grid<RealType>& grid,
              Scheduler&& scheduler,
              Variable<RealType>& variables) {
  const std::size_t n = conf.nx_ * conf.ny_ * conf.nz_;
  const double time = conf.dt_ * conf.nbiter_;

  const auto x = grid.x();
  const auto y = grid.y();
  const auto z = grid.z();

  auto u = variables.u();
  auto un = variables.un();

  auto analytical_solution = stdexec::just()
                 | exec::on( scheduler, stdexec::bulk(n, analytical_solution_functor(conf, time, x, y, z, un)) );
  stdexec::sync_wait( std::move(analytical_solution) );

  // Check errors
  // un: analytical, u: numerical solutions
  auto L2norm = thrust::transform_reduce(
                  thrust::device,
                  counting_iterator(0), counting_iterator(0) + n,
                  [=] MDSPAN_FORCE_INLINE_FUNCTION (const std::size_t idx) {
                    const std::size_t ix  = idx % conf.nx_;
                    const std::size_t iyz = idx / conf.nx_;
                    const std::size_t iy  = iyz % conf.ny_;
                    const std::size_t iz  = iyz / conf.ny_;

                    auto diff = un(ix, iy, iz) - u(ix, iy, iz);
                    return diff * diff;
                  },
                  0.0,
                  thrust::plus<double>()
                );

  L2norm = sqrt(L2norm);
  std::cout << "L2 norm: " << L2norm << std::endl;
}

static void report_performance(const Config& conf, double seconds) {
  const std::size_t n = conf.nx_ * conf.ny_ * conf.nz_;
  const double GBytes = static_cast<double>(n) * static_cast<double>(conf.nbiter_) * 2 * sizeof(double) / 1.e9;

  // 9 Flop per iteration
  const double GFlops = static_cast<double>(n) * static_cast<double>(conf.nbiter_) * 9 / 1.e9;

  std::cout << "Elapsed time: " << seconds << " [s]" << std::endl;
  std::cout << "Bandwidth: " << GBytes / seconds << " [GB/s]" << std::endl;
  std::cout << "Flops: " << GFlops / seconds << " [GFlops]" << std::endl;
}

#endif
