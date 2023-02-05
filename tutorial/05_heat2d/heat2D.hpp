#ifndef __HEAT2D_HPP__
#define __HEAT2D_HPP__

#include <iostream>
#include "types.hpp"

struct Config{
  std::size_t nx_;
  std::size_t ny_;
  std::size_t nbiter_;
  double dt_;
  double dx_;
  double dy_;
  double Lx_;
  double Ly_;
  double umax_;
  double kappa_;

  Config() {
    nx_ = 1024;
    ny_ = 1024;
    nbiter_ = 10000;

    // resolution
    Lx_ = 1.0;
    Ly_ = 1.0;
    dx_ = Lx_ / static_cast<double>(nx_);
    dy_ = Ly_ / static_cast<double>(ny_);

    // Physical constants
    umax_ = 1.0;
    kappa_ = 1.0;
    dt_ = 0.25 * dx_ * dx_ / kappa_; 
  }
};

struct init_functor{
private:
  std::size_t nx_;
  std::size_t ny_;
  double dx_;
  double dy_;
  double Lx_;
  double Ly_;
  double umax_;
  double xmin_;
  double ymin_;

  RealView1D x_;
  RealView1D y_;
  RealView2D u_;
  RealView2D un_;

public:
  init_functor(const Config& conf,
               RealView1D& x,
               RealView1D& y,
               RealView2D& u,
               RealView2D& un)
    : x_(x), y_(y), u_(u), un_(un) {
    nx_ = conf.nx_;
    ny_ = conf.ny_;
    Lx_ = conf.Lx_;
    Ly_ = conf.Ly_;
    dx_ = conf.dx_;
    dy_ = conf.dy_;
    umax_ = conf.umax_;
    xmin_ = -static_cast<double>(nx_/2) * dx_;
    ymin_ = -static_cast<double>(ny_/2) * dy_;
  }

  MDSPAN_FORCE_INLINE_FUNCTION
  void operator()(const std::size_t idx) const {
    const std::size_t ix = idx % nx_;
    const std::size_t iy = idx / nx_;

    x_(ix) = static_cast<double>(ix) * dx_ + xmin_;
    y_(iy) = static_cast<double>(iy) * dy_ + ymin_;
    u_(ix, iy) = umax_ * cos(x_(ix) / Lx_ * 2.0 * M_PI + y_(iy) / Ly_ * 2.0 * M_PI);
    un_(ix, iy) = 0.0;
  }
};

struct analytical_solution_functor{
private:
  std::size_t nx_;
  double Lx_;
  double Ly_;
  double umax_;
  double kappa_;
  double time_;

  RealView1D x_;
  RealView1D y_;
  RealView2D un_;

public:
  analytical_solution_functor(const Config& conf,
                              const double time,
                              RealView1D& x,
                              RealView1D& y,
                              RealView2D& un)
    : time_(time), x_(x), y_(y), un_(un) {
    nx_ = conf.nx_;
    Lx_ = conf.Lx_;
    Ly_ = conf.Ly_;
    umax_ = conf.umax_;
    kappa_ = conf.kappa_;
  }

  MDSPAN_FORCE_INLINE_FUNCTION
  void operator()(const std::size_t idx) const {
    const std::size_t ix = idx % nx_;
    const std::size_t iy = idx / nx_;

    const double u_init = umax_ * cos(x_(ix) / Lx_ * 2.0 * M_PI + y_(iy) / Ly_ * 2.0 * M_PI);
    un_(ix, iy) = u_init * exp(-kappa_ * (pow((2.0 * M_PI / Lx_), 2) + pow((2.0 * M_PI / Ly_), 2)) * time_);
  }
};

struct heat2d_functor{
private:
  std::size_t nx_;
  std::size_t ny_;
  double coef_;

  RealView2D u_;
  RealView2D un_;

public:
  heat2d_functor(const Config& conf,
                 RealView2D& u,
                 RealView2D& un)
    : u_(u), un_(un) {
    nx_ = conf.nx_;
    ny_ = conf.ny_;
    coef_ = conf.kappa_ * conf.dt_ / (conf.dx_*conf.dx_);
  }

  MDSPAN_FORCE_INLINE_FUNCTION
  void operator()(const std::size_t idx) const {
    const std::size_t ix = idx % nx_;
    const std::size_t iy = idx / nx_;

    const int ixp1 = (ix + nx_ + 1) % nx_;
    const int ixm1 = (ix + nx_ - 1) % nx_;
    const int iyp1 = (iy + ny_ + 1) % ny_;
    const int iym1 = (iy + ny_ - 1) % ny_;

    un_(ix, iy) = u_(ix, iy)
                + coef_ * (u_(ixp1, iy) + u_(ixm1, iy) + u_(ix, iyp1) + u_(ix, iym1) - 4. * u_(ix, iy));
  }
};

template <class Scheduler>
void initialize(const Config& conf,
                Scheduler&& scheduler,
                RealView1D& x,
                RealView1D& y,
                RealView2D& u,
                RealView2D& un) {

  const std::size_t n = conf.nx_ * conf.ny_;
  auto initializer = stdexec::just()
                   | exec::on( scheduler, stdexec::bulk(n, init_functor(conf, x, y, u, un)) ); 
  stdexec::sync_wait(initializer);
}

template <class Scheduler>
void finalize(const Config& conf,
              const double time,
              Scheduler&& scheduler,
              RealView1D& x,
              RealView1D& y,
              RealView2D& u,
              RealView2D& un) {
  
  const std::size_t n = conf.nx_ * conf.ny_;
  const std::size_t nx = conf.nx_;

  auto analytical_solution = stdexec::just()
                 | exec::on( scheduler, stdexec::bulk(n, analytical_solution_functor(conf, time, x, y, un)) ); 
  stdexec::sync_wait(analytical_solution);

  // Check errors
  // un: analytical, u: numerical solutions
  auto L2norm = thrust::transform_reduce(
                  thrust::device,
                  counting_iterator(0), counting_iterator(0) + n,
                  [=] MDSPAN_FORCE_INLINE_FUNCTION (const std::size_t idx) {
                    const std::size_t ix = idx % nx;
                    const std::size_t iy = idx / nx;
                    
                    auto diff = un(ix, iy) - u(ix, iy);
                    return diff * diff;
                  },
                  0.0,
                  thrust::plus<double>()
                );

  L2norm = sqrt(L2norm);
  std::cout << "L2 norm: " << L2norm << std::endl;
}

template <class Scheduler>
void step(const Config& conf,
          Scheduler&& scheduler,
          RealView2D& u,
          RealView2D& un) {
  
  const std::size_t n = conf.nx_ * conf.ny_;
  auto time_step = stdexec::just()
                 | exec::on( scheduler, stdexec::bulk(n, heat2d_functor(conf, u, un)) )
                 | stdexec::then( [&]{ std::swap(u, un); });
  stdexec::sync_wait(time_step);
}

static void report_performance(const Config& conf, double seconds) {
  const std::size_t n = conf.nx_ * conf.ny_;
  const double GBytes = static_cast<double>(n) * static_cast<double>(conf.nbiter_) * 2 * sizeof(double) / 1.e9;

  // 7 Flop per iteration
  const double GFlops = static_cast<double>(n) * static_cast<double>(conf.nbiter_) * 7 / 1.e9;

  std::cout << "Elapsed time: " << seconds << " [s]" << std::endl;
  std::cout << "Bandwidth: " << GBytes / seconds << " [GB/s]" << std::endl;
  std::cout << "Flops: " << GFlops / seconds << " [GFlops]" << std::endl;
}

#endif
