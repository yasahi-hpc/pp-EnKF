#ifndef __HEAT3D_HPP__
#define __HEAT3D_HPP__

#include <iostream>
#include <ranges>
#include <execution>
#include "../types.hpp"
#include "../config.hpp"
#include "grid.hpp"
#include "variable.hpp"

template <typename RealType>
struct init_functor{
private:
  using RealView1D = View1D<RealType>;
  using RealView3D = View3D<RealType>;

  Config conf_;
  RealView1D x_, y_, z_;
  RealView3D u_;

public:
  init_functor(const Config& conf,
               const RealView1D& x,
               const RealView1D& y,
               const RealView1D& z,
               RealView3D& u)
    : conf_(conf), x_(x), y_(y), z_(z), u_(u) {}

  void operator()(const std::size_t idx) const {
    const std::size_t ix  = idx % conf_.nx_;
    const std::size_t iyz = idx / conf_.nx_;
    const std::size_t iy  = iyz % conf_.ny_;
    const std::size_t iz  = iyz / conf_.ny_;

    u_(ix, iy, iz) = conf_.umax_ * cos(  x_(ix) / conf_.Lx_ * 2.0 * M_PI
                                       + y_(iy) / conf_.Ly_ * 2.0 * M_PI
                                       + z_(iz) / conf_.Lz_ * 2.0 * M_PI
                                      );
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

template <typename RealType>
void initialize(const Config& conf,
                Grid<RealType>& grid,
                Variable<RealType>& variables) {
  const std::size_t n = conf.nx_ * conf.ny_ * conf.nz_;
  const auto x = grid.x();
  const auto y = grid.y();
  const auto z = grid.z();

  auto u = variables.u();
  auto un = variables.un();

  std::for_each_n(std::execution::par_unseq,
                  std::views::iota(0).begin(), n,
                  init_functor(conf, x, y, z, u));
}

template <typename RealType>
void solve(const Config& conf,
           Variable<RealType>& variables) {
  const std::size_t n = conf.nx_ * conf.ny_ * conf.nz_;

  auto u = variables.u();
  auto un = variables.un();

  for(std::size_t i=0; i<conf.nbiter_; i++) {
    std::for_each_n(std::execution::par_unseq,
                    std::views::iota(0).begin(), n,
                    heat3d_functor(conf, u, un));
    std::swap(u, un);
  }
}

template <typename RealType>
void finalize(const Config& conf,
              Grid<RealType>& grid,
              Variable<RealType>& variables) {
  const std::size_t n = conf.nx_ * conf.ny_ * conf.nz_;
  const double time = conf.dt_ * conf.nbiter_;

  const auto x = grid.x();
  const auto y = grid.y();
  const auto z = grid.z();

  auto u = variables.u();
  auto un = variables.un();

  std::for_each_n(std::execution::par_unseq,
                  std::views::iota(0).begin(), n,
                  analytical_solution_functor(conf, time, x, y, z, un));

  // Check errors
  // un: analytical, u: numerical solutions
  auto L2norm = std::transform_reduce(std::execution::par_unseq,
                                      std::views::iota(0).begin(), std::views::iota(0).begin() + n,
                                      0.0,
                                      std::plus<double>(),
                                      [=] (const std::size_t idx) {
                                        const std::size_t ix  = idx % conf.nx_;
                                        const std::size_t iyz = idx / conf.nx_;
                                        const std::size_t iy  = iyz % conf.ny_;
                                        const std::size_t iz  = iyz / conf.ny_;

                                        auto diff = un(ix, iy, iz) - u(ix, iy, iz);
                                        return diff * diff;
                                      }
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
