#ifndef HEAT3D_HPP
#define HEAT3D_HPP

#include <iostream>
#include <sycl/sycl.hpp>
#include <execution>
#include <memory>
#include "../types.hpp"
#include "../config.hpp"
#include "grid.hpp"
#include "variable.hpp"

template<class T>
using usm_alloc = sycl::usm_allocator<T, sycl::usm::alloc::shared>;

template<class T>
using usm_vector = std::vector<T, usm_alloc<T>>;

template <typename RealType>
class Init_functor {
private:
  using RealView1D = View1D<RealType>;
  using RealView3D = View3D<RealType>;

  Config conf_;
  RealView1D x_, y_, z_;
  RealView3D u_, un_;

public:
  Init_functor(const Config& conf,
               const RealView1D& x,
               const RealView1D& y,
               const RealView1D& z,
               RealView3D& u,
               RealView3D& un
              )
    : conf_(conf), x_(x), y_(y), z_(z), u_(u), un_(un) {}

  void operator()(sycl::nd_item<3> item) const {
    const int ix = item.get_global_id(0);
    const int iy = item.get_global_id(1);
    const int iz = item.get_global_id(2);

    u_(ix, iy, iz) = conf_.umax_ * std::cos(  x_(ix) / conf_.Lx_ * 2.0 * M_PI
                                            + y_(iy) / conf_.Ly_ * 2.0 * M_PI
                                            + z_(iz) / conf_.Lz_ * 2.0 * M_PI
                                           );
    un_(ix, iy, iz) = 0; 
  }
};

template <typename RealType>
class Heat3D_functor {
private:
  using RealView3D = View3D<RealType>;

  Config conf_;
  RealType coef_;
  RealView3D u_, un_;

public:
  Heat3D_functor(const Config& conf,
                 const RealView3D& u,
                 RealView3D& un)
    : conf_(conf), u_(u), un_(un) {
    coef_ = conf_.Kappa_ * conf_.dt_ / (conf_.dx_*conf_.dx_);
  }

  void operator()(sycl::nd_item<3> item) const {
    const int ix = item.get_global_id(0);
    const int iy = item.get_global_id(1);
    const int iz = item.get_global_id(2);
    
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
class AnalyticalSolutionFunctor {
private:
  using RealView1D = View1D<RealType>;
  using RealView3D = View3D<RealType>;

  Config conf_;
  double time_;
  RealView1D x_, y_, z_;
  RealView3D un_;

public:
  AnalyticalSolutionFunctor(const Config& conf,
                            const double time,
                            const RealView1D& x,
                            const RealView1D& y,
                            const RealView1D& z,
                            RealView3D& un)
    : conf_(conf), time_(time), x_(x), y_(y), z_(z), un_(un) {}

  void operator()(sycl::nd_item<3> item) const {
    // Get the global ID of the current work-item
    const int ix = item.get_global_id(0);
    const int iy = item.get_global_id(1);
    const int iz = item.get_global_id(2);

    const auto u_init = conf_.umax_ * cos(  x_(ix) / conf_.Lx_ * 2.0 * M_PI
                                          + y_(iy) / conf_.Ly_ * 2.0 * M_PI
                                          + z_(iz) / conf_.Lz_ * 2.0 * M_PI
                                         );
    un_(ix, iy, iz) = u_init
                    * exp(-conf_.Kappa_ * (pow((2.0*M_PI/conf_.Lx_), 2) + pow(2.0*M_PI/conf_.Ly_, 2) + pow(2.0*M_PI/conf_.Lz_, 2)) * time_ );
  }
};

template <typename RealType>
void initialize(sycl::queue& q,
                const Config& conf,
                Grid<RealType>& grid,
                Variable<RealType>& variables) {
  auto const x = grid.x();
  auto const y = grid.y();
  auto const z = grid.z();

  auto u  = variables.u();
  auto un = variables.un();

  // 3D loop
  sycl::range<3> global_range(conf.nx_, conf.ny_, conf.nz_);
  sycl::range<3> local_range = sycl::range<3>(4, 4, 4);
  sycl::nd_range<3> nd_range(global_range, local_range);

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
      nd_range, 
      Init_functor(conf, x, y, z, u, un)
    );
  }).wait();
}

template <typename RealType>
void solve(sycl::queue& q,
           const Config& conf,
           Variable<RealType>& variables) {
  double coef = conf.Kappa_ * conf.dt_ / (conf.dx_*conf.dx_);

  auto u  = variables.u();
  auto un = variables.un();

  // 3D loop
  sycl::range<3> global_range(conf.nx_, conf.ny_, conf.nz_);
  sycl::range<3> local_range = sycl::range<3>(32, 8, 1);
  sycl::nd_range<3> nd_range(global_range, local_range);

  for(std::size_t i=0; i<conf.nbiter_; i++) {
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
        nd_range, 
        Heat3D_functor(conf, u, un)
      );
    }).wait();
    std::swap(u, un);
  }
}

class reduction_kernel;
template <typename RealType>
void finalize(sycl::queue& q, 
              const Config& conf,
              Grid<RealType>& grid,
              Variable<RealType>& variables) {
  const double time = conf.dt_ * conf.nbiter_;

  auto const x = grid.x();
  auto const y = grid.y();
  auto const z = grid.z();

  auto const u  = variables.u();
  auto       un = variables.un();

  // 3D loop
  sycl::range<3> global_range(conf.nx_, conf.ny_, conf.nz_);
  sycl::range<3> local_range = sycl::range<3>(4, 4, 4);
  sycl::nd_range<3> nd_range(global_range, local_range);

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
      nd_range, 
      AnalyticalSolutionFunctor(conf, time, x, y, z, un)
    );
  }).wait();

  // un: analytical, u: numerical solutions
  usm_alloc<double> alloc(q);
  usm_vector<double> diff2(1, 0.0, alloc);

  q.submit([&](sycl::handler& cgh) {
    auto diff2_reduction = sycl::reduction(diff2.data(), sycl::plus<double>());
    cgh.parallel_for<class reduction_kernel>(
      nd_range, diff2_reduction,
      [=](sycl::nd_item<3> item, auto& sum) {
        const int ix = item.get_global_id(0);
        const int iy = item.get_global_id(1);
        const int iz = item.get_global_id(2);
        auto diff = un(ix, iy, iz) - u(ix, iy, iz);
        sum += diff * diff;
      }
    );
  }).wait();
  auto L2norm = sqrt(diff2.at(0));

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