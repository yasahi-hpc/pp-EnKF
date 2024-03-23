#ifndef FUNCTORS_HPP
#define FUNCTORS_HPP

#include "../config.hpp"
#include "../types.hpp"
#include "grid.hpp"
#include "variable.hpp"

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
    // Get the global ID of the current work-item
    const int ix = item.get_global_id(0);
    const int iy = item.get_global_id(1);
    const int iz = item.get_global_id(2);
    const int h = conf_.halo_width_;

    u_(ix+h, iy+h, iz+h) = conf_.umax_ * cos(   x_(ix) / conf_.Lx_ * 2.0 * M_PI
                                              + y_(iy) / conf_.Ly_ * 2.0 * M_PI
                                              + z_(iz) / conf_.Lz_ * 2.0 * M_PI
                                            );
    un_(ix, iy, iz) = 0; 
  }
};

template <typename RealType>
class Heat3D_functor {
private:
  using RealView1D = View1D<RealType>;
  using RealView3D = View3D<RealType>;

  Config conf_;
  RealType coef_;
  RealView1D x_mask_, y_mask_, z_mask_;
  RealView3D u_, un_;

public:
  Heat3D_functor(const Config& conf,
                 const RealView1D& x_mask,
                 const RealView1D& y_mask,
                 const RealView1D& z_mask,
                 RealView3D& u,
                 RealView3D& un)
    : conf_(conf), x_mask_(x_mask), y_mask_(y_mask), z_mask_(z_mask), u_(u), un_(un) {
    coef_ = conf_.Kappa_ * conf_.dt_ / (conf_.dx_*conf_.dx_);
  }

  /*
   * masks are used for overlapping
   * Values on halo regions should not be used before MPI communications
   * Thus, masks on these regions are set as zero. E.g. x_mask(0) == 0, x_mask(nx+1) == 0
   *
   * If not overlapped, then masks are ones
   */
  void operator()(sycl::nd_item<3> item) const {
    // Get the global ID of the current work-item
    const int h = conf_.halo_width_;
    const int ix = item.get_global_id(0) + h;
    const int iy = item.get_global_id(1) + h;
    const int iz = item.get_global_id(2) + h;
    
    un_(ix, iy, iz) = u_(ix, iy, iz)
                    + coef_ * ( x_mask_(ix+1) * u_(ix+1, iy, iz) + x_mask_(ix-1) * u_(ix-1, iy, iz)
                              + y_mask_(iy+1) * u_(ix, iy+1, iz) + y_mask_(iy-1) * u_(ix, iy-1, iz)
                              + z_mask_(iz+1) * u_(ix, iy, iz+1) + z_mask_(iz-1) * u_(ix, iy, iz-1)
                              - 6.0 * u_(ix, iy, iz) );
  }
};

template <class InputView, class OutputView>
class Heat3DBoundaryfunctor {
private:
  Config conf_;
  double coef_;
  InputView u_left_, u_right_;
  OutputView un_left_, un_right_;

public:
  Heat3DBoundaryfunctor(const Config& conf,
                        const InputView& u_left,
                        const InputView& u_right,
                        OutputView& un_left,
                        OutputView& un_right)
    : conf_(conf), u_left_(u_left), u_right_(u_right), un_left_(un_left), un_right_(un_right) {
    coef_ = conf_.Kappa_ * conf_.dt_ / (conf_.dx_*conf_.dx_);
  }

  void operator()(sycl::nd_item<2> item) const {
    const int i0 = item.get_global_id(0);
    const int i1 = item.get_global_id(1);

    un_left_(i0, i1)  += coef_ * u_left_(i0, i1);
    un_right_(i0, i1) += coef_ * u_right_(i0, i1);
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
    const int h = conf_.halo_width_;
    const int ix = item.get_global_id(0);
    const int iy = item.get_global_id(1);
    const int iz = item.get_global_id(2);

    const auto u_init = conf_.umax_ * cos(  x_(ix) / conf_.Lx_ * 2.0 * M_PI
                                          + y_(iy) / conf_.Ly_ * 2.0 * M_PI
                                          + z_(iz) / conf_.Lz_ * 2.0 * M_PI
                                         );
    un_(ix+h, iy+h, iz+h) = u_init
                          * exp(-conf_.Kappa_ * (pow((2.0*M_PI/conf_.Lx_), 2) + pow(2.0*M_PI/conf_.Ly_, 2) + pow(2.0*M_PI/conf_.Lz_, 2)) * time_ );
  }
};

template <class InputView, class OutputView>
class Copyfunctor {
private:
  InputView src_left_, src_right_;
  OutputView dst_left_, dst_right_;

public:
  Copyfunctor(const InputView& src_left,
              const InputView& src_right,
              OutputView& dst_left,
              OutputView& dst_right
             ) : src_left_(src_left), src_right_(src_right), dst_left_(dst_left), dst_right_(dst_right) {}

  void operator()(sycl::nd_item<2> item) const {
    const int i0 = item.get_global_id(0);
    const int i1 = item.get_global_id(1);

    dst_left_(i0, i1)  = src_left_(i0, i1);
    dst_right_(i0, i1) = src_right_(i0, i1);
  }
};

class reduction_kernel;

// https://registry.khronos.org/SYCL/specs/sycl-2020/html/sycl-2020.html#sec:reduction
template <typename ViewType>
auto check_errors(sycl::queue& q,
                  const Config& conf,
                  ViewType& u,
                  ViewType& un) {
  // Buffers with just 1 element to get the reduction results
  using value_type = typename ViewType::value_type;
  value_type* sum = sycl::malloc_shared<value_type>(1, q);

  const int h = conf.halo_width_;

  // Get the global range for the 3D parallel for loop
  sycl::range<3> global_range(conf.nx_, conf.ny_, conf.nz_);

  // Get the local range for the 3D parallel for loop
  sycl::range<3> local_range = sycl::range<3>(4, 4, 4);

  // Create a 3D nd_range using the global and local ranges
  sycl::nd_range<3> nd_range(global_range, local_range);

  q.submit([&](sycl::handler& cgh) {
    // Create temporary objects describing variables with reduction semantics
    #if defined(__HIPSYCL__) || defined(__OPENSYCL__)
      auto sum_reduction = sycl::reduction(sum, sycl::plus<value_type>());
    #else
      auto sum_reduction = sycl::reduction(sum, sycl::plus<value_type>(), sycl::property::reduction::initialize_to_identity{});
    #endif

    // parallel_for performs two reduction operations
    // For each reduction variable, the implementation:
    // - Creates a corresponding reducer
    // - Passes a reference to the reducer to the lambda as a parameter
    cgh.parallel_for<class reduction_kernel>(
                     nd_range,
                     sum_reduction,
                     [=](sycl::nd_item<3> item, auto& _sum) {
                       // Get the global ID of the current work-item
                       const int ix = item.get_global_id(0) + h;
                       const int iy = item.get_global_id(1) + h;
                       const int iz = item.get_global_id(2) + h;
                       // plus<>() corresponds to += operator, so sum can be
                       // updated via += or combine()
                       auto diff = un(ix, iy, iz) - u(ix, iy, iz);
                       _sum += diff * diff;
                     });
  });

  // Wait for commands to complete. Enforce synchronization on the command queue
  q.wait();

  auto h_sum = *sum;
  sycl::free(sum, q);
  return h_sum;
}

#endif