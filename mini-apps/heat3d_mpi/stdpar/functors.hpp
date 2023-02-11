#ifndef __FUNCTORS_HPP__
#define __FUNCTORS_HPP__

#include <iostream>
#include "config.hpp"
#include "grid.hpp"
#include "variable.hpp"
#include "types.hpp"

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
    const int h = conf_.halo_width_;
    const std::size_t ix  = idx % conf_.nx_;
    const std::size_t iyz = idx / conf_.nx_;
    const std::size_t iy  = iyz % conf_.ny_;
    const std::size_t iz  = iyz / conf_.ny_;

    u_(ix+h, iy+h, iz+h) = conf_.umax_ 
                         * cos(  x_(ix) / conf_.Lx_ * 2.0 * M_PI
                               + y_(iy) / conf_.Ly_ * 2.0 * M_PI
                               + z_(iz) / conf_.Lz_ * 2.0 * M_PI
                              );
  }
};

template <typename RealType>
struct heat3d_functor{
private:
  using RealView1D = View1D<RealType>;
  using RealView3D = View3D<RealType>;

  Config conf_;
  double coef_;
  RealView1D x_mask_, y_mask_, z_mask_;
  RealView3D u_, un_;

public:
  heat3d_functor(const Config& conf,
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
  void operator()(const std::size_t idx) const {
    const std::size_t ix  = idx % conf_.nx_ + conf_.halo_width_;
    const std::size_t iyz = idx / conf_.nx_;
    const std::size_t iy  = iyz % conf_.ny_ + conf_.halo_width_;
    const std::size_t iz  = iyz / conf_.ny_ + conf_.halo_width_;

    un_(ix, iy, iz) = u_(ix, iy, iz)
                    + coef_ * ( x_mask_(ix+1) * u_(ix+1, iy, iz) + x_mask_(ix-1) * u_(ix-1, iy, iz)
                              + y_mask_(iy+1) * u_(ix, iy+1, iz) + y_mask_(iy-1) * u_(ix, iy-1, iz)
                              + z_mask_(iz+1) * u_(ix, iy, iz+1) + z_mask_(iz-1) * u_(ix, iy, iz-1)
                              - 6. * u_(ix, iy, iz) );
  }
};

template <class InputView, class OutputView>
struct heat3d_boundary_functor{
private:
  Config conf_;
  double coef_;
  InputView u_left_, u_right_;
  OutputView un_left_, un_right_;
  std::size_t n0_;

public:
  heat3d_boundary_functor(const Config& conf,
                          const InputView& u_left,
                          const InputView& u_right,
                          OutputView& un_left,
                          OutputView& un_right)
    : conf_(conf), u_left_(u_left), u_right_(u_right), un_left_(un_left), un_right_(un_right) {
    coef_ = conf_.Kappa_ * conf_.dt_ / (conf_.dx_*conf_.dx_);
    n0_ = u_left_.extent(0);
  }

  void operator()(const std::size_t idx) const {
    const std::size_t i0 = idx % n0_;
    const std::size_t i1 = idx / n0_;

    un_left_(i0, i1)  += coef_ * u_left_(i0, i1);
    un_right_(i0, i1) += coef_ * u_right_(i0, i1);
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
    const int h = conf_.halo_width_;
    const std::size_t ix  = idx % conf_.nx_;
    const std::size_t iyz = idx / conf_.nx_;
    const std::size_t iy  = iyz % conf_.ny_;
    const std::size_t iz  = iyz / conf_.ny_;

    const auto u_init = conf_.umax_ * cos(  x_(ix) / conf_.Lx_ * 2.0 * M_PI
                                          + y_(iy) / conf_.Ly_ * 2.0 * M_PI
                                          + z_(iz) / conf_.Lz_ * 2.0 * M_PI
                                         );
    un_(ix+h, iy+h, iz+h) = u_init
                    * exp(-conf_.Kappa_ * (pow((2.0*M_PI/conf_.Lx_), 2) + pow(2.0*M_PI/conf_.Ly_, 2) + pow(2.0*M_PI/conf_.Lz_, 2)) * time_);
  }
};

template <class InputView, class OutputView>
struct copy_functor{
private:
  InputView src_left_, src_right_;
  OutputView dst_left_, dst_right_;
  std::size_t n0_;

public:
  copy_functor(const InputView& src_left,
               const InputView& src_right,
               OutputView& dst_left,
               OutputView& dst_right
              ) : src_left_(src_left), src_right_(src_right), dst_left_(dst_left), dst_right_(dst_right) {
    n0_ = src_left_.extent(0);
  }
  
  void operator()(const std::size_t idx) const {
    const std::size_t i0 = idx % n0_;
    const std::size_t i1 = idx / n0_;

    dst_left_(i0, i1) = src_left_(i0, i1);
    dst_right_(i0, i1) = src_right_(i0, i1);
  }
};

#endif
