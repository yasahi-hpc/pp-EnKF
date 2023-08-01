#ifndef __VARIABLE_HPP__
#define __VARIABLE_HPP__

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "../types.hpp"
#include "../config.hpp"

template <typename RealType>
struct Variable {
private:
  using RealView1D = View1D<RealType>;
  using RealView3D = View3D<RealType>;

  using Shape1D = shape_type<1>;
  using Shape3D = shape_type<3>;
  thrust::device_vector<RealType> u_, un_;
  thrust::device_vector<RealType> x_mask_, y_mask_, z_mask_;
  Shape1D extents_x_, extents_y_, extents_z_;
  Shape3D extents3D_;

public:
  Variable() = delete;
  Variable(Config& conf) {
    extents3D_ = Shape3D({conf.nxh_, conf.nyh_, conf.nzh_});
    extents_x_ = Shape1D{conf.nxh_};
    extents_y_ = Shape1D{conf.nyh_};
    extents_z_ = Shape1D{conf.nzh_};

    size_type size = conf.nxh_ * conf.nyh_ * conf.nzh_;

    // Allocate device vectors
    u_.resize(size, 0);
    un_.resize(size, 0);

    x_mask_.resize(conf.nxh_, 1.0);
    y_mask_.resize(conf.nyh_, 1.0);
    z_mask_.resize(conf.nzh_, 1.0);

    if(conf.is_async_) {
      thrust::host_vector<RealType> x_mask = x_mask_;
      thrust::host_vector<RealType> y_mask = y_mask_;
      thrust::host_vector<RealType> z_mask = z_mask_;

      x_mask[0] = 0.0; x_mask.back() = 0.0;
      y_mask[0] = 0.0; y_mask.back() = 0.0;
      z_mask[0] = 0.0; z_mask.back() = 0.0;

      x_mask_ = x_mask;
      y_mask_ = y_mask;
      z_mask_ = z_mask;
    }
  }

  // Getters
  inline RealView3D  u() const {
    return RealView3D( (RealType *)thrust::raw_pointer_cast(u_.data()), extents3D_ );
  }

  inline RealView3D u() {
    return RealView3D( (RealType *)thrust::raw_pointer_cast(u_.data()), extents3D_ );
  }

  inline RealView3D  un() const {
    return RealView3D( (RealType *)thrust::raw_pointer_cast(un_.data()), extents3D_ );
  }

  inline RealView3D  un() {
    return RealView3D( (RealType *)thrust::raw_pointer_cast(un_.data()), extents3D_ );
  }

  inline RealView1D x_mask() {
    return RealView1D( (RealType *)thrust::raw_pointer_cast(x_mask_.data()), extents_x_ );
  }

  inline RealView1D x_mask() const {
    return RealView1D( (RealType *)thrust::raw_pointer_cast(x_mask_.data()), extents_x_ );
  }

  inline RealView1D y_mask() {
    return RealView1D( (RealType *)thrust::raw_pointer_cast(y_mask_.data()), extents_y_ );
  }

  inline RealView1D y_mask() const {
    return RealView1D( (RealType *)thrust::raw_pointer_cast(y_mask_.data()), extents_y_ );
  }

  inline RealView1D z_mask() {
    return RealView1D( (RealType *)thrust::raw_pointer_cast(z_mask_.data()), extents_z_ );
  }

  inline RealView1D z_mask() const {
    return RealView1D( (RealType *)thrust::raw_pointer_cast(z_mask_.data()), extents_z_ );
  }
};

#endif
