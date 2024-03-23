#ifndef VARIABLE_HPP
#define VARIABLE_HPP

#include <vector>
#include <sycl/sycl.hpp>
#include "../types.hpp"
#include "../config.hpp"

template <typename RealType>
struct Variable {
private:
  using RealView1D = View1D<RealType>;
  using RealView3D = View3D<RealType>;

  using Shape1D = shape_type<1>;
  using Shape3D = shape_type<3>;

  sycl::queue q_;
  RealType *u_, *un_;
  RealType *x_mask_, *y_mask_, *z_mask_;
  Shape1D extents_x_, extents_y_, extents_z_;
  Shape3D extents3D_;

public:
  Variable() = delete;
  Variable(sycl::queue& q, Config& conf) : q_(q) {
    extents3D_ = Shape3D({conf.nxh_, conf.nyh_, conf.nzh_});
    size_type size = conf.nxh_ * conf.nyh_ * conf.nzh_;

    // Allocate device vectors
    u_  = sycl::malloc_shared<RealType>(size, q_);
    un_ = sycl::malloc_shared<RealType>(size, q_);

    x_mask_ = sycl::malloc_shared<RealType>(conf.nxh_, q_);
    y_mask_ = sycl::malloc_shared<RealType>(conf.nyh_, q_);
    z_mask_ = sycl::malloc_shared<RealType>(conf.nzh_, q_);

    for(int i=0; i<conf.nxh_; i++) {
      x_mask_[i] = 1;
    }

    for(int i=0; i<conf.nyh_; i++) {
      y_mask_[i] = 1;
    }

    for(int i=0; i<conf.nzh_; i++) {
      z_mask_[i] = 1;
    }

    if(conf.is_async_) {
      x_mask_[0] = 0.0; x_mask_[conf.nxh_-1] = 0.0;
      y_mask_[0] = 0.0; y_mask_[conf.nyh_-1] = 0.0;
      z_mask_[0] = 0.0; z_mask_[conf.nzh_-1] = 0.0;
    }
  }
  ~Variable() {
    sycl::free(u_,  q_);
    sycl::free(un_, q_);
    sycl::free(x_mask_, q_);
    sycl::free(y_mask_, q_);
    sycl::free(z_mask_, q_);
  }

  // Getters
  inline RealView3D u() { return RealView3D( u_, extents3D_ ); }
  inline RealView3D un() { return RealView3D( un_, extents3D_ ); }

  inline RealView1D x_mask() { return RealView1D( x_mask_, extents_x_ ); }
  inline RealView1D y_mask() { return RealView1D( y_mask_, extents_y_ ); }
  inline RealView1D z_mask() { return RealView1D( z_mask_, extents_z_ ); }
};

#endif