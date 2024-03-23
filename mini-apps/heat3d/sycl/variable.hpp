#ifndef VARIABLE_HPP
#define VARIABLE_HPP

#include <vector>
#include <sycl/sycl.hpp>
#include "../types.hpp"
#include "../config.hpp"

template <typename RealType>
struct Variable {
private:
  using RealView3D = View3D<RealType>;
  using Shape3D = shape_type<3>;

  sycl::queue m_q;
  RealType *u_, *un_;
  Shape3D extents3D_;

public:
  Variable() = delete;
  Variable(sycl::queue& q, Config& conf) : m_q(q) {
    extents3D_ = Shape3D({conf.nx_, conf.ny_, conf.nz_});
    size_type size = conf.nx_ * conf.ny_ * conf.nz_;

    // Allocate device vectors
    u_ = sycl::malloc_shared<RealType>(size, m_q);
    un_ = sycl::malloc_shared<RealType>(size, m_q);
  }
  ~Variable() {
    sycl::free(u_, m_q);
    sycl::free(un_, m_q);
  }

  // Getters
  inline RealView3D u() { return RealView3D( u_, extents3D_ ); }
  inline RealView3D un() { return RealView3D( un_, extents3D_ ); }
};

#endif