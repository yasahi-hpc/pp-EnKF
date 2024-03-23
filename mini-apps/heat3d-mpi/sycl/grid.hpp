#ifndef GRID_HPP
#define GRID_HPP

#include <cmath>
#include <iostream>
#include <vector>
#include <sycl/sycl.hpp>
#include "../types.hpp"
#include "../config.hpp"

template <typename ScalarType>
inline std::vector<ScalarType> arange(const ScalarType start,
                                      const ScalarType stop,
                                      const ScalarType step=1
                                     ) {
  const size_t length = ceil((stop - start) / step);

  std::vector<ScalarType> result;

  ScalarType delta = (stop - start) / length;

  // thrust::sequence
  for(auto i=0; i<length; i++) {
    ScalarType value = start + delta*i;
    result.push_back(value);
  }

  return result;
};

template <typename RealType>
struct Grid {
  using RealView1D = View1D<RealType>;
  using Shape1D = shape_type<1>;

private:
  sycl::queue q_;
  RealType *x_, *y_, *z_;
  Shape1D extents_nx_, extents_ny_, extents_nz_;

public:
  Grid() = delete;
  Grid(sycl::queue& q, Config& conf) : q_(q) {
    // Allocate use memory
    auto gx = arange<RealType>(-conf.Lx_/2., conf.Lx_/2., conf.dx_);
    auto gy = arange<RealType>(-conf.Ly_/2., conf.Ly_/2., conf.dy_);
    auto gz = arange<RealType>(-conf.Lz_/2., conf.Lz_/2., conf.dz_);

    // Keep extents
    auto nx = conf.nx_, ny = conf.ny_, nz = conf.nz_;
    extents_nx_ = Shape1D({nx});
    extents_ny_ = Shape1D({ny});
    extents_nz_ = Shape1D({nz});

    // Allocate data
    x_ = sycl::malloc_shared<RealType>(nx, q_);
    y_ = sycl::malloc_shared<RealType>(ny, q_);
    z_ = sycl::malloc_shared<RealType>(nz, q_);

    // Deep copy to local vectors
    std::copy(gx.begin() + conf.nx_ * conf.rx_,
              gx.begin() + conf.nx_ * (conf.rx_+1),
              x_);

    std::copy(gy.begin() + conf.ny_ * conf.ry_,
              gy.begin() + conf.ny_ * (conf.ry_+1),
              y_);

    std::copy(gz.begin() + conf.nz_ * conf.rz_,
              gz.begin() + conf.nz_ * (conf.rz_+1),
              z_);
  }

  ~Grid() {
    sycl::free(x_, q_);
    sycl::free(y_, q_);
    sycl::free(z_, q_);
  }

  // Getters
  inline RealView1D x() { return RealView1D( x_, extents_nx_ ); }
  inline RealView1D y() { return RealView1D( y_, extents_ny_ ); }
  inline RealView1D z() { return RealView1D( z_, extents_nz_ ); }
};

#endif