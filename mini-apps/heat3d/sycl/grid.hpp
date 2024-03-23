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
  using Shape1D    = shape_type<1>;

private:
  sycl::queue m_q;
  RealType *x_, *y_, *z_;
  Shape1D extents_nx_, extents_ny_, extents_nz_;

public:
  Grid() = delete;
  Grid(sycl::queue& q, Config& conf) : m_q(q) {
    // Allocate use memory
    auto x = arange<RealType>(-conf.Lx_/2., conf.Lx_/2., conf.dx_);
    auto y = arange<RealType>(-conf.Ly_/2., conf.Ly_/2., conf.dy_);
    auto z = arange<RealType>(-conf.Lz_/2., conf.Lz_/2., conf.dz_);

    // Keep extents
    auto nx = x.size(), ny = y.size(), nz = z.size();
    extents_nx_ = Shape1D({nx});
    extents_ny_ = Shape1D({ny});
    extents_nz_ = Shape1D({nz});

    // Allocate data
    x_ = sycl::malloc_shared<RealType>(nx, m_q);
    y_ = sycl::malloc_shared<RealType>(ny, m_q);
    z_ = sycl::malloc_shared<RealType>(nz, m_q);

    for(std::size_t i=0; i<nx; i++) { x_[i] = x.at(i); }
    for(std::size_t i=0; i<ny; i++) { y_[i] = y.at(i); }
    for(std::size_t i=0; i<nz; i++) { z_[i] = z.at(i); }
  }

  ~Grid() {
    sycl::free(x_, m_q);
    sycl::free(y_, m_q);
    sycl::free(z_, m_q);
  }

  // Getters
  inline RealView1D x() { return RealView1D( x_, extents_nx_ ); }
  inline RealView1D y() { return RealView1D( y_, extents_ny_ ); }
  inline RealView1D z() { return RealView1D( z_, extents_nz_ ); }
};

#endif