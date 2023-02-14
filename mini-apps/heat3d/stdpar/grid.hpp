#ifndef __GRID_HPP__
#define __GRID_HPP__

#include <cmath>
#include <iostream>
#include <vector>
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
  std::vector<RealType> x_, y_, z_;
  Shape1D extents_nx_, extents_ny_, extents_nz_;

public:
  Grid() = delete;
  Grid(Config& conf) {
    // Allocate device vectors
    x_ = arange<RealType>(-conf.Lx_/2., conf.Lx_/2., conf.dx_);
    y_ = arange<RealType>(-conf.Ly_/2., conf.Ly_/2., conf.dy_);
    z_ = arange<RealType>(-conf.Lz_/2., conf.Lz_/2., conf.dz_);

    // Keep extents
    extents_nx_ = Shape1D({x_.size()});
    extents_ny_ = Shape1D({y_.size()});
    extents_nz_ = Shape1D({z_.size()});
  }

  // Getters
  inline RealView1D x() {
    return RealView1D( x_.data(), extents_nx_ );
  }

  inline RealView1D y() {
    return RealView1D( y_.data(), extents_ny_ );
  }

  inline RealView1D z() {
    return RealView1D( z_.data(), extents_nz_ );
  }

};

#endif
