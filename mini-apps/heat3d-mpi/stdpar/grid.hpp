#ifndef __GRID_HPP__
#define __GRID_HPP__

#include <cmath>
#include <execution>
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
    // Allocate vectors
    auto gx = arange<RealType>(-conf.Lx_/2., conf.Lx_/2., conf.dx_);
    auto gy = arange<RealType>(-conf.Ly_/2., conf.Ly_/2., conf.dy_);
    auto gz = arange<RealType>(-conf.Lz_/2., conf.Lz_/2., conf.dz_);

    x_.resize(conf.nx_);
    y_.resize(conf.ny_);
    z_.resize(conf.nz_);

    // Deep copy to local vectors
    std::copy(std::execution::par_unseq,
              gx.begin() + conf.nx_ * conf.rx_,
              gx.begin() + conf.nx_ * (conf.rx_+1),
              x_.begin());

    std::copy(std::execution::par_unseq,
              gy.begin() + conf.ny_ * conf.ry_,
              gy.begin() + conf.ny_ * (conf.ry_+1),
              y_.begin());

    std::copy(std::execution::par_unseq,
              gz.begin() + conf.nz_ * conf.rz_,
              gz.begin() + conf.nz_ * (conf.rz_+1),
              z_.begin());

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
