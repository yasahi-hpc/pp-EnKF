#ifndef __VARIABLE_HPP__
#define __VARIABLE_HPP__

#include <vector>
#include "../types.hpp"
#include "../config.hpp"

template <typename RealType>
struct Variable {
private:
  using RealView3D = View3D<RealType>;
  using Shape3D = shape_type<3>;
  std::vector<RealType> u_, un_;
  Shape3D extents3D_;

public:
  Variable() = delete;
  Variable(Config& conf) {
    extents3D_ = Shape3D({conf.nx_, conf.ny_, conf.nz_});
    size_type size = conf.nx_ * conf.ny_ * conf.nz_;

    // Allocate device vectors
    u_.resize(size, 0);
    un_.resize(size, 0);
  }

  // Getters
  inline RealView3D u() {
    return RealView3D( u_.data(), extents3D_ );
  }

  inline RealView3D  un() {
    return RealView3D( un_.data(), extents3D_ );
  }
};

#endif
