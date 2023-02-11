#ifndef __CONFIG_HPP__
#define __CONFIG_HPP__

#include "types.hpp"

struct Config {
  // grids
  size_type nx_, ny_, nz_;

  size_type nbiter_;
  size_type freq_diag_;

  const double Lx_ = 1.0;
  const double Ly_ = 1.0;
  const double Lz_ = 1.0;
  double dx_, dy_, dz_;

  // Physical constants
  const double umax_ = 1.0;
  const double Kappa_ = 1.0;
  double dt_;

  Config() = delete;
  Config(size_type nx,
         size_type ny,
         size_type nz,
         size_type nbiter,
         size_type freq_diag)
    : nx_(nx), ny_(ny), nz_(nz), nbiter_(nbiter), freq_diag_(freq_diag) {
    dx_ = Lx_ / static_cast<double>(nx_);
    dy_ = Ly_ / static_cast<double>(ny_);
    dz_ = Lz_ / static_cast<double>(nz_);
    dt_ = 0.1 * dx_ * dx_ / Kappa_;
  }
};

#endif
