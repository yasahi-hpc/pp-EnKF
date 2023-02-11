#ifndef __CONFIG_HPP__
#define __CONFIG_HPP__

#include "types.hpp"

struct Config {
  // grids
  std::size_t nx_, ny_, nz_;

  std::size_t nbiter_;
  std::size_t freq_diag_;

  const double Lx_ = 1.0;
  const double Ly_ = 1.0;
  const double Lz_ = 1.0;
  double dx_, dy_, dz_;

  // Physical constants
  const double umax_ = 1.0;
  const double Kappa_ = 1.0;
  double dt_;

  Config() = delete;
  Config(std::size_t nx,
         std::size_t ny,
         std::size_t nz,
         std::size_t nbiter,
         std::size_t freq_diag)
    : nx_(nx), ny_(ny), nz_(nz), nbiter_(nbiter), freq_diag_(freq_diag) {
    dx_ = Lx_ / static_cast<double>(nx_);
    dy_ = Ly_ / static_cast<double>(ny_);
    dz_ = Lz_ / static_cast<double>(nz_);
    dt_ = 0.1 * dx_ * dx_ / Kappa_;
  }
};

#endif
