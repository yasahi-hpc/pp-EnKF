#ifndef __CONFIG_HPP__
#define __CONFIG_HPP__

#include <vector>

struct Config {
  // grids
  std::size_t nx_, ny_, nz_;
  std::size_t nxh_, nyh_, nzh_; // Including halos
  std::size_t gnx_, gny_, gnz_;
  int halo_width_ = 1;

  // Parallelization
  int px_, py_, pz_;
  int rx_, ry_, rz_;

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

  // Optimization and Performance measurements
  bool is_async_ = false; // In order to enable overlapping, in senders/receivers version
  bool use_time_stamps_ = false; // for detailed analysis

  Config() = delete;
  Config(std::vector<std::size_t> shape,
         std::vector<int> topology,
         std::vector<int> cart_rank,
         std::size_t nbiter,
         std::size_t freq_diag,
         bool is_async=false,
         bool use_time_stamps=false)
    : nbiter_(nbiter), freq_diag_(freq_diag), is_async_(is_async), use_time_stamps_(use_time_stamps) {

    nx_ = shape[0], ny_ = shape[1], nz_ = shape[2];
    nxh_ = nx_ + 2*halo_width_, nyh_ = ny_ + 2*halo_width_, nzh_ = nz_ + 2*halo_width_;
    px_ = topology[0], py_ = topology[1], pz_ = topology[2];
    rx_ = cart_rank[0], ry_ = cart_rank[1], rz_ = cart_rank[2];

    gnx_ = nx_ * px_;
    gny_ = ny_ * py_;
    gnz_ = nz_ * pz_;

    dx_ = Lx_ / static_cast<double>(gnx_);
    dy_ = Ly_ / static_cast<double>(gny_);
    dz_ = Lz_ / static_cast<double>(gnz_);
    dt_ = 0.1 * dx_ * dx_ / Kappa_;
  }
};

#endif
