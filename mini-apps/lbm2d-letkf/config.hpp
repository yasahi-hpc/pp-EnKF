#ifndef __CONFIG_HPP__
#define __CONFIG_HPP__

#include <array>
#include <cmath>

static constexpr std::size_t DIMENSION = 2;

template <std::size_t N>
using shape_type = std::array<std::size_t, N>;

struct Physics {
  const double rho_ref_ = 1.;
  const double u_ref_ = 1.;
  double nu_ = 1.e-4;
  const double h_ref_ = 2 * M_PI; // reference length

  // friction (Chartkov, PRL 2007; Xia, PRE 2014)
  double friction_rate_ = 5e-4;

  // infection force (Watanabe, PRE 1997)
  double kf_ = 4; // injected wavenumber
  double fkf_ = 5.6;
  double dk_ = 10;
  double sigma_ = 5;
  double p_amp_ = 0.01;

  // LES
  double cs_ = 0.2; /// 0.1 -- 0.2 ??

  // LBM array
  const int Q_ = 9;

  const int q_[2][9] = {
    {-1,0,1,-1,0,1,-1,0,1},
    {-1,-1,-1,0,0,0,1,1,1}
  };

  const double weights_[9] = {
    1.0/36.0, 1.0/9.0, 1.0/36.0,
    1.0/9.0,  4.0/9.0, 1.0/9.0,
    1.0/36.0, 1.0/9.0, 1.0/36.0
  };

  // Observation errors
  double obs_error_rho_ = 0.01;
  double obs_error_u_ = 0.1;
};

struct Settings {
  shape_type<DIMENSION> n_ = {256, 256}; /* Number of points on the finest grid */
  const double cfl_ = 0.05;
  double c_;
  double dx_;
  double dt_;
  double tau_;
  double omega_;

  int spinup_ = 1000; // iterations for spinup
  int nbiter_ = 1000; // iterations for main simulation
  int io_interval_ = 1; // Output interval
  int obs_interval_ = 1; // Observation interval
  int da_interval_ = 1; // Interval for data assimilation
  int ensemble_idx_ = 0; // MPI rank
  int rloc_len_ = 1; // r-localization c = p-1
  double beta_ = 1; // covariance inflation

  bool lyapnov_ = false;
  bool is_les_ = true;
  bool is_reference_ = true; // false for DA cases
  bool is_async_ = false; // In order to enable overlapping, in senders/receivers version of letkf
  double ly_epsilon_ = 1.e-8;

  // data assimilation parameter
  double da_nud_rate_ = 0.1; // nudging rate: by predactibility time t_pred and DA interval t_DA: = t_pred / t_DA
};

struct Config {
  Physics phys_;
  Settings settings_;
};

#endif
