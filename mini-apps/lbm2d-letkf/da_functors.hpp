#ifndef __DA_FUNCTORS_HPP__
#define __DA_FUNCTORS_HPP__

#include "config.hpp"

template <class mdspan3d_type, class mdspan2d_type>
struct nudging_functor {
private:
  Config conf_;
  mdspan3d_type f_;
  mdspan2d_type rho_, u_, v_;
  int nx_, ny_;
  int Q_;
  double c_;
  double alpha_;
  int obs_interval_ = 1;

public:
  nudging_functor(const Config& conf, const mdspan2d_type& rho, const mdspan2d_type& u, const mdspan2d_type& v, mdspan3d_type& f)
    : conf_(conf), rho_(rho), u_(u), v_(v), f_(f) {
    auto [nx, ny] = conf_.settings_.n_;
    nx_ = static_cast<int>(nx);
    ny_ = static_cast<int>(ny);
    Q_ = conf_.phys_.Q_;
    c_ = conf_.settings_.c_;
  }

  MDSPAN_FORCE_INLINE_FUNCTION
  void operator()(const int ix, const int iy) const {
    auto rho_tmp = maybeLinearInterp2D(rho_, ix, iy);
    auto u_tmp   = maybeLinearInterp2D(u_, ix, iy);
    auto v_tmp   = maybeLinearInterp2D(v_, ix, iy);

    for(int q=0; q<Q_; q++) {
      const auto f_calc = f_(ix, iy, q);
      const auto f_obs  = feq(rho_tmp, u_tmp, v_tmp, q); 
      f_(ix, iy, q) = alpha_ * f_obs + (1-alpha_) * f_calc;
    }
  }

private:
  MDSPAN_FORCE_INLINE_FUNCTION
  int periodic(const int i, const int n) const {return (i+n)%n; }

  MDSPAN_FORCE_INLINE_FUNCTION
  double feq(double rho, double u, double v, int q) const {
    auto weight = conf_.phys_.weights_[q];
    auto qx = conf_.phys_.q_[0][q];
    auto qy = conf_.phys_.q_[1][q];

    auto uu = (u*u + v*v) / (c_*c_);
    auto uc = (u*qx + v*qy) / c_;
    return weight * rho * (1.0 + 3.0 * uc + 4.5 * uc * uc - 1.5 * uu);
  }

  MDSPAN_FORCE_INLINE_FUNCTION
  double maybeLinearInterp2D(const mdspan2d_type& var, int ix, int iy) const {
    if(obs_interval_ == 1) {
      return var(ix, iy);
    } else {
      const int iw = (ix / obs_interval_) * obs_interval_;
      const int ie = iw + obs_interval_;
      const int is = (iy / obs_interval_) * obs_interval_;
      const int in = is + obs_interval_;

      const auto dx = (ix - iw) / static_cast<double>(obs_interval_);
      const auto dy = (iy - is) / static_cast<double>(obs_interval_);

      const auto iwp = periodic(iw, nx_);
      const auto iep = periodic(ie, nx_);
      const auto isp = periodic(is, ny_);
      const auto inp = periodic(in, ny_);

      auto var_interp = (1.0 - dx) * (1.0 - dy) * var(iwp, isp)
                      +        dx  * (1.0 - dy) * var(iep, isp)
                      + (1.0 - dx) *        dy  * var(iwp, inp)
                      +        dx  *        dy  * var(iep, inp);
      return var_interp;
    }
  }
};

// LETKF
template <class mdspan3d_type>
struct initialize_rR_functor {
private:
  using value_type = mdspan3d_type::value_type;
  Config conf_;

  mdspan3d_type rR_; // (n_obs, n_obs, n_batch)

  value_type obs_error_cov_u_;
  value_type obs_error_cov_v_;
  value_type obs_error_cov_rho_;
  std::size_t n_obs_x_;
  std::size_t n_obs_x_sq_;
  std::size_t nx_;

  int obs_interval_ = 1;
  int obs_local_; // n_obs_local-obs_offset
  int y_offset_; // i_batch0 / config::nx;
  value_type cutoff_;

public:
  initialize_rR_functor(const Config& conf, const int y_offset, mdspan3d_type& rR)
    : conf_(conf), y_offset_(y_offset), rR_(rR) {
    value_type obs_error_u = static_cast<value_type>(conf.phys_.obs_error_u_);
    value_type obs_error_v = static_cast<value_type>(conf.phys_.obs_error_u_);
    value_type obs_error_rho = static_cast<value_type>(conf.phys_.obs_error_rho_);

    value_type power = -2.0;
    obs_error_cov_u_   = pow(obs_error_u, power);
    obs_error_cov_v_   = pow(obs_error_v, power);
    obs_error_cov_rho_ = pow(obs_error_rho, power);

    // localization
    cutoff_ = static_cast<value_type>(conf.settings_.rloc_len_ * conf.settings_.da_interval_)  + 1.0e-6;
    obs_interval_ = conf_.settings_.obs_interval_;
    obs_local_ = conf_.settings_.rloc_len_ * 2;
    n_obs_x_ = obs_local_ * 2 + 1;
    n_obs_x_sq_ = n_obs_x_ * n_obs_x_;
  }

  MDSPAN_FORCE_INLINE_FUNCTION
  void operator()(const int ix, const int iy, const int ib) const {
    // [TO DO] allow layout abstraction
    const int ixy_obs = flatten(ix, iy);

    // obs error covariance
    rR_(ixy_obs, ixy_obs, ib) = obs_error_cov_u_;
    rR_(ixy_obs + n_obs_x_sq_, ixy_obs + n_obs_x_sq_, ib) = obs_error_cov_v_;
    rR_(ixy_obs + n_obs_x_sq_ * 2, ixy_obs + n_obs_x_sq_ * 2, ib) = obs_error_cov_rho_;

    // R-localization
    if(obs_interval_ > 1) {
      // grid offset by i_batch
      const int ix_stt = ib % nx_;
      const int iy_stt = y_offset_ + ib / nx_;

      const int ix_obs_stt = obs_index(ix_stt, ix);
      const int iy_obs_stt = obs_index(iy_stt, iy);
      const value_type dx = static_cast<value_type>(ix_stt - ix_obs_stt);
      const value_type dy = static_cast<value_type>(iy_stt - iy_obs_stt);
      const value_type d = sqrt(dx*dx + dy*dy) / cutoff_;
      const value_type gaspari_cohn =
          (d <= 0) ? 1 :
          (d <= 1) ? -d*d*d*d*d/4. + d*d*d*d/2. + d*d*d*5./8. - d*d*5./3. + 1 :
          (d <= 2) ? d*d*d*d*d/12. - d*d*d*d/2. + d*d*d*5./8. + d*d*5./3. - d*5 + 4 - 2./3./d :
          0;

      rR_(ixy_obs, ixy_obs, ib) *= gaspari_cohn;
      rR_(ixy_obs + n_obs_x_sq_, ixy_obs + n_obs_x_sq_, ib) *= gaspari_cohn;
      rR_(ixy_obs + n_obs_x_sq_ * 2, ixy_obs + n_obs_x_sq_ * 2, ib) *= gaspari_cohn;
    }
  }

private:
  MDSPAN_FORCE_INLINE_FUNCTION
  int flatten(const int ix, const int iy) const { return ix + iy * n_obs_x_; }

  MDSPAN_FORCE_INLINE_FUNCTION
  int obs_index(int idx_obs, int idx_stt) const {
    return (int(idx_stt / obs_interval_) + idx_obs - obs_local_) * obs_interval_;
  }
};

template <class mdspan3d_type, class mdspan2d_type>
struct pack_y_functor {
private:
  using value_type = mdspan3d_type::value_type;
  Config conf_;

  mdspan3d_type y_; // (n_obs_x * n_obs_x, 3, n_batch*n_ens) which is reshaped from y (n_obs, n_batch, n_ens)
  mdspan2d_type rho_, u_, v_;

  std::size_t n_obs_x_;
  int nx_;
  int ny_;
  int y_offset_ = 0;
  int obs_interval_ = 1;
  int obs_local_; // n_obs_local-obs_offset

public:
  pack_y_functor(const Config& conf, const int y_offset, const mdspan2d_type& rho, const mdspan2d_type& u, const mdspan2d_type& v, mdspan3d_type& y)
    : conf_(conf), y_offset_(y_offset), rho_(rho), u_(u), v_(v), y_(y) {
    auto [nx, ny] = conf_.settings_.n_;
    nx_ = static_cast<int>(nx);
    ny_ = static_cast<int>(ny);
    obs_interval_ = conf_.settings_.obs_interval_;
    obs_local_ = conf_.settings_.rloc_len_ * 2;
    n_obs_x_ = obs_local_ * 2 + 1;
  }

  MDSPAN_FORCE_INLINE_FUNCTION
  void operator()(const int ix, const int iy, const int ibx, const int iby) const {
    const int _iby = iby + y_offset_;
    const int _ix_obs = obs_index(ix, ibx);
    const int _iy_obs = obs_index(iy, _iby);
    const int ix_obs = periodic(_ix_obs, nx_);
    const int iy_obs = periodic(_iy_obs, ny_);

    const int ixy = flatten(ix, iy, n_obs_x_);
    const int ib  = flatten(ibx, iby, nx_);

    auto rho_tmp = rho_(ix_obs, iy_obs);
    auto u_tmp   = u_(ix_obs, iy_obs);
    auto v_tmp   = v_(ix_obs, iy_obs);

    if(out_of_localization(ibx, _iby, _ix_obs, _iy_obs)) {u_tmp=0; v_tmp=0; rho_tmp=0;}

    y_(ixy, 0, ib) = u_tmp;
    y_(ixy, 1, ib) = v_tmp;
    y_(ixy, 2, ib) = rho_tmp;
  }

private:
  MDSPAN_FORCE_INLINE_FUNCTION
  int periodic(const int i, const int n) const {return (i+n)%n; }

  MDSPAN_FORCE_INLINE_FUNCTION
  int flatten(const int ix, const int iy, const int nx) const { return ix + iy * nx; }

  MDSPAN_FORCE_INLINE_FUNCTION
  int obs_index(int idx_obs, int idx_stt) const {
    return (int(idx_stt / obs_interval_) + idx_obs - obs_local_) * obs_interval_;
  }

  MDSPAN_FORCE_INLINE_FUNCTION
  bool out_of_localization(const int ibx, const int iby, const int ix_obs, const int iy_obs) const {
    const int di = abs(ibx - ix_obs);
    const int dj = abs(iby - iy_obs);
    return di > obs_local_ * obs_interval_ || dj > obs_local_ * obs_interval_;
  }
};

#endif
