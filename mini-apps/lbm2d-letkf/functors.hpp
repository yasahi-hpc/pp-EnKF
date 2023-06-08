#ifndef __FUNCTORS_HPP__
#define __FUNCTORS_HPP__

#include "config.hpp"

inline int periodic(const int i, const int n) {return (i+n)%n; }

template <class mdspan3d_type, class mdspan2d_type>
struct init_feq_functor {
private:
  Config conf_;
  mdspan2d_type rho_, u_, v_;
  mdspan3d_type f_;

  double c_;

public:
  init_feq_functor(const Config& conf, const mdspan2d_type& rho, const mdspan2d_type& u, const mdspan2d_type& v, mdspan3d_type& f)
    : conf_(conf), f_(f), rho_(rho), u_(u), v_(v) {
    c_ = conf_.settings_.c_;
  }

  void operator()(const int ix, const int iy, const int q) const {
    auto rho_tmp = rho_(ix, iy);
    auto u_tmp = u_(ix, iy);
    auto v_tmp = v_(ix, iy);
    f_(ix, iy, q) = feq(rho_tmp, u_tmp, v_tmp, q);
  }

private:
  double feq(double rho, double u, double v, int q) const {
    auto weight = conf_.phys_.weights_[q];
    auto qx = conf_.phys_.q_[0][q];
    auto qy = conf_.phys_.q_[1][q];

    auto uu = (u*u + v*v) / (c_*c_);
    auto uc = (u*qx + v*qy) / c_;
    return weight * rho * (1.0 + 3.0 * uc + 4.5 * uc * uc - 1.5 * uu);
  }
};

template <class mdspan2d_type>
struct vorticity_functor {
private:
  Config conf_;
  mdspan2d_type u_, v_, vor_;
  int nx_, ny_;
  double dx_;

public:
  vorticity_functor(const Config& conf, const mdspan2d_type& u, const mdspan2d_type& v, mdspan2d_type& vor) 
    : conf_(conf), u_(u), v_(v), vor_(vor) {
    auto [nx, ny] = conf_.settings_.n_;
    nx_ = static_cast<int>(nx);
    ny_ = static_cast<int>(ny);
    dx_ = conf_.settings_.dx_;
  }

  void operator()(const int ix, const int iy) const {
    const auto ixp1 = periodic(ix+1, nx_);
    const auto ixm1 = periodic(ix-1, nx_);
    const auto iyp1 = periodic(iy+1, ny_);
    const auto iym1 = periodic(iy-1, ny_);

    const auto ux = (u_(ixp1, iy) - u_(ixm1, iy)) / (2*dx_);
    const auto uy = (u_(ix, iyp1) - u_(ix, iym1)) / (2*dx_);
    const auto vx = (v_(ixp1, iy) - v_(ixm1, iy)) / (2*dx_);
    const auto vy = (v_(ix, iyp1) - v_(ix, iym1)) / (2*dx_);
    vor_(ix, iy) = vx - uy;
  }
};

template <class mdspan2d_type>
struct sgs_functor {
private:
  Config conf_;
  mdspan2d_type rho_, u_, v_, nu_;
  int nx_, ny_;
  double dx_, dx_sqr_, cs_sqr_;
  bool is_les_csm_ = false; // This should be false for 2D turb?

public:
  sgs_functor(const Config& conf, const mdspan2d_type& rho, const mdspan2d_type& u, const mdspan2d_type& v, mdspan2d_type& nu) 
    : conf_(conf), rho_(rho), u_(u), v_(v), nu_(nu) {
    auto [nx, ny] = conf_.settings_.n_;
    nx_ = static_cast<int>(nx);
    ny_ = static_cast<int>(ny);

    dx_ = conf_.settings_.dx_;
    dx_sqr_ = dx_ * dx_;
    auto cs = conf_.phys_.cs_; // 0.1 -- 0.2 ??
    cs_sqr_ = cs * cs;
  }

  void operator()(const int ix, const int iy) const {
    const auto ixp1 = periodic(ix+1, nx_);
    const auto ixm1 = periodic(ix-1, nx_);
    const auto iyp1 = periodic(iy+1, ny_);
    const auto iym1 = periodic(iy-1, ny_);

    const auto ux = (u_(ixp1, iy) - u_(ixm1, iy)) / (2*dx_);
    const auto uy = (u_(ix, iyp1) - u_(ix, iym1)) / (2*dx_);
    const auto vx = (v_(ixp1, iy) - v_(ixm1, iy)) / (2*dx_);
    const auto vy = (v_(ix, iyp1) - v_(ix, iym1)) / (2*dx_);
    const auto s00 = ux;
    const auto s01 = static_cast<double>(0.5) * (uy + vx);
    const auto s10 = s01;
    const auto s11 = vy;

    const auto ss = sqrt(2 * (s00*s00 + s01*s10 + s10*s01 + s11*s11));

    if(is_les_csm_) {
      const auto qq = -static_cast<double>(0.5) * (ux*ux + uy*vx + vx*uy + vy*vy);
      const auto ee = static_cast<double>(0.5) * (ux*ux + uy*uy + vx*vx + vy*vy);
      const auto fcs = qq/ee;
      nu_(ix, iy) = 0.05 * pow(abs(fcs), static_cast<double>(1.5)) * dx_sqr_ * ss;
    } else {
      nu_(ix, iy) = cs_sqr_ * dx_sqr_ * ss;
    }
  }
};

template <class mdspan3d_type>
struct streaming_functor {
private:
  Config conf_;
  mdspan3d_type f_, fn_;
  int nx_, ny_;

public:
  streaming_functor(const Config& conf, const mdspan3d_type& f, mdspan3d_type& fn)
    : conf_(conf), f_(f), fn_(fn) {
    auto [nx, ny] = conf_.settings_.n_;
    nx_ = static_cast<int>(nx);
    ny_ = static_cast<int>(ny);
  }

  void operator()(const int ix, const int iy, const int q) const {
    const auto qx = conf_.phys_.q_[0][q];
    const auto qy = conf_.phys_.q_[1][q];
    int ix_out = periodic(ix - qx, nx_);
    int iy_out = periodic(iy - qy, ny_);
    fn_(ix, iy, q) = f_(ix_out, iy_out, q);
  }
};

template <class mdspan3d_type, class mdspan2d_type>
struct macroscopic_functor {
private:
  Config conf_;
  mdspan3d_type f_;
  mdspan2d_type rho_, u_, v_;
  double dt_, c_;
  int Q_;

public:
  macroscopic_functor(const Config& conf, const mdspan3d_type& f,
                      mdspan2d_type& rho, mdspan2d_type& u, mdspan2d_type& v)
    : conf_(conf), f_(f), rho_(rho), u_(u), v_(v) {
    Q_ = conf_.phys_.Q_;
    c_ = conf_.settings_.c_;
    dt_ = conf_.settings_.dt_;
  }

  void operator()(const int ix, const int iy) const {
    double rho_tmp = 0.0;
    double u_tmp = 0.0;
    double v_tmp = 0.0;

    for(int q=0; q<Q_; q++) {
      auto f_tmp = f_(ix, iy, q);
      // 0th order moment
      rho_tmp += f_tmp;

      // 1st order moment
      const auto qx = conf_.phys_.q_[0][q];
      const auto qy = conf_.phys_.q_[1][q];
      u_tmp += f_tmp * c_ * qx;
      v_tmp += f_tmp * c_ * qy;
    }

    rho_(ix, iy) = rho_tmp;
    u_(ix, iy)   = u_tmp / rho_tmp;
    v_(ix, iy)   = v_tmp / rho_tmp;
  }
};

template <class mdspan3d_type, class mdspan2d_type>
struct streaming_macroscopic_functor {
private:
  Config conf_;
  mdspan3d_type f_, fn_;
  mdspan2d_type rho_, u_, v_;
  double dt_, c_;
  int nx_, ny_;

public:
  streaming_macroscopic_functor(const Config& conf, const mdspan3d_type& f,
                                mdspan3d_type& fn, mdspan2d_type& rho, mdspan2d_type& u, mdspan2d_type& v)
    : conf_(conf), f_(f), fn_(fn), rho_(rho), u_(u), v_(v) {
    auto [nx, ny] = conf_.settings_.n_;
    nx_ = static_cast<int>(nx);
    ny_ = static_cast<int>(ny);
    c_ = conf_.settings_.c_;
    dt_ = conf_.settings_.dt_;

  }

  void operator()(const int ix, const int iy) const {
    constexpr int Q = 9;
    double rho_tmp = 0.0;
    double u_tmp = 0.0;
    double v_tmp = 0.0;

    double f_tmp[Q];
    for(int q=0; q<Q; q++) {
      const auto qx = conf_.phys_.q_[0][q];
      const auto qy = conf_.phys_.q_[1][q];
      int ix_out = periodic(ix - qx, nx_);
      int iy_out = periodic(iy - qy, ny_);
      f_tmp[q] = f_(ix_out, iy_out, q);
    }

    for(int q=0; q<Q; q++) { fn_(ix, iy, q) = f_tmp[q]; }

    // Moments
    for(int q=0; q<Q; q++) {
      // 0th order moment
      rho_tmp += f_tmp[q];

      // 1st order moment
      const auto qx = conf_.phys_.q_[0][q];
      const auto qy = conf_.phys_.q_[1][q];
      u_tmp += f_tmp[q] * c_ * qx;
      v_tmp += f_tmp[q] * c_ * qy;
    }

    rho_(ix, iy) = rho_tmp;
    u_(ix, iy)   = u_tmp / rho_tmp;
    v_(ix, iy)   = v_tmp / rho_tmp;
  }
};

template <class mdspan3d_type, class mdspan2d_type>
struct collision_srt_functor {
private:
  Config conf_;
  mdspan3d_type f_;
  mdspan2d_type rho_, u_, v_, fx_, fy_, nu_;
  bool is_les_ = true;
  double c_, dt_;
  double omega_;
  double u_ref_;
  double nu_ref_;
  double friction_rate_;

public:
  collision_srt_functor(const Config& conf, mdspan3d_type& f, const mdspan2d_type& fx, const mdspan2d_type& fy,
                        const mdspan2d_type& rho, const mdspan2d_type& u, const mdspan2d_type& v, const mdspan2d_type& nu)
    : conf_(conf), f_(f), fx_(fx), fy_(fy), rho_(rho), u_(u), v_(v), nu_(nu) {
    c_ = conf_.settings_.c_;
    dt_ = conf_.settings_.dt_;
    omega_ = conf_.settings_.omega_;
    nu_ref_ = conf_.phys_.nu_;
    u_ref_  = conf_.phys_.u_ref_;
    friction_rate_ = conf_.phys_.friction_rate_;
    is_les_ = conf_.settings_.is_les_;
  }

  void operator()(const int ix, const int iy) const {
    // This Q must be a compile time constant, otherwise it fails.
    constexpr int Q = 9;
    double f_tmp[Q];
    for(int q=0; q<Q; q++) {
      f_tmp[q] = f_(ix, iy, q);
    }

    auto omega_tmp = omega_;
    if(is_les_) {
      const auto nu_sgs = nu_(ix, iy);
      const auto nu = nu_ref_ + nu_sgs;
      const auto tau = 0.5 + 3.0 * nu / (c_*c_*dt_);
      omega_tmp = 1.0 / tau;
    }

    // Collision
    auto rho_tmp = rho_(ix, iy);
    auto u_tmp   = u_(ix, iy);
    auto v_tmp   = v_(ix, iy);

    for(int q=0; q<Q; q++) {
      auto f_eq_tmp  = feq(rho_tmp, u_tmp, v_tmp, q);
      auto f_neq_tmp = f_tmp[q] - f_eq_tmp;
      f_tmp[q] = f_tmp[q] - omega_tmp * f_neq_tmp;
    }

    // Forcing
    auto fx_tmp = fx_(ix, iy);
    auto fy_tmp = fy_(ix, iy);
    for(int q=0; q<Q; q++) {
      f_tmp[q] += force_acc(fx_tmp, fy_tmp, q, rho_tmp, u_tmp, v_tmp, omega_tmp);
    }

    // Store
    for(int q=0; q<Q; q++) {
      f_(ix, iy, q) = f_tmp[q];
    }
  }

private:
  double force_acc(double fx, double fy, int q, double rho, double u, double v, double omega) const {
    const auto weight = conf_.phys_.weights_[q];
    const auto qx = conf_.phys_.q_[0][q];
    const auto qy = conf_.phys_.q_[1][q];

    const auto cx = u_ref_ * qx;
    const auto cy = u_ref_ * qy;
    const auto cs2 = c_ * c_ / 3;
    const auto fx_tmp = (fx - friction_rate_ / dt_ * u) * dt_;
    const auto fy_tmp = (fy - friction_rate_ / dt_ * v) * dt_;
    auto f1 = (cx*fx_tmp + cy*fy_tmp) / cs2; // first-order term
    auto ret = f1 * weight * rho;
    return ret;
  }

  double feq(double rho, double u, double v, int q) const {
    auto weight = conf_.phys_.weights_[q];
    auto qx = conf_.phys_.q_[0][q];
    auto qy = conf_.phys_.q_[1][q];

    auto uu = (u*u + v*v) / (c_*c_);
    auto uc = (u*qx + v*qy) / c_;
    return weight * rho * (1.0 + 3.0 * uc + 4.5 * uc * uc - 1.5 * uu);
  }
};

#endif
