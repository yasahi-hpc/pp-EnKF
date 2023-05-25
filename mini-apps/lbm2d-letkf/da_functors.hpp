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
  double dx_;
  double c_, dt_;
  double omega_ = 1.6;
  double alpha_;
  bool da_w_itp_;
  int xyprune_;
  const int q_[2][9] = {
    {1,1,1,0,0,0,-1,-1,-1},
    {1,0,-1,1,0,-1,1,0,-1}
  };

  const double weights_[9] = {1.0/216.0, 1.0/54.0, 1.0/216.0, 1.0/54.0, 8.0/27.0, 1.0/54.0, 1.0/216.0, 1.0/54.0, 1.0/216.0};

public:
  nudging_functor(const Config& conf, const mdspan2d_type& rho, const mdspan2d_type& u, const mdspan2d_type& v, mdspan3d_type& f)
    : conf_(conf), rho_(rho), u_(u), v_(v), f_(f) {
  }

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
  int periodic(const int i, const int n) const {return (i+n)%n; }

  double feq(double rho, double u, double v, int q) const {
    auto uu = (u*u + v*v) / (c_*c_);
    auto uc = (u * q_[0][q] + v * q_[1][q]) / c_;
    return weights_[q] * rho * (1.0 + 3.0 * uc + 4.5 * uc * uc - 1.5 * uu); 
  }

  double maybeLinearInterp2D(const mdspan2d_type& var, int ix, int iy) const {
    if(xyprune_ == 1 || !da_w_itp_) {
      return var(ix, iy);
    } else {
      const int iw = (ix / xyprune_) * xyprune_;
      const int ie = iw + xyprune_;
      const int is = (iy / xyprune_) * xyprune_;
      const int in = is + xyprune_;

      const auto dx = (ix - iw) / static_cast<double>(xyprune_);
      const auto dy = (iy - is) / static_cast<double>(xyprune_);

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

#endif
