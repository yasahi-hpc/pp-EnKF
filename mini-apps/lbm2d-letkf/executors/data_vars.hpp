#ifndef __DATAVARS_HPP__
#define __DATAVARS_HPP__

#include "types.hpp"
#include "../config.hpp"

struct DataVars {
private:
  // Variables used for simulation
  RealView3D f_;
  RealView3D fn_;
  RealView2D rho_;
  RealView2D u_;
  RealView2D v_;

  // Variables used for data assimilation
  RealView2D rho_obs_;
  RealView2D u_obs_;
  RealView2D v_obs_;

public:
  DataVars()=delete;
  DataVars(Config& conf) {
    auto [nx, ny] = conf.settings_.n_;
    auto Q = conf.phys_.Q_;
    f_ = RealView3D("f", nx, ny, Q);
    fn_ = RealView3D("fn", nx, ny, Q);
    rho_ = RealView2D("rho", nx, ny);
    u_   = RealView2D("u", nx, ny);
    v_   = RealView2D("v", nx, ny);

    rho_obs_ = RealView2D("rho_obs", nx, ny);
    u_obs_   = RealView2D("u_obs", nx, ny);
    v_obs_   = RealView2D("v_obs", nx, ny);
  }

  ~DataVars() {}

  // Getters
  auto& f() { return f_; }
  auto& fn() { return fn_; }
  auto& rho() { return rho_; }
  auto& u() { return u_; }
  auto& v() { return v_; }

  auto& rho_obs() { return rho_obs_; }
  auto& u_obs() { return u_obs_; }
  auto& v_obs() { return v_obs_; }
};

#endif
