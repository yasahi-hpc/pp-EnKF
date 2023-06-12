#ifndef __LBM2D_HPP__
#define __LBM2D_HPP__

#include <string>
#include <vector>
#include <map>
#include <sys/stat.h>
#include <fstream>
#include <cstdio>
#include <iomanip>
#include <utils/string_utils.hpp>
#include <utils/system_utils.hpp>
#include <executors/Parallel_For.hpp>
#include <executors/Parallel_Reduce.hpp>
#include <Random.hpp>
#include "../functors.hpp"
#include "../config.hpp"
#include "force.hpp"
#include "models.hpp"
#include "types.hpp"

class LBM2D : public Model {
private:
  bool is_master_ = true;
  bool is_reference_ = true;

  // Variables used only in this class
  RealView2D vor_, nu_;
  RealView2D vor_obs_;
  RealView2D noise_;

  // Observation
  Impl::Random<double> rand_;

  // Force term
  std::unique_ptr<Force> force_;

  // IO
  std::map<std::string, std::string> directory_names_;
  
public:
  LBM2D()=delete;
  LBM2D(Config& conf) : Model(conf) {}
  ~LBM2D() override {}

public:
  // Methods
  void initialize(std::unique_ptr<DataVars>& data_vars) {
    // tmp val for stream function
    auto [nx, ny] = conf_.settings_.n_;
    double p_amp = conf_.phys_.p_amp_;
    int nkx = nx/2;
    double k0 = conf_.phys_.kf_;
    double dk = conf_.phys_.dk_;
    double sigma = conf_.phys_.sigma_;
    int ensemble_id = conf_.settings_.ensemble_idx_;
    is_master_ = (ensemble_id == 0);

    // Allocate data
    vor_ = RealView2D("vor", nx, ny);
    nu_  = RealView2D("nu", nx, ny);
    vor_obs_ = RealView2D("vor_obs", nx, ny);
    noise_   = RealView2D("noise", nx, ny);

    RealView2D p("p", nkx, nkx);
    RealView2D theta("theta", nkx, nkx);

    // init val (mainly on host)
    auto rand_engine = std::mt19937(ensemble_id);
    auto rand_dist = std::uniform_real_distribution<double>(-1, 1);
    auto rand = [&]() { return rand_dist(rand_engine); };
    for(int iky=0; iky<nkx; iky++) {
      for(int ikx=0; ikx<nkx; ikx++) {
        const double k = std::sqrt(static_cast<double>(ikx*ikx + iky*iky));
        const auto pk = (k0-dk <= k and k <= k0+dk)
          ? std::exp( - (k-k0)*(k-k0) / sigma ) 
          : 0; // forced maltrud91
        p(ikx, iky) = pk;
        theta(ikx, iky) = rand() * M_PI;
      }
    }
    p.updateDevice();
    theta.updateDevice();

    // Initialize variables on device
    auto rho = data_vars->rho().mdspan();
    auto u   = data_vars->u().mdspan();
    auto v   = data_vars->v().mdspan();
    const auto _p  = p.mdspan();
    const auto _theta = theta.mdspan();
    double rho_ref = conf_.phys_.rho_ref_;

    auto init_fluid_moments = [=] MDSPAN_FORCE_INLINE_FUNCTION (const int ix, const int iy) {
      // fluid
      double u_tmp = 0;
      double v_tmp = 0;
      for(int iky=0; iky<nkx; iky++) {
        const double ky = 2 * M_PI * iky / ny;
        for(int ikx=0; ikx<nkx; ikx++) {
          const double kx = 2 * M_PI * ikx / nx;
          const auto p_tmp = _p(ikx, iky);
          const auto theta_tmp = _theta(ikx, iky);
          u_tmp += p_tmp * ky    * cos(kx * ix + ky * iy + theta_tmp);
          v_tmp += p_tmp * (-kx) * cos(kx * ix + ky * iy + theta_tmp);
        }
      }
      rho(ix, iy) = rho_ref;
      u(ix, iy) = u_tmp;
      v(ix, iy) = v_tmp;
    };
    Iterate_policy<2> policy2d({0, 0}, {nx, ny});
    Impl::for_each(policy2d, init_fluid_moments);

    // Modify u, v by max. vs cfl
    double u_ref = conf_.phys_.u_ref_;
    double vmax = 1e-30 * u_ref;
    auto max_speed = [=](const int ix, const int iy) {
      auto _u = u(ix, iy);
      auto _v = v(ix, iy);
      return std::sqrt(_u*_u + _v*_v);
    };

    auto max_operator = 
      [=](const auto& lhs, const auto& rhs) { return std::max(lhs, rhs); };

    Impl::transform_reduce(policy2d, max_operator, max_speed, vmax);
    Impl::for_each(policy2d,
      [=](const int ix, const int iy) { 
        u(ix, iy) *= u_ref / vmax * p_amp;
        v(ix, iy) *= u_ref / vmax * p_amp;
    });

    // Compute feq
    auto Q = conf_.phys_.Q_;
    auto f = data_vars->f().mdspan();
    Iterate_policy<3> policy3d({0, 0, 0}, {nx, ny, Q});
    Impl::for_each(policy3d, init_feq_functor(conf_, rho, u, v, f));

    // Initialize force term
    force_ = std::move( std::unique_ptr<Force>(new Force(conf_)) );

    // Initialize IO
    const std::string prefix = "io";

    // Create directories if not exist
    std::vector<std::string> dirs({"calc", "nature", "observed"});
    for(auto dir_name : dirs) {
      std::string full_path = prefix + "/" + dir_name + "/ens" + Impl::zfill(ensemble_id);
      directory_names_[dir_name] = full_path;
    }

    Impl::mkdirs(directory_names_.at("calc"), 0755);
    if(is_master_) {
      Impl::mkdirs(directory_names_.at("nature"), 0755);
      Impl::mkdirs(directory_names_.at("observed"), 0755);
    }

    is_reference_ = conf_.settings_.is_reference_;
  }

  void reset(std::unique_ptr<DataVars>& data_vars, const std::string mode) {
    // Always reset counts
    it_ = 0;
    diag_it_ = 0;

    if(mode == "purturbulate") {
      purturbulate(data_vars);
    }
  }

  void solve(std::unique_ptr<DataVars>& data_vars) {
    // Update force
    update_forces();

    streaming_macroscopic(data_vars);

    // Large Eddy Simulation
    sgs(data_vars);

    // Collision
    collision(data_vars);

    // Swap and increment count
    auto& f  = data_vars->f();
    auto& fn = data_vars->fn();
    fn.swap(f);

    it_++;
  }

  void diag(std::unique_ptr<DataVars>& data_vars){
    /* 
     * 0. Nature run or perturbed run (as reference)
     *    Save rho, u, v and vor into /nature (as is) and /observed (with noise)
     *
     * 1. Run with DA
     *    Save rho, u, v and vor into /calc (as is)
     *
     * */
    if(it_ % conf_.settings_.io_interval_ != 0) return;
    if(is_master_) inspect(data_vars);

    // Save values calculated by this ensemble member
    // Save simulation results without noises
    std::string sim_result_name = is_reference_ ? "nature" : "calc";
    auto rho = data_vars->rho();
    auto u = data_vars->u();
    auto v = data_vars->v();
    save_to_files(sim_result_name, rho, u, v, it_);

    // Save noisy results
    if(is_reference_) {
      observe(data_vars); // adding noise to u_obs_, v_obs_, rho_obs_
      auto rho_obs = data_vars->rho_obs();
      auto u_obs = data_vars->u_obs();
      auto v_obs = data_vars->v_obs();
      save_to_files("observed", rho_obs, u_obs, v_obs, it_);
    }
  }

  void finalize() {}

private:
  // Related to initialization
  void purturbulate(std::unique_ptr<DataVars>& data_vars) {
    auto [nx, ny] = conf_.settings_.n_;
    const auto Q = conf_.phys_.Q_;
    const auto epsilon = conf_.settings_.ly_epsilon_;
    const int ensemble_idx = conf_.settings_.ensemble_idx_ + 132333;

    std::mt19937 engine(ensemble_idx);
    std::normal_distribution<double> dist(0, 1);

    // f with purturbulation
    //     purturbulation using the context of newtonian nudging
    auto _f = data_vars->f(); // Access via Views
    auto _rho = data_vars->rho();

    auto feq = [=, this](double rho, double u, double v, int q) {
      auto c = conf_.settings_.c_;
      auto uu = (u*u + v*v) / (c*c);
      auto uc = (u * conf_.phys_.q_[0][q] + v * conf_.phys_.q_[1][q]) / c;
      return conf_.phys_.weights_[q] * rho * (1.0 + 3.0 * uc + 4.5 * uc * uc - 1.5 * uu);
    };

    _f.updateSelf();
    _rho.updateSelf();
    for(int iy=0; iy<ny; iy++) {
      for(int ix=0; ix<nx; ix++) {
        const auto rho_tmp = _rho(ix, iy);
        const auto u_tmp = conf_.phys_.u_ref_ * dist(engine);
        const auto v_tmp = conf_.phys_.u_ref_ * dist(engine);

        for(int q=0; q<Q; q++) {
          const auto fo = feq(rho_tmp, u_tmp, v_tmp, q);
          _f(ix, iy, q) = epsilon * fo + (1-epsilon) * _f(ix, iy, q);
        }
      }
    }
    _f.updateDevice();
  }

  // Related to computations
private:
  void update_forces() {
    force_->update_forces();
  }

  void streaming(std::unique_ptr<DataVars>& data_vars) {
    auto [nx, ny] = conf_.settings_.n_;
    const auto Q = conf_.phys_.Q_;
    const auto f = data_vars->f().mdspan();
    auto      fn = data_vars->fn().mdspan();

    Iterate_policy<3> policy3d({0, 0, 0}, {nx, ny, Q});
    Impl::for_each(policy3d, streaming_functor(conf_, f, fn));
  }

  void macroscopic(std::unique_ptr<DataVars>& data_vars) {
    auto [nx, ny] = conf_.settings_.n_;
    const auto fn = data_vars->fn().mdspan();
    auto rho = data_vars->rho().mdspan();
    auto u   = data_vars->u().mdspan();
    auto v   = data_vars->v().mdspan();

    Iterate_policy<2> policy2d({0, 0}, {nx, ny});
    Impl::for_each(policy2d, macroscopic_functor(conf_, fn, rho, u, v));
  }

  void streaming_macroscopic(std::unique_ptr<DataVars>& data_vars) {
    auto [nx, ny] = conf_.settings_.n_;
    const auto f  = data_vars->f().mdspan();
    auto fn  = data_vars->fn().mdspan();
    auto rho = data_vars->rho().mdspan();
    auto u   = data_vars->u().mdspan();
    auto v   = data_vars->v().mdspan();

    Iterate_policy<2> policy2d({0, 0}, {nx, ny});
    Impl::for_each(policy2d, streaming_macroscopic_functor(conf_, f, fn, rho, u, v));
  }

  void sgs(std::unique_ptr<DataVars>& data_vars) {
    auto [nx, ny] = conf_.settings_.n_;
    const auto rho = data_vars->rho().mdspan();
    const auto u   = data_vars->u().mdspan();
    const auto v   = data_vars->v().mdspan();
    auto nu  = nu_.mdspan();

    Iterate_policy<2> policy2d({0, 0}, {nx, ny});
    Impl::for_each(policy2d, sgs_functor(conf_, rho, u, v, nu));
  }

  void collision(std::unique_ptr<DataVars>& data_vars) {
    auto [nx, ny] = conf_.settings_.n_;
    auto fn = data_vars->fn().mdspan();
    const auto fx = force_->fx().mdspan();
    const auto fy = force_->fy().mdspan();
    const auto rho = data_vars->rho().mdspan();
    const auto u   = data_vars->u().mdspan();
    const auto v   = data_vars->v().mdspan();
    const auto nu  = nu_.mdspan();

    Iterate_policy<2> policy2d({0, 0}, {nx, ny});
    Impl::for_each(policy2d, collision_srt_functor(conf_, fn, fx, fy, rho, u, v, nu));
  }

private:
  void inspect(std::unique_ptr<DataVars>& data_vars) {
    auto [nx, ny] = conf_.settings_.n_;
    auto dx = conf_.settings_.dx_;
    auto u_ref = conf_.phys_.u_ref_;

    data_vars->rho().updateSelf();
    data_vars->u().updateSelf();
    data_vars->v().updateSelf();
    nu_.updateSelf();
    auto rho = data_vars->rho().mdspan();
    auto u   = data_vars->u().mdspan();
    auto v   = data_vars->v().mdspan();
    const auto nu = nu_.mdspan();

    using moment_type = std::tuple<double, double, double, double, double, double, double, double, double>;
    moment_type moments = {0, 0, 0, 0, 0, 0, 0, 0, 0};

    auto moment_kernel = 
      [=](const int ix, const int iy) {
        auto tmp_rho = rho(ix, iy);
        auto tmp_u   = u(ix, iy);
        auto tmp_v   = v(ix, iy);

        auto momentum_x = tmp_rho * tmp_u;
        auto momentum_y = tmp_rho * tmp_v;
        auto energy     = tmp_u * tmp_u + tmp_v * tmp_v;
        auto nus        = nu(ix, iy);
        auto mass       = tmp_rho;

        // derivatives
        const auto ixp1 = periodic(ix+1, nx);
        const auto ixm1 = periodic(ix-1, nx);
        const auto iyp1 = periodic(iy+1, ny);
        const auto iym1 = periodic(iy-1, ny);

        const auto ux = (u(ixp1, iy) - u(ixm1, iy)) / (2*dx);
        const auto uy = (u(ix, iyp1) - u(ix, iym1)) / (2*dx);
        const auto vx = (v(ixp1, iy) - v(ixm1, iy)) / (2*dx);
        const auto vy = (v(ix, iyp1) - v(ix, iym1)) / (2*dx);

        const auto enstrophy = ( (vx - uy) * (vx - uy) + (ux + vy) * (ux + vy) ) / 2;
        const auto divu = ux + vy;
        const auto divu2 = (ux + vy) * (ux + vy);
        const auto vel2 = tmp_u * tmp_u + tmp_v * tmp_v;

        return moment_type {momentum_x, momentum_y, energy, enstrophy, nus, mass, divu2, divu, vel2};
    };

    auto sum_operator =
      [=] (const moment_type& left, const moment_type& right) {
        return moment_type {std::get<0>(left) + std::get<0>(right),
                            std::get<1>(left) + std::get<1>(right),
                            std::get<2>(left) + std::get<2>(right),
                            std::get<3>(left) + std::get<3>(right),
                            std::get<4>(left) + std::get<4>(right),
                            std::get<5>(left) + std::get<5>(right),
                            std::get<6>(left) + std::get<6>(right),
                            std::get<7>(left) + std::get<7>(right),
                            std::get<8>(left) + std::get<8>(right)
                           };
    };

    Iterate_policy<2> policy2d({0, 0}, {nx, ny});
    Impl::transform_reduce(policy2d, sum_operator, moment_kernel, moments);

    /* [FIX THIS] transform reduce to get multiple max elements does not work correctly???
    using maximum_type = std::tuple<double, double, double>;
    maximum_type maximums = {0, 0, 0};
    // Compute maximum
    auto maximum_kernel = 
      [=](const int ix, const int iy) {
        auto tmp_rho = rho(ix, iy);
        auto tmp_u   = u(ix, iy);
        auto tmp_v   = v(ix, iy);

        // derivatives
        const auto ixp1 = periodic(ix+1, nx);
        const auto ixm1 = periodic(ix-1, nx);
        const auto iyp1 = periodic(iy+1, ny);
        const auto iym1 = periodic(iy-1, ny);

        const auto ux = (u(ixp1, iy) - u(ixm1, iy)) / (2*dx);
        const auto uy = (u(ix, iyp1) - u(ix, iym1)) / (2*dx);
        const auto vx = (v(ixp1, iy) - v(ixm1, iy)) / (2*dx);
        const auto vy = (v(ix, iyp1) - v(ix, iym1)) / (2*dx);

        auto maxdivu = std::abs(ux + vy);
        auto maxvel2 = tmp_u * tmp_u + tmp_v * tmp_v;

        return maximum_type {maxdivu, maxvel2, tmp_rho};
    };

    auto max_operator =
      [=] (const maximum_type& left, const maximum_type& right) {
        return maximum_type {std::max( std::get<0>(left), std::get<0>(right) ),
                             std::max( std::get<1>(left), std::get<1>(right) ),
                             std::max( std::get<2>(left), std::get<2>(right) )
                            };
    };
    Impl::transform_reduce(policy2d, max_operator, maximum_kernel, maximums);

    // Compute minimum
    double rho_min = 9999; // some large number
    auto minimum_kernel = 
      [=](const int ix, const int iy) { return rho(ix, iy); };

    auto min_operator =
      [=] (const auto& left, const auto& right) { return std::min(left, right); };
    Impl::transform_reduce(policy2d, min_operator, minimum_kernel, rho_min);
    auto maxvel2 = std::get<0>(maximums);
    auto maxdivu = std::get<1>(maximums);
    auto rho_max = std::get<2>(maximums);
    */

    // To be removed
    double maxdivu = 0;
    double maxvel2 = 0;
    double rho_max = 0;
    double rho_min = 9999;

    auto _rho = data_vars->rho();
    auto _u = data_vars->u();
    auto _v = data_vars->v();

    _rho.updateSelf();
    _u.updateSelf();
    _v.updateSelf();
    for(int iy=0; iy<ny; iy++) {
      for(int ix=0; ix<nx; ix++) {
        auto tmp_rho = _rho(ix, iy);
        auto tmp_u   = _u(ix, iy);
        auto tmp_v   = _v(ix, iy);

        // derivatives
        const int ixp1 = periodic(ix+1, nx);
        const int ixm1 = periodic(ix-1, nx);
        const int iyp1 = periodic(iy+1, ny);
        const int iym1 = periodic(iy-1, ny);

        const auto ux = (_u(ixp1, iy) - _u(ixm1, iy)) / (2*dx);
        const auto uy = (_u(ix, iyp1) - _u(ix, iym1)) / (2*dx);
        const auto vx = (_v(ixp1, iy) - _v(ixm1, iy)) / (2*dx);
        const auto vy = (_v(ix, iyp1) - _v(ix, iym1)) / (2*dx);

        maxdivu = std::max(maxdivu, std::abs(ux + vy));
        maxvel2 = std::max(maxvel2, tmp_u * tmp_u + tmp_v * tmp_v);
        rho_max = std::max(rho_max, tmp_rho);
        rho_min = std::min(rho_min, tmp_rho);
      }
    }
    auto momentum_x_total = std::get<0>(moments) / (nx * ny);
    auto momentum_y_total = std::get<1>(moments) / (nx * ny);
    auto energy           = std::get<2>(moments) / (nx * ny);
    auto enstrophy        = std::get<3>(moments) / (nx * ny);
    auto nus_total        = std::get<4>(moments) / (nx * ny);
    auto mass             = std::get<5>(moments) / (nx * ny);
    auto divu2            = std::get<6>(moments) / (nx * ny);
    auto divu             = std::get<7>(moments) / (nx * ny);
    auto vel2             = std::get<8>(moments) / (nx * ny);

    std::cout << std::scientific << std::setprecision(16) << std::flush;
    std::cout << " RMS, max speed: " << std::sqrt(vel2) << ", " << std::sqrt(maxvel2) << " [m/s]" << std::endl;
    //std::cout << " mean energy: " << energy << " [m2/s2]" << std::endl;
    //std::cout << " mean enstrophy: " << enstrophy << " [/s2]" << std::endl;
    std::cout << " mean les visc: " << nus_total << " [m2/s]" << std::endl;
    std::cout << " mean mass: " << mass << " [kg]" << std::endl;
    std::cout << " delta_rho: " << rho_max - rho_min << " [kg/m3]" << std::endl;
    std::cout << " max vel divergence (times dx, devided by uref): " << maxdivu * dx / u_ref << " []" << std::endl;

    std::cout << std::resetiosflags(std::ios_base::floatfield);
    std::cout << std::endl;
  }

  // Related to diagnostics
  template <class ViewType>
  void update_vorticity(const ViewType& u, const ViewType& v, ViewType& vor) {
    auto [nx, ny] = conf_.settings_.n_;
    const auto u_tmp = u.mdspan();
    const auto v_tmp = v.mdspan();
    auto vor_tmp = vor.mdspan();

    Iterate_policy<2> policy2d({0, 0}, {nx, ny});
    Impl::for_each(policy2d, vorticity_functor(conf_, u_tmp, v_tmp, vor_tmp));
  }

  template <class ViewType>
  void add_noise(const ViewType& value, ViewType& noisy_value, const double error=0.0) {
    auto [nx, ny] = conf_.settings_.n_;
    const auto value_tmp = value.mdspan();
    auto noisy_value_tmp = noisy_value.mdspan();
    const auto noise_tmp = noise_.mdspan();

    const double mean = 0.0, stddev = 1.0;
    rand_.normal(noise_.data(), nx*ny, mean, stddev);

    Iterate_policy<2> policy2d({0, 0}, {nx, ny});
    Impl::for_each(policy2d, 
      [=](const int ix, const int iy) {
        noisy_value_tmp(ix, iy) = value_tmp(ix, iy) + error * noise_tmp(ix, iy);
      });
  }

  void observe(std::unique_ptr<DataVars>& data_vars) {
    /* Update rho_obs, u_obs and v_obs */
    const auto rho = data_vars->rho();
    const auto u   = data_vars->u();
    const auto v   = data_vars->v();
    auto rho_obs = data_vars->rho_obs();
    auto u_obs   = data_vars->u_obs();
    auto v_obs   = data_vars->v_obs();

    add_noise(rho, rho_obs, conf_.phys_.obs_error_rho_);
    add_noise(u,   u_obs,   conf_.phys_.obs_error_u_);
    add_noise(v,   v_obs,   conf_.phys_.obs_error_u_);
  }

  template <class ViewType>
  void save_to_files(std::string case_name, ViewType& rho, ViewType& u, ViewType& v, const int it) {
    update_vorticity(u, v, vor_);
    to_file(case_name, rho, it);
    to_file(case_name, u,   it);
    to_file(case_name, v,   it);
    to_file(case_name, vor_, it);
  }

  template <class ViewType>
  void to_file(std::string case_name, ViewType& value, const int it) {
    auto dir_name = directory_names_.at(case_name);
    value.updateSelf();
    std::string file_name = dir_name + "/" + value.name() + "_step"
                          + Impl::zfill(it / conf_.settings_.io_interval_, 10) + ".dat";
    Impl::to_binary(file_name, value.host_mdspan());
  }
};

#endif
