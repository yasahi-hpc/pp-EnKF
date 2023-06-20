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
#include <stdpar/Parallel_For.hpp>
#include <stdpar/Parallel_Reduce.hpp>
#include <Random.hpp>
#include "../functors.hpp"
#include "../config.hpp"
#include "force.hpp"
#include "models.hpp"
#include "types.hpp"

class LBM2D : public Model {
private:
  using value_type = RealView2D::value_type;
  bool is_master_ = true;
  bool is_reference_ = true;

  // Variables used only in this class
  RealView2D vor_, nu_;
  RealView2D vor_obs_;
  RealView2D noise_;

  // Observation
  Impl::Random<value_type> rand_;

  // Force term
  std::unique_ptr<Force> force_;

  // IO
  std::map<std::string, std::string> directory_names_;
  
public:
  LBM2D()=delete;
  LBM2D(Config& conf, IOConfig& io_conf) : Model(conf, io_conf) {}
  ~LBM2D() override {}

public:
  // Methods
  void initialize(std::unique_ptr<DataVars>& data_vars) {
    // tmp val for stream function
    auto [nx, ny] = conf_.settings_.n_;
    value_type p_amp = static_cast<value_type>(conf_.phys_.p_amp_);
    int nkx = nx/2;
    value_type k0 = static_cast<value_type>(conf_.phys_.kf_);
    value_type dk = static_cast<value_type>(conf_.phys_.dk_);
    value_type sigma = static_cast<value_type>(conf_.phys_.sigma_);
    int ensemble_id = conf_.settings_.ensemble_idx_;
    is_master_ = (ensemble_id == 0);
    is_reference_ = conf_.settings_.is_reference_;
    if(!is_reference_) {
      ensemble_id += 334; // For DA case, starting with different initial condition
    }

    // Allocate data
    vor_     = RealView2D("vor", nx, ny);
    nu_      = RealView2D("nu", nx, ny);
    vor_obs_ = RealView2D("vor_obs", nx, ny);
    noise_   = RealView2D("noise", nx, ny);

    RealView2D p("p", nkx, nkx);
    RealView2D theta("theta", nkx, nkx);

    // init val (mainly on host)
    auto rand_engine = std::mt19937(ensemble_id);
    auto rand_dist = std::uniform_real_distribution<value_type>(-1, 1);
    auto rand = [&]() { return rand_dist(rand_engine); };
    for(int iky=0; iky<nkx; iky++) {
      for(int ikx=0; ikx<nkx; ikx++) {
        const value_type k = std::sqrt(static_cast<value_type>(ikx*ikx + iky*iky));
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
    value_type rho_ref = static_cast<value_type>(conf_.phys_.rho_ref_);

    auto init_fluid_moments = [=](const int ix, const int iy) {
      // fluid
      value_type u_tmp = 0.0;
      value_type v_tmp = 0.0;
      for(int iky=0; iky<nkx; iky++) {
        const value_type ky = 2 * M_PI * iky / ny;
        for(int ikx=0; ikx<nkx; ikx++) {
          const value_type kx = 2 * M_PI * ikx / nx;
          const auto p_tmp     = _p(ikx, iky);
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
    value_type u_ref = static_cast<value_type>(conf_.phys_.u_ref_);
    value_type vmax  = 1e-30 * u_ref;
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
    const std::string out_dir = io_conf_.base_dir_ + "/" + io_conf_.case_name_;

    // Create directories if not exist
    std::vector<std::string> dirs({"calc"});
    if(is_reference_) dirs.push_back("observed");

    for(auto dir_name : dirs) {
      std::string full_path = out_dir + "/" + dir_name + "/ens" + Impl::zfill(conf_.settings_.ensemble_idx_);
      directory_names_[dir_name] = full_path;
    }

    Impl::mkdirs(directory_names_.at("calc"), 0755);
    if(is_master_ && is_reference_) {
      Impl::mkdirs(directory_names_.at("observed"), 0755);
    }
  }

  void reset(std::unique_ptr<DataVars>& data_vars, const std::string mode) {
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
  }

  void diag(std::unique_ptr<DataVars>& data_vars, const int it){
    /* 
     * 0. Nature run or perturbed run (as reference)
     *    Save rho, u, v and vor into /nature (as is) and /observed (with noise)
     *
     * 1. Run with DA
     *    Save rho, u, v and vor into /calc (as is)
     *
     * */
    if(it % conf_.settings_.io_interval_ != 0) return;
    if(is_master_) inspect(data_vars, it);

    // Save values calculated by this ensemble member
    // Save simulation results without noises
    std::string sim_result_name = "calc";
    auto rho = data_vars->rho();
    auto u = data_vars->u();
    auto v = data_vars->v();
    save_to_files(sim_result_name, rho, u, v, it);

    // Save noisy results
    if(is_reference_) {
      observe(data_vars); // adding noise to u_obs_, v_obs_, rho_obs_
      auto rho_obs = data_vars->rho_obs();
      auto u_obs = data_vars->u_obs();
      auto v_obs = data_vars->v_obs();
      save_to_files("observed", rho_obs, u_obs, v_obs, it);
    }
  }

  void finalize() {}

private:
  // Related to initialization
  void purturbulate(std::unique_ptr<DataVars>& data_vars) {
    auto [nx, ny] = conf_.settings_.n_;
    const auto Q = conf_.phys_.Q_;
    const value_type epsilon = static_cast<value_type>(conf_.settings_.ly_epsilon_);
    const int ensemble_idx = conf_.settings_.ensemble_idx_ + 132333;

    auto rand_engine = std::mt19937(ensemble_idx);
    auto rand_dist = std::normal_distribution<value_type>(0, 1);
    auto rand = [&]() { return rand_dist(rand_engine); };

    // f with purturbulation
    //     purturbulation using the context of newtonian nudging
    auto _f   = data_vars->f(); // Access via Views
    auto _rho = data_vars->rho();
    value_type c = static_cast<value_type>(conf_.settings_.c_);

    auto feq = [=, this](value_type rho, value_type u, value_type v, int q) {
      value_type weight = static_cast<value_type>(conf_.phys_.weights_[q]);
      value_type qx = static_cast<value_type>(conf_.phys_.q_[0][q]);
      value_type qy = static_cast<value_type>(conf_.phys_.q_[1][q]);
      value_type uu = (u*u + v*v) / (c*c);
      value_type uc = (u*qx + v*qy) / c;
      return weight * rho * (1.0 + 3.0 * uc + 4.5 * uc * uc - 1.5 * uu);
    };

    _f.updateSelf();
    _rho.updateSelf();
    for(int iy=0; iy<ny; iy++) {
      for(int ix=0; ix<nx; ix++) {
        const auto rho_tmp = _rho(ix, iy);
        const value_type u_tmp = static_cast<value_type>(conf_.phys_.u_ref_) * rand();
        const value_type v_tmp = static_cast<value_type>(conf_.phys_.u_ref_) * rand();

        for(int q=0; q<Q; q++) {
          const auto fo = feq(rho_tmp, u_tmp, v_tmp, q);
          _f(ix, iy, q) = epsilon * fo + (1.0-epsilon) * _f(ix, iy, q);
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
  void inspect(std::unique_ptr<DataVars>& data_vars, const int it) {
    auto [nx, ny] = conf_.settings_.n_;
    value_type dx    = static_cast<value_type>(conf_.settings_.dx_);
    value_type u_ref = static_cast<value_type>(conf_.phys_.u_ref_);

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

        double momentum_x = tmp_rho * tmp_u;
        double momentum_y = tmp_rho * tmp_v;
        double energy     = tmp_u * tmp_u + tmp_v * tmp_v;
        double nus        = nu(ix, iy);
        double mass       = tmp_rho;

        // derivatives
        const auto ixp1 = periodic(ix+1, nx);
        const auto ixm1 = periodic(ix-1, nx);
        const auto iyp1 = periodic(iy+1, ny);
        const auto iym1 = periodic(iy-1, ny);

        const auto ux = (u(ixp1, iy) - u(ixm1, iy)) / (2*dx);
        const auto uy = (u(ix, iyp1) - u(ix, iym1)) / (2*dx);
        const auto vx = (v(ixp1, iy) - v(ixm1, iy)) / (2*dx);
        const auto vy = (v(ix, iyp1) - v(ix, iym1)) / (2*dx);

        const double enstrophy = ( (vx - uy) * (vx - uy) + (ux + vy) * (ux + vy) ) / 2;
        const double divu = ux + vy;
        const double divu2 = (ux + vy) * (ux + vy);
        const double vel2 = tmp_u * tmp_u + tmp_v * tmp_v;

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
    value_type maxdivu = 0;
    value_type maxvel2 = 0;
    value_type rho_max = 0;
    value_type rho_min = 9999;

    for(int iy=0; iy<ny; iy++) {
      for(int ix=0; ix<nx; ix++) {
        auto tmp_rho = rho(ix, iy);
        auto tmp_u   = u(ix, iy);
        auto tmp_v   = v(ix, iy);

        // derivatives
        const int ixp1 = periodic(ix+1, nx);
        const int ixm1 = periodic(ix-1, nx);
        const int iyp1 = periodic(iy+1, ny);
        const int iym1 = periodic(iy-1, ny);

        const value_type ux = (u(ixp1, iy) - u(ixm1, iy)) / (2*dx);
        const value_type uy = (u(ix, iyp1) - u(ix, iym1)) / (2*dx);
        const value_type vx = (v(ixp1, iy) - v(ixm1, iy)) / (2*dx);
        const value_type vy = (v(ix, iyp1) - v(ix, iym1)) / (2*dx);

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
    std::cout << " it/nbiter: " << it << "/" << conf_.settings_.nbiter_ << std::endl;
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
  void add_noise(const ViewType& value, ViewType& noisy_value, const value_type error=0.0) {
    auto [nx, ny] = conf_.settings_.n_;
    const auto value_tmp = value.mdspan();
    auto noisy_value_tmp = noisy_value.mdspan();
    const auto noise_tmp = noise_.mdspan();

    const value_type mean = 0.0, stddev = 1.0;
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
                          + Impl::zfill(it, 10) + ".dat";
    Impl::to_binary(file_name, value.mdspan());
  }
};

#endif
