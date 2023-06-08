#ifndef __SOLVER_HPP__
#define __SOLVER_HPP__

#include <string>
#include <vector>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <utils/commandline_utils.hpp>
#include "../config.hpp"
#include "../mpi_config.hpp"
#include "models.hpp"
#include "model_factories.hpp"
#include "data_vars.hpp"

using json = nlohmann::json;

class Solver {
  MPIConfig mpi_conf_;
  Config conf_;
  std::string case_name_;
  std::unique_ptr<Model> model_;
  std::unique_ptr<DA_Model> da_model_;
  std::unique_ptr<DataVars> data_vars_;

public:
  void initialize(int* argc, char*** argv){
    // Load args
    auto kwargs = Impl::parse(*argc, *argv);
    std::string filename = Impl::get(kwargs, "filename", "nature.json");

    // Initialize MPI
    mpi_conf_.initialize(argc, argv);

    // Initialize Configuration from the input json file
    initialize_conf(filename, conf_);

    // Allocate attributes
    data_vars_ = std::move( std::unique_ptr<DataVars>(new DataVars(conf_)) );
    model_     = std::move( model_factory(case_name_, conf_) );
    da_model_  = std::move( da_model_factory(case_name_, conf_, mpi_conf_) );

    model_->initialize(data_vars_);
    da_model_->initialize();

    // spinup
    for(int it=0; it<conf_.settings_.spinup_; it++) {
      model_->solve(data_vars_);
    }
    model_->reset(data_vars_, "count");
    if(mpi_conf_.is_master()) {
      std::cout << "spin-up finished" << std::endl;
    }

    if(conf_.settings_.lyapnov_) {
      model_->reset(data_vars_, "purturbulate");
    }
  };

  void run(){
    for(int it=0; it<conf_.settings_.nbiter_; it++) {
      da_model_->apply(data_vars_, it);
      model_->diag(data_vars_);
      model_->solve(data_vars_);
    }
  }

  void finalize(){
    mpi_conf_.finalize();
  }

private:
  void initialize_conf(std::string& filename, Config& conf) {
    std::ifstream f(filename);
    assert(f.is_open());
    json json_data = json::parse(f);
    if(mpi_conf_.is_master()) {
      std::cout << "Input: \n" << json_data.dump(4) << std::endl;
    }

    // Set Physics
    case_name_                 = json_data["Physics"]["case_name"].get<std::string>();
    conf_.phys_.nu_            = json_data["Physics"]["nu"].get<double>();
    conf_.phys_.friction_rate_ = json_data["Physics"]["friction_rate"].get<double>();
    conf_.phys_.kf_            = json_data["Physics"]["kf"].get<double>();
    conf_.phys_.fkf_           = json_data["Physics"]["fkf"].get<double>();
    conf_.phys_.dk_            = json_data["Physics"]["dk"].get<double>();
    conf_.phys_.sigma_         = json_data["Physics"]["sigma"].get<double>();
    conf_.phys_.p_amp_         = json_data["Physics"]["p_amp"].get<double>();
    conf_.phys_.obs_error_rho_ = json_data["Physics"]["obs_error_rho"].get<double>();
    conf_.phys_.obs_error_u_   = json_data["Physics"]["obs_error_u"].get<double>();

    // Set settings
    conf_.settings_.spinup_       = json_data["Settings"]["spinup"].get<int>();
    conf_.settings_.nbiter_       = json_data["Settings"]["nbiter"].get<int>();
    conf_.settings_.io_interval_  = json_data["Settings"]["io_interval"].get<int>();
    conf_.settings_.obs_interval_ = json_data["Settings"]["obs_interval"].get<int>();
    conf_.settings_.da_interval_  = json_data["Settings"]["da_interval"].get<int>();
    conf_.settings_.lyapnov_      = json_data["Settings"]["lyapnov"].get<bool>();
    conf_.settings_.is_les_       = json_data["Settings"]["les"].get<bool>();
    conf_.settings_.da_nud_rate_  = json_data["Settings"]["da_nud_rate"].get<double>();

    if(case_name_ == "letkf") {
      conf_.settings_.rloc_len_ = json_data["Settings"]["rloc_len"].get<int>();
      conf_.settings_.beta_     = json_data["Settings"]["beta"].get<double>();
    }
    conf_.settings_.is_reference_ = (case_name_ == "nature");

    auto nx = json_data["Settings"]["nx"].get<int>();
    auto ny = json_data["Settings"]["ny"].get<int>();
    conf_.settings_.n_ = {nx, ny};

    auto c  = conf_.phys_.u_ref_ / conf_.settings_.cfl_;
    auto dx = conf_.phys_.h_ref_ / static_cast<double>(ny-1);
    auto dt = dx / c;
    const auto tau = 0.5 + 3. * conf_.phys_.nu_ / (c*c) / dt;
    const auto omega = 1.0 / tau;
    conf_.settings_.c_ = c;
    conf_.settings_.dx_ = dx;
    conf_.settings_.dt_ = dt;
    conf_.settings_.tau_ = tau;
    conf_.settings_.omega_ = omega;

    // Set ensemble idx from mpi rank
    conf_.settings_.ensemble_idx_ = mpi_conf_.rank();

    // Print settings
    auto nu    = conf_.phys_.nu_;
    auto u_ref = conf_.phys_.u_ref_;
    auto h_ref = conf_.phys_.h_ref_;
    auto io_interval = conf_.settings_.io_interval_;

    if(mpi_conf_.is_master()) {
      std::cout
          << "  nx = " << nx << std::endl
          << "  nu = " << nu << " m2/s" << std::endl
          << "  u_ref = " << u_ref << " m/s" << std::endl
          << "  h_ref = " << h_ref << " meter" << std::endl
          << "  Re = " << u_ref*h_ref / nu << std::endl
          << "  omega = " << omega << std::endl
          << "  dx = " << dx << " meter" << std::endl
          << "  dt = " << dt << " sec" << std::endl
          << "  io_interval = " << io_interval << std::endl
          << "  dt_io = " << dt*io_interval << std::endl
          << "  c = " << c << " m/s" << std::endl
          << "  Area = " << dx*dx*nx*ny << std::endl
          ;
    }
  }
};

#endif
