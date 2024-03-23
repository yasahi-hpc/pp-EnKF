#ifndef DA_MODELS_HPP
#define DA_MODELS_HPP

#include <cassert>
#include <iostream>
#include <utils/string_utils.hpp>
#include <utils/file_utils.hpp>
#include <utils/io_utils.hpp>
#include "../config.hpp"
#include "../io_config.hpp"
#include "../mpi_config.hpp"
#include "../timer.hpp"
#include "data_vars.hpp"

class DA_Model {
protected:
  sycl::queue queue_;
  Config conf_;
  IOConfig io_conf_;
  std::string base_dir_name_;

public:
  DA_Model(sycl::queue& queue, Config& conf, IOConfig& io_conf) : queue_(queue), conf_(conf), io_conf_(io_conf) {
    base_dir_name_ = io_conf_.base_dir_ + "/" + io_conf_.in_case_name_ + "/observed/ens0000";
  }

  DA_Model(sycl::queue& queue, Config& conf, IOConfig& io_conf, MPIConfig& mpi_conf) : queue_(queue), conf_(conf), io_conf_(io_conf) {
    base_dir_name_ = io_conf_.base_dir_ + "/" + io_conf_.in_case_name_ + "/observed/ens0000";
  }

  virtual ~DA_Model(){}
  virtual void initialize()=0;
  virtual void apply(std::unique_ptr<DataVars>& data_vars, const int it, std::vector<Timer*>& timers)=0;
  virtual void diag()=0;
  virtual void finalize()=0;

protected:
  void setFileInfo() {
    int nb_expected_files = conf_.settings_.nbiter_ / conf_.settings_.io_interval_;
    std::string variables[3] = {"rho", "u", "v"};
    for(int it=0; it<nb_expected_files; it++) {
      for(int i=0; i<3; i++) {
        auto step = it * conf_.settings_.io_interval_;
        auto file_name = base_dir_name_ + "/" + variables[i] + "_obs_step" + Impl::zfill(step, 10) + ".dat";
        if(!Impl::isFileExists(file_name)) {
          std::runtime_error("Expected observation file does not exist." + file_name);
        }
      }
    }
  }

  void load(std::unique_ptr<DataVars>& data_vars, const int it) {
    from_file(data_vars->rho_obs(), it);
    from_file(data_vars->u_obs(), it);
    from_file(data_vars->v_obs(), it);
  }

private:
  template <class ViewType>
  void from_file(ViewType& value, const int step) {
    auto file_name = base_dir_name_ + "/" + value.name() + "_step" + Impl::zfill(step, 10) + ".dat";
    auto mdspan = value.mdspan();
    Impl::from_binary(file_name, mdspan);
  }

};

/* If DA is not applied at all
 */
class NonDA : public DA_Model {
public:
  NonDA(sycl::queue& queue, Config& conf, IOConfig& io_conf) : DA_Model(queue, conf, io_conf) {}
  NonDA(sycl::queue& queue, Config& conf, IOConfig& io_conf, MPIConfig& mpi_conf) : DA_Model(queue, conf, io_conf, mpi_conf) {}
  virtual ~NonDA(){}
  void initialize() {}
  void apply(std::unique_ptr<DataVars>& data_vars, const int it, std::vector<Timer*>& timers){};
  void diag(){};
  void finalize(){};
};

#endif
