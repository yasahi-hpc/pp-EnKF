#ifndef __DA_MODELS_HPP__
#define __DA_MODELS_HPP__

#include <cassert>
#include <iostream>
#include <utils/string_utils.hpp>
#include <utils/file_utils.hpp>
#include <utils/io_utils.hpp>
#include "../config.hpp"
#include "../mpi_config.hpp"
#include "data_vars.hpp"

class DA_Model {
protected:
  Config conf_;
  std::string base_dir_name_;

public:
  DA_Model(Config& conf) : base_dir_name_("io/observed/ens0000"), conf_(conf) {}
  DA_Model(Config& conf, MPIConfig& mpi_conf) : base_dir_name_("io/observed/ens0000"), conf_(conf) {}
  virtual ~DA_Model(){}
  virtual void initialize()=0;
  virtual void apply(std::unique_ptr<DataVars>& data_vars, const int it)=0;
  virtual void diag()=0;
  virtual void finalize()=0;

protected:
  void setFileInfo() {
    int nb_expected_files = conf_.settings_.nbiter_ / conf_.settings_.io_interval_;
    bool expected_files_exist = true;
    std::string variables[3] = {"rho", "u", "v"};
    for(int it=0; it<nb_expected_files; it++) {
      for(int i=0; i<3; i++) {
        auto file_name = base_dir_name_ + "/" + variables[i] + "obs_step" + Impl::zfill(it, 10) + ".dat";
        if(!Impl::isFileExists(file_name)) {
          expected_files_exist = false;
        }
      }
    }
    assert(expected_files_exist);
  }

  void load(std::unique_ptr<DataVars>& data_vars, const int it) {
    auto step = it / conf_.settings_.io_interval_;
    if(step % conf_.settings_.da_interval_ != 0) {
      std::cout << __PRETTY_FUNCTION__ << ": t=" << it << ": skip" << std::endl;
      return;
    };
    from_file(data_vars->rho_obs(), step);
    from_file(data_vars->u_obs(), step);
    from_file(data_vars->v_obs(), step);
  }

private:
  template <class ViewType>
  void from_file(ViewType& value, const int step) {
    auto file_name = base_dir_name_ + "/" + value.name() + "_step"
                   + Impl::zfill(step, 10) + ".dat";
    auto mdspan = value.host_mdspan();
    Impl::from_binary(file_name, mdspan);
    value.updateDevice();
  }

};

/* If DA is not applied at all
 */
class NonDA : public DA_Model {
public:
  NonDA(Config& conf) : DA_Model(conf) {}
  NonDA(Config& conf, MPIConfig& mpi_conf) : DA_Model(conf, mpi_conf) {}
  virtual ~NonDA(){}
  void initialize() {}
  void apply(std::unique_ptr<DataVars>& data_vars, const int it){};
  void diag(){};
  void finalize(){};
};


#endif
