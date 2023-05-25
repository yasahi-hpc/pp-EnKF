#ifndef __DA_MODELS_HPP__
#define __DA_MODELS_HPP__

#include <cassert>
#include <iostream>
#include <utils/string_utils.hpp>
#include <utils/file_utils.hpp>
#include "../config.hpp"
#include "data_vars.hpp"

class DA_Model {
protected:
  Config conf_;
  std::string base_dir_name_;

public:
  DA_Model(Config& conf) : base_dir_name_("io/observed/ens0000"), conf_(conf) {}
  virtual ~DA_Model(){}
  virtual void initialize()=0;
  virtual void apply(std::unique_ptr<DataVars>& data_vars, const int it)=0;
  virtual void diag()=0;
  virtual void finalize()=0;

protected:
  void setFileInfo() {
    int nb_expected_files = 1;
    bool expected_files_exist = true;
    std::string variables[3] = {"rho", "u", "v"};
    for(int it=0; it<nb_expected_files; it++) {
      for(int i=0; i<3; i++) {
        auto file_name = base_dir_name_ + "/" + variables[i] + "_step" + Impl::zfill(it, 10) + ".dat";
        if(!Impl::isFileExists(file_name)) {
          expected_files_exist = false;
        }
      }
    }
    assert(expected_files_exist);
  }

  void load(std::unique_ptr<DataVars>& data_vars, const int it) {
    auto step = it / conf_.settings_.io_interval_;
    // load u
    {
      auto file_name = base_dir_name_ + "/u_step" + Impl::zfill(step, 10) + ".dat";
      auto file = std::ifstream(file_name, std::ios::binary);
      assert(file.is_open());
      auto* uo_ptr = data_vars->u_obs().data();
      std::size_t size = sizeof(double) * data_vars->u_obs().size();
      file.read(reinterpret_cast<char*>(uo_ptr), size);
    }

    // load v
    {
      auto file_name = base_dir_name_ + "/v_step" + Impl::zfill(step, 10) + ".dat";
      auto file = std::ifstream(file_name, std::ios::binary);
      assert(file.is_open());
      auto* vo_ptr = data_vars->v_obs().data();
      std::size_t size = sizeof(double) * data_vars->v_obs().size();
      file.read(reinterpret_cast<char*>(vo_ptr), size);
    }

    // load rho
    {
      auto file_name = base_dir_name_ + "/rho_step" + Impl::zfill(step, 10) + ".dat";
      auto file = std::ifstream(file_name, std::ios::binary);
      assert(file.is_open());
      auto* rho_ptr = data_vars->rho_obs().data();
      std::size_t size = sizeof(double) * data_vars->rho_obs().size();
      file.read(reinterpret_cast<char*>(rho_ptr), size);
    }
  }
};

/* If DA is not applied at all
 */
class NonDA : public DA_Model {
public:
  NonDA(Config& conf) : DA_Model(conf) {}
  virtual ~NonDA(){}
  void initialize() {}
  void apply(std::unique_ptr<DataVars>& data_vars, const int it){};
  void diag(){};
  void finalize(){};
};


#endif
