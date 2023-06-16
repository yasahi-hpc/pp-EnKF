#ifndef __MODELS_HPP__
#define __MODELS_HPP__

#include <string>
#include "../config.hpp"
#include "../io_config.hpp"
#include "data_vars.hpp"

class Model {
protected:
  Config conf_;
  IOConfig io_conf_;
  int it_;
  int diag_it_;

public:
  Model()=delete;
  Model(Config& conf, IOConfig& io_conf) : it_(0), diag_it_(0), conf_(conf), io_conf_(io_conf) {}
  virtual ~Model(){}
  virtual void initialize(std::unique_ptr<DataVars>& data_vars)=0;
  virtual void reset(std::unique_ptr<DataVars>& data_vars, const std::string mode)=0;
  virtual void solve(std::unique_ptr<DataVars>& data_vars)=0;
  virtual void diag(std::unique_ptr<DataVars>& data_vars)=0;
  virtual void finalize()=0;
};

#endif
