#ifndef MODELS_HPP
#define MODELS_HPP

#include <string>
#include "../config.hpp"
#include "../io_config.hpp"
#include "../timer.hpp"
#include "data_vars.hpp"

class Model {
protected:
  Config conf_;
  IOConfig io_conf_;
  sycl::queue queue_;
  
public:
  Model()=delete;
  Model(sycl::queue& queue, Config& conf, IOConfig& io_conf) : queue_(queue), conf_(conf), io_conf_(io_conf) {}
  virtual ~Model(){}
  virtual void initialize(std::unique_ptr<DataVars>& data_vars)=0;
  virtual void reset(std::unique_ptr<DataVars>& data_vars, const std::string mode)=0;
  virtual void solve(std::unique_ptr<DataVars>& data_vars)=0;
  virtual void diag(std::unique_ptr<DataVars>& data_vars, const int it, std::vector<Timer*>& timers)=0;
  virtual void finalize()=0;
};

#endif
