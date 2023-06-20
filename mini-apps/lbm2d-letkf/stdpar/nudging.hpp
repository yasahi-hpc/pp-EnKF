#ifndef __NUDGING_HPP__
#define __NUDGING_HPP__

#include <stdpar/Parallel_For.hpp>
#include "../da_functors.hpp"
#include "da_models.hpp"

class Nudging : public DA_Model {
public:
  Nudging(Config& conf, IOConfig& io_conf) : DA_Model(conf, io_conf) {}
  Nudging(Config& conf, IOConfig& io_conf, MPIConfig& mpi_conf)=delete;
  virtual ~Nudging(){}
  void initialize() {
    setFileInfo();
  }

  void apply(std::unique_ptr<DataVars>& data_vars, const int it, std::vector<Timer*>& timers){
    if(it == 0) return;
    if(it % conf_.settings_.da_interval_ != 0) {
      std::cout << __PRETTY_FUNCTION__ << ": t=" << it << ": skip" << std::endl;
      return;
    };
    timers[DA_Load]->begin();
    load(data_vars, it); // loading rho_obs, u_obs, v_obs
    timers[DA_Load]->end();

    auto f       = data_vars->f().mdspan();
    auto rho_obs = data_vars->rho_obs().mdspan();
    auto u_obs   = data_vars->u_obs().mdspan();
    auto v_obs   = data_vars->v_obs().mdspan();

    auto [nx, ny] = conf_.settings_.n_;

    Iterate_policy<2> policy2d({0, 0}, {nx, ny});
    timers[DA_Update]->begin();
    Impl::for_each(policy2d, nudging_functor(conf_, rho_obs, u_obs, v_obs, f));
    timers[DA_Update]->end();
  }
  void diag(){}
  void finalize(){}
};

#endif
