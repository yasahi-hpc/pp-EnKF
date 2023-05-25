#ifndef __MODEL_FACTORIES_HPP__
#define __MODEL_FACTORIES_HPP__

#include <string>
#include "../config.hpp"
#include "nudging.hpp"
#include "lbm2d.hpp"

static std::unique_ptr<Model> model_factory(std::string model, Config& conf) {
  // For the moment, we only implement LBM2D model
  if(model == "nature") {
    return std::unique_ptr<LBM2D>( new LBM2D(conf) );
  }
  return std::unique_ptr<LBM2D>( new LBM2D(conf) );
};

static std::unique_ptr<DA_Model> da_model_factory(std::string da_model, Config& conf) {
  if(da_model == "nudging") {
    return std::unique_ptr<Nudging>(new Nudging(conf));
  }
  return std::unique_ptr<NonDA>(new NonDA(conf));
};

#endif
