#ifndef MODEL_FACTORIES_HPP
#define MODEL_FACTORIES_HPP

#include <string>
#include <sycl/sycl.hpp>
#include "../config.hpp"
#include "../io_config.hpp"
#include "../mpi_config.hpp"
#include "nudging.hpp"
#include "letkf.hpp"
#include "lbm2d.hpp"

static std::unique_ptr<Model> model_factory(std::string model, sycl::queue& queue, Config& conf, IOConfig& io_conf) {
  // For the moment, we only implement LBM2D model
  if(model == "nature") {
    return std::unique_ptr<LBM2D>( new LBM2D(queue, conf, io_conf) );
  }
  return std::unique_ptr<LBM2D>( new LBM2D(queue, conf, io_conf) );
};

static std::unique_ptr<DA_Model> da_model_factory(std::string da_model, sycl::queue& queue, Config& conf, IOConfig& io_conf, MPIConfig& mpi_conf) {
  if(da_model == "nudging") {
    return std::unique_ptr<Nudging>(new Nudging(queue, conf, io_conf));
  } else if(da_model == "letkf") {
    return std::unique_ptr<LETKF>(new LETKF(queue, conf, io_conf, mpi_conf));
  }
  return std::unique_ptr<NonDA>(new NonDA(queue, conf, io_conf));
};

#endif
