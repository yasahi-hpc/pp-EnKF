#ifndef DIAGS_HPP
#define DIAGS_HPP

#include "../config.hpp"
#include "types.hpp"
#include "efield.hpp"

struct Diags {
private:
  RealView1D nrj_;
  RealView1D mass_;

public:
  Diags(sycl::queue& q, const Config& conf);
  virtual ~Diags() {}

  void compute(sycl::queue& q, const Config& conf, std::unique_ptr<Efield>& ef, int iter);
  void save(const Config& conf);
};

#endif
