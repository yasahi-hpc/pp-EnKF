#ifndef __DIAGS_HPP__
#define __DIAGS_HPP__

#include "../config.hpp"
#include "types.hpp"
#include "efield.hpp"

struct Diags {
private:
  RealView1D nrj_;
  RealView1D mass_;

public:
  Diags(const Config& conf);
  virtual ~Diags() {};

  void compute(const Config& conf, Efield* ef, int iter);
  void save(const Config& conf);
};

#endif
