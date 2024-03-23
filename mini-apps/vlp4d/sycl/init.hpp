#ifndef INIT_HPP
#define INIT_HPP

#include "efield.hpp"
#include "diags.hpp"
#include "types.hpp"

void init(const char* file,
          sycl::queue& q,
          Config& conf,
          RealView4D& fn,
          RealView4D& fnp1,
          std::unique_ptr<Efield>& ef,
          std::unique_ptr<Diags>& dg);
void finalize(Config& conf, std::unique_ptr<Diags>& dg);

#endif