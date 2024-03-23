#ifndef FIELD_HPP
#define FIELD_HPP

#include "../config.hpp"
#include "types.hpp"
#include "efield.hpp"
#include "diags.hpp"

void field_rho(sycl::queue& q,
               const Config& conf,
               const RealView4D& fn,
               std::unique_ptr<Efield>& ef);

void field_poisson(sycl::queue& q,
                   const Config& conf,
                   std::unique_ptr<Efield>& ef);

#endif
