#ifndef __FIELD_HPP__
#define __FIELD_HPP__

#include "../config.hpp"
#include "types.hpp"
#include "efield.hpp"
#include "diags.hpp"

/*
 * @param[in] fn
 * @param[out] ef.rho_ (Updated by the integral of fn)
 * @param[out] ef.ex_ (zero initialization)
 * @param[out] ef.ey_ (zero initialization)
 * @param[out] ef.phi_ (zero initialization)
 */
void field_rho(const Config& conf, const RealView4D& fn, Efield* ef);
void field_poisson(const Config& conf, Efield* ef);

#endif
