#ifndef __CONFIG_HPP__
#define __CONFIG_HPP__

#include <array>

static constexpr std::size_t DIMENSION = 4;

template <std::size_t N>
using shape_type = std::array<std::size_t, N>;

struct Domain {
  shape_type<DIMENSION> nxmax_; /* Number of points on the finest grid  */
  int nbiter_, ifreq_; /* number of iteration and diagnostic frequency */
  int idcase_; /* Identifier of the test to run */

  double minPhy_[DIMENSION]; /* min and max coordinates in x and v */
  double maxPhy_[DIMENSION];
  double dt_; /*step in time */
  double dx_[DIMENSION]; /* step in x and v */
  char limitCondition_[DIMENSION];
  int fxvx_;
};

struct Physics {
  /*Physic parameter */
  double z_; /* position in accelerator */
  double S_; /* lattice period */
  double omega02_; /* coefficient of applied field */
  double vbeam_; /* beam velocity */
  double beta_, psi_; /* parameters for applied field */
  double eps0_; /*permittivity of free space */
  double echarge_; /* charge of particle */
  double mass_; /*mass of particle */
};

struct Config {
  Physics phys_;
  Domain dom_;
};

#endif
