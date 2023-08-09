#include "diags.hpp"
#include <executors/Parallel_Reduce.hpp>
#include <cstdio>

Diags::Diags(const Config& conf) {
  const int nbiter = conf.dom_.nbiter_ + 1;

  nrj_  = RealView1D("nrj", nbiter);
  mass_ = RealView1D("mass", nbiter);
}

void Diags::compute(const Config& conf, Efield* ef, int iter) {
  const auto dom = conf.dom_;
  int nx = dom.nxmax_[0], ny = dom.nxmax_[1];

  assert(iter >= 0 && iter <= dom.nbiter_);
  auto ex  = ef->ex();
  auto ey  = ef->ey();
  auto rho = ef->rho();
 
  using moment_type = thrust::tuple<float64, float64>;
  moment_type moments = {0, 0};

  Iterate_policy<2> policy2d({0, 0}, {nx, ny});

  auto moment_kernel =
    [=] MDSPAN_FORCE_INLINE_FUNCTION (const int ix, const int iy) {
      const float64 _ex  = ex(ix, iy);
      const float64 _ey  = ey(ix, iy);
      const float64 _rho = rho(ix, iy);
      const float64 nrj = _ex*_ex + _ey*_ey;
      return moment_type {_rho, nrj};
  };
 
  auto binary_operator =
    [=] MDSPAN_FORCE_INLINE_FUNCTION (const moment_type& left, const moment_type& right) {
      return moment_type {thrust::get<0>(left) + thrust::get<0>(right),
                          thrust::get<1>(left) + thrust::get<1>(right),
                         };
  };
 
  Impl::transform_reduce(policy2d, binary_operator, moment_kernel, moments);

  float64 iter_mass = thrust::get<0>(moments);
  float64 iter_nrj  = thrust::get<1>(moments);
  iter_nrj = sqrt(iter_nrj * dom.dx_[0] * dom.dx_[1]);
  iter_mass *= dom.dx_[0] + dom.dx_[1];

  iter_nrj = iter_nrj > 1.e-30 ? log(iter_nrj) : -1.e9;
  nrj_(iter) = iter_nrj;
  mass_(iter) = iter_mass;
}

void Diags::save(const Config& conf) {
  const auto dom = conf.dom_;

  char filename[16];
  sprintf(filename, "nrj.out");

  FILE* fileid = fopen(filename, "w");
  for(int iter=0; iter<=dom.nbiter_; ++iter)
    fprintf(fileid, "%17.13e %17.13e %17.13e\n", iter * dom.dt_, nrj_(iter), mass_(iter));

  fclose(fileid);
}
