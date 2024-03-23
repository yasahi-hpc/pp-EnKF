#include "diags.hpp"
#include <executors/Parallel_Reduce.hpp>
#include <cstdio>

class moment_kernel;

Diags::Diags(sycl::queue& q, const Config& conf) {
  const int nbiter = conf.dom_.nbiter_ + 1;

  nrj_  = RealView1D(q, "nrj", nbiter);
  mass_ = RealView1D(q, "mass", nbiter);
}

void Diags::compute(sycl::queue& q, const Config& conf, std::unique_ptr<Efield>& ef, int iter) {
  const auto dom = conf.dom_;
  int nx = dom.nxmax_[0], ny = dom.nxmax_[1];

  assert(iter >= 0 && iter <= dom.nbiter_);
  auto ex  = ef->ex();
  auto ey  = ef->ey();
  auto rho = ef->rho();

  /*
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
  */

  float64* iter_mass_ptr; 
  float64* iter_nrj_ptr;
  iter_mass_ptr = sycl::malloc_shared<float64>(1, q);
  iter_nrj_ptr  = sycl::malloc_shared<float64>(1, q);
  *iter_mass_ptr = 0;
  *iter_nrj_ptr = 0;

  sycl::range<2> global_range(nx, ny);
  sycl::range<2> local_range = sycl::range<2>(32, 8);
  sycl::nd_range<2> nd_range(global_range, local_range); 

  q.submit([&](sycl::handler& cgh) {
    #if defined(__HIPSYCL__) || defined(__OPENSYCL__)
      auto mass_reduction = sycl::reduction(iter_mass_ptr, sycl::plus<float64>());
      auto nrj_reduction  = sycl::reduction(iter_nrj_ptr, sycl::plus<float64>());
    #else
      auto mass_reduction = sycl::reduction(iter_mass_ptr, sycl::plus<float64>(), sycl::property::reduction::initialize_to_identity{});
      auto nrj_reduction  = sycl::reduction(iter_nrj_ptr, sycl::plus<float64>(), sycl::property::reduction::initialize_to_identity{});
    #endif
    cgh.parallel_for<class moment_kernel>(
      nd_range, mass_reduction, nrj_reduction,
        [=](sycl::nd_item<2> item, auto& mass_reduction, auto& nrj_reduction) {
          const int ix = item.get_global_id(0);
          const int iy = item.get_global_id(1);

          const float64 _ex  = ex(ix, iy);
          const float64 _ey  = ey(ix, iy);
          const float64 _rho = rho(ix, iy);
          const float64 nrj  = _ex*_ex + _ey*_ey;
          mass_reduction += _rho;
          nrj_reduction  += nrj;
        }
    );
  }).wait();

  float64 iter_nrj  = sqrt(*iter_nrj_ptr * dom.dx_[0] * dom.dx_[1]);
  float64 iter_mass = *iter_mass_ptr * dom.dx_[0] * dom.dx_[1];

  iter_nrj = iter_nrj > 1.e-30 ? log(iter_nrj) : -1.e9;
  nrj_(iter)  = iter_nrj;
  mass_(iter) = iter_mass;

  sycl::free(iter_mass_ptr, q);
  sycl::free(iter_nrj_ptr, q);
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
