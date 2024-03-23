#ifndef EFIELD_HPP
#define EFIELD_HPP

#include <memory>
#include <FFT.hpp>
#include "../config.hpp"
#include "types.hpp"

struct Efield {
  using value_type  = RealView2D::value_type;
  using layout_type = RealView2D::layout_type;
  using fft_type = Impl::FFT<value_type, layout_type>;

private:
  float64 xmax_, ymax_;

  // These variables may be updated
  RealView2D rho_;
  RealView2D ex_;
  RealView2D ey_;
  RealView2D phi_;

  std::unique_ptr<fft_type> fft_;

  RealView1D filter_; // [YA added] In order to avoid conditional to keep (0, 0) component 0

  // a 2D complex buffer of size nx1h * nx2 (renamed)
  ComplexView2D rho_hat_;
  ComplexView2D ex_hat_;
  ComplexView2D ey_hat_;

public:
  Efield(sycl::queue& q, Config& conf);
  virtual ~Efield(){}

public:
  // methods and getters
  void solve_poisson_fftw(sycl::queue& q);
  auto rho() {return rho_.mdspan();}
  auto ex()  {return ex_.mdspan();}
  auto ey()  {return ey_.mdspan();}
  auto phi() {return phi_.mdspan();}
};

#endif
