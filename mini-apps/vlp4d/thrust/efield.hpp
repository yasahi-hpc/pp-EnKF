#ifndef __EFIELD_HPP__
#define __EFIELD_HPP__

#include "../config.hpp"
#include "types.hpp"
#include <FFT.hpp>

struct Efield {
  using value_type = RealView2D::value_type;
  using layout_type = RealView2D::layout_type;

private:
  float64 xmax_, ymax_;

  // These variables may be updated
  RealView2D rho_;
  RealView2D ex_;
  RealView2D ey_;
  RealView2D phi_;

  Impl::FFT<value_type, layout_type> *fft_;

  RealView1D filter_; // [YA added] In order to avoid conditional to keep (0, 0) component 0

  // a 2D complex buffer of size nx1h * nx2 (renamed)
  ComplexView2D rho_hat_;
  ComplexView2D ex_hat_;
  ComplexView2D ey_hat_;

public:
  Efield(Config& conf);
  virtual ~Efield(){ if(fft_ != nullptr) delete fft_; };

public:
  // methods and getters
  void solve_poisson_fftw();
  auto rho() {return rho_.mdspan();}
  auto ex()  {return ex_.mdspan();}
  auto ey()  {return ey_.mdspan();}
  auto phi() {return phi_.mdspan();}
};

#endif
