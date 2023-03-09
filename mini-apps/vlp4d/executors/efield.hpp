#ifndef __EFIELD_HPP__
#define __EFIELD_HPP__

#include <stdexec/execution.hpp>
#include "exec/on.hpp"
#include <FFT.hpp>
#include "../config.hpp"
#include "types.hpp"

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
  Efield(Config& conf) {
    auto nx = conf.dom_.nxmax_[0], ny = conf.dom_.nxmax_[1];
    rho_ = RealView2D("rho", nx, ny);
    ex_  = RealView2D("ex",  nx, ny);
    ey_  = RealView2D("ey",  nx, ny);
    phi_ = RealView2D("phi", nx, ny);
 
    // Initialize fft helper
    xmax_ = conf.dom_.maxPhy_[0];
    ymax_ = conf.dom_.maxPhy_[1];
    float64 kx0 = 2 * M_PI / xmax_;
 
    fft_ = new Impl::FFT<value_type, layout_type>(nx, ny, 1);
    int nx1h = fft_->nx1h();
 
    rho_hat_ = ComplexView2D("rho_hat", nx1h, ny);
    ex_hat_  = ComplexView2D("ex_hat", nx1h, ny);
    ey_hat_  = ComplexView2D("ey_hat", nx1h, ny);
 
    filter_ = RealView1D("filter", nx1h);
 
    // Initialize filter (0,1/k2,1/k2,1/k2,...)
    filter_(0) = 0.;
    for(int ix = 1; ix < nx1h; ix++) {
      float64 kx = ix * kx0;
      float64 k2 = kx * kx;
      filter_(ix) = 1./k2;
    }
 
    filter_.updateDevice();
    rho_.updateDevice(); ex_.updateDevice(); ey_.updateDevice(); phi_.updateDevice();
    rho_hat_.updateDevice(); ex_hat_.updateDevice(); ey_hat_.updateDevice();
  }

  virtual ~Efield(){ if(fft_ != nullptr) delete fft_; };

public:
  // Getters
  auto rho() { return rho_.mdspan(); }
  auto ex()  { return ex_.mdspan(); }
  auto ey()  { return ey_.mdspan(); }
  auto phi() { return phi_.mdspan() ;}

  template <class Scheduler>
  void solve_poisson_fftw(Scheduler&& scheduler) { 
    float64 kx0 = 2 * M_PI / xmax_;
    float64 ky0 = 2 * M_PI / ymax_;
    int nx1  = fft_->nx1();
    int nx1h = fft_->nx1h();
    int nx2  = fft_->nx2();
    int nx2h = fft_->nx2h();
    float64 normcoeff = fft_->normcoeff();
    const complex128 I = complex128(0.0, 1.0);

    // Access via mdspans
    auto rho = rho_.mdspan();
    auto ex  = ex_.mdspan();
    auto ey  = ey_.mdspan();
    auto phi = phi_.mdspan();
    auto rho_hat = rho_hat_.mdspan();
    auto ex_hat  = ex_hat_.mdspan();
    auto ey_hat  = ey_hat_.mdspan();
    auto filter  = filter_.mdspan();

    // Forward 2D FFT (Real to Complex)
    fft_->rfft2(rho.data_handle(), rho_hat.data_handle());

    // Solve Poisson equation in Fourier space
    // In order to avoid zero division in vectorized way
    // filter[0] == 0, and filter[0:] == 1./(ix*kx0)**2
    auto solve_poisson = [=](const int ix1) {
      float64 kx = ix1 * kx0;
      {
        int ix2 = 0;
        float64 kx = ix1 * kx0;
        ex_hat(ix1, ix2) = -kx * I * rho_hat(ix1, ix2) * filter(ix1) * normcoeff;
        ey_hat(ix1, ix2) = 0.;
        rho_hat(ix1, ix2) = rho_hat(ix1, ix2) * filter(ix1) * normcoeff;
      }

      for(int ix2=1; ix2<nx2h; ix2++) {
        float64 ky = ix2 * ky0;
        float64 k2 = kx * kx + ky * ky;

        ex_hat(ix1, ix2) = -(kx/k2) * I * rho_hat(ix1, ix2) * normcoeff;
        ey_hat(ix1, ix2) = -(ky/k2) * I * rho_hat(ix1, ix2) * normcoeff;
        rho_hat(ix1, ix2) = rho_hat(ix1, ix2) / k2 * normcoeff;
      }

      for(int ix2=nx2h; ix2<nx2; ix2++) {
        float64 ky = (ix2-nx2) * ky0;
        float64 k2 = kx*kx + ky*ky;

        ex_hat(ix1, ix2) = -(kx/k2) * I * rho_hat(ix1, ix2) * normcoeff;
        ey_hat(ix1, ix2) = -(ky/k2) * I * rho_hat(ix1, ix2) * normcoeff;
        rho_hat(ix1, ix2) = rho_hat(ix1, ix2) / k2 * normcoeff;
      }
    };

    auto poisson_fourier_task = stdexec::just()
      | stdexec::then(
        [&]{ 
        // Forward 2D FFT (Real to Complex)
        fft_->rfft2(rho.data_handle(), rho_hat.data_handle()); 
        } )
      | exec::on( scheduler, stdexec::bulk(nx1h, solve_poisson) )
      | stdexec::then(
        [&]{ 
          // Backward 2D FFT (Complex to Real)
          fft_->irfft2(rho_hat.data_handle(), rho.data_handle());
          fft_->irfft2(ex_hat.data_handle(),  ex.data_handle());
          fft_->irfft2(ey_hat.data_handle(),  ey.data_handle());
        } );
    stdexec::sync_wait( std::move(poisson_fourier_task) );
  }
};

#endif
