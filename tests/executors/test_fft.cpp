#include <gtest/gtest.h>
#include "Types.hpp"
#include <FFT.hpp>
#include <executors/Parallel_For.hpp>
#include <cmath>

void test_fft_2d() {
  constexpr int Nx = 16;
  constexpr int Ny = 16;
  constexpr int Nz = 2;
  constexpr int Nkx = Nx/2 + 1;
  constexpr double Lx = 1.0;
  constexpr double Ly = 1.0;
  constexpr double dx = Lx / static_cast<double>(Nx);
  constexpr double dy = Ly / static_cast<double>(Ny);
  constexpr double eps = 1.e-8;

  RealView3D u("u", Nx, Ny, Nz);
  RealView3D u_ref("u_ref", Nx, Ny, Nz);
  ComplexView3D u_hat("u", Nkx, Ny, Nz);

  // Initialization on host
  for(int iz=0; iz<Nz; iz++) {
    for(int iy=0; iy<Ny; iy++) {
      for(int ix=0; ix<Nx; ix++) {
        const double xtmp = static_cast<double>(ix) * dx;
        const double ytmp = static_cast<double>(iy) * dy;
        u(ix, iy, iz) = sin(xtmp * 2. * M_PI / Lx)
                      + cos(ytmp * 3. * M_PI / Ly);
        u_ref(ix, iy, iz) = u(ix, iy, iz);
      }
    }
  }
  u.updateDevice();

  // FFT helper
  using value_type = RealView3D::value_type;
  using layout_type = RealView3D::layout_type;
  Impl::FFT<value_type, layout_type> fft(Nx, Ny, Nz);
  float64 normcoeff = fft.normcoeff();

  // Forward Fourier Transform (Nx, Ny, Nz) => (Nx/2+1, Ny, Nz)
  fft.rfft2(u.data(), u_hat.data());

  // Normalization on GPUs
  auto _u_hat = u_hat.mdspan();
  auto normalization = [=] (const int ix, const int iy, const int iz) {
    _u_hat(ix, iy, iz) *= normcoeff;
  };

  Iterate_policy<3> policy3d({0, 0, 0}, {Nkx, Ny, Nz});
  Impl::for_each(policy3d, normalization);

  // Backward Fourier transform (Nx/2+1, Ny, Nz) => (Nx, Ny, Nz)
  fft.irfft2(u_hat.data(), u.data());

  // Device to host copy
  u.updateSelf();

  for(int iz=0; iz<Nz; iz++) {
    for(int iy=0; iy<Ny; iy++) {
      for(int ix=0; ix<Nx; ix++) {
        ASSERT_NEAR( u(ix, iy, iz), u_ref(ix, iy, iz), eps );
      }
    }
  }
}

TEST( EXECUTORS_FFT, DIM2 ) {
  test_fft_2d();
}
