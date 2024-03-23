#include <gtest/gtest.h>
#include <FFT.hpp>
#include "Types.hpp"
#include "Test_Helper.hpp"

class TestFFT2 : public ::testing::Test {
protected:
  std::unique_ptr<sycl::queue> queue_;

  virtual void SetUp() {
    auto selector = sycl::gpu_selector_v;
    try {
      queue_ = std::make_unique<sycl::queue>(selector, 
        exception_handler, sycl::property_list{sycl::property::queue::in_order{}});
      queue_->wait();
    } catch (std::exception const& e) {
      std::cout << "An exception is caught intializing a queue.\n";
      std::terminate();
    }
  }
};

void test_fft_2d(sycl::queue& q) {
  constexpr int Nx = 16;
  constexpr int Ny = 16;
  constexpr int Nz = 4;
  constexpr int Nkx = Nx/2 + 1;
  constexpr double Lx = 1.0;
  constexpr double Ly = 1.0;
  constexpr double dx = Lx / static_cast<double>(Nx);
  constexpr double dy = Ly / static_cast<double>(Ny);
  constexpr double eps = 1.e-8;

  RealView3D u(q, "u", Nx, Ny, Nz);
  RealView3D u_ref(q, "u_ref", Nx, Ny, Nz);
  ComplexView3D u_hat(q, "u", Nkx, Ny, Nz);

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

  // FFT helper
  using value_type = RealView3D::value_type;
  using layout_type = RealView3D::layout_type;
  Impl::FFT<value_type, layout_type> fft(Nx, Ny, Nz);
  fft.set_stream(q);
  float64 normcoeff = fft.normcoeff();

  // Forward Fourier Transform (Nx, Ny, Nz) => (Nx/2+1, Ny, Nz)
  fft.rfft2(u.data(), u_hat.data());

  // Normalization on GPUs
  auto _u_hat = u_hat.mdspan();

  // Sycl range
  sycl::range<3> global_range(Nkx, Ny, Nz);

  // Get the local range for the 3D parallel for loop
  sycl::range<3> local_range = sycl::range<3>(Nkx, 4, 1);

  // Create a 3D nd_range using the global and local ranges
  sycl::nd_range<3> nd_range(global_range, local_range);

  q.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
      nd_range,
      [=](sycl::nd_item<3> item) {
        const int ix = item.get_global_id(0);
        const int iy = item.get_global_id(1);
        const int iz = item.get_global_id(2);

        _u_hat(ix, iy, iz) *= normcoeff;
      }
    );
  });

  // Backward Fourier transform (Nx/2+1, Ny, Nz) => (Nx, Ny, Nz)
  fft.irfft2(u_hat.data(), u.data());
  q.wait();

  for(int iz=0; iz<Nz; iz++) {
    for(int iy=0; iy<Ny; iy++) {
      for(int ix=0; ix<Nx; ix++) {
        EXPECT_NEAR( u(ix, iy, iz), u_ref(ix, iy, iz), eps );
      }
    }
  }
}

TEST_F( TestFFT2, Real ) {
  test_fft_2d(*queue_);
}