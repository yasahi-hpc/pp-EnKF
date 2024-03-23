#include <gtest/gtest.h>
#include <random>
#include <functional>
#include <sycl/Transpose.hpp>
#include "Types.hpp"
#include "Test_Helper.hpp"

using test_types = ::testing::Types<float, double>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct Transpose2D : public ::testing::Test {
protected:
  using float_type = T;
  std::unique_ptr<sycl::queue> queue_;

  virtual void SetUp() {
    auto selector = sycl::gpu_selector_v;
    try {
      queue_ = std::make_unique<sycl::queue>(selector, exception_handler, sycl::property_list{sycl::property::queue::in_order{}});
      queue_->wait();
    } catch (std::exception const& e) {
      std::cout << "An exception is caught intializing a queue.\n";
      std::terminate();
    }
  }
};

TYPED_TEST_SUITE(Transpose2D, test_types);

template <typename RealType>
void test_transpose2D(sycl::queue& q) {
  const std::size_t rows = 16;
  const std::size_t rows2 = 15;
  const std::size_t cols = rows * 2;
  const std::size_t cols2 = rows2 * 2;

  // Original arrays: A, Transposed Arrays: B
  View2D<RealType> A_even(q, "A_even", rows, cols);
  View2D<RealType> A_odd(q, "A_odd", rows2, cols2);
  View2D<RealType> B_even(q, "B_even", cols, rows);
  View2D<RealType> B_odd(q, "B_odd", cols2, rows2);

  View2D<RealType> Ref_even(q, "Ref_even", cols, rows);
  View2D<RealType> Ref_odd(q, "Ref_odd", cols2, rows2);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<RealType>(-1, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set random numbers to 2D view
  for(int i=0; i<cols; i++) {
    for(int j=0; j<rows; j++) {
      A_even(j, i) = rand_gen();
      Ref_even(i, j) = A_even(j, i);
    }
  }

  for(int i=0; i<cols2; i++) {
    for(int j=0; j<rows2; j++) {
      A_odd(j, i) = rand_gen();
      Ref_odd(i, j) = A_odd(j, i);
    }
  }

  constexpr RealType eps = 1.e-5;
  auto _A_even = A_even.mdspan();
  auto _B_even = B_even.mdspan();
  Impl::transpose(_A_even, _B_even);

  for(int i=0; i<cols; i++) {
    for(int j=0; j<rows; j++) {
      EXPECT_NEAR( B_even(i, j), Ref_even(i, j), eps );
    }
  }

  auto _A_odd = A_odd.mdspan();
  auto _B_odd = B_odd.mdspan();
  Impl::transpose(_A_odd, _B_odd);

  for(int i=0; i<cols2; i++) {
    for(int j=0; j<rows2; j++) {
      EXPECT_NEAR( B_odd(i, j), Ref_odd(i, j), eps );
    }
  }
}

template <typename RealType>
void test_transpose2D_batched(sycl::queue& q) {
  const std::size_t n = 16, m = 15, l = 4;

  // Original arrays
  View3D<RealType> x(q, "x", n, m, l);
  View3D<RealType> x0(q, "x0", n, m, l);
  View3D<RealType> x1(q, "x1", n, l, m);
  View3D<RealType> x2(q, "x2", m, n, l);
  View3D<RealType> x3(q, "x3", m, l, n);
  View3D<RealType> x4(q, "x4", l, n, m);
  View3D<RealType> x5(q, "x5", l, m, n);

  View3D<RealType> ref0(q, "ref0", n, m, l);
  View3D<RealType> ref1(q, "ref1", n, l, m);
  View3D<RealType> ref2(q, "ref2", m, n, l);
  View3D<RealType> ref3(q, "ref3", m, l, n);
  View3D<RealType> ref4(q, "ref4", l, n, m);
  View3D<RealType> ref5(q, "ref5", l, m, n);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<RealType>(-1, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set random numbers to 3D view
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        x(ix, iy, iz) = rand_gen();

        ref0(ix, iy, iz) = x(ix, iy, iz);
        ref1(ix, iz, iy) = x(ix, iy, iz);
        ref2(iy, ix, iz) = x(ix, iy, iz);
        ref3(iy, iz, ix) = x(ix, iy, iz);
        ref4(iz, ix, iy) = x(ix, iy, iz);
        ref5(iz, iy, ix) = x(ix, iy, iz);
      }
    }
  }

  auto _x  = x.mdspan();
  auto _x0 = x0.mdspan();
  auto _x1 = x1.mdspan();
  auto _x2 = x2.mdspan();
  auto _x3 = x3.mdspan();
  auto _x4 = x4.mdspan();
  auto _x5 = x5.mdspan();

  Impl::transpose(_x, _x0, {0, 1, 2});
  Impl::transpose(_x, _x1, {0, 2, 1});
  Impl::transpose(_x, _x2, {1, 0, 2});
  Impl::transpose(_x, _x3, {1, 2, 0});
  Impl::transpose(_x, _x4, {2, 0, 1});
  Impl::transpose(_x, _x5, {2, 1, 0});

  constexpr RealType eps = 1.e-12;
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        EXPECT_NEAR( x0(ix, iy, iz), ref0(ix, iy, iz), eps );
        EXPECT_NEAR( x1(ix, iz, iy), ref1(ix, iz, iy), eps );
        EXPECT_NEAR( x2(iy, ix, iz), ref2(iy, ix, iz), eps );
        EXPECT_NEAR( x3(iy, iz, ix), ref3(iy, iz, ix), eps );
        EXPECT_NEAR( x4(iz, ix, iy), ref4(iz, ix, iy), eps );
        EXPECT_NEAR( x5(iz, iy, ix), ref5(iz, iy, ix), eps );
      }
    }
  }
}

TYPED_TEST(Transpose2D, Single) {
  using float_type = typename TestFixture::float_type;
  test_transpose2D<float_type>(*this->queue_);
}

TYPED_TEST(Transpose2D, Batched) {
  using float_type = typename TestFixture::float_type;
  test_transpose2D_batched<float_type>(*this->queue_);
}