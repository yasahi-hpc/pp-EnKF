#include <gtest/gtest.h>
#include <random>
#include <functional>
#include <tuple>
#include <vector>
#include <sycl/numpy_like.hpp>
#include "Types.hpp"
#include "Test_Helper.hpp"

class NumpyLikeSimple : public ::testing::Test {
protected:
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

using test_types = testing::Types<int, float, double>;

template <typename T>
using param_triplet = std::vector<std::tuple<T, T, T>>;

static std::tuple<
  param_triplet<int>,
  param_triplet<float>,
  param_triplet<double>
> all_params_arange {
  {
    // (start, stop, step) for int
    std::make_tuple(0, 5, 1),
    std::make_tuple(0, 8, 2),
  },
  {
    // (start, stop, step) for float
    std::make_tuple(0, 5, 1),
    std::make_tuple(-10.0, 10.0, 0.5),
  },
  {
    // (start, stop, step) for double
    std::make_tuple(0, 5, 1),
    std::make_tuple(-10.0, 10.0, 0.5),
  },
};

template <typename T>
struct SYCLFixtureTyped : public ::testing::Test {
protected:
  using param_type = param_triplet<T>;
  using float_type = T;
  param_type params_;
  std::unique_ptr<sycl::queue> queue_;

  virtual void SetUp() {
    params_ = std::get<param_type>(all_params_arange);
    auto selector = sycl::gpu_selector_v;
    try {
      queue_ = std::make_unique<sycl::queue>(selector, exception_handler);
      //std::cout << "Running on device: "
      //          << queue_->get_device().get_info<sycl::info::device::name>() << "\n";
      queue_->wait();
    } catch (std::exception const& e) {
      std::cout << "An exception is caught intializing a queue.\n";
      std::terminate();
    }
  }
};

TYPED_TEST_SUITE_P(SYCLFixtureTyped);

template <typename ElementType>
void test_arange(sycl::queue& q,
                 ElementType start,
                 ElementType stop,
                 ElementType step
                ) {

  auto a = Impl::arange(start, stop, step);
  if constexpr(std::is_same_v<ElementType, int>) {
    // np.arange(5) or np.arange(0, 5, 1)
    auto b = Impl::arange(start, stop);

    for(int i=0; i<b.size(); i++) {
      EXPECT_EQ(b.at(i), start + i);
    }

    ASSERT_EQ( a.size(), b.size() / step );
    for(int i=0; i<a.size(); i++) {
      EXPECT_EQ(a.at(i), start + i * step);
    }
  } else {
    // np.arange(start, stop, step)
    const int len = a.size();
    const int expected_len = ceil( (stop-start) / step );
    const ElementType expected_dt = (stop-start) / static_cast<ElementType>(expected_len);
    const auto dt = a.at(1) - a.at(0);

    constexpr ElementType eps = std::is_same_v<ElementType, float> ? 1.0e-6 : 1.0e-13;
    ASSERT_EQ( typeid(decltype(a.front())), typeid(ElementType) );
    ASSERT_EQ( len, expected_len );
    EXPECT_NEAR( dt, expected_dt, eps );
    EXPECT_NEAR( a.front(), start, eps );
    EXPECT_NEAR( a.back(), stop-step, eps ); // because the last element should be unreached
  }
}

template <typename OutputType, typename InputType>
void test_linspace(sycl::queue& q,
                   InputType start,
                   InputType stop
                  ) {

  // np.linspace(0, 1)
  auto a = Impl::linspace<OutputType>(start, stop);
  const size_t len = a.size();
  const size_t expected_len = 50;
  const double expected_dt = static_cast<OutputType>(stop-start) / static_cast<OutputType>(expected_len-1);
  const auto dt = a.at(1) - a.at(0);

  constexpr OutputType eps = std::is_same_v<OutputType, float> ? 1.0e-6 : 1.0e-13;
  ASSERT_EQ( typeid(decltype(a.front())), typeid(OutputType) );
  ASSERT_EQ( len, expected_len );
  EXPECT_NEAR( dt, expected_dt, eps );
  EXPECT_NEAR( a.front(), static_cast<OutputType>(start), eps );
  EXPECT_NEAR( a.back(), static_cast<OutputType>(stop), eps ); // because the last element should be included
}

TYPED_TEST_P(SYCLFixtureTyped, Arange) {
  for(auto const& [start, stop, step] : this->params_) {
    test_arange(*this->queue_, start, stop, step);
  }
}

TYPED_TEST_P(SYCLFixtureTyped, Linspace) {
  using out_float_type = double;
  for(auto const& [start, stop, _] : this->params_) {
    test_linspace<out_float_type>(*this->queue_, start, stop);
  }
}

REGISTER_TYPED_TEST_SUITE_P(SYCLFixtureTyped, Arange, Linspace);
INSTANTIATE_TYPED_TEST_SUITE_P(NumpyLike, SYCLFixtureTyped, test_types);
TYPED_TEST_SUITE(SYCLFixtureTyped, test_types);

void test_mean_3D_to_3D(sycl::queue& q) {
  const std::size_t n = 3, m = 2, l = 5;
  RealView3D a(q, "a", n, m, l);
  RealView3D b0(q, "b0", 1, m, l);
  RealView3D b1(q, "b1", n, 1, l);
  RealView3D b2(q, "b2", n, m, 1);

  RealView3D ref0(q, "ref0", 1, m, l);
  RealView3D ref1(q, "ref1", n, 1, l);
  RealView3D ref2(q, "ref2", n, m, 1);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<double>(-1, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set random numbers to 3D view
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        a(ix, iy, iz) = rand_gen();
      }
    }
  }

  // Reduction along first axis
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      double sum = 0.0;
      for(int ix=0; ix<n; ix++) {
        sum += a(ix, iy, iz);
      }
      ref0(0, iy, iz) = sum / static_cast<double>(n);
    }
  }
  auto _a = a.mdspan();
  auto _b0 = b0.mdspan();
  Impl::mean(q, _a, _b0, 0);

  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      EXPECT_DOUBLE_EQ( b0(0, iy, iz), ref0(0, iy, iz) );
    }
  }

  // Reduction along second axis
  for(int iz=0; iz<l; iz++) {
    for(int ix=0; ix<n; ix++) {
      double sum = 0.0;
      for(int iy=0; iy<m; iy++) {
        sum += a(ix, iy, iz);
      }
      ref1(ix, 0, iz) = sum / static_cast<double>(m);
    }
  }
  auto _b1 = b1.mdspan();
  Impl::mean(q, _a, _b1, 1);

  for(int iz=0; iz<l; iz++) {
    for(int ix=0; ix<n; ix++) {
      EXPECT_DOUBLE_EQ( b1(ix, 0, iz), ref1(ix, 0, iz) );
    }
  }

  // Reduction along third axis
  for(int iy=0; iy<m; iy++) {
    for(int ix=0; ix<n; ix++) {
      double sum = 0.0;
      for(int iz=0; iz<l; iz++) {
        sum += a(ix, iy, iz);
      }
      ref2(ix, iy, 0) = sum / static_cast<double>(l);
    }
  }
  auto _b2 = b2.mdspan();
  Impl::mean(q, _a, _b2, -1);

  for(int iy=0; iy<m; iy++) {
    for(int ix=0; ix<n; ix++) {
      EXPECT_DOUBLE_EQ( b2(ix, iy, 0), ref2(ix, iy, 0) );
    }
  }
}

void test_mean_3D_to_2D(sycl::queue& q) {
  const std::size_t n = 3, m = 2, l = 5;

  RealView3D a(q, "a", n, m, l);
  RealView2D b0(q, "b0", m, l);
  RealView2D b1(q, "b1", n, l);
  RealView2D b2(q, "b2", n, m);

  RealView2D ref0(q, "ref0", m, l);
  RealView2D ref1(q, "ref1", n, l);
  RealView2D ref2(q, "ref2", n, m);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<double>(-1, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set random numbers to 3D view
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        a(ix, iy, iz) = rand_gen();
      }
    }
  }

  // Reduction along first axis
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      double sum = 0.0;
      for(int ix=0; ix<n; ix++) {
        sum += a(ix, iy, iz);
      }
      ref0(iy, iz) = sum / static_cast<double>(n);
    }
  }
  auto _a = a.mdspan();
  auto _b0 = b0.mdspan();
  Impl::mean(q, _a, _b0, 0);

  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      EXPECT_DOUBLE_EQ( b0(iy, iz), ref0(iy, iz) );
    }
  }

  // Reduction along second axis
  for(int iz=0; iz<l; iz++) {
    for(int ix=0; ix<n; ix++) {
      double sum = 0.0;
      for(int iy=0; iy<m; iy++) {
        sum += a(ix, iy, iz);
      }
      ref1(ix, iz) = sum / static_cast<double>(m);
    }
  }

  auto _b1 = b1.mdspan();
  Impl::mean(q, _a, _b1, 1);

  for(int iz=0; iz<l; iz++) {
    for(int ix=0; ix<n; ix++) {
      EXPECT_DOUBLE_EQ( b1(ix, iz), ref1(ix, iz) );
    }
  }

  // Reduction along third axis
  for(int iy=0; iy<m; iy++) {
    for(int ix=0; ix<n; ix++) {
      double sum = 0.0;
      for(int iz=0; iz<l; iz++) {
        sum += a(ix, iy, iz);
      }
      ref2(ix, iy) = sum / static_cast<double>(l);
    }
  }

  auto _b2 = b2.mdspan();
  Impl::mean(q, _a, _b2, -1);

  for(int iy=0; iy<m; iy++) {
    for(int ix=0; ix<n; ix++) {
      EXPECT_DOUBLE_EQ( b2(ix, iy), ref2(ix, iy) );
    }
  }
}

void test_axpy_1D_plus_1D(sycl::queue& q) {
  const std::size_t n = 16;
  const double alpha = 1.5, beta = 1.71;
  const double scalar = 2.31;

  RealView1D x0(q, "x0", n);
  RealView1D x1(q, "x1", n);
  RealView1D x2(q, "x2", n);
  RealView1D y0(q, "y0", n);
  RealView1D y1(q, "y1", 1);
  RealView1D z0(q, "z0", n);
  RealView1D z1(q, "z1", n);
  RealView1D z2(q, "z2", n);
  RealView1D ref0(q, "ref0", n);
  RealView1D ref1(q, "ref1", n);
  RealView1D ref2(q, "ref2", n);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<double>(-1, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set random numbers to 1D view
  // ref = alpha * x + beta * y;
  for(int ix=0; ix<n; ix++) {
    x0(ix) = rand_gen();
    x1(ix) = rand_gen();
    x2(ix) = rand_gen();
    y0(ix) = rand_gen();
  }

  y1(0) = rand_gen();

  // ref = alpha * x + beta * y;
  for(int ix=0; ix<n; ix++) {
    ref0(ix) = alpha * x0(ix) + beta * y0(ix);
    ref1(ix) = alpha * x1(ix) + beta * y1(0);
    ref2(ix) = alpha * x2(ix) + beta * scalar;
  }

  // Axpy (Outplace first then inplace)
  auto _x0 = x0.mdspan();
  auto _x1 = x1.mdspan();
  auto _x2 = x2.mdspan();
  auto _y0 = y0.mdspan();
  auto _y1 = y1.mdspan();
  auto _z0 = z0.mdspan();
  auto _z1 = z1.mdspan();
  auto _z2 = z2.mdspan();

  Impl::axpy(q, _x0, _y0, _z0, beta, alpha);
  Impl::axpy(q, _x1, _y1, _z1, beta, alpha);
  Impl::axpy(q, _x2, scalar, _z2, beta, alpha);

  Impl::axpy(q, _x0, _y0, beta, alpha);
  Impl::axpy(q, _x1, _y1, beta, alpha);
  Impl::axpy(q, _x2, scalar, beta, alpha);

  constexpr double eps = 1.e-13;
  for(int ix=0; ix<n; ix++) {
    EXPECT_NEAR( x0(ix), ref0(ix), eps );
    EXPECT_NEAR( x1(ix), ref1(ix), eps );
    EXPECT_NEAR( x2(ix), ref2(ix), eps );

    EXPECT_NEAR( z0(ix), ref0(ix), eps );
    EXPECT_NEAR( z1(ix), ref1(ix), eps );
    EXPECT_NEAR( z2(ix), ref2(ix), eps );
  }
}

void test_axpy_2D_plus_1D(sycl::queue& q) {
  const std::size_t n = 3, m = 2;
  const double alpha = 1.5, beta = 1.71;
  const double scalar = 2.31;

  RealView2D x0(q, "x0", n, m);
  RealView2D x1(q, "x1", n, m);
  RealView2D x2(q, "x2", n, m);
  RealView2D x3(q, "x3", n, m);
  RealView2D y0(q, "y0", n, 1);
  RealView2D y1(q, "y1", 1, m);
  RealView2D y2(q, "y2", 1, 1);
  RealView2D z0(q, "z0", n, m);
  RealView2D z1(q, "z1", n, m);
  RealView2D z2(q, "z2", n, m);
  RealView2D z3(q, "z3", n, m);
  RealView2D ref0(q, "ref0", n, m);
  RealView2D ref1(q, "ref1", n, m);
  RealView2D ref2(q, "ref2", n, m);
  RealView2D ref3(q, "ref3", n, m);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<double>(-1, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set random numbers to 2D view
  // ref = alpha * x + beta * y;
  for(int iy=0; iy<m; iy++) {
    for(int ix=0; ix<n; ix++) {
      x0(ix, iy) = rand_gen();
      x1(ix, iy) = rand_gen();
      x2(ix, iy) = rand_gen();
      x3(ix, iy) = rand_gen();
    }
  }

  for(int ix=0; ix<n; ix++) {
    y0(ix, 0) = rand_gen();
  }

  for(int iy=0; iy<m; iy++) {
    y1(0, iy) = rand_gen();
  }

  y2(0, 0) = rand_gen();

  // ref = alpha * x + beta * y;
  for(int iy=0; iy<m; iy++) {
    for(int ix=0; ix<n; ix++) {
      ref0(ix, iy) = alpha * x0(ix, iy) + beta * y0(ix, 0);
      ref1(ix, iy) = alpha * x1(ix, iy) + beta * y1(0, iy);
      ref2(ix, iy) = alpha * x2(ix, iy) + beta * y2(0, 0);
      ref3(ix, iy) = alpha * x3(ix, iy) + beta * scalar;
    }
  }

  // Axpy (Outplace first then inplace)
  auto _x0 = x0.mdspan(), _x1 = x1.mdspan(), _x2 = x2.mdspan(), _x3 = x3.mdspan();
  auto _y0 = y0.mdspan(), _y1 = y1.mdspan(), _y2 = y2.mdspan();
  auto _z0 = z0.mdspan(), _z1 = z1.mdspan(), _z2 = z2.mdspan(), _z3 = z3.mdspan();

  Impl::axpy(q, _x0, _y0, _z0, beta, alpha);
  Impl::axpy(q, _x1, _y1, _z1, beta, alpha);
  Impl::axpy(q, _x2, _y2, _z2, beta, alpha);
  Impl::axpy(q, _x3, scalar, _z3, beta, alpha);

  Impl::axpy(q, _x0, _y0, beta, alpha);
  Impl::axpy(q, _x1, _y1, beta, alpha);
  Impl::axpy(q, _x2, _y2, beta, alpha);
  Impl::axpy(q, _x3, scalar, beta, alpha);
  for(int iy=0; iy<m; iy++) {
    for(int ix=0; ix<n; ix++) {
      EXPECT_DOUBLE_EQ( x0(ix, iy), ref0(ix, iy) );
      EXPECT_DOUBLE_EQ( x1(ix, iy), ref1(ix, iy) );
      EXPECT_DOUBLE_EQ( x2(ix, iy), ref2(ix, iy) );
      EXPECT_DOUBLE_EQ( x3(ix, iy), ref3(ix, iy) );

      EXPECT_DOUBLE_EQ( z0(ix, iy), ref0(ix, iy) );
      EXPECT_DOUBLE_EQ( z1(ix, iy), ref1(ix, iy) );
      EXPECT_DOUBLE_EQ( z2(ix, iy), ref2(ix, iy) );
      EXPECT_DOUBLE_EQ( z3(ix, iy), ref3(ix, iy) );
    }
  }
}

void test_axpy_2D_plus_2D(sycl::queue& q) {
  const std::size_t n = 3, m = 2;
  const double alpha = 1.5, beta = 1.71;

  RealView2D x(q, "x", n, m);
  RealView2D y(q, "y", n, m);
  RealView2D z(q, "z", n, m);
  RealView2D ref(q, "ref", n, m);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<double>(-1, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set random numbers to 2D view
  for(int iy=0; iy<m; iy++) {
    for(int ix=0; ix<n; ix++) {
      x(ix, iy) = rand_gen();
      y(ix, iy) = rand_gen();
    }
  }

  // ref = alpha * x + beta * y;
  for(int iy=0; iy<m; iy++) {
    for(int ix=0; ix<n; ix++) {
      ref(ix, iy) = alpha * x(ix, iy) + beta * y(ix, iy);
    }
  }

  // Axpy (Outplace first then inplace)
  auto _x = x.mdspan();
  auto _y = y.mdspan();
  auto _z = z.mdspan();

  Impl::axpy(q, _x, _y, _z, beta, alpha);
  Impl::axpy(q, _x, _y, beta, alpha);

  constexpr double eps = 1.e-13;
  for(int iy=0; iy<m; iy++) {
    for(int ix=0; ix<n; ix++) {
      EXPECT_NEAR( x(ix, iy), ref(ix, iy), eps );
      EXPECT_NEAR( z(ix, iy), ref(ix, iy), eps );
    }
  }
}

void test_axpy_3D_plus_1D(sycl::queue& q) {
  const std::size_t n = 3, m = 2, l = 5;
  const double alpha = 1.5, beta = 1.71;
  const double scalar = 2.31;

  RealView3D x0(q, "x0", n, m, l);
  RealView3D x1(q, "x1", n, m, l);
  RealView3D x2(q, "x2", n, m, l);
  RealView3D x3(q, "x3", n, m, l);
  RealView3D x4(q, "x4", n, m, l);
  RealView3D y0(q, "y0", n, 1, 1);
  RealView3D y1(q, "y1", 1, m, 1);
  RealView3D y2(q, "y2", 1, 1, l);
  RealView3D y3(q, "y3", 1, 1, 1);
  RealView3D z0(q, "z0", n, m, l);
  RealView3D z1(q, "z1", n, m, l);
  RealView3D z2(q, "z2", n, m, l);
  RealView3D z3(q, "z3", n, m, l);
  RealView3D z4(q, "z4", n, m, l);

  RealView3D ref0(q, "ref0", n, m, l);
  RealView3D ref1(q, "ref1", n, m, l);
  RealView3D ref2(q, "ref2", n, m, l);
  RealView3D ref3(q, "ref3", n, m, l);
  RealView3D ref4(q, "ref4", n, m, l);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<double>(-1, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set random numbers to 3D view
  // ref = alpha * x + beta * y;
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        x0(ix, iy, iz) = rand_gen();
        x1(ix, iy, iz) = rand_gen();
        x2(ix, iy, iz) = rand_gen();
        x3(ix, iy, iz) = rand_gen();
        x4(ix, iy, iz) = rand_gen();
      }
    }
  }

  for(int ix=0; ix<n; ix++) {
    y0(ix, 0, 0) = rand_gen();
  }

  for(int iy=0; iy<m; iy++) {
    y1(0, iy, 0) = rand_gen();
  }

  for(int iz=0; iz<l; iz++) {
    y2(0, 0, iz) = rand_gen();
  }
  y3(0, 0, 0) = rand_gen();

  // ref = alpha * x + beta * y;
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        ref0(ix, iy, iz) = alpha * x0(ix, iy, iz) + beta * y0(ix, 0, 0);
        ref1(ix, iy, iz) = alpha * x1(ix, iy, iz) + beta * y1(0, iy, 0);
        ref2(ix, iy, iz) = alpha * x2(ix, iy, iz) + beta * y2(0, 0, iz);
        ref3(ix, iy, iz) = alpha * x3(ix, iy, iz) + beta * y3(0, 0, 0);
        ref4(ix, iy, iz) = alpha * x4(ix, iy, iz) + beta * scalar;
      }
    }
  }

  // Axpy (Outplace first then inplace)
  auto _x0 = x0.mdspan(), _x1 = x1.mdspan(), _x2 = x2.mdspan();
  auto _x3 = x3.mdspan(), _x4 = x4.mdspan();
  auto _y0 = y0.mdspan(), _y1 = y1.mdspan(), _y2 = y2.mdspan(), _y3 = y3.mdspan();
  auto _z0 = z0.mdspan(), _z1 = z1.mdspan(), _z2 = z2.mdspan();
  auto _z3 = z3.mdspan(), _z4 = z4.mdspan();
  Impl::axpy(q, _x0, _y0, _z0, beta, alpha);
  Impl::axpy(q, _x1, _y1, _z1, beta, alpha);
  Impl::axpy(q, _x2, _y2, _z2, beta, alpha);
  Impl::axpy(q, _x3, _y3, _z3, beta, alpha);
  Impl::axpy(q, _x4, scalar, _z4, beta, alpha);

  Impl::axpy(q, _x0, _y0, beta, alpha);
  Impl::axpy(q, _x1, _y1, beta, alpha);
  Impl::axpy(q, _x2, _y2, beta, alpha);
  Impl::axpy(q, _x3, _y3, beta, alpha);
  Impl::axpy(q, _x4, scalar, beta, alpha);

  constexpr double eps = 1.e-13;
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        EXPECT_NEAR( x0(ix, iy, iz), ref0(ix, iy, iz), eps );
        EXPECT_NEAR( x1(ix, iy, iz), ref1(ix, iy, iz), eps );
        EXPECT_NEAR( x2(ix, iy, iz), ref2(ix, iy, iz), eps );
        EXPECT_NEAR( x3(ix, iy, iz), ref3(ix, iy, iz), eps );
        EXPECT_NEAR( x4(ix, iy, iz), ref4(ix, iy, iz), eps );

        EXPECT_NEAR( z0(ix, iy, iz), ref0(ix, iy, iz), eps );
        EXPECT_NEAR( z1(ix, iy, iz), ref1(ix, iy, iz), eps );
        EXPECT_NEAR( z2(ix, iy, iz), ref2(ix, iy, iz), eps );
        EXPECT_NEAR( z3(ix, iy, iz), ref3(ix, iy, iz), eps );
        EXPECT_NEAR( z4(ix, iy, iz), ref4(ix, iy, iz), eps );
      }
    }
  }
}

void test_axpy_3D_plus_2D(sycl::queue& q) {
  const std::size_t n = 3, m = 2, l = 5;
  const double alpha = 1.5, beta = 1.71;

  RealView3D x0(q, "x0", n, m, l);
  RealView3D x1(q, "x1", n, m, l);
  RealView3D x2(q, "x2", n, m, l);
  RealView3D y0(q, "y0", 1, m, l);
  RealView3D y1(q, "y1", n, 1, l);
  RealView3D y2(q, "y2", n, m, 1);
  RealView3D z0(q, "z0", n, m, l);
  RealView3D z1(q, "z1", n, m, l);
  RealView3D z2(q, "z2", n, m, l);

  RealView3D ref0(q, "ref0", n, m, l);
  RealView3D ref1(q, "ref1", n, m, l);
  RealView3D ref2(q, "ref2", n, m, l);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<double>(-1, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set random numbers to 3D view
  // ref = alpha * x + beta * y;
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        x0(ix, iy, iz) = rand_gen();
        x1(ix, iy, iz) = rand_gen();
        x2(ix, iy, iz) = rand_gen();
      }
      y0(0, iy, iz) = rand_gen();
    }

    for(int ix=0; ix<n; ix++) {
      y1(ix, 0, iz) = rand_gen();
    }
  }

  for(int iy=0; iy<m; iy++) {
    for(int ix=0; ix<n; ix++) {
      y2(ix, iy, 0) = rand_gen();
    }
  }

  // ref = alpha * x + beta * y;
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        ref0(ix, iy, iz) = alpha * x0(ix, iy, iz) + beta * y0(0, iy, iz);
        ref1(ix, iy, iz) = alpha * x1(ix, iy, iz) + beta * y1(ix, 0, iz);
        ref2(ix, iy, iz) = alpha * x2(ix, iy, iz) + beta * y2(ix, iy, 0);
      }
    }
  }

  // Axpy (Outplace first then inplace)
  auto _x0 = x0.mdspan(), _x1 = x1.mdspan(), _x2 = x2.mdspan();
  auto _y0 = y0.mdspan(), _y1 = y1.mdspan(), _y2 = y2.mdspan();
  auto _z0 = z0.mdspan(), _z1 = z1.mdspan(), _z2 = z2.mdspan();

  Impl::axpy(q, _x0, _y0, _z0, beta, alpha);
  Impl::axpy(q, _x1, _y1, _z1, beta, alpha);
  Impl::axpy(q, _x2, _y2, _z2, beta, alpha);

  Impl::axpy(q, _x0, _y0, beta, alpha);
  Impl::axpy(q, _x1, _y1, beta, alpha);
  Impl::axpy(q, _x2, _y2, beta, alpha);
  constexpr double eps = 1.e-13;
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        EXPECT_NEAR( x0(ix, iy, iz), ref0(ix, iy, iz), eps );
        EXPECT_NEAR( x1(ix, iy, iz), ref1(ix, iy, iz), eps );
        EXPECT_NEAR( x2(ix, iy, iz), ref2(ix, iy, iz), eps );

        EXPECT_NEAR( z0(ix, iy, iz), ref0(ix, iy, iz), eps );
        EXPECT_NEAR( z1(ix, iy, iz), ref1(ix, iy, iz), eps );
        EXPECT_NEAR( z2(ix, iy, iz), ref2(ix, iy, iz), eps );
      }
    }
  }
}

void test_axpy_3D_plus_3D(sycl::queue& q) {
  const std::size_t n = 3, m = 2, l = 5;
  const double alpha = 1.5, beta = 1.71;

  RealView3D x(q, "x", n, m, l);
  RealView3D y(q, "y", n, m, l);
  RealView3D z(q, "z", n, m, l);
  RealView3D ref(q, "ref", n, m, l);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<double>(-1, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set random numbers to 3D view
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        x(ix, iy, iz) = rand_gen();
        y(ix, iy, iz) = rand_gen();
      }
    }
  }

  // ref = alpha * x + beta * y;
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        ref(ix, iy, iz) = alpha * x(ix, iy, iz) + beta * y(ix, iy, iz);
      }
    }
  }

  // Axpy (Outplace first then inplace)
  auto _x = x.mdspan();
  auto _y = y.mdspan();
  auto _z = z.mdspan();

  Impl::axpy(q, _x, _y, _z, beta, alpha);
  Impl::axpy(q, _x, _y, beta, alpha);

  constexpr double eps = 1.e-13;
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        EXPECT_NEAR( x(ix, iy, iz), ref(ix, iy, iz), eps );
        EXPECT_NEAR( z(ix, iy, iz), ref(ix, iy, iz), eps );
      }
    }
  }
}

void test_zeroslike_1D_to_3D(sycl::queue& q) {
  const std::size_t n = 3, m = 2, l = 5;

  RealView1D x0(q, "x0", n);
  RealView2D x1(q, "x1", n, m);
  RealView3D x2(q, "x2", n, m, l);
  RealView1D zeros0(q, "zeros0", n);
  RealView2D zeros1(q, "zeros1", n, m);
  RealView3D zeros2(q, "zeros2", n, m, l);
  RealView1D ref0(q, "ref0", n);
  RealView2D ref1(q, "ref1", n, m);
  RealView3D ref2(q, "ref2", n, m, l);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<double>(-1, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set random numbers to 3D view
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        x0(ix) = rand_gen();
        x1(ix, iy) = rand_gen();
        x2(ix, iy, iz) = rand_gen();

        zeros0(ix) = rand_gen();
        zeros1(ix, iy) = rand_gen();
        zeros2(ix, iy, iz) = rand_gen();
      }
    }
  }

  //  (Outplace first then inplace)
  auto _x0 = x0.mdspan();
  auto _x1 = x1.mdspan();
  auto _x2 = x2.mdspan();
  auto _zeros0 = zeros0.mdspan();
  auto _zeros1 = zeros1.mdspan();
  auto _zeros2 = zeros2.mdspan();

  Impl::zeros_like(q, _x0, _zeros0);
  Impl::zeros_like(q, _x1, _zeros1);
  Impl::zeros_like(q, _x2, _zeros2);

  Impl::zeros_like(q, _x0);
  Impl::zeros_like(q, _x1);
  Impl::zeros_like(q, _x2);

  for(int ix=0; ix<n; ix++) {
    EXPECT_EQ( zeros0(ix), ref0(ix) );
    EXPECT_EQ( x0(ix), ref0(ix) );
  }

  for(int iy=0; iy<m; iy++) {
    for(int ix=0; ix<n; ix++) {
      EXPECT_EQ( zeros1(ix, iy), ref1(ix, iy) );
      EXPECT_EQ( x1(ix, iy), ref1(ix, iy) );
    }
  }

  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        EXPECT_EQ( zeros2(ix, iy, iz), ref2(ix, iy, iz) );
        EXPECT_EQ( x2(ix, iy, iz), ref2(ix, iy, iz) );
      }
    }
  }
}

void test_oneslike_1D_to_3D(sycl::queue& q) {
  const std::size_t n = 3, m = 2, l = 5;

  RealView1D x0(q, "x0", n);
  RealView2D x1(q, "x1", n, m);
  RealView3D x2(q, "x2", n, m, l);
  RealView1D ones0(q, "ones0", n);
  RealView2D ones1(q, "ones1", n, m);
  RealView3D ones2(q, "ones2", n, m, l);
  RealView1D ref0(q, "ref0", n);
  RealView2D ref1(q, "ref1", n, m);
  RealView3D ref2(q, "ref2", n, m, l);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<double>(-1, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set random numbers to 3D view
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        x0(ix) = rand_gen();
        x1(ix, iy) = rand_gen();
        x2(ix, iy, iz) = rand_gen();

        ones0(ix) = rand_gen();
        ones1(ix, iy) = rand_gen();
        ones2(ix, iy, iz) = rand_gen();

        ref0(ix) = 1.0;
        ref1(ix, iy) = 1.0;
        ref2(ix, iy, iz) = 1.0;
      }
    }
  }

  //  (Outplace first then inplace)
  auto _x0 = x0.mdspan();
  auto _x1 = x1.mdspan();
  auto _x2 = x2.mdspan();
  auto _ones0 = ones0.mdspan();
  auto _ones1 = ones1.mdspan();
  auto _ones2 = ones2.mdspan();

  Impl::ones_like(q, _x0, _ones0);
  Impl::ones_like(q, _x1, _ones1);
  Impl::ones_like(q, _x2, _ones2);

  Impl::ones_like(q, _x0);
  Impl::ones_like(q, _x1);
  Impl::ones_like(q, _x2);

  for(int ix=0; ix<n; ix++) {
    EXPECT_EQ( ones0(ix), ref0(ix) );
    EXPECT_EQ( x0(ix), ref0(ix) );
  }

  for(int iy=0; iy<m; iy++) {
    for(int ix=0; ix<n; ix++) {
      EXPECT_EQ( ones1(ix, iy), ref1(ix, iy) );
      EXPECT_EQ( x1(ix, iy), ref1(ix, iy) );
    }
  }

  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        EXPECT_EQ( ones2(ix, iy, iz), ref2(ix, iy, iz) );
        EXPECT_EQ( x2(ix, iy, iz), ref2(ix, iy, iz) );
      }
    }
  }
}

void test_identity_2D_to_3D(sycl::queue& q) {
  const std::size_t m = 3, l = 5;

  RealView2D a(q, "a", m, m);
  RealView3D a_batch(q, "a_batch", m, m, l);
  RealView2D ref(q, "ref", m, m);
  RealView3D ref_batch(q, "ref_batch", m, m, l);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<double>(-1, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set random numbers to 3D view
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<m; ix++) {
        a(ix, iy) = rand_gen();
        a_batch(ix, iy, iz) = rand_gen();
      }
    }
  }

  // Set identity matrix
  for(int iz=0; iz<l; iz++) {
    for(int id=0; id<m; id++) {
      ref(id, id) = 1.0;
      ref_batch(id, id, iz) = 1.0;
    }
  }

  auto _a = a.mdspan();
  auto _a_batch = a_batch.mdspan();

  Impl::identity(q, _a);
  Impl::identity(q, _a_batch);

  for(int iy=0; iy<m; iy++) {
    for(int ix=0; ix<m; ix++) {
      EXPECT_EQ( a(ix, iy), ref(ix, iy) );
    }
  }

  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<m; ix++) {
        EXPECT_EQ( a_batch(ix, iy, iz), ref_batch(ix, iy, iz) );
      }
    }
  }
}

void test_diag_2D_to_3D(sycl::queue& q) {
  const std::size_t m = 3, l = 5;
  double exponent = -0.5;

  RealView2D a(q, "a", m, m);
  RealView3D a_batch(q, "a_batch", m, m, l);
  RealView1D v(q, "v", m);
  RealView2D v_batch(q, "v_batch", m, l);
  RealView2D ref(q, "ref", m, m);
  RealView3D ref_batch(q, "ref_batch", m, m, l);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<double>(0, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  double threshold = 1.e-3;
  auto nonzero_gen = [&]() {
    bool non_zero_found = false;
    double rand = 0;
    while(!non_zero_found) {
      rand = rand_gen();
      if(abs(rand) > threshold) {
        non_zero_found = true;
      }
    }
    return rand;
  };

  // Set random numbers to 3D view
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<m; ix++) {
        a(ix, iy) = rand_gen();
        a_batch(ix, iy, iz) = rand_gen();
      }
    }
    for(int ix=0; ix<m; ix++) {
      v(ix) = nonzero_gen();
      v_batch(ix, iz) = nonzero_gen();
    }
  }

  // Set diagonal matrix
  for(int iz=0; iz<l; iz++) {
    for(int id=0; id<m; id++) {
      ref(id, id) = pow(v(id), exponent);
      ref_batch(id, id, iz) = pow(v_batch(id, iz), exponent);
    }
  }

  auto _a = a.mdspan();
  auto _a_batch = a_batch.mdspan();
  auto _v = v.mdspan();
  auto _v_batch = v_batch.mdspan();

  Impl::diag(q, _v, _a, exponent);
  Impl::diag(q, _v_batch, _a_batch, exponent);

  constexpr double eps = 1.e-13;
  for(int iy=0; iy<m; iy++) {
    for(int ix=0; ix<m; ix++) {
      EXPECT_NEAR( a(ix, iy), ref(ix, iy), eps );
    }
  }

  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<m; ix++) {
        EXPECT_NEAR( a_batch(ix, iy, iz), ref_batch(ix, iy, iz), eps );
      }
    }
  }
}

void test_squeeze_2D(sycl::queue& q) {
  const std::size_t m = 3, n = 4;

  RealView2D a0(q, "a0", 1, n);
  RealView2D a1(q, "a1", m, 1);

  RealView1D ref0(q, "ref0", n);
  RealView1D ref1(q, "ref1", m);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<double>(0, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set random numbers to 2D view
  for(int iy=0; iy<n; iy++) {
    a0(0, iy) = rand_gen();
    ref0(iy) = a0(0, iy);
  }

  for(int ix=0; ix<m; ix++) {
    a1(ix, 0) = rand_gen();
    ref1(ix) = a1(ix, 0);
  }

  auto _a0 = a0.mdspan();
  auto _a1 = a1.mdspan();

  // auto out = Impl::squeeze(View2D in, int axis=-1);
  auto b0 = Impl::squeeze(_a0, 0);
  auto b1 = Impl::squeeze(_a1, 1);
  auto c1 = Impl::squeeze(_a1, -1);

  for(int iy=0; iy<n; iy++) {
    EXPECT_EQ( b0(iy), ref0(iy) );
  }

  for(int ix=0; ix<m; ix++) {
    EXPECT_EQ( b1(ix), ref1(ix) );
    EXPECT_EQ( c1(ix), ref1(ix) );
  }
}

void test_squeeze_3D(sycl::queue& q) {
  const std::size_t m = 3, n = 4, l = 5;
  RealView3D a0(q, "a0", 1, n, l);
  RealView3D a1(q, "a1", m, 1, l);
  RealView3D a2(q, "a2", m, n, 1);

  RealView2D ref0(q, "ref0", n, l);
  RealView2D ref1(q, "ref1", m, l);
  RealView2D ref2(q, "ref2", m, n);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<double>(0, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set random numbers to 3D view
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<n; iy++) {
      a0(0, iy, iz) = rand_gen();
      ref0(iy, iz) = a0(0, iy, iz);
    }
  }

  for(int iz=0; iz<l; iz++) {
    for(int ix=0; ix<m; ix++) {
      a1(ix, 0, iz) = rand_gen();
      ref1(ix, iz) = a1(ix, 0, iz);
    }
  }

  for(int iy=0; iy<n; iy++) {
    for(int ix=0; ix<m; ix++) {
      a2(ix, iy, 0) = rand_gen();
      ref2(ix, iy) = a2(ix, iy, 0);
    }
  }

  // auto out = Impl::squeeze(View3D in, int axis=-1);
  auto _a0 = a0.mdspan();
  auto _a1 = a1.mdspan();
  auto _a2 = a2.mdspan();

  auto b0 = Impl::squeeze(_a0, 0);
  auto b1 = Impl::squeeze(_a1, 1);
  auto b2 = Impl::squeeze(_a2, 2);
  auto c2 = Impl::squeeze(_a2, -1);

  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<n; iy++) {
      EXPECT_EQ( b0(iy, iz), ref0(iy, iz) );
    }
  }

  for(int iz=0; iz<l; iz++) {
    for(int ix=0; ix<m; ix++) {
      EXPECT_EQ( b1(ix, iz), ref1(ix, iz) );
    }
  }

  for(int iy=0; iy<n; iy++) {
    for(int ix=0; ix<m; ix++) {
      EXPECT_EQ( b2(ix, iy), ref2(ix, iy) );
      EXPECT_EQ( c2(ix, iy), ref2(ix, iy) );
    }
  }
}

void test_reshape(sycl::queue& q) {
  const std::size_t m = 3, n = 4, l = 5;

  RealView3D a0(q, "a0", m, n, l);
  RealView2D ref0(q, "ref0", m*n, l);
  RealView1D ref1(q, "ref1", m*n*l);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<double>(0, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  using layout_type = RealView3D::layout_type;

  // Set random numbers to 3D view
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<n; iy++) {
      for(int ix=0; ix<m; ix++) {
        a0(ix, iy, iz) = rand_gen();

        if(std::is_same_v<layout_type, stdex::layout_left>) {
          ref0(ix + iy * m, iz) = a0(ix, iy, iz);
          ref1(ix + iy * m + iz * m * n) = a0(ix, iy, iz);
        } else {
          ref0(ix * n + iy, iz) = a0(ix, iy, iz);
          ref1(ix * n * l + iy * l + iz) = a0(ix, iy, iz);
        }
      }
    }
  }

  auto _a0 = a0.mdspan();

  auto b0 = Impl::reshape(_a0, std::array<std::size_t, 2>({m*n, l}));
  auto b1 = Impl::reshape(_a0, std::array<std::size_t, 1>({m*n*l}));

  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<n; iy++) {
      for(int ix=0; ix<m; ix++) {
        if(std::is_same_v<layout_type, stdex::layout_left>) {
          EXPECT_EQ( b0(ix + iy * m, iz), ref0(ix + iy * m, iz) );
          EXPECT_EQ( b1(ix + iy * m + iz * m * n), ref1(ix + iy * m + iz * m * n) );
        } else {
          EXPECT_EQ( b0(ix * n + iy, iz), ref0(ix * n + iy, iz) );
          EXPECT_EQ( b1(ix * n * l + iy * l + iz), ref1(ix * n * l + iy * l + iz) );
        }
      }
    }
  }
}

void test_deep_copy(sycl::queue& q) {
  const std::size_t m = 3, n = 4, l = 5;
  RealView3D a0(q, "a0", m, n, l), b0(q, "b0", m, n, l), ref0(q, "ref0", m, n, l);
  RealView2D a1(q, "a1", m, n), b1(q, "b1", m, n), ref1(q, "ref1", m, n);
  RealView1D a2(q, "a2", m), b2(q, "b2", m), ref2(q, "ref2", m);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<double>(0, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set random numbers to 1D, 2D and 3D views
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<n; iy++) {
      for(int ix=0; ix<m; ix++) {
        a0(ix, iy, iz) = rand_gen();
        a1(ix, iy) = rand_gen();
        a2(ix) = rand_gen();
        ref0(ix, iy, iz) = a0(ix, iy, iz);
        ref1(ix, iy) = a1(ix, iy);
        ref2(ix) = a2(ix);
      }
    }
  }

  auto _a0 = a0.mdspan();
  auto _a1 = a1.mdspan();
  auto _a2 = a2.mdspan();
  auto _b0 = b0.mdspan();
  auto _b1 = b1.mdspan();
  auto _b2 = b2.mdspan();

  Impl::deep_copy(q, _a0, _b0);
  Impl::deep_copy(q, _a1, _b1);
  Impl::deep_copy(q, _a2, _b2);

  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<n; iy++) {
      for(int ix=0; ix<m; ix++) {
        EXPECT_EQ( b0(ix, iy, iz), ref0(ix, iy, iz) );
      }
    }
  }

  for(int iy=0; iy<n; iy++) {
    for(int ix=0; ix<m; ix++) {
      EXPECT_EQ( b1(ix, iy), ref1(ix, iy) );
    }
  }

  for(int ix=0; ix<m; ix++) {
    EXPECT_EQ( b2(ix), ref2(ix) );
  }
}

TEST_F( NumpyLikeSimple, Mean3Dplus3D ) {
  test_mean_3D_to_3D(*queue_);
}

TEST_F( NumpyLikeSimple, Mean3Dplus2D ) {
  test_mean_3D_to_2D(*queue_);
}

TEST_F( NumpyLikeSimple, Axpy1Dplus1D ) {
  test_axpy_1D_plus_1D(*queue_);
}

TEST_F( NumpyLikeSimple, Axpy2Dplus1D ) {
  test_axpy_2D_plus_1D(*queue_);
}

TEST_F( NumpyLikeSimple, Axpy2Dplus2D ) {
  test_axpy_2D_plus_2D(*queue_);
}

TEST_F( NumpyLikeSimple, Axpy3Dplus2D ) {
  test_axpy_3D_plus_2D(*queue_);
}

TEST_F( NumpyLikeSimple, Axpy3Dplus3D ) {
  test_axpy_3D_plus_3D(*queue_);
}

TEST_F( NumpyLikeSimple, Zeroslike2Dto3D ) {
  test_zeroslike_1D_to_3D(*queue_);
}

TEST_F( NumpyLikeSimple, Oneslike2Dto3D ) {
  test_oneslike_1D_to_3D(*queue_);
}

TEST_F( NumpyLikeSimple, Identity2Dto3D ) {
  test_identity_2D_to_3D(*queue_);
}

TEST_F( NumpyLikeSimple, DIAG2Dto3D ) {
  test_diag_2D_to_3D(*queue_);
}

TEST_F( NumpyLikeSimple, SQUEEZE2D ) {
  test_squeeze_2D(*queue_);
}

TEST_F( NumpyLikeSimple, SQUEEZE3D ) {
  test_squeeze_3D(*queue_);
}

TEST_F( NumpyLikeSimple, RESHAPE ) {
  test_reshape(*queue_);
}

TEST_F( NumpyLikeSimple, DEEP_COPY ) {
  test_deep_copy(*queue_);
}