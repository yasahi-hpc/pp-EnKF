#include <random>
#include <functional>
#include <gtest/gtest.h>
#include <stdpar/numpy_like.hpp>
#include "Types.hpp"

TEST( ARANGE, INT ) {
  const int start = 0;
  const int stop = 5;
  const int step = 1;

  // np.arange(5) or np.arange(0, 5, 1)
  auto a = Impl::arange(start, stop);
  auto b = Impl::arange(start, stop, step);
  const size_t len = a.size();

  ASSERT_EQ( a.size(), b.size() );

  for(int i=0; i<len; i++) {
    ASSERT_EQ(a.at(i), i);
    ASSERT_EQ(b.at(i), i);
  }
}

TEST( ARANGE, FLOAT ) {
  const double start = -10.;
  const double stop = 10.;
  const double step = 0.5;

  // np.arange(start, stop, step)
  auto a = Impl::arange(start, stop, step);
  const int len = a.size();

  const int expected_len = ceil( (stop-start) / step );
  const double expected_dt = (stop-start) / static_cast<double>(expected_len);
  const auto dt = a.at(1) - a.at(0);

  ASSERT_EQ( typeid(decltype(a.front())), typeid(double) );
  ASSERT_EQ( len, expected_len );
  ASSERT_EQ( dt, expected_dt );
  ASSERT_EQ( a.front(), start );
  ASSERT_EQ( a.back(), stop-step ); // because the last element should be unreached
}

TEST( LINSPACE, INT ) {
  const int start = 0;
  const int stop = 1;

  // np.linspace(0, 1)
  auto a = Impl::linspace<double>(start, stop);
  const size_t len = a.size();
  const size_t expected_len = 50;
  const double expected_dt = static_cast<double>(stop-start) / static_cast<double>(expected_len-1);
  const auto dt = a.at(1) - a.at(0);

  ASSERT_EQ( typeid(decltype(a.front())), typeid(double) );
  ASSERT_EQ( len, expected_len );
  ASSERT_EQ( dt, expected_dt );
  ASSERT_EQ( a.front(), static_cast<double>(start) );
  ASSERT_EQ( a.back(), static_cast<double>(stop) ); // because the last element should be included
}

TEST( MEAN, 3D_to_3D ) {
  const std::size_t n = 3, m = 2, l = 5;
  std::vector<double> _a(n*m*l);
  std::vector<double> _b0(m*l);
  std::vector<double> _b1(n*l);
  std::vector<double> _b2(n*m);
  std::vector<double> _ref0(m*l);
  std::vector<double> _ref1(n*l);
  std::vector<double> _ref2(n*m);

  Mdspan3D<double> a(_a.data(), n, m, l);
  Mdspan3D<double> b0(_b0.data(), 1, m, l);
  Mdspan3D<double> b1(_b1.data(), n, 1, l);
  Mdspan3D<double> b2(_b2.data(), n, m, 1);

  Mdspan3D<double> ref0(_ref0.data(), 1, m, l);
  Mdspan3D<double> ref1(_ref1.data(), n, 1, l);
  Mdspan3D<double> ref2(_ref2.data(), n, m, 1);

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
  Impl::mean(a, b0, 0);

  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      ASSERT_DOUBLE_EQ( b0(0, iy, iz), ref0(0, iy, iz) );
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
  Impl::mean(a, b1, 1);

  for(int iz=0; iz<l; iz++) {
    for(int ix=0; ix<n; ix++) {
      ASSERT_DOUBLE_EQ( b1(ix, 0, iz), ref1(ix, 0, iz) );
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
  Impl::mean(a, b2, -1);

  for(int iy=0; iy<m; iy++) {
    for(int ix=0; ix<n; ix++) {
      ASSERT_DOUBLE_EQ( b2(ix, iy, 0), ref2(ix, iy, 0) );
    }
  }
}

TEST( MEAN, 3D_to_2D ) {
  const std::size_t n = 3, m = 2, l = 5;
  std::vector<double> _a(n*m*l);
  std::vector<double> _b0(m*l);
  std::vector<double> _b1(n*l);
  std::vector<double> _b2(n*m);
  std::vector<double> _ref0(m*l);
  std::vector<double> _ref1(n*l);
  std::vector<double> _ref2(n*m);

  Mdspan3D<double> a(_a.data(), n, m, l);
  Mdspan2D<double> b0(_b0.data(), m, l);
  Mdspan2D<double> b1(_b1.data(), n, l);
  Mdspan2D<double> b2(_b2.data(), n, m);

  Mdspan2D<double> ref0(_ref0.data(), m, l);
  Mdspan2D<double> ref1(_ref1.data(), n, l);
  Mdspan2D<double> ref2(_ref2.data(), n, m);

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
  Impl::mean(a, b0, 0);

  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      ASSERT_DOUBLE_EQ( b0(iy, iz), ref0(iy, iz) );
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
  Impl::mean(a, b1, 1);

  for(int iz=0; iz<l; iz++) {
    for(int ix=0; ix<n; ix++) {
      ASSERT_DOUBLE_EQ( b1(ix, iz), ref1(ix, iz) );
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
  Impl::mean(a, b2, -1);

  for(int iy=0; iy<m; iy++) {
    for(int ix=0; ix<n; ix++) {
      ASSERT_DOUBLE_EQ( b2(ix, iy), ref2(ix, iy) );
    }
  }
}

TEST( AXPY, 2D_plus_1D ) {
  const std::size_t n = 3, m = 2;
  const double alpha = 1.5, beta = 1.71;
  const double scalar = 2.31;
  std::vector<double> _x0(n*m);
  std::vector<double> _x1(n*m);
  std::vector<double> _x2(n*m);
  std::vector<double> _x3(n*m);
  std::vector<double> _y0(n);
  std::vector<double> _y1(m);
  std::vector<double> _y2(1);
  std::vector<double> _z0(n*m);
  std::vector<double> _z1(n*m);
  std::vector<double> _z2(n*m);
  std::vector<double> _z3(n*m);
  std::vector<double> _ref0(n*m);
  std::vector<double> _ref1(n*m);
  std::vector<double> _ref2(n*m);
  std::vector<double> _ref3(n*m);

  Mdspan2D<double> x0(_x0.data(), n, m);
  Mdspan2D<double> x1(_x1.data(), n, m);
  Mdspan2D<double> x2(_x2.data(), n, m);
  Mdspan2D<double> x3(_x3.data(), n, m);
  Mdspan2D<double> y0(_y0.data(), n, 1);
  Mdspan2D<double> y1(_y1.data(), 1, m);
  Mdspan2D<double> y2(_y2.data(), 1, 1);
  Mdspan2D<double> z0(_z0.data(), n, m);
  Mdspan2D<double> z1(_z1.data(), n, m);
  Mdspan2D<double> z2(_z2.data(), n, m);
  Mdspan2D<double> z3(_z3.data(), n, m);

  Mdspan2D<double> ref0(_ref0.data(), n, m);
  Mdspan2D<double> ref1(_ref1.data(), n, m);
  Mdspan2D<double> ref2(_ref2.data(), n, m);
  Mdspan2D<double> ref3(_ref3.data(), n, m);

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
  Impl::axpy(x0, y0, z0, beta, alpha);
  Impl::axpy(x1, y1, z1, beta, alpha);
  Impl::axpy(x2, y2, z2, beta, alpha);
  Impl::axpy(x3, scalar, z3, beta, alpha);

  Impl::axpy(x0, y0, beta, alpha);
  Impl::axpy(x1, y1, beta, alpha);
  Impl::axpy(x2, y2, beta, alpha);
  Impl::axpy(x3, scalar, beta, alpha);
  for(int iy=0; iy<m; iy++) {
    for(int ix=0; ix<n; ix++) {
      ASSERT_DOUBLE_EQ( x0(ix, iy), ref0(ix, iy) );
      ASSERT_DOUBLE_EQ( x1(ix, iy), ref1(ix, iy) );
      ASSERT_DOUBLE_EQ( x2(ix, iy), ref2(ix, iy) );
      ASSERT_DOUBLE_EQ( x3(ix, iy), ref3(ix, iy) );

      ASSERT_DOUBLE_EQ( z0(ix, iy), ref0(ix, iy) );
      ASSERT_DOUBLE_EQ( z1(ix, iy), ref1(ix, iy) );
      ASSERT_DOUBLE_EQ( z2(ix, iy), ref2(ix, iy) );
      ASSERT_DOUBLE_EQ( z3(ix, iy), ref3(ix, iy) );
    }
  }
}

TEST( AXPY, 2D_plus_2D ) {
  const std::size_t n = 3, m = 2;
  const double alpha = 1.5, beta = 1.71;
  std::vector<double> _x(n*m);
  std::vector<double> _y(n*m);
  std::vector<double> _z(n*m);
  std::vector<double> _ref(n*m);

  Mdspan2D<double> x(_x.data(), n, m);
  Mdspan2D<double> y(_y.data(), n, m);
  Mdspan2D<double> z(_z.data(), n, m);
  Mdspan2D<double> ref(_ref.data(), n, m);

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
  Impl::axpy(x, y, z, beta, alpha);
  Impl::axpy(x, y, beta, alpha);
  for(int iy=0; iy<m; iy++) {
    for(int ix=0; ix<n; ix++) {
      ASSERT_DOUBLE_EQ( x(ix, iy), ref(ix, iy) );
      ASSERT_DOUBLE_EQ( z(ix, iy), ref(ix, iy) );
    }
  }
}

TEST( AXPY, 3D_plus_1D ) {
  const std::size_t n = 3, m = 2, l = 5;
  const double alpha = 1.5, beta = 1.71;
  const double scalar = 2.31;
  std::vector<double> _x0(n*m*l);
  std::vector<double> _x1(n*m*l);
  std::vector<double> _x2(n*m*l);
  std::vector<double> _x3(n*m*l);
  std::vector<double> _x4(n*m*l);
  std::vector<double> _y0(n);
  std::vector<double> _y1(m);
  std::vector<double> _y2(l);
  std::vector<double> _y3(1);
  std::vector<double> _z0(n*m*l);
  std::vector<double> _z1(n*m*l);
  std::vector<double> _z2(n*m*l);
  std::vector<double> _z3(n*m*l);
  std::vector<double> _z4(n*m*l);
  std::vector<double> _ref0(n*m*l);
  std::vector<double> _ref1(n*m*l);
  std::vector<double> _ref2(n*m*l);
  std::vector<double> _ref3(n*m*l);
  std::vector<double> _ref4(n*m*l);

  Mdspan3D<double> x0(_x0.data(), n, m, l);
  Mdspan3D<double> x1(_x1.data(), n, m, l);
  Mdspan3D<double> x2(_x2.data(), n, m, l);
  Mdspan3D<double> x3(_x3.data(), n, m, l);
  Mdspan3D<double> x4(_x4.data(), n, m, l);
  Mdspan3D<double> y0(_y0.data(), n, 1, 1);
  Mdspan3D<double> y1(_y1.data(), 1, m, 1);
  Mdspan3D<double> y2(_y2.data(), 1, 1, l);
  Mdspan3D<double> y3(_y3.data(), 1, 1, 1);
  Mdspan3D<double> z0(_z0.data(), n, m, l);
  Mdspan3D<double> z1(_z1.data(), n, m, l);
  Mdspan3D<double> z2(_z2.data(), n, m, l);
  Mdspan3D<double> z3(_z3.data(), n, m, l);
  Mdspan3D<double> z4(_z4.data(), n, m, l);

  Mdspan3D<double> ref0(_ref0.data(), n, m, l);
  Mdspan3D<double> ref1(_ref1.data(), n, m, l);
  Mdspan3D<double> ref2(_ref2.data(), n, m, l);
  Mdspan3D<double> ref3(_ref3.data(), n, m, l);
  Mdspan3D<double> ref4(_ref4.data(), n, m, l);

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
  Impl::axpy(x0, y0, z0, beta, alpha);
  Impl::axpy(x1, y1, z1, beta, alpha);
  Impl::axpy(x2, y2, z2, beta, alpha);
  Impl::axpy(x3, y3, z3, beta, alpha);
  Impl::axpy(x4, scalar, z4, beta, alpha);

  Impl::axpy(x0, y0, beta, alpha);
  Impl::axpy(x1, y1, beta, alpha);
  Impl::axpy(x2, y2, beta, alpha);
  Impl::axpy(x3, y3, beta, alpha);
  Impl::axpy(x4, scalar, beta, alpha);

  constexpr double eps = 1.e-13;
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        ASSERT_NEAR( x0(ix, iy, iz), ref0(ix, iy, iz), eps );
        ASSERT_NEAR( x1(ix, iy, iz), ref1(ix, iy, iz), eps );
        ASSERT_NEAR( x2(ix, iy, iz), ref2(ix, iy, iz), eps );
        ASSERT_NEAR( x3(ix, iy, iz), ref3(ix, iy, iz), eps );
        ASSERT_NEAR( x4(ix, iy, iz), ref4(ix, iy, iz), eps );

        ASSERT_NEAR( z0(ix, iy, iz), ref0(ix, iy, iz), eps );
        ASSERT_NEAR( z1(ix, iy, iz), ref1(ix, iy, iz), eps );
        ASSERT_NEAR( z2(ix, iy, iz), ref2(ix, iy, iz), eps );
        ASSERT_NEAR( z3(ix, iy, iz), ref3(ix, iy, iz), eps );
        ASSERT_NEAR( z4(ix, iy, iz), ref4(ix, iy, iz), eps );
      }
    }
  }
}

TEST( AXPY, 3D_plus_2D ) {
  const std::size_t n = 3, m = 2, l = 5;
  const double alpha = 1.5, beta = 1.71;
  std::vector<double> _x0(n*m*l);
  std::vector<double> _x1(n*m*l);
  std::vector<double> _x2(n*m*l);
  std::vector<double> _y0(m*l);
  std::vector<double> _y1(n*l);
  std::vector<double> _y2(n*m);
  std::vector<double> _z0(n*m*l);
  std::vector<double> _z1(n*m*l);
  std::vector<double> _z2(n*m*l);

  std::vector<double> _ref0(n*m*l);
  std::vector<double> _ref1(n*m*l);
  std::vector<double> _ref2(n*m*l);

  Mdspan3D<double> x0(_x0.data(), n, m, l);
  Mdspan3D<double> x1(_x1.data(), n, m, l);
  Mdspan3D<double> x2(_x2.data(), n, m, l);
  Mdspan3D<double> y0(_y0.data(), 1, m, l);
  Mdspan3D<double> y1(_y1.data(), n, 1, l);
  Mdspan3D<double> y2(_y2.data(), n, m, 1);
  Mdspan3D<double> z0(_z0.data(), n, m, l);
  Mdspan3D<double> z1(_z1.data(), n, m, l);
  Mdspan3D<double> z2(_z2.data(), n, m, l);

  Mdspan3D<double> ref0(_ref0.data(), n, m, l);
  Mdspan3D<double> ref1(_ref1.data(), n, m, l);
  Mdspan3D<double> ref2(_ref2.data(), n, m, l);

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
  Impl::axpy(x0, y0, z0, beta, alpha);
  Impl::axpy(x1, y1, z1, beta, alpha);
  Impl::axpy(x2, y2, z2, beta, alpha);

  Impl::axpy(x0, y0, beta, alpha);
  Impl::axpy(x1, y1, beta, alpha);
  Impl::axpy(x2, y2, beta, alpha);
  constexpr double eps = 1.e-13;
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        ASSERT_NEAR( x0(ix, iy, iz), ref0(ix, iy, iz), eps );
        ASSERT_NEAR( x1(ix, iy, iz), ref1(ix, iy, iz), eps );
        ASSERT_NEAR( x2(ix, iy, iz), ref2(ix, iy, iz), eps );

        ASSERT_NEAR( z0(ix, iy, iz), ref0(ix, iy, iz), eps );
        ASSERT_NEAR( z1(ix, iy, iz), ref1(ix, iy, iz), eps );
        ASSERT_NEAR( z2(ix, iy, iz), ref2(ix, iy, iz), eps );
      }
    }
  }
}

TEST( AXPY, 3D_plus_3D ) {
  const std::size_t n = 3, m = 2, l = 5;
  const double alpha = 1.5, beta = 1.71;
  std::vector<double> _x(n*m*l);
  std::vector<double> _y(n*m*l);
  std::vector<double> _z(n*m*l);
  std::vector<double> _ref(n*m*l);

  Mdspan3D<double> x(_x.data(), n, m, l);
  Mdspan3D<double> y(_y.data(), n, m, l);
  Mdspan3D<double> z(_z.data(), n, m, l);
  Mdspan3D<double> ref(_ref.data(), n, m, l);

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
  Impl::axpy(x, y, z, beta, alpha);
  Impl::axpy(x, y, beta, alpha);
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        ASSERT_DOUBLE_EQ( x(ix, iy, iz), ref(ix, iy, iz) );
        ASSERT_DOUBLE_EQ( z(ix, iy, iz), ref(ix, iy, iz) );
      }
    }
  }
}

TEST( ZEROS_LIKE, 1D_to_3D ) {
  const std::size_t n = 3, m = 2, l = 5;
  std::vector<double> _x0(n);
  std::vector<double> _x1(n*m);
  std::vector<double> _x2(n*m*l);
  std::vector<double> _zeros0(n);
  std::vector<double> _zeros1(n*m);
  std::vector<double> _zeros2(n*m*l);

  std::vector<double> _ref0(n, 0.0);
  std::vector<double> _ref1(n*m, 0.0);
  std::vector<double> _ref2(n*m*l, 0.0);

  Mdspan1D<double> x0(_x0.data(), n);
  Mdspan2D<double> x1(_x1.data(), n, m);
  Mdspan3D<double> x2(_x2.data(), n, m, l);
  Mdspan1D<double> zeros0(_zeros0.data(), n);
  Mdspan2D<double> zeros1(_zeros1.data(), n, m);
  Mdspan3D<double> zeros2(_zeros2.data(), n, m, l);

  Mdspan1D<double> ref0(_ref0.data(), n);
  Mdspan2D<double> ref1(_ref1.data(), n, m);
  Mdspan3D<double> ref2(_ref2.data(), n, m, l);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<double>(-1, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set random numbers to 3D view
  // Refs are initialized with zeros
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
  Impl::zeros_like(x0, zeros0);
  Impl::zeros_like(x1, zeros1);
  Impl::zeros_like(x2, zeros2);

  Impl::zeros_like(x0);
  Impl::zeros_like(x1);
  Impl::zeros_like(x2);

  for(int ix=0; ix<n; ix++) {
    ASSERT_EQ( zeros0(ix), ref0(ix) );
    ASSERT_EQ( x0(ix), ref0(ix) );
  }

  for(int iy=0; iy<m; iy++) {
    for(int ix=0; ix<n; ix++) {
      ASSERT_EQ( zeros1(ix, iy), ref1(ix, iy) );
      ASSERT_EQ( x1(ix, iy), ref1(ix, iy) );
    }
  }

  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        ASSERT_EQ( zeros2(ix, iy, iz), ref2(ix, iy, iz) );
        ASSERT_EQ( x2(ix, iy, iz), ref2(ix, iy, iz) );
      }
    }
  }
}

TEST( ONES_LIKE, 1D_to_3D ) {
  const std::size_t n = 3, m = 2, l = 5;
  std::vector<double> _x0(n);
  std::vector<double> _x1(n*m);
  std::vector<double> _x2(n*m*l);
  std::vector<double> _ones0(n);
  std::vector<double> _ones1(n*m);
  std::vector<double> _ones2(n*m*l);

  std::vector<double> _ref0(n, 1.0);
  std::vector<double> _ref1(n*m, 1.0);
  std::vector<double> _ref2(n*m*l, 1.0);

  Mdspan1D<double> x0(_x0.data(), n);
  Mdspan2D<double> x1(_x1.data(), n, m);
  Mdspan3D<double> x2(_x2.data(), n, m, l);
  Mdspan1D<double> ones0(_ones0.data(), n);
  Mdspan2D<double> ones1(_ones1.data(), n, m);
  Mdspan3D<double> ones2(_ones2.data(), n, m, l);

  Mdspan1D<double> ref0(_ref0.data(), n);
  Mdspan2D<double> ref1(_ref1.data(), n, m);
  Mdspan3D<double> ref2(_ref2.data(), n, m, l);

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
      }
    }
  }

  //  (Outplace first then inplace)
  Impl::ones_like(x0, ones0);
  Impl::ones_like(x1, ones1);
  Impl::ones_like(x2, ones2);

  Impl::ones_like(x0);
  Impl::ones_like(x1);
  Impl::ones_like(x2);

  for(int ix=0; ix<n; ix++) {
    ASSERT_EQ( ones0(ix), ref0(ix) );
    ASSERT_EQ( x0(ix), ref0(ix) );
  }

  for(int iy=0; iy<m; iy++) {
    for(int ix=0; ix<n; ix++) {
      ASSERT_EQ( ones1(ix, iy), ref1(ix, iy) );
      ASSERT_EQ( x1(ix, iy), ref1(ix, iy) );
    }
  }

  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        ASSERT_EQ( ones2(ix, iy, iz), ref2(ix, iy, iz) );
        ASSERT_EQ( x2(ix, iy, iz), ref2(ix, iy, iz) );
      }
    }
  }
}

TEST( IDENTITY, 2D_and_3D ) {
  const std::size_t m = 3, l = 5;
  std::vector<double> _a(m*m);
  std::vector<double> _a_batch(m*m*l);

  std::vector<double> _ref(m*m);
  std::vector<double> _ref_batch(m*m*l);

  Mdspan2D<double> a(_a.data(), m, m);
  Mdspan3D<double> a_batch(_a_batch.data(), m, m, l);
  Mdspan2D<double> ref(_ref.data(), m, m);
  Mdspan3D<double> ref_batch(_ref_batch.data(), m, m, l);

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

  Impl::identity(a);
  Impl::identity(a_batch);

  for(int iy=0; iy<m; iy++) {
    for(int ix=0; ix<m; ix++) {
      ASSERT_EQ( a(ix, iy), ref(ix, iy) );
    }
  }

  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<m; ix++) {
        ASSERT_EQ( a_batch(ix, iy, iz), ref_batch(ix, iy, iz) );
      }
    }
  }
}

TEST( DIAG, 2D_and_3D ) {
  const std::size_t m = 3, l = 5;
  double exponent = -0.5;
  std::vector<double> _a(m*m);
  std::vector<double> _a_batch(m*m*l);
  std::vector<double> _v(m);
  std::vector<double> _v_batch(m*l);

  std::vector<double> _ref(m*m);
  std::vector<double> _ref_batch(m*m*l);

  Mdspan2D<double> a(_a.data(), m, m);
  Mdspan3D<double> a_batch(_a_batch.data(), m, m, l);
  Mdspan1D<double> v(_v.data(), m);
  Mdspan2D<double> v_batch(_v_batch.data(), m, l);

  Mdspan2D<double> ref(_ref.data(), m, m);
  Mdspan3D<double> ref_batch(_ref_batch.data(), m, m, l);

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

  Impl::diag(v, a, exponent);
  Impl::diag(v_batch, a_batch, exponent);

  constexpr double eps = 1.e-13;
  for(int iy=0; iy<m; iy++) {
    for(int ix=0; ix<m; ix++) {
      ASSERT_NEAR( a(ix, iy), ref(ix, iy), eps );
    }
  }

  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<m; ix++) {
        ASSERT_NEAR( a_batch(ix, iy, iz), ref_batch(ix, iy, iz), eps );
      }
    }
  }
}

TEST( SQUEEZE, 2D ) {
  const std::size_t m = 3, n = 4;
  using RealType = double;
  std::vector<RealType> _a0(n);
  std::vector<RealType> _a1(m);

  std::vector<RealType> _ref0(n);
  std::vector<RealType> _ref1(m);

  Mdspan2D<double> a0(_a0.data(), 1, n);
  Mdspan2D<double> a1(_a1.data(), m, 1);

  Mdspan1D<double> ref0(_ref0.data(), n);
  Mdspan1D<double> ref1(_ref1.data(), m);

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

  // auto out = Impl::squeeze(View2D in, int axis=-1);
  auto b0 = Impl::squeeze(a0, 0);
  auto b1 = Impl::squeeze(a1, 1);
  auto c1 = Impl::squeeze(a1, -1);

  for(int iy=0; iy<n; iy++) {
    ASSERT_EQ( b0(iy), ref0(iy) );
  }

  for(int ix=0; ix<m; ix++) {
    ASSERT_EQ( b1(ix), ref1(ix) );
    ASSERT_EQ( c1(ix), ref1(ix) );
  }
}

TEST( SQUEEZE, 3D ) {
  const std::size_t m = 3, n = 4, l = 5;
  using RealType = double;
  std::vector<RealType> _a0(n*l);
  std::vector<RealType> _a1(m*l);
  std::vector<RealType> _a2(m*n);

  std::vector<RealType> _ref0(n*l);
  std::vector<RealType> _ref1(m*l);
  std::vector<RealType> _ref2(m*n);

  Mdspan3D<double> a0(_a0.data(), 1, n, l);
  Mdspan3D<double> a1(_a1.data(), m, 1, l);
  Mdspan3D<double> a2(_a2.data(), m, n, 1);

  Mdspan2D<double> ref0(_ref0.data(), n, l);
  Mdspan2D<double> ref1(_ref1.data(), m, l);
  Mdspan2D<double> ref2(_ref2.data(), m, n);

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
  auto b0 = Impl::squeeze(a0, 0);
  auto b1 = Impl::squeeze(a1, 1);
  auto b2 = Impl::squeeze(a2, 2);
  auto c2 = Impl::squeeze(a2, -1);

  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<n; iy++) {
      ASSERT_EQ( b0(iy, iz), ref0(iy, iz) );
    }
  }

  for(int iz=0; iz<l; iz++) {
    for(int ix=0; ix<m; ix++) {
      ASSERT_EQ( b1(ix, iz), ref1(ix, iz) );
    }
  }

  for(int iy=0; iy<n; iy++) {
    for(int ix=0; ix<m; ix++) {
      ASSERT_EQ( b2(ix, iy), ref2(ix, iy) );
      ASSERT_EQ( c2(ix, iy), ref2(ix, iy) );
    }
  }
}

TEST( RESHAPE, 3D ) {
  const std::size_t m = 3, n = 4, l = 5;
  using RealType = double;
  std::vector<RealType> _a0(m*n*l);

  std::vector<RealType> _ref0(m*n*l);
  std::vector<RealType> _ref1(m*n*l);

  Mdspan3D<double> a0(_a0.data(), m, n, l);

  Mdspan2D<double> ref0(_ref0.data(), m*n, l);
  Mdspan1D<double> ref1(_ref1.data(), m*n*l);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<double>(0, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  using layout_type = View3D<double>::layout_type;

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

  auto b0 = Impl::reshape(a0, std::array<std::size_t, 2>({m*n, l}));
  auto b1 = Impl::reshape(a0, std::array<std::size_t, 1>({m*n*l}));

  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<n; iy++) {
      for(int ix=0; ix<m; ix++) {
        if(std::is_same_v<layout_type, stdex::layout_left>) {
          ASSERT_EQ( b0(ix + iy * m, iz), ref0(ix + iy * m, iz) );
          ASSERT_EQ( b1(ix + iy * m + iz * m * n), ref1(ix + iy * m + iz * m * n) );
        } else {
          ASSERT_EQ( b0(ix * n + iy, iz), ref0(ix * n + iy, iz) );
          ASSERT_EQ( b1(ix * n * l + iy * l + iz), ref1(ix * n * l + iy * l + iz) );
        }
      }
    }
  }
}

TEST( DEEP_COPY, 1Dto3D ) {
  const std::size_t m = 3, n = 4, l = 5;
  using RealType = double;
  std::vector<RealType> _a0(m*n*l);
  std::vector<RealType> _a1(m*n);
  std::vector<RealType> _a2(m);
  std::vector<RealType> _b0(m*n*l);
  std::vector<RealType> _b1(m*n);
  std::vector<RealType> _b2(m);

  std::vector<RealType> _ref0(m*n*l);
  std::vector<RealType> _ref1(m*n);
  std::vector<RealType> _ref2(m);

  Mdspan3D<double> a0(_a0.data(), m, n, l);
  Mdspan2D<double> a1(_a1.data(), m, n);
  Mdspan1D<double> a2(_a2.data(), m);
  Mdspan3D<double> b0(_b0.data(), m, n, l);
  Mdspan2D<double> b1(_b1.data(), m, n);
  Mdspan1D<double> b2(_b2.data(), m);

  Mdspan3D<double> ref0(_ref0.data(), m, n, l);
  Mdspan2D<double> ref1(_ref1.data(), m, n);
  Mdspan1D<double> ref2(_ref2.data(), m);

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

  Impl::deep_copy(a0, b0);
  Impl::deep_copy(a1, b1);
  Impl::deep_copy(a2, b2);

  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<n; iy++) {
      for(int ix=0; ix<m; ix++) {
        ASSERT_EQ( b0(ix, iy, iz), ref0(ix, iy, iz) );
      }
    }
  }

  for(int iy=0; iy<n; iy++) {
    for(int ix=0; ix<m; ix++) {
      ASSERT_EQ( b1(ix, iy), ref1(ix, iy) );
    }
  }

  for(int ix=0; ix<m; ix++) {
    ASSERT_EQ( b2(ix), ref2(ix) );
  }
}
