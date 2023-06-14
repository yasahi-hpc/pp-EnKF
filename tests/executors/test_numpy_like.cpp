#include <random>
#include <functional>
#include <gtest/gtest.h>
#include <executors/numpy_like.hpp>
#include "Types.hpp"

TEST( MEAN, 3D_to_3D ) {
  const std::size_t n = 3, m = 2, l = 5;
  View3D<double> a("a", n, m, l);
  View3D<double> b0("b0", 1, m, l);
  View3D<double> b1("b1", n, 1, l);
  View3D<double> b2("b2", n, m, 1);

  View3D<double> ref0("ref0", 1, m, l);
  View3D<double> ref1("ref1", n, 1, l);
  View3D<double> ref2("ref2", n, m, 1);

  constexpr double eps = 1.e-13;

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
  a.updateDevice();

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
  Impl::mean(_a, _b0, 0);
  b0.updateSelf();

  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      EXPECT_NEAR( b0(0, iy, iz), ref0(0, iy, iz), eps );
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
  Impl::mean(_a, _b1, 1);
  b1.updateSelf();

  for(int iz=0; iz<l; iz++) {
    for(int ix=0; ix<n; ix++) {
      EXPECT_NEAR( b1(ix, 0, iz), ref1(ix, 0, iz), eps );
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
  Impl::mean(_a, _b2, -1);
  b2.updateSelf();

  for(int iy=0; iy<m; iy++) {
    for(int ix=0; ix<n; ix++) {
      EXPECT_NEAR( b2(ix, iy, 0), ref2(ix, iy, 0), eps );
    }
  }
}

TEST( MEAN, 3D_to_2D ) {
  const std::size_t n = 3, m = 2, l = 5;
  View3D<double> a("a", n, m, l);
  View2D<double> b0("b0", m, l);
  View2D<double> b1("b1", n, l);
  View2D<double> b2("b2", n, m);

  View2D<double> ref0("ref0", m, l);
  View2D<double> ref1("ref1", n, l);
  View2D<double> ref2("ref2", n, m);

  constexpr double eps = 1.e-13;

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
  a.updateDevice();

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
  Impl::mean(_a, _b0, 0);
  b0.updateSelf();

  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      EXPECT_NEAR( b0(iy, iz), ref0(iy, iz), eps );
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
  Impl::mean(_a, _b1, 1);
  b1.updateSelf();

  for(int iz=0; iz<l; iz++) {
    for(int ix=0; ix<n; ix++) {
      EXPECT_NEAR( b1(ix, iz), ref1(ix, iz), eps );
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
  Impl::mean(_a, _b2, -1);
  b2.updateSelf();

  for(int iy=0; iy<m; iy++) {
    for(int ix=0; ix<n; ix++) {
      EXPECT_NEAR( b2(ix, iy), ref2(ix, iy), eps );
    }
  }
}

TEST( AXPY, 1D_plus_1D ) {
  const std::size_t n = 16;
  const double alpha = 1.5, beta = 1.71;
  const double scalar = 2.31;

  View1D<double> x0("x0", n);
  View1D<double> x1("x1", n);
  View1D<double> x2("x2", n);
  View1D<double> y0("y0", n);
  View1D<double> y1("y1", 1);
  View1D<double> z0("z0", n);
  View1D<double> z1("z1", n);
  View1D<double> z2("z2", n);
  View1D<double> ref0("ref0", n);
  View1D<double> ref1("ref1", n);
  View1D<double> ref2("ref2", n);

  constexpr double eps = 1.e-13;

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<double>(-1, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set random numbers to 1D view
  for(int ix=0; ix<n; ix++) {
    x0(ix) = rand_gen();
    x1(ix) = rand_gen();
    x2(ix) = rand_gen();
    y0(ix) = rand_gen();
  }

  y1(0) = rand_gen();

  x0.updateDevice();
  x1.updateDevice();
  x2.updateDevice();
  y0.updateDevice();
  y1.updateDevice();

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
  Impl::axpy(_x0, _y0, _z0, beta, alpha);
  Impl::axpy(_x1, _y1, _z1, beta, alpha);
  Impl::axpy(_x2, scalar, _z2, beta, alpha);
  Impl::axpy(_x0, _y0, beta, alpha);
  Impl::axpy(_x1, _y1, beta, alpha);
  Impl::axpy(_x2, scalar, beta, alpha);

  x0.updateSelf();
  x1.updateSelf();
  x2.updateSelf();
  z0.updateSelf();
  z1.updateSelf();
  z2.updateSelf();
  for(int ix=0; ix<n; ix++) {
    EXPECT_NEAR( x0(ix), ref0(ix), eps );
    EXPECT_NEAR( x1(ix), ref1(ix), eps );
    EXPECT_NEAR( x2(ix), ref2(ix), eps );
    EXPECT_NEAR( z0(ix), ref0(ix), eps );
    EXPECT_NEAR( z1(ix), ref1(ix), eps );
    EXPECT_NEAR( z2(ix), ref2(ix), eps );
  }
}

TEST( AXPY, 2D_plus_1D ) {
  const std::size_t n = 3, m = 2;
  const double alpha = 1.5, beta = 1.71;
  const double scalar = 2.31;

  View2D<double> x0("x0", n, m);
  View2D<double> x1("x1", n, m);
  View2D<double> x2("x2", n, m);
  View2D<double> x3("x3", n, m);
  View2D<double> y0("y0", n, 1);
  View2D<double> y1("y1", 1, m);
  View2D<double> y2("y2", 1, 1);
  View2D<double> z0("z0", n, m);
  View2D<double> z1("z1", n, m);
  View2D<double> z2("z2", n, m);
  View2D<double> z3("z3", n, m);

  View2D<double> ref0("ref0", n, m);
  View2D<double> ref1("ref1", n, m);
  View2D<double> ref2("ref2", n, m);
  View2D<double> ref3("ref3", n, m);

  constexpr double eps = 1.e-13;

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

  x0.updateDevice();
  x1.updateDevice();
  x2.updateDevice();
  x3.updateDevice();
  y0.updateDevice();
  y1.updateDevice();
  y2.updateDevice();

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
  auto _x0 = x0.mdspan();
  auto _x1 = x1.mdspan();
  auto _x2 = x2.mdspan();
  auto _x3 = x3.mdspan();
  auto _y0 = y0.mdspan();
  auto _y1 = y1.mdspan();
  auto _y2 = y2.mdspan();
  auto _z0 = z0.mdspan();
  auto _z1 = z1.mdspan();
  auto _z2 = z2.mdspan();
  auto _z3 = z3.mdspan();
  Impl::axpy(_x0, _y0, _z0, beta, alpha);
  Impl::axpy(_x1, _y1, _z1, beta, alpha);
  Impl::axpy(_x2, _y2, _z2, beta, alpha);
  Impl::axpy(_x3, scalar, _z3, beta, alpha);

  Impl::axpy(_x0, _y0, beta, alpha);
  Impl::axpy(_x1, _y1, beta, alpha);
  Impl::axpy(_x2, _y2, beta, alpha);
  Impl::axpy(_x3, scalar, beta, alpha);

  x0.updateSelf();
  x1.updateSelf();
  x2.updateSelf();
  x3.updateSelf();
  z0.updateSelf();
  z1.updateSelf();
  z2.updateSelf();
  z3.updateSelf();

  for(int iy=0; iy<m; iy++) {
    for(int ix=0; ix<n; ix++) {
      EXPECT_NEAR( x0(ix, iy), ref0(ix, iy), eps );
      EXPECT_NEAR( x1(ix, iy), ref1(ix, iy), eps );
      EXPECT_NEAR( x2(ix, iy), ref2(ix, iy), eps );
      EXPECT_NEAR( x3(ix, iy), ref3(ix, iy), eps );

      EXPECT_NEAR( z0(ix, iy), ref0(ix, iy), eps );
      EXPECT_NEAR( z1(ix, iy), ref1(ix, iy), eps );
      EXPECT_NEAR( z2(ix, iy), ref2(ix, iy), eps );
      EXPECT_NEAR( z3(ix, iy), ref3(ix, iy), eps );
    }
  }
}

TEST( AXPY, 2D_plus_2D ) {
  const std::size_t n = 3, m = 2;
  const double alpha = 1.5, beta = 1.71;

  View2D<double> x("x", n, m);
  View2D<double> y("y", n, m);
  View2D<double> z("z", n, m);
  View2D<double> ref("ref", n, m);

  constexpr double eps = 1.e-13;

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
  x.updateDevice();
  y.updateDevice();

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
  Impl::axpy(_x, _y, _z, beta, alpha);
  Impl::axpy(_x, _y, beta, alpha);

  x.updateSelf();
  z.updateSelf();
  for(int iy=0; iy<m; iy++) {
    for(int ix=0; ix<n; ix++) {
      EXPECT_NEAR( x(ix, iy), ref(ix, iy), eps );
      EXPECT_NEAR( z(ix, iy), ref(ix, iy), eps );
    }
  }
}

TEST( AXPY, 3D_plus_1D ) {
  const std::size_t n = 3, m = 2, l = 5;
  const double alpha = 1.5, beta = 1.71;
  const double scalar = 2.31;

  View3D<double> x0("x0", n, m, l);
  View3D<double> x1("x1", n, m, l);
  View3D<double> x2("x2", n, m, l);
  View3D<double> x3("x3", n, m, l);
  View3D<double> x4("x4", n, m, l);
  View3D<double> y0("y0", n, 1, 1);
  View3D<double> y1("y1", 1, m, 1);
  View3D<double> y2("y2", 1, 1, l);
  View3D<double> y3("y3", 1, 1, 1);
  View3D<double> z0("z0", n, m, l);
  View3D<double> z1("z1", n, m, l);
  View3D<double> z2("z2", n, m, l);
  View3D<double> z3("z3", n, m, l);
  View3D<double> z4("z4", n, m, l);

  View3D<double> ref0("ref0", n, m, l);
  View3D<double> ref1("ref1", n, m, l);
  View3D<double> ref2("ref2", n, m, l);
  View3D<double> ref3("ref3", n, m, l);
  View3D<double> ref4("ref4", n, m, l);

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

  x0.updateDevice();
  x1.updateDevice();
  x2.updateDevice();
  x3.updateDevice();
  x4.updateDevice();
  y0.updateDevice();
  y1.updateDevice();
  y2.updateDevice();
  y3.updateDevice();

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
  auto _x0 = x0.mdspan();
  auto _x1 = x1.mdspan();
  auto _x2 = x2.mdspan();
  auto _x3 = x3.mdspan();
  auto _x4 = x4.mdspan();
  auto _y0 = y0.mdspan();
  auto _y1 = y1.mdspan();
  auto _y2 = y2.mdspan();
  auto _y3 = y3.mdspan();
  auto _z0 = z0.mdspan();
  auto _z1 = z1.mdspan();
  auto _z2 = z2.mdspan();
  auto _z3 = z3.mdspan();
  auto _z4 = z4.mdspan();
  Impl::axpy(_x0, _y0, _z0, beta, alpha);
  Impl::axpy(_x1, _y1, _z1, beta, alpha);
  Impl::axpy(_x2, _y2, _z2, beta, alpha);
  Impl::axpy(_x3, _y3, _z3, beta, alpha);
  Impl::axpy(_x4, scalar, _z4, beta, alpha);

  Impl::axpy(_x0, _y0, beta, alpha);
  Impl::axpy(_x1, _y1, beta, alpha);
  Impl::axpy(_x2, _y2, beta, alpha);
  Impl::axpy(_x3, _y3, beta, alpha);
  Impl::axpy(_x4, scalar, beta, alpha);

  x0.updateSelf();
  x1.updateSelf();
  x2.updateSelf();
  x3.updateSelf();
  x4.updateSelf();
  z0.updateSelf();
  z1.updateSelf();
  z2.updateSelf();
  z3.updateSelf();
  z4.updateSelf();

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

TEST( AXPY, 3D_plus_2D ) {
  const std::size_t n = 3, m = 2, l = 5;
  const double alpha = 1.5, beta = 1.71;

  View3D<double> x0("x0", n, m, l);
  View3D<double> x1("x1", n, m, l);
  View3D<double> x2("x2", n, m, l);
  View3D<double> y0("y0", 1, m, l);
  View3D<double> y1("y1", n, 1, l);
  View3D<double> y2("y2", n, m, 1);
  View3D<double> z0("z0", n, m, l);
  View3D<double> z1("z1", n, m, l);
  View3D<double> z2("z2", n, m, l);

  View3D<double> ref0("ref0", n, m, l);
  View3D<double> ref1("ref1", n, m, l);
  View3D<double> ref2("ref2", n, m, l);

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
  x0.updateDevice();
  x1.updateDevice();
  x2.updateDevice();
  y0.updateDevice();
  y1.updateDevice();
  y2.updateDevice();

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
  auto _x0 = x0.mdspan();
  auto _x1 = x1.mdspan();
  auto _x2 = x2.mdspan();
  auto _y0 = y0.mdspan();
  auto _y1 = y1.mdspan();
  auto _y2 = y2.mdspan();
  auto _z0 = z0.mdspan();
  auto _z1 = z1.mdspan();
  auto _z2 = z2.mdspan();

  Impl::axpy(_x0, _y0, _z0, beta, alpha);
  Impl::axpy(_x1, _y1, _z1, beta, alpha);
  Impl::axpy(_x2, _y2, _z2, beta, alpha);

  Impl::axpy(_x0, _y0, beta, alpha);
  Impl::axpy(_x1, _y1, beta, alpha);
  Impl::axpy(_x2, _y2, beta, alpha);

  x0.updateSelf();
  x1.updateSelf();
  x2.updateSelf();
  z0.updateSelf();
  z1.updateSelf();
  z2.updateSelf();

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

TEST( AXPY, 3D_plus_3D ) {
  const std::size_t n = 3, m = 2, l = 5;
  const double alpha = 1.5, beta = 1.71;
  View3D<double> x("x", n, m, l);
  View3D<double> y("y", n, m, l);
  View3D<double> z("z", n, m, l);
  View3D<double> ref("ref", n, m, l);

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
  x.updateDevice();
  y.updateDevice();

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
  Impl::axpy(_x, _y, _z, beta, alpha);
  Impl::axpy(_x, _y, beta, alpha);

  x.updateSelf();
  z.updateSelf();

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

TEST( ZEROS_LIKE, 1D_to_3D ) {
  const std::size_t n = 3, m = 2, l = 5;

  View1D<double> x0("x0", n);
  View2D<double> x1("x1", n, m);
  View3D<double> x2("x2", n, m, l);
  View1D<double> zeros0("zeros0", n);
  View2D<double> zeros1("zeros1", n, m);
  View3D<double> zeros2("zeros2", n, m, l);

  View1D<double> ref0("ref0", n);
  View2D<double> ref1("ref1", n, m);
  View3D<double> ref2("ref2", n, m, l);

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
  x0.updateDevice();
  x1.updateDevice();
  x2.updateDevice();
  zeros0.updateDevice();
  zeros1.updateDevice();
  zeros2.updateDevice();

  //  (Outplace first then inplace)
  auto _x0 = x0.mdspan();
  auto _x1 = x1.mdspan();
  auto _x2 = x2.mdspan();
  auto _zeros0 = zeros0.mdspan();
  auto _zeros1 = zeros1.mdspan();
  auto _zeros2 = zeros2.mdspan();
  Impl::zeros_like(_x0, _zeros0);
  Impl::zeros_like(_x1, _zeros1);
  Impl::zeros_like(_x2, _zeros2);

  Impl::zeros_like(_x0);
  Impl::zeros_like(_x1);
  Impl::zeros_like(_x2);

  x0.updateSelf();
  x1.updateSelf();
  x2.updateSelf();
  zeros0.updateSelf();
  zeros1.updateSelf();
  zeros2.updateSelf();

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
  View1D<double> x0("x0", n);
  View2D<double> x1("x1", n, m);
  View3D<double> x2("x2", n, m, l);
  View1D<double> ones0("ones0", n);
  View2D<double> ones1("ones1", n, m);
  View3D<double> ones2("ones2", n, m, l);

  View1D<double> ref0("ref0", n);
  View2D<double> ref1("ref1", n, m);
  View3D<double> ref2("ref2", n, m, l);

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
  x0.updateDevice();
  x1.updateDevice();
  x2.updateDevice();
  ones0.updateDevice();
  ones1.updateDevice();
  ones2.updateDevice();

  //  (Outplace first then inplace)
  auto _x0 = x0.mdspan();
  auto _x1 = x1.mdspan();
  auto _x2 = x2.mdspan();
  auto _ones0 = ones0.mdspan();
  auto _ones1 = ones1.mdspan();
  auto _ones2 = ones2.mdspan();
  Impl::ones_like(_x0, _ones0);
  Impl::ones_like(_x1, _ones1);
  Impl::ones_like(_x2, _ones2);

  Impl::ones_like(_x0);
  Impl::ones_like(_x1);
  Impl::ones_like(_x2);

  x0.updateSelf();
  x1.updateSelf();
  x2.updateSelf();
  ones0.updateSelf();
  ones1.updateSelf();
  ones2.updateSelf();

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
  View2D<double> a("a", m, m);
  View3D<double> a_batch("a_batch", m, m, l);
  View2D<double> ref("ref", m, m);
  View3D<double> ref_batch("ref_batch", m, m, l);

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
  a.updateDevice();
  a_batch.updateDevice();

  // Set identity matrix
  for(int iz=0; iz<l; iz++) {
    for(int id=0; id<m; id++) {
      ref(id, id) = 1.0;
      ref_batch(id, id, iz) = 1.0;
    }
  }

  auto _a = a.mdspan();
  auto _a_batch = a_batch.mdspan();
  Impl::identity(_a);
  Impl::identity(_a_batch);

  a.updateSelf();
  a_batch.updateSelf();
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

  View2D<double> a("a", m, m);
  View3D<double> a_batch("a_batch", m, m, l);
  View1D<double> v("v", m);
  View2D<double> v_batch("v_batch", m, l);

  View2D<double> ref("ref", m, m);
  View3D<double> ref_batch("ref_batch", m, m, l);

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
  a.updateDevice();
  a_batch.updateDevice();
  v.updateDevice();
  v_batch.updateDevice();

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

  Impl::diag(_v, _a, exponent);
  Impl::diag(_v_batch, _a_batch, exponent);

  a.updateSelf();
  a_batch.updateSelf();

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

TEST( SQUEEZE, 2D ) {
  const std::size_t m = 3, n = 4;
  using RealType = double;
  View2D<double> a0("a0", 1, n);
  View2D<double> a1("a1", m, 1);

  View1D<double> ref0("ref0", n);
  View1D<double> ref1("ref1", m);
  View1D<double> ref2("ref2", m);

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
    ref2(ix) = a1(ix, 0);
  }
  a0.updateDevice();
  a1.updateDevice();
  ref0.updateDevice();
  ref1.updateDevice();
  ref2.updateDevice();

  // auto out = Impl::squeeze(View2D in, int axis=-1);
  auto _a0 = a0.mdspan();
  auto _a1 = a1.mdspan();
  auto _ref0 = ref0.mdspan();
  auto _ref1 = ref1.mdspan();
  auto _ref2 = ref2.mdspan();
  auto b0 = Impl::squeeze(_a0, 0);
  auto b1 = Impl::squeeze(_a1, 1);
  auto c1 = Impl::squeeze(_a1, -1);

  Impl::axpy(_ref0, b0, -1); // Need to compute difference on GPUs
  Impl::axpy(_ref1, b1, -1); // Need to compute difference on GPUs
  Impl::axpy(_ref2, c1, -1); // Need to compute difference on GPUs
  ref0.updateSelf();
  ref1.updateSelf();
  ref2.updateSelf();

  constexpr double eps = 1.e-13;
  for(int iy=0; iy<n; iy++) {
    EXPECT_LE( abs( ref0(iy) ), eps );
  }

  for(int ix=0; ix<m; ix++) {
    EXPECT_LE( abs( ref1(ix) ), eps );
    EXPECT_LE( abs( ref2(ix) ), eps );
  }
}

TEST( SQUEEZE, 3D ) {
  const std::size_t m = 3, n = 4, l = 5;
  using RealType = double;

  View3D<double> a0("a0", 1, n, l);
  View3D<double> a1("a1", m, 1, l);
  View3D<double> a2("a2", m, n, 1);

  View2D<double> ref0("ref0", n, l);
  View2D<double> ref1("ref1", m, l);
  View2D<double> ref2("ref2", m, n);
  View2D<double> ref3("ref3", m, n);

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
      ref3(ix, iy) = a2(ix, iy, 0);
    }
  }

  a0.updateDevice();
  a1.updateDevice();
  a2.updateDevice();
  ref0.updateDevice();
  ref1.updateDevice();
  ref2.updateDevice();
  ref3.updateDevice();

  // auto out = Impl::squeeze(View3D in, int axis=-1);
  auto _a0 = a0.mdspan();
  auto _a1 = a1.mdspan();
  auto _a2 = a2.mdspan();
  auto _ref0 = ref0.mdspan();
  auto _ref1 = ref1.mdspan();
  auto _ref2 = ref2.mdspan();
  auto _ref3 = ref3.mdspan();

  auto b0 = Impl::squeeze(_a0, 0);
  auto b1 = Impl::squeeze(_a1, 1);
  auto b2 = Impl::squeeze(_a2, 2);
  auto c2 = Impl::squeeze(_a2, -1);

  Impl::axpy(_ref0, b0, -1); // Need to compute difference on GPUs
  Impl::axpy(_ref1, b1, -1); // Need to compute difference on GPUs
  Impl::axpy(_ref2, b2, -1); // Need to compute difference on GPUs
  Impl::axpy(_ref3, c2, -1); // Need to compute difference on GPUs
  ref0.updateSelf();
  ref1.updateSelf();
  ref2.updateSelf();
  ref3.updateSelf();

  constexpr double eps = 1.e-13;

  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<n; iy++) {
      EXPECT_LE( abs( ref0(iy, iz) ), eps );
    }
  }

  for(int iz=0; iz<l; iz++) {
    for(int ix=0; ix<m; ix++) {
      EXPECT_LE( abs( ref1(ix, iz) ), eps );
    }
  }

  for(int iy=0; iy<n; iy++) {
    for(int ix=0; ix<m; ix++) {
      EXPECT_LE( abs( ref2(ix, iy) ), eps );
      EXPECT_LE( abs( ref2(ix, iy) ), eps );
    }
  }
}

TEST( RESHAPE, 3D ) {
  const std::size_t m = 3, n = 4, l = 5;
  using RealType = double;

  View3D<double> a0("a0", m, n, l);
  View2D<double> ref0("ref0", m*n, l);
  View1D<double> ref1("ref1", m*n*l);

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

  a0.updateDevice();
  ref0.updateDevice();
  ref1.updateDevice();

  auto _a0 = a0.mdspan();
  auto _ref0 = ref0.mdspan();
  auto _ref1 = ref1.mdspan();

  auto b0 = Impl::reshape(_a0, std::array<std::size_t, 2>({m*n, l}));
  auto b1 = Impl::reshape(_a0, std::array<std::size_t, 1>({m*n*l}));

  Impl::axpy(_ref0, b0, -1); // Need to compute difference on GPUs
  Impl::axpy(_ref1, b1, -1); // Need to compute difference on GPUs
  ref0.updateSelf();
  ref1.updateSelf();

  constexpr double eps = 1.e-13;
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<n; iy++) {
      for(int ix=0; ix<m; ix++) {
        if(std::is_same_v<layout_type, stdex::layout_left>) {
          EXPECT_LE( abs( ref0(ix + iy * m, iz) ), eps );
          EXPECT_LE( abs( ref1(ix + iy * m + iz * m * n) ), eps );
        } else {
          EXPECT_LE( abs( ref0(ix * n + iy, iz) ), eps );
          EXPECT_LE( abs( ref1(ix * n * l + iy * l + iz) ), eps );
        }
      }
    }
  }
}

TEST( DEEP_COPY, 1Dto3D ) {
  const std::size_t m = 3, n = 4, l = 5;
  using RealType = double;

  View3D<double> a0("a0", m, n, l);
  View2D<double> a1("a1", m, n);
  View1D<double> a2("a2", m);
  View3D<double> b0("b0", m, n, l);
  View2D<double> b1("b1", m, n);
  View1D<double> b2("b2", m);

  View3D<double> ref0("ref0", m, n, l);
  View2D<double> ref1("ref1", m, n);
  View1D<double> ref2("ref2", m);

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
  a0.updateDevice();
  a1.updateDevice();
  a2.updateDevice();

  auto _a0 = a0.mdspan();
  auto _a1 = a1.mdspan();
  auto _a2 = a2.mdspan();
  auto _b0 = b0.mdspan();
  auto _b1 = b1.mdspan();
  auto _b2 = b2.mdspan();

  Impl::deep_copy(_a0, _b0);
  Impl::deep_copy(_a1, _b1);
  Impl::deep_copy(_a2, _b2);

  b0.updateSelf();
  b1.updateSelf();
  b2.updateSelf();

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
