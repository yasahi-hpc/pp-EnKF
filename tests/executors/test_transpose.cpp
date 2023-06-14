#include <gtest/gtest.h>
#include "Types.hpp"
#include <executors/Transpose.hpp>

TEST( TRANSPOSE2D, FP32 ) {
  using RealType = float;
  const std::size_t rows = 16;
  const std::size_t rows2 = 15;
  const std::size_t cols = rows * 2;
  const std::size_t cols2 = rows2 * 2;

  // Original arrays: A, Transposed Arrays: B
  View2D<RealType> A_even("A_even", rows, cols);
  View2D<RealType> A_odd("A_odd", rows2, cols2);
  View2D<RealType> B_even("B_even", cols, rows);
  View2D<RealType> B_odd("B_odd", cols2, rows2);

  View2D<RealType> Ref_even("Ref_even", cols, rows);
  View2D<RealType> Ref_odd("Ref_odd", cols2, rows2);

  // Set random numbers to 2D view
  for(int i=0; i<cols; i++) {
    for(int j=0; j<rows; j++) {
      A_even(j, i) = i * 0.12 + j * 0.5 + 0.001;
      Ref_even(i, j) = A_even(j, i);
    }
  }

  for(int i=0; i<cols2; i++) {
    for(int j=0; j<rows2; j++) {
      A_odd(j, i) = i * 0.13 + j * 0.51 + 0.002;
      Ref_odd(i, j) = A_odd(j, i);
    }
  }
  A_even.updateDevice();
  A_odd.updateDevice();

  constexpr RealType eps = 1.e-5;
  auto _A_even = A_even.mdspan();
  auto _B_even = B_even.mdspan();
  Impl::transpose(_A_even, _B_even);
  B_even.updateSelf();

  for(int i=0; i<cols; i++) {
    for(int j=0; j<rows; j++) {
      EXPECT_NEAR( B_even(i, j), Ref_even(i, j), eps );
    }
  }

  auto _A_odd = A_odd.mdspan();
  auto _B_odd = B_odd.mdspan();
  Impl::transpose(_A_odd, _B_odd);
  B_odd.updateSelf();

  for(int i=0; i<cols2; i++) {
    for(int j=0; j<rows2; j++) {
      EXPECT_NEAR( B_odd(i, j), Ref_odd(i, j), eps );
    }
  }
}

TEST( TRANSPOSE2D, FP64 ) {
  using RealType = double;
  const std::size_t rows = 16;
  const std::size_t rows2 = 15;
  const std::size_t cols = rows * 2;
  const std::size_t cols2 = rows2 * 2;

  // Original arrays: A, Transposed Arrays: B
  View2D<RealType> A_even("A_even", rows, cols);
  View2D<RealType> A_odd("A_odd", rows2, cols2);
  View2D<RealType> B_even("B_even", cols, rows);
  View2D<RealType> B_odd("B_odd", cols2, rows2);

  View2D<RealType> Ref_even("Ref_even", cols, rows);
  View2D<RealType> Ref_odd("Ref_odd", cols2, rows2);

  // Set random numbers to 2D view
  for(int i=0; i<cols; i++) {
    for(int j=0; j<rows; j++) {
      A_even(j, i) = i * 0.12 + j * 0.5 + 0.001;
      Ref_even(i, j) = A_even(j, i);
    }
  }

  for(int i=0; i<cols2; i++) {
    for(int j=0; j<rows2; j++) {
      A_odd(j, i) = i * 0.13 + j * 0.51 + 0.002;
      Ref_odd(i, j) = A_odd(j, i);
    }
  }
  A_even.updateDevice();
  A_odd.updateDevice();

  constexpr RealType eps = 1.e-12;
  auto _A_even = A_even.mdspan();
  auto _B_even = B_even.mdspan();
  Impl::transpose(_A_even, _B_even);
  B_even.updateSelf();

  for(int i=0; i<cols; i++) {
    for(int j=0; j<rows; j++) {
      EXPECT_NEAR( B_even(i, j), Ref_even(i, j), eps );
    }
  }

  auto _A_odd = A_odd.mdspan();
  auto _B_odd = B_odd.mdspan();
  Impl::transpose(_A_odd, _B_odd);
  B_odd.updateSelf();

  for(int i=0; i<cols2; i++) {
    for(int j=0; j<rows2; j++) {
      EXPECT_NEAR( B_odd(i, j), Ref_odd(i, j), eps );
    }
  }
}

TEST( TRANSPOSE2D_batched, FP32 ) {
  using RealType = float;
  const std::size_t n = 16, m = 15, l = 4;

  // Original arrays
  View3D<RealType> x("x", n, m, l);
  View3D<RealType> x0("x0", n, m, l);
  View3D<RealType> x1("x1", n, l, m);
  View3D<RealType> x2("x2", m, n, l);
  View3D<RealType> x3("x3", m, l, n);
  View3D<RealType> x4("x4", l, n, m);
  View3D<RealType> x5("x5", l, m, n);

  View3D<RealType> ref0("ref0", n, m, l);
  View3D<RealType> ref1("ref1", n, l, m);
  View3D<RealType> ref2("ref2", m, n, l);
  View3D<RealType> ref3("ref3", m, l, n);
  View3D<RealType> ref4("ref4", l, n, m);
  View3D<RealType> ref5("ref5", l, m, n);

  // Set random numbers to 3D view
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        x(ix, iy, iz) = ix * 0.01 + iy * 0.203 + iz * 0.07;

        ref0(ix, iy, iz) = x(ix, iy, iz);
        ref1(ix, iz, iy) = x(ix, iy, iz);
        ref2(iy, ix, iz) = x(ix, iy, iz);
        ref3(iy, iz, ix) = x(ix, iy, iz);
        ref4(iz, ix, iy) = x(ix, iy, iz);
        ref5(iz, iy, ix) = x(ix, iy, iz);
      }
    }
  }
  x.updateDevice();

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

  x0.updateSelf();
  x1.updateSelf();
  x2.updateSelf();
  x3.updateSelf();
  x4.updateSelf();
  x5.updateSelf();

  constexpr RealType eps = 1.e-5;
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

TEST( TRANSPOSE2D_batched, FP64 ) {
  using RealType = float;
  const std::size_t n = 16, m = 15, l = 4;

  // Original arrays
  View3D<RealType> x("x", n, m, l);
  View3D<RealType> x0("x0", n, m, l);
  View3D<RealType> x1("x1", n, l, m);
  View3D<RealType> x2("x2", m, n, l);
  View3D<RealType> x3("x3", m, l, n);
  View3D<RealType> x4("x4", l, n, m);
  View3D<RealType> x5("x5", l, m, n);

  View3D<RealType> ref0("ref0", n, m, l);
  View3D<RealType> ref1("ref1", n, l, m);
  View3D<RealType> ref2("ref2", m, n, l);
  View3D<RealType> ref3("ref3", m, l, n);
  View3D<RealType> ref4("ref4", l, n, m);
  View3D<RealType> ref5("ref5", l, m, n);

  // Set random numbers to 3D view
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        x(ix, iy, iz) = ix * 0.01 + iy * 0.203 + iz * 0.07;

        ref0(ix, iy, iz) = x(ix, iy, iz);
        ref1(ix, iz, iy) = x(ix, iy, iz);
        ref2(iy, ix, iz) = x(ix, iy, iz);
        ref3(iy, iz, ix) = x(ix, iy, iz);
        ref4(iz, ix, iy) = x(ix, iy, iz);
        ref5(iz, iy, ix) = x(ix, iy, iz);
      }
    }
  }
  x.updateDevice();

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

  x0.updateSelf();
  x1.updateSelf();
  x2.updateSelf();
  x3.updateSelf();
  x4.updateSelf();
  x5.updateSelf();

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

