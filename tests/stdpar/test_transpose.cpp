#include <gtest/gtest.h>
#include "Types.hpp"
#include <stdpar/Transpose.hpp>

TEST( TRANSPOSE2D, FP32 ) {
  using RealType = float;
  const std::size_t rows = 16;
  const std::size_t rows2 = 15;
  const std::size_t cols = rows * 2;
  const std::size_t cols2 = rows2 * 2;

  // Original arrays
  std::vector<RealType> _a_even(rows * cols);
  std::vector<RealType> _a_odd(rows2 * cols2);

  // Transposed arrays
  std::vector<RealType> _b_even(rows * cols);
  std::vector<RealType> _b_odd(rows2 * cols2);

  std::vector<RealType> _ref_even(rows * cols);
  std::vector<RealType> _ref_odd(rows2 * cols2);

  Mdspan2D<RealType> A_even(_a_even.data(), rows, cols);
  Mdspan2D<RealType> A_odd(_a_odd.data(), rows2, cols2);
  Mdspan2D<RealType> B_even(_b_even.data(), cols, rows);
  Mdspan2D<RealType> B_odd(_b_odd.data(), cols2, rows2);

  Mdspan2D<RealType> Ref_even(_ref_even.data(), cols, rows);
  Mdspan2D<RealType> Ref_odd(_ref_odd.data(), cols2, rows2);

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

  constexpr RealType eps = 1.e-5;
  Impl::transpose(A_even, B_even);
  for(int i=0; i<cols; i++) {
    for(int j=0; j<rows; j++) {
      EXPECT_NEAR( B_even(i, j), Ref_even(i, j), eps );
    }
  }

  Impl::transpose(A_odd, B_odd);
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

  // Original arrays
  std::vector<RealType> _a_even(rows * cols);
  std::vector<RealType> _a_odd(rows2 * cols2);

  // Transposed arrays
  std::vector<RealType> _b_even(rows * cols);
  std::vector<RealType> _b_odd(rows2 * cols2);

  std::vector<RealType> _ref_even(rows * cols);
  std::vector<RealType> _ref_odd(rows2 * cols2);

  Mdspan2D<RealType> A_even(_a_even.data(), rows, cols);
  Mdspan2D<RealType> A_odd(_a_odd.data(), rows2, cols2);
  Mdspan2D<RealType> B_even(_b_even.data(), cols, rows);
  Mdspan2D<RealType> B_odd(_b_odd.data(), cols2, rows2);

  Mdspan2D<RealType> Ref_even(_ref_even.data(), cols, rows);
  Mdspan2D<RealType> Ref_odd(_ref_odd.data(), cols2, rows2);

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

  constexpr RealType eps = 1.e-12;
  Impl::transpose(A_even, B_even);
  for(int i=0; i<cols; i++) {
    for(int j=0; j<rows; j++) {
      EXPECT_NEAR( B_even(i, j), Ref_even(i, j), eps );
    }
  }

  Impl::transpose(A_odd, B_odd);
  for(int i=0; i<cols2; i++) {
    for(int j=0; j<rows2; j++) {
      EXPECT_NEAR( B_odd(i, j), Ref_odd(i, j), eps );
    }
  }
}

TEST( TRANSPOSE2D_batched, fp32 ) {
  using RealType = float;
  const std::size_t n = 16, m = 15, l = 4;

  // Original arrays
  std::vector<RealType> _x(n*m*l);

  std::vector<RealType> _x0(n*m*l);
  std::vector<RealType> _x1(n*m*l);
  std::vector<RealType> _x2(n*m*l);
  std::vector<RealType> _x3(n*m*l);
  std::vector<RealType> _x4(n*m*l);
  std::vector<RealType> _x5(n*m*l);

  std::vector<RealType> _ref0(n*m*l);
  std::vector<RealType> _ref1(n*m*l);
  std::vector<RealType> _ref2(n*m*l);
  std::vector<RealType> _ref3(n*m*l);
  std::vector<RealType> _ref4(n*m*l);
  std::vector<RealType> _ref5(n*m*l);

  Mdspan3D<RealType> x(_x.data(), n, m, l);
  Mdspan3D<RealType> x0(_x0.data(), n, m, l);
  Mdspan3D<RealType> x1(_x1.data(), n, l, m);
  Mdspan3D<RealType> x2(_x2.data(), m, n, l);
  Mdspan3D<RealType> x3(_x3.data(), m, l, n);
  Mdspan3D<RealType> x4(_x4.data(), l, n, m);
  Mdspan3D<RealType> x5(_x5.data(), l, m, n);

  Mdspan3D<RealType> ref0(_ref0.data(), n, m, l);
  Mdspan3D<RealType> ref1(_ref1.data(), n, l, m);
  Mdspan3D<RealType> ref2(_ref2.data(), m, n, l);
  Mdspan3D<RealType> ref3(_ref3.data(), m, l, n);
  Mdspan3D<RealType> ref4(_ref4.data(), l, n, m);
  Mdspan3D<RealType> ref5(_ref5.data(), l, m, n);

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

  Impl::transpose(x, x0, {0, 1, 2});
  Impl::transpose(x, x1, {0, 2, 1});
  Impl::transpose(x, x2, {1, 0, 2});
  Impl::transpose(x, x3, {1, 2, 0});
  Impl::transpose(x, x4, {2, 0, 1});
  Impl::transpose(x, x5, {2, 1, 0});

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

TEST( TRANSPOSE2D_batched, fp64 ) {
  using RealType = double;
  const std::size_t n = 16, m = 15, l = 4;

  // Original arrays
  std::vector<RealType> _x(n*m*l);

  std::vector<RealType> _x0(n*m*l);
  std::vector<RealType> _x1(n*m*l);
  std::vector<RealType> _x2(n*m*l);
  std::vector<RealType> _x3(n*m*l);
  std::vector<RealType> _x4(n*m*l);
  std::vector<RealType> _x5(n*m*l);

  std::vector<RealType> _ref0(n*m*l);
  std::vector<RealType> _ref1(n*m*l);
  std::vector<RealType> _ref2(n*m*l);
  std::vector<RealType> _ref3(n*m*l);
  std::vector<RealType> _ref4(n*m*l);
  std::vector<RealType> _ref5(n*m*l);

  Mdspan3D<RealType> x(_x.data(), n, m, l);
  Mdspan3D<RealType> x0(_x0.data(), n, m, l);
  Mdspan3D<RealType> x1(_x1.data(), n, l, m);
  Mdspan3D<RealType> x2(_x2.data(), m, n, l);
  Mdspan3D<RealType> x3(_x3.data(), m, l, n);
  Mdspan3D<RealType> x4(_x4.data(), l, n, m);
  Mdspan3D<RealType> x5(_x5.data(), l, m, n);

  Mdspan3D<RealType> ref0(_ref0.data(), n, m, l);
  Mdspan3D<RealType> ref1(_ref1.data(), n, l, m);
  Mdspan3D<RealType> ref2(_ref2.data(), m, n, l);
  Mdspan3D<RealType> ref3(_ref3.data(), m, l, n);
  Mdspan3D<RealType> ref4(_ref4.data(), l, n, m);
  Mdspan3D<RealType> ref5(_ref5.data(), l, m, n);

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

  Impl::transpose(x, x0, {0, 1, 2});
  Impl::transpose(x, x1, {0, 2, 1});
  Impl::transpose(x, x2, {1, 0, 2});
  Impl::transpose(x, x3, {1, 2, 0});
  Impl::transpose(x, x4, {2, 0, 1});
  Impl::transpose(x, x5, {2, 1, 0});

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
