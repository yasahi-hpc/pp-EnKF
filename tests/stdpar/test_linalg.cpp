#include <random>
#include <functional>
#include <gtest/gtest.h>
#include "Types.hpp"
#include <linalg.hpp>

TEST( MatrixVectorProduct, N_batched_fp32 ) {
  using RealType = float;
  const std::size_t n = 16, m = 8, l = 4;
  const RealType alpha = 1.5;
  std::vector<RealType> _a(n*m*l);
  std::vector<RealType> _b(m*l);
  std::vector<RealType> _c(n*l);
  std::vector<RealType> _ref(n*l);

  Mdspan3D<RealType> A(_a.data(), n, m, l);
  Mdspan2D<RealType> B(_b.data(), m, l);
  Mdspan2D<RealType> C(_c.data(), n, l);

  Mdspan2D<RealType> Ref(_ref.data(), n, l);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<RealType>(-1, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set random numbers to 3D view
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        A(ix, iy, iz) = rand_gen();
      }
    }

    for(int ix=0; ix<m; ix++) {
      B(ix, iz) = rand_gen();
    }

    for(int ix=0; ix<n; ix++) {
      C(ix, iz) = rand_gen();
    }
  }

  // C = Op(A) * B
  for(int iz=0; iz<l; iz++) {
    for(int ix=0; ix<n; ix++) {
      RealType sum = 0;
      for(int im=0; im<m; im++) {
        sum += alpha * A(ix, im, iz) * B(im, iz);
      }
      Ref(ix, iz) = sum;
    }
  }

  // Matrix vector product using Eigen
  Impl::matrix_vector_product(A, B, C, "N", alpha);
  constexpr RealType eps = 1.e-3;

  for(int iz=0; iz<l; iz++) {
    for(int ix=0; ix<n; ix++) {
      EXPECT_NEAR( Ref(ix, iz), C(ix, iz), eps );
    }
  }
}

TEST( MatrixVectorProduct, N_batched_fp64 ) {
  using RealType = double;
  const std::size_t n = 16, m = 8, l = 4;
  const RealType alpha = 1.5;
  std::vector<RealType> _a(n*m*l);
  std::vector<RealType> _b(m*l);
  std::vector<RealType> _c(n*l);
  std::vector<RealType> _ref(n*l);

  Mdspan3D<RealType> A(_a.data(), n, m, l);
  Mdspan2D<RealType> B(_b.data(), m, l);
  Mdspan2D<RealType> C(_c.data(), n, l);

  Mdspan2D<RealType> Ref(_ref.data(), n, l);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<RealType>(-1, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set random numbers to 3D view
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        A(ix, iy, iz) = rand_gen();
      }
    }

    for(int ix=0; ix<m; ix++) {
      B(ix, iz) = rand_gen();
    }

    for(int ix=0; ix<n; ix++) {
      C(ix, iz) = rand_gen();
    }
  }

  // C = Op(A) * B
  for(int iz=0; iz<l; iz++) {
    for(int ix=0; ix<n; ix++) {
      RealType sum = 0;
      for(int im=0; im<m; im++) {
        sum += alpha * A(ix, im, iz) * B(im, iz);
      }
      Ref(ix, iz) = sum;
    }
  }

  // Matrix vector product using Eigen
  Impl::matrix_vector_product(A, B, C, "N", alpha);
  constexpr RealType eps = 1.e-12;

  for(int iz=0; iz<l; iz++) {
    for(int ix=0; ix<n; ix++) {
      EXPECT_NEAR( Ref(ix, iz), C(ix, iz), eps );
    }
  }
}

TEST( MatrixVectorProduct, T_batched_fp32 ) {
  using RealType = float;
  const std::size_t n = 16, m = 8, l = 4;
  const RealType alpha = 1.5;
  std::vector<RealType> _a(m*n*l);
  std::vector<RealType> _b(m*l);
  std::vector<RealType> _c(n*l);
  std::vector<RealType> _ref(n*l);

  Mdspan3D<RealType> A(_a.data(), m, n, l);
  Mdspan2D<RealType> B(_b.data(), m, l);
  Mdspan2D<RealType> C(_c.data(), n, l);

  Mdspan2D<RealType> Ref(_ref.data(), n, l);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<RealType>(-1, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set random numbers to 3D view
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        A(iy, ix, iz) = rand_gen();
      }
    }

    for(int ix=0; ix<m; ix++) {
      B(ix, iz) = rand_gen();
    }

    for(int ix=0; ix<n; ix++) {
      C(ix, iz) = rand_gen();
    }
  }

  // C = Op(A) * B
  for(int iz=0; iz<l; iz++) {
    for(int ix=0; ix<n; ix++) {
      RealType sum = 0;
      for(int im=0; im<m; im++) {
        sum += alpha * A(im, ix, iz) * B(im, iz);
      }
      Ref(ix, iz) = sum;
    }
  }

  // Matrix vector product using Eigen
  Impl::matrix_vector_product(A, B, C, "T", alpha);
  constexpr RealType eps = 1.e-3;

  for(int iz=0; iz<l; iz++) {
    for(int ix=0; ix<n; ix++) {
      EXPECT_NEAR( Ref(ix, iz), C(ix, iz), eps );
    }
  }
}

TEST( MatrixVectorProduct, T_batched_fp64 ) {
  using RealType = double;
  const std::size_t n = 16, m = 8, l = 4;
  const RealType alpha = 1.5;
  std::vector<RealType> _a(m*n*l);
  std::vector<RealType> _b(m*l);
  std::vector<RealType> _c(n*l);
  std::vector<RealType> _ref(n*l);

  Mdspan3D<RealType> A(_a.data(), m, n, l);
  Mdspan2D<RealType> B(_b.data(), m, l);
  Mdspan2D<RealType> C(_c.data(), n, l);

  Mdspan2D<RealType> Ref(_ref.data(), n, l);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<RealType>(-1, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set random numbers to 3D view
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        A(iy, ix, iz) = rand_gen();
      }
    }

    for(int ix=0; ix<m; ix++) {
      B(ix, iz) = rand_gen();
    }

    for(int ix=0; ix<n; ix++) {
      C(ix, iz) = rand_gen();
    }
  }

  // C = Op(A) * B
  for(int iz=0; iz<l; iz++) {
    for(int ix=0; ix<n; ix++) {
      RealType sum = 0;
      for(int im=0; im<m; im++) {
        sum += alpha * A(im, ix, iz) * B(im, iz);
      }
      Ref(ix, iz) = sum;
    }
  }

  // Matrix vector product using Eigen
  Impl::matrix_vector_product(A, B, C, "T", alpha);
  constexpr RealType eps = 1.e-12;

  for(int iz=0; iz<l; iz++) {
    for(int ix=0; ix<n; ix++) {
      EXPECT_NEAR( Ref(ix, iz), C(ix, iz), eps );
    }
  }
}

TEST( MatrixMatrixProduct, N_N_batched_fp32 ) {
  using RealType = float;
  const std::size_t n = 16, m = 8, k = 24, l = 4;
  const RealType alpha = 1.5, beta = 1.19;
  std::vector<RealType> _a(n*m*l);
  std::vector<RealType> _b(m*k*l);
  std::vector<RealType> _c(n*k*l);
  std::vector<RealType> _ref(n*k*l);

  Mdspan3D<RealType> A(_a.data(), n, m, l);
  Mdspan3D<RealType> B(_b.data(), m, k, l);
  Mdspan3D<RealType> C(_c.data(), n, k, l);

  Mdspan3D<RealType> Ref(_ref.data(), n, k, l);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<RealType>(-1, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set random numbers to 3D view
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        A(ix, iy, iz) = rand_gen();
      }
    }

    for(int iy=0; iy<k; iy++) {
      for(int ix=0; ix<m; ix++) {
        B(ix, iy, iz) = rand_gen();
      }
    }

    for(int iy=0; iy<k; iy++) {
      for(int ix=0; ix<n; ix++) {
        C(ix, iy, iz) = rand_gen();
      }
    }
  }

  // C = alpha * Op(A) * Op(B) + beta * C
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<k; iy++) {
      for(int ix=0; ix<n; ix++) {
        RealType sum = 0;
        for(int im=0; im<m; im++) {
          sum += alpha * A(ix, im, iz) * B(im, iy, iz);
        }
        Ref(ix, iy, iz) = sum + beta * C(ix, iy, iz);
      }
    }
  }

  // Matrix matrix product using Eigen
  Impl::matrix_matrix_product(A, B, C, "N", "N", alpha, beta);
  constexpr RealType eps = 1.0e-3;

  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<k; iy++) {
      for(int ix=0; ix<n; ix++) {
        EXPECT_NEAR( Ref(ix, iy, iz), C(ix, iy, iz), eps );
      }
    }
  }
}

TEST( MatrixMatrixProduct, N_N_batched_fp64 ) {
  using RealType = double;
  const std::size_t n = 16, m = 8, k = 24, l = 4;
  const RealType alpha = 1.5, beta = 1.19;
  std::vector<RealType> _a(n*m*l);
  std::vector<RealType> _b(m*k*l);
  std::vector<RealType> _c(n*k*l);
  std::vector<RealType> _ref(n*k*l);

  Mdspan3D<RealType> A(_a.data(), n, m, l);
  Mdspan3D<RealType> B(_b.data(), m, k, l);
  Mdspan3D<RealType> C(_c.data(), n, k, l);

  Mdspan3D<RealType> Ref(_ref.data(), n, k, l);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<RealType>(-1, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set random numbers to 3D view
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        A(ix, iy, iz) = rand_gen();
      }
    }

    for(int iy=0; iy<k; iy++) {
      for(int ix=0; ix<m; ix++) {
        B(ix, iy, iz) = rand_gen();
      }
    }

    for(int iy=0; iy<k; iy++) {
      for(int ix=0; ix<n; ix++) {
        C(ix, iy, iz) = rand_gen();
      }
    }
  }

  // C = alpha * Op(A) * Op(B) + beta * C
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<k; iy++) {
      for(int ix=0; ix<n; ix++) {
        RealType sum = 0;
        for(int im=0; im<m; im++) {
          sum += alpha * A(ix, im, iz) * B(im, iy, iz);
        }
        Ref(ix, iy, iz) = sum + beta * C(ix, iy, iz);
      }
    }
  }

  // Matrix matrix product using Eigen
  Impl::matrix_matrix_product(A, B, C, "N", "N", alpha, beta);
  constexpr RealType eps = 1.0e-12;

  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<k; iy++) {
      for(int ix=0; ix<n; ix++) {
        EXPECT_NEAR( Ref(ix, iy, iz), C(ix, iy, iz), eps );
      }
    }
  }
}

TEST( MatrixMatrixProduct, T_T_batched_fp32 ) {
  using RealType = float;
  const std::size_t n = 16, m = 8, k = 24, l = 4;
  const RealType alpha = 1.5, beta = 1.19;
  std::vector<RealType> _a(n*m*l);
  std::vector<RealType> _b(m*k*l);
  std::vector<RealType> _c(n*k*l);
  std::vector<RealType> _ref(n*k*l);

  Mdspan3D<RealType> A(_a.data(), m, n, l);
  Mdspan3D<RealType> B(_b.data(), k, m, l);
  Mdspan3D<RealType> C(_c.data(), n, k, l);

  Mdspan3D<RealType> Ref(_ref.data(), n, k, l);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<RealType>(-1, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set random numbers to 3D view
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        A(iy, ix, iz) = rand_gen();
      }
    }

    for(int iy=0; iy<k; iy++) {
      for(int ix=0; ix<m; ix++) {
        B(iy, ix, iz) = rand_gen();
      }
    }

    for(int iy=0; iy<k; iy++) {
      for(int ix=0; ix<n; ix++) {
        C(ix, iy, iz) = rand_gen();
      }
    }
  }

  // C = alpha * Op(A) * Op(B) + beta * C
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<k; iy++) {
      for(int ix=0; ix<n; ix++) {
        RealType sum = 0;
        for(int im=0; im<m; im++) {
          sum += alpha * A(im, ix, iz) * B(iy, im, iz);
        }
        Ref(ix, iy, iz) = sum + beta * C(ix, iy, iz);
      }
    }
  }

  // Matrix matrix product using Eigen
  Impl::matrix_matrix_product(A, B, C, "T", "T", alpha, beta);
  constexpr RealType eps = 1.e-3;

  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<k; iy++) {
      for(int ix=0; ix<n; ix++) {
        EXPECT_NEAR( Ref(ix, iy, iz), C(ix, iy, iz), eps );
      }
    }
  }
}

TEST( MatrixMatrixProduct, T_T_batched_fp64 ) {
  using RealType = double;
  const std::size_t n = 16, m = 8, k = 24, l = 4;
  const RealType alpha = 1.5, beta = 1.19;
  std::vector<RealType> _a(n*m*l);
  std::vector<RealType> _b(m*k*l);
  std::vector<RealType> _c(n*k*l);
  std::vector<RealType> _ref(n*k*l);

  Mdspan3D<RealType> A(_a.data(), m, n, l);
  Mdspan3D<RealType> B(_b.data(), k, m, l);
  Mdspan3D<RealType> C(_c.data(), n, k, l);

  Mdspan3D<RealType> Ref(_ref.data(), n, k, l);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<RealType>(-1, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set random numbers to 3D view
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<n; ix++) {
        A(iy, ix, iz) = rand_gen();
      }
    }

    for(int iy=0; iy<k; iy++) {
      for(int ix=0; ix<m; ix++) {
        B(iy, ix, iz) = rand_gen();
      }
    }

    for(int iy=0; iy<k; iy++) {
      for(int ix=0; ix<n; ix++) {
        C(ix, iy, iz) = rand_gen();
      }
    }
  }

  // C = alpha * Op(A) * Op(B) + beta * C
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<k; iy++) {
      for(int ix=0; ix<n; ix++) {
        RealType sum = 0;
        for(int im=0; im<m; im++) {
          sum += alpha * A(im, ix, iz) * B(iy, im, iz);
        }
        Ref(ix, iy, iz) = sum + beta * C(ix, iy, iz);
      }
    }
  }

  // Matrix matrix product using Eigen
  Impl::matrix_matrix_product(A, B, C, "T", "T", alpha, beta);
  constexpr RealType eps = 1.e-12;

  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<k; iy++) {
      for(int ix=0; ix<n; ix++) {
        EXPECT_NEAR( Ref(ix, iy, iz), C(ix, iy, iz), eps );
      }
    }
  }
}

TEST( EIGEN, BATCHED_INPLACE ) {
  using RealType = double;
  const std::size_t m = 4, l = 128;
  std::vector<RealType> _a(m*m*l);
  std::vector<RealType> _v(m*l);
  std::vector<RealType> _ref_v(m*l);
  std::vector<RealType> _ref_w(m*m*l);

  Mdspan3D<RealType> a(_a.data(), m, m, l);
  Mdspan2D<RealType> v(_v.data(), m, l);
  Mdspan2D<RealType> ref_v(_ref_v.data(), m, l);
  Mdspan3D<RealType> ref_w(_ref_w.data(), m, m, l);

  // Set Diagonal matrix (np.diag(1,2,3))
  for(int iz=0; iz<l; iz++) {
    for(int id=0; id<m; id++) {
      a(id, id, iz) = id+1;
    }
  }

  // The eigen values and vectors should be
  // v == [1, 2, 3], w == np.eye(3)
  for(int iz=0; iz<l; iz++) {
    for(int id=0; id<m; id++) {
      ref_w(id, id, iz) = 1;
      ref_v(id, iz) = id+1;
    }
  }

  // Eigen value decomposition
  Impl::eig(a, v);

  constexpr RealType eps = 1.e-13;
  for(int iz=0; iz<l; iz++) {
    for(int ix=0; ix<m; ix++) {
      EXPECT_NEAR( v(ix, iz), ref_v(ix, iz), eps );
    }
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<m; ix++) {
        EXPECT_NEAR( a(ix, iy, iz), ref_w(ix, iy, iz), eps );
      }
    }
  }
}
