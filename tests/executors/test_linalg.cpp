#include <random>
#include <functional>
#include <gtest/gtest.h>
#include "Types.hpp"
#include <linalg.hpp>

TEST( MatrixVectorProduct, N_batched_fp32 ) {
  using RealType = float;
  const std::size_t n = 16, m = 8, l = 4;
  const RealType alpha = 1.5;
  View3D<RealType> A("A", n, m, l);
  View2D<RealType> B("B", m, l);
  View2D<RealType> C("C", n, l);
  View2D<RealType> Ref("Ref", n, l);

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
  A.updateDevice();
  B.updateDevice();
  C.updateDevice();

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
  auto _A = A.mdspan();
  auto _B = B.mdspan();
  auto _C = C.mdspan();

  // Matrix vector product using Eigen
  Impl::matrix_vector_product(_A, _B, _C, "N", alpha);
  constexpr RealType eps = 1.e-3;

  C.updateSelf();

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
  View3D<RealType> A("A", n, m, l);
  View2D<RealType> B("B", m, l);
  View2D<RealType> C("C", n, l);
  View2D<RealType> Ref("Ref", n, l);

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
  A.updateDevice();
  B.updateDevice();
  C.updateDevice();

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

  auto _A = A.mdspan();
  auto _B = B.mdspan();
  auto _C = C.mdspan();

  // Matrix vector product using Eigen
  Impl::matrix_vector_product(_A, _B, _C, "N", alpha);
  constexpr RealType eps = 1.e-12;

  C.updateSelf();

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

  View3D<RealType> A("A", m, n, l);
  View2D<RealType> B("B", m, l);
  View2D<RealType> C("C", n, l);
  View2D<RealType> Ref("Ref", n, l);

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
  A.updateDevice();
  B.updateDevice();
  C.updateDevice();

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

  auto _A = A.mdspan();
  auto _B = B.mdspan();
  auto _C = C.mdspan();

  // Matrix vector product using Eigen
  Impl::matrix_vector_product(_A, _B, _C, "T", alpha);
  constexpr RealType eps = 1.e-3;

  C.updateSelf();

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
  View3D<RealType> A("A", m, n, l);
  View2D<RealType> B("B", m, l);
  View2D<RealType> C("C", n, l);
  View2D<RealType> Ref("Ref", n, l);

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
  A.updateDevice();
  B.updateDevice();
  C.updateDevice();

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

  auto _A = A.mdspan();
  auto _B = B.mdspan();
  auto _C = C.mdspan();

  // Matrix vector product using Eigen
  Impl::matrix_vector_product(_A, _B, _C, "T", alpha);
  constexpr RealType eps = 1.e-12;

  C.updateSelf();

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
  View3D<RealType> A("A", n, m, l);
  View3D<RealType> B("B", m, k, l);
  View3D<RealType> C("C", n, k, l);
  View3D<RealType> Ref("Ref", n, k, l);

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
  A.updateDevice();
  B.updateDevice();
  C.updateDevice();

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

  auto _A = A.mdspan();
  auto _B = B.mdspan();
  auto _C = C.mdspan();

  // Matrix matrix product using Eigen
  Impl::matrix_matrix_product(_A, _B, _C, "N", "N", alpha, beta);
  constexpr RealType eps = 1.0e-3;

  C.updateSelf();

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
  View3D<RealType> A("A", n, m, l);
  View3D<RealType> B("B", m, k, l);
  View3D<RealType> C("C", n, k, l);
  View3D<RealType> Ref("Ref", n, k, l);

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
  A.updateDevice();
  B.updateDevice();
  C.updateDevice();

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

  auto _A = A.mdspan();
  auto _B = B.mdspan();
  auto _C = C.mdspan();

  // Matrix matrix product using Eigen
  Impl::matrix_matrix_product(_A, _B, _C, "N", "N", alpha, beta);
  constexpr RealType eps = 1.0e-12;

  C.updateSelf();

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
  View3D<RealType> A("A", m, n, l);
  View3D<RealType> B("B", k, m, l);
  View3D<RealType> C("C", n, k, l);
  View3D<RealType> Ref("Ref", n, k, l);

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
  A.updateDevice();
  B.updateDevice();
  C.updateDevice();

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

  auto _A = A.mdspan();
  auto _B = B.mdspan();
  auto _C = C.mdspan();

  // Matrix matrix product using Eigen
  Impl::matrix_matrix_product(_A, _B, _C, "T", "T", alpha, beta);
  constexpr RealType eps = 1.e-3;

  C.updateSelf();

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
  View3D<RealType> A("A", m, n, l);
  View3D<RealType> B("B", k, m, l);
  View3D<RealType> C("C", n, k, l);
  View3D<RealType> Ref("Ref", n, k, l);

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
  A.updateDevice();
  B.updateDevice();
  C.updateDevice();

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

  auto _A = A.mdspan();
  auto _B = B.mdspan();
  auto _C = C.mdspan();

  // Matrix matrix product using Eigen
  Impl::matrix_matrix_product(_A, _B, _C, "T", "T", alpha, beta);
  constexpr RealType eps = 1.e-12;

  C.updateSelf();

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
  View3D<RealType> a("a", m, m, l);
  View2D<RealType> v("v", m, l);
  View3D<RealType> ref_w("ref_w", m, m, l);
  View2D<RealType> ref_v("ref_v", m, l);

  // Set Diagonal matrix (np.diag(1,2,3))
  for(int iz=0; iz<l; iz++) {
    for(int id=0; id<m; id++) {
      a(id, id, iz) = id+1;
    }
  }
  a.updateDevice();

  // The eigen values and vectors should be
  // v == [1, 2, 3], w == np.eye(3)
  for(int iz=0; iz<l; iz++) {
    for(int id=0; id<m; id++) {
      ref_w(id, id, iz) = 1;
      ref_v(id, iz) = id+1;
    }
  }

  auto _a = a.mdspan();
  auto _v = v.mdspan();

  // Eigen value decomposition
  Impl::eig(_a, _v);
  a.updateSelf();
  v.updateSelf();

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
