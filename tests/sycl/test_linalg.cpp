#include <gtest/gtest.h>
#include <random>
#include <functional>
#include <linalg.hpp>
#include "Types.hpp"
#include "Test_Helper.hpp"

using test_types = ::testing::Types<float, double>;

// Basically the same fixtures, used for labeling tests
template <typename T>
struct MatrixVectorProduct : public ::testing::Test {
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

template <typename T>
struct MatrixMatrixProduct : public ::testing::Test {
protected:
  using float_type = T;
  std::unique_ptr<sycl::queue> queue_;

  virtual void SetUp() {
    auto selector = sycl::gpu_selector_v;
    try {
      queue_ = std::make_unique<sycl::queue>(selector, exception_handler);
      queue_->wait();
    } catch (std::exception const& e) {
      std::cout << "An exception is caught intializing a queue.\n";
      std::terminate();
    }
  }
};

template <typename T>
struct Eig : public ::testing::Test {
protected:
  using float_type = T;
  std::unique_ptr<sycl::queue> queue_;

  virtual void SetUp() {
    auto selector = sycl::gpu_selector_v;
    try {
      queue_ = std::make_unique<sycl::queue>(selector, exception_handler);
      queue_->wait();
    } catch (std::exception const& e) {
      std::cout << "An exception is caught intializing a queue.\n";
      std::terminate();
    }
  }
};

TYPED_TEST_SUITE(MatrixVectorProduct, test_types);
TYPED_TEST_SUITE(MatrixMatrixProduct, test_types);
TYPED_TEST_SUITE(Eig,                 test_types);

template <typename RealType>
void test_matrix_vector_product(sycl::queue& q, const std::string& transa) {
  const std::size_t n = 16, m = 8, l = 4;
  const RealType alpha = 1.5;

  std::size_t _n = transa == "N" ? n : m;
  std::size_t _m = transa == "N" ? m : n;

  View3D<RealType> A(q, "a", _n, _m, l);
  View2D<RealType> B(q, "b", m, l);
  View2D<RealType> C(q, "c", n, l);

  View2D<RealType> Ref(q, "ref", n, l);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<RealType>(-1, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set random numbers to 3D view
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<_m; iy++) {
      for(int ix=0; ix<_n; ix++) {
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
        if(transa == "N") {
          sum += alpha * A(ix, im, iz) * B(im, iz);
        } else {
          sum += alpha * A(im, ix, iz) * B(im, iz);
        }
      }
      Ref(ix, iz) = sum;
    }
  }

  // Matrix vector product
  auto _A = A.mdspan();
  auto _B = B.mdspan();
  auto _C = C.mdspan();

  Impl::matrix_vector_product(_A, _B, _C, transa, alpha);
  constexpr RealType eps = std::is_same_v<RealType, float> ? 1.e-3 :  1.e-12;

  for(int iz=0; iz<l; iz++) {
    for(int ix=0; ix<n; ix++) {
      EXPECT_NEAR( Ref(ix, iz), C(ix, iz), eps );
    }
  }
}

template <typename RealType>
void test_matrix_matrix_product(sycl::queue& q, 
                                const std::string& transa,
                                const std::string& transb) {
  const std::size_t n = 16, m = 8, k = 24, l = 4;
  const RealType alpha = 1.5, beta = 1.19;

  std::size_t An = transa == "N" ? n : m;
  std::size_t Bm = transb == "N" ? m : k;
  std::size_t Am = transa == "N" ? m : n;
  std::size_t Bk = transb == "N" ? k : m;

  View3D<RealType> A(q, "A", An, Am, l);
  View3D<RealType> B(q, "B", Bm, Bk, l);
  View3D<RealType> C(q, "C", n, k, l);
  View3D<RealType> Ref(q, "Ref", n, k, l);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<RealType>(-1, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set random numbers to 3D view
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<Am; iy++) {
      for(int ix=0; ix<An; ix++) {
        A(ix, iy, iz) = rand_gen();
      }
    }

    for(int iy=0; iy<Bk; iy++) {
      for(int ix=0; ix<Bm; ix++) {
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
          RealType A_tmp = transa == "N" ? A(ix, im, iz): A(im, ix, iz);
          RealType B_tmp = transb == "N" ? B(im, iy, iz): B(iy, im, iz);

          sum += alpha * A_tmp * B_tmp;
        }
        Ref(ix, iy, iz) = sum + beta * C(ix, iy, iz);
      }
    }
  }

  auto _A = A.mdspan();
  auto _B = B.mdspan();
  auto _C = C.mdspan();

  // Matrix matrix product using Eigen
  Impl::matrix_matrix_product(_A, _B, _C, transa, transb, alpha, beta);
  constexpr RealType eps = 1.0e-3;

  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<k; iy++) {
      for(int ix=0; ix<n; ix++) {
        EXPECT_NEAR( Ref(ix, iy, iz), C(ix, iy, iz), eps );
      }
    }
  }
}

TYPED_TEST(MatrixVectorProduct, N) {
  using float_type = typename TestFixture::float_type;
  test_matrix_vector_product<float_type>(*this->queue_, "N");
}

TYPED_TEST(MatrixVectorProduct, T) {
  using float_type = typename TestFixture::float_type;
  test_matrix_vector_product<float_type>(*this->queue_, "T");
}

TYPED_TEST(MatrixMatrixProduct, N_N) {
  using float_type = typename TestFixture::float_type;
  test_matrix_matrix_product<float_type>(*this->queue_, "N", "N");
}

TYPED_TEST(MatrixMatrixProduct, N_T) {
  using float_type = typename TestFixture::float_type;
  test_matrix_matrix_product<float_type>(*this->queue_, "N", "T");
}

TYPED_TEST(MatrixMatrixProduct, T_N) {
  using float_type = typename TestFixture::float_type;
  test_matrix_matrix_product<float_type>(*this->queue_, "T", "N");
}

TYPED_TEST(MatrixMatrixProduct, T_T) {
  using float_type = typename TestFixture::float_type;
  test_matrix_matrix_product<float_type>(*this->queue_, "T", "T");
}

template <typename RealType>
void test_eigen_inplace(sycl::queue& q) {
  const std::size_t m = 4, l = 128;
  View3D<RealType> a(q, "a", m, m, l);
  View2D<RealType> v(q, "v", m, l);
  View3D<RealType> ref_w(q, "ref_w", m, m, l);
  View2D<RealType> ref_v(q, "ref_v", m, l);

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
  auto _a = a.mdspan();
  auto _v = v.mdspan();
  Impl::eig(_a, _v);

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

TYPED_TEST(Eig, Inplace) {
  using float_type = typename TestFixture::float_type;
  test_eigen_inplace<float_type>(*this->queue_);
}