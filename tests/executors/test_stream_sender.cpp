#include <random>
#include <functional>
#include <gtest/gtest.h>
#include <executors/numpy_like.hpp>
#include <linalg.hpp>
#include "nvexec/stream_context.cuh"
#include <stdexec/execution.hpp>
#include <exec/on.hpp>
#include "Types.hpp"

TEST( AxpyAndThen, Blas ) {
  using RealType = double;
  const std::size_t n = 16, m = 8, k = 24, l = 4;
  const RealType alpha = 1.5, beta = 1.19;
  View3D<RealType> A0("A0", m, n, l), A1("A1", m, n, l), A2("A2", m, n, l), A2_ref("A2_ref", m, n, l);
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
        A0(iy, ix, iz) = rand_gen();
        A1(iy, ix, iz) = rand_gen();
        A2_ref(iy, ix, iz) = A0(iy, ix, iz) + beta * A1(iy, ix, iz);
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
  A0.updateDevice();
  A1.updateDevice();
  B.updateDevice();
  C.updateDevice();

  // C = alpha * Op(D) * Op(B) + beta * C
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<k; iy++) {
      for(int ix=0; ix<n; ix++) {
        RealType sum = 0;
        for(int im=0; im<m; im++) {
          sum += alpha * A2_ref(im, ix, iz) * B(iy, im, iz);
        }
        Ref(ix, iy, iz) = sum + beta * C(ix, iy, iz);
      }
    }
  }

  auto _A0 = A0.mdspan();
  auto _A1 = A1.mdspan();
  auto _A2 = A2.mdspan();
  auto _B = B.mdspan();
  auto _C = C.mdspan();

  // Sender to perform Axpy
  nvexec::stream_context stream_ctx{};
  auto scheduler = stream_ctx.get_scheduler();

  // Get stream from the scheduler
  auto stream_getter = stdexec::schedule(scheduler)
                     | stdexec::let_value([] {
                         return nvexec::get_stream();
                       })
                     | stdexec::then([](cudaStream_t stream) {
                         return stream;
                       });
  auto [stream] = stdexec::sync_wait(std::move(stream_getter)).value();

  auto size = A0.size();
  auto axpy_sender = stdexec::just()
                   | exec::on(scheduler, stdexec::bulk(size,
                     [=] MDSPAN_FORCE_INLINE_FUNCTION (const int idx) {
                       const int iy = idx % m;
                       const int ixz = idx / m;
                       const int ix  = ixz % n;
                       const int iz  = ixz / n;
                       _A2(iy, ix, iz) = _A0(iy, ix, iz) + beta * _A1(iy, ix, iz);
                     })
                   );

  // Matrix matrix product using cublas
  Impl::blasHandle_t blas_handle;
  blas_handle.create();
  blas_handle.set_stream(stream);
  auto blas_sender = axpy_sender
                   | stdexec::then([&]{
                                        Impl::matrix_matrix_product(blas_handle, _A2, _B, _C, "T", "T", alpha, beta);
                                      });

  stdexec::sync_wait( std::move(blas_sender) );
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

TEST( AxpyAndThen, Solver ) {
  using RealType = double;
  const RealType beta = 1.19;

  const std::size_t m = 4, l = 128;
  View3D<RealType> a0("a0", m, m, l), a1("a1", m, m, l), a2("a2", m, m, l), a2_ref("a2-ref", m, m, l);
  View2D<RealType> v("v", m, l);
  View3D<RealType> ref_w("ref_w", m, m, l);
  View2D<RealType> ref_v("ref_v", m, l);

  // Set Diagonal matrix (np.diag(1,2,3) + beta * np.diag(0, 0.37, 0.74))
  for(int iz=0; iz<l; iz++) {
    for(int id=0; id<m; id++) {
      a0(id, id, iz) = id+1;
      a1(id, id, iz) = id*0.37;
      a2_ref(id, id, iz) = a0(id, id, iz) + beta * a1(id, id, iz);
    }
  }
  a0.updateDevice();
  a1.updateDevice();

  // The eigen values and vectors should be
  // v == [1, 2, 3], w == np.eye(3)
  for(int iz=0; iz<l; iz++) {
    for(int id=0; id<m; id++) {
      ref_w(id, id, iz) = 1;
      ref_v(id, iz) = id+1 + beta*(id*0.37);
    }
  }

  auto _a0 = a0.mdspan();
  auto _a1 = a1.mdspan();
  auto _a2 = a2.mdspan();
  auto _v  = v.mdspan();

  // Sender to perform Axpy
  nvexec::stream_context stream_ctx{};
  auto scheduler = stream_ctx.get_scheduler();

  // Get stream from the scheduler
  auto stream_getter = stdexec::schedule(scheduler)
                     | stdexec::let_value([] {
                         return nvexec::get_stream();
                       })
                     | stdexec::then([](cudaStream_t stream) {
                         return stream;
                       });
  auto [stream] = stdexec::sync_wait(std::move(stream_getter)).value();

  auto size = a0.size();
  auto axpy_sender = stdexec::just() |
                   exec::on(scheduler, stdexec::bulk(size,
                     [=] MDSPAN_FORCE_INLINE_FUNCTION (const int idx) {
                       const int ix = idx % m;
                       const int iyz = idx / m;
                       const int iy  = iyz % m;
                       const int iz  = iyz / m;
                       _a2(iy, ix, iz) = _a0(iy, ix, iz) + beta * _a1(iy, ix, iz);
                     })
                   );

  // Eigen value decomposition
  Impl::syevjHandle_t<double> syevj_handle;
  syevj_handle.create(_a2, _v);
  syevj_handle.set_stream(stream);
  auto evd_sender = axpy_sender
                  | stdexec::then([&]{
                                       Impl::eig(syevj_handle, _a2, _v);
                                     });

  stdexec::sync_wait( std::move(evd_sender) );
  a2.updateSelf();
  v.updateSelf();

  constexpr RealType eps = 1.e-13;
  for(int iz=0; iz<l; iz++) {
    for(int ix=0; ix<m; ix++) {
      EXPECT_NEAR( v(ix, iz), ref_v(ix, iz), eps );
    }
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<m; ix++) {
        EXPECT_NEAR( a2(ix, iy, iz), ref_w(ix, iy, iz), eps );
      }
    }
  }
}

TEST( ThenAndAxpy, Blas ) {
  using RealType = double;
  const std::size_t n = 16, m = 8, k = 24, l = 4;
  const RealType alpha = 1.5, beta = 1.19;
  View3D<RealType> A("A", m, n, l);
  View3D<RealType> B("B", k, m, l);
  View3D<RealType> C0("C0", n, k, l), C1("C1", n, k, l), C2("C2", n, k, l);
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
        C0(ix, iy, iz) = rand_gen();
        C1(ix, iy, iz) = rand_gen();
      }
    }
  }
  A.updateDevice();
  B.updateDevice();
  C0.updateDevice();
  C1.updateDevice();

  // C = alpha * Op(D) * Op(B) + beta * C
  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<k; iy++) {
      for(int ix=0; ix<n; ix++) {
        RealType sum = 0;
        for(int im=0; im<m; im++) {
          sum += alpha * A(im, ix, iz) * B(iy, im, iz);
        }
        Ref(ix, iy, iz) = sum + beta * C0(ix, iy, iz);
        Ref(ix, iy, iz) = Ref(ix, iy, iz) + beta * C1(ix, iy, iz);
      }
    }
  }

  auto _A = A.mdspan();
  auto _B = B.mdspan();
  auto _C0 = C0.mdspan();
  auto _C1 = C1.mdspan();
  auto _C2 = C2.mdspan();

  // Sender to perform Axpy
  nvexec::stream_context stream_ctx{};
  auto scheduler = stream_ctx.get_scheduler();

  // Get stream from the scheduler
  auto stream_getter = stdexec::schedule(scheduler)
                     | stdexec::let_value([] {
                         return nvexec::get_stream();
                       })
                     | stdexec::then([](cudaStream_t stream) {
                         return stream;
                       });
  auto [stream] = stdexec::sync_wait(std::move(stream_getter)).value();

  // Matrix matrix product using cublas
  Impl::blasHandle_t blas_handle;
  blas_handle.create();
  blas_handle.set_stream(stream);
  auto blas_sender = stdexec::just()
                   | stdexec::then([&]{
                                        Impl::matrix_matrix_product(blas_handle, _A, _B, _C0, "T", "T", alpha, beta);
                                      });

  auto size = C0.size();
  auto axpy_sender = blas_sender
                   | exec::on(scheduler, stdexec::bulk(size,
                     [=] MDSPAN_FORCE_INLINE_FUNCTION (const int idx) {
                       const int ix  = idx % n;
                       const int iyz = idx / n;
                       const int iy  = iyz % k;
                       const int iz  = iyz / k;
                       _C2(ix, iy, iz) = _C0(ix, iy, iz) + beta * _C1(ix, iy, iz);
                     })
                   );

  stdexec::sync_wait( std::move(axpy_sender) );
  constexpr RealType eps = 1.e-12;

  C2.updateSelf();

  for(int iz=0; iz<l; iz++) {
    for(int iy=0; iy<k; iy++) {
      for(int ix=0; ix<n; ix++) {
        EXPECT_NEAR( Ref(ix, iy, iz), C2(ix, iy, iz), eps );
      }
    }
  }
}

TEST( ThenAndAxpy, Solver ) {
  using RealType = double;
  const RealType beta = 1.19;

  const std::size_t m = 4, l = 128;
  View3D<RealType> a0("a0", m, m, l), a1("a1", m, m, l), a2("a2", m, m, l);
  View2D<RealType> v("v", m, l);
  View3D<RealType> ref_w("ref_w", m, m, l);
  View2D<RealType> ref_v("ref_v", m, l);

  auto rand_engine = std::mt19937(0);
  auto rand_dist = std::uniform_real_distribution<RealType>(-1, 1);
  auto rand_gen = std::bind(rand_dist, rand_engine);

  // Set Diagonal matrix (np.diag(1,2,3))
  for(int iz=0; iz<l; iz++) {
    for(int id=0; id<m; id++) {
      a0(id, id, iz) = id+1;
      a1(id, id, iz) = rand_gen();
    }
  }
  a0.updateDevice();
  a1.updateDevice();

  // The eigen values and vectors should be
  // v == [1, 2, 3], w == np.eye(3)
  for(int iz=0; iz<l; iz++) {
    for(int id=0; id<m; id++) {
      ref_w(id, id, iz) = 1;
      ref_w(id, id, iz) = ref_w(id, id, iz) + beta * a1(id, id, iz);
      ref_v(id, iz) = id+1;
    }
  }

  auto _a0 = a0.mdspan();
  auto _a1 = a1.mdspan();
  auto _a2 = a2.mdspan();
  auto _v  = v.mdspan();

  // Sender to perform Axpy
  nvexec::stream_context stream_ctx{};
  auto scheduler = stream_ctx.get_scheduler();

  // Get stream from the scheduler
  auto stream_getter = stdexec::schedule(scheduler)
                     | stdexec::let_value([] {
                         return nvexec::get_stream();
                       })
                     | stdexec::then([](cudaStream_t stream) {
                         return stream;
                       });
  auto [stream] = stdexec::sync_wait(std::move(stream_getter)).value();

  // Eigen value decomposition
  Impl::syevjHandle_t<double> syevj_handle;
  syevj_handle.create(_a0, _v);
  syevj_handle.set_stream(stream);
  auto evd_sender = stdexec::just()
                  | stdexec::then([&]{
                                       Impl::eig(syevj_handle, _a0, _v);
                                     });

  auto size = a0.size();
  auto axpy_sender = evd_sender
                   | exec::on(scheduler, stdexec::bulk(size,
                     [=] MDSPAN_FORCE_INLINE_FUNCTION (const int idx) {
                       const int ix  = idx % m;
                       const int iyz = idx / m;
                       const int iy  = iyz % m;
                       const int iz  = iyz / m;
                       _a2(iy, ix, iz) = _a0(iy, ix, iz) + beta * _a1(iy, ix, iz);
                     })
                   );

  stdexec::sync_wait( std::move(axpy_sender) );
  a2.updateSelf();
  v.updateSelf();

  constexpr RealType eps = 1.e-13;
  for(int iz=0; iz<l; iz++) {
    for(int ix=0; ix<m; ix++) {
      EXPECT_NEAR( v(ix, iz), ref_v(ix, iz), eps );
    }
    for(int iy=0; iy<m; iy++) {
      for(int ix=0; ix<m; ix++) {
        EXPECT_NEAR( a2(ix, iy, iz), ref_w(ix, iy, iz), eps );
      }
    }
  }
}
