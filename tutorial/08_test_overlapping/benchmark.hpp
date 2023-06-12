#ifndef __BENCHMARK_HPP__
#define __BENCHMARK_HPP__

#include <mpi.h>
#include <cassert>
#include <array>
#include <algorithm>
#include <numeric>
#include <stdexec/execution.hpp>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cublas_v2.h>
#include "exec/on.hpp"
#include "config.hpp"
#include "types.hpp"
#include "timer.hpp"

template <class ViewType>
void all2all(const ViewType& a, ViewType& b) {
  assert( a.extents() == b.extents() );
  const std::size_t size = a.extent(0) * a.extent(1);
  MPI_Alltoall(a.data_handle(),
               size,
               MPI_DOUBLE,
               b.data_handle(),
               size,
               MPI_DOUBLE,
               MPI_COMM_WORLD);
}

template <class InputView, class OutputView>
void transpose(const InputView& in, OutputView& out) {
  // transpose by cublas
  cublasHandle_t handle;
  cublasCreate(&handle);

  constexpr double alpha = 1;
  constexpr double beta = 0;

  cublasDgeam(handle,
              CUBLAS_OP_T,
              CUBLAS_OP_T,
              in.extent(1),
              in.extent(0),
              &alpha,
              in.data_handle(),
              in.extent(0),
              &beta,
              in.data_handle(),
              in.extent(0),
              out.data_handle(),
              out.extent(0));

  cublasDestroy(handle);
} 

template <class Scheduler, class Functor, std::size_t N>
stdexec::sender auto bulk_generator(Scheduler&& scheduler, const std::array<std::size_t, N>& shape, Functor&& functor) {
  const std::size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  const int n0 = shape[0];
  const int n1 = shape[1];
  auto functor_1d = [=] MDSPAN_FORCE_INLINE_FUNCTION (const int idx) {
    const int i0  = idx % n0;
    const int i12 = idx / n0;
    const int i1  = i12%n1;
    const int i2  = i12/n1;
    functor(i0, i1, i2);
  };
  auto bulk = stdexec::just() | exec::on(scheduler, stdexec::bulk(size, functor_1d));
  return bulk;
}

template <class Sender, class Scheduler, class Functor, std::size_t N>
stdexec::sender auto bulk_connector(Sender&& sender, Scheduler&& scheduler, const std::array<std::size_t, N>& shape, Functor&& functor) {
  const std::size_t size = std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
  const int n0 = shape[0];
  const int n1 = shape[1];
  auto functor_1d = [=] MDSPAN_FORCE_INLINE_FUNCTION (const int idx) {
    const int i0  = idx % n0;
    const int i12 = idx / n0;
    const int i1  = i12%n1;
    const int i2  = i12/n1;
    functor(i0, i1, i2);
  };
  auto bulk = sender | exec::on(scheduler, stdexec::bulk(size, functor_1d));
  return bulk;
}

template <class Scheduler, class Functor>
stdexec::sender auto then_generator(Scheduler&& scheduler, Functor&& functor) {
  return stdexec::just() | stdexec::then( [&]{ functor(); } );
}

template <class Sender, class Functor>
stdexec::sender auto then_connector(Sender&& sender, Functor&& functor) {
  return sender | stdexec::then( [&]{ functor(); } );
}

/* Serialized comm */
template <class Scheduler>
void comm_task(Scheduler&& scheduler, const std::size_t size, Timer *timer) {
  thrust::device_vector<double> _A(Config::nx_ * Config::ny_ * size);
  thrust::device_vector<double> _B(Config::nx_ * Config::ny_ * size);
  RealView3D A( (double *)thrust::raw_pointer_cast(_A.data()), Config::nx_, Config::ny_, size);
  RealView3D B( (double *)thrust::raw_pointer_cast(_B.data()), Config::nx_, Config::ny_, size);
  
  timer->begin();
  for(int it=0; it<Config::nbiter_; it++) {
    auto comm = then_generator(scheduler, [&](){ all2all(A, B);} );
    stdexec::sync_wait( std::move(comm) );
  }
  timer->end();
}

template <class Scheduler>
void commh2h_task(Scheduler&& scheduler, const std::size_t size, Timer *timer) {
  thrust::host_vector<double> _A(Config::nx_ * Config::ny_ * size);
  thrust::host_vector<double> _B(Config::nx_ * Config::ny_ * size);
  RealView3D A( (double *)thrust::raw_pointer_cast(_A.data()), Config::nx_, Config::ny_, size);
  RealView3D B( (double *)thrust::raw_pointer_cast(_B.data()), Config::nx_, Config::ny_, size);
  
  timer->begin();
  for(int it=0; it<Config::nbiter_; it++) {
    auto comm = then_generator(scheduler, [&](){ all2all(A, B);} );
    stdexec::sync_wait( std::move(comm) );
  }
  timer->end();
}

/* Serialized transpose (cublas) */
template <class Scheduler>
void transpose_task(Scheduler&& scheduler, const std::size_t size, Timer *timer) {
  thrust::device_vector<double> _C(Config::nx_ * Config::ny_);
  thrust::device_vector<double> _D(Config::nx_ * Config::ny_);
  RealView2D C( (double *)thrust::raw_pointer_cast(_C.data()), Config::nx_, Config::ny_);
  RealView2D D( (double *)thrust::raw_pointer_cast(_D.data()), Config::ny_, Config::nx_);
  
  timer->begin();
  for(int it=0; it<Config::nbiter_; it++) {
    auto trans = then_generator(scheduler, [&](){ transpose(C, D); });
    stdexec::sync_wait( std::move(trans) );
  }
  timer->end();
}

/* Serialized comm and transpose (cublas) */
template <class Scheduler>
void sync_comm_transpose(Scheduler&& scheduler, const std::size_t size, Timer *timer) {
  thrust::device_vector<double> _A(Config::nx_ * Config::ny_ * size);
  thrust::device_vector<double> _B(Config::nx_ * Config::ny_ * size);
  thrust::device_vector<double> _C(Config::nx_ * Config::ny_);
  thrust::device_vector<double> _D(Config::nx_ * Config::ny_);
  RealView3D A( (double *)thrust::raw_pointer_cast(_A.data()), Config::nx_, Config::ny_, size);
  RealView3D B( (double *)thrust::raw_pointer_cast(_B.data()), Config::nx_, Config::ny_, size);
  RealView2D C( (double *)thrust::raw_pointer_cast(_C.data()), Config::nx_, Config::ny_);
  RealView2D D( (double *)thrust::raw_pointer_cast(_D.data()), Config::ny_, Config::nx_);
  
  timer->begin();
  for(int it=0; it<Config::nbiter_; it++) {
    auto comm = then_generator(scheduler, [&](){ all2all(A, B);} );
    auto comm_then = then_connector(std::move(comm), [&](){ transpose(C, D); });
    stdexec::sync_wait( std::move(comm_then) );
  }
  timer->end();
}

/* Asynchronous comm and transpose (cublas) */
template <class Scheduler>
void async_comm_transpose(Scheduler&& scheduler, const std::size_t size, Timer *timer) {
  thrust::device_vector<double> _A(Config::nx_ * Config::ny_ * size);
  thrust::device_vector<double> _B(Config::nx_ * Config::ny_ * size);
  thrust::device_vector<double> _C(Config::nx_ * Config::ny_);
  thrust::device_vector<double> _D(Config::nx_ * Config::ny_);
  RealView3D A( (double *)thrust::raw_pointer_cast(_A.data()), Config::nx_, Config::ny_, size);
  RealView3D B( (double *)thrust::raw_pointer_cast(_B.data()), Config::nx_, Config::ny_, size);
  RealView2D C( (double *)thrust::raw_pointer_cast(_C.data()), Config::nx_, Config::ny_);
  RealView2D D( (double *)thrust::raw_pointer_cast(_D.data()), Config::ny_, Config::nx_);
  
  timer->begin();
  for(int it=0; it<Config::nbiter_; it++) {
    auto comm = then_generator(scheduler, [&](){ all2all(A, B);} );
    auto trans = then_generator(scheduler, [&](){ transpose(C, D); });
    auto async = stdexec::when_all( std::move(comm), std::move(trans) );
    stdexec::sync_wait( std::move(async) );
  }
  timer->end();
}

/* Asynchronous comm (H2H) and transpose (cublas) */
template <class Scheduler>
void async_commh2h_transpose(Scheduler&& scheduler, const std::size_t size, Timer *timer) {
  thrust::host_vector<double> _A(Config::nx_ * Config::ny_ * size);
  thrust::host_vector<double> _B(Config::nx_ * Config::ny_ * size);
  thrust::device_vector<double> _C(Config::nx_ * Config::ny_);
  thrust::device_vector<double> _D(Config::nx_ * Config::ny_);
  RealView3D A( (double *)thrust::raw_pointer_cast(_A.data()), Config::nx_, Config::ny_, size);
  RealView3D B( (double *)thrust::raw_pointer_cast(_B.data()), Config::nx_, Config::ny_, size);
  RealView2D C( (double *)thrust::raw_pointer_cast(_C.data()), Config::nx_, Config::ny_);
  RealView2D D( (double *)thrust::raw_pointer_cast(_D.data()), Config::ny_, Config::nx_);
  
  timer->begin();
  for(int it=0; it<Config::nbiter_; it++) {
    auto comm = then_generator(scheduler, [&](){ all2all(A, B);} );
    auto trans = then_generator(scheduler, [&](){ transpose(C, D); });
    auto async = stdexec::when_all( std::move(comm), std::move(trans) );
    stdexec::sync_wait( std::move(async) );
  }
  timer->end();
}

/* Serialized bulk (axpy) */
template <class Scheduler>
void axpy_task(Scheduler&& scheduler, const std::size_t size, Timer *timer) {
  thrust::device_vector<double> _A(Config::nx_ * Config::ny_ * size);
  thrust::device_vector<double> _B(Config::nx_ * Config::ny_ * size);
  thrust::device_vector<double> _C(Config::nx_ * Config::ny_ * size);
  thrust::device_vector<double> _D(Config::nx_ * Config::ny_ * size);
  RealView3D A( (double *)thrust::raw_pointer_cast(_A.data()), Config::nx_, Config::ny_, size);
  RealView3D B( (double *)thrust::raw_pointer_cast(_B.data()), Config::nx_, Config::ny_, size);
  RealView3D C( (double *)thrust::raw_pointer_cast(_C.data()), Config::nx_, Config::ny_, size);
  RealView3D D( (double *)thrust::raw_pointer_cast(_D.data()), Config::nx_, Config::ny_, size);
  double alpha = 1.0;
  auto axpy = [=] MDSPAN_FORCE_INLINE_FUNCTION (const int i0, const int i1, const int i2) {
    C(i0, i1, i2) = C(i0, i1, i2) + alpha * D(i0, i1, i2);
  };

  timer->begin();
  for(int it=0; it<Config::nbiter_; it++) {
    auto bulk = bulk_connector(stdexec::just(), scheduler, std::array<std::size_t, 3>({Config::nx_, Config::ny_, size}), axpy);
    stdexec::sync_wait( std::move(bulk) );
  }
  timer->end();
}

/* Serialized comm and bulk (axpy) */
template <class Scheduler>
void sync_comm_bulk(Scheduler&& scheduler, const std::size_t size, Timer *timer) {
  thrust::device_vector<double> _A(Config::nx_ * Config::ny_ * size);
  thrust::device_vector<double> _B(Config::nx_ * Config::ny_ * size);
  thrust::device_vector<double> _C(Config::nx_ * Config::ny_ * size);
  thrust::device_vector<double> _D(Config::nx_ * Config::ny_ * size);
  RealView3D A( (double *)thrust::raw_pointer_cast(_A.data()), Config::nx_, Config::ny_, size);
  RealView3D B( (double *)thrust::raw_pointer_cast(_B.data()), Config::nx_, Config::ny_, size);
  RealView3D C( (double *)thrust::raw_pointer_cast(_C.data()), Config::nx_, Config::ny_, size);
  RealView3D D( (double *)thrust::raw_pointer_cast(_D.data()), Config::nx_, Config::ny_, size);
  double alpha = 1.0;
  auto axpy = [=] MDSPAN_FORCE_INLINE_FUNCTION (const int i0, const int i1, const int i2) {
    C(i0, i1, i2) = C(i0, i1, i2) + alpha * D(i0, i1, i2);
  };

  timer->begin();
  for(int it=0; it<Config::nbiter_; it++) {
    auto comm = then_generator(scheduler, [&](){ all2all(A, B);} );
    auto bulk = bulk_connector(std::move(comm), scheduler, std::array<std::size_t, 3>({Config::nx_, Config::ny_, size}), axpy);
    stdexec::sync_wait( std::move(bulk) );
  }
  timer->end();
}

/* Overlapping comm and bulk (axpy) */
template <class Scheduler>
void async_comm_bulk(Scheduler&& scheduler, const std::size_t size, Timer *timer) {
  thrust::device_vector<double> _A(Config::nx_ * Config::ny_ * size);
  thrust::device_vector<double> _B(Config::nx_ * Config::ny_ * size);
  thrust::device_vector<double> _C(Config::nx_ * Config::ny_ * size);
  thrust::device_vector<double> _D(Config::nx_ * Config::ny_ * size);
  RealView3D A( (double *)thrust::raw_pointer_cast(_A.data()), Config::nx_, Config::ny_, size);
  RealView3D B( (double *)thrust::raw_pointer_cast(_B.data()), Config::nx_, Config::ny_, size);
  RealView3D C( (double *)thrust::raw_pointer_cast(_C.data()), Config::nx_, Config::ny_, size);
  RealView3D D( (double *)thrust::raw_pointer_cast(_D.data()), Config::nx_, Config::ny_, size);
  double alpha = 1.0;
  auto axpy = [=] MDSPAN_FORCE_INLINE_FUNCTION (const int i0, const int i1, const int i2) {
    C(i0, i1, i2) = C(i0, i1, i2) + alpha * D(i0, i1, i2);
  };

  timer->begin();
  for(int it=0; it<Config::nbiter_; it++) {
    auto comm = then_generator(scheduler, [&](){ all2all(A, B);} );
    auto bulk = bulk_generator(scheduler, std::array<std::size_t, 3>({Config::nx_, Config::ny_, size}), axpy);
    auto async = stdexec::when_all( std::move(comm), std::move(bulk) );
    stdexec::sync_wait( std::move(async) );
  }
  timer->end();
}

#endif
