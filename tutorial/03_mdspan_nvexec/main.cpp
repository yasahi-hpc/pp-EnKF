#include <iostream>
#include <stdexec/execution.hpp>
#include <experimental/mdspan>
#include <thrust/device_vector.h>
#include "nvexec/stream_context.cuh"
#include "exec/on.hpp"

using stdexec::just; using stdexec::on;
using stdexec::sender; using stdexec::schedule;
using stdexec::then; using stdexec::sync_wait;
using stdexec::run_loop; using stdexec::get_scheduler;
using stdexec::let_value; using stdexec::when_all;
using stdexec::get_stop_token; using stdexec::bulk;
namespace stdex = std::experimental;

int main(int argc, char *argv[]) {
  // Declare a CUDA stream
  nvexec::stream_context stream_ctx{};
  auto sched = stream_ctx.get_scheduler();

  const std::size_t nx = 3, ny = 2;
  const std::size_t n = nx * ny;
  thrust::device_vector<std::size_t> a(n, 1), b(n, 2);
  stdex::mdspan<double, stdex::extents<std::size_t, ny, nx>> dst((double *)thrust::raw_pointer_cast(a.data()));
  stdex::mdspan<double, stdex::extents<std::size_t, ny, nx>> src((double *)thrust::raw_pointer_cast(b.data()));

  auto copy_kernel = [dst, src](const std::size_t idx) {
    std::size_t ix = idx % nx;
    std::size_t iy = idx / nx;
    dst(ix, iy) = src(ix, iy);
  };

  auto init = just()
            | exec::on(sched, bulk(n, copy_kernel)); 
  stdexec::sync_wait(init);

  return 0;
}
