#include <iostream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <array>
#include <stdexec/execution.hpp>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/execution_policy.h>
#if defined(ENABLE_OPENMP)
  #include <exec/static_thread_pool.hpp>
#else
  #include "nvexec/stream_context.cuh"
#endif
#include "exec/on.hpp"
#include "heat2D.hpp"

int main(int argc, char *argv[]) {
  // Set configuration
  Config conf;

  // Declare grid and values
  thrust::device_vector<double> _x(conf.nx_);
  thrust::device_vector<double> _y(conf.ny_);
  thrust::device_vector<double> _u(conf.nx_ * conf.ny_);
  thrust::device_vector<double> _un(conf.nx_ * conf.ny_);

  // Viewed to mdspans
  RealView1D x( (double *)thrust::raw_pointer_cast(_x.data()), std::array<std::size_t, 1>({conf.nx_}) );
  RealView1D y( (double *)thrust::raw_pointer_cast(_y.data()), std::array<std::size_t, 1>({conf.ny_}) );
  RealView2D u( (double *)thrust::raw_pointer_cast(_u.data()), std::array<std::size_t, 2>({conf.nx_, conf.ny_}) );
  RealView2D un( (double *)thrust::raw_pointer_cast(_un.data()), std::array<std::size_t, 2>({conf.nx_, conf.ny_}) );

  // Declare a scheduler
  #if defined(ENABLE_OPENMP)
    exec::static_thread_pool pool{std::thread::hardware_concurrency()};
    auto scheduler = pool.get_scheduler();
  #else
    nvexec::stream_context stream_ctx{};
    auto scheduler = stream_ctx.get_scheduler();
  #endif

  initialize(conf, scheduler, x, y, u, un);
  auto start = std::chrono::high_resolution_clock::now();

  // Run all the operations inside a single repeat_n sender
  auto heat_2d = solve_heat2d(conf, scheduler, u, un);
  stdexec::sync_wait(heat_2d);

  auto end = std::chrono::high_resolution_clock::now();
  double seconds = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();

  double time = conf.dt_ * conf.nbiter_;
  finalize(conf, time, scheduler, x, y, u, un);
  report_performance(conf, seconds);

  return 0;
}