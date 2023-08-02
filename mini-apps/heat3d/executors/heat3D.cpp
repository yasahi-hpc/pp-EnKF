#include <chrono>
#include "../config.hpp"
#include "../parser.hpp"
#include "heat3D.hpp"
#include "variable.hpp"
#include "grid.hpp"

#if defined(ENABLE_OPENMP)
  #include <exec/static_thread_pool.hpp>
#else
  #include "nvexec/stream_context.cuh"
#endif

int main(int argc, char *argv[]) {
  Parser parser(argc, argv);
  auto nx = parser.shape_[0];
  auto ny = parser.shape_[1];
  auto nz = parser.shape_[2];

  Config conf(nx, ny, nz, parser.nbiter_, parser.freq_diag_);

  // Declare grid and variables
  Grid<double> grid(conf);
  Variable<double> variables(conf);

  #if defined(ENABLE_OPENMP)
    exec::static_thread_pool pool{std::thread::hardware_concurrency()};
    auto scheduler = pool.get_scheduler();
  #else
    nvexec::stream_context stream_ctx{};
    auto scheduler = stream_ctx.get_scheduler();
  #endif

  initialize(conf, grid, scheduler, variables);
  auto start = std::chrono::high_resolution_clock::now();
  solve(conf, scheduler, variables);
  auto end = std::chrono::high_resolution_clock::now();
  double seconds = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();

  // Sanity check
  finalize(conf, grid, scheduler, variables);

  // Report GBytes and GFlops
  report_performance(conf, seconds);

  return 0;
}
