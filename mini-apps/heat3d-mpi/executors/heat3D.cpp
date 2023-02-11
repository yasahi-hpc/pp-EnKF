#include <chrono>
#include "nvexec/stream_context.cuh"
#include "../config.hpp"
#include "../parser.hpp"
#include "heat3D.hpp"
#include "mpi_comm.hpp"
#include "variable.hpp"
#include "grid.hpp"

int main(int argc, char *argv[]) {
  Parser parser(argc, argv);
  auto shape = parser.shape_;
  auto topology = parser.topology_;

  // Begin MPI
  Comm comm(argc, argv, shape, topology);
  Config conf(shape, topology, comm.cart_rank(), parser.nbiter_, parser.freq_diag_);

  // Declare grid and variables
  Grid<double> grid(conf);
  Variable<double> variables(conf);

  // Declare a CUDA stream
  nvexec::stream_context stream_ctx{};
  auto scheduler = stream_ctx.get_scheduler();

  initialize(conf, grid, scheduler, comm, variables);
  auto start = std::chrono::high_resolution_clock::now();
  solve(conf, scheduler, comm, variables);
  auto end = std::chrono::high_resolution_clock::now();
  double seconds = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();

  // Sanity check
  finalize(conf, grid, scheduler, comm, variables);

  // Report GBytes and GFlops
  if(comm.is_master()) {
    report_performance(conf, seconds);
  }

  // Finalize MPI
  comm.finalize();

  return 0;
}
