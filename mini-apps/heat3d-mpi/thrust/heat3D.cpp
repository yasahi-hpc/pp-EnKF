#include <chrono>
#include <utils/string_utils.hpp>
#include <utils/system_utils.hpp>
#include <utils/io_utils.hpp>
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
  auto is_async = parser.is_async_;
  auto use_time_stamps = parser.use_time_stamps_;

  // Begin MPI
  Comm comm(argc, argv, shape, topology);
  Config conf(shape, topology, comm.cart_rank(),
              parser.nbiter_, parser.freq_diag_, is_async, use_time_stamps);

  // Declare timers
  std::vector<Timer*> timers;
  defineTimers(timers, use_time_stamps);
  MPI_Barrier(MPI_COMM_WORLD);
  resetTimers(timers); // In order to share the initial time among all the timers

  // Declare grid and variables
  Grid<double> grid(conf);
  Variable<double> variables(conf);

  initialize(conf, grid, comm, variables);

  timers[Total]->begin();
  solve(conf, comm, variables, timers);
  timers[Total]->end();

  // Sanity check
  finalize(conf, grid, comm, variables);

  // Save elapsed time in csv
  const std::string performance_dir = "heat3d-mpi-thrust/performance";
  if(comm.is_master()) {
    Impl::mkdirs(performance_dir, 0755);
  }
  MPI_Barrier(MPI_COMM_WORLD);
 
  const std::string filename = performance_dir + "/" + "elapsed_time_rank" + std::to_string(comm.rank()) + ".csv";
  auto performance_dict = timersToDict(timers);
  Impl::to_csv(filename, performance_dict);
 
  if(conf.use_time_stamps_) {
    const std::string timestamps_filename = performance_dir + "/" + "time_stamps_rank" + std::to_string(comm.rank()) + ".csv";
    auto timestamps_dict = timeStampsToDict(timers);
    Impl::to_csv(timestamps_filename, timestamps_dict);
  }

  // Report GBytes and GFlops
  if(comm.is_master()) {
    report_performance(conf, timers[Total]->seconds());
    printTimers(timers);
  }

  // Finalize MPI
  freeTimers(timers);
  comm.finalize();

  return 0;
}
