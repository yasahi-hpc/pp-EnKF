#include <sycl/sycl.hpp>
#include <iostream>
#include <string>
#include <utils/string_utils.hpp>
#include <utils/system_utils.hpp>
#include <utils/io_utils.hpp>
#include "../config.hpp"
#include "../parser.hpp"
#include "../timer.hpp"
#include "heat3D.hpp"
#include "mpi_comm.hpp"
#include "variable.hpp"
#include "grid.hpp"

static auto exception_handler = [](sycl::exception_list e_list) {
  for (std::exception_ptr const &e : e_list) {
    try {
      std::rethrow_exception(e);
    }
    catch (std::exception const &e) {
#if _DEBUG
      std::cout << "Failure" << std::endl;
#endif
      std::terminate();
    }
  }
};

int main(int argc, char* argv[]) {
  Parser parser(argc, argv);
  auto shape = parser.shape_;
  auto topology = parser.topology_;
  auto is_async = parser.is_async_;
  auto use_time_stamps = parser.use_time_stamps_;

  // The default device selector will select the most performant device.
  #if defined(ENABLE_OPENMP)
    auto selector = sycl::cpu_selector_v;
  #else
    auto selector = sycl::gpu_selector_v;
  #endif

  try {
    std::vector<Timer*> timers;
    defineTimers(timers, use_time_stamps);

    sycl::queue q(selector, exception_handler);

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<sycl::info::device::name>() << "\n";

    // Begin MPI
    Comm comm(argc, argv, q, shape, topology);
    Config conf(shape, topology, comm.cart_rank(),
                parser.nbiter_, parser.freq_diag_, is_async, use_time_stamps);

    // Declare grid and variables
    Grid<double> grid(q, conf);
    Variable<double> variables(q, conf);

    initialize(q, conf, grid, comm, variables);

    MPI_Barrier(MPI_COMM_WORLD);
    resetTimers(timers); // In order to share the initial time among all the timers

    timers[Total]->begin();
    solve(q, conf, comm, variables, timers);
    timers[Total]->end();

    // Sanity check
    finalize(q, conf, grid, comm, variables);

    // Save elapsed time in csv
    const std::string performance_dir = is_async ? "heat3d-mpi-sycl-async/performance" : "heat3d-mpi-sycl/performance";
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
  } catch (std::exception const &e){
    std::cout << "An exception is caught while computing on device.\n";
    std::terminate();
  }

  return 0;
}