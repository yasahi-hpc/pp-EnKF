#include <sycl/sycl.hpp>
#include <iostream>
#include <string>
#include "../config.hpp"
#include "../parser.hpp"
#include "heat3D.hpp"
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
  auto nx = parser.shape_[0];
  auto ny = parser.shape_[1];
  auto nz = parser.shape_[2];

  Config conf(nx, ny, nz, parser.nbiter_, parser.freq_diag_);

  // The default device selector will select the most performant device.
  auto selector = sycl::gpu_selector_v;

  try {
    sycl::queue q(selector, exception_handler);

    // Print out the device information used for the kernel code.
    std::cout << "Running on device: "
              << q.get_device().get_info<sycl::info::device::name>() << "\n";

    // Declare grid and variables
    Grid<double> grid(q, conf);
    Variable<double> variables(q, conf);

    initialize(q, conf, grid, variables);

    auto start = std::chrono::high_resolution_clock::now();
    solve(q, conf, variables);
    auto end = std::chrono::high_resolution_clock::now();
    double seconds = std::chrono::duration_cast<std::chrono::duration<double> >(end - start).count();

    // Sanity check
    finalize(q, conf, grid, variables);

    // Report GBytes and GFlops
    report_performance(conf, seconds);
  } catch (std::exception const &e){
    std::cout << "An exception is caught while computing on device.\n";
    std::terminate();
  }
  return 0;
}