#include "solver.hpp"
#include <stdexcept>
#include <iostream>

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

int main(int argc, char *argv[]) {
  #if defined(ENABLE_OPENMP)
    auto selector = sycl::cpu_selector_v;
  #else
    auto selector = sycl::gpu_selector_v;
  #endif
  //auto selector = sycl::default_selector_v;
  Solver solver;

  try {
    sycl::queue q(selector, exception_handler, sycl::property_list{sycl::property::queue::in_order{}});

    solver.initialize(q, &argc, &argv);
    solver.run();
    solver.finalize();
  } catch(std::runtime_error e) {
    std::cerr << e.what() << std::endl;
    solver.finalize();
  }
  return 0;
}
