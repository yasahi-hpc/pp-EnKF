#include "solver.hpp"
#include <stdexcept>
#include <iostream>

int main(int argc, char *argv[]) {
  Solver solver;

  try {
    solver.initialize(&argc, &argv);
    solver.run();
    solver.finalize();

    return 0;
  } catch(std::runtime_error e) {
    std::cerr << e.what() << std::endl;
    solver.finalize();
  }
}
