add_executable(08_test_overlapping benchmark.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)
target_link_libraries(08_test_overlapping PUBLIC math_lib Threads::Threads STDEXEC::stdexec STDEXEC::nvexec MPI::MPI_CXX)
target_compile_features(08_test_overlapping PUBLIC cxx_std_20)
