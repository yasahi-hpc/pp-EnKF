add_executable(heat3d-mpi-executors heat3D.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)
target_link_libraries(heat3d-mpi-executors PUBLIC Threads::Threads STDEXEC::stdexec STDEXEC::nvexec MPI::MPI_CXX math_lib)
target_compile_features(heat3d-mpi-executors PUBLIC cxx_std_20)
