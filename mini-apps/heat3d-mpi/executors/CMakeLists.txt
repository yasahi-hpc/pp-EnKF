add_executable(heat3d-mpi-exec heat3D.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)
target_link_libraries(heat3d-mpi-exec PUBLIC Threads::Threads STDEXEC::stdexec STDEXEC::nvexec MPI::MPI_CXX)
target_compile_features(heat3d-mpi-exec PUBLIC cxx_std_20)

# Enable macro
target_compile_definitions(heat3d-mpi-exec PUBLIC $<$<BOOL:${OVERLAP}>: OVERLAP>)
