add_executable(heat3d-mpi-sycl heat3D.cpp)

find_package(AdaptiveCpp CONFIG REQUIRED)
find_package(MPI REQUIRED COMPONENTS CXX)

add_sycl_to_target(TARGET heat3d-mpi-sycl SOURCES heat3D.cpp)

target_link_libraries(heat3d-mpi-sycl PUBLIC std::mdspan MPI::MPI_CXX math_lib)
target_compile_features(heat3d-mpi-sycl PUBLIC cxx_std_17)
target_compile_options(heat3d-mpi-sycl PUBLIC -O3)