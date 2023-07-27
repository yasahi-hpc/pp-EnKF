add_executable(heat3d-mpi-stdpar heat3D.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)

if(BACKEND STREQUAL "CUDA")
    target_compile_options(heat3d-mpi-stdpar PUBLIC -stdpar=gpu)
    target_link_options(heat3d-mpi-stdpar PUBLIC -stdpar=gpu)
elseif(BACKEND STREQUAL "OPENMP")
    target_compile_options(heat3d-mpi-stdpar PUBLIC -stdpar=multicore)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, and OPENMP must be On.")
endif()

target_link_libraries(heat3d-mpi-stdpar PUBLIC MPI::MPI_CXX math_lib)
target_compile_features(heat3d-mpi-stdpar PUBLIC cxx_std_20)
