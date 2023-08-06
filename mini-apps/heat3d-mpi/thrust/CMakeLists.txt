add_executable(heat3d-mpi-thrust heat3D.cpp)

find_package(MPI REQUIRED COMPONENTS CXX)

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")
if((BACKEND STREQUAL "CUDA") OR (BACKEND STREQUAL "OPENMP"))
    target_link_libraries(heat3d-mpi-thrust PUBLIC Threads::Threads STDEXEC::stdexec STDEXEC::nvexec)
    target_compile_features(heat3d-mpi-thrust PUBLIC cxx_std_20)
elseif(BACKEND STREQUAL "HIP")
    enable_language(HIP)
    target_compile_features(heat3d-mpi-thrust PUBLIC cxx_std_17)
    target_compile_options(heat3d-mpi-thrust PUBLIC -std=c++1z)
    set_source_files_properties(heat3D.cpp PROPERTIES LANGUAGE HIP)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, HIP and OPENMP must be On.")
endif()

target_link_libraries(heat3d-mpi-thrust PUBLIC MPI::MPI_CXX math_lib)
