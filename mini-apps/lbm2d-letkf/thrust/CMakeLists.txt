file(GLOB SOURCES "*.cpp")
add_executable(lbm2d-letkf-thrust ${SOURCES})

find_package(MPI REQUIRED COMPONENTS CXX)

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")
if(BACKEND STREQUAL "CUDA")
    target_link_libraries(lbm2d-letkf-thrust PUBLIC Threads::Threads STDEXEC::stdexec STDEXEC::nvexec)
    target_compile_features(lbm2d-letkf-thrust PUBLIC cxx_std_20)
elseif(BACKEND STREQUAL "OPENMP")
    target_link_libraries(lbm2d-letkf-thrust PUBLIC Threads::Threads STDEXEC::stdexec STDEXEC::nvexec)
    target_compile_options(lbm2d-letkf-thrust PUBLIC ${OpenMP_CXX_FLAGS})
    target_compile_features(lbm2d-letkf-thrust PUBLIC cxx_std_20)
elseif(BACKEND STREQUAL "HIP")
    target_compile_features(lbm2d-letkf-thrust PUBLIC cxx_std_20)
    target_compile_options(lbm2d-letkf-thrust PUBLIC -std=c++20)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, HIP and OPENMP must be On.")
endif()

# Enable macro
target_compile_definitions(lbm2d-letkf-thrust PUBLIC $<$<BOOL:${USE_SINGLE_PRECISION}>: USE_SINGLE_PRECISION>)
target_link_libraries(lbm2d-letkf-thrust PUBLIC MPI::MPI_CXX math_lib nlohmann_json::nlohmann_json)
