file(GLOB _SOURCES "*.cpp")
add_executable(lbm2d-letkf-sycl ${_SOURCES})

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(AdaptiveCpp CONFIG REQUIRED)
add_sycl_to_target(TARGET lbm2d-letkf-sycl SOURCES ${_SOURCES})

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")
if(BACKEND STREQUAL "CUDA")
    target_link_options(lbm2d-letkf-sycl PUBLIC -cudalib=cusolver,cublas,curand)
elseif(BACKEND STREQUAL "OPENMP")
    target_compile_definitions(lbm2d-letkf-sycl PUBLIC SIMD)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, and OPENMP must be On.")
endif()

# Enable macro
target_compile_definitions(lbm2d-letkf-sycl PUBLIC $<$<BOOL:${USE_SINGLE_PRECISION}>: USE_SINGLE_PRECISION>)
target_link_libraries(lbm2d-letkf-sycl PUBLIC MPI::MPI_CXX math_lib nlohmann_json::nlohmann_json)