file(GLOB _SOURCES "*.cpp")
add_executable(vlp4d-sycl ${_SOURCES})

find_package(AdaptiveCpp CONFIG REQUIRED)
add_sycl_to_target(TARGET vlp4d-sycl SOURCES ${_SOURCES})

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")
if(BACKEND STREQUAL "CUDA")
    target_link_options(vlp4d-sycl PUBLIC -cudalib=cufft,cublas)
elseif(BACKEND STREQUAL "OPENMP")
    target_compile_definitions(vlp4d-sycl PUBLIC SIMD)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, and OPENMP must be On.")
endif()

target_link_libraries(vlp4d-sycl PUBLIC math_lib)