add_executable(google-tests-sycl
    test_view.cpp
    test_fft.cpp
    test_transpose.cpp
    test_numpy_like.cpp
    test_linalg.cpp
)
target_compile_features(google-tests-sycl PUBLIC cxx_std_17)

find_package(AdaptiveCpp CONFIG REQUIRED)
add_sycl_to_target(TARGET google-tests-sycl SOURCES ${SOURCES})

# Compiler version
if(BACKEND STREQUAL "CUDA")
    # NVIDIA GPU (compiler AdaptiveCpp: with nvhpc backend)
    target_compile_options(google-tests-sycl PUBLIC -O3 -stdpar=gpu)
    target_link_options(google-tests-sycl PUBLIC -stdpar=gpu -cudalib=cufft,cublas,cusolver)
elseif(BACKEND STREQUAL "HIP")
    # AMD GPU (compiler AdaptiveCpp: with hipcc backend)
elseif(BACKEND STREQUAL "SYCL")
    # Intel GPU (compile dpcpp)
elseif(BACKEND STREQUAL "OPENMP")
    # CPU (compiler AdaptiveCpp: with nvhpc backend)
    target_compile_options(google-tests-sycl PUBLIC -O3 -stdpar=multicore -mp)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, HIP, SYCL and OPENMP must be On.")
endif()

target_link_libraries(google-tests-sycl PUBLIC math_lib GTest::gtest_main)

# Enable GoogleTest
include(GoogleTest)
gtest_discover_tests(google-tests-sycl PROPERTIES DISCOVERY_TIMEOUT 600)