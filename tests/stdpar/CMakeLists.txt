add_executable(google-tests-stdpar
    test_view.cpp
    test_fft.cpp
    test_transpose.cpp
)
target_compile_features(google-tests-stdpar PUBLIC cxx_std_20)

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")

# Compiler version
if(BACKEND STREQUAL "CUDA")
    target_compile_options(google-tests-stdpar PUBLIC -O3 -stdpar=gpu)
    target_link_options(google-tests-stdpar PUBLIC -stdpar=gpu -cudalib=cufft,cublas)
elseif(BACKEND STREQUAL "OPENMP")
    target_compile_options(google-tests-stdpar PUBLIC -O3 -stdpar=multicore -mp)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, and OPENMP must be On.")
endif()

target_link_libraries(google-tests-stdpar PUBLIC math_lib GTest::gtest_main)

# Enable GoogleTest
include(GoogleTest)
gtest_discover_tests(google-tests-stdpar PROPERTIES DISCOVERY_TIMEOUT 600)
