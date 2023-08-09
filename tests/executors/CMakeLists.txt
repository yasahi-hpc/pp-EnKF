add_executable(google-tests-executors
    test_view.cpp
    test_fft.cpp
    test_transpose.cpp
    test_numpy_like.cpp
    test_linalg.cpp
)

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")
if(BACKEND STREQUAL "CUDA")
    target_compile_features(google-tests-executors PUBLIC cxx_std_20)
    target_link_libraries(google-tests-executors PUBLIC Threads::Threads STDEXEC::stdexec STDEXEC::nvexec)
elseif(BACKEND STREQUAL "HIP")
    target_compile_features(google-tests-executors PUBLIC cxx_std_20)
    target_compile_options(google-tests-executors PUBLIC -std=c++20)
elseif(BACKEND STREQUAL "OPENMP")
    target_compile_options(google-tests-executors PUBLIC ${OpenMP_CXX_FLAGS})
    target_compile_features(google-tests-executors PUBLIC cxx_std_20)
    target_link_libraries(google-tests-executors PUBLIC Threads::Threads STDEXEC::stdexec STDEXEC::nvexec)
endif()
target_link_libraries(google-tests-executors PUBLIC math_lib GTest::gtest_main)

# Enable GoogleTest
include(GoogleTest)
gtest_discover_tests(google-tests-executors PROPERTIES DISCOVERY_TIMEOUT 600)
