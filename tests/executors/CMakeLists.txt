add_executable(google-tests-executors
    test_view.cpp
    test_fft.cpp
    test_transpose.cpp
)
target_compile_features(google-tests-executors PUBLIC cxx_std_20)

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")

# Compiler version
if(BACKEND STREQUAL "OPENMP")
    target_compile_options(google-tests-executors PUBLIC ${OpenMP_CXX_FLAGS})
endif()

target_link_libraries(google-tests-executors PUBLIC math_lib GTest::gtest_main Threads::Threads STDEXEC::stdexec STDEXEC::nvexec)

# Enable GoogleTest
include(GoogleTest)
gtest_discover_tests(google-tests-executors PROPERTIES DISCOVERY_TIMEOUT 600)
