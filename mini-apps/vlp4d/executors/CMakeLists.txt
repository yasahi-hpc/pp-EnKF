file(GLOB SOURCES "*.cpp")
add_executable(vlp4d-executors ${SOURCES})

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")
  
# Compiler version
if(BACKEND STREQUAL "OPENMP")
    target_compile_options(vlp4d-executors PUBLIC ${OpenMP_CXX_FLAGS})
endif()

target_link_libraries(vlp4d-executors PUBLIC math_lib Threads::Threads STDEXEC::stdexec STDEXEC::nvexec)
target_compile_features(vlp4d-executors PUBLIC cxx_std_20)
