file(GLOB SOURCES "*.cpp")
add_executable(lbm2d-letkf-executors ${SOURCES})

find_package(MPI REQUIRED COMPONENTS CXX)
set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")

# Compiler version
if(BACKEND STREQUAL "OPENMP")
    target_compile_options(lbm2d-letkf-executors PUBLIC ${OpenMP_CXX_FLAGS})
endif()

target_link_libraries(lbm2d-letkf-executors PUBLIC math_lib Threads::Threads STDEXEC::stdexec STDEXEC::nvexec MPI::MPI_CXX nlohmann_json::nlohmann_json)
target_compile_features(lbm2d-letkf-executors PUBLIC cxx_std_20)
