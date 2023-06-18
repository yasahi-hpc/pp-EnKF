file(GLOB SOURCES "*.cpp")
add_executable(lbm2d-letkf-stdpar ${SOURCES})

find_package(MPI REQUIRED COMPONENTS CXX)

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")
if(BACKEND STREQUAL "CUDA")
    target_compile_options(lbm2d-letkf-stdpar PUBLIC -stdpar=gpu)
    target_link_options(lbm2d-letkf-stdpar PUBLIC -stdpar=gpu -cudalib=cusolver,cublas,curand)
elseif(BACKEND STREQUAL "OPENMP")
    target_compile_options(lbm2d-letkf-stdpar PUBLIC -stdpar=multicore)
    target_compile_definitions(lbm2d-letkf-stdpar PUBLIC SIMD)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, and OPENMP must be On.")
endif()

# Enable macro
target_compile_definitions(lbm2d-letkf-stdpar PUBLIC $<$<BOOL:${USE_SINGLE_PRECISION}>: USE_SINGLE_PRECISION>)
target_link_libraries(lbm2d-letkf-stdpar PUBLIC MPI::MPI_CXX math_lib nlohmann_json::nlohmann_json)
