set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")

if(BACKEND STREQUAL "CUDA")
    target_compile_definitions(math_lib INTERFACE ENABLE_CUDA ENABLE_STDPAR)
    target_compile_options(math_lib INTERFACE -O3 -cudalib=cufft,cublas)
elseif(BACKEND STREQUAL "OPENMP")
    find_package(OpenMP REQUIRED)
    find_package(FFTW REQUIRED COMPONENTS DOUBLE_LIB)
    target_link_libraries(math_lib INTERFACE OpenMP::OpenMP_CXX fftw3)
    target_compile_definitions(math_lib INTERFACE ENABLE_OPENMP ENABLE_STDPAR)
    target_compile_options(math_lib INTERFACE -O3)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, and OPENMP must be On.")
endif()

# Compiler versions
target_compile_features(math_lib INTERFACE cxx_std_20)
