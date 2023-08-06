set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")

if(BACKEND STREQUAL "CUDA")
    target_compile_definitions(math_lib INTERFACE ENABLE_CUDA ENABLE_STDPAR)
    target_compile_options(math_lib INTERFACE -O3 -cudalib=cufft,cublas,curand)
elseif(BACKEND STREQUAL "OPENMP")
    find_package(OpenMP REQUIRED)
    find_package(FFTW REQUIRED COMPONENTS DOUBLE_LIB)
    target_link_libraries(math_lib INTERFACE OpenMP::OpenMP_CXX fftw3 eigen)
    target_compile_definitions(math_lib INTERFACE ENABLE_OPENMP ENABLE_STDPAR)
    target_compile_options(math_lib INTERFACE -O3)
    if(${CMAKE_CXX_COMPILER_ID} STREQUAL NVHPC)
        # with NVHPC, Eigen warns a lot. Need to suppress warnings in order to perform build testing on github.
        target_compile_options(math_lib INTERFACE --diag_suppress=subscript_out_of_range,integer_sign_change,incompatible_vectors_conversion,code_is_unreachable)
    endif()
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, and OPENMP must be On.")
endif()

# Compiler versions
target_link_libraries(math_lib INTERFACE std::mdspan)
target_compile_features(math_lib INTERFACE cxx_std_20)
