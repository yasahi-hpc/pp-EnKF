add_executable(heat3d-stdpar heat3D.cpp)

set(BACKEND AUTO CACHE STRING "CHOICE OF PARALLEL BACKEND")
if(BACKEND STREQUAL "CUDA")
    target_compile_options(heat3d-stdpar PUBLIC -stdpar=gpu)
    target_link_options(heat3d-stdpar PUBLIC -stdpar=gpu)
elseif(BACKEND STREQUAL "OPENMP")
    target_compile_options(heat3d-stdpar PUBLIC -stdpar=multicore)
else()
    message(FATAL_ERROR "No parallel backend specified. One of CUDA, and OPENMP must be On.")
endif()

target_compile_features(heat3d-stdpar PUBLIC cxx_std_20)
