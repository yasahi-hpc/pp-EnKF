add_executable(heat3d-sycl heat3D.cpp)

find_package(AdaptiveCpp CONFIG REQUIRED)
add_sycl_to_target(TARGET heat3d-sycl SOURCES heat3D.cpp)

target_link_libraries(heat3d-sycl PUBLIC std::mdspan)
target_compile_features(heat3d-sycl PUBLIC cxx_std_17)