add_executable(heat3d-exec heat3D.cpp)
target_link_libraries(heat3d-exec PUBLIC Threads::Threads STDEXEC::stdexec STDEXEC::nvexec)
target_compile_features(heat3d-exec PUBLIC cxx_std_20)
