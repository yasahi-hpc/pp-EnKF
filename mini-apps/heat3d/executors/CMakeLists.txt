add_executable(heat3d heat3D.cpp)
target_link_libraries(heat3d PUBLIC Threads::Threads STDEXEC::stdexec STDEXEC::nvexec)
target_compile_features(heat3d PUBLIC cxx_std_20)
