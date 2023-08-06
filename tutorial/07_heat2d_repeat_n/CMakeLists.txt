add_executable(07_heat2d_repeat_n heat2D.cpp)
target_link_libraries(07_heat2d_repeat_n PUBLIC Threads::Threads STDEXEC::stdexec STDEXEC::nvexec std::mdspan)
target_compile_features(07_heat2d_repeat_n PUBLIC cxx_std_20)
