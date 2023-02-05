add_executable(05_heat_test heat2D.cpp)
target_link_libraries(05_heat_test PUBLIC Threads::Threads STDEXEC::stdexec STDEXEC::nvexec)
target_compile_features(05_heat_test PUBLIC cxx_std_20)
