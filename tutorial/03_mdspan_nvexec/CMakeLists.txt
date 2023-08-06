add_executable(03_mdspan_test main.cpp)
target_link_libraries(03_mdspan_test PUBLIC Threads::Threads STDEXEC::stdexec STDEXEC::nvexec std::mdspan)
target_compile_features(03_mdspan_test PUBLIC cxx_std_20)
