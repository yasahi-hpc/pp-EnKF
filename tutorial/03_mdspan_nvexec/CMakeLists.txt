add_executable(03_mdspan_test main.cpp)
target_link_libraries(03_mdspan_test PUBLIC Threads::Threads STDEXEC::stdexec CUDASchedulers)
