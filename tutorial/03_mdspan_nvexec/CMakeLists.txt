add_executable(03_mdspan_test main.cpp)
target_link_libraries(03_mdspan_test PUBLIC Threads::Threads STDEXEC::stdexec CUDASchedulers)
target_compile_options(03_mdspan_test PUBLIC -O3 -stdpar=gpu)
target_link_options(03_mdspan_test PUBLIC -O3 -stdpar=gpu)
