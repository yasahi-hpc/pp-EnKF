add_executable(02_hello_world main.cpp)
target_link_libraries(02_hello_world PUBLIC Threads::Threads STDEXEC::stdexec CUDASchedulers)
#target_compile_options(02_hello_world PUBLIC -O3 -stdpar=gpu)
#target_link_options(02_hello_world PUBLIC -O3 -stdpar=gpu)
