add_executable(02_hello_world main.cpp)
target_link_libraries(02_hello_world PUBLIC Threads::Threads STDEXEC::stdexec CUDASchedulers)
target_compile_features(02_hello_world PUBLIC cxx_std_20)
