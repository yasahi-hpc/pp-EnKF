add_executable(01_hello_world main.cpp)
target_link_libraries(01_hello_world PUBLIC Threads::Threads STDEXEC::stdexec)
