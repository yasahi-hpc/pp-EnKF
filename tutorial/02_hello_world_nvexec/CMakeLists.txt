add_executable(02_hello_world main.cpp)
target_link_libraries(02_hello_world PUBLIC Threads::Threads STDEXEC::stdexec STDEXEC::nvexec)
target_compile_features(02_hello_world PUBLIC cuda_std_20)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
