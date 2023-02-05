add_executable(04_stream_test stream.cpp)
target_link_libraries(04_stream_test PUBLIC Threads::Threads STDEXEC::stdexec STDEXEC::nvexec)
target_compile_features(04_stream_test PUBLIC cxx_std_20)
