add_executable(06_sender_from_function heat2D.cpp)
target_link_libraries(06_sender_from_function PUBLIC Threads::Threads STDEXEC::stdexec STDEXEC::nvexec)
target_compile_features(06_sender_from_function PUBLIC cxx_std_20)
