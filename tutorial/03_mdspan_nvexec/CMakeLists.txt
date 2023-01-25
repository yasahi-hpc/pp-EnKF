add_executable(03_mdspan_test main.cpp)
target_link_libraries(03_mdspan_test PUBLIC Threads::Threads STDEXEC::stdexec STDEXEC::nvexec)
target_compile_features(03_mdspan_test PUBLIC cuda_std_20)
set_source_files_properties(main.cpp PROPERTIES LANGUAGE CUDA)
