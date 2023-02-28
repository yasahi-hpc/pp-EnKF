cmake_minimum_required(VERSION 3.23)
project(executors_tutorial LANGUAGES CXX)

# Add cmake helpers
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_SOURCE_DIR}/cmake")

# Find external libraries
find_package(Threads REQUIRED)
find_package(stdexec CONFIG)
if(NOT stdexec_FOUND)
    add_subdirectory(ext_lib/stdexec)
endif()

# Libraries
add_subdirectory(lib)

# Test
include(CTest)
if(BUILD_TESTING)
    find_package(GTest CONFIG)
    if(NOT GTest_FOUND)
        add_subdirectory(ext_lib/googletest)
    endif()
    add_subdirectory(tests)
endif()

# Tutorial
add_subdirectory(tutorial)

# Mini-apps
add_subdirectory(mini-apps)
