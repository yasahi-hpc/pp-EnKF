cmake_minimum_required(VERSION 3.23)
project(executors_tutorial LANGUAGES CXX)

# Find libraries
find_package(Threads REQUIRED)
find_package(stdexec CONFIG)
if(NOT stdexec_FOUND)
    add_subdirectory(ext_lib/stdexec)
endif()

# Tutorial
add_subdirectory(tutorial)
