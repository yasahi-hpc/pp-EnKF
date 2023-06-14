cmake_minimum_required(VERSION 3.23)
project(executors_tutorial LANGUAGES CXX)

# Add cmake helpers
list(INSERT CMAKE_MODULE_PATH 0 "${CMAKE_SOURCE_DIR}/cmake")

# Keep BUILD_TESTING
set(PRIVATE_BUILD_TESTING ${BUILD_TESTING})

# Find external libraries
find_package(Threads REQUIRED)
find_package(stdexec CONFIG)

# In order to disable testing for stdexec
set(STDEXEC_BUILD_EXAMPLES OFF)
set(STDEXEC_BUILD_TESTS OFF)
set(BUILD_TESTING OFF)
if(NOT stdexec_FOUND)
    add_subdirectory(ext_lib/stdexec)
endif()

# Libraries
add_subdirectory(lib)
add_subdirectory(ext_lib/json)
add_subdirectory(ext_lib/eigen)

# Test
set(BUILD_TESTING ${PRIVATE_BUILD_TESTING})
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
