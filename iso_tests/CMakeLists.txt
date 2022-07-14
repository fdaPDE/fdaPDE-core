cmake_minimum_required(VERSION 3.14)

# Project name
project(TEST)

# Set C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable warnings and optimizations
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "-g -O2 -march=native")

# Enable testing functionality using gtest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
  DOWNLOAD_EXTRACT_TIMESTAMP true
)

# Make sure googletest is available
FetchContent_MakeAvailable(googletest)

# Find Eigen3
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# Include directories
include_directories(../fdaPDE)

# Add the executable
add_executable(IsoTest iso_test.cpp)

# Link libraries
target_link_libraries(IsoTest PRIVATE Eigen3::Eigen gtest gtest_main)

# Enable testing
enable_testing()
add_test(NAME IsoTest COMMAND $<TARGET_FILE:IsoTest>)
set_tests_properties(IsoTest PROPERTIES WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})

# Discover Google Tests
include(GoogleTest)
gtest_discover_tests(IsoTest)