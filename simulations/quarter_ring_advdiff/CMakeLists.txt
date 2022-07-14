cmake_minimum_required(VERSION 3.14)

# Project name
project(TEST)

# Set C++ standard to C++20
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Enable warnings and optimizations
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "-g -O2 -Wno-psabi -march=native")


# Find Eigen3
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

# Include directories
include_directories(../../fdaPDE)

# Add the executable
add_executable(AdvDiffRing advdiff_ring.cpp)

# Link libraries
target_link_libraries(AdvDiffRing PRIVATE Eigen3::Eigen)
