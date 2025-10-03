cmake_minimum_required(VERSION 3.14)

project(TEST LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Warnings
set(WARNING_FLAGS -Wall -Wextra -Wpedantic -Werror)

# Dependencies
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/f8d7d77c06936315286eb55f8de22cd23c188571.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# Main test target
add_executable(fdapde_test main.cpp)

target_compile_options(fdapde_test PRIVATE ${WARNING_FLAGS})
target_include_directories(fdapde_test
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../
        ${CMAKE_CURRENT_SOURCE_DIR}/../fdaPDE/core/
        ${CMAKE_CURRENT_SOURCE_DIR}/../src/
        ${CMAKE_CURRENT_SOURCE_DIR}/../../fdaPDE/
)

target_link_libraries(fdapde_test
    PRIVATE
        Eigen3::Eigen
        gtest_main
)

# Testing setup
enable_testing()
include(GoogleTest)
gtest_discover_tests(fdapde_test)
