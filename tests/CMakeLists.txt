cmake_minimum_required(VERSION 3.14)

project(TEST LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Warnings
set(WARNING_FLAGS -Wall -Wextra -Wpedantic -Werror)

# Dependencies
find_package(Eigen3 REQUIRED NO_MODULE)

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

# Execution topic tests share one suite and one worker pool.
find_package(Threads REQUIRED)
add_executable(fdapde_execution_test execution.cpp)
target_compile_options(fdapde_execution_test PRIVATE ${WARNING_FLAGS})
target_include_directories(fdapde_execution_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(fdapde_execution_test PRIVATE gtest_main Threads::Threads)
gtest_discover_tests(fdapde_execution_test PROPERTIES TIMEOUT 120)

add_executable(fdapde_execution_odr
  public_headers/execution_odr_a.cpp public_headers/execution_odr_b.cpp
  public_headers/execution_odr_main.cpp)
target_compile_options(fdapde_execution_odr PRIVATE ${WARNING_FLAGS})
target_include_directories(fdapde_execution_odr PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
target_link_libraries(fdapde_execution_odr PRIVATE Threads::Threads)
add_test(NAME execution_multi_translation_unit COMMAND fdapde_execution_odr)
set_tests_properties(execution_multi_translation_unit PROPERTIES TIMEOUT 30)
