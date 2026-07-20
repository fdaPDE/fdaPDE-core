cmake_minimum_required(VERSION 3.14)

project(TEST LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(WARNING_FLAGS -Wall -Wextra -Wpedantic -Werror)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/f8d7d77c06936315286eb55f8de22cd23c188571.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

if(APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_definitions(gtest PRIVATE _Static_assert=static_assert)
endif()

add_executable(fdapde_test main.cpp linear_algebra/native_dense.cpp linear_algebra/native_structured.cpp)
target_compile_options(fdapde_test PRIVATE ${WARNING_FLAGS})
target_include_directories(fdapde_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(fdapde_test PRIVATE Eigen3::Eigen gtest_main)

foreach(module IN ITEMS core linear_algebra optimization)
  add_library(fdapde_${module}_header_check OBJECT public_headers/${module}.cpp)
  target_compile_options(fdapde_${module}_header_check PRIVATE ${WARNING_FLAGS})
  target_include_directories(fdapde_${module}_header_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
  target_link_libraries(fdapde_${module}_header_check PRIVATE Eigen3::Eigen)
endforeach()

enable_testing()
include(GoogleTest)
gtest_discover_tests(fdapde_test)
