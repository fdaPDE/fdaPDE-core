cmake_minimum_required(VERSION 3.14)

project(TEST)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_WARNINGS_FLAGS "-Wall -Wpedantic -Wextra -Werror")
set(CMAKE_CXX_FLAGS "-g -O2 -march=native -std=c++20 ${CMAKE_WARNINGS_FLAGS} -I${CMAKE_CURRENT_SOURCE_DIR}/../ -I${CMAKE_CURRENT_SOURCE_DIR}/../fdaPDE/core/")

find_package (Eigen3 3.4 REQUIRED NO_MODULE)

include_directories(../src/)
include_directories(../../fdaPDE/)

# enable testing functionality using gtest
include(FetchContent)
FetchContent_Declare(
  googletest
  # version 1.14.0
  URL https://github.com/google/googletest/archive/f8d7d77c06936315286eb55f8de22cd23c188571.zip
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()

add_executable(fdapde_test main.cpp)
target_link_libraries (fdapde_test Eigen3::Eigen)
target_link_libraries (fdapde_test gtest_main)

add_test(NAME fdapde_test
	 COMMAND $<TARGET_FILE:fdapde_test>
  )

include(GoogleTest)
gtest_discover_tests(fdapde_test)
