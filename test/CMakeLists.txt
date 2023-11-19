cmake_minimum_required(VERSION 3.14)

project(TEST)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS "-O2 -march=x86-64 -std=c++17 -I${CMAKE_CURRENT_SOURCE_DIR}/../ -g")

find_package (Eigen3 3.3 REQUIRED NO_MODULE)

include_directories(../src/)
include_directories(../../fdaPDE/)

# enable testing functionality using gtest
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/609281088cfefc76f9d0ce82e1ff6c30cc3591e5.zip
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
