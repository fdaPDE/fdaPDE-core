cmake_minimum_required(VERSION 3.20)
project(fdapde_integration_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(POLICY CMP0135)
  cmake_policy(SET CMP0135 NEW)
endif()
include(FetchContent)
FetchContent_Declare(googletest
  URL https://github.com/google/googletest/archive/f8d7d77c06936315286eb55f8de22cd23c188571.zip)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

enable_testing()
include(GoogleTest)

function(fdapde_configure_target target)
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /WX)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic -Werror)
  endif()
endfunction()

add_executable(fdapde_assert_test utility/assert.cpp)
fdapde_configure_target(fdapde_assert_test)
target_link_libraries(fdapde_assert_test PRIVATE GTest::gtest_main)
gtest_discover_tests(fdapde_assert_test PROPERTIES TIMEOUT 30)

# a single focused test proves disabled macro argument erasure; the suite is not duplicated
add_executable(fdapde_assert_disabled_test utility/assert_disabled.cpp)
fdapde_configure_target(fdapde_assert_disabled_test)
target_compile_definitions(fdapde_assert_disabled_test PRIVATE FDAPDE_NO_DEBUG)
target_link_libraries(fdapde_assert_disabled_test PRIVATE GTest::gtest_main)
gtest_discover_tests(fdapde_assert_disabled_test PROPERTIES TIMEOUT 30)

add_library(fdapde_core_header_check OBJECT public_headers/core.cpp)
fdapde_configure_target(fdapde_core_header_check)
target_link_libraries(fdapde_core_header_check PRIVATE Eigen3::Eigen)

add_executable(fdapde_stable_callers_test utility/stable_callers.cpp utility/assert_callers.cpp)
fdapde_configure_target(fdapde_stable_callers_test)
target_link_libraries(fdapde_stable_callers_test PRIVATE GTest::gtest_main Eigen3::Eigen)
gtest_discover_tests(fdapde_stable_callers_test PROPERTIES TIMEOUT 30)
