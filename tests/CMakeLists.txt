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

add_executable(fdapde_stable_callers_test utility/stable_callers.cpp utility/assert_callers.cpp utility/numeric_callers.cpp)
fdapde_configure_target(fdapde_stable_callers_test)
target_link_libraries(fdapde_stable_callers_test PRIVATE GTest::gtest_main Eigen3::Eigen)
gtest_discover_tests(fdapde_stable_callers_test PROPERTIES TIMEOUT 30)

add_executable(fdapde_numeric_test utility/numeric.cpp utility/meta.cpp)
fdapde_configure_target(fdapde_numeric_test)
target_link_libraries(fdapde_numeric_test PRIVATE GTest::gtest_main)
gtest_discover_tests(fdapde_numeric_test PROPERTIES TIMEOUT 30)

add_library(fdapde_utility_header_check OBJECT public_headers/utility.cpp)
fdapde_configure_target(fdapde_utility_header_check)

add_library(fdapde_linear_algebra_header_check OBJECT public_headers/linear_algebra.cpp)
fdapde_configure_target(fdapde_linear_algebra_header_check)
target_link_libraries(fdapde_linear_algebra_header_check PRIVATE Eigen3::Eigen)

set(dense_sources
  linear_algebra/block.cpp
  linear_algebra/bool.cpp
  linear_algebra/diagonal.cpp
  linear_algebra/evd.cpp
  linear_algebra/historical_boolean.cpp
  linear_algebra/matrix.cpp
  linear_algebra/matrix_runtime_contracts.cpp
  linear_algebra/mdarray.cpp
  linear_algebra/native_dense.cpp
  linear_algebra/orthogonal.cpp
  linear_algebra/partial_piv_lu.cpp
  linear_algebra/permutation.cpp
  linear_algebra/qr.cpp
  linear_algebra/skew.cpp
  linear_algebra/symmetric.cpp
  linear_algebra/triangular.cpp
  linear_algebra/xpr.cpp
)
add_executable(fdapde_dense_test ${dense_sources})
fdapde_configure_target(fdapde_dense_test)
target_link_libraries(fdapde_dense_test PRIVATE GTest::gtest_main Eigen3::Eigen)
gtest_discover_tests(fdapde_dense_test PROPERTIES TIMEOUT 60)

add_executable(fdapde_dense_callers_test integration/dense_callers.cpp)
fdapde_configure_target(fdapde_dense_callers_test)
target_link_libraries(fdapde_dense_callers_test PRIVATE GTest::gtest_main Eigen3::Eigen)
gtest_discover_tests(fdapde_dense_callers_test PROPERTIES TIMEOUT 60)

# each rejected program must fail for its intended static assertion
function(fdapde_reject_program name diagnostic)
  unset(compiles CACHE)
  try_compile(compiles "${CMAKE_CURRENT_BINARY_DIR}/compile_fail/${name}"
    "${CMAKE_CURRENT_SOURCE_DIR}/compile_fail/${name}.cpp"
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${CMAKE_CURRENT_SOURCE_DIR}/.."
    LINK_LIBRARIES Eigen3::Eigen
    OUTPUT_VARIABLE output)
  file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/compile_fail/${name}.log" "${output}")
  if(compiles OR NOT output MATCHES "${diagnostic}")
    message(FATAL_ERROR "${name} did not fail with ${diagnostic}:\n${output}")
  endif()
endfunction()
fdapde_reject_program(col_major_skew_symmetric_matrix PACKED_COL_MAJOR_STRUCTURED_STORAGE_IS_NOT_SUPPORTED)
fdapde_reject_program(col_major_skew_symmetric_view PACKED_COL_MAJOR_STRUCTURED_STORAGE_IS_NOT_SUPPORTED)
fdapde_reject_program(col_major_symmetric_matrix PACKED_COL_MAJOR_STRUCTURED_STORAGE_IS_NOT_SUPPORTED)
fdapde_reject_program(col_major_symmetric_view PACKED_COL_MAJOR_STRUCTURED_STORAGE_IS_NOT_SUPPORTED)
fdapde_reject_program(col_major_triangular_matrix PACKED_COL_MAJOR_STRUCTURED_STORAGE_IS_NOT_SUPPORTED)
fdapde_reject_program(col_major_triangular_view PACKED_COL_MAJOR_STRUCTURED_STORAGE_IS_NOT_SUPPORTED)
fdapde_reject_program(oversized_boolean_block MATRIX_SIZE_EXCEEDS_SUPPORTED_RANGE)
fdapde_reject_program(oversized_boolean_matrix MATRIX_SIZE_EXCEEDS_SUPPORTED_RANGE)
fdapde_reject_program(oversized_evd EVD_DENSE_WORKSPACE_SIZE_EXCEEDS_SUPPORTED_RANGE)
fdapde_reject_program(oversized_matrix_block MATRIX_SIZE_EXCEEDS_SUPPORTED_RANGE)
fdapde_reject_program(oversized_mdarray_extents MDARRAY_STATIC_SIZE_EXCEEDS_SUPPORTED_RANGE)
fdapde_reject_program(oversized_reshape MATRIX_SIZE_EXCEEDS_SUPPORTED_RANGE)
fdapde_reject_program(reshape_static_size_mismatch INVALID_RESHAPE__DIFFERENT_STATIC_SIZE)
fdapde_reject_program(zero_static_boolean_block INVALID_BLOCK_DIMENSIONS)
fdapde_reject_program(zero_static_reshape INVALID_RESHAPE_DIMENSIONS)

add_library(fdapde_module_header_checks OBJECT
  public_headers/fields.cpp public_headers/geometry.cpp public_headers/io.cpp
  public_headers/finite_elements.cpp public_headers/splines.cpp
  public_headers/optimization.cpp public_headers/geoframe.cpp)
fdapde_configure_target(fdapde_module_header_checks)
target_link_libraries(fdapde_module_header_checks PRIVATE Eigen3::Eigen)
