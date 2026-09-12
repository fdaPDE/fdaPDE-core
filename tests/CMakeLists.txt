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
option(FDAPDE_NATIVE_ONLY "Build native dense/SPD geometry checks with Eigen unavailable" OFF)
if(FDAPDE_NATIVE_ONLY)
  include(CheckIncludeFileCXX)
  unset(FDAPDE_EIGEN_DISCOVERABLE CACHE)
  check_include_file_cxx("Eigen/Eigen" FDAPDE_EIGEN_DISCOVERABLE)
  if(FDAPDE_EIGEN_DISCOVERABLE)
    message(FATAL_ERROR "The native-only lane requires Eigen to be unavailable")
  endif()
else()
  find_package(Eigen3 3.4 REQUIRED NO_MODULE)
endif()

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

if(NOT FDAPDE_NATIVE_ONLY)
  add_library(fdapde_core_header_check OBJECT public_headers/core.cpp)
  fdapde_configure_target(fdapde_core_header_check)
  target_link_libraries(fdapde_core_header_check PRIVATE Eigen3::Eigen)

  add_executable(fdapde_stable_callers_test utility/stable_callers.cpp utility/assert_callers.cpp utility/numeric_callers.cpp)
  fdapde_configure_target(fdapde_stable_callers_test)
  target_link_libraries(fdapde_stable_callers_test PRIVATE GTest::gtest_main Eigen3::Eigen)
  gtest_discover_tests(fdapde_stable_callers_test PROPERTIES TIMEOUT 30)
endif()

add_executable(fdapde_numeric_test utility/numeric.cpp utility/meta.cpp)
fdapde_configure_target(fdapde_numeric_test)
target_link_libraries(fdapde_numeric_test PRIVATE GTest::gtest_main)
gtest_discover_tests(fdapde_numeric_test PROPERTIES TIMEOUT 30)

add_library(fdapde_utility_header_check OBJECT public_headers/utility.cpp)
fdapde_configure_target(fdapde_utility_header_check)

if(NOT FDAPDE_NATIVE_ONLY)
  add_library(fdapde_linear_algebra_header_check OBJECT public_headers/linear_algebra.cpp)
  fdapde_configure_target(fdapde_linear_algebra_header_check)
  target_link_libraries(fdapde_linear_algebra_header_check PRIVATE Eigen3::Eigen)
endif()

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
target_link_libraries(fdapde_dense_test PRIVATE GTest::gtest_main)
gtest_discover_tests(fdapde_dense_test PROPERTIES TIMEOUT 60)

add_executable(fdapde_spd_test linear_algebra/spd.cpp linear_algebra/spectral.cpp)
fdapde_configure_target(fdapde_spd_test)
target_link_libraries(fdapde_spd_test PRIVATE GTest::gtest_main)
gtest_discover_tests(fdapde_spd_test PROPERTIES TIMEOUT 60)

if(NOT FDAPDE_NATIVE_ONLY)
  add_executable(fdapde_dense_callers_test integration/dense_callers.cpp)
  fdapde_configure_target(fdapde_dense_callers_test)
  target_link_libraries(fdapde_dense_callers_test PRIVATE GTest::gtest_main Eigen3::Eigen)
  gtest_discover_tests(fdapde_dense_callers_test PROPERTIES TIMEOUT 60)
endif()

# each rejected program must fail for its intended static assertion
function(fdapde_reject_program name diagnostic)
  unset(compiles CACHE)
  try_compile(compiles "${CMAKE_CURRENT_BINARY_DIR}/compile_fail/${name}"
    "${CMAKE_CURRENT_SOURCE_DIR}/compile_fail/${name}.cpp"
    CMAKE_FLAGS "-DINCLUDE_DIRECTORIES:STRING=${CMAKE_CURRENT_SOURCE_DIR}/.."
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

fdapde_reject_program(col_major_spd_matrix PACKED_COL_MAJOR_STRUCTURED_STORAGE_IS_NOT_SUPPORTED)
fdapde_reject_program(oversized_spd_matrix SPD_DENSE_WORKSPACE_SIZE_EXCEEDS_SUPPORTED_RANGE)
fdapde_reject_program(partial_dynamic_spd_matrix SPD_MATRICES_MUST_BE_SQUARE_AND_EITHER_FULLY_STATIC_OR_FULLY_DYNAMIC)
fdapde_reject_program(qualified_scalar_spd_matrix SPD_MATRICES_REQUIRE_AN_UNQUALIFIED_FLOATING_POINT_SCALAR)

if(NOT FDAPDE_NATIVE_ONLY)
  add_library(fdapde_module_header_checks OBJECT
    public_headers/fields.cpp public_headers/geometry.cpp public_headers/io.cpp
    public_headers/finite_elements.cpp public_headers/splines.cpp
    public_headers/optimization.cpp public_headers/geoframe.cpp)
  fdapde_configure_target(fdapde_module_header_checks)
  target_link_libraries(fdapde_module_header_checks PRIVATE Eigen3::Eigen)
endif()

add_executable(fdapde_spd_geometry_test
  manifold_optimization/log_euclidean_spd.cpp manifold_optimization/affine_invariant_spd.cpp
  manifold_optimization/spd_geometry.cpp)
fdapde_configure_target(fdapde_spd_geometry_test)
target_link_libraries(fdapde_spd_geometry_test PRIVATE GTest::gtest_main)
gtest_discover_tests(fdapde_spd_geometry_test PROPERTIES TIMEOUT 60)

# Focused public error/ownership contracts remain checked when debug assertions are disabled.
add_executable(fdapde_spd_geometry_contract_test manifold_optimization/contracts.cpp)
fdapde_configure_target(fdapde_spd_geometry_contract_test)
target_compile_definitions(fdapde_spd_geometry_contract_test PRIVATE FDAPDE_NO_DEBUG)
target_link_libraries(fdapde_spd_geometry_contract_test PRIVATE GTest::gtest_main)
gtest_discover_tests(fdapde_spd_geometry_contract_test PROPERTIES TIMEOUT 60)

add_library(fdapde_native_header_checks OBJECT
  public_headers/dense_linear_algebra.cpp public_headers/manifold_optimization.cpp)
fdapde_configure_target(fdapde_native_header_checks)

if(NOT FDAPDE_NATIVE_ONLY)
  add_executable(fdapde_geometry_header_order_test
    public_headers/manifold_then_linear_algebra.cpp public_headers/linear_algebra_then_manifold.cpp)
  fdapde_configure_target(fdapde_geometry_header_order_test)
  target_link_libraries(fdapde_geometry_header_order_test PRIVATE Eigen3::Eigen)
  add_test(NAME GeometryHeaderOrder COMMAND fdapde_geometry_header_order_test)
endif()

fdapde_reject_program(qualified_scalar_spd_geometry SPD_GEOMETRIES_REQUIRE_AN_UNQUALIFIED_FLOATING_POINT_SCALAR)
fdapde_reject_program(zero_order_spd_geometry INVALID_SPD_GEOMETRY_ORDER)
fdapde_reject_program(oversized_spd_geometry SPD_GEOMETRY_DENSE_WORKSPACE_SIZE_EXCEEDS_SUPPORTED_RANGE)

add_executable(fdapde_gmres_test linear_algebra/gmres.cpp linear_algebra/preconditioners.cpp)
fdapde_configure_target(fdapde_gmres_test)
target_link_libraries(fdapde_gmres_test PRIVATE GTest::gtest_main)
gtest_discover_tests(fdapde_gmres_test PROPERTIES TIMEOUT 60)

fdapde_reject_program(gmres_integer_scalar GMRES_REQUIRES_FLOATING_POINT_SCALARS)
fdapde_reject_program(gmres_matrix_rhs GMRES_REQUIRES_COLUMN_VECTOR_RIGHT_HAND_SIDE_AND_INITIAL_GUESS)
fdapde_reject_program(preconditioner_rectangular THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY)

if(NOT FDAPDE_NATIVE_ONLY)
  add_executable(fdapde_gmres_eigen_test linear_algebra/gmres_eigen.cpp)
  fdapde_configure_target(fdapde_gmres_eigen_test)
  target_link_libraries(fdapde_gmres_eigen_test PRIVATE GTest::gtest_main Eigen3::Eigen)
  gtest_discover_tests(fdapde_gmres_eigen_test PROPERTIES TIMEOUT 60)
endif()

add_executable(fdapde_sparse_test linear_algebra/sparse.cpp)
fdapde_configure_target(fdapde_sparse_test)
target_link_libraries(fdapde_sparse_test PRIVATE GTest::gtest_main)
gtest_discover_tests(fdapde_sparse_test PROPERTIES TIMEOUT 60)

add_library(fdapde_sparse_header_check OBJECT public_headers/sparse_linear_algebra.cpp)
fdapde_configure_target(fdapde_sparse_header_check)

if(NOT FDAPDE_NATIVE_ONLY)
  add_executable(fdapde_sparse_oracle_test linear_algebra/sparse_oracle.cpp)
  fdapde_configure_target(fdapde_sparse_oracle_test)
  target_link_libraries(fdapde_sparse_oracle_test PRIVATE GTest::gtest_main Eigen3::Eigen)
  gtest_discover_tests(fdapde_sparse_oracle_test PROPERTIES TIMEOUT 60)

  add_executable(fdapde_sparse_benchmark EXCLUDE_FROM_ALL benchmarks/sparse.cpp)
  fdapde_configure_target(fdapde_sparse_benchmark)
  target_link_libraries(fdapde_sparse_benchmark PRIVATE Eigen3::Eigen)
endif()

fdapde_reject_program(sparse_boolean "SparseMatrix<bool> is not supported")
