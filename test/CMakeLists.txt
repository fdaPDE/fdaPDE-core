cmake_minimum_required(VERSION 3.14)

project(TEST LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(WARNING_FLAGS -Wall -Wextra -Wpedantic -Werror)

find_package(Eigen3 3.4 REQUIRED NO_MODULE)

try_compile(
  FDAPDE_GFE_INTERNAL_INCLUDE_COMPILES
  ${CMAKE_CURRENT_BINARY_DIR}/geometric_finite_elements_internal_include
  SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/public_headers/geometric_finite_elements_internal_include.cpp
  CMAKE_FLAGS
    -DCMAKE_CXX_STANDARD=20
    -DCMAKE_CXX_STANDARD_REQUIRED=ON
    -DCMAKE_CXX_EXTENSIONS=OFF
  COMPILE_DEFINITIONS -I${CMAKE_CURRENT_SOURCE_DIR}/..
  LINK_LIBRARIES Eigen3::Eigen
  OUTPUT_VARIABLE FDAPDE_GFE_INTERNAL_INCLUDE_OUTPUT
)
if(FDAPDE_GFE_INTERNAL_INCLUDE_COMPILES)
  message(FATAL_ERROR "The internal GFE header compiled without the public module header.")
endif()
string(
  FIND
  "${FDAPDE_GFE_INTERNAL_INCLUDE_OUTPUT}"
  "Include fdaPDE/geometric_finite_elements.h instead of including internal headers directly."
  FDAPDE_GFE_INTERNAL_INCLUDE_ERROR
)
if(FDAPDE_GFE_INTERNAL_INCLUDE_ERROR EQUAL -1)
  message(FATAL_ERROR "The internal GFE header failed for an unexpected reason:\n${FDAPDE_GFE_INTERNAL_INCLUDE_OUTPUT}")
endif()

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

add_executable(
  fdapde_test
  main.cpp
  linear_algebra/native_dense.cpp
  linear_algebra/native_structured.cpp
  linear_algebra/native_skew.cpp
  linear_algebra/native_decompositions.cpp
  linear_algebra/native_gmres.cpp
  linear_algebra/native_mdarray.cpp
  linear_algebra/native_spd.cpp
  linear_algebra/native_matrix_batch.cpp
  manifold_optimization/manifold.cpp
  manifold_optimization/problem.cpp
  manifold_optimization/power_geometry.cpp
  manifold_optimization/steepest_descent.cpp
  manifold_optimization/trust_region.cpp
  manifold_optimization/weighted_karcher_mean.cpp
  manifold_optimization/log_euclidean_spd.cpp
  manifold_optimization/affine_invariant_spd.cpp
  manifold_optimization/spd_mean.cpp
  geometric_finite_elements/p1_value.cpp
  geometric_finite_elements/affine_invariant_p1_value.cpp
  src/dg_dof_test.cpp
  src/dg_expression_test.cpp
  src/dg_facet_coefficient_test.cpp
  src/dg_facet_test.cpp
  src/dg_high_order_quadrature_test.cpp
  src/dg_linear_facet_test.cpp
  src/dg_poisson_test.cpp
)
target_compile_options(fdapde_test PRIVATE ${WARNING_FLAGS})
target_include_directories(fdapde_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(fdapde_test PRIVATE Eigen3::Eigen gtest_main)

add_executable(fdapde_native_mdarray_no_debug linear_algebra/native_mdarray_no_debug.cpp)
target_compile_definitions(fdapde_native_mdarray_no_debug PRIVATE FDAPDE_NO_DEBUG)
target_compile_options(fdapde_native_mdarray_no_debug PRIVATE ${WARNING_FLAGS})
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(
    fdapde_native_mdarray_no_debug
    PRIVATE -Wno-error=unused-parameter -Wno-error=unused-local-typedef
  )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(
    fdapde_native_mdarray_no_debug
    PRIVATE -Wno-error=unused-parameter -Wno-error=unused-local-typedefs
  )
endif()
target_include_directories(fdapde_native_mdarray_no_debug PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(fdapde_native_mdarray_no_debug PRIVATE Eigen3::Eigen)

add_executable(fdapde_native_spd_no_debug linear_algebra/native_spd_no_debug.cpp)
target_compile_definitions(fdapde_native_spd_no_debug PRIVATE FDAPDE_NO_DEBUG)
target_compile_options(fdapde_native_spd_no_debug PRIVATE ${WARNING_FLAGS})
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(
    fdapde_native_spd_no_debug
    PRIVATE -Wno-error=unused-parameter -Wno-error=unused-local-typedef
  )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(
    fdapde_native_spd_no_debug
    PRIVATE -Wno-error=unused-parameter -Wno-error=unused-local-typedefs
  )
endif()
target_include_directories(fdapde_native_spd_no_debug PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(fdapde_native_spd_no_debug PRIVATE Eigen3::Eigen)

add_executable(fdapde_native_matrix_batch_no_debug linear_algebra/native_matrix_batch_no_debug.cpp)
target_compile_definitions(fdapde_native_matrix_batch_no_debug PRIVATE FDAPDE_NO_DEBUG)
target_compile_options(fdapde_native_matrix_batch_no_debug PRIVATE ${WARNING_FLAGS})
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(
    fdapde_native_matrix_batch_no_debug
    PRIVATE -Wno-error=unused-parameter -Wno-error=unused-local-typedef
  )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(
    fdapde_native_matrix_batch_no_debug
    PRIVATE -Wno-error=unused-parameter -Wno-error=unused-local-typedefs
  )
endif()
target_include_directories(fdapde_native_matrix_batch_no_debug PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(fdapde_native_matrix_batch_no_debug PRIVATE Eigen3::Eigen)

add_executable(
  fdapde_geometric_finite_elements_no_debug
  geometric_finite_elements/p1_value_no_debug.cpp
)
target_compile_definitions(fdapde_geometric_finite_elements_no_debug PRIVATE FDAPDE_NO_DEBUG)
target_compile_options(fdapde_geometric_finite_elements_no_debug PRIVATE ${WARNING_FLAGS})
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(
    fdapde_geometric_finite_elements_no_debug
    PRIVATE -Wno-error=unused-parameter -Wno-error=unused-local-typedef
  )
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  target_compile_options(
    fdapde_geometric_finite_elements_no_debug
    PRIVATE -Wno-error=unused-parameter -Wno-error=unused-local-typedefs
  )
endif()
target_include_directories(fdapde_geometric_finite_elements_no_debug PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(fdapde_geometric_finite_elements_no_debug PRIVATE Eigen3::Eigen)

add_executable(
  fdapde_execution_test
  execution/parallel_algorithms.cpp
  execution/runtime.cpp
  execution/task_graph.cpp
)
target_compile_options(fdapde_execution_test PRIVATE ${WARNING_FLAGS})
target_include_directories(fdapde_execution_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(fdapde_execution_test PRIVATE Eigen3::Eigen gtest_main)

add_executable(fdapde_execution_configuration_test execution/configuration.cpp)
target_compile_options(fdapde_execution_configuration_test PRIVATE ${WARNING_FLAGS})
target_include_directories(fdapde_execution_configuration_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(fdapde_execution_configuration_test PRIVATE Eigen3::Eigen gtest_main)

add_executable(fdapde_execution_queue_capacity_test execution/queue_capacity.cpp)
target_compile_options(fdapde_execution_queue_capacity_test PRIVATE ${WARNING_FLAGS})
target_include_directories(fdapde_execution_queue_capacity_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(fdapde_execution_queue_capacity_test PRIVATE Eigen3::Eigen gtest_main)

foreach(module IN ITEMS core linear_algebra optimization manifold_optimization geometric_finite_elements execution)
  add_library(fdapde_${module}_header_check OBJECT public_headers/${module}.cpp)
  target_compile_options(fdapde_${module}_header_check PRIVATE ${WARNING_FLAGS})
  target_include_directories(fdapde_${module}_header_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
  target_link_libraries(fdapde_${module}_header_check PRIVATE Eigen3::Eigen)
endforeach()

add_library(
  fdapde_execution_header_composition_check
  OBJECT
  public_headers/execution_then_linear_algebra.cpp
  public_headers/linear_algebra_then_execution.cpp
)
target_compile_options(fdapde_execution_header_composition_check PRIVATE ${WARNING_FLAGS})
target_include_directories(fdapde_execution_header_composition_check PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..)
target_link_libraries(fdapde_execution_header_composition_check PRIVATE Eigen3::Eigen)

add_library(
  fdapde_geometric_finite_elements_header_composition_check
  OBJECT
  public_headers/geometric_finite_elements_then_manifold_optimization.cpp
  public_headers/manifold_optimization_then_geometric_finite_elements.cpp
)
target_compile_options(fdapde_geometric_finite_elements_header_composition_check PRIVATE ${WARNING_FLAGS})
target_include_directories(
  fdapde_geometric_finite_elements_header_composition_check
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/..
)
target_link_libraries(fdapde_geometric_finite_elements_header_composition_check PRIVATE Eigen3::Eigen)

enable_testing()
include(GoogleTest)
gtest_discover_tests(fdapde_test)
gtest_discover_tests(fdapde_execution_test)
gtest_discover_tests(fdapde_execution_configuration_test)
gtest_discover_tests(fdapde_execution_queue_capacity_test PROPERTIES TIMEOUT 30)
add_test(NAME NativeMdArray.NoDebugContracts COMMAND fdapde_native_mdarray_no_debug)
add_test(NAME NativeSPD.NoDebugContracts COMMAND fdapde_native_spd_no_debug)
add_test(NAME NativeMatrixBatchView.NoDebugContracts COMMAND fdapde_native_matrix_batch_no_debug)
add_test(
  NAME GeometricFiniteElements.NoDebugContracts
  COMMAND fdapde_geometric_finite_elements_no_debug
)
