cmake_minimum_required(VERSION 3.20)
project(fdapde_examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)
include(CTest)
option(FDAPDE_EXAMPLES_WITH_FEM "Build spatial mesh examples requiring Eigen3" OFF)
option(FDAPDE_EXAMPLES_WITH_EIGEN "Build linear algebra examples requiring Eigen3" OFF)

function(fdapde_example source)
  get_filename_component(name "${source}" NAME_WE)
  add_executable(${name} ${source})
  target_include_directories(${name} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/..")
  target_link_libraries(${name} PRIVATE Threads::Threads)
  if(BUILD_TESTING)
    add_test(NAME example_${name} COMMAND ${name})
    set_tests_properties(example_${name} PROPERTIES TIMEOUT 120)
  endif()
endfunction()

foreach(example IN ITEMS
    linear_algebra/01_dense_storage_and_views.cpp
    linear_algebra/02_dense_expressions_and_products.cpp
    linear_algebra/03_diagonal_and_triangular_solve.cpp
    linear_algebra/04_symmetric_and_skew_storage.cpp
    linear_algebra/05_orthogonal_permutation_and_rotation.cpp
    linear_algebra/06_dense_factorizations.cpp
    linear_algebra/07_spd_spectral_and_caches.cpp
    linear_algebra/08_boolean_and_multidimensional_storage.cpp
    linear_algebra/09_matrix_batch_arithmetic.cpp
    linear_algebra/10_spd_batch_operations.cpp
    manifold/01_spd_metrics_and_means.cpp
    manifold/02_so_maps_and_branches.cpp
    manifold/03_so_product_optimization.cpp
    geometric_finite_elements/01_so_p1_interpolation.cpp
    geometric_finite_elements/02_cheeger_p1_interpolation.cpp
    geometric_finite_elements/03_spd_discrete_tension.cpp
    benchmarks/01_matrix_function_caches.cpp
    benchmarks/02_spd_log_euclidean_interpolation.cpp
    benchmarks/03_spd_airm_interpolation.cpp)
  fdapde_example(${example})
endforeach()

if(FDAPDE_EXAMPLES_WITH_EIGEN OR FDAPDE_EXAMPLES_WITH_FEM)
  find_package(Eigen3 3.4 REQUIRED NO_MODULE)
endif()

if(FDAPDE_EXAMPLES_WITH_EIGEN)
  fdapde_example(linear_algebra/11_eigen_sparse_blocks_and_products.cpp)
  target_link_libraries(11_eigen_sparse_blocks_and_products PRIVATE Eigen3::Eigen)
endif()

if(FDAPDE_EXAMPLES_WITH_FEM)
  foreach(example IN ITEMS geometric_finite_elements/04_spd_p1_field.cpp
                           geometric_finite_elements/05_spd_field_estimation.cpp
                           geometric_discretization/01_spd_spline_field.cpp)
    fdapde_example(${example})
    get_filename_component(name "${example}" NAME_WE)
    target_link_libraries(${name} PRIVATE Eigen3::Eigen)
  endforeach()
endif()
