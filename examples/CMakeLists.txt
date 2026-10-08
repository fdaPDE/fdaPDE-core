cmake_minimum_required(VERSION 3.20)
project(fdapde_examples LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Threads REQUIRED)
include(CTest)
option(FDAPDE_EXAMPLES_WITH_FEM "Build spatial mesh examples requiring Eigen3" OFF)

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
    linear_algebra/matrix_batch_arithmetic.cpp
    linear_algebra/spd_batch_operations.cpp
    manifold/spd_metrics_and_means.cpp
    manifold/so_maps_and_branches.cpp
    manifold/so_product_optimization.cpp
    geometric_finite_elements/so_p1_interpolation.cpp
    geometric_finite_elements/cheeger_p1_interpolation.cpp
    geometric_finite_elements/spd_discrete_tension.cpp
    benchmarks/spd_airm_interpolation.cpp
    benchmarks/spd_log_euclidean_interpolation.cpp
    benchmarks/matrix_function_caches.cpp)
  fdapde_example(${example})
endforeach()

if(FDAPDE_EXAMPLES_WITH_FEM)
  find_package(Eigen3 3.4 REQUIRED NO_MODULE)
  foreach(example IN ITEMS geometric_finite_elements/spd_p1_field.cpp
                           geometric_finite_elements/spd_field_estimation.cpp)
    fdapde_example(${example})
    get_filename_component(name "${example}" NAME_WE)
    target_link_libraries(${name} PRIVATE Eigen3::Eigen)
  endforeach()
endif()
