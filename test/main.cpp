#include <gtest/gtest.h> // testing framework
// include eigen now to avoid possible linking errors
#include <Eigen/Dense>
#include <Eigen/Sparse>

// fields
#include "src/scalar_field_test.cpp"
#include "src/vector_field_test.cpp"
// mesh
#include "src/element_test.cpp"
#include "src/mesh_test.cpp"
#include "src/point_location_test.cpp"
// linear_algebra
#include "src/vector_space_test.cpp"
#include "src/kronecker_product_test.cpp"
// finite_elements
#include "src/integration_test.cpp"
#include "src/lagrangian_basis_test.cpp"
#include "src/fem_operators_test.cpp"
#include "src/fem_pde_test.cpp"
// optimization
#include "src/optimization_test.cpp"

// space-time test suites
// #include "core/SplineTest.cpp"

// #include "core/ThreadPoolTest.cpp"

int main(int argc, char **argv){
  // start testing
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
