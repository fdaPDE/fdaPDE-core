// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#include <gtest/gtest.h>   // testing framework
// include eigen now to avoid possible linking errors
#include <Eigen/Dense>
#include <Eigen/Sparse>

// utils
#include "src/scalar_field_test.cpp"
#include "src/vector_field_test.cpp"
#include "src/matrix_field_test.cpp"
#include "src/type_erasure_test.cpp"
// mesh
#include "src/element_test.cpp"
#include "src/mesh_test.cpp"
#include "src/point_location_test.cpp"
// linear_algebra
#include "src/kronecker_product_test.cpp"
#include "src/vector_space_test.cpp"
#include "src/binary_matrix_test.cpp"
// finite_elements
#include "src/fem_operators_test.cpp"
#include "src/fem_pde_test.cpp"
#include "src/integration_test.cpp"
#include "src/lagrangian_basis_test.cpp"
// optimization
#include "src/optimization_test.cpp"
//splines
#include "src/spline_test.cpp"

int main(int argc, char** argv) {
    // start testing
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
