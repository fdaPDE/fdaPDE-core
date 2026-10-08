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

#include <fdaPDE/dense_linear_algebra.h>
#include <gtest/gtest.h>

// disabled debug checks must preserve the orthogonalization performed by both input constructors
TEST(OrthogonalDisabled, OrthogonalizesVectorAndArrayInputs) {
    using namespace fdapde;
    const OrthogonalMatrix<double, Dynamic, Dynamic> from_vector(std::vector<double> {2., 1., 0., 3.}, orthogonalize);
    const double values[] = {2., 1., 0., 3.};
    const OrthogonalMatrix<double, 2, 2> from_array(values, orthogonalize);
    const Matrix<double, 2, 2> identity({1., 0., 0., 1.});
    // the vector input has full rank but is not orthogonal until the constructor executes its algorithm
    EXPECT_NEAR((from_vector.transpose() * from_vector - identity).norm(), 0., 1e-12);
    // the fixed array overload must execute the same algorithm independently of assertion mode
    EXPECT_NEAR((from_array.transpose() * from_array - identity).norm(), 0., 1e-12);
    // transpose-based inversion is valid only after the orthogonal invariant has been established
    EXPECT_NEAR((from_array.inv() * from_array - identity).norm(), 0., 1e-12);
}
