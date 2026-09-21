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

#include <fdaPDE/core.h>
#include <gtest/gtest.h>

// verifies the strict comparison helpers still support stable spatial range queries
TEST(NumericCallers, KDTreeIncludesBoundaryPoints) {
    Eigen::Matrix<double, 5, 2> points;
    points << 0, 0, 1, 1, 2, 2, 3, 3, 4, 4;
    fdapde::KDTree<2> tree(points);
    auto result = tree.range_search({Eigen::Vector2d(1, 1), Eigen::Vector2d(3, 3)});
    // compares the complete point-index set including both query boundaries
    EXPECT_EQ(result, (std::unordered_set<int> {1, 2, 3}));
}

// verifies shared traits still recognize Eigen vectors and preserve stable expression nesting
TEST(NumericCallers, EigenTraitsAndStableMatrixExpressions) {
    // verifies Eigen column vectors retain vector-like dispatch
    static_assert(fdapde::internals::is_vector_like_v<Eigen::VectorXd>);
    // verifies Eigen matrices are not mistaken for vectors
    static_assert(!fdapde::internals::is_vector_like_v<Eigen::MatrixXd>);
    fdapde::Matrix<double, 2, 2> matrix(std::vector<double> {1, 2, 3, 4});
    fdapde::Matrix<double, 2, 2> result = matrix + matrix;
    // checks that the first coefficient is read through the original owning matrix
    EXPECT_DOUBLE_EQ(result(0, 0), 2.0);
    // checks the last coefficient survives expression assignment
    EXPECT_DOUBLE_EQ(result(1, 1), 8.0);
}

// verifies finite-element assembly still reaches the shared combinatorial and expression utilities
TEST(NumericCallers, LinearFiniteElementMass) {
    fdapde::Triangulation<1, 1> domain(0.0, 1.0, 3);
    fdapde::FeSpace space(domain, fdapde::P1<1>);
    fdapde::TrialFunction f(space);
    fdapde::TestFunction v(space);
    auto mass = fdapde::integral(domain)(f * v).assemble();
    // checks that the mass matrix includes each degree of freedom
    EXPECT_EQ(mass.rows(), 3);
    // checks the integral of the partition of unity over the unit interval
    EXPECT_NEAR(mass.sum(), 1.0, 1e-12);
    // compares the middle basis-function mass with the analytic P1 value
    EXPECT_NEAR(mass.coeff(1, 1), 1.0 / 3.0, 1e-12);
}
