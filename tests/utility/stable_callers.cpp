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

// verifies migrated checks still support stable multidimensional grid evaluation
TEST(StableCallers, GridSearchKeepsMatrixLayoutAndRejectsWrongDimension) {
    fdapde::GridSearch<2> search;
    Eigen::Matrix<double, 3, 2> grid;
    grid << 4, 2, 1, 3, 2, 1;
    auto objective = [](const Eigen::Vector2d& point) -> double { return point.squaredNorm(); };
    auto optimum = search.optimize(objective, grid);
    // compares both coordinates with the unique minimum row in the input grid
    EXPECT_TRUE(optimum.isApprox(Eigen::Vector2d(2, 1)));
    // checks the numerical objective at the selected row
    EXPECT_DOUBLE_EQ(search.value(), 5.0);
    // instantiates a migrated constructor failure through the public stable API
    EXPECT_THROW(fdapde::GridSearch<2>(3), std::invalid_argument);
}

// verifies geometry and its binary markers instantiate with the migrated assertion signature
TEST(StableCallers, IntervalPreservesNodesAndRejectsEmptyInput) {
    fdapde::Triangulation<1, 1> interval(0.0, 1.0, 5);
    // checks that construction retained all requested nodes
    EXPECT_EQ(interval.n_nodes(), 5);
    // checks that consecutive nodes form four intervals
    EXPECT_EQ(interval.n_cells(), 4);
    Eigen::VectorXd empty;
    // an empty coordinate vector must fail the migrated debug precondition
    EXPECT_THROW((fdapde::Triangulation<1, 1>(empty)), std::invalid_argument);
}
