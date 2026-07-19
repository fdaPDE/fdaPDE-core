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

#include <fdaPDE/linear_algebra.h>
#include <fdaPDE/optimization.h>

#include <gtest/gtest.h>

#include <array>

TEST(PublicApiSmoke, LegacyMatrixArithmetic) {
    const fdapde::Matrix<double, 2, 2> lhs(std::array<double, 4> {1.0, 2.0, 3.0, 4.0});
    const fdapde::Matrix<double, 2, 2> rhs(std::array<double, 4> {10.0, 20.0, 30.0, 40.0});
    const fdapde::Matrix<double, 2, 2> sum = lhs + rhs;

    EXPECT_DOUBLE_EQ(sum(0, 0), 11.0);
    EXPECT_DOUBLE_EQ(sum(0, 1), 22.0);
    EXPECT_DOUBLE_EQ(sum(1, 0), 33.0);
    EXPECT_DOUBLE_EQ(sum(1, 1), 44.0);
}

TEST(PublicApiSmoke, GridSearchHonorsEigenStorageOrder) {
    Eigen::Matrix<double, 4, 2, Eigen::RowMajor> row_major_grid;
    row_major_grid << -1.0, -1.0,
                       0.0,  0.0,
                       1.0,  1.0,
                       2.0,  2.0;

    const auto objective = [](const Eigen::Vector2d& x) -> double { return x.squaredNorm(); };
    const auto check_grid = [&objective](const auto& grid) {
        fdapde::GridSearch<2> optimizer;
        const auto optimum = optimizer.optimize(objective, grid);

        EXPECT_DOUBLE_EQ(optimum[0], 0.0);
        EXPECT_DOUBLE_EQ(optimum[1], 0.0);
        EXPECT_DOUBLE_EQ(optimizer.value(), 0.0);
        EXPECT_EQ(optimizer.values().size(), 4u);
    };

    check_grid(row_major_grid);
    const Eigen::Matrix<double, 4, 2, Eigen::ColMajor> column_major_grid = row_major_grid;
    check_grid(column_major_grid);
}
