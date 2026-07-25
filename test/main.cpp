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

TEST(PublicApiSmoke, NelderMeadDefaultsMatchTheDocumentedConfiguration) {
    const auto objective = [](const Eigen::Vector2d& x) -> double {
        return (x - Eigen::Vector2d(1.0, -2.0)).squaredNorm();
    };
    const Eigen::Vector2d x0(4.0, 3.0);

    fdapde::NelderMead<2> default_optimizer;
    fdapde::NelderMead<2> configured_optimizer(500, 1e-5, fdapde::random_seed);
    const Eigen::Vector2d default_optimum = default_optimizer.optimize(objective, x0);
    const Eigen::Vector2d configured_optimum = configured_optimizer.optimize(objective, x0);

    EXPECT_TRUE(default_optimum.isApprox(configured_optimum, 1e-12));
    EXPECT_DOUBLE_EQ(default_optimizer.value(), configured_optimizer.value());
    EXPECT_EQ(default_optimizer.n_iter(), configured_optimizer.n_iter());

    fdapde::NelderMead<2> iteration_limited_optimizer;
    const auto unbounded_objective = [](const Eigen::Vector2d& x) -> double { return x[0]; };
    iteration_limited_optimizer.optimize(unbounded_objective, Eigen::Vector2d::Zero());
    EXPECT_EQ(iteration_limited_optimizer.n_iter(), 500);
}

TEST(PublicApiSmoke, NelderMeadCanBeReused) {
    const auto objective = [](const Eigen::Vector2d& x) -> double {
        return (x - Eigen::Vector2d(-1.0, 2.0)).squaredNorm();
    };
    const Eigen::Vector2d x0(3.0, -4.0);

    fdapde::NelderMead<2> reused_optimizer(200, 1e-8, 7);
    const Eigen::Vector2d first_optimum = reused_optimizer.optimize(objective, x0);
    const int first_iterations = reused_optimizer.n_iter();
    const Eigen::Vector2d second_optimum = reused_optimizer.optimize(objective, x0);

    fdapde::NelderMead<2> fresh_optimizer(200, 1e-8, 7);
    const Eigen::Vector2d fresh_optimum = fresh_optimizer.optimize(objective, x0);

    EXPECT_TRUE(first_optimum.isApprox(fresh_optimum, 1e-12));
    EXPECT_TRUE(second_optimum.isApprox(fresh_optimum, 1e-12));
    EXPECT_EQ(first_iterations, fresh_optimizer.n_iter());
    EXPECT_EQ(reused_optimizer.n_iter(), fresh_optimizer.n_iter());
    EXPECT_DOUBLE_EQ(reused_optimizer.value(), fresh_optimizer.value());
}
