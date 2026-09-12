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

#include <Eigen/LU>

// compares restarted native GMRES with an independent direct solver on nonsymmetric systems
TEST(linear_algebra, gmres_eigen_lu_oracle) {
    using matrix_type = fdapde::Matrix<double, fdapde::Dynamic, fdapde::Dynamic>;
    using vector_type = fdapde::Vector<double, fdapde::Dynamic>;
    for (int n : {3, 7, 12}) {
        matrix_type matrix(n, n);
        vector_type rhs(n);
        Eigen::MatrixXd reference(n, n);
        Eigen::VectorXd reference_rhs(n);
        for (int i = 0; i < n; ++i) {
            rhs[i] = reference_rhs[i] = (i % 3) - 0.5;
            for (int j = 0; j < n; ++j) {
                const double value = i == j ? n + 2.0 : ((3 * i + j) % 7 - 3) * 0.1;
                matrix(i, j) = reference(i, j) = value;
            }
        }
        const Eigen::VectorXd expected = reference.partialPivLu().solve(reference_rhs);
        fdapde::GMRES solver(matrix, fdapde::DiagonalPreconditioner<matrix_type> {}, 100, 2, 1e-12);
        const auto solution = solver.solve(rhs);
        // restart length two reaches the requested tolerance for each diagonally dominant system
        ASSERT_TRUE(solver.converged());
        for (int i = 0; i < n; ++i) {
            // each native coefficient agrees with the independent pivoted LU solution
            EXPECT_NEAR(solution[i], expected[i], 1e-11);
        }
        // recomputing the physical residual independently confirms the returned solution
        EXPECT_LT((matrix * solution - rhs).norm() / rhs.norm(), 1e-11);
    }
}
