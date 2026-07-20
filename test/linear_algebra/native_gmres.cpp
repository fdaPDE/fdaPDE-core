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
#include <gtest/gtest.h>

#include <cmath>

namespace {

namespace native = fdapde::linalg;

TEST(NativeGMRES, IdentityAndDiagonalPreconditioningConverge) {
    native::Matrix<double, 2, 2> matrix({4.0, 1.0, 1.0, 3.0});
    const native::Vector<double, 2> rhs({1.0, 2.0});

    using matrix_type = decltype(matrix);
    using identity_type = native::IdentityPreconditioner<matrix_type>;
    native::GMRES identity_solver(matrix, identity_type {}, 20, 2, 1.0e-12);
    const auto identity_solution = identity_solver.solve(rhs);
    EXPECT_TRUE(identity_solver.converged());
    EXPECT_LE(identity_solver.iterations(), 2);
    EXPECT_LT((matrix * identity_solution - rhs).norm(), 1.0e-10);

    using diagonal_type = native::DiagonalPreconditioner<matrix_type>;
    native::GMRES diagonal_solver(matrix, diagonal_type {}, 20, 2, 1.0e-12);
    const auto diagonal_solution = diagonal_solver.solve(rhs);
    EXPECT_TRUE(diagonal_solver.converged());
    EXPECT_LE(diagonal_solver.iterations(), 2);
    EXPECT_LT((matrix * diagonal_solution - rhs).norm(), 1.0e-10);
}

TEST(NativeGMRES, RestartOneConvergesAndHonorsInitialGuess) {
    const native::Matrix<double, 3, 3> matrix({4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0});
    const native::Vector<double, 3> expected({1.0, -2.0, 0.5});
    const native::Vector<double, 3> rhs(matrix * expected);
    using matrix_type = decltype(matrix);
    using preconditioner_type = native::DiagonalPreconditioner<matrix_type>;

    native::GMRES solver(matrix, preconditioner_type {}, 500, 1, 1.0e-10);
    const auto solution = solver.solve(rhs);
    EXPECT_TRUE(solver.converged());
    EXPECT_GT(solver.iterations(), 1);
    EXPECT_LT((matrix * solution - rhs).norm() / rhs.norm(), 1.0e-9);

    native::GMRES exact_guess_solver(matrix, preconditioner_type {}, 10, 1, 1.0e-12);
    const auto exact_solution = exact_guess_solver.solve(rhs, expected);
    EXPECT_TRUE(exact_guess_solver.converged());
    EXPECT_EQ(exact_guess_solver.iterations(), 0);
    EXPECT_EQ(exact_solution, expected);
}

TEST(NativeGMRES, DynamicSystemAndZeroRightHandSide) {
    native::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> matrix(3, 3);
    matrix(0, 0) = 3.0;
    matrix(0, 1) = -1.0;
    matrix(0, 2) = 0.0;
    matrix(1, 0) = 2.0;
    matrix(1, 1) = 4.0;
    matrix(1, 2) = 1.0;
    matrix(2, 0) = 0.0;
    matrix(2, 1) = 1.0;
    matrix(2, 2) = 2.0;
    const native::Vector<double, fdapde::Dynamic> expected(std::vector<double> {2.0, -1.0, 3.0});
    const native::Vector<double, fdapde::Dynamic> rhs(matrix * expected);
    using matrix_type = decltype(matrix);
    using preconditioner_type = native::IdentityPreconditioner<matrix_type>;
    native::GMRES solver(matrix, preconditioner_type {}, 50, 2, 1.0e-12);
    const auto solution = solver.solve(rhs);
    EXPECT_TRUE(solver.converged());
    EXPECT_LT((matrix * solution - rhs).norm() / rhs.norm(), 1.0e-10);

    const native::Vector<double, fdapde::Dynamic> zero(native::Vector<double, fdapde::Dynamic>::Zero(3));
    native::GMRES zero_solver(matrix, preconditioner_type {}, 10, 2, 1.0e-12);
    const auto zero_solution = zero_solver.solve(zero);
    EXPECT_TRUE(zero_solver.converged());
    EXPECT_EQ(zero_solver.iterations(), 0);
    EXPECT_EQ(zero_solution, zero);
}

TEST(NativeGMRES, OwnsCoefficientMatrixAndReportsBreakdown) {
    native::Matrix<double, 2, 2> matrix({2.0, 0.0, 0.0, 4.0});
    const native::Vector<double, 2> rhs({2.0, 8.0});
    using matrix_type = decltype(matrix);
    using preconditioner_type = native::IdentityPreconditioner<matrix_type>;
    native::GMRES solver(matrix, preconditioner_type {}, 10, 2, 1.0e-12);
    matrix(0, 0) = 100.0;
    const auto solution = solver.solve(rhs);
    EXPECT_TRUE(solver.converged());
    EXPECT_NEAR(solution[0], 1.0, 1.0e-12);
    EXPECT_NEAR(solution[1], 2.0, 1.0e-12);

    const native::Matrix<double, 2, 2> zero(native::Matrix<double, 2, 2>::Zero());
    native::GMRES breakdown_solver(zero, preconditioner_type {}, 5, 2, 1.0e-12);
    const auto breakdown_solution = breakdown_solver.solve(rhs);
    EXPECT_FALSE(breakdown_solver.converged());
    EXPECT_EQ(breakdown_solver.iterations(), 0);
    EXPECT_TRUE(std::isfinite(breakdown_solver.residual()));
    EXPECT_EQ(breakdown_solution, (native::Vector<double, 2>({0.0, 0.0})));
}

TEST(NativeGMRES, DoesNotMistakeAnOverflowingReferenceNormForConvergence) {
    const native::Matrix<double, 2, 2> identity = native::IdentityMatrix<double, 2, 2>();
    const native::Vector<double, 2> rhs({1.4e308, 1.4e308});
    const native::Vector<double, 2> initial({1.4e308, 0.0});
    using matrix_type = decltype(identity);
    using preconditioner_type = native::IdentityPreconditioner<matrix_type>;
    native::GMRES zero_guess_solver(identity, preconditioner_type {}, 4, 2, 1.0e-12);
    const auto zero_guess_solution = zero_guess_solver.solve(rhs);
    EXPECT_TRUE(zero_guess_solver.converged());
    EXPECT_GT(zero_guess_solver.iterations(), 0);
    EXPECT_NEAR(zero_guess_solution[0] / rhs[0], 1.0, 1.0e-12);
    EXPECT_NEAR(zero_guess_solution[1] / rhs[1], 1.0, 1.0e-12);

    native::GMRES solver(identity, preconditioner_type {}, 4, 2, 1.0e-12);
    const auto solution = solver.solve(rhs, initial);
    EXPECT_TRUE(solver.converged());
    EXPECT_GT(solver.iterations(), 0);
    EXPECT_EQ(solution, rhs);
}

}   // namespace
