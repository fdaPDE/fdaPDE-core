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
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using namespace fdapde;

template <typename XprType_> class RejectingPreconditioner {
    using XprType = std::decay_t<XprType_>;
   public:
    using Scalar = std::remove_cv_t<typename XprType::Scalar>;

    template <typename MatrixType> constexpr void compute(const MatrixExpr<MatrixType>&) { }

    template <typename RhsType> constexpr auto solve(const MatrixExpr<RhsType>& rhs) const {
        return Matrix<Scalar, RhsType::Rows, RhsType::Cols>(rhs);
    }

    constexpr bool valid() const { return false; }
};

template <typename MatrixType, typename SolutionType, typename RhsType>
void expect_small_residual(const MatrixType& matrix, const SolutionType& solution, const RhsType& rhs) {
    EXPECT_LT((matrix * solution - rhs).norm(), 1.0e-10);
}

template <int StorageOrder> void check_gmres_happy_paths() {
    using fixed_matrix = Matrix<double, 2, 2, StorageOrder>;
    using fixed_vector = Matrix<double, 2, 1, StorageOrder>;
    const fixed_matrix matrix({4.0, 1.0, 1.0, 3.0});
    const fixed_vector rhs(std::vector<double> {1.0, 2.0});

    using identity_type = IdentityPreconditioner<fixed_matrix>;
    GMRES identity_solver(matrix, identity_type {}, 20, 2, 1.0e-12);
    const auto identity_solution = identity_solver.solve(rhs);
    EXPECT_TRUE(identity_solver.initialized());
    EXPECT_TRUE(identity_solver.converged());
    EXPECT_LE(identity_solver.iterations(), 2);
    EXPECT_TRUE(std::isfinite(identity_solver.residual()));
    expect_small_residual(matrix, identity_solution, rhs);

    using diagonal_type = DiagonalPreconditioner<fixed_matrix>;
    GMRES diagonal_solver(matrix, diagonal_type {}, 20, 2, 1.0e-12);
    const auto diagonal_solution = diagonal_solver.solve(rhs);
    EXPECT_TRUE(diagonal_solver.converged());
    EXPECT_LE(diagonal_solver.iterations(), 2);
    expect_small_residual(matrix, diagonal_solution, rhs);

    const MatrixView<const double, 2, 2, StorageOrder> const_view(matrix.data());
    using const_view_type = decltype(const_view);
    GMRES const_view_solver(const_view, IdentityPreconditioner<const_view_type> {}, 20, 2, 1.0e-12);
    expect_small_residual(matrix, const_view_solver.solve(rhs), rhs);

    using partial_matrix = Matrix<double, Dynamic, 2, StorageOrder>;
    using partial_vector = Matrix<double, Dynamic, 1, StorageOrder>;
    const partial_matrix partial(matrix);
    const partial_vector partial_rhs(rhs);
    GMRES partial_solver(partial, IdentityPreconditioner<partial_matrix> {}, 20, 2, 1.0e-12);
    const auto partial_solution = partial_solver.solve(partial_rhs);
    EXPECT_TRUE(partial_solver.converged());
    EXPECT_LT((partial * partial_solution - partial_rhs).norm(), 1.0e-10);

    using partial_cols_matrix = Matrix<double, 2, Dynamic, StorageOrder>;
    const partial_cols_matrix partial_cols(matrix);
    GMRES partial_cols_solver(partial_cols, IdentityPreconditioner<partial_cols_matrix> {}, 20, 2, 1.0e-12);
    const auto partial_cols_solution = partial_cols_solver.solve(rhs);
    EXPECT_TRUE(partial_cols_solver.converged());
    EXPECT_LT((partial_cols * partial_cols_solution - rhs).norm(), 1.0e-10);
}

template <int StorageOrder> void check_gmres_restart_and_dynamic_state() {
    using fixed_matrix = Matrix<double, 3, 3, StorageOrder>;
    using fixed_vector = Matrix<double, 3, 1, StorageOrder>;
    const fixed_matrix fixed({4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0});
    const fixed_vector expected(std::vector<double> {1.0, -2.0, 0.5});
    const fixed_vector rhs(fixed * expected);
    using fixed_preconditioner = DiagonalPreconditioner<fixed_matrix>;

    GMRES restart_solver(fixed, fixed_preconditioner {}, 500, 1, 1.0e-10);
    const auto solution = restart_solver.solve(rhs);
    EXPECT_TRUE(restart_solver.converged());
    EXPECT_GT(restart_solver.iterations(), 1);
    EXPECT_LT((fixed * solution - rhs).norm() / rhs.norm(), 1.0e-9);

    const auto exact_solution = restart_solver.solve(rhs, expected);
    EXPECT_TRUE(restart_solver.converged());
    EXPECT_EQ(restart_solver.iterations(), 0);
    EXPECT_EQ(exact_solution, expected);

    using dynamic_matrix = Matrix<double, Dynamic, Dynamic, StorageOrder>;
    using dynamic_vector = Matrix<double, Dynamic, 1, StorageOrder>;
    const dynamic_matrix dynamic(fixed);
    const dynamic_vector dynamic_expected(expected);
    const dynamic_vector dynamic_rhs(dynamic * dynamic_expected);
    using dynamic_preconditioner = IdentityPreconditioner<dynamic_matrix>;
    GMRES dynamic_solver(dynamic, dynamic_preconditioner {}, 50, 2, 1.0e-12);
    const auto dynamic_solution = dynamic_solver.solve(dynamic_rhs);
    EXPECT_TRUE(dynamic_solver.converged());
    EXPECT_LT((dynamic * dynamic_solution - dynamic_rhs).norm() / dynamic_rhs.norm(), 1.0e-10);

    const dynamic_vector zero = dynamic_vector::Zero(3);
    const auto zero_solution = dynamic_solver.solve(zero);
    EXPECT_TRUE(dynamic_solver.converged());
    EXPECT_EQ(dynamic_solver.iterations(), 0);
    EXPECT_EQ(zero_solution, zero);
}

template <int StorageOrder> void check_gmres_ownership_and_breakdown() {
    using matrix_type = Matrix<double, 2, 2, StorageOrder>;
    using vector_type = Matrix<double, 2, 1, StorageOrder>;
    const vector_type rhs(std::vector<double> {2.0, 8.0});

    matrix_type source({2.0, 0.0, 0.0, 4.0});
    GMRES retained_source_solver(source, IdentityPreconditioner<matrix_type> {}, 10, 2, 1.0e-12);
    source(0, 0) = 100.0;
    const auto retained_source_solution = retained_source_solver.solve(rhs);
    EXPECT_TRUE(retained_source_solver.converged());
    EXPECT_NEAR(retained_source_solution[0], 1.0, 1.0e-12);
    EXPECT_NEAR(retained_source_solution[1], 2.0, 1.0e-12);

    auto retained_expression_solver = [] {
        const matrix_type matrix({2.0, 0.0, 0.0, 4.0});
        const matrix_type zero = matrix_type::Zero();
        using expression_type = decltype(matrix + zero);
        return GMRES(matrix + zero, IdentityPreconditioner<expression_type> {}, 10, 2, 1.0e-12);
    }();
    const auto retained_expression_solution = retained_expression_solver.solve(rhs);
    EXPECT_TRUE(retained_expression_solver.converged());
    EXPECT_NEAR(retained_expression_solution[0], 1.0, 1.0e-12);
    EXPECT_NEAR(retained_expression_solution[1], 2.0, 1.0e-12);

    const matrix_type zero = matrix_type::Zero();
    GMRES breakdown_solver(zero, IdentityPreconditioner<matrix_type> {}, 5, 2, 1.0e-12);
    const auto breakdown_solution = breakdown_solver.solve(rhs);
    EXPECT_FALSE(breakdown_solver.converged());
    EXPECT_EQ(breakdown_solver.iterations(), 0);
    EXPECT_TRUE(std::isfinite(breakdown_solver.residual()));
    EXPECT_EQ(breakdown_solution, vector_type::Zero());
}

template <int StorageOrder> void check_gmres_extreme_scale() {
    using matrix_type = Matrix<double, 2, 2, StorageOrder>;
    using vector_type = Matrix<double, 2, 1, StorageOrder>;
    const matrix_type identity({1.0, 0.0, 0.0, 1.0});
    const vector_type rhs(std::vector<double> {1.4e308, 1.4e308});
    const vector_type initial(std::vector<double> {1.4e308, 0.0});
    using preconditioner_type = IdentityPreconditioner<matrix_type>;

    GMRES zero_guess_solver(identity, preconditioner_type {}, 4, 2, 1.0e-12);
    const auto zero_guess_solution = zero_guess_solver.solve(rhs);
    EXPECT_TRUE(zero_guess_solver.converged());
    EXPECT_GT(zero_guess_solver.iterations(), 0);
    EXPECT_NEAR(zero_guess_solution[0] / rhs[0], 1.0, 1.0e-12);
    EXPECT_NEAR(zero_guess_solution[1] / rhs[1], 1.0, 1.0e-12);

    GMRES solver(identity, preconditioner_type {}, 4, 2, 1.0e-12);
    const auto solution = solver.solve(rhs, initial);
    EXPECT_TRUE(solver.converged());
    EXPECT_GT(solver.iterations(), 0);
    EXPECT_EQ(solution, rhs);
}

template <int StorageOrder> void check_gmres_invalid_contracts() {
    using dynamic_matrix = Matrix<double, Dynamic, Dynamic, StorageOrder>;
    using dynamic_vector = Matrix<double, Dynamic, 1, StorageOrder>;
    using identity_type = IdentityPreconditioner<dynamic_matrix>;
    using solver_type = GMRES<dynamic_matrix, identity_type>;

    EXPECT_THROW(static_cast<void>(solver_type(identity_type {}, 0, 2, 1.0e-12)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(solver_type(identity_type {}, 10, 0, 1.0e-12)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(solver_type(identity_type {}, 10, 2, 0.0)), std::invalid_argument);
    EXPECT_THROW(
      static_cast<void>(solver_type(identity_type {}, 10, 2, std::numeric_limits<double>::quiet_NaN())),
      std::invalid_argument);
    EXPECT_THROW(
      static_cast<void>(solver_type(identity_type {}, 10, 2, std::numeric_limits<double>::infinity())),
      std::invalid_argument);

    const dynamic_matrix valid(Matrix<double, 2, 2, StorageOrder>({4.0, 1.0, 1.0, 3.0}));
    const dynamic_vector rhs(std::vector<double> {1.0, 2.0});
    solver_type solver(identity_type {}, 10, 2, 1.0e-12);
    EXPECT_FALSE(solver.initialized());
    EXPECT_THROW(static_cast<void>(solver.solve(rhs)), std::domain_error);

    solver.compute(valid);
    ASSERT_TRUE(solver.initialized());
    Matrix<double, Dynamic, Dynamic, StorageOrder> nonfinite(valid);
    nonfinite(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(solver.compute(nonfinite), std::invalid_argument);
    EXPECT_FALSE(solver.initialized());
    EXPECT_FALSE(solver.converged());
    EXPECT_EQ(solver.iterations(), 0);
    EXPECT_TRUE(std::isinf(solver.residual()));
    EXPECT_THROW(static_cast<void>(solver.solve(rhs)), std::domain_error);

    EXPECT_THROW(solver.compute(dynamic_matrix()), std::invalid_argument);
    EXPECT_FALSE(solver.initialized());
    EXPECT_THROW(solver.compute(dynamic_matrix(2, 3)), std::invalid_argument);
    EXPECT_FALSE(solver.initialized());

    solver.compute(valid);
    dynamic_vector wrong_rows(3);
    EXPECT_THROW(static_cast<void>(solver.solve(wrong_rows)), std::invalid_argument);
    EXPECT_TRUE(solver.initialized());
    EXPECT_FALSE(solver.converged());
    EXPECT_EQ(solver.iterations(), 0);
    EXPECT_TRUE(std::isinf(solver.residual()));

    dynamic_vector nonfinite_rhs(rhs);
    nonfinite_rhs[0] = std::numeric_limits<double>::infinity();
    EXPECT_THROW(static_cast<void>(solver.solve(nonfinite_rhs)), std::invalid_argument);
    dynamic_vector nonfinite_initial = dynamic_vector::Zero(2);
    nonfinite_initial[1] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(static_cast<void>(solver.solve(rhs, nonfinite_initial)), std::invalid_argument);

    const dynamic_matrix recompute_matrix(
      Matrix<double, 3, 3, StorageOrder>({2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 2.0}));
    const dynamic_vector recompute_expected(std::vector<double> {1.0, -2.0, 0.5});
    const dynamic_vector recompute_rhs(recompute_matrix * recompute_expected);
    solver.compute(recompute_matrix);
    ASSERT_TRUE(solver.initialized());
    const auto recompute_solution = solver.solve(recompute_rhs);
    EXPECT_TRUE(solver.converged());
    EXPECT_LT((recompute_matrix * recompute_solution - recompute_rhs).norm(), 1.0e-10);

    using partial_matrix = Matrix<double, Dynamic, 2, StorageOrder>;
    using partial_preconditioner = IdentityPreconditioner<partial_matrix>;
    GMRES<partial_matrix, partial_preconditioner> partial_solver(partial_preconditioner {}, 10, 2, 1.0e-12);
    EXPECT_THROW(partial_solver.compute(partial_matrix(3, 2)), std::invalid_argument);
    EXPECT_FALSE(partial_solver.initialized());

    using rejecting_type = RejectingPreconditioner<dynamic_matrix>;
    GMRES<dynamic_matrix, rejecting_type> rejecting_solver(rejecting_type {}, 10, 2, 1.0e-12);
    EXPECT_THROW(rejecting_solver.compute(valid), std::domain_error);
    EXPECT_FALSE(rejecting_solver.initialized());

    using diagonal_type = DiagonalPreconditioner<dynamic_matrix>;
    GMRES<dynamic_matrix, diagonal_type> diagonal_solver(diagonal_type {}, 10, 2, 1.0e-12);
    const dynamic_matrix zero = dynamic_matrix::Zero(2, 2);
    EXPECT_THROW(diagonal_solver.compute(zero), std::domain_error);
    EXPECT_FALSE(diagonal_solver.initialized());
}

TEST(linear_algebra, gmres) {
    check_gmres_happy_paths<RowMajor>();
    check_gmres_happy_paths<ColMajor>();
    check_gmres_restart_and_dynamic_state<RowMajor>();
    check_gmres_restart_and_dynamic_state<ColMajor>();
    check_gmres_ownership_and_breakdown<RowMajor>();
    check_gmres_ownership_and_breakdown<ColMajor>();
    check_gmres_extreme_scale<RowMajor>();
    check_gmres_extreme_scale<ColMajor>();
    check_gmres_invalid_contracts<RowMajor>();
    check_gmres_invalid_contracts<ColMajor>();
}

}   // namespace
