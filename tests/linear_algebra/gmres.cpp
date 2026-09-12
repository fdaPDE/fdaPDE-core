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

#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace {

using namespace fdapde;

/// @brief supplies a preconditioner that explicitly rejects initialization
template <typename XprType_> class RejectingPreconditioner {
    using XprType = std::decay_t<XprType_>;
   public:
    using Scalar = std::remove_cv_t<typename XprType::Scalar>;

    /// @brief accepts the operator without changing the rejected state
    template <typename MatrixType> constexpr void compute(const MatrixExpr<MatrixType>&) { }

    /// @brief returns an owning copy if invoked despite the rejected state
    template <typename RhsType> constexpr auto solve(const MatrixExpr<RhsType>& rhs) const {
        return Matrix<Scalar, RhsType::Rows, RhsType::Cols>(rhs);
    }

    /// @brief reports rejection to the solver
    constexpr bool valid() const { return false; }
};

// recomputes the unpreconditioned residual against the supplied matrix and right-hand side
template <typename MatrixType, typename SolutionType, typename RhsType>
void expect_small_residual(const MatrixType& matrix, const SolutionType& solution, const RhsType& rhs) {
    // its Euclidean norm is below the absolute residual tolerance
    EXPECT_LT((matrix * solution - rhs).norm(), 1.0e-10);
}

// checks both preconditioners on fixed and partially dynamic systems
template <int StorageOrder> void check_gmres_happy_paths() {
    using fixed_matrix = Matrix<double, 2, 2, StorageOrder>;
    using fixed_vector = Matrix<double, 2, 1, StorageOrder>;
    const fixed_matrix matrix({4.0, 1.0, 1.0, 3.0});
    const fixed_vector rhs(std::vector<double> {1.0, 2.0});

    using identity_type = IdentityPreconditioner<fixed_matrix>;
    GMRES identity_solver(matrix, identity_type {}, 20, 2, 1.0e-12);
    const auto identity_solution = identity_solver.solve(rhs);
    // construction has initialized the operator and preconditioner
    EXPECT_TRUE(identity_solver.initialized());
    // identity preconditioning reaches the requested relative residual tolerance
    EXPECT_TRUE(identity_solver.converged());
    // the two-dimensional system needs no more than two Arnoldi steps
    EXPECT_LE(identity_solver.iterations(), 2);
    // the reported preconditioned residual is representable
    EXPECT_TRUE(std::isfinite(identity_solver.residual()));
    // independently check the identity-preconditioned solution against A x = b
    expect_small_residual(matrix, identity_solution, rhs);

    using diagonal_type = DiagonalPreconditioner<fixed_matrix>;
    GMRES diagonal_solver(matrix, diagonal_type {}, 20, 2, 1.0e-12);
    const auto diagonal_solution = diagonal_solver.solve(rhs);
    // diagonal preconditioning also reaches the requested tolerance
    EXPECT_TRUE(diagonal_solver.converged());
    // a full two-dimensional Krylov space solves the diagonally preconditioned system
    EXPECT_LE(diagonal_solver.iterations(), 2);
    // independently check the diagonally preconditioned solution against A x = b
    expect_small_residual(matrix, diagonal_solution, rhs);

    const MatrixView<const double, 2, 2, StorageOrder> const_view(matrix.data());
    using const_view_type = decltype(const_view);
    GMRES const_view_solver(const_view, IdentityPreconditioner<const_view_type> {}, 20, 2, 1.0e-12);
    // construction from a const view gives the same small physical residual
    expect_small_residual(matrix, const_view_solver.solve(rhs), rhs);

    using partial_matrix = Matrix<double, Dynamic, 2, StorageOrder>;
    using partial_vector = Matrix<double, Dynamic, 1, StorageOrder>;
    const partial_matrix partial(matrix);
    const partial_vector partial_rhs(rhs);
    GMRES partial_solver(partial, IdentityPreconditioner<partial_matrix> {}, 20, 2, 1.0e-12);
    const auto partial_solution = partial_solver.solve(partial_rhs);
    // dynamic row extent preserves convergence
    EXPECT_TRUE(partial_solver.converged());
    // the partial-row solution satisfies the original equations
    EXPECT_LT((partial * partial_solution - partial_rhs).norm(), 1.0e-10);

    using partial_cols_matrix = Matrix<double, 2, Dynamic, StorageOrder>;
    const partial_cols_matrix partial_cols(matrix);
    GMRES partial_cols_solver(partial_cols, IdentityPreconditioner<partial_cols_matrix> {}, 20, 2, 1.0e-12);
    const auto partial_cols_solution = partial_cols_solver.solve(rhs);
    // dynamic column extent also preserves convergence
    EXPECT_TRUE(partial_cols_solver.converged());
    // the partial-column solution satisfies the original equations
    EXPECT_LT((partial_cols * partial_cols_solution - rhs).norm(), 1.0e-10);
}

// checks restart cycles and observer reset between solves
template <int StorageOrder> void check_gmres_restart_and_dynamic_state() {
    using fixed_matrix = Matrix<double, 3, 3, StorageOrder>;
    using fixed_vector = Matrix<double, 3, 1, StorageOrder>;
    const fixed_matrix fixed({4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0});
    const fixed_vector expected(std::vector<double> {1.0, -2.0, 0.5});
    const fixed_vector rhs(fixed * expected);
    using fixed_preconditioner = DiagonalPreconditioner<fixed_matrix>;

    GMRES restart_solver(fixed, fixed_preconditioner {}, 500, 1, 1.0e-10);
    const auto solution = restart_solver.solve(rhs);
    // restart length one converges after repeated cycles
    EXPECT_TRUE(restart_solver.converged());
    // more than one step confirms that restart cycles were exercised
    EXPECT_GT(restart_solver.iterations(), 1);
    // the physical relative residual confirms the restarted solution
    EXPECT_LT((fixed * solution - rhs).norm() / rhs.norm(), 1.0e-9);

    const auto exact_solution = restart_solver.solve(rhs, expected);
    // an exact initial guess is recognized as converged
    EXPECT_TRUE(restart_solver.converged());
    // an exact initial guess needs no Arnoldi steps
    EXPECT_EQ(restart_solver.iterations(), 0);
    // the exact initial guess is returned unchanged
    EXPECT_EQ(exact_solution, expected);

    using dynamic_matrix = Matrix<double, Dynamic, Dynamic, StorageOrder>;
    using dynamic_vector = Matrix<double, Dynamic, 1, StorageOrder>;
    const dynamic_matrix dynamic(fixed);
    const dynamic_vector dynamic_expected(expected);
    const dynamic_vector dynamic_rhs(dynamic * dynamic_expected);
    using dynamic_preconditioner = IdentityPreconditioner<dynamic_matrix>;
    GMRES dynamic_solver(dynamic, dynamic_preconditioner {}, 50, 2, 1.0e-12);
    const auto dynamic_solution = dynamic_solver.solve(dynamic_rhs);
    // a fully dynamic system converges with a truncated Krylov basis
    EXPECT_TRUE(dynamic_solver.converged());
    // its physical relative residual satisfies the independent tolerance
    EXPECT_LT((dynamic * dynamic_solution - dynamic_rhs).norm() / dynamic_rhs.norm(), 1.0e-10);

    const dynamic_vector zero = dynamic_vector::Zero(3);
    const auto zero_solution = dynamic_solver.solve(zero);
    // the zero right-hand side is solved by the zero initial guess
    EXPECT_TRUE(dynamic_solver.converged());
    // observer reset reports zero steps for the second solve
    EXPECT_EQ(dynamic_solver.iterations(), 0);
    // the returned vector is exactly zero
    EXPECT_EQ(zero_solution, zero);
}

// checks independent ownership and singular Arnoldi breakdown
template <int StorageOrder> void check_gmres_ownership_and_breakdown() {
    using matrix_type = Matrix<double, 2, 2, StorageOrder>;
    using vector_type = Matrix<double, 2, 1, StorageOrder>;
    const vector_type rhs(std::vector<double> {2.0, 8.0});

    matrix_type source({2.0, 0.0, 0.0, 4.0});
    GMRES retained_source_solver(source, IdentityPreconditioner<matrix_type> {}, 10, 2, 1.0e-12);
    source(0, 0) = 100.0;
    const auto retained_source_solution = retained_source_solver.solve(rhs);
    // mutating the source does not prevent the retained operator from converging
    EXPECT_TRUE(retained_source_solver.converged());
    // the first coefficient matches the original diagonal system, not the mutated source
    EXPECT_NEAR(retained_source_solution[0], 1.0, 1.0e-12);
    // the second coefficient matches eight divided by four
    EXPECT_NEAR(retained_source_solution[1], 2.0, 1.0e-12);

    auto retained_expression_solver = [] {
        const matrix_type matrix({2.0, 0.0, 0.0, 4.0});
        const matrix_type zero = matrix_type::Zero();
        using expression_type = decltype(matrix + zero);
        return GMRES(matrix + zero, IdentityPreconditioner<expression_type> {}, 10, 2, 1.0e-12);
    }();
    const auto retained_expression_solution = retained_expression_solver.solve(rhs);
    // a solver built from a destroyed expression still converges
    EXPECT_TRUE(retained_expression_solver.converged());
    // the first coefficient survives destruction of the source expression
    EXPECT_NEAR(retained_expression_solution[0], 1.0, 1.0e-12);
    // the second coefficient survives destruction of the source expression
    EXPECT_NEAR(retained_expression_solution[1], 2.0, 1.0e-12);

    const matrix_type zero = matrix_type::Zero();
    GMRES breakdown_solver(zero, IdentityPreconditioner<matrix_type> {}, 5, 2, 1.0e-12);
    const auto breakdown_solution = breakdown_solver.solve(rhs);
    // a zero operator with nonzero right-hand side cannot converge
    EXPECT_FALSE(breakdown_solver.converged());
    // breakdown before the first valid rotation completes no Arnoldi steps
    EXPECT_EQ(breakdown_solver.iterations(), 0);
    // the residual remains finite after singular breakdown
    EXPECT_TRUE(std::isfinite(breakdown_solver.residual()));
    // the last valid iterate remains the zero initial guess
    EXPECT_EQ(breakdown_solution, vector_type::Zero());
}

// checks residual normalization when the right-hand-side norm exceeds double range
template <int StorageOrder> void check_gmres_extreme_scale() {
    using matrix_type = Matrix<double, 2, 2, StorageOrder>;
    using vector_type = Matrix<double, 2, 1, StorageOrder>;
    const matrix_type identity({1.0, 0.0, 0.0, 1.0});
    const vector_type rhs(std::vector<double> {1.4e308, 1.4e308});
    const vector_type initial(std::vector<double> {1.4e308, 0.0});
    using preconditioner_type = IdentityPreconditioner<matrix_type>;

    GMRES zero_guess_solver(identity, preconditioner_type {}, 4, 2, 1.0e-12);
    const auto zero_guess_solution = zero_guess_solver.solve(rhs);
    // scaled norm comparison permits convergence despite an unrepresentable initial norm
    EXPECT_TRUE(zero_guess_solver.converged());
    // the zero guess must take a step before satisfying this nonzero system
    EXPECT_GT(zero_guess_solver.iterations(), 0);
    // the first large coefficient matches the identity-system oracle in relative scale
    EXPECT_NEAR(zero_guess_solution[0] / rhs[0], 1.0, 1.0e-12);
    // the second large coefficient matches the same oracle
    EXPECT_NEAR(zero_guess_solution[1] / rhs[1], 1.0, 1.0e-12);

    GMRES solver(identity, preconditioner_type {}, 4, 2, 1.0e-12);
    const auto solution = solver.solve(rhs, initial);
    // a partially exact large initial guess also converges
    EXPECT_TRUE(solver.converged());
    // the missing second coefficient requires a correction
    EXPECT_GT(solver.iterations(), 0);
    // the final solution equals the right-hand side for the identity operator
    EXPECT_EQ(solution, rhs);
}

// checks public validation, invalidation and recovery after failed computations
template <int StorageOrder> void check_gmres_invalid_contracts() {
    using dynamic_matrix = Matrix<double, Dynamic, Dynamic, StorageOrder>;
    using dynamic_vector = Matrix<double, Dynamic, 1, StorageOrder>;
    using identity_type = IdentityPreconditioner<dynamic_matrix>;
    using solver_type = GMRES<dynamic_matrix, identity_type>;

    // a zero iteration budget is rejected
    EXPECT_THROW(static_cast<void>(solver_type(identity_type {}, 0, 2, 1.0e-12)), std::invalid_argument);
    // a zero restart length is rejected
    EXPECT_THROW(static_cast<void>(solver_type(identity_type {}, 10, 0, 1.0e-12)), std::invalid_argument);
    // a zero tolerance is rejected
    EXPECT_THROW(static_cast<void>(solver_type(identity_type {}, 10, 2, 0.0)), std::invalid_argument);
    // a NaN tolerance is rejected
    EXPECT_THROW(
      static_cast<void>(solver_type(identity_type {}, 10, 2, std::numeric_limits<double>::quiet_NaN())),
      std::invalid_argument);
    // an infinite tolerance is rejected
    EXPECT_THROW(
      static_cast<void>(solver_type(identity_type {}, 10, 2, std::numeric_limits<double>::infinity())),
      std::invalid_argument);

    const dynamic_matrix valid(Matrix<double, 2, 2, StorageOrder>({4.0, 1.0, 1.0, 3.0}));
    const dynamic_vector rhs(std::vector<double> {1.0, 2.0});
    solver_type solver(identity_type {}, 10, 2, 1.0e-12);
    // configuration alone does not initialize an operator
    EXPECT_FALSE(solver.initialized());
    // solving before compute reports an unavailable solver
    EXPECT_THROW(static_cast<void>(solver.solve(rhs)), std::domain_error);

    solver.compute(valid);
    // a valid compute initializes the solver before failure probes
    ASSERT_TRUE(solver.initialized());
    Matrix<double, Dynamic, Dynamic, StorageOrder> nonfinite(valid);
    nonfinite(0, 0) = std::numeric_limits<double>::quiet_NaN();
    // compute rejects a matrix containing NaN
    EXPECT_THROW(solver.compute(nonfinite), std::invalid_argument);
    // failed compute invalidates the previously usable solver
    EXPECT_FALSE(solver.initialized());
    // failed compute clears the previous convergence result
    EXPECT_FALSE(solver.converged());
    // failed compute clears the iteration count
    EXPECT_EQ(solver.iterations(), 0);
    // failed compute marks the residual as unavailable
    EXPECT_TRUE(std::isinf(solver.residual()));
    // solve cannot reuse the stale matrix after failed compute
    EXPECT_THROW(static_cast<void>(solver.solve(rhs)), std::domain_error);

    // an empty operator is rejected
    EXPECT_THROW(solver.compute(dynamic_matrix()), std::invalid_argument);
    // empty-input failure leaves the solver unavailable
    EXPECT_FALSE(solver.initialized());
    // a rectangular operator is rejected
    EXPECT_THROW(solver.compute(dynamic_matrix(2, 3)), std::invalid_argument);
    // rectangular-input failure leaves the solver unavailable
    EXPECT_FALSE(solver.initialized());

    solver.compute(valid);
    dynamic_vector wrong_rows(3);
    // a right-hand side with the wrong row count is rejected
    EXPECT_THROW(static_cast<void>(solver.solve(wrong_rows)), std::invalid_argument);
    // invalid solve input preserves the successfully computed operator
    EXPECT_TRUE(solver.initialized());
    // invalid solve input clears convergence from any earlier solve
    EXPECT_FALSE(solver.converged());
    // invalid solve input resets the step count
    EXPECT_EQ(solver.iterations(), 0);
    // invalid solve input leaves no reported finite residual
    EXPECT_TRUE(std::isinf(solver.residual()));

    dynamic_vector nonfinite_rhs(rhs);
    nonfinite_rhs[0] = std::numeric_limits<double>::infinity();
    // infinite right-hand-side coefficients are rejected
    EXPECT_THROW(static_cast<void>(solver.solve(nonfinite_rhs)), std::invalid_argument);
    dynamic_vector nonfinite_initial = dynamic_vector::Zero(2);
    nonfinite_initial[1] = std::numeric_limits<double>::quiet_NaN();
    // nonfinite initial-guess coefficients containing NaN are rejected
    EXPECT_THROW(static_cast<void>(solver.solve(rhs, nonfinite_initial)), std::invalid_argument);

    const dynamic_matrix recompute_matrix(
      Matrix<double, 3, 3, StorageOrder>({2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 2.0}));
    const dynamic_vector recompute_expected(std::vector<double> {1.0, -2.0, 0.5});
    const dynamic_vector recompute_rhs(recompute_matrix * recompute_expected);
    solver.compute(recompute_matrix);
    // recomputing with a different valid order reinitializes the workspaces
    ASSERT_TRUE(solver.initialized());
    const auto recompute_solution = solver.solve(recompute_rhs);
    // the resized solver converges on the new system
    EXPECT_TRUE(solver.converged());
    // the resized solution satisfies its independently constructed right-hand side
    EXPECT_LT((recompute_matrix * recompute_solution - recompute_rhs).norm(), 1.0e-10);

    using partial_matrix = Matrix<double, Dynamic, 2, StorageOrder>;
    using partial_preconditioner = IdentityPreconditioner<partial_matrix>;
    GMRES<partial_matrix, partial_preconditioner> partial_solver(partial_preconditioner {}, 10, 2, 1.0e-12);
    // a partial-static operator must still be square
    EXPECT_THROW(partial_solver.compute(partial_matrix(3, 2)), std::invalid_argument);
    // failed partial-static compute leaves the solver unavailable
    EXPECT_FALSE(partial_solver.initialized());

    using rejecting_type = RejectingPreconditioner<dynamic_matrix>;
    GMRES<dynamic_matrix, rejecting_type> rejecting_solver(rejecting_type {}, 10, 2, 1.0e-12);
    // a custom preconditioner reporting invalid state is rejected
    EXPECT_THROW(rejecting_solver.compute(valid), std::domain_error);
    // rejected preconditioning leaves the solver unavailable
    EXPECT_FALSE(rejecting_solver.initialized());

    using diagonal_type = DiagonalPreconditioner<dynamic_matrix>;
    GMRES<dynamic_matrix, diagonal_type> diagonal_solver(diagonal_type {}, 10, 2, 1.0e-12);
    const dynamic_matrix zero = dynamic_matrix::Zero(2, 2);
    // zero diagonal entries cannot initialize diagonal preconditioning
    EXPECT_THROW(diagonal_solver.compute(zero), std::domain_error);
    // failed diagonal preconditioning leaves the solver unavailable
    EXPECT_FALSE(diagonal_solver.initialized());
}

// checks solutions with both dense storage orders
TEST(linear_algebra, gmres_solutions) {
    // run the happy paths oracles with RowMajor input storage
    check_gmres_happy_paths<RowMajor>();
    // run the happy paths oracles with ColMajor input storage
    check_gmres_happy_paths<ColMajor>();
}

// checks restart with both dense storage orders
TEST(linear_algebra, gmres_restart) {
    // run the restart and dynamic state oracles with RowMajor input storage
    check_gmres_restart_and_dynamic_state<RowMajor>();
    // run the restart and dynamic state oracles with ColMajor input storage
    check_gmres_restart_and_dynamic_state<ColMajor>();
}

// checks ownership with both dense storage orders
TEST(linear_algebra, gmres_ownership) {
    // run the ownership and breakdown oracles with RowMajor input storage
    check_gmres_ownership_and_breakdown<RowMajor>();
    // run the ownership and breakdown oracles with ColMajor input storage
    check_gmres_ownership_and_breakdown<ColMajor>();
}

// checks scales with both dense storage orders
TEST(linear_algebra, gmres_scales) {
    // run the extreme scale oracles with RowMajor input storage
    check_gmres_extreme_scale<RowMajor>();
    // run the extreme scale oracles with ColMajor input storage
    check_gmres_extreme_scale<ColMajor>();
}

// checks contracts with both dense storage orders
TEST(linear_algebra, gmres_contracts) {
    // run the invalid contracts oracles with RowMajor input storage
    check_gmres_invalid_contracts<RowMajor>();
    // run the invalid contracts oracles with ColMajor input storage
    check_gmres_invalid_contracts<ColMajor>();
}

// verifies nonsymmetric systems against an independently chosen solution in each floating scalar type
template <typename Scalar> void check_nonsymmetric_solution() {
    using matrix_type = Matrix<Scalar, 3, 3>;
    using vector_type = Vector<Scalar, 3>;
    const matrix_type matrix({4, 1, -1, 0, 3, 1, 2, 0, 5});
    const vector_type expected(std::vector<Scalar> {1, -2, 3});
    const vector_type rhs(matrix * expected);
    const Scalar tolerance = std::is_same_v<Scalar, float> ? Scalar(1e-5) : Scalar(1e-12);
    GMRES solver(matrix, DiagonalPreconditioner<matrix_type> {}, 20, 3, tolerance);
    const auto solution = solver.solve(rhs);
    // the left-preconditioned iteration converges on a matrix that is not symmetric
    EXPECT_TRUE(solver.converged());
    // the returned coefficients recover the independently selected solution
    EXPECT_LT((solution - expected).norm(), Scalar(20) * tolerance);
    const DiagonalPreconditioner<matrix_type> preconditioner(matrix);
    const vector_type raw_residual(rhs - matrix * solution);
    const auto checked_residual = preconditioner.solve(raw_residual);
    // the residual observer matches independent inverse-diagonal scaling of b - A x
    EXPECT_NEAR(solver.residual(), checked_residual.norm(), tolerance);
}

// checks nonsymmetric systems in single and double precision using a known solution
TEST(linear_algebra, gmres_nonsymmetric_scalars) {
    // run the nonsymmetric solution and residual oracles with float coefficients
    check_nonsymmetric_solution<float>();
    // run the same oracles with double coefficients and a tighter tolerance
    check_nonsymmetric_solution<double>();
}

// checks finite nonconvergence at the iteration budget and independence of copied solver state
TEST(linear_algebra, gmres_iteration_limit_and_copy) {
    using matrix_type = Matrix<double, 3, 3>;
    using vector_type = Vector<double, 3>;
    const matrix_type matrix({4, 1, -1, 0, 3, 1, 2, 0, 5});
    const vector_type rhs(std::vector<double> {1, 2, 3});
    GMRES solver(matrix, IdentityPreconditioner<matrix_type> {}, 1, 3, 1e-14);
    auto copied = solver;
    const auto limited = solver.solve(rhs);
    // a single Arnoldi step cannot solve this system to the requested tolerance
    EXPECT_FALSE(solver.converged());
    // reaching the configured budget reports exactly one completed step
    EXPECT_EQ(solver.iterations(), 1);
    // the finite last iterate retains its independently recomputed physical residual
    EXPECT_NEAR(solver.residual(), (rhs - matrix * limited).norm(), 1e-13);
    const auto original_residual = solver.residual();
    const auto zero = vector_type::Zero();
    const auto copy_result = copied.solve(zero);
    // solving the copy with a zero right-hand side returns the exact zero solution
    EXPECT_EQ(copy_result, zero);
    // the copied solver can converge independently of the original failed solve
    EXPECT_TRUE(copied.converged());
    // solving the copy does not overwrite the original residual observer
    EXPECT_EQ(solver.residual(), original_residual);
    // the original convergence flag remains unchanged by the copied solve
    EXPECT_FALSE(solver.converged());
}

}   // namespace
