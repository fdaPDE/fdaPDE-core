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

#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

namespace {

using namespace fdapde;

template <typename Actual, typename Expected>
void expect_matrix_near(const Actual& actual, const Expected& expected, double tolerance = 1.0e-11) {
    // the result has the independently assembled system solution row count
    ASSERT_EQ(actual.rows(), expected.rows());
    // the result has one column per oracle right-hand side
    ASSERT_EQ(actual.cols(), expected.cols());
    for (int row = 0; row < actual.rows(); ++row) {
        for (int col = 0; col < actual.cols(); ++col) {
            // each coefficient agrees with the direct or analytic solution within the supplied tolerance
            EXPECT_NEAR(static_cast<double>(actual(row, col)), static_cast<double>(expected(row, col)), tolerance);
        }
    }
}

template <int StorageOrder> void check_woodbury_against_direct_dense_solve() {
    using base_matrix = Matrix<double, 3, 3, StorageOrder>;
    using update_matrix = Matrix<double, 3, 2, StorageOrder>;
    using inverse_core = Matrix<double, 2, 2, StorageOrder>;
    using transpose_update = Matrix<double, 2, 3, StorageOrder>;
    using vector_rhs = Matrix<double, 3, 1, StorageOrder>;
    using matrix_rhs = Matrix<double, 3, 2, StorageOrder>;

    const base_matrix base({4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0});
    update_matrix u({1.0, 0.0, 0.0, 1.0, 1.0, -1.0});
    const update_matrix original_u(u);
    const inverse_core inverse_c({2.0, 0.0, 0.0, 3.0});
    transpose_update v({1.0, 0.0, 1.0, 0.0, 1.0, -1.0});
    const transpose_update original_v(v);
    const matrix_rhs rhs({1.0, 2.0, 3.0, -1.0, 2.0, 4.0});
    const vector_rhs second_rhs({-2.0, 1.0, 3.0});

    const inverse_core c(inverse_c.inverse());
    const base_matrix full_system(base + original_u * c * original_v);
    const PartialPivLU direct_solver(full_system);
    const matrix_rhs expected(direct_solver.solve(rhs));
    const vector_rhs second_expected(direct_solver.solve(second_rhs));

    PartialPivLU base_solver(base);
    Woodbury cached(base_solver, u, inverse_c, v);
    // the cached update matches a direct solve of the explicitly assembled full system
    expect_matrix_near(cached.solve(rhs), expected);
    // a second right-hand side matches the direct solve without rebuilding the update
    expect_matrix_near(cached.solve(second_rhs), second_expected);

    auto temporary_owned = [&] {
        return Woodbury(
          PartialPivLU(base), original_u + update_matrix::Zero(), inverse_c + inverse_core::Zero(),
          original_v + transpose_update::Zero());
    }();
    // materialized update expressions outlive the temporaries used at construction
    expect_matrix_near(temporary_owned.solve(rhs), expected);

    base_matrix identity;
    identity.set_zero();
    for (int i = 0; i < 3; ++i) identity(i, i) = 1.0;
    base_solver.compute(identity);
    u.set_zero();
    v.set_zero();
    // the cached update matches a direct solve of the explicitly assembled full system
    expect_matrix_near(cached.solve(rhs), expected);

    auto copied = cached;
    auto moved = std::move(copied);
    Woodbury<decltype(base_solver)> assigned;
    assigned = cached;
    // moving a decomposition makes its source unavailable for subsequent solves
    EXPECT_THROW(static_cast<void>(copied.solve(rhs)), std::domain_error);
    // the moved cache reproduces the independent direct-system solution
    expect_matrix_near(moved.solve(rhs), expected);
    // copy assignment retains an independent usable decomposition
    expect_matrix_near(assigned.solve(rhs), expected);
}

/// @brief counts column solves while implementing an identity inverse
struct CountingIdentitySolver {
    std::shared_ptr<int> solve_calls;

    /// @brief copies the right-hand side and increments the shared solve counter
    template <typename RhsType> auto solve(const MatrixExpr<RhsType>& rhs) {
        ++*solve_calls;
        return Matrix<double, Dynamic, Dynamic>(rhs);
    }
};

/// @brief injects a base-solver result with one extra row
struct WrongShapeSolver {
    /// @brief returns an extra row to exercise backend shape validation
    template <typename RhsType> auto solve(const MatrixExpr<RhsType>& rhs) {
        return Matrix<double, Dynamic, Dynamic>(rhs.rows() + 1, rhs.cols());
    }
};

/// @brief injects an infinite coefficient into the base-solver result
struct NonfiniteSolver {
    /// @brief returns infinity to exercise backend coefficient validation
    template <typename RhsType> auto solve(const MatrixExpr<RhsType>& rhs) {
        Matrix<double, Dynamic, Dynamic> result(rhs);
        result(0, 0) = std::numeric_limits<double>::infinity();
        return result;
    }
};

/// @brief returns a valid initialization result followed by an invalid solve shape
struct LateWrongShapeSolver {
    int solve_calls = 0;

    /// @brief returns the correct shape once and an extra row on later calls
    template <typename RhsType> auto solve(const MatrixExpr<RhsType>& rhs) {
        ++solve_calls;
        if (solve_calls == 1) return Matrix<double, Dynamic, Dynamic>(rhs);
        return Matrix<double, Dynamic, Dynamic>(rhs.rows() + 1, rhs.cols());
    }
};

using fixed_solver = PartialPivLU<Matrix<double, 3, 3>>;
using fixed_woodbury = Woodbury<fixed_solver>;
using fixed_rhs = Matrix<double, 3, 1>;
using fixed_result = decltype(std::declval<fixed_woodbury&>().solve(std::declval<const fixed_rhs&>()));
using fixed_float_rhs = Matrix<float, 3, 2, ColMajor>;
using fixed_float_result = decltype(std::declval<fixed_woodbury&>().solve(std::declval<const fixed_float_rhs&>()));
using partial_float_rhs = Matrix<float, Dynamic, 2, ColMajor>;
using partial_float_result = decltype(std::declval<fixed_woodbury&>().solve(std::declval<const partial_float_rhs&>()));
using partial_cols_float_rhs = Matrix<float, 3, Dynamic, RowMajor>;
using partial_cols_float_result =
  decltype(std::declval<fixed_woodbury&>().solve(std::declval<const partial_cols_float_rhs&>()));
using dynamic_float_rhs = Matrix<float, Dynamic, Dynamic, ColMajor>;
using dynamic_float_result = decltype(std::declval<fixed_woodbury&>().solve(std::declval<const dynamic_float_rhs&>()));
using fixed_int_rhs = Matrix<int, 3, 1, ColMajor>;
using fixed_int_result = decltype(std::declval<fixed_woodbury&>().solve(std::declval<const fixed_int_rhs&>()));
using fixed_float_expression_result =
  decltype(std::declval<fixed_woodbury&>().solve(2.0 * std::declval<const fixed_float_rhs&>()));
using fixed_float_container = Matrix<float, 3, 3, ColMajor>;
using fixed_float_view = decltype(std::declval<fixed_float_container&>().template block<3, 2>(0, 0));
using fixed_float_view_result =
  decltype(std::declval<fixed_woodbury&>().solve(std::declval<const fixed_float_view&>()));
using fixed_update = Matrix<double, 3, 1>;
using fixed_inverse_core = Matrix<double, 1, 1>;
using fixed_transpose_update = Matrix<double, 1, 3>;
// the working scalar is explicitly double regardless of right-hand-side storage
static_assert(std::is_same_v<typename fixed_woodbury::Scalar, double>);
// the decomposition owns the supplied base-solver type
static_assert(std::is_same_v<typename fixed_woodbury::SolverType, fixed_solver>);
// the cached update uses dynamic native double storage
static_assert(std::is_same_v<typename fixed_woodbury::MatrixType, Matrix<double, Dynamic, Dynamic>>);
// the small correction system uses native pivoted LU
static_assert(std::is_same_v<typename fixed_woodbury::DenseSolver, PartialPivLU<Matrix<double, Dynamic, Dynamic>>>);
// a fixed column-vector right-hand side returns a fixed owning double vector
static_assert(std::is_same_v<fixed_result, Matrix<double, 3, 1, RowMajor>>);
// float input promotes to double while preserving its fixed shape and column-major layout
static_assert(std::is_same_v<fixed_float_result, Matrix<double, 3, 2, ColMajor>>);
// dynamic rows and fixed columns survive promotion to a double result
static_assert(std::is_same_v<partial_float_result, Matrix<double, Dynamic, 2, ColMajor>>);
// fixed rows and dynamic columns survive promotion to a double result
static_assert(std::is_same_v<partial_cols_float_result, Matrix<double, 3, Dynamic, RowMajor>>);
// fully dynamic input keeps its column-major layout in the double result
static_assert(std::is_same_v<dynamic_float_result, Matrix<double, Dynamic, Dynamic, ColMajor>>);
// integer input returns double coefficients so fractional solutions are representable
static_assert(std::is_same_v<fixed_int_result, Matrix<double, 3, 1, ColMajor>>);
// an expression result retains its static shape and column-major layout
static_assert(std::is_same_v<fixed_float_expression_result, Matrix<double, 3, 2, ColMajor>>);
// a block view produces owning double storage with its fixed block dimensions
static_assert(std::is_same_v<fixed_float_view_result, Matrix<double, 3, 2, ColMajor>>);
// a returned solution must not borrow mutable solver workspaces
static_assert(!std::is_reference_v<fixed_result>);
// the returned concrete matrix follows the native owning-expression lifetime contract
static_assert(fixed_float_result::NestAsRef == 1);

// both dense layouts match explicit low-rank system assembly and direct LU solves
TEST(linear_algebra, woodbury_matches_direct_dense_system) {
    // row-major inputs exercise repeated solves, expression ownership, copy and move behavior
    check_woodbury_against_direct_dense_solve<RowMajor>();
    // column-major inputs exercise the same coefficient oracle and ownership paths
    check_woodbury_against_direct_dense_solve<ColMajor>();
}

// a counting identity backend proves the update is solved once and reused for each right-hand side
TEST(linear_algebra, woodbury_caches_base_update_for_repeated_solves) {
    const auto calls = std::make_shared<int>(0);
    CountingIdentitySolver solver {calls};
    const Matrix<double, 2, 1> u({1.0, 0.0});
    const Matrix<double, 1, 1> inverse_c(1.0);
    const Matrix<double, 1, 2> v({1.0, 0.0});
    const Matrix<double, 2, 1> rhs({2.0, 3.0});

    Woodbury cached(solver, u, inverse_c, v);
    // initialization performs exactly one solve for the single update column
    EXPECT_EQ(*calls, 1);
    // the rank-one update of identity halves only the first right-hand-side coefficient
    expect_matrix_near(cached.solve(rhs), Matrix<double, 2, 1>({1.0, 3.0}));
    // the first right-hand side adds one solve without recomputing the update
    EXPECT_EQ(*calls, 2);
    // doubling the right-hand side doubles the analytic solution using the existing cache
    expect_matrix_near(cached.solve(2.0 * rhs), Matrix<double, 2, 1>({2.0, 6.0}));
    // the second right-hand side adds only one further solve
    EXPECT_EQ(*calls, 3);
}

// fixed, dynamic, view and expression inputs return independent solutions with their declared layouts
TEST(linear_algebra, woodbury_owns_results_and_preserves_rhs_shape_and_storage) {
    using base_matrix = Matrix<double, 3, 3, ColMajor>;
    const base_matrix base({4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0});
    const fixed_update u({1.0, 0.0, 1.0});
    const fixed_inverse_core inverse_c(2.0);
    const fixed_transpose_update v({1.0, 0.0, 1.0});
    const fixed_float_rhs rhs({1.25f, -0.5f, 3.0f, 2.0f, -1.0f, 4.5f});

    fixed_solver base_solver(base);
    fixed_woodbury decomposition(base_solver, u, inverse_c, v);
    const auto member_result = decomposition.solve(rhs);

    const Matrix<double, 1, 1> c(inverse_c.inverse());
    const Matrix<double, 3, 3> full_system(base + u * c * v);
    const PartialPivLU direct_solver(full_system);
    const Matrix<double, 3, 2> expected(direct_solver.solve(Matrix<double, 3, 2>(rhs)));
    // the native cached solve matches LU on the explicitly assembled full system
    expect_matrix_near(member_result, expected, 1.0e-5);

    partial_float_rhs partial(rhs);
    // partially dynamic row storage reproduces the same direct-system oracle
    expect_matrix_near(decomposition.solve(partial), expected, 1.0e-5);
    dynamic_float_rhs dynamic(rhs);
    // fully dynamic storage reproduces the same direct-system oracle
    expect_matrix_near(decomposition.solve(dynamic), expected, 1.0e-5);
    partial_cols_float_rhs partial_cols(rhs);
    // partially dynamic column storage reproduces the same direct-system oracle
    expect_matrix_near(decomposition.solve(partial_cols), expected, 1.0e-5);

    fixed_float_container container;
    for (int row = 0; row < rhs.rows(); ++row) {
        for (int col = 0; col < rhs.cols(); ++col) container(row, col) = rhs(row, col);
        container(row, 2) = 99.0f;
    }
    auto view = container.block<3, 2>(0, 0);
    const auto view_result = decomposition.solve(view);
    for (int row = 0; row < rhs.rows(); ++row) {
        for (int col = 0; col < rhs.cols(); ++col) container(row, col) = 0.0f;
    }
    // overwriting the source block does not alter the already returned solution
    expect_matrix_near(view_result, expected, 1.0e-5);

    const auto expression_result = [&] {
        fixed_float_rhs temporary(rhs);
        return decomposition.solve(2.0 * temporary);
    }();
    // a result from a destroyed local expression retains the doubled direct-system solution
    expect_matrix_near(expression_result, 2.0 * expected, 1.0e-11);
}

// integer right-hand sides produce floating solutions instead of truncating fractional coefficients
TEST(linear_algebra, woodbury_promotes_integer_rhs) {
    const Matrix<double, 2, 2> base({2., 0., 0., 2.});
    const Matrix<double, 2, 1> u({0., 0.});
    const Matrix<double, 1, 1> inverse_c(1.);
    const Matrix<double, 1, 2> v({0., 0.});
    Woodbury decomposition {PartialPivLU(base), u, inverse_c, v};
    const Matrix<int, 2, 1> rhs({1, 3});
    // the analytic solution of 2I x = rhs retains both half-integer coefficients
    expect_matrix_near(decomposition.solve(rhs), Matrix<double, 2, 1>({0.5, 1.5}));
}

// unavailable states, singular corrections and invalid backend results throw typed errors
TEST(linear_algebra, woodbury_default_and_numerical_failures_are_checked) {
    using matrix_type = Matrix<double, Dynamic, Dynamic>;
    const matrix_type identity(Matrix<double, 2, 2>({1.0, 0.0, 0.0, 1.0}));
    PartialPivLU<matrix_type> solver(identity);
    const matrix_type u(Matrix<double, 2, 1>({1.0, 0.0}));
    const matrix_type inverse_c(Matrix<double, 1, 1>(1.0));
    const matrix_type v(Matrix<double, 1, 2>({1.0, 0.0}));
    const matrix_type rhs(Matrix<double, 2, 1>({2.0, 3.0}));

    Woodbury<decltype(solver)> unavailable;
    // an uninitialized decomposition cannot return a solution
    EXPECT_THROW(static_cast<void>(unavailable.solve(rhs)), std::domain_error);

    const matrix_type singular_inverse_c(Matrix<double, 1, 1>(-1.0));
    // a zero scalar correction matrix is rejected as singular
    EXPECT_THROW(static_cast<void>(Woodbury(solver, u, singular_inverse_c, v)), std::domain_error);
    // initialization rejects a backend result with an extra row
    EXPECT_THROW(static_cast<void>(Woodbury(WrongShapeSolver {}, u, inverse_c, v)), std::domain_error);
    // initialization rejects an infinite backend result
    EXPECT_THROW(static_cast<void>(Woodbury(NonfiniteSolver {}, u, inverse_c, v)), std::domain_error);

    Woodbury late_failure(LateWrongShapeSolver {}, u, inverse_c, v);
    // backend shape validation also runs on later right-hand-side solves
    EXPECT_THROW(static_cast<void>(late_failure.solve(rhs)), std::domain_error);
}

// public update shapes and finite coefficients are checked before cache construction or solving
TEST(linear_algebra, woodbury_rejects_invalid_public_inputs) {
    using matrix_type = Matrix<double, Dynamic, Dynamic>;
    const matrix_type identity(Matrix<double, 2, 2>({1.0, 0.0, 0.0, 1.0}));
    PartialPivLU<matrix_type> solver(identity);
    const matrix_type u(Matrix<double, 2, 1>({1.0, 0.0}));
    const matrix_type inverse_c(Matrix<double, 1, 1>(1.0));
    const matrix_type v(Matrix<double, 1, 2>({1.0, 0.0}));

    // an empty base dimension cannot define a low-rank system
    EXPECT_THROW(static_cast<void>(Woodbury(solver, matrix_type(0, 1), inverse_c, v)), std::invalid_argument);
    // the inverse core must be square with the update rank
    EXPECT_THROW(static_cast<void>(Woodbury(solver, u, matrix_type(2, 2), v)), std::invalid_argument);
    // the right update must match both rank and base dimension
    EXPECT_THROW(static_cast<void>(Woodbury(solver, u, inverse_c, matrix_type(2, 2))), std::invalid_argument);

    matrix_type nonfinite_u(u);
    nonfinite_u(0, 0) = std::numeric_limits<double>::quiet_NaN();
    // a NaN update coefficient is rejected before solving with the backend
    EXPECT_THROW(static_cast<void>(Woodbury(solver, nonfinite_u, inverse_c, v)), std::invalid_argument);
    matrix_type nonfinite_inverse_c(inverse_c);
    nonfinite_inverse_c(0, 0) = std::numeric_limits<double>::infinity();
    // an infinite inverse-core coefficient is rejected before correction factorization
    EXPECT_THROW(static_cast<void>(Woodbury(solver, u, nonfinite_inverse_c, v)), std::invalid_argument);

    Woodbury valid(solver, u, inverse_c, v);
    // right-hand-side rows must match the initialized base dimension
    EXPECT_THROW(static_cast<void>(valid.solve(matrix_type(3, 1))), std::invalid_argument);
    // a right-hand side must supply at least one solution column
    EXPECT_THROW(static_cast<void>(valid.solve(matrix_type(2, 0))), std::invalid_argument);
    matrix_type nonfinite_rhs(Matrix<double, 2, 1>({1.0, 2.0}));
    nonfinite_rhs(1, 0) = std::numeric_limits<double>::quiet_NaN();
    // a NaN right-hand-side coefficient is rejected before the base solve
    EXPECT_THROW(static_cast<void>(valid.solve(nonfinite_rhs)), std::invalid_argument);
}

}   // namespace
