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

#include <limits>
#include <memory>
#include <type_traits>
#include <utility>

namespace {

using namespace fdapde;

template <typename Actual, typename Expected>
void expect_matrix_near(const Actual& actual, const Expected& expected, double tolerance = 1.0e-11) {
    ASSERT_EQ(actual.rows(), expected.rows());
    ASSERT_EQ(actual.cols(), expected.cols());
    for (int row = 0; row < actual.rows(); ++row) {
        for (int col = 0; col < actual.cols(); ++col) {
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
    expect_matrix_near(cached.solve(rhs), expected);
    expect_matrix_near(cached.solve(second_rhs), second_expected);
    expect_matrix_near(woodbury_system_solve(base_solver, u, inverse_c, v, rhs), expected);
    const PartialPivLU<base_matrix> const_base_solver(base);
    expect_matrix_near(woodbury_system_solve(const_base_solver, u, inverse_c, v, rhs), expected);

    auto temporary_owned = [&] {
        return Woodbury(
          PartialPivLU(base), original_u + update_matrix::Zero(), inverse_c + inverse_core::Zero(),
          original_v + transpose_update::Zero());
    }();
    expect_matrix_near(temporary_owned.solve(rhs), expected);

    base_matrix identity;
    identity.set_zero();
    for (int i = 0; i < 3; ++i) identity(i, i) = 1.0;
    base_solver.compute(identity);
    u.set_zero();
    v.set_zero();
    expect_matrix_near(cached.solve(rhs), expected);

    auto copied = cached;
    auto moved = std::move(copied);
    Woodbury<decltype(base_solver)> assigned;
    assigned = cached;
    EXPECT_THROW(static_cast<void>(copied.solve(rhs)), std::domain_error);
    expect_matrix_near(moved.solve(rhs), expected);
    expect_matrix_near(assigned.solve(rhs), expected);
}

TEST(linear_algebra, woodbury_accepts_backend_neutral_gmres_solver) {
    using matrix_type = Matrix<double, 3, 3>;
    const matrix_type base({4.0, 1.0, 0.0, 1.0, 3.0, 1.0, 0.0, 1.0, 2.0});
    const Matrix<double, 3, 1> u({1.0, 0.0, 1.0});
    const Matrix<double, 1, 1> inverse_c(2.0);
    const Matrix<double, 1, 3> v({1.0, 0.0, 1.0});
    const Matrix<double, 3, 1> rhs({1.0, 3.0, 2.0});

    const Matrix<double, 1, 1> c(inverse_c.inverse());
    const matrix_type full_system(base + u * c * v);
    const PartialPivLU<matrix_type> direct_solver(full_system);
    const Matrix<double, 3, 1> expected(direct_solver.solve(rhs));
    GMRES base_solver(base, IdentityPreconditioner<matrix_type> {}, 50, 3, 1.0e-12);
    Woodbury decomposition(std::move(base_solver), u, inverse_c, v);
    expect_matrix_near(decomposition.solve(rhs), expected);
}

struct CountingIdentitySolver {
    std::shared_ptr<int> solve_calls;

    template <typename RhsType> auto solve(const MatrixExpr<RhsType>& rhs) {
        ++*solve_calls;
        return Matrix<double, Dynamic, Dynamic>(rhs);
    }
};

struct WrongShapeSolver {
    template <typename RhsType> auto solve(const MatrixExpr<RhsType>& rhs) {
        return Matrix<double, Dynamic, Dynamic>(rhs.rows() + 1, rhs.cols());
    }
};

struct NonfiniteSolver {
    template <typename RhsType> auto solve(const MatrixExpr<RhsType>& rhs) {
        Matrix<double, Dynamic, Dynamic> result(rhs);
        result(0, 0) = std::numeric_limits<double>::infinity();
        return result;
    }
};

struct LateWrongShapeSolver {
    int solve_calls = 0;

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
static_assert(std::is_same_v<typename fixed_woodbury::Scalar, double>);
static_assert(std::is_same_v<typename fixed_woodbury::SparseSolver, fixed_solver>);
static_assert(std::is_same_v<typename fixed_woodbury::MatrixType, Matrix<double, Dynamic, Dynamic>>);
static_assert(std::is_same_v<fixed_result, Matrix<double, Dynamic, Dynamic>>);
static_assert(!std::is_reference_v<fixed_result>);

TEST(linear_algebra, woodbury_matches_direct_dense_system) {
    check_woodbury_against_direct_dense_solve<RowMajor>();
    check_woodbury_against_direct_dense_solve<ColMajor>();
}

TEST(linear_algebra, woodbury_caches_base_update_for_repeated_solves) {
    const auto calls = std::make_shared<int>(0);
    CountingIdentitySolver solver {calls};
    const Matrix<double, 2, 1> u({1.0, 0.0});
    const Matrix<double, 1, 1> inverse_c(1.0);
    const Matrix<double, 1, 2> v({1.0, 0.0});
    const Matrix<double, 2, 1> rhs({2.0, 3.0});

    Woodbury cached(solver, u, inverse_c, v);
    EXPECT_EQ(*calls, 1);
    expect_matrix_near(cached.solve(rhs), Matrix<double, 2, 1>({1.0, 3.0}));
    EXPECT_EQ(*calls, 2);
    expect_matrix_near(cached.solve(2.0 * rhs), Matrix<double, 2, 1>({2.0, 6.0}));
    EXPECT_EQ(*calls, 3);
}

TEST(linear_algebra, woodbury_default_and_numerical_failures_are_checked) {
    using matrix_type = Matrix<double, Dynamic, Dynamic>;
    const matrix_type identity(Matrix<double, 2, 2>({1.0, 0.0, 0.0, 1.0}));
    PartialPivLU<matrix_type> solver(identity);
    const matrix_type u(Matrix<double, 2, 1>({1.0, 0.0}));
    const matrix_type inverse_c(Matrix<double, 1, 1>(1.0));
    const matrix_type v(Matrix<double, 1, 2>({1.0, 0.0}));
    const matrix_type rhs(Matrix<double, 2, 1>({2.0, 3.0}));

    Woodbury<decltype(solver)> unavailable;
    EXPECT_THROW(static_cast<void>(unavailable.solve(rhs)), std::domain_error);

    const matrix_type singular_inverse_c(Matrix<double, 1, 1>(-1.0));
    EXPECT_THROW(static_cast<void>(Woodbury(solver, u, singular_inverse_c, v)), std::domain_error);
    EXPECT_THROW(static_cast<void>(Woodbury(WrongShapeSolver {}, u, inverse_c, v)), std::domain_error);
    EXPECT_THROW(static_cast<void>(Woodbury(NonfiniteSolver {}, u, inverse_c, v)), std::domain_error);

    Woodbury late_failure(LateWrongShapeSolver {}, u, inverse_c, v);
    EXPECT_THROW(static_cast<void>(late_failure.solve(rhs)), std::domain_error);
}

TEST(linear_algebra, woodbury_contracts_remain_active_without_debug_assertions) {
    using matrix_type = Matrix<double, Dynamic, Dynamic>;
    const matrix_type identity(Matrix<double, 2, 2>({1.0, 0.0, 0.0, 1.0}));
    PartialPivLU<matrix_type> solver(identity);
    const matrix_type u(Matrix<double, 2, 1>({1.0, 0.0}));
    const matrix_type inverse_c(Matrix<double, 1, 1>(1.0));
    const matrix_type v(Matrix<double, 1, 2>({1.0, 0.0}));

    EXPECT_THROW(static_cast<void>(Woodbury(solver, matrix_type(0, 1), inverse_c, v)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(Woodbury(solver, u, matrix_type(2, 2), v)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(Woodbury(solver, u, inverse_c, matrix_type(2, 2))), std::invalid_argument);

    matrix_type nonfinite_u(u);
    nonfinite_u(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(static_cast<void>(Woodbury(solver, nonfinite_u, inverse_c, v)), std::invalid_argument);
    matrix_type nonfinite_inverse_c(inverse_c);
    nonfinite_inverse_c(0, 0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW(static_cast<void>(Woodbury(solver, u, nonfinite_inverse_c, v)), std::invalid_argument);

    Woodbury valid(solver, u, inverse_c, v);
    EXPECT_THROW(static_cast<void>(valid.solve(matrix_type(3, 1))), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(valid.solve(matrix_type(2, 0))), std::invalid_argument);
    matrix_type nonfinite_rhs(Matrix<double, 2, 1>({1.0, 2.0}));
    nonfinite_rhs(1, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(static_cast<void>(valid.solve(nonfinite_rhs)), std::invalid_argument);
}

}   // namespace
