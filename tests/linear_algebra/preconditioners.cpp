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
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace fdapde;

using const_view_identity = IdentityPreconditioner<MatrixView<const double, 2, 2>>;
using const_view_diagonal = DiagonalPreconditioner<MatrixView<const double, 2, 2>>;
using fixed_rhs = Matrix<double, 2, 1>;
// const-view identity preconditioning exposes an unqualified value scalar
static_assert(std::is_same_v<typename const_view_identity::Scalar, double>);
// const-view diagonal preconditioning exposes an unqualified value scalar
static_assert(std::is_same_v<typename const_view_diagonal::Scalar, double>);
// identity preconditioning retains both fixed extents
static_assert(const_view_identity::Rows == 2 && const_view_identity::Cols == 2);
// diagonal preconditioning retains both fixed extents
static_assert(const_view_diagonal::Rows == 2 && const_view_diagonal::Cols == 2);
// identity solve returns a value rather than a borrowed reference
static_assert(
  !std::is_reference_v<decltype(std::declval<const const_view_identity&>().solve(std::declval<const fixed_rhs&>()))>);
// diagonal solve returns a value rather than a borrowed reference
static_assert(
  !std::is_reference_v<decltype(std::declval<const const_view_diagonal&>().solve(std::declval<const fixed_rhs&>()))>);

// compares matrix shapes and coefficients with an explicit expected result
template <typename Actual, typename Expected> void expect_matrix_equal(const Actual& actual, const Expected& expected) {
    // row counts must agree before coefficient indexing
    ASSERT_EQ(actual.rows(), expected.rows());
    // column counts must agree before coefficient indexing
    ASSERT_EQ(actual.cols(), expected.cols());
    for (int row = 0; row < actual.rows(); ++row) {
        for (int col = 0; col < actual.cols(); ++col) {
            // each coefficient agrees with the explicit identity or inverse-diagonal oracle
            EXPECT_DOUBLE_EQ(actual(row, col), expected(row, col));
        }
    }
}

// checks vector and multiple-right-hand-side actions for a supplied shape
template <typename MatrixType, int StorageOrder> void check_preconditioner_happy_paths(const MatrixType& source) {
    using vector_type = Matrix<double, MatrixType::Rows, 1, StorageOrder>;
    using multiple_rhs_type = Matrix<double, MatrixType::Rows, 2, StorageOrder>;
    const vector_type rhs(std::vector<double> {6.0, 8.0});
    const Matrix<double, 2, 2, StorageOrder> fixed_multiple_rhs({2.0, 4.0, 6.0, 8.0});
    const multiple_rhs_type multiple_rhs(fixed_multiple_rhs);

    const IdentityPreconditioner<MatrixType> identity(source);
    // a valid square matrix initializes identity preconditioning
    ASSERT_TRUE(identity.valid());
    const auto identity_vector = identity.solve(rhs);
    const auto identity_multiple = identity.solve(multiple_rhs);
    // identity action preserves every vector coefficient
    expect_matrix_equal(identity_vector, rhs);
    // identity action preserves both right-hand-side columns
    expect_matrix_equal(identity_multiple, multiple_rhs);

    const DiagonalPreconditioner<MatrixType> diagonal(source);
    // a finite nonzero diagonal initializes diagonal preconditioning
    ASSERT_TRUE(diagonal.valid());
    const vector_type expected_vector(std::vector<double> {3.0, 2.0});
    const Matrix<double, 2, 2, StorageOrder> fixed_expected_multiple({1.0, 2.0, 1.5, 2.0});
    const multiple_rhs_type expected_multiple(fixed_expected_multiple);
    // each vector entry is divided by its corresponding diagonal entry
    expect_matrix_equal(diagonal.solve(rhs), expected_vector);
    // each right-hand-side column is scaled by the same inverse diagonal
    expect_matrix_equal(diagonal.solve(multiple_rhs), expected_multiple);
}

// checks fixed and dynamic extents, const views and independent result ownership
template <int StorageOrder> void check_preconditioner_shapes_and_lifetimes() {
    using fixed_matrix = Matrix<double, 2, 2, StorageOrder>;
    const fixed_matrix source({2.0, 1.0, -1.0, 4.0});
    // check both preconditioners with fully fixed dimensions
    check_preconditioner_happy_paths<fixed_matrix, StorageOrder>(source);

    const Matrix<double, Dynamic, Dynamic, StorageOrder> dynamic(source);
    // check the same actions with fully dynamic dimensions
    check_preconditioner_happy_paths<decltype(dynamic), StorageOrder>(dynamic);
    const Matrix<double, Dynamic, 2, StorageOrder> dynamic_rows(source);
    // check the same actions with dynamic rows only
    check_preconditioner_happy_paths<decltype(dynamic_rows), StorageOrder>(dynamic_rows);
    const Matrix<double, 2, Dynamic, StorageOrder> dynamic_cols(source);
    // check the same actions with dynamic columns only
    check_preconditioner_happy_paths<decltype(dynamic_cols), StorageOrder>(dynamic_cols);

    fixed_matrix mutable_source(source);
    const DiagonalPreconditioner<fixed_matrix> retained_diagonal(mutable_source);
    mutable_source(0, 0) = 100.0;
    const Matrix<double, 2, 1, StorageOrder> rhs(std::vector<double> {6.0, 8.0});
    // source mutation does not alter the stored inverse diagonal
    expect_matrix_equal(
      retained_diagonal.solve(rhs), Matrix<double, 2, 1, StorageOrder>(std::vector<double> {3.0, 2.0}));

    const auto retained_expression_preconditioner = [] {
        const fixed_matrix matrix({2.0, 1.0, -1.0, 4.0});
        const fixed_matrix zero = fixed_matrix::Zero();
        using expression_type = decltype(matrix + zero);
        return DiagonalPreconditioner<expression_type>(matrix + zero);
    }();
    auto owned_solution = retained_expression_preconditioner.solve(rhs + Matrix<double, 2, 1, StorageOrder>::Zero());
    // the result remains correct after destruction of temporary source expressions
    expect_matrix_equal(owned_solution, Matrix<double, 2, 1, StorageOrder>(std::vector<double> {3.0, 2.0}));

    const MatrixView<const double, 2, 2, StorageOrder> const_view(source.data());
    const IdentityPreconditioner<decltype(const_view)> const_view_identity(const_view);
    const DiagonalPreconditioner<decltype(const_view)> const_view_diagonal(const_view);
    // identity action through a const view preserves the right-hand side
    expect_matrix_equal(const_view_identity.solve(rhs), rhs);
    // diagonal action through a const view matches the explicit divided coefficients
    expect_matrix_equal(
      const_view_diagonal.solve(rhs), Matrix<double, 2, 1, StorageOrder>(std::vector<double> {3.0, 2.0}));
}

// checks unavailable states, shape rejection and invalid diagonal entries
template <int StorageOrder> void check_preconditioner_failure_contracts() {
    using fixed_matrix = Matrix<double, 2, 2, StorageOrder>;
    using vector_type = Matrix<double, 2, 1, StorageOrder>;
    const fixed_matrix valid({2.0, 0.0, 0.0, 4.0});
    const vector_type rhs(std::vector<double> {6.0, 8.0});

    IdentityPreconditioner<fixed_matrix> identity;
    DiagonalPreconditioner<fixed_matrix> diagonal;
    // a default identity preconditioner has no matrix shape
    EXPECT_FALSE(identity.valid());
    // a default diagonal preconditioner has no inverse diagonal
    EXPECT_FALSE(diagonal.valid());
    // identity solve before compute is rejected
    EXPECT_THROW(static_cast<void>(identity.solve(rhs)), std::domain_error);
    // diagonal solve before compute is rejected
    EXPECT_THROW(static_cast<void>(diagonal.solve(rhs)), std::domain_error);

    identity.compute(valid);
    diagonal.compute(valid);
    // valid compute activates identity preconditioning
    ASSERT_TRUE(identity.valid());
    // valid compute activates diagonal preconditioning
    ASSERT_TRUE(diagonal.valid());
    Matrix<double, Dynamic, Dynamic, StorageOrder> wrong_shape(3, 3);
    // identity compute rejects a runtime order conflicting with its static extent
    EXPECT_THROW(identity.compute(wrong_shape), std::invalid_argument);
    // failed identity compute invalidates the prior state
    EXPECT_FALSE(identity.valid());
    // identity solve cannot reuse state after failed compute
    EXPECT_THROW(static_cast<void>(identity.solve(rhs)), std::domain_error);
    // diagonal compute rejects a runtime order conflicting with its static extent
    EXPECT_THROW(diagonal.compute(wrong_shape), std::invalid_argument);
    // failed diagonal compute invalidates the prior state
    EXPECT_FALSE(diagonal.valid());
    // diagonal solve cannot reuse state after failed compute
    EXPECT_THROW(static_cast<void>(diagonal.solve(rhs)), std::domain_error);

    Matrix<double, Dynamic, Dynamic, StorageOrder> empty;
    // identity compute rejects an empty matrix
    EXPECT_THROW(identity.compute(empty), std::invalid_argument);
    // empty-input rejection leaves identity preconditioning unavailable
    EXPECT_FALSE(identity.valid());
    Matrix<double, Dynamic, Dynamic, StorageOrder> rectangular(2, 3);
    // diagonal compute rejects a rectangular matrix
    EXPECT_THROW(diagonal.compute(rectangular), std::invalid_argument);
    // rectangular-input rejection leaves diagonal preconditioning unavailable
    EXPECT_FALSE(diagonal.valid());

    identity.compute(valid);
    diagonal.compute(valid);
    Matrix<double, Dynamic, 1, StorageOrder> wrong_rows(3);
    Matrix<double, Dynamic, Dynamic, StorageOrder> no_columns(2, 0);
    // identity solve rejects an incompatible row count
    EXPECT_THROW(static_cast<void>(identity.solve(wrong_rows)), std::invalid_argument);
    // diagonal solve rejects an incompatible row count
    EXPECT_THROW(static_cast<void>(diagonal.solve(wrong_rows)), std::invalid_argument);
    // identity solve rejects a right-hand side with no columns
    EXPECT_THROW(static_cast<void>(identity.solve(no_columns)), std::invalid_argument);
    // diagonal solve rejects a right-hand side with no columns
    EXPECT_THROW(static_cast<void>(diagonal.solve(no_columns)), std::invalid_argument);

    fixed_matrix invalid_diagonal(valid);
    invalid_diagonal(0, 0) = 0.0;
    // a zero diagonal entry is rejected before reciprocal evaluation
    EXPECT_THROW(diagonal.compute(invalid_diagonal), std::domain_error);
    // zero-diagonal rejection leaves the preconditioner unavailable
    EXPECT_FALSE(diagonal.valid());
    // solve cannot reuse the earlier inverse after zero-diagonal rejection
    EXPECT_THROW(static_cast<void>(diagonal.solve(rhs)), std::domain_error);

    invalid_diagonal = valid;
    invalid_diagonal(0, 0) = std::numeric_limits<double>::quiet_NaN();
    // a NaN diagonal entry is rejected
    EXPECT_THROW(diagonal.compute(invalid_diagonal), std::invalid_argument);
    // rejecting NaN leaves the preconditioner unavailable
    EXPECT_FALSE(diagonal.valid());
    invalid_diagonal(0, 0) = std::numeric_limits<double>::infinity();
    // an infinite diagonal entry is rejected
    EXPECT_THROW(diagonal.compute(invalid_diagonal), std::invalid_argument);
    // infinite-input rejection leaves the preconditioner unavailable
    EXPECT_FALSE(diagonal.valid());
    invalid_diagonal(0, 0) = std::numeric_limits<double>::denorm_min();
    // a finite subnormal diagonal whose reciprocal overflows is rejected
    EXPECT_THROW(diagonal.compute(invalid_diagonal), std::domain_error);
    // reciprocal overflow leaves the preconditioner unavailable
    EXPECT_FALSE(diagonal.valid());
}

// checks values and ownership with both dense storage orders
TEST(linear_algebra, preconditioners_values_and_ownership) {
    // run the shapes and lifetimes oracles with RowMajor input storage
    check_preconditioner_shapes_and_lifetimes<RowMajor>();
    // run the shapes and lifetimes oracles with ColMajor input storage
    check_preconditioner_shapes_and_lifetimes<ColMajor>();
}

// checks contracts with both dense storage orders
TEST(linear_algebra, preconditioners_contracts) {
    // run the failure contracts oracles with RowMajor input storage
    check_preconditioner_failure_contracts<RowMajor>();
    // run the failure contracts oracles with ColMajor input storage
    check_preconditioner_failure_contracts<ColMajor>();
}

}   // namespace
