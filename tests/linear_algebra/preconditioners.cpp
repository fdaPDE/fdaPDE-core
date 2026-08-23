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
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using namespace fdapde;

using const_view_identity = IdentityPreconditioner<MatrixView<const double, 2, 2>>;
using const_view_diagonal = DiagonalPreconditioner<MatrixView<const double, 2, 2>>;
using fixed_rhs = Matrix<double, 2, 1>;
static_assert(std::is_same_v<typename const_view_identity::Scalar, double>);
static_assert(std::is_same_v<typename const_view_diagonal::Scalar, double>);
static_assert(const_view_identity::Rows == 2 && const_view_identity::Cols == 2);
static_assert(const_view_diagonal::Rows == 2 && const_view_diagonal::Cols == 2);
static_assert(
  !std::is_reference_v<decltype(std::declval<const const_view_identity&>().solve(std::declval<const fixed_rhs&>()))>);
static_assert(
  !std::is_reference_v<decltype(std::declval<const const_view_diagonal&>().solve(std::declval<const fixed_rhs&>()))>);

template <typename Actual, typename Expected> void expect_matrix_equal(const Actual& actual, const Expected& expected) {
    ASSERT_EQ(actual.rows(), expected.rows());
    ASSERT_EQ(actual.cols(), expected.cols());
    for (int row = 0; row < actual.rows(); ++row) {
        for (int col = 0; col < actual.cols(); ++col) EXPECT_DOUBLE_EQ(actual(row, col), expected(row, col));
    }
}

template <typename MatrixType, int StorageOrder> void check_preconditioner_happy_paths(const MatrixType& source) {
    using vector_type = Matrix<double, MatrixType::Rows, 1, StorageOrder>;
    using multiple_rhs_type = Matrix<double, MatrixType::Rows, 2, StorageOrder>;
    const vector_type rhs(std::vector<double> {6.0, 8.0});
    const Matrix<double, 2, 2, StorageOrder> fixed_multiple_rhs({2.0, 4.0, 6.0, 8.0});
    const multiple_rhs_type multiple_rhs(fixed_multiple_rhs);

    const IdentityPreconditioner<MatrixType> identity(source);
    ASSERT_TRUE(identity.valid());
    const auto identity_vector = identity.solve(rhs);
    const auto identity_multiple = identity.solve(multiple_rhs);
    expect_matrix_equal(identity_vector, rhs);
    expect_matrix_equal(identity_multiple, multiple_rhs);

    const DiagonalPreconditioner<MatrixType> diagonal(source);
    ASSERT_TRUE(diagonal.valid());
    const vector_type expected_vector(std::vector<double> {3.0, 2.0});
    const Matrix<double, 2, 2, StorageOrder> fixed_expected_multiple({1.0, 2.0, 1.5, 2.0});
    const multiple_rhs_type expected_multiple(fixed_expected_multiple);
    expect_matrix_equal(diagonal.solve(rhs), expected_vector);
    expect_matrix_equal(diagonal.solve(multiple_rhs), expected_multiple);
}

template <int StorageOrder> void check_preconditioner_shapes_and_lifetimes() {
    using fixed_matrix = Matrix<double, 2, 2, StorageOrder>;
    const fixed_matrix source({2.0, 1.0, -1.0, 4.0});
    check_preconditioner_happy_paths<fixed_matrix, StorageOrder>(source);

    const Matrix<double, Dynamic, Dynamic, StorageOrder> dynamic(source);
    check_preconditioner_happy_paths<decltype(dynamic), StorageOrder>(dynamic);
    const Matrix<double, Dynamic, 2, StorageOrder> dynamic_rows(source);
    check_preconditioner_happy_paths<decltype(dynamic_rows), StorageOrder>(dynamic_rows);
    const Matrix<double, 2, Dynamic, StorageOrder> dynamic_cols(source);
    check_preconditioner_happy_paths<decltype(dynamic_cols), StorageOrder>(dynamic_cols);

    fixed_matrix mutable_source(source);
    const DiagonalPreconditioner<fixed_matrix> retained_diagonal(mutable_source);
    mutable_source(0, 0) = 100.0;
    const Matrix<double, 2, 1, StorageOrder> rhs(std::vector<double> {6.0, 8.0});
    expect_matrix_equal(
      retained_diagonal.solve(rhs), Matrix<double, 2, 1, StorageOrder>(std::vector<double> {3.0, 2.0}));

    const auto retained_expression_preconditioner = [] {
        const fixed_matrix matrix({2.0, 1.0, -1.0, 4.0});
        const fixed_matrix zero = fixed_matrix::Zero();
        using expression_type = decltype(matrix + zero);
        return DiagonalPreconditioner<expression_type>(matrix + zero);
    }();
    auto owned_solution = retained_expression_preconditioner.solve(rhs + Matrix<double, 2, 1, StorageOrder>::Zero());
    expect_matrix_equal(owned_solution, Matrix<double, 2, 1, StorageOrder>(std::vector<double> {3.0, 2.0}));

    const MatrixView<const double, 2, 2, StorageOrder> const_view(source.data());
    const IdentityPreconditioner<decltype(const_view)> const_view_identity(const_view);
    const DiagonalPreconditioner<decltype(const_view)> const_view_diagonal(const_view);
    expect_matrix_equal(const_view_identity.solve(rhs), rhs);
    expect_matrix_equal(
      const_view_diagonal.solve(rhs), Matrix<double, 2, 1, StorageOrder>(std::vector<double> {3.0, 2.0}));
}

template <int StorageOrder> void check_preconditioner_failure_contracts() {
    using fixed_matrix = Matrix<double, 2, 2, StorageOrder>;
    using vector_type = Matrix<double, 2, 1, StorageOrder>;
    const fixed_matrix valid({2.0, 0.0, 0.0, 4.0});
    const vector_type rhs(std::vector<double> {6.0, 8.0});

    IdentityPreconditioner<fixed_matrix> identity;
    DiagonalPreconditioner<fixed_matrix> diagonal;
    EXPECT_FALSE(identity.valid());
    EXPECT_FALSE(diagonal.valid());
    EXPECT_THROW(static_cast<void>(identity.solve(rhs)), std::domain_error);
    EXPECT_THROW(static_cast<void>(diagonal.solve(rhs)), std::domain_error);

    identity.compute(valid);
    diagonal.compute(valid);
    ASSERT_TRUE(identity.valid());
    ASSERT_TRUE(diagonal.valid());
    Matrix<double, Dynamic, Dynamic, StorageOrder> wrong_shape(3, 3);
    EXPECT_THROW(identity.compute(wrong_shape), std::invalid_argument);
    EXPECT_FALSE(identity.valid());
    EXPECT_THROW(static_cast<void>(identity.solve(rhs)), std::domain_error);
    EXPECT_THROW(diagonal.compute(wrong_shape), std::invalid_argument);
    EXPECT_FALSE(diagonal.valid());
    EXPECT_THROW(static_cast<void>(diagonal.solve(rhs)), std::domain_error);

    Matrix<double, Dynamic, Dynamic, StorageOrder> empty;
    EXPECT_THROW(identity.compute(empty), std::invalid_argument);
    EXPECT_FALSE(identity.valid());
    Matrix<double, Dynamic, Dynamic, StorageOrder> rectangular(2, 3);
    EXPECT_THROW(diagonal.compute(rectangular), std::invalid_argument);
    EXPECT_FALSE(diagonal.valid());

    identity.compute(valid);
    diagonal.compute(valid);
    Matrix<double, Dynamic, 1, StorageOrder> wrong_rows(3);
    Matrix<double, Dynamic, Dynamic, StorageOrder> no_columns(2, 0);
    EXPECT_THROW(static_cast<void>(identity.solve(wrong_rows)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(diagonal.solve(wrong_rows)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(identity.solve(no_columns)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(diagonal.solve(no_columns)), std::invalid_argument);

    fixed_matrix invalid_diagonal(valid);
    invalid_diagonal(0, 0) = 0.0;
    EXPECT_THROW(diagonal.compute(invalid_diagonal), std::domain_error);
    EXPECT_FALSE(diagonal.valid());
    EXPECT_THROW(static_cast<void>(diagonal.solve(rhs)), std::domain_error);

    invalid_diagonal = valid;
    invalid_diagonal(0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(diagonal.compute(invalid_diagonal), std::invalid_argument);
    EXPECT_FALSE(diagonal.valid());
    invalid_diagonal(0, 0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW(diagonal.compute(invalid_diagonal), std::invalid_argument);
    EXPECT_FALSE(diagonal.valid());
    invalid_diagonal(0, 0) = std::numeric_limits<double>::denorm_min();
    EXPECT_THROW(diagonal.compute(invalid_diagonal), std::domain_error);
    EXPECT_FALSE(diagonal.valid());
}

TEST(linear_algebra, preconditioners) {
    check_preconditioner_shapes_and_lifetimes<RowMajor>();
    check_preconditioner_shapes_and_lifetimes<ColMajor>();
    check_preconditioner_failure_contracts<RowMajor>();
    check_preconditioner_failure_contracts<ColMajor>();
}

}   // namespace
