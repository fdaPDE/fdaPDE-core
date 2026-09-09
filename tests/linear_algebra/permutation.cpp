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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#include <fdaPDE/linear_algebra.h>
#include <gtest/gtest.h>

#include <limits>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

using namespace fdapde;

namespace {

template <typename MatrixType>
concept permits_coefficient_write = requires(MatrixType& matrix) { matrix(0, 0) = 1; };

template <typename MatrixType>
concept permits_permutation_write = requires(MatrixType& matrix) { matrix.permutation()[0] = 1; };

template <typename MatrixType>
concept exposes_owning_rvalue_derived = requires(MatrixType& matrix) { std::move(matrix).derived(); };

template <typename MatrixType>
concept permits_owning_rvalue_inverse = requires(MatrixType& matrix) { std::move(matrix).inverse(); };

template <typename MatrixType>
concept permits_owning_rvalue_assignment = requires(MatrixType& lhs, MatrixType& rhs) { std::move(lhs) = rhs; };

template <typename Permutation, typename Dense>
concept permits_left_owning_rvalue_action =
  requires(Permutation& permutation, Dense& dense) { std::move(permutation) * dense; };

template <typename Permutation, typename Dense>
concept permits_right_owning_rvalue_action =
  requires(Permutation& permutation, Dense& dense) { dense * std::move(permutation); };

using fixed_permutation = PermutationMatrix<3, 3>;
using fixed_dense = Matrix<double, 3, 3>;

// checks at compile time: fixed_permutation::ReadOnly == 1
static_assert(fixed_permutation::ReadOnly == 1);
// checks at compile time: fixed_permutation::NestAsRef == 1
static_assert(fixed_permutation::NestAsRef == 1);
// checks at compile time: is_permutation_matrix_v<const fixed_permutation&>
static_assert(is_permutation_matrix_v<const fixed_permutation&>);
// checks at compile time: is_orthogonal_matrix_v<const fixed_permutation&>
static_assert(is_orthogonal_matrix_v<const fixed_permutation&>);
// checks at compile time: !permits_coefficient_write<fixed_permutation>
static_assert(!permits_coefficient_write<fixed_permutation>);
// checks at compile time: !permits_permutation_write<fixed_permutation>
static_assert(!permits_permutation_write<fixed_permutation>);
// checks at compile time: !exposes_owning_rvalue_derived<fixed_permutation>
static_assert(!exposes_owning_rvalue_derived<fixed_permutation>);
// checks at compile time: !permits_owning_rvalue_inverse<fixed_permutation>
static_assert(!permits_owning_rvalue_inverse<fixed_permutation>);
// checks at compile time: !permits_owning_rvalue_assignment<fixed_permutation>
static_assert(!permits_owning_rvalue_assignment<fixed_permutation>);
// checks at compile time: !permits_left_owning_rvalue_action<fixed_permutation, fixed_dense>
static_assert(!permits_left_owning_rvalue_action<fixed_permutation, fixed_dense>);
// checks at compile time: !permits_right_owning_rvalue_action<fixed_permutation, fixed_dense>
static_assert(!permits_right_owning_rvalue_action<fixed_permutation, fixed_dense>);

template <typename Actual, typename Expected> void expect_matrix_equal(const Actual& actual, const Expected& expected) {
    // compares actual.rows(), expected.rows() using eq semantics
    ASSERT_EQ(actual.rows(), expected.rows());
    // compares actual.cols(), expected.cols() using eq semantics
    ASSERT_EQ(actual.cols(), expected.cols());
    for (int i = 0; i < actual.rows(); ++i) {
        // compares actual(i, j), expected(i, j) using eq semantics
        for (int j = 0; j < actual.cols(); ++j) { EXPECT_EQ(actual(i, j), expected(i, j)); }
    }
}

template <int StorageOrder> void check_permutation_actions_and_dispatch() {
    const int mapping[] = {2, 0, 1};
    const PermutationMatrix<3, 3> permutation(mapping);

    const Matrix<double, 3, 2, StorageOrder> left_operand({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    const Matrix<double, 3, 2, StorageOrder> expected_left({5.0, 6.0, 1.0, 2.0, 3.0, 4.0});
    const Matrix<double, 3, 2, StorageOrder> left_result(permutation * left_operand);
    expect_matrix_equal(left_result, expected_left);

    const Matrix<double, 2, 3, StorageOrder> right_operand({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    const Matrix<double, 2, 3, StorageOrder> expected_right({2.0, 3.0, 1.0, 5.0, 6.0, 4.0});
    const Matrix<double, 2, 3, StorageOrder> right_result(right_operand * permutation);
    expect_matrix_equal(right_result, expected_right);
    // checks at compile time: std::is_same_v<typename decltype(right_result)::Scalar, double>
    static_assert(std::is_same_v<typename decltype(right_result)::Scalar, double>);

    Matrix<double, 3, 2, StorageOrder> aliased_left(left_operand);
    aliased_left = permutation * aliased_left;
    expect_matrix_equal(aliased_left, expected_left);
    Matrix<double, 2, 3, StorageOrder> aliased_right(right_operand);
    aliased_right = aliased_right * permutation;
    expect_matrix_equal(aliased_right, expected_right);

    const double identity_data[] = {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
    const OrthogonalMatrix<double, 3, 3, StorageOrder> orthogonal(identity_data, checked);
    // checks at compile time: is_orthogonal_matrix_v<decltype(permutation * orthogonal)>
    static_assert(is_orthogonal_matrix_v<decltype(permutation * orthogonal)>);
    // checks at compile time: is_orthogonal_matrix_v<decltype(orthogonal * permutation)>
    static_assert(is_orthogonal_matrix_v<decltype(orthogonal * permutation)>);
    expect_matrix_equal(
      Matrix<double, 3, 3, StorageOrder>(permutation * orthogonal), Matrix<double, 3, 3, StorageOrder>(permutation));
    expect_matrix_equal(
      Matrix<double, 3, 3, StorageOrder>(orthogonal * permutation), Matrix<double, 3, 3, StorageOrder>(permutation));

    const DiagonalMatrix<double, 3> diagonal({2.0, 3.0, 4.0});
    const LowerTriangularMatrix<double, 3, 3> triangular({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    const SymmetricMatrix<double, 3, 3> symmetric({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    const Matrix<double, 3, 3, StorageOrder> dense_permutation(permutation);
    const Matrix<double, 3, 3, StorageOrder> dense_diagonal(diagonal);
    const Matrix<double, 3, 3, StorageOrder> dense_triangular(triangular);
    const Matrix<double, 3, 3, StorageOrder> dense_symmetric(symmetric);
    expect_matrix_equal(
      Matrix<double, 3, 3, StorageOrder>(permutation * diagonal),
      Matrix<double, 3, 3, StorageOrder>(dense_permutation * dense_diagonal));
    expect_matrix_equal(
      Matrix<double, 3, 3, StorageOrder>(diagonal * permutation),
      Matrix<double, 3, 3, StorageOrder>(dense_diagonal * dense_permutation));
    expect_matrix_equal(
      Matrix<double, 3, 3, StorageOrder>(triangular * permutation),
      Matrix<double, 3, 3, StorageOrder>(dense_triangular * dense_permutation));
    expect_matrix_equal(
      Matrix<double, 3, 3, StorageOrder>(permutation * triangular),
      Matrix<double, 3, 3, StorageOrder>(dense_permutation * dense_triangular));
    expect_matrix_equal(
      Matrix<double, 3, 3, StorageOrder>(permutation * symmetric),
      Matrix<double, 3, 3, StorageOrder>(dense_permutation * dense_symmetric));
    expect_matrix_equal(
      Matrix<double, 3, 3, StorageOrder>(symmetric * permutation),
      Matrix<double, 3, 3, StorageOrder>(dense_symmetric * dense_permutation));
}

// verifies permutation through the public algebra API
TEST(linear_algebra, permutation) {
    const int mapping[] = {2, 0, 1};
    const fixed_permutation permutation(mapping);
    // compares permutation.permutation(), (Vector<int, 3>({2, 0, 1})) using eq semantics
    EXPECT_EQ(permutation.permutation(), (Vector<int, 3>({2, 0, 1})));
    // compares permutation.image(0), 2 using eq semantics
    EXPECT_EQ(permutation.image(0), 2);
    // compares permutation.determinant(), 1 using eq semantics
    EXPECT_EQ(permutation.determinant(), 1);
    // compares the computed and expected values within the stated absolute tolerance
    EXPECT_NEAR(permutation.norm(), fdapde::sqrt(3.0), 1.0e-12);

    const auto inverse = permutation.inverse();
    // checks at compile time: is_permutation_matrix_v<decltype(inverse)>
    static_assert(is_permutation_matrix_v<decltype(inverse)>);
    // compares inverse.permutation(), (Vector<int, 3>({1, 2, 0})) using eq semantics
    EXPECT_EQ(inverse.permutation(), (Vector<int, 3>({1, 2, 0})));
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(inverse.image(-1)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(inverse(0, 3)), std::out_of_range);

    const int transposition_mapping[] = {1, 0, 2};
    const fixed_permutation transposition(transposition_mapping);
    // compares transposition.determinant(), -1 using eq semantics
    EXPECT_EQ(transposition.determinant(), -1);
    const auto composition = permutation * transposition;
    // checks at compile time: is_permutation_matrix_v<decltype(composition)>
    static_assert(is_permutation_matrix_v<decltype(composition)>);
    // compares composition.permutation(), (Vector<int, 3>({2, 1, 0})) using eq semantics
    EXPECT_EQ(composition.permutation(), (Vector<int, 3>({2, 1, 0})));

    const fixed_permutation fixed_identity;
    const PermutationMatrix<3, Dynamic> fixed_rows_identity;
    const PermutationMatrix<Dynamic, 3> fixed_cols_identity;
    // compares fixed_identity.permutation(), (Vector<int, 3>({0, 1, 2})) using eq semantics
    EXPECT_EQ(fixed_identity.permutation(), (Vector<int, 3>({0, 1, 2})));
    // compares fixed_rows_identity.permutation(), (Vector<int, 3>({0, 1, 2})) using eq semantics
    EXPECT_EQ(fixed_rows_identity.permutation(), (Vector<int, 3>({0, 1, 2})));
    // compares fixed_cols_identity.rows(), 3 using eq semantics
    ASSERT_EQ(fixed_cols_identity.rows(), 3);
    // compares fixed_cols_identity.image(i), i using eq semantics
    for (int i = 0; i < fixed_cols_identity.rows(); ++i) { EXPECT_EQ(fixed_cols_identity.image(i), i); }

    const PermutationMatrix<Dynamic, Dynamic> dynamic(std::vector<int> {1, 2, 0});
    // compares dynamic.rows(), 3 using eq semantics
    EXPECT_EQ(dynamic.rows(), 3);
    // compares dynamic.cols(), 3 using eq semantics
    EXPECT_EQ(dynamic.cols(), 3);
    // compares dynamic.image(2), 0 using eq semantics
    EXPECT_EQ(dynamic.image(2), 0);
    const PermutationMatrix<Dynamic, Dynamic> empty;
    // compares empty.rows(), 0 using eq semantics
    EXPECT_EQ(empty.rows(), 0);
    std::ostringstream stream;
    stream << empty;
    // checks stream.str().empty()
    EXPECT_TRUE(stream.str().empty());

    const Matrix<int, 1, 3> row_mapping({1, 2, 0});
    const fixed_permutation from_expression(row_mapping);
    // compares from_expression.permutation(), (Vector<int, 3>({1, 2, 0})) using eq semantics
    EXPECT_EQ(from_expression.permutation(), (Vector<int, 3>({1, 2, 0})));
    const fixed_permutation from_wide_integral(std::vector<long long> {1, 2, 0});
    // compares from_wide_integral.permutation(), (Vector<int, 3>({1, 2, 0})) using eq semantics
    EXPECT_EQ(from_wide_integral.permutation(), (Vector<int, 3>({1, 2, 0})));

    check_permutation_actions_and_dispatch<RowMajor>();
    check_permutation_actions_and_dispatch<ColMajor>();

    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(fixed_permutation(std::vector<int> {0, 1})), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(fixed_permutation(std::vector<int> {0, 0, 2})), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(fixed_permutation(std::vector<int> {-1, 0, 2})), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(fixed_permutation(std::vector<int> {0, 1, 3})), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(
      static_cast<void>(
        fixed_permutation(std::vector<long long> {0, 1, static_cast<long long>(std::numeric_limits<int>::max()) + 1})),
      std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(PermutationMatrix<Dynamic, 3>(std::vector<int> {0, 1})), std::invalid_argument);
    Matrix<int, Dynamic, 1> short_mapping(2);
    short_mapping[0] = 1;
    short_mapping[1] = 0;
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(fixed_permutation(short_mapping)), std::invalid_argument);
    Matrix<int, Dynamic, Dynamic> rectangular_mapping(2, 2);
    rectangular_mapping(0, 0) = 0;
    rectangular_mapping(0, 1) = 1;
    rectangular_mapping(1, 0) = 2;
    rectangular_mapping(1, 1) = 3;
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(PermutationMatrix<Dynamic, Dynamic>(rectangular_mapping)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(permutation.image(-1)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(permutation(3, 0)), std::out_of_range);

    const PermutationMatrix<Dynamic, Dynamic> short_permutation(std::vector<int> {1, 0});
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(dynamic * short_permutation), std::invalid_argument);
    Matrix<double, Dynamic, 1> short_vector(2);
    short_vector[0] = 1.0;
    short_vector[1] = 2.0;
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(dynamic * short_vector), std::invalid_argument);

    const Matrix<double, 2, 2> pivot_input({0.0, 1.0, 2.0, 3.0});
    const PartialPivLU<Matrix<double, 2, 2>> lu(pivot_input);
    // compares lu.P().permutation(), (Vector<int, 2>({1, 0})) using eq semantics
    EXPECT_EQ(lu.P().permutation(), (Vector<int, 2>({1, 0})));
    const auto lower = lu.L();
    const auto upper = lu.U();
    expect_matrix_equal(Matrix<double, 2, 2>(lu.P() * pivot_input), Matrix<double, 2, 2>(lower * upper));
}

}   // namespace
