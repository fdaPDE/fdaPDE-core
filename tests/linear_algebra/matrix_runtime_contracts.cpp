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

#include <initializer_list>
#include <limits>
#include <sstream>
#include <type_traits>
#include <utility>
#include <vector>

namespace fdapde {
namespace {

template <typename MatrixType>
concept permits_norm = requires(const MatrixType& matrix) { matrix.norm(); };

template <typename MatrixType>
concept permits_rowwise_norm = requires(const MatrixType& matrix) { matrix.rowwise().norm(); };

template <typename MatrixType>
concept permits_colwise_norm = requires(const MatrixType& matrix) { matrix.colwise().norm(); };

template <typename MatrixType>
concept permits_coefficient_sqrt = requires(const MatrixType& matrix) { matrix.cwise().sqrt(); };

template <typename MatrixType>
concept permits_temporary_integer_division = requires { MatrixType {} / 2; };

template <typename MatrixType>
concept permits_const_temporary_integer_division = requires(const MatrixType&& matrix) { std::move(matrix) / 2; };

using floating_norm_matrix = Matrix<double, 2, 2>;
using integral_norm_matrix = Matrix<int, 2, 2>;
// checks at compile time: permits_norm<floating_norm_matrix>
static_assert(permits_norm<floating_norm_matrix>);
// checks at compile time: permits_rowwise_norm<floating_norm_matrix>
static_assert(permits_rowwise_norm<floating_norm_matrix>);
// checks at compile time: permits_colwise_norm<floating_norm_matrix>
static_assert(permits_colwise_norm<floating_norm_matrix>);
// checks at compile time: permits_coefficient_sqrt<floating_norm_matrix>
static_assert(permits_coefficient_sqrt<floating_norm_matrix>);
// checks at compile time: !permits_norm<integral_norm_matrix>
static_assert(!permits_norm<integral_norm_matrix>);
// checks at compile time: !permits_rowwise_norm<integral_norm_matrix>
static_assert(!permits_rowwise_norm<integral_norm_matrix>);
// checks at compile time: !permits_colwise_norm<integral_norm_matrix>
static_assert(!permits_colwise_norm<integral_norm_matrix>);
// checks at compile time: !permits_coefficient_sqrt<integral_norm_matrix>
static_assert(!permits_coefficient_sqrt<integral_norm_matrix>);

template <int StorageOrder> void check_matrix_block_runtime_contracts() {
    using matrix_type = Matrix<int, 3, 4, StorageOrder>;
    matrix_type matrix({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.row(-1)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.row(3)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.col(-1)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.col(4)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.block(-1, 0, 1, 1)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.block(0, -1, 1, 1)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.block(2, 3, 2, 2)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.block(0, 0, 0, 2)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.block(0, 0, -1, 2)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.template block<2, 2>(2, 3)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.top_rows(0)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.bottom_rows(0)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.left_cols(0)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.right_cols(0)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.top_rows(4)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.bottom_rows(4)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.left_cols(5)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.right_cols(5)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.bottom_rows(std::numeric_limits<int>::min())), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.right_cols(std::numeric_limits<int>::min())), std::invalid_argument);

    const auto zero = [](int, int) { return 0.0; };
    ProceduralMatrix<decltype(zero), Dynamic, 1> long_column(50000, zero);
    ProceduralMatrix<decltype(zero), 1, Dynamic> long_row(50000, zero);
    auto oversized_outer_product = long_column * long_row;
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(oversized_outer_product.block(0, 0, 50000, 50000)), std::length_error);

    auto block = matrix.template block<2, 2>(1, 1);
    const auto& const_block = block;
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(block(-1, 0)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(block(0, 2)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(const_block(-1, 0)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(const_block(0, 2)), std::out_of_range);

    auto row = matrix.row(0);
    const auto& const_row = row;
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(row[-1]), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(row[row.size()]), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(const_row[-1]), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(const_row[const_row.size()]), std::out_of_range);
    const matrix_type original = matrix;
    const std::initializer_list<int> short_row {20, 21};
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(row = short_row, std::invalid_argument);
    // compares matrix, original using eq semantics
    EXPECT_EQ(matrix, original);
}

template <int StorageOrder> void check_matrix_reshape_runtime_contracts() {
    using matrix_type = Matrix<int, Dynamic, Dynamic, StorageOrder>;
    matrix_type matrix(2, 2);

    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.reshape(-1, 4)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.reshape(1, -1)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.reshape(3, 2)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.reshape(3)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.reshape(std::numeric_limits<int>::max(), 2)), std::length_error);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(ReshapeOp<2, Dynamic, matrix_type>(matrix, 3, 2)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(ReshapeOp<Dynamic, 2, matrix_type>(matrix, 2, 3)), std::invalid_argument);

    auto reshaped = matrix.template reshape<1, 4>();
    const auto& const_reshaped = reshaped;
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(reshaped(-1, 0)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(reshaped(0, 4)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(const_reshaped(-1, 0)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(const_reshaped(0, 4)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(reshaped[-1]), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(reshaped[4]), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(const_reshaped[-1]), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(const_reshaped[4]), std::out_of_range);

    auto column = matrix.reshape(4);
    const auto& const_column = column;
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(column[-1]), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(column[4]), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(const_column[-1]), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(const_column[4]), std::out_of_range);

    matrix_type empty;
    const auto empty_matrix = empty.reshape(0, 5);
    const auto empty_column = empty.reshape(0);
    // compares empty_matrix.rows(), 0 using eq semantics
    EXPECT_EQ(empty_matrix.rows(), 0);
    // compares empty_matrix.cols(), 5 using eq semantics
    EXPECT_EQ(empty_matrix.cols(), 5);
    // compares empty_matrix.size(), 0 using eq semantics
    EXPECT_EQ(empty_matrix.size(), 0);
    // compares empty_column.rows(), 0 using eq semantics
    EXPECT_EQ(empty_column.rows(), 0);
    // compares empty_column.cols(), 1 using eq semantics
    EXPECT_EQ(empty_column.cols(), 1);
    // compares empty_column.size(), 0 using eq semantics
    EXPECT_EQ(empty_column.size(), 0);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.reshape(0, 5)), std::invalid_argument);

    const auto zero = [](int, int) { return 0.0; };
    ProceduralMatrix<decltype(zero), Dynamic, 1> long_column(50000, zero);
    ProceduralMatrix<decltype(zero), 1, Dynamic> long_row(50000, zero);
    auto oversized_outer_product = long_column * long_row;
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(oversized_outer_product.reshape(1, 1)), std::length_error);

    Matrix<bool, Dynamic, Dynamic, StorageOrder> boolean_matrix(2, 2);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(boolean_matrix.reshape(3, 2)), std::invalid_argument);
}

template <int StorageOrder> void check_matrix_coeffwise_runtime_contracts() {
    using matrix_type = Matrix<double, Dynamic, Dynamic, StorageOrder>;
    matrix_type lhs(2, 2);
    matrix_type rhs(1, 4);
    lhs.cwise() = 1.0;
    rhs.cwise() = 2.0;
    const matrix_type original = lhs;
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(lhs.cwise() + rhs.cwise()), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(lhs.cwise() += rhs.cwise(), std::invalid_argument);
    // compares lhs, original using eq semantics
    EXPECT_EQ(lhs, original);

    Matrix<double, Dynamic, 2, StorageOrder> partial_lhs(2, 2);
    Matrix<double, Dynamic, 2, StorageOrder> partial_rhs(3, 2);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(partial_lhs.cwise() + partial_rhs.cwise()), std::invalid_argument);

    auto cwise = lhs.cwise();
    const auto& const_cwise = cwise;
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(cwise(-1, 0)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(cwise(0, 2)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(const_cwise(-1, 0)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(const_cwise(0, 2)), std::out_of_range);

    auto mwise = cwise.mwise();
    const auto& const_mwise = mwise;
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(mwise(-1, 0)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(mwise(0, 2)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(const_mwise(-1, 0)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(const_mwise(0, 2)), std::out_of_range);

    auto transformed = lhs.cwise().sqrt();
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(transformed(-1, 0)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(transformed(0, 2)), std::out_of_range);

    auto binary = lhs.cwise() + lhs.cwise();
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(binary(-1, 0)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(binary(0, 2)), std::out_of_range);

    Matrix<double, 1, 3, StorageOrder> row({1.0, 2.0, 3.0});
    auto row_cwise = row.cwise();
    auto row_mwise = row_cwise.mwise();
    auto row_binary = row.cwise() + row.cwise();
    // compares row_cwise[2], 3.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(row_cwise[2], 3.0);
    // compares row_mwise[2], 3.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(row_mwise[2], 3.0);
    // compares row_binary[2], 6.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(row_binary[2], 6.0);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(row_cwise[-1]), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(row_cwise[3]), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(row_mwise[-1]), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(row_mwise[3]), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(row_binary[-1]), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(row_binary[3]), std::out_of_range);
}

template <int StorageOrder> void check_matrix_vectorwise_runtime_contracts() {
    using dynamic_matrix = Matrix<double, Dynamic, Dynamic, StorageOrder>;
    dynamic_matrix matrix = Matrix<double, 2, 3, StorageOrder>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    const dynamic_matrix original = matrix;
    const Matrix<double, Dynamic, 1, StorageOrder> wrong_rows(std::vector<double> {1.0, 2.0, 3.0});
    const Matrix<double, 1, Dynamic, StorageOrder> wrong_cols(std::vector<double> {1.0, 2.0, 3.0, 4.0});

    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.rowwise() = wrong_rows), std::invalid_argument);
    // compares matrix, original using eq semantics
    EXPECT_EQ(matrix, original);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.rowwise() += wrong_rows), std::invalid_argument);
    // compares matrix, original using eq semantics
    EXPECT_EQ(matrix, original);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.rowwise() -= wrong_rows), std::invalid_argument);
    // compares matrix, original using eq semantics
    EXPECT_EQ(matrix, original);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.colwise() = wrong_cols), std::invalid_argument);
    // compares matrix, original using eq semantics
    EXPECT_EQ(matrix, original);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.colwise() += wrong_cols), std::invalid_argument);
    // compares matrix, original using eq semantics
    EXPECT_EQ(matrix, original);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.colwise() -= wrong_cols), std::invalid_argument);
    // compares matrix, original using eq semantics
    EXPECT_EQ(matrix, original);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.rowwise() == wrong_rows), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix.colwise() == wrong_cols), std::invalid_argument);

    const dynamic_matrix valid_rows = Matrix<double, 2, 1, StorageOrder>(std::vector<double> {10.0, 20.0});
    matrix.rowwise() = valid_rows;
    // compares matrix, (Matrix<double, 2, 3, StorageOrder>({10.0, 10.0, 10.0, 20.0, 20.0, 20.0})) using eq
    // semantics
    EXPECT_EQ(matrix, (Matrix<double, 2, 3, StorageOrder>({10.0, 10.0, 10.0, 20.0, 20.0, 20.0})));
    // checks matrix.rowwise() == valid_rows
    EXPECT_TRUE(matrix.rowwise() == valid_rows);
    matrix = original;
    matrix.rowwise() += valid_rows;
    // compares matrix, (Matrix<double, 2, 3, StorageOrder>({11.0, 12.0, 13.0, 24.0, 25.0, 26.0})) using eq
    // semantics
    EXPECT_EQ(matrix, (Matrix<double, 2, 3, StorageOrder>({11.0, 12.0, 13.0, 24.0, 25.0, 26.0})));
    matrix = original;
    matrix.rowwise() -= valid_rows;
    // compares matrix, (Matrix<double, 2, 3, StorageOrder>({-9.0, -8.0, -7.0, -16.0, -15.0, -14.0})) using eq
    // semantics
    EXPECT_EQ(matrix, (Matrix<double, 2, 3, StorageOrder>({-9.0, -8.0, -7.0, -16.0, -15.0, -14.0})));

    const dynamic_matrix valid_cols = Matrix<double, 1, 3, StorageOrder>(std::vector<double> {10.0, 20.0, 30.0});
    matrix.colwise() = valid_cols;
    // compares matrix, (Matrix<double, 2, 3, StorageOrder>({10.0, 20.0, 30.0, 10.0, 20.0, 30.0})) using eq
    // semantics
    EXPECT_EQ(matrix, (Matrix<double, 2, 3, StorageOrder>({10.0, 20.0, 30.0, 10.0, 20.0, 30.0})));
    // checks matrix.colwise() == valid_cols
    EXPECT_TRUE(matrix.colwise() == valid_cols);
    matrix = original;
    matrix.colwise() += valid_cols;
    // compares matrix, (Matrix<double, 2, 3, StorageOrder>({11.0, 22.0, 33.0, 14.0, 25.0, 36.0})) using eq
    // semantics
    EXPECT_EQ(matrix, (Matrix<double, 2, 3, StorageOrder>({11.0, 22.0, 33.0, 14.0, 25.0, 36.0})));
    matrix = original;
    matrix.colwise() -= valid_cols;
    // compares matrix, (Matrix<double, 2, 3, StorageOrder>({-9.0, -18.0, -27.0, -6.0, -15.0, -24.0})) using eq
    // semantics
    EXPECT_EQ(matrix, (Matrix<double, 2, 3, StorageOrder>({-9.0, -18.0, -27.0, -6.0, -15.0, -24.0})));

    Matrix<double, Dynamic, 3, StorageOrder> partial_rows(2, 3);
    const Matrix<double, Dynamic, 1, StorageOrder> valid_partial_rows(std::vector<double> {2.0, 3.0});
    partial_rows.rowwise() = valid_partial_rows;
    // compares the expression result with the explicitly specified fixture
    EXPECT_EQ(partial_rows, (Matrix<double, 2, 3, StorageOrder>({2.0, 2.0, 2.0, 3.0, 3.0, 3.0})));
    const auto partial_rows_snapshot = partial_rows;
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(partial_rows.rowwise() = wrong_rows), std::invalid_argument);
    // compares partial_rows, partial_rows_snapshot using eq semantics
    EXPECT_EQ(partial_rows, partial_rows_snapshot);

    Matrix<double, 2, Dynamic, StorageOrder> partial_cols(2, 3);
    const Matrix<double, 1, Dynamic, StorageOrder> valid_partial_cols(std::vector<double> {2.0, 3.0, 4.0});
    partial_cols.colwise() = valid_partial_cols;
    // compares the expression result with the explicitly specified fixture
    EXPECT_EQ(partial_cols, (Matrix<double, 2, 3, StorageOrder>({2.0, 3.0, 4.0, 2.0, 3.0, 4.0})));
    const auto partial_cols_snapshot = partial_cols;
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(partial_cols.colwise() = wrong_cols), std::invalid_argument);
    // compares partial_cols, partial_cols_snapshot using eq semantics
    EXPECT_EQ(partial_cols, partial_cols_snapshot);

    auto row_sums = original.rowwise().sum();
    // compares row_sums(1, 0), 15.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(row_sums(1, 0), 15.0);
    // compares row_sums[1], 15.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(row_sums[1], 15.0);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(row_sums(-1, 0)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(row_sums(2, 0)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(row_sums(0, -1)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(row_sums(0, 1)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(row_sums[-1]), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(row_sums[2]), std::out_of_range);

    auto col_sums = original.colwise().sum();
    // compares col_sums(0, 2), 9.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(col_sums(0, 2), 9.0);
    // compares col_sums[2], 9.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(col_sums[2], 9.0);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(col_sums(-1, 0)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(col_sums(1, 0)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(col_sums(0, -1)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(col_sums(0, 3)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(col_sums[-1]), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(col_sums[3]), std::out_of_range);
}

template <int StorageOrder> void check_matrix_reduction_runtime_contracts() {
    const auto expect_empty_shape = [](const auto& result, int rows, int cols) {
        // compares result.rows(), rows using eq semantics
        EXPECT_EQ(result.rows(), rows);
        // compares result.cols(), cols using eq semantics
        EXPECT_EQ(result.cols(), cols);
        // compares result.size(), 0 using eq semantics
        EXPECT_EQ(result.size(), 0);
    };

    const Matrix<double, 2, 3, StorageOrder> negative({-4.0, -2.0, -3.0, -9.0, -8.0, -7.0});
    // compares negative.max(), -2.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(negative.max(), -2.0);
    // compares the expression result with the explicitly specified fixture
    EXPECT_EQ(negative.rowwise().max(), (Matrix<double, 2, 1, StorageOrder>({-2.0, -7.0})));
    // compares the expression result with the explicitly specified fixture
    EXPECT_EQ(negative.colwise().max(), (Matrix<double, 1, 3, StorageOrder>({-4.0, -2.0, -3.0})));

    const Matrix<int, Dynamic, Dynamic, StorageOrder> dynamic_negative =
      Matrix<int, 2, 3, StorageOrder>({-6, -5, -4, -3, -2, -1});
    // compares dynamic_negative.max(), -1 using eq semantics
    EXPECT_EQ(dynamic_negative.max(), -1);

    const Matrix<double, Dynamic, Dynamic, StorageOrder> empty;
    // compares empty.sum(), 0.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(empty.sum(), 0.0);
    // compares empty.prod(), 1.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(empty.prod(), 1.0);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(empty.mean()), std::domain_error);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(empty.max()), std::domain_error);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(empty.min()), std::domain_error);

    const Matrix<int, Dynamic, Dynamic, StorageOrder> empty_int;
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(empty_int.mean()), std::domain_error);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(empty_int.max()), std::domain_error);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(empty_int.min()), std::domain_error);

    const Matrix<double, 2, Dynamic, StorageOrder> empty_row_axes(2, 0);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(empty_row_axes.mean()), std::domain_error);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(empty_row_axes.max()), std::domain_error);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(empty_row_axes.min()), std::domain_error);
    // compares the expression result with the explicitly specified fixture
    EXPECT_EQ(empty_row_axes.rowwise().sum(), (Matrix<double, 2, 1, StorageOrder>({0.0, 0.0})));
    // compares the expression result with the explicitly specified fixture
    EXPECT_EQ(empty_row_axes.rowwise().prod(), (Matrix<double, 2, 1, StorageOrder>({1.0, 1.0})));
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(empty_row_axes.rowwise().mean()(0, 0)), std::domain_error);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(empty_row_axes.rowwise().max()(0, 0)), std::domain_error);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(empty_row_axes.rowwise().min()(0, 0)), std::domain_error);
    expect_empty_shape(empty_row_axes.colwise().mean(), 1, 0);
    expect_empty_shape(empty_row_axes.colwise().max(), 1, 0);
    expect_empty_shape(empty_row_axes.colwise().min(), 1, 0);

    const Matrix<double, Dynamic, 3, StorageOrder> empty_col_axes(0, 3);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(empty_col_axes.mean()), std::domain_error);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(empty_col_axes.max()), std::domain_error);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(empty_col_axes.min()), std::domain_error);
    // compares the expression result with the explicitly specified fixture
    EXPECT_EQ(empty_col_axes.colwise().sum(), (Matrix<double, 1, 3, StorageOrder>({0.0, 0.0, 0.0})));
    // compares the expression result with the explicitly specified fixture
    EXPECT_EQ(empty_col_axes.colwise().prod(), (Matrix<double, 1, 3, StorageOrder>({1.0, 1.0, 1.0})));
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(empty_col_axes.colwise().mean()(0, 0)), std::domain_error);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(empty_col_axes.colwise().max()(0, 0)), std::domain_error);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(empty_col_axes.colwise().min()(0, 0)), std::domain_error);
    expect_empty_shape(empty_col_axes.rowwise().mean(), 0, 1);
    expect_empty_shape(empty_col_axes.rowwise().max(), 0, 1);
    expect_empty_shape(empty_col_axes.rowwise().min(), 0, 1);

    expect_empty_shape(empty_int.rowwise().mean(), 0, 1);
    expect_empty_shape(empty_int.rowwise().max(), 0, 1);
    expect_empty_shape(empty_int.rowwise().min(), 0, 1);
    expect_empty_shape(empty_int.colwise().mean(), 1, 0);
    expect_empty_shape(empty_int.colwise().max(), 1, 0);
    expect_empty_shape(empty_int.colwise().min(), 1, 0);

    const Matrix<int, 2, 2, StorageOrder> integral({-1, -2, -4, -7});
    const auto row_means = integral.rowwise().mean();
    const auto col_means = integral.colwise().mean();
    using RowMean = decltype(row_means);
    using ColMean = decltype(col_means);
    // checks at compile time: std::is_same_v<typename RowMean::Scalar, int>
    static_assert(std::is_same_v<typename RowMean::Scalar, int>);
    // checks at compile time: std::is_same_v<typename ColMean::Scalar, int>
    static_assert(std::is_same_v<typename ColMean::Scalar, int>);
    // checks at compile time: std::is_same_v<decltype(row_means(0, 0)), int>
    static_assert(std::is_same_v<decltype(row_means(0, 0)), int>);
    // checks at compile time: std::is_same_v<decltype(col_means(0, 0)), int>
    static_assert(std::is_same_v<decltype(col_means(0, 0)), int>);
    // checks at compile time: RowMean::Rows == 2 && RowMean::Cols == 1
    static_assert(RowMean::Rows == 2 && RowMean::Cols == 1);
    // checks at compile time: ColMean::Rows == 1 && ColMean::Cols == 2
    static_assert(ColMean::Rows == 1 && ColMean::Cols == 2);
    // compares row_means, (Matrix<int, 2, 1, StorageOrder>({-1, -5})) using eq semantics
    EXPECT_EQ(row_means, (Matrix<int, 2, 1, StorageOrder>({-1, -5})));
    // compares col_means, (Matrix<int, 1, 2, StorageOrder>({-2, -4})) using eq semantics
    EXPECT_EQ(col_means, (Matrix<int, 1, 2, StorageOrder>({-2, -4})));
}

template <int StorageOrder> void check_matrix_norm_runtime_contracts() {
    using pair_type = Matrix<double, 1, 2, StorageOrder>;
    const pair_type tiny({3.0e-8, 4.0e-8});
    const pair_type underflowing_square({3.0e-200, 4.0e-200});
    const pair_type overflowing_square({3.0e200, 4.0e200});
    // compares tiny.norm(), 5.0e-8 using double_eq semantics
    EXPECT_DOUBLE_EQ(tiny.norm(), 5.0e-8);
    // compares underflowing_square.norm(), 5.0e-200 using double_eq semantics
    EXPECT_DOUBLE_EQ(underflowing_square.norm(), 5.0e-200);
    // compares overflowing_square.norm(), 5.0e200 using double_eq semantics
    EXPECT_DOUBLE_EQ(overflowing_square.norm(), 5.0e200);

    const Matrix<double, 3, 2, StorageOrder> rows({3.0e-8, 4.0e-8, 3.0e-200, 4.0e-200, 3.0e200, 4.0e200});
    const auto row_norms = rows.rowwise().norm();
    // compares row_norms[0], 5.0e-8 using double_eq semantics
    EXPECT_DOUBLE_EQ(row_norms[0], 5.0e-8);
    // compares row_norms[1], 5.0e-200 using double_eq semantics
    EXPECT_DOUBLE_EQ(row_norms[1], 5.0e-200);
    // compares row_norms[2], 5.0e200 using double_eq semantics
    EXPECT_DOUBLE_EQ(row_norms[2], 5.0e200);

    const Matrix<double, 2, 3, StorageOrder> cols({3.0e-8, 3.0e-200, 3.0e200, 4.0e-8, 4.0e-200, 4.0e200});
    const auto col_norms = cols.colwise().norm();
    // compares col_norms[0], 5.0e-8 using double_eq semantics
    EXPECT_DOUBLE_EQ(col_norms[0], 5.0e-8);
    // compares col_norms[1], 5.0e-200 using double_eq semantics
    EXPECT_DOUBLE_EQ(col_norms[1], 5.0e-200);
    // compares col_norms[2], 5.0e200 using double_eq semantics
    EXPECT_DOUBLE_EQ(col_norms[2], 5.0e200);

    const pair_type tiny_squares({1.0e-16, 4.0e-16});
    const pair_type roots = tiny_squares.cwise().sqrt();
    // compares roots[0], 1.0e-8 using double_eq semantics
    EXPECT_DOUBLE_EQ(roots[0], 1.0e-8);
    // compares roots[1], 2.0e-8 using double_eq semantics
    EXPECT_DOUBLE_EQ(roots[1], 2.0e-8);

    const double view_data[2] {3.0, 4.0};
    const MatrixView<const double, 1, 2, StorageOrder> const_view(view_data);
    // compares const_view.norm(), 5.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(const_view.norm(), 5.0);
}

template <int StorageOrder> void check_matrix_inf_norm_runtime_contracts() {
    const Matrix<double, 2, 3, StorageOrder> zero = Matrix<double, 2, 3, StorageOrder>::Zero();
    // compares zero.inf_norm(), 0.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(zero.inf_norm(), 0.0);
    // compares zero.rowwise().inf_norm(), (Matrix<double, 2, 1, StorageOrder>({0.0, 0.0})) using eq semantics
    EXPECT_EQ(zero.rowwise().inf_norm(), (Matrix<double, 2, 1, StorageOrder>({0.0, 0.0})));
    // compares the expression result with the explicitly specified fixture
    EXPECT_EQ(zero.colwise().inf_norm(), (Matrix<double, 1, 3, StorageOrder>({0.0, 0.0, 0.0})));

    const double denormal = std::numeric_limits<double>::denorm_min();
    const Matrix<double, 1, 1, StorageOrder> subnormal({denormal});
    // compares subnormal.inf_norm(), denormal using double_eq semantics
    EXPECT_DOUBLE_EQ(subnormal.inf_norm(), denormal);
    // compares subnormal.rowwise().inf_norm()[0], denormal using double_eq semantics
    EXPECT_DOUBLE_EQ(subnormal.rowwise().inf_norm()[0], denormal);
    // compares subnormal.colwise().inf_norm()[0], denormal using double_eq semantics
    EXPECT_DOUBLE_EQ(subnormal.colwise().inf_norm()[0], denormal);

    const Matrix<double, Dynamic, Dynamic, StorageOrder> empty;
    // compares empty.inf_norm(), 0.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(empty.inf_norm(), 0.0);

    const Matrix<double, 2, Dynamic, StorageOrder> empty_row_axes(2, 0);
    // compares the expression result with the explicitly specified fixture
    EXPECT_EQ(empty_row_axes.rowwise().inf_norm(), (Matrix<double, 2, 1, StorageOrder>({0.0, 0.0})));

    const Matrix<double, Dynamic, 3, StorageOrder> empty_col_axes(0, 3);
    // compares the expression result with the explicitly specified fixture
    EXPECT_EQ(empty_col_axes.colwise().inf_norm(), (Matrix<double, 1, 3, StorageOrder>({0.0, 0.0, 0.0})));

    const Matrix<double, 2, 3, StorageOrder> view_owner({-4.0, 0.0, 2.0, 1.0, -3.0, 5.0});
    const MatrixView<const double, 2, 3, StorageOrder> const_view(view_owner.data());
    // compares const_view.inf_norm(), 5.0 using double_eq semantics
    EXPECT_DOUBLE_EQ(const_view.inf_norm(), 5.0);
    // compares the expression result with the explicitly specified fixture
    EXPECT_EQ(const_view.rowwise().inf_norm(), (Matrix<double, 2, 1, StorageOrder>({4.0, 5.0})));
    // compares the expression result with the explicitly specified fixture
    EXPECT_EQ(const_view.colwise().inf_norm(), (Matrix<double, 1, 3, StorageOrder>({4.0, 3.0, 5.0})));
}

template <int StorageOrder> void check_empty_boolean_runtime_contracts() {
    using dynamic_matrix = Matrix<bool, Dynamic, Dynamic, StorageOrder>;

    const dynamic_matrix empty;
    // checks empty.all()
    EXPECT_TRUE(empty.all());
    // checks empty.any()
    EXPECT_FALSE(empty.any());
    // compares empty.count(), 0 using eq semantics
    EXPECT_EQ(empty.count(), 0);
    // checks empty == dynamic_matrix()
    EXPECT_TRUE(empty == dynamic_matrix());
    // checks empty != dynamic_matrix()
    EXPECT_FALSE(empty != dynamic_matrix());

    const dynamic_matrix zero_rows(0, 3);
    const dynamic_matrix zero_cols(3, 0);
    // checks zero_rows.all()
    EXPECT_TRUE(zero_rows.all());
    // checks zero_rows.any()
    EXPECT_FALSE(zero_rows.any());
    // compares zero_rows.count(), 0 using eq semantics
    EXPECT_EQ(zero_rows.count(), 0);
    // checks zero_rows == dynamic_matrix(0, 3)
    EXPECT_TRUE(zero_rows == dynamic_matrix(0, 3));
    // checks zero_cols.all()
    EXPECT_TRUE(zero_cols.all());
    // checks zero_cols.any()
    EXPECT_FALSE(zero_cols.any());
    // compares zero_cols.count(), 0 using eq semantics
    EXPECT_EQ(zero_cols.count(), 0);
    // checks zero_cols == dynamic_matrix(3, 0)
    EXPECT_TRUE(zero_cols == dynamic_matrix(3, 0));
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(zero_rows == zero_cols), std::invalid_argument);
    dynamic_matrix mismatched_assignment(0, 3);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(mismatched_assignment &= zero_cols, std::invalid_argument);
    // compares mismatched_assignment.rows(), 0 using eq semantics
    EXPECT_EQ(mismatched_assignment.rows(), 0);
    // compares mismatched_assignment.cols(), 3 using eq semantics
    EXPECT_EQ(mismatched_assignment.cols(), 3);

    dynamic_matrix assigned(1, 2);
    assigned(0, 1) = true;
    assigned = empty;
    // compares assigned.rows(), 0 using eq semantics
    EXPECT_EQ(assigned.rows(), 0);
    // compares assigned.cols(), 0 using eq semantics
    EXPECT_EQ(assigned.cols(), 0);
    // checks assigned.all()
    EXPECT_TRUE(assigned.all());
    // checks assigned.any()
    EXPECT_FALSE(assigned.any());
    // compares assigned.count(), 0 using eq semantics
    EXPECT_EQ(assigned.count(), 0);

    dynamic_matrix expression_assigned(2, 1);
    expression_assigned(1, 0) = true;
    expression_assigned = ~empty;
    // compares expression_assigned.rows(), 0 using eq semantics
    EXPECT_EQ(expression_assigned.rows(), 0);
    // compares expression_assigned.cols(), 0 using eq semantics
    EXPECT_EQ(expression_assigned.cols(), 0);
    // checks expression_assigned.all()
    EXPECT_TRUE(expression_assigned.all());
    // checks expression_assigned.any()
    EXPECT_FALSE(expression_assigned.any());
    // compares expression_assigned.count(), 0 using eq semantics
    EXPECT_EQ(expression_assigned.count(), 0);

    std::ostringstream stream;
    stream << empty << zero_rows << zero_cols;
    // checks stream.str().empty()
    EXPECT_TRUE(stream.str().empty());

    MatrixView<bool, Dynamic, Dynamic, StorageOrder> empty_view;
    // compares empty_view.bitpacks(), 0 using eq semantics
    EXPECT_EQ(empty_view.bitpacks(), 0);
    empty_view.set();
    empty_view.clear();
}

}   // namespace

// verifies dynamic square operations remain available through the public algebra API
TEST(LinearAlgebraRuntimeContracts, DynamicSquareOperationsRemainAvailable) {
    Matrix<double, Dynamic, Dynamic> matrix = Matrix<double, 2, 2>({2, 1, 1, 3});
    const Matrix<double, 2, 2> zero = Matrix<double, 2, 2>::Zero();
    const Vector<double, 2> expected_diagonal({2, 3});
    const Matrix<double, Dynamic, Dynamic> inverse = matrix.inverse();

    // compares matrix.symm_part(), matrix using eq semantics
    EXPECT_EQ(matrix.symm_part(), matrix);
    // compares matrix.skew_part(), zero using eq semantics
    EXPECT_EQ(matrix.skew_part(), zero);
    // checks almost_equal(inverse * matrix, Matrix<double, 2, 2>({1, 0, 0, 1}))
    EXPECT_TRUE(almost_equal(inverse * matrix, Matrix<double, 2, 2>({1, 0, 0, 1})));
    // compares matrix.determinant(), 5 using eq semantics
    EXPECT_EQ(matrix.determinant(), 5);
    // compares matrix.diagonal(), expected_diagonal using eq semantics
    EXPECT_EQ(matrix.diagonal(), expected_diagonal);
}

// verifies square operations reject rectangular matrices through the public algebra API
TEST(LinearAlgebraRuntimeContracts, SquareOperationsRejectRectangularMatrices) {
    Matrix<double, Dynamic, Dynamic> matrix(2, 3);

    // checks the exception category for the supplied invalid operation
    EXPECT_THROW((void)matrix.symm_part(), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW((void)matrix.skew_part(), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW((void)matrix.inverse(), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW((void)matrix.determinant(), std::invalid_argument);
}

// verifies dynamic binary expressions reject incompatible shapes through the public algebra API
TEST(LinearAlgebraRuntimeContracts, DynamicBinaryExpressionsRejectIncompatibleShapes) {
    Matrix<double, Dynamic, Dynamic> lhs(2, 3);
    Matrix<double, Dynamic, Dynamic> wrong_cols_rhs(2, 2);
    Matrix<double, Dynamic, Dynamic> wrong_rows_rhs(3, 3);
    Matrix<double, Dynamic, Dynamic> product_rhs(2, 4);
    Vector<double, Dynamic> short_vector(2);
    Vector<double, Dynamic> vector(3);
    Matrix<double, Dynamic, Dynamic> non_column_vector(3, 2);

    // checks the exception category for the supplied invalid operation
    EXPECT_THROW((void)(lhs + wrong_cols_rhs), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW((void)(lhs - wrong_rows_rhs), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW((void)(lhs * product_rhs), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW((void)short_vector.cross(vector), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW((void)non_column_vector.cross(vector), std::invalid_argument);

    Vector<double, Dynamic> x(3);
    Vector<double, Dynamic> y(3);
    x[0] = 1;
    y[1] = 1;
    const Vector<double, 3> expected_cross({0, 0, 1});
    // compares x.cross(y), expected_cross using eq semantics
    EXPECT_EQ(x.cross(y), expected_cross);
}

// verifies dense diagonal view checks shape and indexes through the public algebra API
TEST(LinearAlgebraRuntimeContracts, DenseDiagonalViewChecksShapeAndIndexes) {
    Matrix<double, Dynamic, Dynamic> rectangular(2, 3);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW((void)rectangular.diagonal(), std::invalid_argument);

    Matrix<double, 2, 2> matrix({1, 2, 3, 4});
    auto diagonal = matrix.diagonal();
    const auto& const_diagonal = diagonal;

    // checks the exception category for the supplied invalid operation
    EXPECT_THROW((void)diagonal(-1, 0), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW((void)diagonal(2, 0), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW((void)diagonal(0, -1), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW((void)diagonal(0, 1), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW((void)diagonal[-1], std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW((void)diagonal[2], std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW((void)const_diagonal(-1, 0), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW((void)const_diagonal(2, 0), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW((void)const_diagonal(0, -1), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW((void)const_diagonal(0, 1), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW((void)const_diagonal[-1], std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW((void)const_diagonal[2], std::out_of_range);
}

// verifies matrix owner rejects invalid shapes and sizes through the public algebra API
TEST(LinearAlgebraRuntimeContracts, MatrixOwnerRejectsInvalidShapesAndSizes) {
    using dynamic_matrix = Matrix<int, Dynamic, Dynamic>;
    using partial_rows_matrix = Matrix<int, Dynamic, 3>;
    using partial_cols_matrix = Matrix<int, 2, Dynamic>;

    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(dynamic_matrix(-1, 2)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(partial_rows_matrix(2, 4)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(partial_cols_matrix(3, 2)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(dynamic_matrix(std::numeric_limits<int>::max(), 2)), std::length_error);

    partial_rows_matrix partial(2, 3);
    partial(1, 2) = 17;
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(partial.resize(2, 4), std::invalid_argument);
    // compares partial.rows(), 2 using eq semantics
    EXPECT_EQ(partial.rows(), 2);
    // compares partial.cols(), 3 using eq semantics
    EXPECT_EQ(partial.cols(), 3);
    // compares partial.size(), 6 using eq semantics
    EXPECT_EQ(partial.size(), 6);
    // compares partial(1, 2), 17 using eq semantics
    EXPECT_EQ(partial(1, 2), 17);

    dynamic_matrix bounded(1, 2);
    bounded(0, 1) = 23;
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(bounded.resize(std::numeric_limits<int>::max(), 2), std::length_error);
    // compares bounded.rows(), 1 using eq semantics
    EXPECT_EQ(bounded.rows(), 1);
    // compares bounded.cols(), 2 using eq semantics
    EXPECT_EQ(bounded.cols(), 2);
    // compares bounded(0, 1), 23 using eq semantics
    EXPECT_EQ(bounded(0, 1), 23);
}

// verifies matrix owner checks inputs assignments and indexes through the public algebra API
TEST(LinearAlgebraRuntimeContracts, MatrixOwnerChecksInputsAssignmentsAndIndexes) {
    using dynamic_matrix = Matrix<int, Dynamic, Dynamic>;
    using dynamic_vector = Vector<int, Dynamic>;
    using fixed_matrix = Matrix<int, 2, 3>;
    using fixed_vector = Vector<int, 3>;

    const std::vector<int> short_input {1, 2, 3, 4, 5};
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(fixed_matrix(short_input)), std::invalid_argument);

    const dynamic_matrix wrong_matrix_shape(2, 4);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(fixed_matrix(wrong_matrix_shape)), std::invalid_argument);
    const dynamic_matrix non_vector_shape(2, 2);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(fixed_vector(non_vector_shape)), std::invalid_argument);

    fixed_vector vector({1, 2, 3});
    const std::initializer_list<int> short_vector {4, 5};
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(vector = short_vector, std::invalid_argument);
    // compares vector, fixed_vector({1, 2, 3}) using eq semantics
    EXPECT_EQ(vector, fixed_vector({1, 2, 3}));

    fixed_matrix matrix({1, 2, 3, 4, 5, 6});
    const fixed_matrix& const_matrix = matrix;
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix(-1, 0)), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(const_matrix(0, 3)), std::out_of_range);
    const fixed_vector& const_vector = vector;
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(vector[-1]), std::out_of_range);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(const_vector[3]), std::out_of_range);

    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(dynamic_vector::LinSpaced(1, 0, 1)), std::invalid_argument);
}

// verifies procedural matrix checks shape size and indexes through the public algebra API
TEST(LinearAlgebraRuntimeContracts, ProceduralMatrixChecksShapeSizeAndIndexes) {
    const auto ones = [](int, int) { return 1; };
    using partial_procedural = ProceduralMatrix<decltype(ones), 2, Dynamic>;
    using dynamic_procedural = ProceduralMatrix<decltype(ones), Dynamic, Dynamic>;
    using fixed_procedural_vector = ProceduralMatrix<decltype(ones), 3, 1>;

    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(partial_procedural(3, 3, ones)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(fixed_procedural_vector(2, ones)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(dynamic_procedural(std::numeric_limits<int>::max(), 2, ones)), std::length_error);

    partial_procedural matrix(2, 3, ones);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(matrix.resize(3, 3), std::invalid_argument);
    // compares matrix.rows(), 2 using eq semantics
    EXPECT_EQ(matrix.rows(), 2);
    // compares matrix.cols(), 3 using eq semantics
    EXPECT_EQ(matrix.cols(), 3);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(matrix(2, 0)), std::out_of_range);
}

// verifies matrix view rejects invalid runtime shapes through the public algebra API
TEST(LinearAlgebraRuntimeContracts, MatrixViewRejectsInvalidRuntimeShapes) {
    using dynamic_matrix_view = MatrixView<int, Dynamic, Dynamic>;
    using partial_matrix_view = MatrixView<int, Dynamic, 3>;
    using dynamic_vector_view = MatrixView<int, Dynamic, 1>;
    using fixed_vector_view = MatrixView<int, 3, 1>;
    int data[6] {};

    dynamic_matrix_view empty;
    // compares empty.rows(), 0 using eq semantics
    EXPECT_EQ(empty.rows(), 0);
    // compares empty.cols(), 0 using eq semantics
    EXPECT_EQ(empty.cols(), 0);
    // compares empty.data(), nullptr using eq semantics
    EXPECT_EQ(empty.data(), nullptr);

    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(dynamic_matrix_view(data, -1, 3)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(dynamic_matrix_view(data, 0, 3)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(dynamic_matrix_view(data, 2, 0)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(partial_matrix_view(data, 2, 2)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(dynamic_vector_view(data, 0)), std::invalid_argument);
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(static_cast<void>(fixed_vector_view(data, 2)), std::invalid_argument);

    int destination_data[4] {1, 2, 3, 4};
    int source_data[6] {6, 5, 4, 3, 2, 1};
    dynamic_matrix_view destination(destination_data, 2, 2);
    dynamic_matrix_view source(source_data, 2, 3);
    int* const destination_binding = destination.data();
    // checks the exception category for the supplied invalid operation
    EXPECT_THROW(destination = source, std::invalid_argument);
    // compares destination.data(), destination_binding using eq semantics
    EXPECT_EQ(destination.data(), destination_binding);
    // compares destination.rows(), 2 using eq semantics
    EXPECT_EQ(destination.rows(), 2);
    // compares destination.cols(), 2 using eq semantics
    EXPECT_EQ(destination.cols(), 2);
    // compares destination(0, 0), 1 using eq semantics
    EXPECT_EQ(destination(0, 0), 1);
    // compares destination(1, 1), 4 using eq semantics
    EXPECT_EQ(destination(1, 1), 4);
}

// verifies matrix block checks bounds and assignments through the public algebra API
TEST(LinearAlgebraRuntimeContracts, MatrixBlockChecksBoundsAndAssignments) {
    check_matrix_block_runtime_contracts<RowMajor>();
    check_matrix_block_runtime_contracts<ColMajor>();
}

// verifies matrix reshape checks shapes sizes and indexes through the public algebra API
TEST(LinearAlgebraRuntimeContracts, MatrixReshapeChecksShapesSizesAndIndexes) {
    check_matrix_reshape_runtime_contracts<RowMajor>();
    check_matrix_reshape_runtime_contracts<ColMajor>();
}

// verifies matrix coeff wise checks shapes and indexes through the public algebra API
TEST(LinearAlgebraRuntimeContracts, MatrixCoeffWiseChecksShapesAndIndexes) {
    check_matrix_coeffwise_runtime_contracts<RowMajor>();
    check_matrix_coeffwise_runtime_contracts<ColMajor>();
}

// verifies matrix vector wise checks shapes and indexes through the public algebra API
TEST(LinearAlgebraRuntimeContracts, MatrixVectorWiseChecksShapesAndIndexes) {
    check_matrix_vectorwise_runtime_contracts<RowMajor>();
    check_matrix_vectorwise_runtime_contracts<ColMajor>();
}

// verifies matrix reductions honor empty and integral contracts through the public algebra API
TEST(LinearAlgebraRuntimeContracts, MatrixReductionsHonorEmptyAndIntegralContracts) {
    check_matrix_reduction_runtime_contracts<RowMajor>();
    check_matrix_reduction_runtime_contracts<ColMajor>();
}

// verifies matrix norms remain scale safe through the public algebra API
TEST(LinearAlgebraRuntimeContracts, MatrixNormsRemainScaleSafe) {
    check_matrix_norm_runtime_contracts<RowMajor>();
    check_matrix_norm_runtime_contracts<ColMajor>();
}

// verifies matrix infinity norms use zero identity through the public algebra API
TEST(LinearAlgebraRuntimeContracts, MatrixInfinityNormsUseZeroIdentity) {
    check_matrix_inf_norm_runtime_contracts<RowMajor>();
    check_matrix_inf_norm_runtime_contracts<ColMajor>();
}

// verifies empty boolean expressions are terminal safe through the public algebra API
TEST(LinearAlgebraRuntimeContracts, EmptyBooleanExpressionsAreTerminalSafe) {
    check_empty_boolean_runtime_contracts<RowMajor>();
    check_empty_boolean_runtime_contracts<ColMajor>();
}

// verifies scalar division uses coefficient arithmetic and safe nesting through the public algebra API
TEST(LinearAlgebraRuntimeContracts, ScalarDivisionUsesCoefficientArithmeticAndSafeNesting) {
    const auto check_storage = []<int StorageOrder>() {
        using matrix_type = Matrix<double, 2, 2, StorageOrder>;
        // checks at compile time: !permits_temporary_integer_division<matrix_type>
        static_assert(!permits_temporary_integer_division<matrix_type>);
        // checks at compile time: !permits_const_temporary_integer_division<matrix_type>
        static_assert(!permits_const_temporary_integer_division<matrix_type>);
        constexpr matrix_type source({3.0, -5.0, 7.0, -9.0});
        constexpr double first = (source / 2)(0, 0);
        // compares first, 1.5 using double_eq semantics
        EXPECT_DOUBLE_EQ(first, 1.5);
        const matrix_type expected({1.5, -2.5, 3.5, -4.5});
        const auto divided = source / 2;
        // checks at compile time: std::is_same_v<typename decltype(divided)::Scalar, double>
        static_assert(std::is_same_v<typename decltype(divided)::Scalar, double>);
        // checks at compile time: decltype(divided)::ReadOnly == 1 && decltype(divided)::StorageOrder ==
        // StorageOrder
        static_assert(decltype(divided)::ReadOnly == 1 && decltype(divided)::StorageOrder == StorageOrder);
        // compares matrix_type(divided), expected using eq semantics
        EXPECT_EQ(matrix_type(divided), expected);
        // compares matrix_type(source / 2.0), expected using eq semantics
        EXPECT_EQ(matrix_type(source / 2.0), expected);
        // compares matrix_type(source / -2), matrix_type(expected * -1) using eq semantics
        EXPECT_EQ(matrix_type(source / -2), matrix_type(expected * -1));
        matrix_type compound(source);
        compound /= 2;
        // compares compound, expected using eq semantics
        EXPECT_EQ(compound, expected);
        matrix_type alias(source);
        alias = alias.transpose() / 2;
        // compares alias, matrix_type({1.5, 3.5, -2.5, -4.5}) using eq semantics
        EXPECT_EQ(alias, matrix_type({1.5, 3.5, -2.5, -4.5}));
        const auto nested = [&] {
            const auto block = source.template block<2, 2>(0, 0);
            const int divisor = 2;
            return (block + block) / divisor;
        }();
        // compares matrix_type(nested), source using eq semantics
        EXPECT_EQ(matrix_type(nested), source);
        matrix_type view_owner(source);
        const auto view_divided = MatrixView<double, 2, 2, StorageOrder>(view_owner.data()) / 2;
        // compares matrix_type(view_divided), expected using eq semantics
        EXPECT_EQ(matrix_type(view_divided), expected);

        const Matrix<int, 2, 2, StorageOrder> integers({3, -5, 7, -9});
        const auto integer_divided = integers / 2;
        // checks at compile time: std::is_same_v<typename decltype(integer_divided)::Scalar, int>
        static_assert(std::is_same_v<typename decltype(integer_divided)::Scalar, int>);
        // compares the expression result with the explicitly specified fixture
        EXPECT_EQ(
          (Matrix<int, 2, 2, StorageOrder>(integer_divided)), (Matrix<int, 2, 2, StorageOrder>({1, -2, 3, -4})));
        const auto promoted = integers / 2.0;
        // checks at compile time: std::is_same_v<typename decltype(promoted)::Scalar, double>
        static_assert(std::is_same_v<typename decltype(promoted)::Scalar, double>);
        // compares matrix_type(promoted), expected using eq semantics
        EXPECT_EQ(matrix_type(promoted), expected);
        const Matrix<float, Dynamic, 1, StorageOrder> vector(std::vector<float> {3.0f, -5.0f});
        const auto vector_divided = vector / 2;
        // checks at compile time: std::is_same_v<typename decltype(vector_divided)::Scalar, float>
        static_assert(std::is_same_v<typename decltype(vector_divided)::Scalar, float>);
        // compares vector_divided.rows(), 2 using eq semantics
        EXPECT_EQ(vector_divided.rows(), 2);
        // compares vector_divided.cols(), 1 using eq semantics
        EXPECT_EQ(vector_divided.cols(), 1);
        // compares vector_divided[0], 1.5f using float_eq semantics
        EXPECT_FLOAT_EQ(vector_divided[0], 1.5f);
        // compares vector_divided[1], -2.5f using float_eq semantics
        EXPECT_FLOAT_EQ(vector_divided[1], -2.5f);
        const Matrix<short, 1, 1> narrow(static_cast<short>(7));
        const auto narrow_divided = narrow / static_cast<short>(2);
        // checks at compile time: std::is_same_v<typename decltype(narrow_divided)::Scalar, short>
        static_assert(std::is_same_v<typename decltype(narrow_divided)::Scalar, short>);
        // compares narrow_divided(0, 0), 3 using eq semantics
        EXPECT_EQ(narrow_divided(0, 0), 3);
    };
    check_storage.template operator()<RowMajor>();
    check_storage.template operator()<ColMajor>();
}

// verifies scalar division preserves structured matrix categories through the public algebra API
TEST(LinearAlgebraRuntimeContracts, ScalarDivisionPreservesStructuredMatrixCategories) {
    const double symmetric_values[] {3.0, -5.0, 7.0};
    const SymmetricMatrix<double, 2, 2> symmetric(symmetric_values);
    const auto symmetric_divided = symmetric / 2;
    // checks at compile time: is_symmetric_matrix_v<decltype(symmetric_divided)>
    static_assert(is_symmetric_matrix_v<decltype(symmetric_divided)>);
    // checks at compile time: !permits_temporary_integer_division<SymmetricMatrix<double, 2, 2>>
    static_assert(!permits_temporary_integer_division<SymmetricMatrix<double, 2, 2>>);
    // compares (Matrix<double, 2, 2>(symmetric_divided)), (Matrix<double, 2, 2>({1.5, -2.5, -2.5, 3.5})) using
    // eq semantics
    EXPECT_EQ((Matrix<double, 2, 2>(symmetric_divided)), (Matrix<double, 2, 2>({1.5, -2.5, -2.5, 3.5})));
    const auto symmetric_nested = [&] { return (symmetric + symmetric) / 2; }();
    // compares (Matrix<double, 2, 2>(symmetric_nested)), (Matrix<double, 2, 2>(symmetric)) using eq semantics
    EXPECT_EQ((Matrix<double, 2, 2>(symmetric_nested)), (Matrix<double, 2, 2>(symmetric)));

    const double skew_values[] {3.0, -5.0, 7.0};
    const SkewSymmetricMatrix<double, 3> skew(skew_values);
    const auto skew_divided = skew / 2;
    // checks at compile time: is_skew_symmetric_matrix_v<decltype(skew_divided)>
    static_assert(is_skew_symmetric_matrix_v<decltype(skew_divided)>);
    // checks at compile time: !permits_temporary_integer_division<SkewSymmetricMatrix<double, 3>>
    static_assert(!permits_temporary_integer_division<SkewSymmetricMatrix<double, 3>>);
    // compares the expression result with the explicitly specified fixture
    EXPECT_EQ(
      (Matrix<double, 3, 3>(skew_divided)), (Matrix<double, 3, 3>({0.0, 1.5, -2.5, -1.5, 0.0, 3.5, 2.5, -3.5, 0.0})));
    const auto skew_nested = [&] { return (skew + skew) / 2; }();
    // compares (Matrix<double, 3, 3>(skew_nested)), (Matrix<double, 3, 3>(skew)) using eq semantics
    EXPECT_EQ((Matrix<double, 3, 3>(skew_nested)), (Matrix<double, 3, 3>(skew)));

    const Vector<double, 2> diagonal_values({3.0, 5.0});
    const auto diagonal_divided = diagonal_values.as_diagonal() / 2;
    // checks at compile time: is_diagonal_matrix_v<decltype(diagonal_divided)>
    static_assert(is_diagonal_matrix_v<decltype(diagonal_divided)>);
    // compares (Matrix<double, 2, 2>(diagonal_divided)), (Matrix<double, 2, 2>({1.5, 0.0, 0.0, 2.5})) using eq
    // semantics
    EXPECT_EQ((Matrix<double, 2, 2>(diagonal_divided)), (Matrix<double, 2, 2>({1.5, 0.0, 0.0, 2.5})));
    const LowerTriangularMatrix<double, 2, 2> triangular(symmetric_values);
    const auto triangular_divided = triangular / 2;
    // checks at compile time: is_triangular_matrix_v<decltype(triangular_divided)>
    static_assert(is_triangular_matrix_v<decltype(triangular_divided)>);
    // compares (Matrix<double, 2, 2>(triangular_divided)), (Matrix<double, 2, 2>({1.5, 0.0, -2.5, 3.5})) using
    // eq semantics
    EXPECT_EQ((Matrix<double, 2, 2>(triangular_divided)), (Matrix<double, 2, 2>({1.5, 0.0, -2.5, 3.5})));
}

}   // namespace fdapde
