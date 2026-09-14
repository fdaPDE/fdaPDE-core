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
// floating-point matrices expose a Euclidean norm
static_assert(permits_norm<floating_norm_matrix>);
// floating-point matrices expose rowwise Euclidean norms
static_assert(permits_rowwise_norm<floating_norm_matrix>);
// floating-point matrices expose columnwise Euclidean norms
static_assert(permits_colwise_norm<floating_norm_matrix>);
// floating-point matrices expose coefficientwise square roots
static_assert(permits_coefficient_sqrt<floating_norm_matrix>);
// integral matrices do not expose a truncating Euclidean norm
static_assert(!permits_norm<integral_norm_matrix>);
// integral matrices do not expose truncating rowwise Euclidean norms
static_assert(!permits_rowwise_norm<integral_norm_matrix>);
// integral matrices do not expose truncating columnwise Euclidean norms
static_assert(!permits_colwise_norm<integral_norm_matrix>);
// integral matrices do not expose truncating coefficientwise square roots
static_assert(!permits_coefficient_sqrt<integral_norm_matrix>);

template <int StorageOrder> void check_matrix_block_runtime_contracts() {
    using matrix_type = Matrix<int, 3, 4, StorageOrder>;
    matrix_type matrix({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    // row extraction rejects a negative row
    EXPECT_THROW(static_cast<void>(matrix.row(-1)), std::out_of_range);
    // row extraction rejects a row equal to the matrix height
    EXPECT_THROW(static_cast<void>(matrix.row(3)), std::out_of_range);
    // column extraction rejects a negative column
    EXPECT_THROW(static_cast<void>(matrix.col(-1)), std::out_of_range);
    // column extraction rejects a column equal to the matrix width
    EXPECT_THROW(static_cast<void>(matrix.col(4)), std::out_of_range);
    // block extraction rejects a negative row origin
    EXPECT_THROW(static_cast<void>(matrix.block(-1, 0, 1, 1)), std::out_of_range);
    // block extraction rejects a negative column origin
    EXPECT_THROW(static_cast<void>(matrix.block(0, -1, 1, 1)), std::out_of_range);
    // block extraction rejects a rectangle extending beyond the matrix
    EXPECT_THROW(static_cast<void>(matrix.block(2, 3, 2, 2)), std::out_of_range);
    // block extraction rejects a zero row extent
    EXPECT_THROW(static_cast<void>(matrix.block(0, 0, 0, 2)), std::invalid_argument);
    // block extraction rejects a negative row extent
    EXPECT_THROW(static_cast<void>(matrix.block(0, 0, -1, 2)), std::invalid_argument);
    // static block extraction rejects an out-of-bounds rectangle
    EXPECT_THROW(static_cast<void>(matrix.template block<2, 2>(2, 3)), std::out_of_range);
    // top-row extraction rejects an empty selection
    EXPECT_THROW(static_cast<void>(matrix.top_rows(0)), std::invalid_argument);
    // bottom-row extraction rejects an empty selection
    EXPECT_THROW(static_cast<void>(matrix.bottom_rows(0)), std::invalid_argument);
    // left-column extraction rejects an empty selection
    EXPECT_THROW(static_cast<void>(matrix.left_cols(0)), std::invalid_argument);
    // right-column extraction rejects an empty selection
    EXPECT_THROW(static_cast<void>(matrix.right_cols(0)), std::invalid_argument);
    // top-row extraction rejects more rows than the matrix contains
    EXPECT_THROW(static_cast<void>(matrix.top_rows(4)), std::out_of_range);
    // bottom-row extraction rejects more rows than the matrix contains
    EXPECT_THROW(static_cast<void>(matrix.bottom_rows(4)), std::out_of_range);
    // left-column extraction rejects more columns than the matrix contains
    EXPECT_THROW(static_cast<void>(matrix.left_cols(5)), std::out_of_range);
    // right-column extraction rejects more columns than the matrix contains
    EXPECT_THROW(static_cast<void>(matrix.right_cols(5)), std::out_of_range);
    // bottom-row extraction rejects the minimum int count without arithmetic overflow
    EXPECT_THROW(static_cast<void>(matrix.bottom_rows(std::numeric_limits<int>::min())), std::invalid_argument);
    // right-column extraction rejects the minimum int count without arithmetic overflow
    EXPECT_THROW(static_cast<void>(matrix.right_cols(std::numeric_limits<int>::min())), std::invalid_argument);

    const auto zero = [](int, int) { return 0.0; };
    ProceduralMatrix<decltype(zero), Dynamic, 1> long_column(50000, zero);
    ProceduralMatrix<decltype(zero), 1, Dynamic> long_row(50000, zero);
    auto oversized_outer_product = long_column * long_row;
    // block construction rejects a logical size exceeding the supported integer range
    EXPECT_THROW(static_cast<void>(oversized_outer_product.block(0, 0, 50000, 50000)), std::length_error);

    auto block = matrix.template block<2, 2>(1, 1);
    const auto& const_block = block;
    // mutable block access rejects a negative local row
    EXPECT_THROW(static_cast<void>(block(-1, 0)), std::out_of_range);
    // mutable block access rejects a local column equal to the block width
    EXPECT_THROW(static_cast<void>(block(0, 2)), std::out_of_range);
    // const block access rejects a negative local row
    EXPECT_THROW(static_cast<void>(const_block(-1, 0)), std::out_of_range);
    // const block access rejects a local column equal to the block width
    EXPECT_THROW(static_cast<void>(const_block(0, 2)), std::out_of_range);

    auto row = matrix.row(0);
    const auto& const_row = row;
    // mutable row indexing rejects a negative index
    EXPECT_THROW(static_cast<void>(row[-1]), std::out_of_range);
    // mutable row indexing rejects an index equal to the row length
    EXPECT_THROW(static_cast<void>(row[row.size()]), std::out_of_range);
    // const row indexing rejects a negative index
    EXPECT_THROW(static_cast<void>(const_row[-1]), std::out_of_range);
    // const row indexing rejects an index equal to the row length
    EXPECT_THROW(static_cast<void>(const_row[const_row.size()]), std::out_of_range);
    const matrix_type original = matrix;
    const std::initializer_list<int> short_row {20, 21};
    // row assignment rejects a source with incompatible length
    EXPECT_THROW(row = short_row, std::invalid_argument);
    // failed block and row operations preserve the complete original matrix
    EXPECT_EQ(matrix, original);
}

template <int StorageOrder> void check_matrix_reshape_runtime_contracts() {
    using matrix_type = Matrix<int, Dynamic, Dynamic, StorageOrder>;
    matrix_type matrix(2, 2);

    // reshape rejects a negative row extent
    EXPECT_THROW(static_cast<void>(matrix.reshape(-1, 4)), std::invalid_argument);
    // reshape rejects a negative column extent
    EXPECT_THROW(static_cast<void>(matrix.reshape(1, -1)), std::invalid_argument);
    // reshape rejects a target matrix with a different coefficient count
    EXPECT_THROW(static_cast<void>(matrix.reshape(3, 2)), std::invalid_argument);
    // vector reshape rejects a target length with a different coefficient count
    EXPECT_THROW(static_cast<void>(matrix.reshape(3)), std::invalid_argument);
    // reshape rejects an overflowing target coefficient count
    EXPECT_THROW(static_cast<void>(matrix.reshape(std::numeric_limits<int>::max(), 2)), std::length_error);
    // direct reshape construction rejects rows inconsistent with its static extent
    EXPECT_THROW(static_cast<void>(ReshapeOp<2, Dynamic, matrix_type>(matrix, 3, 2)), std::invalid_argument);
    // direct reshape construction rejects columns inconsistent with its static extent
    EXPECT_THROW(static_cast<void>(ReshapeOp<Dynamic, 2, matrix_type>(matrix, 2, 3)), std::invalid_argument);

    auto reshaped = matrix.template reshape<1, 4>();
    const auto& const_reshaped = reshaped;
    // mutable reshape access rejects a negative row
    EXPECT_THROW(static_cast<void>(reshaped(-1, 0)), std::out_of_range);
    // mutable reshape access rejects a column equal to its width
    EXPECT_THROW(static_cast<void>(reshaped(0, 4)), std::out_of_range);
    // const reshape access rejects a negative row
    EXPECT_THROW(static_cast<void>(const_reshaped(-1, 0)), std::out_of_range);
    // const reshape access rejects a column equal to its width
    EXPECT_THROW(static_cast<void>(const_reshaped(0, 4)), std::out_of_range);
    // mutable row-reshape indexing rejects a negative index
    EXPECT_THROW(static_cast<void>(reshaped[-1]), std::out_of_range);
    // mutable row-reshape indexing rejects an index equal to its length
    EXPECT_THROW(static_cast<void>(reshaped[4]), std::out_of_range);
    // const row-reshape indexing rejects a negative index
    EXPECT_THROW(static_cast<void>(const_reshaped[-1]), std::out_of_range);
    // const row-reshape indexing rejects an index equal to its length
    EXPECT_THROW(static_cast<void>(const_reshaped[4]), std::out_of_range);

    auto column = matrix.reshape(4);
    const auto& const_column = column;
    // mutable column-reshape indexing rejects a negative index
    EXPECT_THROW(static_cast<void>(column[-1]), std::out_of_range);
    // mutable column-reshape indexing rejects an index equal to its length
    EXPECT_THROW(static_cast<void>(column[4]), std::out_of_range);
    // const column-reshape indexing rejects a negative index
    EXPECT_THROW(static_cast<void>(const_column[-1]), std::out_of_range);
    // const column-reshape indexing rejects an index equal to its length
    EXPECT_THROW(static_cast<void>(const_column[4]), std::out_of_range);

    matrix_type empty;
    const auto empty_matrix = empty.reshape(0, 5);
    const auto empty_column = empty.reshape(0);
    // reshaping an empty source permits zero rows
    EXPECT_EQ(empty_matrix.rows(), 0);
    // reshaping an empty source preserves the requested five columns
    EXPECT_EQ(empty_matrix.cols(), 5);
    // the empty matrix reshape contains no coefficients
    EXPECT_EQ(empty_matrix.size(), 0);
    // an empty vector reshape has zero rows
    EXPECT_EQ(empty_column.rows(), 0);
    // an empty vector reshape retains one column
    EXPECT_EQ(empty_column.cols(), 1);
    // an empty vector reshape contains no coefficients
    EXPECT_EQ(empty_column.size(), 0);
    // a nonempty source cannot be reshaped to an empty target
    EXPECT_THROW(static_cast<void>(matrix.reshape(0, 5)), std::invalid_argument);

    const auto zero = [](int, int) { return 0.0; };
    ProceduralMatrix<decltype(zero), Dynamic, 1> long_column(50000, zero);
    ProceduralMatrix<decltype(zero), 1, Dynamic> long_row(50000, zero);
    auto oversized_outer_product = long_column * long_row;
    // reshape rejects an overflowing source size before comparing target dimensions
    EXPECT_THROW(static_cast<void>(oversized_outer_product.reshape(1, 1)), std::length_error);

    Matrix<bool, Dynamic, Dynamic, StorageOrder> boolean_matrix(2, 2);
    // boolean reshape rejects a target with a different logical bit count
    EXPECT_THROW(static_cast<void>(boolean_matrix.reshape(3, 2)), std::invalid_argument);
}

template <int StorageOrder> void check_matrix_coeffwise_runtime_contracts() {
    using matrix_type = Matrix<double, Dynamic, Dynamic, StorageOrder>;
    matrix_type lhs(2, 2);
    matrix_type rhs(1, 4);
    lhs.cwise() = 1.0;
    rhs.cwise() = 2.0;
    const matrix_type original = lhs;
    // coefficientwise addition rejects incompatible runtime shapes
    EXPECT_THROW(static_cast<void>(lhs.cwise() + rhs.cwise()), std::invalid_argument);
    // coefficientwise compound addition rejects incompatible runtime shapes
    EXPECT_THROW(lhs.cwise() += rhs.cwise(), std::invalid_argument);
    // failed coefficientwise compound assignment preserves the destination
    EXPECT_EQ(lhs, original);

    Matrix<double, Dynamic, 2, StorageOrder> partial_lhs(2, 2);
    Matrix<double, Dynamic, 2, StorageOrder> partial_rhs(3, 2);
    // partially dynamic coefficientwise addition validates its unresolved extents
    EXPECT_THROW(static_cast<void>(partial_lhs.cwise() + partial_rhs.cwise()), std::invalid_argument);

    auto cwise = lhs.cwise();
    const auto& const_cwise = cwise;
    // mutable coefficientwise access rejects a negative row
    EXPECT_THROW(static_cast<void>(cwise(-1, 0)), std::out_of_range);
    // mutable coefficientwise access rejects a column equal to its width
    EXPECT_THROW(static_cast<void>(cwise(0, 2)), std::out_of_range);
    // const coefficientwise access rejects a negative row
    EXPECT_THROW(static_cast<void>(const_cwise(-1, 0)), std::out_of_range);
    // const coefficientwise access rejects a column equal to its width
    EXPECT_THROW(static_cast<void>(const_cwise(0, 2)), std::out_of_range);

    auto mwise = cwise.mwise();
    const auto& const_mwise = mwise;
    // matrix-semantics adaptor access rejects a negative row
    EXPECT_THROW(static_cast<void>(mwise(-1, 0)), std::out_of_range);
    // matrix-semantics adaptor access rejects a column equal to its width
    EXPECT_THROW(static_cast<void>(mwise(0, 2)), std::out_of_range);
    // const matrix-semantics adaptor access rejects a negative row
    EXPECT_THROW(static_cast<void>(const_mwise(-1, 0)), std::out_of_range);
    // const matrix-semantics adaptor access rejects a column equal to its width
    EXPECT_THROW(static_cast<void>(const_mwise(0, 2)), std::out_of_range);

    auto transformed = lhs.cwise().sqrt();
    // transformed coefficientwise access rejects a negative row
    EXPECT_THROW(static_cast<void>(transformed(-1, 0)), std::out_of_range);
    // transformed coefficientwise access rejects a column equal to its width
    EXPECT_THROW(static_cast<void>(transformed(0, 2)), std::out_of_range);

    auto binary = lhs.cwise() + lhs.cwise();
    // binary coefficientwise access rejects a negative row
    EXPECT_THROW(static_cast<void>(binary(-1, 0)), std::out_of_range);
    // binary coefficientwise access rejects a column equal to its width
    EXPECT_THROW(static_cast<void>(binary(0, 2)), std::out_of_range);

    Matrix<double, 1, 3, StorageOrder> row({1.0, 2.0, 3.0});
    auto row_cwise = row.cwise();
    auto row_mwise = row_cwise.mwise();
    auto row_binary = row.cwise() + row.cwise();
    // coefficientwise row indexing reads the third source coefficient
    EXPECT_DOUBLE_EQ(row_cwise[2], 3.0);
    // returning to matrix semantics preserves the third row coefficient
    EXPECT_DOUBLE_EQ(row_mwise[2], 3.0);
    // binary row-expression indexing evaluates the doubled third coefficient
    EXPECT_DOUBLE_EQ(row_binary[2], 6.0);
    // coefficientwise row indexing rejects a negative index
    EXPECT_THROW(static_cast<void>(row_cwise[-1]), std::out_of_range);
    // coefficientwise row indexing rejects an index equal to the row length
    EXPECT_THROW(static_cast<void>(row_cwise[3]), std::out_of_range);
    // matrix-semantics row indexing rejects a negative index
    EXPECT_THROW(static_cast<void>(row_mwise[-1]), std::out_of_range);
    // matrix-semantics row indexing rejects an index equal to the row length
    EXPECT_THROW(static_cast<void>(row_mwise[3]), std::out_of_range);
    // binary row-expression indexing rejects a negative index
    EXPECT_THROW(static_cast<void>(row_binary[-1]), std::out_of_range);
    // binary row-expression indexing rejects an index equal to the row length
    EXPECT_THROW(static_cast<void>(row_binary[3]), std::out_of_range);
}

template <int StorageOrder> void check_matrix_vectorwise_runtime_contracts() {
    using dynamic_matrix = Matrix<double, Dynamic, Dynamic, StorageOrder>;
    dynamic_matrix matrix = Matrix<double, 2, 3, StorageOrder>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    const dynamic_matrix original = matrix;
    const Matrix<double, Dynamic, 1, StorageOrder> wrong_rows(std::vector<double> {1.0, 2.0, 3.0});
    const Matrix<double, 1, Dynamic, StorageOrder> wrong_cols(std::vector<double> {1.0, 2.0, 3.0, 4.0});

    // rowwise assignment rejects a source with incompatible dimensions
    EXPECT_THROW(static_cast<void>(matrix.rowwise() = wrong_rows), std::invalid_argument);
    // failed rowwise assignment preserves every destination coefficient
    EXPECT_EQ(matrix, original);
    // rowwise addition assignment rejects a source with incompatible dimensions
    EXPECT_THROW(static_cast<void>(matrix.rowwise() += wrong_rows), std::invalid_argument);
    // failed rowwise addition preserves every destination coefficient
    EXPECT_EQ(matrix, original);
    // rowwise subtraction assignment rejects a source with incompatible dimensions
    EXPECT_THROW(static_cast<void>(matrix.rowwise() -= wrong_rows), std::invalid_argument);
    // failed rowwise subtraction preserves every destination coefficient
    EXPECT_EQ(matrix, original);
    // columnwise assignment rejects a source with incompatible dimensions
    EXPECT_THROW(static_cast<void>(matrix.colwise() = wrong_cols), std::invalid_argument);
    // failed columnwise assignment preserves every destination coefficient
    EXPECT_EQ(matrix, original);
    // columnwise addition assignment rejects a source with incompatible dimensions
    EXPECT_THROW(static_cast<void>(matrix.colwise() += wrong_cols), std::invalid_argument);
    // failed columnwise addition preserves every destination coefficient
    EXPECT_EQ(matrix, original);
    // columnwise subtraction assignment rejects a source with incompatible dimensions
    EXPECT_THROW(static_cast<void>(matrix.colwise() -= wrong_cols), std::invalid_argument);
    // failed columnwise subtraction preserves every destination coefficient
    EXPECT_EQ(matrix, original);
    // rowwise comparison rejects an incompatible source shape
    EXPECT_THROW(static_cast<void>(matrix.rowwise() == wrong_rows), std::invalid_argument);
    // columnwise comparison rejects an incompatible source shape
    EXPECT_THROW(static_cast<void>(matrix.colwise() == wrong_cols), std::invalid_argument);

    const dynamic_matrix valid_rows = Matrix<double, 2, 1, StorageOrder>(std::vector<double> {10.0, 20.0});
    matrix.rowwise() = valid_rows;
    // valid rowwise assignment broadcasts each supplied row value across its row
    EXPECT_EQ(matrix, (Matrix<double, 2, 3, StorageOrder>({10.0, 10.0, 10.0, 20.0, 20.0, 20.0})));
    // rowwise comparison recognizes the broadcast reference values
    EXPECT_TRUE(matrix.rowwise() == valid_rows);
    matrix = original;
    matrix.rowwise() += valid_rows;
    // valid rowwise addition adds the supplied value to each coefficient in its row
    EXPECT_EQ(matrix, (Matrix<double, 2, 3, StorageOrder>({11.0, 12.0, 13.0, 24.0, 25.0, 26.0})));
    matrix = original;
    matrix.rowwise() -= valid_rows;
    // valid rowwise subtraction subtracts the supplied value from each coefficient in its row
    EXPECT_EQ(matrix, (Matrix<double, 2, 3, StorageOrder>({-9.0, -8.0, -7.0, -16.0, -15.0, -14.0})));

    const dynamic_matrix valid_cols = Matrix<double, 1, 3, StorageOrder>(std::vector<double> {10.0, 20.0, 30.0});
    matrix.colwise() = valid_cols;
    // valid columnwise assignment broadcasts each supplied column value down its column
    EXPECT_EQ(matrix, (Matrix<double, 2, 3, StorageOrder>({10.0, 20.0, 30.0, 10.0, 20.0, 30.0})));
    // columnwise comparison recognizes the broadcast reference values
    EXPECT_TRUE(matrix.colwise() == valid_cols);
    matrix = original;
    matrix.colwise() += valid_cols;
    // valid columnwise addition adds the supplied value to each coefficient in its column
    EXPECT_EQ(matrix, (Matrix<double, 2, 3, StorageOrder>({11.0, 22.0, 33.0, 14.0, 25.0, 36.0})));
    matrix = original;
    matrix.colwise() -= valid_cols;
    // valid columnwise subtraction subtracts the supplied value from each coefficient in its column
    EXPECT_EQ(matrix, (Matrix<double, 2, 3, StorageOrder>({-9.0, -18.0, -27.0, -6.0, -15.0, -24.0})));

    Matrix<double, Dynamic, 3, StorageOrder> partial_rows(2, 3);
    const Matrix<double, Dynamic, 1, StorageOrder> valid_partial_rows(std::vector<double> {2.0, 3.0});
    partial_rows.rowwise() = valid_partial_rows;
    // partially dynamic rowwise assignment broadcasts the two compatible values
    EXPECT_EQ(partial_rows, (Matrix<double, 2, 3, StorageOrder>({2.0, 2.0, 2.0, 3.0, 3.0, 3.0})));
    const auto partial_rows_snapshot = partial_rows;
    // partially dynamic rowwise assignment rejects incompatible runtime extents
    EXPECT_THROW(static_cast<void>(partial_rows.rowwise() = wrong_rows), std::invalid_argument);
    // failed partially dynamic rowwise assignment preserves the original coefficients
    EXPECT_EQ(partial_rows, partial_rows_snapshot);

    Matrix<double, 2, Dynamic, StorageOrder> partial_cols(2, 3);
    const Matrix<double, 1, Dynamic, StorageOrder> valid_partial_cols(std::vector<double> {2.0, 3.0, 4.0});
    partial_cols.colwise() = valid_partial_cols;
    // partially dynamic columnwise assignment broadcasts the three compatible values
    EXPECT_EQ(partial_cols, (Matrix<double, 2, 3, StorageOrder>({2.0, 3.0, 4.0, 2.0, 3.0, 4.0})));
    const auto partial_cols_snapshot = partial_cols;
    // partially dynamic columnwise assignment rejects incompatible runtime extents
    EXPECT_THROW(static_cast<void>(partial_cols.colwise() = wrong_cols), std::invalid_argument);
    // failed partially dynamic columnwise assignment preserves the original coefficients
    EXPECT_EQ(partial_cols, partial_cols_snapshot);

    auto row_sums = original.rowwise().sum();
    // row-sum coordinate access returns the second row's sum
    EXPECT_DOUBLE_EQ(row_sums(1, 0), 15.0);
    // row-sum vector indexing returns the same second-row sum
    EXPECT_DOUBLE_EQ(row_sums[1], 15.0);
    // row-sum access rejects a negative row
    EXPECT_THROW(static_cast<void>(row_sums(-1, 0)), std::out_of_range);
    // row-sum access rejects a row equal to the result height
    EXPECT_THROW(static_cast<void>(row_sums(2, 0)), std::out_of_range);
    // row-sum access rejects a negative column
    EXPECT_THROW(static_cast<void>(row_sums(0, -1)), std::out_of_range);
    // row-sum access rejects a column beyond its single result column
    EXPECT_THROW(static_cast<void>(row_sums(0, 1)), std::out_of_range);
    // row-sum vector indexing rejects a negative index
    EXPECT_THROW(static_cast<void>(row_sums[-1]), std::out_of_range);
    // row-sum vector indexing rejects an index equal to the result length
    EXPECT_THROW(static_cast<void>(row_sums[2]), std::out_of_range);

    auto col_sums = original.colwise().sum();
    // column-sum coordinate access returns the final column's sum
    EXPECT_DOUBLE_EQ(col_sums(0, 2), 9.0);
    // column-sum vector indexing returns the same final-column sum
    EXPECT_DOUBLE_EQ(col_sums[2], 9.0);
    // column-sum access rejects a negative row
    EXPECT_THROW(static_cast<void>(col_sums(-1, 0)), std::out_of_range);
    // column-sum access rejects a row beyond its single result row
    EXPECT_THROW(static_cast<void>(col_sums(1, 0)), std::out_of_range);
    // column-sum access rejects a negative column
    EXPECT_THROW(static_cast<void>(col_sums(0, -1)), std::out_of_range);
    // column-sum access rejects a column equal to the result width
    EXPECT_THROW(static_cast<void>(col_sums(0, 3)), std::out_of_range);
    // column-sum vector indexing rejects a negative index
    EXPECT_THROW(static_cast<void>(col_sums[-1]), std::out_of_range);
    // column-sum vector indexing rejects an index equal to the result length
    EXPECT_THROW(static_cast<void>(col_sums[3]), std::out_of_range);
}

template <int StorageOrder> void check_matrix_reduction_runtime_contracts() {
    const auto expect_empty_shape = [](const auto& result, int rows, int cols) {
        // an empty reduction result retains the expected row extent
        EXPECT_EQ(result.rows(), rows);
        // an empty reduction result retains the expected column extent
        EXPECT_EQ(result.cols(), cols);
        // an empty reduction result contains no coefficients
        EXPECT_EQ(result.size(), 0);
    };

    const Matrix<double, 2, 3, StorageOrder> negative({-4.0, -2.0, -3.0, -9.0, -8.0, -7.0});
    // maximum over negative floating-point values returns the least negative coefficient
    EXPECT_DOUBLE_EQ(negative.max(), -2.0);
    // row maxima remain negative when every coefficient in each row is negative
    EXPECT_EQ(negative.rowwise().max(), (Matrix<double, 2, 1, StorageOrder>({-2.0, -7.0})));
    // column maxima remain negative when every coefficient in each column is negative
    EXPECT_EQ(negative.colwise().max(), (Matrix<double, 1, 3, StorageOrder>({-4.0, -2.0, -3.0})));

    const Matrix<int, Dynamic, Dynamic, StorageOrder> dynamic_negative =
      Matrix<int, 2, 3, StorageOrder>({-6, -5, -4, -3, -2, -1});
    // maximum over negative integers returns the least negative coefficient
    EXPECT_EQ(dynamic_negative.max(), -1);

    const Matrix<double, Dynamic, Dynamic, StorageOrder> empty;
    // sum over an empty floating-point matrix uses the additive identity zero
    EXPECT_DOUBLE_EQ(empty.sum(), 0.0);
    // product over an empty floating-point matrix uses the multiplicative identity one
    EXPECT_DOUBLE_EQ(empty.prod(), 1.0);
    // mean of an empty floating-point matrix is undefined
    EXPECT_THROW(static_cast<void>(empty.mean()), std::domain_error);
    // maximum of an empty floating-point matrix is undefined
    EXPECT_THROW(static_cast<void>(empty.max()), std::domain_error);
    // minimum of an empty floating-point matrix is undefined
    EXPECT_THROW(static_cast<void>(empty.min()), std::domain_error);

    const Matrix<int, Dynamic, Dynamic, StorageOrder> empty_int;
    // mean of an empty integral matrix is undefined
    EXPECT_THROW(static_cast<void>(empty_int.mean()), std::domain_error);
    // maximum of an empty integral matrix is undefined
    EXPECT_THROW(static_cast<void>(empty_int.max()), std::domain_error);
    // minimum of an empty integral matrix is undefined
    EXPECT_THROW(static_cast<void>(empty_int.min()), std::domain_error);

    const Matrix<double, 2, Dynamic, StorageOrder> empty_row_axes(2, 0);
    // whole-matrix mean is undefined when every row is empty
    EXPECT_THROW(static_cast<void>(empty_row_axes.mean()), std::domain_error);
    // whole-matrix maximum is undefined when every row is empty
    EXPECT_THROW(static_cast<void>(empty_row_axes.max()), std::domain_error);
    // whole-matrix minimum is undefined when every row is empty
    EXPECT_THROW(static_cast<void>(empty_row_axes.min()), std::domain_error);
    // summing each empty row yields the additive identity zero
    EXPECT_EQ(empty_row_axes.rowwise().sum(), (Matrix<double, 2, 1, StorageOrder>({0.0, 0.0})));
    // multiplying each empty row yields the multiplicative identity one
    EXPECT_EQ(empty_row_axes.rowwise().prod(), (Matrix<double, 2, 1, StorageOrder>({1.0, 1.0})));
    // evaluating the mean of an empty row raises domain_error
    EXPECT_THROW(static_cast<void>(empty_row_axes.rowwise().mean()(0, 0)), std::domain_error);
    // evaluating the maximum of an empty row raises domain_error
    EXPECT_THROW(static_cast<void>(empty_row_axes.rowwise().max()(0, 0)), std::domain_error);
    // evaluating the minimum of an empty row raises domain_error
    EXPECT_THROW(static_cast<void>(empty_row_axes.rowwise().min()(0, 0)), std::domain_error);
    expect_empty_shape(empty_row_axes.colwise().mean(), 1, 0);
    expect_empty_shape(empty_row_axes.colwise().max(), 1, 0);
    expect_empty_shape(empty_row_axes.colwise().min(), 1, 0);

    const Matrix<double, Dynamic, 3, StorageOrder> empty_col_axes(0, 3);
    // whole-matrix mean is undefined when every column is empty
    EXPECT_THROW(static_cast<void>(empty_col_axes.mean()), std::domain_error);
    // whole-matrix maximum is undefined when every column is empty
    EXPECT_THROW(static_cast<void>(empty_col_axes.max()), std::domain_error);
    // whole-matrix minimum is undefined when every column is empty
    EXPECT_THROW(static_cast<void>(empty_col_axes.min()), std::domain_error);
    // summing each empty column yields the additive identity zero
    EXPECT_EQ(empty_col_axes.colwise().sum(), (Matrix<double, 1, 3, StorageOrder>({0.0, 0.0, 0.0})));
    // multiplying each empty column yields the multiplicative identity one
    EXPECT_EQ(empty_col_axes.colwise().prod(), (Matrix<double, 1, 3, StorageOrder>({1.0, 1.0, 1.0})));
    // evaluating the mean of an empty column raises domain_error
    EXPECT_THROW(static_cast<void>(empty_col_axes.colwise().mean()(0, 0)), std::domain_error);
    // evaluating the maximum of an empty column raises domain_error
    EXPECT_THROW(static_cast<void>(empty_col_axes.colwise().max()(0, 0)), std::domain_error);
    // evaluating the minimum of an empty column raises domain_error
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
    // row means preserve the integral scalar type
    static_assert(std::is_same_v<typename RowMean::Scalar, int>);
    // column means preserve the integral scalar type
    static_assert(std::is_same_v<typename ColMean::Scalar, int>);
    // row-mean evaluation returns an integer value
    static_assert(std::is_same_v<decltype(row_means(0, 0)), int>);
    // column-mean evaluation returns an integer value
    static_assert(std::is_same_v<decltype(col_means(0, 0)), int>);
    // row means retain the fixed two-row column-vector shape
    static_assert(RowMean::Rows == 2 && RowMean::Cols == 1);
    // column means retain the fixed two-column row-vector shape
    static_assert(ColMean::Rows == 1 && ColMean::Cols == 2);
    // integral row means use integer division toward zero
    EXPECT_EQ(row_means, (Matrix<int, 2, 1, StorageOrder>({-1, -5})));
    // integral column means use integer division toward zero
    EXPECT_EQ(col_means, (Matrix<int, 1, 2, StorageOrder>({-2, -4})));
}

template <int StorageOrder> void check_matrix_norm_runtime_contracts() {
    using pair_type = Matrix<double, 1, 2, StorageOrder>;
    const pair_type tiny({3.0e-8, 4.0e-8});
    const pair_type underflowing_square({3.0e-200, 4.0e-200});
    const pair_type overflowing_square({3.0e200, 4.0e200});
    // the Euclidean norm resolves a small 3-4-5 vector accurately
    EXPECT_DOUBLE_EQ(tiny.norm(), 5.0e-8);
    // scaled norm evaluation avoids underflow from squaring tiny coefficients
    EXPECT_DOUBLE_EQ(underflowing_square.norm(), 5.0e-200);
    // scaled norm evaluation avoids overflow from squaring huge coefficients
    EXPECT_DOUBLE_EQ(overflowing_square.norm(), 5.0e200);

    const Matrix<double, 3, 2, StorageOrder> rows({3.0e-8, 4.0e-8, 3.0e-200, 4.0e-200, 3.0e200, 4.0e200});
    const auto row_norms = rows.rowwise().norm();
    // rowwise norm resolves the small-scale 3-4-5 row
    EXPECT_DOUBLE_EQ(row_norms[0], 5.0e-8);
    // rowwise norm avoids square underflow in the tiny row
    EXPECT_DOUBLE_EQ(row_norms[1], 5.0e-200);
    // rowwise norm avoids square overflow in the huge row
    EXPECT_DOUBLE_EQ(row_norms[2], 5.0e200);

    const Matrix<double, 2, 3, StorageOrder> cols({3.0e-8, 3.0e-200, 3.0e200, 4.0e-8, 4.0e-200, 4.0e200});
    const auto col_norms = cols.colwise().norm();
    // columnwise norm resolves the small-scale 3-4-5 column
    EXPECT_DOUBLE_EQ(col_norms[0], 5.0e-8);
    // columnwise norm avoids square underflow in the tiny column
    EXPECT_DOUBLE_EQ(col_norms[1], 5.0e-200);
    // columnwise norm avoids square overflow in the huge column
    EXPECT_DOUBLE_EQ(col_norms[2], 5.0e200);

    const pair_type tiny_squares({1.0e-16, 4.0e-16});
    const pair_type roots = tiny_squares.cwise().sqrt();
    // coefficientwise square root resolves the first tiny positive coefficient
    EXPECT_DOUBLE_EQ(roots[0], 1.0e-8);
    // coefficientwise square root resolves the second tiny positive coefficient
    EXPECT_DOUBLE_EQ(roots[1], 2.0e-8);

    const double view_data[2] {3.0, 4.0};
    const MatrixView<const double, 1, 2, StorageOrder> const_view(view_data);
    // euclidean norm is available through a const-storage view
    EXPECT_DOUBLE_EQ(const_view.norm(), 5.0);
}

template <int StorageOrder> void check_matrix_inf_norm_runtime_contracts() {
    const Matrix<double, 2, 3, StorageOrder> zero = Matrix<double, 2, 3, StorageOrder>::Zero();
    // the infinity norm of a zero matrix is zero
    EXPECT_DOUBLE_EQ(zero.inf_norm(), 0.0);
    // each all-zero row has infinity norm zero
    EXPECT_EQ(zero.rowwise().inf_norm(), (Matrix<double, 2, 1, StorageOrder>({0.0, 0.0})));
    // each all-zero column has infinity norm zero
    EXPECT_EQ(zero.colwise().inf_norm(), (Matrix<double, 1, 3, StorageOrder>({0.0, 0.0, 0.0})));

    const double denormal = std::numeric_limits<double>::denorm_min();
    const Matrix<double, 1, 1, StorageOrder> subnormal({denormal});
    // the infinity norm preserves a subnormal maximum rather than replacing it with a positive identity
    EXPECT_DOUBLE_EQ(subnormal.inf_norm(), denormal);
    // rowwise infinity norm preserves the subnormal maximum
    EXPECT_DOUBLE_EQ(subnormal.rowwise().inf_norm()[0], denormal);
    // columnwise infinity norm preserves the subnormal maximum
    EXPECT_DOUBLE_EQ(subnormal.colwise().inf_norm()[0], denormal);

    const Matrix<double, Dynamic, Dynamic, StorageOrder> empty;
    // the infinity norm of an empty matrix uses zero as its identity
    EXPECT_DOUBLE_EQ(empty.inf_norm(), 0.0);

    const Matrix<double, 2, Dynamic, StorageOrder> empty_row_axes(2, 0);
    // each empty row has infinity norm zero
    EXPECT_EQ(empty_row_axes.rowwise().inf_norm(), (Matrix<double, 2, 1, StorageOrder>({0.0, 0.0})));

    const Matrix<double, Dynamic, 3, StorageOrder> empty_col_axes(0, 3);
    // each empty column has infinity norm zero
    EXPECT_EQ(empty_col_axes.colwise().inf_norm(), (Matrix<double, 1, 3, StorageOrder>({0.0, 0.0, 0.0})));

    const Matrix<double, 2, 3, StorageOrder> view_owner({-4.0, 0.0, 2.0, 1.0, -3.0, 5.0});
    const MatrixView<const double, 2, 3, StorageOrder> const_view(view_owner.data());
    // a const view's infinity norm equals its largest absolute coefficient
    EXPECT_DOUBLE_EQ(const_view.inf_norm(), 5.0);
    // rowwise infinity norm through a const view takes each row's largest absolute coefficient
    EXPECT_EQ(const_view.rowwise().inf_norm(), (Matrix<double, 2, 1, StorageOrder>({4.0, 5.0})));
    // columnwise infinity norm through a const view takes each column's largest absolute coefficient
    EXPECT_EQ(const_view.colwise().inf_norm(), (Matrix<double, 1, 3, StorageOrder>({4.0, 3.0, 5.0})));
}

template <int StorageOrder> void check_empty_boolean_runtime_contracts() {
    using dynamic_matrix = Matrix<bool, Dynamic, Dynamic, StorageOrder>;

    const dynamic_matrix empty;
    // all uses the vacuous true identity for an empty Boolean matrix
    EXPECT_TRUE(empty.all());
    // any uses the false identity for an empty Boolean matrix
    EXPECT_FALSE(empty.any());
    // count returns zero for an empty Boolean matrix
    EXPECT_EQ(empty.count(), 0);
    // two default empty Boolean matrices compare equal
    EXPECT_TRUE(empty == dynamic_matrix());
    // two default empty Boolean matrices do not compare unequal
    EXPECT_FALSE(empty != dynamic_matrix());

    const dynamic_matrix zero_rows(0, 3);
    const dynamic_matrix zero_cols(3, 0);
    // all is true for a Boolean matrix with zero rows
    EXPECT_TRUE(zero_rows.all());
    // any is false for a Boolean matrix with zero rows
    EXPECT_FALSE(zero_rows.any());
    // count is zero for a Boolean matrix with zero rows
    EXPECT_EQ(zero_rows.count(), 0);
    // empty Boolean matrices with the same zero-row shape compare equal
    EXPECT_TRUE(zero_rows == dynamic_matrix(0, 3));
    // all is true for a Boolean matrix with zero columns
    EXPECT_TRUE(zero_cols.all());
    // any is false for a Boolean matrix with zero columns
    EXPECT_FALSE(zero_cols.any());
    // count is zero for a Boolean matrix with zero columns
    EXPECT_EQ(zero_cols.count(), 0);
    // empty Boolean matrices with the same zero-column shape compare equal
    EXPECT_TRUE(zero_cols == dynamic_matrix(3, 0));
    // boolean comparison rejects incompatible empty shapes
    EXPECT_THROW(static_cast<void>(zero_rows == zero_cols), std::invalid_argument);
    dynamic_matrix mismatched_assignment(0, 3);
    // compound Boolean assignment rejects incompatible empty shapes
    EXPECT_THROW(mismatched_assignment &= zero_cols, std::invalid_argument);
    // failed empty-shape assignment preserves the destination's zero rows
    EXPECT_EQ(mismatched_assignment.rows(), 0);
    // failed empty-shape assignment preserves its nonzero column extent
    EXPECT_EQ(mismatched_assignment.cols(), 3);

    dynamic_matrix assigned(1, 2);
    assigned(0, 1) = true;
    assigned = empty;
    // copy assignment from an empty owner adopts zero rows
    EXPECT_EQ(assigned.rows(), 0);
    // copy assignment from an empty owner adopts zero columns
    EXPECT_EQ(assigned.cols(), 0);
    // all remains vacuously true after assignment from an empty owner
    EXPECT_TRUE(assigned.all());
    // any remains false after assignment from an empty owner
    EXPECT_FALSE(assigned.any());
    // count remains zero after assignment from an empty owner
    EXPECT_EQ(assigned.count(), 0);

    dynamic_matrix expression_assigned(2, 1);
    expression_assigned(1, 0) = true;
    expression_assigned = ~empty;
    // assignment from an empty expression adopts zero rows
    EXPECT_EQ(expression_assigned.rows(), 0);
    // assignment from an empty expression adopts zero columns
    EXPECT_EQ(expression_assigned.cols(), 0);
    // all remains vacuously true after assignment from an empty expression
    EXPECT_TRUE(expression_assigned.all());
    // any remains false after assignment from an empty expression
    EXPECT_FALSE(expression_assigned.any());
    // count remains zero after assignment from an empty expression
    EXPECT_EQ(expression_assigned.count(), 0);

    std::ostringstream stream;
    stream << empty << zero_rows << zero_cols;
    // streaming an empty Boolean expression emits no coefficients
    EXPECT_TRUE(stream.str().empty());

    MatrixView<bool, Dynamic, Dynamic, StorageOrder> empty_view;
    // an empty Boolean view reports no packed storage words
    EXPECT_EQ(empty_view.bitpacks(), 0);
    empty_view.set();
    empty_view.clear();
}

}   // namespace

// exercise square-only operations on matrices whose dimensions are known at runtime
TEST(LinearAlgebraRuntimeContracts, DynamicSquareOperationsRemainAvailable) {
    Matrix<double, Dynamic, Dynamic> matrix = Matrix<double, 2, 2>({2, 1, 1, 3});
    const Matrix<double, 2, 2> zero = Matrix<double, 2, 2>::Zero();
    const Vector<double, 2> expected_diagonal({2, 3});
    const Matrix<double, Dynamic, Dynamic> inverse = matrix.inverse();

    // the symmetric part of an already symmetric matrix equals the input
    EXPECT_EQ(matrix.symm_part(), matrix);
    // the skew part of a symmetric matrix is zero
    EXPECT_EQ(matrix.skew_part(), zero);
    // the dynamic inverse multiplies its source to the identity within tolerance
    EXPECT_TRUE(almost_equal(inverse * matrix, Matrix<double, 2, 2>({1, 0, 0, 1})));
    // the dynamic determinant matches the explicit value five
    EXPECT_EQ(matrix.determinant(), 5);
    // dynamic diagonal extraction returns the expected two entries
    EXPECT_EQ(matrix.diagonal(), expected_diagonal);
}

// reject square-only operations on runtime rectangular matrices
TEST(LinearAlgebraRuntimeContracts, SquareOperationsRejectRectangularMatrices) {
    Matrix<double, Dynamic, Dynamic> matrix(2, 3);

    // symmetric-part extraction rejects a rectangular matrix
    EXPECT_THROW((void)matrix.symm_part(), std::invalid_argument);
    // skew-part extraction rejects a rectangular matrix
    EXPECT_THROW((void)matrix.skew_part(), std::invalid_argument);
    // inverse rejects a rectangular matrix
    EXPECT_THROW((void)matrix.inverse(), std::invalid_argument);
    // determinant rejects a rectangular matrix
    EXPECT_THROW((void)matrix.determinant(), std::invalid_argument);
}

// validate runtime arithmetic dimensions and the three-component cross-product contract
TEST(LinearAlgebraRuntimeContracts, DynamicBinaryExpressionsRejectIncompatibleShapes) {
    Matrix<double, Dynamic, Dynamic> lhs(2, 3);
    Matrix<double, Dynamic, Dynamic> wrong_cols_rhs(2, 2);
    Matrix<double, Dynamic, Dynamic> wrong_rows_rhs(3, 3);
    Matrix<double, Dynamic, Dynamic> product_rhs(2, 4);
    Vector<double, Dynamic> short_vector(2);
    Vector<double, Dynamic> vector(3);
    Matrix<double, Dynamic, Dynamic> non_column_vector(3, 2);

    // addition rejects a right operand with incompatible columns
    EXPECT_THROW((void)(lhs + wrong_cols_rhs), std::invalid_argument);
    // subtraction rejects a right operand with incompatible rows
    EXPECT_THROW((void)(lhs - wrong_rows_rhs), std::invalid_argument);
    // matrix multiplication rejects incompatible inner dimensions
    EXPECT_THROW((void)(lhs * product_rhs), std::invalid_argument);
    // cross product rejects a vector shorter than three components
    EXPECT_THROW((void)short_vector.cross(vector), std::invalid_argument);
    // cross product rejects a matrix that is not a column vector
    EXPECT_THROW((void)non_column_vector.cross(vector), std::invalid_argument);

    Vector<double, Dynamic> x(3);
    Vector<double, Dynamic> y(3);
    x[0] = 1;
    y[1] = 1;
    const Vector<double, 3> expected_cross({0, 0, 1});
    // the valid three-dimensional cross product matches the explicit reference
    EXPECT_EQ(x.cross(y), expected_cross);
}

// validate dense diagonal extraction and local coordinate bounds for mutable and const access
TEST(LinearAlgebraRuntimeContracts, DenseDiagonalViewChecksShapeAndIndexes) {
    Matrix<double, Dynamic, Dynamic> rectangular(2, 3);
    // diagonal extraction rejects a rectangular source
    EXPECT_THROW((void)rectangular.diagonal(), std::invalid_argument);

    Matrix<double, 2, 2> matrix({1, 2, 3, 4});
    auto diagonal = matrix.diagonal();
    const auto& const_diagonal = diagonal;

    // mutable diagonal access rejects a negative row
    EXPECT_THROW((void)diagonal(-1, 0), std::out_of_range);
    // mutable diagonal access rejects a row equal to its length
    EXPECT_THROW((void)diagonal(2, 0), std::out_of_range);
    // mutable diagonal access rejects a negative column
    EXPECT_THROW((void)diagonal(0, -1), std::out_of_range);
    // mutable diagonal access rejects a column beyond its single vector column
    EXPECT_THROW((void)diagonal(0, 1), std::out_of_range);
    // mutable diagonal indexing rejects a negative index
    EXPECT_THROW((void)diagonal[-1], std::out_of_range);
    // mutable diagonal indexing rejects an index equal to its length
    EXPECT_THROW((void)diagonal[2], std::out_of_range);
    // const diagonal access rejects a negative row
    EXPECT_THROW((void)const_diagonal(-1, 0), std::out_of_range);
    // const diagonal access rejects a row equal to its length
    EXPECT_THROW((void)const_diagonal(2, 0), std::out_of_range);
    // const diagonal access rejects a negative column
    EXPECT_THROW((void)const_diagonal(0, -1), std::out_of_range);
    // const diagonal access rejects a column beyond its single vector column
    EXPECT_THROW((void)const_diagonal(0, 1), std::out_of_range);
    // const diagonal indexing rejects a negative index
    EXPECT_THROW((void)const_diagonal[-1], std::out_of_range);
    // const diagonal indexing rejects an index equal to its length
    EXPECT_THROW((void)const_diagonal[2], std::out_of_range);
}

// validate owner dimensions and preserve existing storage when resize fails
TEST(LinearAlgebraRuntimeContracts, MatrixOwnerRejectsInvalidShapesAndSizes) {
    using dynamic_matrix = Matrix<int, Dynamic, Dynamic>;
    using partial_rows_matrix = Matrix<int, Dynamic, 3>;
    using partial_cols_matrix = Matrix<int, 2, Dynamic>;

    // owner construction rejects a negative dynamic extent
    EXPECT_THROW(static_cast<void>(dynamic_matrix(-1, 2)), std::invalid_argument);
    // owner construction rejects columns inconsistent with a fixed axis
    EXPECT_THROW(static_cast<void>(partial_rows_matrix(2, 4)), std::invalid_argument);
    // owner construction rejects rows inconsistent with a fixed axis
    EXPECT_THROW(static_cast<void>(partial_cols_matrix(3, 2)), std::invalid_argument);
    // owner construction rejects an overflowing coefficient count
    EXPECT_THROW(static_cast<void>(dynamic_matrix(std::numeric_limits<int>::max(), 2)), std::length_error);

    partial_rows_matrix partial(2, 3);
    partial(1, 2) = 17;
    // resize rejects columns inconsistent with a fixed axis
    EXPECT_THROW(partial.resize(2, 4), std::invalid_argument);
    // failed fixed-axis resize preserves the row count
    EXPECT_EQ(partial.rows(), 2);
    // failed fixed-axis resize preserves the column count
    EXPECT_EQ(partial.cols(), 3);
    // failed fixed-axis resize preserves the coefficient count
    EXPECT_EQ(partial.size(), 6);
    // failed fixed-axis resize preserves the previously written coefficient
    EXPECT_EQ(partial(1, 2), 17);

    dynamic_matrix bounded(1, 2);
    bounded(0, 1) = 23;
    // resize rejects an overflowing coefficient count
    EXPECT_THROW(bounded.resize(std::numeric_limits<int>::max(), 2), std::length_error);
    // failed overflowing resize preserves the row count
    EXPECT_EQ(bounded.rows(), 1);
    // failed overflowing resize preserves the column count
    EXPECT_EQ(bounded.cols(), 2);
    // failed overflowing resize preserves the previously written coefficient
    EXPECT_EQ(bounded(0, 1), 23);
}

// validate owner initialization, vector assignments and coordinate bounds
TEST(LinearAlgebraRuntimeContracts, MatrixOwnerChecksInputsAssignmentsAndIndexes) {
    using dynamic_matrix = Matrix<int, Dynamic, Dynamic>;
    using dynamic_vector = Vector<int, Dynamic>;
    using fixed_matrix = Matrix<int, 2, 3>;
    using fixed_vector = Vector<int, 3>;

    const std::vector<int> short_input {1, 2, 3, 4, 5};
    // fixed owner construction rejects an initializer with too few coefficients
    EXPECT_THROW(static_cast<void>(fixed_matrix(short_input)), std::invalid_argument);

    const dynamic_matrix wrong_matrix_shape(2, 4);
    // fixed owner construction rejects a matrix with incompatible shape
    EXPECT_THROW(static_cast<void>(fixed_matrix(wrong_matrix_shape)), std::invalid_argument);
    const dynamic_matrix non_vector_shape(2, 2);
    // fixed vector construction rejects a source that is not a vector
    EXPECT_THROW(static_cast<void>(fixed_vector(non_vector_shape)), std::invalid_argument);

    fixed_vector vector({1, 2, 3});
    const std::initializer_list<int> short_vector {4, 5};
    // fixed vector assignment rejects an incompatible source length
    EXPECT_THROW(vector = short_vector, std::invalid_argument);
    // failed vector assignment preserves the original three coefficients
    EXPECT_EQ(vector, fixed_vector({1, 2, 3}));

    fixed_matrix matrix({1, 2, 3, 4, 5, 6});
    const fixed_matrix& const_matrix = matrix;
    // mutable owner access rejects a negative row
    EXPECT_THROW(static_cast<void>(matrix(-1, 0)), std::out_of_range);
    // const owner access rejects a column equal to its width
    EXPECT_THROW(static_cast<void>(const_matrix(0, 3)), std::out_of_range);
    const fixed_vector& const_vector = vector;
    // mutable vector indexing rejects a negative index
    EXPECT_THROW(static_cast<void>(vector[-1]), std::out_of_range);
    // const vector indexing rejects an index equal to its length
    EXPECT_THROW(static_cast<void>(const_vector[3]), std::out_of_range);

    // equally spaced construction rejects a single sample that cannot span both endpoints
    EXPECT_THROW(static_cast<void>(dynamic_vector::LinSpaced(1, 0, 1)), std::invalid_argument);
}

// validate procedural dimensions, coefficient-count overflow and coordinate bounds
TEST(LinearAlgebraRuntimeContracts, ProceduralMatrixChecksShapeSizeAndIndexes) {
    const auto ones = [](int, int) { return 1; };
    using partial_procedural = ProceduralMatrix<decltype(ones), 2, Dynamic>;
    using dynamic_procedural = ProceduralMatrix<decltype(ones), Dynamic, Dynamic>;
    using fixed_procedural_vector = ProceduralMatrix<decltype(ones), 3, 1>;

    // procedural construction rejects a runtime extent conflicting with a fixed axis
    EXPECT_THROW(static_cast<void>(partial_procedural(3, 3, ones)), std::invalid_argument);
    // fixed procedural vector construction rejects an inconsistent runtime length
    EXPECT_THROW(static_cast<void>(fixed_procedural_vector(2, ones)), std::invalid_argument);
    // procedural construction rejects an overflowing coefficient count
    EXPECT_THROW(static_cast<void>(dynamic_procedural(std::numeric_limits<int>::max(), 2, ones)), std::length_error);

    partial_procedural matrix(2, 3, ones);
    // procedural resize rejects a runtime extent conflicting with a fixed axis
    EXPECT_THROW(matrix.resize(3, 3), std::invalid_argument);
    // failed procedural resize preserves the row count
    EXPECT_EQ(matrix.rows(), 2);
    // failed procedural resize preserves the column count
    EXPECT_EQ(matrix.cols(), 3);
    // procedural coefficient access rejects a row equal to its height
    EXPECT_THROW(static_cast<void>(matrix(2, 0)), std::out_of_range);
}

// validate explicit numeric view dimensions and preserve bindings after failed assignment
TEST(LinearAlgebraRuntimeContracts, MatrixViewRejectsInvalidRuntimeShapes) {
    using dynamic_matrix_view = MatrixView<int, Dynamic, Dynamic>;
    using partial_matrix_view = MatrixView<int, Dynamic, 3>;
    using dynamic_vector_view = MatrixView<int, Dynamic, 1>;
    using fixed_vector_view = MatrixView<int, 3, 1>;
    int data[6] {};

    dynamic_matrix_view empty;
    // a default dynamic numeric view has zero rows
    EXPECT_EQ(empty.rows(), 0);
    // a default dynamic numeric view has zero columns
    EXPECT_EQ(empty.cols(), 0);
    // a default dynamic numeric view has a null storage binding
    EXPECT_EQ(empty.data(), nullptr);

    // explicit numeric view construction rejects a negative row extent
    EXPECT_THROW(static_cast<void>(dynamic_matrix_view(data, -1, 3)), std::invalid_argument);
    // explicit numeric view construction rejects a zero row extent
    EXPECT_THROW(static_cast<void>(dynamic_matrix_view(data, 0, 3)), std::invalid_argument);
    // explicit numeric view construction rejects a zero column extent
    EXPECT_THROW(static_cast<void>(dynamic_matrix_view(data, 2, 0)), std::invalid_argument);
    // partially dynamic view construction rejects an extent conflicting with a fixed axis
    EXPECT_THROW(static_cast<void>(partial_matrix_view(data, 2, 2)), std::invalid_argument);
    // explicit vector-view construction rejects a zero length
    EXPECT_THROW(static_cast<void>(dynamic_vector_view(data, 0)), std::invalid_argument);
    // fixed vector-view construction rejects an inconsistent runtime length
    EXPECT_THROW(static_cast<void>(fixed_vector_view(data, 2)), std::invalid_argument);

    int destination_data[4] {1, 2, 3, 4};
    int source_data[6] {6, 5, 4, 3, 2, 1};
    dynamic_matrix_view destination(destination_data, 2, 2);
    dynamic_matrix_view source(source_data, 2, 3);
    int* const destination_binding = destination.data();
    // view assignment rejects a source with incompatible shape
    EXPECT_THROW(destination = source, std::invalid_argument);
    // failed view assignment preserves the destination's storage binding
    EXPECT_EQ(destination.data(), destination_binding);
    // failed view assignment preserves its row count
    EXPECT_EQ(destination.rows(), 2);
    // failed view assignment preserves its column count
    EXPECT_EQ(destination.cols(), 2);
    // failed view assignment preserves the first destination coefficient
    EXPECT_EQ(destination(0, 0), 1);
    // failed view assignment preserves the final destination coefficient
    EXPECT_EQ(destination(1, 1), 4);
}

// exercise block bounds and nonmutation on failed row assignment in both storage orders
TEST(LinearAlgebraRuntimeContracts, MatrixBlockChecksBoundsAndAssignments) {
    check_matrix_block_runtime_contracts<RowMajor>();
    check_matrix_block_runtime_contracts<ColMajor>();
}

// exercise reshape size preservation, local bounds and empty targets in both storage orders
TEST(LinearAlgebraRuntimeContracts, MatrixReshapeChecksShapesSizesAndIndexes) {
    check_matrix_reshape_runtime_contracts<RowMajor>();
    check_matrix_reshape_runtime_contracts<ColMajor>();
}

// exercise coefficientwise shape checks and bounds through plain and transformed adaptors
TEST(LinearAlgebraRuntimeContracts, MatrixCoeffWiseChecksShapesAndIndexes) {
    check_matrix_coeffwise_runtime_contracts<RowMajor>();
    check_matrix_coeffwise_runtime_contracts<ColMajor>();
}

// exercise vectorwise broadcast shape checks, nonmutation and reduction-result bounds
TEST(LinearAlgebraRuntimeContracts, MatrixVectorWiseChecksShapesAndIndexes) {
    check_matrix_vectorwise_runtime_contracts<RowMajor>();
    check_matrix_vectorwise_runtime_contracts<ColMajor>();
}

// exercise empty reduction identities, undefined empty reductions and integral mean semantics
TEST(LinearAlgebraRuntimeContracts, MatrixReductionsHonorEmptyAndIntegralContracts) {
    check_matrix_reduction_runtime_contracts<RowMajor>();
    check_matrix_reduction_runtime_contracts<ColMajor>();
}

// exercise stable Euclidean norms and tiny square roots across extreme scales
TEST(LinearAlgebraRuntimeContracts, MatrixNormsRemainScaleSafe) {
    check_matrix_norm_runtime_contracts<RowMajor>();
    check_matrix_norm_runtime_contracts<ColMajor>();
}

// exercise infinity norms for zeros, empty axes, subnormals and const views
TEST(LinearAlgebraRuntimeContracts, MatrixInfinityNormsUseZeroIdentity) {
    check_matrix_inf_norm_runtime_contracts<RowMajor>();
    check_matrix_inf_norm_runtime_contracts<ColMajor>();
}

// exercise empty Boolean reductions, comparisons, assignment and streaming
TEST(LinearAlgebraRuntimeContracts, EmptyBooleanExpressionsAreTerminalSafe) {
    check_empty_boolean_runtime_contracts<RowMajor>();
    check_empty_boolean_runtime_contracts<ColMajor>();
}

// exercise scalar-division promotion, integer truncation, nesting and aliased assignment
TEST(LinearAlgebraRuntimeContracts, ScalarDivisionUsesCoefficientArithmeticAndSafeNesting) {
    const auto check_storage = []<int StorageOrder>() {
        using matrix_type = Matrix<double, 2, 2, StorageOrder>;
        // integer scalar division cannot borrow a temporary matrix owner
        static_assert(!permits_temporary_integer_division<matrix_type>);
        // integer scalar division cannot borrow a const temporary matrix owner
        static_assert(!permits_const_temporary_integer_division<matrix_type>);
        constexpr matrix_type source({3.0, -5.0, 7.0, -9.0});
        constexpr double first = (source / 2)(0, 0);
        // dividing a double coefficient by an integer preserves the fractional quotient
        EXPECT_DOUBLE_EQ(first, 1.5);
        const matrix_type expected({1.5, -2.5, 3.5, -4.5});
        const auto divided = source / 2;
        // dividing double coefficients by an integer retains double scalar results
        static_assert(std::is_same_v<typename decltype(divided)::Scalar, double>);
        // scalar division is read-only and preserves the requested storage order
        static_assert(decltype(divided)::ReadOnly == 1 && decltype(divided)::StorageOrder == StorageOrder);
        // integer scalar division produces the explicit floating-point quotient matrix
        EXPECT_EQ(matrix_type(divided), expected);
        // floating scalar division produces the same quotient matrix
        EXPECT_EQ(matrix_type(source / 2.0), expected);
        // a negative divisor reverses every quotient's sign
        EXPECT_EQ(matrix_type(source / -2), matrix_type(expected * -1));
        matrix_type compound(source);
        compound /= 2;
        // compound scalar division produces the same coefficientwise quotients
        EXPECT_EQ(compound, expected);
        matrix_type alias(source);
        alias = alias.transpose() / 2;
        // aliased division snapshots the transposed source before assignment
        EXPECT_EQ(alias, matrix_type({1.5, 3.5, -2.5, -4.5}));
        const auto nested = [&] {
            const auto block = source.template block<2, 2>(0, 0);
            const int divisor = 2;
            return (block + block) / divisor;
        }();
        // a stored division chain evaluates back to the original matrix
        EXPECT_EQ(matrix_type(nested), source);
        matrix_type view_owner(source);
        const auto view_divided = MatrixView<double, 2, 2, StorageOrder>(view_owner.data()) / 2;
        // division of a temporary view safely retains the external storage binding
        EXPECT_EQ(matrix_type(view_divided), expected);

        const Matrix<int, 2, 2, StorageOrder> integers({3, -5, 7, -9});
        const auto integer_divided = integers / 2;
        // integer coefficients divided by an integer retain the integer scalar type
        static_assert(std::is_same_v<typename decltype(integer_divided)::Scalar, int>);
        // integer division truncates positive and negative quotients toward zero
        EXPECT_EQ(
          (Matrix<int, 2, 2, StorageOrder>(integer_divided)), (Matrix<int, 2, 2, StorageOrder>({1, -2, 3, -4})));
        const auto promoted = integers / 2.0;
        // integer coefficients divided by a floating scalar promote to double
        static_assert(std::is_same_v<typename decltype(promoted)::Scalar, double>);
        // promoted scalar division preserves the fractional quotients
        EXPECT_EQ(matrix_type(promoted), expected);
        const Matrix<float, Dynamic, 1, StorageOrder> vector(std::vector<float> {3.0f, -5.0f});
        const auto vector_divided = vector / 2;
        // float vector division by an integer retains float coefficients
        static_assert(std::is_same_v<typename decltype(vector_divided)::Scalar, float>);
        // vector scalar division preserves the two-row length
        EXPECT_EQ(vector_divided.rows(), 2);
        // vector scalar division preserves the single-column orientation
        EXPECT_EQ(vector_divided.cols(), 1);
        // the first float quotient preserves its positive fractional part
        EXPECT_FLOAT_EQ(vector_divided[0], 1.5f);
        // the second float quotient preserves its negative fractional part
        EXPECT_FLOAT_EQ(vector_divided[1], -2.5f);
        const Matrix<short, 1, 1> narrow(static_cast<short>(7));
        const auto narrow_divided = narrow / static_cast<short>(2);
        // same-type short division preserves the declared short scalar result
        static_assert(std::is_same_v<typename decltype(narrow_divided)::Scalar, short>);
        // short coefficient division produces the exact integral quotient
        EXPECT_EQ(narrow_divided(0, 0), 3);
    };
    check_storage.template operator()<RowMajor>();
    check_storage.template operator()<ColMajor>();
}

// exercise scalar division without losing symmetric, skew, diagonal or triangular structure
TEST(LinearAlgebraRuntimeContracts, ScalarDivisionPreservesStructuredMatrixCategories) {
    const double symmetric_values[] {3.0, -5.0, 7.0};
    const SymmetricMatrix<double, 2, 2> symmetric(symmetric_values);
    const auto symmetric_divided = symmetric / 2;
    // scalar division preserves the symmetric expression tag
    static_assert(is_symmetric_matrix_v<decltype(symmetric_divided)>);
    // division cannot borrow a temporary symmetric owner
    static_assert(!permits_temporary_integer_division<SymmetricMatrix<double, 2, 2>>);
    // dividing a symmetric matrix scales both reflected entries consistently
    EXPECT_EQ((Matrix<double, 2, 2>(symmetric_divided)), (Matrix<double, 2, 2>({1.5, -2.5, -2.5, 3.5})));
    const auto symmetric_nested = [&] { return (symmetric + symmetric) / 2; }();
    // a stored symmetric division chain reconstructs the original matrix
    EXPECT_EQ((Matrix<double, 2, 2>(symmetric_nested)), (Matrix<double, 2, 2>(symmetric)));

    const double skew_values[] {3.0, -5.0, 7.0};
    const SkewSymmetricMatrix<double, 3> skew(skew_values);
    const auto skew_divided = skew / 2;
    // scalar division preserves the skew-symmetric expression tag
    static_assert(is_skew_symmetric_matrix_v<decltype(skew_divided)>);
    // division cannot borrow a temporary skew-symmetric owner
    static_assert(!permits_temporary_integer_division<SkewSymmetricMatrix<double, 3>>);
    // dividing a skew matrix preserves its zero diagonal and signed reflections
    EXPECT_EQ(
      (Matrix<double, 3, 3>(skew_divided)), (Matrix<double, 3, 3>({0.0, 1.5, -2.5, -1.5, 0.0, 3.5, 2.5, -3.5, 0.0})));
    const auto skew_nested = [&] { return (skew + skew) / 2; }();
    // a stored skew division chain reconstructs the original matrix
    EXPECT_EQ((Matrix<double, 3, 3>(skew_nested)), (Matrix<double, 3, 3>(skew)));

    const Vector<double, 2> diagonal_values({3.0, 5.0});
    const auto diagonal_divided = diagonal_values.as_diagonal() / 2;
    // scalar division preserves the diagonal expression tag
    static_assert(is_diagonal_matrix_v<decltype(diagonal_divided)>);
    // dividing a diagonal matrix scales only its stored diagonal entries
    EXPECT_EQ((Matrix<double, 2, 2>(diagonal_divided)), (Matrix<double, 2, 2>({1.5, 0.0, 0.0, 2.5})));
    const LowerTriangularMatrix<double, 2, 2> triangular(symmetric_values);
    const auto triangular_divided = triangular / 2;
    // scalar division preserves the triangular expression tag
    static_assert(is_triangular_matrix_v<decltype(triangular_divided)>);
    // dividing a triangular matrix scales its stored entries and preserves implicit zeros
    EXPECT_EQ((Matrix<double, 2, 2>(triangular_divided)), (Matrix<double, 2, 2>({1.5, 0.0, -2.5, 3.5})));
}

}   // namespace fdapde
