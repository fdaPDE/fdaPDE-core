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
#include <type_traits>
#include <vector>

namespace fdapde {
namespace {

template <int StorageOrder> void check_matrix_block_runtime_contracts() {
    using matrix_type = Matrix<int, 3, 4, StorageOrder>;
    matrix_type matrix({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12});

    EXPECT_THROW(static_cast<void>(matrix.row(-1)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(matrix.row(3)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(matrix.col(-1)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(matrix.col(4)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(matrix.block(-1, 0, 1, 1)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(matrix.block(0, -1, 1, 1)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(matrix.block(2, 3, 2, 2)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(matrix.block(0, 0, 0, 2)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(matrix.block(0, 0, -1, 2)), std::invalid_argument);
    EXPECT_THROW(
      static_cast<void>(matrix.template block<2, 2>(2, 3)),
      std::out_of_range);
    EXPECT_THROW(static_cast<void>(matrix.top_rows(0)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(matrix.bottom_rows(0)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(matrix.left_cols(0)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(matrix.right_cols(0)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(matrix.top_rows(4)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(matrix.bottom_rows(4)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(matrix.left_cols(5)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(matrix.right_cols(5)), std::out_of_range);
    EXPECT_THROW(
      static_cast<void>(matrix.bottom_rows(std::numeric_limits<int>::min())),
      std::invalid_argument);
    EXPECT_THROW(
      static_cast<void>(matrix.right_cols(std::numeric_limits<int>::min())),
      std::invalid_argument);

    const auto zero = [](int, int) { return 0.0; };
    ProceduralMatrix<decltype(zero), Dynamic, 1> long_column(50000, zero);
    ProceduralMatrix<decltype(zero), 1, Dynamic> long_row(50000, zero);
    auto oversized_outer_product = long_column * long_row;
    EXPECT_THROW(
      static_cast<void>(oversized_outer_product.block(0, 0, 50000, 50000)),
      std::length_error);

    auto block = matrix.template block<2, 2>(1, 1);
    const auto& const_block = block;
    EXPECT_THROW(static_cast<void>(block(-1, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(block(0, 2)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_block(-1, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_block(0, 2)), std::out_of_range);

    auto row = matrix.row(0);
    const auto& const_row = row;
    EXPECT_THROW(static_cast<void>(row[-1]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(row[row.size()]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_row[-1]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_row[const_row.size()]), std::out_of_range);
    const matrix_type original = matrix;
    const std::initializer_list<int> short_row {20, 21};
    EXPECT_THROW(row = short_row, std::invalid_argument);
    EXPECT_EQ(matrix, original);
}

template <int StorageOrder> void check_matrix_reshape_runtime_contracts() {
    using matrix_type = Matrix<int, Dynamic, Dynamic, StorageOrder>;
    matrix_type matrix(2, 2);

    EXPECT_THROW(static_cast<void>(matrix.reshape(-1, 4)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(matrix.reshape(1, -1)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(matrix.reshape(3, 2)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(matrix.reshape(3)), std::invalid_argument);
    EXPECT_THROW(
      static_cast<void>(matrix.reshape(std::numeric_limits<int>::max(), 2)),
      std::length_error);
    EXPECT_THROW(
      static_cast<void>(ReshapeOp<2, Dynamic, matrix_type>(matrix, 3, 2)),
      std::invalid_argument);
    EXPECT_THROW(
      static_cast<void>(ReshapeOp<Dynamic, 2, matrix_type>(matrix, 2, 3)),
      std::invalid_argument);

    auto reshaped = matrix.template reshape<1, 4>();
    const auto& const_reshaped = reshaped;
    EXPECT_THROW(static_cast<void>(reshaped(-1, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(reshaped(0, 4)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_reshaped(-1, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_reshaped(0, 4)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(reshaped[-1]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(reshaped[4]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_reshaped[-1]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_reshaped[4]), std::out_of_range);

    auto column = matrix.reshape(4);
    const auto& const_column = column;
    EXPECT_THROW(static_cast<void>(column[-1]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(column[4]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_column[-1]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_column[4]), std::out_of_range);

    matrix_type empty;
    const auto empty_matrix = empty.reshape(0, 5);
    const auto empty_column = empty.reshape(0);
    EXPECT_EQ(empty_matrix.rows(), 0);
    EXPECT_EQ(empty_matrix.cols(), 5);
    EXPECT_EQ(empty_matrix.size(), 0);
    EXPECT_EQ(empty_column.rows(), 0);
    EXPECT_EQ(empty_column.cols(), 1);
    EXPECT_EQ(empty_column.size(), 0);
    EXPECT_THROW(static_cast<void>(matrix.reshape(0, 5)), std::invalid_argument);

    const auto zero = [](int, int) { return 0.0; };
    ProceduralMatrix<decltype(zero), Dynamic, 1> long_column(50000, zero);
    ProceduralMatrix<decltype(zero), 1, Dynamic> long_row(50000, zero);
    auto oversized_outer_product = long_column * long_row;
    EXPECT_THROW(static_cast<void>(oversized_outer_product.reshape(1, 1)), std::length_error);

    Matrix<bool, Dynamic, Dynamic, StorageOrder> boolean_matrix(2, 2);
    EXPECT_THROW(static_cast<void>(boolean_matrix.reshape(3, 2)), std::invalid_argument);
}

template <int StorageOrder> void check_matrix_coeffwise_runtime_contracts() {
    using matrix_type = Matrix<double, Dynamic, Dynamic, StorageOrder>;
    matrix_type lhs(2, 2);
    matrix_type rhs(1, 4);
    lhs.cwise() = 1.0;
    rhs.cwise() = 2.0;
    const matrix_type original = lhs;
    EXPECT_THROW(static_cast<void>(lhs.cwise() + rhs.cwise()), std::invalid_argument);
    EXPECT_THROW(lhs.cwise() += rhs.cwise(), std::invalid_argument);
    EXPECT_EQ(lhs, original);

    Matrix<double, Dynamic, 2, StorageOrder> partial_lhs(2, 2);
    Matrix<double, Dynamic, 2, StorageOrder> partial_rhs(3, 2);
    EXPECT_THROW(static_cast<void>(partial_lhs.cwise() + partial_rhs.cwise()), std::invalid_argument);

    auto cwise = lhs.cwise();
    const auto& const_cwise = cwise;
    EXPECT_THROW(static_cast<void>(cwise(-1, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(cwise(0, 2)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_cwise(-1, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_cwise(0, 2)), std::out_of_range);

    auto mwise = cwise.mwise();
    const auto& const_mwise = mwise;
    EXPECT_THROW(static_cast<void>(mwise(-1, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(mwise(0, 2)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_mwise(-1, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_mwise(0, 2)), std::out_of_range);

    auto transformed = lhs.cwise().sqrt();
    EXPECT_THROW(static_cast<void>(transformed(-1, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(transformed(0, 2)), std::out_of_range);

    auto binary = lhs.cwise() + lhs.cwise();
    EXPECT_THROW(static_cast<void>(binary(-1, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(binary(0, 2)), std::out_of_range);

    Matrix<double, 1, 3, StorageOrder> row({1.0, 2.0, 3.0});
    auto row_cwise = row.cwise();
    auto row_mwise = row_cwise.mwise();
    auto row_binary = row.cwise() + row.cwise();
    EXPECT_DOUBLE_EQ(row_cwise[2], 3.0);
    EXPECT_DOUBLE_EQ(row_mwise[2], 3.0);
    EXPECT_DOUBLE_EQ(row_binary[2], 6.0);
    EXPECT_THROW(static_cast<void>(row_cwise[-1]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(row_cwise[3]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(row_mwise[-1]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(row_mwise[3]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(row_binary[-1]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(row_binary[3]), std::out_of_range);
}

template <int StorageOrder> void check_matrix_vectorwise_runtime_contracts() {
    using dynamic_matrix = Matrix<double, Dynamic, Dynamic, StorageOrder>;
    dynamic_matrix matrix = Matrix<double, 2, 3, StorageOrder>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0});
    const dynamic_matrix original = matrix;
    const Matrix<double, Dynamic, 1, StorageOrder> wrong_rows(std::vector<double> {1.0, 2.0, 3.0});
    const Matrix<double, 1, Dynamic, StorageOrder> wrong_cols(std::vector<double> {1.0, 2.0, 3.0, 4.0});

    EXPECT_THROW(static_cast<void>(matrix.rowwise() = wrong_rows), std::invalid_argument);
    EXPECT_EQ(matrix, original);
    EXPECT_THROW(static_cast<void>(matrix.rowwise() += wrong_rows), std::invalid_argument);
    EXPECT_EQ(matrix, original);
    EXPECT_THROW(static_cast<void>(matrix.rowwise() -= wrong_rows), std::invalid_argument);
    EXPECT_EQ(matrix, original);
    EXPECT_THROW(static_cast<void>(matrix.colwise() = wrong_cols), std::invalid_argument);
    EXPECT_EQ(matrix, original);
    EXPECT_THROW(static_cast<void>(matrix.colwise() += wrong_cols), std::invalid_argument);
    EXPECT_EQ(matrix, original);
    EXPECT_THROW(static_cast<void>(matrix.colwise() -= wrong_cols), std::invalid_argument);
    EXPECT_EQ(matrix, original);
    EXPECT_THROW(static_cast<void>(matrix.rowwise() == wrong_rows), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(matrix.colwise() == wrong_cols), std::invalid_argument);

    const dynamic_matrix valid_rows =
      Matrix<double, 2, 1, StorageOrder>(std::vector<double> {10.0, 20.0});
    matrix.rowwise() = valid_rows;
    EXPECT_EQ(matrix, (Matrix<double, 2, 3, StorageOrder>({10.0, 10.0, 10.0, 20.0, 20.0, 20.0})));
    EXPECT_TRUE(matrix.rowwise() == valid_rows);
    matrix = original;
    matrix.rowwise() += valid_rows;
    EXPECT_EQ(matrix, (Matrix<double, 2, 3, StorageOrder>({11.0, 12.0, 13.0, 24.0, 25.0, 26.0})));
    matrix = original;
    matrix.rowwise() -= valid_rows;
    EXPECT_EQ(matrix, (Matrix<double, 2, 3, StorageOrder>({-9.0, -8.0, -7.0, -16.0, -15.0, -14.0})));

    const dynamic_matrix valid_cols =
      Matrix<double, 1, 3, StorageOrder>(std::vector<double> {10.0, 20.0, 30.0});
    matrix.colwise() = valid_cols;
    EXPECT_EQ(matrix, (Matrix<double, 2, 3, StorageOrder>({10.0, 20.0, 30.0, 10.0, 20.0, 30.0})));
    EXPECT_TRUE(matrix.colwise() == valid_cols);
    matrix = original;
    matrix.colwise() += valid_cols;
    EXPECT_EQ(matrix, (Matrix<double, 2, 3, StorageOrder>({11.0, 22.0, 33.0, 14.0, 25.0, 36.0})));
    matrix = original;
    matrix.colwise() -= valid_cols;
    EXPECT_EQ(matrix, (Matrix<double, 2, 3, StorageOrder>({-9.0, -18.0, -27.0, -6.0, -15.0, -24.0})));

    Matrix<double, Dynamic, 3, StorageOrder> partial_rows(2, 3);
    const Matrix<double, Dynamic, 1, StorageOrder> valid_partial_rows(std::vector<double> {2.0, 3.0});
    partial_rows.rowwise() = valid_partial_rows;
    EXPECT_EQ(
      partial_rows,
      (Matrix<double, 2, 3, StorageOrder>({2.0, 2.0, 2.0, 3.0, 3.0, 3.0})));
    const auto partial_rows_snapshot = partial_rows;
    EXPECT_THROW(static_cast<void>(partial_rows.rowwise() = wrong_rows), std::invalid_argument);
    EXPECT_EQ(partial_rows, partial_rows_snapshot);

    Matrix<double, 2, Dynamic, StorageOrder> partial_cols(2, 3);
    const Matrix<double, 1, Dynamic, StorageOrder> valid_partial_cols(std::vector<double> {2.0, 3.0, 4.0});
    partial_cols.colwise() = valid_partial_cols;
    EXPECT_EQ(
      partial_cols,
      (Matrix<double, 2, 3, StorageOrder>({2.0, 3.0, 4.0, 2.0, 3.0, 4.0})));
    const auto partial_cols_snapshot = partial_cols;
    EXPECT_THROW(static_cast<void>(partial_cols.colwise() = wrong_cols), std::invalid_argument);
    EXPECT_EQ(partial_cols, partial_cols_snapshot);

    auto row_sums = original.rowwise().sum();
    EXPECT_DOUBLE_EQ(row_sums(1, 0), 15.0);
    EXPECT_DOUBLE_EQ(row_sums[1], 15.0);
    EXPECT_THROW(static_cast<void>(row_sums(-1, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(row_sums(2, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(row_sums(0, -1)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(row_sums(0, 1)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(row_sums[-1]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(row_sums[2]), std::out_of_range);

    auto col_sums = original.colwise().sum();
    EXPECT_DOUBLE_EQ(col_sums(0, 2), 9.0);
    EXPECT_DOUBLE_EQ(col_sums[2], 9.0);
    EXPECT_THROW(static_cast<void>(col_sums(-1, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(col_sums(1, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(col_sums(0, -1)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(col_sums(0, 3)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(col_sums[-1]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(col_sums[3]), std::out_of_range);
}

template <int StorageOrder> void check_matrix_reduction_runtime_contracts() {
    const auto expect_empty_shape = [](const auto& result, int rows, int cols) {
        EXPECT_EQ(result.rows(), rows);
        EXPECT_EQ(result.cols(), cols);
        EXPECT_EQ(result.size(), 0);
    };

    const Matrix<double, 2, 3, StorageOrder> negative(
      {-4.0, -2.0, -3.0, -9.0, -8.0, -7.0});
    EXPECT_DOUBLE_EQ(negative.max(), -2.0);
    EXPECT_EQ(
      negative.rowwise().max(),
      (Matrix<double, 2, 1, StorageOrder>({-2.0, -7.0})));
    EXPECT_EQ(
      negative.colwise().max(),
      (Matrix<double, 1, 3, StorageOrder>({-4.0, -2.0, -3.0})));

    const Matrix<int, Dynamic, Dynamic, StorageOrder> dynamic_negative =
      Matrix<int, 2, 3, StorageOrder>({-6, -5, -4, -3, -2, -1});
    EXPECT_EQ(dynamic_negative.max(), -1);

    const Matrix<double, Dynamic, Dynamic, StorageOrder> empty;
    EXPECT_DOUBLE_EQ(empty.sum(), 0.0);
    EXPECT_DOUBLE_EQ(empty.prod(), 1.0);
    EXPECT_THROW(static_cast<void>(empty.mean()), std::domain_error);
    EXPECT_THROW(static_cast<void>(empty.max()), std::domain_error);
    EXPECT_THROW(static_cast<void>(empty.min()), std::domain_error);

    const Matrix<int, Dynamic, Dynamic, StorageOrder> empty_int;
    EXPECT_THROW(static_cast<void>(empty_int.mean()), std::domain_error);
    EXPECT_THROW(static_cast<void>(empty_int.max()), std::domain_error);
    EXPECT_THROW(static_cast<void>(empty_int.min()), std::domain_error);

    const Matrix<double, 2, Dynamic, StorageOrder> empty_row_axes(2, 0);
    EXPECT_THROW(static_cast<void>(empty_row_axes.mean()), std::domain_error);
    EXPECT_THROW(static_cast<void>(empty_row_axes.max()), std::domain_error);
    EXPECT_THROW(static_cast<void>(empty_row_axes.min()), std::domain_error);
    EXPECT_EQ(
      empty_row_axes.rowwise().sum(),
      (Matrix<double, 2, 1, StorageOrder>({0.0, 0.0})));
    EXPECT_EQ(
      empty_row_axes.rowwise().prod(),
      (Matrix<double, 2, 1, StorageOrder>({1.0, 1.0})));
    EXPECT_THROW(static_cast<void>(empty_row_axes.rowwise().mean()(0, 0)), std::domain_error);
    EXPECT_THROW(static_cast<void>(empty_row_axes.rowwise().max()(0, 0)), std::domain_error);
    EXPECT_THROW(static_cast<void>(empty_row_axes.rowwise().min()(0, 0)), std::domain_error);
    expect_empty_shape(empty_row_axes.colwise().mean(), 1, 0);
    expect_empty_shape(empty_row_axes.colwise().max(), 1, 0);
    expect_empty_shape(empty_row_axes.colwise().min(), 1, 0);

    const Matrix<double, Dynamic, 3, StorageOrder> empty_col_axes(0, 3);
    EXPECT_THROW(static_cast<void>(empty_col_axes.mean()), std::domain_error);
    EXPECT_THROW(static_cast<void>(empty_col_axes.max()), std::domain_error);
    EXPECT_THROW(static_cast<void>(empty_col_axes.min()), std::domain_error);
    EXPECT_EQ(
      empty_col_axes.colwise().sum(),
      (Matrix<double, 1, 3, StorageOrder>({0.0, 0.0, 0.0})));
    EXPECT_EQ(
      empty_col_axes.colwise().prod(),
      (Matrix<double, 1, 3, StorageOrder>({1.0, 1.0, 1.0})));
    EXPECT_THROW(static_cast<void>(empty_col_axes.colwise().mean()(0, 0)), std::domain_error);
    EXPECT_THROW(static_cast<void>(empty_col_axes.colwise().max()(0, 0)), std::domain_error);
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
    static_assert(std::is_same_v<typename RowMean::Scalar, int>);
    static_assert(std::is_same_v<typename ColMean::Scalar, int>);
    static_assert(std::is_same_v<decltype(row_means(0, 0)), int>);
    static_assert(std::is_same_v<decltype(col_means(0, 0)), int>);
    static_assert(RowMean::Rows == 2 && RowMean::Cols == 1);
    static_assert(ColMean::Rows == 1 && ColMean::Cols == 2);
    EXPECT_EQ(row_means, (Matrix<int, 2, 1, StorageOrder>({-1, -5})));
    EXPECT_EQ(col_means, (Matrix<int, 1, 2, StorageOrder>({-2, -4})));
}

}   // namespace

TEST(LinearAlgebraRuntimeContracts, DynamicSquareOperationsRemainAvailable) {
    Matrix<double, Dynamic, Dynamic> matrix = Matrix<double, 2, 2>({2, 1, 1, 3});
    const Matrix<double, 2, 2> zero = Matrix<double, 2, 2>::Zero();
    const Vector<double, 2> expected_diagonal({2, 3});
    const Matrix<double, Dynamic, Dynamic> inverse = matrix.inverse();

    EXPECT_EQ(matrix.symm_part(), matrix);
    EXPECT_EQ(matrix.skew_part(), zero);
    EXPECT_TRUE(almost_equal(inverse * matrix, Matrix<double, 2, 2>({1, 0, 0, 1})));
    EXPECT_EQ(matrix.determinant(), 5);
    EXPECT_EQ(matrix.diagonal(), expected_diagonal);
}

TEST(LinearAlgebraRuntimeContracts, SquareOperationsRejectRectangularMatrices) {
    Matrix<double, Dynamic, Dynamic> matrix(2, 3);

    EXPECT_THROW((void)matrix.symm_part(), std::invalid_argument);
    EXPECT_THROW((void)matrix.skew_part(), std::invalid_argument);
    EXPECT_THROW((void)matrix.inverse(), std::invalid_argument);
    EXPECT_THROW((void)matrix.determinant(), std::invalid_argument);
}

TEST(LinearAlgebraRuntimeContracts, DynamicBinaryExpressionsRejectIncompatibleShapes) {
    Matrix<double, Dynamic, Dynamic> lhs(2, 3);
    Matrix<double, Dynamic, Dynamic> wrong_cols_rhs(2, 2);
    Matrix<double, Dynamic, Dynamic> wrong_rows_rhs(3, 3);
    Matrix<double, Dynamic, Dynamic> product_rhs(2, 4);
    Vector<double, Dynamic> short_vector(2);
    Vector<double, Dynamic> vector(3);
    Matrix<double, Dynamic, Dynamic> non_column_vector(3, 2);

    EXPECT_THROW((void)(lhs + wrong_cols_rhs), std::invalid_argument);
    EXPECT_THROW((void)(lhs - wrong_rows_rhs), std::invalid_argument);
    EXPECT_THROW((void)(lhs * product_rhs), std::invalid_argument);
    EXPECT_THROW((void)short_vector.cross(vector), std::invalid_argument);
    EXPECT_THROW((void)non_column_vector.cross(vector), std::invalid_argument);

    Vector<double, Dynamic> x(3);
    Vector<double, Dynamic> y(3);
    x[0] = 1;
    y[1] = 1;
    const Vector<double, 3> expected_cross({0, 0, 1});
    EXPECT_EQ(x.cross(y), expected_cross);
}

TEST(LinearAlgebraRuntimeContracts, DenseDiagonalViewChecksShapeAndIndexes) {
    Matrix<double, Dynamic, Dynamic> rectangular(2, 3);
    EXPECT_THROW((void)rectangular.diagonal(), std::invalid_argument);

    Matrix<double, 2, 2> matrix({1, 2, 3, 4});
    auto diagonal = matrix.diagonal();
    const auto& const_diagonal = diagonal;

    EXPECT_THROW((void)diagonal(-1, 0), std::out_of_range);
    EXPECT_THROW((void)diagonal(2, 0), std::out_of_range);
    EXPECT_THROW((void)diagonal(0, -1), std::out_of_range);
    EXPECT_THROW((void)diagonal(0, 1), std::out_of_range);
    EXPECT_THROW((void)diagonal[-1], std::out_of_range);
    EXPECT_THROW((void)diagonal[2], std::out_of_range);
    EXPECT_THROW((void)const_diagonal(-1, 0), std::out_of_range);
    EXPECT_THROW((void)const_diagonal(2, 0), std::out_of_range);
    EXPECT_THROW((void)const_diagonal(0, -1), std::out_of_range);
    EXPECT_THROW((void)const_diagonal(0, 1), std::out_of_range);
    EXPECT_THROW((void)const_diagonal[-1], std::out_of_range);
    EXPECT_THROW((void)const_diagonal[2], std::out_of_range);
}

TEST(LinearAlgebraRuntimeContracts, MatrixOwnerRejectsInvalidShapesAndSizes) {
    using dynamic_matrix = Matrix<int, Dynamic, Dynamic>;
    using partial_rows_matrix = Matrix<int, Dynamic, 3>;
    using partial_cols_matrix = Matrix<int, 2, Dynamic>;

    EXPECT_THROW(static_cast<void>(dynamic_matrix(-1, 2)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(partial_rows_matrix(2, 4)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(partial_cols_matrix(3, 2)), std::invalid_argument);
    EXPECT_THROW(
      static_cast<void>(dynamic_matrix(std::numeric_limits<int>::max(), 2)), std::length_error);

    partial_rows_matrix partial(2, 3);
    partial(1, 2) = 17;
    EXPECT_THROW(partial.resize(2, 4), std::invalid_argument);
    EXPECT_EQ(partial.rows(), 2);
    EXPECT_EQ(partial.cols(), 3);
    EXPECT_EQ(partial.size(), 6);
    EXPECT_EQ(partial(1, 2), 17);

    dynamic_matrix bounded(1, 2);
    bounded(0, 1) = 23;
    EXPECT_THROW(bounded.resize(std::numeric_limits<int>::max(), 2), std::length_error);
    EXPECT_EQ(bounded.rows(), 1);
    EXPECT_EQ(bounded.cols(), 2);
    EXPECT_EQ(bounded(0, 1), 23);
}

TEST(LinearAlgebraRuntimeContracts, MatrixOwnerChecksInputsAssignmentsAndIndexes) {
    using dynamic_matrix = Matrix<int, Dynamic, Dynamic>;
    using dynamic_vector = Vector<int, Dynamic>;
    using fixed_matrix = Matrix<int, 2, 3>;
    using fixed_vector = Vector<int, 3>;

    const std::vector<int> short_input {1, 2, 3, 4, 5};
    EXPECT_THROW(static_cast<void>(fixed_matrix(short_input)), std::invalid_argument);

    const dynamic_matrix wrong_matrix_shape(2, 4);
    EXPECT_THROW(static_cast<void>(fixed_matrix(wrong_matrix_shape)), std::invalid_argument);
    const dynamic_matrix non_vector_shape(2, 2);
    EXPECT_THROW(static_cast<void>(fixed_vector(non_vector_shape)), std::invalid_argument);

    fixed_vector vector({1, 2, 3});
    const std::initializer_list<int> short_vector {4, 5};
    EXPECT_THROW(vector = short_vector, std::invalid_argument);
    EXPECT_EQ(vector, fixed_vector({1, 2, 3}));

    fixed_matrix matrix({1, 2, 3, 4, 5, 6});
    const fixed_matrix& const_matrix = matrix;
    EXPECT_THROW(static_cast<void>(matrix(-1, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_matrix(0, 3)), std::out_of_range);
    const fixed_vector& const_vector = vector;
    EXPECT_THROW(static_cast<void>(vector[-1]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_vector[3]), std::out_of_range);

    EXPECT_THROW(static_cast<void>(dynamic_vector::LinSpaced(1, 0, 1)), std::invalid_argument);
}

TEST(LinearAlgebraRuntimeContracts, ProceduralMatrixChecksShapeSizeAndIndexes) {
    const auto ones = [](int, int) { return 1; };
    using partial_procedural = ProceduralMatrix<decltype(ones), 2, Dynamic>;
    using dynamic_procedural = ProceduralMatrix<decltype(ones), Dynamic, Dynamic>;
    using fixed_procedural_vector = ProceduralMatrix<decltype(ones), 3, 1>;

    EXPECT_THROW(static_cast<void>(partial_procedural(3, 3, ones)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(fixed_procedural_vector(2, ones)), std::invalid_argument);
    EXPECT_THROW(
      static_cast<void>(dynamic_procedural(std::numeric_limits<int>::max(), 2, ones)),
      std::length_error);

    partial_procedural matrix(2, 3, ones);
    EXPECT_THROW(matrix.resize(3, 3), std::invalid_argument);
    EXPECT_EQ(matrix.rows(), 2);
    EXPECT_EQ(matrix.cols(), 3);
    EXPECT_THROW(static_cast<void>(matrix(2, 0)), std::out_of_range);
}

TEST(LinearAlgebraRuntimeContracts, MatrixViewRejectsInvalidRuntimeShapes) {
    using dynamic_matrix_view = MatrixView<int, Dynamic, Dynamic>;
    using partial_matrix_view = MatrixView<int, Dynamic, 3>;
    using dynamic_vector_view = MatrixView<int, Dynamic, 1>;
    using fixed_vector_view = MatrixView<int, 3, 1>;
    int data[6] {};

    dynamic_matrix_view empty;
    EXPECT_EQ(empty.rows(), 0);
    EXPECT_EQ(empty.cols(), 0);
    EXPECT_EQ(empty.data(), nullptr);

    EXPECT_THROW(static_cast<void>(dynamic_matrix_view(data, -1, 3)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(dynamic_matrix_view(data, 0, 3)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(dynamic_matrix_view(data, 2, 0)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(partial_matrix_view(data, 2, 2)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(dynamic_vector_view(data, 0)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(fixed_vector_view(data, 2)), std::invalid_argument);

    int destination_data[4] {1, 2, 3, 4};
    int source_data[6] {6, 5, 4, 3, 2, 1};
    dynamic_matrix_view destination(destination_data, 2, 2);
    dynamic_matrix_view source(source_data, 2, 3);
    int* const destination_binding = destination.data();
    EXPECT_THROW(destination = source, std::invalid_argument);
    EXPECT_EQ(destination.data(), destination_binding);
    EXPECT_EQ(destination.rows(), 2);
    EXPECT_EQ(destination.cols(), 2);
    EXPECT_EQ(destination(0, 0), 1);
    EXPECT_EQ(destination(1, 1), 4);
}

TEST(LinearAlgebraRuntimeContracts, MatrixBlockChecksBoundsAndAssignments) {
    check_matrix_block_runtime_contracts<RowMajor>();
    check_matrix_block_runtime_contracts<ColMajor>();
}

TEST(LinearAlgebraRuntimeContracts, MatrixReshapeChecksShapesSizesAndIndexes) {
    check_matrix_reshape_runtime_contracts<RowMajor>();
    check_matrix_reshape_runtime_contracts<ColMajor>();
}

TEST(LinearAlgebraRuntimeContracts, MatrixCoeffWiseChecksShapesAndIndexes) {
    check_matrix_coeffwise_runtime_contracts<RowMajor>();
    check_matrix_coeffwise_runtime_contracts<ColMajor>();
}

TEST(LinearAlgebraRuntimeContracts, MatrixVectorWiseChecksShapesAndIndexes) {
    check_matrix_vectorwise_runtime_contracts<RowMajor>();
    check_matrix_vectorwise_runtime_contracts<ColMajor>();
}

TEST(LinearAlgebraRuntimeContracts, MatrixReductionsHonorEmptyAndIntegralContracts) {
    check_matrix_reduction_runtime_contracts<RowMajor>();
    check_matrix_reduction_runtime_contracts<ColMajor>();
}

}   // namespace fdapde
