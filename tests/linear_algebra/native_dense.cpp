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

#include <array>
#include <initializer_list>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

namespace fdapde {
namespace {

static_assert(!std::is_default_constructible_v<MatrixView<int, 2, 2>>);
static_assert(std::is_default_constructible_v<MatrixView<int, Dynamic, Dynamic>>);
static_assert(std::is_default_constructible_v<MatrixView<int, Dynamic, 3>>);
static_assert(std::is_same_v<decltype(std::declval<MatrixView<int, 2, 3>&>().data()), int*>);
static_assert(std::is_same_v<decltype(std::declval<const MatrixView<int, 2, 3>&>().data()), const int*>);
static_assert(std::is_same_v<decltype(std::declval<MatrixView<const int, 2, 3>&>().data()), const int*>);

template <typename Matrix>
concept permits_left_temporary_add = requires(Matrix& named) { Matrix {} + named; };

template <typename Matrix>
concept permits_right_temporary_add = requires(Matrix& named) { named + Matrix {}; };

template <typename Matrix>
concept permits_left_temporary_subtract = requires(Matrix& named) { Matrix {} - named; };

template <typename Matrix>
concept permits_right_temporary_subtract = requires(Matrix& named) { named - Matrix {}; };

template <typename Matrix>
concept permits_temporary_scalar_multiply = requires { Matrix {} * 2.0; };

template <typename Matrix>
concept permits_scalar_temporary_multiply = requires { 2.0 * Matrix {}; };

template <typename Matrix>
concept permits_temporary_scalar_divide = requires { Matrix {} / 2.0; };

template <typename Matrix>
concept permits_left_temporary_product = requires(Matrix& named) { Matrix {} * named; };

template <typename Matrix>
concept permits_right_temporary_product = requires(Matrix& named) { named * Matrix {}; };

template <typename Matrix>
concept permits_left_temporary_kron = requires(Matrix& named) { kron(Matrix {}, named); };

template <typename Matrix>
concept permits_right_temporary_kron = requires(Matrix& named) { kron(named, Matrix {}); };

template <typename Vector>
concept permits_left_temporary_cross = requires(Vector& named) { Vector {}.cross(named); };

template <typename Vector>
concept permits_right_temporary_cross = requires(Vector& named) { named.cross(Vector {}); };

template <typename Matrix>
concept exposes_temporary_derived = requires { Matrix {}.derived(); };

template <typename Matrix>
concept permits_temporary_transpose = requires { Matrix {}.transpose(); };

template <typename Matrix>
concept permits_temporary_symm_part = requires { Matrix {}.symm_part(); };

template <typename Matrix>
concept permits_temporary_skew_part = requires { Matrix {}.skew_part(); };

template <typename Matrix>
concept permits_temporary_assignment_add = requires(Matrix& named) { (Matrix {} = named) + named; };

template <typename Matrix>
concept permits_temporary_add_assignment = requires(Matrix& named) { Matrix {} += named; };

template <typename Matrix>
concept permits_temporary_subtract_assignment = requires(Matrix& named) { Matrix {} -= named; };

template <typename Matrix>
concept permits_temporary_scalar_multiply_assignment = requires { Matrix {} *= 2.0; };

template <typename Matrix>
concept permits_temporary_scalar_divide_assignment = requires { Matrix {} /= 2.0; };

template <typename Matrix>
concept permits_temporary_product_assignment = requires(Matrix& named) { Matrix {} *= named; };

template <typename Matrix>
concept permits_temporary_initializer_assignment =
  requires(std::initializer_list<typename Matrix::Scalar> values) { Matrix {} = values; };

template <typename Matrix>
concept permits_safe_expression_chaining = requires(Matrix& a, Matrix& b, Matrix& c) {
    (a + b) + c;
    (a + b).transpose();
    (a + b).symm_part();
    (a + b).skew_part();
};

template <typename Vector>
concept permits_safe_cross_chaining =
  requires(Vector& a, Vector& b, Vector& c) { (a + b).cross(c - b); };

template <typename View>
concept permits_temporary_view_add =
  requires(View& named, std::add_pointer_t<typename View::Scalar> data) { View(data) + named; };

template <typename Matrix>
concept permits_temporary_static_block = requires(Matrix&& matrix) {
    std::move(matrix).template block<1, 1>(0, 0);
};

template <typename Matrix>
concept permits_temporary_dynamic_block = requires(Matrix&& matrix) {
    std::move(matrix).block(0, 0, 1, 1);
};

template <typename Matrix>
concept permits_temporary_row = requires(Matrix&& matrix) { std::move(matrix).row(0); };

template <typename Matrix>
concept permits_temporary_col = requires(Matrix&& matrix) { std::move(matrix).col(0); };

template <typename Matrix>
concept permits_temporary_static_top_rows = requires(Matrix&& matrix) {
    std::move(matrix).template top_rows<1>();
};

template <typename Matrix>
concept permits_temporary_dynamic_top_rows = requires(Matrix&& matrix) {
    std::move(matrix).top_rows(1);
};

template <typename Matrix>
concept permits_temporary_static_bottom_rows = requires(Matrix&& matrix) {
    std::move(matrix).template bottom_rows<1>();
};

template <typename Matrix>
concept permits_temporary_dynamic_bottom_rows = requires(Matrix&& matrix) {
    std::move(matrix).bottom_rows(1);
};

template <typename Matrix>
concept permits_temporary_static_left_cols = requires(Matrix&& matrix) {
    std::move(matrix).template left_cols<1>();
};

template <typename Matrix>
concept permits_temporary_dynamic_left_cols = requires(Matrix&& matrix) {
    std::move(matrix).left_cols(1);
};

template <typename Matrix>
concept permits_temporary_static_right_cols = requires(Matrix&& matrix) {
    std::move(matrix).template right_cols<1>();
};

template <typename Matrix>
concept permits_temporary_dynamic_right_cols = requires(Matrix&& matrix) {
    std::move(matrix).right_cols(1);
};

template <typename Matrix, int ExpectedReadOnly>
concept permits_all_temporary_block_accessors = requires(Matrix&& matrix) {
    requires (decltype(std::move(matrix).template block<1, 1>(0, 0))::ReadOnly == ExpectedReadOnly);
    requires (decltype(std::move(matrix).block(0, 0, 1, 1))::ReadOnly == ExpectedReadOnly);
    requires (decltype(std::move(matrix).row(0))::ReadOnly == ExpectedReadOnly);
    requires (decltype(std::move(matrix).col(0))::ReadOnly == ExpectedReadOnly);
    requires (decltype(std::move(matrix).template top_rows<1>())::ReadOnly == ExpectedReadOnly);
    requires (decltype(std::move(matrix).top_rows(1))::ReadOnly == ExpectedReadOnly);
    requires (decltype(std::move(matrix).template bottom_rows<1>())::ReadOnly == ExpectedReadOnly);
    requires (decltype(std::move(matrix).bottom_rows(1))::ReadOnly == ExpectedReadOnly);
    requires (decltype(std::move(matrix).template left_cols<1>())::ReadOnly == ExpectedReadOnly);
    requires (decltype(std::move(matrix).left_cols(1))::ReadOnly == ExpectedReadOnly);
    requires (decltype(std::move(matrix).template right_cols<1>())::ReadOnly == ExpectedReadOnly);
    requires (decltype(std::move(matrix).right_cols(1))::ReadOnly == ExpectedReadOnly);
};

template <typename Matrix>
concept permits_safe_temporary_block_access = requires(Matrix& a, Matrix& b) {
    (a + b).template block<1, 1>(0, 0);
    (a + b).row(0);
};

template <typename View>
concept permits_temporary_fixed_view_row = requires(std::add_pointer_t<typename View::Scalar> data) {
    View(data).row(0);
};

template <typename View>
concept permits_temporary_dynamic_view_row = requires(std::add_pointer_t<typename View::Scalar> data) {
    View(data, 2, 2).row(0);
};

template <typename Matrix>
concept permits_direct_temporary_vector_block = requires(Matrix&& matrix) {
    MatrixBlock<1, 1, Matrix> {std::move(matrix), 0};
};

template <typename Matrix>
concept permits_direct_temporary_static_block = requires(Matrix&& matrix) {
    MatrixBlock<1, 1, Matrix> {std::move(matrix), 0, 0};
};

template <typename Matrix>
concept permits_direct_temporary_dynamic_block = requires(Matrix&& matrix) {
    MatrixBlock<Dynamic, Dynamic, Matrix> {std::move(matrix), 0, 0, 1, 1};
};

template <typename Block>
concept permits_const_block_coefficient_write = requires(const Block& block) { block(0, 0) = 1.0; };

template <typename Block>
concept exposes_temporary_block_iterator = requires(Block&& block) { std::move(block).begin(); };

template <typename Block>
concept has_const_bidirectional_block_iterator = requires {
    requires std::bidirectional_iterator<typename Block::const_iterator>;
};

using lifetime_matrix = Matrix<double, 2, 2>;
using lifetime_const_matrix = const lifetime_matrix;
using lifetime_vector = Matrix<double, 3, 1>;
using lifetime_initializer_vector = Matrix<double, 2, 1>;
using lifetime_expression = decltype(
  std::declval<lifetime_matrix&>() + std::declval<lifetime_matrix&>());
using lifetime_view = MatrixView<double, 2, 2>;
using lifetime_const_view = MatrixView<const double, 2, 2>;
using lifetime_block = decltype(std::declval<lifetime_matrix&>().template block<1, 1>(0, 0));
using lifetime_const_block = decltype(std::declval<const lifetime_matrix&>().template block<1, 1>(0, 0));
using lifetime_const_view_block =
  decltype(std::declval<lifetime_const_view&>().template block<1, 1>(0, 0));
using lifetime_row = decltype(std::declval<lifetime_matrix&>().row(0));
static_assert(!permits_left_temporary_add<lifetime_matrix>);
static_assert(!permits_right_temporary_add<lifetime_matrix>);
static_assert(!permits_left_temporary_subtract<lifetime_matrix>);
static_assert(!permits_right_temporary_subtract<lifetime_matrix>);
static_assert(!permits_temporary_scalar_multiply<lifetime_matrix>);
static_assert(!permits_scalar_temporary_multiply<lifetime_matrix>);
static_assert(!permits_temporary_scalar_divide<lifetime_matrix>);
static_assert(!permits_left_temporary_product<lifetime_matrix>);
static_assert(!permits_right_temporary_product<lifetime_matrix>);
static_assert(!permits_left_temporary_kron<lifetime_matrix>);
static_assert(!permits_right_temporary_kron<lifetime_matrix>);
static_assert(!permits_left_temporary_cross<lifetime_vector>);
static_assert(!permits_right_temporary_cross<lifetime_vector>);
static_assert(!exposes_temporary_derived<lifetime_matrix>);
static_assert(!permits_temporary_transpose<lifetime_matrix>);
static_assert(!permits_temporary_symm_part<lifetime_matrix>);
static_assert(!permits_temporary_skew_part<lifetime_matrix>);
static_assert(!permits_temporary_assignment_add<lifetime_matrix>);
static_assert(!permits_temporary_add_assignment<lifetime_matrix>);
static_assert(!permits_temporary_subtract_assignment<lifetime_matrix>);
static_assert(!permits_temporary_scalar_multiply_assignment<lifetime_matrix>);
static_assert(!permits_temporary_scalar_divide_assignment<lifetime_matrix>);
static_assert(!permits_temporary_product_assignment<lifetime_matrix>);
static_assert(!permits_temporary_initializer_assignment<lifetime_initializer_vector>);
static_assert(permits_safe_expression_chaining<lifetime_matrix>);
static_assert(permits_safe_cross_chaining<lifetime_vector>);
static_assert(permits_temporary_view_add<MatrixView<double, 2, 2>>);
static_assert(!permits_temporary_static_block<lifetime_matrix>);
static_assert(!permits_temporary_dynamic_block<lifetime_matrix>);
static_assert(!permits_temporary_row<lifetime_matrix>);
static_assert(!permits_temporary_col<lifetime_matrix>);
static_assert(!permits_temporary_static_top_rows<lifetime_matrix>);
static_assert(!permits_temporary_dynamic_top_rows<lifetime_matrix>);
static_assert(!permits_temporary_static_bottom_rows<lifetime_matrix>);
static_assert(!permits_temporary_dynamic_bottom_rows<lifetime_matrix>);
static_assert(!permits_temporary_static_left_cols<lifetime_matrix>);
static_assert(!permits_temporary_dynamic_left_cols<lifetime_matrix>);
static_assert(!permits_temporary_static_right_cols<lifetime_matrix>);
static_assert(!permits_temporary_dynamic_right_cols<lifetime_matrix>);
static_assert(!permits_temporary_static_block<lifetime_const_matrix>);
static_assert(!permits_temporary_dynamic_block<lifetime_const_matrix>);
static_assert(!permits_temporary_row<lifetime_const_matrix>);
static_assert(!permits_temporary_col<lifetime_const_matrix>);
static_assert(!permits_temporary_static_top_rows<lifetime_const_matrix>);
static_assert(!permits_temporary_dynamic_top_rows<lifetime_const_matrix>);
static_assert(!permits_temporary_static_bottom_rows<lifetime_const_matrix>);
static_assert(!permits_temporary_dynamic_bottom_rows<lifetime_const_matrix>);
static_assert(!permits_temporary_static_left_cols<lifetime_const_matrix>);
static_assert(!permits_temporary_dynamic_left_cols<lifetime_const_matrix>);
static_assert(!permits_temporary_static_right_cols<lifetime_const_matrix>);
static_assert(!permits_temporary_dynamic_right_cols<lifetime_const_matrix>);
static_assert(permits_all_temporary_block_accessors<lifetime_expression, 1>);
static_assert(permits_all_temporary_block_accessors<lifetime_view, 0>);
static_assert(permits_all_temporary_block_accessors<const lifetime_view, 1>);
static_assert(permits_safe_temporary_block_access<lifetime_matrix>);
static_assert(permits_temporary_fixed_view_row<MatrixView<double, 2, 2>>);
static_assert(permits_temporary_dynamic_view_row<MatrixView<double, Dynamic, Dynamic>>);
static_assert(!permits_direct_temporary_vector_block<lifetime_matrix>);
static_assert(!permits_direct_temporary_static_block<lifetime_matrix>);
static_assert(!permits_direct_temporary_dynamic_block<lifetime_matrix>);
static_assert(!permits_direct_temporary_vector_block<lifetime_const_matrix>);
static_assert(!permits_direct_temporary_static_block<lifetime_const_matrix>);
static_assert(!permits_direct_temporary_dynamic_block<lifetime_const_matrix>);
static_assert(lifetime_const_block::ReadOnly == 1);
static_assert(lifetime_const_view_block::ReadOnly == 1);
static_assert(!permits_const_block_coefficient_write<lifetime_block>);
static_assert(!permits_const_block_coefficient_write<lifetime_const_view_block>);
static_assert(std::is_same_v<decltype(std::declval<const lifetime_block&>()(0, 0)), const double&>);
static_assert(std::is_same_v<decltype(std::declval<lifetime_const_view_block&>()(0, 0)), const double&>);
static_assert(std::is_same_v<decltype(std::declval<const lifetime_row&>()[0]), const double&>);
static_assert(!exposes_temporary_block_iterator<lifetime_row>);
static_assert(std::bidirectional_iterator<typename lifetime_row::iterator>);
static_assert(has_const_bidirectional_block_iterator<lifetime_row>);
static_assert(std::is_same_v<std::iter_reference_t<typename lifetime_row::const_iterator>, const double&>);
using temporary_row_initializer_result = decltype(
  std::declval<lifetime_row&&>() = std::declval<const std::initializer_list<double>&>());
static_assert(!std::is_lvalue_reference_v<temporary_row_initializer_result>);

template <int StorageOrder> void check_owner_behavior() {
    using fixed_matrix = Matrix<int, 2, 3, StorageOrder>;
    using dynamic_column = Matrix<int, Dynamic, 1, StorageOrder>;

    constexpr int fixed_input[6] {1, 2, 3, 4, 5, 6};
    constexpr fixed_matrix matrix(fixed_input);
    static_assert(matrix.rows() == 2);
    static_assert(matrix.cols() == 3);
    static_assert(matrix(1, 2) == 6);

    const std::array<int, 6> expected_storage = StorageOrder == RowMajor
      ? std::array<int, 6> {1, 2, 3, 4, 5, 6}
      : std::array<int, 6> {1, 4, 2, 5, 3, 6};
    for (int i = 0; i < matrix.size(); ++i) { EXPECT_EQ(matrix.data()[i], expected_storage[i]); }

    const std::vector<int> input {1, 2, 3, 4, 5, 6};
    const fixed_matrix from_vector(input);
    EXPECT_EQ(from_vector, matrix);

    Matrix<int, Dynamic, 3, StorageOrder> dynamic_rows(2, 3);
    EXPECT_EQ(dynamic_rows.end() - dynamic_rows.begin(), 6);
    dynamic_rows.resize(4, 3);
    EXPECT_EQ(dynamic_rows.rows(), 4);
    EXPECT_EQ(dynamic_rows.cols(), 3);
    EXPECT_EQ(dynamic_rows.end() - dynamic_rows.begin(), 12);

    Matrix<int, 2, Dynamic, StorageOrder> dynamic_cols(2, 3);
    EXPECT_EQ(dynamic_cols.end() - dynamic_cols.begin(), 6);
    dynamic_cols.resize(2, 4);
    EXPECT_EQ(dynamic_cols.rows(), 2);
    EXPECT_EQ(dynamic_cols.cols(), 4);
    EXPECT_EQ(dynamic_cols.end() - dynamic_cols.begin(), 8);

    dynamic_column column(std::vector<int> {1, 2, 3});
    EXPECT_EQ(column.rows(), 3);
    EXPECT_EQ(column.cols(), 1);
    column = {4, 5};
    EXPECT_EQ(column.rows(), 2);
    EXPECT_EQ(column.cols(), 1);
    EXPECT_EQ(column[0], 4);
    EXPECT_EQ(column[1], 5);

    const Matrix<int, 1, 3, StorageOrder> row({7, 8, 9});
    const Matrix<int, 3, 1, StorageOrder> fixed_column_from_row(row);
    EXPECT_EQ(fixed_column_from_row[0], 7);
    EXPECT_EQ(fixed_column_from_row[1], 8);
    EXPECT_EQ(fixed_column_from_row[2], 9);

    const dynamic_column column_from_row(row);
    EXPECT_EQ(column_from_row.rows(), 3);
    EXPECT_EQ(column_from_row.cols(), 1);
    EXPECT_EQ(column_from_row[0], 7);
    EXPECT_EQ(column_from_row[1], 8);
    EXPECT_EQ(column_from_row[2], 9);

    const Matrix<int, 1, 1, StorageOrder> scalar_vector(11);
    const dynamic_column column_from_scalar(scalar_vector);
    EXPECT_EQ(column_from_scalar.rows(), 1);
    EXPECT_EQ(column_from_scalar[0], 11);

    constexpr int OtherStorageOrder = StorageOrder == RowMajor ? ColMajor : RowMajor;
    const Matrix<int, 2, 3, OtherStorageOrder> other_order(matrix);
    EXPECT_EQ(other_order(0, 1), 2);
    EXPECT_EQ(other_order(1, 2), 6);
}

template <int StorageOrder> void check_numeric_view_behavior() {
    using fixed_view = MatrixView<int, 2, 3, StorageOrder>;
    static_assert(std::is_same_v<
                  decltype(std::declval<fixed_view&&>() = std::declval<const fixed_view&>()), fixed_view>);

    std::array<int, 6> view_storage {};
    fixed_view view(view_storage.data());
    view(0, 1) = 7;
    constexpr int view_index = StorageOrder == RowMajor ? 1 : 2;
    EXPECT_EQ(view_storage[view_index], 7);
    const auto& const_view = view;
    static_assert(std::is_same_v<decltype(const_view(0, 0)), const int&>);

    std::array<int, 6> source_storage {};
    fixed_view source_view(source_storage.data());
    const auto source_alias = source_view;
    EXPECT_EQ(source_alias.data(), source_view.data());
    source_view(1, 2) = 11;
    int* const destination = view.data();
    view = source_view;
    EXPECT_EQ(view.data(), destination);
    EXPECT_EQ(view(1, 2), 11);

    const Matrix<int, 2, 3, StorageOrder> matrix({1, 2, 3, 4, 5, 6});
    view = matrix;
    EXPECT_EQ(view.data(), destination);
    EXPECT_EQ(view, matrix);

    std::array<int, 6> temporary_destination {};
    const auto assigned_temporary = fixed_view(temporary_destination.data()) = source_view;
    EXPECT_EQ(assigned_temporary.data(), temporary_destination.data());
    EXPECT_EQ(assigned_temporary(1, 2), 11);

    std::array<int, 3> vector_storage {3, 2, 1};
    MatrixView<int, 1, Dynamic, StorageOrder> row_view(vector_storage.data(), 3);
    MatrixView<int, Dynamic, 1, StorageOrder> column_view(vector_storage.data(), 3);
    EXPECT_EQ(row_view.rows(), 1);
    EXPECT_EQ(row_view.cols(), 3);
    EXPECT_EQ(row_view(0, 2), 1);
    EXPECT_EQ(column_view.rows(), 3);
    EXPECT_EQ(column_view.cols(), 1);
    EXPECT_EQ(column_view(2, 0), 1);
}

template <int StorageOrder> void check_arithmetic_expression_nesting() {
    using matrix_type = Matrix<double, 2, 3, StorageOrder>;
    const matrix_type matrix({-4.0, 0.0, 2.0, 1.0, -3.0, 5.0});

    const matrix_type chained_sum = (matrix + matrix) + matrix;
    EXPECT_EQ(chained_sum, (matrix_type({-12.0, 0.0, 6.0, 3.0, -9.0, 15.0})));

    const Matrix<double, 3, 2, StorageOrder> transposed_sum = (matrix + matrix).transpose();
    EXPECT_EQ(
      transposed_sum,
      (Matrix<double, 3, 2, StorageOrder>({-8.0, 2.0, 0.0, -6.0, 4.0, 10.0})));

    const Matrix<double, 2, 2, StorageOrder> gram = matrix * matrix.transpose();
    EXPECT_EQ(gram, (Matrix<double, 2, 2, StorageOrder>({20.0, 6.0, 6.0, 35.0})));

    using square_matrix_type = Matrix<double, 2, 2, StorageOrder>;
    const square_matrix_type square({1.0, 2.0, 3.0, 4.0});
    const square_matrix_type symmetric = (square + square).symm_part();
    const square_matrix_type skew = (square + square).skew_part();
    EXPECT_EQ(symmetric, (square_matrix_type({2.0, 5.0, 5.0, 8.0})));
    EXPECT_EQ(skew, (square_matrix_type({0.0, -1.0, 1.0, 0.0})));

    using vector_type = Matrix<double, 3, 1, StorageOrder>;
    const vector_type a({1.0, 0.0, 0.0});
    const vector_type b({0.0, 1.0, 0.0});
    const vector_type c({0.0, 0.0, 1.0});
    const vector_type cross = (a + b).cross(c - b);
    EXPECT_EQ(cross, (vector_type({1.0, -1.0, -1.0})));

    using cross_expression = decltype(a.cross(b));
    using kron_expression = decltype(kron(matrix, matrix));
    static_assert(cross_expression::StorageOrder == StorageOrder);
    static_assert(kron_expression::StorageOrder == StorageOrder);
}

template <int StorageOrder> void check_assignment_alias_materialization() {
    using matrix_type = Matrix<double, 2, 2, StorageOrder>;

    matrix_type aliased({1.0, 2.0, 3.0, 4.0});
    aliased += aliased.transpose();
    EXPECT_EQ(aliased, (matrix_type({2.0, 5.0, 5.0, 8.0})));

    aliased = matrix_type({1.0, 2.0, 3.0, 4.0});
    aliased -= aliased.transpose();
    EXPECT_EQ(aliased, (matrix_type({0.0, -1.0, 1.0, 0.0})));

    aliased = matrix_type({1.0, 2.0, 3.0, 4.0});
    aliased *= aliased;
    EXPECT_EQ(aliased, (matrix_type({7.0, 10.0, 15.0, 22.0})));

    matrix_type overlapping({1.0, 2.0, 3.0, 4.0});
    overlapping.template block<2, 2>(0, 0) =
      overlapping.template block<2, 2>(0, 0).transpose();
    EXPECT_EQ(overlapping, (matrix_type({1.0, 3.0, 2.0, 4.0})));

    const matrix_type source_owner({1.0, 2.0, 3.0, 4.0});
    double destination_data[4] {};
    MatrixView<const double, 2, 2, StorageOrder> source(source_owner.data());
    MatrixView<double, 2, 2, StorageOrder> destination(destination_data);
    destination = source;
    EXPECT_EQ(destination, source_owner);
}

template <int StorageOrder> void check_block_view_behavior() {
    using matrix_type = Matrix<double, 3, 4, StorageOrder>;
    matrix_type matrix({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0});

    Matrix<double, 1, 3, StorageOrder> row_vector({1.0, 2.0, 3.0});
    auto scalar_col = row_vector.col(2);
    EXPECT_EQ(scalar_col.rows(), 1);
    EXPECT_EQ(scalar_col.cols(), 1);
    EXPECT_DOUBLE_EQ(scalar_col(0, 0), 3.0);
    scalar_col(0, 0) = 4.0;
    EXPECT_DOUBLE_EQ(row_vector(0, 2), 4.0);

    Matrix<double, 3, 1, StorageOrder> column_vector({1.0, 2.0, 3.0});
    auto scalar_row = column_vector.row(2);
    EXPECT_EQ(scalar_row.rows(), 1);
    EXPECT_EQ(scalar_row.cols(), 1);
    EXPECT_DOUBLE_EQ(scalar_row(0, 0), 3.0);
    scalar_row(0, 0) = 4.0;
    EXPECT_DOUBLE_EQ(column_vector(2, 0), 4.0);

    const Matrix<double, 2, 2, StorageOrder> fixed = matrix.template block<2, 2>(1, 1);
    EXPECT_EQ(fixed, (Matrix<double, 2, 2, StorageOrder>({6.0, 7.0, 10.0, 11.0})));
    const Matrix<double, 3, 2, StorageOrder> dynamic = matrix.block(0, 2, 3, 2);
    EXPECT_EQ(dynamic, (Matrix<double, 3, 2, StorageOrder>({3.0, 4.0, 7.0, 8.0, 11.0, 12.0})));

    auto safe_expression_block = (matrix + matrix).template block<1, 2>(0, 0);
    const Matrix<double, 1, 2, StorageOrder> safe_expression_value = safe_expression_block;
    EXPECT_EQ(safe_expression_value, (Matrix<double, 1, 2, StorageOrder>({2.0, 4.0})));

    auto temporary_const_view_row =
      MatrixView<const double, Dynamic, Dynamic, StorageOrder>(matrix.data(), 3, 4).row(1);
    const Matrix<double, 1, 4, StorageOrder> temporary_view_value = temporary_const_view_row;
    EXPECT_EQ(temporary_view_value, (Matrix<double, 1, 4, StorageOrder>({5.0, 6.0, 7.0, 8.0})));

    double mutable_view_data[4] {1.0, 2.0, 3.0, 4.0};
    MatrixView<double, 2, 2, StorageOrder> mutable_view(mutable_view_data);
    auto temporary_mutable_view_row =
      MatrixView<double, 2, 2, StorageOrder>(mutable_view_data).row(1);
    static_assert(decltype(temporary_mutable_view_row)::ReadOnly == 0);
    temporary_mutable_view_row(0, 0) = 9.0;
    EXPECT_DOUBLE_EQ(mutable_view(1, 0), 9.0);

    auto computed = matrix + matrix;
    auto computed_row = computed.row(0);
    static_assert(std::bidirectional_iterator<typename decltype(computed_row)::iterator>);
    double computed_sum = 0.0;
    for (const double value : computed_row) { computed_sum += value; }
    EXPECT_DOUBLE_EQ(computed_sum, 20.0);

    auto row = matrix.row(1);
    int count = 0;
    double sum = 0.0;
    for (const double value : row) {
        ++count;
        sum += value;
    }
    EXPECT_EQ(count, row.size());
    EXPECT_DOUBLE_EQ(sum, 26.0);

    matrix.row(2) = {30.0, 31.0, 32.0, 33.0};
    EXPECT_EQ(matrix.row(2), (Matrix<double, 1, 4, StorageOrder>({30.0, 31.0, 32.0, 33.0})));

    matrix_type named_assignment({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    auto destination = named_assignment.template block<2, 2>(0, 0);
    auto source = named_assignment.template block<2, 2>(1, 2);
    destination = source;
    EXPECT_EQ(
      named_assignment,
      (matrix_type({7.0, 8.0, 3.0, 4.0, 11.0, 12.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0})));

    matrix_type temporary_assignment({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    temporary_assignment.template block<2, 2>(0, 0) = temporary_assignment.template block<2, 2>(1, 2);
    EXPECT_EQ(
      temporary_assignment,
      (matrix_type({7.0, 8.0, 3.0, 4.0, 11.0, 12.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0})));

    matrix_type view_owner({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0});
    MatrixView<double, 3, 4, StorageOrder> view(view_owner.data());
    double* const binding = view.data();
    auto view_destination = view.template block<2, 2>(0, 0);
    auto view_source = view.template block<2, 2>(1, 2);
    view_destination = view_source;
    EXPECT_EQ(view.data(), binding);
    EXPECT_EQ(
      view_owner,
      (matrix_type({7.0, 8.0, 3.0, 4.0, 11.0, 12.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0})));
}

}   // namespace

TEST(NativeDenseMatrix, OwnerShapeStorageAndVectorCopy) {
    check_owner_behavior<RowMajor>();
    check_owner_behavior<ColMajor>();

    const auto ones = [](int, int) { return 1; };
    ProceduralMatrix<decltype(ones), 2, Dynamic> procedural(2, 3, ones);
    procedural.resize(2, 4);
    EXPECT_EQ(procedural.rows(), 2);
    EXPECT_EQ(procedural.cols(), 4);
    EXPECT_EQ(procedural(1, 3), 1);
}

TEST(NativeDenseMatrix, NumericViewBindingConstnessAndStorage) {
    check_numeric_view_behavior<RowMajor>();
    check_numeric_view_behavior<ColMajor>();
}

TEST(NativeDenseMatrix, ArithmeticExpressionNesting) {
    check_arithmetic_expression_nesting<RowMajor>();
    check_arithmetic_expression_nesting<ColMajor>();
}

TEST(NativeDenseMatrix, AssignmentOperationsMaterializeAliases) {
    check_assignment_alias_materialization<RowMajor>();
    check_assignment_alias_materialization<ColMajor>();
}

TEST(NativeDenseMatrix, BlocksRemainBoundedViews) {
    check_block_view_behavior<RowMajor>();
    check_block_view_behavior<ColMajor>();
}

}   // namespace fdapde
