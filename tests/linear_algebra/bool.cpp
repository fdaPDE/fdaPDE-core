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
#include <functional>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

static_assert(fdapde::internals::bitpack_count(0, 64) == 0);
static_assert(fdapde::internals::bitpack_count(1, 64) == 1);
static_assert(fdapde::internals::bitpack_count(63, 64) == 1);
static_assert(fdapde::internals::bitpack_count(64, 64) == 1);
static_assert(fdapde::internals::bitpack_count(65, 64) == 2);
static_assert(
  fdapde::internals::bitpack_count(std::numeric_limits<int>::max(), 64) == 33'554'432);

template <typename Lhs, typename Rhs, typename Operation>
concept permits_boolean_binary = requires(Lhs&& lhs, Rhs&& rhs, Operation operation) {
    operation(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs));
};

template <typename Matrix>
concept permits_boolean_negation = requires(Matrix&& matrix) { ~std::forward<Matrix>(matrix); };

template <typename Matrix>
concept exposes_boolean_rvalue_derived = requires(Matrix&& matrix) { std::move(matrix).derived(); };

template <typename Matrix>
concept permits_temporary_boolean_assignment = requires(Matrix&& matrix, const Matrix& rhs) {
    std::move(matrix) = rhs;
};

template <typename Matrix>
concept permits_temporary_boolean_and_assignment = requires(Matrix&& matrix, const Matrix& rhs) {
    std::move(matrix) &= rhs;
};

template <typename Matrix>
concept permits_temporary_boolean_or_assignment = requires(Matrix&& matrix, const Matrix& rhs) {
    std::move(matrix) |= rhs;
};

template <typename Matrix>
concept permits_temporary_boolean_xor_assignment = requires(Matrix&& matrix, const Matrix& rhs) {
    std::move(matrix) ^= rhs;
};

template <typename Matrix>
concept permits_named_boolean_chain = requires(Matrix& lhs, Matrix& rhs) { ~((lhs | rhs) ^ (lhs & rhs)); };

using boolean_mask = fdapde::Matrix<bool, 2, 2>;
using boolean_unary_node =
  fdapde::BoolMatrixBitWiseOp<boolean_mask, std::logical_not<>, std::bit_not<>>;
using boolean_binary_node =
  fdapde::BoolMatrixBinOp<boolean_mask, boolean_mask, std::bit_and<>, std::bit_and<>>;

static_assert(permits_boolean_binary<boolean_mask&, boolean_mask&, std::bit_and<>>);
static_assert(!permits_boolean_binary<boolean_mask, boolean_mask&, std::bit_and<>>);
static_assert(!permits_boolean_binary<boolean_mask&, boolean_mask, std::bit_and<>>);
static_assert(!permits_boolean_binary<boolean_mask, boolean_mask&, std::bit_or<>>);
static_assert(!permits_boolean_binary<boolean_mask&, boolean_mask, std::bit_or<>>);
static_assert(!permits_boolean_binary<boolean_mask, boolean_mask&, std::bit_xor<>>);
static_assert(!permits_boolean_binary<boolean_mask&, boolean_mask, std::bit_xor<>>);
static_assert(permits_boolean_negation<boolean_mask&>);
static_assert(!permits_boolean_negation<boolean_mask>);
static_assert(!exposes_boolean_rvalue_derived<boolean_mask>);
static_assert(!permits_temporary_boolean_assignment<boolean_mask>);
static_assert(!permits_temporary_boolean_and_assignment<boolean_mask>);
static_assert(!permits_temporary_boolean_or_assignment<boolean_mask>);
static_assert(!permits_temporary_boolean_xor_assignment<boolean_mask>);
static_assert(permits_named_boolean_chain<boolean_mask>);
static_assert(std::is_constructible_v<
              boolean_unary_node, const boolean_mask&, std::logical_not<>, std::bit_not<>>);
static_assert(!std::is_constructible_v<
              boolean_unary_node, boolean_mask&&, std::logical_not<>, std::bit_not<>>);
static_assert(std::is_constructible_v<
              boolean_binary_node,
              const boolean_mask&,
              const boolean_mask&,
              std::bit_and<>,
              std::bit_and<>>);
static_assert(!std::is_constructible_v<
              boolean_binary_node, boolean_mask&&, const boolean_mask&, std::bit_and<>, std::bit_and<>>);
static_assert(!std::is_constructible_v<
              boolean_binary_node, const boolean_mask&, boolean_mask&&, std::bit_and<>, std::bit_and<>>);
static_assert(fdapde::is_boolean_matrix_v<boolean_mask&>);
static_assert(fdapde::is_boolean_matrix_v<const boolean_mask&>);
static_assert(fdapde::is_boolean_vector_v<const fdapde::Matrix<bool, 1, 2>&>);
static_assert(!fdapde::is_boolean_vector_v<int>);

template <typename Condition, typename TrueXpr, typename FalseXpr>
concept permits_boolean_selection = requires(Condition&& condition, TrueXpr&& true_xpr, FalseXpr&& false_xpr) {
    std::forward<Condition>(condition).select(
      std::forward<TrueXpr>(true_xpr), std::forward<FalseXpr>(false_xpr));
};

using selection_mask = fdapde::Matrix<bool, 2, 2>;
using selection_values = fdapde::Matrix<double, 2, 2>;
using selection_condition_expression = decltype(~std::declval<selection_mask&>());
using selection_value_expression =
  decltype(std::declval<selection_values&>() + std::declval<selection_values&>());
using selection_mask_view = fdapde::MatrixView<const bool, 2, 2>;
using selection_value_view = fdapde::MatrixView<const double, 2, 2>;
using direct_ternary = fdapde::TernaryOp<selection_mask, selection_values, selection_values>;
using safe_selection = decltype(std::declval<selection_condition_expression>().select(
  std::declval<selection_value_expression>(), std::declval<selection_value_expression>()));

static_assert(permits_boolean_selection<selection_mask&, selection_values&, selection_values&>);
static_assert(!permits_boolean_selection<selection_mask, selection_values&, selection_values&>);
static_assert(!permits_boolean_selection<const selection_mask, selection_values&, selection_values&>);
static_assert(!permits_boolean_selection<selection_mask&, selection_values, selection_values&>);
static_assert(!permits_boolean_selection<selection_mask&, selection_values&, selection_values>);
static_assert(permits_boolean_selection<
              selection_condition_expression, selection_value_expression, selection_value_expression>);
static_assert(permits_boolean_selection<selection_mask_view, selection_value_view, selection_values&>);
static_assert(safe_selection::ReadOnly == 1);
static_assert(safe_selection::StorageOrder == fdapde::RowMajor);
static_assert(std::same_as<typename safe_selection::Scalar, double>);
static_assert(std::is_constructible_v<
              direct_ternary, const selection_mask&, const selection_values&, const selection_values&>);
static_assert(!std::is_constructible_v<
              direct_ternary, selection_mask&&, const selection_values&, const selection_values&>);
static_assert(!std::is_constructible_v<
              direct_ternary, const selection_mask&, selection_values&&, const selection_values&>);
static_assert(!std::is_constructible_v<
              direct_ternary, const selection_mask&, const selection_values&, selection_values&&>);

template <typename Matrix>
concept exposes_boolean_repeat = requires(Matrix&& matrix) {
    std::forward<Matrix>(matrix).repeat(2, 3);
};

using repeat_owner = fdapde::Matrix<bool, 2, 3>;
using repeat_expression = decltype(~std::declval<repeat_owner&>());
using repeat_view = fdapde::MatrixView<const bool, 2, 3>;
using repeat_result = decltype(std::declval<const repeat_owner&>().repeat(2, 3));
using direct_repeat = fdapde::BoolMatrixRepeatOp<repeat_owner>;

static_assert(exposes_boolean_repeat<repeat_owner&>);
static_assert(exposes_boolean_repeat<const repeat_owner&>);
static_assert(!exposes_boolean_repeat<repeat_owner>);
static_assert(!exposes_boolean_repeat<const repeat_owner>);
static_assert(exposes_boolean_repeat<repeat_expression>);
static_assert(exposes_boolean_repeat<repeat_view>);
static_assert(repeat_result::Rows == fdapde::Dynamic);
static_assert(repeat_result::Cols == fdapde::Dynamic);
static_assert(repeat_result::StorageOrder == fdapde::RowMajor);
static_assert(repeat_result::ReadOnly == 1);
static_assert(repeat_result::NestAsRef == 0);
static_assert(std::same_as<typename repeat_result::Scalar, bool>);
static_assert(std::is_constructible_v<direct_repeat, const repeat_owner&, int, int>);
static_assert(!std::is_constructible_v<direct_repeat, repeat_owner&&, int, int>);

template <typename Matrix>
concept exposes_static_boolean_block = requires(Matrix&& matrix) {
    std::forward<Matrix>(matrix).template block<1, 2>(0, 0);
};

template <typename Matrix>
concept exposes_dynamic_boolean_block = requires(Matrix&& matrix) {
    std::forward<Matrix>(matrix).block(0, 0, 1, 2);
};

template <typename Matrix>
concept exposes_boolean_row = requires(Matrix&& matrix) { std::forward<Matrix>(matrix).row(0); };

template <typename Matrix>
concept exposes_boolean_col = requires(Matrix&& matrix) { std::forward<Matrix>(matrix).col(0); };

template <typename Matrix>
concept exposes_static_boolean_top_rows = requires(Matrix&& matrix) {
    std::forward<Matrix>(matrix).template top_rows<1>();
};

template <typename Matrix>
concept exposes_dynamic_boolean_top_rows = requires(Matrix&& matrix) {
    std::forward<Matrix>(matrix).top_rows(1);
};

template <typename Matrix>
concept exposes_static_boolean_bottom_rows = requires(Matrix&& matrix) {
    std::forward<Matrix>(matrix).template bottom_rows<1>();
};

template <typename Matrix>
concept exposes_dynamic_boolean_bottom_rows = requires(Matrix&& matrix) {
    std::forward<Matrix>(matrix).bottom_rows(1);
};

template <typename Matrix>
concept exposes_static_boolean_left_cols = requires(Matrix&& matrix) {
    std::forward<Matrix>(matrix).template left_cols<1>();
};

template <typename Matrix>
concept exposes_dynamic_boolean_left_cols = requires(Matrix&& matrix) {
    std::forward<Matrix>(matrix).left_cols(1);
};

template <typename Matrix>
concept exposes_static_boolean_right_cols = requires(Matrix&& matrix) {
    std::forward<Matrix>(matrix).template right_cols<1>();
};

template <typename Matrix>
concept exposes_dynamic_boolean_right_cols = requires(Matrix&& matrix) {
    std::forward<Matrix>(matrix).right_cols(1);
};

template <typename Block>
concept permits_boolean_block_coordinate_assignment = requires(Block& block) {
    block(0, 0) = true;
};

template <typename Block>
concept permits_boolean_block_vector_assignment = requires(Block& block) { block[0] = true; };

template <typename Block>
concept permits_boolean_block_mutation = requires(Block& block) {
    block.set();
    block.set(0, 0);
    block.clear();
    block.clear(0, 0);
};

using boolean_block_owner = fdapde::Matrix<bool, 3, 4>;
using mutable_boolean_block =
  decltype(std::declval<boolean_block_owner&>().template block<1, 2>(0, 0));
using const_boolean_block =
  decltype(std::declval<const boolean_block_owner&>().template block<1, 2>(0, 0));
using mutable_boolean_row = decltype(std::declval<boolean_block_owner&>().row(0));
using const_boolean_row = decltype(std::declval<const boolean_block_owner&>().row(0));
using safe_boolean_block_expression =
  decltype(std::declval<boolean_block_owner&>() | std::declval<boolean_block_owner&>());
using safe_boolean_expression_block = decltype(
  std::declval<safe_boolean_block_expression>().template block<1, 2>(0, 0));

static_assert(exposes_static_boolean_block<boolean_block_owner&>);
static_assert(exposes_dynamic_boolean_block<boolean_block_owner&>);
static_assert(exposes_boolean_row<boolean_block_owner&>);
static_assert(exposes_boolean_col<boolean_block_owner&>);
static_assert(exposes_static_boolean_top_rows<boolean_block_owner&>);
static_assert(exposes_dynamic_boolean_top_rows<boolean_block_owner&>);
static_assert(exposes_static_boolean_bottom_rows<boolean_block_owner&>);
static_assert(exposes_dynamic_boolean_bottom_rows<boolean_block_owner&>);
static_assert(exposes_static_boolean_left_cols<boolean_block_owner&>);
static_assert(exposes_dynamic_boolean_left_cols<boolean_block_owner&>);
static_assert(exposes_static_boolean_right_cols<boolean_block_owner&>);
static_assert(exposes_dynamic_boolean_right_cols<boolean_block_owner&>);
static_assert(
  !exposes_static_boolean_block<boolean_block_owner> &&
  !exposes_dynamic_boolean_block<boolean_block_owner> && !exposes_boolean_row<boolean_block_owner> &&
  !exposes_boolean_col<boolean_block_owner> && !exposes_static_boolean_top_rows<boolean_block_owner> &&
  !exposes_dynamic_boolean_top_rows<boolean_block_owner> &&
  !exposes_static_boolean_bottom_rows<boolean_block_owner> &&
  !exposes_dynamic_boolean_bottom_rows<boolean_block_owner> &&
  !exposes_static_boolean_left_cols<boolean_block_owner> &&
  !exposes_dynamic_boolean_left_cols<boolean_block_owner> &&
  !exposes_static_boolean_right_cols<boolean_block_owner> &&
  !exposes_dynamic_boolean_right_cols<boolean_block_owner>);
static_assert(
  !exposes_static_boolean_block<const boolean_block_owner> &&
  !exposes_dynamic_boolean_block<const boolean_block_owner> &&
  !exposes_boolean_row<const boolean_block_owner> && !exposes_boolean_col<const boolean_block_owner> &&
  !exposes_static_boolean_top_rows<const boolean_block_owner> &&
  !exposes_dynamic_boolean_top_rows<const boolean_block_owner> &&
  !exposes_static_boolean_bottom_rows<const boolean_block_owner> &&
  !exposes_dynamic_boolean_bottom_rows<const boolean_block_owner> &&
  !exposes_static_boolean_left_cols<const boolean_block_owner> &&
  !exposes_dynamic_boolean_left_cols<const boolean_block_owner> &&
  !exposes_static_boolean_right_cols<const boolean_block_owner> &&
  !exposes_dynamic_boolean_right_cols<const boolean_block_owner>);
static_assert(
  exposes_static_boolean_block<safe_boolean_block_expression> &&
  exposes_dynamic_boolean_block<safe_boolean_block_expression> &&
  exposes_boolean_row<safe_boolean_block_expression> &&
  exposes_boolean_col<safe_boolean_block_expression> &&
  exposes_static_boolean_top_rows<safe_boolean_block_expression> &&
  exposes_dynamic_boolean_top_rows<safe_boolean_block_expression> &&
  exposes_static_boolean_bottom_rows<safe_boolean_block_expression> &&
  exposes_dynamic_boolean_bottom_rows<safe_boolean_block_expression> &&
  exposes_static_boolean_left_cols<safe_boolean_block_expression> &&
  exposes_dynamic_boolean_left_cols<safe_boolean_block_expression> &&
  exposes_static_boolean_right_cols<safe_boolean_block_expression> &&
  exposes_dynamic_boolean_right_cols<safe_boolean_block_expression>);
static_assert(safe_boolean_expression_block::ReadOnly == 1);

static_assert(mutable_boolean_block::ReadOnly == 0);
static_assert(const_boolean_block::ReadOnly == 1);
static_assert(permits_boolean_block_coordinate_assignment<mutable_boolean_block>);
static_assert(permits_boolean_block_vector_assignment<mutable_boolean_row>);
static_assert(!permits_boolean_block_coordinate_assignment<const mutable_boolean_block>);
static_assert(!permits_boolean_block_vector_assignment<const mutable_boolean_row>);
static_assert(!permits_boolean_block_coordinate_assignment<const_boolean_block>);
static_assert(!permits_boolean_block_vector_assignment<const_boolean_row>);
static_assert(!permits_boolean_block_mutation<const_boolean_block>);

using direct_boolean_row_block =
  fdapde::BoolMatrixBlock<1, boolean_block_owner::Cols, const boolean_block_owner>;
using direct_static_boolean_block =
  fdapde::BoolMatrixBlock<1, 2, const boolean_block_owner>;
using direct_dynamic_boolean_block =
  fdapde::BoolMatrixBlock<fdapde::Dynamic, fdapde::Dynamic, const boolean_block_owner>;
static_assert(!std::is_constructible_v<direct_boolean_row_block, boolean_block_owner&&, int>);
static_assert(!std::is_constructible_v<direct_static_boolean_block, boolean_block_owner&&, int, int>);
static_assert(
  !std::is_constructible_v<direct_dynamic_boolean_block, boolean_block_owner&&, int, int, int, int>);

template <typename Matrix>
concept exposes_static_boolean_matrix_reshape = requires(Matrix&& matrix) {
    std::forward<Matrix>(matrix).template reshape<2, 2>();
};

template <typename Matrix>
concept exposes_static_boolean_vector_reshape = requires(Matrix&& matrix) {
    std::forward<Matrix>(matrix).template reshape<4>();
};

template <typename Matrix>
concept exposes_dynamic_boolean_matrix_reshape = requires(Matrix&& matrix) {
    std::forward<Matrix>(matrix).reshape(2, 2);
};

template <typename Matrix>
concept exposes_dynamic_boolean_vector_reshape = requires(Matrix&& matrix) {
    std::forward<Matrix>(matrix).reshape(4);
};

template <typename Matrix, int ExpectedReadOnly>
concept exposes_all_safe_boolean_reshape_accessors = requires(Matrix&& matrix) {
    requires (decltype(std::forward<Matrix>(matrix).template reshape<2, 2>())::ReadOnly == ExpectedReadOnly);
    requires (decltype(std::forward<Matrix>(matrix).template reshape<4>())::ReadOnly == ExpectedReadOnly);
    requires (decltype(std::forward<Matrix>(matrix).reshape(2, 2))::ReadOnly == ExpectedReadOnly);
    requires (decltype(std::forward<Matrix>(matrix).reshape(4))::ReadOnly == ExpectedReadOnly);
};

template <typename Matrix>
concept permits_direct_temporary_static_boolean_reshape = requires(Matrix&& matrix) {
    fdapde::BoolReshapeOp<2, 2, Matrix> {std::move(matrix)};
};

template <typename Matrix>
concept permits_direct_temporary_dynamic_boolean_reshape = requires(Matrix&& matrix) {
    fdapde::BoolReshapeOp<fdapde::Dynamic, fdapde::Dynamic, Matrix> {std::move(matrix), 2, 2};
};

template <typename Matrix>
concept permits_direct_temporary_vector_boolean_reshape = requires(Matrix&& matrix) {
    fdapde::BoolReshapeOp<fdapde::Dynamic, 1, Matrix> {std::move(matrix), 4};
};

template <typename Reshape>
concept permits_boolean_reshape_coordinate_write = requires(Reshape& reshape) { reshape(0, 0) = true; };

template <typename Reshape>
concept permits_boolean_reshape_vector_write = requires(Reshape& reshape) { reshape[0] = true; };

template <typename Reshape>
concept permits_boolean_reshape_assignment = requires(Reshape& lhs, const Reshape& rhs) { lhs = rhs; };

using boolean_reshape_owner = fdapde::Matrix<bool, 2, 2>;
using mutable_boolean_matrix_reshape =
  decltype(std::declval<boolean_reshape_owner&>().template reshape<2, 2>());
using mutable_boolean_vector_reshape =
  decltype(std::declval<boolean_reshape_owner&>().template reshape<4>());
using const_owner_boolean_matrix_reshape =
  decltype(std::declval<const boolean_reshape_owner&>().template reshape<2, 2>());
using const_owner_boolean_vector_reshape =
  decltype(std::declval<const boolean_reshape_owner&>().template reshape<4>());
using safe_boolean_reshape_expression =
  decltype(std::declval<boolean_reshape_owner&>() | std::declval<boolean_reshape_owner&>());

static_assert(exposes_static_boolean_matrix_reshape<boolean_reshape_owner&>);
static_assert(exposes_static_boolean_vector_reshape<boolean_reshape_owner&>);
static_assert(exposes_dynamic_boolean_matrix_reshape<boolean_reshape_owner&>);
static_assert(exposes_dynamic_boolean_vector_reshape<boolean_reshape_owner&>);
static_assert(
  !exposes_static_boolean_matrix_reshape<boolean_reshape_owner> &&
  !exposes_static_boolean_vector_reshape<boolean_reshape_owner> &&
  !exposes_dynamic_boolean_matrix_reshape<boolean_reshape_owner> &&
  !exposes_dynamic_boolean_vector_reshape<boolean_reshape_owner>);
static_assert(
  !exposes_static_boolean_matrix_reshape<const boolean_reshape_owner> &&
  !exposes_static_boolean_vector_reshape<const boolean_reshape_owner> &&
  !exposes_dynamic_boolean_matrix_reshape<const boolean_reshape_owner> &&
  !exposes_dynamic_boolean_vector_reshape<const boolean_reshape_owner>);
static_assert(exposes_all_safe_boolean_reshape_accessors<safe_boolean_reshape_expression, 1>);
static_assert(exposes_all_safe_boolean_reshape_accessors<const safe_boolean_reshape_expression, 1>);
static_assert(!permits_direct_temporary_static_boolean_reshape<boolean_reshape_owner>);
static_assert(!permits_direct_temporary_dynamic_boolean_reshape<boolean_reshape_owner>);
static_assert(!permits_direct_temporary_vector_boolean_reshape<boolean_reshape_owner>);
static_assert(!permits_direct_temporary_static_boolean_reshape<const boolean_reshape_owner>);
static_assert(!permits_direct_temporary_dynamic_boolean_reshape<const boolean_reshape_owner>);
static_assert(!permits_direct_temporary_vector_boolean_reshape<const boolean_reshape_owner>);
static_assert(mutable_boolean_matrix_reshape::ReadOnly == 0);
static_assert(const_owner_boolean_matrix_reshape::ReadOnly == 1);
static_assert(permits_boolean_reshape_coordinate_write<mutable_boolean_matrix_reshape>);
static_assert(permits_boolean_reshape_vector_write<mutable_boolean_vector_reshape>);
static_assert(!permits_boolean_reshape_coordinate_write<const mutable_boolean_matrix_reshape>);
static_assert(!permits_boolean_reshape_vector_write<const mutable_boolean_vector_reshape>);
static_assert(!permits_boolean_reshape_coordinate_write<const_owner_boolean_matrix_reshape>);
static_assert(!permits_boolean_reshape_vector_write<const_owner_boolean_vector_reshape>);
static_assert(permits_boolean_reshape_assignment<mutable_boolean_matrix_reshape>);
static_assert(!permits_boolean_reshape_assignment<const_owner_boolean_matrix_reshape>);
using temporary_boolean_reshape_assignment_result = decltype(
  std::declval<mutable_boolean_matrix_reshape&&>() =
  std::declval<const mutable_boolean_matrix_reshape&>());
static_assert(std::is_same_v<temporary_boolean_reshape_assignment_result, mutable_boolean_matrix_reshape>);

template <typename View>
concept permits_boolean_view_coordinate_write = requires(View& view) { view(0, 0) = true; };

template <typename View>
concept permits_boolean_view_vector_write = requires(View& view) { view[0] = true; };

template <typename View>
concept permits_boolean_view_bulk_mutation = requires(View& view) {
    view.set();
    view.clear();
};

template <typename View, typename Rhs>
concept permits_boolean_view_assignment = requires(View& view, const Rhs& rhs) { view = rhs; };

using fixed_boolean_view = fdapde::MatrixView<bool, 2, 3>;
using fixed_const_boolean_view = fdapde::MatrixView<const bool, 2, 3>;
using dynamic_boolean_view = fdapde::MatrixView<bool, fdapde::Dynamic, fdapde::Dynamic>;
using partial_boolean_view = fdapde::MatrixView<bool, 2, fdapde::Dynamic>;
using dynamic_const_boolean_view =
  fdapde::MatrixView<const bool, fdapde::Dynamic, fdapde::Dynamic>;
using boolean_view_word = typename fixed_boolean_view::bitpack_t;

static_assert(!std::is_default_constructible_v<fixed_boolean_view>);
static_assert(!std::is_default_constructible_v<fixed_const_boolean_view>);
static_assert(std::is_default_constructible_v<dynamic_boolean_view>);
static_assert(std::is_default_constructible_v<partial_boolean_view>);
static_assert(std::is_default_constructible_v<dynamic_const_boolean_view>);
static_assert(std::is_constructible_v<fixed_boolean_view, boolean_view_word*>);
static_assert(!std::is_constructible_v<fixed_boolean_view, const boolean_view_word*>);
static_assert(!std::is_constructible_v<fixed_boolean_view, bool*>);
static_assert(!std::is_constructible_v<fixed_boolean_view, unsigned char*>);
static_assert(std::is_constructible_v<fixed_const_boolean_view, boolean_view_word*>);
static_assert(std::is_constructible_v<fixed_const_boolean_view, const boolean_view_word*>);
static_assert(!std::is_constructible_v<fixed_const_boolean_view, const bool*>);
static_assert(std::is_same_v<decltype(std::declval<fixed_boolean_view&>().data()), boolean_view_word*>);
static_assert(
  std::is_same_v<decltype(std::declval<const fixed_boolean_view&>().data()), const boolean_view_word*>);
static_assert(
  std::is_same_v<decltype(std::declval<fixed_const_boolean_view&>().data()), const boolean_view_word*>);
static_assert(fdapde::is_boolean_matrix_v<fixed_const_boolean_view>);
static_assert(fixed_const_boolean_view::ReadOnly == 1);
static_assert(permits_boolean_view_coordinate_write<fixed_boolean_view>);
static_assert(!permits_boolean_view_coordinate_write<const fixed_boolean_view>);
static_assert(!permits_boolean_view_coordinate_write<fixed_const_boolean_view>);
static_assert(permits_boolean_view_vector_write<fdapde::MatrixView<bool, 1, 3>>);
static_assert(!permits_boolean_view_vector_write<const fdapde::MatrixView<bool, 1, 3>>);
static_assert(!permits_boolean_view_vector_write<fdapde::MatrixView<const bool, 1, 3>>);
static_assert(permits_boolean_view_bulk_mutation<fixed_boolean_view>);
static_assert(!permits_boolean_view_bulk_mutation<const fixed_boolean_view>);
static_assert(!permits_boolean_view_bulk_mutation<fixed_const_boolean_view>);
static_assert(permits_boolean_view_assignment<fixed_boolean_view, fixed_boolean_view>);
static_assert(!permits_boolean_view_assignment<fixed_const_boolean_view, fixed_const_boolean_view>);
using temporary_boolean_view_assignment_result =
  decltype(std::declval<fixed_boolean_view&&>() = std::declval<const fixed_boolean_view&>());
static_assert(std::is_same_v<temporary_boolean_view_assignment_result, fixed_boolean_view>);

template <int StorageOrder> void check_exact_boolean_pack_accounting() {
    using exact_pack = fdapde::Matrix<bool, 8, 8, StorageOrder>;
    using partial_pack = fdapde::Matrix<bool, 5, 13, StorageOrder>;
    using dynamic_column = fdapde::Matrix<bool, fdapde::Dynamic, 1, StorageOrder>;
    using partial_owner = fdapde::Matrix<bool, 2, fdapde::Dynamic, StorageOrder>;

    static_assert(exact_pack::StorageSize == 1);
    static_assert(partial_pack::StorageSize == 2);

    exact_pack bits;
    EXPECT_EQ(bits.bitpacks(), 1);
    EXPECT_EQ(bits.bitpack(0), 0u);
    bits(1, 6) = true;
    constexpr int expected_bit = StorageOrder == fdapde::RowMajor ? 14 : 49;
    EXPECT_NE(bits.bitpack(0) & (typename exact_pack::bitpack_t(1) << expected_bit), 0u);
    const exact_pack copied = bits;
    EXPECT_TRUE(copied == bits);

    const partial_pack partial;
    EXPECT_EQ(partial.bitpacks(), 2);
    EXPECT_EQ(partial.bitpack(0), 0u);
    EXPECT_EQ(partial.bitpack(1), 0u);

    partial_owner normalized_owner(2, 64);
    EXPECT_EQ(normalized_owner.rows(), 2);
    EXPECT_EQ(normalized_owner.cols(), 64);
    EXPECT_EQ(normalized_owner.bitpacks(), 2);
    normalized_owner(1, 63) = true;
    EXPECT_TRUE(normalized_owner(1, 63));

    constexpr std::array<int, 6> sizes {0, 1, 63, 64, 65, 130};
    constexpr std::array<int, 6> expected_counts {0, 1, 1, 1, 2, 3};
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        const dynamic_column dynamic(sizes[i]);
        EXPECT_EQ(dynamic.bitpacks(), expected_counts[i]);
    }

    exact_pack block_owner;
    auto whole = block_owner.template block<8, 8>(0, 0);
    EXPECT_EQ(whole.bitpacks(), 1);
    whole.set();
    for (int i = 0; i < block_owner.rows(); ++i) {
        for (int j = 0; j < block_owner.cols(); ++j) EXPECT_TRUE(block_owner(i, j));
    }
    whole.clear();

    auto trailing = block_owner.template block<2, 3>(6, 5);
    trailing.set();
    for (int i = 0; i < block_owner.rows(); ++i) {
        for (int j = 0; j < block_owner.cols(); ++j) {
            EXPECT_EQ(bool(block_owner(i, j)), i >= 6 && j >= 5);
        }
    }
    trailing.clear();
    for (int i = 0; i < block_owner.rows(); ++i) {
        for (int j = 0; j < block_owner.cols(); ++j) EXPECT_FALSE(block_owner(i, j));
    }

    using view_type = fdapde::MatrixView<bool, 8, 8, StorageOrder>;
    using bitpack_t = typename view_type::bitpack_t;
    constexpr bitpack_t canary = bitpack_t(0x5a5a);
    std::array<bitpack_t, 2> storage {bitpack_t(0), canary};
    view_type view(storage.data());
    EXPECT_EQ(view.bitpacks(), 1);
    view.set();
    EXPECT_EQ(storage[0], std::numeric_limits<bitpack_t>::max());
    EXPECT_EQ(storage[1], canary);
    EXPECT_TRUE(view == exact_pack(true));
    view.clear();
    EXPECT_EQ(storage[0], bitpack_t(0));
    EXPECT_EQ(storage[1], canary);

    using partial_view = fdapde::MatrixView<bool, 2, fdapde::Dynamic, StorageOrder>;
    std::array<bitpack_t, 3> partial_storage {bitpack_t(0), bitpack_t(0), canary};
    partial_view normalized_view(partial_storage.data(), 2, 64);
    EXPECT_EQ(normalized_view.rows(), 2);
    EXPECT_EQ(normalized_view.cols(), 64);
    EXPECT_EQ(normalized_view.bitpacks(), 2);
    normalized_view.set(1, 63);
    EXPECT_TRUE(normalized_view(1, 63));
    EXPECT_EQ(partial_storage[2], canary);
}

template <int StorageOrder> void check_boolean_owner_contracts() {
    using fixed_matrix = fdapde::Matrix<bool, 2, 3, StorageOrder>;
    using dynamic_matrix = fdapde::Matrix<bool, fdapde::Dynamic, fdapde::Dynamic, StorageOrder>;
    using partial_matrix = fdapde::Matrix<bool, 2, fdapde::Dynamic, StorageOrder>;
    using dynamic_row = fdapde::Matrix<bool, 1, fdapde::Dynamic, StorageOrder>;
    using dynamic_column = fdapde::Matrix<bool, fdapde::Dynamic, 1, StorageOrder>;
    constexpr int OppositeOrder = StorageOrder == fdapde::RowMajor ? fdapde::ColMajor : fdapde::RowMajor;

    EXPECT_THROW(static_cast<void>(dynamic_matrix(-1, 2)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(partial_matrix(3, 4)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(fixed_matrix(3, 2)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(dynamic_column(-1)), std::invalid_argument);
    EXPECT_THROW(
      static_cast<void>(dynamic_matrix(std::numeric_limits<int>::max(), 2)), std::length_error);

    const fixed_matrix fixed_zero(2, 3);
    EXPECT_FALSE(fixed_zero.any());
    EXPECT_EQ(fixed_zero.count(), 0);
    const dynamic_matrix dynamic_zero(2, 65);
    EXPECT_FALSE(dynamic_zero.any());
    EXPECT_EQ(dynamic_zero.count(), 0);
    const dynamic_matrix dynamic_ones(2, 65, true);
    EXPECT_TRUE(dynamic_ones.all());
    EXPECT_EQ(dynamic_ones.count(), 130);

    const std::array<bool, 6> expected {false, true, false, true, true, false};
    const auto expect_logical_values = [&expected](const auto& matrix) {
        EXPECT_EQ(matrix.rows(), 2);
        EXPECT_EQ(matrix.cols(), 3);
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                EXPECT_EQ(bool(matrix(i, j)), expected[static_cast<std::size_t>(i * matrix.cols() + j)]);
            }
        }
    };

    const bool array_data[6] {false, true, false, true, true, false};
    const fixed_matrix from_array(array_data);
    expect_logical_values(from_array);
    const std::vector<bool> vector_data {false, true, false, true, true, false};
    const fixed_matrix from_vector(vector_data);
    expect_logical_values(from_vector);
    const std::vector<bool> short_vector_data(5);
    EXPECT_THROW(static_cast<void>(fixed_matrix(short_vector_data)), std::invalid_argument);

    std::vector<bool> boundary_data(130);
    for (int i : {0, 63, 64, 129}) boundary_data[static_cast<std::size_t>(i)] = true;
    const dynamic_column boundary_vector(boundary_data);
    EXPECT_EQ(boundary_vector.size(), 130);
    for (int i = 0; i < boundary_vector.size(); ++i) {
        EXPECT_EQ(bool(boundary_vector[i]), i == 0 || i == 63 || i == 64 || i == 129);
    }

    const fdapde::Matrix<int, 2, 3, OppositeOrder> numeric({0, 2, -3, 0, 4, 0});
    const fixed_matrix converted(numeric);
    const std::array<bool, 6> converted_expected {false, true, true, false, true, false};
    for (int i = 0; i < converted.rows(); ++i) {
        for (int j = 0; j < converted.cols(); ++j) {
            EXPECT_EQ(
              bool(converted(i, j)), converted_expected[static_cast<std::size_t>(i * converted.cols() + j)]);
        }
    }
    const fdapde::Matrix<int, fdapde::Dynamic, fdapde::Dynamic, OppositeOrder> smaller_numeric(1, 2);
    EXPECT_THROW(static_cast<void>(fixed_matrix(smaller_numeric)), std::invalid_argument);

    fdapde::Matrix<bool, 1, 2, StorageOrder> proxy_values;
    proxy_values(0, 1) = true;
    proxy_values(0, 0) = proxy_values(0, 1);
    proxy_values(0, 1) = false;
    EXPECT_TRUE(proxy_values(0, 0));

    dynamic_matrix copy_source(2, 65);
    copy_source(0, 0) = true;
    copy_source(1, 64) = true;
    dynamic_matrix copy_target(1, 2, true);
    copy_target = copy_source;
    EXPECT_EQ(copy_target.rows(), 2);
    EXPECT_EQ(copy_target.cols(), 65);
    EXPECT_EQ(copy_target.bitpacks(), 3);
    EXPECT_EQ(copy_target.count(), 2);
    EXPECT_TRUE(copy_target(0, 0));
    EXPECT_TRUE(copy_target(1, 64));
    copy_source.clear();
    EXPECT_EQ(copy_target.count(), 2);

    const auto row_zero_xpr = dynamic_row::Zero(5);
    const auto row_ones_xpr = dynamic_row::Ones(5);
    EXPECT_EQ(row_zero_xpr.rows(), 1);
    EXPECT_EQ(row_zero_xpr.cols(), 5);
    EXPECT_EQ(row_ones_xpr.rows(), 1);
    EXPECT_EQ(row_ones_xpr.cols(), 5);
    if (row_zero_xpr.rows() == 1 && row_zero_xpr.cols() == 5 && row_ones_xpr.rows() == 1 &&
        row_ones_xpr.cols() == 5) {
        const dynamic_row row_zero(row_zero_xpr);
        const dynamic_row row_ones(row_ones_xpr);
        EXPECT_FALSE(row_zero.any());
        EXPECT_TRUE(row_ones.all());
    }
    const dynamic_column column_zero = dynamic_column::Zero(5);
    const dynamic_column column_ones = dynamic_column::Ones(5);
    EXPECT_EQ(column_zero.rows(), 5);
    EXPECT_EQ(column_zero.cols(), 1);
    EXPECT_FALSE(column_zero.any());
    EXPECT_TRUE(column_ones.all());

    dynamic_column retained(10);
    retained[1] = true;
    retained[9] = true;
    retained.resize(5);
    retained.resize(10);
    EXPECT_TRUE(retained[1]);
    EXPECT_FALSE(retained[9]);

    dynamic_column exposed_padding(5, true);
    exposed_padding.resize(10);
    for (int i = 0; i < 5; ++i) EXPECT_TRUE(exposed_padding[i]);
    for (int i = 5; i < 10; ++i) EXPECT_FALSE(exposed_padding[i]);

    partial_matrix partial(2, 3);
    partial(1, 2) = true;
    const int partial_bitpacks = partial.bitpacks();
    EXPECT_THROW(partial.resize(3, 3), std::invalid_argument);
    EXPECT_EQ(partial.rows(), 2);
    EXPECT_EQ(partial.cols(), 3);
    EXPECT_EQ(partial.bitpacks(), partial_bitpacks);
    EXPECT_TRUE(partial(1, 2));

    dynamic_matrix bounded(1, 2);
    bounded(0, 1) = true;
    const int bounded_bitpacks = bounded.bitpacks();
    EXPECT_THROW(bounded.resize(std::numeric_limits<int>::max(), 2), std::length_error);
    EXPECT_EQ(bounded.rows(), 1);
    EXPECT_EQ(bounded.cols(), 2);
    EXPECT_EQ(bounded.bitpacks(), bounded_bitpacks);
    if (bounded.rows() == 1 && bounded.cols() == 2 && bounded.bitpacks() == bounded_bitpacks) {
        EXPECT_TRUE(bounded(0, 1));
    }
}

template <int StorageOrder> void check_boolean_expression_contracts() {
    using fixed_matrix = fdapde::Matrix<bool, 2, 2, StorageOrder>;
    using layout_matrix = fdapde::Matrix<bool, 2, 3, StorageOrder>;
    using dynamic_matrix = fdapde::Matrix<bool, fdapde::Dynamic, fdapde::Dynamic, StorageOrder>;
    using overlap_matrix = fdapde::Matrix<bool, 1, 4, StorageOrder>;
    constexpr int OppositeOrder = StorageOrder == fdapde::RowMajor ? fdapde::ColMajor : fdapde::RowMajor;
    using opposite_layout_matrix = fdapde::Matrix<bool, 2, 3, OppositeOrder>;

    const auto expect_values = [](const auto& matrix, const auto& expected) {
        ASSERT_EQ(matrix.size(), static_cast<int>(expected.size()));
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                EXPECT_EQ(
                  bool(matrix(i, j)), expected[static_cast<std::size_t>(i * matrix.cols() + j)]);
            }
        }
    };

    fixed_matrix lhs({true, false, true, false});
    fixed_matrix rhs({true, true, false, false});
    const auto stored_expression = [&lhs, &rhs] { return ~((lhs | rhs) ^ (lhs & rhs)); }();
    const fixed_matrix stored_result(stored_expression);
    expect_values(stored_result, std::array {true, false, false, true});

    const std::array layout_values {false, true, true, true, false, false};
    const layout_matrix layout(std::vector<bool>(layout_values.begin(), layout_values.end()));
    const opposite_layout_matrix other_layout(std::vector<bool>(layout_values.begin(), layout_values.end()));
    EXPECT_EQ((layout ^ other_layout).bitpack(0), typename layout_matrix::bitpack_t(0));

    const layout_matrix constructed(other_layout);
    expect_values(constructed, layout_values);
    layout_matrix assigned;
    assigned = other_layout;
    expect_values(assigned, layout_values);
    layout_matrix compounded(layout);
    compounded ^= other_layout;
    expect_values(compounded, std::array {false, false, false, false, false, false});

    dynamic_matrix dynamic_lhs(2, 3);
    dynamic_lhs(0, 1) = true;
    const dynamic_matrix dynamic_rhs(3, 2);
    EXPECT_THROW(static_cast<void>(dynamic_lhs & dynamic_rhs), std::invalid_argument);
    EXPECT_THROW(dynamic_lhs |= dynamic_rhs, std::invalid_argument);
    EXPECT_EQ(dynamic_lhs.rows(), 2);
    EXPECT_EQ(dynamic_lhs.cols(), 3);
    EXPECT_TRUE(dynamic_lhs(0, 1));
    EXPECT_EQ(dynamic_lhs.count(), 1);

    overlap_matrix assignment_overlap({true, false, true, false});
    assignment_overlap.right_cols(3) = assignment_overlap.template left_cols<3>();
    expect_values(assignment_overlap, std::array {true, true, false, true});

    overlap_matrix and_overlap({true, false, true, true});
    and_overlap.right_cols(3) &= and_overlap.template left_cols<3>();
    expect_values(and_overlap, std::array {true, false, false, true});

    overlap_matrix or_overlap({false, true, false, false});
    or_overlap.right_cols(3) |= or_overlap.template left_cols<3>();
    expect_values(or_overlap, std::array {false, true, true, false});

    overlap_matrix xor_overlap({true, false, true, false});
    xor_overlap.right_cols(3) ^= xor_overlap.template left_cols<3>();
    expect_values(xor_overlap, std::array {true, true, true, true});
}

template <int StorageOrder> void check_boolean_block_contracts() {
    using fixed_matrix = fdapde::Matrix<bool, 3, 4, StorageOrder>;
    using dynamic_matrix = fdapde::Matrix<bool, fdapde::Dynamic, fdapde::Dynamic, StorageOrder>;

    const auto expect_values = [](const auto& matrix, const auto& expected) {
        ASSERT_EQ(matrix.size(), static_cast<int>(expected.size()));
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                EXPECT_EQ(
                  bool(matrix(i, j)), expected[static_cast<std::size_t>(i * matrix.cols() + j)]);
            }
        }
    };

    const fixed_matrix source(
      {false, true, false, true, true, false, true, false, false, true, true, false});
    const auto static_block = source.template block<2, 2>(1, 1);
    EXPECT_EQ(static_block.rows(), 2);
    EXPECT_EQ(static_block.cols(), 2);
    expect_values(static_block, std::array {false, true, true, true});
    const fdapde::Matrix<bool, 2, 2, StorageOrder> static_materialized(static_block);
    expect_values(static_materialized, std::array {false, true, true, true});

    const auto dynamic_block = source.block(1, 1, 2, 2);
    EXPECT_EQ(dynamic_block.rows(), 2);
    EXPECT_EQ(dynamic_block.cols(), 2);
    expect_values(dynamic_block, std::array {false, true, true, true});
    const dynamic_matrix dynamic_materialized(dynamic_block);
    expect_values(dynamic_materialized, std::array {false, true, true, true});

    expect_values(source.row(1), std::array {true, false, true, false});
    expect_values(source.col(2), std::array {false, true, true});
    expect_values(source.template top_rows<1>(), std::array {false, true, false, true});
    expect_values(source.top_rows(1), std::array {false, true, false, true});
    expect_values(source.template bottom_rows<1>(), std::array {false, true, true, false});
    expect_values(source.bottom_rows(1), std::array {false, true, true, false});
    expect_values(source.template left_cols<1>(), std::array {false, true, false});
    expect_values(source.left_cols(1), std::array {false, true, false});
    expect_values(
      source.template right_cols<2>(), std::array {false, true, true, false, true, false});
    expect_values(source.right_cols(2), std::array {false, true, true, false, true, false});

    fdapde::Matrix<bool, 1, 3, StorageOrder> row_vector({false, true, true});
    auto row_scalar = row_vector.col(2);
    EXPECT_EQ(row_scalar.rows(), 1);
    EXPECT_EQ(row_scalar.cols(), 1);
    EXPECT_TRUE(row_scalar(0, 0));
    row_scalar(0, 0) = false;
    EXPECT_FALSE(row_vector(0, 2));

    fdapde::Matrix<bool, 3, 1, StorageOrder> column_vector({false, true, true});
    auto column_scalar = column_vector.row(2);
    EXPECT_EQ(column_scalar.rows(), 1);
    EXPECT_EQ(column_scalar.cols(), 1);
    EXPECT_TRUE(column_scalar(0, 0));
    column_scalar(0, 0) = false;
    EXPECT_FALSE(column_vector(2, 0));

    fdapde::Matrix<bool, 5, 25, StorageOrder> packed_owner;
    packed_owner(1, 2) = true;
    packed_owner(3, 1) = true;
    packed_owner(2, 22) = true;
    packed_owner(3, 23) = true;
    const auto packed_block = packed_owner.block(1, 1, 3, 23);
    using bitpack_t = typename decltype(packed_owner)::bitpack_t;
    EXPECT_EQ(packed_block.bitpacks(), 2);
    if constexpr (StorageOrder == fdapde::RowMajor) {
        EXPECT_EQ(
          packed_block.bitpack(0),
          (bitpack_t(1) << 1) | (bitpack_t(1) << 44) | (bitpack_t(1) << 46));
        EXPECT_EQ(packed_block.bitpack(1), bitpack_t(1) << 4);
    } else {
        EXPECT_EQ(packed_block.bitpack(0), (bitpack_t(1) << 2) | (bitpack_t(1) << 3));
        EXPECT_EQ(packed_block.bitpack(1), (bitpack_t(1) << 0) | (bitpack_t(1) << 4));
    }

    fdapde::Matrix<bool, 2, 4, StorageOrder> named_assignment(
      {false, false, true, false, false, false, false, true});
    auto named_destination = named_assignment.template block<2, 2>(0, 0);
    const auto named_source = named_assignment.template block<2, 2>(0, 2);
    named_destination = named_source;
    expect_values(named_assignment, std::array {true, false, true, false, false, true, false, true});
    named_destination(0, 0) = false;
    EXPECT_FALSE(named_assignment(0, 0));
    EXPECT_TRUE(named_assignment(0, 2));

    fdapde::Matrix<bool, 1, 4, StorageOrder> overlapping_assignment({true, false, true, false});
    overlapping_assignment.template block<1, 3>(0, 1) =
      overlapping_assignment.template block<1, 3>(0, 0);
    expect_values(overlapping_assignment, std::array {true, true, false, true});

    fixed_matrix expression_lhs(
      {false, false, false, false, false, true, false, true, false, false, false, false});
    fixed_matrix expression_rhs(
      {false, false, false, false, true, false, true, false, false, false, false, false});
    const auto stored_expression_block = [&expression_lhs, &expression_rhs] {
        return (expression_lhs | expression_rhs).template block<1, 2>(1, 1);
    }();
    const fdapde::Matrix<bool, 1, 2, StorageOrder> stored_expression_result(stored_expression_block);
    expect_values(stored_expression_result, std::array {true, true});

    dynamic_matrix bounds(3, 4);
    bounds(1, 1) = true;
    EXPECT_THROW(static_cast<void>(bounds.row(-1)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(bounds.row(bounds.rows())), std::out_of_range);
    EXPECT_THROW(static_cast<void>(bounds.col(-1)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(bounds.col(bounds.cols())), std::out_of_range);
    EXPECT_THROW(static_cast<void>(bounds.template block<2, 2>(-1, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(bounds.template block<2, 2>(2, 3)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(bounds.block(0, 0, 0, 1)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(bounds.block(0, 0, 1, -1)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(bounds.block(-1, 0, 1, 1)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(bounds.block(0, -1, 1, 1)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(bounds.block(2, 3, 2, 2)), std::out_of_range);
    EXPECT_THROW(
      static_cast<void>(bounds.bottom_rows(std::numeric_limits<int>::min())), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(bounds.bottom_rows(4)), std::out_of_range);
    EXPECT_THROW(
      static_cast<void>(bounds.right_cols(std::numeric_limits<int>::min())), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(bounds.right_cols(5)), std::out_of_range);

    auto local_block = bounds.block(0, 0, 2, 2);
    const auto& const_local_block = local_block;
    EXPECT_THROW(static_cast<void>(local_block(-1, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(local_block(2, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_local_block(0, -1)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_local_block(0, 2)), std::out_of_range);
    auto local_row = bounds.row(1);
    const auto& const_local_row = local_row;
    EXPECT_THROW(static_cast<void>(local_row[-1]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(local_row[local_row.size()]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_local_row[-1]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_local_row[const_local_row.size()]), std::out_of_range);
    EXPECT_THROW(local_block.set(-1, 0), std::out_of_range);
    EXPECT_THROW(local_block.clear(0, 2), std::out_of_range);
    EXPECT_THROW(static_cast<void>(local_block.bitpack(-1)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(local_block.bitpack(local_block.bitpacks())), std::out_of_range);
    EXPECT_TRUE(bounds(1, 1));
}

template <int StorageOrder> void check_boolean_reshape_contracts() {
    using source_matrix = fdapde::Matrix<bool, 2, 3, StorageOrder>;
    using target_matrix = fdapde::Matrix<bool, 3, 2, StorageOrder>;
    using dynamic_matrix =
      fdapde::Matrix<bool, fdapde::Dynamic, fdapde::Dynamic, StorageOrder>;

    const auto expect_values = [](const auto& matrix, const auto& expected) {
        ASSERT_EQ(matrix.size(), static_cast<int>(expected.size()));
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                EXPECT_EQ(
                  bool(matrix(i, j)), expected[static_cast<std::size_t>(i * matrix.cols() + j)]);
            }
        }
    };

    source_matrix source({false, true, true, false, true, false});
    const std::array<bool, 6> reshaped_expected = StorageOrder == fdapde::RowMajor
      ? std::array<bool, 6> {false, true, true, false, true, false}
      : std::array<bool, 6> {false, true, false, true, true, false};
    const std::array<bool, 6> flat_expected = StorageOrder == fdapde::RowMajor
      ? std::array<bool, 6> {false, true, true, false, true, false}
      : std::array<bool, 6> {false, false, true, true, true, false};

    auto static_reshape = source.template reshape<3, 2>();
    auto dynamic_reshape = source.reshape(3, 2);
    expect_values(target_matrix(static_reshape), reshaped_expected);
    expect_values(target_matrix(dynamic_reshape), reshaped_expected);
    EXPECT_EQ(static_reshape.bitpacks(), source.bitpacks());
    EXPECT_EQ(static_reshape.bitpack(0), source.bitpack(0));

    dynamic_matrix packed_source(5, 13);
    packed_source(0, 0) = true;
    packed_source(4, 12) = true;
    const auto packed_static_reshape = packed_source.template reshape<1, 65>();
    const auto packed_dynamic_reshape = packed_source.reshape(1, 65);
    const auto packed_column_reshape = packed_source.reshape(65);
    EXPECT_EQ(packed_static_reshape.bitpacks(), 2);
    EXPECT_EQ(packed_dynamic_reshape.bitpacks(), 2);
    for (int i = 0; i < packed_source.bitpacks(); ++i) {
        EXPECT_EQ(packed_static_reshape.bitpack(i), packed_source.bitpack(i));
        EXPECT_EQ(packed_dynamic_reshape.bitpack(i), packed_source.bitpack(i));
    }
    EXPECT_TRUE(packed_static_reshape[64]);
    EXPECT_TRUE(packed_column_reshape[64]);

    const dynamic_matrix packed_zero(5, 13);
    const auto stored_packed_expression = [&packed_source, &packed_zero] {
        return (packed_source | packed_zero).template reshape<1, 65>();
    }();
    EXPECT_EQ(stored_packed_expression.bitpacks(), 2);
    for (int i = 0; i < packed_source.bitpacks(); ++i) {
        EXPECT_EQ(stored_packed_expression.bitpack(i), packed_source.bitpack(i));
    }

    auto row = source.template reshape<1, 6>();
    auto column = source.template reshape<6>();
    const fdapde::BoolReshapeOp<1, fdapde::Dynamic, source_matrix> direct_row(source, source.size());
    EXPECT_EQ(direct_row.rows(), 1);
    EXPECT_EQ(direct_row.cols(), source.size());
    for (int i = 0; i < source.size(); ++i) {
        EXPECT_EQ(bool(row[i]), flat_expected[static_cast<std::size_t>(i)]);
        EXPECT_EQ(bool(column[i]), flat_expected[static_cast<std::size_t>(i)]);
        EXPECT_EQ(bool(direct_row[i]), flat_expected[static_cast<std::size_t>(i)]);
    }

    static_reshape(0, 1) = false;
    if constexpr (StorageOrder == fdapde::RowMajor) {
        EXPECT_FALSE(source(0, 1));
        EXPECT_TRUE(source(1, 1));
    } else {
        EXPECT_TRUE(source(0, 1));
        EXPECT_FALSE(source(1, 1));
    }

    const source_matrix lifetime_source({false, true, true, false, true, false});
    const source_matrix lifetime_zero;
    const auto stored_expression = [&lifetime_source, &lifetime_zero] {
        return (lifetime_source | lifetime_zero).template reshape<3, 2>();
    }();
    expect_values(target_matrix(stored_expression), reshaped_expected);

    source_matrix named_destination;
    source_matrix named_source({true, false, true, true, false, true});
    auto destination_reshape = named_destination.template reshape<3, 2>();
    const auto source_reshape = named_source.template reshape<3, 2>();
    destination_reshape = source_reshape;
    expect_values(named_destination, std::array {true, false, true, true, false, true});
    destination_reshape(0, 0) = false;
    EXPECT_TRUE(named_source(0, 0));

    source_matrix temporary_destination;
    temporary_destination.template reshape<3, 2>() = named_source.template reshape<3, 2>();
    expect_values(temporary_destination, std::array {true, false, true, true, false, true});

    fdapde::Matrix<bool, 4, 1, StorageOrder> vector_destination;
    const fdapde::Matrix<bool, 2, 2, StorageOrder> matrix_source({true, false, false, true});
    vector_destination.template reshape<2, 2>() = matrix_source;
    const auto matrix_shaped_destination = vector_destination.template reshape<2, 2>();
    expect_values(matrix_shaped_destination, std::array {true, false, false, true});

    fdapde::Matrix<bool, 1, 5, StorageOrder> overlapping({true, false, true, false, false});
    overlapping.template block<1, 4>(0, 1).template reshape<1, 4>() =
      overlapping.template block<1, 4>(0, 0).template reshape<1, 4>();
    expect_values(overlapping, std::array {true, true, false, true, false});

    const auto& const_static_reshape = static_reshape;
    EXPECT_THROW(static_cast<void>(static_reshape(-1, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(static_reshape(3, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_static_reshape(0, -1)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_static_reshape(0, 2)), std::out_of_range);
    const auto& const_row = row;
    EXPECT_THROW(static_cast<void>(row[-1]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(row[row.size()]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_row[-1]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_row[const_row.size()]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(static_reshape.bitpack(-1)), std::out_of_range);
    EXPECT_THROW(
      static_cast<void>(static_reshape.bitpack(static_reshape.bitpacks())), std::out_of_range);

    dynamic_matrix empty;
    const auto empty_matrix = empty.reshape(0, 5);
    const auto empty_column = empty.reshape(0);
    EXPECT_EQ(empty_matrix.rows(), 0);
    EXPECT_EQ(empty_matrix.cols(), 5);
    EXPECT_EQ(empty_matrix.size(), 0);
    EXPECT_EQ(empty_matrix.bitpacks(), 0);
    EXPECT_EQ(empty_column.rows(), 0);
    EXPECT_EQ(empty_column.cols(), 1);
    EXPECT_EQ(empty_column.size(), 0);
    EXPECT_EQ(empty_column.bitpacks(), 0);
}

template <int StorageOrder> void check_boolean_selection_contracts() {
    constexpr int OppositeOrder = StorageOrder == fdapde::RowMajor ? fdapde::ColMajor : fdapde::RowMajor;
    using mask_matrix = fdapde::Matrix<bool, 2, 2, StorageOrder>;
    using value_matrix = fdapde::Matrix<double, 2, 2, StorageOrder>;
    using opposite_value_matrix = fdapde::Matrix<double, 2, 2, OppositeOrder>;
    using mask_view = fdapde::MatrixView<const bool, 2, 2, StorageOrder>;
    using value_view = fdapde::MatrixView<const double, 2, 2, StorageOrder>;
    using dynamic_mask =
      fdapde::Matrix<bool, fdapde::Dynamic, fdapde::Dynamic, StorageOrder>;
    using dynamic_values =
      fdapde::Matrix<double, fdapde::Dynamic, fdapde::Dynamic, StorageOrder>;

    const auto expect_values = [](const auto& matrix, const auto& expected) {
        ASSERT_EQ(matrix.size(), static_cast<int>(expected.size()));
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                EXPECT_DOUBLE_EQ(
                  matrix(i, j), expected[static_cast<std::size_t>(i * matrix.cols() + j)]);
            }
        }
    };

    const mask_matrix diagonal({true, false, false, true});
    const value_matrix true_values({1.0, 2.0, 3.0, 4.0});
    const opposite_value_matrix false_values({10.0, 20.0, 30.0, 40.0});
    const value_matrix selected(diagonal.select(true_values, false_values));
    expect_values(selected, std::array {1.0, 20.0, 30.0, 4.0});
    const value_matrix inverse_selected((~diagonal).select(true_values, false_values));
    expect_values(inverse_selected, std::array {10.0, 2.0, 3.0, 40.0});

    const value_matrix zero;
    const opposite_value_matrix opposite_zero;
    const auto stored_expression = [&] {
        return (~diagonal).select(true_values + zero, false_values - opposite_zero);
    }();
    const value_matrix stored_result(stored_expression);
    expect_values(stored_result, std::array {10.0, 2.0, 3.0, 40.0});

    const auto stored_views =
      mask_view(diagonal.data()).select(value_view(true_values.data()), false_values);
    const value_matrix view_result(stored_views);
    expect_values(view_result, std::array {1.0, 20.0, 30.0, 4.0});

    value_matrix aliased(true_values);
    aliased = diagonal.select(aliased, false_values);
    expect_values(aliased, std::array {1.0, 20.0, 30.0, 4.0});

    dynamic_mask condition(2, 3);
    condition(0, 0) = true;
    condition(1, 2) = true;
    dynamic_values dynamic_true(2, 3);
    dynamic_values dynamic_false(2, 3);
    dynamic_true(0, 0) = 1.0;
    dynamic_true(1, 2) = 2.0;
    dynamic_false(0, 1) = 3.0;
    const dynamic_values dynamic_selected(condition.select(dynamic_true, dynamic_false));
    EXPECT_EQ(dynamic_selected.rows(), 2);
    EXPECT_EQ(dynamic_selected.cols(), 3);
    EXPECT_DOUBLE_EQ(dynamic_selected(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(dynamic_selected(0, 1), 3.0);
    EXPECT_DOUBLE_EQ(dynamic_selected(1, 2), 2.0);

    const dynamic_values wrong_shape(3, 2);
    EXPECT_THROW(static_cast<void>(condition.select(wrong_shape, dynamic_false)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(condition.select(dynamic_true, wrong_shape)), std::invalid_argument);

    const dynamic_mask empty_condition(0, 3);
    const dynamic_values empty_true(0, 3);
    const dynamic_values empty_false(0, 3);
    const auto empty_selection = empty_condition.select(empty_true, empty_false);
    EXPECT_EQ(empty_selection.rows(), 0);
    EXPECT_EQ(empty_selection.cols(), 3);
    EXPECT_EQ(empty_selection.size(), 0);

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const fdapde::Matrix<double, 2, 2, StorageOrder> nan_values({0.0, nan, 2.0, nan});
    const auto nan_mask = fdapde::nan_indicator(nan_values);
    EXPECT_EQ(nan_mask.rows(), 2);
    EXPECT_EQ(nan_mask.cols(), 2);
    EXPECT_FALSE(bool(nan_mask(0, 0)));
    EXPECT_TRUE(bool(nan_mask(0, 1)));
    EXPECT_FALSE(bool(nan_mask(1, 0)));
    EXPECT_TRUE(bool(nan_mask(1, 1)));

    const fdapde::Matrix<double, 1, 3, StorageOrder> nan_row({nan, 1.0, nan});
    const auto nan_row_mask = fdapde::nan_indicator(nan_row);
    EXPECT_EQ(nan_row_mask.rows(), 1);
    EXPECT_EQ(nan_row_mask.cols(), 3);
    EXPECT_TRUE(bool(nan_row_mask(0, 0)));
    EXPECT_FALSE(bool(nan_row_mask(0, 1)));
    EXPECT_TRUE(bool(nan_row_mask(0, 2)));
}

template <int StorageOrder> void check_boolean_repeat_contracts() {
    using dynamic_matrix =
      fdapde::Matrix<bool, fdapde::Dynamic, fdapde::Dynamic, StorageOrder>;
    using dynamic_column = fdapde::Matrix<bool, fdapde::Dynamic, 1, StorageOrder>;

    dynamic_matrix source(3, 4);
    source(0, 0) = true;
    source(0, 2) = true;
    source(1, 1) = true;
    source(2, 0) = true;
    source(2, 1) = true;
    source(2, 3) = true;

    const auto repeated = source.repeat(2, 4);
    EXPECT_EQ(repeated.rows(), 6);
    EXPECT_EQ(repeated.cols(), 16);
    EXPECT_EQ(repeated.bitpacks(), 2);
    EXPECT_TRUE(repeated.any());
    EXPECT_FALSE(repeated.all());
    EXPECT_EQ(repeated.count(), source.count() * 8);

    dynamic_matrix expected(6, 16);
    for (int i = 0; i < expected.rows(); ++i) {
        for (int j = 0; j < expected.cols(); ++j) {
            expected(i, j) = bool(source(i % source.rows(), j % source.cols()));
            EXPECT_EQ(bool(repeated(i, j)), bool(expected(i, j)));
        }
    }
    for (int i = 0; i < repeated.bitpacks(); ++i) {
        EXPECT_EQ(repeated.bitpack(i), expected.bitpack(i));
    }

    dynamic_column column(5);
    column[1] = true;
    column[4] = true;
    const auto tiled_column = column.repeat(1, 4);
    EXPECT_EQ(tiled_column.rows(), 5);
    EXPECT_EQ(tiled_column.cols(), 4);
    for (int i = 0; i < tiled_column.rows(); ++i) {
        for (int j = 0; j < tiled_column.cols(); ++j) {
            EXPECT_EQ(bool(tiled_column(i, j)), bool(column[i]));
        }
    }

    const auto stored_expression = [&] { return (~source).repeat(1, 2); }();
    const dynamic_matrix stored_result(stored_expression);
    EXPECT_EQ(stored_result.rows(), 3);
    EXPECT_EQ(stored_result.cols(), 8);
    for (int i = 0; i < stored_result.rows(); ++i) {
        for (int j = 0; j < stored_result.cols(); ++j) {
            EXPECT_EQ(bool(stored_result(i, j)), !bool(source(i, j % source.cols())));
        }
    }

    const auto stored_view =
      fdapde::MatrixView<const bool, 3, 4, StorageOrder>(source.data()).repeat(2, 1);
    const dynamic_matrix view_result(stored_view);
    EXPECT_EQ(view_result.rows(), 6);
    EXPECT_EQ(view_result.cols(), 4);
    for (int i = 0; i < view_result.rows(); ++i) {
        for (int j = 0; j < view_result.cols(); ++j) {
            EXPECT_EQ(bool(view_result(i, j)), bool(source(i % source.rows(), j)));
        }
    }

    dynamic_matrix aliased(1, 3);
    aliased(0, 0) = true;
    aliased(0, 2) = true;
    aliased = aliased.repeat(2, 1);
    EXPECT_EQ(aliased.rows(), 2);
    EXPECT_EQ(aliased.cols(), 3);
    for (int i = 0; i < aliased.rows(); ++i) {
        EXPECT_TRUE(bool(aliased(i, 0)));
        EXPECT_FALSE(bool(aliased(i, 1)));
        EXPECT_TRUE(bool(aliased(i, 2)));
    }

    const dynamic_matrix zero_rows(0, 3);
    const dynamic_matrix zero_cols(3, 0);
    const auto repeated_zero_rows = zero_rows.repeat(2, 4);
    const auto repeated_zero_cols = zero_cols.repeat(2, 4);
    EXPECT_EQ(repeated_zero_rows.rows(), 0);
    EXPECT_EQ(repeated_zero_rows.cols(), 12);
    EXPECT_EQ(repeated_zero_rows.bitpacks(), 0);
    EXPECT_EQ(repeated_zero_cols.rows(), 6);
    EXPECT_EQ(repeated_zero_cols.cols(), 0);
    EXPECT_EQ(repeated_zero_cols.bitpacks(), 0);

    EXPECT_THROW(static_cast<void>(source.repeat(0, 1)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(source.repeat(1, 0)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(source.repeat(-1, 1)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(source.repeat(1, -1)), std::invalid_argument);
    EXPECT_THROW(
      static_cast<void>(source.repeat(std::numeric_limits<int>::max(), 1)), std::length_error);
    EXPECT_THROW(
      static_cast<void>(source.repeat(1, std::numeric_limits<int>::max())), std::length_error);
    EXPECT_THROW(static_cast<void>(source.repeat(20'000, 20'000)), std::length_error);

    EXPECT_THROW(static_cast<void>(repeated(-1, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(repeated(repeated.rows(), 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(repeated(0, -1)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(repeated(0, repeated.cols())), std::out_of_range);
    EXPECT_THROW(static_cast<void>(repeated.bitpack(-1)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(repeated.bitpack(repeated.bitpacks())), std::out_of_range);
}

template <int StorageOrder> void check_boolean_terminal_contracts() {
    using exact_matrix = fdapde::Matrix<bool, 8, 8, StorageOrder>;
    using tail_matrix = fdapde::Matrix<bool, 5, 13, StorageOrder>;
    using layout_matrix = fdapde::Matrix<bool, 2, 3, StorageOrder>;
    using dynamic_matrix =
      fdapde::Matrix<bool, fdapde::Dynamic, fdapde::Dynamic, StorageOrder>;
    constexpr int OppositeOrder = StorageOrder == fdapde::RowMajor ? fdapde::ColMajor : fdapde::RowMajor;
    using opposite_tail_matrix = fdapde::Matrix<bool, 5, 13, OppositeOrder>;
    using opposite_layout_matrix = fdapde::Matrix<bool, 2, 3, OppositeOrder>;

    exact_matrix exact(true);
    EXPECT_TRUE(exact.all());
    EXPECT_TRUE(exact.any());
    EXPECT_EQ(exact.count(), 64);
    exact(7, 7) = false;
    EXPECT_FALSE(exact.all());
    EXPECT_TRUE(exact.any());
    EXPECT_EQ(exact.count(), 63);

    const tail_matrix zeros;
    const tail_matrix logical_ones(std::vector<bool>(65, true));
    const auto inverted_zeros = ~zeros;
    const auto hidden_only = ~logical_ones;
    EXPECT_TRUE(inverted_zeros.all());
    EXPECT_TRUE(inverted_zeros.any());
    EXPECT_EQ(inverted_zeros.count(), 65);
    EXPECT_TRUE(logical_ones.all());
    EXPECT_EQ(logical_ones.count(), 65);
    EXPECT_FALSE(hidden_only.all());
    EXPECT_FALSE(hidden_only.any());
    EXPECT_EQ(hidden_only.count(), 0);
    EXPECT_TRUE(inverted_zeros == logical_ones);
    tail_matrix missing_last(logical_ones);
    missing_last(4, 12) = false;
    EXPECT_FALSE(inverted_zeros == missing_last);

    std::vector<bool> tail_values(65);
    for (const int index : {1, 12, 13, 51, 64}) {
        tail_values[static_cast<std::size_t>(index)] = true;
    }
    const tail_matrix tail_layout(tail_values);
    opposite_tail_matrix other_tail_layout(tail_values);
    EXPECT_TRUE(tail_layout == other_tail_layout);
    EXPECT_FALSE(tail_layout != other_tail_layout);
    const auto stored_tail_expression = [&tail_layout, &zeros] { return tail_layout | zeros; }();
    EXPECT_TRUE(stored_tail_expression == other_tail_layout);
    other_tail_layout(3, 12) = !bool(other_tail_layout(3, 12));
    EXPECT_FALSE(tail_layout == other_tail_layout);
    EXPECT_TRUE(tail_layout != other_tail_layout);

    const std::array<bool, 6> layout_values {false, true, true, true, false, false};
    const layout_matrix layout(std::vector<bool>(layout_values.begin(), layout_values.end()));
    const opposite_layout_matrix other_layout(
      std::vector<bool>(layout_values.begin(), layout_values.end()));
    EXPECT_TRUE(layout == other_layout);
    EXPECT_EQ(layout.which(true), (std::vector<int> {1, 2, 3}));
    EXPECT_EQ(layout.which(false), (std::vector<int> {0, 4, 5}));
    EXPECT_EQ(fdapde::which(layout), (std::vector<int> {1, 2, 3}));

    const layout_matrix layout_zeros;
    const auto stored_layout_expression = [&layout, &layout_zeros] { return layout | layout_zeros; }();
    EXPECT_EQ(stored_layout_expression.which(true), (std::vector<int> {1, 2, 3}));
    EXPECT_EQ(fdapde::which(stored_layout_expression), (std::vector<int> {1, 2, 3}));

    const dynamic_matrix empty;
    const dynamic_matrix zero_rows(0, 3);
    const dynamic_matrix zero_cols(3, 0);
    const dynamic_matrix mismatched_rows(2, 3);
    const dynamic_matrix mismatched_cols(3, 2);
    EXPECT_THROW(static_cast<void>(mismatched_rows == mismatched_cols), std::invalid_argument);
    EXPECT_TRUE(empty.which(true).empty());
    EXPECT_TRUE(empty.which(false).empty());
    EXPECT_TRUE(fdapde::which(empty).empty());
    EXPECT_TRUE(zero_rows.which(true).empty());
    EXPECT_TRUE(zero_cols.which(false).empty());
}

template <int StorageOrder> void check_boolean_view_contracts() {
    using fixed_view = fdapde::MatrixView<bool, 2, 3, StorageOrder>;
    using dynamic_view =
      fdapde::MatrixView<bool, fdapde::Dynamic, fdapde::Dynamic, StorageOrder>;
    using const_dynamic_view =
      fdapde::MatrixView<const bool, fdapde::Dynamic, fdapde::Dynamic, StorageOrder>;
    using partial_view = fdapde::MatrixView<bool, 5, fdapde::Dynamic, StorageOrder>;
    using row_view = fdapde::MatrixView<bool, 1, fdapde::Dynamic, StorageOrder>;
    using column_view = fdapde::MatrixView<bool, fdapde::Dynamic, 1, StorageOrder>;
    using fixed_column_view = fdapde::MatrixView<bool, 3, 1, StorageOrder>;
    using bitpack_t = typename fixed_view::bitpack_t;
    constexpr int OppositeOrder = StorageOrder == fdapde::RowMajor ? fdapde::ColMajor : fdapde::RowMajor;
    using opposite_view = fdapde::MatrixView<bool, 2, 3, OppositeOrder>;

    const bitpack_t outside_canary = bitpack_t(0x5a5a);
    const bitpack_t hidden_canary = bitpack_t(1) << (fixed_view::PackSize - 1);
    std::array<bitpack_t, 2> storage {bitpack_t(0), outside_canary};
    fixed_view view(storage.data());
    view(1, 0) = true;
    constexpr int mapped_bit = StorageOrder == fdapde::RowMajor ? 3 : 1;
    EXPECT_NE(storage[0] & (bitpack_t(1) << mapped_bit), bitpack_t(0));
    EXPECT_EQ(storage[1], outside_canary);
    const auto& const_view_handle = view;
    EXPECT_TRUE(bool(const_view_handle(1, 0)));

    const fixed_view shallow_alias(view);
    EXPECT_EQ(shallow_alias.data(), view.data());

    std::array<bitpack_t, 1> source_storage {bitpack_t(0)};
    fixed_view source(source_storage.data());
    source(0, 1) = true;
    source(1, 2) = true;
    storage[0] = hidden_canary;
    bitpack_t* const binding = view.data();
    view = source;
    EXPECT_EQ(view.data(), binding);
    EXPECT_TRUE(bool(view(0, 1)));
    EXPECT_TRUE(bool(view(1, 2)));
    EXPECT_EQ(storage[0] & hidden_canary, hidden_canary);

    const fdapde::Matrix<bool, 2, 3, StorageOrder> owner(
      std::vector<bool> {true, false, true, false, true, false});
    const fdapde::Matrix<bool, 2, 3, StorageOrder> owner_zero;
    view = owner | owner_zero;
    EXPECT_EQ(view.data(), binding);
    for (int i = 0; i < owner.rows(); ++i) {
        for (int j = 0; j < owner.cols(); ++j) { EXPECT_EQ(bool(view(i, j)), bool(owner(i, j))); }
    }
    EXPECT_EQ(storage[0] & hidden_canary, hidden_canary);

    std::array<bitpack_t, 1> opposite_storage {bitpack_t(0)};
    opposite_view other(opposite_storage.data());
    other(0, 0) = true;
    other(1, 1) = true;
    view = other;
    EXPECT_TRUE(bool(view(0, 0)));
    EXPECT_TRUE(bool(view(1, 1)));
    EXPECT_FALSE(bool(view(0, 1)));

    std::array<bitpack_t, 1> temporary_storage {bitpack_t(0)};
    const auto assigned_temporary = fixed_view(temporary_storage.data()) = source;
    EXPECT_EQ(assigned_temporary.data(), temporary_storage.data());
    EXPECT_TRUE(bool(assigned_temporary(0, 1)));
    EXPECT_TRUE(bool(assigned_temporary(1, 2)));

    const auto stored_expression = [&source_storage] { return ~fixed_view(source_storage.data()); }();
    const fdapde::Matrix<bool, 2, 3, StorageOrder> stored_result(stored_expression);
    for (int i = 0; i < source.rows(); ++i) {
        for (int j = 0; j < source.cols(); ++j) {
            EXPECT_EQ(bool(stored_result(i, j)), !bool(source(i, j)));
        }
    }

    std::array<bitpack_t, 3> tail_storage {bitpack_t(0), hidden_canary, outside_canary};
    dynamic_view tail(tail_storage.data(), 5, 13);
    partial_view partial(tail_storage.data(), 5, 13);
    EXPECT_EQ(tail.bitpacks(), 2);
    EXPECT_EQ(partial.bitpacks(), 2);
    tail(4, 12) = true;
    EXPECT_EQ(tail.bitpack(1), bitpack_t(1));
    EXPECT_TRUE(bool(partial(4, 12)));
    EXPECT_EQ(tail_storage[2], outside_canary);
    tail.set();
    EXPECT_EQ(tail.bitpack(0), std::numeric_limits<bitpack_t>::max());
    EXPECT_EQ(tail.bitpack(1), bitpack_t(1));
    EXPECT_EQ(tail_storage[1] & hidden_canary, hidden_canary);
    EXPECT_EQ(tail_storage[2], outside_canary);
    tail.clear();
    EXPECT_EQ(tail.bitpack(0), bitpack_t(0));
    EXPECT_EQ(tail.bitpack(1), bitpack_t(0));
    EXPECT_EQ(tail_storage[1] & hidden_canary, hidden_canary);
    EXPECT_EQ(tail_storage[2], outside_canary);
    const const_dynamic_view read_only_tail(tail_storage.data(), 5, 13);
    EXPECT_EQ(read_only_tail.data(), tail_storage.data());
    EXPECT_FALSE(bool(read_only_tail(4, 12)));

    std::array<bitpack_t, 1> vector_storage {bitpack_t(0)};
    row_view row(vector_storage.data(), 5);
    column_view column(vector_storage.data(), 5);
    row[4] = true;
    EXPECT_TRUE(bool(column[4]));
    column[4] = false;
    EXPECT_FALSE(bool(row[4]));

    dynamic_view empty;
    partial_view partial_empty;
    const_dynamic_view const_empty;
    EXPECT_EQ(empty.rows(), 0);
    EXPECT_EQ(empty.cols(), 0);
    EXPECT_EQ(empty.data(), nullptr);
    EXPECT_EQ(empty.bitpacks(), 0);
    EXPECT_EQ(partial_empty.rows(), 5);
    EXPECT_EQ(partial_empty.cols(), 0);
    EXPECT_EQ(partial_empty.data(), nullptr);
    EXPECT_EQ(partial_empty.bitpacks(), 0);
    EXPECT_EQ(const_empty.data(), nullptr);
    empty.set();
    empty.clear();
    partial_empty.set();
    partial_empty.clear();

    EXPECT_THROW(static_cast<void>(dynamic_view(tail_storage.data(), -1, 2)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(dynamic_view(tail_storage.data(), 0, 2)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(dynamic_view(tail_storage.data(), 2, 0)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(partial_view(tail_storage.data(), 4, 13)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(row_view(vector_storage.data(), -1)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(row_view(vector_storage.data(), 0)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(fixed_column_view(vector_storage.data(), 2)), std::invalid_argument);
    EXPECT_THROW(
      static_cast<void>(dynamic_view(tail_storage.data(), std::numeric_limits<int>::max(), 2)),
      std::length_error);

    std::array<bitpack_t, 1> mismatch_destination_storage {hidden_canary};
    std::array<bitpack_t, 1> mismatch_source_storage {bitpack_t(3)};
    dynamic_view mismatch_destination(mismatch_destination_storage.data(), 2, 2);
    dynamic_view mismatch_source(mismatch_source_storage.data(), 2, 3);
    bitpack_t* const mismatch_binding = mismatch_destination.data();
    const bitpack_t mismatch_snapshot = mismatch_destination_storage[0];
    EXPECT_THROW(mismatch_destination = mismatch_source, std::invalid_argument);
    EXPECT_EQ(mismatch_destination.data(), mismatch_binding);
    EXPECT_EQ(mismatch_destination.rows(), 2);
    EXPECT_EQ(mismatch_destination.cols(), 2);
    EXPECT_EQ(mismatch_destination_storage[0], mismatch_snapshot);

    const bitpack_t bounds_snapshot = storage[0];
    EXPECT_THROW(static_cast<void>(view(-1, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(view(2, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_view_handle(0, -1)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(const_view_handle(0, 3)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(row[-1]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(row[row.size()]), std::out_of_range);
    EXPECT_THROW(static_cast<void>(tail.bitpack(-1)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(tail.bitpack(tail.bitpacks())), std::out_of_range);
    EXPECT_EQ(storage[0], bounds_snapshot);
}

}   // namespace

TEST(linear_algebra, boolean) {
    fdapde::Matrix<bool, 2, 3> fixed({true, false, true, false, true, false});
    EXPECT_EQ(fixed.rows(), 2);
    EXPECT_EQ(fixed.cols(), 3);
    EXPECT_TRUE(fixed(0, 0));
    EXPECT_FALSE(fixed(0, 1));

    fixed(1, 0).set();
    fixed(0, 2).clear();
    EXPECT_TRUE(fixed(1, 0));
    EXPECT_FALSE(fixed(0, 2));

    using dynamic_bool_matrix = fdapde::Matrix<bool, fdapde::Dynamic, fdapde::Dynamic>;
    constexpr int pack_size = static_cast<int>(dynamic_bool_matrix::PackSize);
    dynamic_bool_matrix dynamic(1, pack_size + 1);
    dynamic(0, pack_size - 1) = true;
    dynamic(0, pack_size) = true;

    dynamic_bool_matrix copy = dynamic;
    EXPECT_TRUE(copy(0, pack_size - 1));
    EXPECT_TRUE(copy(0, pack_size));
    copy(0, pack_size - 1).clear();
    EXPECT_FALSE(copy(0, pack_size - 1));
    EXPECT_TRUE(copy(0, pack_size));
    EXPECT_TRUE(dynamic(0, pack_size - 1));

    check_exact_boolean_pack_accounting<fdapde::RowMajor>();
    check_exact_boolean_pack_accounting<fdapde::ColMajor>();
    check_boolean_owner_contracts<fdapde::RowMajor>();
    check_boolean_owner_contracts<fdapde::ColMajor>();
    check_boolean_expression_contracts<fdapde::RowMajor>();
    check_boolean_expression_contracts<fdapde::ColMajor>();
    check_boolean_block_contracts<fdapde::RowMajor>();
    check_boolean_block_contracts<fdapde::ColMajor>();
    check_boolean_reshape_contracts<fdapde::RowMajor>();
    check_boolean_reshape_contracts<fdapde::ColMajor>();
    check_boolean_selection_contracts<fdapde::RowMajor>();
    check_boolean_selection_contracts<fdapde::ColMajor>();
    check_boolean_repeat_contracts<fdapde::RowMajor>();
    check_boolean_repeat_contracts<fdapde::ColMajor>();
    check_boolean_terminal_contracts<fdapde::RowMajor>();
    check_boolean_terminal_contracts<fdapde::ColMajor>();
    check_boolean_view_contracts<fdapde::RowMajor>();
    check_boolean_view_contracts<fdapde::ColMajor>();

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const std::vector<double> nan_values {nan, 1.0, nan};
    const auto nan_mask = fdapde::nan_indicator(nan_values);
    EXPECT_EQ(nan_mask.rows(), 3);
    EXPECT_EQ(nan_mask.cols(), 1);
    EXPECT_TRUE(bool(nan_mask(0, 0)));
    EXPECT_FALSE(bool(nan_mask(1, 0)));
    EXPECT_TRUE(bool(nan_mask(2, 0)));

    const std::vector<int> markers {2, 1, 2, 3};
    const auto marker_mask = fdapde::value_indicator(markers.cbegin(), markers.cend(), 2);
    EXPECT_EQ(marker_mask.rows(), 4);
    EXPECT_EQ(marker_mask.cols(), 1);
    EXPECT_TRUE(bool(marker_mask[0]));
    EXPECT_FALSE(bool(marker_mask[1]));
    EXPECT_TRUE(bool(marker_mask[2]));
    EXPECT_FALSE(bool(marker_mask[3]));
    const auto empty_marker_mask = fdapde::value_indicator(markers.cend(), markers.cend(), 2);
    EXPECT_EQ(empty_marker_mask.rows(), 0);
    EXPECT_EQ(empty_marker_mask.cols(), 1);
}

// Current regression adapted from 86ff6d12:tests/linear_algebra/bool.cpp.
// Stable source: a2a9c88:test/src/binary_matrix_test.cpp.
// Stable declarations (9): static_sized_matrix, dynamic_sized_matrix, binary_vector, block_operations,
// binary_expresssions, visitors, block_repeat, eigen_assignment_and_construct, and reshaped.
// TODO(P4-B): cover the remaining two-dimensional resize policy.
// Replace the historical Eigen assignment/construct assertion with native numeric-matrix conversion; do not
// restore an implicit Eigen bridge.
