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

// zero logical bits require no packed storage words
static_assert(fdapde::internals::bitpack_count(0, 64) == 0);
// one logical bit requires one packed storage word
static_assert(fdapde::internals::bitpack_count(1, 64) == 1);
// sixty-three logical bits fit in one sixty-four-bit word
static_assert(fdapde::internals::bitpack_count(63, 64) == 1);
// an exact word boundary does not allocate an extra word
static_assert(fdapde::internals::bitpack_count(64, 64) == 1);
// one bit beyond a word boundary requires a second word
static_assert(fdapde::internals::bitpack_count(65, 64) == 2);
// packed-word counting rounds up the maximum supported bit count without overflow
static_assert(fdapde::internals::bitpack_count(std::numeric_limits<int>::max(), 64) == 33'554'432);

template <typename Lhs, typename Rhs, typename Operation>
concept permits_boolean_binary =
  requires(Lhs&& lhs, Rhs&& rhs, Operation operation) { operation(std::forward<Lhs>(lhs), std::forward<Rhs>(rhs)); };

template <typename Matrix>
concept permits_boolean_negation = requires(Matrix&& matrix) { ~std::forward<Matrix>(matrix); };

template <typename Matrix>
concept exposes_boolean_rvalue_derived = requires(Matrix&& matrix) { std::move(matrix).derived(); };

template <typename Matrix>
concept permits_temporary_boolean_assignment =
  requires(Matrix&& matrix, const Matrix& rhs) { std::move(matrix) = rhs; };

template <typename Matrix>
concept permits_temporary_boolean_and_assignment =
  requires(Matrix&& matrix, const Matrix& rhs) { std::move(matrix) &= rhs; };

template <typename Matrix>
concept permits_temporary_boolean_or_assignment =
  requires(Matrix&& matrix, const Matrix& rhs) { std::move(matrix) |= rhs; };

template <typename Matrix>
concept permits_temporary_boolean_xor_assignment =
  requires(Matrix&& matrix, const Matrix& rhs) { std::move(matrix) ^= rhs; };

template <typename Matrix>
concept permits_named_boolean_chain = requires(Matrix& lhs, Matrix& rhs) { ~((lhs | rhs) ^ (lhs & rhs)); };

using boolean_mask = fdapde::Matrix<bool, 2, 2>;
using boolean_unary_node = fdapde::BoolMatrixBitWiseOp<boolean_mask, std::logical_not<>, std::bit_not<>>;
using boolean_binary_node = fdapde::BoolMatrixBinOp<boolean_mask, boolean_mask, std::bit_and<>, std::bit_and<>>;

// boolean AND can borrow two named owners
static_assert(permits_boolean_binary<boolean_mask&, boolean_mask&, std::bit_and<>>);
// boolean AND rejects a temporary left owner that would leave a dangling operand
static_assert(!permits_boolean_binary<boolean_mask, boolean_mask&, std::bit_and<>>);
// boolean AND rejects a temporary right owner that would leave a dangling operand
static_assert(!permits_boolean_binary<boolean_mask&, boolean_mask, std::bit_and<>>);
// boolean OR rejects a temporary left owner that would leave a dangling operand
static_assert(!permits_boolean_binary<boolean_mask, boolean_mask&, std::bit_or<>>);
// boolean OR rejects a temporary right owner that would leave a dangling operand
static_assert(!permits_boolean_binary<boolean_mask&, boolean_mask, std::bit_or<>>);
// boolean XOR rejects a temporary left owner that would leave a dangling operand
static_assert(!permits_boolean_binary<boolean_mask, boolean_mask&, std::bit_xor<>>);
// boolean XOR rejects a temporary right owner that would leave a dangling operand
static_assert(!permits_boolean_binary<boolean_mask&, boolean_mask, std::bit_xor<>>);
// boolean negation can borrow a named owner
static_assert(permits_boolean_negation<boolean_mask&>);
// boolean negation rejects a temporary owner that would leave a dangling operand
static_assert(!permits_boolean_negation<boolean_mask>);
// a temporary Boolean owner cannot expose a dangling derived reference
static_assert(!exposes_boolean_rvalue_derived<boolean_mask>);
// a temporary Boolean owner cannot return a borrow through assignment
static_assert(!permits_temporary_boolean_assignment<boolean_mask>);
// boolean AND assignment cannot return a borrow into a temporary Boolean owner
static_assert(!permits_temporary_boolean_and_assignment<boolean_mask>);
// boolean OR assignment cannot return a borrow into a temporary Boolean owner
static_assert(!permits_temporary_boolean_or_assignment<boolean_mask>);
// exclusive OR assignment cannot return a borrow into a temporary Boolean owner
static_assert(!permits_temporary_boolean_xor_assignment<boolean_mask>);
// chained Boolean expressions can safely borrow named owners
static_assert(permits_named_boolean_chain<boolean_mask>);
// a unary Boolean node can bind a named const owner
static_assert(std::is_constructible_v<boolean_unary_node, const boolean_mask&, std::logical_not<>, std::bit_not<>>);
// direct unary-node construction cannot bypass temporary-owner lifetime protection
static_assert(!std::is_constructible_v<boolean_unary_node, boolean_mask&&, std::logical_not<>, std::bit_not<>>);
// a binary Boolean node can bind two named const owners
static_assert(std::is_constructible_v<
              boolean_binary_node, const boolean_mask&, const boolean_mask&, std::bit_and<>, std::bit_and<>>);
// direct binary-node construction rejects a temporary left owner
static_assert(
  !std::is_constructible_v<boolean_binary_node, boolean_mask&&, const boolean_mask&, std::bit_and<>, std::bit_and<>>);
// direct binary-node construction rejects a temporary right owner
static_assert(
  !std::is_constructible_v<boolean_binary_node, const boolean_mask&, boolean_mask&&, std::bit_and<>, std::bit_and<>>);
// the Boolean matrix trait recognizes an owner reference
static_assert(fdapde::is_boolean_matrix_v<boolean_mask&>);
// the Boolean matrix trait recognizes a const owner reference
static_assert(fdapde::is_boolean_matrix_v<const boolean_mask&>);
// the Boolean vector trait recognizes a const reference to a Boolean row vector
static_assert(fdapde::is_boolean_vector_v<const fdapde::Matrix<bool, 1, 2>&>);
// the Boolean vector trait rejects an unrelated scalar type
static_assert(!fdapde::is_boolean_vector_v<int>);

template <typename Condition, typename TrueXpr, typename FalseXpr>
concept permits_boolean_selection = requires(Condition&& condition, TrueXpr&& true_xpr, FalseXpr&& false_xpr) {
    std::forward<Condition>(condition).select(std::forward<TrueXpr>(true_xpr), std::forward<FalseXpr>(false_xpr));
};

using selection_mask = fdapde::Matrix<bool, 2, 2>;
using selection_values = fdapde::Matrix<double, 2, 2>;
using selection_condition_expression = decltype(~std::declval<selection_mask&>());
using selection_value_expression = decltype(std::declval<selection_values&>() + std::declval<selection_values&>());
using selection_mask_view = fdapde::MatrixView<const bool, 2, 2>;
using selection_value_view = fdapde::MatrixView<const double, 2, 2>;
using direct_ternary = fdapde::TernaryOp<selection_mask, selection_values, selection_values>;
using safe_selection = decltype(std::declval<selection_condition_expression>().select(
  std::declval<selection_value_expression>(), std::declval<selection_value_expression>()));

// selection can borrow a named mask and two named value owners
static_assert(permits_boolean_selection<selection_mask&, selection_values&, selection_values&>);
// selection rejects a temporary mask owner
static_assert(!permits_boolean_selection<selection_mask, selection_values&, selection_values&>);
// selection rejects a const temporary mask owner
static_assert(!permits_boolean_selection<const selection_mask, selection_values&, selection_values&>);
// selection rejects a temporary true-branch owner
static_assert(!permits_boolean_selection<selection_mask&, selection_values, selection_values&>);
// selection rejects a temporary false-branch owner
static_assert(!permits_boolean_selection<selection_mask&, selection_values&, selection_values>);
// selection can store temporary expression nodes whose owners remain alive
static_assert(
  permits_boolean_selection<selection_condition_expression, selection_value_expression, selection_value_expression>);
// selection can store temporary views while borrowing a named value owner
static_assert(permits_boolean_selection<selection_mask_view, selection_value_view, selection_values&>);
// a selection expression exposes read-only coefficients
static_assert(safe_selection::ReadOnly == 1);
// selection retains its specified row-major expression layout
static_assert(safe_selection::StorageOrder == fdapde::RowMajor);
// selection preserves the value branches' double scalar type
static_assert(std::same_as<typename safe_selection::Scalar, double>);
// a ternary node can bind three named const owners
static_assert(
  std::is_constructible_v<direct_ternary, const selection_mask&, const selection_values&, const selection_values&>);
// direct ternary-node construction rejects a temporary condition owner
static_assert(
  !std::is_constructible_v<direct_ternary, selection_mask&&, const selection_values&, const selection_values&>);
// direct ternary-node construction rejects a temporary true-branch owner
static_assert(
  !std::is_constructible_v<direct_ternary, const selection_mask&, selection_values&&, const selection_values&>);
// direct ternary-node construction rejects a temporary false-branch owner
static_assert(
  !std::is_constructible_v<direct_ternary, const selection_mask&, const selection_values&, selection_values&&>);

template <typename Matrix>
concept exposes_boolean_repeat = requires(Matrix&& matrix) { std::forward<Matrix>(matrix).repeat(2, 3); };

using repeat_owner = fdapde::Matrix<bool, 2, 3>;
using repeat_expression = decltype(~std::declval<repeat_owner&>());
using repeat_view = fdapde::MatrixView<const bool, 2, 3>;
using repeat_result = decltype(std::declval<const repeat_owner&>().repeat(2, 3));
using direct_repeat = fdapde::BoolMatrixRepeatOp<repeat_owner>;

// repeat can borrow a named Boolean owner
static_assert(exposes_boolean_repeat<repeat_owner&>);
// repeat can borrow a named const Boolean owner
static_assert(exposes_boolean_repeat<const repeat_owner&>);
// repeat rejects a temporary owner that would leave a dangling tile
static_assert(!exposes_boolean_repeat<repeat_owner>);
// repeat rejects a const temporary owner that would leave a dangling tile
static_assert(!exposes_boolean_repeat<const repeat_owner>);
// repeat can store a temporary Boolean expression node
static_assert(exposes_boolean_repeat<repeat_expression>);
// repeat can store a temporary view with externally owned storage
static_assert(exposes_boolean_repeat<repeat_view>);
// runtime row repetition produces a dynamic row extent
static_assert(repeat_result::Rows == fdapde::Dynamic);
// runtime column repetition produces a dynamic column extent
static_assert(repeat_result::Cols == fdapde::Dynamic);
// the repeated expression uses its declared row-major layout
static_assert(repeat_result::StorageOrder == fdapde::RowMajor);
// repeat exposes read-only coefficients
static_assert(repeat_result::ReadOnly == 1);
// repeat nodes are nested by value in later expressions
static_assert(repeat_result::NestAsRef == 0);
// repeat retains Boolean scalar coefficients
static_assert(std::same_as<typename repeat_result::Scalar, bool>);
// a repeat node can bind a named const owner and runtime repetition counts
static_assert(std::is_constructible_v<direct_repeat, const repeat_owner&, int, int>);
// direct repeat-node construction rejects a temporary owner
static_assert(!std::is_constructible_v<direct_repeat, repeat_owner&&, int, int>);

template <typename Matrix>
concept exposes_static_boolean_block =
  requires(Matrix&& matrix) { std::forward<Matrix>(matrix).template block<1, 2>(0, 0); };

template <typename Matrix>
concept exposes_dynamic_boolean_block = requires(Matrix&& matrix) { std::forward<Matrix>(matrix).block(0, 0, 1, 2); };

template <typename Matrix>
concept exposes_boolean_row = requires(Matrix&& matrix) { std::forward<Matrix>(matrix).row(0); };

template <typename Matrix>
concept exposes_boolean_col = requires(Matrix&& matrix) { std::forward<Matrix>(matrix).col(0); };

template <typename Matrix>
concept exposes_static_boolean_top_rows =
  requires(Matrix&& matrix) { std::forward<Matrix>(matrix).template top_rows<1>(); };

template <typename Matrix>
concept exposes_dynamic_boolean_top_rows = requires(Matrix&& matrix) { std::forward<Matrix>(matrix).top_rows(1); };

template <typename Matrix>
concept exposes_static_boolean_bottom_rows =
  requires(Matrix&& matrix) { std::forward<Matrix>(matrix).template bottom_rows<1>(); };

template <typename Matrix>
concept exposes_dynamic_boolean_bottom_rows =
  requires(Matrix&& matrix) { std::forward<Matrix>(matrix).bottom_rows(1); };

template <typename Matrix>
concept exposes_static_boolean_left_cols =
  requires(Matrix&& matrix) { std::forward<Matrix>(matrix).template left_cols<1>(); };

template <typename Matrix>
concept exposes_dynamic_boolean_left_cols = requires(Matrix&& matrix) { std::forward<Matrix>(matrix).left_cols(1); };

template <typename Matrix>
concept exposes_static_boolean_right_cols =
  requires(Matrix&& matrix) { std::forward<Matrix>(matrix).template right_cols<1>(); };

template <typename Matrix>
concept exposes_dynamic_boolean_right_cols = requires(Matrix&& matrix) { std::forward<Matrix>(matrix).right_cols(1); };

template <typename Block>
concept permits_boolean_block_coordinate_assignment = requires(Block& block) { block(0, 0) = true; };

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
using mutable_boolean_block = decltype(std::declval<boolean_block_owner&>().template block<1, 2>(0, 0));
using const_boolean_block = decltype(std::declval<const boolean_block_owner&>().template block<1, 2>(0, 0));
using mutable_boolean_row = decltype(std::declval<boolean_block_owner&>().row(0));
using const_boolean_row = decltype(std::declval<const boolean_block_owner&>().row(0));
using safe_boolean_block_expression =
  decltype(std::declval<boolean_block_owner&>() | std::declval<boolean_block_owner&>());
using safe_boolean_expression_block =
  decltype(std::declval<safe_boolean_block_expression>().template block<1, 2>(0, 0));

// a named Boolean owner exposes a block with static extents
static_assert(exposes_static_boolean_block<boolean_block_owner&>);
// a named Boolean owner exposes a block with runtime extents
static_assert(exposes_dynamic_boolean_block<boolean_block_owner&>);
// a named Boolean owner exposes a row view
static_assert(exposes_boolean_row<boolean_block_owner&>);
// a named Boolean owner exposes a column view
static_assert(exposes_boolean_col<boolean_block_owner&>);
// a named Boolean owner exposes a static prefix of rows
static_assert(exposes_static_boolean_top_rows<boolean_block_owner&>);
// a named Boolean owner exposes a runtime prefix of rows
static_assert(exposes_dynamic_boolean_top_rows<boolean_block_owner&>);
// a named Boolean owner exposes a static suffix of rows
static_assert(exposes_static_boolean_bottom_rows<boolean_block_owner&>);
// a named Boolean owner exposes a runtime suffix of rows
static_assert(exposes_dynamic_boolean_bottom_rows<boolean_block_owner&>);
// a named Boolean owner exposes a static prefix of columns
static_assert(exposes_static_boolean_left_cols<boolean_block_owner&>);
// a named Boolean owner exposes a runtime prefix of columns
static_assert(exposes_dynamic_boolean_left_cols<boolean_block_owner&>);
// a named Boolean owner exposes a static suffix of columns
static_assert(exposes_static_boolean_right_cols<boolean_block_owner&>);
// a named Boolean owner exposes a runtime suffix of columns
static_assert(exposes_dynamic_boolean_right_cols<boolean_block_owner&>);
// a temporary Boolean owner cannot lend any block, row, column or edge view
static_assert(
  !exposes_static_boolean_block<boolean_block_owner> && !exposes_dynamic_boolean_block<boolean_block_owner> &&
  !exposes_boolean_row<boolean_block_owner> && !exposes_boolean_col<boolean_block_owner> &&
  !exposes_static_boolean_top_rows<boolean_block_owner> && !exposes_dynamic_boolean_top_rows<boolean_block_owner> &&
  !exposes_static_boolean_bottom_rows<boolean_block_owner> &&
  !exposes_dynamic_boolean_bottom_rows<boolean_block_owner> && !exposes_static_boolean_left_cols<boolean_block_owner> &&
  !exposes_dynamic_boolean_left_cols<boolean_block_owner> && !exposes_static_boolean_right_cols<boolean_block_owner> &&
  !exposes_dynamic_boolean_right_cols<boolean_block_owner>);
// a const temporary Boolean owner cannot lend any block, row, column or edge view
static_assert(
  !exposes_static_boolean_block<const boolean_block_owner> &&
  !exposes_dynamic_boolean_block<const boolean_block_owner> && !exposes_boolean_row<const boolean_block_owner> &&
  !exposes_boolean_col<const boolean_block_owner> && !exposes_static_boolean_top_rows<const boolean_block_owner> &&
  !exposes_dynamic_boolean_top_rows<const boolean_block_owner> &&
  !exposes_static_boolean_bottom_rows<const boolean_block_owner> &&
  !exposes_dynamic_boolean_bottom_rows<const boolean_block_owner> &&
  !exposes_static_boolean_left_cols<const boolean_block_owner> &&
  !exposes_dynamic_boolean_left_cols<const boolean_block_owner> &&
  !exposes_static_boolean_right_cols<const boolean_block_owner> &&
  !exposes_dynamic_boolean_right_cols<const boolean_block_owner>);
// a temporary expression can retain every block accessor while borrowing live operands
static_assert(
  exposes_static_boolean_block<safe_boolean_block_expression> &&
  exposes_dynamic_boolean_block<safe_boolean_block_expression> && exposes_boolean_row<safe_boolean_block_expression> &&
  exposes_boolean_col<safe_boolean_block_expression> &&
  exposes_static_boolean_top_rows<safe_boolean_block_expression> &&
  exposes_dynamic_boolean_top_rows<safe_boolean_block_expression> &&
  exposes_static_boolean_bottom_rows<safe_boolean_block_expression> &&
  exposes_dynamic_boolean_bottom_rows<safe_boolean_block_expression> &&
  exposes_static_boolean_left_cols<safe_boolean_block_expression> &&
  exposes_dynamic_boolean_left_cols<safe_boolean_block_expression> &&
  exposes_static_boolean_right_cols<safe_boolean_block_expression> &&
  exposes_dynamic_boolean_right_cols<safe_boolean_block_expression>);
// a block of a computed Boolean expression is read-only
static_assert(safe_boolean_expression_block::ReadOnly == 1);

// a block of a mutable Boolean owner permits writes
static_assert(mutable_boolean_block::ReadOnly == 0);
// a block of a const Boolean owner is read-only
static_assert(const_boolean_block::ReadOnly == 1);
// a mutable Boolean block permits coordinate assignment
static_assert(permits_boolean_block_coordinate_assignment<mutable_boolean_block>);
// a mutable Boolean row permits vector-index assignment
static_assert(permits_boolean_block_vector_assignment<mutable_boolean_row>);
// a const block object cannot expose writable coordinate proxies
static_assert(!permits_boolean_block_coordinate_assignment<const mutable_boolean_block>);
// a const row object cannot expose writable vector proxies
static_assert(!permits_boolean_block_vector_assignment<const mutable_boolean_row>);
// a block borrowed from a const owner rejects coordinate writes
static_assert(!permits_boolean_block_coordinate_assignment<const_boolean_block>);
// a row borrowed from a const owner rejects vector-index writes
static_assert(!permits_boolean_block_vector_assignment<const_boolean_row>);
// a block borrowed from a const owner rejects bulk mutation
static_assert(!permits_boolean_block_mutation<const_boolean_block>);

using direct_boolean_row_block = fdapde::BoolMatrixBlock<1, boolean_block_owner::Cols, const boolean_block_owner>;
using direct_static_boolean_block = fdapde::BoolMatrixBlock<1, 2, const boolean_block_owner>;
using direct_dynamic_boolean_block =
  fdapde::BoolMatrixBlock<fdapde::Dynamic, fdapde::Dynamic, const boolean_block_owner>;
// direct row-block construction cannot borrow a temporary Boolean owner
static_assert(!std::is_constructible_v<direct_boolean_row_block, boolean_block_owner&&, int>);
// direct static-block construction cannot borrow a temporary Boolean owner
static_assert(!std::is_constructible_v<direct_static_boolean_block, boolean_block_owner&&, int, int>);
// direct runtime-block construction cannot borrow a temporary Boolean owner
static_assert(!std::is_constructible_v<direct_dynamic_boolean_block, boolean_block_owner&&, int, int, int, int>);

template <typename Matrix>
concept exposes_static_boolean_matrix_reshape =
  requires(Matrix&& matrix) { std::forward<Matrix>(matrix).template reshape<2, 2>(); };

template <typename Matrix>
concept exposes_static_boolean_vector_reshape =
  requires(Matrix&& matrix) { std::forward<Matrix>(matrix).template reshape<4>(); };

template <typename Matrix>
concept exposes_dynamic_boolean_matrix_reshape =
  requires(Matrix&& matrix) { std::forward<Matrix>(matrix).reshape(2, 2); };

template <typename Matrix>
concept exposes_dynamic_boolean_vector_reshape = requires(Matrix&& matrix) { std::forward<Matrix>(matrix).reshape(4); };

template <typename Matrix, int ExpectedReadOnly>
concept exposes_all_safe_boolean_reshape_accessors = requires(Matrix&& matrix) {
    requires(decltype(std::forward<Matrix>(matrix).template reshape<2, 2>())::ReadOnly == ExpectedReadOnly);
    requires(decltype(std::forward<Matrix>(matrix).template reshape<4>())::ReadOnly == ExpectedReadOnly);
    requires(decltype(std::forward<Matrix>(matrix).reshape(2, 2))::ReadOnly == ExpectedReadOnly);
    requires(decltype(std::forward<Matrix>(matrix).reshape(4))::ReadOnly == ExpectedReadOnly);
};

template <typename Matrix>
concept permits_direct_temporary_static_boolean_reshape =
  requires(Matrix&& matrix) { fdapde::BoolReshapeOp<2, 2, Matrix> {std::move(matrix)}; };

template <typename Matrix>
concept permits_direct_temporary_dynamic_boolean_reshape = requires(Matrix&& matrix) {
    fdapde::BoolReshapeOp<fdapde::Dynamic, fdapde::Dynamic, Matrix> {std::move(matrix), 2, 2};
};

template <typename Matrix>
concept permits_direct_temporary_vector_boolean_reshape =
  requires(Matrix&& matrix) { fdapde::BoolReshapeOp<fdapde::Dynamic, 1, Matrix> {std::move(matrix), 4}; };

template <typename Reshape>
concept permits_boolean_reshape_coordinate_write = requires(Reshape& reshape) { reshape(0, 0) = true; };

template <typename Reshape>
concept permits_boolean_reshape_vector_write = requires(Reshape& reshape) { reshape[0] = true; };

template <typename Reshape>
concept permits_boolean_reshape_assignment = requires(Reshape& lhs, const Reshape& rhs) { lhs = rhs; };

using boolean_reshape_owner = fdapde::Matrix<bool, 2, 2>;
using mutable_boolean_matrix_reshape = decltype(std::declval<boolean_reshape_owner&>().template reshape<2, 2>());
using mutable_boolean_vector_reshape = decltype(std::declval<boolean_reshape_owner&>().template reshape<4>());
using const_owner_boolean_matrix_reshape =
  decltype(std::declval<const boolean_reshape_owner&>().template reshape<2, 2>());
using const_owner_boolean_vector_reshape = decltype(std::declval<const boolean_reshape_owner&>().template reshape<4>());
using safe_boolean_reshape_expression =
  decltype(std::declval<boolean_reshape_owner&>() | std::declval<boolean_reshape_owner&>());

// a named Boolean owner exposes a matrix reshape with static extents
static_assert(exposes_static_boolean_matrix_reshape<boolean_reshape_owner&>);
// a named Boolean owner exposes a vector reshape with static extent
static_assert(exposes_static_boolean_vector_reshape<boolean_reshape_owner&>);
// a named Boolean owner exposes a matrix reshape with runtime extents
static_assert(exposes_dynamic_boolean_matrix_reshape<boolean_reshape_owner&>);
// a named Boolean owner exposes a vector reshape with runtime extent
static_assert(exposes_dynamic_boolean_vector_reshape<boolean_reshape_owner&>);
// a temporary Boolean owner cannot lend any reshape accessor
static_assert(
  !exposes_static_boolean_matrix_reshape<boolean_reshape_owner> &&
  !exposes_static_boolean_vector_reshape<boolean_reshape_owner> &&
  !exposes_dynamic_boolean_matrix_reshape<boolean_reshape_owner> &&
  !exposes_dynamic_boolean_vector_reshape<boolean_reshape_owner>);
// a const temporary Boolean owner cannot lend any reshape accessor
static_assert(
  !exposes_static_boolean_matrix_reshape<const boolean_reshape_owner> &&
  !exposes_static_boolean_vector_reshape<const boolean_reshape_owner> &&
  !exposes_dynamic_boolean_matrix_reshape<const boolean_reshape_owner> &&
  !exposes_dynamic_boolean_vector_reshape<const boolean_reshape_owner>);
// a temporary Boolean expression supports all reshape accessors as read-only expressions
static_assert(exposes_all_safe_boolean_reshape_accessors<safe_boolean_reshape_expression, 1>);
// a const temporary Boolean expression supports all reshape accessors as read-only expressions
static_assert(exposes_all_safe_boolean_reshape_accessors<const safe_boolean_reshape_expression, 1>);
// direct static-reshape construction rejects a temporary Boolean owner
static_assert(!permits_direct_temporary_static_boolean_reshape<boolean_reshape_owner>);
// direct runtime-reshape construction rejects a temporary Boolean owner
static_assert(!permits_direct_temporary_dynamic_boolean_reshape<boolean_reshape_owner>);
// direct vector-reshape construction rejects a temporary Boolean owner
static_assert(!permits_direct_temporary_vector_boolean_reshape<boolean_reshape_owner>);
// direct static-reshape construction rejects a const temporary Boolean owner
static_assert(!permits_direct_temporary_static_boolean_reshape<const boolean_reshape_owner>);
// direct runtime-reshape construction rejects a const temporary Boolean owner
static_assert(!permits_direct_temporary_dynamic_boolean_reshape<const boolean_reshape_owner>);
// direct vector-reshape construction rejects a const temporary Boolean owner
static_assert(!permits_direct_temporary_vector_boolean_reshape<const boolean_reshape_owner>);
// a reshape borrowed from a mutable Boolean owner permits writes
static_assert(mutable_boolean_matrix_reshape::ReadOnly == 0);
// a reshape borrowed from a const Boolean owner is read-only
static_assert(const_owner_boolean_matrix_reshape::ReadOnly == 1);
// a mutable matrix reshape permits coordinate writes
static_assert(permits_boolean_reshape_coordinate_write<mutable_boolean_matrix_reshape>);
// a mutable vector reshape permits vector-index writes
static_assert(permits_boolean_reshape_vector_write<mutable_boolean_vector_reshape>);
// a const matrix-reshape object rejects coordinate writes
static_assert(!permits_boolean_reshape_coordinate_write<const mutable_boolean_matrix_reshape>);
// a const vector-reshape object rejects vector-index writes
static_assert(!permits_boolean_reshape_vector_write<const mutable_boolean_vector_reshape>);
// a matrix reshape borrowed from a const owner rejects coordinate writes
static_assert(!permits_boolean_reshape_coordinate_write<const_owner_boolean_matrix_reshape>);
// a vector reshape borrowed from a const owner rejects vector-index writes
static_assert(!permits_boolean_reshape_vector_write<const_owner_boolean_vector_reshape>);
// a mutable Boolean reshape accepts assignment into its source storage
static_assert(permits_boolean_reshape_assignment<mutable_boolean_matrix_reshape>);
// a reshape borrowed from a const owner rejects assignment
static_assert(!permits_boolean_reshape_assignment<const_owner_boolean_matrix_reshape>);
using temporary_boolean_reshape_assignment_result =
  decltype(std::declval<mutable_boolean_matrix_reshape&&>() = std::declval<const mutable_boolean_matrix_reshape&>());
// assignment to a temporary reshape returns a value rather than a reference to the expired wrapper
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

template <typename View>
concept permits_boolean_view_compound_assignment = requires(View& lhs, const View& rhs) {
    lhs &= rhs;
    lhs |= rhs;
    lhs ^= rhs;
};

using fixed_boolean_view = fdapde::MatrixView<bool, 2, 3>;
using fixed_const_boolean_view = fdapde::MatrixView<const bool, 2, 3>;
using dynamic_boolean_view = fdapde::MatrixView<bool, fdapde::Dynamic, fdapde::Dynamic>;
using partial_boolean_view = fdapde::MatrixView<bool, 2, fdapde::Dynamic>;
using dynamic_const_boolean_view = fdapde::MatrixView<const bool, fdapde::Dynamic, fdapde::Dynamic>;
using boolean_view_word = typename fixed_boolean_view::bitpack_t;

// a fixed nonempty Boolean view requires an explicit storage binding
static_assert(!std::is_default_constructible_v<fixed_boolean_view>);
// a fixed const-storage Boolean view requires an explicit storage binding
static_assert(!std::is_default_constructible_v<fixed_const_boolean_view>);
// a dynamic Boolean view permits an initially empty binding
static_assert(std::is_default_constructible_v<dynamic_boolean_view>);
// a partially dynamic Boolean view permits an initially empty binding
static_assert(std::is_default_constructible_v<partial_boolean_view>);
// a dynamic const-storage Boolean view permits an initially empty binding
static_assert(std::is_default_constructible_v<dynamic_const_boolean_view>);
// a mutable Boolean view binds packed-word storage
static_assert(std::is_constructible_v<fixed_boolean_view, boolean_view_word*>);
// a mutable Boolean view cannot bind const packed-word storage
static_assert(!std::is_constructible_v<fixed_boolean_view, const boolean_view_word*>);
// a Boolean view rejects unpacked bool storage
static_assert(!std::is_constructible_v<fixed_boolean_view, bool*>);
// a Boolean view rejects byte storage with the wrong packed-word type
static_assert(!std::is_constructible_v<fixed_boolean_view, unsigned char*>);
// a const Boolean view can borrow mutable packed-word storage read-only
static_assert(std::is_constructible_v<fixed_const_boolean_view, boolean_view_word*>);
// a const Boolean view can borrow const packed-word storage
static_assert(std::is_constructible_v<fixed_const_boolean_view, const boolean_view_word*>);
// a const Boolean view rejects unpacked bool storage
static_assert(!std::is_constructible_v<fixed_const_boolean_view, const bool*>);
// a mutable Boolean view exposes its packed words through a mutable pointer
static_assert(std::is_same_v<decltype(std::declval<fixed_boolean_view&>().data()), boolean_view_word*>);
// a const Boolean view object exposes its packed words through a const pointer
static_assert(std::is_same_v<decltype(std::declval<const fixed_boolean_view&>().data()), const boolean_view_word*>);
// a const-storage Boolean view never exposes a mutable packed-word pointer
static_assert(std::is_same_v<decltype(std::declval<fixed_const_boolean_view&>().data()), const boolean_view_word*>);
// the Boolean matrix trait recognizes a const-storage view
static_assert(fdapde::is_boolean_matrix_v<fixed_const_boolean_view>);
// a const-storage Boolean view advertises read-only access
static_assert(fixed_const_boolean_view::ReadOnly == 1);
// a mutable Boolean view permits coordinate writes
static_assert(permits_boolean_view_coordinate_write<fixed_boolean_view>);
// a const Boolean view object rejects coordinate writes
static_assert(!permits_boolean_view_coordinate_write<const fixed_boolean_view>);
// a const-storage Boolean view rejects coordinate writes
static_assert(!permits_boolean_view_coordinate_write<fixed_const_boolean_view>);
// a mutable Boolean row view permits vector-index writes
static_assert(permits_boolean_view_vector_write<fdapde::MatrixView<bool, 1, 3>>);
// a const Boolean row-view object rejects vector-index writes
static_assert(!permits_boolean_view_vector_write<const fdapde::MatrixView<bool, 1, 3>>);
// a const-storage Boolean row view rejects vector-index writes
static_assert(!permits_boolean_view_vector_write<fdapde::MatrixView<const bool, 1, 3>>);
// a mutable Boolean view permits bulk bit mutation
static_assert(permits_boolean_view_bulk_mutation<fixed_boolean_view>);
// a const Boolean view object rejects bulk bit mutation
static_assert(!permits_boolean_view_bulk_mutation<const fixed_boolean_view>);
// a const-storage Boolean view rejects bulk bit mutation
static_assert(!permits_boolean_view_bulk_mutation<fixed_const_boolean_view>);
// a mutable Boolean view accepts assignment from another view
static_assert(permits_boolean_view_assignment<fixed_boolean_view, fixed_boolean_view>);
// a const-storage Boolean view rejects assignment from another view
static_assert(!permits_boolean_view_assignment<fixed_const_boolean_view, fixed_const_boolean_view>);
// a mutable Boolean view permits compound bitwise assignment
static_assert(permits_boolean_view_compound_assignment<fixed_boolean_view>);
// a const-storage Boolean view rejects compound bitwise assignment
static_assert(!permits_boolean_view_compound_assignment<fixed_const_boolean_view>);
using temporary_boolean_view_assignment_result =
  decltype(std::declval<fixed_boolean_view&&>() = std::declval<const fixed_boolean_view&>());
// assignment to a temporary Boolean view returns a value with the same storage binding
static_assert(std::is_same_v<temporary_boolean_view_assignment_result, fixed_boolean_view>);

template <int StorageOrder> void check_exact_boolean_pack_accounting() {
    using exact_pack = fdapde::Matrix<bool, 8, 8, StorageOrder>;
    using partial_pack = fdapde::Matrix<bool, 5, 13, StorageOrder>;
    using dynamic_column = fdapde::Matrix<bool, fdapde::Dynamic, 1, StorageOrder>;
    using partial_owner = fdapde::Matrix<bool, 2, fdapde::Dynamic, StorageOrder>;

    // an exactly full fixed Boolean shape requires one packed word
    static_assert(exact_pack::StorageSize == 1);
    // a fixed shape crossing the word boundary requires two packed words
    static_assert(partial_pack::StorageSize == 2);

    exact_pack bits;
    // the exactly full owner reports one packed word
    EXPECT_EQ(bits.bitpacks(), 1);
    // default construction clears the owner's full storage word
    EXPECT_EQ(bits.bitpack(0), 0u);
    bits(1, 6) = true;
    constexpr int expected_bit = StorageOrder == fdapde::RowMajor ? 14 : 49;
    // coordinate assignment sets the expected physical bit for the selected storage order
    EXPECT_NE(bits.bitpack(0) & (typename exact_pack::bitpack_t(1) << expected_bit), 0u);
    const exact_pack copied = bits;
    // copy construction preserves the complete logical Boolean matrix
    EXPECT_TRUE(copied == bits);

    const partial_pack partial;
    // a partially filled second word is included in the packed count
    EXPECT_EQ(partial.bitpacks(), 2);
    // the first word of a default partial owner is zero
    EXPECT_EQ(partial.bitpack(0), 0u);
    // the second word of a default partial owner is zero
    EXPECT_EQ(partial.bitpack(1), 0u);

    partial_owner normalized_owner(2, 64);
    // normalizing dynamic dimensions retains the requested two rows
    EXPECT_EQ(normalized_owner.rows(), 2);
    // normalizing dynamic dimensions retains the requested sixty-four columns
    EXPECT_EQ(normalized_owner.cols(), 64);
    // the normalized 128-bit owner requires two words
    EXPECT_EQ(normalized_owner.bitpacks(), 2);
    normalized_owner(1, 63) = true;
    // the normalized owner's last coordinate remains writable and readable
    EXPECT_TRUE(normalized_owner(1, 63));

    constexpr std::array<int, 6> sizes {0, 1, 63, 64, 65, 130};
    constexpr std::array<int, 6> expected_counts {0, 1, 1, 1, 2, 3};
    for (std::size_t i = 0; i < sizes.size(); ++i) {
        const dynamic_column dynamic(sizes[i]);
        // runtime allocation uses the expected rounded-up word count at each boundary
        EXPECT_EQ(dynamic.bitpacks(), expected_counts[i]);
    }

    exact_pack block_owner;
    auto whole = block_owner.template block<8, 8>(0, 0);
    // an exactly word-sized block reports one packed word
    EXPECT_EQ(whole.bitpacks(), 1);
    whole.set();
    for (int i = 0; i < block_owner.rows(); ++i) {
        // setting the full block makes every owner coefficient true
        for (int j = 0; j < block_owner.cols(); ++j) EXPECT_TRUE(block_owner(i, j));
    }
    whole.clear();

    auto trailing = block_owner.template block<2, 3>(6, 5);
    trailing.set();
    for (int i = 0; i < block_owner.rows(); ++i) {
        for (int j = 0; j < block_owner.cols(); ++j) {
            // clearing outside the selected corner leaves precisely the corner bits set
            EXPECT_EQ(bool(block_owner(i, j)), i >= 6 && j >= 5);
        }
    }
    trailing.clear();
    for (int i = 0; i < block_owner.rows(); ++i) {
        // clearing the remaining block restores every owner bit to false
        for (int j = 0; j < block_owner.cols(); ++j) EXPECT_FALSE(block_owner(i, j));
    }

    using view_type = fdapde::MatrixView<bool, 8, 8, StorageOrder>;
    using bitpack_t = typename view_type::bitpack_t;
    constexpr bitpack_t canary = bitpack_t(0x5a5a);
    std::array<bitpack_t, 2> storage {bitpack_t(0), canary};
    view_type view(storage.data());
    // an exactly full Boolean view reports one packed word
    EXPECT_EQ(view.bitpacks(), 1);
    view.set();
    // bulk set fills every bit of the bound storage word
    EXPECT_EQ(storage[0], std::numeric_limits<bitpack_t>::max());
    // bulk set leaves the following canary word unchanged
    EXPECT_EQ(storage[1], canary);
    // the filled view equals an explicitly all-true owner
    EXPECT_TRUE(view == exact_pack(true));
    view.clear();
    // bulk clear zeroes the bound storage word
    EXPECT_EQ(storage[0], bitpack_t(0));
    // bulk clear leaves the following canary word unchanged
    EXPECT_EQ(storage[1], canary);

    using partial_view = fdapde::MatrixView<bool, 2, fdapde::Dynamic, StorageOrder>;
    std::array<bitpack_t, 3> partial_storage {bitpack_t(0), bitpack_t(0), canary};
    partial_view normalized_view(partial_storage.data(), 2, 64);
    // normalizing a dynamic view retains its two rows
    EXPECT_EQ(normalized_view.rows(), 2);
    // normalizing a dynamic view retains its sixty-four columns
    EXPECT_EQ(normalized_view.cols(), 64);
    // the normalized 128-bit view spans two words
    EXPECT_EQ(normalized_view.bitpacks(), 2);
    normalized_view.set(1, 63);
    // the normalized view reads its final logical bit
    EXPECT_TRUE(normalized_view(1, 63));
    // mutating a partial-word view leaves the following canary word unchanged
    EXPECT_EQ(partial_storage[2], canary);
}

template <int StorageOrder> void check_boolean_owner_contracts() {
    using fixed_matrix = fdapde::Matrix<bool, 2, 3, StorageOrder>;
    using dynamic_matrix = fdapde::Matrix<bool, fdapde::Dynamic, fdapde::Dynamic, StorageOrder>;
    using partial_matrix = fdapde::Matrix<bool, 2, fdapde::Dynamic, StorageOrder>;
    using dynamic_row = fdapde::Matrix<bool, 1, fdapde::Dynamic, StorageOrder>;
    using dynamic_column = fdapde::Matrix<bool, fdapde::Dynamic, 1, StorageOrder>;
    constexpr int OppositeOrder = StorageOrder == fdapde::RowMajor ? fdapde::ColMajor : fdapde::RowMajor;

    // boolean construction rejects a negative row extent
    EXPECT_THROW(static_cast<void>(dynamic_matrix(-1, 2)), std::invalid_argument);
    // boolean construction rejects a runtime row extent that conflicts with a fixed axis
    EXPECT_THROW(static_cast<void>(partial_matrix(3, 4)), std::invalid_argument);
    // fixed Boolean construction rejects runtime dimensions that conflict with its shape
    EXPECT_THROW(static_cast<void>(fixed_matrix(3, 2)), std::invalid_argument);
    // boolean vector construction rejects a negative length
    EXPECT_THROW(static_cast<void>(dynamic_column(-1)), std::invalid_argument);
    // boolean construction rejects an overflowing logical coefficient count
    EXPECT_THROW(static_cast<void>(dynamic_matrix(std::numeric_limits<int>::max(), 2)), std::length_error);

    const fixed_matrix fixed_zero(2, 3);
    // a fixed zero expression materializes with no true coefficients
    EXPECT_FALSE(fixed_zero.any());
    // a fixed zero expression materializes with a zero true-bit count
    EXPECT_EQ(fixed_zero.count(), 0);
    const dynamic_matrix dynamic_zero(2, 65);
    // a dynamic zero expression materializes with no true coefficients
    EXPECT_FALSE(dynamic_zero.any());
    // a dynamic zero expression materializes with a zero true-bit count
    EXPECT_EQ(dynamic_zero.count(), 0);
    const dynamic_matrix dynamic_ones(2, 65, true);
    // a dynamic ones expression sets every logical coefficient
    EXPECT_TRUE(dynamic_ones.all());
    // count includes all 130 true bits and excludes storage padding
    EXPECT_EQ(dynamic_ones.count(), 130);

    const std::array<bool, 6> expected {false, true, false, true, true, false};
    const auto expect_logical_values = [&expected](const auto& matrix) {
        // initializer construction adopts the expected two rows
        EXPECT_EQ(matrix.rows(), 2);
        // initializer construction adopts the expected three columns
        EXPECT_EQ(matrix.cols(), 3);
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                // initializer values appear at the expected logical row-major coordinates
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
    // a fixed Boolean owner rejects an initializer with too few coefficients
    EXPECT_THROW(static_cast<void>(fixed_matrix(short_vector_data)), std::invalid_argument);

    std::vector<bool> boundary_data(130);
    for (int i : {0, 63, 64, 129}) boundary_data[static_cast<std::size_t>(i)] = true;
    const dynamic_column boundary_vector(boundary_data);
    // a boundary-spanning vector retains all 130 logical positions
    EXPECT_EQ(boundary_vector.size(), 130);
    for (int i = 0; i < boundary_vector.size(); ++i) {
        // only the first, word-edge and last explicitly initialized vector bits are true
        EXPECT_EQ(bool(boundary_vector[i]), i == 0 || i == 63 || i == 64 || i == 129);
    }

    const fdapde::Matrix<int, 2, 3, OppositeOrder> numeric({0, 2, -3, 0, 4, 0});
    const fixed_matrix converted(numeric);
    const std::array<bool, 6> converted_expected {false, true, true, false, true, false};
    for (int i = 0; i < converted.rows(); ++i) {
        for (int j = 0; j < converted.cols(); ++j) {
            // numeric conversion maps each logical scalar to its expected Boolean value
            EXPECT_EQ(bool(converted(i, j)), converted_expected[static_cast<std::size_t>(i * converted.cols() + j)]);
        }
    }
    const fdapde::Matrix<int, fdapde::Dynamic, fdapde::Dynamic, OppositeOrder> smaller_numeric(1, 2);
    // a fixed Boolean owner rejects conversion from an incompatible numeric shape
    EXPECT_THROW(static_cast<void>(fixed_matrix(smaller_numeric)), std::invalid_argument);

    fdapde::Matrix<bool, 1, 2, StorageOrder> proxy_values;
    proxy_values(0, 1) = true;
    proxy_values(0, 0) = proxy_values(0, 1);
    proxy_values(0, 1) = false;
    // assignment between bit proxies copies the source bit's value
    EXPECT_TRUE(proxy_values(0, 0));

    dynamic_matrix copy_source(2, 65);
    copy_source(0, 0) = true;
    copy_source(1, 64) = true;
    dynamic_matrix copy_target(1, 2, true);
    copy_target = copy_source;
    // copy assignment adopts the source's two rows
    EXPECT_EQ(copy_target.rows(), 2);
    // copy assignment adopts the source's sixty-five columns
    EXPECT_EQ(copy_target.cols(), 65);
    // copy assignment resizes packed storage to three words
    EXPECT_EQ(copy_target.bitpacks(), 3);
    // copy assignment preserves exactly the two set source bits
    EXPECT_EQ(copy_target.count(), 2);
    // copy assignment preserves the first source bit
    EXPECT_TRUE(copy_target(0, 0));
    // copy assignment preserves the final source bit
    EXPECT_TRUE(copy_target(1, 64));
    copy_source.clear();
    // self-assignment preserves both set bits
    EXPECT_EQ(copy_target.count(), 2);

    const auto row_zero_xpr = dynamic_row::Zero(5);
    const auto row_ones_xpr = dynamic_row::Ones(5);
    // a row-vector zero factory retains its single row
    EXPECT_EQ(row_zero_xpr.rows(), 1);
    // a row-vector zero factory uses the requested length as its columns
    EXPECT_EQ(row_zero_xpr.cols(), 5);
    // a row-vector ones factory retains its single row
    EXPECT_EQ(row_ones_xpr.rows(), 1);
    // a row-vector ones factory uses the requested length as its columns
    EXPECT_EQ(row_ones_xpr.cols(), 5);
    if (row_zero_xpr.rows() == 1 && row_zero_xpr.cols() == 5 && row_ones_xpr.rows() == 1 && row_ones_xpr.cols() == 5) {
        const dynamic_row row_zero(row_zero_xpr);
        const dynamic_row row_ones(row_ones_xpr);
        // materializing a row zero expression leaves every bit false
        EXPECT_FALSE(row_zero.any());
        // materializing a row ones expression makes every bit true
        EXPECT_TRUE(row_ones.all());
    }
    const dynamic_column column_zero = dynamic_column::Zero(5);
    const dynamic_column column_ones = dynamic_column::Ones(5);
    // a column-vector zero factory uses the requested length as its rows
    EXPECT_EQ(column_zero.rows(), 5);
    // a column-vector zero factory retains its single column
    EXPECT_EQ(column_zero.cols(), 1);
    // materializing a column zero expression leaves every bit false
    EXPECT_FALSE(column_zero.any());
    // materializing a column ones expression makes every bit true
    EXPECT_TRUE(column_ones.all());

    dynamic_column retained(10);
    retained[1] = true;
    retained[9] = true;
    retained.resize(5);
    retained.resize(10);
    // shrinking and regrowing a statically oriented Boolean vector preserves its retained prefix
    EXPECT_TRUE(retained[1]);
    // a bit removed by shrinking does not reappear when the vector regrows
    EXPECT_FALSE(retained[9]);

    dynamic_column exposed_padding(5, true);
    exposed_padding.resize(10);
    // growing a statically oriented Boolean vector preserves its original five bits
    for (int i = 0; i < 5; ++i) EXPECT_TRUE(exposed_padding[i]);
    // growing a Boolean vector initializes the newly exposed positions to false
    for (int i = 5; i < 10; ++i) EXPECT_FALSE(exposed_padding[i]);

    dynamic_matrix stable_resize(5, 100);
    stable_resize(0, 0) = true;
    stable_resize(4, 99) = true;
    stable_resize.resize(20, 20);
    // shape-changing resize adopts the requested twenty rows
    EXPECT_EQ(stable_resize.rows(), 20);
    // shape-changing resize adopts the requested twenty columns
    EXPECT_EQ(stable_resize.cols(), 20);
    // shape-changing resize clears the previous Boolean coefficients
    EXPECT_FALSE(stable_resize.any());

    dynamic_matrix same_size_resize(2, 3);
    same_size_resize(0, 0) = true;
    same_size_resize(0, 2) = true;
    same_size_resize(1, 1) = true;
    same_size_resize.resize(2, 3);
    // resize to the identical shape preserves the three set bits
    EXPECT_EQ(same_size_resize.count(), 3);
    same_size_resize.resize(3, 2);
    // reshape-like resize with equal total size still adopts the new row count
    EXPECT_EQ(same_size_resize.rows(), 3);
    // reshape-like resize with equal total size still adopts the new column count
    EXPECT_EQ(same_size_resize.cols(), 2);
    // changing shape clears Boolean coefficients even when the total size is unchanged
    EXPECT_FALSE(same_size_resize.any());

    partial_matrix partial_resize(2, 32);
    partial_resize(0, 0) = true;
    partial_resize(1, 31) = true;
    partial_resize.resize(2, 33);
    // resizing across a word boundary allocates the required two words
    EXPECT_EQ(partial_resize.bitpacks(), 2);
    // resizing across a word boundary clears every logical bit
    EXPECT_FALSE(partial_resize.any());

    dynamic_matrix runtime_row_resize(1, 3, true);
    runtime_row_resize.resize(1, 4);
    // resizing a runtime row vector clears its prior coefficients
    EXPECT_FALSE(runtime_row_resize.any());

    partial_matrix partial(2, 3);
    partial(1, 2) = true;
    const int partial_bitpacks = partial.bitpacks();
    // resize rejects a row extent inconsistent with a fixed axis
    EXPECT_THROW(partial.resize(3, 3), std::invalid_argument);
    // failed fixed-axis resize preserves the original row count
    EXPECT_EQ(partial.rows(), 2);
    // failed fixed-axis resize preserves the original column count
    EXPECT_EQ(partial.cols(), 3);
    // failed fixed-axis resize preserves the packed-word count
    EXPECT_EQ(partial.bitpacks(), partial_bitpacks);
    // failed fixed-axis resize preserves the original set bit
    EXPECT_TRUE(partial(1, 2));

    dynamic_matrix bounded(1, 2);
    bounded(0, 1) = true;
    const int bounded_bitpacks = bounded.bitpacks();
    // resize rejects a coefficient count beyond the supported range
    EXPECT_THROW(bounded.resize(std::numeric_limits<int>::max(), 2), std::length_error);
    // failed overflowing resize preserves the row count
    EXPECT_EQ(bounded.rows(), 1);
    // failed overflowing resize preserves the column count
    EXPECT_EQ(bounded.cols(), 2);
    // failed overflowing resize preserves the packed-word count
    EXPECT_EQ(bounded.bitpacks(), bounded_bitpacks);
    if (bounded.rows() == 1 && bounded.cols() == 2 && bounded.bitpacks() == bounded_bitpacks) {
        // failed overflowing resize preserves the original set bit
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
        // logical bit comparison requires the same number of expected coefficients
        ASSERT_EQ(matrix.size(), static_cast<int>(expected.size()));
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                // each expression bit matches the explicit reference at its logical coordinate
                EXPECT_EQ(bool(matrix(i, j)), expected[static_cast<std::size_t>(i * matrix.cols() + j)]);
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
    // exclusive OR of equal logical matrices in different layouts produces a zero packed word
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
    // bitwise AND rejects operands with incompatible shapes
    EXPECT_THROW(static_cast<void>(dynamic_lhs & dynamic_rhs), std::invalid_argument);
    // compound OR rejects a source with incompatible shape
    EXPECT_THROW(dynamic_lhs |= dynamic_rhs, std::invalid_argument);
    // failed compound OR preserves the destination row count
    EXPECT_EQ(dynamic_lhs.rows(), 2);
    // failed compound OR preserves the destination column count
    EXPECT_EQ(dynamic_lhs.cols(), 3);
    // failed compound OR preserves the original set bit
    EXPECT_TRUE(dynamic_lhs(0, 1));
    // failed compound OR introduces no additional set bits
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
        // block comparison requires the expected number of logical coefficients
        ASSERT_EQ(matrix.size(), static_cast<int>(expected.size()));
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                // each block coefficient matches its explicit logical reference
                EXPECT_EQ(bool(matrix(i, j)), expected[static_cast<std::size_t>(i * matrix.cols() + j)]);
            }
        }
    };

    const fixed_matrix source({false, true, false, true, true, false, true, false, false, true, true, false});
    const auto static_block = source.template block<2, 2>(1, 1);
    // a static block retains its requested two rows
    EXPECT_EQ(static_block.rows(), 2);
    // a static block retains its requested two columns
    EXPECT_EQ(static_block.cols(), 2);
    expect_values(static_block, std::array {false, true, true, true});
    const fdapde::Matrix<bool, 2, 2, StorageOrder> static_materialized(static_block);
    expect_values(static_materialized, std::array {false, true, true, true});

    const auto dynamic_block = source.block(1, 1, 2, 2);
    // a runtime block retains its requested two rows
    EXPECT_EQ(dynamic_block.rows(), 2);
    // a runtime block retains its requested two columns
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
    expect_values(source.template right_cols<2>(), std::array {false, true, true, false, true, false});
    expect_values(source.right_cols(2), std::array {false, true, true, false, true, false});

    fdapde::Matrix<bool, 1, 3, StorageOrder> row_vector({false, true, true});
    auto row_scalar = row_vector.col(2);
    // selecting a scalar from a row yields one row
    EXPECT_EQ(row_scalar.rows(), 1);
    // selecting a scalar from a row yields one column
    EXPECT_EQ(row_scalar.cols(), 1);
    // the row-derived scalar reads the selected true bit
    EXPECT_TRUE(row_scalar(0, 0));
    row_scalar(0, 0) = false;
    // the row vector reads the expected false coefficient at its third position
    EXPECT_FALSE(row_vector(0, 2));

    fdapde::Matrix<bool, 3, 1, StorageOrder> column_vector({false, true, true});
    auto column_scalar = column_vector.row(2);
    // selecting a scalar from a column yields one row
    EXPECT_EQ(column_scalar.rows(), 1);
    // selecting a scalar from a column yields one column
    EXPECT_EQ(column_scalar.cols(), 1);
    // the column-derived scalar reads the selected true bit
    EXPECT_TRUE(column_scalar(0, 0));
    column_scalar(0, 0) = false;
    // the column vector reads the expected false coefficient at its third position
    EXPECT_FALSE(column_vector(2, 0));

    fdapde::Matrix<bool, 5, 25, StorageOrder> packed_owner;
    packed_owner(1, 2) = true;
    packed_owner(3, 1) = true;
    packed_owner(2, 22) = true;
    packed_owner(3, 23) = true;
    const auto packed_block = packed_owner.block(1, 1, 3, 23);
    using bitpack_t = typename decltype(packed_owner)::bitpack_t;
    // the selected multiword block reports two packed words
    EXPECT_EQ(packed_block.bitpacks(), 2);
    if constexpr (StorageOrder == fdapde::RowMajor) {
        // row-major block packing places its true coefficients at the expected first-word offsets
        EXPECT_EQ(packed_block.bitpack(0), (bitpack_t(1) << 1) | (bitpack_t(1) << 44) | (bitpack_t(1) << 46));
        // row-major block packing places its remaining true coefficient in the second word
        EXPECT_EQ(packed_block.bitpack(1), bitpack_t(1) << 4);
    } else {
        // column-major block packing places its true coefficients at the expected first-word offsets
        EXPECT_EQ(packed_block.bitpack(0), (bitpack_t(1) << 2) | (bitpack_t(1) << 3));
        // column-major block packing places its remaining true coefficients in the second word
        EXPECT_EQ(packed_block.bitpack(1), (bitpack_t(1) << 0) | (bitpack_t(1) << 4));
    }

    fdapde::Matrix<bool, 2, 4, StorageOrder> named_assignment({false, false, true, false, false, false, false, true});
    auto named_destination = named_assignment.template block<2, 2>(0, 0);
    const auto named_source = named_assignment.template block<2, 2>(0, 2);
    named_destination = named_source;
    expect_values(named_assignment, std::array {true, false, true, false, false, true, false, true});
    named_destination(0, 0) = false;
    // assignment through the named block clears its first selected coefficient
    EXPECT_FALSE(named_assignment(0, 0));
    // writing the destination block leaves the separate source block coefficient unchanged
    EXPECT_TRUE(named_assignment(0, 2));

    fdapde::Matrix<bool, 1, 4, StorageOrder> overlapping_assignment({true, false, true, false});
    overlapping_assignment.template block<1, 3>(0, 1) = overlapping_assignment.template block<1, 3>(0, 0);
    expect_values(overlapping_assignment, std::array {true, true, false, true});

    fixed_matrix expression_lhs({false, false, false, false, false, true, false, true, false, false, false, false});
    fixed_matrix expression_rhs({false, false, false, false, true, false, true, false, false, false, false, false});
    const auto stored_expression_block = [&expression_lhs, &expression_rhs] {
        return (expression_lhs | expression_rhs).template block<1, 2>(1, 1);
    }();
    const fdapde::Matrix<bool, 1, 2, StorageOrder> stored_expression_result(stored_expression_block);
    expect_values(stored_expression_result, std::array {true, true});

    dynamic_matrix bounds(3, 4);
    bounds(1, 1) = true;
    // row extraction rejects a negative row
    EXPECT_THROW(static_cast<void>(bounds.row(-1)), std::out_of_range);
    // row extraction rejects a row equal to the owner's row count
    EXPECT_THROW(static_cast<void>(bounds.row(bounds.rows())), std::out_of_range);
    // column extraction rejects a negative column
    EXPECT_THROW(static_cast<void>(bounds.col(-1)), std::out_of_range);
    // column extraction rejects a column equal to the owner's column count
    EXPECT_THROW(static_cast<void>(bounds.col(bounds.cols())), std::out_of_range);
    // static block extraction rejects a negative origin
    EXPECT_THROW(static_cast<void>(bounds.template block<2, 2>(-1, 0)), std::out_of_range);
    // static block extraction rejects a block extending beyond the owner
    EXPECT_THROW(static_cast<void>(bounds.template block<2, 2>(2, 3)), std::out_of_range);
    // runtime block extraction rejects a zero row extent
    EXPECT_THROW(static_cast<void>(bounds.block(0, 0, 0, 1)), std::invalid_argument);
    // runtime block extraction rejects a negative column extent
    EXPECT_THROW(static_cast<void>(bounds.block(0, 0, 1, -1)), std::invalid_argument);
    // runtime block extraction rejects a negative row origin
    EXPECT_THROW(static_cast<void>(bounds.block(-1, 0, 1, 1)), std::out_of_range);
    // runtime block extraction rejects a negative column origin
    EXPECT_THROW(static_cast<void>(bounds.block(0, -1, 1, 1)), std::out_of_range);
    // runtime block extraction rejects an out-of-bounds rectangle
    EXPECT_THROW(static_cast<void>(bounds.block(2, 3, 2, 2)), std::out_of_range);
    // bottom-row extraction rejects the minimum int count without arithmetic overflow
    EXPECT_THROW(static_cast<void>(bounds.bottom_rows(std::numeric_limits<int>::min())), std::invalid_argument);
    // bottom-row extraction rejects a count larger than the owner
    EXPECT_THROW(static_cast<void>(bounds.bottom_rows(4)), std::out_of_range);
    // right-column extraction rejects the minimum int count without arithmetic overflow
    EXPECT_THROW(static_cast<void>(bounds.right_cols(std::numeric_limits<int>::min())), std::invalid_argument);
    // right-column extraction rejects a count larger than the owner
    EXPECT_THROW(static_cast<void>(bounds.right_cols(5)), std::out_of_range);

    auto local_block = bounds.block(0, 0, 2, 2);
    const auto& const_local_block = local_block;
    // mutable block access rejects a negative local row
    EXPECT_THROW(static_cast<void>(local_block(-1, 0)), std::out_of_range);
    // mutable block access rejects a local row equal to the block height
    EXPECT_THROW(static_cast<void>(local_block(2, 0)), std::out_of_range);
    // const block access rejects a negative local column
    EXPECT_THROW(static_cast<void>(const_local_block(0, -1)), std::out_of_range);
    // const block access rejects a local column equal to the block width
    EXPECT_THROW(static_cast<void>(const_local_block(0, 2)), std::out_of_range);
    auto local_row = bounds.row(1);
    const auto& const_local_row = local_row;
    // mutable row indexing rejects a negative local index
    EXPECT_THROW(static_cast<void>(local_row[-1]), std::out_of_range);
    // mutable row indexing rejects an index equal to the row length
    EXPECT_THROW(static_cast<void>(local_row[local_row.size()]), std::out_of_range);
    // const row indexing rejects a negative local index
    EXPECT_THROW(static_cast<void>(const_local_row[-1]), std::out_of_range);
    // const row indexing rejects an index equal to the row length
    EXPECT_THROW(static_cast<void>(const_local_row[const_local_row.size()]), std::out_of_range);
    // block set rejects a negative local coordinate
    EXPECT_THROW(local_block.set(-1, 0), std::out_of_range);
    // block clear rejects a local coordinate outside the block
    EXPECT_THROW(local_block.clear(0, 2), std::out_of_range);
    // block packed access rejects a negative word index
    EXPECT_THROW(static_cast<void>(local_block.bitpack(-1)), std::out_of_range);
    // block packed access rejects an index equal to the packed-word count
    EXPECT_THROW(static_cast<void>(local_block.bitpack(local_block.bitpacks())), std::out_of_range);
    // failed block operations preserve the owner's original set bit
    EXPECT_TRUE(bounds(1, 1));
}

template <int StorageOrder> void check_boolean_reshape_contracts() {
    using source_matrix = fdapde::Matrix<bool, 2, 3, StorageOrder>;
    using target_matrix = fdapde::Matrix<bool, 3, 2, StorageOrder>;
    using dynamic_matrix = fdapde::Matrix<bool, fdapde::Dynamic, fdapde::Dynamic, StorageOrder>;

    const auto expect_values = [](const auto& matrix, const auto& expected) {
        // reshape comparison requires the expected logical coefficient count
        ASSERT_EQ(matrix.size(), static_cast<int>(expected.size()));
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                // each reshaped bit matches the explicit logical reference
                EXPECT_EQ(bool(matrix(i, j)), expected[static_cast<std::size_t>(i * matrix.cols() + j)]);
            }
        }
    };

    source_matrix source({false, true, true, false, true, false});
    const std::array<bool, 6> reshaped_expected = StorageOrder == fdapde::RowMajor ?
                                                    std::array<bool, 6> {false, true, true, false, true, false} :
                                                    std::array<bool, 6> {false, true, false, true, true, false};
    const std::array<bool, 6> flat_expected = StorageOrder == fdapde::RowMajor ?
                                                std::array<bool, 6> {false, true, true, false, true, false} :
                                                std::array<bool, 6> {false, false, true, true, true, false};

    auto static_reshape = source.template reshape<3, 2>();
    auto dynamic_reshape = source.reshape(3, 2);
    expect_values(target_matrix(static_reshape), reshaped_expected);
    expect_values(target_matrix(dynamic_reshape), reshaped_expected);
    // reshape preserves the number of packed storage words
    EXPECT_EQ(static_reshape.bitpacks(), source.bitpacks());
    // reshape preserves the first word's bit sequence
    EXPECT_EQ(static_reshape.bitpack(0), source.bitpack(0));

    dynamic_matrix packed_source(5, 13);
    packed_source(0, 0) = true;
    packed_source(4, 12) = true;
    const auto packed_static_reshape = packed_source.template reshape<1, 65>();
    const auto packed_dynamic_reshape = packed_source.reshape(1, 65);
    const auto packed_column_reshape = packed_source.reshape(65);
    // a static reshape of the multiword source still spans two words
    EXPECT_EQ(packed_static_reshape.bitpacks(), 2);
    // a runtime reshape of the multiword source still spans two words
    EXPECT_EQ(packed_dynamic_reshape.bitpacks(), 2);
    for (int i = 0; i < packed_source.bitpacks(); ++i) {
        // static reshape preserves every source packed word
        EXPECT_EQ(packed_static_reshape.bitpack(i), packed_source.bitpack(i));
        // runtime reshape preserves every source packed word
        EXPECT_EQ(packed_dynamic_reshape.bitpack(i), packed_source.bitpack(i));
    }
    // a row reshape reads the set bit immediately after the first word boundary
    EXPECT_TRUE(packed_static_reshape[64]);
    // a column reshape reads the same set bit after the word boundary
    EXPECT_TRUE(packed_column_reshape[64]);

    const dynamic_matrix packed_zero(5, 13);
    const auto stored_packed_expression = [&packed_source, &packed_zero] {
        return (packed_source | packed_zero).template reshape<1, 65>();
    }();
    // a stored reshaped expression retains its two packed words
    EXPECT_EQ(stored_packed_expression.bitpacks(), 2);
    for (int i = 0; i < packed_source.bitpacks(); ++i) {
        // a stored reshaped expression preserves every expected packed word
        EXPECT_EQ(stored_packed_expression.bitpack(i), packed_source.bitpack(i));
    }

    auto row = source.template reshape<1, 6>();
    auto column = source.template reshape<6>();
    const fdapde::BoolReshapeOp<1, fdapde::Dynamic, source_matrix> direct_row(source, source.size());
    // direct vector reshape uses a single row
    EXPECT_EQ(direct_row.rows(), 1);
    // direct vector reshape uses the source size as its column count
    EXPECT_EQ(direct_row.cols(), source.size());
    for (int i = 0; i < source.size(); ++i) {
        // the row reshape follows the explicit flattened coefficient sequence
        EXPECT_EQ(bool(row[i]), flat_expected[static_cast<std::size_t>(i)]);
        // the column reshape follows the same flattened coefficient sequence
        EXPECT_EQ(bool(column[i]), flat_expected[static_cast<std::size_t>(i)]);
        // the directly constructed row reshape follows the same flattened sequence
        EXPECT_EQ(bool(direct_row[i]), flat_expected[static_cast<std::size_t>(i)]);
    }

    static_reshape(0, 1) = false;
    if constexpr (StorageOrder == fdapde::RowMajor) {
        // writing through reshape clears the corresponding first-row source bit
        EXPECT_FALSE(source(0, 1));
        // writing through reshape sets the corresponding second-row source bit
        EXPECT_TRUE(source(1, 1));
    } else {
        // subsequent reshape assignment restores the first-row source bit
        EXPECT_TRUE(source(0, 1));
        // subsequent reshape assignment clears the second-row source bit
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
    // assignment through a named reshape updates its owner's first bit
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
    // mutable reshape access rejects a negative row
    EXPECT_THROW(static_cast<void>(static_reshape(-1, 0)), std::out_of_range);
    // mutable reshape access rejects a row equal to its height
    EXPECT_THROW(static_cast<void>(static_reshape(3, 0)), std::out_of_range);
    // const reshape access rejects a negative column
    EXPECT_THROW(static_cast<void>(const_static_reshape(0, -1)), std::out_of_range);
    // const reshape access rejects a column equal to its width
    EXPECT_THROW(static_cast<void>(const_static_reshape(0, 2)), std::out_of_range);
    const auto& const_row = row;
    // mutable reshaped-vector indexing rejects a negative index
    EXPECT_THROW(static_cast<void>(row[-1]), std::out_of_range);
    // mutable reshaped-vector indexing rejects an index equal to its length
    EXPECT_THROW(static_cast<void>(row[row.size()]), std::out_of_range);
    // const reshaped-vector indexing rejects a negative index
    EXPECT_THROW(static_cast<void>(const_row[-1]), std::out_of_range);
    // const reshaped-vector indexing rejects an index equal to its length
    EXPECT_THROW(static_cast<void>(const_row[const_row.size()]), std::out_of_range);
    // reshape packed access rejects a negative word index
    EXPECT_THROW(static_cast<void>(static_reshape.bitpack(-1)), std::out_of_range);
    // reshape packed access rejects an index equal to its word count
    EXPECT_THROW(static_cast<void>(static_reshape.bitpack(static_reshape.bitpacks())), std::out_of_range);

    dynamic_matrix empty;
    const auto empty_matrix = empty.reshape(0, 5);
    const auto empty_column = empty.reshape(0);
    // an empty matrix reshape retains the requested zero rows
    EXPECT_EQ(empty_matrix.rows(), 0);
    // an empty matrix reshape retains its nonzero column extent
    EXPECT_EQ(empty_matrix.cols(), 5);
    // an empty matrix reshape contains no logical bits
    EXPECT_EQ(empty_matrix.size(), 0);
    // an empty matrix reshape requires no packed words
    EXPECT_EQ(empty_matrix.bitpacks(), 0);
    // an empty column reshape has zero rows
    EXPECT_EQ(empty_column.rows(), 0);
    // an empty column reshape retains its single column
    EXPECT_EQ(empty_column.cols(), 1);
    // an empty column reshape contains no logical bits
    EXPECT_EQ(empty_column.size(), 0);
    // an empty column reshape requires no packed words
    EXPECT_EQ(empty_column.bitpacks(), 0);
}

template <int StorageOrder> void check_boolean_selection_contracts() {
    constexpr int OppositeOrder = StorageOrder == fdapde::RowMajor ? fdapde::ColMajor : fdapde::RowMajor;
    using mask_matrix = fdapde::Matrix<bool, 2, 2, StorageOrder>;
    using value_matrix = fdapde::Matrix<double, 2, 2, StorageOrder>;
    using opposite_value_matrix = fdapde::Matrix<double, 2, 2, OppositeOrder>;
    using mask_view = fdapde::MatrixView<const bool, 2, 2, StorageOrder>;
    using value_view = fdapde::MatrixView<const double, 2, 2, StorageOrder>;
    using dynamic_mask = fdapde::Matrix<bool, fdapde::Dynamic, fdapde::Dynamic, StorageOrder>;
    using dynamic_values = fdapde::Matrix<double, fdapde::Dynamic, fdapde::Dynamic, StorageOrder>;

    const auto expect_values = [](const auto& matrix, const auto& expected) {
        // selection comparison requires the expected number of coefficients
        ASSERT_EQ(matrix.size(), static_cast<int>(expected.size()));
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                // each selected numeric coefficient agrees with the explicit branch reference
                EXPECT_DOUBLE_EQ(matrix(i, j), expected[static_cast<std::size_t>(i * matrix.cols() + j)]);
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

    const auto stored_views = mask_view(diagonal.data()).select(value_view(true_values.data()), false_values);
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
    // dynamic selection adopts the mask's two rows
    EXPECT_EQ(dynamic_selected.rows(), 2);
    // dynamic selection adopts the mask's three columns
    EXPECT_EQ(dynamic_selected.cols(), 3);
    // selection reads the expected first coefficient from the chosen branch
    EXPECT_DOUBLE_EQ(dynamic_selected(0, 0), 1.0);
    // selection reads the expected neighboring coefficient from the alternate branch
    EXPECT_DOUBLE_EQ(dynamic_selected(0, 1), 3.0);
    // selection reads the expected final coefficient from the chosen branch
    EXPECT_DOUBLE_EQ(dynamic_selected(1, 2), 2.0);

    const dynamic_values wrong_shape(3, 2);
    // selection rejects a true branch with incompatible shape
    EXPECT_THROW(static_cast<void>(condition.select(wrong_shape, dynamic_false)), std::invalid_argument);
    // selection rejects a false branch with incompatible shape
    EXPECT_THROW(static_cast<void>(condition.select(dynamic_true, wrong_shape)), std::invalid_argument);

    const dynamic_mask empty_condition(0, 3);
    const dynamic_values empty_true(0, 3);
    const dynamic_values empty_false(0, 3);
    const auto empty_selection = empty_condition.select(empty_true, empty_false);
    // selection over an empty mask retains zero rows
    EXPECT_EQ(empty_selection.rows(), 0);
    // selection over an empty mask retains three columns
    EXPECT_EQ(empty_selection.cols(), 3);
    // selection over an empty mask has no coefficients
    EXPECT_EQ(empty_selection.size(), 0);

    const double nan = std::numeric_limits<double>::quiet_NaN();
    const fdapde::Matrix<double, 2, 2, StorageOrder> nan_values({0.0, nan, 2.0, nan});
    const auto nan_mask = fdapde::nan_indicator(nan_values);
    // a coefficientwise NaN predicate preserves the input row count
    EXPECT_EQ(nan_mask.rows(), 2);
    // a coefficientwise NaN predicate preserves the input column count
    EXPECT_EQ(nan_mask.cols(), 2);
    // the NaN predicate is false for the first non-NaN coefficient
    EXPECT_FALSE(bool(nan_mask(0, 0)));
    // the NaN predicate detects the NaN in the first row
    EXPECT_TRUE(bool(nan_mask(0, 1)));
    // the NaN predicate is false for the non-NaN coefficient in the second row
    EXPECT_FALSE(bool(nan_mask(1, 0)));
    // the NaN predicate detects the NaN in the second row
    EXPECT_TRUE(bool(nan_mask(1, 1)));

    const fdapde::Matrix<double, 1, 3, StorageOrder> nan_row({nan, 1.0, nan});
    const auto nan_row_mask = fdapde::nan_indicator(nan_row);
    // a row-vector NaN predicate retains its single row
    EXPECT_EQ(nan_row_mask.rows(), 1);
    // a row-vector NaN predicate retains its three columns
    EXPECT_EQ(nan_row_mask.cols(), 3);
    // the NaN predicate detects the first NaN in the row vector
    EXPECT_TRUE(bool(nan_row_mask(0, 0)));
    // the NaN predicate is false for the middle non-NaN coefficient
    EXPECT_FALSE(bool(nan_row_mask(0, 1)));
    // the NaN predicate detects the last NaN in the row vector
    EXPECT_TRUE(bool(nan_row_mask(0, 2)));
}

template <int StorageOrder> void check_boolean_repeat_contracts() {
    using dynamic_matrix = fdapde::Matrix<bool, fdapde::Dynamic, fdapde::Dynamic, StorageOrder>;
    using dynamic_column = fdapde::Matrix<bool, fdapde::Dynamic, 1, StorageOrder>;

    dynamic_matrix source(3, 4);
    source(0, 0) = true;
    source(0, 2) = true;
    source(1, 1) = true;
    source(2, 0) = true;
    source(2, 1) = true;
    source(2, 3) = true;

    const auto repeated = source.repeat(2, 4);
    // tiling doubles the source's three rows
    EXPECT_EQ(repeated.rows(), 6);
    // tiling quadruples the source's four columns
    EXPECT_EQ(repeated.cols(), 16);
    // the ninety-six-bit repeated expression spans two words
    EXPECT_EQ(repeated.bitpacks(), 2);
    // the tiled pattern contains true bits
    EXPECT_TRUE(repeated.any());
    // the tiled pattern retains its false bits
    EXPECT_FALSE(repeated.all());
    // eight copies of the source multiply its true-bit count by eight
    EXPECT_EQ(repeated.count(), source.count() * 8);

    dynamic_matrix expected(6, 16);
    for (int i = 0; i < expected.rows(); ++i) {
        for (int j = 0; j < expected.cols(); ++j) {
            expected(i, j) = bool(source(i % source.rows(), j % source.cols()));
            // each repeated coefficient matches the explicitly tiled matrix
            EXPECT_EQ(bool(repeated(i, j)), bool(expected(i, j)));
        }
    }
    for (int i = 0; i < repeated.bitpacks(); ++i) {
        // every repeated packed word matches the explicitly tiled reference
        EXPECT_EQ(repeated.bitpack(i), expected.bitpack(i));
    }

    dynamic_column column(5);
    column[1] = true;
    column[4] = true;
    const auto tiled_column = column.repeat(1, 4);
    // repeating a column preserves its five rows
    EXPECT_EQ(tiled_column.rows(), 5);
    // repeating a column four times produces four columns
    EXPECT_EQ(tiled_column.cols(), 4);
    for (int i = 0; i < tiled_column.rows(); ++i) {
        for (int j = 0; j < tiled_column.cols(); ++j) {
            // each repeated column copies the original vector coefficient
            EXPECT_EQ(bool(tiled_column(i, j)), bool(column[i]));
        }
    }

    const auto stored_expression = [&] { return (~source).repeat(1, 2); }();
    const dynamic_matrix stored_result(stored_expression);
    // a stored negated repeat retains its three rows
    EXPECT_EQ(stored_result.rows(), 3);
    // a stored negated repeat doubles its columns to eight
    EXPECT_EQ(stored_result.cols(), 8);
    for (int i = 0; i < stored_result.rows(); ++i) {
        for (int j = 0; j < stored_result.cols(); ++j) {
            // each stored repeated coefficient equals the complement of the corresponding source tile
            EXPECT_EQ(bool(stored_result(i, j)), !bool(source(i, j % source.cols())));
        }
    }

    const auto stored_view = fdapde::MatrixView<const bool, 3, 4, StorageOrder>(source.data()).repeat(2, 1);
    const dynamic_matrix view_result(stored_view);
    // repeating a temporary view doubles its rows to six
    EXPECT_EQ(view_result.rows(), 6);
    // repeating a temporary view retains its four columns
    EXPECT_EQ(view_result.cols(), 4);
    for (int i = 0; i < view_result.rows(); ++i) {
        for (int j = 0; j < view_result.cols(); ++j) {
            // the stored repeat reads the expected coefficient through the retained view
            EXPECT_EQ(bool(view_result(i, j)), bool(source(i % source.rows(), j)));
        }
    }

    dynamic_matrix aliased(1, 3);
    aliased(0, 0) = true;
    aliased(0, 2) = true;
    aliased = aliased.repeat(2, 1);
    // aliased repeat assignment adopts the expected two rows
    EXPECT_EQ(aliased.rows(), 2);
    // aliased repeat assignment adopts the expected three columns
    EXPECT_EQ(aliased.cols(), 3);
    for (int i = 0; i < aliased.rows(); ++i) {
        // overlap-safe repeat assignment preserves each leading true bit
        EXPECT_TRUE(bool(aliased(i, 0)));
        // overlap-safe repeat assignment preserves each middle false bit
        EXPECT_FALSE(bool(aliased(i, 1)));
        // overlap-safe repeat assignment preserves each final true bit
        EXPECT_TRUE(bool(aliased(i, 2)));
    }

    const dynamic_matrix zero_rows(0, 3);
    const dynamic_matrix zero_cols(3, 0);
    const auto repeated_zero_rows = zero_rows.repeat(2, 4);
    const auto repeated_zero_cols = zero_cols.repeat(2, 4);
    // repeating an empty-row source keeps the result row count zero
    EXPECT_EQ(repeated_zero_rows.rows(), 0);
    // repeating an empty-row source still scales its column extent
    EXPECT_EQ(repeated_zero_rows.cols(), 12);
    // repeating an empty-row source requires no storage words
    EXPECT_EQ(repeated_zero_rows.bitpacks(), 0);
    // repeating an empty-column source still scales its row extent
    EXPECT_EQ(repeated_zero_cols.rows(), 6);
    // repeating an empty-column source keeps the result column count zero
    EXPECT_EQ(repeated_zero_cols.cols(), 0);
    // repeating an empty-column source requires no storage words
    EXPECT_EQ(repeated_zero_cols.bitpacks(), 0);

    // repeat rejects a zero row repetition count
    EXPECT_THROW(static_cast<void>(source.repeat(0, 1)), std::invalid_argument);
    // repeat rejects a zero column repetition count
    EXPECT_THROW(static_cast<void>(source.repeat(1, 0)), std::invalid_argument);
    // repeat rejects a negative row repetition count
    EXPECT_THROW(static_cast<void>(source.repeat(-1, 1)), std::invalid_argument);
    // repeat rejects a negative column repetition count
    EXPECT_THROW(static_cast<void>(source.repeat(1, -1)), std::invalid_argument);
    // repeat rejects an overflowing repeated row extent
    EXPECT_THROW(static_cast<void>(source.repeat(std::numeric_limits<int>::max(), 1)), std::length_error);
    // repeat rejects an overflowing repeated column extent
    EXPECT_THROW(static_cast<void>(source.repeat(1, std::numeric_limits<int>::max())), std::length_error);
    // repeat rejects an overflowing total coefficient count
    EXPECT_THROW(static_cast<void>(source.repeat(20'000, 20'000)), std::length_error);

    // repeated coefficient access rejects a negative row
    EXPECT_THROW(static_cast<void>(repeated(-1, 0)), std::out_of_range);
    // repeated coefficient access rejects a row equal to the result height
    EXPECT_THROW(static_cast<void>(repeated(repeated.rows(), 0)), std::out_of_range);
    // repeated coefficient access rejects a negative column
    EXPECT_THROW(static_cast<void>(repeated(0, -1)), std::out_of_range);
    // repeated coefficient access rejects a column equal to the result width
    EXPECT_THROW(static_cast<void>(repeated(0, repeated.cols())), std::out_of_range);
    // repeated packed access rejects a negative word index
    EXPECT_THROW(static_cast<void>(repeated.bitpack(-1)), std::out_of_range);
    // repeated packed access rejects an index equal to the word count
    EXPECT_THROW(static_cast<void>(repeated.bitpack(repeated.bitpacks())), std::out_of_range);
}

template <int StorageOrder> void check_boolean_terminal_contracts() {
    using exact_matrix = fdapde::Matrix<bool, 8, 8, StorageOrder>;
    using tail_matrix = fdapde::Matrix<bool, 5, 13, StorageOrder>;
    using layout_matrix = fdapde::Matrix<bool, 2, 3, StorageOrder>;
    using dynamic_matrix = fdapde::Matrix<bool, fdapde::Dynamic, fdapde::Dynamic, StorageOrder>;
    constexpr int OppositeOrder = StorageOrder == fdapde::RowMajor ? fdapde::ColMajor : fdapde::RowMajor;
    using opposite_tail_matrix = fdapde::Matrix<bool, 5, 13, OppositeOrder>;
    using opposite_layout_matrix = fdapde::Matrix<bool, 2, 3, OppositeOrder>;

    exact_matrix exact(true);
    // all recognizes an exactly full all-true storage word
    EXPECT_TRUE(exact.all());
    // any recognizes true bits in an exactly full storage word
    EXPECT_TRUE(exact.any());
    // count includes all sixty-four logical bits of the full word
    EXPECT_EQ(exact.count(), 64);
    exact(7, 7) = false;
    // all detects the single cleared bit in the full word
    EXPECT_FALSE(exact.all());
    // any remains true when sixty-three bits are still set
    EXPECT_TRUE(exact.any());
    // count decreases to sixty-three after clearing one bit
    EXPECT_EQ(exact.count(), 63);

    const tail_matrix zeros;
    const tail_matrix logical_ones(std::vector<bool>(65, true));
    const auto inverted_zeros = ~zeros;
    const auto hidden_only = ~logical_ones;
    // all recognizes logical ones produced by negating zeros across a word boundary
    EXPECT_TRUE(inverted_zeros.all());
    // any recognizes logical ones produced by negating zeros across a word boundary
    EXPECT_TRUE(inverted_zeros.any());
    // count excludes padding bits in the negated sixty-five-bit expression
    EXPECT_EQ(inverted_zeros.count(), 65);
    // all recognizes the equivalent materialized sixty-five-bit pattern
    EXPECT_TRUE(logical_ones.all());
    // count includes only the materialized owner's sixty-five logical bits
    EXPECT_EQ(logical_ones.count(), 65);
    // padding bits alone cannot make all true
    EXPECT_FALSE(hidden_only.all());
    // padding bits alone cannot make any true
    EXPECT_FALSE(hidden_only.any());
    // padding bits alone do not contribute to count
    EXPECT_EQ(hidden_only.count(), 0);
    // logical equality ignores differences in unused tail bits
    EXPECT_TRUE(inverted_zeros == logical_ones);
    tail_matrix missing_last(logical_ones);
    missing_last(4, 12) = false;
    // logical equality detects a missing final logical bit
    EXPECT_FALSE(inverted_zeros == missing_last);

    std::vector<bool> tail_values(65);
    for (const int index : {1, 12, 13, 51, 64}) { tail_values[static_cast<std::size_t>(index)] = true; }
    const tail_matrix tail_layout(tail_values);
    opposite_tail_matrix other_tail_layout(tail_values);
    // logical equality compares coefficients across storage orders with a partial tail
    EXPECT_TRUE(tail_layout == other_tail_layout);
    // logical inequality is false for equivalent cross-order tail patterns
    EXPECT_FALSE(tail_layout != other_tail_layout);
    const auto stored_tail_expression = [&tail_layout, &zeros] { return tail_layout | zeros; }();
    // a stored expression compares equal to the same cross-order logical pattern
    EXPECT_TRUE(stored_tail_expression == other_tail_layout);
    other_tail_layout(3, 12) = !bool(other_tail_layout(3, 12));
    // logical equality detects a changed bit in the cross-order pattern
    EXPECT_FALSE(tail_layout == other_tail_layout);
    // logical inequality detects the same changed bit
    EXPECT_TRUE(tail_layout != other_tail_layout);

    const std::array<bool, 6> layout_values {false, true, true, true, false, false};
    const layout_matrix layout(std::vector<bool>(layout_values.begin(), layout_values.end()));
    const opposite_layout_matrix other_layout(std::vector<bool>(layout_values.begin(), layout_values.end()));
    // logical equality compares matching matrices independently of storage order
    EXPECT_TRUE(layout == other_layout);
    // which(true) returns set-bit indices in logical row order
    EXPECT_EQ(layout.which(true), (std::vector<int> {1, 2, 3}));
    // which(false) returns clear-bit indices in logical row order
    EXPECT_EQ(layout.which(false), (std::vector<int> {0, 4, 5}));
    // the free which function returns the same logical set-bit indices
    EXPECT_EQ(fdapde::which(layout), (std::vector<int> {1, 2, 3}));

    const layout_matrix layout_zeros;
    const auto stored_layout_expression = [&layout, &layout_zeros] { return layout | layout_zeros; }();
    // which on a stored expression preserves logical row ordering
    EXPECT_EQ(stored_layout_expression.which(true), (std::vector<int> {1, 2, 3}));
    // the free which function accepts the stored expression
    EXPECT_EQ(fdapde::which(stored_layout_expression), (std::vector<int> {1, 2, 3}));

    const dynamic_matrix empty;
    const dynamic_matrix zero_rows(0, 3);
    const dynamic_matrix zero_cols(3, 0);
    const dynamic_matrix mismatched_rows(2, 3);
    const dynamic_matrix mismatched_cols(3, 2);
    // comparison rejects different shapes even when their total sizes match
    EXPECT_THROW(static_cast<void>(mismatched_rows == mismatched_cols), std::invalid_argument);
    // which(true) returns no indices for an empty matrix
    EXPECT_TRUE(empty.which(true).empty());
    // which(false) returns no indices for an empty matrix
    EXPECT_TRUE(empty.which(false).empty());
    // the free which function returns no indices for an empty matrix
    EXPECT_TRUE(fdapde::which(empty).empty());
    // which(true) returns no indices when the row extent is zero
    EXPECT_TRUE(zero_rows.which(true).empty());
    // which(false) returns no indices when the column extent is zero
    EXPECT_TRUE(zero_cols.which(false).empty());
}

template <int StorageOrder> void check_boolean_view_contracts() {
    using fixed_view = fdapde::MatrixView<bool, 2, 3, StorageOrder>;
    using dynamic_view = fdapde::MatrixView<bool, fdapde::Dynamic, fdapde::Dynamic, StorageOrder>;
    using const_dynamic_view = fdapde::MatrixView<const bool, fdapde::Dynamic, fdapde::Dynamic, StorageOrder>;
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
    // writing through a Boolean view sets the expected physical storage bit
    EXPECT_NE(storage[0] & (bitpack_t(1) << mapped_bit), bitpack_t(0));
    // writing through a Boolean view preserves the next storage word
    EXPECT_EQ(storage[1], outside_canary);
    const auto& const_view_handle = view;
    // a const view handle observes the bit written through the mutable handle
    EXPECT_TRUE(bool(const_view_handle(1, 0)));

    const fixed_view shallow_alias(view);
    // copy construction of a view shares the source's storage address
    EXPECT_EQ(shallow_alias.data(), view.data());

    std::array<bitpack_t, 1> source_storage {bitpack_t(0)};
    fixed_view source(source_storage.data());
    source(0, 1) = true;
    source(1, 2) = true;
    storage[0] = hidden_canary;
    bitpack_t* const binding = view.data();
    view = source;
    // view assignment preserves the destination's original storage address
    EXPECT_EQ(view.data(), binding);
    // view assignment copies the source's first set coordinate
    EXPECT_TRUE(bool(view(0, 1)));
    // view assignment copies the source's second set coordinate
    EXPECT_TRUE(bool(view(1, 2)));
    // view assignment preserves unrelated bits beyond its logical extent
    EXPECT_EQ(storage[0] & hidden_canary, hidden_canary);

    const fdapde::Matrix<bool, 2, 3, StorageOrder> owner(std::vector<bool> {true, false, true, false, true, false});
    const fdapde::Matrix<bool, 2, 3, StorageOrder> owner_zero;
    view = owner | owner_zero;
    // owner-to-view assignment preserves the destination binding
    EXPECT_EQ(view.data(), binding);
    for (int i = 0; i < owner.rows(); ++i) {
        // owner-to-view assignment copies every logical coefficient
        for (int j = 0; j < owner.cols(); ++j) { EXPECT_EQ(bool(view(i, j)), bool(owner(i, j))); }
    }
    // owner-to-view assignment preserves unused bits in the destination word
    EXPECT_EQ(storage[0] & hidden_canary, hidden_canary);

    std::array<bitpack_t, 1> opposite_storage {bitpack_t(0)};
    opposite_view other(opposite_storage.data());
    other(0, 0) = true;
    other(1, 1) = true;
    view = other;
    // compound view operations leave the expected leading bit true
    EXPECT_TRUE(bool(view(0, 0)));
    // compound view operations leave the expected diagonal bit true
    EXPECT_TRUE(bool(view(1, 1)));
    // compound view operations clear the expected off-diagonal bit
    EXPECT_FALSE(bool(view(0, 1)));

    std::array<bitpack_t, 1> temporary_storage {bitpack_t(0)};
    const auto assigned_temporary = fixed_view(temporary_storage.data()) = source;
    // assignment to a temporary view returns the original destination binding
    EXPECT_EQ(assigned_temporary.data(), temporary_storage.data());
    // the returned temporary view exposes the first copied set bit
    EXPECT_TRUE(bool(assigned_temporary(0, 1)));
    // the returned temporary view exposes the second copied set bit
    EXPECT_TRUE(bool(assigned_temporary(1, 2)));

    const auto stored_expression = [&source_storage] { return ~fixed_view(source_storage.data()); }();
    const fdapde::Matrix<bool, 2, 3, StorageOrder> stored_result(stored_expression);
    for (int i = 0; i < source.rows(); ++i) {
        for (int j = 0; j < source.cols(); ++j) {
            // a stored expression over a temporary view reads each complemented source bit
            EXPECT_EQ(bool(stored_result(i, j)), !bool(source(i, j)));
        }
    }

    std::array<bitpack_t, 3> tail_storage {bitpack_t(0), hidden_canary, outside_canary};
    dynamic_view tail(tail_storage.data(), 5, 13);
    partial_view partial(tail_storage.data(), 5, 13);
    // a sixty-five-bit dynamic view spans two words
    EXPECT_EQ(tail.bitpacks(), 2);
    // a partially dynamic sixty-five-bit view spans two words
    EXPECT_EQ(partial.bitpacks(), 2);
    tail(4, 12) = true;
    // tail packed access masks unused bits and exposes only the final logical bit
    EXPECT_EQ(tail.bitpack(1), bitpack_t(1));
    // the partially dynamic view reads the final logical bit
    EXPECT_TRUE(bool(partial(4, 12)));
    // writing the view's tail leaves the following canary word unchanged
    EXPECT_EQ(tail_storage[2], outside_canary);
    tail.set();
    // bulk set fills the first complete storage word
    EXPECT_EQ(tail.bitpack(0), std::numeric_limits<bitpack_t>::max());
    // bulk set exposes only the valid bit in the final partial word
    EXPECT_EQ(tail.bitpack(1), bitpack_t(1));
    // bulk set preserves hidden bits outside the view's logical tail
    EXPECT_EQ(tail_storage[1] & hidden_canary, hidden_canary);
    // bulk set preserves the following canary word
    EXPECT_EQ(tail_storage[2], outside_canary);
    tail.clear();
    // bulk clear clears the first complete storage word
    EXPECT_EQ(tail.bitpack(0), bitpack_t(0));
    // bulk clear clears the valid bit in the final partial word
    EXPECT_EQ(tail.bitpack(1), bitpack_t(0));
    // bulk clear preserves hidden bits outside the view's logical tail
    EXPECT_EQ(tail_storage[1] & hidden_canary, hidden_canary);
    // bulk clear preserves the following canary word
    EXPECT_EQ(tail_storage[2], outside_canary);
    const const_dynamic_view read_only_tail(tail_storage.data(), 5, 13);
    // a read-only tail view keeps the supplied storage binding
    EXPECT_EQ(read_only_tail.data(), tail_storage.data());
    // a read-only tail view observes the previously cleared final logical bit
    EXPECT_FALSE(bool(read_only_tail(4, 12)));

    std::array<bitpack_t, 1> vector_storage {bitpack_t(0)};
    row_view row(vector_storage.data(), 5);
    column_view column(vector_storage.data(), 5);
    row[4] = true;
    // vector indexing on a column view reads the selected true bit
    EXPECT_TRUE(bool(column[4]));
    column[4] = false;
    // vector indexing on a row view reads the selected false bit
    EXPECT_FALSE(bool(row[4]));

    dynamic_view empty;
    partial_view partial_empty;
    const_dynamic_view const_empty;
    // a default fully dynamic view has zero rows
    EXPECT_EQ(empty.rows(), 0);
    // a default fully dynamic view has zero columns
    EXPECT_EQ(empty.cols(), 0);
    // a default fully dynamic view has no storage binding
    EXPECT_EQ(empty.data(), nullptr);
    // a default fully dynamic view spans no packed words
    EXPECT_EQ(empty.bitpacks(), 0);
    // a default partially dynamic view retains its fixed five rows
    EXPECT_EQ(partial_empty.rows(), 5);
    // a default partially dynamic view starts with zero dynamic columns
    EXPECT_EQ(partial_empty.cols(), 0);
    // a default partially dynamic view has no storage binding
    EXPECT_EQ(partial_empty.data(), nullptr);
    // a default partially dynamic view spans no packed words
    EXPECT_EQ(partial_empty.bitpacks(), 0);
    // a default const-storage dynamic view has no storage binding
    EXPECT_EQ(const_empty.data(), nullptr);
    empty.set();
    empty.clear();
    partial_empty.set();
    partial_empty.clear();

    // explicit view construction rejects a negative row extent
    EXPECT_THROW(static_cast<void>(dynamic_view(tail_storage.data(), -1, 2)), std::invalid_argument);
    // explicit view construction rejects a zero row extent
    EXPECT_THROW(static_cast<void>(dynamic_view(tail_storage.data(), 0, 2)), std::invalid_argument);
    // explicit view construction rejects a zero column extent
    EXPECT_THROW(static_cast<void>(dynamic_view(tail_storage.data(), 2, 0)), std::invalid_argument);
    // view construction rejects a row extent inconsistent with its static axis
    EXPECT_THROW(static_cast<void>(partial_view(tail_storage.data(), 4, 13)), std::invalid_argument);
    // row-view construction rejects a negative length
    EXPECT_THROW(static_cast<void>(row_view(vector_storage.data(), -1)), std::invalid_argument);
    // row-view construction rejects a zero length
    EXPECT_THROW(static_cast<void>(row_view(vector_storage.data(), 0)), std::invalid_argument);
    // fixed column-view construction rejects an inconsistent runtime length
    EXPECT_THROW(static_cast<void>(fixed_column_view(vector_storage.data(), 2)), std::invalid_argument);
    // view construction rejects an overflowing logical coefficient count
    EXPECT_THROW(
      static_cast<void>(dynamic_view(tail_storage.data(), std::numeric_limits<int>::max(), 2)), std::length_error);

    std::array<bitpack_t, 1> mismatch_destination_storage {hidden_canary};
    std::array<bitpack_t, 1> mismatch_source_storage {bitpack_t(3)};
    dynamic_view mismatch_destination(mismatch_destination_storage.data(), 2, 2);
    dynamic_view mismatch_source(mismatch_source_storage.data(), 2, 3);
    bitpack_t* const mismatch_binding = mismatch_destination.data();
    const bitpack_t mismatch_snapshot = mismatch_destination_storage[0];
    // view assignment rejects a source with incompatible dimensions
    EXPECT_THROW(mismatch_destination = mismatch_source, std::invalid_argument);
    // failed view assignment preserves the storage binding
    EXPECT_EQ(mismatch_destination.data(), mismatch_binding);
    // failed view assignment preserves the two-row extent
    EXPECT_EQ(mismatch_destination.rows(), 2);
    // failed view assignment preserves the two-column extent
    EXPECT_EQ(mismatch_destination.cols(), 2);
    // failed view assignment preserves the complete destination storage word
    EXPECT_EQ(mismatch_destination_storage[0], mismatch_snapshot);

    const bitpack_t bounds_snapshot = storage[0];
    // mutable view access rejects a negative row
    EXPECT_THROW(static_cast<void>(view(-1, 0)), std::out_of_range);
    // mutable view access rejects a row equal to the view height
    EXPECT_THROW(static_cast<void>(view(2, 0)), std::out_of_range);
    // const view access rejects a negative column
    EXPECT_THROW(static_cast<void>(const_view_handle(0, -1)), std::out_of_range);
    // const view access rejects a column equal to the view width
    EXPECT_THROW(static_cast<void>(const_view_handle(0, 3)), std::out_of_range);
    // row-view indexing rejects a negative index
    EXPECT_THROW(static_cast<void>(row[-1]), std::out_of_range);
    // row-view indexing rejects an index equal to the row length
    EXPECT_THROW(static_cast<void>(row[row.size()]), std::out_of_range);
    // view packed access rejects a negative word index
    EXPECT_THROW(static_cast<void>(tail.bitpack(-1)), std::out_of_range);
    // view packed access rejects an index equal to the word count
    EXPECT_THROW(static_cast<void>(tail.bitpack(tail.bitpacks())), std::out_of_range);
    // failed bounds checks leave the original storage word unchanged
    EXPECT_EQ(storage[0], bounds_snapshot);
}

}   // namespace

// exercise packed Boolean storage, expression lifetimes, views, selection, repeat and boundary contracts
TEST(linear_algebra, boolean) {
    fdapde::Matrix<bool, 2, 3> fixed({true, false, true, false, true, false});
    // fixed initializer construction retains two rows
    EXPECT_EQ(fixed.rows(), 2);
    // fixed initializer construction retains three columns
    EXPECT_EQ(fixed.cols(), 3);
    // the initializer preserves the first true bit
    EXPECT_TRUE(fixed(0, 0));
    // the initializer preserves the neighboring false bit
    EXPECT_FALSE(fixed(0, 1));

    fixed(1, 0).set();
    fixed(0, 2).clear();
    // the initializer preserves the second row's first true bit
    EXPECT_TRUE(fixed(1, 0));
    // the initializer preserves the first row's final false bit
    EXPECT_FALSE(fixed(0, 2));

    using dynamic_bool_matrix = fdapde::Matrix<bool, fdapde::Dynamic, fdapde::Dynamic>;
    constexpr int pack_size = static_cast<int>(dynamic_bool_matrix::PackSize);
    dynamic_bool_matrix dynamic(1, pack_size + 1);
    dynamic(0, pack_size - 1) = true;
    dynamic(0, pack_size) = true;

    dynamic_bool_matrix copy = dynamic;
    // copy construction preserves the set bit immediately before a word boundary
    EXPECT_TRUE(copy(0, pack_size - 1));
    // copy construction preserves the set bit immediately after a word boundary
    EXPECT_TRUE(copy(0, pack_size));
    copy(0, pack_size - 1).clear();
    // clearing the copied pre-boundary bit makes only that bit false
    EXPECT_FALSE(copy(0, pack_size - 1));
    // clearing the pre-boundary bit leaves the adjacent word's set bit intact
    EXPECT_TRUE(copy(0, pack_size));
    // mutating the copy leaves the source's pre-boundary bit true
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
    // a column-vector NaN mask retains its three rows
    EXPECT_EQ(nan_mask.rows(), 3);
    // a column-vector NaN mask retains its single column
    EXPECT_EQ(nan_mask.cols(), 1);
    // the NaN mask detects the first NaN vector coefficient
    EXPECT_TRUE(bool(nan_mask(0, 0)));
    // the NaN mask rejects the middle finite vector coefficient
    EXPECT_FALSE(bool(nan_mask(1, 0)));
    // the NaN mask detects the final NaN vector coefficient
    EXPECT_TRUE(bool(nan_mask(2, 0)));

    const std::vector<int> markers {2, 1, 2, 3};
    const auto marker_mask = fdapde::value_indicator(markers.cbegin(), markers.cend(), 2);
    // the custom marker predicate preserves the vector's four rows
    EXPECT_EQ(marker_mask.rows(), 4);
    // the custom marker predicate preserves the vector's single column
    EXPECT_EQ(marker_mask.cols(), 1);
    // the marker predicate recognizes the first marked coefficient
    EXPECT_TRUE(bool(marker_mask[0]));
    // the marker predicate rejects the second unmarked coefficient
    EXPECT_FALSE(bool(marker_mask[1]));
    // the marker predicate recognizes the third marked coefficient
    EXPECT_TRUE(bool(marker_mask[2]));
    // the marker predicate rejects the final unmarked coefficient
    EXPECT_FALSE(bool(marker_mask[3]));
    const auto empty_marker_mask = fdapde::value_indicator(markers.cend(), markers.cend(), 2);
    // applying the marker predicate to an empty vector preserves zero rows
    EXPECT_EQ(empty_marker_mask.rows(), 0);
    // applying the marker predicate to an empty vector preserves its single column
    EXPECT_EQ(empty_marker_mask.cols(), 1);
}
