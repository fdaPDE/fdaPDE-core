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
}

// Current regression adapted from 86ff6d12:tests/linear_algebra/bool.cpp.
// Stable source: a2a9c88:test/src/binary_matrix_test.cpp.
// Stable declarations (9): static_sized_matrix, dynamic_sized_matrix, binary_vector, block_operations,
// binary_expresssions, visitors, block_repeat, eigen_assignment_and_construct, and reshaped.
// TODO(P4-B): cover packed MatrixView contracts, reshape/select lifetime seams, reductions/equality/which,
// repeat, and the remaining two-dimensional resize policy.
// Replace the historical Eigen assignment/construct assertion with native numeric-matrix conversion; do not
// restore an implicit Eigen bridge.
