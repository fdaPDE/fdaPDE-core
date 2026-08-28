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
#include <cstdint>
#include <limits>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using block_matrix = fdapde::SparseBlockMatrix<double, 2, 2>;
using explicit_block_matrix = fdapde::SparseBlockMatrix<double, 2, 2, fdapde::ColMajor, int>;
using narrow_block_matrix = fdapde::SparseBlockMatrix<double, 2, 2, fdapde::ColMajor, std::int8_t>;

template <int Options, typename StorageIndex>
concept valid_block_matrix = requires { typename fdapde::SparseBlockMatrix<double, 2, 2, Options, StorageIndex>; };

static_assert(std::is_same_v<block_matrix, explicit_block_matrix>);
static_assert(std::is_same_v<typename block_matrix::Scalar, double>);
static_assert(std::is_same_v<typename block_matrix::Index, int>);
static_assert(std::is_same_v<typename block_matrix::StorageIndex, int>);
static_assert(std::is_same_v<typename block_matrix::Nested, const block_matrix&>);
static_assert(block_matrix::StorageOrder == fdapde::ColMajor);
static_assert(valid_block_matrix<fdapde::ColMajor, short>);
static_assert(!valid_block_matrix<fdapde::RowMajor, int>);
static_assert(!valid_block_matrix<fdapde::ColMajor, unsigned>);
static_assert(std::is_same_v<decltype(std::declval<block_matrix&>().block(0, 0)), fdapde::SparseMatrix<double>&>);
static_assert(
  std::is_same_v<decltype(std::declval<const block_matrix&>().block(0, 0)), const fdapde::SparseMatrix<double>&>);
static_assert(!std::is_constructible_v<block_matrix, fdapde::SparseMatrix<std::string>, int, int, int>);

block_matrix make_mixed_blocks() {
    return block_matrix(
      fdapde::Matrix<double, 1, 2>({
        1.0, 0.0
    }),
      fdapde::SparseMatrix<int>(1, 1, {{0, 0, 2}}), 0, fdapde::Matrix<float, 2, 1>({3.0F, 4.0F}));
}

TEST(linear_algebra, sparse_block_matrix_owns_heterogeneous_blocks) {
    const block_matrix matrix = make_mixed_blocks();

    EXPECT_EQ(matrix.rows(), 3);
    EXPECT_EQ(matrix.cols(), 3);
    EXPECT_EQ(matrix.block_rows(), 2);
    EXPECT_EQ(matrix.block_cols(), 2);
    EXPECT_EQ(matrix.blockRows(), 2);
    EXPECT_EQ(matrix.blockCols(), 2);
    EXPECT_EQ(matrix.innerSize(), 3);
    EXPECT_EQ(matrix.outerSize(), 3);
    EXPECT_EQ(matrix.non_zeros(), 4U);
    EXPECT_EQ(matrix.nonZerosEstimate(), 4);
    EXPECT_TRUE(matrix.isCompressed());

    EXPECT_DOUBLE_EQ(matrix.coeff(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 2), 2.0);
    EXPECT_DOUBLE_EQ(matrix.coeff(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(matrix.coeff(1, 2), 3.0);
    EXPECT_DOUBLE_EQ(matrix.coeff(2, 2), 4.0);
    EXPECT_EQ(matrix.block(1, 0).rows(), 2);
    EXPECT_EQ(matrix.block(1, 0).cols(), 2);
    EXPECT_EQ(matrix.block(1, 0).non_zeros(), 0);
}

TEST(linear_algebra, sparse_block_matrix_copies_are_independent) {
    const block_matrix original = make_mixed_blocks();
    block_matrix copied(original);
    block_matrix assigned;
    assigned = original;

    copied.coeffRef(0, 0) = 9.0;
    assigned.block(1, 1).value_ref(1, 0) = 8.0;

    EXPECT_DOUBLE_EQ(original.coeff(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(original.coeff(2, 2), 4.0);
    EXPECT_DOUBLE_EQ(copied.coeff(0, 0), 9.0);
    EXPECT_DOUBLE_EQ(assigned.coeff(2, 2), 8.0);
    EXPECT_THROW(static_cast<void>(copied.value_ref(0, 1)), std::out_of_range);

    block_matrix moved(std::move(copied));
    block_matrix move_assigned;
    move_assigned = std::move(assigned);
    swap(moved, move_assigned);
    EXPECT_DOUBLE_EQ(moved.coeff(2, 2), 8.0);
    EXPECT_DOUBLE_EQ(move_assigned.coeff(0, 0), 9.0);
}

TEST(linear_algebra, sparse_block_matrix_rebuilds_global_and_local_coordinates) {
    block_matrix matrix(2, 3);
    EXPECT_EQ(matrix.rows(), 4);
    EXPECT_EQ(matrix.cols(), 6);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            EXPECT_EQ(matrix.block(i, j).rows(), 2);
            EXPECT_EQ(matrix.block(i, j).cols(), 3);
        }
    }

    matrix.setFromTriplets(
      std::vector<fdapde::Triplet<double>> {
        {0, 0, 1.0 },
        {1, 4, 2.0 },
        {2, 1, 3.0 },
        {2, 1, -1.0},
        {3, 5, 4.0 }
    });
    EXPECT_DOUBLE_EQ(matrix.block(0, 0).coeff(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(matrix.block(0, 1).coeff(1, 1), 2.0);
    EXPECT_DOUBLE_EQ(matrix.block(1, 0).coeff(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(matrix.block(1, 1).coeff(1, 2), 4.0);
    EXPECT_EQ(matrix.non_zeros(), 4U);

    EXPECT_EQ(matrix.innerBlockIndex(0), 0);
    EXPECT_EQ(matrix.innerBlockIndex(2), 1);
    EXPECT_EQ(matrix.outerBlockIndex(2), 0);
    EXPECT_EQ(matrix.outerBlockIndex(3), 1);
    EXPECT_EQ(matrix.indexToBlockInner(3), 1);
    EXPECT_EQ(matrix.indexToBlockOuter(5), 2);

    matrix.setBlockFromTriplets<0, 1>(std::vector<fdapde::Triplet<double>> {
      {0, 2, 7.0}
    });
    matrix.setBlockFromTriplets(
      1, 0,
      std::vector<fdapde::Triplet<double>> {
        {1, 0, 6.0}
    });
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 5), 7.0);
    EXPECT_DOUBLE_EQ(matrix.coeff(3, 0), 6.0);
    matrix.makeCompressed();
}

TEST(linear_algebra, sparse_block_matrix_inserts_and_iterates_stored_entries) {
    block_matrix matrix(std::array<int, 2> {1, 2}, std::array<int, 2> {2, 1});
    matrix.rebuild({
      {0, 0, 1.0},
      {0, 2, 5.0},
      {1, 0, 2.0},
      {1, 2, 6.0},
      {2, 0, 3.0},
      {2, 1, 4.0}
    });

    double& inserted = matrix.coeffRef(2, 2);
    EXPECT_DOUBLE_EQ(inserted, 0.0);
    EXPECT_TRUE(matrix.contains(2, 2));
    EXPECT_EQ(matrix.block(1, 1).non_zeros(), 2);
    EXPECT_THROW(static_cast<void>(matrix.value_ref(0, 1)), std::out_of_range);

    std::vector<std::tuple<int, int, double>> entries;
    for (int outer = 0; outer < matrix.outerSize(); ++outer) {
        for (block_matrix::InnerIterator entry(matrix, outer); entry; ++entry) {
            entries.emplace_back(entry.row(), entry.col(), entry.value());
            EXPECT_EQ(entry.outer(), outer);
            EXPECT_EQ(entry.index(), entry.row());
            if (entry.row() == 1 && entry.col() == 0) entry.valueRef() = 8.0;
        }
    }
    EXPECT_EQ(
      entries, (std::vector<std::tuple<int, int, double>> {
                 {0, 0, 1.0},
                 {1, 0, 2.0},
                 {2, 0, 3.0},
                 {2, 1, 4.0},
                 {0, 2, 5.0},
                 {1, 2, 6.0},
                 {2, 2, 0.0}
    }));
    EXPECT_DOUBLE_EQ(matrix.coeff(1, 0), 8.0);

    const block_matrix& const_matrix = matrix;
    block_matrix::InnerIterator const_entry(const_matrix, 1);
    ASSERT_TRUE(const_entry);
    EXPECT_EQ(const_entry.row(), 2);
    EXPECT_DOUBLE_EQ(const_entry.value(), 4.0);
    EXPECT_THROW(static_cast<void>(const_entry.valueRef()), std::logic_error);
    EXPECT_THROW(static_cast<void>(block_matrix::InnerIterator(matrix, -1)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(block_matrix::InnerIterator(matrix, matrix.cols())), std::out_of_range);

    block_matrix empty_extents(std::array<int, 2> {0, 2}, std::array<int, 2> {1, 0});
    empty_extents.rebuild({
      {0, 0, 9.0}
    });
    block_matrix::InnerIterator after_empty_row(empty_extents, 0);
    ASSERT_TRUE(after_empty_row);
    EXPECT_EQ(after_empty_row.row(), 0);
    EXPECT_DOUBLE_EQ(after_empty_row.value(), 9.0);
    EXPECT_FALSE(++after_empty_row);
}

TEST(linear_algebra, sparse_block_matrix_materializes_owning_native_matrices) {
    block_matrix source = make_mixed_blocks();
    source.coeffRef(2, 0);
    const auto sparse = source.to_sparse();
    const auto dense = source.to_dense();
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(sparse)>, fdapde::SparseMatrix<double>>);
    static_assert(
      std::is_same_v<std::remove_cvref_t<decltype(dense)>, fdapde::Matrix<double, fdapde::Dynamic, fdapde::Dynamic>>);

    EXPECT_EQ(sparse.rows(), 3);
    EXPECT_EQ(sparse.cols(), 3);
    EXPECT_TRUE(sparse.contains(2, 0));
    EXPECT_DOUBLE_EQ(sparse.coeff(0, 2), 2.0);
    EXPECT_DOUBLE_EQ(dense(1, 2), 3.0);
    EXPECT_DOUBLE_EQ(dense(2, 2), 4.0);

    source.coeffRef(0, 0) = 9.0;
    EXPECT_DOUBLE_EQ(sparse.coeff(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(dense(0, 0), 1.0);

    const auto from_temporary = make_mixed_blocks().to_sparse();
    EXPECT_DOUBLE_EQ(from_temporary.coeff(2, 2), 4.0);
}

TEST(linear_algebra, sparse_block_matrix_eagerly_owns_nested_blocks_and_sparse_patterns) {
    using nested_row = fdapde::SparseBlockMatrix<double, 1, 2>;
    using nested_owner = fdapde::SparseBlockMatrix<double, 2, 1>;

    fdapde::SparseMatrix<double> stored_zero(1, 1);
    stored_zero.coeffRef(0, 0);
    nested_row nested(
      stored_zero, fdapde::SparseMatrix<double>(
                     1, 1,
                     {
                       {0, 0, 2.0}
    }));
    nested_owner owner(nested, fdapde::Matrix<double, 1, 2>({3.0, 4.0}));

    EXPECT_TRUE(owner.block(0, 0).contains(0, 0));
    EXPECT_DOUBLE_EQ(owner.coeff(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(owner.coeff(0, 1), 2.0);
    EXPECT_DOUBLE_EQ(owner.coeff(1, 0), 3.0);
    EXPECT_DOUBLE_EQ(owner.coeff(1, 1), 4.0);

    stored_zero.coeffRef(0, 0) = 8.0;
    nested.coeffRef(0, 1) = 9.0;
    EXPECT_DOUBLE_EQ(owner.coeff(0, 0), 0.0);
    EXPECT_DOUBLE_EQ(owner.coeff(0, 1), 2.0);
    const auto flattened = owner.to_sparse();
    EXPECT_TRUE(flattened.contains(0, 0));
}

TEST(linear_algebra, sparse_block_matrix_rebuilds_unit_constraints_atomically) {
    block_matrix matrix(std::array<int, 2> {1, 2}, std::array<int, 2> {2, 1});
    matrix.rebuild({
      {0, 0, 2.0},
      {0, 1, 3.0},
      {1, 0, 4.0},
      {1, 1, 5.0},
      {1, 2, 6.0},
      {2, 1, 7.0},
      {2, 2, 8.0}
    });
    matrix.rebuild_with_constraints({1});

    EXPECT_DOUBLE_EQ(matrix.coeff(0, 0), 2.0);
    EXPECT_DOUBLE_EQ(matrix.coeff(1, 1), 1.0);
    EXPECT_DOUBLE_EQ(matrix.coeff(2, 2), 8.0);
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 1), 0.0);
    EXPECT_DOUBLE_EQ(matrix.coeff(1, 0), 0.0);
    EXPECT_DOUBLE_EQ(matrix.coeff(1, 2), 0.0);
    EXPECT_DOUBLE_EQ(matrix.coeff(2, 1), 0.0);
    EXPECT_EQ(matrix.non_zeros(), 3U);

    EXPECT_THROW(matrix.rebuild_with_constraints({3}), std::out_of_range);
    EXPECT_DOUBLE_EQ(matrix.coeff(1, 1), 1.0);
    EXPECT_EQ(matrix.non_zeros(), 3U);
    EXPECT_THROW(block_matrix(1, 2).rebuild_with_constraints({0}), std::invalid_argument);
}

TEST(linear_algebra, sparse_block_matrix_supports_explicit_and_empty_extents) {
    const std::vector<int> row_extents {1, 2};
    const std::vector<int> col_extents {2, 1};
    block_matrix explicit_extents(row_extents, col_extents);
    EXPECT_EQ(explicit_extents.rows(), 3);
    EXPECT_EQ(explicit_extents.cols(), 3);
    EXPECT_EQ(explicit_extents.block(0, 0).rows(), 1);
    EXPECT_EQ(explicit_extents.block(1, 1).rows(), 2);
    EXPECT_EQ(explicit_extents.block(1, 1).cols(), 1);

    block_matrix with_empty(std::array<int, 2> {0, 2}, std::array<int, 2> {1, 0});
    EXPECT_EQ(with_empty.rows(), 2);
    EXPECT_EQ(with_empty.cols(), 1);
    with_empty.rebuild({
      {0, 0, 5.0}
    });
    EXPECT_EQ(with_empty.innerBlockIndex(0), 1);
    EXPECT_EQ(with_empty.outerBlockIndex(0), 0);
    EXPECT_DOUBLE_EQ(with_empty.block(1, 0).coeff(0, 0), 5.0);

    const block_matrix placeholders(0, 0, 0, 0);
    EXPECT_EQ(placeholders.rows(), 2);
    EXPECT_EQ(placeholders.cols(), 2);
    EXPECT_EQ(placeholders.non_zeros(), 0U);

    const block_matrix empty;
    EXPECT_EQ(empty.rows(), 0);
    EXPECT_EQ(empty.cols(), 0);
    EXPECT_EQ(empty.nonZerosEstimate(), 0);
    EXPECT_TRUE(empty.isCompressed());
    EXPECT_THROW(static_cast<void>(empty.coeff(0, 0)), std::out_of_range);
}

TEST(linear_algebra, sparse_block_matrix_contracts_remain_active_without_debug_assertions) {
    EXPECT_THROW(static_cast<void>(block_matrix(-1, 2)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(block_matrix(std::vector<int> {1}, std::vector<int> {1, 1})), std::invalid_argument);
    EXPECT_THROW(
      static_cast<void>(block_matrix(std::array<int, 2> {1, -1}, std::array<int, 2> {1, 1})), std::invalid_argument);
    EXPECT_THROW(
      static_cast<void>(block_matrix(
        fdapde::Matrix<double, 1, 1>(), fdapde::Matrix<double, 2, 1>(), 0, fdapde::Matrix<double, 1, 1>())),
      std::invalid_argument);
    EXPECT_THROW(
      static_cast<void>(block_matrix(
        fdapde::Matrix<double, 1, 1>(), 0, fdapde::Matrix<double, 1, 2>(), fdapde::Matrix<double, 1, 1>())),
      std::invalid_argument);
    EXPECT_THROW(static_cast<void>(block_matrix(0, 0, 1, 0)), std::invalid_argument);

    block_matrix matrix(2, 3);
    matrix.rebuild({
      {0, 0, 1.0}
    });
    EXPECT_THROW(static_cast<void>(matrix.block(-1, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(matrix.block(0, 2)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(matrix.coeff(-1, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(matrix.coeff(4, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(matrix.innerBlockIndex(4)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(matrix.outerBlockIndex(6)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(matrix.indexToBlockInner(-1)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(matrix.indexToBlockOuter(6)), std::out_of_range);
    const std::size_t old_nonzeros = matrix.non_zeros();
    EXPECT_THROW(static_cast<void>(matrix.coeffRef(-1, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(matrix.coeffRef(0, 6)), std::out_of_range);
    EXPECT_EQ(matrix.non_zeros(), old_nonzeros);

    EXPECT_THROW(
      matrix.rebuild({
        {0, 0, 9.0},
        {4, 0, 2.0}
    }),
      std::out_of_range);
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 0), 1.0);
    EXPECT_THROW(
      matrix.rebuild_block(
        0, 0,
        {
          {2, 0, 1.0}
    }),
      std::out_of_range);
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 0), 1.0);

    matrix.block(0, 0).resize(1, 1);
    EXPECT_THROW(static_cast<void>(matrix.coeff(0, 0)), std::invalid_argument);

    EXPECT_THROW(
      static_cast<void>(
        block_matrix(std::array<int, 2> {std::numeric_limits<int>::max(), 1}, std::array<int, 2> {0, 0})),
      std::length_error);
    narrow_block_matrix narrow_boundary(std::array<int, 2> {64, 64}, std::array<int, 2> {100, 100});
    EXPECT_EQ(narrow_boundary.rows(), 128);
    EXPECT_EQ(narrow_boundary.cols(), 200);
    narrow_boundary.coeffRef(127, 199) = 3.0;
    narrow_block_matrix::InnerIterator narrow_entry(narrow_boundary, 199);
    ASSERT_TRUE(narrow_entry);
    EXPECT_EQ(static_cast<int>(narrow_entry.index()), 127);
    EXPECT_THROW(
      static_cast<void>(narrow_block_matrix(std::array<int, 2> {64, 65}, std::array<int, 2> {1, 1})),
      std::length_error);

    const fdapde::Matrix<double, 8, 8> full = fdapde::Matrix<double, 8, 8>::Ones();
    const narrow_block_matrix narrow_full(full, full, full, full);
    EXPECT_EQ(narrow_full.nonZerosEstimate(), 256);
}

}   // namespace
