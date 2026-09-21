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

#include <fdaPDE/sparse_linear_algebra.h>
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
// a mutable lvalue grid exposes its native CSR block by reference
static_assert(std::is_same_v<decltype(std::declval<block_matrix&>().block(0, 0)), fdapde::SparseMatrix<double>&>);
// a const lvalue grid exposes only const block storage
static_assert(
  std::is_same_v<decltype(std::declval<const block_matrix&>().block(0, 0)), const fdapde::SparseMatrix<double>&>);
// nonconvertible sparse scalar types cannot participate in block construction
static_assert(!std::is_constructible_v<block_matrix, fdapde::SparseMatrix<std::string>, int, int, int>);

block_matrix make_mixed_blocks() {
    return block_matrix(
      fdapde::Matrix<double, 1, 2>({
        1.0, 0.0
    }),
      fdapde::SparseMatrix<int>(1, 1, {{0, 0, 2}}), 0, fdapde::Matrix<float, 2, 1>({3.0F, 4.0F}));
}

// mixed dense and CSR temporaries are materialized into a consistent owning partition
TEST(linear_algebra, sparse_block_matrix_owns_heterogeneous_blocks) {
    const block_matrix matrix = make_mixed_blocks();

    // the block-row extents one and two sum to three global rows
    EXPECT_EQ(matrix.rows(), 3);
    // the block-column extents two and one sum to three global columns
    EXPECT_EQ(matrix.cols(), 3);
    // the grid retains its two compile-time block rows
    EXPECT_EQ(matrix.block_rows(), 2);
    // the grid retains its two compile-time block columns
    EXPECT_EQ(matrix.block_cols(), 2);
    // the input blocks contribute exactly four stored coefficients
    EXPECT_EQ(matrix.non_zeros(), 4U);

    // the top-left dense coefficient is copied to global coordinates
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 0), 1.0);
    // the top-right CSR coefficient is placed after the first block column
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 2), 2.0);
    // the zero placeholder contributes no coefficient in the lower-left block
    EXPECT_DOUBLE_EQ(matrix.coeff(1, 0), 0.0);
    // the first coefficient of the lower-right dense block is preserved
    EXPECT_DOUBLE_EQ(matrix.coeff(1, 2), 3.0);
    // the second coefficient of the lower-right dense block is preserved
    EXPECT_DOUBLE_EQ(matrix.coeff(2, 2), 4.0);
    // the zero placeholder inherits two rows from its block-row neighbor
    EXPECT_EQ(matrix.block(1, 0).rows(), 2);
    // the zero placeholder inherits two columns from its block-column neighbor
    EXPECT_EQ(matrix.block(1, 0).cols(), 2);
    // the inferred zero block allocates no stored coefficients
    EXPECT_EQ(matrix.block(1, 0).non_zeros(), 0);
}

// copy construction and assignment isolate coefficient mutations while moves and swap transfer ownership
TEST(linear_algebra, sparse_block_matrix_copies_are_independent) {
    const block_matrix original = make_mixed_blocks();
    block_matrix copied(original);
    block_matrix assigned;
    assigned = original;

    copied.coeff_ref(0, 0) = 9.0;
    assigned.block(1, 1).value_ref(1, 0) = 8.0;

    // writing the copied top-left block leaves the original coefficient unchanged
    EXPECT_DOUBLE_EQ(original.coeff(0, 0), 1.0);
    // writing an assigned block leaves the original lower-right coefficient unchanged
    EXPECT_DOUBLE_EQ(original.coeff(2, 2), 4.0);
    // the copied grid contains its independently assigned top-left value
    EXPECT_DOUBLE_EQ(copied.coeff(0, 0), 9.0);
    // the assigned grid contains its independently assigned lower-right value
    EXPECT_DOUBLE_EQ(assigned.coeff(2, 2), 8.0);
    // value_ref refuses to insert an absent coefficient
    EXPECT_THROW(static_cast<void>(copied.value_ref(0, 1)), std::out_of_range);

    block_matrix moved(std::move(copied));
    block_matrix move_assigned;
    move_assigned = std::move(assigned);
    swap(moved, move_assigned);
    // move and swap transfer the assigned lower-right coefficient to its final owner
    EXPECT_DOUBLE_EQ(moved.coeff(2, 2), 8.0);
    // move and swap transfer the copied top-left coefficient to its final owner
    EXPECT_DOUBLE_EQ(move_assigned.coeff(0, 0), 9.0);
}

// global triplets are partitioned into local coordinates and block rebuilds affect only their selected region
TEST(linear_algebra, sparse_block_matrix_rebuilds_global_and_local_coordinates) {
    block_matrix matrix(2, 3);
    // two uniform block rows of height two produce four global rows
    EXPECT_EQ(matrix.rows(), 4);
    // two uniform block columns of width three produce six global columns
    EXPECT_EQ(matrix.cols(), 6);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            // every block inherits the requested uniform height
            EXPECT_EQ(matrix.block(i, j).rows(), 2);
            // every block inherits the requested uniform width
            EXPECT_EQ(matrix.block(i, j).cols(), 3);
        }
    }

    matrix.rebuild(
      std::vector<fdapde::Triplet<double>> {
        {0, 0, 1.0 },
        {1, 4, 2.0 },
        {2, 1, 3.0 },
        {2, 1, -1.0},
        {3, 5, 4.0 }
    });
    // the first global triplet lands at the top-left block origin
    EXPECT_DOUBLE_EQ(matrix.block(0, 0).coeff(0, 0), 1.0);
    // global column four becomes local column one in the second block column
    EXPECT_DOUBLE_EQ(matrix.block(0, 1).coeff(1, 1), 2.0);
    // duplicate global triplets combine to two within the lower-left block
    EXPECT_DOUBLE_EQ(matrix.block(1, 0).coeff(0, 1), 2.0);
    // the last global triplet maps to the final local coefficient
    EXPECT_DOUBLE_EQ(matrix.block(1, 1).coeff(1, 2), 4.0);
    // duplicate combination leaves exactly four stored global entries
    EXPECT_EQ(matrix.non_zeros(), 4U);

    // the first global row belongs to the first block row
    EXPECT_EQ(matrix.row_block(0), 0);
    // the row at the partition boundary belongs to the second block row
    EXPECT_EQ(matrix.row_block(2), 1);
    // the final column before the boundary belongs to the first block column
    EXPECT_EQ(matrix.col_block(2), 0);
    // the column at the partition boundary belongs to the second block column
    EXPECT_EQ(matrix.col_block(3), 1);
    // global row three has local row one after subtracting the row offset
    EXPECT_EQ(matrix.local_row(3), 1);
    // global column five has local column two after subtracting the column offset
    EXPECT_EQ(matrix.local_col(5), 2);

    matrix.rebuild_block(
      0, 1,
      std::vector<fdapde::Triplet<double>> {
        {0, 2, 7.0}
    });
    matrix.rebuild_block(
      1, 0,
      std::vector<fdapde::Triplet<double>> {
        {1, 0, 6.0}
    });
    // rebuilding the upper-right block translates its local column two into global column five
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 5), 7.0);
    // rebuilding the lower-left block translates its local row one into global row three
    EXPECT_DOUBLE_EQ(matrix.coeff(3, 0), 6.0);
}

// inserting through global coordinates updates one block and flattening preserves the stored zero
// global insertion creates a stored zero in exactly one block and flattening preserves the pattern
TEST(linear_algebra, sparse_block_matrix_inserts_stored_entries) {
    block_matrix matrix(std::array<int, 2> {1, 2}, std::array<int, 2> {2, 1});
    matrix.rebuild({
      {0, 0, 1.},
      {0, 2, 5.},
      {1, 0, 2.},
      {1, 2, 6.},
      {2, 0, 3.},
      {2, 1, 4.}
    });
    double& inserted = matrix.coeff_ref(2, 2);
    // an absent position is initialized to zero before a mutable reference is returned
    EXPECT_DOUBLE_EQ(inserted, 0.);
    // the inserted zero becomes part of the global stored pattern
    EXPECT_TRUE(matrix.contains(2, 2));
    // only the lower-right block gains an additional stored coefficient
    EXPECT_EQ(matrix.block(1, 1).non_zeros(), 2);
    // value_ref still rejects missing entries after another coefficient is inserted
    EXPECT_THROW(static_cast<void>(matrix.value_ref(0, 1)), std::out_of_range);
    inserted = 9.;
    // writing the returned global reference updates the corresponding local block coefficient
    EXPECT_DOUBLE_EQ(matrix.block(1, 1).coeff(1, 0), 9.);
    inserted = 0.;
    const auto flattened = matrix.to_sparse();
    // flattening retains an explicitly stored zero in its CSR pattern
    EXPECT_TRUE(flattened.contains(2, 2));
    // flattening emits the original six entries and the inserted zero exactly once
    EXPECT_EQ(flattened.non_zeros(), 7);
}

// dense and CSR conversions return independent storage and preserve global coefficient placement
TEST(linear_algebra, sparse_block_matrix_materializes_owning_native_matrices) {
    block_matrix source = make_mixed_blocks();
    source.coeff_ref(2, 0);
    const auto sparse = source.to_sparse();
    const auto dense = source.to_dense();
    // CSR conversion returns a concrete native sparse owner
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(sparse)>, fdapde::SparseMatrix<double>>);
    // dense conversion returns a concrete native matrix owner
    static_assert(
      std::is_same_v<std::remove_cvref_t<decltype(dense)>, fdapde::Matrix<double, fdapde::Dynamic, fdapde::Dynamic>>);

    // the flattened CSR retains the global row count
    EXPECT_EQ(sparse.rows(), 3);
    // the flattened CSR retains the global column count
    EXPECT_EQ(sparse.cols(), 3);
    // the flattened CSR retains an explicitly inserted zero
    EXPECT_TRUE(sparse.contains(2, 0));
    // the flattened top-right coefficient keeps its global column coordinate
    EXPECT_DOUBLE_EQ(sparse.coeff(0, 2), 2.0);
    // the dense conversion places the first lower-right coefficient correctly
    EXPECT_DOUBLE_EQ(dense(1, 2), 3.0);
    // the dense conversion places the second lower-right coefficient correctly
    EXPECT_DOUBLE_EQ(dense(2, 2), 4.0);

    source.coeff_ref(0, 0) = 9.0;
    // subsequent source mutation leaves the existing sparse result unchanged
    EXPECT_DOUBLE_EQ(sparse.coeff(0, 0), 1.0);
    // subsequent source mutation leaves the existing dense result unchanged
    EXPECT_DOUBLE_EQ(dense(0, 0), 1.0);

    const auto from_temporary = make_mixed_blocks().to_sparse();
    // conversion from a temporary grid returns storage that survives the grid destruction
    EXPECT_DOUBLE_EQ(from_temporary.coeff(2, 2), 4.0);
}

// nested grids and sparse blocks are eagerly copied with their stored-zero pattern
TEST(linear_algebra, sparse_block_matrix_eagerly_owns_nested_blocks_and_sparse_patterns) {
    using nested_row = fdapde::SparseBlockMatrix<double, 1, 2>;
    using nested_owner = fdapde::SparseBlockMatrix<double, 2, 1>;

    fdapde::SparseMatrix<double> stored_zero(1, 1);
    stored_zero.coeff_ref(0, 0);
    nested_row nested(
      stored_zero, fdapde::SparseMatrix<double>(
                     1, 1,
                     {
                       {0, 0, 2.0}
    }));
    nested_owner owner(nested, fdapde::Matrix<double, 1, 2>({3.0, 4.0}));

    // a nested stored zero survives materialization into its parent block
    EXPECT_TRUE(owner.block(0, 0).contains(0, 0));
    // the copied stored-zero coefficient remains numerically zero
    EXPECT_DOUBLE_EQ(owner.coeff(0, 0), 0.0);
    // the nested upper-right coefficient is placed in its parent block
    EXPECT_DOUBLE_EQ(owner.coeff(0, 1), 2.0);
    // the lower dense block contributes its first coefficient
    EXPECT_DOUBLE_EQ(owner.coeff(1, 0), 3.0);
    // the lower dense block contributes its second coefficient
    EXPECT_DOUBLE_EQ(owner.coeff(1, 1), 4.0);

    stored_zero.coeff_ref(0, 0) = 8.0;
    nested.coeff_ref(0, 1) = 9.0;
    // mutating the original sparse block does not change the parent copy
    EXPECT_DOUBLE_EQ(owner.coeff(0, 0), 0.0);
    // mutating the nested grid does not change the parent copy
    EXPECT_DOUBLE_EQ(owner.coeff(0, 1), 2.0);
    const auto flattened = owner.to_sparse();
    // flattening the parent retains the nested stored zero
    EXPECT_TRUE(flattened.contains(0, 0));
}

// unit constraints clear the selected global row and column and commit replacement blocks atomically
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

    // the unconstrained first diagonal retains its original coefficient
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 0), 2.0);
    // the constrained diagonal becomes exactly one
    EXPECT_DOUBLE_EQ(matrix.coeff(1, 1), 1.0);
    // the unconstrained last diagonal retains its original coefficient
    EXPECT_DOUBLE_EQ(matrix.coeff(2, 2), 8.0);
    // the first entry of the constrained column becomes zero
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 1), 0.0);
    // the first entry of the constrained row becomes zero
    EXPECT_DOUBLE_EQ(matrix.coeff(1, 0), 0.0);
    // the last entry of the constrained row becomes zero
    EXPECT_DOUBLE_EQ(matrix.coeff(1, 2), 0.0);
    // the last entry of the constrained column becomes zero
    EXPECT_DOUBLE_EQ(matrix.coeff(2, 1), 0.0);
    // constraint rebuilding leaves only the three surviving diagonal entries
    EXPECT_EQ(matrix.non_zeros(), 3U);

    // an out-of-range constraint is rejected before replacing any block
    EXPECT_THROW(matrix.rebuild_with_constraints({3}), std::out_of_range);
    // a rejected constraint retains the previously imposed unit diagonal
    EXPECT_DOUBLE_EQ(matrix.coeff(1, 1), 1.0);
    // a rejected constraint retains the previous sparse pattern
    EXPECT_EQ(matrix.non_zeros(), 3U);
    // nonempty constraints require a square global matrix
    EXPECT_THROW(block_matrix(1, 2).rebuild_with_constraints({0}), std::invalid_argument);
}

// explicit partitions and empty block extents preserve global-to-local coordinate mapping
TEST(linear_algebra, sparse_block_matrix_supports_explicit_and_empty_extents) {
    const std::vector<int> row_extents {1, 2};
    const std::vector<int> col_extents {2, 1};
    block_matrix explicit_extents(row_extents, col_extents);
    // explicit row extents sum to three global rows
    EXPECT_EQ(explicit_extents.rows(), 3);
    // explicit column extents sum to three global columns
    EXPECT_EQ(explicit_extents.cols(), 3);
    // the first block row retains its specified height one
    EXPECT_EQ(explicit_extents.block(0, 0).rows(), 1);
    // the second block row retains its specified height two
    EXPECT_EQ(explicit_extents.block(1, 1).rows(), 2);
    // the second block column retains its specified width one
    EXPECT_EQ(explicit_extents.block(1, 1).cols(), 1);

    block_matrix with_empty(std::array<int, 2> {0, 2}, std::array<int, 2> {1, 0});
    // a zero-height block row adds no global rows
    EXPECT_EQ(with_empty.rows(), 2);
    // a zero-width block column adds no global columns
    EXPECT_EQ(with_empty.cols(), 1);
    with_empty.rebuild({
      {0, 0, 5.0}
    });
    // row lookup skips the leading empty block-row interval
    EXPECT_EQ(with_empty.row_block(0), 1);
    // column lookup selects the nonempty first block-column interval
    EXPECT_EQ(with_empty.col_block(0), 0);
    // a global triplet after an empty extent maps to local row zero
    EXPECT_DOUBLE_EQ(with_empty.block(1, 0).coeff(0, 0), 5.0);

    const block_matrix placeholders(0, 0, 0, 0);
    // all-zero scalar placeholders infer unit block-row extents
    EXPECT_EQ(placeholders.rows(), 2);
    // all-zero scalar placeholders infer unit block-column extents
    EXPECT_EQ(placeholders.cols(), 2);
    // all-zero placeholders contribute no stored coefficients
    EXPECT_EQ(placeholders.non_zeros(), 0U);

    const block_matrix empty;
    // a default grid has no global rows
    EXPECT_EQ(empty.rows(), 0);
    // a default grid has no global columns
    EXPECT_EQ(empty.cols(), 0);
    // a default empty grid rejects coefficient access
    EXPECT_THROW(static_cast<void>(empty.coeff(0, 0)), std::out_of_range);
}

// invalid partitions, coordinates and rebuilds fail before publishing structural changes
TEST(linear_algebra, sparse_block_matrix_checks_partition_and_public_indices) {
    // negative uniform block dimensions are rejected
    EXPECT_THROW(static_cast<void>(block_matrix(-1, 2)), std::invalid_argument);
    // the number of explicit row extents must match the fixed grid
    EXPECT_THROW(static_cast<void>(block_matrix(std::vector<int> {1}, std::vector<int> {1, 1})), std::invalid_argument);
    // a negative entry in an explicit extent list is rejected
    EXPECT_THROW(
      static_cast<void>(block_matrix(std::array<int, 2> {1, -1}, std::array<int, 2> {1, 1})), std::invalid_argument);
    // blocks in the same block row must have equal heights
    EXPECT_THROW(
      static_cast<void>(block_matrix(
        fdapde::Matrix<double, 1, 1>(), fdapde::Matrix<double, 2, 1>(), 0, fdapde::Matrix<double, 1, 1>())),
      std::invalid_argument);
    // blocks in the same block column must have equal widths
    EXPECT_THROW(
      static_cast<void>(block_matrix(
        fdapde::Matrix<double, 1, 1>(), 0, fdapde::Matrix<double, 1, 2>(), fdapde::Matrix<double, 1, 1>())),
      std::invalid_argument);
    // a scalar placeholder must be exactly zero
    EXPECT_THROW(static_cast<void>(block_matrix(0, 0, 1, 0)), std::invalid_argument);

    block_matrix matrix(2, 3);
    matrix.rebuild({
      {0, 0, 1.0}
    });
    // a negative block-row coordinate is rejected
    EXPECT_THROW(static_cast<void>(matrix.block(-1, 0)), std::out_of_range);
    // a block-column coordinate at the grid bound is rejected
    EXPECT_THROW(static_cast<void>(matrix.block(0, 2)), std::out_of_range);
    // a negative global row coordinate is rejected
    EXPECT_THROW(static_cast<void>(matrix.coeff(-1, 0)), std::out_of_range);
    // a global row coordinate at the dimension bound is rejected
    EXPECT_THROW(static_cast<void>(matrix.coeff(4, 0)), std::out_of_range);
    // row-to-block lookup rejects an out-of-range global row
    EXPECT_THROW(static_cast<void>(matrix.row_block(4)), std::out_of_range);
    // column-to-block lookup rejects an out-of-range global column
    EXPECT_THROW(static_cast<void>(matrix.col_block(6)), std::out_of_range);
    // local-row lookup rejects negative global coordinates
    EXPECT_THROW(static_cast<void>(matrix.local_row(-1)), std::out_of_range);
    // local-column lookup rejects a coordinate at the dimension bound
    EXPECT_THROW(static_cast<void>(matrix.local_col(6)), std::out_of_range);
    const std::size_t old_nonzeros = matrix.non_zeros();
    // coefficient insertion rejects negative row coordinates
    EXPECT_THROW(static_cast<void>(matrix.coeff_ref(-1, 0)), std::out_of_range);
    // coefficient insertion rejects a column at the dimension bound
    EXPECT_THROW(static_cast<void>(matrix.coeff_ref(0, 6)), std::out_of_range);
    // invalid insertion requests leave the complete stored-entry count unchanged
    EXPECT_EQ(matrix.non_zeros(), old_nonzeros);

    // an invalid global triplet rejects the whole replacement build
    EXPECT_THROW(
      matrix.rebuild({
        {0, 0, 9.0},
        {4, 0, 2.0}
    }),
      std::out_of_range);
    // a rejected global rebuild retains the previously stored coefficient
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 0), 1.0);
    // an invalid local triplet rejects the selected block replacement
    EXPECT_THROW(
      matrix.rebuild_block(
        0, 0,
        {
          {2, 0, 1.0}
    }),
      std::out_of_range);
    // a rejected local rebuild retains the previously stored coefficient
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 0), 1.0);

    matrix.block(0, 0).resize(1, 1);
    // structurally resizing a borrowed block invalidates global access until its shape is restored
    EXPECT_THROW(static_cast<void>(matrix.coeff(0, 0)), std::invalid_argument);

    // partition prefix sums reject global row-count overflow before block allocation
    EXPECT_THROW(
      static_cast<void>(
        block_matrix(std::array<int, 2> {std::numeric_limits<int>::max(), 1}, std::array<int, 2> {0, 0})),
      std::length_error);
}

template <typename T>
concept borrows_temporary_block = requires(T&& value) { std::move(value).block(0, 0); };
// a block reference cannot outlive a temporary grid owner
static_assert(!borrows_temporary_block<block_matrix>);

// mixed scalar materialization rejects out-of-range integral conversions before constructing a block
TEST(linear_algebra, sparse_block_matrix_checks_scalar_conversion) {
    using integer_blocks = fdapde::SparseBlockMatrix<int, 1, 2>;
    const fdapde::Matrix<double, 1, 1> large(std::numeric_limits<double>::infinity());
    const fdapde::SparseMatrix<double> sparse_large(
      1, 1,
      {
        {0, 0, std::numeric_limits<double>::max()}
    });
    // nonfinite dense coefficients cannot be converted to int storage
    EXPECT_THROW((integer_blocks(large, 0)), std::overflow_error);
    // finite sparse coefficients outside the int range are also rejected before narrowing
    EXPECT_THROW((integer_blocks(sparse_large, 0)), std::overflow_error);
    const fdapde::Matrix<double, 1, 1> fraction(3.75);
    const integer_blocks valid(fraction, 0);
    // representable scalar conversion follows the documented truncation toward zero
    EXPECT_EQ(valid.coeff(0, 0), 3);
}

// inserting new CSR entries preserves row ordering, existing coefficients and no-op reference stability
TEST(linear_algebra, sparse_coefficient_insertion_is_ordered) {
    fdapde::SparseMatrix<double> matrix(
      3, 4,
      {
        {1, 2, 5.}
    });
    double* existing = &matrix.value_ref(1, 2);
    // accessing an existing coefficient does not rebuild storage or invalidate its address
    EXPECT_EQ(&matrix.coeff_ref(1, 2), existing);
    matrix.coeff_ref(1, 0) = 2.;
    matrix.coeff_ref(0, 3) = 3.;
    matrix.coeff_ref(2, 1) = 4.;
    // insertion in preceding and following rows preserves the original value
    EXPECT_DOUBLE_EQ(matrix.coeff(1, 2), 5.);
    const auto row = matrix.row(1);
    auto it = row.begin();
    // inserting before the original column keeps row traversal sorted
    EXPECT_EQ((*it).column(), 0);
    ++it;
    // the original column follows the newly inserted smaller column
    EXPECT_EQ((*it).column(), 2);
    // invalid coordinates fail without inserting an entry
    EXPECT_THROW(matrix.coeff_ref(3, 0), std::out_of_range);
    // the failed insertion leaves the exact four-entry pattern unchanged
    EXPECT_EQ(matrix.non_zeros(), 4);
}

}   // namespace
