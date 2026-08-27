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

#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

template <typename Matrix>
concept has_inserting_coeff_ref = requires(Matrix& matrix) { matrix.coeffRef(0, 0); };

using sparse_double = fdapde::SparseMatrix<double>;
using sparse_entry = decltype(*std::declval<const sparse_double&>().row(0).begin());

// the sparse owner exposes its double coefficient type
static_assert(std::is_same_v<typename sparse_double::Scalar, double>);
// the CSR index type is the checked int storage index
static_assert(std::is_same_v<typename sparse_double::Index, int>);
// row entries expose coefficients through const references
static_assert(std::is_same_v<decltype(std::declval<sparse_entry>().value()), const double&>);
// coefficient lookup cannot implicitly insert into the CSR pattern
static_assert(!has_inserting_coeff_ref<sparse_double>);

// collects stored row entries for explicit pattern and coefficient comparisons
template <typename Scalar>
std::vector<std::pair<int, Scalar>> collect_row(const fdapde::SparseMatrix<Scalar>& matrix, int row) {
    std::vector<std::pair<int, Scalar>> result;
    for (const auto entry : matrix.row(row)) result.emplace_back(entry.column(), entry.value());
    return result;
}

// checks duplicate compression, stored-value mutation and independent ownership
void check_sparse_construction_and_access() {
    fdapde::Triplet<double> mutable_triplet(2, 3, 1.0);
    mutable_triplet.value() = 4.0;
    // changing a triplet value preserves its row index
    EXPECT_EQ(mutable_triplet.row(), 2);
    // changing a triplet value preserves its column index
    EXPECT_EQ(mutable_triplet.col(), 3);
    // const access observes the explicitly replaced triplet value
    EXPECT_DOUBLE_EQ(std::as_const(mutable_triplet).value(), 4.0);

    const std::vector<fdapde::Triplet<double>> triplets {
      mutable_triplet, {0, 1, 2.0 },
       {2, 1, -1.0},
       {0, 1, 3.0 },
       {1, 0, 0.0 },
      {2, 1, 1.0 },
       {0, 3, -2.0},
       {0, 0, 1.0 },
       {0, 3, 2.0 },
    };
    sparse_double matrix(3, 4, triplets);

    // the rectangular owner retains its requested row count
    EXPECT_EQ(matrix.rows(), 3);
    // the rectangular owner retains its requested column count
    EXPECT_EQ(matrix.cols(), 4);
    // duplicates and exact cancellations leave three stored entries
    EXPECT_EQ(matrix.non_zeros(), 3);
    // a unique triplet retains its coefficient
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 0), 1.0);
    // duplicate coefficients at zero, one sum to five
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 1), 5.0);
    // an absent position reads as zero
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 2), 0.0);
    // the mutated triplet is stored at its original coordinates
    EXPECT_DOUBLE_EQ(matrix.coeff(2, 3), 4.0);
    // an implicit zero does not belong to the stored pattern
    EXPECT_FALSE(matrix.contains(0, 2));
    // the nonzero triplet position belongs to the stored pattern
    EXPECT_TRUE(matrix.contains(2, 3));
    // row iteration emits sorted columns and the analytically combined values
    EXPECT_EQ(
      collect_row(matrix, 0), (std::vector<std::pair<int, double>> {
                                {0, 1.0},
                                {1, 5.0}
    }));
    // explicit zeros and duplicate cancellation leave the middle row empty
    EXPECT_TRUE(matrix.row(1).empty());
    // the last row retains only its uncancelled triplet
    EXPECT_EQ(
      collect_row(matrix, 2), (std::vector<std::pair<int, double>> {
                                {3, 4.0}
    }));

    const sparse_double repeated(3, 4, triplets);
    for (int row = 0; row < matrix.rows(); ++row) {
        // repeated construction yields identical ordered row contents
        EXPECT_EQ(collect_row(repeated, row), collect_row(matrix, row));
    }

    const sparse_double ordered_sum(
      1, 1,
      {
        {0, 0, 1.0e16 },
        {0, 0, -1.0e16},
        {0, 0, 1.0    }
    });
    // left-to-right duplicate summation preserves the final unit after large cancellation
    EXPECT_DOUBLE_EQ(ordered_sum.coeff(0, 0), 1.0);

    matrix.value_ref(0, 1) = 7.0;
    // mutation through value_ref updates the existing coefficient
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 1), 7.0);
    // value_ref rejects an absent position instead of inserting
    EXPECT_THROW(static_cast<void>(matrix.value_ref(0, 2)), std::out_of_range);
    matrix.value_ref(0, 1) = 0.0;
    // writing zero through a reference preserves the stored position
    EXPECT_TRUE(matrix.contains(0, 1));
    // writing zero does not change the stored entry count
    EXPECT_EQ(matrix.non_zeros(), 3);
    matrix.rebuild({
      {0, 0, 1.0},
      {0, 1, 0.0},
      {2, 3, 4.0}
    });
    // rebuild removes an explicitly supplied zero
    EXPECT_FALSE(matrix.contains(0, 1));
    // rebuild retains only the two nonzero entries
    EXPECT_EQ(matrix.non_zeros(), 2);

    sparse_double copy(matrix);
    copy.value_ref(0, 0) = 9.0;
    // mutating a deep copy leaves the original coefficient unchanged
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 0), 1.0);
    // the copied owner contains its independently modified coefficient
    EXPECT_DOUBLE_EQ(copy.coeff(0, 0), 9.0);

    sparse_double assigned(1, 1);
    assigned = matrix;
    // copy assignment replaces the destination row count
    EXPECT_EQ(assigned.rows(), 3);
    // copy assignment replaces the destination column count
    EXPECT_EQ(assigned.cols(), 4);
    // copy assignment preserves the source coefficient
    EXPECT_DOUBLE_EQ(assigned.coeff(2, 3), 4.0);

    sparse_double moved(std::move(copy));
    // move construction transfers the independently modified value
    EXPECT_DOUBLE_EQ(moved.coeff(0, 0), 9.0);
    sparse_double move_assigned;
    move_assigned = std::move(assigned);
    // move assignment transfers the source row count
    EXPECT_EQ(move_assigned.rows(), 3);
    // move assignment transfers the source column count
    EXPECT_EQ(move_assigned.cols(), 4);
    // move assignment transfers the source coefficient
    EXPECT_DOUBLE_EQ(move_assigned.coeff(2, 3), 4.0);
    copy.resize(0, 0);
    // a moved-from owner can be resized to zero rows
    EXPECT_EQ(copy.rows(), 0);
    // a moved-from owner can be resized to zero columns
    EXPECT_EQ(copy.cols(), 0);
    // the resized moved-from owner has no stored entries
    EXPECT_EQ(copy.non_zeros(), 0);

    moved.resize(3, 4);
    // resize sets the requested row count
    EXPECT_EQ(moved.rows(), 3);
    // resize sets the requested column count
    EXPECT_EQ(moved.cols(), 4);
    // resize discards all previously stored entries
    EXPECT_EQ(moved.non_zeros(), 0);
    // lookup after resize reads an implicit zero
    EXPECT_DOUBLE_EQ(moved.coeff(0, 0), 0.0);
}

// checks degenerate shapes, integral adjacency and extremely wide storage
void check_sparse_empty_and_integral_contracts() {
    const sparse_double empty;
    // a default owner has zero rows
    EXPECT_EQ(empty.rows(), 0);
    // a default owner has zero columns
    EXPECT_EQ(empty.cols(), 0);
    // a default owner stores no entries
    EXPECT_EQ(empty.non_zeros(), 0);

    const sparse_double zero_rows(0, 5);
    const sparse_double zero_cols(3, 0);
    // zero-row construction retains an empty row dimension
    EXPECT_EQ(zero_rows.rows(), 0);
    // zero-row construction still retains the requested column dimension
    EXPECT_EQ(zero_rows.cols(), 5);
    // zero-column construction retains the requested row dimension
    EXPECT_EQ(zero_cols.rows(), 3);
    // zero-column construction retains its empty column dimension
    EXPECT_EQ(zero_cols.cols(), 0);
    // a row in a zero-column matrix is empty
    EXPECT_TRUE(zero_cols.row(0).empty());

    const fdapde::SparseMatrix<int> adjacency(
      3, 3,
      std::vector<fdapde::Triplet<int>> {
        {0, 1, 1},
        {2, 0, 1},
        {0, 1, 2},
        {1, 2, 0}
    });
    // integral duplicate compression leaves two adjacency entries
    EXPECT_EQ(adjacency.non_zeros(), 2);
    // integral duplicate weights are summed exactly
    EXPECT_EQ(adjacency.coeff(0, 1), 3);
    // a unique integral adjacency entry is preserved
    EXPECT_EQ(adjacency.coeff(2, 0), 1);

    const fdapde::SparseMatrix<int> wide(
      1, std::numeric_limits<int>::max(),
      std::vector<fdapde::Triplet<int>> {
        {0, std::numeric_limits<int>::max() - 1, 5}
    });
    // an extremely wide matrix retains the maximum supported column count
    EXPECT_EQ(wide.cols(), std::numeric_limits<int>::max());
    // the final valid column is accessible without a column-sized allocation
    EXPECT_EQ(wide.coeff(0, std::numeric_limits<int>::max() - 1), 5);
}

// checks public bounds and preservation of the prior matrix after failure
void check_sparse_failure_contracts() {
    // negative row counts are rejected
    EXPECT_THROW(static_cast<void>(sparse_double(-1, 2)), std::invalid_argument);
    // negative column counts are rejected
    EXPECT_THROW(static_cast<void>(sparse_double(2, -1)), std::invalid_argument);
    // the maximum int row count is rejected because CSR needs one extra offset
    EXPECT_THROW(static_cast<void>(sparse_double(std::numeric_limits<int>::max(), 0)), std::length_error);

    // construction rejects a triplet with a negative row
    EXPECT_THROW(
      static_cast<void>(sparse_double(
        2, 2,
        std::vector<fdapde::Triplet<double>> {
          {-1, 0, 1.0}
    })),
      std::out_of_range);
    // construction rejects a triplet at the column upper bound
    EXPECT_THROW(
      static_cast<void>(sparse_double(
        2, 2,
        std::vector<fdapde::Triplet<double>> {
          {0, 2, 1.0}
    })),
      std::out_of_range);

    sparse_double matrix(
      2, 2,
      std::vector<fdapde::Triplet<double>> {
        {0, 0, 3.0},
        {1, 1, 4.0}
    });
    // lookup rejects a negative row
    EXPECT_THROW(static_cast<void>(matrix.coeff(-1, 0)), std::out_of_range);
    // lookup rejects the column upper bound
    EXPECT_THROW(static_cast<void>(matrix.coeff(0, 2)), std::out_of_range);
    // row access rejects the row upper bound
    EXPECT_THROW(static_cast<void>(matrix.row(2)), std::out_of_range);
    // resize rejects negative dimensions
    EXPECT_THROW(matrix.resize(-1, 2), std::invalid_argument);
    // failed resize preserves the previous row count
    EXPECT_EQ(matrix.rows(), 2);
    // failed resize preserves the previous column count
    EXPECT_EQ(matrix.cols(), 2);
    // failed resize preserves the previous stored pattern size
    EXPECT_EQ(matrix.non_zeros(), 2);
    // failed resize preserves the previous coefficient
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 0), 3.0);

    // rebuild rejects an out-of-range triplet before publishing changes
    EXPECT_THROW(
      matrix.rebuild(
        std::vector<fdapde::Triplet<double>> {
          {0, 0, 8.0},
          {2, 0, 1.0}
    }),
      std::out_of_range);
    // failed rebuild preserves the stored entry count
    EXPECT_EQ(matrix.non_zeros(), 2);
    // failed rebuild preserves the previous coefficient even after valid earlier triplets
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 0), 3.0);

    fdapde::SparseMatrix<int> integral(
      1, 1,
      std::vector<fdapde::Triplet<int>> {
        {0, 0, 7}
    });
    // integral duplicate summation rejects overflow before evaluating signed addition
    EXPECT_THROW(
      integral.rebuild(
        std::vector<fdapde::Triplet<int>> {
          {0, 0, std::numeric_limits<int>::max()},
          {0, 0, 1                              }
    }),
      std::overflow_error);
    // failed overflow rebuild preserves the prior pattern size
    EXPECT_EQ(integral.non_zeros(), 1);
    // failed overflow rebuild preserves the prior integral value
    EXPECT_EQ(integral.coeff(0, 0), 7);

    matrix.rebuild(
      std::vector<fdapde::Triplet<double>> {
        {1, 0, 6.0},
        {0, 1, 2.0}
    });
    // a subsequent valid rebuild replaces the pattern successfully
    EXPECT_EQ(matrix.non_zeros(), 2);
    // the new first row contains only the supplied off-diagonal entry
    EXPECT_EQ(
      collect_row(matrix, 0), (std::vector<std::pair<int, double>> {
                                {1, 2.0}
    }));
    // the new second row contains only its supplied off-diagonal entry
    EXPECT_EQ(
      collect_row(matrix, 1), (std::vector<std::pair<int, double>> {
                                {0, 6.0}
    }));
}

// verifies construction, mutation and ownership against explicit row and coefficient oracles
TEST(linear_algebra, sparse_construction_and_access) {
    // exercise the construction, mutation and ownership cases
    check_sparse_construction_and_access();
}

// verifies empty shapes and integer storage against explicit row and coefficient oracles
TEST(linear_algebra, sparse_empty_and_integral_contracts) {
    // exercise the empty shapes and integer storage cases
    check_sparse_empty_and_integral_contracts();
}

// verifies bounds and failure guarantees against explicit row and coefficient oracles
TEST(linear_algebra, sparse_failure_contracts) {
    // exercise the bounds and failure guarantees cases
    check_sparse_failure_contracts();
}

// requires borrowing only from an owner whose lifetime can outlast the returned row
template <typename Matrix>
concept borrows_temporary_row = requires(Matrix&& matrix) { std::move(matrix).row(0); };
// requires coefficient references to come from an lvalue matrix owner
template <typename Matrix>
concept borrows_temporary_value = requires(Matrix&& matrix) { std::move(matrix).value_ref(0, 0); };

// temporary owners cannot expose row views that would immediately dangle
static_assert(!borrows_temporary_row<sparse_double>);
// const temporary owners are rejected by the same borrowing boundary
static_assert(!borrows_temporary_row<const sparse_double>);
// temporary owners cannot expose mutable coefficient references
static_assert(!borrows_temporary_value<sparse_double>);
// the declared forward iterator supports the standard C++20 iterator requirements
static_assert(std::forward_iterator<sparse_double::ConstRowView::const_iterator>);

// checks the row-sort construction path and multipass row iteration on an extremely wide matrix
TEST(linear_algebra, sparse_wide_duplicates_and_iterators) {
    const sparse_double matrix(
      2, std::numeric_limits<int>::max(),
      {
        {0, 9, 1e16 },
        {0, 2, 4    },
        {0, 9, -1e16},
        {0, 9, 1    },
        {1, 3, 2    },
        {1, 3, -2   }
    });
    // row sorting preserves input order among duplicates and emits increasing columns
    EXPECT_EQ(
      collect_row(matrix, 0), (std::vector<std::pair<int, double>> {
                                {2, 4},
                                {9, 1}
    }));
    // cancelling duplicates disappear in the wide-shape construction path
    EXPECT_TRUE(matrix.row(1).empty());
    const auto row = matrix.row(0);
    auto first = row.begin();
    auto copy = first;
    const auto previous = first++;
    // postfix increment returns the iterator's previous position
    EXPECT_EQ(previous, copy);
    // incrementing one iterator does not change the position of its copy
    EXPECT_EQ((*copy).column(), 2);
    // the incremented iterator reaches the next stored column
    EXPECT_EQ((*first).column(), 9);
    // standard forward traversal counts exactly the stored entries
    EXPECT_EQ(std::distance(row.begin(), row.end()), 2);
    // the row-size observer agrees with traversal
    EXPECT_EQ(row.size(), 2);
}

// checks both signed overflow directions and unsigned overflow without changing the previous matrix
TEST(linear_algebra, sparse_integral_overflow_preserves_values) {
    fdapde::SparseMatrix<int> signed_matrix(
      1, 1,
      {
        {0, 0, 7}
    });
    // adding below the signed minimum is rejected before undefined signed arithmetic
    EXPECT_THROW(
      signed_matrix.rebuild({
        {0, 0, std::numeric_limits<int>::min()},
        {0, 0, -1                             }
    }),
      std::overflow_error);
    // the previous signed coefficient survives failed duplicate compression
    EXPECT_EQ(signed_matrix.coeff(0, 0), 7);
    fdapde::SparseMatrix<unsigned> unsigned_matrix(
      1, 100,
      {
        {0, 99, 8}
    });
    // the row-sort path rejects unsigned wraparound during duplicate summation
    EXPECT_THROW(
      unsigned_matrix.rebuild({
        {0, 99, std::numeric_limits<unsigned>::max()},
        {0, 99, 1                                   }
    }),
      std::overflow_error);
    // failed unsigned compression also preserves the previous coefficient
    EXPECT_EQ(unsigned_matrix.coeff(0, 99), 8u);
}

// moved-from owners can be queried, rejected on bounds, and rebuilt into usable storage
TEST(linear_algebra, sparse_moved_from_recovery) {
    sparse_double source(
      1, 1,
      {
        {0, 0, 3}
    });
    sparse_double destination(std::move(source));
    // move construction transfers the stored coefficient to its destination
    EXPECT_EQ(destination.coeff(0, 0), 3);
    // the source has zero rows after moving
    EXPECT_EQ(source.rows(), 0);
    // the source has zero columns after moving
    EXPECT_EQ(source.cols(), 0);
    // an empty moved-from source rejects coefficient access before touching its buffers
    EXPECT_THROW(source.coeff(0, 0), std::out_of_range);
    source.rebuild({});
    // rebuilding the empty moved-from shape yields no stored entries
    EXPECT_EQ(source.non_zeros(), 0);
    source.resize(1, 2);
    source.rebuild({
      {0, 1, 5}
    });
    // resize and rebuild make the moved-from owner reusable with a different shape
    EXPECT_EQ(source.coeff(0, 1), 5);
}

// constructs an asymmetric rectangular fixture with five known nonzero coefficients
sparse_double make_rectangular_fixture() {
    return sparse_double(
      3, 4,
      {
        {0, 0, 2.0 },
        {0, 2, -1.0},
        {1, 1, 3.0 },
        {1, 3, 4.0 },
        {2, 0, 5.0 }
    });
}

// checks sparse transforms against explicit patterns, values and invalid inputs
TEST(linear_algebra, sparse_matrix_transforms) {
    const sparse_double matrix = make_rectangular_fixture();
    const auto transposed = matrix.transpose();
    // the transpose has one row per original column
    EXPECT_EQ(transposed.rows(), 4);
    // original rows become the three transpose columns
    EXPECT_EQ(transposed.cols(), 3);
    // transposition preserves all five nonzero coefficients
    EXPECT_EQ(transposed.non_zeros(), 5);
    // the first transpose row contains the original first-column entries in sorted order
    EXPECT_EQ(
      collect_row(transposed, 0), (std::vector<std::pair<int, double>> {
                                    {0, 2.0},
                                    {2, 5.0}
    }));
    // the second transpose row contains only the original coefficient at row one
    EXPECT_EQ(
      collect_row(transposed, 1), (std::vector<std::pair<int, double>> {
                                    {1, 3.0}
    }));
    // the negative coefficient moves from column two into transpose row two
    EXPECT_EQ(
      collect_row(transposed, 2), (std::vector<std::pair<int, double>> {
                                    {0, -1.0}
    }));
    // the fourth transpose row contains the last-column coefficient
    EXPECT_EQ(
      collect_row(transposed, 3), (std::vector<std::pair<int, double>> {
                                    {1, 4.0}
    }));
    // creating a transpose leaves the source pattern intact
    EXPECT_EQ(matrix.non_zeros(), 5);
    // creating a transpose preserves the negative source coefficient
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 2), -1.0);
    const auto repeated_transpose = matrix.transpose();
    for (int row = 0; row < transposed.rows(); ++row) {
        // repeating the transpose yields identical ordered rows
        EXPECT_EQ(collect_row(repeated_transpose, row), collect_row(transposed, row));
    }

    sparse_double retained_zero(
      1, 2,
      {
        {0, 1, 1.0}
    });
    retained_zero.value_ref(0, 1) = 0.0;
    // assigning zero through a reference retains the source entry
    EXPECT_EQ(retained_zero.non_zeros(), 1);
    // transposition removes the explicitly stored zero
    EXPECT_EQ(retained_zero.transpose().non_zeros(), 0);

    const auto zero_rows_transposed = sparse_double(0, 4).transpose();
    const auto zero_cols_transposed = sparse_double(3, 0).transpose();
    // a zero-row matrix becomes a transpose with four rows
    EXPECT_EQ(zero_rows_transposed.rows(), 4);
    // the transpose retains the original empty axis as zero columns
    EXPECT_EQ(zero_rows_transposed.cols(), 0);
    // transposing a zero-row shape creates no entries
    EXPECT_EQ(zero_rows_transposed.non_zeros(), 0);
    // a zero-column matrix becomes a transpose with zero rows
    EXPECT_EQ(zero_cols_transposed.rows(), 0);
    // the original three rows become three columns
    EXPECT_EQ(zero_cols_transposed.cols(), 3);
    // transposing a zero-column shape creates no entries
    EXPECT_EQ(zero_cols_transposed.non_zeros(), 0);

    const sparse_double lower(
      3, 3,
      {
        {0, 0, 2.0 },
        {1, 0, -1.0},
        {1, 1, 3.0 },
        {2, 0, 4.0 },
        {2, 2, 5.0 }
    });
    const sparse_double upper(
      3, 3,
      {
        {0, 0, 2.0 },
        {0, 1, -1.0},
        {0, 2, 4.0 },
        {1, 1, 3.0 },
        {2, 2, 5.0 }
    });
    const auto expanded_lower = lower.symmetric_expanded(fdapde::Lower);
    const auto expanded_upper = upper.symmetric_expanded(fdapde::Upper);
    // mirroring two lower off-diagonal entries adds exactly two entries
    EXPECT_EQ(expanded_lower.non_zeros(), 7);
    // mirroring the upper representation gives the same seven-entry pattern
    EXPECT_EQ(expanded_upper.non_zeros(), 7);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            // both authoritative triangles expand to identical full coefficients
            EXPECT_DOUBLE_EQ(expanded_lower.coeff(i, j), expanded_upper.coeff(i, j));
        }
    }
    // the lower coefficient at one-zero is mirrored into zero-one
    EXPECT_DOUBLE_EQ(expanded_lower.coeff(0, 1), -1.0);
    // the lower coefficient remains unchanged at its original position
    EXPECT_DOUBLE_EQ(expanded_lower.coeff(1, 0), -1.0);
    // the lower coefficient at two-zero is mirrored into zero-two
    EXPECT_DOUBLE_EQ(expanded_lower.coeff(0, 2), 4.0);
    // expansion preserves the original coefficient at two-zero
    EXPECT_DOUBLE_EQ(expanded_lower.coeff(2, 0), 4.0);
    const auto repeated_expansion = lower.symmetric_expanded(fdapde::Lower);
    for (int row = 0; row < expanded_lower.rows(); ++row) {
        // repeated expansion yields identical ordered rows
        EXPECT_EQ(collect_row(repeated_expansion, row), collect_row(expanded_lower, row));
    }

    // expansion rejects the rectangular fixture
    EXPECT_THROW(static_cast<void>(matrix.symmetric_expanded(fdapde::Lower)), std::invalid_argument);
    // upper-only entries are rejected when the declared triangle is lower
    EXPECT_THROW(static_cast<void>(upper.symmetric_expanded(fdapde::Lower)), std::invalid_argument);
    // lower-only entries are rejected when the declared triangle is upper
    EXPECT_THROW(static_cast<void>(lower.symmetric_expanded(fdapde::Upper)), std::invalid_argument);
    // an unknown triangle selector raises the public argument error
    EXPECT_THROW(static_cast<void>(lower.symmetric_expanded(17)), std::invalid_argument);
    sparse_double retained_opposite_zero(
      2, 2,
      {
        {0, 1, 1.0}
    });
    retained_opposite_zero.value_ref(0, 1) = 0.0;
    // a stored opposite-triangle entry is rejected even after assigning zero
    EXPECT_THROW(static_cast<void>(retained_opposite_zero.symmetric_expanded(fdapde::Lower)), std::invalid_argument);

    const auto sums = matrix.row_sums();
    // row sums have one entry per sparse row
    EXPECT_EQ(sums.rows(), 3);
    // row sums are returned as a column vector
    EXPECT_EQ(sums.cols(), 1);
    // the first row sum combines two and minus one
    EXPECT_DOUBLE_EQ(sums[0], 1.0);
    // the second row sum combines three and four
    EXPECT_DOUBLE_EQ(sums[1], 7.0);
    // the last row sum preserves its single coefficient
    EXPECT_DOUBLE_EQ(sums[2], 5.0);
    // diagonal extraction rejects the rectangular fixture
    EXPECT_THROW(static_cast<void>(matrix.diagonal()), std::invalid_argument);

    const auto diagonal = lower.diagonal();
    // the square fixture yields three diagonal entries
    EXPECT_EQ(diagonal.size(), 3);
    // the first diagonal entry matches the lower fixture
    EXPECT_DOUBLE_EQ(diagonal[0], 2.0);
    // the middle diagonal entry matches the lower fixture
    EXPECT_DOUBLE_EQ(diagonal[1], 3.0);
    // the final diagonal entry matches the lower fixture
    EXPECT_DOUBLE_EQ(diagonal[2], 5.0);

    const fdapde::Vector<double, 4> diagonal_values({2.0, 0.0, -1.0, 4.0});
    const auto diagonal_matrix = sparse_double::from_diagonal(diagonal_values);
    // diagonal construction uses the vector length as its row count
    EXPECT_EQ(diagonal_matrix.rows(), 4);
    // diagonal construction produces the matching square column count
    EXPECT_EQ(diagonal_matrix.cols(), 4);
    // the zero diagonal coefficient is omitted from storage
    EXPECT_EQ(diagonal_matrix.non_zeros(), 3);
    // the first supplied value occupies the first diagonal position
    EXPECT_DOUBLE_EQ(diagonal_matrix.coeff(0, 0), 2.0);
    // the omitted diagonal entry reads back as implicit zero
    EXPECT_DOUBLE_EQ(diagonal_matrix.coeff(1, 1), 0.0);
    // the negative supplied value is retained on the diagonal
    EXPECT_DOUBLE_EQ(diagonal_matrix.coeff(2, 2), -1.0);
    // the last supplied value occupies the last diagonal position
    EXPECT_DOUBLE_EQ(diagonal_matrix.coeff(3, 3), 4.0);

    const sparse_double empty;
    // expanding the empty square matrix produces no entries
    EXPECT_EQ(empty.symmetric_expanded(fdapde::Lower).non_zeros(), 0);
    // summing zero rows yields an empty vector
    EXPECT_EQ(empty.row_sums().size(), 0);
    // extracting a zero-length diagonal yields an empty vector
    EXPECT_EQ(empty.diagonal().size(), 0);
    // an empty diagonal vector creates a sparse matrix without entries
    EXPECT_EQ(sparse_double::from_diagonal(fdapde::Vector<double, fdapde::Dynamic>()).non_zeros(), 0);
}

// checks eager sparse products across layouts, views, expressions and empty dimensions
TEST(linear_algebra, sparse_matrix_products) {
    const sparse_double matrix = make_rectangular_fixture();
    const fdapde::Vector<int, 4> vector({1, 2, 3, 4});
    const auto product = matrix * vector;
    // the vector product has one output per sparse row
    EXPECT_EQ(product.rows(), 3);
    // the vector product remains a column vector
    EXPECT_EQ(product.cols(), 1);
    // the first sparse dot product evaluates to two minus three
    EXPECT_DOUBLE_EQ(product[0], -1.0);
    // the second sparse dot product evaluates to six plus sixteen
    EXPECT_DOUBLE_EQ(product[1], 22.0);
    // the last sparse dot product evaluates to five times one
    EXPECT_DOUBLE_EQ(product[2], 5.0);

    const auto expression_product = matrix * (vector + vector);
    // doubling the vector expression doubles the first dot product
    EXPECT_DOUBLE_EQ(expression_product[0], -2.0);
    // doubling the vector expression doubles the second dot product
    EXPECT_DOUBLE_EQ(expression_product[1], 44.0);
    // doubling the vector expression doubles the final dot product
    EXPECT_DOUBLE_EQ(expression_product[2], 10.0);

    const int rhs_values[8] {1, 2, 3, 4, 5, 6, 7, 8};
    const fdapde::Matrix<int, 4, 2, fdapde::ColMajor> col_major_rhs(rhs_values);
    const auto dense_product = matrix * col_major_rhs;
    // the dense product retains the three sparse rows
    EXPECT_EQ(dense_product.rows(), 3);
    // the dense product retains both right-hand-side columns
    EXPECT_EQ(dense_product.cols(), 2);
    // the first column-major dot product evaluates to two minus five
    EXPECT_DOUBLE_EQ(dense_product(0, 0), -3.0);
    // the second column-major dot product evaluates to four minus six
    EXPECT_DOUBLE_EQ(dense_product(0, 1), -2.0);
    // the middle-row first dot product evaluates to nine plus twenty-eight
    EXPECT_DOUBLE_EQ(dense_product(1, 0), 37.0);
    // the middle-row second dot product evaluates to twelve plus thirty-two
    EXPECT_DOUBLE_EQ(dense_product(1, 1), 44.0);
    // the last-row first dot product evaluates to five times one
    EXPECT_DOUBLE_EQ(dense_product(2, 0), 5.0);
    // the last-row second dot product evaluates to five times two
    EXPECT_DOUBLE_EQ(dense_product(2, 1), 10.0);

    const fdapde::Matrix<int, 4, 2> row_major_rhs(rhs_values);
    const auto row_major_product = matrix * row_major_rhs;
    for (int i = 0; i < dense_product.rows(); ++i) {
        for (int j = 0; j < dense_product.cols(); ++j) {
            // row-major storage produces the same coefficients as column-major storage
            EXPECT_DOUBLE_EQ(row_major_product(i, j), dense_product(i, j));
        }
    }

    const fdapde::MatrixView<const int, 4, 2, fdapde::ColMajor> rhs_view(col_major_rhs.data());
    const auto view_product = matrix * rhs_view;
    // a const column-major view reproduces the first owning-matrix dot product
    EXPECT_DOUBLE_EQ(view_product(0, 0), -3.0);
    // a const column-major view reproduces the middle-row second dot product
    EXPECT_DOUBLE_EQ(view_product(1, 1), 44.0);

    const auto dense_expression_product = matrix * (col_major_rhs + col_major_rhs);
    // a doubled matrix expression doubles the first dense result
    EXPECT_DOUBLE_EQ(dense_expression_product(0, 0), -6.0);
    // a doubled matrix expression doubles the middle-row second result
    EXPECT_DOUBLE_EQ(dense_expression_product(1, 1), 88.0);

    const sparse_double single_column(
      3, 1,
      {
        {0, 0, 2.0 },
        {2, 0, -1.0}
    });
    const int one_row_values[2] {3, 4};
    const fdapde::Matrix<int, 1, 2> one_row_rhs(one_row_values);
    const auto one_row_product = single_column * one_row_rhs;
    // a row-vector right-hand side produces one output row per sparse row
    EXPECT_EQ(one_row_product.rows(), 3);
    // a row-vector right-hand side preserves its two columns
    EXPECT_EQ(one_row_product.cols(), 2);
    // the first outer-product entry multiplies two by three
    EXPECT_DOUBLE_EQ(one_row_product(0, 0), 6.0);
    // the second outer-product entry multiplies two by four
    EXPECT_DOUBLE_EQ(one_row_product(0, 1), 8.0);
    // the empty sparse middle row yields zero in the first output column
    EXPECT_DOUBLE_EQ(one_row_product(1, 0), 0.0);
    // the empty sparse middle row yields zero in the second output column
    EXPECT_DOUBLE_EQ(one_row_product(1, 1), 0.0);
    // the last outer-product row multiplies minus one by three
    EXPECT_DOUBLE_EQ(one_row_product(2, 0), -3.0);
    // the last outer-product row multiplies minus one by four
    EXPECT_DOUBLE_EQ(one_row_product(2, 1), -4.0);

    const fdapde::Vector<double, fdapde::Dynamic> empty_vector;
    const auto zero_row_product = sparse_double(0, 4) * vector;
    const auto zero_col_product = sparse_double(3, 0) * empty_vector;
    // a product with zero output rows returns an empty vector
    EXPECT_EQ(zero_row_product.size(), 0);
    // a zero inner dimension still preserves the three output rows
    EXPECT_EQ(zero_col_product.size(), 3);
    // the first empty dot product is zero
    EXPECT_DOUBLE_EQ(zero_col_product[0], 0.0);
    // the middle empty dot product is zero
    EXPECT_DOUBLE_EQ(zero_col_product[1], 0.0);
    // the final empty dot product is zero
    EXPECT_DOUBLE_EQ(zero_col_product[2], 0.0);

    const fdapde::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> empty_rhs(0, 2);
    const auto zero_col_dense_product = sparse_double(3, 0) * empty_rhs;
    // a zero inner dimension preserves the dense output row count
    EXPECT_EQ(zero_col_dense_product.rows(), 3);
    // a zero inner dimension preserves the dense output column count
    EXPECT_EQ(zero_col_dense_product.cols(), 2);
    // an empty dense dot product initializes its output to zero
    EXPECT_DOUBLE_EQ(zero_col_dense_product(2, 1), 0.0);

    const fdapde::Vector<double, 3> short_vector({1.0, 2.0, 3.0});
    const fdapde::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> short_matrix(3, 2);
    // a vector with too few entries raises a dimension error
    EXPECT_THROW(static_cast<void>(matrix * short_vector), std::invalid_argument);
    // a dense right-hand side with too few rows raises a dimension error
    EXPECT_THROW(static_cast<void>(matrix * short_matrix), std::invalid_argument);
    // failed products leave the source pattern unchanged
    EXPECT_EQ(matrix.non_zeros(), 5);
    // failed products leave the last-column coefficient unchanged
    EXPECT_DOUBLE_EQ(matrix.coeff(1, 3), 4.0);
}

// checks quadratic forms against a hand calculation and dimension failures
TEST(linear_algebra, sparse_matrix_quadratic_form) {
    const sparse_double lower(
      3, 3,
      {
        {0, 0, 2.0 },
        {1, 0, -1.0},
        {1, 1, 3.0 },
        {2, 0, 4.0 },
        {2, 2, 5.0 }
    });
    const auto matrix = lower.symmetric_expanded(fdapde::Lower);
    const fdapde::Vector<int, 3> vector({1, 2, 3});
    // the expanded symmetric fixture gives the hand-computed quadratic value seventy-nine
    EXPECT_DOUBLE_EQ(matrix.quadratic_form(vector), 79.0);
    // the empty quadratic form returns the additive identity
    EXPECT_DOUBLE_EQ(sparse_double().quadratic_form(fdapde::Vector<double, fdapde::Dynamic>()), 0.0);

    const fdapde::Vector<double, 2> short_vector({1.0, 2.0});
    // a short quadratic vector raises a dimension error
    EXPECT_THROW(static_cast<void>(matrix.quadratic_form(short_vector)), std::invalid_argument);
    // a rectangular matrix cannot define the square quadratic form
    EXPECT_THROW(static_cast<void>(make_rectangular_fixture().quadratic_form(vector)), std::invalid_argument);
}

// checks eager result types, ownership and boundary shapes through public sparse operations
TEST(linear_algebra, sparse_operation_boundaries) {
    using vector_product = decltype(std::declval<const sparse_double&>() * std::declval<fdapde::Vector<int, 4>>());
    // mixed-scalar vector multiplication returns an owning dynamic double vector
    static_assert(std::is_same_v<vector_product, fdapde::Vector<double, fdapde::Dynamic>>);
    using matrix_product = decltype(std::declval<const sparse_double&>() * std::declval<fdapde::Matrix<int, 4, 2>>());
    // mixed-scalar matrix multiplication returns an owning dynamic double matrix
    static_assert(std::is_same_v<matrix_product, fdapde::Matrix<double, fdapde::Dynamic, fdapde::Dynamic>>);
    const sparse_double wide(0, std::numeric_limits<int>::max());
    // the transpose rejects a row count that cannot accommodate its terminal CSR offset
    EXPECT_THROW(static_cast<void>(wide.transpose()), std::length_error);
    sparse_double source(
      2, 2,
      {
        {0, 1, 3.0}
    });
    const auto transposed = source.transpose();
    const auto product = source * fdapde::Vector<double, 2>({2.0, 4.0});
    source.value_ref(0, 1) = 9.0;
    // a source mutation cannot change the independent transpose coefficient
    EXPECT_DOUBLE_EQ(transposed.coeff(1, 0), 3.0);
    // a source mutation cannot change the already evaluated dense product
    EXPECT_DOUBLE_EQ(product[0], 12.0);
    const auto diagonal = source.diagonal();
    // a missing first diagonal entry is initialized to zero
    EXPECT_DOUBLE_EQ(diagonal[0], 0.0);
    // a missing final diagonal entry is initialized to zero
    EXPECT_DOUBLE_EQ(diagonal[1], 0.0);
    const double row_values[2] {2.0, 4.0};
    const fdapde::Matrix<double, 1, 2> row_vector(row_values);
    // the row-vector form evaluates two times nine times four without requiring a column view
    EXPECT_DOUBLE_EQ(source.quadratic_form(row_vector), 72.0);
    const auto row_diagonal = sparse_double::from_diagonal(row_vector + row_vector);
    // diagonal construction evaluates the doubled row expression at its second entry
    EXPECT_DOUBLE_EQ(row_diagonal.coeff(1, 1), 8.0);
    const fdapde::Vector<double, 2> vector({2.0, 4.0});
    const fdapde::MatrixView<const double, 2, 1> view(vector.data());
    // a const vector view uses the same coefficient ordering as its owning vector
    EXPECT_DOUBLE_EQ((source * view)[0], 36.0);
    const fdapde::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> no_columns(2, 0);
    const auto empty_product = source * no_columns;
    // a right-hand side with no columns retains the two output rows
    EXPECT_EQ(empty_product.rows(), 2);
    // a right-hand side with no columns produces no output coefficients
    EXPECT_EQ(empty_product.size(), 0);
}

// checks signed and unsigned arithmetic boundaries before overflowing products or reductions are evaluated
TEST(linear_algebra, sparse_integral_operation_overflow) {
    using sparse_int = fdapde::SparseMatrix<int>;
    constexpr int max = std::numeric_limits<int>::max();
    constexpr int min = std::numeric_limits<int>::min();
    const fdapde::Vector<int, 1> two(2);
    const fdapde::Vector<int, 1> minus_one(-1);
    // doubling the largest signed coefficient is rejected before signed multiplication
    EXPECT_THROW(
      static_cast<void>(
        sparse_int(
          1, 1,
          {
            {0, 0, max}
    }) *
        two),
      std::overflow_error);
    // negating the smallest signed coefficient is rejected before signed multiplication
    EXPECT_THROW(
      static_cast<void>(
        sparse_int(
          1, 1,
          {
            {0, 0, min}
    }) *
        minus_one),
      std::overflow_error);
    const sparse_int sum(
      1, 2,
      {
        {0, 0, max},
        {0, 1, 1  }
    });
    // individually valid coefficients cannot overflow while being summed into a row total
    EXPECT_THROW(static_cast<void>(sum.row_sums()), std::overflow_error);
    const fdapde::Vector<int, 2> ones({1, 1});
    // individually valid products cannot overflow their dot-product accumulator
    EXPECT_THROW(static_cast<void>(sum * ones), std::overflow_error);
    const fdapde::Matrix<int, 2, 2> dense_ones(1);
    // the dense matrix kernel checks the same accumulator overflow as the vector kernel
    EXPECT_THROW(static_cast<void>(sum * dense_ones), std::overflow_error);
    const fdapde::Vector<int, 2> zeros({0, 0});
    // the expression path retains the integral accumulator check
    EXPECT_THROW(static_cast<void>(sum * (ones + zeros)), std::overflow_error);
    // quadratic multiplication checks intermediate products before the final reduction
    EXPECT_THROW(
      static_cast<void>(sparse_int(
                          1, 1,
                          {
                            {0, 0, max}
    })
                          .quadratic_form(two)),
      std::overflow_error);
    const sparse_int quadratic_sum(
      2, 2,
      {
        {0, 0, max},
        {1, 1, 1  }
    });
    // quadratic accumulation rejects a sum overflow even when all products fit
    EXPECT_THROW(static_cast<void>(quadratic_sum.quadratic_form(ones)), std::overflow_error);
    const sparse_int negative(
      1, 1,
      {
        {0, 0, min}
    });
    // multiplying the minimum by one reaches the exact lower boundary without throwing
    EXPECT_EQ((negative * fdapde::Vector<int, 1>(1))[0], min);
    // multiplying the minimum by zero returns zero without dividing by zero in validation
    EXPECT_EQ((negative * fdapde::Vector<int, 1>(0))[0], 0);
    // two negative operands with a representable product are accepted
    EXPECT_EQ(
      (sparse_int(
         1, 1,
         {
           {0, 0, -3}
    }) *
       minus_one)[0],
      3);
    const fdapde::Vector<double, 1> floating_two(2.0);
    // promotion to double occurs before multiplication and avoids a false int overflow
    EXPECT_DOUBLE_EQ(
      (sparse_int(
         1, 1,
         {
           {0, 0, max}
    }) *
       floating_two)[0],
      2.0 * max);
    using sparse_unsigned = fdapde::SparseMatrix<unsigned>;
    const sparse_unsigned unsigned_max(
      1, 1,
      {
        {0, 0, std::numeric_limits<unsigned>::max()}
    });
    // unsigned multiplication rejects wraparound at the maximum coefficient
    EXPECT_THROW(static_cast<void>(unsigned_max * fdapde::Vector<unsigned, 1>(2)), std::overflow_error);
    const sparse_unsigned unsigned_sum(
      1, 2,
      {
        {0, 0, std::numeric_limits<unsigned>::max()},
        {0, 1, 1                                   }
    });
    // unsigned row summation rejects wraparound in the reduction
    EXPECT_THROW(static_cast<void>(unsigned_sum.row_sums()), std::overflow_error);
    // a failed reduction leaves the original maximum coefficient intact
    EXPECT_EQ(sum.coeff(0, 0), max);
}

// checks diagonal conversion at integer boundaries, with fractional, nonfinite and narrow input values
TEST(linear_algebra, sparse_diagonal_conversion) {
    using sparse_int = fdapde::SparseMatrix<int>;
    const fdapde::Vector<double, 3> fractions({2.9, -3.9, 0.9});
    const auto truncated = sparse_int::from_diagonal(fractions);
    // a positive fractional coefficient truncates toward zero
    EXPECT_EQ(truncated.coeff(0, 0), 2);
    // a negative fractional coefficient truncates toward zero
    EXPECT_EQ(truncated.coeff(1, 1), -3);
    // a fraction that truncates to zero is omitted from the canonical pattern
    EXPECT_EQ(truncated.non_zeros(), 2);
    const fdapde::Vector<double, 1> out_of_range(std::ldexp(1.0, std::numeric_limits<int>::digits));
    // the first value above the signed range is rejected before floating-to-integer conversion
    EXPECT_THROW(static_cast<void>(sparse_int::from_diagonal(out_of_range)), std::overflow_error);
    const fdapde::Vector<double, 1> nan(std::numeric_limits<double>::quiet_NaN());
    // a NaN cannot be converted into an integral diagonal coefficient
    EXPECT_THROW(static_cast<void>(sparse_int::from_diagonal(nan)), std::overflow_error);
    const fdapde::Vector<double, 1> infinity(std::numeric_limits<double>::infinity());
    // infinity cannot be converted into an integral diagonal coefficient
    EXPECT_THROW(static_cast<void>(sparse_int::from_diagonal(infinity)), std::overflow_error);
    const fdapde::Vector<unsigned long long, 1> large(std::numeric_limits<unsigned long long>::max());
    // integral narrowing rejects a coefficient above the signed destination range
    EXPECT_THROW(static_cast<void>(sparse_int::from_diagonal(large)), std::overflow_error);
    const fdapde::Vector<int, 1> negative(-1);
    // integral conversion into an unsigned diagonal rejects negative values
    EXPECT_THROW(static_cast<void>(fdapde::SparseMatrix<unsigned>::from_diagonal(negative)), std::overflow_error);
    const auto narrow = fdapde::SparseMatrix<char>::from_diagonal(fdapde::Vector<int, 1>(7));
    // a representable plain-char destination supports checked integral conversion
    EXPECT_EQ(narrow.coeff(0, 0), 7);
}

// compares dimensions and complete ordered row contents against a sparse snapshot
template <typename Scalar>
void expect_same_sparse(const fdapde::SparseMatrix<Scalar>& lhs, const fdapde::SparseMatrix<Scalar>& rhs) {
    // the snapshot comparison requires identical row counts
    ASSERT_EQ(lhs.rows(), rhs.rows());
    // the snapshot comparison requires identical column counts
    ASSERT_EQ(lhs.cols(), rhs.cols());
    // the snapshot comparison checks stored zeros through the stored-entry count
    ASSERT_EQ(lhs.non_zeros(), rhs.non_zeros());
    for (int row = 0; row < lhs.rows(); ++row) {
        // every ordered row must preserve the same column indices and coefficients
        EXPECT_EQ(collect_row(lhs, row), collect_row(rhs, row));
    }
}

// checks unit constraints, symmetry, duplicate indices, zero pruning and validation before mutation
TEST(linear_algebra, sparse_matrix_constraint_rebuilding) {
    const sparse_double source(
      4, 4,
      {
        {0, 0, 2.0 },
        {0, 1, 3.0 },
        {0, 3, 4.0 },
        {1, 0, 5.0 },
        {1, 2, 6.0 },
        {2, 1, 7.0 },
        {2, 2, 8.0 },
        {2, 3, 9.0 },
        {3, 0, 10.0},
        {3, 2, 11.0},
        {3, 3, 12.0}
    });

    sparse_double constrained(source);
    constrained.rebuild_with_constraints(std::vector<int> {1});
    // clearing column one removes only its entry from the first row
    EXPECT_EQ(
      collect_row(constrained, 0), (std::vector<std::pair<int, double>> {
                                     {0, 2.0},
                                     {3, 4.0}
    }));
    // the constrained row becomes a single unit diagonal even when that diagonal was missing
    EXPECT_EQ(
      collect_row(constrained, 1), (std::vector<std::pair<int, double>> {
                                     {1, 1.0}
    }));
    // clearing column one preserves the remaining sorted entries in row two
    EXPECT_EQ(
      collect_row(constrained, 2), (std::vector<std::pair<int, double>> {
                                     {2, 8.0},
                                     {3, 9.0}
    }));
    // an unrelated row retains every original coefficient
    EXPECT_EQ(
      collect_row(constrained, 3), (std::vector<std::pair<int, double>> {
                                     {0, 10.0},
                                     {2, 11.0},
                                     {3, 12.0}
    }));
    // constraining a copy leaves the source column coefficient unchanged
    EXPECT_DOUBLE_EQ(source.coeff(0, 1), 3.0);
    // constraining a copy leaves the source row coefficient unchanged
    EXPECT_DOUBLE_EQ(source.coeff(1, 2), 6.0);

    sparse_double duplicated(source);
    duplicated.rebuild_with_constraints(std::vector<int> {3, 1, 3});
    sparse_double permuted(source);
    permuted.rebuild_with_constraints(std::vector<int> {1, 3});
    // duplicate and permuted constraint indices yield identical compressed matrices
    expect_same_sparse(duplicated, permuted);

    const sparse_double symmetric = sparse_double(
                                      4, 4,
                                      {
                                        {0, 0, 4.0 },
                                        {1, 0, 1.0 },
                                        {1, 1, 5.0 },
                                        {2, 0, 2.0 },
                                        {2, 1, 3.0 },
                                        {2, 2, 6.0 },
                                        {3, 0, 7.0 },
                                        {3, 1, 8.0 },
                                        {3, 2, 9.0 },
                                        {3, 3, 10.0}
    })
                                      .symmetric_expanded(fdapde::Lower);
    sparse_double symmetric_constrained(symmetric);
    symmetric_constrained.rebuild_with_constraints(std::vector<int> {1, 3});
    for (int row = 0; row < symmetric_constrained.rows(); ++row) {
        for (int col = 0; col < symmetric_constrained.cols(); ++col) {
            // simultaneous row and column removal preserves coefficient symmetry
            EXPECT_DOUBLE_EQ(symmetric_constrained.coeff(row, col), symmetric_constrained.coeff(col, row));
        }
    }
    // the first selected diagonal is replaced by one
    EXPECT_DOUBLE_EQ(symmetric_constrained.coeff(1, 1), 1.0);
    // the second selected diagonal is replaced by one
    EXPECT_DOUBLE_EQ(symmetric_constrained.coeff(3, 3), 1.0);

    sparse_double no_op(source);
    no_op.value_ref(0, 1) = 0.0;
    const sparse_double no_op_snapshot(no_op);
    no_op.rebuild_with_constraints(std::vector<int> {});
    // an empty constraint list preserves the exact original pattern and values
    expect_same_sparse(no_op, no_op_snapshot);
    // an empty constraint list retains an explicitly stored zero
    EXPECT_TRUE(no_op.contains(0, 1));

    sparse_double retained_zero(
      3, 3,
      {
        {0, 1, 4.0},
        {1, 1, 5.0},
        {1, 2, 6.0},
        {2, 1, 7.0},
        {2, 2, 8.0}
    });
    retained_zero.value_ref(0, 1) = 0.0;
    retained_zero.value_ref(1, 1) = 0.0;
    retained_zero.rebuild_with_constraints(std::vector<int> {1});
    // the rebuilt constrained row contains its diagonal entry
    EXPECT_TRUE(retained_zero.contains(1, 1));
    // a stored zero on the selected diagonal is replaced by one
    EXPECT_DOUBLE_EQ(retained_zero.coeff(1, 1), 1.0);
    // an explicitly zero entry in the constrained column is removed
    EXPECT_FALSE(retained_zero.contains(0, 1));
    // a nonzero entry in the constrained column is removed
    EXPECT_FALSE(retained_zero.contains(2, 1));
    // the unconstrained final row retains only its unaffected diagonal
    EXPECT_EQ(
      collect_row(retained_zero, 2), (std::vector<std::pair<int, double>> {
                                       {2, 8.0}
    }));

    sparse_double failed(source);
    // a negative index after a valid index raises the bounds error
    EXPECT_THROW(failed.rebuild_with_constraints(std::vector<int> {1, -1}), std::out_of_range);
    // negative-index rejection leaves every original row unchanged
    expect_same_sparse(failed, source);
    // an index equal to the row count raises the bounds error
    EXPECT_THROW(failed.rebuild_with_constraints(std::vector<int> {1, 4}), std::out_of_range);
    // upper-bound rejection leaves every original row unchanged
    expect_same_sparse(failed, source);

    sparse_double rectangular = make_rectangular_fixture();
    const sparse_double rectangular_snapshot(rectangular);
    // a nonempty constraint list rejects a rectangular matrix
    EXPECT_THROW(rectangular.rebuild_with_constraints(std::vector<int> {1}), std::invalid_argument);
    // shape rejection preserves the rectangular matrix exactly
    expect_same_sparse(rectangular, rectangular_snapshot);
}

// checks empty lists, complete constraints, repeated rebuilding and zero pruning outside selected rows
TEST(linear_algebra, sparse_constraint_boundaries) {
    sparse_double rectangular = make_rectangular_fixture();
    const auto rectangular_snapshot = rectangular;
    const double* borrowed = &rectangular.value_ref(0, 0);
    rectangular.rebuild_with_constraints({});
    // an empty list accepts rectangular shapes without changing any row contents
    expect_same_sparse(rectangular, rectangular_snapshot);
    // a no-op preserves the address of a previously borrowed coefficient
    EXPECT_EQ(&rectangular.value_ref(0, 0), borrowed);
    sparse_double empty;
    // a zero-by-zero matrix accepts the empty constraint list
    EXPECT_NO_THROW(empty.rebuild_with_constraints({}));
    // a zero-by-zero matrix rejects its first nonexistent row
    EXPECT_THROW(empty.rebuild_with_constraints({0}), std::out_of_range);
    fdapde::SparseMatrix<int> all(
      3, 3,
      {
        {0, 2, 5 },
        {2, 1, -7}
    });
    all.rebuild_with_constraints({2, 0, 1, 0});
    // constraining every row creates exactly three unit diagonals, including previously empty rows
    EXPECT_EQ(all.non_zeros(), 3);
    for (int row = 0; row < 3; ++row) {
        // each fully constrained integer row contains only its unit diagonal
        EXPECT_EQ(
          collect_row(all, row), (std::vector<std::pair<int, int>> {
                                   {row, 1}
        }));
    }
    const auto all_snapshot = all;
    all.rebuild_with_constraints({0, 1, 2});
    // rebuilding an already constrained matrix is structurally and numerically idempotent
    expect_same_sparse(all, all_snapshot);
    sparse_double zeros(
      3, 3,
      {
        {0, 0, 2.0},
        {2, 1, 3.0}
    });
    zeros.value_ref(2, 1) = 0.0;
    zeros.rebuild_with_constraints({0});
    // a nonempty rebuild prunes stored zeros even outside the selected row and column
    EXPECT_FALSE(zeros.contains(2, 1));
    // the rebuilt pattern contains only the selected unit diagonal
    EXPECT_EQ(zeros.non_zeros(), 1);
    sparse_double nonfinite(
      2, 2,
      {
        {0, 1, std::numeric_limits<double>::infinity() },
        {1, 0, std::numeric_limits<double>::quiet_NaN()}
    });
    nonfinite.rebuild_with_constraints({0});
    // structural elimination removes constrained nonfinite entries without multiplying them by zero
    EXPECT_EQ(nonfinite.non_zeros(), 1);
    // nonfinite source couplings do not contaminate the restored unit diagonal
    EXPECT_DOUBLE_EQ(nonfinite.coeff(0, 0), 1.0);
}

/// @brief injects a copy failure after a controlled number of successful coefficient copies
struct constraint_copy_probe {
    int value = 0;
    static inline int copies_left = -1;
    /// @brief constructs a coefficient without consuming the copy budget
    constraint_copy_probe(int coefficient = 0) : value(coefficient) { }
    /// @brief copies a coefficient or throws when the enabled budget is exhausted
    constraint_copy_probe(const constraint_copy_probe& other) : value(other.value) {
        fdapde_strong_assert(copies_left != 0, std::runtime_error, "injected coefficient copy failure");
        if (copies_left > 0) --copies_left;
    }
    /// @brief assigns a coefficient without affecting the copy-construction probe
    constraint_copy_probe& operator=(const constraint_copy_probe&) = default;
    /// @brief compares coefficient values during sparse zero pruning
    friend bool operator==(const constraint_copy_probe&, const constraint_copy_probe&) = default;
    /// @brief adds duplicate input coefficients during fixture construction
    friend constraint_copy_probe operator+(const constraint_copy_probe& lhs, const constraint_copy_probe& rhs) {
        return constraint_copy_probe(lhs.value + rhs.value);
    }
};

// checks that a coefficient-copy exception cannot publish a partially rebuilt matrix
TEST(linear_algebra, sparse_constraint_copy_failure) {
    fdapde::SparseMatrix<constraint_copy_probe> matrix(
      3, 3,
      {
        {0, 2, 4},
        {1, 1, 5},
        {2, 2, 6}
    });
    const auto snapshot = matrix;
    constraint_copy_probe::copies_left = 1;
    // copying the retained second row fails after the replacement already contains its first unit diagonal
    EXPECT_THROW(matrix.rebuild_with_constraints({0}), std::runtime_error);
    constraint_copy_probe::copies_left = -1;
    // the exception leaves the complete source pattern and coefficients equal to the saved snapshot
    expect_same_sparse(matrix, snapshot);
}

template <int StorageOrder> void check_dense_lumping() {
    const fdapde::Matrix<double, 3, 3, StorageOrder> matrix({1.0, 2.0, -3.0, 4.0, -1.0, 2.0, 0.5, 1.5, 2.0});
    const auto lumped = fdapde::lump(matrix);
    EXPECT_EQ(lumped.rows(), 3);
    EXPECT_EQ(lumped.cols(), 3);
    EXPECT_DOUBLE_EQ(lumped[0], 0.0);
    EXPECT_DOUBLE_EQ(lumped[1], 5.0);
    EXPECT_DOUBLE_EQ(lumped[2], 4.0);
    EXPECT_DOUBLE_EQ(lumped(0, 1), 0.0);

    const auto expression = fdapde::lump(matrix + matrix);
    EXPECT_DOUBLE_EQ(expression[0], 0.0);
    EXPECT_DOUBLE_EQ(expression[1], 10.0);
    EXPECT_DOUBLE_EQ(expression[2], 8.0);

    const auto temporary = fdapde::lump(fdapde::Matrix<double, 2, 2, StorageOrder>({1.0, 2.0, 3.0, 4.0}));
    EXPECT_DOUBLE_EQ(temporary[0], 3.0);
    EXPECT_DOUBLE_EQ(temporary[1], 7.0);
}

void check_matrix_lumping() {
    check_dense_lumping<fdapde::RowMajor>();
    check_dense_lumping<fdapde::ColMajor>();

    const sparse_double source(
      3, 3,
      {
        {0, 0, 2.0 },
        {0, 2, -2.0},
        {1, 1, 3.0 },
        {2, 0, -1.0},
        {2, 2, 4.0 }
    });
    const auto sparse_lumped = fdapde::lump(source);
    EXPECT_EQ(sparse_lumped.rows(), 3);
    EXPECT_EQ(sparse_lumped.cols(), 3);
    EXPECT_EQ(sparse_lumped.non_zeros(), 2);
    EXPECT_TRUE(sparse_lumped.row(0).empty());
    EXPECT_DOUBLE_EQ(sparse_lumped.coeff(1, 1), 3.0);
    EXPECT_DOUBLE_EQ(sparse_lumped.coeff(2, 2), 3.0);
    EXPECT_EQ(source.non_zeros(), 5);
    EXPECT_DOUBLE_EQ(source.coeff(0, 2), -2.0);

    const auto sparse_empty = fdapde::lump(sparse_double());
    EXPECT_EQ(sparse_empty.rows(), 0);
    EXPECT_EQ(sparse_empty.cols(), 0);
    EXPECT_EQ(sparse_empty.non_zeros(), 0);
    const auto dense_empty = fdapde::lump(fdapde::Matrix<double, fdapde::Dynamic, fdapde::Dynamic>(0, 0));
    EXPECT_EQ(dense_empty.rows(), 0);
    EXPECT_EQ(dense_empty.cols(), 0);

    EXPECT_THROW(static_cast<void>(fdapde::lump(make_rectangular_fixture())), std::invalid_argument);
    const fdapde::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> rectangular(2, 3);
    EXPECT_THROW(static_cast<void>(fdapde::lump(rectangular)), std::invalid_argument);
}

TEST(linear_algebra, matrix_lumping) { check_matrix_lumping(); }

}   // namespace
