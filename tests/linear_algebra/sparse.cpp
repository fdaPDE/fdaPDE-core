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

}   // namespace
