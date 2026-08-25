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

#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

template <typename Matrix>
concept has_inserting_coeff_ref = requires(Matrix& matrix) { matrix.coeffRef(0, 0); };

using sparse_double = fdapde::SparseMatrix<double>;
using sparse_entry = decltype(*std::declval<const sparse_double&>().row(0).begin());

static_assert(std::is_same_v<typename sparse_double::Scalar, double>);
static_assert(std::is_same_v<typename sparse_double::Index, int>);
static_assert(std::is_same_v<decltype(std::declval<sparse_entry>().value()), const double&>);
static_assert(!has_inserting_coeff_ref<sparse_double>);

template <typename Scalar>
std::vector<std::pair<int, Scalar>> collect_row(const fdapde::SparseMatrix<Scalar>& matrix, int row) {
    std::vector<std::pair<int, Scalar>> result;
    for (const auto entry : matrix.row(row)) result.emplace_back(entry.column(), entry.value());
    return result;
}

void check_sparse_construction_and_access() {
    fdapde::Triplet<double> mutable_triplet(2, 3, 1.0);
    mutable_triplet.value() = 4.0;
    EXPECT_EQ(mutable_triplet.row(), 2);
    EXPECT_EQ(mutable_triplet.col(), 3);
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

    EXPECT_EQ(matrix.rows(), 3);
    EXPECT_EQ(matrix.cols(), 4);
    EXPECT_EQ(matrix.non_zeros(), 3);
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 1), 5.0);
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 2), 0.0);
    EXPECT_DOUBLE_EQ(matrix.coeff(2, 3), 4.0);
    EXPECT_FALSE(matrix.contains(0, 2));
    EXPECT_TRUE(matrix.contains(2, 3));
    EXPECT_EQ(
      collect_row(matrix, 0), (std::vector<std::pair<int, double>> {
                                {0, 1.0},
                                {1, 5.0}
    }));
    EXPECT_TRUE(matrix.row(1).empty());
    EXPECT_EQ(
      collect_row(matrix, 2), (std::vector<std::pair<int, double>> {
                                {3, 4.0}
    }));

    const sparse_double repeated(3, 4, triplets);
    for (int row = 0; row < matrix.rows(); ++row) { EXPECT_EQ(collect_row(repeated, row), collect_row(matrix, row)); }

    const sparse_double ordered_sum(
      1, 1,
      {
        {0, 0, 1.0e16 },
        {0, 0, -1.0e16},
        {0, 0, 1.0    }
    });
    EXPECT_DOUBLE_EQ(ordered_sum.coeff(0, 0), 1.0);

    matrix.value_ref(0, 1) = 7.0;
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 1), 7.0);
    EXPECT_THROW(static_cast<void>(matrix.value_ref(0, 2)), std::out_of_range);
    matrix.value_ref(0, 1) = 0.0;
    EXPECT_TRUE(matrix.contains(0, 1));
    EXPECT_EQ(matrix.non_zeros(), 3);
    matrix.rebuild({
      {0, 0, 1.0},
      {0, 1, 0.0},
      {2, 3, 4.0}
    });
    EXPECT_FALSE(matrix.contains(0, 1));
    EXPECT_EQ(matrix.non_zeros(), 2);

    sparse_double copy(matrix);
    copy.value_ref(0, 0) = 9.0;
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 0), 1.0);
    EXPECT_DOUBLE_EQ(copy.coeff(0, 0), 9.0);

    sparse_double assigned(1, 1);
    assigned = matrix;
    EXPECT_EQ(assigned.rows(), 3);
    EXPECT_EQ(assigned.cols(), 4);
    EXPECT_DOUBLE_EQ(assigned.coeff(2, 3), 4.0);

    sparse_double moved(std::move(copy));
    EXPECT_DOUBLE_EQ(moved.coeff(0, 0), 9.0);
    sparse_double move_assigned;
    move_assigned = std::move(assigned);
    EXPECT_EQ(move_assigned.rows(), 3);
    EXPECT_EQ(move_assigned.cols(), 4);
    EXPECT_DOUBLE_EQ(move_assigned.coeff(2, 3), 4.0);
    copy.resize(0, 0);
    EXPECT_EQ(copy.rows(), 0);
    EXPECT_EQ(copy.cols(), 0);
    EXPECT_EQ(copy.non_zeros(), 0);

    moved.resize(3, 4);
    EXPECT_EQ(moved.rows(), 3);
    EXPECT_EQ(moved.cols(), 4);
    EXPECT_EQ(moved.non_zeros(), 0);
    EXPECT_DOUBLE_EQ(moved.coeff(0, 0), 0.0);
}

void check_sparse_empty_and_integral_contracts() {
    const sparse_double empty;
    EXPECT_EQ(empty.rows(), 0);
    EXPECT_EQ(empty.cols(), 0);
    EXPECT_EQ(empty.non_zeros(), 0);

    const sparse_double zero_rows(0, 5);
    const sparse_double zero_cols(3, 0);
    EXPECT_EQ(zero_rows.rows(), 0);
    EXPECT_EQ(zero_rows.cols(), 5);
    EXPECT_EQ(zero_cols.rows(), 3);
    EXPECT_EQ(zero_cols.cols(), 0);
    EXPECT_TRUE(zero_cols.row(0).empty());

    const fdapde::SparseMatrix<int> adjacency(
      3, 3,
      std::vector<fdapde::Triplet<int>> {
        {0, 1, 1},
        {2, 0, 1},
        {0, 1, 2},
        {1, 2, 0}
    });
    EXPECT_EQ(adjacency.non_zeros(), 2);
    EXPECT_EQ(adjacency.coeff(0, 1), 3);
    EXPECT_EQ(adjacency.coeff(2, 0), 1);

    const fdapde::SparseMatrix<int> wide(
      1, std::numeric_limits<int>::max(),
      std::vector<fdapde::Triplet<int>> {
        {0, std::numeric_limits<int>::max() - 1, 5}
    });
    EXPECT_EQ(wide.cols(), std::numeric_limits<int>::max());
    EXPECT_EQ(wide.coeff(0, std::numeric_limits<int>::max() - 1), 5);
}

void check_sparse_failure_contracts() {
    EXPECT_THROW(static_cast<void>(sparse_double(-1, 2)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(sparse_double(2, -1)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(sparse_double(std::numeric_limits<int>::max(), 0)), std::length_error);

    EXPECT_THROW(
      static_cast<void>(sparse_double(
        2, 2,
        std::vector<fdapde::Triplet<double>> {
          {-1, 0, 1.0}
    })),
      std::out_of_range);
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
    EXPECT_THROW(static_cast<void>(matrix.coeff(-1, 0)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(matrix.coeff(0, 2)), std::out_of_range);
    EXPECT_THROW(static_cast<void>(matrix.row(2)), std::out_of_range);
    EXPECT_THROW(matrix.resize(-1, 2), std::invalid_argument);
    EXPECT_EQ(matrix.rows(), 2);
    EXPECT_EQ(matrix.cols(), 2);
    EXPECT_EQ(matrix.non_zeros(), 2);
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 0), 3.0);

    EXPECT_THROW(
      matrix.rebuild(
        std::vector<fdapde::Triplet<double>> {
          {0, 0, 8.0},
          {2, 0, 1.0}
    }),
      std::out_of_range);
    EXPECT_EQ(matrix.non_zeros(), 2);
    EXPECT_DOUBLE_EQ(matrix.coeff(0, 0), 3.0);

    fdapde::SparseMatrix<int> integral(
      1, 1,
      std::vector<fdapde::Triplet<int>> {
        {0, 0, 7}
    });
    EXPECT_THROW(
      integral.rebuild(
        std::vector<fdapde::Triplet<int>> {
          {0, 0, std::numeric_limits<int>::max()},
          {0, 0, 1                              }
    }),
      std::overflow_error);
    EXPECT_EQ(integral.non_zeros(), 1);
    EXPECT_EQ(integral.coeff(0, 0), 7);

    matrix.rebuild(
      std::vector<fdapde::Triplet<double>> {
        {1, 0, 6.0},
        {0, 1, 2.0}
    });
    EXPECT_EQ(matrix.non_zeros(), 2);
    EXPECT_EQ(
      collect_row(matrix, 0), (std::vector<std::pair<int, double>> {
                                {1, 2.0}
    }));
    EXPECT_EQ(
      collect_row(matrix, 1), (std::vector<std::pair<int, double>> {
                                {0, 6.0}
    }));
}

TEST(linear_algebra, sparse_matrix) {
    check_sparse_construction_and_access();
    check_sparse_empty_and_integral_contracts();
    check_sparse_failure_contracts();
}

}   // namespace
