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

using fdapde::Matrix;
using fdapde::SparseMatrix;

template <typename Dense>
concept permits_left_temporary_kronecker = requires(Dense& named) { fdapde::kronecker(Dense {}, named); };

template <typename Dense>
concept permits_right_temporary_kronecker = requires(Dense& named) { fdapde::kronecker(named, Dense {}); };

using dense_matrix = Matrix<double, 2, 2>;
using promoted_sparse_product =
  decltype(fdapde::kron(std::declval<const SparseMatrix<int>&>(), std::declval<const SparseMatrix<double>&>()));

static_assert(!permits_left_temporary_kronecker<dense_matrix>);
static_assert(!permits_right_temporary_kronecker<dense_matrix>);
static_assert(std::is_same_v<promoted_sparse_product, SparseMatrix<double>>);

template <typename Scalar>
void expect_sparse(
  const SparseMatrix<Scalar>& actual, int rows, int cols, int non_zeros, const std::vector<Scalar>& expected) {
    ASSERT_EQ(actual.rows(), rows);
    ASSERT_EQ(actual.cols(), cols);
    ASSERT_EQ(actual.non_zeros(), non_zeros);
    ASSERT_EQ(expected.size(), static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            EXPECT_EQ(actual.coeff(i, j), expected[static_cast<std::size_t>(i * cols + j)]);
        }
    }
}

TEST(linear_algebra, kronecker_dense_square) {
    const Matrix<double, 2, 2> lhs({1.0, 0.0, 0.0, 1.0});
    const Matrix<double, 2, 2> rhs({1.0, 2.0, 3.0, 4.0});
    const Matrix<double, 4, 4> expected({
      1.0,
      2.0,
      0.0,
      0.0,
      3.0,
      4.0,
      0.0,
      0.0,
      0.0,
      0.0,
      1.0,
      2.0,
      0.0,
      0.0,
      3.0,
      4.0,
    });

    EXPECT_EQ((Matrix<double, 4, 4>(fdapde::kron(lhs, rhs))), expected);
    EXPECT_EQ((Matrix<double, 4, 4>(fdapde::kronecker(lhs, rhs))), expected);
}

TEST(linear_algebra, kronecker_dense_rectangular) {
    const Matrix<double, 2, 3, fdapde::ColMajor> lhs({1.0, 2.0, 0.0, 4.0, 0.0, 1.0});
    const Matrix<double, 2, 2> rhs({1.0, 1.0, 0.0, 1.0});
    const Matrix<double, 4, 6> expected({
      1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0,
      4.0, 4.0, 0.0, 0.0, 1.0, 1.0, 0.0, 4.0, 0.0, 0.0, 0.0, 1.0,
    });

    EXPECT_EQ((Matrix<double, 4, 6>(fdapde::kron(lhs, rhs))), expected);

    const fdapde::ZeroMatrix<double, fdapde::Dynamic, fdapde::Dynamic> wide(1, std::numeric_limits<int>::max());
    const fdapde::ZeroMatrix<double, fdapde::Dynamic, fdapde::Dynamic> two_columns(1, 2);
    const auto overflowing = fdapde::kron(wide, two_columns);
    EXPECT_THROW(static_cast<void>(overflowing.cols()), std::length_error);
}

TEST(linear_algebra, kronecker_sparse_square) {
    const SparseMatrix<double> lhs(
      3, 3,
      {
        {0, 0, 1.0},
        {1, 1, 1.0},
        {2, 2, 1.0}
    });
    const SparseMatrix<double> rhs(
      3, 3,
      {
        {0, 0, 2.0},
        {1, 0, 1.0},
        {1, 1, 3.0},
        {2, 2, 1.0}
    });
    const std::vector<double> expected {
      2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0,
    };

    expect_sparse(fdapde::kron(lhs, rhs), 9, 9, 12, expected);
    expect_sparse(fdapde::kronecker(lhs, rhs), 9, 9, 12, expected);
    EXPECT_EQ(lhs.non_zeros(), 3);
    EXPECT_EQ(rhs.non_zeros(), 4);
}

TEST(linear_algebra, kronecker_sparse_rectangular) {
    SparseMatrix<int> lhs(
      2, 3,
      {
        {0, 0, 2},
        {1, 0, 1},
        {1, 1, 3},
        {1, 2, 1}
    });
    const SparseMatrix<double> rhs(
      3, 2,
      {
        {0, 0, 5.0},
        {0, 1, 6.0},
        {1, 1, 1.0}
    });
    const std::vector<double> expected {
      10.0, 12.0, 0.0,  0.0,  0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      5.0,  6.0,  15.0, 18.0, 5.0, 6.0, 0.0, 1.0, 0.0, 3.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
    };

    expect_sparse(fdapde::kron(lhs, rhs), 6, 6, 12, expected);

    lhs.value_ref(0, 0) = 0;
    const auto zero_elided = fdapde::kron(
      lhs, SparseMatrix<double>(
             1, 1,
             {
               {0, 0, 2.0}
    }));
    EXPECT_EQ(zero_elided.rows(), 2);
    EXPECT_EQ(zero_elided.cols(), 3);
    EXPECT_EQ(zero_elided.non_zeros(), 3);
    EXPECT_FALSE(zero_elided.contains(0, 0));

    const auto empty = fdapde::kron(SparseMatrix<int>(0, 2), SparseMatrix<double>(3, 4));
    EXPECT_EQ(empty.rows(), 0);
    EXPECT_EQ(empty.cols(), 8);
    EXPECT_EQ(empty.non_zeros(), 0);

    const auto empty_columns = fdapde::kron(SparseMatrix<int>(2, 3), SparseMatrix<double>(4, 0));
    EXPECT_EQ(empty_columns.rows(), 8);
    EXPECT_EQ(empty_columns.cols(), 0);
    EXPECT_EQ(empty_columns.non_zeros(), 0);

    const SparseMatrix<int> wide_sparse(1, std::numeric_limits<int>::max());
    const SparseMatrix<int> two_sparse_columns(1, 2);
    EXPECT_THROW(static_cast<void>(fdapde::kron(wide_sparse, two_sparse_columns)), std::length_error);
}

}   // namespace
