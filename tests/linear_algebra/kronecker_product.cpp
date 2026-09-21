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

using fdapde::Matrix;
using fdapde::SparseMatrix;

template <typename Dense>
concept permits_left_temporary_kronecker = requires(Dense& named) { fdapde::kron(Dense {}, named); };

template <typename Dense>
concept permits_right_temporary_kronecker = requires(Dense& named) { fdapde::kron(named, Dense {}); };

using dense_matrix = Matrix<double, 2, 2>;
using promoted_sparse_product =
  decltype(fdapde::kron(std::declval<const SparseMatrix<int>&>(), std::declval<const SparseMatrix<double>&>()));

// a borrowed dense expression must reject a temporary left owner
static_assert(!permits_left_temporary_kronecker<dense_matrix>);
// a borrowed dense expression must reject a temporary right owner
static_assert(!permits_right_temporary_kronecker<dense_matrix>);
// mixed integral and floating sparse coefficients promote to an owning double CSR
static_assert(std::is_same_v<promoted_sparse_product, SparseMatrix<double>>);

template <typename Scalar>
void expect_sparse(
  const SparseMatrix<Scalar>& actual, int rows, int cols, int non_zeros, const std::vector<Scalar>& expected) {
    // the row count matches the independently specified output shape
    ASSERT_EQ(actual.rows(), rows);
    // the column count matches the independently specified output shape
    ASSERT_EQ(actual.cols(), cols);
    // the stored pattern has exactly the expected number of nonzero products
    ASSERT_EQ(actual.non_zeros(), non_zeros);
    // the row-major oracle supplies a coefficient for every output position
    ASSERT_EQ(expected.size(), static_cast<std::size_t>(rows) * static_cast<std::size_t>(cols));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // every stored or absent position matches the explicit row-major oracle
            EXPECT_EQ(actual.coeff(i, j), expected[static_cast<std::size_t>(i * cols + j)]);
        }
    }
}

// the historical identity tensor product reproduces two explicit diagonal blocks
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

    // materializing the dense expression reproduces the historical 4-by-4 coefficient oracle
    EXPECT_EQ((Matrix<double, 4, 4>(fdapde::kron(lhs, rhs))), expected);
}

// mixed dense storage orders preserve the historical rectangular tensor product and reject oversized shapes
TEST(linear_algebra, kronecker_dense_rectangular) {
    const Matrix<double, 2, 3, fdapde::ColMajor> lhs({1.0, 2.0, 0.0, 4.0, 0.0, 1.0});
    const Matrix<double, 2, 2> rhs({1.0, 1.0, 0.0, 1.0});
    const Matrix<double, 4, 6> expected({
      1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 2.0, 0.0, 0.0,
      4.0, 4.0, 0.0, 0.0, 1.0, 1.0, 0.0, 4.0, 0.0, 0.0, 0.0, 1.0,
    });

    // column-major input produces the historical row-major 4-by-6 coefficient oracle
    EXPECT_EQ((Matrix<double, 4, 6>(fdapde::kron(lhs, rhs))), expected);

    const fdapde::ZeroMatrix<double, fdapde::Dynamic, fdapde::Dynamic> wide(1, std::numeric_limits<int>::max());
    const fdapde::ZeroMatrix<double, fdapde::Dynamic, fdapde::Dynamic> two_columns(1, 2);
    const auto overflowing = fdapde::kron(wide, two_columns);
    // querying an unrepresentable dense output width throws before index multiplication
    EXPECT_THROW(static_cast<void>(overflowing.cols()), std::length_error);
}

// the historical sparse identity tensor product reproduces three explicit diagonal blocks
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

    // all 81 coefficients and the 12-entry pattern match the historical block-diagonal oracle
    expect_sparse(fdapde::kron(lhs, rhs), 9, 9, 12, expected);
    // the left input retains its original three stored identity coefficients
    EXPECT_EQ(lhs.non_zeros(), 3);
    // the right input retains its original four stored coefficients
    EXPECT_EQ(rhs.non_zeros(), 4);
}

// mixed sparse scalars preserve rectangular coefficients, prune zeros and retain empty output shapes
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

    // all 36 coefficients and the 12-entry pattern match the historical rectangular oracle
    expect_sparse(fdapde::kron(lhs, rhs), 6, 6, 12, expected);

    lhs.value_ref(0, 0) = 0;
    const auto zero_elided = fdapde::kron(
      lhs, SparseMatrix<double>(
             1, 1,
             {
               {0, 0, 2.0}
    }));
    // a scalar right factor preserves the two input rows
    EXPECT_EQ(zero_elided.rows(), 2);
    // a scalar right factor preserves the three input columns
    EXPECT_EQ(zero_elided.cols(), 3);
    // the explicitly zeroed input contributes no stored output product
    EXPECT_EQ(zero_elided.non_zeros(), 3);
    // the zero product is absent from the resulting CSR pattern
    EXPECT_FALSE(zero_elided.contains(0, 0));

    const auto empty = fdapde::kron(SparseMatrix<int>(0, 2), SparseMatrix<double>(3, 4));
    // a zero-row operand produces zero output rows
    EXPECT_EQ(empty.rows(), 0);
    // zero output rows do not discard the product of column counts
    EXPECT_EQ(empty.cols(), 8);
    // a zero-row output has no stored coefficients
    EXPECT_EQ(empty.non_zeros(), 0);

    const auto empty_columns = fdapde::kron(SparseMatrix<int>(2, 3), SparseMatrix<double>(4, 0));
    // zero output columns do not discard the product of row counts
    EXPECT_EQ(empty_columns.rows(), 8);
    // a zero-column operand produces zero output columns
    EXPECT_EQ(empty_columns.cols(), 0);
    // a zero-column output has no stored coefficients
    EXPECT_EQ(empty_columns.non_zeros(), 0);

    const SparseMatrix<int> wide_sparse(1, std::numeric_limits<int>::max());
    const SparseMatrix<int> two_sparse_columns(1, 2);
    // unrepresentable sparse output width throws before allocating storage
    EXPECT_THROW(static_cast<void>(fdapde::kron(wide_sparse, two_sparse_columns)), std::length_error);
}

// integral tensor coefficients reject overflow while boundary products and temporary owners remain valid
TEST(linear_algebra, kronecker_checked_integral_products) {
    const SparseMatrix<int> maximum(
      1, 1,
      {
        {0, 0, std::numeric_limits<int>::max()}
    });
    const SparseMatrix<int> minimum(
      1, 1,
      {
        {0, 0, std::numeric_limits<int>::min()}
    });
    // a positive product beyond INT_MAX is rejected before signed multiplication
    EXPECT_THROW(
      fdapde::kron(
        maximum, SparseMatrix<int>(
                   1, 1,
                   {
                     {0, 0, 2}
    })),
      std::overflow_error);
    // negating INT_MIN cannot be represented in the promoted signed scalar
    EXPECT_THROW(
      fdapde::kron(
        minimum, SparseMatrix<int>(
                   1, 1,
                   {
                     {0, 0, -1}
    })),
      std::overflow_error);
    // multiplication by one preserves the smallest signed coefficient exactly
    EXPECT_EQ(
      fdapde::kron(
        minimum, SparseMatrix<int>(
                   1, 1,
                   {
                     {0, 0, 1}
    }))
        .coeff(0, 0),
      std::numeric_limits<int>::min());
    const SparseMatrix<unsigned> unsigned_max(
      1, 1,
      {
        {0, 0, std::numeric_limits<unsigned>::max()}
    });
    // unsigned arithmetic must not wrap when the mathematical product exceeds its range
    EXPECT_THROW(
      fdapde::kron(
        unsigned_max, SparseMatrix<unsigned>(
                        1, 1,
                        {
                          {0, 0, 2u}
    })),
      std::overflow_error);
    const auto owned = fdapde::kron(
      SparseMatrix<float>(
        1, 1,
        {
          {0, 0, 3.f}
    }),
      SparseMatrix<double>(1, 1, {{0, 0, 2.}}));
    // the result still owns its coefficient after both temporary inputs are destroyed
    EXPECT_DOUBLE_EQ(owned.coeff(0, 0), 6.);
}

}   // namespace
