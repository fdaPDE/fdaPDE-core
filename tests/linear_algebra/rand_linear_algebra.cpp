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

#include <cmath>
#include <limits>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace {

using namespace fdapde;

template <typename Approximation>
concept exposes_rvalue_factor = requires(Approximation&& approximation) { std::move(approximation).matrixL(); };

template <int StorageOrder> auto low_rank_spd(int columns) {
    Matrix<double, Dynamic, Dynamic, StorageOrder> generator(12, columns);
    for (int row = 0; row < generator.rows(); ++row) {
        for (int col = 0; col < generator.cols(); ++col) {
            generator(row, col) = static_cast<double>((row + 1) * (col + 2)) / 17.0 + (row == 2 * col ? 1.0 : 0.0);
        }
    }
    return Matrix<double, Dynamic, Dynamic, StorageOrder>(generator * generator.transpose());
}

template <typename MatrixType, typename FactorType>
double relative_reconstruction_error(const MatrixType& matrix, const FactorType& factor) {
    const Matrix<double, Dynamic, Dynamic, MatrixType::StorageOrder> reconstructed(factor * factor.transpose());
    return (matrix - reconstructed).norm() / matrix.norm();
}

template <int StorageOrder> void check_reconstruction(int block_size, int rank, int seed) {
    using matrix_type = Matrix<double, Dynamic, Dynamic, StorageOrder>;
    const matrix_type source = low_rank_spd<StorageOrder>(rank);
    const int max_iterations = block_size == 1 ? rank : 1;
    RpChol<matrix_type> approximation(source, 1.0e-10, block_size, max_iterations, seed);

    EXPECT_LT(relative_reconstruction_error(source, approximation.matrixL()), 1.0e-10);
    EXPECT_EQ(approximation.matrixL().rows(), source.rows());
    EXPECT_EQ(approximation.matrixL().cols(), approximation.rank());
    EXPECT_EQ(approximation.pivotSet().size(), static_cast<std::size_t>(approximation.rank()));
    EXPECT_LE(approximation.rank(), source.rows());
}

TEST(nys_approximation, block_equal_one) {
    check_reconstruction<RowMajor>(1, 3, 1729);
    check_reconstruction<ColMajor>(1, 3, 1729);
}

TEST(nys_approximation, block_larger_than_one) {
    check_reconstruction<RowMajor>(7, 5, 8675309);
    check_reconstruction<ColMajor>(7, 5, 8675309);
}

TEST(nys_approximation, state_is_reusable_and_failures_are_atomic) {
    using matrix_type = Matrix<double, Dynamic, Dynamic>;
    using approximation_type = RpChol<matrix_type>;
    static_assert(std::is_same_v<typename approximation_type::Scalar, double>);
    static_assert(std::is_same_v<typename approximation_type::MatrixType, matrix_type>);
    static_assert(
      std::is_same_v<decltype(std::declval<const approximation_type&>().pivotSet()), const std::unordered_set<int>&>);
    static_assert(!exposes_rvalue_factor<approximation_type>);

    approximation_type approximation(2, 12, 314159);
    const matrix_type first = low_rank_spd<RowMajor>(4);
    approximation.compute(first, 1.0e-10);
    EXPECT_LT(relative_reconstruction_error(first, approximation.matrixL()), 1.0e-10);

    const matrix_type second = low_rank_spd<RowMajor>(2);
    approximation.compute(second, 1.0e-10);
    EXPECT_LT(relative_reconstruction_error(second, approximation.matrixL()), 1.0e-10);
    EXPECT_LE(approximation.rank(), second.rows());

    const matrix_type retained_factor(approximation.matrixL());
    const auto retained_pivots = approximation.pivotSet();
    matrix_type nonsquare(2, 3);
    nonsquare.set_zero();
    EXPECT_THROW(approximation.compute(nonsquare, 1.0e-3), std::invalid_argument);
    EXPECT_EQ(approximation.pivotSet(), retained_pivots);
    EXPECT_DOUBLE_EQ((approximation.matrixL() - retained_factor).norm(), 0.0);

    matrix_type indefinite(2, 2);
    indefinite(0, 0) = 1.0;
    indefinite(0, 1) = 2.0;
    indefinite(1, 0) = 2.0;
    indefinite(1, 1) = 1.0;
    EXPECT_THROW(approximation.compute(indefinite, 1.0e-3), std::domain_error);
    EXPECT_EQ(approximation.pivotSet(), retained_pivots);
    EXPECT_DOUBLE_EQ((approximation.matrixL() - retained_factor).norm(), 0.0);
}

TEST(nys_approximation, contracts_remain_active_without_debug_assertions) {
    using matrix_type = Matrix<double, Dynamic, Dynamic>;
    EXPECT_THROW(static_cast<void>(RpChol<matrix_type>(0, 10, 1)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(RpChol<matrix_type>(1, 0, 1)), std::invalid_argument);

    RpChol<matrix_type> approximation(1, 10, 1);
    const matrix_type valid = low_rank_spd<RowMajor>(2);
    EXPECT_THROW(approximation.compute(valid, -1.0), std::invalid_argument);
    EXPECT_THROW(approximation.compute(valid, 1.0), std::invalid_argument);
    EXPECT_THROW(approximation.compute(valid, std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);

    matrix_type empty;
    EXPECT_THROW(approximation.compute(empty, 1.0e-3), std::invalid_argument);

    matrix_type nonfinite(valid);
    nonfinite(0, 0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW(approximation.compute(nonfinite, 1.0e-3), std::invalid_argument);

    matrix_type asymmetric(valid);
    asymmetric(0, 1) += 1.0;
    EXPECT_THROW(approximation.compute(asymmetric, 1.0e-3), std::invalid_argument);

    matrix_type indefinite(2, 2);
    indefinite(0, 0) = 1.0;
    indefinite(0, 1) = 2.0;
    indefinite(1, 0) = 2.0;
    indefinite(1, 1) = 1.0;
    EXPECT_THROW(approximation.compute(indefinite, 1.0e-3), std::domain_error);

    matrix_type extreme(2, 2);
    extreme.set_zero();
    extreme(0, 0) = std::numeric_limits<double>::max();
    extreme(1, 1) = std::numeric_limits<double>::max();
    RpChol<matrix_type> scale_safe(1, 2, 3);
    EXPECT_NO_THROW(scale_safe.compute(extreme, 0.5));
    ASSERT_EQ(scale_safe.rank(), 2);
    for (int row = 0; row < 2; ++row) {
        double diagonal = 0.0;
        for (int col = 0; col < scale_safe.rank(); ++col) {
            diagonal += scale_safe.matrixL()(row, col) * scale_safe.matrixL()(row, col);
        }
        EXPECT_NEAR(diagonal / std::numeric_limits<double>::max(), 1.0, 1.0e-12);
    }

    matrix_type small_mode(2, 2);
    small_mode.set_zero();
    small_mode(0, 0) = 1.0;
    small_mode(1, 1) = 1.0e-20;
    RpChol<matrix_type> zero_tolerance(1, 2, 4);
    EXPECT_NO_THROW(zero_tolerance.compute(small_mode, 0.0));
    ASSERT_EQ(zero_tolerance.rank(), 2);
    const matrix_type reconstructed(zero_tolerance.matrixL() * zero_tolerance.matrixL().transpose());
    EXPECT_DOUBLE_EQ(reconstructed(0, 0), 1.0);
    EXPECT_NEAR(reconstructed(1, 1), 1.0e-20, 1.0e-35);

    using long_matrix_type = Matrix<long double, Dynamic, Dynamic>;
    long_matrix_type wide_range(2, 2);
    wide_range.set_zero();
    wide_range(0, 0) = 1.0L;
    wide_range(1, 1) = std::numeric_limits<long double>::min();
    RpChol<long_matrix_type> wide_range_approximation(2, 1, 5);
    EXPECT_NO_THROW(wide_range_approximation.compute(wide_range, 0.0L));
    ASSERT_EQ(wide_range_approximation.rank(), 2);
    const long_matrix_type wide_range_reconstructed(
      wide_range_approximation.matrixL() * wide_range_approximation.matrixL().transpose());
    EXPECT_NEAR(
      static_cast<double>(wide_range_reconstructed(1, 1) / std::numeric_limits<long double>::min()), 1.0, 1.0e-12);
}

// Source: a2a9c88:test/src/rand_linear_algebra_test.cpp.
// TODO(P4-M): restore the four seeded RSI/RBKI and NysRSI/NysRBKI leading-spectrum assertions in their own native
// compact-decomposition slice. Eigen may be an opt-in test oracle only, never a production dependency.

}   // namespace
