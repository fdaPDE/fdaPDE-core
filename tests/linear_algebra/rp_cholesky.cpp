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

#include <fdaPDE/dense_linear_algebra.h>
#include <gtest/gtest.h>

#include <cmath>
#include <limits>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace {

using namespace fdapde;

template <typename Approximation>
concept exposes_rvalue_factor = requires(Approximation&& approximation) { std::move(approximation).factor(); };

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

    // the explicit factor product reconstructs the generated low-rank matrix within the requested relative tolerance
    EXPECT_LT(relative_reconstruction_error(source, approximation.factor()), 1.0e-10);
    // the factor retains one row per input coordinate
    EXPECT_EQ(approximation.factor().rows(), source.rows());
    // the factor column count equals the reported approximation rank
    EXPECT_EQ(approximation.factor().cols(), approximation.rank());
    // one recorded pivot corresponds to each emitted factor column
    EXPECT_EQ(approximation.pivots().size(), static_cast<std::size_t>(approximation.rank()));
    // the factor rank cannot exceed the input order
    EXPECT_LE(approximation.rank(), source.rows());
}

// single-pivot sampling reconstructs a rank-three positive-semidefinite matrix in both layouts
TEST(nys_approximation, block_equal_one) {
    // row-major storage meets the independent reconstruction residual and shape checks
    check_reconstruction<RowMajor>(1, 3, 1729);
    // column-major storage meets the same reconstruction residual and shape checks
    check_reconstruction<ColMajor>(1, 3, 1729);
}

// batched distinct pivots reconstruct a rank-five matrix when the batch exceeds its rank
TEST(nys_approximation, block_larger_than_one) {
    // row-major batched sampling meets the independent reconstruction and rank checks
    check_reconstruction<RowMajor>(7, 5, 8675309);
    // column-major batched sampling meets the same reconstruction and rank checks
    check_reconstruction<ColMajor>(7, 5, 8675309);
}

// recomputation replaces results on success and retains the previous factor and pivots on failure
TEST(nys_approximation, state_is_reusable_and_failures_are_atomic) {
    using matrix_type = Matrix<double, Dynamic, Dynamic>;
    using float_matrix_type = Matrix<float, Dynamic, Dynamic>;
    using approximation_type = RpChol<matrix_type>;
    using deduced_with_default_seed = decltype(RpChol(std::declval<const matrix_type&>(), 1.0e-3, 2, 12));
    using deduced_with_explicit_seed = decltype(RpChol(std::declval<const matrix_type&>(), 1.0e-3, 2, 12, 314159));
    using deduced_float_with_double_tolerance =
      decltype(RpChol(std::declval<const float_matrix_type&>(), 1.0e-3, 2, 12, 314159));
    // the approximation retains the double input coefficient type
    static_assert(std::is_same_v<typename approximation_type::Scalar, double>);
    // the declared matrix type remains the native input type
    static_assert(std::is_same_v<typename approximation_type::MatrixType, matrix_type>);
    // class deduction works when the random seed is omitted
    static_assert(std::is_same_v<deduced_with_default_seed, approximation_type>);
    // class deduction works with an explicit seed
    static_assert(std::is_same_v<deduced_with_explicit_seed, approximation_type>);
    // converting a double tolerance does not change a float matrix factor type
    static_assert(std::is_same_v<deduced_float_with_double_tolerance, RpChol<float_matrix_type>>);
    // pivot access borrows an ordered index vector from a live owner
    static_assert(
      std::is_same_v<decltype(std::declval<const approximation_type&>().pivots()), const std::vector<int>&>);
    // factor borrowing from temporary approximations is rejected
    static_assert(!exposes_rvalue_factor<approximation_type>);

    approximation_type approximation(2, 12, 314159);
    const matrix_type first = low_rank_spd<RowMajor>(4);
    approximation.compute(first, 1.0e-10);
    // an initial rank-four input meets the independently reconstructed relative error
    EXPECT_LT(relative_reconstruction_error(first, approximation.factor()), 1.0e-10);

    const matrix_type second = low_rank_spd<RowMajor>(2);
    approximation.compute(second, 1.0e-10);
    // recomputation on a rank-two input meets its own reconstruction oracle
    EXPECT_LT(relative_reconstruction_error(second, approximation.factor()), 1.0e-10);
    // the recomputed factor cannot exceed the new matrix order
    EXPECT_LE(approximation.rank(), second.rows());

    const matrix_type retained_factor(approximation.factor());
    const auto retained_pivots = approximation.pivots();
    matrix_type nonsquare(2, 3);
    nonsquare.set_zero();
    // rectangular input is rejected before replacing the previous state
    EXPECT_THROW(approximation.compute(nonsquare, 1.0e-3), std::invalid_argument);
    // a rejected rectangular input leaves the pivot sequence unchanged
    EXPECT_EQ(approximation.pivots(), retained_pivots);
    // a rejected rectangular input leaves every factor coefficient unchanged
    EXPECT_DOUBLE_EQ((approximation.factor() - retained_factor).norm(), 0.0);

    matrix_type indefinite(2, 2);
    indefinite(0, 0) = 1.0;
    indefinite(0, 1) = 2.0;
    indefinite(1, 0) = 2.0;
    indefinite(1, 1) = 1.0;
    // an indefinite two-by-two matrix is rejected by the positive-semidefinite precondition checks
    EXPECT_THROW(approximation.compute(indefinite, 1.0e-3), std::domain_error);
    // a rejected indefinite input leaves the pivot sequence unchanged
    EXPECT_EQ(approximation.pivots(), retained_pivots);
    // a rejected indefinite input leaves every factor coefficient unchanged
    EXPECT_DOUBLE_EQ((approximation.factor() - retained_factor).norm(), 0.0);
}

// invalid configuration and matrix inputs fail while normalization preserves extreme positive scales
TEST(nys_approximation, rejects_invalid_inputs_and_preserves_extreme_scales) {
    using matrix_type = Matrix<double, Dynamic, Dynamic>;
    // a zero sampling block size is rejected
    EXPECT_THROW(static_cast<void>(RpChol<matrix_type>(0, 10, 1)), std::invalid_argument);
    // a zero iteration limit is rejected
    EXPECT_THROW(static_cast<void>(RpChol<matrix_type>(1, 0, 1)), std::invalid_argument);

    RpChol<matrix_type> approximation(1, 10, 1);
    const matrix_type valid = low_rank_spd<RowMajor>(2);
    // a negative relative residual tolerance is rejected
    EXPECT_THROW(approximation.compute(valid, -1.0), std::invalid_argument);
    // a tolerance of one is outside the supported half-open interval
    EXPECT_THROW(approximation.compute(valid, 1.0), std::invalid_argument);
    // a NaN tolerance is rejected before sampling
    EXPECT_THROW(approximation.compute(valid, std::numeric_limits<double>::quiet_NaN()), std::invalid_argument);

    matrix_type empty;
    // an empty matrix cannot define the approximation
    EXPECT_THROW(approximation.compute(empty, 1.0e-3), std::invalid_argument);

    matrix_type nonfinite(valid);
    nonfinite(0, 0) = std::numeric_limits<double>::infinity();
    // an infinite source coefficient is rejected before normalization
    EXPECT_THROW(approximation.compute(nonfinite, 1.0e-3), std::invalid_argument);

    matrix_type asymmetric(valid);
    asymmetric(0, 1) += 1.0;
    // asymmetric off-diagonal coefficients are rejected before sampling
    EXPECT_THROW(approximation.compute(asymmetric, 1.0e-3), std::invalid_argument);

    matrix_type indefinite(2, 2);
    indefinite(0, 0) = 1.0;
    indefinite(0, 1) = 2.0;
    indefinite(1, 0) = 2.0;
    indefinite(1, 1) = 1.0;
    // an off-diagonal coefficient beyond the two-by-two PSD bound is rejected
    EXPECT_THROW(approximation.compute(indefinite, 1.0e-3), std::domain_error);

    matrix_type extreme(2, 2);
    extreme.set_zero();
    extreme(0, 0) = std::numeric_limits<double>::max();
    extreme(1, 1) = std::numeric_limits<double>::max();
    RpChol<matrix_type> scale_safe(1, 2, 3);
    // normalization avoids overflow for a diagonal matrix at the largest finite double scale
    EXPECT_NO_THROW(scale_safe.compute(extreme, 0.5));
    // both equally large independent diagonal modes are retained
    ASSERT_EQ(scale_safe.rank(), 2);
    for (int row = 0; row < 2; ++row) {
        double diagonal = 0.0;
        for (int col = 0; col < scale_safe.rank(); ++col) {
            diagonal += scale_safe.factor()(row, col) * scale_safe.factor()(row, col);
        }
        // each reconstructed diagonal agrees with its largest-finite analytic value after scaling
        EXPECT_NEAR(diagonal / std::numeric_limits<double>::max(), 1.0, 1.0e-12);
    }

    matrix_type small_mode(2, 2);
    small_mode.set_zero();
    small_mode(0, 0) = 1.0;
    small_mode(1, 1) = 1.0e-20;
    RpChol<matrix_type> zero_tolerance(1, 2, 4);
    // zero tolerance retains a tiny but positive independent diagonal mode
    EXPECT_NO_THROW(zero_tolerance.compute(small_mode, 0.0));
    // both positive diagonal modes contribute a factor column
    ASSERT_EQ(zero_tolerance.rank(), 2);
    const matrix_type reconstructed(zero_tolerance.factor() * zero_tolerance.factor().transpose());
    // the unit diagonal mode reconstructs exactly
    EXPECT_DOUBLE_EQ(reconstructed(0, 0), 1.0);
    // the tiny diagonal mode reconstructs within its scale-specific tolerance
    EXPECT_NEAR(reconstructed(1, 1), 1.0e-20, 1.0e-35);

    using long_matrix_type = Matrix<long double, Dynamic, Dynamic>;
    long_matrix_type wide_range(2, 2);
    wide_range.set_zero();
    wide_range(0, 0) = 1.0L;
    wide_range(1, 1) = std::numeric_limits<long double>::min();
    RpChol<long_matrix_type> wide_range_approximation(2, 1, 5);
    // sampling probabilities preserve the smallest positive long-double mode after rescaling
    EXPECT_NO_THROW(wide_range_approximation.compute(wide_range, 0.0L));
    // both long-double diagonal modes contribute a factor column
    ASSERT_EQ(wide_range_approximation.rank(), 2);
    const long_matrix_type wide_range_reconstructed(
      wide_range_approximation.factor() * wide_range_approximation.factor().transpose());
    // the reconstructed smallest positive mode agrees relatively with its analytic value
    EXPECT_NEAR(
      static_cast<double>(wide_range_reconstructed(1, 1) / std::numeric_limits<long double>::min()), 1.0, 1.0e-12);
}

// a fixed seed reproduces factors and pivot order, and a zero PSD matrix needs no sampled columns
TEST(nys_approximation, seeds_and_zero_matrix) {
    using matrix_type = Matrix<double, Dynamic, Dynamic>;
    const auto source = low_rank_spd<RowMajor>(3);
    const RpChol<matrix_type> first(source, 1.e-10, 2, 8, 73);
    const RpChol<matrix_type> second(source, 1.e-10, 2, 8, 73);
    // restarting the same standard random engine yields the same ordered pivot sequence
    EXPECT_EQ(first.pivots(), second.pivots());
    // identical sampled columns reproduce exactly the same native factor coefficients
    EXPECT_EQ(first.factor(), second.factor());
    const std::unordered_set<int> unique(first.pivots().begin(), first.pivots().end());
    // every emitted factor column comes from a distinct pivot index
    EXPECT_EQ(unique.size(), first.pivots().size());
    const matrix_type zero(3, 3);
    const RpChol<matrix_type> empty_factor(zero, 0., 2, 3, 73);
    // zero residual terminates without adding any factor columns
    EXPECT_EQ(empty_factor.rank(), 0);
    // an empty factor still records all three input rows
    EXPECT_EQ(empty_factor.factor().rows(), 3);
    // no pivot is sampled for the exact zero matrix
    EXPECT_TRUE(empty_factor.pivots().empty());
}

}   // namespace
