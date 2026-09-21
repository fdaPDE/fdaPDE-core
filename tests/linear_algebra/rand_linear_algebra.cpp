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
#include <utility>

namespace {

using namespace fdapde;

template <typename Approximation>
concept exposes_rvalue_left_singular_vectors =
  requires(Approximation&& approximation) { std::move(approximation).left_vectors(); };

template <typename Approximation>
concept exposes_rvalue_right_singular_vectors =
  requires(Approximation&& approximation) { std::move(approximation).right_vectors(); };

template <typename Approximation>
concept exposes_rvalue_singular_values =
  requires(Approximation&& approximation) { std::move(approximation).singular_values(); };

template <typename Approximation>
concept exposes_rvalue_eigenvectors =
  requires(Approximation&& approximation) { std::move(approximation).eigenvectors(); };

template <typename Approximation>
concept exposes_rvalue_eigenvalues =
  requires(Approximation&& approximation) { std::move(approximation).eigenvalues(); };

template <int StorageOrder> auto low_rank_spd(int columns) {
    Matrix<double, Dynamic, Dynamic, StorageOrder> generator(12, columns);
    for (int row = 0; row < generator.rows(); ++row) {
        for (int col = 0; col < generator.cols(); ++col) {
            generator(row, col) = static_cast<double>((row + 1) * (col + 2)) / 17.0 + (row == 2 * col ? 1.0 : 0.0);
        }
    }
    return Matrix<double, Dynamic, Dynamic, StorageOrder>(generator * generator.transpose());
}

template <int StorageOrder> auto diagonal_spectrum(int rows, int cols, double scale = 1.0) {
    Matrix<double, Dynamic, Dynamic, StorageOrder> matrix(rows, cols);
    matrix.set_zero();
    const double values[] = {9.0, 7.0, 5.0, 3.0, 1.0};
    for (int i = 0; i < fdapde::min(rows, cols); ++i) matrix(i, i) = scale * values[i];
    return matrix;
}

template <typename MatrixType, typename Approximation>
double rsi_residual(const MatrixType& matrix, const Approximation& approximation) {
    double maximum = 0.0;
    for (int col = 0; col < approximation.rank(); ++col) {
        double squared_norm = 0.0;
        for (int row = 0; row < matrix.rows(); ++row) {
            double projected = 0.0;
            for (int inner = 0; inner < matrix.cols(); ++inner) {
                projected += matrix(row, inner) * approximation.right_vectors()(inner, col);
            }
            const double residual =
              projected - approximation.left_vectors()(row, col) * approximation.singular_values()[col];
            squared_norm += residual * residual;
        }
        maximum = fdapde::max(maximum, std::sqrt(squared_norm));
    }
    return maximum;
}

template <typename MatrixType, typename Approximation>
double nys_eigen_residual(const MatrixType& matrix, const Approximation& approximation) {
    double maximum = 0.0;
    for (int col = 0; col < approximation.rank(); ++col) {
        double norm = 0.0;
        for (int row = 0; row < matrix.rows(); ++row) {
            double value = 0.0;
            for (int inner = 0; inner < matrix.cols(); ++inner) {
                value += matrix(row, inner) * approximation.eigenvectors()(inner, col);
            }
            value -= approximation.eigenvectors()(row, col) * approximation.eigenvalues()[col];
            norm = std::hypot(norm, value);
        }
        maximum = fdapde::max(maximum, norm);
    }
    return std::sqrt(2.0) * maximum;
}

template <typename MatrixType, typename Approximation>
double relative_svd_reconstruction_error(const MatrixType& matrix, const Approximation& approximation) {
    Matrix<double, Dynamic, Dynamic, MatrixType::StorageOrder> reconstructed(matrix.rows(), matrix.cols());
    reconstructed.set_zero();
    for (int row = 0; row < matrix.rows(); ++row) {
        for (int col = 0; col < matrix.cols(); ++col) {
            for (int k = 0; k < approximation.rank(); ++k) {
                reconstructed(row, col) += approximation.left_vectors()(row, k) * approximation.singular_values()[k] *
                                           approximation.right_vectors()(col, k);
            }
        }
    }
    return (matrix - reconstructed).norm() / matrix.norm();
}

template <int StorageOrder> void check_rsi_spectrum(int rows, int cols) {
    using matrix_type = Matrix<double, Dynamic, Dynamic, StorageOrder>;
    const matrix_type source = diagonal_spectrum<StorageOrder>(rows, cols);
    const RSI<matrix_type> approximation(source, 3, 1.0e-12, 8, 1729);

    // the published number of components matches the explicitly requested or capped rank
    EXPECT_EQ(approximation.rank(), 3);
    // left or eigenvector basis rows match the input row space
    EXPECT_EQ(approximation.left_vectors().rows(), rows);
    // each output basis has one column per requested component
    EXPECT_EQ(approximation.left_vectors().cols(), 3);
    // right singular vectors occupy the input column space
    EXPECT_EQ(approximation.right_vectors().rows(), cols);
    // each output basis has one column per requested component
    EXPECT_EQ(approximation.right_vectors().cols(), 3);
    // one spectral value is stored for each of the three requested components
    EXPECT_EQ(approximation.singular_values().rows(), 3);
    // the known diagonal spectrum supplies the expected leading value 9.0
    EXPECT_NEAR(approximation.singular_values()[0], 9.0, 1.0e-10);
    // the known diagonal spectrum supplies the expected leading value 7.0
    EXPECT_NEAR(approximation.singular_values()[1], 7.0, 1.0e-10);
    // the known diagonal spectrum supplies the expected leading value 5.0
    EXPECT_NEAR(approximation.singular_values()[2], 5.0, 1.0e-10);
    // an independently accumulated singular-triplet residual verifies the returned factors
    EXPECT_LT(rsi_residual(source, approximation), 1.0e-10);
}

// known diagonal spectra, zero modes and explicit residuals check the requested decomposition
TEST(rand_svd_test, rsi_square_tall_and_wide_leading_spectra) {
    // the fixed diagonal spectrum supplies exact leading values and dimensions in this storage order
    check_rsi_spectrum<RowMajor>(5, 5);
    // the fixed diagonal spectrum supplies exact leading values and dimensions in this storage order
    check_rsi_spectrum<ColMajor>(5, 5);
    // the fixed diagonal spectrum supplies exact leading values and dimensions in this storage order
    check_rsi_spectrum<RowMajor>(8, 5);
    // the fixed diagonal spectrum supplies exact leading values and dimensions in this storage order
    check_rsi_spectrum<ColMajor>(8, 5);
    // the fixed diagonal spectrum supplies exact leading values and dimensions in this storage order
    check_rsi_spectrum<RowMajor>(5, 8);
    // the fixed diagonal spectrum supplies exact leading values and dimensions in this storage order
    check_rsi_spectrum<ColMajor>(5, 8);

    using matrix_type = Matrix<double, Dynamic, Dynamic>;
    matrix_type rank_deficient(4, 5);
    rank_deficient.set_zero();
    rank_deficient(0, 0) = 9.0;
    const RSI<matrix_type> deficient(rank_deficient, 3, 1.0e-12, 4, 1729);
    // the published number of components matches the explicitly requested or capped rank
    ASSERT_EQ(deficient.rank(), 3);
    // the known diagonal spectrum supplies the expected leading value 9.0
    EXPECT_NEAR(deficient.singular_values()[0], 9.0, 1.0e-10);
    // an exactly zero mode remains zero in the returned spectrum
    EXPECT_DOUBLE_EQ(deficient.singular_values()[1], 0.0);
    // an exactly zero mode remains zero in the returned spectrum
    EXPECT_DOUBLE_EQ(deficient.singular_values()[2], 0.0);
    // an independently accumulated singular-triplet residual verifies the returned factors
    EXPECT_LT(rsi_residual(rank_deficient, deficient), 1.0e-10);

    matrix_type zero(3, 4);
    zero.set_zero();
    const RSI<matrix_type> zero_approximation(zero, 2, 0.0, 1, 1729);
    // the published number of components matches the explicitly requested or capped rank
    ASSERT_EQ(zero_approximation.rank(), 2);
    // left or eigenvector basis rows match the input row space
    EXPECT_EQ(zero_approximation.left_vectors().rows(), 3);
    // each output basis has one column per requested component
    EXPECT_EQ(zero_approximation.left_vectors().cols(), 2);
    // right singular vectors occupy the input column space
    EXPECT_EQ(zero_approximation.right_vectors().rows(), 4);
    // each output basis has one column per requested component
    EXPECT_EQ(zero_approximation.right_vectors().cols(), 2);
    // an exactly zero mode remains zero in the returned spectrum
    EXPECT_DOUBLE_EQ(zero_approximation.singular_values()[0], 0.0);
    // an exactly zero mode remains zero in the returned spectrum
    EXPECT_DOUBLE_EQ(zero_approximation.singular_values()[1], 0.0);
}

// rescaling the same input distinguishes absolute stopping tolerance from a relative criterion
TEST(rand_svd_test, rsi_uses_absolute_tolerance_and_returns_best_capped_result) {
    using matrix_type = Matrix<double, Dynamic, Dynamic>;
    matrix_type source(4, 4);
    source.set_zero();
    source(0, 0) = 1.0;
    source(1, 1) = 0.45;
    source(2, 2) = 0.2;
    source(3, 3) = 0.05;

    RSI<matrix_type> initial(std::numeric_limits<double>::max(), 1, 271828);
    initial.compute(source, 1, 1);
    const double initial_residual = rsi_residual(source, initial);
    // the seeded initial subspace has nonzero error so the scaling check exercises stopping
    ASSERT_GT(initial_residual, 0.0);

    const double tolerance = 1.1 * initial_residual;
    RSI<matrix_type> unscaled(tolerance, 1, 271828);
    unscaled.compute(source, 1, 1);
    // a tolerance above the initial residual stops before changing the sampled approximation
    EXPECT_NEAR(rsi_residual(source, unscaled), initial_residual, 1.0e-13);

    const matrix_type scaled_source(source * 16.0);
    RSI<matrix_type> scaled(tolerance, 1, 271828);
    scaled.compute(scaled_source, 1, 1);
    // scaling the source forces further work under the same absolute tolerance and reduces normalized error
    EXPECT_LT(rsi_residual(scaled_source, scaled) / 16.0, 0.8 * initial_residual);

    RSI<matrix_type> capped(0.0, 1, 271828);
    // exhausting the iteration cap returns the computed approximation without claiming convergence
    EXPECT_NO_THROW(capped.compute(source, 1, 1));
    // the published number of components matches the explicitly requested or capped rank
    EXPECT_EQ(capped.rank(), 1);
    // the iteration cap leaves a measurable residual even though computation returns normally
    EXPECT_GT(rsi_residual(source, capped), 0.0);
    // the capped approximation retains a finite computed spectral value
    EXPECT_TRUE(std::isfinite(capped.singular_values()[0]));
}

// native type deduction, recomputation and rejected inputs preserve the documented result state
TEST(rand_svd_test, rsi_state_is_reusable_and_failures_are_atomic) {
    using matrix_type = Matrix<double, Dynamic, Dynamic>;
    using float_matrix_type = Matrix<float, Dynamic, Dynamic>;
    using approximation_type = RSI<matrix_type>;
    using deduced_from_rank = decltype(RSI(std::declval<const matrix_type&>(), 2));
    using deduced_with_default_seed = decltype(RSI(std::declval<const matrix_type&>(), 2, 1.0e-5, 8));
    using deduced_with_explicit_seed = decltype(RSI(std::declval<const matrix_type&>(), 2, 1.0e-5, 8, 314159));
    using deduced_float_with_double_tolerance =
      decltype(RSI(std::declval<const float_matrix_type&>(), 2, 1.0e-5, 8, 314159));
    // the public scalar alias retains double coefficients
    static_assert(std::is_same_v<typename approximation_type::Scalar, double>);
    // the public input alias retains the native matrix type
    static_assert(std::is_same_v<typename approximation_type::MatrixType, matrix_type>);
    // the public factor alias retains the native matrix type and layout
    static_assert(std::is_same_v<typename approximation_type::FactorType, matrix_type>);
    // matrix-and-rank deduction selects the native decomposition type
    static_assert(std::is_same_v<deduced_from_rank, approximation_type>);
    // omitting the seed preserves native matrix type deduction
    static_assert(std::is_same_v<deduced_with_default_seed, approximation_type>);
    // an explicit seed preserves native matrix type deduction
    static_assert(std::is_same_v<deduced_with_explicit_seed, approximation_type>);
    // a double tolerance does not promote the float matrix coefficient type
    static_assert(std::is_same_v<deduced_float_with_double_tolerance, RSI<float_matrix_type>>);
    // temporary owners cannot expose a borrowed left singular basis
    static_assert(!exposes_rvalue_left_singular_vectors<approximation_type>);
    // temporary owners cannot expose a borrowed right singular basis
    static_assert(!exposes_rvalue_right_singular_vectors<approximation_type>);
    // temporary owners cannot expose a borrowed singular-value vector
    static_assert(!exposes_rvalue_singular_values<approximation_type>);

    approximation_type approximation(1.0e-12, 8, 314159);
    approximation.compute(diagonal_spectrum<RowMajor>(5, 5), 3, 3);
    // the published number of components matches the explicitly requested or capped rank
    EXPECT_EQ(approximation.rank(), 3);

    const matrix_type second = diagonal_spectrum<RowMajor>(8, 5, 0.5);
    approximation.compute(second, 2, 3);
    // the published number of components matches the explicitly requested or capped rank
    ASSERT_EQ(approximation.rank(), 2);
    // the known diagonal spectrum supplies the expected leading value 4.5
    EXPECT_NEAR(approximation.singular_values()[0], 4.5, 1.0e-10);
    // the known diagonal spectrum supplies the expected leading value 3.5
    EXPECT_NEAR(approximation.singular_values()[1], 3.5, 1.0e-10);

    const matrix_type retained_u(approximation.left_vectors());
    const matrix_type retained_v(approximation.right_vectors());
    const Vector<double, Dynamic> retained_values(approximation.singular_values());

    matrix_type nonfinite(second);
    nonfinite(0, 0) = std::numeric_limits<double>::infinity();
    // an infinite input coefficient is rejected before publishing replacement state
    EXPECT_THROW(approximation.compute(nonfinite, 2, 3), std::invalid_argument);
    // a zero requested rank is rejected
    EXPECT_THROW(approximation.compute(second, 0, 3), std::invalid_argument);
    // a subspace narrower than the requested RSI rank is rejected
    EXPECT_THROW(approximation.compute(second, 3, 2), std::invalid_argument);
    // failed recomputations leave the previously saved left basis exactly unchanged
    EXPECT_DOUBLE_EQ((approximation.left_vectors() - retained_u).norm(), 0.0);
    // failed recomputations leave the previously saved right basis exactly unchanged
    EXPECT_DOUBLE_EQ((approximation.right_vectors() - retained_v).norm(), 0.0);
    // failed recomputations leave the previously saved spectral values exactly unchanged
    EXPECT_DOUBLE_EQ((approximation.singular_values() - retained_values).norm(), 0.0);

    // a negative stopping tolerance is rejected during configuration
    EXPECT_THROW(static_cast<void>(approximation_type(-1.0, 8, 1)), std::invalid_argument);
    // a zero iteration cap is rejected during configuration
    EXPECT_THROW(static_cast<void>(approximation_type(1.0e-5, 0, 1)), std::invalid_argument);
    // a NaN stopping tolerance is rejected during configuration
    EXPECT_THROW(
      static_cast<void>(approximation_type(std::numeric_limits<double>::quiet_NaN(), 8, 1)), std::invalid_argument);
}

template <int StorageOrder> void check_rbki_spectrum(int rows, int cols) {
    using matrix_type = Matrix<double, Dynamic, Dynamic, StorageOrder>;
    const matrix_type source = diagonal_spectrum<StorageOrder>(rows, cols);
    const RBKI<matrix_type> approximation(source, 3, 1.0e-12, 8, 1729);

    // the published number of components matches the explicitly requested or capped rank
    EXPECT_EQ(approximation.rank(), 3);
    // left or eigenvector basis rows match the input row space
    EXPECT_EQ(approximation.left_vectors().rows(), rows);
    // each output basis has one column per requested component
    EXPECT_EQ(approximation.left_vectors().cols(), 3);
    // right singular vectors occupy the input column space
    EXPECT_EQ(approximation.right_vectors().rows(), cols);
    // each output basis has one column per requested component
    EXPECT_EQ(approximation.right_vectors().cols(), 3);
    // one spectral value is stored for each of the three requested components
    EXPECT_EQ(approximation.singular_values().rows(), 3);
    // the known diagonal spectrum supplies the expected leading value 9.0
    EXPECT_NEAR(approximation.singular_values()[0], 9.0, 1.0e-10);
    // the known diagonal spectrum supplies the expected leading value 7.0
    EXPECT_NEAR(approximation.singular_values()[1], 7.0, 1.0e-10);
    // the known diagonal spectrum supplies the expected leading value 5.0
    EXPECT_NEAR(approximation.singular_values()[2], 5.0, 1.0e-10);
    // an independently accumulated singular-triplet residual verifies the returned factors
    EXPECT_LT(rsi_residual(source, approximation), 1.0e-10);
}

template <int StorageOrder> void check_rbki_rectangular_reconstruction() {
    using matrix_type = Matrix<double, Dynamic, Dynamic, StorageOrder>;
    const double left[][2] = {
      {1.0,  2.0 },
      {2.0,  -1.0},
      {-1.0, 3.0 },
      {4.0,  0.5 }
    };
    const double right[][3] = {
      {1.0,  0.0, 2.0},
      {-2.0, 1.0, 0.5}
    };
    matrix_type tall(4, 3);
    for (int row = 0; row < tall.rows(); ++row) {
        for (int col = 0; col < tall.cols(); ++col) {
            tall(row, col) = left[row][0] * right[0][col] + left[row][1] * right[1][col];
        }
    }

    const RBKI<matrix_type> tall_approximation(tall, 2, 1.0e-12, 4, 1729);
    // explicit rank-two input factors are recovered within relative Frobenius tolerance
    EXPECT_LT(relative_svd_reconstruction_error(tall, tall_approximation), 1.0e-10);

    const matrix_type wide(tall.transpose());
    const RBKI<matrix_type> wide_approximation(wide, 2, 1.0e-12, 4, 1729);
    // explicit rank-two input factors are recovered within relative Frobenius tolerance
    EXPECT_LT(relative_svd_reconstruction_error(wide, wide_approximation), 1.0e-10);
}

// known diagonal spectra, zero modes and explicit residuals check the requested decomposition
TEST(rand_svd_test, rbki_square_tall_and_wide_leading_spectra) {
    // the fixed diagonal spectrum supplies exact leading values and dimensions in this storage order
    check_rbki_spectrum<RowMajor>(5, 5);
    // the fixed diagonal spectrum supplies exact leading values and dimensions in this storage order
    check_rbki_spectrum<ColMajor>(5, 5);
    // the fixed diagonal spectrum supplies exact leading values and dimensions in this storage order
    check_rbki_spectrum<RowMajor>(8, 5);
    // the fixed diagonal spectrum supplies exact leading values and dimensions in this storage order
    check_rbki_spectrum<ColMajor>(8, 5);
    // the fixed diagonal spectrum supplies exact leading values and dimensions in this storage order
    check_rbki_spectrum<RowMajor>(5, 8);
    // the fixed diagonal spectrum supplies exact leading values and dimensions in this storage order
    check_rbki_spectrum<ColMajor>(5, 8);
    // explicit rank-two factors provide the tall and transposed reconstruction oracle in this storage order
    check_rbki_rectangular_reconstruction<RowMajor>();
    // explicit rank-two factors provide the tall and transposed reconstruction oracle in this storage order
    check_rbki_rectangular_reconstruction<ColMajor>();

    using matrix_type = Matrix<double, Dynamic, Dynamic>;
    matrix_type rank_deficient(4, 5);
    rank_deficient.set_zero();
    rank_deficient(0, 0) = 9.0;
    RBKI<matrix_type> deficient(1.0e-12, 4, 1729);
    deficient.compute(rank_deficient, 3, 3);
    // the published number of components matches the explicitly requested or capped rank
    ASSERT_EQ(deficient.rank(), 3);
    // the known diagonal spectrum supplies the expected leading value 9.0
    EXPECT_NEAR(deficient.singular_values()[0], 9.0, 1.0e-10);
    // an exactly zero mode remains zero in the returned spectrum
    EXPECT_DOUBLE_EQ(deficient.singular_values()[1], 0.0);
    // an exactly zero mode remains zero in the returned spectrum
    EXPECT_DOUBLE_EQ(deficient.singular_values()[2], 0.0);
    // an independently accumulated singular-triplet residual verifies the returned factors
    EXPECT_LT(rsi_residual(rank_deficient, deficient), 1.0e-10);

    matrix_type zero(3, 4);
    zero.set_zero();
    RBKI<matrix_type> zero_explicit(0.0, 1, 1729);
    zero_explicit.compute(zero, 2, 2);
    // the published number of components matches the explicitly requested or capped rank
    ASSERT_EQ(zero_explicit.rank(), 2);
    // an exactly zero mode remains zero in the returned spectrum
    EXPECT_DOUBLE_EQ(zero_explicit.singular_values()[0], 0.0);
    // an exactly zero mode remains zero in the returned spectrum
    EXPECT_DOUBLE_EQ(zero_explicit.singular_values()[1], 0.0);

    const RBKI<matrix_type> zero_default(zero, 2, 0.0, 1, 1729);
    // early termination retains the default one-column Krylov result
    EXPECT_EQ(zero_default.rank(), 1);
    // an exactly zero mode remains zero in the returned spectrum
    EXPECT_DOUBLE_EQ(zero_default.singular_values()[0], 0.0);
}

// rescaling the same input distinguishes absolute stopping tolerance from a relative criterion
TEST(rand_svd_test, rbki_uses_absolute_tolerance_and_respects_iteration_cap) {
    using matrix_type = Matrix<double, Dynamic, Dynamic>;
    matrix_type source(4, 4);
    source.set_zero();
    source(0, 0) = 1.0;
    source(1, 1) = 0.45;
    source(2, 2) = 0.2;
    source(3, 3) = 0.05;

    RBKI<matrix_type> initial(std::numeric_limits<double>::max(), 1, 271828);
    initial.compute(source, 1, 1);
    const double initial_residual = rsi_residual(source, initial);
    // the seeded initial subspace has nonzero error so the scaling check exercises stopping
    ASSERT_GT(initial_residual, 0.0);

    const double tolerance = 1.1 * initial_residual;
    RBKI<matrix_type> unscaled(tolerance, 1, 271828);
    unscaled.compute(source, 1, 1);
    // a tolerance above the initial residual stops before changing the sampled approximation
    EXPECT_NEAR(rsi_residual(source, unscaled), initial_residual, 1.0e-13);

    const matrix_type scaled_source(source * 16.0);
    RBKI<matrix_type> scaled(tolerance, 1, 271828);
    scaled.compute(scaled_source, 1, 1);
    // scaling the source forces further work under the same absolute tolerance and reduces normalized error
    EXPECT_LT(rsi_residual(scaled_source, scaled) / 16.0, 0.8 * initial_residual);

    RBKI<matrix_type> capped(0.0, 1, 271828);
    // exhausting the iteration cap returns the computed approximation without claiming convergence
    EXPECT_NO_THROW(capped.compute(source, 3, 1));
    // one Krylov expansion can produce only two columns from an initial one-column block
    EXPECT_EQ(capped.rank(), 2);
    // the capped approximation retains a finite computed spectral value
    EXPECT_TRUE(std::isfinite(capped.singular_values()[0]));
    // the capped approximation retains a finite computed spectral value
    EXPECT_TRUE(std::isfinite(capped.singular_values()[1]));
    // the iteration cap leaves a measurable residual even though computation returns normally
    EXPECT_GT(rsi_residual(source, capped), 0.0);

    matrix_type large_source(101, 101);
    large_source.set_zero();
    for (int i = 0; i < large_source.rows(); ++i) large_source(i, i) = 1.0;
    RBKI<matrix_type> large_default(std::numeric_limits<double>::max(), 1, 271828);
    large_default.compute(large_source, 11);
    // the default ten-column Krylov block limits the initial result despite a requested rank of eleven
    EXPECT_EQ(large_default.rank(), 10);
}

// native type deduction, recomputation and rejected inputs preserve the documented result state
TEST(rand_svd_test, rbki_state_is_reusable_and_failures_are_atomic) {
    using matrix_type = Matrix<double, Dynamic, Dynamic>;
    using float_matrix_type = Matrix<float, Dynamic, Dynamic>;
    using approximation_type = RBKI<matrix_type>;
    using deduced_from_rank = decltype(RBKI(std::declval<const matrix_type&>(), 2));
    using deduced_with_default_seed = decltype(RBKI(std::declval<const matrix_type&>(), 2, 1.0e-5, 8));
    using deduced_with_explicit_seed = decltype(RBKI(std::declval<const matrix_type&>(), 2, 1.0e-5, 8, 314159));
    using deduced_float_with_double_tolerance =
      decltype(RBKI(std::declval<const float_matrix_type&>(), 2, 1.0e-5, 8, 314159));
    // the public scalar alias retains double coefficients
    static_assert(std::is_same_v<typename approximation_type::Scalar, double>);
    // the public input alias retains the native matrix type
    static_assert(std::is_same_v<typename approximation_type::MatrixType, matrix_type>);
    // the public factor alias retains the native matrix type and layout
    static_assert(std::is_same_v<typename approximation_type::FactorType, matrix_type>);
    // matrix-and-rank deduction selects the native decomposition type
    static_assert(std::is_same_v<deduced_from_rank, approximation_type>);
    // omitting the seed preserves native matrix type deduction
    static_assert(std::is_same_v<deduced_with_default_seed, approximation_type>);
    // an explicit seed preserves native matrix type deduction
    static_assert(std::is_same_v<deduced_with_explicit_seed, approximation_type>);
    // a double tolerance does not promote the float matrix coefficient type
    static_assert(std::is_same_v<deduced_float_with_double_tolerance, RBKI<float_matrix_type>>);
    // temporary owners cannot expose a borrowed left singular basis
    static_assert(!exposes_rvalue_left_singular_vectors<approximation_type>);
    // temporary owners cannot expose a borrowed right singular basis
    static_assert(!exposes_rvalue_right_singular_vectors<approximation_type>);
    // temporary owners cannot expose a borrowed singular-value vector
    static_assert(!exposes_rvalue_singular_values<approximation_type>);

    approximation_type approximation(1.0e-12, 8, 314159);
    approximation.compute(diagonal_spectrum<RowMajor>(5, 5), 3, 1);
    // the published number of components matches the explicitly requested or capped rank
    EXPECT_EQ(approximation.rank(), 3);

    const matrix_type second = diagonal_spectrum<RowMajor>(8, 5, 0.5);
    approximation.compute(second, 2, 2);
    // the published number of components matches the explicitly requested or capped rank
    ASSERT_EQ(approximation.rank(), 2);
    // the known diagonal spectrum supplies the expected leading value 4.5
    EXPECT_NEAR(approximation.singular_values()[0], 4.5, 1.0e-10);
    // the known diagonal spectrum supplies the expected leading value 3.5
    EXPECT_NEAR(approximation.singular_values()[1], 3.5, 1.0e-10);

    const matrix_type retained_u(approximation.left_vectors());
    const matrix_type retained_v(approximation.right_vectors());
    const Vector<double, Dynamic> retained_values(approximation.singular_values());

    matrix_type nonfinite(second);
    nonfinite(0, 0) = std::numeric_limits<double>::infinity();
    // an infinite input coefficient is rejected before publishing replacement state
    EXPECT_THROW(approximation.compute(nonfinite, 2, 2), std::invalid_argument);
    // a zero requested rank is rejected
    EXPECT_THROW(approximation.compute(second, 0, 2), std::invalid_argument);
    // a requested rank beyond the ambient dimension is rejected
    EXPECT_THROW(approximation.compute(second, 6, 2), std::invalid_argument);
    // a zero sampling block width is rejected
    EXPECT_THROW(approximation.compute(second, 2, 0), std::invalid_argument);
    // a sampling block wider than the ambient dimension is rejected
    EXPECT_THROW(approximation.compute(second, 2, 6), std::invalid_argument);
    matrix_type empty(0, 0);
    // an empty input cannot supply a positive requested rank
    EXPECT_THROW(approximation.compute(empty, 1), std::invalid_argument);
    // failed recomputations leave the previously saved left basis exactly unchanged
    EXPECT_DOUBLE_EQ((approximation.left_vectors() - retained_u).norm(), 0.0);
    // failed recomputations leave the previously saved right basis exactly unchanged
    EXPECT_DOUBLE_EQ((approximation.right_vectors() - retained_v).norm(), 0.0);
    // failed recomputations leave the previously saved spectral values exactly unchanged
    EXPECT_DOUBLE_EQ((approximation.singular_values() - retained_values).norm(), 0.0);

    // a negative stopping tolerance is rejected during configuration
    EXPECT_THROW(static_cast<void>(approximation_type(-1.0, 8, 1)), std::invalid_argument);
    // a zero iteration cap is rejected during configuration
    EXPECT_THROW(static_cast<void>(approximation_type(1.0e-5, 0, 1)), std::invalid_argument);
    // a NaN stopping tolerance is rejected during configuration
    EXPECT_THROW(
      static_cast<void>(approximation_type(std::numeric_limits<double>::quiet_NaN(), 8, 1)), std::invalid_argument);
}

template <int StorageOrder> void check_nysrsi_spectrum() {
    using matrix_type = Matrix<double, Dynamic, Dynamic, StorageOrder>;
    const matrix_type source = diagonal_spectrum<StorageOrder>(5, 5);
    const NysRSI<matrix_type> approximation(source, 3, 1.0e-12, 8, 1729);

    // the published number of components matches the explicitly requested or capped rank
    EXPECT_EQ(approximation.rank(), 3);
    // left or eigenvector basis rows match the input row space
    EXPECT_EQ(approximation.eigenvectors().rows(), 5);
    // each output basis has one column per requested component
    EXPECT_EQ(approximation.eigenvectors().cols(), 3);
    // one spectral value is stored for each of the three requested components
    EXPECT_EQ(approximation.eigenvalues().rows(), 3);
    // the known diagonal spectrum supplies the expected leading value 9.0
    EXPECT_NEAR(approximation.eigenvalues()[0], 9.0, 1.0e-10);
    // the known diagonal spectrum supplies the expected leading value 7.0
    EXPECT_NEAR(approximation.eigenvalues()[1], 7.0, 1.0e-10);
    // the known diagonal spectrum supplies the expected leading value 5.0
    EXPECT_NEAR(approximation.eigenvalues()[2], 5.0, 1.0e-10);
    // an independently accumulated eigenpair residual verifies the returned factors
    EXPECT_LT(nys_eigen_residual(source, approximation), 1.0e-10);
}

// known diagonal spectra, zero modes and explicit residuals check the requested decomposition
TEST(rand_evd_test, nysrsi_leading_spectra_and_psd_rank_contract) {
    // the fixed diagonal spectrum supplies exact leading values and dimensions in this storage order
    check_nysrsi_spectrum<RowMajor>();
    // the fixed diagonal spectrum supplies exact leading values and dimensions in this storage order
    check_nysrsi_spectrum<ColMajor>();

    const auto randomized_source = low_rank_spd<RowMajor>(3);
    NysRSI<decltype(randomized_source)> seeded_first(0.0, 2, 8675309);
    NysRSI<decltype(randomized_source)> seeded_second(0.0, 2, 8675309);
    seeded_first.compute(randomized_source, 2, 3);
    seeded_second.compute(randomized_source, 2, 3);
    // the same seed and source reproduce exactly the same basis coefficients
    EXPECT_DOUBLE_EQ((seeded_first.eigenvectors() - seeded_second.eigenvectors()).norm(), 0.0);
    // the same seed and source reproduce exactly the same spectral values
    EXPECT_DOUBLE_EQ((seeded_first.eigenvalues() - seeded_second.eigenvalues()).norm(), 0.0);
    // an independently accumulated eigenpair residual verifies the returned factors
    EXPECT_LT(nys_eigen_residual(randomized_source, seeded_first), 1.0e-10);

    using matrix_type = Matrix<double, Dynamic, Dynamic>;
    matrix_type rank_deficient(4, 4);
    rank_deficient.set_zero();
    rank_deficient(0, 0) = 9.0;
    NysRSI<matrix_type> deficient(1.0e-12, 4, 1729);
    deficient.compute(rank_deficient, 3, 4);
    // the published number of components matches the explicitly requested or capped rank
    ASSERT_EQ(deficient.rank(), 3);
    // the known diagonal spectrum supplies the expected leading value 9.0
    EXPECT_NEAR(deficient.eigenvalues()[0], 9.0, 1.0e-10);
    // an exactly zero mode remains zero in the returned spectrum
    EXPECT_DOUBLE_EQ(deficient.eigenvalues()[1], 0.0);
    // an exactly zero mode remains zero in the returned spectrum
    EXPECT_DOUBLE_EQ(deficient.eigenvalues()[2], 0.0);
    // an independently accumulated eigenpair residual verifies the returned factors
    EXPECT_LT(nys_eigen_residual(rank_deficient, deficient), 1.0e-10);

    matrix_type zero(3, 3);
    zero.set_zero();
    NysRSI<matrix_type> zero_approximation(0.0, 1, 1729);
    zero_approximation.compute(zero, 2, 2);
    // the published number of components matches the explicitly requested or capped rank
    ASSERT_EQ(zero_approximation.rank(), 2);
    // left or eigenvector basis rows match the input row space
    EXPECT_EQ(zero_approximation.eigenvectors().rows(), 3);
    // each output basis has one column per requested component
    EXPECT_EQ(zero_approximation.eigenvectors().cols(), 2);
    // an exactly zero mode remains zero in the returned spectrum
    EXPECT_DOUBLE_EQ(zero_approximation.eigenvalues()[0], 0.0);
    // an exactly zero mode remains zero in the returned spectrum
    EXPECT_DOUBLE_EQ(zero_approximation.eigenvalues()[1], 0.0);
}

// rescaling the same input distinguishes absolute stopping tolerance from a relative criterion
TEST(rand_evd_test, nysrsi_uses_absolute_tolerance_and_respects_iteration_cap) {
    using matrix_type = Matrix<double, Dynamic, Dynamic>;
    matrix_type source(4, 4);
    source.set_zero();
    source(0, 0) = 1.0;
    source(1, 1) = 0.45;
    source(2, 2) = 0.2;
    source(3, 3) = 0.05;

    NysRSI<matrix_type> initial(std::numeric_limits<double>::max(), 1, 271828);
    initial.compute(source, 1, 1);
    const double initial_residual = nys_eigen_residual(source, initial);
    // the seeded initial subspace has nonzero error so the scaling check exercises stopping
    ASSERT_GT(initial_residual, 0.0);

    const double tolerance = 1.1 * initial_residual;
    NysRSI<matrix_type> unscaled(tolerance, 2, 271828);
    unscaled.compute(source, 1, 1);
    // a tolerance above the initial residual stops before changing the sampled approximation
    EXPECT_NEAR(nys_eigen_residual(source, unscaled), initial_residual, 1.0e-13);

    const matrix_type scaled_source(source * 16.0);
    NysRSI<matrix_type> scaled(tolerance, 2, 271828);
    scaled.compute(scaled_source, 1, 1);
    // scaling the source forces further work under the same absolute tolerance and reduces normalized error
    EXPECT_LT(nys_eigen_residual(scaled_source, scaled) / 16.0, 0.8 * initial_residual);

    NysRSI<matrix_type> capped(0.0, 1, 271828);
    // exhausting the iteration cap returns the computed approximation without claiming convergence
    EXPECT_NO_THROW(capped.compute(source, 1, 1));
    // the published number of components matches the explicitly requested or capped rank
    EXPECT_EQ(capped.rank(), 1);
    // the iteration cap leaves a measurable residual even though computation returns normally
    EXPECT_GT(nys_eigen_residual(source, capped), 0.0);
    // the capped approximation retains a finite computed spectral value
    EXPECT_TRUE(std::isfinite(capped.eigenvalues()[0]));
}

// native type deduction, recomputation and rejected inputs preserve the documented result state
TEST(rand_evd_test, nysrsi_state_is_reusable_and_failures_are_atomic) {
    using matrix_type = Matrix<double, Dynamic, Dynamic>;
    using float_matrix_type = Matrix<float, Dynamic, Dynamic>;
    using approximation_type = NysRSI<matrix_type>;
    using deduced_from_rank = decltype(NysRSI(std::declval<const matrix_type&>(), 2));
    using deduced_with_default_seed = decltype(NysRSI(std::declval<const matrix_type&>(), 2, 1.0e-5, 8));
    using deduced_with_explicit_seed = decltype(NysRSI(std::declval<const matrix_type&>(), 2, 1.0e-5, 8, 314159));
    using deduced_float_with_double_tolerance =
      decltype(NysRSI(std::declval<const float_matrix_type&>(), 2, 1.0e-5, 8, 314159));
    // the public scalar alias retains double coefficients
    static_assert(std::is_same_v<typename approximation_type::Scalar, double>);
    // the public input alias retains the native matrix type
    static_assert(std::is_same_v<typename approximation_type::MatrixType, matrix_type>);
    // the public factor alias retains the native matrix type and layout
    static_assert(std::is_same_v<typename approximation_type::FactorType, matrix_type>);
    // the public spectral-value alias is an owning native double vector
    static_assert(std::is_same_v<typename approximation_type::EigenValuesType, Vector<double, Dynamic>>);
    // matrix-and-rank deduction selects the native decomposition type
    static_assert(std::is_same_v<deduced_from_rank, approximation_type>);
    // omitting the seed preserves native matrix type deduction
    static_assert(std::is_same_v<deduced_with_default_seed, approximation_type>);
    // an explicit seed preserves native matrix type deduction
    static_assert(std::is_same_v<deduced_with_explicit_seed, approximation_type>);
    // a double tolerance does not promote the float matrix coefficient type
    static_assert(std::is_same_v<deduced_float_with_double_tolerance, NysRSI<float_matrix_type>>);
    // temporary owners cannot expose a borrowed eigenvector matrix
    static_assert(!exposes_rvalue_eigenvectors<approximation_type>);
    // temporary owners cannot expose a borrowed eigenvalue vector
    static_assert(!exposes_rvalue_eigenvalues<approximation_type>);

    approximation_type approximation(1.0e-12, 8, 314159);
    approximation.compute(diagonal_spectrum<RowMajor>(5, 5), 3, 3);
    // the published number of components matches the explicitly requested or capped rank
    EXPECT_EQ(approximation.rank(), 3);

    matrix_type second = diagonal_spectrum<RowMajor>(5, 5, 0.5);
    for (int i = 2; i < second.rows(); ++i) second(i, i) = 0.0;
    approximation.compute(second, 2, 2);
    // the published number of components matches the explicitly requested or capped rank
    ASSERT_EQ(approximation.rank(), 2);
    // the known diagonal spectrum supplies the expected leading value 4.5
    EXPECT_NEAR(approximation.eigenvalues()[0], 4.5, 1.0e-10);
    // the known diagonal spectrum supplies the expected leading value 3.5
    EXPECT_NEAR(approximation.eigenvalues()[1], 3.5, 1.0e-10);

    const matrix_type retained_vectors(approximation.eigenvectors());
    const Vector<double, Dynamic> retained_values(approximation.eigenvalues());

    matrix_type empty;
    matrix_type nonsquare(2, 3);
    nonsquare.set_zero();
    matrix_type nonfinite(second);
    nonfinite(0, 0) = std::numeric_limits<double>::infinity();
    matrix_type asymmetric(second);
    asymmetric(0, 1) = 1.0;
    matrix_type indefinite(2, 2);
    indefinite(0, 0) = 1.0;
    indefinite(0, 1) = 2.0;
    indefinite(1, 0) = 2.0;
    indefinite(1, 1) = 1.0;
    matrix_type projected_indefinite(3, 3);
    for (int row = 0; row < projected_indefinite.rows(); ++row) {
        for (int col = 0; col < projected_indefinite.cols(); ++col) {
            projected_indefinite(row, col) = row == col ? 1.0 : -0.9;
        }
    }

    // an empty input cannot supply a positive requested rank
    EXPECT_THROW(approximation.compute(empty, 1), std::invalid_argument);
    // a rectangular input violates the Nyström square-matrix contract
    EXPECT_THROW(approximation.compute(nonsquare, 1), std::invalid_argument);
    // an infinite input coefficient is rejected before publishing replacement state
    EXPECT_THROW(approximation.compute(nonfinite, 2), std::invalid_argument);
    // unequal transposed entries reject the nonsymmetric input
    EXPECT_THROW(approximation.compute(asymmetric, 2), std::invalid_argument);
    // the two-by-two necessary PSD condition rejects the indefinite input
    EXPECT_THROW(approximation.compute(indefinite, 1), std::domain_error);
    // a negative projected eigenvalue triggers numerical PSD breakdown
    EXPECT_THROW(approximation.compute(projected_indefinite, 1, 3), std::domain_error);
    // a zero requested rank is rejected
    EXPECT_THROW(approximation.compute(second, 0), std::invalid_argument);
    // a requested rank beyond the ambient dimension is rejected
    EXPECT_THROW(approximation.compute(second, 6), std::invalid_argument);
    // a subspace narrower than the requested Nyström RSI rank is rejected
    EXPECT_THROW(approximation.compute(second, 2, 1), std::invalid_argument);
    // a sampling block wider than the ambient dimension is rejected
    EXPECT_THROW(approximation.compute(second, 2, 6), std::invalid_argument);
    // failed recomputations leave the previously saved eigenvector basis exactly unchanged
    EXPECT_DOUBLE_EQ((approximation.eigenvectors() - retained_vectors).norm(), 0.0);
    // failed recomputations leave the previously saved spectral values exactly unchanged
    EXPECT_DOUBLE_EQ((approximation.eigenvalues() - retained_values).norm(), 0.0);

    // a negative stopping tolerance is rejected during configuration
    EXPECT_THROW(static_cast<void>(approximation_type(-1.0, 8, 1)), std::invalid_argument);
    // a zero iteration cap is rejected during configuration
    EXPECT_THROW(static_cast<void>(approximation_type(1.0e-5, 0, 1)), std::invalid_argument);
    // a NaN stopping tolerance is rejected during configuration
    EXPECT_THROW(
      static_cast<void>(approximation_type(std::numeric_limits<double>::quiet_NaN(), 8, 1)), std::invalid_argument);
}

template <int StorageOrder> void check_nysrbki_spectrum() {
    using matrix_type = Matrix<double, Dynamic, Dynamic, StorageOrder>;
    const matrix_type source = diagonal_spectrum<StorageOrder>(5, 5);
    const NysRBKI<matrix_type> approximation(source, 3, 1.0e-12, 8, 1729);

    // the published number of components matches the explicitly requested or capped rank
    EXPECT_EQ(approximation.rank(), 3);
    // left or eigenvector basis rows match the input row space
    EXPECT_EQ(approximation.eigenvectors().rows(), 5);
    // each output basis has one column per requested component
    EXPECT_EQ(approximation.eigenvectors().cols(), 3);
    // one spectral value is stored for each of the three requested components
    EXPECT_EQ(approximation.eigenvalues().rows(), 3);
    // the known diagonal spectrum supplies the expected leading value 9.0
    EXPECT_NEAR(approximation.eigenvalues()[0], 9.0, 1.0e-10);
    // the known diagonal spectrum supplies the expected leading value 7.0
    EXPECT_NEAR(approximation.eigenvalues()[1], 7.0, 1.0e-10);
    // the known diagonal spectrum supplies the expected leading value 5.0
    EXPECT_NEAR(approximation.eigenvalues()[2], 5.0, 1.0e-10);
    // an independently accumulated eigenpair residual verifies the returned factors
    EXPECT_LT(nys_eigen_residual(source, approximation), 1.0e-10);
}

// known diagonal spectra, zero modes and explicit residuals check the requested decomposition
TEST(rand_evd_test, nysrbki_leading_spectra_and_psd_rank_contract) {
    // the fixed diagonal spectrum supplies exact leading values and dimensions in this storage order
    check_nysrbki_spectrum<RowMajor>();
    // the fixed diagonal spectrum supplies exact leading values and dimensions in this storage order
    check_nysrbki_spectrum<ColMajor>();

    using matrix_type = Matrix<double, Dynamic, Dynamic>;
    matrix_type rank_deficient(4, 4);
    rank_deficient.set_zero();
    rank_deficient(0, 0) = 9.0;
    NysRBKI<matrix_type> deficient(1.0e-12, 4, 1729);
    deficient.compute(rank_deficient, 3, 3);
    // the published number of components matches the explicitly requested or capped rank
    ASSERT_EQ(deficient.rank(), 3);
    // the known diagonal spectrum supplies the expected leading value 9.0
    EXPECT_NEAR(deficient.eigenvalues()[0], 9.0, 1.0e-10);
    // an exactly zero mode remains zero in the returned spectrum
    EXPECT_DOUBLE_EQ(deficient.eigenvalues()[1], 0.0);
    // an exactly zero mode remains zero in the returned spectrum
    EXPECT_DOUBLE_EQ(deficient.eigenvalues()[2], 0.0);
    // an independently accumulated eigenpair residual verifies the returned factors
    EXPECT_LT(nys_eigen_residual(rank_deficient, deficient), 1.0e-10);

    matrix_type zero(3, 3);
    zero.set_zero();
    NysRBKI<matrix_type> zero_approximation(0.0, 1, 1729);
    zero_approximation.compute(zero, 2, 2);
    // the published number of components matches the explicitly requested or capped rank
    ASSERT_EQ(zero_approximation.rank(), 2);
    // left or eigenvector basis rows match the input row space
    EXPECT_EQ(zero_approximation.eigenvectors().rows(), 3);
    // each output basis has one column per requested component
    EXPECT_EQ(zero_approximation.eigenvectors().cols(), 2);
    // an exactly zero mode remains zero in the returned spectrum
    EXPECT_DOUBLE_EQ(zero_approximation.eigenvalues()[0], 0.0);
    // an exactly zero mode remains zero in the returned spectrum
    EXPECT_DOUBLE_EQ(zero_approximation.eigenvalues()[1], 0.0);
}

// rescaling the same input distinguishes absolute stopping tolerance from a relative criterion
TEST(rand_evd_test, nysrbki_uses_absolute_tolerance_and_respects_iteration_cap) {
    using matrix_type = Matrix<double, Dynamic, Dynamic>;
    matrix_type source(4, 4);
    source.set_zero();
    source(0, 0) = 1.0;
    source(1, 1) = 0.45;
    source(2, 2) = 0.2;
    source(3, 3) = 0.05;

    NysRBKI<matrix_type> initial(std::numeric_limits<double>::max(), 1, 271828);
    initial.compute(source, 1, 1);
    const double initial_residual = nys_eigen_residual(source, initial);
    // the seeded initial subspace has nonzero error so the scaling check exercises stopping
    ASSERT_GT(initial_residual, 0.0);

    const double tolerance = 1.1 * initial_residual;
    NysRBKI<matrix_type> unscaled(tolerance, 2, 271828);
    unscaled.compute(source, 1, 1);
    // a tolerance above the initial residual stops before changing the sampled approximation
    EXPECT_NEAR(nys_eigen_residual(source, unscaled), initial_residual, 1.0e-13);

    const matrix_type scaled_source(source * 16.0);
    NysRBKI<matrix_type> scaled(tolerance, 2, 271828);
    scaled.compute(scaled_source, 1, 1);
    // scaling the source forces further work under the same absolute tolerance and reduces normalized error
    EXPECT_LT(nys_eigen_residual(scaled_source, scaled) / 16.0, 0.8 * initial_residual);

    NysRBKI<matrix_type> capped(0.0, 1, 271828);
    // exhausting the iteration cap returns the computed approximation without claiming convergence
    EXPECT_NO_THROW(capped.compute(source, 1, 1));
    // the published number of components matches the explicitly requested or capped rank
    EXPECT_EQ(capped.rank(), 1);
    // the iteration cap leaves a measurable residual even though computation returns normally
    EXPECT_GT(nys_eigen_residual(source, capped), 0.0);
    // the capped approximation retains a finite computed spectral value
    EXPECT_TRUE(std::isfinite(capped.eigenvalues()[0]));

    NysRBKI<matrix_type> default_block(std::numeric_limits<double>::max(), 1, 271828);
    default_block.compute(source, 3);
    // early termination retains the default one-column Krylov result
    EXPECT_EQ(default_block.rank(), 1);

    matrix_type large_source(101, 101);
    large_source.set_zero();
    for (int i = 0; i < large_source.rows(); ++i) large_source(i, i) = 1.0;
    NysRBKI<matrix_type> large_default(std::numeric_limits<double>::max(), 1, 271828);
    large_default.compute(large_source, 11);
    // the default ten-column Krylov block limits the initial result despite a requested rank of eleven
    EXPECT_EQ(large_default.rank(), 10);
}

// native type deduction, recomputation and rejected inputs preserve the documented result state
TEST(rand_evd_test, nysrbki_state_is_reusable_and_failures_are_atomic) {
    using matrix_type = Matrix<double, Dynamic, Dynamic>;
    using float_matrix_type = Matrix<float, Dynamic, Dynamic>;
    using approximation_type = NysRBKI<matrix_type>;
    using deduced_from_rank = decltype(NysRBKI(std::declval<const matrix_type&>(), 2));
    using deduced_with_default_seed = decltype(NysRBKI(std::declval<const matrix_type&>(), 2, 1.0e-5, 8));
    using deduced_with_explicit_seed = decltype(NysRBKI(std::declval<const matrix_type&>(), 2, 1.0e-5, 8, 314159));
    using deduced_float_with_double_tolerance =
      decltype(NysRBKI(std::declval<const float_matrix_type&>(), 2, 1.0e-5, 8, 314159));
    // the public scalar alias retains double coefficients
    static_assert(std::is_same_v<typename approximation_type::Scalar, double>);
    // the public input alias retains the native matrix type
    static_assert(std::is_same_v<typename approximation_type::MatrixType, matrix_type>);
    // the public factor alias retains the native matrix type and layout
    static_assert(std::is_same_v<typename approximation_type::FactorType, matrix_type>);
    // the public spectral-value alias is an owning native double vector
    static_assert(std::is_same_v<typename approximation_type::EigenValuesType, Vector<double, Dynamic>>);
    // matrix-and-rank deduction selects the native decomposition type
    static_assert(std::is_same_v<deduced_from_rank, approximation_type>);
    // omitting the seed preserves native matrix type deduction
    static_assert(std::is_same_v<deduced_with_default_seed, approximation_type>);
    // an explicit seed preserves native matrix type deduction
    static_assert(std::is_same_v<deduced_with_explicit_seed, approximation_type>);
    // a double tolerance does not promote the float matrix coefficient type
    static_assert(std::is_same_v<deduced_float_with_double_tolerance, NysRBKI<float_matrix_type>>);
    // temporary owners cannot expose a borrowed eigenvector matrix
    static_assert(!exposes_rvalue_eigenvectors<approximation_type>);
    // temporary owners cannot expose a borrowed eigenvalue vector
    static_assert(!exposes_rvalue_eigenvalues<approximation_type>);

    approximation_type approximation(1.0e-12, 8, 314159);
    approximation.compute(diagonal_spectrum<RowMajor>(5, 5), 3, 1);
    // the published number of components matches the explicitly requested or capped rank
    EXPECT_EQ(approximation.rank(), 3);

    matrix_type second = diagonal_spectrum<RowMajor>(5, 5, 0.5);
    for (int i = 2; i < second.rows(); ++i) second(i, i) = 0.0;
    approximation.compute(second, 2, 2);
    // the published number of components matches the explicitly requested or capped rank
    ASSERT_EQ(approximation.rank(), 2);
    // the known diagonal spectrum supplies the expected leading value 4.5
    EXPECT_NEAR(approximation.eigenvalues()[0], 4.5, 1.0e-10);
    // the known diagonal spectrum supplies the expected leading value 3.5
    EXPECT_NEAR(approximation.eigenvalues()[1], 3.5, 1.0e-10);
    // an independently accumulated eigenpair residual verifies the returned factors
    EXPECT_LT(nys_eigen_residual(second, approximation), 1.0e-10);

    approximation_type fresh(1.0e-12, 8, 314159);
    fresh.compute(second, 2, 2);
    // the same seed and source reproduce exactly the same basis coefficients
    EXPECT_DOUBLE_EQ((approximation.eigenvectors() - fresh.eigenvectors()).norm(), 0.0);
    // the same seed and source reproduce exactly the same spectral values
    EXPECT_DOUBLE_EQ((approximation.eigenvalues() - fresh.eigenvalues()).norm(), 0.0);

    const matrix_type retained_vectors(approximation.eigenvectors());
    const Vector<double, Dynamic> retained_values(approximation.eigenvalues());

    matrix_type empty;
    matrix_type nonsquare(2, 3);
    nonsquare.set_zero();
    matrix_type nonfinite(second);
    nonfinite(0, 0) = std::numeric_limits<double>::infinity();
    matrix_type asymmetric(second);
    asymmetric(0, 1) = 1.0;
    matrix_type indefinite(2, 2);
    indefinite(0, 0) = 1.0;
    indefinite(0, 1) = 2.0;
    indefinite(1, 0) = 2.0;
    indefinite(1, 1) = 1.0;
    matrix_type projected_indefinite(3, 3);
    for (int row = 0; row < projected_indefinite.rows(); ++row) {
        for (int col = 0; col < projected_indefinite.cols(); ++col) {
            projected_indefinite(row, col) = row == col ? 1.0 : -0.9;
        }
    }

    // an empty input cannot supply a positive requested rank
    EXPECT_THROW(approximation.compute(empty, 1), std::invalid_argument);
    // a rectangular input violates the Nyström square-matrix contract
    EXPECT_THROW(approximation.compute(nonsquare, 1), std::invalid_argument);
    // an infinite input coefficient is rejected before publishing replacement state
    EXPECT_THROW(approximation.compute(nonfinite, 2), std::invalid_argument);
    // unequal transposed entries reject the nonsymmetric input
    EXPECT_THROW(approximation.compute(asymmetric, 2), std::invalid_argument);
    // the two-by-two necessary PSD condition rejects the indefinite input
    EXPECT_THROW(approximation.compute(indefinite, 1), std::domain_error);
    // a negative projected eigenvalue triggers numerical PSD breakdown
    EXPECT_THROW(approximation.compute(projected_indefinite, 1, 3), std::domain_error);
    // a zero requested rank is rejected
    EXPECT_THROW(approximation.compute(second, 0), std::invalid_argument);
    // a requested rank beyond the ambient dimension is rejected
    EXPECT_THROW(approximation.compute(second, 6), std::invalid_argument);
    // a zero sampling block width is rejected
    EXPECT_THROW(approximation.compute(second, 2, 0), std::invalid_argument);
    // a sampling block wider than the ambient dimension is rejected
    EXPECT_THROW(approximation.compute(second, 2, 6), std::invalid_argument);
    // failed recomputations leave the previously saved eigenvector basis exactly unchanged
    EXPECT_DOUBLE_EQ((approximation.eigenvectors() - retained_vectors).norm(), 0.0);
    // failed recomputations leave the previously saved spectral values exactly unchanged
    EXPECT_DOUBLE_EQ((approximation.eigenvalues() - retained_values).norm(), 0.0);

    // a negative stopping tolerance is rejected during configuration
    EXPECT_THROW(static_cast<void>(approximation_type(-1.0, 8, 1)), std::invalid_argument);
    // a zero iteration cap is rejected during configuration
    EXPECT_THROW(static_cast<void>(approximation_type(1.0e-5, 0, 1)), std::invalid_argument);
    // a NaN stopping tolerance is rejected during configuration
    EXPECT_THROW(
      static_cast<void>(approximation_type(std::numeric_limits<double>::quiet_NaN(), 8, 1)), std::invalid_argument);
}

}   // namespace
