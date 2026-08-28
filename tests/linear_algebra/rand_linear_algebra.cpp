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

template <typename Approximation>
concept exposes_rvalue_left_singular_vectors =
  requires(Approximation&& approximation) { std::move(approximation).matrixU(); };

template <typename Approximation>
concept exposes_rvalue_right_singular_vectors =
  requires(Approximation&& approximation) { std::move(approximation).matrixV(); };

template <typename Approximation>
concept exposes_rvalue_singular_values =
  requires(Approximation&& approximation) { std::move(approximation).singularValues(); };

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
                projected += matrix(row, inner) * approximation.matrixV()(inner, col);
            }
            const double residual = projected - approximation.matrixU()(row, col) * approximation.singularValues()[col];
            squared_norm += residual * residual;
        }
        maximum = fdapde::max(maximum, std::sqrt(squared_norm));
    }
    return maximum;
}

template <typename MatrixType, typename Approximation>
double relative_svd_reconstruction_error(const MatrixType& matrix, const Approximation& approximation) {
    Matrix<double, Dynamic, Dynamic, MatrixType::StorageOrder> reconstructed(matrix.rows(), matrix.cols());
    reconstructed.set_zero();
    for (int row = 0; row < matrix.rows(); ++row) {
        for (int col = 0; col < matrix.cols(); ++col) {
            for (int k = 0; k < approximation.rank(); ++k) {
                reconstructed(row, col) +=
                  approximation.matrixU()(row, k) * approximation.singularValues()[k] * approximation.matrixV()(col, k);
            }
        }
    }
    return (matrix - reconstructed).norm() / matrix.norm();
}

template <int StorageOrder> void check_rsi_spectrum(int rows, int cols) {
    using matrix_type = Matrix<double, Dynamic, Dynamic, StorageOrder>;
    const matrix_type source = diagonal_spectrum<StorageOrder>(rows, cols);
    const RSI<matrix_type> approximation(source, 3, 1.0e-12, 8, 1729);

    EXPECT_EQ(approximation.rank(), 3);
    EXPECT_EQ(approximation.matrixU().rows(), rows);
    EXPECT_EQ(approximation.matrixU().cols(), 3);
    EXPECT_EQ(approximation.matrixV().rows(), cols);
    EXPECT_EQ(approximation.matrixV().cols(), 3);
    EXPECT_EQ(approximation.singularValues().rows(), 3);
    EXPECT_NEAR(approximation.singularValues()[0], 9.0, 1.0e-10);
    EXPECT_NEAR(approximation.singularValues()[1], 7.0, 1.0e-10);
    EXPECT_NEAR(approximation.singularValues()[2], 5.0, 1.0e-10);
    EXPECT_LT(rsi_residual(source, approximation), 1.0e-10);
}

TEST(rand_svd_test, rsi_square_tall_and_wide_leading_spectra) {
    check_rsi_spectrum<RowMajor>(5, 5);
    check_rsi_spectrum<ColMajor>(5, 5);
    check_rsi_spectrum<RowMajor>(8, 5);
    check_rsi_spectrum<ColMajor>(8, 5);
    check_rsi_spectrum<RowMajor>(5, 8);
    check_rsi_spectrum<ColMajor>(5, 8);

    using matrix_type = Matrix<double, Dynamic, Dynamic>;
    matrix_type rank_deficient(4, 5);
    rank_deficient.set_zero();
    rank_deficient(0, 0) = 9.0;
    const RSI<matrix_type> deficient(rank_deficient, 3, 1.0e-12, 4, 1729);
    ASSERT_EQ(deficient.rank(), 3);
    EXPECT_NEAR(deficient.singularValues()[0], 9.0, 1.0e-10);
    EXPECT_DOUBLE_EQ(deficient.singularValues()[1], 0.0);
    EXPECT_DOUBLE_EQ(deficient.singularValues()[2], 0.0);
    EXPECT_LT(rsi_residual(rank_deficient, deficient), 1.0e-10);

    matrix_type zero(3, 4);
    zero.set_zero();
    const RSI<matrix_type> zero_approximation(zero, 2, 0.0, 1, 1729);
    ASSERT_EQ(zero_approximation.rank(), 2);
    EXPECT_EQ(zero_approximation.matrixU().rows(), 3);
    EXPECT_EQ(zero_approximation.matrixU().cols(), 2);
    EXPECT_EQ(zero_approximation.matrixV().rows(), 4);
    EXPECT_EQ(zero_approximation.matrixV().cols(), 2);
    EXPECT_DOUBLE_EQ(zero_approximation.singularValues()[0], 0.0);
    EXPECT_DOUBLE_EQ(zero_approximation.singularValues()[1], 0.0);
}

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
    ASSERT_GT(initial_residual, 0.0);

    const double tolerance = 1.1 * initial_residual;
    RSI<matrix_type> unscaled(tolerance, 1, 271828);
    unscaled.compute(source, 1, 1);
    EXPECT_NEAR(rsi_residual(source, unscaled), initial_residual, 1.0e-13);

    const matrix_type scaled_source(source * 16.0);
    RSI<matrix_type> scaled(tolerance, 1, 271828);
    scaled.compute(scaled_source, 1, 1);
    EXPECT_LT(rsi_residual(scaled_source, scaled) / 16.0, 0.8 * initial_residual);

    RSI<matrix_type> capped(0.0, 1, 271828);
    EXPECT_NO_THROW(capped.compute(source, 1, 1));
    EXPECT_EQ(capped.rank(), 1);
    EXPECT_GT(rsi_residual(source, capped), 0.0);
    EXPECT_TRUE(std::isfinite(capped.singularValues()[0]));
}

TEST(rand_svd_test, rsi_state_is_reusable_and_failures_are_atomic) {
    using matrix_type = Matrix<double, Dynamic, Dynamic>;
    using approximation_type = RSI<matrix_type>;
    static_assert(std::is_same_v<typename approximation_type::Scalar, double>);
    static_assert(std::is_same_v<typename approximation_type::MatrixType, matrix_type>);
    static_assert(std::is_same_v<typename approximation_type::FactorType, matrix_type>);
    static_assert(!exposes_rvalue_left_singular_vectors<approximation_type>);
    static_assert(!exposes_rvalue_right_singular_vectors<approximation_type>);
    static_assert(!exposes_rvalue_singular_values<approximation_type>);

    approximation_type approximation(1.0e-12, 8, 314159);
    approximation.compute(diagonal_spectrum<RowMajor>(5, 5), 3, 3);
    EXPECT_EQ(approximation.rank(), 3);

    const matrix_type second = diagonal_spectrum<RowMajor>(8, 5, 0.5);
    approximation.compute(second, 2, 3);
    ASSERT_EQ(approximation.rank(), 2);
    EXPECT_NEAR(approximation.singularValues()[0], 4.5, 1.0e-10);
    EXPECT_NEAR(approximation.singularValues()[1], 3.5, 1.0e-10);

    const matrix_type retained_u(approximation.matrixU());
    const matrix_type retained_v(approximation.matrixV());
    const Vector<double, Dynamic> retained_values(approximation.singularValues());

    matrix_type nonfinite(second);
    nonfinite(0, 0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW(approximation.compute(nonfinite, 2, 3), std::invalid_argument);
    EXPECT_THROW(approximation.compute(second, 0, 3), std::invalid_argument);
    EXPECT_THROW(approximation.compute(second, 3, 2), std::invalid_argument);
    EXPECT_DOUBLE_EQ((approximation.matrixU() - retained_u).norm(), 0.0);
    EXPECT_DOUBLE_EQ((approximation.matrixV() - retained_v).norm(), 0.0);
    EXPECT_DOUBLE_EQ((approximation.singularValues() - retained_values).norm(), 0.0);

    EXPECT_THROW(static_cast<void>(approximation_type(-1.0, 8, 1)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(approximation_type(1.0e-5, 0, 1)), std::invalid_argument);
    EXPECT_THROW(
      static_cast<void>(approximation_type(std::numeric_limits<double>::quiet_NaN(), 8, 1)), std::invalid_argument);
}

template <int StorageOrder> void check_rbki_spectrum(int rows, int cols) {
    using matrix_type = Matrix<double, Dynamic, Dynamic, StorageOrder>;
    const matrix_type source = diagonal_spectrum<StorageOrder>(rows, cols);
    const RBKI<matrix_type> approximation(source, 3, 1.0e-12, 8, 1729);

    EXPECT_EQ(approximation.rank(), 3);
    EXPECT_EQ(approximation.matrixU().rows(), rows);
    EXPECT_EQ(approximation.matrixU().cols(), 3);
    EXPECT_EQ(approximation.matrixV().rows(), cols);
    EXPECT_EQ(approximation.matrixV().cols(), 3);
    EXPECT_EQ(approximation.singularValues().rows(), 3);
    EXPECT_NEAR(approximation.singularValues()[0], 9.0, 1.0e-10);
    EXPECT_NEAR(approximation.singularValues()[1], 7.0, 1.0e-10);
    EXPECT_NEAR(approximation.singularValues()[2], 5.0, 1.0e-10);
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
    EXPECT_LT(relative_svd_reconstruction_error(tall, tall_approximation), 1.0e-10);

    const matrix_type wide(tall.transpose());
    const RBKI<matrix_type> wide_approximation(wide, 2, 1.0e-12, 4, 1729);
    EXPECT_LT(relative_svd_reconstruction_error(wide, wide_approximation), 1.0e-10);
}

TEST(rand_svd_test, rbki_square_tall_and_wide_leading_spectra) {
    check_rbki_spectrum<RowMajor>(5, 5);
    check_rbki_spectrum<ColMajor>(5, 5);
    check_rbki_spectrum<RowMajor>(8, 5);
    check_rbki_spectrum<ColMajor>(8, 5);
    check_rbki_spectrum<RowMajor>(5, 8);
    check_rbki_spectrum<ColMajor>(5, 8);
    check_rbki_rectangular_reconstruction<RowMajor>();
    check_rbki_rectangular_reconstruction<ColMajor>();

    using matrix_type = Matrix<double, Dynamic, Dynamic>;
    matrix_type rank_deficient(4, 5);
    rank_deficient.set_zero();
    rank_deficient(0, 0) = 9.0;
    RBKI<matrix_type> deficient(1.0e-12, 4, 1729);
    deficient.compute(rank_deficient, 3, 3);
    ASSERT_EQ(deficient.rank(), 3);
    EXPECT_NEAR(deficient.singularValues()[0], 9.0, 1.0e-10);
    EXPECT_DOUBLE_EQ(deficient.singularValues()[1], 0.0);
    EXPECT_DOUBLE_EQ(deficient.singularValues()[2], 0.0);
    EXPECT_LT(rsi_residual(rank_deficient, deficient), 1.0e-10);

    matrix_type zero(3, 4);
    zero.set_zero();
    RBKI<matrix_type> zero_explicit(0.0, 1, 1729);
    zero_explicit.compute(zero, 2, 2);
    ASSERT_EQ(zero_explicit.rank(), 2);
    EXPECT_DOUBLE_EQ(zero_explicit.singularValues()[0], 0.0);
    EXPECT_DOUBLE_EQ(zero_explicit.singularValues()[1], 0.0);

    const RBKI<matrix_type> zero_default(zero, 2, 0.0, 1, 1729);
    EXPECT_EQ(zero_default.rank(), 1);
    EXPECT_DOUBLE_EQ(zero_default.singularValues()[0], 0.0);
}

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
    ASSERT_GT(initial_residual, 0.0);

    const double tolerance = 1.1 * initial_residual;
    RBKI<matrix_type> unscaled(tolerance, 1, 271828);
    unscaled.compute(source, 1, 1);
    EXPECT_NEAR(rsi_residual(source, unscaled), initial_residual, 1.0e-13);

    const matrix_type scaled_source(source * 16.0);
    RBKI<matrix_type> scaled(tolerance, 1, 271828);
    scaled.compute(scaled_source, 1, 1);
    EXPECT_LT(rsi_residual(scaled_source, scaled) / 16.0, 0.8 * initial_residual);

    RBKI<matrix_type> capped(0.0, 1, 271828);
    EXPECT_NO_THROW(capped.compute(source, 3, 1));
    EXPECT_EQ(capped.rank(), 2);
    EXPECT_TRUE(std::isfinite(capped.singularValues()[0]));
    EXPECT_TRUE(std::isfinite(capped.singularValues()[1]));
    EXPECT_GT(rsi_residual(source, capped), 0.0);
}

TEST(rand_svd_test, rbki_state_is_reusable_and_failures_are_atomic) {
    using matrix_type = Matrix<double, Dynamic, Dynamic>;
    using approximation_type = RBKI<matrix_type>;
    static_assert(std::is_same_v<typename approximation_type::Scalar, double>);
    static_assert(std::is_same_v<typename approximation_type::MatrixType, matrix_type>);
    static_assert(std::is_same_v<typename approximation_type::FactorType, matrix_type>);
    static_assert(!exposes_rvalue_left_singular_vectors<approximation_type>);
    static_assert(!exposes_rvalue_right_singular_vectors<approximation_type>);
    static_assert(!exposes_rvalue_singular_values<approximation_type>);

    approximation_type approximation(1.0e-12, 8, 314159);
    approximation.compute(diagonal_spectrum<RowMajor>(5, 5), 3, 1);
    EXPECT_EQ(approximation.rank(), 3);

    const matrix_type second = diagonal_spectrum<RowMajor>(8, 5, 0.5);
    approximation.compute(second, 2, 2);
    ASSERT_EQ(approximation.rank(), 2);
    EXPECT_NEAR(approximation.singularValues()[0], 4.5, 1.0e-10);
    EXPECT_NEAR(approximation.singularValues()[1], 3.5, 1.0e-10);

    const matrix_type retained_u(approximation.matrixU());
    const matrix_type retained_v(approximation.matrixV());
    const Vector<double, Dynamic> retained_values(approximation.singularValues());

    matrix_type nonfinite(second);
    nonfinite(0, 0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW(approximation.compute(nonfinite, 2, 2), std::invalid_argument);
    EXPECT_THROW(approximation.compute(second, 0, 2), std::invalid_argument);
    EXPECT_THROW(approximation.compute(second, 6, 2), std::invalid_argument);
    EXPECT_THROW(approximation.compute(second, 2, 0), std::invalid_argument);
    EXPECT_THROW(approximation.compute(second, 2, 6), std::invalid_argument);
    matrix_type empty(0, 0);
    EXPECT_THROW(approximation.compute(empty, 1), std::invalid_argument);
    EXPECT_DOUBLE_EQ((approximation.matrixU() - retained_u).norm(), 0.0);
    EXPECT_DOUBLE_EQ((approximation.matrixV() - retained_v).norm(), 0.0);
    EXPECT_DOUBLE_EQ((approximation.singularValues() - retained_values).norm(), 0.0);

    EXPECT_THROW(static_cast<void>(approximation_type(-1.0, 8, 1)), std::invalid_argument);
    EXPECT_THROW(static_cast<void>(approximation_type(1.0e-5, 0, 1)), std::invalid_argument);
    EXPECT_THROW(
      static_cast<void>(approximation_type(std::numeric_limits<double>::quiet_NaN(), 8, 1)), std::invalid_argument);
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
// TODO(P4-M): restore seeded NysRSI and NysRBKI leading-spectrum assertions in their own native
// compact-decomposition slices. Eigen may be an opt-in test oracle only, never a production dependency.

}   // namespace
