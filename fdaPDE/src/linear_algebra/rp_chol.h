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

#ifndef __FDAPDE_LINALG_RP_CHOL_H__
#define __FDAPDE_LINALG_RP_CHOL_H__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "header_check.h"

namespace fdapde {

/// @brief builds a randomized low-rank Cholesky factor for a symmetric positive-semidefinite matrix
template <internals::matrix_expression MatrixType_> class RpChol {
   public:
    using MatrixType = std::remove_cvref_t<MatrixType_>;
    using Scalar = std::remove_cv_t<typename MatrixType::Scalar>;
    using FactorType = Matrix<Scalar, Dynamic, Dynamic, MatrixType::StorageOrder>;
    fdapde_static_assert(std::is_floating_point_v<Scalar>, RP_CHOL_REQUIRES_FLOATING_POINT_SCALARS);
    fdapde_static_assert(
      MatrixType::Rows == Dynamic || MatrixType::Cols == Dynamic || MatrixType::Rows == MatrixType::Cols,
      RP_CHOL_REQUIRES_A_SQUARE_MATRIX);

    /// @brief constructs an empty approximation with default sampling parameters
    RpChol() : seed_(resolve_seed_(random_seed)) { }
    /// @brief validates sampling limits and resolves the optional random seed
    RpChol(int block_size, int max_iterations, int seed = random_seed) :
        block_size_(block_size), max_iterations_(max_iterations), seed_(resolve_seed_(seed)) {
        validate_configuration_();
    }
    /// @brief constructs and computes an approximation with the requested relative residual tolerance
    RpChol(const MatrixType& matrix, Scalar tolerance, int block_size, int max_iterations, int seed = random_seed) :
        RpChol(block_size, max_iterations, seed) {
        compute(matrix, tolerance);
    }

    /// @brief builds replacement factor storage and pivots, preserving previous results on failure
    void compute(const MatrixType& matrix, Scalar tolerance) {
        validate_configuration_();
        fdapde_strong_assert(
          !(!std::isfinite(tolerance) || tolerance < Scalar(0) || tolerance >= Scalar(1)), std::invalid_argument,
          "RpChol tolerance must be finite and in [0, 1)");
        const int rows = matrix.rows();
        const int cols = matrix.cols();
        fdapde_strong_assert(
          !(rows <= 0 || rows != cols), std::invalid_argument, "RpChol requires a nonempty square matrix");
        (void)internals::checked_matrix_size(rows, cols);

        FactorType source(rows, cols);
        Scalar scale = Scalar(0);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                const Scalar value = static_cast<Scalar>(matrix(row, col));
                fdapde_strong_assert(
                  std::isfinite(value), std::invalid_argument, "RpChol requires finite matrix coefficients");
                source(row, col) = value;
                scale = std::max(scale, fdapde::abs(value));
            }
        }
        if (scale != Scalar(0)) {
            for (int row = 0; row < rows; ++row) {
                for (int col = 0; col < cols; ++col) source(row, col) /= scale;
            }
        }
        const Scalar roundoff = Scalar(64) * std::numeric_limits<Scalar>::epsilon() * static_cast<Scalar>(rows);
        // ponytail: these necessary PSD checks are not a certificate; use full factorization when certification is
        // needed
        for (int row = 0; row < rows; ++row) {
            fdapde_strong_assert(
              !(source(row, row) < -roundoff), std::domain_error, "RpChol requires a positive-semidefinite matrix");
            for (int col = row + 1; col < cols; ++col) {
                fdapde_strong_assert(
                  !(fdapde::abs(source(row, col) - source(col, row)) > roundoff), std::invalid_argument,
                  "RpChol requires a symmetric matrix");
                const Scalar diagonal_product = source(row, row) * source(col, col);
                const Scalar coefficient_square = source(row, col) * source(row, col);
                fdapde_strong_assert(
                  !(coefficient_square > diagonal_product + roundoff), std::domain_error,
                  "RpChol requires a positive-semidefinite matrix");
            }
        }

        const std::int64_t requested_capacity =
          static_cast<std::int64_t>(block_size_) * static_cast<std::int64_t>(max_iterations_);
        const int capacity = static_cast<int>(std::min<std::int64_t>(rows, requested_capacity));
        FactorType workspace(rows, capacity);
        workspace.set_zero();
        std::vector<Scalar> residual_diagonal(static_cast<std::size_t>(rows));
        for (int i = 0; i < rows; ++i) residual_diagonal[static_cast<std::size_t>(i)] = source(i, i);
        std::vector<bool> selected(static_cast<std::size_t>(rows), false);
        std::vector<int> pivots;
        pivots.reserve(static_cast<std::size_t>(capacity));
        std::mt19937 random_engine(seed_);

        const Scalar source_norm = source.norm();
        Scalar residual_norm = source_norm;
        int factor_rank = 0;
        while (factor_rank < capacity && residual_norm > tolerance * source_norm) {
            int available = 0;
            for (int i = 0; i < rows; ++i) {
                const Scalar residual = residual_diagonal[static_cast<std::size_t>(i)];
                if (!selected[static_cast<std::size_t>(i)] && residual > Scalar(0)) ++available;
            }
            fdapde_strong_assert(
              !(available == 0), std::domain_error, "RpChol factorization broke down before reaching its tolerance");

            const int batch_size = std::min({block_size_, capacity - factor_rank, available});
            std::vector<int> batch;
            batch.reserve(static_cast<std::size_t>(batch_size));
            std::vector<bool> batch_selected(selected);
            std::vector<double> weights(static_cast<std::size_t>(rows), 0.0);
            for (int i = 0; i < batch_size; ++i) {
                Scalar weight_scale = Scalar(0);
                for (int row = 0; row < rows; ++row) {
                    if (!batch_selected[static_cast<std::size_t>(row)]) {
                        weight_scale = std::max(weight_scale, residual_diagonal[static_cast<std::size_t>(row)]);
                    }
                }
                for (int row = 0; row < rows; ++row) {
                    const Scalar residual = residual_diagonal[static_cast<std::size_t>(row)];
                    const bool eligible = !batch_selected[static_cast<std::size_t>(row)] && residual > Scalar(0);
                    weights[static_cast<std::size_t>(row)] =
                      eligible ? static_cast<double>(residual / weight_scale) : 0.0;
                }
                std::discrete_distribution<int> distribution(weights.begin(), weights.end());
                const int pivot = distribution(random_engine);
                batch.push_back(pivot);
                batch_selected[static_cast<std::size_t>(pivot)] = true;
            }

            const int previous_rank = factor_rank;
            for (const int pivot : batch) {
                Scalar pivot_residual = source(pivot, pivot);
                for (int k = 0; k < factor_rank; ++k) { pivot_residual -= workspace(pivot, k) * workspace(pivot, k); }
                fdapde_strong_assert(
                  !(pivot_residual < -roundoff), std::domain_error, "RpChol requires a positive-semidefinite matrix");
                if (!(pivot_residual > Scalar(0))) continue;

                const Scalar denominator = std::sqrt(pivot_residual);
                for (int row = 0; row < rows; ++row) {
                    Scalar value = source(row, pivot);
                    for (int k = 0; k < factor_rank; ++k) { value -= workspace(row, k) * workspace(pivot, k); }
                    value /= denominator;
                    fdapde_strong_assert(
                      std::isfinite(value), std::domain_error, "RpChol produced a nonfinite factor coefficient");
                    workspace(row, factor_rank) = value;
                }
                selected[static_cast<std::size_t>(pivot)] = true;
                pivots.push_back(pivot);
                ++factor_rank;

                for (int row = 0; row < rows; ++row) {
                    Scalar& residual = residual_diagonal[static_cast<std::size_t>(row)];
                    residual -= workspace(row, factor_rank - 1) * workspace(row, factor_rank - 1);
                    fdapde_strong_assert(
                      !(residual < -roundoff), std::domain_error, "RpChol requires a positive-semidefinite matrix");
                    if (residual < Scalar(0)) residual = Scalar(0);
                }
            }
            fdapde_strong_assert(
              !(factor_rank == previous_rank), std::domain_error, "RpChol factorization made no numerical progress");
            residual_norm = residual_norm_(source, workspace, factor_rank);
        }

        FactorType factor(rows, factor_rank);
        const Scalar factor_scale = std::sqrt(scale);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < factor_rank; ++col) {
                factor(row, col) = workspace(row, col) * factor_scale;
                fdapde_strong_assert(
                  std::isfinite(factor(row, col)), std::domain_error,
                  "RpChol factor coefficients are not representable");
            }
        }
        factor_ = std::move(factor);
        pivots_ = std::move(pivots);
    }

    /// @brief borrows the owning rectangular Cholesky factor from a live lvalue approximation
    const FactorType& factor() const& { return factor_; }
    /// @brief rejects borrowing factor storage from a temporary approximation
    void factor() const&& = delete;
    /// @brief borrows the distinct pivot indices in their selection order
    const std::vector<int>& pivots() const& { return pivots_; }
    /// @brief rejects borrowing pivot storage from a temporary approximation
    void pivots() const&& = delete;
    /// @brief returns the number of columns in the current factor
    int rank() const { return factor_.cols(); }
   private:
    /// @brief uses the supplied seed or obtains one nondeterministic seed at construction
    static unsigned int resolve_seed_(int seed) {
        return seed == random_seed ? std::random_device {}() : static_cast<unsigned int>(seed);
    }

    /// @brief checks positive block size and iteration limit before sampling
    void validate_configuration_() const {
        fdapde_strong_assert(!(block_size_ <= 0), std::invalid_argument, "RpChol block size must be positive");
        fdapde_strong_assert(
          !(max_iterations_ <= 0), std::invalid_argument, "RpChol maximum iterations must be positive");
    }

    /// @brief accumulates a scale-safe Frobenius reconstruction residual for the current factor rank
    static Scalar residual_norm_(const FactorType& matrix, const FactorType& factor, int rank) {
        Scalar norm = Scalar(0);
        for (int row = 0; row < matrix.rows(); ++row) {
            for (int col = 0; col < matrix.cols(); ++col) {
                Scalar residual = matrix(row, col);
                for (int k = 0; k < rank; ++k) residual -= factor(row, k) * factor(col, k);
                fdapde_strong_assert(
                  std::isfinite(residual), std::domain_error, "RpChol produced a nonfinite reconstruction residual");
                norm = internals::scale_safe_hypot(norm, residual);
            }
        }
        return norm;
    }

    FactorType factor_;
    std::vector<int> pivots_;
    int block_size_ = 1;
    int max_iterations_ = 50;
    unsigned int seed_;
};

template <internals::matrix_expression MatrixType>
RpChol(const MatrixType&, typename std::remove_cvref_t<MatrixType>::Scalar, int, int)
  -> RpChol<std::remove_cvref_t<MatrixType>>;

template <internals::matrix_expression MatrixType>
RpChol(const MatrixType&, typename std::remove_cvref_t<MatrixType>::Scalar, int, int, int)
  -> RpChol<std::remove_cvref_t<MatrixType>>;

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_RP_CHOL_H__
