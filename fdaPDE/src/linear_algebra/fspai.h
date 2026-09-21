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

#ifndef __FDAPDE_LINALG_FSPAI_H__
#define __FDAPDE_LINALG_FSPAI_H__

#include <algorithm>
#include <cmath>
#include <concepts>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "header_check.h"

namespace fdapde {
namespace internals {

/// @brief solves an internal positive-definite principal system using scaled Cholesky and triangular substitution
template <std::floating_point Scalar>
Vector<Scalar, Dynamic>
fspai_spd_solve(const Matrix<Scalar, Dynamic, Dynamic>& matrix, const Vector<Scalar, Dynamic>& rhs) {
    const int n = matrix.rows();
    fdapde_assert(
      !(n <= 0 || matrix.cols() != n || rhs.size() != n), std::invalid_argument,
      "FSPAI local solve requires a nonempty square system and matching right-hand side");

    Scalar scale = Scalar(0);
    for (int row = 0; row < n; ++row) {
        fdapde_assert(std::isfinite(rhs[row]), std::invalid_argument, "FSPAI local solve requires finite coefficients");
        for (int col = 0; col < n; ++col) {
            const Scalar value = matrix(row, col);
            fdapde_assert(
              std::isfinite(value), std::invalid_argument, "FSPAI local solve requires finite coefficients");
            scale = std::max(scale, fdapde::abs(value));
        }
    }
    fdapde_strong_assert(
      scale > Scalar(0), std::domain_error, "FSPAI local Cholesky factorization requires positive pivots");

    Matrix<Scalar, Dynamic, Dynamic> lower(n, n);
    lower.set_zero();
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col <= row; ++col) {
            Scalar value = matrix(row, col) / scale;
            for (int k = 0; k < col; ++k) value -= lower(row, k) * lower(col, k);
            if (row == col) {
                fdapde_strong_assert(
                  value > Scalar(0) && std::isfinite(value), std::domain_error,
                  "FSPAI local Cholesky factorization requires positive finite pivots");
                lower(row, col) = std::sqrt(value);
            } else {
                value /= lower(col, col);
                fdapde_strong_assert(
                  std::isfinite(value), std::domain_error,
                  "FSPAI local Cholesky factorization produced a nonfinite coefficient");
                lower(row, col) = value;
            }
        }
    }

    Vector<Scalar, Dynamic> solution(n);
    for (int row = 0; row < n; ++row) {
        Scalar value = rhs[row] / scale;
        for (int col = 0; col < row; ++col) value -= lower(row, col) * solution[col];
        solution[row] = value / lower(row, row);
    }
    for (int row = n - 1; row >= 0; --row) {
        Scalar value = solution[row];
        for (int col = row + 1; col < n; ++col) value -= lower(col, row) * solution[col];
        solution[row] = value / lower(row, row);
        fdapde_strong_assert(
          std::isfinite(solution[row]), std::domain_error,
          "FSPAI local Cholesky solve produced a nonfinite coefficient");
    }
    return solution;
}

}   // namespace internals

/// @brief computes an owning lower factor L whose product L L-transpose approximates an SPD inverse
template <typename Scalar_> class FSPAI {
   public:
    using Scalar = std::remove_cvref_t<Scalar_>;
    using Index = int;
    using SparseMatrixType = SparseMatrix<Scalar>;
    using DenseMatrixType = Matrix<Scalar, Dynamic, Dynamic>;
    using DenseVectorType = Vector<Scalar, Dynamic>;

    fdapde_static_assert(
      (std::is_same_v<Scalar_, Scalar> && std::is_floating_point_v<Scalar>),
      FSPAI_REQUIRES_AN_UNQUALIFIED_FLOATING_POINT_SCALAR);

    /// @brief constructs an unavailable factorization with default update parameters
    FSPAI() = default;
    /// @brief copies the sparse factor and update parameters into independent storage
    FSPAI(const FSPAI&) = default;
    /// @brief transfers the factor and marks its source unavailable
    FSPAI(FSPAI&& other) noexcept :
        factor_(std::move(other.factor_)),
        alpha_(other.alpha_),
        beta_(other.beta_),
        epsilon_(other.epsilon_),
        ready_(std::exchange(other.ready_, false)) { }
    /// @brief copies the factor and update parameters with sparse-owner assignment semantics
    FSPAI& operator=(const FSPAI&) = default;
    /// @brief transfers the factor and parameters, tolerating self-move
    FSPAI& operator=(FSPAI&& other) noexcept {
        if (this == &other) return *this;
        factor_ = std::move(other.factor_);
        alpha_ = other.alpha_;
        beta_ = other.beta_;
        epsilon_ = other.epsilon_;
        ready_ = std::exchange(other.ready_, false);
        return *this;
    }

    /// @brief computes a factor using the supplied pattern-update limits and score tolerance
    explicit FSPAI(const SparseMatrixType& matrix, int alpha = 10, int beta = 10, Scalar epsilon = Scalar(0.005)) :
        alpha_(alpha), beta_(beta), epsilon_(epsilon) {
        compute(matrix);
    }

    /// @brief recomputes using the parameters from the last successful computation
    void compute(const SparseMatrixType& matrix) { compute_impl_(matrix, alpha_, beta_, epsilon_); }
    /// @brief computes and commits a new factor and parameters only after all checks succeed
    void compute(const SparseMatrixType& matrix, int alpha, int beta, Scalar epsilon) {
        compute_impl_(matrix, alpha, beta, epsilon);
    }

    /// @brief returns the factor row count, or zero before computation and after a move
    Index rows() const { return factor_.rows(); }
    /// @brief returns the factor column count, or zero before computation and after a move
    Index cols() const { return factor_.cols(); }
    /// @brief borrows the lower factor from a successfully computed lvalue owner
    const SparseMatrixType& lower_factor() const& {
        require_ready_();
        return factor_;
    }
    /// @brief rejects borrowing a factor from a temporary owner
    void lower_factor() const&& = delete;
    /// @brief returns an independent transpose of the computed lower factor
    SparseMatrixType upper_factor() const {
        require_ready_();
        return factor_.transpose();
    }
    /// @brief materializes the complete approximate inverse L L-transpose
    SparseMatrixType inverse() const {
        require_ready_();
        return factor_ * factor_.transpose();
    }

    /// @brief applies the complete approximate inverse to a matching dense right-hand side
    template <internals::matrix_expression RhsXprType>
        requires(std::convertible_to<typename RhsXprType::Scalar, Scalar>)
    Matrix<Scalar, Dynamic, Dynamic> solve(const MatrixExpr<RhsXprType>& rhs) const {
        require_ready_();
        const RhsXprType& rhs_derived = rhs.derived();
        validate_dense_rhs_(rhs_derived);
        const Matrix<Scalar, Dynamic, Dynamic> rhs_owned(rhs_derived);
        const Matrix<Scalar, Dynamic, Dynamic> result(inverse() * rhs_owned);
        validate_finite_dense_result_(result);
        return result;
    }

    /// @brief applies the complete approximate inverse to a matching sparse right-hand side
    SparseMatrixType solve(const SparseMatrixType& rhs) const {
        require_ready_();
        validate_sparse_rhs_(rhs);
        const SparseMatrixType result = inverse() * rhs;
        validate_finite_sparse_result_(result);
        return result;
    }

    /// @brief replaces a dense right-hand side only after its complete solution is available
    template <int Rows, int Cols, int StorageOrder>
    void solve_in_place(Matrix<Scalar, Rows, Cols, StorageOrder>& rhs) const {
        const Matrix<Scalar, Dynamic, Dynamic> result = solve(rhs);
        Matrix<Scalar, Rows, Cols, StorageOrder> replacement(result);
        rhs = replacement;
    }

    /// @brief replaces sparse right-hand-side storage only after its complete solution is available
    void solve_in_place(SparseMatrixType& rhs) const {
        SparseMatrixType replacement = solve(rhs);
        rhs.swap(replacement);
    }
   private:
    /// @brief checks public dense right-hand-side shape and finite converted coefficients
    template <internals::matrix_expression RhsXprType> void validate_dense_rhs_(const RhsXprType& rhs) const {
        fdapde_strong_assert(
          !(rhs.rows() != rows() || rhs.cols() <= 0), std::invalid_argument,
          "FSPAI solve requires a matching nonempty dense right-hand side");
        for (int row = 0; row < rhs.rows(); ++row) {
            for (int col = 0; col < rhs.cols(); ++col) {
                fdapde_strong_assert(
                  std::isfinite(static_cast<Scalar>(rhs(row, col))), std::invalid_argument,
                  "FSPAI solve requires finite right-hand-side coefficients");
            }
        }
    }

    /// @brief checks public sparse right-hand-side shape and finite stored coefficients
    void validate_sparse_rhs_(const SparseMatrixType& rhs) const {
        fdapde_strong_assert(
          !(rhs.rows() != rows() || rhs.cols() <= 0), std::invalid_argument,
          "FSPAI solve requires a matching nonempty sparse right-hand side");
        for (int row = 0; row < rhs.rows(); ++row) {
            for (const auto entry : rhs.row(row)) {
                fdapde_strong_assert(
                  std::isfinite(entry.value()), std::invalid_argument,
                  "FSPAI solve requires finite right-hand-side coefficients");
            }
        }
    }

    /// @brief rejects nonfinite dense coefficients produced by the approximate inverse
    static void validate_finite_dense_result_(const Matrix<Scalar, Dynamic, Dynamic>& result) {
        for (int row = 0; row < result.rows(); ++row) {
            for (int col = 0; col < result.cols(); ++col) {
                fdapde_strong_assert(
                  std::isfinite(result(row, col)), std::domain_error, "FSPAI solve produced a nonfinite coefficient");
            }
        }
    }

    /// @brief rejects nonfinite sparse coefficients produced by the approximate inverse
    static void validate_finite_sparse_result_(const SparseMatrixType& result) {
        for (int row = 0; row < result.rows(); ++row) {
            for (const auto entry : result.row(row)) {
                fdapde_strong_assert(
                  std::isfinite(entry.value()), std::domain_error, "FSPAI solve produced a nonfinite coefficient");
            }
        }
    }

    /// @brief validates the SPD preconditions and constructs replacement factor storage before publishing it
    void compute_impl_(const SparseMatrixType& matrix, int alpha, int beta, Scalar epsilon) {
        fdapde_strong_assert(
          !(alpha < 0 || beta < 0 || !std::isfinite(epsilon) || epsilon < Scalar(0)), std::invalid_argument,
          "FSPAI requires nonnegative update parameters and a finite nonnegative tolerance");
        const int n = matrix.rows();
        fdapde_strong_assert(
          !(n <= 0 || matrix.cols() != n), std::invalid_argument, "FSPAI requires a nonempty square matrix");

        Scalar scale = Scalar(0);
        std::vector<Triplet<Scalar>> normalized_triplets;
        normalized_triplets.reserve(static_cast<std::size_t>(matrix.non_zeros()));
        for (int row = 0; row < n; ++row) {
            for (const auto entry : matrix.row(row)) {
                fdapde_strong_assert(
                  std::isfinite(entry.value()), std::invalid_argument, "FSPAI requires finite matrix coefficients");
                scale = std::max(scale, fdapde::abs(entry.value()));
            }
        }
        fdapde_strong_assert(scale > Scalar(0), std::domain_error, "FSPAI requires a positive-definite matrix");
        for (int row = 0; row < n; ++row) {
            for (const auto entry : matrix.row(row)) {
                const Scalar value = entry.value() / scale;
                if (value != Scalar {}) normalized_triplets.emplace_back(row, entry.column(), value);
            }
        }
        const SparseMatrixType source(n, n, normalized_triplets);

        const Scalar roundoff = Scalar(64) * std::numeric_limits<Scalar>::epsilon();
        // ponytail: these sparse checks cannot certify global SPD; add a sparse factorization if certification is
        // needed
        std::vector<Scalar> diagonal(static_cast<std::size_t>(n));
        for (int row = 0; row < n; ++row) {
            const Scalar value = source.coeff(row, row);
            fdapde_strong_assert(
              value > Scalar(0) && std::isfinite(value), std::domain_error,
              "FSPAI requires positive finite diagonal coefficients");
            diagonal[static_cast<std::size_t>(row)] = value;
            for (const auto entry : source.row(row)) {
                const int col = entry.column();
                if (col == row) continue;
                fdapde_strong_assert(
                  !(fdapde::abs(entry.value() - source.coeff(col, row)) > roundoff), std::invalid_argument,
                  "FSPAI requires a symmetric matrix");
                if (col < row) continue;
                const Scalar limit = std::sqrt(value) * std::sqrt(diagonal_value_(source, diagonal, col));
                fdapde_strong_assert(
                  !(fdapde::abs(entry.value()) >= limit), std::domain_error,
                  "FSPAI requires a positive-definite matrix");
            }
        }

        // grow each lower-factor column independently from its diagonal seed
        std::vector<Triplet<Scalar>> factor_triplets;
        std::vector<Scalar> column(static_cast<std::size_t>(n), Scalar(0));
        std::vector<unsigned char> candidate_mask(static_cast<std::size_t>(n), 0);
        std::vector<unsigned char> pattern_mask(static_cast<std::size_t>(n), 0);
        for (int k = 0; k < n; ++k) {
            std::fill(column.begin(), column.end(), Scalar(0));
            std::fill(candidate_mask.begin(), candidate_mask.end(), 0);
            std::fill(pattern_mask.begin(), pattern_mask.end(), 0);
            std::vector<int> candidates;
            std::vector<int> pattern {k};
            std::vector<int> added {k};
            pattern_mask[static_cast<std::size_t>(k)] = 1;
            column[static_cast<std::size_t>(k)] = Scalar(1) / std::sqrt(diagonal[static_cast<std::size_t>(k)]);

            for (int step = 0; step < alpha && !added.empty(); ++step) {
                if (step != 0) update_column_(source, diagonal, k, pattern, column);

                // new pattern entries expose neighboring candidates for the next local principal solve
                for (const int row : added) {
                    for (const auto entry : source.row(row)) {
                        const int candidate = entry.column();
                        if (candidate > k && candidate_mask[static_cast<std::size_t>(candidate)] == 0) {
                            candidate_mask[static_cast<std::size_t>(candidate)] = 1;
                            candidates.push_back(candidate);
                        }
                    }
                }
                added.clear();

                // rank candidate gradient contributions relative to their diagonal curvature
                std::vector<std::pair<int, Scalar>> scores;
                scores.reserve(candidates.size());
                Scalar score_sum = Scalar(0);
                Scalar maximum_score = Scalar(0);
                for (const int candidate : candidates) {
                    if (pattern_mask[static_cast<std::size_t>(candidate)] != 0) continue;
                    Scalar value = Scalar(0);
                    for (const int index : pattern) {
                        value += Scalar(2) * source.coeff(candidate, index) * column[static_cast<std::size_t>(index)];
                    }
                    const Scalar score = value * value / diagonal[static_cast<std::size_t>(candidate)];
                    fdapde_strong_assert(
                      std::isfinite(score), std::domain_error, "FSPAI sparsity update produced a nonfinite score");
                    scores.emplace_back(candidate, score);
                    score_sum += score;
                    maximum_score = std::max(maximum_score, score);
                }
                fdapde_strong_assert(
                  std::isfinite(score_sum), std::domain_error, "FSPAI sparsity update produced a nonfinite score");
                if (scores.empty() || !(maximum_score > epsilon)) continue;

                // retain at most beta candidates above the mean score, with deterministic index tie-breaking
                const Scalar mean = score_sum / static_cast<Scalar>(scores.size());
                std::sort(scores.begin(), scores.end(), [](const auto& lhs, const auto& rhs) {
                    return lhs.second != rhs.second ? lhs.second > rhs.second : lhs.first < rhs.first;
                });
                int selected = 0;
                for (const auto& [index, score] : scores) {
                    if (selected == beta || score < mean) break;
                    pattern.push_back(index);
                    added.push_back(index);
                    pattern_mask[static_cast<std::size_t>(index)] = 1;
                    ++selected;
                }
            }

            for (const int row : pattern) {
                const Scalar value = column[static_cast<std::size_t>(row)] / std::sqrt(scale);
                fdapde_strong_assert(
                  std::isfinite(value), std::domain_error, "FSPAI factor coefficients are not representable");
                if (value != Scalar {}) factor_triplets.emplace_back(row, k, value);
            }
        }

        // publish the completed factor and its settings only after every column succeeds
        SparseMatrixType replacement(n, n, factor_triplets);
        factor_.swap(replacement);
        alpha_ = alpha;
        beta_ = beta;
        epsilon_ = epsilon;
        ready_ = true;
    }

    /// @brief loads a diagonal coefficient into the local cache on first access
    static Scalar diagonal_value_(const SparseMatrixType& matrix, std::vector<Scalar>& cached_diagonal, int index) {
        Scalar& value = cached_diagonal[static_cast<std::size_t>(index)];
        if (value == Scalar(0)) value = matrix.coeff(index, index);
        return value;
    }

    /// @brief solves the selected principal system and normalizes its inverse-factor column
    static void update_column_(
      const SparseMatrixType& matrix, const std::vector<Scalar>& diagonal, int k, const std::vector<int>& pattern,
      std::vector<Scalar>& column) {
        const int size = static_cast<int>(pattern.size()) - 1;
        Matrix<Scalar, Dynamic, Dynamic> system(size, size);
        Vector<Scalar, Dynamic> rhs(size);
        for (int row = 0; row < size; ++row) {
            const int source_row = pattern[static_cast<std::size_t>(row + 1)];
            rhs[row] = matrix.coeff(source_row, k);
            for (int col = 0; col < size; ++col) {
                system(row, col) = matrix.coeff(source_row, pattern[static_cast<std::size_t>(col + 1)]);
            }
        }
        const Vector<Scalar, Dynamic> solution = internals::fspai_spd_solve(system, rhs);
        Scalar correction = Scalar(0);
        for (int i = 0; i < size; ++i) correction += rhs[i] * solution[i];
        const Scalar schur = diagonal[static_cast<std::size_t>(k)] - correction;
        fdapde_strong_assert(
          schur > Scalar(0) && std::isfinite(schur), std::domain_error,
          "FSPAI update requires a positive finite Schur complement");
        const Scalar diagonal_factor = Scalar(1) / std::sqrt(schur);
        fdapde_strong_assert(
          std::isfinite(diagonal_factor), std::domain_error, "FSPAI factor coefficients are not representable");
        column[static_cast<std::size_t>(k)] = diagonal_factor;
        for (int i = 0; i < size; ++i) {
            const Scalar value = -diagonal_factor * solution[i];
            fdapde_strong_assert(
              std::isfinite(value), std::domain_error, "FSPAI factor coefficients are not representable");
            column[static_cast<std::size_t>(pattern[static_cast<std::size_t>(i + 1)])] = value;
        }
    }

    /// @brief rejects factor access and solves before a successful computation
    void require_ready_() const {
        fdapde_strong_assert(!(!ready_), std::domain_error, "FSPAI factor access requires a successful computation");
    }

    SparseMatrixType factor_;
    int alpha_ = 10;
    int beta_ = 10;
    Scalar epsilon_ = Scalar(0.005);
    bool ready_ = false;
};

template <typename Scalar> FSPAI(const SparseMatrix<Scalar>&) -> FSPAI<Scalar>;
template <typename Scalar, typename Epsilon>
    requires std::convertible_to<Epsilon, Scalar>
FSPAI(const SparseMatrix<Scalar>&, int, int, Epsilon) -> FSPAI<Scalar>;

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_FSPAI_H__
