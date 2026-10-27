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

#ifndef __FDAPDE_LINALG_QR_H__
#define __FDAPDE_LINALG_QR_H__

#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "header_check.h"

namespace fdapde {

// full Householder QR following Golub and Van Loan, Matrix Computations,
// algorithm 5.1.1: A = Q * R, Q is rows-by-rows and R has A's shape
/// @brief factors a rectangular matrix using Householder reflections
template <typename Scalar_, int Rows_, int Cols_> class HouseholderQR {
   public:
    using Scalar = std::remove_cv_t<Scalar_>;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    fdapde_static_assert(std::is_floating_point_v<Scalar>, QR_DECOMPOSITION_REQUIRES_FLOATING_POINT_SCALARS);

    /// @brief creates an uncomputed decomposition with no available factors
    constexpr HouseholderQR() = default;
    /// @brief computes full Householder QR factors from a finite, nonempty matrix
    template <typename MatrixType> constexpr explicit HouseholderQR(const MatrixExpr<MatrixType>& matrix) {
        compute(matrix);
    }

    /// @brief computes the factorization of the supplied matrix
    template <typename MatrixType> constexpr void compute(const MatrixExpr<MatrixType>& matrix) {
        fdapde_static_assert(
          MatrixType::Rows == Dynamic || Rows == Dynamic || MatrixType::Rows == Rows, INVALID_QR_MATRIX_STATIC_SHAPE);
        fdapde_static_assert(
          MatrixType::Cols == Dynamic || Cols == Dynamic || MatrixType::Cols == Cols, INVALID_QR_MATRIX_STATIC_SHAPE);

        computed_ = false;
        rank_ = 0;
        const int rows = matrix.rows();
        const int cols = matrix.cols();
        const bool shape_valid =
          rows > 0 && cols > 0 && (Rows == Dynamic || rows == Rows) && (Cols == Dynamic || cols == Cols);
        fdapde_strong_assert(
          !(!shape_valid), std::invalid_argument, "HouseholderQR requires a nonempty matrix matching its static shape");

        Matrix<Scalar, Rows, Rows> q;
        if constexpr (Rows == Dynamic) q.resize(rows, rows);
        if constexpr (Rows == Dynamic || Cols == Dynamic) R_.resize(rows, cols);
        Scalar scale = Scalar(0);
        for (int row = 0; row < rows; ++row) {
            for (int col = 0; col < cols; ++col) {
                const Scalar value = static_cast<Scalar>(matrix.derived()(row, col));
                fdapde_strong_assert(
                  !(!is_finite_(value)), std::invalid_argument, "HouseholderQR requires finite matrix coefficients");
                scale = fdapde::max(scale, fdapde::abs(value));
                R_(row, col) = value;
            }
        }

        const Scalar normalization = scale == Scalar(0) ? Scalar(1) : scale;
        R_ /= normalization;
        q.set_zero();
        for (int i = 0; i < rows; ++i) q(i, i) = Scalar(1);

        const int reflectors = fdapde::min(rows, cols);
        for (int k = 0; k < reflectors; ++k) {
            const int length = rows - k;
            std::vector<Scalar> reflector(static_cast<std::size_t>(length));
            Scalar norm = Scalar(0);
            for (int i = 0; i < length; ++i) {
                reflector[static_cast<std::size_t>(i)] = R_(k + i, k);
                norm = internals::scale_safe_hypot(norm, reflector[static_cast<std::size_t>(i)]);
            }
            if (norm == Scalar(0)) continue;

            for (Scalar& value : reflector) value /= norm;
            const Scalar alpha = -std::copysign(Scalar(1), reflector[0]);
            reflector[0] -= alpha;
            Scalar squared_norm = Scalar(0);
            for (const Scalar value : reflector) squared_norm += value * value;
            if (squared_norm == Scalar(0)) continue;
            const Scalar beta = Scalar(2) / squared_norm;

            for (int col = k; col < cols; ++col) {
                Scalar dot = Scalar(0);
                for (int i = 0; i < length; ++i) { dot += reflector[static_cast<std::size_t>(i)] * R_(k + i, col); }
                dot *= beta;
                for (int i = 0; i < length; ++i) { R_(k + i, col) -= reflector[static_cast<std::size_t>(i)] * dot; }
            }
            R_(k, k) = alpha * norm;
            for (int i = k + 1; i < rows; ++i) R_(i, k) = Scalar(0);

            // q <- Q * H. Householder reflectors are symmetric
            for (int row = 0; row < rows; ++row) {
                Scalar dot = Scalar(0);
                for (int i = 0; i < length; ++i) { dot += q(row, k + i) * reflector[static_cast<std::size_t>(i)]; }
                dot *= beta;
                for (int i = 0; i < length; ++i) { q(row, k + i) -= dot * reflector[static_cast<std::size_t>(i)]; }
            }
        }

        const Scalar tolerance = std::numeric_limits<Scalar>::epsilon() * static_cast<Scalar>(fdapde::max(rows, cols));
        const int rank = numerical_rank_(R_, tolerance);
        R_ *= normalization;
        Q_.emplace(q);
        rank_ = rank;
        computed_ = true;
    }

    /// @brief returns the full square orthogonal factor after a completed decomposition
    constexpr const OrthogonalMatrix<Scalar, Rows>& Q() const& {
        fdapde_strong_assert(computed_, std::logic_error, "QR factorization has not been computed");
        return *Q_;
    }
    /// @brief rejects access through a temporary decomposition to prevent dangling factor references
    constexpr void Q() const&& = delete;
    /// @brief returns the upper-trapezoidal factor with the input matrix shape and scale
    constexpr const Matrix<Scalar, Rows, Cols>& R() const& {
        fdapde_assert(computed_, std::logic_error, "QR factorization has not been computed");
        return R_;
    }
    /// @brief rejects access through a temporary decomposition to prevent dangling factor references
    constexpr void R() const&& = delete;
    /// @brief returns the numerical rank
    constexpr int rank() const { return rank_; }
    /// @brief reports whether a factorization has been computed
    constexpr bool computed() const { return computed_; }

    /// @brief solves square or overdetermined full-column-rank systems for one or more finite right-hand sides
    /// @details returns the unique least-squares solution and rejects underdetermined or numerically deficient input
    template <typename RhsType> auto solve(const MatrixExpr<RhsType>& rhs) const {
        fdapde_strong_assert(computed_, std::logic_error, "QR factorization has not been computed");
        fdapde_static_assert(
          RhsType::Rows == Dynamic || Rows == Dynamic || RhsType::Rows == Rows, INVALID_QR_RHS_STATIC_SHAPE);
        const int rows = R_.rows(), cols = R_.cols();
        fdapde_strong_assert(
          rows >= cols, std::invalid_argument, "QR solve requires a square or overdetermined system");
        fdapde_strong_assert(rank_ == cols, std::domain_error, "QR solve requires full numerical column rank");
        fdapde_strong_assert(
          rhs.rows() == rows && rhs.cols() > 0, std::invalid_argument,
          "QR solve requires matching nonempty right-hand sides");
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < rhs.cols(); ++j)
                fdapde_strong_assert(
                  is_finite_(static_cast<Scalar>(rhs.derived()(i, j))), std::invalid_argument,
                  "QR solve requires finite right-hand sides");
        Scalar factor_scale = Scalar(0);
        for (int i = 0; i < cols; ++i) {
            for (int j = i; j < cols; ++j) {
                fdapde_strong_assert(
                  is_finite_(R_(i, j)), std::domain_error, "QR solve requires representable triangular factors");
                factor_scale = std::max(factor_scale, std::abs(R_(i, j)));
            }
            fdapde_strong_assert(
              R_(i, i) != Scalar(0), std::domain_error, "QR solve requires nonzero triangular pivots");
        }

        // normalize both factors and loads so projection and substitution do not depend on their absolute scale
        Matrix<long double, Cols, RhsType::Cols> projected;
        if constexpr (Cols == Dynamic || RhsType::Cols == Dynamic) projected.resize(cols, rhs.cols());
        int factor_exponent = 0;
        const long double factor_fraction = std::frexp(static_cast<long double>(factor_scale), &factor_exponent);
        for (int col = 0; col < rhs.cols(); ++col) {
            Scalar rhs_scale = Scalar(0);
            for (int k = 0; k < rows; ++k)
                rhs_scale = std::max(rhs_scale, std::abs(static_cast<Scalar>(rhs.derived()(k, col))));
            const Scalar normalization = rhs_scale == Scalar(0) ? Scalar(1) : rhs_scale;
            for (int i = 0; i < cols; ++i) {
                long double value = 0;
                for (int k = 0; k < rows; ++k)
                    value += static_cast<long double>((*Q_)(k, i)) *
                             (static_cast<long double>(static_cast<Scalar>(rhs.derived()(k, col))) / normalization);
                projected(i, col) = value;
            }
            for (int i = cols - 1; i >= 0; --i) {
                long double value = projected(i, col);
                for (int k = i + 1; k < cols; ++k)
                    value -= (static_cast<long double>(R_(i, k)) / factor_scale) * projected(k, col);
                projected(i, col) = value / (static_cast<long double>(R_(i, i)) / factor_scale);
            }
            // combine binary exponents without overflowing the scale ratio before restoring the solution
            int rhs_exponent = 0;
            const long double rhs_fraction = std::frexp(static_cast<long double>(normalization), &rhs_exponent);
            for (int i = 0; i < cols; ++i) {
                int exponent = 0;
                const long double fraction = std::frexp(projected(i, col), &exponent);
                projected(i, col) =
                  std::scalbn(fraction * rhs_fraction / factor_fraction, exponent + rhs_exponent - factor_exponent);
                fdapde_strong_assert(
                  is_finite_(static_cast<Scalar>(projected(i, col))), std::domain_error,
                  "QR solution is not representable");
            }
        }
        return Matrix<Scalar, Cols, RhsType::Cols>(projected);
    }
   private:
    /// @brief returns whether the scalar is finite and not NaN
    static constexpr bool is_finite_(Scalar value) {
        const Scalar infinity = std::numeric_limits<Scalar>::infinity();
        return value == value && value != infinity && value != -infinity;
    }

    /// @brief returns the estimated numerical rank
    template <typename MatrixType> static constexpr int numerical_rank_(const MatrixType& matrix, Scalar tolerance) {
        Matrix<Scalar, Dynamic, Dynamic> echelon(matrix);
        int pivot_row = 0;
        for (int col = 0; col < echelon.cols() && pivot_row < echelon.rows(); ++col) {
            int pivot = pivot_row;
            Scalar largest = Scalar(0);
            for (int row = pivot_row; row < echelon.rows(); ++row) {
                const Scalar candidate = fdapde::abs(echelon(row, col));
                if (candidate > largest) {
                    largest = candidate;
                    pivot = row;
                }
            }
            if (!(largest > tolerance)) continue;
            if (pivot != pivot_row) {
                for (int j = col; j < echelon.cols(); ++j) std::swap(echelon(pivot_row, j), echelon(pivot, j));
            }
            for (int row = pivot_row + 1; row < echelon.rows(); ++row) {
                const Scalar multiplier = echelon(row, col) / echelon(pivot_row, col);
                for (int j = col; j < echelon.cols(); ++j) echelon(row, j) -= multiplier * echelon(pivot_row, j);
            }
            ++pivot_row;
        }
        return pivot_row;
    }

    std::optional<OrthogonalMatrix<Scalar, Rows>> Q_;
    Matrix<Scalar, Rows, Cols> R_;
    int rank_ = 0;
    bool computed_ = false;
};

template <typename XprType>
HouseholderQR(const MatrixExpr<XprType>&) -> HouseholderQR<typename XprType::Scalar, XprType::Rows, XprType::Cols>;

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_QR_H__
