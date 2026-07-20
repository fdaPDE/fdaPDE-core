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

#ifndef __FDAPDE_LINALG_NATIVE_QR_H__
#define __FDAPDE_LINALG_NATIVE_QR_H__

#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

#include "header_check.h"

namespace fdapde::linalg {

// Full Householder QR following Golub and Van Loan, Matrix Computations,
// Algorithm 5.1.1: A = Q * R, Q is rows-by-rows and R has A's shape.
template <typename Scalar_, int Rows_, int Cols_> class HouseholderQR {
   public:
    using Scalar = std::remove_cv_t<Scalar_>;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    fdapde_static_assert(std::is_floating_point_v<Scalar>, QR_DECOMPOSITION_REQUIRES_FLOATING_POINT_SCALARS);

    constexpr HouseholderQR() = default;
    template <typename MatrixType> constexpr explicit HouseholderQR(const MatrixExpr<MatrixType>& matrix) {
        compute(matrix);
    }

    template <typename MatrixType> constexpr void compute(const MatrixExpr<MatrixType>& matrix) {
        fdapde_static_assert(
          MatrixType::Rows == Dynamic || Rows == Dynamic || MatrixType::Rows == Rows, INVALID_QR_MATRIX_STATIC_SHAPE);
        fdapde_static_assert(
          MatrixType::Cols == Dynamic || Cols == Dynamic || MatrixType::Cols == Cols, INVALID_QR_MATRIX_STATIC_SHAPE);
        const int rows = matrix.rows();
        const int cols = matrix.cols();
        const bool shape_valid =
          rows > 0 && cols > 0 && (Rows == Dynamic || rows == Rows) && (Cols == Dynamic || cols == Cols);
        if (!shape_valid) {
            fdapde_assert(shape_valid);
            computed_ = false;
            rank_ = 0;
            return;
        }

        if constexpr (Rows == Dynamic) Q_.resize(rows, rows);
        if constexpr (Rows == Dynamic || Cols == Dynamic) R_.resize(rows, cols);
        Scalar scale = Scalar(0);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                const Scalar value = static_cast<Scalar>(matrix.derived()(i, j));
                if (!std::isfinite(value)) {
                    fdapde_assert(std::isfinite(value));
                    computed_ = false;
                    rank_ = 0;
                    return;
                }
                scale = fdapde::max(scale, fdapde::abs(value));
                R_(i, j) = value;
            }
        }
        const Scalar normalization = scale == Scalar(0) ? Scalar(1) : scale;
        R_ /= normalization;
        Q_.set_zero();
        for (int i = 0; i < rows; ++i) Q_(i, i) = Scalar(1);

        const int reflectors = fdapde::min(rows, cols);
        for (int k = 0; k < reflectors; ++k) {
            const int length = rows - k;
            std::vector<Scalar> reflector(static_cast<std::size_t>(length));
            Scalar norm = Scalar(0);
            for (int i = 0; i < length; ++i) {
                reflector[static_cast<std::size_t>(i)] = R_(k + i, k);
                norm = std::hypot(norm, reflector[static_cast<std::size_t>(i)]);
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
                for (int i = 0; i < length; ++i) dot += reflector[static_cast<std::size_t>(i)] * R_(k + i, col);
                dot *= beta;
                for (int i = 0; i < length; ++i) R_(k + i, col) -= reflector[static_cast<std::size_t>(i)] * dot;
            }
            R_(k, k) = alpha * norm;
            for (int i = k + 1; i < rows; ++i) R_(i, k) = Scalar(0);

            // Q <- Q * H. Householder reflectors are symmetric.
            for (int row = 0; row < rows; ++row) {
                Scalar dot = Scalar(0);
                for (int i = 0; i < length; ++i) dot += Q_(row, k + i) * reflector[static_cast<std::size_t>(i)];
                dot *= beta;
                for (int i = 0; i < length; ++i) Q_(row, k + i) -= dot * reflector[static_cast<std::size_t>(i)];
            }
        }
        const Scalar tolerance = std::numeric_limits<Scalar>::epsilon() * static_cast<Scalar>(fdapde::max(rows, cols));
        rank_ = numerical_rank_(R_, tolerance);
        R_ *= normalization;
        computed_ = true;
    }

    constexpr const Matrix<Scalar, Rows, Rows>& Q() const& {
        fdapde_assert(computed_);
        return Q_;
    }
    constexpr void Q() const&& = delete;
    constexpr const Matrix<Scalar, Rows, Cols>& R() const& {
        fdapde_assert(computed_);
        return R_;
    }
    constexpr void R() const&& = delete;
    constexpr int rank() const { return rank_; }
    constexpr bool computed() const { return computed_; }
   private:
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

    Matrix<Scalar, Rows, Rows> Q_;
    Matrix<Scalar, Rows, Cols> R_;
    int rank_ = 0;
    bool computed_ = false;
};

template <typename XprType>
HouseholderQR(const MatrixExpr<XprType>&) -> HouseholderQR<typename XprType::Scalar, XprType::Rows, XprType::Cols>;

}   // namespace fdapde::linalg

#endif   // __FDAPDE_LINALG_NATIVE_QR_H__
