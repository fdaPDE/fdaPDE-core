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

#ifndef __FDAPDE_LINALG_NATIVE_EVD_H__
#define __FDAPDE_LINALG_NATIVE_EVD_H__

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "header_check.h"

namespace fdapde::linalg {

template <typename XprType_> class EVD {
    using XprType = std::decay_t<XprType_>;
    fdapde_static_assert(
      XprType::Rows == Dynamic || XprType::Cols == Dynamic || XprType::Rows == XprType::Cols,
      THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
    fdapde_static_assert(
      XprType::Rows == Dynamic || XprType::Cols == Dynamic ||
        std::int64_t(XprType::Rows) * std::int64_t(XprType::Cols) <= std::numeric_limits<int>::max(),
      EVD_DENSE_WORKSPACE_SIZE_EXCEEDS_SUPPORTED_RANGE);
   public:
    using Scalar = std::remove_cv_t<typename XprType::Scalar>;
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    fdapde_static_assert(std::is_floating_point_v<Scalar>, EVD_REQUIRES_FLOATING_POINT_SCALARS);

    constexpr EVD() = default;
    template <typename MatrixType> constexpr explicit EVD(const SymmetricMatrixExpr<MatrixType>& matrix) {
        compute(matrix);
    }

    // Maximum-pivot Jacobi rotations with scale normalization; see Golub and
    // Van Loan, Matrix Computations, Section 8.5.
    template <typename MatrixType> constexpr void compute(const SymmetricMatrixExpr<MatrixType>& matrix) {
        computed_ = false;
        fdapde_static_assert(
          MatrixType::Rows == Dynamic || Rows == Dynamic || MatrixType::Rows == Rows, INVALID_EVD_MATRIX_STATIC_SHAPE);
        fdapde_static_assert(
          MatrixType::Cols == Dynamic || Cols == Dynamic || MatrixType::Cols == Cols, INVALID_EVD_MATRIX_STATIC_SHAPE);
        const int n = matrix.rows();
        const bool shape_valid =
          n > 0 && n == matrix.cols() && (Rows == Dynamic || n == Rows) && (Cols == Dynamic || n == Cols);
        if (!shape_valid) {
            fdapde_assert(shape_valid);
            computed_ = false;
            return;
        }
        const std::int64_t dense_dimension = n;
        if (dense_dimension * dense_dimension > std::numeric_limits<int>::max()) {
            throw std::length_error("EVD: dense workspace size exceeds supported range");
        }

        Matrix<Scalar, Rows, Cols> diagonalized(matrix);
        Scalar matrix_scale = Scalar(0);
        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
                const Scalar value = diagonalized(row, col);
                if (!std::isfinite(value)) {
                    fdapde_assert(std::isfinite(value));
                    computed_ = false;
                    return;
                }
                matrix_scale = fdapde::max(matrix_scale, fdapde::abs(value));
            }
        }
        const Scalar normalization = matrix_scale == Scalar(0) ? Scalar(1) : matrix_scale;
        diagonalized /= normalization;

        if constexpr (Rows == Dynamic || Cols == Dynamic) eigenvectors_.resize(n, n);
        eigenvectors_.set_zero();
        for (int i = 0; i < n; ++i) eigenvectors_(i, i) = Scalar(1);

        const Scalar epsilon = std::numeric_limits<Scalar>::epsilon();
        const Scalar tolerance = fdapde::max(std::numeric_limits<Scalar>::min(), epsilon * static_cast<Scalar>(n));
        constexpr std::size_t iteration_factor = 50;
        const std::size_t dimension = static_cast<std::size_t>(n);
        const std::size_t max_size = std::numeric_limits<std::size_t>::max();
        const std::size_t max_iterations = dimension != 0 && dimension > max_size / iteration_factor / dimension ?
                                             max_size :
                                             iteration_factor * dimension * dimension;

        for (std::size_t iteration = 0;; ++iteration) {
            int p = 0;
            int q = 0;
            Scalar largest_off_diagonal = Scalar(0);
            for (int row = 0; row < n; ++row) {
                for (int col = row + 1; col < n; ++col) {
                    const Scalar candidate = fdapde::abs(diagonalized(row, col));
                    if (candidate > largest_off_diagonal) {
                        largest_off_diagonal = candidate;
                        p = row;
                        q = col;
                    }
                }
            }
            if (largest_off_diagonal <= tolerance) break;
            if (iteration == max_iterations) throw std::runtime_error("EVD: Jacobi iteration did not converge");

            const Scalar app = diagonalized(p, p);
            const Scalar aqq = diagonalized(q, q);
            const Scalar apq = diagonalized(p, q);
            const Scalar tau = (aqq - app) / (Scalar(2) * apq);
            const Scalar tangent = std::copysign(Scalar(1), tau) / (fdapde::abs(tau) + std::hypot(Scalar(1), tau));
            const Scalar cosine = Scalar(1) / std::hypot(Scalar(1), tangent);
            const Scalar sine = tangent * cosine;

            diagonalized(p, p) = app - tangent * apq;
            diagonalized(q, q) = aqq + tangent * apq;
            diagonalized(p, q) = diagonalized(q, p) = Scalar(0);
            for (int row = 0; row < n; ++row) {
                if (row == p || row == q) continue;
                const Scalar arp = diagonalized(row, p);
                const Scalar arq = diagonalized(row, q);
                diagonalized(row, p) = diagonalized(p, row) = cosine * arp - sine * arq;
                diagonalized(row, q) = diagonalized(q, row) = sine * arp + cosine * arq;
            }
            for (int row = 0; row < n; ++row) {
                const Scalar erp = eigenvectors_(row, p);
                const Scalar erq = eigenvectors_(row, q);
                eigenvectors_(row, p) = cosine * erp - sine * erq;
                eigenvectors_(row, q) = sine * erp + cosine * erq;
            }
        }

        if constexpr (Rows == Dynamic) eigenvalues_.resize(n);
        for (int i = 0; i < n; ++i) eigenvalues_[i] = diagonalized(i, i) * normalization;
        computed_ = true;
    }

    constexpr const Vector<Scalar, Rows>& eigenvalues() const& {
        fdapde_assert(computed_);
        return eigenvalues_;
    }
    constexpr void eigenvalues() const&& = delete;
    constexpr auto eigenvectors() const& {
        fdapde_assert(computed_);
        return internals::orthogonal_cast(eigenvectors_);
    }
    constexpr void eigenvectors() const&& = delete;
    constexpr bool computed() const { return computed_; }
   private:
    Matrix<Scalar, Rows, Cols> eigenvectors_;
    Vector<Scalar, Rows> eigenvalues_;
    bool computed_ = false;
};

template <typename XprType> EVD(const SymmetricMatrixExpr<XprType>&) -> EVD<XprType>;

}   // namespace fdapde::linalg

#endif   // __FDAPDE_LINALG_NATIVE_EVD_H__
