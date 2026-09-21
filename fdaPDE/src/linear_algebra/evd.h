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

#ifndef __FDAPDE_LINALG_EVD_H__
#define __FDAPDE_LINALG_EVD_H__

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <type_traits>

#include "header_check.h"

namespace fdapde {

/// @brief computes the eigendecomposition of a symmetric matrix
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

    /// @brief creates an uncomputed decomposition with no available factors
    constexpr EVD() = default;
    /// @brief computes Jacobi eigenpairs from a finite, nonempty symmetric matrix
    template <typename MatrixType> constexpr explicit EVD(const SymmetricMatrixExpr<MatrixType>& matrix) {
        compute(matrix);
    }

    // maximum-pivot Jacobi rotations with scale normalization; see Golub and
    // van Loan, Matrix Computations, Section 8.5
    /// @brief computes symmetric eigenpairs with scaled Jacobi rotations and throws if iteration does not converge
    template <typename MatrixType> constexpr void compute(const SymmetricMatrixExpr<MatrixType>& matrix) {
        computed_ = false;
        fdapde_static_assert(
          MatrixType::Rows == Dynamic || Rows == Dynamic || MatrixType::Rows == Rows, INVALID_EVD_MATRIX_STATIC_SHAPE);
        fdapde_static_assert(
          MatrixType::Cols == Dynamic || Cols == Dynamic || MatrixType::Cols == Cols, INVALID_EVD_MATRIX_STATIC_SHAPE);

        const int n = matrix.rows();
        const bool shape_valid =
          n > 0 && n == matrix.cols() && (Rows == Dynamic || n == Rows) && (Cols == Dynamic || n == Cols);
        fdapde_strong_assert(
          !(!shape_valid), std::invalid_argument, "EVD requires a nonempty square matrix matching its static shape");
        const std::int64_t dense_dimension = n;
        fdapde_strong_assert(
          !(dense_dimension * dense_dimension > std::numeric_limits<int>::max()), std::length_error,
          "EVD: dense workspace size exceeds supported range");

        Matrix<Scalar, Rows, Cols> diagonalized(matrix);
        Scalar matrix_scale = Scalar(0);
        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
                const Scalar value = diagonalized(row, col);
                fdapde_strong_assert(
                  !(!std::isfinite(value)), std::invalid_argument, "EVD requires finite matrix coefficients");
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
        const std::size_t max_iterations =
          dimension > max_size / iteration_factor / dimension ? max_size : iteration_factor * dimension * dimension;

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

    /// @brief returns unsorted eigenvalues in the same order as the eigenvector columns
    constexpr const Vector<Scalar, Rows>& eigenvalues() const& {
        fdapde_assert(computed_, std::logic_error, "eigendecomposition has not been computed");
        return eigenvalues_;
    }
    /// @brief rejects access through a temporary decomposition to prevent dangling factor references
    constexpr void eigenvalues() const&& = delete;
    /// @brief returns a borrowed orthogonal adaptor whose columns match the computed eigenvalues
    constexpr auto eigenvectors() const& {
        fdapde_assert(computed_, std::logic_error, "eigendecomposition has not been computed");
        return internals::orthogonal_cast(eigenvectors_);
    }
    /// @brief rejects access through a temporary decomposition to prevent dangling factor references
    constexpr void eigenvectors() const&& = delete;
    /// @brief reports whether a factorization has been computed
    constexpr bool computed() const { return computed_; }
   private:
    Matrix<Scalar, Rows, Cols> eigenvectors_;
    Vector<Scalar, Rows> eigenvalues_;
    bool computed_ = false;
};

template <typename XprType> EVD(const SymmetricMatrixExpr<XprType>&) -> EVD<XprType>;

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_EVD_H__
