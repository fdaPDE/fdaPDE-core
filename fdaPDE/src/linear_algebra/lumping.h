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

#ifndef __FDAPDE_LINALG_LUMPING_H__
#define __FDAPDE_LINALG_LUMPING_H__

#include <cmath>
#include <complex>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "header_check.h"

namespace fdapde {
namespace internals {

template <typename Scalar> inline constexpr bool is_std_complex_v = false;
template <typename Scalar> inline constexpr bool is_std_complex_v<std::complex<Scalar>> = true;

template <typename Scalar> bool finite_lumping_scalar(const Scalar& value) {
    if constexpr (std::is_floating_point_v<Scalar>) {
        return std::isfinite(value);
    } else if constexpr (is_std_complex_v<Scalar>) {
        return std::isfinite(value.real()) && std::isfinite(value.imag());
    } else {
        return true;
    }
}

template <typename Scalar> Scalar checked_lumping_add(const Scalar& lhs, const Scalar& rhs) {
    if constexpr (std::is_floating_point_v<Scalar> || is_std_complex_v<Scalar>) {
        if (!finite_lumping_scalar(lhs) || !finite_lumping_scalar(rhs)) {
            throw std::invalid_argument("matrix lumping requires finite coefficients");
        }
        const Scalar result = lhs + rhs;
        if (!finite_lumping_scalar(result)) { throw std::overflow_error("matrix lumping row sum is not finite"); }
        return result;
    } else if constexpr (std::is_integral_v<Scalar>) {
        if constexpr (std::is_signed_v<Scalar>) {
            if (
              (rhs > 0 && lhs > std::numeric_limits<Scalar>::max() - rhs) ||
              (rhs < 0 && lhs < std::numeric_limits<Scalar>::lowest() - rhs)) {
                throw std::overflow_error("matrix lumping row sum exceeds the scalar range");
            }
        } else if (lhs > std::numeric_limits<Scalar>::max() - rhs) {
            throw std::overflow_error("matrix lumping row sum exceeds the scalar range");
        }
        return lhs + rhs;
    } else {
        Scalar result = lhs;
        result += rhs;
        return result;
    }
}

}   // namespace internals

// Row-sum lumping of a sparse matrix.
template <typename Scalar> SparseMatrix<Scalar> lump(const SparseMatrix<Scalar>& matrix) {
    if (matrix.rows() != matrix.cols()) { throw std::invalid_argument("matrix lumping requires a square matrix"); }
    SparseMatrix<Scalar> result(matrix.rows(), matrix.cols());
    result.column_indices_.reserve(static_cast<std::size_t>(matrix.rows()));
    result.values_.reserve(static_cast<std::size_t>(matrix.rows()));
    for (int row = 0; row < matrix.rows(); ++row) {
        Scalar row_sum {};
        for (const auto entry : matrix.row(row)) { row_sum = internals::checked_lumping_add(row_sum, entry.value()); }
        result.column_indices_.push_back(row);
        result.values_.push_back(std::move(row_sum));
        result.row_offsets_[row + 1] = static_cast<int>(result.values_.size());
    }
    return result;
}

// Row-sum lumping of a dense matrix expression.
template <internals::matrix_expression XprType> auto lump(const XprType& matrix) {
    using Xpr = std::remove_cvref_t<XprType>;
    using Scalar = std::remove_cv_t<typename Xpr::Scalar>;
    fdapde_static_assert(
      Xpr::Rows == Dynamic || Xpr::Cols == Dynamic || Xpr::Rows == Xpr::Cols, THIS_METHODS_IS_FOR_SQUARE_MATRICES_ONLY);
    if (matrix.rows() != matrix.cols()) { throw std::invalid_argument("matrix lumping requires a square matrix"); }
    DiagonalMatrix<Scalar, Dynamic> result(matrix.rows());
    for (int i = 0; i < matrix.rows(); ++i) {
        Scalar row_sum {};
        for (int j = 0; j < matrix.cols(); ++j) {
            row_sum = internals::checked_lumping_add(row_sum, static_cast<Scalar>(matrix(i, j)));
        }
        result[i] = row_sum;
    }
    return result;
}

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_LUMPING_H__
