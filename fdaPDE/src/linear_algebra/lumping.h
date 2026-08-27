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

#include "header_check.h"

namespace fdapde {

// Row-sum lumping of a sparse matrix.
template <typename Scalar> SparseMatrix<Scalar> lump(const SparseMatrix<Scalar>& matrix) {
    if (matrix.rows() != matrix.cols()) { throw std::invalid_argument("matrix lumping requires a square matrix"); }
    return SparseMatrix<Scalar>::from_diagonal(matrix.row_sums());
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
        for (int j = 0; j < matrix.cols(); ++j) { row_sum += static_cast<Scalar>(matrix(i, j)); }
        result[i] = row_sum;
    }
    return result;
}

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_LUMPING_H__
