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

#ifndef __FDAPDE_LUMPING_H__
#define __FDAPDE_LUMPING_H__

#include "header_check.h"

namespace fdapde {

// returns the lumped matrix of a sparse expression. row-sum lumping operator
template <typename ExprType>
Eigen::SparseMatrix<typename ExprType::Scalar> lump(const Eigen::SparseMatrixBase<ExprType>& expr) {
    fdapde_assert(expr.rows() == expr.cols());   // stop if not square
    using Scalar_ = typename ExprType::Scalar;
    // reserve space for triplets
    std::vector<Triplet<Scalar_>> triplet_list;
    triplet_list.reserve(expr.rows());
    for (int i = 0; i < expr.rows(); ++i) { triplet_list.emplace_back(i, i, expr.row(i).sum()); }
    // matrix lumping
    Eigen::SparseMatrix<Scalar_> lumped_matrix(expr.rows(), expr.rows());
    lumped_matrix.setFromTriplets(triplet_list.begin(), triplet_list.end());
    lumped_matrix.makeCompressed();
    return lumped_matrix;
}

// returns the lumped matrix of a dense expression. row-sum lumping operator
template <typename ExprType>
Eigen::DiagonalMatrix<typename ExprType::Scalar, Dynamic, Dynamic> lump(const Eigen::MatrixBase<ExprType>& expr) {
    fdapde_assert(expr.rows() == expr.cols());   // stop if not square
    using Scalar_ = typename ExprType::Scalar;
    // matrix lumping
    Eigen::Matrix<Scalar_, Dynamic, 1> lumped_matrix = expr.array().rowwise().sum();
    return lumped_matrix.asDiagonal();
}

}   // namespace fdapde

#endif   // __FDAPDE_LUMPING_H__
