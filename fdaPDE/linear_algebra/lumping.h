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

#ifndef __LUMPING_H__
#define __LUMPING_H__

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "../utils/assert.h"
#include "../utils/symbols.h"

namespace fdapde {
namespace core {

// returns the lumped matrix of a sparse expression. Implements a row-sum lumping operator
template <typename ExprType> SpMatrix<typename ExprType::Scalar> lump(const Eigen::SparseMatrixBase<ExprType>& expr) {
    fdapde_assert(expr.rows() == expr.cols());   // stop if not square
    using Scalar_ = typename ExprType::Scalar;
    // reserve space for triplets
    std::vector<fdapde::Triplet<Scalar_>> triplet_list;
    triplet_list.reserve(expr.rows());
    for (std::size_t i = 0; i < expr.rows(); ++i) { triplet_list.emplace_back(i, i, expr.row(i).sum()); }
    // matrix lumping
    SpMatrix<Scalar_> lumped_matrix(expr.rows(), expr.rows());
    lumped_matrix.setFromTriplets(triplet_list.begin(), triplet_list.end());
    lumped_matrix.makeCompressed();
    return lumped_matrix;
}

// returns the lumped matrix of a dense expression. Implements a row-sum lumping operator
template <typename ExprType> DiagMatrix<typename ExprType::Scalar> lump(const Eigen::MatrixBase<ExprType>& expr) {
    fdapde_assert(expr.rows() == expr.cols());   // stop if not square
    using Scalar_ = typename ExprType::Scalar;
    // matrix lumping
    DVector<Scalar_> lumped_matrix = expr.array().rowwise().sum();
    return lumped_matrix.asDiagonal();
}

}   // namespace core
}   // namespace fdapde

#endif   // __LUMPING_H__
