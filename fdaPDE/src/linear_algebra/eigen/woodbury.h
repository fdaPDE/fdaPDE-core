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

#ifndef __FDAPDE_WOODBURY_H__
#define __FDAPDE_WOODBURY_H__

#include "header_check.h"

namespace fdapde {

// linear system solver based on the Sherman–Morrison–Woodbury formula
template <typename SparseSolver_> struct Woodbury {
   public:
    using Scalar = double;
    using MatrixType   = Eigen::Matrix<Scalar, Dynamic, Dynamic>;
    using SparseSolver = std::decay_t<SparseSolver_>;
    using DenseSolver  = Eigen::PartialPivLU<MatrixType>;

    Woodbury() = default;
    Woodbury(const SparseSolver_& invA, const MatrixType& U, const MatrixType& invC, const MatrixType& V) :
        invA_(std::addressof(invA)), V_(std::addressof(V)), U_(std::addressof(U)) {
        MatrixType Y = invA_->solve(U);
        MatrixType G = invC + V * Y;      // compute qxq dense matrix G = C^{-1} + V*A^{-1}*U = C^{-1} + V*y
        invG_.compute(G);
    }
    // solves linear system (A + U*C^{-1}*V)x = b
    template <typename RhsType> RhsType solve(const Eigen::MatrixBase<RhsType>& b) {
        fdapde_static_assert(internals::is_eigen_dense_xpr_v<RhsType>, THIS_METHOD_IS_ONLY_FOR_DENSE_EIGEN_RHS);
        MatrixType y = invA_->solve(b);
        MatrixType t = invG_ .solve((*V_) * y);
        MatrixType v = invA_->solve((*U_) * t);
        return y - v;
    }
   private:
    DenseSolver invG_;
    std::add_pointer_t<const SparseSolver> invA_;
    std::add_pointer_t<const MatrixType> V_, U_;
};

// solves linear system (A + U*C^{-1}*V)x = b using woodbury formula. For repated applications, use Woodbury class
template <typename SparseSolver, typename MatrixU, typename MatrixC, typename MatrixV, typename RhsType>
Eigen::Matrix<double, Dynamic, Dynamic> woodbury_system_solve(
  const SparseSolver& invA, const MatrixU& U, const MatrixC& invC, const MatrixV& V,
  const Eigen::MatrixBase<RhsType>& b) {
    fdapde_static_assert(
      internals::is_eigen_dense_xpr_v<MatrixU> && internals::is_eigen_dense_xpr_v<MatrixC> &&
      internals::is_eigen_dense_xpr_v<MatrixV> && internals::is_eigen_dense_xpr_v<RhsType>,
      THIS_METHOD_IS_ONLY_FOR_DENSE_EIGEN_MATRICES);
    using MatrixType = Eigen::Matrix<double, Dynamic, Dynamic>;
    MatrixType y = invA.solve(b);   // y = A^{-1}b
    MatrixType Y = invA.solve(U);   // Y = A^{-1}U. Heavy step of the method. solver is more efficient for small q
    MatrixType G = invC + V * Y;    // compute dense matrix G = C^{-1} + V*A^{-1}*U = C^{-1} + V*y
    Eigen::PartialPivLU<MatrixType> invG;
    invG.compute(G);   // factorize qxq dense matrix G
    MatrixType t = invG.solve(V * y);
    // v = A^{-1}*U*t = A^{-1}*U*(C^{-1} + V*A^{-1}*U)^{-1}*V*A^{-1}*b by solving linear system A*v = U*t
    MatrixType v = invA.solve(U * t);
    return y - v;   // return system solution
}

}   // namespace fdapde

#endif   // __FDAPDE_WOODBURY_H__
