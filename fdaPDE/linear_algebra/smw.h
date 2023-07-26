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

#ifndef __SMW_H__
#define __SMW_H__

#include <Eigen/LU>
#include <Eigen/SparseLU>

#include "../utils/symbols.h"

namespace fdapde {
namespace core {

// A general implementation of a system solver using the Sherman–Morrison–Woodbury formula for matrix inversion

// Given  a linear system Mx = b, under the assumption that we are able to decompose matrix M as (A + UCV), according
// to the Shermann-Morrison-Woodbury formula we have
//
//        M^{-1} = (A + UCV)^{-1} = A^{-1} - A^{-1}*U*(C^{-1} + V*A^{-1}*U)^{-1}*V*A^{-1}
//
// If A is sparse and C a small dense matrix, computing M^{-1} using the above decomposition is much more efficient
// than computing M^{-1} directly

template <typename SparseSolver = fdapde::SparseLU<SpMatrix<double>>,
	  typename DenseSolver  = Eigen::PartialPivLU<DMatrix<double>>>
struct SMW {
    // constructor
    SMW() = default;

    // solves linear system (A + U*C^{-1}*V)x = b, assume to supply the already computed inversion of the dense matrix C
    DMatrix<double> solve(
      const SparseSolver& invA, const DMatrix<double>& U, const DMatrix<double>& invC, const DMatrix<double>& V,
      const DMatrix<double>& b) {
        DMatrix<double> y = invA.solve(b);   // y = A^{-1}b
        // Y = A^{-1}U. Heavy step of the method. SMW is more and more efficient as q gets smaller and smaller
        DMatrix<double> Y = invA.solve(U);
        // compute dense matrix G = C^{-1} + V*A^{-1}*U = C^{-1} + V*y
        DMatrix<double> G = invC + V * Y;
        DenseSolver invG;
        invG.compute(G);   // factorize qxq dense matrix G
        DMatrix<double> t = invG.solve(V * y);
        // v = A^{-1}*U*t = A^{-1}*U*(C^{-1} + V*A^{-1}*U)^{-1}*V*A^{-1}*b by solving linear system A*v = U*t
        DMatrix<double> v = invA.solve(U * t);
        return y - v;   // return system solution
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // _SMW_H__
