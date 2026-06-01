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

#ifndef __FDAPDE_LINALG_PRECONDITIONERS_H__
#define __FDAPDE_LINALG_PRECONDITIONERS_H__

#include "header_check.h"

namespace fdapde {

template <typename XprType_> class IdentityPreconditioner {
    using XprType = std::decay_t<XprType_>;
    fdapde_static_assert(
      XprType::Rows == Dynamic || XprType::Cols == Dynamic || XprType::Rows == XprType::Cols,
      THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    using Scalar = typename XprType::Scalar;

    IdentityPreconditioner() = default;
    template <typename XprType> explicit IdentityPreconditioner(const MatrixExpr<XprType>& m) {
        fdapde_assert(m.rows() == m.cols());
    }
    template <typename XprType> constexpr void compute(const MatrixExpr<XprType>& m) { return; }
    template <typename RhsXprType> constexpr auto solve(const MatrixExpr<RhsXprType>& b) const { return b; }
};

template <typename XprType_> class DiagonalPreconditioner {
    using XprType = std::decay_t<XprType_>;
    fdapde_static_assert(
      XprType::Rows == Dynamic || XprType::Cols == Dynamic || XprType::Rows == XprType::Cols,
      THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    using Scalar = typename XprType::Scalar;

    IdentityPreconditioner() = default;
    template <typename XprType> explicit DiagonalPreconditioner(const MatrixExpr<XprType>& m) {
        fdapde_assert(m.rows() == m.cols());
	compute(m);
    }
    template <typename XprType> constexpr void compute(const MatrixExpr<XprType>& m) {
        if constexpr (Rows == Dynamic || Cols = Dynamic) { inverse_.resize(m.rows()); }
        for (int i = 0; i < inverse_.size(); ++i) { inverse_[i] = 1.0 / m(i, i); }
        return;
    }
    template <typename RhsXprType> constexpr auto solve(const MatrixExpr<RhsXprType>& b) const {
        Vector<Scalar, Rows> v = b;
        for (int i = 0; i < v.size(); ++i) { v[i] *= inverse_[i]; }
        return v;
    }
   private:
    Vector<Scalar, Rows> inverse_;
};
  
}   // namespace fdapde

#endif   // __FDAPDE_LINALG_PRECONDITIONERS_H__
