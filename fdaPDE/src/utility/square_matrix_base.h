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

#ifndef __FDAPDE_SQUARE_MATRIX_BASE_H__
#define __FDAPDE_SQUARE_MATRIX_BASE_H__

#include "header_check.h"
#include "matrix_base.h"

namespace fdapde {

template <int Size, typename Derived>
struct SquareMatrixBase : public MatrixBase<Size, Size, Derived> {
    using Base = MatrixBase<Size, Size, Derived>;
    using Base::derived;

    // trace of matrix
    constexpr auto trace() const {
        typename Derived::Scalar trace_ = 0;
        for (int i = 0; i < Size; ++i) trace_ += derived().operator()(i, i);
        return trace_;
    }

    // TODO: Eigenvalue Decomposition

};

}

#endif //__FDAPDE_SQUARE_MATRIX_BASE_H__