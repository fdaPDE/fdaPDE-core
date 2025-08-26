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

#ifndef __FDAPDE_ORTHOGONAL_MATRIX_H__
#define __FDAPDE_ORTHOGONAL_MATRIX_H__

#include "header_check.h"

namespace fdapde {

template <typename Scalar_, int Size_, int StorageOrder, typename MatrixType>
struct OrthogonalMatrixBase : public MatrixBase<Scalar_, Size_, Size_, StorageOrder, MatrixType> { };

// A matrix with enforced orthogonality check (i.e., M == M^\top)
template <typename Scalar_, int Size_, int StorageOrder_ = RowMajor>
struct OrthogonalMatrix :
    public OrthogonalMatrixBase<Scalar_, Size_, StorageOrder_, OrthogonalMatrix<Scalar_, Size_, StorageOrder_>> {
    using Base = OrthogonalMatrixBase<Scalar_, Size_, StorageOrder_, OrthogonalMatrix<Scalar_, Size_, StorageOrder_>>;
};

}   // namespace fdapde

#endif   // _FDAPDE_ORTHOGONAL_MATRIX_H__
