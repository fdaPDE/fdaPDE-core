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

#ifndef __FDAPDE_MATRIX_ALGORITHMS_H__
#define __FDAPDE_MATRIX_ALGORITHMS_H__

#include "header_check.h"
#include "square_matrix_base.h"

namespace fdapde {

// forward substitution for lower-triangular matrix (view)
template <typename Matrix, typename Rhs>
constexpr auto forward_sub(const Matrix& L, const Rhs& b)
requires (internals::is_lower_triangular_v<Matrix>) {
    fdapde_static_assert(Matrix::Rows == Matrix::Cols, FS_IS_ONLY_FOR_SQUARE_INVERTIBLE_MATRICES);
    fdapde_static_assert(std::is_same_v<typename Matrix::Scalar FDAPDE_COMMA typename Rhs::Scalar>, OPERANDS_HAVE_DIFFERENT_SCALAR_TYPES);

    using Scalar = typename Matrix::Scalar;
    constexpr int N = Matrix::Rows;
    Vector<Scalar, N> x;
    x[0] = b[0] / L(0,0);
    for (int i = 1; i < N; ++i) {
        Scalar sum = 0;
        for (int j = 0; j < i; ++j) sum += L(i,j) * x[j];
        x[i] = (b[i] - sum) / L(i,i);
    }
    return x;
}

// backward substitution for upper-triangular matrix (view)
template <typename Matrix, typename Rhs>
constexpr auto backward_sub(const Matrix& U, const Rhs& b)
requires (internals::is_upper_triangular_v<Matrix>) {
    fdapde_static_assert(Matrix::Rows == Matrix::Cols, BS_IS_ONLY_FOR_SQUARE_INVERTIBLE_MATRICES);
    fdapde_static_assert(std::is_same_v<typename Matrix::Scalar FDAPDE_COMMA typename Rhs::Scalar>, OPERANDS_HAVE_DIFFERENT_SCALAR_TYPES);

    using Scalar = typename Matrix::Scalar;
    constexpr int N = Matrix::Rows;
    Vector<Scalar, N> x;
    x[N-1] = b[N-1] / U(N-1,N-1);
    for (int i = N-2; i >= 0; --i) {
        Scalar sum = 0;
        for (int j = i+1; j < N; ++j) sum += U(i,j) * x[j];
        x[i] = (b[i] - sum) / U(i,i);
    }
    return x;
}

}

#endif   // __FDAPDE_MATRIX_ALGORITHMS_H__