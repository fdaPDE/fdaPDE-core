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

#ifndef __FDAPDE_SQUARE_MATRIX_H__
#define __FDAPDE_SQUARE_MATRIX_H__

#include "header_check.h"

namespace fdapde {

// forward/backward substitution for unit-triangular L and general U
template <typename Matrix, typename Rhs>
constexpr auto forward_sub(const Matrix& A, const Rhs& b) {
    fdapde_static_assert(Matrix::Rows == Matrix::Cols, FS_IS_ONLY_FOR_SQUARE_INVERTIBLE_MATRICES);
    fdapde_static_assert(std::is_same_v<typename Matrix::Scalar FDAPDE_COMMA typename Rhs::Scalar>, OPERANDS_HAVE_DIFFERENT_SCALAR_TYPES);

    using Scalar = typename Matrix::Scalar;
    constexpr int N = Matrix::Rows;
    Vector<Scalar, N> x;
    x[0] = b[0] / A(0,0);
    for (int i = 1; i < N; ++i) {
        Scalar sum = 0;
        for (int j = 0; j < i; ++j) sum += A(i,j) * x[j];
        x[i] = (b[i] - sum) / A(i,i);
    }
    return x;
}

template <typename Matrix, typename Rhs>
constexpr auto backward_sub(const Matrix& A, const Rhs& b) {
    fdapde_static_assert(Matrix::Rows == Matrix::Cols, BS_IS_ONLY_FOR_SQUARE_INVERTIBLE_MATRICES);
    fdapde_static_assert(std::is_same_v<typename Matrix::Scalar FDAPDE_COMMA typename Rhs::Scalar>, OPERANDS_HAVE_DIFFERENT_SCALAR_TYPES);

    using Scalar = typename Matrix::Scalar;
    constexpr int N = Matrix::Rows;
    Vector<Scalar, N> x;
    x[N-1] = b[N-1] / A(N-1,N-1);
    for (int i = N-2; i >= 0; --i) {
        Scalar sum = 0;
        for (int j = i+1; j < N; ++j) sum += A(i,j) * x[j];
        x[i] = (b[i] - sum) / A(i,i);
    }
    return x;
}

// LU with partial (row) pivoting
template <typename MatrixType>
class PartialPivLU {
    fdapde_static_assert(MatrixType::Rows == MatrixType::Cols, LU_FACTORIZATION_IS_ONLY_FOR_SQUARE_INVERTIBLE_MATRICES);

    static constexpr int N = MatrixType::Rows;
    using Scalar = typename MatrixType::Scalar;

    Matrix<Scalar, N, N> lu_;  // will hold both L (unit lower) and U (upper)
    PermutationMatrix<N> P_; // row‐permutation matrix

public:
    constexpr PartialPivLU() : lu_(), P_() { }

    template <typename Xpr>
    constexpr explicit PartialPivLU(const SquareMatrixBase<N, Xpr>& m) { // : lu_(m) { // TODO:: with SquareMatrixBase -> segfault
        lu_ = m;
        compute(m);
    }

    // Factorize: overwrite `lu_` in place, build P_
    template <typename Xpr>
    constexpr void compute(const SquareMatrixBase<N, Xpr>& m) {
        // copy input
        lu_ = m;

        // track row swaps in a simple array first
        std::array<int,N> perm;
        for (int i = 0; i < N; ++i) perm[i] = i;

        // Doolittle with partial pivoting
        for (int i = 0; i < N-1; ++i) {
            // 1) find pivot row among i..N-1
            int pivot_index = i;
            Scalar maxval = Scalar(0);
            for (int r = i; r < N; ++r) {
                Scalar av = fdapde::abs(lu_(r,i));
                if (av > maxval) {
                    maxval = av;
                    pivot_index = r;
                }
            }

            // 2) swap rows i <-> pivot_index if needed
            if (pivot_index != i) {
                std::swap(perm[i], perm[pivot_index]);
                for (int c = 0; c < N; ++c)
                    std::swap(lu_(i,c), lu_(pivot_index,c));
            }

            // 3) eliminate below pivot
            for (int r = i+1; r < N; ++r) {
                Scalar alpha = lu_(r,i) / lu_(i,i);
                lu_(r,i) = alpha;  // store L
                for (int c = i+1; c < N; ++c) {
                    lu_(r,c) -= alpha * lu_(i,c);
                }
            }
        }

        // build the final PermutationMatrix
        P_ = PermutationMatrix<N>(perm);
    }

    // Access the permutation
    constexpr PermutationMatrix<N> P() const { return P_; }

    // Solve Ax = b via PA = LU
    template <typename Rhs>
    constexpr Vector<Scalar,N> solve(const Rhs& b) const {
        fdapde_static_assert(std::is_same_v<Scalar FDAPDE_COMMA typename Rhs::Scalar>,
                             INVALID_SCALAR_TYPE_FOR_RHS_OPERAND);
        fdapde_constexpr_assert(b.rows() == N && b.cols() == 1);

        // apply P to RHS, then forward/backward substitute
        auto y = P_ * b;
        auto z = forward_sub(lu_.template triangular_view<UnitLower>(), y);
        return backward_sub(lu_.template triangular_view<Upper>(), z);

    }
};

}

#endif   // __FDAPDE_SQUARE_MATRIX_H__