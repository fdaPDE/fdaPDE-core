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

// has_identity trait
namespace internals {

// Matrix<Scalar, N, N> => has identity
template <typename Scalar_, int N_, int NestAsRefBit_>
struct has_identity<Matrix<Scalar_, N_, N_, NestAsRefBit_>> : std::true_type {};

}

template <typename Matrix, typename Rhs> constexpr auto backward_sub(const Matrix& A, const Rhs& b) {

    fdapde_static_assert(Matrix::Rows == Matrix::Cols, BS_IS_ONLY_FOR_SQUARE_INVERTIBLE_MATRICES);
    fdapde_static_assert(std::is_same_v<typename Matrix::Scalar FDAPDE_COMMA typename Rhs::Scalar>, OPERANDS_HAVE_DIFFERENT_SCALAR_TYPES);

    // check dimensions
    using Scalar = typename Matrix::Scalar;
    constexpr int rows = Matrix::Rows;
    Vector<Scalar, rows> res;
    int i = rows - 1;
    res[i] = b[i] / A(i, i);
    i--;
    for (; i >= 0; --i) {
        Scalar tmp = 0;
        for (int j = i + 1; j < rows; ++j) tmp += A(i, j) * res[j];
        res[i] = 1. / A(i, i) * (b[i] - tmp);
    }
    return res;
}

template <typename Matrix, typename Rhs> constexpr auto forward_sub(const Matrix& A, const Rhs& b) {

    fdapde_static_assert(Matrix::Rows == Matrix::Cols, FS_IS_ONLY_FOR_SQUARE_INVERTIBLE_MATRICES);
    fdapde_static_assert(std::is_same_v<typename Matrix::Scalar FDAPDE_COMMA typename Rhs::Scalar>, OPERANDS_HAVE_DIFFERENT_SCALAR_TYPES);

    // check dimensions
    using Scalar = typename Matrix::Scalar;
    constexpr int rows = Matrix::Rows;
    Vector<Scalar, rows> res;
    int i = 0;
    res[i] = b[i] / A(i, i);
    i++;
    for (; i < rows; ++i) {
        Scalar tmp = 0;
        for (int j = 0; j < i; ++j) tmp += A(i, j) * res[j];
        res[i] = 1. / A(i, i) * (b[i] - tmp);
    }
    return res;
}

// LU factorization of SquareMatrixBase expressions with partial pivoting
template <typename MatrixType> class PartialPivLU {

    fdapde_static_assert(MatrixType::Rows == MatrixType::Cols, LU_FACTORIZATION_IS_ONLY_FOR_SQUARE_INVERTIBLE_MATRICES);

    static constexpr int N = MatrixType::Rows;
    using Scalar = typename MatrixType::Scalar;
    MatrixType m_;
    PermutationMatrix<N> P_;

public:

    // constructors
    constexpr PartialPivLU() : m_(), P_() {};
    template <typename XprType>
    constexpr explicit  PartialPivLU(const SquareMatrixBase<N, XprType>& m) : m_() { compute(m); }

    // computes the LU factorization of matrix m with partial (row) pivoting
    template <typename XprType> constexpr void compute(const SquareMatrixBase<N, XprType>& m) {
        m_ = m;
        std::array<int, N> P;
        for (int i = 0; i < N; ++i) { P[i] = i; }
        int pivot_index = 0;
        int hh, kk;
        for (int i = 0; i < N - 1; ++i) {
            // find pivotal element
            Scalar pivot = -std::numeric_limits<Scalar>::infinity();
            for (int j = i; j < N; ++j) {
                Scalar abs_ = fdapde::abs(m_(P[j], i));
                if (pivot < abs_) {
                    pivot = abs_;
                    pivot_index = j;
                }
            }
            // perform gaussian elimination step in place
            for (int j = i; j < N; ++j) {
                if (P[j] != P[pivot_index]) {   // avoid to subtract row with itself
                    Scalar l = m_(P[j], i) / m_(P[pivot_index], i);
                    m_(P[j], i) = l;
                    for (int k = i + 1; k < N; ++k) {
                        m_(P[j], k) = m_(P[j], k) - l * m_(P[pivot_index], k);
                    }
                }
            }
            // swap rows
            hh = P[i], kk = P[pivot_index];
            P[pivot_index] = hh;
            P[i] = kk;
        }
        P_ = PermutationMatrix<N>(P);
        m_ = P_ * m_;
    }

    // permutation matrix
    constexpr PermutationMatrix<N> P() const { return P_; }

    // solve linear system Ax = b using A factorization PA = LU
    template <typename RhsType> constexpr Vector<Scalar, N> solve(const RhsType& rhs) {
        fdapde_static_assert(
          std::is_same_v<Scalar FDAPDE_COMMA typename RhsType::Scalar>, INVALID_SCALAR_TYPE_FOR_RHS_OPERAND);
        fdapde_constexpr_assert(rhs.rows() == N && rhs.cols() == 1);
        Vector<Scalar, N> x;
        // evaluate U^{-1} * (L^{-1} * (P * rhs))
        x = P_ * rhs;
        x = forward_sub(m_.template triangular_view<UnitLower>(), x);
        x = backward_sub(m_.template triangular_view<Upper>(), x);
        return x;
    }
};

}

#endif   // __FDAPDE_SQUARE_MATRIX_H__