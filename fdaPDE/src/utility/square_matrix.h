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
template <int Size_> struct PermutationMatrix : public SquareMatrixBase<Size_, PermutationMatrix<Size_>> {
    using Base = SquareMatrixBase<Size_, PermutationMatrix<Size_>>;
    using Scalar = int;
    using XprType = PermutationMatrix<Size_>;
    static constexpr int Size = Size_;
    static constexpr int Rows = Size_;
    static constexpr int Cols = Size_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;
    static constexpr int XprBits = int(matrix_flags::square);

    constexpr PermutationMatrix() = default;
    constexpr explicit PermutationMatrix(const std::array<int, Size>& permutation) : permutation_(permutation) { }
    constexpr int rows() const { return Rows; }
    constexpr int cols() const { return Cols; }
    // left multiplication by permutation matrix
    template <int RhsRows, int RhsCols, typename RhsType>
    constexpr Matrix<typename RhsType::Scalar, Rows, RhsCols>
    operator*(const MatrixBase<RhsRows, RhsCols, RhsType>& rhs) const {
        fdapde_static_assert(Cols == RhsRows, INVALID_OPERAND_DIMENSIONS_FOR_MATRIX_MATRIX_PRODUCT);
        using Scalar = typename RhsType::Scalar;
        Matrix<Scalar, Rows, RhsCols> permuted;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < RhsCols; ++j) { permuted(i, j) = rhs.derived().operator()(permutation_[i], j); }
        }
        return permuted;
    }
    // right multiplication by permutation matrix
    template <int RhsRows, int RhsCols, typename RhsType>
    constexpr friend Matrix<typename RhsType::Scalar, Rows, RhsCols>
    operator*(const MatrixBase<RhsRows, RhsCols, RhsType>& lhs, const PermutationMatrix<Size_>& rhs) {
        fdapde_static_assert(Cols == RhsRows, INVALID_OPERANDS_DIMENSION_FOR_MATRIX_MATRIX_PRODUCT);
        using Scalar = typename RhsType::Scalar;
        Matrix<Scalar, Rows, RhsCols> permuted;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < RhsCols; ++j) { permuted(j, i) = lhs.derived().operator()(j, rhs.permutation()[i]); }
        }
        return permuted;
    }
    constexpr int operator()(int i, int j) const {
        return permutation_[i] == j ? 1 : 0;
    }
    constexpr const std::array<int, Size_>& permutation() const { return permutation_; }
private:
    std::array<int, Size> permutation_;
};

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

// LU factorization of matrix with partial pivoting
template <typename MatrixType> class PartialPivLU {
    fdapde_static_assert(MatrixType::Rows == MatrixType::Cols, LU_FACTORIZATION_IS_ONLY_FOR_SQUARE_INVERTIBLE_MATRICES);
    static constexpr int Size = MatrixType::Rows;
    using Scalar = typename MatrixType::Scalar;
    MatrixType m_;
    PermutationMatrix<Size> P_;
public:
    constexpr PartialPivLU() : m_(), P_() {};
    template <typename XprType> constexpr PartialPivLU(const SquareMatrixBase<Size, XprType>& m) : m_() { compute(m); std::cout << MatrixType::XprBits << std::endl; }

    // computes the LU factorization of matrix m with partial (row) pivoting
    template <typename XprType> constexpr void compute(const SquareMatrixBase<Size, XprType>& m) {
        m_ = m;
        std::array<int, Size> P;
        for (int i = 0; i < Size; ++i) { P[i] = i; }
        int pivot_index = 0;
        int h, k;
        for (int i = 0; i < Size - 1; ++i) {
            // find pivotal element
            Scalar pivot = -std::numeric_limits<Scalar>::infinity();
            for (int j = i; j < Size; ++j) {
                Scalar abs_ = fdapde::abs(m_(P[j], i));
                if (pivot < abs_) {
                    pivot = abs_;
                    pivot_index = j;
                }
            }
            // perform gaussian elimination step in place
            for (int j = i; j < Size; ++j) {
                if (P[j] != P[pivot_index]) {   // avoid to subtract row with itself
                    Scalar l = m_(P[j], i) / m_(P[pivot_index], i);
                    m_(P[j], i) = l;
                    for (int k = i + 1; k < Size; ++k) { m_(P[j], k) = m_(P[j], k) - l * m_(P[pivot_index], k); }
                }
            }
            // swap rows
            h = P[i], k = P[pivot_index];
            P[pivot_index] = h;
            P[i] = k;
        }
        P_ = PermutationMatrix<Size>(P);
        m_ = P_ * m_;
    }
    constexpr PermutationMatrix<Size> P() const { return P_; }
    // solve linear system Ax = b using A factorization PA = LU
    template <typename RhsType> constexpr Vector<Scalar, Size> solve(const RhsType& rhs) {
        fdapde_static_assert(
          std::is_same_v<Scalar FDAPDE_COMMA typename RhsType::Scalar>, INVALID_SCALAR_TYPE_FOR_RHS_OPERAND);
        fdapde_constexpr_assert(rhs.rows() == Size && rhs.cols() == 1);
        Vector<Scalar, Size> x;
        // evaluate U^{-1} * (L^{-1} * (P * rhs))
        x = P_ * rhs;
        x = forward_sub(m_.template triangular_view<UnitLower>(), x);
        x = backward_sub(m_.template triangular_view<Upper>(), x);
        return x;
    }
};

}

#endif   // _FDAPDE_SQUARE_MATRIX_H__