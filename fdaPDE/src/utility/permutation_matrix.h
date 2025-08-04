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

#ifndef __FDAPDE_PERMUTATION_MATRIX_H__
#define __FDAPDE_PERMUTATION_MATRIX_H__

#include "header_check.h"

namespace fdapde {

// forward declaration to break circular dependency
template <int N_> struct PermutationMatrix;

// has_identity trait
namespace internals {

// PermutationMatrix<N> => has identity
template <int N_>
struct has_identity<PermutationMatrix<N_>> : std::true_type {};

}

// permutation matrix
template <int N_> struct PermutationMatrix : public SquareMatrixBase<N_, PermutationMatrix<N_>> {
    using Base = SquareMatrixBase<N_, PermutationMatrix<N_>>;
    using Scalar = int;
    using XprType = PermutationMatrix<N_>;
    static constexpr int N = N_;
    static constexpr int Rows = N_;
    static constexpr int Cols = N_;
    static constexpr int NestAsRefBit = 0;
    static constexpr int ReadOnly = 1;
    static constexpr int XprBits = int(matrix_flags::square);

    // constructors
    constexpr PermutationMatrix() = default;
    constexpr explicit PermutationMatrix(const std::array<int, N>& permutation) : permutation_(permutation) { }

    // constexpr int rows() const { return Rows; } // This shouldn't be necessary
    // constexpr int cols() const { return Cols; } // This shouldn't be necessary

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
    operator*(const MatrixBase<RhsRows, RhsCols, RhsType>& lhs, const PermutationMatrix<N_>& rhs) {
        fdapde_static_assert(Cols == RhsRows, INVALID_OPERANDS_DIMENSION_FOR_MATRIX_MATRIX_PRODUCT);
        using Scalar = typename RhsType::Scalar;
        Matrix<Scalar, Rows, RhsCols> permuted;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < RhsCols; ++j) { permuted(j, i) = lhs.derived().operator()(j, rhs.permutation()[i]); }
        }
        return permuted;
    }

    // const access
    constexpr int operator()(int i, int j) const {
        return permutation_[i] == j ? 1 : 0;
    }

    // data
    constexpr const std::array<int, N>& permutation() const { return permutation_; }

private:
    std::array<int, N> permutation_;
};

}

#endif   // __FDAPDE_PERMUTATION_MATRIX_H__