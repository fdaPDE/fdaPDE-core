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

#ifndef __MATRIX_FIELD_H__
#define __MATRIX_FIELD_H__

#include <array>
#include <type_traits>

#include "../utils/symbols.h"
#include "matrix_expressions.h"
#include "scalar_field.h"
#include "vector_field.h"

namespace fdapde {
namespace core {

// a template class for handling matrix fields. A matrix field is a function mapping N-dimensional points to
// M x K dimensional matrices. Supports expression template arithmetic
template <int N, int M = N, int K = N, typename F = std::function<double(SVector<N>)>>
class MatrixField : public MatrixExpr<N, M, K, MatrixField<N, M, K, F>> {
    static_assert(std::is_invocable<F, SVector<N>>::value &&
                  std::is_same<typename std::invoke_result<F, SVector<N>>::type, double>::value);
   private:
    // an M dimensional array of K dimensional array of N dimensional ScalarField
    std::array<std::array<ScalarField<N, F>, K>, M> field_;
   public:
    // constructors
    MatrixField() = default;
    MatrixField(std::array<std::array<F, K>, M> field) {   // construct from array of callables
        for (std::size_t i = 0; i < M; ++i) {
            for (std::size_t j = 0; j < K; ++j) field_[i][j] = ScalarField<N, F>(field[i][j]);
        }
    }
    // call operator (evaluate field at point)
    SMatrix<M, K> operator()(const SVector<N>& point) const;
    // const and non-const access operations
    const ScalarField<N, F>& operator()(std::size_t i, std::size_t j) const { return field_[i][j]; };
    ScalarField<N, F>& operator()(std::size_t i, std::size_t j) { return field_[i][j]; };
    const ScalarField<N, F>& coeff(std::size_t i, std::size_t j) const { return operator()(i, j); };
};

// implementation details

template <int N, int M, int K, typename F>
SMatrix<M, K> MatrixField<N, M, K, F>::operator()(const SVector<N>& point) const {
    SMatrix<M, K> result;
    for (std::size_t i = 0; i < M; ++i) {
        for (std::size_t j = 0; j < K; ++j) result(i, j) = field_[i][j](point);
    }
    return result;
}

// out of class definitions of MatrixField arithmetic
// rhs multiplication by SVector
template <int N, int M, int K, typename F>
MatrixVectorProduct<N, M, K, MatrixField<N, M, K, F>, VectorConst<N, K>> operator*(const MatrixField<N, M, K, F>& op1,
                                                                                   const SVector<K>& op2) {
    return MatrixVectorProduct<N, M, K, MatrixField<N, M, K, F>, VectorConst<N, K>>(op1, VectorConst<N, K>(op2));
}
// rhs multiplication by VectorField
template <int N, int M, int K, typename F1, typename F2>
MatrixVectorProduct<N, M, K, MatrixField<N, M, K, F1>, VectorField<N, K, F2>>
operator*(const MatrixField<N, M, K, F1>& op1, const VectorField<N, K, F2>& op2) {
    return MatrixVectorProduct<N, M, K, MatrixField<N, M, K, F1>, VectorField<N, K, F2>>(op1, op2);
}

}   // namespace core
}   // namespace fdapde

#endif   // __MATRIX_FIELD_H__
