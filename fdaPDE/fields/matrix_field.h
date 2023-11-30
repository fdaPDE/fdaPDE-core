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

// a template class for handling general matrix fields from \mathbb{R}^M to \mathbb{R}^{N \times K}.
template <int M, int N = M, int K = M, // input/output space dimensions (fdapde::Dynamic accepted)
	  typename F = std::function<double(static_dynamic_vector_selector_t<M>)>>
class MatrixField : public MatrixExpr<M, N, K, MatrixField<M, N, K, F>> {
   public:
    typedef F FieldType;   // type of each field's component
    typedef typename static_dynamic_vector_selector<M>::type InnerVectorType;
    typedef typename static_dynamic_matrix_selector<N, K>::type OuterMatrixType;
    typedef MatrixExpr<M, N, K, MatrixField<M, N, K, F>> Base;
    using Base::inner_size;   // \mathbb{R}^M
    using Base::outer_size;   // \mathbb{R}^{N \times K}
    using Base::outer_rows;   // number of rows in result
    using Base::outer_cols;   // number of cols in result
    static_assert(
      std::is_invocable<F, InnerVectorType>::value &&
      std::is_same<typename std::invoke_result<F, InnerVectorType>::type, double>::value);
    // static constructors
    template <int N_ = N, int K_ = K, typename std::enable_if<N_ != Dynamic && K_ != Dynamic, int>::type = 0>
    MatrixField() { field_.resize(N * K); };
    template <int N_ = N, int K_ = K, typename std::enable_if<N_ != Dynamic && K_ != Dynamic, int>::type = 0>
    MatrixField(const std::vector<FieldType>& v) {
      fdapde_assert(v.size() == outer_size());
      field_.reserve(v.size());
      for (std::size_t i = 0; i < v.size(); ++i) { field_.emplace_back(v[i]); }
    }
    // fully dynamic constructor
    template <int N_ = N, int K_ = K, typename std::enable_if<N_ == Dynamic || K_ == Dynamic, int>::type = 0>
    MatrixField(int m, int n, int k) : Base(m, n, k) {
        field_.resize(n * k, ScalarField<M, FieldType>(m));
    };
    // call operator (evaluate field at point)
    OuterMatrixType operator()(const InnerVectorType& point) const {
      OuterMatrixType result;
      if constexpr(N == Dynamic || K == Dynamic) result.resize(outer_rows(), outer_cols());
      for (std::size_t i = 0; i < outer_rows(); ++i) {
          for (std::size_t j = 0; j < outer_cols(); ++j) result(i, j) = field_[j + i * outer_cols()](point);
      }
      return result;
    }
    // const and non-const access operations
    const ScalarField<M, FieldType>& operator()(std::size_t i, std::size_t j) const {
      return field_[j + i * outer_cols()];
    };
    ScalarField<M, FieldType>& operator()(std::size_t i, std::size_t j) { return field_[j + i * outer_cols()]; };
    const ScalarField<M, FieldType>& coeff(std::size_t i, std::size_t j) const { return operator()(i, j); };
   protected:
    std::vector<ScalarField<M, FieldType>> field_ {}; // stored in row-major order
};

// out of class definitions of MatrixField arithmetic
// rhs multiplication by SVector
template <int M, int N, int K, typename F>
MatrixVectorProduct<M, N, K, MatrixField<M, N, K, F>, VectorConst<M, K>>
operator*(const MatrixField<M, N, K, F>& op1, const static_dynamic_vector_selector_t<K>& op2) {
    return MatrixVectorProduct<M, N, K, MatrixField<M, N, K, F>, VectorConst<M, K>>(
      op1, VectorConst<M, K>(op2, op1.inner_size(), op1.outer_cols()));
}
// rhs multiplication by VectorField
template <int M, int N, int K, typename F1, typename F2>
MatrixVectorProduct<M, N, K, MatrixField<M, N, K, F1>, VectorField<M, K, F2>>
operator*(const MatrixField<M, N, K, F1>& op1, const VectorField<M, K, F2>& op2) {
    return MatrixVectorProduct<M, N, K, MatrixField<M, N, K, F1>, VectorField<M, K, F2>>(op1, op2);
}

}   // namespace core
}   // namespace fdapde

#endif   // __MATRIX_FIELD_H__
