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

#ifndef __VECTOR_FIELD_H__
#define __VECTOR_FIELD_H__

#include <type_traits>

#include "../utils/symbols.h"
#include "../utils/assert.h"
#include "dot_product.h"
#include "scalar_field.h"
#include "vector_expressions.h"

namespace fdapde {
namespace core {
  
// a template class for handling general vector fields from \mathbb{R}^M to \mathbb{R}^N.
template <int M, int N = M,   // input/output space dimensions (fdapde::Dynamic accepted)
	  typename F = std::function<double(static_dynamic_vector_selector_t<M>)>>
class VectorField : public VectorExpr<M, N, VectorField<M, N, F>> {
   public:
    typedef F FieldType;   // type of each field's component
    typedef typename static_dynamic_vector_selector<M>::type InnerVectorType;
    typedef typename static_dynamic_vector_selector<N>::type OuterVectorType;
    typedef VectorExpr<M, N, VectorField<M, N, F>> Base;
    using Base::inner_size;   // \mathbb{R}^M
    using Base::outer_size;   // \mathbb{R}^N
    static_assert(
      std::is_invocable<F, InnerVectorType>::value &&
      std::is_same<typename std::invoke_result<F, InnerVectorType>::type, double>::value);
    // constructors
    template <int N_ = N, typename std::enable_if<N_ != Dynamic, int>::type = 0> VectorField() { field_.resize(N); };
    template <int N_ = N, typename std::enable_if<N_ == Dynamic, int>::type = 0>
    VectorField(int m, int n) : Base(m, n) {
        field_.resize(n, ScalarField<M, FieldType>(m));
    };

    VectorField(const std::vector<FieldType>& v) {
      fdapde_assert(v.size() == outer_size());
      field_.reserve(v.size());
      for (std::size_t i = 0; i < v.size(); ++i) { field_.emplace_back(v[i]); }
    }
    // wrap a VectorExpr into a valid VectorField
    template <
      typename E, typename U = F,
      typename std::enable_if<std::is_same<U, std::function<double(SVector<N>)>>::value, int>::type = 0>
    VectorField(const VectorExpr<M, N, E>& expr) {
      if constexpr(N == Dynamic) fdapde_assert(outer_size() == expr.outer_size());
      field_.resize(outer_size());
      for (std::size_t i = 0; i < field_.size(); ++i) { field_[i] = expr[i]; }
    }
    // initializer for a zero field
    static VectorField<N, N, ZeroField<N>> Zero() {
      return VectorField<N, N, ZeroField<N>>(std::vector<ZeroField<N>>(outer_size()));
    }
    // call operator
    inline OuterVectorType operator()(const InnerVectorType& x) const;
    // subscript operator
    inline const ScalarField<M, F>& operator[](size_t i) const { return field_[i]; }   // const access to i-th element
    inline ScalarField<M, F>& operator[](size_t i) { return field_[i]; }   // non-const access to i-th element

    // inner product VectorField.dot(VectorField)
    DotProduct<M, VectorField<M, N, F>, VectorConst<M, N>> dot(const OuterVectorType& rhs) const;
    // Inner product VectorField.dot(VectorExpr)
    template <typename E> DotProduct<M, VectorField<M, N, F>, E> dot(const VectorExpr<M, N, E>& expr) const;
   protected:
    std::vector<ScalarField<M, FieldType>> field_ {};
};

// implementation details

template <int M, int N, typename F>
typename VectorField<M, N, F>::OuterVectorType VectorField<M, N, F>::operator()(const InnerVectorType& x) const {
    if constexpr(M == Dynamic) fdapde_assert(inner_size() == x.rows());
    OuterVectorType result(outer_size());
    for (int i = 0; i < outer_size(); ++i) { result[i] = field_[i](x); }
    return result;
}
// out of class definitions of VectorField arithmetic
// forward declaration
template <int N, int M, int K> class MatrixConst;
// VectorField-VectorField inner product
template <int M, int N, typename F>
DotProduct<M, VectorField<M, N, F>, VectorConst<M, N>> VectorField<M, N, F>::dot(const OuterVectorType& rhs) const {
    return DotProduct<M, VectorField<M, N, F>, VectorConst<M, N>>(
      *this, VectorConst<M, N>(rhs, inner_size(), outer_size()));
}
// VectorField-VectorExpr inner product
template <int M, int N, typename F>
template <typename E>
DotProduct<M, VectorField<M, N, F>, E> VectorField<M, N, F>::dot(const VectorExpr<M, N, E>& rhs) const {
    return DotProduct<M, VectorField<M, N, F>, E>(*this, rhs.get());
}

}   // namespace core
}   // namespace fdapde

#endif   // __VECTOR_FIELD_H__
