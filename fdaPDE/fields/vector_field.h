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
    using FieldType = F;   // type of each field's component
    using InnerVectorType = typename static_dynamic_vector_selector<M>::type;
    using OuterVectorType = typename static_dynamic_vector_selector<N>::type;
    using Base = VectorExpr<M, N, VectorField<M, N, F>>;
    using This = VectorField<M, N, FieldType>;
    using Base::inner_size;   // \mathbb{R}^M
    using Base::outer_size;   // \mathbb{R}^N
    static_assert(
      std::is_invocable<F, InnerVectorType>::value &&
      std::is_same<typename std::invoke_result<F, InnerVectorType>::type, double>::value);
    // constructors
    VectorField() requires(N != Dynamic) { field_.resize(N); }
    VectorField(int m, int n) requires(N == Dynamic) : Base(m, n) { field_.resize(n, ScalarField<M, FieldType>(m)); }
    explicit VectorField(const std::vector<FieldType>& v) {
        fdapde_assert(int(v.size()) == outer_size());
        field_.reserve(v.size());
        for (std::size_t i = 0; i < v.size(); ++i) { field_.emplace_back(v[i]); }
    }
    // wrap a VectorExpr into a valid VectorField
    template <typename E>
    VectorField(const VectorExpr<M, N, E>& expr)
        requires(std::is_same<FieldType, std::function<double(InnerVectorType)>>::value) {
        if constexpr(N == Dynamic) fdapde_assert(outer_size() == expr.outer_size());
        field_.resize(outer_size());
        for (std::size_t i = 0; i < field_.size(); ++i) { field_[i] = expr[i]; }
    }
    // initializer for a zero field
    static VectorField<N, N, ZeroField<N>> Zero() {
        return VectorField<N, N, ZeroField<N>>(std::vector<ZeroField<N>>(outer_size()));
    }
    // call operator
    inline OuterVectorType operator()(const InnerVectorType& x) const {
        if constexpr (M == Dynamic) fdapde_assert(inner_size() == x.rows());
        OuterVectorType result(outer_size());
        for (int i = 0; i < outer_size(); ++i) { result[i] = field_[i](x); }
        return result;
    }
    // subscript operator
    inline const ScalarField<M, F>& operator[](size_t i) const { return field_[i]; }   // const access to i-th element
    inline ScalarField<M, F>& operator[](size_t i) { return field_[i]; }   // non-const access to i-th element

    // inner product VectorField.dot(VectorField)
    DotProduct<M, This, Vector<M, N>> dot(const OuterVectorType& rhs) const {
        return DotProduct<M, This, Vector<M, N>>(*this, Vector<M, N>(rhs, inner_size(), outer_size()));
    }
    // Inner product VectorField.dot(VectorExpr)
    template <typename E> DotProduct<M, This, E> dot(const VectorExpr<M, N, E>& rhs) const {
        return DotProduct<M, This, E>(*this, rhs.get());
    }
   protected:
    std::vector<ScalarField<M, FieldType>> field_ {};
};

}   // namespace core
}   // namespace fdapde

#endif   // __VECTOR_FIELD_H__
