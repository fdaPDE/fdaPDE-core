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
#include "dot_product.h"
#include "scalar_field.h"
#include "vector_expressions.h"

namespace fdapde {
namespace core {

// template class representing a general vector field from an M-dimensional to an N-dimensional space.
// Support expression template arithmetic.
template <int M, int N = M, typename F = std::function<double(SVector<M>)>>
class VectorField : public VectorExpr<M, N, VectorField<M, N, F>> {
    static_assert(
      std::is_invocable<F, SVector<N>>::value &&
      std::is_same<typename std::invoke_result<F, SVector<N>>::type, double>::value);
   private:
    // each array element is a functor which computes the i-th component of the vector
    std::array<ScalarField<M, F>, N> field_;
   public:
    // constructor
    VectorField() = default;
    VectorField(const std::array<F, N>& field) {
        for (std::size_t i = 0; i < N; ++i) {   // assign a ScalarField to each component of the VectorField
            field_[i] = ScalarField<M, F>(field[i]);
        }
    }
    // wrap a VectorExpr into a valid VectorField
    template <
      typename E, typename U = F,
      typename std::enable_if<std::is_same<U, std::function<double(SVector<N>)>>::value, int>::type = 0>
    VectorField(const VectorExpr<M, N, E>& expr) {
        for (std::size_t i = 0; i < N; ++i) { field_[i] = expr[i]; }
    }
    // initializer for a zero field
    static VectorField<N, N, ZeroField<N>> Zero() {
        return VectorField<N, N, ZeroField<N>>(std::array<ZeroField<N>, N> {});
    }
    // call operator
    inline SVector<N> operator()(const SVector<M>& point) const;
    // subscript operator
    inline const ScalarField<M, F>& operator[](size_t i) const { return field_[i]; }   // const access to i-th element
    inline ScalarField<M, F>& operator[](size_t i) { return field_[i]; }   // non-const access to i-th element

    // inner product VectorField.dot(VectorField)
    DotProduct<M, VectorField<M, N, F>, VectorConst<M, N>> dot(const SVector<N>& rhs) const;
    // Inner product VectorField.dot(VectorExpr)
    template <typename E> DotProduct<M, VectorField<M, N, F>, E> dot(const VectorExpr<M, N, E>& expr) const;
};

// implementation details

template <int M, int N, typename F> SVector<N> VectorField<M, N, F>::operator()(const SVector<M>& point) const {
    SVector<N> result;
    for (size_t i = 0; i < N; ++i) {
        // call callable for each dimension of the vector field
        result[i] = field_[i](point);
    }
    return result;
}

// out of class definitions of VectorField arithmetic
// forward declaration
template <int N, int M, int K> class MatrixConst;
// VectorField-VectorField inner product
template <int M, int N, typename F>
DotProduct<M, VectorField<M, N, F>, VectorConst<M, N>> VectorField<M, N, F>::dot(const SVector<N>& rhs) const {
    return DotProduct<M, VectorField<M, N, F>, VectorConst<M, N>>(*this, VectorConst<M, N>(rhs));
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
