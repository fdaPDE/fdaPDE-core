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

#ifndef __DIFFERENTIABLE_FIELD_H__
#define __DIFFERENTIABLE_FIELD_H__

#include <functional>

#include "../utils/symbols.h"
#include "scalar_field.h"

namespace fdapde {
namespace core {

// forward declarations
template <int M, int N, typename F> class VectorField;
template <int M, int N, int K, typename F> class MatrixField;

template <int N,   // input space dimension
	  typename F = std::function<double(static_dynamic_vector_selector_t<N>)>,
	  typename G = std::function<double(static_dynamic_vector_selector_t<N>)>>
class DifferentiableScalarField : public ScalarField<N, F> {
   protected:
    typedef ScalarField<N, F> Base;
    typedef F FieldType;
    typedef VectorField<N, N, G> GradientType;
    typedef typename static_dynamic_vector_selector<N>::type VectorType;
    static_assert(std::is_invocable<G, VectorType>::value);

    GradientType df_ {};   // gradient vector of scalar field f
   public:
    DifferentiableScalarField(const FieldType& f, const GradientType& df) : Base(f), df_(df) {};
    template <typename Args>
    DifferentiableScalarField(const FieldType& f, const std::vector<Args>& df) : Base(f), df_(GradientType(df)) {};
    GradientType derive() { return df_; }   // return analytical gradient
};

template <int N,   // input space dimension
	  typename F = std::function<double(static_dynamic_vector_selector_t<N>)>,
	  typename G = std::function<double(static_dynamic_vector_selector_t<N>)>,
	  typename H = std::function<double(static_dynamic_vector_selector_t<N>)>>
class TwiceDifferentiableScalarField : public DifferentiableScalarField<N, F, G> {
   protected:
    typedef DifferentiableScalarField<N, F, G> Base;
    typedef F FieldType;
    typedef VectorField<N, N, G> GradientType;
    typedef MatrixField<N, N, N, H> HessianType;
    typedef typename static_dynamic_vector_selector<N>::type VectorType;
    typedef typename static_dynamic_matrix_selector<N, N>::type MatrixType;
    static_assert(std::is_invocable<H, VectorType>::value);

    HessianType ddf_ {};   // hessian matrix of scalar field f
   public:
    TwiceDifferentiableScalarField(const FieldType& f, const GradientType& df, const HessianType& ddf) :
        Base(f, df), ddf_(ddf) {};
    template <typename Args>
    TwiceDifferentiableScalarField(const FieldType& f, const std::vector<Args>& df, const HessianType& ddf) :
        Base(f, df), ddf_(ddf) {};
    HessianType derive_twice() { return ddf_; }   // return analytical hessian
};

}   // namespace core
}   // namespace fdapde

#endif   // __DIFFERENTIABLE_FIELD_H__
