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
  
  // The following classes can be used to force particular regularity conditions on a scalar field
  // If GradientType_ and HessianType_ do not expose an interface compatible with VectorField and MatrixField
  // respectively, do not expect derive() and derive_twice() to be accepted by the field arithmetic
  template <int N, typename F = std::function<double(SVector<N>)>,
    	    typename GradientType_ = VectorField<N,N,std::function<double(SVector<N>)>>,
	    typename HessianType_  = MatrixField<N,N,N,std::function<double(SVector<N>)>>
	    >
  class DifferentiableScalarField : public ScalarField<N,F,GradientType_,HessianType_> {
  protected:
    typedef ScalarField<N,F,GradientType_,HessianType_> Base;
    typedef GradientType_ GradientType;
    typedef HessianType_  HessianType;
    
    GradientType df_{}; // gradient vector of scalar field f
  public:
    // constructors (f: expression of the field, df: expression of its gradient)
    DifferentiableScalarField(const F& f, const GradientType& df)
      : Base(f), df_(df) {};
    // require array-initialization of GradientType with elements of type Args
    template <typename Args>
    DifferentiableScalarField(const F& f, const std::array<Args, N>& df)
      : Base(f), df_(GradientType(df)) {};
    GradientType derive() { return df_; } // return analytical gradient
  };
  
  template <int N, typename F = std::function<double(SVector<N>)>,
    	    typename GradientType_ = VectorField<N,N,std::function<double(SVector<N>)>>,
	    typename HessianType_  = MatrixField<N,N,N,std::function<double(SVector<N>)>>
	    >
  class TwiceDifferentiableScalarField : public DifferentiableScalarField<N,F,GradientType_,HessianType_> {
  protected:
    typedef DifferentiableScalarField<N,F,GradientType_,HessianType_> Base;
    typedef GradientType_ GradientType;
    typedef HessianType_  HessianType;
    
    HessianType ddf_{}; // hessian matrix of scalar field f
  public:
    // constructors (f: expression of the field, df: expression of its gradient, ddf: expression of its hessian)
    TwiceDifferentiableScalarField(const F& f, const GradientType& df, const HessianType& ddf)
      : Base(f, df), ddf_(ddf) {};
    // require array-initialization of GradientType with elements of type Args
    template <typename Args>
    TwiceDifferentiableScalarField(const F& f, const std::array<Args, N>& df, const HessianType& ddf)
      : Base(f, df), ddf_(ddf) {};
    HessianType derive_twice() { return ddf_; } // return analytical hessian
  };
  
}}

#endif // __DIFFERENTIABLE_FIELD_H__
