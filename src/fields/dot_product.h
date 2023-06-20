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

#ifndef __DOT_PRODUCT_H__
#define __DOT_PRODUCT_H__

#include "../utils/traits.h"
#include "../utils/symbols.h"
#include "scalar_expressions.h"

namespace fdapde {
namespace core {
  
  // A functor to represent an inner product. T1 and T2 must provide a subscript operator []. The result of applying
  // [] to an object of type T1 or T2 must return a callable accepting an SVector<N> as argument
  template <typename T1, typename T2>
  class DotProduct : public ScalarExpr<DotProduct<T1, T2>> {
  private:
    T1 op1_; T2 op2_; // operands of inner product

    // number of single additions involved in the dot product
    static constexpr std::size_t ct_rows();
    // trait returning true if T.operator[](std::size_t) returns a double
    template <typename T>
    struct subscript_to_double {
      static constexpr bool value = std::is_same<
	typename subscript_result_of<T, std::size_t>::type, double>
	::value;
    };
    
  public:
    // constructor
    DotProduct(const T1& op1, const T2& op2) : op1_(op1), op2_(op2) {}
    template <int N> inline double operator()(const SVector<N>& x) const; // evaluate dot(op1, op2) at point x
    template <typename T> const DotProduct<T1,T2>& eval_parameters(T i);  // triggers parameter evaluation on operands
  };

  // implementation details

  template <typename T1, typename T2>
  constexpr std::size_t DotProduct<T1,T2>::ct_rows() {
    if((T1::cols == T2::cols == 1)) {
      return T1::rows;
    } else {
      return (T1::cols == T2::rows) ? T1::cols : T1::rows;
    }      
  }
  
  template <typename T1, typename T2>
  template <int N>
  double DotProduct<T1,T2>::operator()(const SVector<N>& x) const {
      // check operands dimensions are correct
      static_assert(((T1::cols == T2::cols == 1) && (T1::rows == T2::rows)) ||
		     (T1::cols == T2::rows) || (T1::rows == T2::cols));
      // implementation of the scalar product operation
      double result = 0;
      for(size_t i = 0; i < ct_rows(); ++i){
	if constexpr (!subscript_to_double<T1>::value && !subscript_to_double<T2>::value) {
	  result += (op1_[i]*op2_[i])(x);
	} else {
	  if constexpr(subscript_to_double<T1>::value && !subscript_to_double<T2>::value) {
	    result += op1_[i]*op2_[i](x);
	  } else {
	    result += op1_[i](x)*op2_[i];
	  }
	}
      }
      return result;   
  }

  template <typename T1, typename T2>
  template <typename T>
  const DotProduct<T1,T2>& DotProduct<T1,T2>::eval_parameters(T i) {
    op1_.eval_parameters(i); op2_.eval_parameters(i);
    return *this;
  }
  
}}

#endif // __DOT_PRODUCT_H__
