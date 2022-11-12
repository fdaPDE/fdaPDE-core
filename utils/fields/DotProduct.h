#ifndef __INNER_PRODUCT_H__
#define __INNER_PRODUCT_H__

#include <type_traits>
#include "../Symbols.h"
#include "expressions/ScalarExpressions.h"
#include <iostream>

namespace fdaPDE{
namespace core{

  // macro to detect if a type, once subscripted, returns directly a non-callable scalar
#define RETURN_SCALAR(T)					       \
  std::is_same<decltype(std::declval<T>().operator[](std::size_t())),  \
	       double>::value					       \
  
  // A functor to represent an inner product. T1 and T2 must provide a subscript operator []. The result of applying
  // [] to an object of type T1 or T2 must return a callable (must provide an implementation for operator()) taking an SVector as argument
  template <typename T1, typename T2>
  class DotProduct : public ScalarExpr<DotProduct<T1, T2>> {
  private:
    T1 op1_; T2 op2_; // operands of inner product

    static constexpr std::size_t ct_rows() {
      if((T1::cols == T2::cols == 1)) return T1::rows;
      else return (T1::cols == T2::rows) ? T1::cols : T1::rows;
    };
    
  public:
    // constructor
    DotProduct(const T1& op1, const T2& op2) : op1_(op1), op2_(op2) {}
    // consider the analytical expression of the inner product as a scalar field, evaluate it at point x
    template <int N>
    inline double operator()(const SVector<N>& x) const{
      // check operands dimensions are correct
      static_assert(((T1::cols == T2::cols == 1) && (T1::rows == T2::rows)) ||
		     (T1::cols == T2::rows) || (T1::rows == T2::cols));
      // implementation of the scalar product operation
      double result = 0;
      for(size_t i = 0; i < ct_rows(); ++i){ // compiler should unroll this loop
	if constexpr(!RETURN_SCALAR(T1) && !RETURN_SCALAR(T2))
	  result += (op1_[i]*op2_[i])(x); // op1_[i] and op2_[i] possibly inlined
	else{
	  if constexpr(RETURN_SCALAR(T1) && !RETURN_SCALAR(T2))
	    result += op1_[i]*op2_[i](x);
	  else
	    result += op1_[i](x)*op2_[i];
	}
      }
      return result;
    }
    // call parameter evaluation on operands (required for supporting parametric operands)
    template <typename T> const DotProduct<T1, T2>& eval_parameters(T i) {
      op1_.eval_parameters(i); op2_.eval_parameters(i);
      return *this;
    }
  };
}}

#endif // __INNER_PRODUCT_H__
