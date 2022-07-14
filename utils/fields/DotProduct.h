#ifndef __INNER_PRODUCT_H__
#define __INNER_PRODUCT_H__

#include "../Symbols.h"
#include "ScalarFieldExpressions.h"

namespace fdaPDE{
namespace core{
  
  // A functor to represent an inner product. T1 and T2 must provide a subscript operator []. The result of applying
  // [] to an object of type T1 or T2 must return a callable (must provide an implementation for operator()) taking an SVector as argument
  template <typename T1, typename T2>
  struct DotProduct : public FieldExpr<DotProduct<T1, T2>> {
    // operands of the inner product operation
    T1 lhs_;
    T2 rhs_;

    // constructor
    DotProduct(const T1& lhs, const T2& rhs) : lhs_(lhs), rhs_(rhs) {}

    // evaluate the form on two different points (this is like evaluating one scalar field at one point,
    // the second one at another point and then perform the inner product between the two numerical results)
    template <int N>
    double operator()(const SVector<N>& x, const SVector<N>& y) const;
    // consider the analytical expression of the inner product as a scalar field, evaluate it at point x
    template <int N>
    double operator()(const SVector<N>& x) const;
  };

  // evaluate the form on two different points (this is like evaluating one scalar field at one point,
  // the second one at another point and then perform the inner product between the two numerical results)
  template <typename T1, typename T2>
  template <int N>
  double DotProduct<T1,T2>::operator()(const SVector<N>& x, const SVector<N>& y) const{
    // implementation of the scalar product operation
    double result = 0;
    for(size_t i = 0; i < N; ++i){
      result += lhs_[i](x)*rhs_[i](y);
    }
    return result;
  }

  // consider the analytical expression of the inner product as a scalar field, evaluate it at point x
  template <typename T1, typename T2>
  template <int N>
  double DotProduct<T1,T2>::operator()(const SVector<N>& x) const{
    // implementation of the scalar product operation
    double result = 0;
    for(size_t i = 0; i < N; ++i){
      result += lhs_[i](x)*rhs_[i](x);
    }
    return result;
  }

}}

#endif // __INNER_PRODUCT_H__
