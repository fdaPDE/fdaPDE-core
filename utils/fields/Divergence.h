#ifndef __DIVERGENCE_H__
#define __DIVERGENCE_H__

#include "expressions/ScalarExpressions.h"
#include "VectorField.h"
#include <cstddef>
#include <functional>

namespace fdaPDE{
namespace core{

  // a functor representing the divergence of a VectorField<N>
  template <int N>
  class Divergence : public ScalarExpr<Divergence<N>>{
  private:
    // the function encoding the divergence operator
    std::function<double(SVector<N>)> div_;

  public:
    Divergence(VectorField<N>& field) {
      // set up the functor encoding the divergence operator
      std::function<double(SVector<N>)> div = [field](SVector<N> x) mutable -> double {
	double result = 0;
	for(std::size_t i = 0; i < N; ++i){
	  result += field[i].derive()(x)[i]; // sum of derivatives evaluated at point x
	}
	return result;
      };
      div_ = div;
    };
    // call operator, const and non-const version
    double operator()(const SVector<N>& x) { return div_(x); };
    double operator()(const SVector<N>& x) const { return div_(x); };
  };

  template <int N>
  Divergence<N> div(VectorField<N>& field){
    return Divergence<N>(field);
  }
  
}};
#endif // __DIVERGENCE_H__
