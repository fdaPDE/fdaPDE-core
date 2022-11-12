#ifndef __MATRIX_FIELD_H__
#define __MATRIX_FIELD_H__

#include <array>
#include <cstddef>
#include <type_traits>
#include "../Symbols.h"
#include "ScalarField.h"
using fdaPDE::core::ScalarField;
#include "VectorField.h"
using fdaPDE::core::VectorField;
#include "expressions/MatrixExpressions.h"

namespace fdaPDE{
namespace core {

  // a template class for handling matrix field. A matrix field is a function mapping N-dimensional points to M x K dimensional
  // matrices. Supports expression template arithmetic
  template <int N, int M = N, int K = N, typename F = std::function<double(SVector<N>)>>
  class MatrixField : public MatrixExpr<N,M,K, MatrixField<N,M,K,F>> {
    static_assert(std::is_invocable<F, SVector<N>>::value &&		   
		  std::is_same<typename std::invoke_result<F,SVector<N>>::type,
		                 double>::value);
  private:
    // an M dimensional array of K dimensional array of N dimensional ScalarField
    std::array<std::array<ScalarField<N,F>, K>, M> field_;
  public:
    // constructors
    MatrixField() = default;
    MatrixField(std::array<std::array<F,K>,M> field) {
      for(std::size_t i = 0; i < M; ++i){
	for(std::size_t j = 0; j < K; ++j){
	  field_[i][j] = ScalarField<N,F>(field[i][j]);
	}
      }
    };
    // call operator
    SMatrix<M, K> operator()(const SVector<N>& point) const;
    // access operator
    const ScalarField<N,F>& operator()(std::size_t i, std::size_t j) const;
    const ScalarField<N,F>& coeff(std::size_t i, std::size_t j) const; // required by MatrixExpr
    ScalarField<N,F>& operator()(std::size_t i, std::size_t j); // non-const version
  };
  
#include "MatrixField.tpp"
}}

#endif // __MATRIX_FIELD_H__
