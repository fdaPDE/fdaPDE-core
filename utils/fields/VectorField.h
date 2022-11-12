#ifndef __VECTOR_FIELD_H__
#define __VECTOR_FIELD_H__

#include <cstddef>
#include <type_traits>
#include "../Symbols.h"
#include "ScalarField.h"
using fdaPDE::core::ScalarField;
using fdaPDE::core::ZeroField;
#include "expressions/VectorExpressions.h"
#include "DotProduct.h"
//#include "MatrixFieldExpressions.h"

namespace fdaPDE{
namespace core{
  
  // template class representing a general vector field from an M-dimensional to an N-dimensional space. Support expression template arithmetic.
  template <int M, int N = M, typename F = std::function<double(SVector<M>)>>
  class VectorField : public VectorExpr<M,N, VectorField<M,N,F>>{
    static_assert(std::is_invocable<F, SVector<N>>::value &&		   
		  std::is_same<typename std::invoke_result<F,SVector<N>>::type,
		                 double>::value);				   
  private:
    // each array element is a lambda which computes the i-th component of the vector
    std::array<ScalarField<M,F>,N> field_;
  public:
    // expose compile time informations
    static constexpr int rows = N;
    static constexpr int cols = 1;
    // constructor
    VectorField() = default;
    // construct from an array of F objects
    VectorField(const std::array<F,N>& field){
      for(std::size_t i = 0; i < N; ++i) // assign a ScalarField to each component of the VectorField
	field_[i] = ScalarField<M,F>(field[i]);
    }
    // wrap a VectorExpr into a valid VectorField
    template <typename E, typename U = F,
              typename std::enable_if<
		std::is_same<U, std::function<double(SVector<N>)>>::value,
		int>::type = 0>
    VectorField(const VectorExpr<M,N,E>& expr){
      for(std::size_t i = 0; i < N; ++i)
	field_[i] = expr[i];
    }
    // initializer for a zero field
    static VectorField<N,N,ZeroField<N>> Zero() {
      return VectorField<N,N,ZeroField<N>>(std::array<ZeroField<N>, N>{});
    }
    // call operator
    inline SVector<N> operator()(const SVector<M>& point) const;
    // subscript operator
    inline const ScalarField<M,F>& operator[](size_t i) const;
    inline ScalarField<M,F>& operator[](size_t i);

    // inner product VectorField.dot(VectorField)
    DotProduct<VectorField<M,N,F>, VectorConst<M,N>> dot(const SVector<N>& rhs) const;
    // Inner product VectorField.dot(VectorExpr)
    template <typename E>
    DotProduct<VectorField<M,N,F>, E> dot(const VectorExpr<M,N,E>& expr) const;
  };

#include "VectorField.tpp"
}}

#endif // __VECTOR_FIELD_H__
