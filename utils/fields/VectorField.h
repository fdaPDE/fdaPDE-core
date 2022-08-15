#ifndef __VECTOR_FIELD_H__
#define __VECTOR_FIELD_H__

#include <cstddef>
#include <initializer_list>
#include "../Symbols.h"
#include "ScalarField.h"
#include "VectorFieldExpressions.h"
#include "DotProduct.h"

namespace fdaPDE{
namespace core{
  
  // template class representing a general vector field from an M-dimensional to an N-dimensional space. Support expression template arithmetic.
  template <int M, int N = M>
  class VectorField : public VectExpr<M,N, VectorField<M,N>>{
  private:
    // each array element is a lambda which computes the i-th component of the vector
    std::array<ScalarField<M>, N> field_;
  public:
    // constructor
    VectorField() = default;
    VectorField(std::array<std::function<double(SVector<M>)>, N> field);
    // allow braced-list initialization
    VectorField(std::initializer_list<std::function<double(SVector<M>)>> field);
    // allow for the construction of a VectorField from a single vectorial lambda. Observe that by doing so evaluating the
    // field along direction i only still requires the evaluation of all other dimesions. Use with care.
    VectorField(const std::function<SVector<N>(SVector<M>)>& field);
    // wrap a VectExpr into a valid VectorField
    template <typename E>
    VectorField(const VectExpr<M, N, E>& expr);    
    // call operator
    SVector<N> operator()(const SVector<M>& point) const;
    // subscript operator
    const ScalarField<M>& operator[](size_t i) const;
    ScalarField<M>& operator[](size_t i);

    // inner product VectorField.dot(VectorField)
    DotProduct<VectorField<M,N>, VectorField<M,N>> dot(const VectorField<M,N>& rhs) const;
    // Inner product VectorField.dot(SVector)
    ScalarField<M> dot(const SVector<N>& op) const;

    // expose compile time informations
    static constexpr int size = N;
  };

  // template argument deduction guide
  template <int M, int N> VectorField(std::array<std::function<double(SVector<M>)>, N>) -> VectorField<M,N>;
  
#include "VectorField.tpp"
}}

#endif // __VECTOR_FIELD_H__
