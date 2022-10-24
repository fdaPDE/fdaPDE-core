#ifndef __MATRIX_FIELD_H__
#define __MATRIX_FIELD_H__

#include <array>
#include <cstddef>
#include <initializer_list>
#include "../Symbols.h"
#include "ScalarField.h"
using fdaPDE::core::ScalarField;
#include "VectorField.h"
using fdaPDE::core::VectorField;
#include "MatrixFieldExpressions.h"

namespace fdaPDE{
namespace core {

  // forward declarations
  template <int N> class ScalarField;
  template <int N, int M> class VectorField;

  // a template class for handling matrix field. A matrix field is a function mapping N-dimensional points to M x K dimensional
  // matrices. Supports expression template arithmetic
  template <int N, int M = N, int K = N>
  class MatrixField : public MatrixExpr<N,M,K, MatrixField<N,M,K>> {
  private:
    // an M dimensional array of K dimensional array of N dimensional ScalarField
    std::array<std::array<ScalarField<N>, K>, M> field_;
  public:
    // constructors
    MatrixField() = default;
    MatrixField(std::array<std::array<std::function<double(SVector<N>)>, K>, M> field);
    // allow braced-list initialization. ScalarFields are given row-wise (each list refers to a MatrixField row)
    MatrixField(std::initializer_list<std::initializer_list<std::function<double(SVector<N>)>>> field);

    // call operator
    SMatrix<M, K> operator()(const SVector<N>& point) const;
    // access operator
    const ScalarField<N>& operator()(std::size_t i, std::size_t j) const;
    const ScalarField<N>& coeff(std::size_t i, std::size_t j) const; // required by MatrixExpr
    ScalarField<N>& operator()(std::size_t i, std::size_t j); // non-const version

    // block access
    VectorField<N, K> row(std::size_t i) const;
    VectorField<N, M> col(std::size_t i) const;
  };
  
#include "MatrixField.tpp"
}}

#endif // __MATRIX_FIELD_H__
