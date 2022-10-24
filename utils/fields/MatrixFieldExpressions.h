#ifndef __MATRIX_FIELD_EXPRESSIONS_H__
#define __MATRIX_FIELD_EXPRESSIONS_H__

#include "../Symbols.h"
#include "ScalarField.h"
#include "VectorField.h"
#include <cstddef>
using fdaPDE::core::ScalarField;

namespace fdaPDE{
namespace core{

  // forward declaration
  template <int N, int M, int K> class MatrixField;
  
// macro for the definition of standard operations between matrix fields
#define DEF_MATRIX_EXPR_OPERATOR(OPERATOR, FUNCTOR)			                   \
  template <int N, int M, int K, typename E1, typename E2>		                   \
  MatrixBinOp<N,M,K, E1, E2, MatrixFunctorOP<N,M,K, FUNCTOR>> OPERATOR(	                   \
      const MatrixExpr<N,M,K, E1> &op1, const MatrixExpr<N,M,K, E2> &op2) {                \
  return MatrixBinOp<N,M,K, E1, E2, MatrixFunctorOP<N,M,K, FUNCTOR>>(	                   \
	op1.get(), op2.get(), MatrixFunctorOP<N,M,K, FUNCTOR>(FUNCTOR()));                 \
  }                                                                                        \
                                                                                           \
  template <int N, int M, int K, typename E>				                   \
  MatrixBinOp<N,M,K, MatrixConst<N,M,K>, E, MatrixFunctorOP<N,M,K, FUNCTOR>> OPERATOR(     \
      SMatrix<M,K> op1, const MatrixExpr<N, M, K, E> &op2) {                               \
    return MatrixBinOp<N,M,K, MatrixConst<N,M,K>, E, MatrixFunctorOP<N,M,K, FUNCTOR>>(     \
	  MatrixConst<N,M,K>(op1), op2.get(), MatrixFunctorOP<N,M,K, FUNCTOR>(FUNCTOR())); \
  }                                                                                        \
  									                   \
  template <int N, int M, int K, typename E>				                   \
  MatrixBinOp<N,M,K, E, MatrixConst<N,M,K>, MatrixFunctorOP<N,M,K, FUNCTOR>> OPERATOR(     \
      const MatrixExpr<N,M,K, E> &op1, SMatrix<M,K> op2) {                                 \
    return MatrixBinOp<N,M,K, E, MatrixConst<N,M,K>, MatrixFunctorOP<N,M,K, FUNCTOR>>(     \
	  op1.get(), MatrixConst<N,M,K>(op2), MatrixFunctorOP<N,M,K, FUNCTOR>(FUNCTOR())); \
  }                                                                                        \
  
  // Base class for any MatrixField type
  struct MatrixBase {};

  // base class for matrix expressions
  template <int N, int M, int K, typename E> struct MatrixExpr : public MatrixBase {
    // access operator on (i,j)-th element on the base type E
    ScalarField<N> coeff(std::size_t i, std::size_t j) const {
      return static_cast<const E&>(*this).coeff(i,j);
    }

    // get underyling type composing the expression node
    const E& get() const { return static_cast<const E&>(*this); }

    // evaluate the expression at point p
    SMatrix<M, K> operator()(const SVector<N>& p) const{
      SMatrix<M, K> result;
      for(std::size_t i = 0; i < M; ++i){
	for(std::size_t j = 0; j < K; ++j){
	  // trigger evaluation on each element of the expression template. This will produce for the (i,j)-th element
	  // a callable object, evaluate this passing p
	  result(i,j) = coeff(i,j)(p);
	}
      }
      return result;
    }

    // block access to i-th row/column of MatrixExpr
    VectorField<N, K> row(std::size_t i) const;
    VectorField<N, M> col(std::size_t i) const;
    
    // allow rhs multiplication by a VectorField
    VectorField<N, M> operator*(const VectorField<N, K>& op) const;
    // allor rhs multiplication by constant SVector
    VectorField<N, M> operator*(const SVector<K>& op) const;
  };

  // access i-th row of MatrixExpr
  template <int N, int M, int K, typename E>
  VectorField<N, K> MatrixExpr<N,M,K, E>::row(std::size_t i) const {
    VectorField<N ,K> result;
    for(std::size_t j = 0; j < K; ++j){
      result[j] = coeff(i,j);
    }
    return result;
  }
  // access i-th column of MatrixExpr
  template <int N, int M, int K, typename E>
  VectorField<N, M> MatrixExpr<N,M,K, E>::col(std::size_t i) const {
    VectorField<N ,M> result;
    for(std::size_t j = 0; j < M; ++j){
      result[j] = coeff(j,i);
    }
    return result;
  }
  
  // an expression node representing a constant matrix
  template <unsigned int N, unsigned int M, unsigned int K>
  class MatrixConst : public MatrixExpr<N,M,K, MatrixConst<N,M,K>> {
  private:
    SMatrix<M,K> value_;
  public:
    MatrixConst(SMatrix<M,K> value) : value_(value) { }
  
    // access operator to (i,j)-th element of underyling SMatrix
    ScalarField<N> coeff(std::size_t i, std::size_t j) const {
      std::function<double(SVector<N>)> lambda;
      lambda = [*this, i, j](const SVector<N>& p) -> double {
	return value_(i,j);
      };
      return ScalarField<N>(lambda); 
    }
  };

  // a generic binary operation node
  template <int N, int M, int K, typename OP1, typename OP2, typename BinaryOperation>
  class MatrixBinOp : public MatrixExpr<N,M,K, MatrixBinOp<N,M,K, OP1, OP2, BinaryOperation>> {
  private:
    typename std::remove_reference<OP1>::type op1_;   // first  operand
    typename std::remove_reference<OP2>::type op2_;   // second operand
    BinaryOperation f_;                               // operation to apply
  public:
    // constructor
    MatrixBinOp(const OP1& op1, const OP2& op2, BinaryOperation f) : op1_(op1), op2_(op2), f_(f) { };

    // access operator. Apply the functor to each accessed element. This returns a callable object
    ScalarField<N> coeff(std::size_t i, std::size_t j) const{
      return f_(op1_.coeff(i,j), op2_.coeff(i,j));
    }
  };

  // functor returning a callable representing the action of OP on two callable operands
  template <int N, int M, int K, typename OP> class MatrixFunctorOP {
  private:
    OP oper_;
  public:
    // constructor
    MatrixFunctorOP(const OP& oper) : oper_(oper) {}
  
    // call operator
    ScalarField<N> operator()(const ScalarField<N>& op1, const ScalarField<N>& op2) const{
      std::function<double(SVector<N>)> lambda;
      // build a lambda expression which return the evaluation of the functional op1 OP op2 at point p
      lambda = [*this, op1, op2](const SVector<N>& p) -> double{
	return oper_(op1(p),op2(p));
      };
      return ScalarField<N>(lambda);
    }
  };

  DEF_MATRIX_EXPR_OPERATOR(operator+, std::plus<> )
  DEF_MATRIX_EXPR_OPERATOR(operator-, std::minus<>)

  // allow MatrixExpr times VectorField
  template <int N, int M, int K, typename E>
  VectorField<N, M> MatrixExpr<N,M,K, E>::operator*(const VectorField<N, K>& op) const {
    VectorField<N, M> result;
    for(std::size_t i = 0; i < M; ++i){
      result[i] = row(i).dot(op); // use VectorField-VectorField DotProduct
    }
    return result;    
  }
  
  // allow MatrixExpr times SVector
  template <int N, int M, int K, typename E>
  VectorField<N, M> MatrixExpr<N,M,K, E>::operator*(const SVector<K>& op) const {
    VectorField<N, M> result;
    for(std::size_t i = 0; i < M; ++i){
      result[i] = row(i).dot(op); // use VectorField-SVector DotProduct
    }
    return result;    
  }

  // allow multiplication of MatrixExpr by scalar, this results in the multiplication of each element of the field by the scalar
  
  // class to represent a single scalar node in an MatrixExpr.
  template <int N, int M, int K>
  class MatrixScalar : public MatrixExpr<N,M,K, MatrixScalar<N,M,K>>{
  private:
    double value_;
  public:
    MatrixScalar(double value) : value_(value) { }

    // fake access operator, returns always the stored value_
    ScalarField<N> coeff(std::size_t i, std::size_t j) const {
      std::function<double(SVector<N>)> lambda;
      lambda = [*this](const SVector<N>& p) -> double { return value_; };
      return ScalarField<N>(lambda); 
    }
  };
  template <int N, int M, int K, typename E>
  MatrixBinOp<N,M,K, MatrixScalar<N,M,K>, E, MatrixFunctorOP<N,M,K, std::multiplies<>> >
  operator*(double op1, const MatrixExpr<N,M,K, E> &op2) {
    return MatrixBinOp<N,M,K, MatrixScalar<N,M,K>, E, MatrixFunctorOP<N,M,K, std::multiplies<>> >
      (MatrixScalar<N,M,K>(op1), op2.get(), MatrixFunctorOP<N,M,K, std::multiplies<>>(std::multiplies<>()));
  }
  template <int N, int M, int K, typename E>
  MatrixBinOp<N,M,K, E, MatrixScalar<N,M,K>, MatrixFunctorOP<N,M,K, std::multiplies<>> >
  operator*(const MatrixExpr<N,M,K, E> &op1, double op2) {
    return MatrixBinOp<N,M,K, E, MatrixScalar<N,M,K>, MatrixFunctorOP<N,M,K, std::multiplies<>> >
      (op1.get(), MatrixScalar<N,M,K>(op2), MatrixFunctorOP<N,M,K, std::multiplies<>>(std::multiplies<>()));
  }

  // support for MatrixExpr * MatrixExpr operation
  template <int N, int M, int K, int H, typename E1, typename E2>
  MatrixField<N,M,H> operator*(const MatrixExpr<N,M,K, E1>& op1, const MatrixExpr<N,K,H, E2>& op2){
    MatrixField<N,M,H> result;
    // implementation of simple schoolbook O(n^3) matrix multiplication
    for(std::size_t i = 0; i < M; ++i){
      for(std::size_t j = 0; j < H; ++j){
	// store the dot product of i-th row of op1 with j-th column of op2
	result(i,j) = op1.row(i).dot(op2.col(j)); 
      }
    }
    return result;
  }
  
}};

#endif // __MATRIX_FIELD_EXPRESSIONS_H__
