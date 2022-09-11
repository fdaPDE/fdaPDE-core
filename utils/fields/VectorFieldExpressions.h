#ifndef __VECTOR_FIELD_EXPRESSIONS_H__
#define __VECTOR_FIELD_EXPRESSIONS_H__

#include "../Symbols.h"
#include <functional>

namespace fdaPDE{
namespace core{

  // forward declaration
  template <int N> class ScalarField;
  
// macro for the definition of standard operations between vector fields
#define DEF_VECT_EXPR_OPERATOR(OPERATOR, FUNCTOR)                                \
  template <int M, int N, typename E1, typename E2>			         \
  VectBinOp<M,N, E1, E2, VectFunctorOP<M,N, FUNCTOR>> OPERATOR(	                 \
      const VectExpr<M,N, E1> &op1, const VectExpr<M,N, E2> &op2) {              \
  return VectBinOp<M,N, E1, E2, VectFunctorOP<M,N, FUNCTOR>>(		         \
	op1.get(), op2.get(), VectFunctorOP<M,N, FUNCTOR>(FUNCTOR()));           \
  }                                                                              \
                                                                                 \
  template <int M, int N, typename E>					         \
  VectBinOp<M,N, VectConst<M,N>, E, VectFunctorOP<M,N, FUNCTOR>> OPERATOR(       \
      SVector<N> op1, const VectExpr<M, N, E> &op2) {                            \
  return VectBinOp<M,N, VectConst<M,N>, E, VectFunctorOP<M,N, FUNCTOR>>(         \
	VectConst<M,N>(op1), op2.get(), VectFunctorOP<M,N, FUNCTOR>(FUNCTOR())); \
  }                                                                              \
  									         \
  template <int M, int N, typename E>					         \
  VectBinOp<M,N, E, VectConst<M,N>, VectFunctorOP<M,N, FUNCTOR>> OPERATOR(       \
      const VectExpr<M,N, E> &op1, SVector<N> op2) {                             \
  return VectBinOp<M,N, E, VectConst<M,N>, VectFunctorOP<M,N, FUNCTOR>>(         \
        op1.get(), VectConst<M,N>(op2), VectFunctorOP<M,N, FUNCTOR>(FUNCTOR())); \
  }                                                                              \

  // Base class for vectorial expressions
  // M dimension of the space where the field is defined, N dimension of the arriving space
  template <int M, int N, typename E> struct VectExpr {
    // call operator[] on the base type E
    ScalarField<M> operator[](size_t i) const {
      return static_cast<const E&>(*this)[i];
    }

    // get underyling type composing the expression node
    const E& get() const { return static_cast<const E&>(*this); }

    // evaluate the expression at point p
    SVector<N> operator()(const SVector<M>& p) const{
      SVector<N> result;
      for(size_t i = 0; i < N; ++i){
	// trigger evaluation, call subscript of the underyling type. This will produce along the dimension i
	// a callable object, evaluate this passing the point p to get a double
	result[i] = operator[](i)(p);
      }
      return result;
    }

    // dot product between VectExpr and SVector
    virtual ScalarField<M> dot(const SVector<N>& op) const;
  };

  // an expression node representing a constant vector
  template <unsigned int M, unsigned int N>
  class VectConst : public VectExpr<M,N, VectConst<M,N>> {
  private:
    SVector<N> value_;
  public:
    VectConst(SVector<N> value) : value_(value) { }
  
    // subsript operator, returns a fake lambda just to provide a call operator
    ScalarField<M> operator[](size_t i) const {
      std::function<double(SVector<M>)> lambda;
      lambda = [*this, i](const SVector<M>& p) -> double {
	return value_[i];
      };
      return ScalarField<M>(lambda); 
    }
  };
  
  // a generic binary operation node
  template <int M, int N, typename OP1, typename OP2, typename BinaryOperation>
  class VectBinOp : public VectExpr<M,N, VectBinOp<M,N, OP1, OP2, BinaryOperation>> {
  private:
    typename std::remove_reference<OP1>::type op1_;   // first  operand
    typename std::remove_reference<OP2>::type op2_;   // second operand
    BinaryOperation f_;                               // operation to apply

  public:
    // constructor
    VectBinOp(const OP1& op1, const OP2& op2, BinaryOperation f) : op1_(op1), op2_(op2), f_(f) { };

    // subscript operator. Apply the functor to each subscripted operand. This returns a callable object
    ScalarField<M> operator[](size_t i) const{
      return f_(op1_[i], op2_[i]); // converting constructor ScalarField(std::function<double(SVector<N>)>) called
    }
  };

  // functor returning a callable representing the action of OP on two callable operands
  template <int M, int N, typename OP> class VectFunctorOP {
  private:
    OP oper_;

  public:
    // constructor
    VectFunctorOP(const OP& oper) : oper_(oper) {}
  
    // call operator
    ScalarField<M> operator()(const ScalarField<M>& op1, const ScalarField<M>& op2) const{
      std::function<double(SVector<M>)> lambda;
      // build a lambda expression which return the evaluation of the functional op1 OP op2 at point p
      lambda = [*this, op1, op2](const SVector<M>& p) -> double{
	return oper_(op1(p),op2(p));
      };
      return ScalarField<M>(lambda);
    }
  };

  DEF_VECT_EXPR_OPERATOR(operator+, std::plus<> )
  DEF_VECT_EXPR_OPERATOR(operator-, std::minus<>)

  // special logic to handle double*VectExpr operation. The application of operator* between a double and a
  // VectExpr results in the multiplication of the double value to each dimension of the VectExpr

  // class to represent a single scalar in an expression.
  template <unsigned int M, unsigned int N>
  class VectScalar : public VectExpr<M,N, VectScalar<M,N>>{
  private:
    double value_;
  public:
    VectScalar(double value) : value_(value) { }

    // fake subsript operator, returns always the stored value_
    std::function<double(SVector<M>)> operator[](size_t i) const {
      std::function<double(SVector<M>)> lambda;
      lambda = [*this](const SVector<M>& p) -> double { return value_; };
      return lambda; 
    }
  };

  template <int M, int N, typename E>
  VectBinOp<M,N, VectScalar<M,N>, E, VectFunctorOP<M,N, std::multiplies<>> >
  operator*(double op1, const VectExpr<M,N, E> &op2) {
    return VectBinOp<M,N, VectScalar<M,N>, E, VectFunctorOP<M,N, std::multiplies<>> >
      (VectScalar<M,N>(op1), op2.get(), VectFunctorOP<M,N, std::multiplies<>>(std::multiplies<>()));
  }  

  template <int M, int N, typename E>
  VectBinOp<M,N, E, VectScalar<M,N>, VectFunctorOP<M,N, std::multiplies<>> >
  operator*(const VectExpr<M,N, E> &op1, double op2) {
    return VectBinOp<M,N, E, VectScalar<M,N>, VectFunctorOP<M,N, std::multiplies<>> >
      (op1.get(), VectScalar<M,N>(op2), VectFunctorOP<M,N, std::multiplies<>>(std::multiplies<>()));
  }

  // allow dot product between a VectExpr and an (eigen) SVector.
  template <int M, int N, typename E>
  ScalarField<M> VectExpr<M,N, E>::dot(const SVector<N>& op) const {
    std::function<double(SVector<M>)> result;
    // build lambda expressing inner product
    result = [*this, op](const SVector<M>& x) -> double{
      double y = 0;
      for(size_t i = 0; i < N; ++i)
	y += op[i]*operator[](i)(x);
      
      return y;
    };
    return ScalarField<M>(result);
  }
}}

#endif // __VECTOR_FIELD_EXPRESSIONS_H__
