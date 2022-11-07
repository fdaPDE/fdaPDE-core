#ifndef __SCALAR_FIELD_EXPRESSIONS_H__
#define __SCALAR_FIELD_EXPRESSIONS_H__

#include "../Symbols.h"

#include <cstddef>
#include <cmath>
#include <functional>
#include <type_traits>

namespace fdaPDE{
namespace core{

  // Base class for any ScalarField type
  struct ScalarBase {};
  
// macro to define an arithmetic operator between scalar fields.
#define DEF_FIELD_EXPR_OPERATOR(OPERATOR, FUNCTOR)			\
  template <typename E1, typename E2>					\
  FieldBinOp<E1, E2, FUNCTOR>						\
  OPERATOR(const FieldExpr<E1>& op1, const FieldExpr<E2>& op2) {	\
    return FieldBinOp<E1, E2, FUNCTOR>					\
      {op1.get(), op2.get(), FUNCTOR()};				\
  }									\
  									\
  template <typename E>							\
  FieldBinOp<E, FieldScalar, FUNCTOR>					\
  OPERATOR(const FieldExpr<E>& op1, double op2) {			\
  return FieldBinOp<E, FieldScalar, FUNCTOR>				\
      (op1.get(), FieldScalar(op2), FUNCTOR());				\
  }									\
  									\
  template <typename E>							\
  FieldBinOp<FieldScalar, E, FUNCTOR>					\
  OPERATOR(double op1, const FieldExpr<E>& op2) {			\
    return FieldBinOp<FieldScalar, E, FUNCTOR>				\
      {FieldScalar(op1), op2.get(), FUNCTOR()};				\
  }									\
// macro for the definition of unary operations. FUNCTION must accept a double and return a double. Internally FUNCTION
// is wrapped by a lambda expression to make it callable (std::sin, std::cos, std::exp, ... are indeed not functor)
#define DEF_FIELD_UNARY_OPERATOR(OPERATOR, FUNCTION)			\
  template <typename E1>						\
  FieldUnOp<E1, std::function<double(double)> >				\
  OPERATOR(const FieldExpr<E1>& op1) {					\
    std::function<double(double)> OPERATOR_ =				\
      [](double x) -> double { return FUNCTION(x); };			\
									\
    return FieldUnOp<E1, std::function<double(double)> >		\
      {op1.get(), OPERATOR_};						\
  }									\

  // Base class for scalar field expressions
  template <typename E> struct FieldExpr : public ScalarBase {
    // call operator() on the base type E
    template <int N>
    inline double operator()(const SVector<N>& p) const {
      return static_cast<const E&>(*this)(p);
    }
    // get underyling type composing the expression node
    const E& get() const { return static_cast<const E&>(*this); }    
    // evaluate parametric nodes in the expression, does nothing if not redefined in derived classes
    template <typename T> void eval_parameters(T i) const { return; }
  };
    
  // an expression node representing a scalar value
  class FieldScalar : public FieldExpr<FieldScalar> {
  private:
    double value_;
  public:
    FieldScalar(double value) : value_(value) { }
    // call operator, return always the stored value
    template <int N>
    inline double operator()(const SVector<N>& p) const { return value_; };
  };

  // a parameter node
  template <typename F, typename T>
  class FieldParam : public FieldExpr<FieldParam<F,T>> {
    // check F is callable with type T and returns a double
    static_assert(std::is_same<decltype(std::declval<F>().operator()(T())), double>::value);
  private:
    const F& f_;
    double value_;
  public:
    // default constructor
    FieldParam() = default;
    FieldParam(const F& f) : f_(f) {};
    // call operator, match with any possible set of arguments
    template <typename... Args>
    double operator()(Args... args) const { return value_; }
    void eval_parameters(T i) { value_ = f_(i); }
  };
  
  // expression template based arithmetic
  template <typename OP1, typename OP2, typename BinaryOperation>
  class FieldBinOp : public FieldExpr<FieldBinOp<OP1, OP2, BinaryOperation>> {
  private:
    typename std::remove_reference<OP1>::type op1_;   // first  operand
    typename std::remove_reference<OP2>::type op2_;   // second operand
    BinaryOperation f_;                               // operation to apply
  public:
    // constructor
    FieldBinOp(const OP1& op1, const OP2& op2, BinaryOperation f) : op1_(op1), op2_(op2), f_(f) { };
    // call operator, performs the expression evaluation
    template <int N>
    double operator()(const SVector<N>& p) const{
      return f_(op1_(p), op2_(p));
    }
    // call parameter evaluation on operands
    template <typename T> const FieldBinOp<OP1, OP2, BinaryOperation>& eval_parameters(T i) {
      op1_.eval_parameters(i); op2_.eval_parameters(i);
      return *this;
    }
  };
  DEF_FIELD_EXPR_OPERATOR(operator+, std::plus<>      )
  DEF_FIELD_EXPR_OPERATOR(operator-, std::minus<>     )
  DEF_FIELD_EXPR_OPERATOR(operator*, std::multiplies<>)
  DEF_FIELD_EXPR_OPERATOR(operator/, std::divides<>   )
  
  // definition of unary operation nodes
  template <typename OP1, typename UnaryOperation>
  class FieldUnOp : public FieldExpr<FieldUnOp<OP1, UnaryOperation>> {
  private:
    typename std::remove_reference<OP1>::type op1_; // operand
    UnaryOperation f_; // operation to apply
  public:
    // constructor
    FieldUnOp(const OP1& op1, UnaryOperation f) : op1_(op1), f_(f) { };
    // call operator, performs the expression evaluation
    template <int N>
    double operator()(const SVector<N>& p) const{
      return f_(op1_(p));
    }    
  };
  DEF_FIELD_UNARY_OPERATOR(sin, std::sin)
  DEF_FIELD_UNARY_OPERATOR(cos, std::cos)
  DEF_FIELD_UNARY_OPERATOR(tan, std::tan)
  DEF_FIELD_UNARY_OPERATOR(exp, std::exp)
  DEF_FIELD_UNARY_OPERATOR(log, std::log)
  
}}

#endif // __SCALAR_FIELD_EXPRESSIONS_H__
