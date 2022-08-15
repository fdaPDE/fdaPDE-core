#ifndef __SCALAR_FIELD_H__
#define __SCALAR_FIELD_H__

#include <cmath>
#include <functional>
#include <initializer_list>
#include "../Symbols.h"
#include "ScalarFieldExpressions.h"

namespace fdaPDE{
namespace core{

  // forward declaration
  template <int M, int N> class VectorField;
  
  // macro for the definition of application of trascendental functions to ScalarFields
#define DEF_FIELD_UNARY_FUNCTOR(FUN_NAME)			\
  template <int N>							\
  ScalarField<N> FUN_NAME(const ScalarField<N>& op){			\
    std::function<double(SVector<N>)> result =				\
    [=](SVector<N> x) -> double {					\
       return std::FUN_NAME(op(x));					\
    };									\
									\
    return ScalarField<N>(result);					\
  }									\
  
  // a template class for handling general scalar fields. A field wrapped by this template doesn't guarantee any regularity condition.
  // N is the domain dimension. The wrapped field must be encoded in a lambda expression receiving an SVector<N> in input and returning a double
  // extend FieldExpr to allow for expression templates
  template <int N>
  class ScalarField : public FieldExpr<ScalarField<N>> {
  private:
    // approximation of first and second derivative using central differences
    double approxFirstDerivative (const SVector<N>& x, std::size_t i, double step) const;
    double approxSecondDerivative(const SVector<N>& x, std::size_t i, std::size_t j, double step) const;
    
  protected:
    std::function<double(SVector<N>)> f_{}; // the function this class wraps
    double step_ = 0.001; // the step size used in the approximation of derivatives in .derive() and .deriveTwice() method
    
  public:
    // default constructor
    ScalarField() = default;
    // construct a scalar field from a std::function object
    ScalarField(const std::function<double(SVector<N>)>& f) : f_(f) {};
    // converting constructor from field expression to ScalarField
    template <typename E>
    ScalarField(const FieldExpr<E>& f) {
      // wraps field expression in lambda
      E op = f.get();
      std::function<double(SVector<N>)> fieldExpr = [op](SVector<N> x) -> double {
	return op(x);
      };
      f_ = fieldExpr;
    };

    // assignment from lambda expression
    template <typename L>
    ScalarField& operator=(const L& lambda) {
      f_ = lambda;
      return *this;
    }
    
    // preserve std::function syntax for evaluating a function at point, required for expression templates
    double operator()(const SVector<N>& x) const { return f_(x); };

    // approximation of gradient vector and hessian matrix without construction of a VectorField object
    SVector<N> approxGradient (const SVector<N>& x, double step) const;
    SMatrix<N> approxHessian  (const SVector<N>& x, double step) const;
    
    VectorField<N, N> derive(double step) const;
    virtual VectorField<N, N> derive() const; // uses the value of step_ as step size in central difference formula
    
    std::function<SMatrix<N>(SVector<N>)> deriveTwice(double step) const;
    virtual std::function<SMatrix<N>(SVector<N>)> deriveTwice() const; // uses the value of step_ as step size in central difference formula

    void setStep(double step) { step_ = step; }
  };

  // definition of most common mathematical functions. This allows, e.g. sin(ScalarField)
  DEF_FIELD_UNARY_FUNCTOR(sin);
  DEF_FIELD_UNARY_FUNCTOR(cos);
  DEF_FIELD_UNARY_FUNCTOR(tan);
  DEF_FIELD_UNARY_FUNCTOR(exp);
  DEF_FIELD_UNARY_FUNCTOR(log);
  
  // the following classes can be used to force particular regularity conditions on the field which might be required for some numerical methods.
  // i.e. Using a DifferentiableScalarfield as argument for a function forces to define an analytical expression for the gradient vector
  template <int N>
  class DifferentiableScalarField : public ScalarField<N> {
  protected:
    // gradient vector of scalar field f
    VectorField<N,N> df_{};
  
  public:
    // constructor
    DifferentiableScalarField(const std::function<double(SVector<N>)>& f, // base function
			      const std::initializer_list<std::function<double(SVector<N>)>>& df // gradient function
			      ) : ScalarField<N>(f), df_(df) {};

    // allow the construction of the gradient VectorField from a single lambda
    DifferentiableScalarField(const std::function<double(SVector<N>)>& f, // base function
			      const std::function<SVector<N>(SVector<N>)>& df // gradient function
			      ) : ScalarField<N>(f), df_(VectorField<N,N>(df)) {};
    
    VectorField<N,N> derive() const override { return df_; };
  };

  template <int N>
  class TwiceDifferentiableScalarField : public DifferentiableScalarField<N> {
  protected:
    // hessian matrix of scalar field f
    std::function<SMatrix<N>(SVector<N>)> ddf_{};
  
  public:
    // constructor
    TwiceDifferentiableScalarField(const std::function<double(SVector<N>)>& f, // base function
				   const std::initializer_list<std::function<double(SVector<N>)>>& df, // gradient function
				   const std::function<SMatrix<N>(SVector<N>)>& ddf // hessian function
				   ) : DifferentiableScalarField<N>(f, df), ddf_(ddf) {};

    // allow the construction of the gradient VectorField from a single lambda
    TwiceDifferentiableScalarField(const std::function<double(SVector<N>)>& f, // base function
			           const std::function<SVector<N>(SVector<N>)>& df, // gradient function
				   const std::function<SMatrix<N>(SVector<N>)>& ddf // hessian function
				   ) : DifferentiableScalarField<N>(f, df), ddf_(ddf) {};
    
    std::function<SMatrix<N>(SVector<N>)> deriveTwice() const override { return ddf_; }
  };
  
#include "ScalarField.tpp"
}}
  
#endif // __SCALAR_FIELD_H__
