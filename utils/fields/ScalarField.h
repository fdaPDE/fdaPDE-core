#ifndef __SCALAR_FIELD_H__
#define __SCALAR_FIELD_H__

#include <cmath>
#include <type_traits>
#include "../Symbols.h"
#include "expressions/ScalarExpressions.h"
using fdaPDE::core::ScalarExpr;

namespace fdaPDE{
namespace core{

  // forward declaration
  template <int M, int N, typename F> class VectorField;

  // a functor representing a zero field
  template <int N>
  struct ZeroField : public ScalarExpr<ZeroField<N>> { 
    inline double operator()(const SVector<N>& p) const { return 0; }
  };
  // a functor representing a constant field
  template <int N>
  class ConstantField : public ScalarExpr<ConstantField<N>> {
  private:
    double c_;
  public:
    ConstantField(double c) : c_(c) {};
    inline double operator()(const SVector<N>& p) const { return c_; }
  };
  
  // a template class for handling general scalar fields. A ScalarFiels is a wrapper for a generic callable F having an implementation for
  // double operator()(const SVector<N>&)
  
  // In general using F = std::function<double(SVector<N>)> is fine but must be avoided at any performance-critical point of the library
  // (any point inside the core). Indeed std::function is an highly polymorphic type with a non-zero run-time cost.
  template <int N, typename F = std::function<double(SVector<N>)>>
  class ScalarField : public ScalarExpr<ScalarField<N,F>> {
    static_assert(std::is_invocable<F, SVector<N>>::value &&		   
		  std::is_same<typename std::invoke_result<F,SVector<N>>::type, 
		                 double>::value);				   
    private:
    // approximation of first and second derivative using central differences
    double approxFirstDerivative (const SVector<N>& x, std::size_t i, double step) const;
    double approxSecondDerivative(const SVector<N>& x, std::size_t i, std::size_t j, double step) const;
  protected:
    F f_{}; // the function this class wraps
    double step_ = 0.001; // the step size used in the approximation of derivatives in .derive() and .deriveTwice() method
  public:
    // default constructor
    ScalarField() = default;
    // construct a scalar field from a std::function object
    ScalarField(const F& f) : f_(f) {};
    // assignement and constructor from a ScalarExpr requires the base type F to be a std::function for type erasure
    template <typename E, typename U = F,
              typename std::enable_if<
		std::is_same<U, std::function<double(SVector<N>)>>::value,
		int>::type = 0>
    ScalarField(const ScalarExpr<E>& f) {
      // wraps field expression in lambda
      E op = f.get();
      std::function<double(SVector<N>)> fieldExpr = [op](SVector<N> x) -> double {
	return op(x);
      };
      f_ = fieldExpr;      
    };
    template <typename E, typename U = F>
    typename std::enable_if<
      std::is_same<U, std::function<double(SVector<N>)>>::value,
      ScalarField<N>&>::type
    operator=(const ScalarExpr<E>& f) {
      // wraps field expression in lambda
      E op = f.get();
      std::function<double(SVector<N>)> fieldExpr = [op](SVector<N> x) -> double {
	return op(x);
      };
      f_ = fieldExpr;
      return *this;
    };
    // assignment from lambda expression. Be carefull of the fact that the lambda will be wrapped in a std::function object, which has
    // NOT zero run-time cost. Use this only if strictly necessary.
    template <typename L, typename U = F>
    typename std::enable_if<
      std::is_same<U, std::function<double(SVector<N>)>>::value,
      ScalarField<N>&>::type
    operator=(const L& lambda) {
      // assign lambda to std::function object
      f_ = lambda;
      return *this;
    }
    // initializer for a zero field
    static ScalarField<N, ZeroField<N>> Zero() { return ScalarField<N, ZeroField<N>>(ZeroField<N>()); }
    // initializer for a constant field
    static ScalarField<N, ConstantField<N>> Const(double c) {
      return ScalarField<N, ConstantField<N>>(ConstantField<N>(c));
    }
    
    // preserve callable syntax for evaluating a function at point
    inline double operator()(const SVector<N>& x) const { return f_(x); };
    // approximation of gradient vector and hessian matrix without construction of a VectorField object
    void setStep(double step) { step_ = step; } // set step size used for numerical approximations
    SVector<N> approxGradient (const SVector<N>& x, double step) const;
    SMatrix<N> approxHessian  (const SVector<N>& x, double step) const;
    
    // gradient objects
    VectorField<N,N,std::function<double(SVector<N>)>> derive(double step) const;
    virtual VectorField<N,N,std::function<double(SVector<N>)>> derive() const; // uses stored step_ value
    // hessian objects
    std::function<SMatrix<N>(SVector<N>)> deriveTwice(double step) const;
    virtual std::function<SMatrix<N>(SVector<N>)> deriveTwice() const;// uses stored step_ value
  };
  // template argument deduction rule for the special case F = std::function<double(SVector<N>)>
  template <int N> ScalarField(const std::function<double(SVector<N>)>&)
    -> ScalarField<N, std::function<double(SVector<N>)>>;
  
  // The following classes can be used to force particular regularity conditions on the field.
  template <int N, typename F1 = std::function<double(SVector<N>)>, typename F2 = F1>
  class DifferentiableScalarField : public ScalarField<N,F1> {
  protected:
    // gradient vector of scalar field f
    VectorField<N,N,F2> df_{};
  public:
    // constructors (f: expression of the field, df: expression of its gradient)
    DifferentiableScalarField
    (const F1& f, const VectorField<N,N,F2>& df)
      : ScalarField<N,F1>(f), df_(df) {};
    DifferentiableScalarField
    (const F1& f, const std::array<F2,N>& df)
      : ScalarField<N,F1>(f), df_(VectorField<N,N,F2>(df)) {};
    // return analytical gradient
    VectorField<N,N,F2> derive() const override { return df_; };
  };
  
  template <int N, typename F1 = std::function<double(SVector<N>)>, typename F2 = F1>
  class TwiceDifferentiableScalarField : public DifferentiableScalarField<N,F1,F2> {
  protected:
    // hessian matrix of scalar field f
    std::function<SMatrix<N>(SVector<N>)> ddf_{};
  public:
    // constructors (f: expression of the field, df: expression of its gradient, ddf: expression of its hessian)
    TwiceDifferentiableScalarField
    (const F1& f, const VectorField<N,N,F2>& df, const std::function<SMatrix<N>(SVector<N>)>& ddf)
      : DifferentiableScalarField<N,F1,F2>(f, df), ddf_(ddf) {};
    TwiceDifferentiableScalarField
    (const F1& f, const std::array<F2,N>& df, const std::function<SMatrix<N>(SVector<N>)>& ddf)
      : DifferentiableScalarField<N,F1,F2>(f, df), ddf_(ddf) {};
    // return analytical hessian
    std::function<SMatrix<N>(SVector<N>)> deriveTwice() const override { return ddf_; }
    };

#include "ScalarField.tpp"
}}
  
#endif // __SCALAR_FIELD_H__
