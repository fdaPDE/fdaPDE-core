// This file is part of fdaPDE, a C++ library for physics-informed
// spatial and functional data analysis.
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __SCALAR_FIELD_H__
#define __SCALAR_FIELD_H__

#include <type_traits>
#include "../utils/symbols.h"
#include "scalar_expressions.h"

namespace fdapde {
namespace core {

  // forward declaration
  template <int M, int N, typename F> class VectorField;
  template <int M, int N, int K, typename F> class MatrixField;
  
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
  
  // a template class for handling general scalar fields. A ScalarFiels is a wrapper for a generic callable
  // F having an implementation for "double operator()(const SVector<N>&)"
  // In general using F = std::function<double(SVector<N>)> is fine but must be avoided at any performance-critical
  // point of the library (any point inside the core library system). std::function is an highly polymorphic type with
  // a non-zero run-time cost due to virtual dispatching.
  template <int N, typename F = std::function<double(SVector<N>)>,
	    typename GradientType_ = VectorField<N,N,std::function<double(SVector<N>)>>,
	    typename HessianType_  = MatrixField<N,N,N,std::function<double(SVector<N>)>>
	    >
  class ScalarField : public ScalarExpr<ScalarField<N,F,GradientType_,HessianType_>> {
    static_assert(std::is_invocable<F, SVector<N>>::value &&		   
		  std::is_same<typename std::invoke_result<F,SVector<N>>::type, 
		                 double>::value);				   
    private:
    // approximation of df(x_1, x_2, ... x_N)/dx_i via central finite differences
    double approx_first_derivative (const SVector<N>& x, std::size_t i, double step);
    // approximation of df^2(x_1, x_2, ... x_N)/(dx_i*dx_j) via central finite differences
    double approx_second_derivative(const SVector<N>& x, std::size_t i, std::size_t j, double step);
  protected:
    F f_{};
    double step_ = 0.001; // step size used in the approximation of derivatives in .derive() and .derive_twice()
  public:
    typedef F             FieldType;    // type of wrapped functor
    typedef GradientType_ GradientType; // return type of approximated gradient vector
    typedef HessianType_  HessianType;  // return type of approximated hessian matrix
    
    // constructors
    ScalarField() = default;
    ScalarField(const F& f) : f_(f) {};

    // assignement and constructor from a ScalarExpr requires the base type F to be a std::function for type erasure
    template <typename E, typename U = F,
              typename std::enable_if<
		std::is_same<U, std::function<double(SVector<N>)>>::value,
		int>::type = 0>
    ScalarField(const ScalarExpr<E>& f) {
      // wraps field expression in lambda
      E op = f.get();
      std::function<double(SVector<N>)> fieldExpr = [op](SVector<N> x) -> double { return op(x); };
      f_ = fieldExpr;      
    };
    template <typename E, typename U = F>
    typename std::enable_if<
      std::is_same<U, std::function<double(SVector<N>)>>::value,
      ScalarField<N>&>::type
    operator=(const ScalarExpr<E>& f) {
      // wraps field expression in lambda
      E op = f.get();
      std::function<double(SVector<N>)> fieldExpr = [op](SVector<N> x) -> double { return op(x); };
      f_ = fieldExpr;
      return *this;
    };
    // assignment from lambda expression. Note: the lambda expression get's wrapped in a std::function object,
    // with a NOT zero run-time cost.
    template <typename L, typename U = F>
    typename std::enable_if<
      std::is_same<U, std::function<double(SVector<N>)>>::value,
      ScalarField<N>&>::type
    operator=(const L& lambda) {
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
    inline double operator()(const SVector<N>& x) { return f_(x); };
    // approximation of gradient vector and hessian matrix
    void set_step(double step) { step_ = step; } // set step size used for numerical approximations
    SVector<N> approx_gradient (const SVector<N>& x, double step);
    SMatrix<N> approx_hessian  (const SVector<N>& x, double step);
    
    // approximate gradient vector
    template <typename U = GradientType>
    typename std::enable_if<
      std::is_constructible<U, std::array<std::function<double(SVector<N>)>, N>>::value,
      GradientType>::type derive(double step);
    GradientType derive() { return derive(step_); };
    // approximate hessian matrix
    template <typename U = HessianType>
    typename std::enable_if<
      std::is_constructible<U, std::array<std::array<std::function<double(SVector<N>)>,N>,N>>::value,
      HessianType>:: type derive_twice(double step);
    HessianType derive_twice() { return derive_twice(step_); };
  };
  // template argument deduction rule for the special case F = std::function<double(SVector<N>)>
  template <int N> ScalarField(const std::function<double(SVector<N>)>&)
    -> ScalarField<N, std::function<double(SVector<N>)>>;

  // implementation details

  template <int N, typename F, typename G, typename H>
  double ScalarField<N,F,G,H>::approx_first_derivative(const SVector<N>& x, std::size_t i, double step) {
    // variation around point x along direction i
    SVector<N> h = SVector<N>::Zero();
    h[i] = step; 
    return (f_(x + h) - f_(x - h))/(2*step);
  }

  template <int N, typename F, typename G, typename H>
  double ScalarField<N,F,G,H>::approx_second_derivative(const SVector<N>& x, std::size_t i, std::size_t j, double step) {
    SVector<N> h_i = SVector<N>::Zero();
    h_i[i] = step; // variation along i-th direction
    if(i == j)
      return (-f_(x + 2*h_i) + 16*f_(x + h_i) - 30*f_(x) + 16*f_(x - h_i) - f_(x - 2*h_i))/(12*pow(step,2));
    else{ 
      SVector<N> h_j = SVector<N>::Zero();
      h_j[j] = step; // variation along j-th direction
      return (f_(x + h_i + h_j) - f_(x + h_i - h_j) - f_(x - h_i + h_j) + f_(x - h_i - h_j))/(4*pow(step,2));	
    }
  }

  // gradient vector computation
  // approximate computation of gradient via central finite differences
  template <int N, typename F, typename G, typename H>
  SVector<N> ScalarField<N,F,G,H>::approx_gradient(const SVector<N>& x, double step) {
    SVector<N> gradient;
    for(size_t i = 0; i < N; ++i){
      // approximation of i-th partial derivative at point x
      gradient[i] = approx_first_derivative(x, i, step);
    }
    return gradient;
  }
  // return gradient as callable object
  template <int N, typename F, typename G, typename H>
  template <typename U>
  typename std::enable_if<
    std::is_constructible<U, std::array<std::function<double(SVector<N>)>, N>>::value,
    typename ScalarField<N,F,G,H>::GradientType>::type
  ScalarField<N,F,G,H>::derive(double step) {
    std::array<std::function<double(SVector<N>)>,N> components;
    for(std::size_t i = 0; i < N; ++i){
      std::function<double(SVector<N>)> gradient_approx = [=](SVector<N> x) -> double {
	return approx_first_derivative(x, i, step);
      };
      components[i] = gradient_approx;
    }
    return GradientType(components);
  }

  // hessian matrix computation
  // approximate computation of hessian matrix via central finite differences
  template <int N, typename F, typename G, typename H>
  SMatrix<N> ScalarField<N,F,G,H>::approx_hessian(const SVector<N>& x, double step) {
    SMatrix<N> hessian = SMatrix<N>::Zero();
    // hessian matrix is symmetric, compute just the lower triangular part
    for(size_t i = 0; i<N; ++i){
      for(size_t j = 0; j<=i; ++j){
	// approximation of (i,j)-th partial derivative at point x
	hessian(i,j) = approx_second_derivative(x, i, j, step);
	hessian(j,i) = hessian(i,j); // exploit symmetry of hessian matrix
      }
    }
    return hessian;
  }
  // return hessian as callable object
  template <int N, typename F, typename G, typename H>
  template <typename U>
  typename std::enable_if<
    std::is_constructible<U, std::array<std::array<std::function<double(SVector<N>)>,N>,N>>::value,
    typename ScalarField<N,F,G,H>::HessianType>::type
  ScalarField<N,F,G,H>::derive_twice(double step) {
    std::array<std::array<std::function<double(SVector<N>)>,N>,N> components;
    for(std::size_t i = 0; i < N; ++i) {
      for(std::size_t j = 0; j < N; ++j) {
	std::function<double(SVector<N>)> hessian_approx = [=](SVector<N> x) -> double {
	  return approx_second_derivative(x, i, j, step);
	};
	components[i][j] = hessian_approx;
      }
    }
    return HessianType(components);
  }
  
}}
  
#endif // __SCALAR_FIELD_H__
