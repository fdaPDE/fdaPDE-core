#ifndef __MULTIVARIATE_POLYNOMIAL_H__
#define __MULTIVARIATE_POLYNOMIAL_H__

#include "../../utils/CompileTime.h"
#include "../../utils/Symbols.h"
#include "../../utils/fields/ScalarFieldExpressions.h"
#include "../../utils/fields/VectorField.h"

using fdaPDE::core::VectorField;
using fdaPDE::core::FieldExpr;

#include <cstddef>
#include <functional>
#include <array>

// This file provides an operative definition of a space of polynomials on an N-dimensional simplex of global degree R.
// C++14 required

/* A polynomial is given by the sum of ct_binomial_coefficient(R+N, R) monomials x1^i1*x2^i2*...*xN^iN. Fixed N and R we can
   precompute the exponents (i1, i2, ..., iN) at compile time.
   For example for a quadratic polinomial p(x) over a 3D space we would obtain a table (named exp table) as follow (R = 2):

   i1  i2  i3 | R
   --------------
   0   0   0  | 0
   0   0   1  | 1
   0   0   2  | 2
   0   1   0  | 1
   0   1   1  | 2
   0   2   0  | 2
   1   0   0  | 1
   1   0   1  | 2
   1   1   0  | 2
   2   0   0  | 2

   Each row of the above table represent a single monomial of the polynomial p(x). ct_poly_exp() computes the above table for any
   choice of N and R at compile time.
 */
template <unsigned int N, unsigned int R>
using expTable = std::array<std::array<unsigned, N>, ct_binomial_coefficient(R+N, R)>;

template <unsigned int N, unsigned int R>
constexpr expTable<N,R> ct_poly_exp(){
  const int monomials = ct_binomial_coefficient(R+N, R); // number of monomials in polynomial p(x)
  // initialize empty array
  std::array<std::array<unsigned, N>, monomials> coeff{};

  // auxiliary array. At each iteration this will be a row of the exp table
  std::array<unsigned, N> tmp{};
  int j = 0;
  while(j < monomials){
    // compute exp vector for monomial j
    int i = 0;
    bool found = false;
    while(i < N && !found){
      if(tmp[i] <= R && ct_array_sum<N>(tmp) <= R){
	found = true;
	// copy exp vector for monomial j into result
	for(int k = 0; k < N; ++k) coeff[j] = tmp;
	tmp[0]++; // increment for next iteration
	j++;      // next monomial
      }else{
	// propagate carry to next element
	tmp[i] = 0;
	tmp[++i]++;
      }
    }
  }
  return coeff;
}

/* The computation of the gradient of a multivariate polynomial can be made fast by taking advantage of the
   peculiar structure of a polynomial. This allow us to precompute at compile time the expTable_ of the gradient vector.
   For example for a quadratic polinomial p(x) over a 3D space we would obtain a table as follow:

      Recall that if we have x1^i1*x2^i2*x3^i3 its derivative wrt x1 is [m1*x1^(i1-1)]*x2^i2*x3^i3 with m1 = i1
      The value of m1 is not stored in the gradExpTable_ for compatibility with MonomialProduct<> and can be obtained
      from expTable_ directly

   i1  i2  i3 | R          i1  i2  i3 | i1  i2  i3 | i1  i2  i3
   --------------          ------------------------------------
   0   0   0  | 0          0   0   0  | 0   0   0  | 0   0   0
   0   0   1  | 1          0   0   1  | 0   0   1  | 0   0   0  
   0   0   2  | 2          0   0   2  | 0   0   2  | 0   0   1  
   0   1   0  | 1          0   1   0  | 0   0   0  | 0   1   0  
   0   1   1  | 2  ----->  0   1   1  | 0   0   1  | 0   1   0  
   0   2   0  | 2          0   2   0  | 0   1   0  | 0   2   0  
   1   0   0  | 1          0   0   0  | 1   0   0  | 1   0   0  
   1   0   1  | 2          0   0   1  | 1   0   1  | 1   0   0  
   1   1   0  | 2          0   1   0  | 1   0   0  | 1   1   0  
   2   0   0  | 2          1   0   0  | 2   0   0  | 2   0   0 
                               *   *    *       *        *   *   
      expTable_                        gradExpTable_

      * indicates that the column is equal to the corresponding one in expTable_
      
   ct_grad_exp() computes the polyExpTable_ above from the expTable_ for any choice of N and R at compile time.   
 */
template <unsigned int N, unsigned int R>
using gradExpTable = std::array<std::array<std::array<unsigned, N>, ct_binomial_coefficient(R+N, R)>, N>;

template <unsigned int N, unsigned int R>
constexpr gradExpTable<N,R> ct_grad_exp(const expTable<N,R> expTable_){
  const int monomials = ct_binomial_coefficient(R+N, R); // number of monomials in polynomial p(x)
  // initialize empty array
  std::array<std::array<std::array<unsigned, N>, monomials>, N> coeff{};

  // auxiliary array. At each iteration this will be a gradExpTable_ subtable
  std::array<std::array<unsigned, N>, monomials> tmp{};
  for(size_t i = 0; i < N; ++i){           // differentiation dimension (subtable index)
    for(size_t j = 0; j < monomials; ++j){ // row index
      for(size_t z = 0; z < N; ++z){       // column index in subtable
	  tmp[j][z] = i == z ? (expTable_[j][z] == 0 ? 0 : expTable_[j][z] - 1) : expTable_[j][z];
      }
    }
    coeff[i] = tmp; // copy subtable in coeff
  }
  return coeff;
}

// recursive template based unfolding of monomial product x1^i1*x2^i2*...*xN^iN.
// This allows to evaluate monomials at point P without explicitly looping over its individual components

template<unsigned int N, // template recursion loop variable
	 typename P,     // point where to evaluate the monomial
	 typename V>     // a row of the polynomial expTable_ (i.e. an array of coefficients [i1 i2 ... iN])
struct MonomialProduct{
  static constexpr double unfold(const P& p, const V& v){
    return v[N] == 0 ? MonomialProduct<N-1, P, V>::unfold(p, v) : std::pow(p[N], v[N]) * MonomialProduct<N-1, P, V>::unfold(p, v);
  }
};
// end of recursion
template <typename P, typename V>
struct MonomialProduct<0, P, V> { // base case
  static constexpr double unfold(const P& p, const V& v){
    return v[0] == 0 ? 1 : std::pow(p[0], v[0]);
  }
};

// unfold the sum of monomials at compile time to produce the complete polynomial expression

template <unsigned int I, // template recursion loop variable
	  unsigned int N, // total number of monomials to unfold
	  unsigned int M, // polynomial space dimension 
	  typename P,     // point where to evaluate the polynomial
	  typename V>     // the whole polynomial expTable_
struct MonomialSum {
  static constexpr double unfold(const std::array<double, N>& c, const P& p, const V& v){
    return (c[I]*MonomialProduct<M - 1, SVector<M>, std::array<unsigned, M>>::unfold(p, v[I])) + MonomialSum<I-1, N, M, P, V>::unfold(c,p,v);
  }
};
// end of recursion
template <unsigned int N, unsigned int M, typename P, typename V>
struct MonomialSum<0, N, M, P, V> {
  static constexpr double unfold(const std::array<double, N>& c, const P& p, const V& v){
    return c[0]*MonomialProduct<M - 1, SVector<M>, std::array<unsigned, M>>::unfold(p, v[0]);
  }  
};

// class representing an N-dimensional multivariate polynomial of degree R
template <unsigned int N, unsigned int R>
class MultivariatePolynomial : public FieldExpr<MultivariatePolynomial<N, R>> {
private:
  // vector of coefficients
  static const constexpr unsigned MON = ct_binomial_coefficient(R+N, R);
  std::array<double, MON> coeffVector_;

  // callable gradient
  VectorField<N> gradient_;

public:
  // compute this at compile time once, let public access
  static const constexpr expTable<N, R>     expTable_     = ct_poly_exp<N,R>();
  static const constexpr gradExpTable<N, R> gradExpTable_ = ct_grad_exp<N,R>(expTable_);
  
  // constructor
  MultivariatePolynomial() = default;
  MultivariatePolynomial(const std::array<double, MON>& coeffVector) : coeffVector_(coeffVector) {

    std::array<std::function<double(SVector<N>)>, N> gradient;
    
    // define i-th element of gradient field
    for(size_t i = 0; i < N; ++i){
      std::function<double(SVector<N>)> item = [=](SVector<N> point) -> double{
	double value = 0;
	// cycle over monomials
	for(size_t m = 0; m < MON; ++m){
	  if(expTable_[m][i] != 0){ // skip powers of zero, their derivative is zero
	    value += coeffVector[m]*expTable_[m][i]*MonomialProduct<N-1, SVector<N>, std::array<unsigned, N>>::unfold(point, gradExpTable_[i][m]);;
	  }
	}
	return value; // return partial derivative
      };
      gradient[i] = item;
    }

    // store gradient
    gradient_ = VectorField<N>(gradient);
  };

  double operator()(const SVector<N>& point) const;  // evaluate polynomial at point
  VectorField<N> gradient() const;  // return callable gradient
  
  // getter
  std::array<double, MON> getCoeff() const { return coeffVector_; }
};

template <unsigned int N, unsigned int R>
double MultivariatePolynomial<N, R>::operator()(const SVector<N> &point) const {
  return MonomialSum<MON - 1, MON, N, SVector<N>, std::array<std::array<unsigned, N>, MON>>::unfold(coeffVector_, point, expTable_);
}

template <unsigned int N, unsigned int R>
VectorField<N> MultivariatePolynomial<N, R>::gradient() const {
  return gradient_;
}

#endif // __MULTIVARIATE_POLYNOMIAL_H__
