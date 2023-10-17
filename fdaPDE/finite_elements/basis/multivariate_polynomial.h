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

#ifndef __MULTIVARIATE_POLYNOMIAL_H__
#define __MULTIVARIATE_POLYNOMIAL_H__

#include <array>

#include "../../utils/compile_time.h"
#include "../../utils/symbols.h"
#include "../../fields/scalar_expressions.h"
#include "../../fields/vector_field.h"

namespace fdapde {
namespace core {

// Definition of multivariate polynomial over an N-dimensional space of global degree R.

// A polynomial is given by the sum of ct_binomial_coefficient(R+N, R) monomials x1^i1*x2^i2*...*xN^iN. Fixed N and R we
// can precompute the exponents (i1, i2, ..., iN) at compile time. As an example, consider a quadratic polinomial p(x)
// over a 3D space, then we can represent all its monomials in the following table
//
//   i1  i2  i3 | R
//   --------------
//   0   0   0  | 0
//   0   0   1  | 1
//   0   0   2  | 2
//   0   1   0  | 1
//   0   1   1  | 2
//   0   2   0  | 2
//   1   0   0  | 1
//   1   0   1  | 2
//   1   1   0  | 2
//   2   0   0  | 2
//
// ct_poly_exp() computes the above table for any choice of N and R at compile time.
template <int N, int R> using PolyTable = std::array<std::array<unsigned, N>, ct_binomial_coefficient(R + N, R)>;

template <int N, int R> constexpr PolyTable<N, R> ct_poly_exp() {
    const int monomials = ct_binomial_coefficient(R + N, R);   // number of monomials in polynomial p(x)
    // initialize empty array
    PolyTable<N, R> coeff {};

    // auxiliary array. At each iteration this will be a row of the exp table
    std::array<unsigned, N> tmp {};
    int j = 0;
    while (j < monomials) {
        // compute exp vector for monomial j
        int i = 0;
        bool found = false;
        while (i < N && !found) {
            if (tmp[i] <= R && ct_array_sum<unsigned, N>(tmp) <= R) {
                found = true;
                // copy exp vector for monomial j into result
                for (int k = 0; k < N; ++k) coeff[j] = tmp;
                tmp[0]++;   // increment for next iteration
                j++;        // next monomial
            } else {
                // propagate carry to next element
                tmp[i] = 0;
                tmp[++i]++;
            }
        }
    }
    return coeff;
}

// The structure of the gradient vector of a multivariate polynomial is known in advantage taking into account the fact
// that if we have p(x) = x1^i1 * x2^i2 * x3^i3 its derivative wrt x1 is [i1*x1^(i1-1)] * x2^i2 * x3^i3. For a quadratic
// polynomial over a 3D space, this accounts to compute the PolyGradTable below
//
//   i1  i2  i3 | R          i1  i2  i3 | i1  i2  i3 | i1  i2  i3
//   --------------          ------------------------------------
//   0   0   0  | 0          0   0   0  | 0   0   0  | 0   0   0
//   0   0   1  | 1          0   0   1  | 0   0   1  | 0   0   0
//   0   0   2  | 2          0   0   2  | 0   0   2  | 0   0   1
//   0   1   0  | 1          0   1   0  | 0   0   0  | 0   1   0
//   0   1   1  | 2  ----->  0   1   1  | 0   0   1  | 0   1   0
//   0   2   0  | 2          0   2   0  | 0   1   0  | 0   2   0
//   1   0   0  | 1          0   0   0  | 1   0   0  | 1   0   0
//   1   0   1  | 2          0   0   1  | 1   0   1  | 1   0   0
//   1   1   0  | 2          0   1   0  | 1   0   0  | 1   1   0
//   2   0   0  | 2          1   0   0  | 2   0   0  | 2   0   0
//     PolyTable                         PolyGradTable
//
// ct_grad_exp() computes the PolyGradTable above from a PolyTable for any choice of N and R at compile time.
template <int N, int R>
using PolyGradTable = std::array<std::array<std::array<unsigned, N>, ct_binomial_coefficient(R + N, R)>, N>;

template <int N, int R> constexpr PolyGradTable<N, R> ct_grad_exp(const PolyTable<N, R> poly_table) {
    const int monomials = ct_binomial_coefficient(R + N, R);   // number of monomials in polynomial p(x)
    // initialize empty array
    PolyGradTable<N,R> coeff {};

    // auxiliary array. At each iteration this will be a PolyGradTable subtable
    std::array<std::array<unsigned, N>, monomials> tmp {};
    for (size_t i = 0; i < N; ++i) {               // differentiation dimension (subtable index)
        for (size_t j = 0; j < monomials; ++j) {   // row index
            for (size_t z = 0; z < N; ++z) {       // column index in subtable
                tmp[j][z] = i == z ? (poly_table[j][z] == 0 ? 0 : poly_table[j][z] - 1) : poly_table[j][z];
            }
        }
        coeff[i] = tmp;   // copy subtable in coeff
    }
    return coeff;
}

// recursive template based unfolding of monomial product x1^i1*x2^i2*...*xN^iN.
template <int N,        // template recursion loop variable
          typename P,   // point where to evaluate the monomial
          typename V>   // a row of the polynomial PolyTable (i.e. an array of coefficients [i1 i2 ... iN])
struct MonomialProduct {
    static constexpr double unfold(const P& p, const V& v) {
        return v[N] == 0 ? MonomialProduct<N - 1, P, V>::unfold(p, v) :
                           std::pow(p[N], v[N]) * MonomialProduct<N - 1, P, V>::unfold(p, v);
    }
};
// end of recursion
template <typename P, typename V> struct MonomialProduct<0, P, V> {   // base case
    static constexpr double unfold(const P& p, const V& v) { return v[0] == 0 ? 1 : std::pow(p[0], v[0]); }
};

// unfold the sum of monomials at compile time to produce the complete polynomial expression
template <int I,        // template recursion loop variable
          int N,        // total number of monomials to unfold
          int M,        // polynomial space dimension
          typename P,   // point where to evaluate the polynomial
          typename V>   // the whole polynomial PolyTable
struct MonomialSum {
    static constexpr double unfold(const std::array<double, N>& c, const P& p, const V& v) {
        return (c[I] * MonomialProduct<M - 1, SVector<M>, std::array<unsigned, M>>::unfold(p, v[I])) +
               MonomialSum<I - 1, N, M, P, V>::unfold(c, p, v);
    }
};
// end of recursion
template <int N, int M, typename P, typename V> struct MonomialSum<0, N, M, P, V> {
    static constexpr double unfold(const std::array<double, N>& c, const P& p, const V& v) {
        return c[0] * MonomialProduct<M - 1, SVector<M>, std::array<unsigned, M>>::unfold(p, v[0]);
    }
};

// functor implementing the derivative of a multivariate N-dimensional polynomial of degree R along a given direction
template <int N, int R>
class PolynomialDerivative : public VectorExpr<N, N, PolynomialDerivative<N, R>> {
   private:
    // compile time informations
    static const constexpr unsigned n_monomials = ct_binomial_coefficient(R + N, R);
    static const constexpr PolyTable<N, R> poly_table_ = ct_poly_exp<N, R>();
    static const constexpr PolyGradTable<N, R> poly_grad_table_ = ct_grad_exp<N, R>(poly_table_);

    std::size_t i_;                                  // direction along which the derivative is computed
    std::array<double, n_monomials> coeff_vector_;   // coefficients of the polynomial whose derivative must be computed
   public:
    // constructor
    PolynomialDerivative() = default;
    PolynomialDerivative(const std::array<double, n_monomials>& coeff_vector, std::size_t i) :
        coeff_vector_(coeff_vector), i_(i) {};
    // call operator
    inline double operator()(const SVector<N>& p) const {
        double value = 0;
        // cycle over monomials
        for (size_t m = 0; m < n_monomials; ++m) {
            if (poly_table_[m][i_] != 0)   // skip powers of zero, their derivative is zero
                value +=
                  coeff_vector_[m] * poly_table_[m][i_] *
                  MonomialProduct<N - 1, SVector<N>, std::array<unsigned, N>>::unfold(p, poly_grad_table_[i_][m]);
        }
        return value;   // return partial derivative
    }
};

// class representing a multivariate polynomial of degree R defined over a space of dimension N
template <int N, int R>
class MultivariatePolynomial : public ScalarExpr<N, MultivariatePolynomial<N, R>> {
   private:
    static const constexpr int n_monomials = ct_binomial_coefficient(R + N, R);
    std::array<double, n_monomials> coeff_vector_;   // vector of coefficients
    VectorField<N, N, PolynomialDerivative<N, R>> gradient_;
   public:
    // compute this at compile time once, let public access
    static constexpr PolyTable<N, R> poly_table = ct_poly_exp<N, R>();
    static constexpr int order = R;                   // polynomial order
    static constexpr int input_space_dimension = N;   // input space dimension

    // constructor
    MultivariatePolynomial() = default;
    MultivariatePolynomial(const std::array<double, n_monomials>& coeff_vector) : coeff_vector_(coeff_vector) {
        // prepare gradient vector
        std::vector<PolynomialDerivative<N, R>> gradient;
	gradient.reserve(N);
        // define i-th element of gradient field
        for (std::size_t i = 0; i < N; ++i) { gradient.emplace_back(coeff_vector, i); }
        gradient_ = VectorField<N, N, PolynomialDerivative<N, R>>(gradient);
    };
    // evaluate polynomial at point
    double operator()(const SVector<N>& point) const {
        return MonomialSum<
          n_monomials - 1, n_monomials, N, SVector<N>,
          std::array<std::array<unsigned, N>, n_monomials>>::unfold(coeff_vector_, point, poly_table);
    };
    VectorField<N, N, PolynomialDerivative<N, R>> derive() const { return gradient_; };   // return callable gradient
    std::array<double, n_monomials> getCoeff() const { return coeff_vector_; }
};

}   // namespace core
}   // namespace fdapde

#endif   // __MULTIVARIATE_POLYNOMIAL_H__
