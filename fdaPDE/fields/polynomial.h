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

#ifndef __POLYNOMIAL_H__
#define __POLYNOMIAL_H__

#include <array>

#include "../fields/scalar_expressions.h"
#include "../fields/vector_field.h"
#include "../utils/combinatorics.h"
#include "../utils/symbols.h"

namespace fdapde {
namespace core {

// multivariate polynomial of degree R over \mathbb{R}^N
template <int N, int R> class Polynomial : public ScalarExpr<N, Polynomial<N, R>> {
    static_assert(N != Dynamic && R != Dynamic);
   private:
    static constexpr int n_monomials = ct_binomial_coefficient(R + N, R);
    std::array<double, n_monomials> coeff_vector_;   // polynomial coefficient vector
    using Base = ScalarExpr<N, Polynomial<N, R>>;
    using VectorType = typename Base::VectorType;
   public:
    // computes the table of monomial exponents (i1, i2, ..., iN)_j for a polynomial of degree R over \mathbb{R}^N
    static consteval std::array<std::array<int, N>, ct_binomial_coefficient(R + N, R)> ct_poly_exp() {
        constexpr int n_monomials = ct_binomial_coefficient(R + N, R);   // number of monomials in polynomial p(x)
        std::array<std::array<int, N>, n_monomials> coeff {};
        std::array<int, N> tmp {};   // fixed j, table row (i1, i2, ..., iN)_j
        int j = 0;
        while (j < n_monomials) {
            // compute exponents vector (i1, i2, ..., iN)_j for monomial j
            int i = 0;
            bool found = false;
            while (i < N && !found) {
                if (tmp[i] <= R && ct_array_sum<int, N>(tmp) <= R) {
                    found = true;
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
    static constexpr std::array<std::array<int, N>, n_monomials> poly_table_ = Polynomial<N, R>::ct_poly_exp();

    // computes the tables of monomial exponents (i1, i2, ..., iN)_j of d/dx_i p(x) for each i = 1, ...,N
    static consteval std::array<std::array<std::array<int, N>, ct_binomial_coefficient(R + N, R)>, N>
    ct_grad_poly_exp(std::array<std::array<int, N>, ct_binomial_coefficient(R + N, R)> poly_table) {
        constexpr int n_monomials = ct_binomial_coefficient(R + N, R);   // number of monomials in polynomial p(x)
        std::array<std::array<std::array<int, N>, n_monomials>, N> coeff {};
        std::array<std::array<int, N>, n_monomials> tmp {};   // coefficient gradient subtable {(i1, i2, ..., iN)_j}_k
        for (int k = 0; k < N; ++k) {
            for (int j = 0; j < n_monomials; ++j) {   // row index
                for (int z = 0; z < N; ++z) {         // column index in subtable
                    tmp[j][z] = (k == z ? (poly_table[j][z] == 0 ? 0 : poly_table[j][z] - 1) : poly_table[j][z]);
                }
            }
            coeff[k] = tmp;
        }
        return coeff;
    }
   private:
    // i-th derivative functor
    class Derivative : public ScalarExpr<N, Derivative> {
       private:
        static constexpr std::array<std::array<std::array<int, N>, n_monomials>, N> poly_grad_table_ =
          Polynomial<N, R>::ct_grad_poly_exp(poly_table_);
        using Base = ScalarExpr<N, Derivative>;
        using VectorType = typename Base::VectorType;
        std::array<double, n_monomials> coeff_vector_;   // polynomial coefficient vector
        int i_;
       public:
        // constructor
        Derivative() = default;
        template <typename CoeffVectorType> Derivative(const CoeffVectorType& coeff_vector, int i) : i_(i) {
            for (int j = 0; j < n_monomials; ++j) { coeff_vector_[j] = coeff_vector[j]; }
        }
        // evaluate d/dx_i p(x) at point
        double operator()(const VectorType& p) const {
            if constexpr (R == 0) return 0;
            if constexpr (R == 1) {   // polynomials of form: c_0 + c_1*x_1 + c_2*x_2 + ... + c_N*x_N
                return coeff_vector_[i_];
            } else {
                double value = 0;
                // cycle over monomials
                for (int m = 0; m < n_monomials; ++m) {
                    int exp = poly_table_[m][i_];
                    if (exp) {   // skip powers of zero, their derivative is zero
                        double tmp = 1;
                        for (int n = 0; n < N; ++n) {
                            int grad_exp = poly_grad_table_[i_][m][n];
                            if (grad_exp) tmp *= std::pow(p[n], grad_exp);
                        }
                        value += coeff_vector_[m] * exp * tmp;
                    }
                }
                return value;   // return partial derivative
            }
        }
    };
    VectorField<N, N, Derivative> gradient_;
   public:
    static constexpr int order = R;   // polynomial order
    using GradientType = VectorField<N, N, Derivative>;
    // constructor
    Polynomial() = default;
    template <typename CoeffVectorType> explicit Polynomial(const CoeffVectorType& coeff_vector) {
        fdapde_assert(int(coeff_vector.size()) == n_monomials);
        for (int i = 0; i < n_monomials; ++i) { coeff_vector_[i] = coeff_vector[i]; }
        for (int i = 0; i < N; ++i) { gradient_[i] = Derivative(coeff_vector, i); }
    }
    // evaluate polynomial at point
    double operator()(const VectorType& p) const {
        double value = coeff_vector_[0];
        if constexpr (R == 0) return value;
        if constexpr (R == 1) {   // polynomials of form: c_0 + c_1*x_1 + c_2*x_2 + ... + c_N*x_N
            for (int n = 0; n < N; ++n) { value += coeff_vector_[n + 1] * p[n]; }
        } else {
            for (int m = 1; m < n_monomials; ++m) {
                double tmp = 1;
                for (int n = 0; n < N; ++n) {
                    int exp = poly_table_[m][n];
                    if (exp) tmp *= std::pow(p[n], exp);
                }
                value += coeff_vector_[m] * tmp;
            }
        }
        return value;
    };
    static Polynomial<N, R> Zero() { return Polynomial<N, R>(DVector<double>::Zero(n_monomials)); }
    DVector<double> operator()(const Eigen::Matrix<double, Eigen::Dynamic, N>& ps) const {
        DVector<double> result(ps.rows());
        for (int i = 0; i < ps.rows(); ++i) result[i] = operator()(ps.row(i));
        return result;
    }
    const VectorField<N, N, Derivative>& derive() const { return gradient_; };
    const std::array<double, n_monomials>& coeff_vector() const { return coeff_vector_; }
    std::array<double, n_monomials>& coeff_vector() { return coeff_vector_; }
    int degree() const { return R; }
};

}   // namespace core
}   // namespace fdapde

#endif   // __POLYNOMIAL_H__
