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

#include "../linear_algebra/constexpr_matrix.h"
#include "../utils/constexpr.h"
#include "scalar_field.h"

namespace fdapde {

// multivariate polynomial of degree R over \mathbb{R}^N
template <int StaticInputSize_, int Order_>
class Polynomial : public ScalarBase<StaticInputSize_, Polynomial<StaticInputSize_, Order_>> {
    fdapde_static_assert(
      StaticInputSize_ != Dynamic && Order_ != Dynamic,
      POLYNOMIAL_CANNOT_HAVE_DYNAMIC_STATIC_INPUT_SIZE_NOR_DYNAMIC_ORDER);
    static constexpr int n_monomials = cexpr::binomial_coefficient(StaticInputSize_ + Order_, Order_);
   public:
    using Base = ScalarBase<StaticInputSize_, Polynomial<StaticInputSize_, Order_>>;
    using Scalar = double;
    using InputType = cexpr::Vector<Scalar, StaticInputSize_>;
    static constexpr int StaticInputSize = StaticInputSize_;
    static constexpr int NestAsRef = 0;
    static constexpr int XprBits = 0;
    static constexpr int Order = Order_;

    // computes the table of monomial exponents (i1, i2, ..., iN)_j for a polynomial of degree Order over
    // \mathbb{R}^StaticInnerSize
    static constexpr cexpr::Matrix<int, n_monomials, StaticInputSize> poly_exponents() {
        constexpr int n_monomials = cexpr::binomial_coefficient(StaticInputSize + Order, Order);
        cexpr::Matrix<int, n_monomials, StaticInputSize> coeff {};
        std::array<int, StaticInputSize> tmp {};   // fixed j, table row (i1, i2, ..., iN)_j
        int j = 0;
        while (j < n_monomials) {
            // compute exponents vector (i1, i2, ..., iN)_j for monomial j
            int i = 0;
            bool found = false;
            while (i < StaticInputSize && !found) {
                if (tmp[i] <= Order && cexpr::array_sum<int, StaticInputSize>(tmp) <= Order) {
                    found = true;
                    for (int k = 0; k < StaticInputSize; ++k) coeff(j, k) = tmp[k];
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
    static constexpr cexpr::Matrix<int, n_monomials, StaticInputSize> poly_table_ =
      Polynomial<StaticInputSize, Order>::poly_exponents();

    // computes the tables of monomial exponents (i1, i2, ..., iN)_j of d/dx_i p(x) for each i = 1, ...,N
    static constexpr std::array<cexpr::Matrix<int, n_monomials, StaticInputSize>, StaticInputSize>
    grad_poly_exponents(cexpr::Matrix<int, n_monomials, StaticInputSize> poly_table) {
        constexpr int n_monomials = cexpr::binomial_coefficient(StaticInputSize + Order, Order);
        std::array<cexpr::Matrix<int, n_monomials, StaticInputSize>, StaticInputSize> coeff {};
        cexpr::Matrix<int, n_monomials, StaticInputSize> tmp {};   // gradient subtable {(i1, i2, ..., iN)_j}_k
        for (int k = 0; k < StaticInputSize; ++k) {
            for (int j = 0; j < n_monomials; ++j) {           // row index
                for (int z = 0; z < StaticInputSize; ++z) {   // column index in subtable
                    tmp(j, z) = (k == z ? (poly_table(j, z) == 0 ? 0 : poly_table(j, z) - 1) : poly_table(j, z));
                }
            }
            coeff[k] = tmp;
        }
        return coeff;
    }
   private:
    // i-th derivative functor
    class Derivative : public ScalarBase<StaticInputSize, Derivative> {
       private:
        static constexpr std::array<cexpr::Matrix<int, n_monomials, StaticInputSize>, StaticInputSize>
          poly_grad_table_ = Polynomial<StaticInputSize, Order>::grad_poly_exponents(poly_table_);
        std::array<double, n_monomials> coeff_vector_;   // polynomial coefficient vector
        int i_;
       public:
        using Base = ScalarBase<StaticInputSize, Derivative>;
        using Scalar = double;
        using InputType = cexpr::Vector<Scalar, StaticInputSize_>;
      
        // constructor
        constexpr Derivative() = default;
        template <typename CoeffVectorType> constexpr Derivative(const CoeffVectorType& coeff_vector, int i) : i_(i) {
            for (int j = 0; j < n_monomials; ++j) { coeff_vector_[j] = coeff_vector[j]; }
        }
        // evaluate d/dx_i p(x) at point
        constexpr Scalar operator()(const InputType& p) const {
            if constexpr (Order == 0) return 0;
            if constexpr (Order == 1) {   // polynomials of form: c_0 + c_1*x_1 + c_2*x_2 + ... + c_N*x_N
                return coeff_vector_[i_ + 1];
            } else {
                Scalar value = 0;
                // cycle over monomials
                for (int m = 0; m < n_monomials; ++m) {
                    int exp = poly_table_(m, i_);
                    if (exp) {   // skip powers of zero, their derivative is zero
                        Scalar tmp = 1;
                        for (int n = 0; n < StaticInputSize; ++n) {
                            int grad_exp = poly_grad_table_[i_](m, n);
                            if (grad_exp) tmp *= std::pow(p[n], grad_exp);
                        }
                        value += coeff_vector_[m] * exp * tmp;
                    }
                }
                return value;   // return partial derivative
            }
        }
    };
    std::array<double, n_monomials> coeff_vector_;   // polynomial coefficient vector
    VectorField<StaticInputSize, StaticInputSize, Derivative> gradient_;
   public:
    // constructor
    constexpr Polynomial() = default;
    template <typename CoeffVectorType>
    explicit constexpr Polynomial(const CoeffVectorType& coeff_vector) : coeff_vector_(), gradient_() {
        fdapde_static_assert(int(coeff_vector.size()) == n_monomials, INVALID_COEFFICIENT_VECTOR);
        for (int i = 0; i < n_monomials; ++i) { coeff_vector_[i] = coeff_vector[i]; }
        for (int i = 0; i < StaticInputSize; ++i) { gradient_[i] = Derivative(coeff_vector_, i); }
    }
    // evaluate polynomial at point
    template <typename InputType_> constexpr Scalar operator()(const InputType_& p) const {
        Scalar value = coeff_vector_[0];
        if constexpr (Order == 0) return value;
        if constexpr (Order == 1) {   // polynomials of form: c_0 + c_1*x_1 + c_2*x_2 + ... + c_N*x_N
            for (int n = 0; n < StaticInputSize; ++n) { value += coeff_vector_[n + 1] * p[n]; }
        } else {
            for (int m = 1; m < n_monomials; ++m) {
                Scalar tmp = 1;
                for (int n = 0; n < StaticInputSize; ++n) {
                    int exp = poly_table_(m, n);
                    if (exp) tmp *= std::pow(p[n], exp);
                }
                value += coeff_vector_[m] * tmp;
            }
        }
        return value;
    };
    constexpr static Polynomial<StaticInputSize, Order> Zero() {
        return Polynomial<StaticInputSize, Order>(cexpr::Vector<double, n_monomials>::Zero());
    }
    constexpr const VectorField<StaticInputSize, StaticInputSize, Derivative>& gradient() const { return gradient_; };
    constexpr const std::array<double, n_monomials>& coeff_vector() const { return coeff_vector_; }
    constexpr std::array<double, n_monomials>& coeff_vector() { return coeff_vector_; }
    constexpr int order() const { return Order; }
    constexpr int input_size() const { return StaticInputSize; }
};

template <int StaticInputSize, int Order> constexpr auto grad(const Polynomial<StaticInputSize, Order>& poly) {
    return poly.gradient();
}

}   // namespace fdapde

#endif   // __POLYNOMIAL_H__
