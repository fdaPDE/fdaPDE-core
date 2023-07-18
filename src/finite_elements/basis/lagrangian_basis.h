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

#ifndef __LAGRANGIAN_BASIS_H__
#define __LAGRANGIAN_BASIS_H__

#include "../../mesh/element.h"
#include "../../mesh/reference_element.h"
#include "../../utils/compile_time.h"
#include "../../utils/symbols.h"
#include "multivariate_polynomial.h"

namespace fdapde {
namespace core {

// A Lagrangian basis of degree R over an M-dimensional space
template <unsigned int M, unsigned int R> class LagrangianBasis {
   private:
    std::array<std::array<double, M>, ct_binomial_coefficient(M + R, R)> nodes_;   // nodes of the Lagrangian basis
    std::array<MultivariatePolynomial<M, R>, ct_binomial_coefficient(M + R, R)> basis_;
    // solves the Vandermonde system for the computation of polynomial coefficients
    void compute_coefficients_(const std::array<std::array<double, M>, ct_binomial_coefficient(M + R, R)>& nodes);
   public:
    // expose basis order
    static constexpr unsigned int order = R;
    typedef MultivariatePolynomial<M, R> ElementType;
    typedef typename std::array<MultivariatePolynomial<M, R>, ct_binomial_coefficient(M + R, R)>::const_iterator
      const_iterator;

    // construct from a given set of nodes
    LagrangianBasis(const std::array<std::array<double, M>, ct_binomial_coefficient(M + R, R)>& nodes)
      : nodes_(nodes) {
        compute_coefficients_(nodes_);
    };
    // construct over the referece M-dimensional unit simplex
    LagrangianBasis() : LagrangianBasis<M, R>(ReferenceElement<M, R>::nodes) {};

    // getters
    const MultivariatePolynomial<M, R>& operator[](size_t i) const { return basis_[i]; }
    int size() const { return basis_.size(); }
    const_iterator begin() const { return basis_.cbegin(); }
    const_iterator end() const { return basis_.cend(); }
};

// compute coefficients via Vandermonde matrix
template <unsigned int M, unsigned int R>
void LagrangianBasis<M, R>::compute_coefficients_(
  const std::array<std::array<double, M>, ct_binomial_coefficient(M + R, R)>& nodes) {
    // build vandermonde matrix
    constexpr unsigned int n_basis = ct_binomial_coefficient(M + R, R);
    constexpr std::array<std::array<unsigned, M>, n_basis> poly_table = MultivariatePolynomial<M, R>::poly_table;

    // Vandermonde matrix construction
    SMatrix<n_basis> V = Eigen::Matrix<double, n_basis, n_basis>::Ones();
    for (size_t i = 0; i < n_basis; ++i) {
        for (size_t j = 1; j < n_basis; ++j) {
            V(i, j) =
              MonomialProduct<M - 1, std::array<double, M>, std::array<unsigned, M>>::unfold(nodes_[i], poly_table[j]);
        }
    }

    // solve the vandermonde system V*a = b with b vector having 1 at position i and 0 everywhere else.
    Eigen::PartialPivLU<SMatrix<n_basis>> invV(V);
    SVector<n_basis> b = Eigen::Matrix<double, n_basis, 1>::Zero(); // rhs of linear system
    for (size_t i = 0; i < n_basis; ++i) {
        b[i] = 1;
	// solve linear system V*a = b
        SVector<n_basis> a = invV.solve(b);
	// store basis
        std::array<double, n_basis> coeff;
        std::copy(a.data(), a.data() + n_basis, coeff.begin());
        basis_[i] = MultivariatePolynomial<M, R>(coeff);
	// restore rhs b
        b[i] = 0;
    }
}

}   // namespace core
}   // namespace fdapde

#endif   // __LAGRANGIAN_BASIS_H__
