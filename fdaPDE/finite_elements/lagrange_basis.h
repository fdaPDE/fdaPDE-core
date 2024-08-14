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

#ifndef __LAGRANGE_BASIS_H__
#define __LAGRANGE_BASIS_H__

#include "../fields/polynomial.h"
#include "../linear_algebra/constexpr_matrix.h"
#include "../utils/symbols.h"

namespace fdapde {

// given N+1 nodes n_0, n_1, n_2, ..., n_N, this class represents the set {l_0(x), l_1(x), ..., l_N(x) : l_j(n_i) = 1
// \iff i = j, 0 otherwise}. this polynomial system makes a basis for the set of polynomials up to order R in R^N
template <int StaticInputSize_, int Order_> class LagrangeBasis {
   public:
    static constexpr int n_basis = cexpr::binomial_coefficient(StaticInputSize_ + Order_, Order_);
    static constexpr int StaticInputSize = StaticInputSize_;
    static constexpr int Order = Order_;
    using PolynomialType = Polynomial<StaticInputSize, Order>;
    // constructors
    constexpr LagrangeBasis() = default;
    template <int n_nodes>
    constexpr explicit LagrangeBasis(const cexpr::Matrix<double, n_nodes, StaticInputSize>& nodes) {
        fdapde_static_assert(n_nodes == n_basis, WRONG_NUMBER_OF_NODES_FOR_DEFINITION_OF_LAGRANGE_BASIS);
        // build Vandermonde matrix
        constexpr auto poly_table = Polynomial<StaticInputSize, Order>::poly_exponents();
	cexpr::Matrix<double, n_basis, n_basis> V = cexpr::Matrix<double, n_basis, n_basis>::Ones();
        for (int i = 0; i < n_basis; ++i) {
            for (int j = 1; j < n_basis; ++j) {
                for (int k = 0; k < StaticInputSize; ++k) {
                    int exp = poly_table(j, k);
                    if (exp) V(i, j) *= std::pow(nodes(i, k), exp);
                }
            }
        }
        // solve the vandermonde system V*a = b_i with b_i[j] = 1 \iff j = i, 0 otherwise
        cexpr::PartialPivLU<cexpr::Matrix<double, n_basis, n_basis>> invV(V);
	cexpr::Vector<double, n_basis> b = cexpr::Matrix<double, n_basis, 1>::Zero();   // rhs of linear system
        for (int i = 0; i < n_basis; ++i) {
            b[i] = 1;
            // solve linear system V*a = b
	    cexpr::Vector<double, n_basis> a = invV.solve(b);
            basis_[i] = Polynomial<StaticInputSize, Order>(a);
            b[i] = 0;
        }
    }
    // getters
    constexpr const Polynomial<StaticInputSize, Order>& operator[](int i) const { return basis_[i]; }
    constexpr int size() const { return basis_.size(); }
   private:
    std::array<Polynomial<StaticInputSize_, Order_>, n_basis> basis_;
};

}   // namespace fdapde

#endif   // __LAGRANGE_BASIS_H__
