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

#include "../utils/compile_time.h"
#include "../utils/symbols.h"
#include "../fields/polynomial.h"

namespace fdapde {
namespace core {

// given N+1 nodes n_0, n_1, n_2, ..., n_N, this class represents the set {l_0(x), l_1(x), ..., l_N(x) : l_j(n_i) = 1
// \iff i = j, 0 otherwise}. this polynomial system makes a basis for the set of polynomials up to order R in R^N
template <int N, int R> class LagrangeBasis {
   private:
    std::array<Polynomial<N, R>, ct_binomial_coefficient(N + R, R)> basis_;
    Eigen::Matrix<double, Eigen::Dynamic, N> nodes_;
   public:
    using PolynomialType = Polynomial<N, R>;
    static constexpr int n_basis = ct_binomial_coefficient(N + R, R);
    static constexpr int order = R;
    // constructors
    LagrangeBasis() = default;
    explicit LagrangeBasis(const Eigen::Matrix<double, Eigen::Dynamic, N>& nodes) : nodes_(nodes) {
        fdapde_assert(nodes.rows() == ct_binomial_coefficient(N + R, R));
        // build Vandermonde matrix
        constexpr auto poly_table = Polynomial<N, R>::ct_poly_exp();
        SMatrix<n_basis> V = Eigen::Matrix<double, n_basis, n_basis>::Ones();
        for (int i = 0; i < n_basis; ++i) {
            for (int j = 1; j < n_basis; ++j) {
                for (int k = 0; k < N; ++k) {
                    int exp = poly_table[j][k];
                    if (exp) V(i, j) *= std::pow(nodes(i, k), exp);
                }
            }
        }
        // solve the vandermonde system V*a = b_i with b_i[j] = 1 \iff j = i, 0 otherwise
        Eigen::PartialPivLU<SMatrix<n_basis>> invV(V);
        SVector<n_basis> b = Eigen::Matrix<double, n_basis, 1>::Zero();   // rhs of linear system
        for (int i = 0; i < n_basis; ++i) {
            b[i] = 1;
            // solve linear system V*a = b
            SVector<n_basis> a = invV.solve(b);
            basis_[i] = Polynomial<N, R>(a);
            b[i] = 0;
        }
    }
    // getters
    const Polynomial<N, R>& operator[](int i) const { return basis_[i]; }   // i-th lagrange polynomial
    const Eigen::Matrix<double, Eigen::Dynamic, N>& nodes() const { return nodes_; }
    int size() const { return basis_.size(); }
};

}   // namespace core
}   // namespace fdapde

#endif   // __LAGRANGE_BASIS_H__
