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

template <typename DomainType, int order> class LagrangianBasis {
   private:
    std::size_t size_ = 0;       // number of basis functions over physical domain
    const DomainType* domain_;   // physical domain of definition

    // A Lagrangian basis of degree R over an M-dimensional element
    template <int M_, int R_> class LagrangianElement {
       public:
        // compile time informations
        static constexpr int R = R_;   // basis order
        static constexpr int M = M_;   // input space dimension
        static constexpr int n_basis = ct_binomial_coefficient(M + R, R);
        typedef MultivariatePolynomial<M, R> ElementType;
        typedef typename std::array<MultivariatePolynomial<M, R>, n_basis>::const_iterator const_iterator;
        typedef Integrator<M, R> Quadrature;   // a quadrature rule which exactly integrates the elements of this basis

        // construct from a given set of nodes
        LagrangianElement(const std::array<std::array<double, M>, n_basis>& nodes) : nodes_(nodes) {
            compute_coefficients_(nodes_);
        };
        // construct over the referece M-dimensional unit simplex
        LagrangianElement() : LagrangianElement<M, R>(ReferenceElement<M, R>::nodes) {};

        // getters
        const MultivariatePolynomial<M, R>& operator[](size_t i) const { return basis_[i]; }
        int size() const { return basis_.size(); }
        const_iterator begin() const { return basis_.cbegin(); }
        const_iterator end() const { return basis_.cend(); }
       private:
        std::array<std::array<double, M_>, n_basis> nodes_;   // nodes of the Lagrangian basis
        std::array<MultivariatePolynomial<M_, R_>, n_basis> basis_;
        // solves the Vandermonde system for the computation of polynomial coefficients
        void compute_coefficients_(const std::array<std::array<double, M_>, n_basis>& nodes) {
            // build vandermonde matrix
            constexpr int n_basis = ct_binomial_coefficient(M_ + R_, R_);
            constexpr std::array<std::array<unsigned, M_>, n_basis> poly_table =
              MultivariatePolynomial<M, R>::poly_table;

            // Vandermonde matrix construction
            SMatrix<n_basis> V = Eigen::Matrix<double, n_basis, n_basis>::Ones();
            for (size_t i = 0; i < n_basis; ++i) {
                for (size_t j = 1; j < n_basis; ++j) {
                    V(i, j) = MonomialProduct<M - 1, std::array<double, M>, std::array<unsigned, M>>::unfold(
                      nodes_[i], poly_table[j]);
                }
            }

            // solve the vandermonde system V*a = b with b vector having 1 at position i and 0 everywhere else.
            Eigen::PartialPivLU<SMatrix<n_basis>> invV(V);
            SVector<n_basis> b = Eigen::Matrix<double, n_basis, 1>::Zero();   // rhs of linear system
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
    };
    LagrangianElement<DomainType::local_dimension, order> ref_basis_ {};
   public:
    static constexpr int R = order;                         // basis order
    static constexpr int M = DomainType::local_dimension;   // input space dimension
    using ReferenceBasis = LagrangianElement<M, R>;
    // constructor
    LagrangianBasis() = default;
    LagrangianBasis(const DomainType& domain, std::size_t size) : domain_(&domain), size_(size) {};

    // returns a pair of matrices (\Psi, D) where: \Psi is the matrix of basis functions evaluations according
    // to the given policy, D is a policy-dependent vector (see the specific policy for details)
    template <template <typename> typename EvaluationPolicy>
    std::pair<SpMatrix<double>, DVector<double>> eval(const DMatrix<double>& locs) const {
        return EvaluationPolicy<LagrangianBasis<DomainType, R>>::eval(*domain_, ref_basis_, locs, size_);
    }
    std::size_t size() const { return size_; }
    static ReferenceBasis ref_basis() { return ReferenceBasis {}; }
    // given a coefficient vector c \in \mathbb{R}^size_, evaluates the corresponding basis expansion at locs
    DVector<double> operator()(const DVector<double>& c, const DMatrix<double>& locs) const {
        fdapde_assert(c.rows() == size_ && locs.cols() == DomainType::embedding_dimension);
        // locate elements
        DVector<int> element_ids = domain_->locate(locs);
        DVector<double> result = DVector<double>::Zero(locs.rows());
        for (std::size_t i = 0; i < locs.rows(); ++i) {
            auto e = domain_->element(element_ids[i]);
            // evaluate basis expansion \sum_{i=1}^size_ c_i \psi_i(x) at p
            for (std::size_t h = 0; h < ref_basis_.size(); ++h) {
                result[i] +=
                  c[e.node_ids()[h]] * ref_basis_[h](e.inv_barycentric_matrix() * (locs.row(i) - e.coords()[0]));
            }
        }
        return result;
    }
};

template <typename T> struct pointwise_evaluation;
template <typename DomainType, int order> struct pointwise_evaluation<LagrangianBasis<DomainType, order>> {
    using BasisType = typename LagrangianBasis<DomainType, order>::ReferenceBasis;
    static constexpr int N = DomainType::embedding_dimension;
    // computes a matrix \Psi such that [\Psi]_{ij} = \psi_j(p_i), D is a vector of ones
    static std::pair<SpMatrix<double>, DVector<double>>
    eval(const DomainType& domain, const BasisType& basis, const DMatrix<double>& locs, std::size_t n_basis) {
        fdapde_assert(locs.size() != 0 && locs.cols() == DomainType::embedding_dimension);
        // preallocate space
        SpMatrix<double> Psi(locs.rows(), n_basis);
        std::vector<fdapde::Triplet<double>> triplet_list;
        triplet_list.reserve(locs.rows() * basis.size());
        // locate points
        DVector<int> element_ids = domain.locate(locs);
        // build \Psi matrix
        for (std::size_t i = 0, n = locs.rows(); i < n; ++i) {
            SVector<N> p_i(locs.row(i));
            auto e = domain.element(element_ids[i]);
            // update \Psi matrix
            for (std::size_t h = 0, n_basis = basis.size(); h < n_basis; ++h) {
                triplet_list.emplace_back(
                  i, e.node_ids()[h],
                  basis[h](e.inv_barycentric_matrix() * (p_i - e.coords()[0])));   // \psi_j(p_i)
            }
        }
        // finalize construction
        Psi.setFromTriplets(triplet_list.begin(), triplet_list.end());
        Psi.makeCompressed();
        return std::pair(std::move(Psi), DVector<double>::Ones(locs.rows()));
    }
};

template <typename T> struct areal_evaluation;
template <typename DomainType, int order> struct areal_evaluation<LagrangianBasis<DomainType, order>> {
    using BasisType = typename LagrangianBasis<DomainType, order>::ReferenceBasis;
    static constexpr int N = DomainType::embedding_dimension;
    // computes a matrix \Psi such that [\Psi]_{ij} = \int_{D_j} \psi_i, D contains the measures of subdomains
    static std::pair<SpMatrix<double>, DVector<double>>
    eval(const DomainType& domain, const BasisType& basis, const DMatrix<double>& locs, std::size_t n_basis) {
        fdapde_assert(locs.size() != 0 && locs.cols() == domain.n_elements());
        typename BasisType::Quadrature integrator {};
        // preallocate space
        SpMatrix<double> Psi(locs.rows(), n_basis);
        std::vector<fdapde::Triplet<double>> triplet_list;
        triplet_list.reserve(locs.rows() * n_basis);

        DVector<double> D(locs.rows());   // measure of subdomains
        // start construction of \Psi matrix
        std::size_t tail = 0;
        for (std::size_t k = 0, n = locs.rows(); k < n; ++k) {
            std::size_t head = 0;
            double Di = 0;   // measure of subdomain D_i
            for (std::size_t l = 0, N_ = locs.cols(); l < N_; ++l) {
                if (locs(k, l) == 1) {   // element with ID l belongs to k-th subdomain
                    // get element with this ID
                    auto e = domain.element(l);
                    // compute \int_e \psi_h \forall \psi_h defined on e
                    for (std::size_t h = 0, n_basis = basis.size(); h < n_basis; ++h) {
                        triplet_list.emplace_back(
                          k, e.node_ids()[h],
                          integrator.integrate(e, [&e, &psi_h = basis[h]](const SVector<N>& q) -> double {
                              return psi_h(e.inv_barycentric_matrix() * (q - e.coords()[0]));
                          }));
                        head++;
                    }
                    Di += e.measure();   // update measure of subdomain D_i
                }
            }
            // divide each \int_{D_i} \psi_j by the measure of subdomain D_i
            for (std::size_t j = 0; j < head; ++j) { triplet_list[tail + j].value() /= Di; }
            D[k] = Di;   // store measure of subdomain
            tail += head;
        }
        // finalize construction
        Psi.setFromTriplets(triplet_list.begin(), triplet_list.end());
        Psi.makeCompressed();
        return std::make_pair(std::move(Psi), std::move(D));
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __LAGRANGIAN_BASIS_H__
