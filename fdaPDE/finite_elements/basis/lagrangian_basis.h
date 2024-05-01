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

#include <unordered_set>

#include "../../pde/symbols.h"
#include "../../utils/compile_time.h"
#include "../../utils/symbols.h"
#include "multivariate_polynomial.h"
#include "reference_element.h"

namespace fdapde {
namespace core {

template <typename MeshType, int Order> class LagrangianBasis {
   private:
    int size_ = 0;                          // number of basis functions over physical domain
    DMatrix<int> dofs_;                     // for each element, the degrees of freedom associated to it
    BinaryVector<Dynamic> boundary_dofs_;   // unknowns on the boundary of the domain
    const MeshType* domain_;                // physical domain of definition

    // A Lagrangian basis of degree R over an M-dimensional element
    template <int M_, int R_> class LagrangianElement {
       public:
        // compile time informations
        static constexpr int R = R_;   // basis order
        static constexpr int M = M_;   // input space dimension
        static constexpr int n_basis = ct_binomial_coefficient(M + R, R);
        typedef MultivariatePolynomial<M, R> ElementType;
        typedef typename std::array<MultivariatePolynomial<M, R>, n_basis>::const_iterator const_iterator;
        typedef Integrator<FEM, M, R> Quadrature;

        // construct from a given set of nodes
        LagrangianElement(const std::array<std::array<double, M>, n_basis>& nodes) : nodes_(nodes) {
            compute_coefficients_(nodes_);
        };
        // construct over the referece M-dimensional unit simplex
        LagrangianElement() : LagrangianElement<M, R>(ReferenceElement<M, R>::nodes) {};

        // getters
        const MultivariatePolynomial<M, R>& operator[](int i) const { return basis_[i]; }
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
            constexpr std::array<std::array<int, M_>, n_basis> poly_table = MultivariatePolynomial<M, R>::poly_table;
            // Vandermonde matrix construction
            SMatrix<n_basis> V = Eigen::Matrix<double, n_basis, n_basis>::Ones();
            for (int i = 0; i < n_basis; ++i) {
                for (int j = 1; j < n_basis; ++j) {
                    V(i, j) = MonomialProduct<M - 1, std::array<double, M>, std::array<int, M>>::unfold(
                      nodes[i], poly_table[j]);
                }
            }
            // solve the vandermonde system V*a = b with b vector having 1 at position i and 0 everywhere else.
            Eigen::PartialPivLU<SMatrix<n_basis>> invV(V);
            SVector<n_basis> b = Eigen::Matrix<double, n_basis, 1>::Zero();   // rhs of linear system
            for (int i = 0; i < n_basis; ++i) {
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
    LagrangianElement<MeshType::local_dim, Order> ref_basis_ {};
    void enumerate_dofs() requires(Order <= 2) {       // produce the matrix of dof coordinates
        if (size_ != 0) return;   // return early if dofs already computed
        if constexpr (Order == 1) {
            size_ = domain_->n_nodes();
            dofs_ = domain_->cells();
            boundary_dofs_ = domain_->boundary_nodes();
        } else {
            dofs_.resize(domain_->n_cells(), this->n_dof_per_element);
            dofs_.leftCols(MeshType::n_nodes_per_cell) =
              domain_->cells();   // copy dofs associated to geometric vertices

            int next = domain_->n_nodes();   // next valid ID to assign
            auto edge_pattern = combinations<MeshType::n_nodes_per_edge, MeshType::n_nodes_per_cell>();
            std::unordered_set<int> boundary_set;
            // cycle over mesh edges
            std::array<int, MeshType::n_nodes_per_edge> e {};
            for (typename MeshType::edge_iterator edge = domain_->edges_begin(); edge != domain_->edges_end(); ++edge) {
                for (int i = 0; i < 2; ++i) {
                    int id = edge->adjacent_cells()[i];
                    if (id >= 0) {
                        // search for dof insertion point
                        int j = 0;
                        for (; j < MeshType::n_edges_per_cell; ++j) {
                            for (int k = 0; k < MeshType::n_nodes_per_edge; ++k) {
                                e[k] = domain_->cells()(id, edge_pattern(j, k));
                            }
			    std::sort(e.begin(), e.end());
                            if (edge->node_ids()[0] == e[0] && edge->node_ids()[1] == e[1]) break;
                        }
                        dofs_(id, MeshType::n_nodes_per_cell + j) = next;
                        if (edge->on_boundary()) boundary_set.insert(next);
                    }
                }
                next++;
            }
            size_ = next;   // store number of unknowns
            // update boundary
            boundary_dofs_.resize(size_);
            boundary_dofs_.topRows(domain_->n_nodes()) = domain_->boundary_nodes();
            for (auto it = boundary_set.begin(); it != boundary_set.end(); ++it) { boundary_dofs_.set(*it); }
        }
        return;
    }
   public:
    static constexpr int R = Order;                 // basis order
    static constexpr int M = MeshType::local_dim;   // input space dimension
    static constexpr int n_dof_per_element = ct_nnodes(MeshType::local_dim, Order);
    static constexpr int n_dof_per_edge = Order - 1;
    static constexpr int n_dof_internal =
      n_dof_per_element - (MeshType::local_dim + 1) - MeshType::n_facets_per_element * (Order - 1);
    using ReferenceBasis = LagrangianElement<M, R>;
    // constructor
    LagrangianBasis() = default;
    LagrangianBasis(const MeshType& domain) : domain_(&domain) { enumerate_dofs(); };

    // returns a pair of matrices (\Psi, D) where: \Psi is the matrix of basis functions evaluations according
    // to the given policy, D is a policy-dependent vector (see the specific policy for details)
    template <template <typename> typename EvaluationPolicy>
    std::pair<SpMatrix<double>, DVector<double>> eval(const DMatrix<double>& locs) const {
        return EvaluationPolicy<LagrangianBasis<MeshType, R>>::eval(*domain_, ref_basis_, locs, size_, dofs_);
    }
    int size() const { return size_; }
    //getters
    DMatrix<int> dofs() const { return dofs_; }
    const BinaryVector<Dynamic>& boundary_dofs() const { return boundary_dofs_; }
    DMatrix<double> dofs_coords() {   // computes the physical coordinates of the degrees of freedom
        if constexpr (Order == 1)
            return domain_->nodes();   // for order 1 dofs coincide with mesh vertices
        else {
            // allocate space
            DMatrix<double> coords;
            coords.resize(size_, MeshType::embed_dim);
            coords.topRows(domain_->n_nodes()) = domain_->nodes();   // copy coordinates of elements' vertices
            std::unordered_set<int> visited;                         // set of already visited dofs
            std::array<SVector<MeshType::local_dim + 1>, n_dof_per_element> ref_coords =
              ReferenceElement<MeshType::local_dim, R>().bary_coords;
            // cycle over all mesh elements
            for (typename MeshType::cell_iterator e = domain_->cells_begin(); e != domain_->cells_end(); ++e) {
                auto dofs = dofs_.row(e->id());
                for (int j = MeshType::n_nodes_per_cell; j < n_dof_per_element; ++j) {
                    if (visited.find(dofs[j]) == visited.end()) {
                        // map point from reference to physical element
                        coords.row(dofs[j]) = e->J() * ref_coords[j].template tail<MeshType::local_dim>() + e->node(0);
                        visited.insert(dofs[j]);
                    }
                }
            }
            return coords;
        }
    }
    static ReferenceBasis ref_basis() { return ReferenceBasis {}; }
    // given a coefficient vector c \in \mathbb{R}^size_, evaluates the corresponding basis expansion at locs
    DVector<double> operator()(const DVector<double>& c, const DMatrix<double>& locs) const {
        fdapde_assert(c.rows() == size_ && locs.cols() == MeshType::embed_dim);
        // locate elements
        DVector<int> element_ids = domain_->locate(locs);
        DVector<double> result = DVector<double>::Zero(locs.rows());
        for (int i = 0; i < locs.rows(); ++i) {
            auto e = domain_->cell(element_ids[i]);
            // evaluate basis expansion \sum_{i=1}^size_ c_i \psi_i(x) at p
            SVector<MeshType::embed_dim> p(locs.row(i));
            for (int h = 0; h < ref_basis_.size(); ++h) {
                result[i] += c[e.node_ids()[h]] * ref_basis_[h](e.invJ() * (p - e.node(0)));
            }
        }
        return result;
    }
};

template <typename MeshType, int order> struct pointwise_evaluation<LagrangianBasis<MeshType, order>> {
    using BasisType = typename LagrangianBasis<MeshType, order>::ReferenceBasis;
    static constexpr int N = MeshType::embed_dim;
    // computes a matrix \Psi such that [\Psi]_{ij} = \psi_j(p_i), D is a vector of ones
    static std::pair<SpMatrix<double>, DVector<double>> eval(
      const MeshType& domain, const BasisType& basis, const DMatrix<double>& locs, int n_basis,
      const DMatrix<int>& dofs) {
      fdapde_assert(locs.size() != 0 && locs.cols() == MeshType::embed_dim);
      // preallocate space
      SpMatrix<double> Psi(locs.rows(), n_basis);
      std::vector<fdapde::Triplet<double>> triplet_list;
      triplet_list.reserve(locs.rows() * basis.size());
      // locate points
      DVector<int> element_ids = domain.locate(locs);
      // build \Psi matrix
      for (int i = 0, n = locs.rows(); i < n; ++i) {
          SVector<N> p_i(locs.row(i));
          if (element_ids[i] != -1) {   // point falls inside domain
              auto e = domain.cell(element_ids[i]);
              // update \Psi matrix
              for (int h = 0, n_basis = basis.size(); h < n_basis; ++h) {
                  triplet_list.emplace_back(
                    i, dofs(e.id(), h),
                    basis[h](e.invJ() * (p_i - e.node(0))));   // \psi_j(p_i)
              }
          }
      }
      // finalize construction
      Psi.setFromTriplets(triplet_list.begin(), triplet_list.end());
      Psi.makeCompressed();
      return std::pair(std::move(Psi), DVector<double>::Ones(locs.rows()));
    }
};

template <typename MeshType, int order> struct areal_evaluation<LagrangianBasis<MeshType, order>> {
    using BasisType = typename LagrangianBasis<MeshType, order>::ReferenceBasis;
    static constexpr int N = MeshType::embed_dim;
    // computes a matrix \Psi such that [\Psi]_{ij} = \int_{D_j} \psi_i, D contains the measures of subdomains
    static std::pair<SpMatrix<double>, DVector<double>> eval(
      const MeshType& domain, const BasisType& basis, const DMatrix<double>& locs, int n_basis,
      const DMatrix<int>& dofs) {
      fdapde_assert(locs.size() != 0 && locs.cols() == domain.n_cells());
      typename BasisType::Quadrature integrator {};
      // preallocate space
      SpMatrix<double> Psi(locs.rows(), n_basis);
      std::vector<fdapde::Triplet<double>> triplet_list;
      triplet_list.reserve(locs.rows() * n_basis);

      DVector<double> D(locs.rows());   // measure of subdomains
      // start construction of \Psi matrix
      int tail = 0;
      for (int k = 0, n = locs.rows(); k < n; ++k) {
            int head = 0;
            double Di = 0;   // measure of subdomain D_i
            for (int l = 0, N_ = locs.cols(); l < N_; ++l) {
                if (locs(k, l) == 1) {   // element with ID l belongs to k-th subdomain
                    // get element with this ID
                    auto e = domain.cell(l);
                    // compute \int_e \psi_h \forall \psi_h defined on e
                    for (int h = 0, n_basis = basis.size(); h < n_basis; ++h) {
                        triplet_list.emplace_back(
                          k, dofs(e.id(), h),
                          integrator.integrate_cell(e, [&e, &psi_h = basis[h]](const SVector<N>& q) -> double {
                              return psi_h(e.invJ() * (q - e.node(0)));
                          }));
                        head++;
                    }
                    Di += e.measure();   // update measure of subdomain D_i
                }
            }
            // divide each \int_{D_i} \psi_j by the measure of subdomain D_i
            for (int j = 0; j < head; ++j) { triplet_list[tail + j].value() /= Di; }
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
