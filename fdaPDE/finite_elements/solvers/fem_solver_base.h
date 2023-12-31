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

#ifndef __FEM_SOLVER_BASE_H__
#define __FEM_SOLVER_BASE_H__

#include <exception>

#include "../../utils/integration/integrator.h"
#include "../../utils/symbols.h"
#include "../../utils/traits.h"
#include "../../utils/combinatorics.h"
#include "../basis/lagrangian_basis.h"
#include "../fem_assembler.h"
#include "../fem_symbols.h"
#include "../operators/reaction.h"   // for mass-matrix computation
#include "../../pde/symbols.h"

namespace fdapde {
namespace core {
  
// base class for the definition of a general solver based on the Finite Element Method
template <typename D, typename E, typename F, typename... Ts> class FEMSolverBase {
   public:
    typedef std::tuple<Ts...> SolverArgs;
    enum {
        fem_order = std::tuple_element<0, SolverArgs>::type::value,
        n_dof_per_element = ct_nnodes(D::local_dimension, fem_order),
        n_dof_per_edge = fem_order - 1,
        n_dof_internal =
          n_dof_per_element - (D::local_dimension + 1) - D::n_facets_per_element * (fem_order - 1)   // > 0 \iff R > 2
    };
    using DomainType = D;
    using FunctionalBasis = LagrangianBasis<DomainType, fem_order>;
    using ReferenceBasis = typename FunctionalBasis::ReferenceBasis;
    using Quadrature = typename ReferenceBasis::Quadrature;
    // constructor
    FEMSolverBase() = default;
    FEMSolverBase(const DomainType& domain) : basis_(domain) {}
    // getters
    const DMatrix<double>& solution() const { return solution_; }
    const DMatrix<double>& force() const { return force_; }
    const SpMatrix<double>& stiff() const { return stiff_; }
    const SpMatrix<double>& mass() const { return mass_; }
    const Quadrature& integrator() const { return integrator_; }
    const ReferenceBasis& reference_basis() const { return reference_basis_; }
    const FunctionalBasis& basis() const { return basis_; }
    std::size_t n_dofs() const { return n_dofs_; }   // number of degrees of freedom (FEM linear system's unknowns)
    const DMatrix<int>& dofs() const { return dofs_; }
    DMatrix<double> dofs_coords() { return basis_.dofs_coords(); };   // computes the physical coordinates of dofs
    // flags
    bool is_init = false;   // notified true if initialization occurred with no errors
    bool success = false;   // notified true if problem solved with no errors

    template <typename PDE> void init(const PDE& pde);
    template <typename PDE> void set_dirichlet_bc(const PDE& pde);
    
    struct boundary_dofs_iterator {   // range-for loop over boundary dofs
       private:
        friend FEMSolverBase;
        const FEMSolverBase* fem_solver_;
        int index_;   // current boundary dof
        boundary_dofs_iterator(const FEMSolverBase* fem_solver, int index) : fem_solver_(fem_solver), index_(index) {};
       public:
        // fetch next boundary dof
        boundary_dofs_iterator& operator++() {
            index_++;
            // scan until all nodes have been visited or a boundary node is not found
            for (; index_ < fem_solver_->n_dofs_ && fem_solver_->boundary_dofs_(index_,0) == 0; ++index_);
            return *this;
        }
        int operator*() const { return index_; }
        friend bool operator!=(const boundary_dofs_iterator& lhs, const boundary_dofs_iterator& rhs) {
            return lhs.index_ != rhs.index_;
        }
    };
    boundary_dofs_iterator boundary_dofs_begin() const { return boundary_dofs_iterator(this, 0); }
    boundary_dofs_iterator boundary_dofs_end() const { return boundary_dofs_iterator(this, n_dofs_); }
   protected:
    Quadrature integrator_ {};            // default to a quadrature rule which is exact for the considered FEM order
    FunctionalBasis basis_ {};            // basis system defined over the pyhisical domain
    ReferenceBasis reference_basis_ {};   // function basis on the reference unit simplex
    DMatrix<double> solution_;            // vector of coefficients of the approximate solution
    DMatrix<double> force_;               // discretized force [u]_i = \int_D f*\psi_i
    SpMatrix<double> stiff_;              // [stiff_]_{ij} = a(\psi_i, \psi_j), being a(.,.) the bilinear form
    SpMatrix<double> mass_;               // mass matrix, [mass_]_{ij} = \int_D (\psi_i * \psi_j)

    std::size_t n_dofs_ = 0;        // degrees of freedom, i.e. the maximum ID in the dof_table_
    DMatrix<int> dofs_;             // for each element, the degrees of freedom associated to it
    DMatrix<int> boundary_dofs_;    // unknowns on the boundary of the domain, for boundary conditions prescription
};

// implementative details

// initialize solver
template <typename D, typename E, typename F, typename... Ts>
template <typename PDE>
void FEMSolverBase<D, E, F, Ts...>::init(const PDE& pde) {
    fdapde_static_assert(is_pde<PDE>::value, THIS_METHOD_IS_FOR_PDE_ONLY);
    n_dofs_ = basis_.size();
    dofs_ = basis_.dofs();
    boundary_dofs_ = basis_.boundary_dofs();
    // assemble discretization matrix for given operator
    Assembler<FEM, DomainType, ReferenceBasis, Quadrature> assembler(pde.domain(), integrator_, n_dofs_, dofs_);
    stiff_ = assembler.discretize_operator(pde.differential_operator());
    stiff_.makeCompressed();
    // assemble forcing vector
    std::size_t n = n_dofs_;   // degrees of freedom in space
    std::size_t m;             // number of time points
    if constexpr (!std::is_base_of<ScalarBase, F>::value) {
        m = pde.forcing_data().cols();
        force_.resize(n * m, 1);
        force_.block(0, 0, n, 1) = assembler.discretize_forcing(pde.forcing_data().col(0));

        // iterate over time steps if a space-time PDE is supplied
        if constexpr (is_parabolic<E>::value) {
            for (std::size_t i = 1; i < m; ++i) {
                force_.block(n * i, 0, n, 1) = assembler.discretize_forcing(pde.forcing_data().col(i));
            }
        }
    } else {
        // TODO: support space-time callable forcing for parabolic problems
        m = 1;
        force_.resize(n * m, 1);
        force_.block(0, 0, n, 1) = assembler.discretize_forcing(pde.forcing_data());
    }
    // compute mass matrix [mass]_{ij} = \int_{\Omega} \phi_i \phi_j
    mass_ = assembler.discretize_operator(Reaction<FEM, double>(1.0));
    is_init = true;
    return;
}

// impose dirichlet boundary conditions
template <typename D, typename E, typename F, typename... Ts>
template <typename PDE>
void FEMSolverBase<D, E, F, Ts...>::set_dirichlet_bc(const PDE& pde) {
    fdapde_static_assert(is_pde<PDE>::value, THIS_METHOD_IS_FOR_PDE_ONLY);
    if (!is_init) throw std::runtime_error("solver must be initialized first!");
    for (auto it = boundary_dofs_begin(); it != boundary_dofs_end(); ++it) {
        stiff_.row(*it) *= 0;            // zero all entries of this row
        stiff_.coeffRef(*it, *it) = 1;   // set diagonal element to 1 to impose equation u_j = b_j

        // TODO: currently only space-only case supported (reason of [0] below)
        force_.coeffRef(*it, 0) = pde.boundary_data()(*it, 0);   // impose boundary value on forcing term
    }
    return;
}

}   // namespace core
}   // namespace fdapde

#endif   // __FEM_SOLVER_BASE_H__
