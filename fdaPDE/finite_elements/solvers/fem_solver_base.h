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
#include "../basis/finite_element_basis.h"
#include "../basis/lagrangian_element.h"
#include "../fem_assembler.h"
#include "../fem_symbols.h"
#include "../operators/reaction.h"   // for mass-matrix computation

namespace fdapde {
namespace core {

// forward declaration
template <typename PDE> struct is_pde;
  
// base class for the definition of a general solver based on the Finite Element Method
template <typename D, typename E, typename F> class FEMSolverBase {
   public:
    typedef Integrator<D::local_dimension, D::order> QuadratureRule;
    typedef LagrangianElement<D::local_dimension, D::order> FunctionSpace;
    typedef FiniteElementBasis<FunctionSpace> FunctionBasis;

    // constructor
    FEMSolverBase() = default;

    // getters
    const DMatrix<double>& solution() const { return solution_; }
    const DMatrix<double>& force() const { return force_; }
    const SpMatrix<double>& R1() const { return R1_; }
    const SpMatrix<double>& R0() const { return R0_; }
    const QuadratureRule& integrator() const { return integrator_; }
    const FunctionSpace& reference_basis() const { return reference_basis; }
    FunctionBasis& basis() const { return fe_basis_; }
    // flags
    bool is_init = false;   // notified true if initialization occurred with no errors
    bool success = false;   // notified true if problem solved with no errors

    template <typename PDE> void init(const PDE& pde);
    template <typename PDE> void set_dirichlet_bc(const PDE& pde);
  
   protected:
    QuadratureRule integrator_ {};       // default to a quadrature rule which is exact for the considered FEM order
    FunctionSpace reference_basis_ {};   // function basis on the reference unit simplex
    FunctionBasis fe_basis_ {};          // basis over the whole domain
    DMatrix<double> solution_;           // vector of coefficients of the approximate solution
    DMatrix<double> force_;              // discretized force [u]_i = \int_D f*\psi_i
    SpMatrix<double> R1_;   // [R1_]_{ij} = a(\psi_i, \psi_j), being a(.,.) the bilinear form of the problem
    SpMatrix<double> R0_;   // mass matrix
};

// implementative details

// initialize solver
template <typename D, typename E, typename F>
template <typename PDE>
void FEMSolverBase<D, E, F>::init(const PDE& pde) {
    static_assert(is_pde<PDE>::value, "not a valid PDE");
    Assembler<FEM, D, FunctionSpace, QuadratureRule> assembler(pde.domain(), integrator_);
    // assemble discretization matrix for given operator
    R1_ = assembler.discretize_operator(pde.differential_operator());
    R1_.makeCompressed();
    // assemble forcing vector
    std::size_t n = pde.domain().dof();   // degrees of freedom in space
    std::size_t m;                        // number of time points
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
    // compute mass matrix [R0]_{ij} = \int_{\Omega} \phi_i \phi_j
    R0_ = assembler.discretize_operator(Reaction<FEM, double>(1.0));
    is_init = true;
    return;
}

// impose dirichlet boundary conditions
template <typename D, typename E, typename F>
template <typename PDE>
void FEMSolverBase<D, E, F>::set_dirichlet_bc(const PDE& pde) {
    static_assert(is_pde<PDE>::value, "not a valid PDE");
    if (!is_init) throw std::runtime_error("solver must be initialized first!");
    for (auto it = pde.domain().boundary_begin(); it != pde.domain().boundary_end(); ++it) {
        R1_.row(*it) *= 0;            // zero all entries of this row
        R1_.coeffRef(*it, *it) = 1;   // set diagonal element to 1 to impose equation u_j = b_j

        // boundaryData is a map (nodeID, boundary value).
        // TODO: currently only space-only case supported (reason of [0] below)
        force_.coeffRef(*it, 0) = pde.boundary_data().at(*it)[0];   // impose boundary value on forcing term
    }
    return;
}

}   // namespace core
}   // namespace fdapde

#endif   // __FEM_SOLVER_BASE_H__
