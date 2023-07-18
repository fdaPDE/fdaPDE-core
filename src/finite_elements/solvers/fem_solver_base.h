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

#include "../../utils/symbols.h"
#include "../../utils/traits.h"
#include "../assembler.h"
#include "../basis/lagrangian_basis.h"
#include "../integration/integrator.h"

namespace fdapde {
namespace core {

// forward declarations
template <typename E> struct is_pde;
  
// base class for the definition of a general solver based on the Finite Element Method
class FEMSolverBase {
   protected:
    DMatrix<double> solution_;   // vector of coefficients of the approximate solution
    DMatrix<double> force_;      // right-hand side of the FEM linear system
    SpMatrix<double> R1_;        // result of the discretization of the bilinear form
    SpMatrix<double> R0_;        // mass matrix, i.e. discretization of the identity operator
    bool init_ = false;          // set to true by init() at the end of solver initialization
   public:
    // flag used to notify is something was wrong during computation
    bool success = true;
    // constructor
    FEMSolverBase() = default;
    // getters
    const DMatrix<double>& solution() const { return solution_; }
    const DMatrix<double>& force() const { return force_; }
    const SpMatrix<double>& R1() const { return R1_; }
    const SpMatrix<double>& R0() const { return R0_; }

    // initializes internal FEM solver status
    template <typename E>
    void init(const E& pde) {
        static_assert(is_pde<E>::value, "pde is not a valid PDE object");
        Assembler<E::M, E::N, E::R, typename E::BasisType, typename E::IntegratorType> assembler(
          pde.domain(), pde.integrator());
        // fill discretization matrix for current operator
        R1_ = assembler.discretize_operator(pde.differential_operator());
        R1_.makeCompressed();

        // fill forcing vector
        std::size_t n = pde.domain().dof();   // degrees of freedom in space
        std::size_t m;                        // number of time points

        if constexpr (!std::is_base_of<ScalarBase, typename E::ForcingType>::value) {
            m = pde.forcing_data().cols();
            force_.resize(n * m, 1);
            force_.block(0, 0, n, 1) = assembler.discretize_forcing(pde.forcing_data().col(0));

            // iterate over time steps if a space-time PDE is supplied
            if constexpr (is_parabolic<typename E::OperatorType>::value) {
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

        // compute mass matrix [R0]_{ij} = \int_{\Omega} \phi_i \phi_j by discretization of the identity operator
        R0_ = assembler.discretize_operator(Identity(1.0));
        init_ = true;
        return;
    }

    // impose dirichlet boundary conditions
    template <typename E>
    void set_dirichlet_bc(const E& pde) {
        static_assert(is_pde<E>::value, "pde is not a valid PDE object");
	
        if (!init_) throw std::runtime_error("solver must be initialized first!");
        for (auto it = pde.domain().boundary_begin(); it != pde.domain().boundary_end(); ++it) {
            R1_.row(*it) *= 0;            // zero all entries of this row
            R1_.coeffRef(*it, *it) = 1;   // set diagonal element to 1 to impose equation u_j = b_j

            // boundaryData is a map (nodeID, boundary value).
            // TODO: currently only space-only case supported (reason of [0] below)
            force_.coeffRef(*it, 0) = pde.boundary_data().at(*it)[0];   // impose boundary value on forcing term
        }
        return;
    };
};

}   // namespace core
}   // namespace fdapde

#endif   // __FEM_SOLVER_BASE_H__
