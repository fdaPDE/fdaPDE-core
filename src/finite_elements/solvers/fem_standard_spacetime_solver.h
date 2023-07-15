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

#ifndef __FEM_STANDARD_SPACE_TIME_SOLVER_H__
#define __FEM_STANDARD_SPACE_TIME_SOLVER_H__

#include <exception>

#include "../../utils/symbols.h"
#include "../assembler.h"
#include "fem_solver_base.h"

namespace fdapde {
namespace core {

struct FEMStandardSpaceTimeSolver : public FEMSolverBase {
    // constructor
    FEMStandardSpaceTimeSolver() = default;

    // solves the PDE using a forward-euler scheme
    template <
      unsigned int M, unsigned int N, unsigned int R, typename E, typename F, typename B, typename I, typename S>
    void solve(const PDE<M, N, R, E, F, B, I, S>& pde, double deltaT);
};

template <unsigned int M, unsigned int N, unsigned int R, typename E, typename F, typename B, typename I, typename S>
void FEMStandardSpaceTimeSolver::solve(const PDE<M, N, R, E, F, B, I, S>& pde, double deltaT) {
    if (!init_) throw std::runtime_error("solver must be initialized first!");

    // define eigen system solver, use SparseLU decomposition.
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
    std::size_t n = pde.domain().dof();          // degrees of freedom in space
    std::size_t m = pde.forcing_data().cols();   // number of iterations for time loop

    this->solution_.resize(pde.domain().dof(), m - 1);
    this->solution_.col(0) = pde.initial_condition();   // impose initial condition
    DVector<double> rhs = ((this->R0_) / deltaT) * pde.initial_condition() + this->force_.block(0, 0, n, 1);

    // Observe that K is time invariant only for homogeneous boundary conditions. In general we need to recompute K at
    // each time instant, anyway we can avoid the recomputation of K at each iteration by just keeping overwriting it at
    // the boundary indexes positions.
    Eigen::SparseMatrix<double> K = (this->R0_) / deltaT + this->R1_;   // build system matrix

    // prepare system matrix to handle dirichlet boundary conditions
    for (auto it = pde.domain().boundary_begin(); it != pde.domain().boundary_end(); ++it) {
        K.row(*it) *= 0;            // zero all entries of this row
        K.coeffRef(*it, *it) = 1;   // set diagonal element to 1 to impose equation u_j = b_j
    }
    // execute temporal loop to solve ODE system via forward-euler scheme
    for (std::size_t i = 1; i < m - 1; ++i) {
        // impose boundary conditions
        for (auto it = pde.domain().boundary_begin(); it != pde.domain().boundary_end(); ++it) {
            rhs[*it] = pde.boundary_data().at(*it)[i];
        }
        solver.compute(K);                       // prepare solver
        if (solver.info() != Eigen::Success) {   // stop if something was wrong...
            this->success = false;
            return;
        }
        DVector<double> u_i = solver.solve(rhs);   // solve linear system
        this->solution_.col(i) = u_i;              // append time step solution to solution matrix
        rhs = ((this->R0_) / deltaT) * u_i + this->force_.block(n * i, 0, n, 1);   // update rhs for next iteration
    }
    return;
}

}   // namespace core
}   // namespace fdapde

#endif   // __FEM_STANDARD_SPACE_TIME_SOLVER_H__
