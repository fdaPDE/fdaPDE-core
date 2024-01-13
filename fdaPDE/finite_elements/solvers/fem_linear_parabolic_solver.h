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

#ifndef __FEM_LINEAR_PARABOLIC_SOLVER_H__
#define __FEM_LINEAR_PARABOLIC_SOLVER_H__

#include <exception>
#include "../../utils/symbols.h"
#include "fem_solver_base.h"

namespace fdapde {
namespace core {

template <typename D, typename E, typename F, typename... Ts>
class FEMLinearParabolicSolver : public FEMSolverBase<D, E, F, Ts...> {
   private:
    double deltaT_ = 1e-2;
   public:
    using Base = FEMSolverBase<D, E, F, Ts...>;
    FEMLinearParabolicSolver(const D& domain) : Base(domain) { }
    void set_deltaT(double deltaT) { deltaT_ = deltaT; }

    // solves the PDE using a forward-euler scheme
    template <typename PDE> void solve(const PDE& pde) {
        fdapde_static_assert(is_pde<PDE>::value, THIS_METHOD_IS_FOR_PDE_ONLY);
        if (!this->is_init) throw std::runtime_error("solver must be initialized first!");

        Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
        this->set_deltaT((pde.time_domain()[1] - pde.time_domain()[0]));
        std::size_t n = this->n_dofs();              // degrees of freedom in space
        std::size_t m = pde.forcing_data().cols();   // number of iterations for time loop

        this->solution_.resize(this->n_dofs(), m);
        this->solution_.col(0) = pde.initial_condition();   // impose initial condition
        DVector<double> rhs(n, 1);
        SpMatrix<double> K = (this->mass_) / deltaT_ + this->stiff_;   // build system matrix
        // set dirichlet boundary conditions
        for (auto it = this->boundary_dofs_begin(); it != this->boundary_dofs_end(); ++it) {
            K.row(*it) *= 0;            // zero all entries of this row
            K.coeffRef(*it, *it) = 1;   // set diagonal element to 1 to impose equation u_j = b_j
        }
        K.makeCompressed();
        solver.compute(K);                       // prepare solver
        if (solver.info() != Eigen::Success) {   // stop if something was wrong...
            this->success = false;
            return;
        }
        // execute temporal loop to solve ODE system via forward-euler scheme
        for (std::size_t i = 0; i < m - 1; ++i) {
            rhs = ((this->mass_) / deltaT_) * this->solution_.col(i) + this->force_.block(n * (i + 1), 0, n, 1);
            // impose boundary conditions
            for (auto it = this->boundary_dofs_begin(); it != this->boundary_dofs_end(); ++it) {
                rhs[*it] = pde.boundary_data()(*it, i + 1);
            }
            this->solution_.col(i + 1) = solver.solve(rhs);   // append time step solution to solution matrix
        }
        this->success = true;
        return;
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __FEM_LINEAR_PARABOLIC_SOLVER_H__
