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

#ifndef __FEM_STANDARD_SPACE_SOLVER_H__
#define __FEM_STANDARD_SPACE_SOLVER_H__

#include <exception>

#include "../../utils/symbols.h"
#include "../assembler.h"
#include "fem_solver_base.h"

namespace fdapde {
namespace core {

struct FEMStandardSpaceSolver : public FEMSolverBase {
    // constructor
    FEMStandardSpaceSolver() = default;

    // solves linear system R1_*u = b, where R1_ : stiff matrix, b : forcing term discretization
    template <
      unsigned int M, unsigned int N, unsigned int R, typename E, typename F, typename B, typename I, typename S>
    void solve(const PDE<M, N, R, E, F, B, I, S>& pde);
};

template <unsigned int M, unsigned int N, unsigned int R, typename E, typename F, typename B, typename I, typename S>
void FEMStandardSpaceSolver::solve(const PDE<M, N, R, E, F, B, I, S>& pde) {
    if (!init_) throw std::runtime_error("solver must be initialized first!");
    // define eigen system solver, use sparse LU decomposition.
    Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
    solver.compute(this->R1_);
    // stop if something was wrong...
    if (solver.info() != Eigen::Success) {
        this->success = false;
        return;
    }
    // solve FEM linear system: R1_*solution_ = force_;
    this->solution_ = solver.solve(this->force_);
    return;
}

}   // namespace core
}   // namespace fdapde

#endif   // __SPACE_SOLVER_H__
