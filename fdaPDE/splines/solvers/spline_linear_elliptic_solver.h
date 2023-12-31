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

#ifndef __SPLINE_LINEAR_ELLIPTIC_SOLVER_H__
#define __SPLINE_LINEAR_ELLIPTIC_SOLVER_H__

#include <exception>

#include "../../utils/symbols.h"
#include "spline_solver_base.h"

namespace fdapde {
namespace core {
  
template <typename D, typename E, typename F, typename... Ts>
struct SplineLinearEllipticSolver : public SplineSolverBase<D, E, F, Ts...> {
    using Base = SplineSolverBase<D, E, F, Ts...>;
    SplineLinearEllipticSolver(const D& domain) : Base(domain) { }
    
    // solves linear system R1_*u = b, where R1_ : stiff matrix, b : discretized force
    template <typename PDE> void solve(const PDE& pde) {
        fdapde_static_assert(is_pde<PDE>::value, THIS_METHOD_IS_FOR_PDE_ONLY);
        if (!this->is_init) throw std::runtime_error("solver must be initialized first!");

        typedef Eigen::SparseLU<SpMatrix<double>, Eigen::COLAMDOrdering<int>> SystemSolverType;
        SystemSolverType solver;
        solver.compute(this->stiff_);
        // stop if something was wrong
        if (solver.info() != Eigen::Success) {
            this->success = false;
            return;
        }
        // solve linear system: R1_*solution_ = force_;
        this->solution_ = solver.solve(this->force_);        
        this->success = true;
        return;
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __SPLINE_LINEAR_ELLIPTIC_SOLVER_H__
