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

#ifndef __FEM_NONLINEAR_FIXEDPOINT_SOLVER_H__
#define __FEM_NONLINEAR_FIXEDPOINT_SOLVER_H__

#include <exception>

#include "../../utils/symbols.h"
#include "fem_solver_base.h"

#include <unsupported/Eigen/IterativeSolvers>

namespace fdapde {
namespace core {

template <typename D, typename E, typename F, typename... Ts>
struct FEMNonLinearFixedPointSolver : public FEMSolverBase<D, E, F, Ts...> {

protected:
    size_t MaxIter_ = 60;      // maximum number of iterations
    double tol_     = 1e-3;    // tolerance for convergence

public:
    typedef std::tuple<Ts...> SolverArgs;
    enum {fem_order = std::tuple_element <0, SolverArgs>::type::value };
    typedef D DomainType;
    // typedef Integrator<DomainType::local_dimension, fem_order> QuadratureRule;
    // typedef LagrangianElement<DomainType::local_dimension, fem_order> FunctionSpace;
    using FunctionalBasis = LagrangianBasis<DomainType, fem_order>;
    using ReferenceBasis = typename FunctionalBasis::ReferenceBasis;
    using Quadrature = typename ReferenceBasis::Quadrature;

    using Base = FEMSolverBase<D, E, F, Ts...>;
    FEMNonLinearFixedPointSolver(const D& domain) : Base(domain) { }

    // setter for MaxIter and tol (should we add them to the constructor?)
    void set_MaxIter(size_t MaxIter) { MaxIter_ = MaxIter; }
    void set_tol(double tol) { tol_ = tol; }

    // solves the nonlinear PDE using a fixed-point method
    template <typename PDE>
    void solve(const PDE& pde) {
        static_assert(is_pde<PDE>::value, "pde is not a valid PDE object");

        if (!this->is_init) throw std::runtime_error("solver must be initialized first!");

        // define eigen system solver, use SparseLU decomposition.
        Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver; //swap LU-GMRES

        this->solution_.resize(this->n_dofs_,1);

        DVector<double> f_prev;   // solution at the previous step
        f_prev.resize(this->n_dofs_);
        f_prev = pde.initial_condition();  // set initial guess

        // Execute nonlinear loop to solve nonlinear system via fixedpoint method. Here the nonlinear loop have
        // a maximum number of iterations and a convergence check criterion.
        std::size_t i;
        for (i = 0; i < MaxIter_; ++i) {
            
            // Perform LU decomposition of the system matrix at every step
            solver.compute(this->stiff_);                // prepare solver
            if (solver.info() != Eigen::Success) {    // stop if something was wrong...
                this->success = false;
                // std::cout << "Return due to success=false at iteration " << i << std::endl;
                return;
            }
            this->solution_ = solver.solve(this->force_);   // solve linear system

            // Check convergence to stop early
            auto incr = this->solution_ - f_prev;
            if (incr.norm() < tol_) break;

            f_prev = this->solution_;

            // Update the system matrix for the next iteration.
            Assembler<FEM, DomainType, ReferenceBasis, Quadrature> assembler(pde.domain(), this->integrator_, this->n_dofs_, this->dofs_, f_prev);
            this->stiff_ = assembler.discretize_operator(pde.differential_operator());
            this->stiff_.makeCompressed();
            
            // set dirichlet boundary conditions on the system matrix
            this->set_dirichlet_bc(pde);
        }
        // std::cout << "\n\t Fixedpoint ended with " << i + 1 << " iterations" << std::endl;
        this->success = true;
        return;
    } // end solve
};

}   // namespace core
}   // namespace fdapde

#endif   // __FEM_NONLINEAR_FIXEDPOINT_SOLVER_H__
