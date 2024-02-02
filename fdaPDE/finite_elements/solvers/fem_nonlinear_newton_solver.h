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

#ifndef __FEM_NONLINEAR_NEWTON_SOLVER_H__
#define __FEM_NONLINEAR_NEWTON_SOLVER_H__

#include <exception>

#include "../../utils/symbols.h"
#include "fem_solver_base.h"
#include "../../fields/non_linear_reaction.h"

using fdapde::core::laplacian;
using fdapde::core::ScalarField;
using fdapde::core::NonLinearReactionPrime;

namespace fdapde {
namespace core {

template <typename D, typename E, typename F, typename... Ts>
struct FEMNonLinearNewtonSolver : public FEMSolverBase<D, E, F, Ts...> {

protected:
    size_t MaxIter_ = 15;       // maximum number of iterations
    double tol_     = 1e-8;    // tolerance for convergence
    bool convergence_test_ = false; 

public:
    typedef std::tuple<Ts...> SolverArgs;
    enum { fem_order = std::tuple_element <0, SolverArgs>::type::value };
    typedef D DomainType;
    using FunctionalBasis = LagrangianBasis<DomainType, fem_order>;
    using ReferenceBasis = typename FunctionalBasis::ReferenceBasis;
    using Quadrature = typename ReferenceBasis::Quadrature;

    using Base = FEMSolverBase<D, E, F, Ts...>;
    FEMNonLinearNewtonSolver(const D& domain) : Base(domain) { }

    // setter for MaxIter and tol (should we add them to the constructor?)
    void set_MaxIter(size_t MaxIter) { MaxIter_ = MaxIter; }
    void set_tol(double tol) { tol_ = tol; }
    void set_conv_test(bool b) { convergence_test_ = b; }

    template <typename PDE>
    void solve(PDE& pde) {
        static_assert(is_pde<PDE>::value, "pde is not a valid PDE object");

        if (!this->is_init) throw std::runtime_error("solver must be initialized first!");

        // first solve the linear associated pde (initial guess = 0)
        Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
        this->solution_.resize(this->n_dofs_,1);
        solver.compute(this->stiff_);
        if (solver.info() != Eigen::Success) {    // stop if something went wrong...
            this->success = false;
            // std::cout << "LU return due to success=false" << std::endl;
            return;
        }
        this->solution_ = solver.solve(this->force_);
        DVector<double> f_prev = this->solution_;   // solution at the previous step

        // declare the known term that will go on the right hand side and will contain h'
        NonLinearReactionPrime<DomainType::local_dimension, ReferenceBasis> h_prime(pde.non_linear_reaction());
        auto Lprime = non_linear_op<FEM>(h_prime);

        DVector<double> force_backup = this->force_;

        // Execute nonlinear loop to solve nonlinear system
        std::size_t i;
        for (i = 1; i < MaxIter_; ++i) {

            // re-declare the assembler with f_prev updated.
            Assembler<FEM, DomainType, ReferenceBasis, Quadrature> assembler(pde.domain(), this->integrator_, this->n_dofs_, this->dofs_, f_prev);

            // discretize stiffness matrix
            this->stiff_ = assembler.discretize_operator(pde.differential_operator());
            this->stiff_.makeCompressed();

            // discretize derived nonlinearity
            SpMatrix<double> R2 = assembler.discretize_operator(Lprime);
            R2.makeCompressed(); 

            this->force_ = force_backup + R2*f_prev;
            this->stiff_ += R2;
            this->set_dirichlet_bc(pde);

            // Perform LU decomposition of the system matrix at every step
            solver.compute(this->stiff_);                // prepare solver
            if (solver.info() != Eigen::Success) {    // stop if something was wrong...
                this->success = false;
                // std::cout << "Return due to success=false at iteration " << i << std::endl;
                return;
            }

            this->solution_ = solver.solve(this->force_);

            // Check convergence to stop early
            double incr = (this->solution_ - f_prev).norm();
            if (incr < tol_) break;

            // std::cout << "\nNewton iter: " << i << std::endl;
            // std::cout << "Newton -> || Au - f || = " << (this->stiff_ * this->solution_ - this->force_ ).norm() << std::endl;
            // std::cout << "Newton -> ||Increment|| = " << incr << std::endl;

            f_prev = this->solution_;
        }
        this->success = true;
        return;
    } // end solve
};


}   // namespace core
}   // namespace fdapde

#endif   // __FEM_NONLINEAR_NEWTON_SOLVER_H__