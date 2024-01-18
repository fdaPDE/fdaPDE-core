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

#ifndef __FEM_NONLINEAR_BROYDEN_SOLVER_H__
#define __FEM_NONLINEAR_BROYDEN_SOLVER_H__

#include <exception>

#include "../../utils/symbols.h"
#include "fem_solver_base.h"
#include "../../optimization/broyden.h"

using fdapde::Dynamic;

namespace fdapde {
namespace core {

template <typename D, typename E, typename F, typename... Ts>
struct FEMNonLinearBroydenSolver : public FEMSolverBase<D, E, F, Ts...> {

protected:
    size_t MaxIter_ = 500;      // maximum number of iterations
    double tol_     = 1e-3;     // tolerance for convergence

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
    FEMNonLinearBroydenSolver(const D& domain) : Base(domain) { }

    // setter for MaxIter and tol (should we add them to the constructor?)
    void set_MaxIter(size_t MaxIter) { MaxIter_ = MaxIter; }
    void set_tol(double tol) { tol_ = tol; }

    // solves the nonlinear PDE using a fixed-point method
    template <typename PDE>
    void solve(const PDE& pde) {
        static_assert(is_pde<PDE>::value, "pde is not a valid PDE object");

        if (!this->is_init) throw std::runtime_error("solver must be initialized first!");

        // first solve the linear associated pde (initial guess = 0)
        DVector<double> f_prev = DVector<double>::Zero(this->n_dofs_);
        // Assembler<FEM, D, FunctionSpace, QuadratureRule> assembler(pde.domain(), this->integrator_, this->n_dofs_, this->dofs_, f_prev); //old
        Assembler<FEM, DomainType, ReferenceBasis, Quadrature> assembler(pde.domain(), this->integrator_, this->n_dofs_, this->dofs_, f_prev);
        SpMatrix<double> A = assembler.discretize_operator(pde.differential_operator());
        A.makeCompressed();
        // set B.C.s
        for (auto it = this->boundary_dofs_begin(); it != this->boundary_dofs_end(); ++it) {
            A.row(*it) *= 0;            // zero all entries of this row
            A.coeffRef(*it, *it) = 1;   // set diagonal element to 1 to impose equation u_j = b_j
        }

        Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
        solver.compute(A);
        if (solver.info() != Eigen::Success) {    // stop if something went wrong...
            this->success = false;
            // std::cout << "LU return due to success=false" << std::endl;
            return;
        }
        DVector<double> x0 = solver.solve(this->force_);

        // F(u) = A(u)*u - b
        std::function<DVector<double>(DVector<double>)> Fun;

        Fun = [&](DVector<double> u) -> DVector<double> {
            // A(u) * u - f = 0
            this->solution_ = u;
            this->init(pde);
            this->set_dirichlet_bc(pde);
            return this->stiff_ * u - this->force_;
        };

        // solve with Broyden Line Search - Weijun Zhou, Li Zhang
        Broyden<Dynamic> br(MaxIter_, tol_);
        this->solution_ = br.solve_modified(Fun, x0);

        // solve with Broyden Trust Region
        // this -> solution_ = br.solve_trust_region(Fun, x0);

        this->success = true;
        return;
    } // end solve
};

}   // namespace core
}   // namespace fdapde

#endif   // __FEM_NONLINEAR_BROYDEN_SOLVER_H__
