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

#ifndef __FEM_CRANK_NICOLSON_SEMI_IMPLICIT_H__
#define __FEM_CRANK_NICOLSON_SEMI_IMPLICIT_H__

#include <exception>

#include "../../utils/symbols.h"
#include "fem_solver_base.h"


namespace fdapde {
namespace core {

template <typename D, typename E, typename F, typename... Ts>
struct FEMCrankNicolsonSemiImplicit : public FEMSolverBase<D, E, F, Ts...> {

protected:
    double deltaT_ = 1e-2;

public:
    typedef std::tuple<Ts...> SolverArgs;
    enum { fem_order = std::tuple_element <0, SolverArgs>::type::value };
    typedef D DomainType;
    using FunctionalBasis = LagrangianBasis<DomainType, fem_order>;
    using ReferenceBasis = typename FunctionalBasis::ReferenceBasis;
    using Quadrature = typename ReferenceBasis::Quadrature;

    using Base = FEMSolverBase<D, E, F, Ts...>;
    FEMCrankNicolsonSemiImplicit(const D& domain) : Base(domain) { }
    FEMCrankNicolsonSemiImplicit(const D& domain, const DMatrix<short int>& BMtrx) : Base(domain, BMtrx){ }

    void set_deltaT(double deltaT) { deltaT_ = deltaT; }

    template <typename PDE>
    void solve(PDE& pde) {
        fdapde_static_assert(is_pde<PDE>::value, THIS_METHOD_IS_FOR_PDE_ONLY);
        if (!this->is_init) throw std::runtime_error("solver must be initialized first!");

        // define eigen system solver, use SparseLU decomposition.
        Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
        this->set_deltaT((pde.time_domain()[1] - pde.time_domain()[0]));
        std::size_t n = this->n_dofs();              // degrees of freedom in space
        std::size_t m = pde.forcing_data().cols();   // number of iterations for time loop

        this->solution_.resize(n, m);
        this->solution_.col(0) = pde.initial_condition();   // impose initial condition
        DVector<double> rhs(n, 1);

        // Crank-Nicolson with semi-implicit treatment of the nonlinear term
        SpMatrix<double> K; // stifness matrix
        for (std::size_t i = 0; i < m-1; ++i) {
            if (i==0) { // first step with implicit Eler
                // declare the assembler with the solution at the previous step updated
                // allows to assemble the nonlinearity defined in the test u*h(u) in a semi-implicit way at each time-step as u_{k+1}*h(u_{k})
                // u_{k} = this->solution_.col(i), u_{k+1} = this->solution_.col(i+1)
                Assembler<FEM, DomainType, ReferenceBasis, Quadrature> assembler(pde.domain(), this->integrator_, this->n_dofs_, this->dofs_, this->solution_.col(i));

                // discretize stiffness matrix
                this->stiff_ = assembler.discretize_operator(pde.differential_operator());

                // build system matrix and rhs
                K = (this->mass_) / deltaT_ + this->stiff_;
                rhs = ((this->mass_) / deltaT_) * this->solution_.col(i) + this->force_.block(n * (i + 1), 0, n, 1);

                // set Robin boundary conditions
                if (this->boundary_dofs_begin_Robin() != this->boundary_dofs_end_Robin()) {
                   K += this->robin_;
                }
            }
            else { // Crank-Nicolson for all the other iterations
                // declare the assembler passing w = 3*u_{k}/2 - u_{k-1}/2 where u_k = solution at the previous step
                // allows to assemble the nonlinearity defined in the test u*h(u) in a semi-implicit way at each time-step as u_{k+1}*h(w)
                auto w = 3*this->solution_.col(i)/2 - this->solution_.col(i-1)/2;
                Assembler<FEM, DomainType, ReferenceBasis, Quadrature> assembler(pde.domain(), this->integrator_, this->n_dofs_, this->dofs_, w);

                // discretize stiffness matrix
                this->stiff_ = assembler.discretize_operator(pde.differential_operator());

                // build system matrix and rhs
                K = (this->mass_) / deltaT_ + this->stiff_ / 2;
                rhs = ((this->mass_) / deltaT_  - (this->stiff_) / 2) * this->solution_.col(i) + (this->force_.block(n * (i + 1), 0, n, 1) + this->force_.block(n * i, 0, n, 1))/2;

                // set Robin boundary conditions
                if (this->boundary_dofs_begin_Robin() != this->boundary_dofs_end_Robin()) {
                   K += this->robin_ / 2;
                   rhs += -(this->robin_ / 2) * this->solution_.col(i);
                }
            }

            // set dirichlet boundary conditions
            for (auto it = this->boundary_dofs_begin_Dirichlet(); it != this->boundary_dofs_end_Dirichlet(); ++it) {
                K.row(*it) *= 0;            // zero all entries of this row
                K.coeffRef(*it, *it) = 1;   // set diagonal element to 1 to impose equation u_j = b_j
                rhs[*it] = pde.dirichlet_boundary_data()((*it) + n*(i+1), 0);
            }
            K.makeCompressed();
            // Perform LU decomposition of the system matrix at every step
            solver.compute(K);                        // prepare solver
            if (solver.info() != Eigen::Success) {    // stop if something was wrong...
                this->success = false;
                return;
            }
            
            this->solution_.col(i + 1) = solver.solve(rhs);   // append time step solution to solution matrix
            // std::cout << i << std::endl;
        }

        this->success = true;
        return;
    } // end solve
};


}   // namespace core
}   // namespace fdapde

#endif   // __FEM_CRANK_NICOLSON_SEMI_IMPLICIT_H__