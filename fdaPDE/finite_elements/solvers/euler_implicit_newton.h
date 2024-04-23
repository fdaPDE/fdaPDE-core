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

#ifndef __FEM_EULER_IMPLICIT_NEWTON_H__
#define __FEM_EULER_IMPLICIT_NEWTON_H__

#include <exception>

#include "../../utils/symbols.h"
#include "fem_solver_base.h"


namespace fdapde {
namespace core {

template <typename D, typename E, typename F, typename... Ts>
struct FEMEulerImplicitNewton : public FEMSolverBase<D, E, F, Ts...> {

protected:
    double deltaT_ = 1e-2;
    size_t MaxIter_ = 20;      // maximum number of iterations
    double tol_     = 1e-3;    // tolerance for convergence

public:
    typedef std::tuple<Ts...> SolverArgs;
    enum { fem_order = std::tuple_element <0, SolverArgs>::type::value };
    typedef D DomainType;
    using FunctionalBasis = LagrangianBasis<DomainType, fem_order>;
    using ReferenceBasis = typename FunctionalBasis::ReferenceBasis;
    using Quadrature = typename ReferenceBasis::Quadrature;

    using Base = FEMSolverBase<D, E, F, Ts...>;
    FEMEulerImplicitNewton(const D& domain) : Base(domain) { }
    FEMEulerImplicitNewton(const D& domain, const DMatrix<short int>& BMtrx) : Base(domain, BMtrx){ }

    void set_deltaT(double deltaT) { deltaT_ = deltaT; }
    void set_MaxIter(double MaxIter) { MaxIter_ = MaxIter; }
    void set_tol(double tol) { tol_ = tol; }

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
        DVector<double> f_prev(n, 1);  // solution at the previous step for the Newton method

        // For Newton method: declare the known term that will go on the right hand side and will contain h'
        NonLinearReactionPrime<DomainType::local_dimension, ReferenceBasis> h_prime(pde.non_linear_reaction());
        auto Lprime = non_linear_op<FEM>(h_prime);
        

        // Implicit Euler
        for (std::size_t i = 0; i < m-1; ++i) {
            // std::cout << "Time iteration = " << i << std::endl;

            f_prev = this->solution_.col(i);

            // At each iteration in time, we have a loop to solve the nonlinear equation arising:
            // (u_{n+1} - u_n)/dt - L(u_{n+1}) = f_{n+1} where the nonlinear term is in L and is given by alpha*u_{n+1}*h(u_{n+1})

            // Use NEWTON

            // 1) Solve the linear associated pde (initial guess = 0)

            // // Define the lhs and the rhs of the equation corresponding to the associate linear system
            // f_prev = DMatrix<double>::Zero(n, 1);
            // Assembler<FEM, DomainType, ReferenceBasis, Quadrature> assembler(pde.domain(), this->integrator_, this->n_dofs_, this->dofs_, f_prev);
            // this->stiff_ = assembler.discretize_operator(pde.differential_operator());
            // this->stiff_.makeCompressed();
            // SpMatrix<double> K = (this->mass_) / deltaT_ + this->stiff_;
            // rhs = ((this->mass_) / deltaT_) * this->solution_.col(i) + this->force_.block(n * (i + 1), 0, n, 1);
            // // set dirichlet boundary conditions
            // for (auto it = this->boundary_dofs_begin_Dirichlet(); it != this->boundary_dofs_end_Dirichlet(); ++it) {
            //     K.row(*it) *= 0;            // zero all entries of this row
            //     K.coeffRef(*it, *it) = 1;   // set diagonal element to 1 to impose equation u_j = b_j
            //     rhs[*it] = pde.dirichlet_boundary_data()(*it, i + 1);
            // }
            // K.makeCompressed();
            
            // solver.compute(K);
            // if (solver.info() != Eigen::Success) {    // stop if something went wrong...
            //     this->success = false;
            //     return;
            // }
            // this->solution_.col(i+1) = solver.solve(rhs);
            // f_prev = this->solution_.col(i+1);   // solution at the previous step

            // 2) Execute nonlinear loop to solve nonlinear system
            for (std::size_t j = 1; j < MaxIter_; ++j) {
                // std::cout << "Newton iteration = " << j << std::endl;

                // Update the system matrix for the next iteration.
                Assembler<FEM, DomainType, ReferenceBasis, Quadrature> assembler(pde.domain(), this->integrator_, this->n_dofs_, this->dofs_, f_prev);
                this->stiff_ = assembler.discretize_operator(pde.differential_operator());
                this->stiff_.makeCompressed();

                // discretize derived nonlinearity
                SpMatrix<double> R2 = assembler.discretize_operator(Lprime);
                R2.makeCompressed(); 

                // set Robin boundary conditions
                if (this->boundary_dofs_begin_Robin() != this->boundary_dofs_end_Robin()) {
                    this->stiff_ += this->robin_;
                }

                // build system matrix and rhs
                SpMatrix<double> K = (this->mass_) / deltaT_ + this->stiff_ + R2;
                rhs = ((this->mass_) / deltaT_) * this->solution_.col(i) + this->force_.block(n * (i + 1), 0, n, 1) + R2*f_prev;

                // set dirichlet boundary conditions
                for (auto it = this->boundary_dofs_begin_Dirichlet(); it != this->boundary_dofs_end_Dirichlet(); ++it) {
                    K.row(*it) *= 0;            // zero all entries of this row
                    K.coeffRef(*it, *it) = 1;   // set diagonal element to 1 to impose equation u_j = b_j
                    rhs[*it] = pde.dirichlet_boundary_data()(*it, i + 1);
                }
                K.makeCompressed();

                // Perform LU decomposition of the system matrix at every step
                solver.compute(K);                // prepare solver
                if (solver.info() != Eigen::Success) {    // stop if something was wrong...
                    this->success = false;
                    return;
                }

                this->solution_.col(i+1) = solver.solve(rhs);   // solve linear system

                // Check convergence to stop early
                auto incr = this->solution_.col(i+1) - f_prev;
                // std::cout << "Increment = " << incr.norm() << std::endl;
                if (incr.norm() < tol_) break;
                // update the solution at the previous step in this internal for loop

                f_prev = this->solution_.col(i+1);
            }
        }

        this->success = true;
        return;
    } // end solve
};

}   // namespace core
}   // namespace fdapde

#endif   // __FEM_EULER_IMPLICIT_NEWTON_H__