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

#ifndef __FEM_NONLINEAR_SOLVER_H__
#define __FEM_NONLINEAR_SOLVER_H__

#include <exception>

#include "../../utils/symbols.h"
#include "fem_solver_base.h"
#include "../../optimization.h"
#include "fem_nonlinear_fixedpoint_solver.h"
#include "fem_nonlinear_broyden_solver.h"
using fdapde::core::BFGS;
using fdapde::core::BacktrackingLineSearch;
using fdapde::core::WolfeLineSearch;

// using fdapde::core::ScalarField;
using fdapde::Dynamic;

namespace fdapde {
namespace core {

    template <int N, typename B>
    class NonLinearReactionBase {
    protected:
        static constexpr std::size_t n_basis_ = B::n_basis;
        typedef std::shared_ptr<DVector<double>> VecP;

        B basis_ {};  //calls default constructor
        mutable VecP f_prev_;  // pointer to vector containing the solution on one element at the previous time step
        std::function<double(SVector<N>, SVector<1>)> h = [](SVector<N> x, SVector<1> ff) -> double {return 1 - ff[0];};

        // protected method that preforms $\sum_i {f_i*\psi_i(x)}$
        SVector<1> f(const SVector<N>& x) const{
            SVector<1> result;
            result[0] = 0;
            for (std::size_t i = 0; i < n_basis_; i++){
                result[0] += (*f_prev_)[i] * basis_[i](x); }
            return result;
        }

    public:
        // constructor
        NonLinearReactionBase() = default;
        NonLinearReactionBase(std::function<double(SVector<N>, SVector<1>)> h_) : h(h_) {}

        //setter for the nonlinear function
        void set_nonlinearity(std::function<double(SVector<N>, SVector<1>)> h_) {h = h_;}
    }; // end of NonLinearReactionBase

    template <int N, typename B>
    class NonLinearReaction : public NonLinearReactionBase<N, B>,
                              public ScalarExpr<N, NonLinearReaction<N, B>>  {
    public:
        typedef std::shared_ptr<DVector<double>> VecP;
        
        auto operator()(VecP f_prev) const{
            this->f_prev_ = f_prev;
            return *this;
        }

        double operator()(const SVector<N>& x) const{
            return this->h(x, this->f(x));
        }
    }; // end of NonLinearReaction

    template <int N, typename B>
    class NonLinearReactionPrime: public NonLinearReactionBase<N, B>,
                                  public ScalarExpr<N, NonLinearReactionPrime<N, B>>  {
    public:
        typedef std::shared_ptr<DVector<double>> VecP;
    
        double operator()(const SVector<N>& x) const{
            std::function<double(SVector<1>)> lambda_fun = [&] (SVector<1> ff) -> double {return this->h(x, ff);};
            ScalarField<1> lambda_field(lambda_fun);
            SVector<1> au;
            au << this->f(x);
            return lambda_field.derive()(au)[0] * this->f(x)[0];    // per newton method
        }

        auto operator()(VecP f_prev) const{
            this->f_prev_ = f_prev;
            return *this;
        }
    }; // end of NonLinearReactionPrime


template <typename D, typename E, typename F, typename... Ts>
struct FEMNonLinearSolver : public FEMSolverBase<D, E, F, Ts...> {

protected:
    size_t FixedpointIter_  = 1;          // maximum number of iterations
    size_t BroydenIter_     = 500;        // maximum number of iterations
    size_t NewtonIter_      = 15;
    size_t RestartBroyden_  = BroydenIter_ + 1;
    double tol_             = 1e-7;    // tolerance for convergence

public:

    typedef std::tuple<Ts...> SolverArgs;
    enum {fem_order = std::tuple_element <0, SolverArgs>::type::value };
    typedef D DomainType;
    
    using FunctionalBasis = LagrangianBasis<DomainType, fem_order>;
    using ReferenceBasis = typename FunctionalBasis::ReferenceBasis;
    using Quadrature = typename ReferenceBasis::Quadrature;

    using Base = FEMSolverBase<D, E, F, Ts...>;
    FEMNonLinearSolver(const D& domain) : Base(domain) { }

    // setters
    void setFixedpointIter(const size_t FixedpointIter) { FixedpointIter_ = FixedpointIter; }
    void setBroydenIter(const size_t BroydenIter) { BroydenIter_=BroydenIter; }
    void setNewtonIter(const size_t NewtonIter) { NewtonIter_=NewtonIter; }
    void setRestartBroyden(const size_t RestartBroyden) { RestartBroyden_=RestartBroyden; }
    void setTol(const double tol) { tol_=tol; }

    // solves the nonlinear PDE with different methods
    template <typename PDE>
    void solve(PDE& pde) {
        solveFixedPoint(pde);
        // std::cout << "\n\n\t done with fixedpoint \n"<<std::endl;
        pde.set_initial_condition(this->solution_);
        // solveBroyden(pde);
        solveNewton(pde);

        this->success = true;
        return;
    } // end solve

    template <typename PDE>
    void solveFixedPoint(PDE& pde){

        // add this for the convergence test
        auto solutionExpr = [](SVector<2> x) -> double { return 3*x[0]*x[0] + 2*x[1]*x[1]; };
        DMatrix<double> nodes_ = pde.dof_coords();
        DMatrix<double> solution_ex(nodes_.rows(), 1);
        for (int i = 0; i < nodes_.rows(); ++i) {
            solution_ex(i) = solutionExpr(nodes_.row(i));
        }

        static_assert(is_pde<PDE>::value, "pde is not a valid PDE object");

        if (!this->is_init) throw std::runtime_error("solver must be initialized first!");

        // vector for convergence test
        DVector<double> error_L2 = DVector<double>::Zero(FixedpointIter_);

        // define eigen system solver, use SparseLU decomposition.
        Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver; //swap LU-GMRES

        this->solution_.resize(this->n_dofs_,1);

        DVector<double> f_prev;   // solution at the previous step
        f_prev.resize(this->n_dofs_);
        f_prev = pde.initial_condition();  // set initial guess

        // Execute nonlinear loop to solve nonlinear system via fixedpoint method. Here the nonlinear loop have
        // a maximum number of iterations and a convergence check criterion.
        std::size_t i;
        for (i = 0; i < FixedpointIter_; ++i) {

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

            f_prev = this->solution_;

            // Update the system matrix for the next iteration.
            Assembler<FEM, DomainType, ReferenceBasis, Quadrature> assembler(pde.domain(), this->integrator_, this->n_dofs_, this->dofs_, f_prev);
            this->stiff_ = assembler.discretize_operator(pde.differential_operator());
            this->stiff_.makeCompressed();

            // set dirichlet boundary conditions on the system matrix
            this->set_dirichlet_bc(pde);

            // compute the error as in the tests
            DMatrix<double> error_ = solution_ex - this->solution_;
            error_L2(i) = (this->mass_ * error_.cwiseProduct(error_)).sum();

            // std::cout << "Error: " << error_L2(i) << std::endl;
            // std::cout << "Iter: " << i << std::endl;
            // std::cout << "|| Au - f || = " << (this->stiff_ * this->solution_ - this->force_).norm() << std::endl;
            // std::cout << "||Increment|| : " << incr.norm() << std::endl;

            if (error_L2(i) < tol_) break;
        }

        //save convergence test results
        std::ofstream file("convergence_test_fixedpoint.txt");    //it will be exported in the current build directory
        if (file.is_open()){
            file << error_L2;
            file.close();
        } else {
            std::cerr << "fixedpoint unable to save convergence test" << std::endl;
        }
    } // end fixedpoint

    template <typename PDE>
    void solveBroyden(PDE& pde){
        static_assert(is_pde<PDE>::value, "pde is not a valid PDE object");
        if (!this->is_init) throw std::runtime_error("solver must be initialized first!");

        // exact solution for the convergence test
        auto solutionExpr = [](SVector<2> x) -> double { return 3*x[0]*x[0] + 2*x[1]*x[1]; };
        DMatrix<double> nodes_ = pde.dof_coords();
        DMatrix<double> solution_ex(nodes_.rows(), 1);
        for (int i = 0; i < nodes_.rows(); ++i) {
            solution_ex(i) = solutionExpr(nodes_.row(i));
        }
        // vector for convergence test
        DVector<double> error_L2 = DVector<double>::Zero(BroydenIter_);

        // F(u) = A(u)*u - b
        std::function<DVector<double>(DVector<double>)> Fun;

        Fun = [&](DVector<double> u) -> DVector<double> {
            // A(u) * u - f = 0
            this->solution_ = u;
            this->init(pde);
            this->set_dirichlet_bc(pde);
            return this->stiff_ * u - this->force_;
        };

        // std::cout << "Fun(fixedpoint) = " << Fun(this->solution_).norm() << std::endl;
        // std::cout << "starting broyden.h with " << BroydenIter_ << " iterations" << std::endl;

        // initial condition
        DVector<double> x = pde.initial_condition();
        // error at zeroth iteration
        DMatrix<double> error_ = solution_ex - x;
        error_L2(0) = (this->mass_ * error_.cwiseProduct(error_)).sum();
        // std::cout << "\nBroyden iter: 0" << std::endl;
        // std::cout << "Error: " << error_L2(0) << std::endl;

        int dim = x.size(); // = f_(x).size();

        DVector<double> d_B(dim), d_MB(dim), d(dim);
        DVector<double> s(dim), y(dim);

        DMatrix<double> B = DMatrix<double>::Identity(dim, dim);

        int itc = 0;   // current iteration

        // parameters
        double tau = 1.;
        double rho = sqrt(0.9);
        double alfa_k = 1.;
        double theta_k = 1.;
        double r = 0.1;
        auto eps = [](int n) -> double {return 1./((n+1)*(n+1));};
        double sigma_1 = 0.1;
        double sigma_2 = 0.01;
        bool step_5 = false;

        double norm_fx = 0;
        DVector<double> fx(dim);
        double norm_fz = 0;
        DVector<double> fz(dim);

        while (itc < BroydenIter_ ){
            itc += 1;
            step_5 = false;
            fx = Fun(x);
            norm_fx = fx.norm();

            // std::cout << "Current iteration: " << itc << std::endl;
            // std::cout << "||f(x)|| = " << norm_fx << std::endl;

            // step 2
            if (norm_fx < tol_) {this->solution_ = x; return;}

            // step 3
            // compute s_B_n+1
            Eigen::GMRES<DMatrix<double>> solver(B);
            d_B = solver.solve(-fx);   // solve linear system

            auto z = x + d_B;
            fz = Fun(z);
            norm_fz = fz.norm();

            if (norm_fz > tau*(norm_fx)){
                d = d_B;
                // and go to step 4
            }
            else {
                d_MB = solver.solve(-fz);
                if (Fun(x + d_B + d_MB).norm() <= rho*norm_fx - sigma_1*(d_B + d_MB).squaredNorm()){
                    d = d_B + d_MB;
                    alfa_k = 1.;
                    // and go to step 5
                    step_5 = true;
                }
                else d = d_B; // and go to step 4
            }

            // step 4
            if (step_5 == false){
                if (Fun(x+d).norm() <= rho*norm_fx - sigma_1*d.squaredNorm())
                    alfa_k = 1.;
                else {
                    alfa_k = 1.;
                    while (Fun(x+alfa_k*d).norm() > norm_fx - sigma_2*(alfa_k*d).squaredNorm() + eps(itc)*norm_fx)
                        alfa_k *= r;
                    // std::cout << "iteration " << itc << " alfa = " << alfa_k << std::endl;
                    // compute alfa_k = max{1, r, r^2, ...} s.t.
                    // f_(x+alfa_k*d.col(n)).norm() <= f_(x).norm() - sigma_2*(alfa_k*d.col(n)).norm() + eps(n)*f_(x).norm()
                }
            }

            // step 5
            auto x_new = x + alfa_k*d;

            // step 6
            // compute B_n+1
            s = x_new - x;
            y = Fun(x_new) - fx;
            B = B + theta_k * (y - B*s)*s.transpose() / s.squaredNorm();

            DMatrix<double> error_ = solution_ex - x;
            error_L2(itc) = (this->mass_ * error_.cwiseProduct(error_)).sum();

            // std::cout << "\nBroyden iter: " << itc << std::endl;
            // std::cout << "Error: " << error_L2(itc) << std::endl;
            // std::cout << "Broyden -> || Au - f || = " << norm_fx << std::endl;
            // std::cout << "Broyden -> ||Increment|| = " << s.norm() << std::endl;
            if (error_L2(itc) < tol_) break;

            x = x_new;
        } 
        
        this->solution_ = x;
        this->success = true;

        // save convergence test results
        std::ofstream file("convergence_test_broyden.txt");    //it will be exported in the current build directory
        if (file.is_open()){
            file << error_L2;
            file.close();
        } else {
            std::cerr << "broyden unable to save convergence test" << std::endl;
        }

        return;
    }

    template <typename PDE>
    void solveNewton(PDE& pde) {

        static_assert(is_pde<PDE>::value, "pde is not a valid PDE object");
        if (!this->is_init) throw std::runtime_error("solver must be initialized first!");

        // exact solution for the convergence test
        auto solutionExpr = [](SVector<2> x) -> double { return 3*x[0]*x[0] + 2*x[1]*x[1]; };
        DMatrix<double> nodes_ = pde.dof_coords();
        DMatrix<double> solution_ex(nodes_.rows(), 1);
        for (int i = 0; i < nodes_.rows(); ++i) {
            solution_ex(i) = solutionExpr(nodes_.row(i));
        }
        // vector for convergence test
        DVector<double> error_L2 = DVector<double>::Zero(BroydenIter_);

        // first solve the linear associated pde (initial guess = 0)
        
        this->solution_ = pde.initial_condition();
        DVector<double> f_prev = pde.initial_condition();   // solution at the previous step
        // error at zeroth iteration
        DMatrix<double> error_ = solution_ex - this->solution_;
        error_L2(0) = (this->mass_ * error_.cwiseProduct(error_)).sum();
        // std::cout << "\nNewton iter: 0" << std::endl;
        // std::cout << "Error: " << error_L2(0) << std::endl;

        // declare the known term that will go on the right hand side and will contain h'
        NonLinearReactionPrime<DomainType::local_dimension, ReferenceBasis> h_prime;
        auto Lprime = non_linear_op<FEM>(h_prime);

        DVector<double> force_backup = this->force_;

        Eigen::SparseLU<Eigen::SparseMatrix<double>, Eigen::COLAMDOrdering<int>> solver;
        this->solution_.resize(this->n_dofs_,1);

        // Execute nonlinear loop to solve nonlinear system
        std::size_t i;
        for (i = 1; i < NewtonIter_; ++i) {

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

            // DVector<double> sN = solver.solve(this->force_ - this->stiff_*f_prev);  // Newton direction
            // LINE SEARCH O TRUST REGION
            // this->solution_ += sN;
            this->solution_ = solver.solve(this->force_);

            // Check convergence to stop early
            double incr = (this->solution_ - f_prev).norm();

            DMatrix<double> error_ = solution_ex - this->solution_;
            error_L2(i) = (this->mass_ * error_.cwiseProduct(error_)).sum();

            // std::cout << "\nNewton iter: " << i << std::endl;
            // std::cout << "Error: " << error_L2(i) << std::endl;
            // std::cout << "Newton -> || Au - f || = " << (this->stiff_ * this->solution_ - this->force_ ).norm() << std::endl;
            // std::cout << "Newton -> ||Increment|| = " << incr << std::endl;
            if (error_L2(i) < tol_) break;
            f_prev = this->solution_;

        }
        this->success = true;

        //save convergence test results
        std::ofstream file("convergence_test_newtonPDE.txt");    //it will be exported in the current build directory
        if (file.is_open()){
            file << error_L2;
            file.close();
        } else {
            std::cerr << "newtonPDE unable to save convergence test" << std::endl;
        }

        return;
    } // end solve

};

}   // namespace core
}   // namespace fdapde

#endif   // __FEM_NONLINEAR_SOLVER_H__
