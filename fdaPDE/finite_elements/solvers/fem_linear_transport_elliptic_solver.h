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

#ifndef __FEM_LINEAR_TRANSPORT_ELLIPTIC_SOLVER_H__
#define __FEM_LINEAR_TRANSPORT_ELLIPTIC_SOLVER_H__

#include <exception>

#include "../../utils/symbols.h"
#include "fem_solver_base.h"
#include "../fem_symbols.h"

using fdapde::core::PDEparameters;

namespace fdapde {
namespace core {

template <typename D, typename E, typename F, typename... Ts>
struct FEMLinearTransportEllipticSolver : public FEMSolverBase<D, E, F, Ts...> {

    using Base = FEMSolverBase<D, E, F, Ts...>;
    FEMLinearTransportEllipticSolver(const D& domain) : Base(domain){ }
    FEMLinearTransportEllipticSolver(const D& domain, const DMatrix<short int>& BMtrx) : Base(domain, BMtrx){ }

    typedef std::tuple<Ts...> SolverArgs;
    enum { fem_order = std::tuple_element <0, SolverArgs>::type::value };
    typedef D DomainType;
    using FunctionalBasis = LagrangianBasis<DomainType, fem_order>;
    using ReferenceBasis = typename FunctionalBasis::ReferenceBasis;
    using Quadrature = typename ReferenceBasis::Quadrature;

    double delta_ = 1.;
    void set_stab_param(const double delta) { delta_ = delta; }

    template <typename PDE> void solve(const PDE& pde) {
        fdapde_static_assert(is_pde<PDE>::value, THIS_METHOD_IS_FOR_PDE_ONLY);
        if (!this->is_init) throw std::runtime_error("solver must be initialized first!");

        // define eigen system solver, exploit spd of operator if possible
        typedef Eigen::SparseLU<SpMatrix<double>, Eigen::COLAMDOrdering<int>> SystemSolverType;
        SystemSolverType solver;

        SpMatrix<double> stab_(this->n_dofs_, this->n_dofs_);
        stab_.setZero();
        DVector<double> stab_rhs_ = DVector<double>::Zero(this->n_dofs_);

        // if there's an advection term -> stabilization
        if constexpr (fdapde::core::is_advection<E>::value){

            // diffusion - advection - reaction equation
            if constexpr (fdapde::core::is_reaction<E>::value){ 

                // initialize empty value with the CORRECT TYPE using default constructor
                auto mu_temp = std::tuple_element_t<1, SolverArgs>();
                auto b_temp = std::tuple_element_t<2, SolverArgs>();
                auto c_temp = std::tuple_element_t<3, SolverArgs>();

                // retrieve the values of the PDE parameters from the singleton
                PDEparameters<decltype(mu_temp), decltype(b_temp), decltype(c_temp)> &PDEparams =
                        PDEparameters<decltype(mu_temp), decltype(b_temp), decltype(c_temp)>::getInstance(mu_temp, b_temp, c_temp);

                auto mu = std::get<0>(PDEparams.getData());
                auto b = std::get<1>(PDEparams.getData());
                auto c = std::get<2>(PDEparams.getData());

                Assembler<FEM, DomainType, ReferenceBasis, Quadrature> assembler(pde.domain(), this->integrator_,
                                                                                 this->n_dofs_, this->dofs_);

                double normb = 0;
                if constexpr (std::is_base_of<VectorBase, decltype(b)>::value) {
                    normb = assembler.normVectorField(b);
                } else {
                    normb = b.norm();
                }

                double h = std::sqrt(this-> basis_.get_element_size() * 2); // assuming triangles

                double Pe = 0;
                std::cout << "mu is a double: " << std::is_same<decltype(mu), double>::value << std::endl;
                if constexpr (std::is_same<decltype(mu), double>::value) {
                    Pe = normb * h * 0.5 / mu;
                }else{
                    Pe = normb * h * 0.5 / mu.norm();
                }
                // Pe = normb * h * 0.5 / mu.norm();
                std::cout << "Pe: " << Pe << std::endl;

                if (Pe > 1) {
                    auto SUPGtuple = std::make_tuple(mu, b, c, normb, delta_);
                    auto StrongStab = SUPG_ADV_DIFF_REACT<FEM, decltype(SUPGtuple)>(SUPGtuple);
                    
                    stab_ = assembler.discretize_operator(StrongStab);
                    stab_rhs_ = assembler.discretize_SUPG_RHS(b, normb, pde.forcing_data(), delta_);
                }
            
            }  else {   // diffusion - advection equation
                // initialize empty value with the CORRECT TYPE using default constructor
                auto mu_temp = std::tuple_element_t<1, SolverArgs>();
                auto b_temp = std::tuple_element_t<2, SolverArgs>();

                PDEparameters<decltype(mu_temp), decltype(b_temp)> &PDEparams =
                        PDEparameters<decltype(mu_temp), decltype(b_temp)>::getInstance(mu_temp, b_temp);

                auto mu = std::get<0>(PDEparams.getData());
                auto b = std::get<1>(PDEparams.getData());

                Assembler<FEM, DomainType, ReferenceBasis, Quadrature> assembler(pde.domain(), this->integrator_,
                                                                                 this->n_dofs_, this->dofs_);

                double normb = 0;
                if constexpr (std::is_base_of<VectorBase, decltype(b)>::value) {
                    normb = assembler.normVectorField(b);
                } else {
                    normb = b.norm();
                }

                double h = std::sqrt(this-> basis_.get_element_size() * 2);
                double Pe = normb * h / (2 * mu);

                if (Pe > 1) {
                    auto SUPGtuple = std::make_tuple(mu, b, normb, delta_);
                    auto StrongStab = SUPG_ADV_DIFF<FEM, decltype(SUPGtuple)>(SUPGtuple);
                    
                    stab_ = assembler.discretize_operator(StrongStab);
                    stab_rhs_ = assembler.discretize_SUPG_RHS(b, normb, pde.forcing_data(), delta_);
                }
            }   // end is_reaction
        }   // end is_advection

        solver.compute(this->stiff_ + stab_);
        // stop if something was wrong
        if (solver.info() != Eigen::Success) {
            this->success = false;
            return;
        }

        this->solution_ = solver.solve(this->force_ + stab_rhs_);

        this->success = true;
        return;
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __FEM_LINEAR_TRANSPORT_ELLIPTIC_SOLVER_H__
