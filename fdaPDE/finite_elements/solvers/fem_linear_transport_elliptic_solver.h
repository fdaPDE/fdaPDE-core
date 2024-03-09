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

    typedef std::tuple<Ts...> SolverArgs;
    enum { fem_order = std::tuple_element <0, SolverArgs>::type::value };
    typedef D DomainType;
    using FunctionalBasis = LagrangianBasis<DomainType, fem_order>;
    using ReferenceBasis = typename FunctionalBasis::ReferenceBasis;
    using Quadrature = typename ReferenceBasis::Quadrature;

    template <typename PDE> void solve(const PDE& pde) {
        fdapde_static_assert(is_pde<PDE>::value, THIS_METHOD_IS_FOR_PDE_ONLY);
        if (!this->is_init) throw std::runtime_error("solver must be initialized first!");

        // define eigen system solver, exploit spd of operator if possible
        typedef Eigen::SparseLU<SpMatrix<double>, Eigen::COLAMDOrdering<int>> SystemSolverType;
        SystemSolverType solver;

        SpMatrix<double> stab_(this->n_dofs_, this->n_dofs_);
        // IF THERE IS AN ADVECTION TERM CHECK IF THE EQUATION NEEDS STABILIZATION
        if constexpr (fdapde::core::is_advection<E>::value){

            // initialize empty value with the CORRECT TYPE using default constructor
            auto mu_temp = std::tuple_element_t<1, SolverArgs>();
            auto b_temp = std::tuple_element_t<2, SolverArgs>();

            // retrieve the values of the PDE parameters from the singleton
            PDEparameters<decltype(mu_temp), decltype(b_temp)> &PDEparams = PDEparameters<decltype(mu_temp),
                    decltype(b_temp)>::getInstance(mu_temp, b_temp);

            auto mu = std::get<0>(PDEparams.getData());
            auto b = std::get<1>(PDEparams.getData());

            // std::cout << "type of b is: " << typeid(b).name() << std::endl;
            // std::cout << "b.norm() is: " << b.norm() << std::endl;

            // add stabilization method IF IT IS NECESSARY
            // I cannot avoid having introduced a getter in lagrangian_basis.h, domain_ is declared private
            double h = this->basis_.get_element_size(); //todo: take the maximum of every element...
            // double Pe = b.norm() * h / (2 * mu);
            // todo: check Peclet number and decide whether to use stabilizer

            Assembler<FEM, DomainType, ReferenceBasis, Quadrature> assembler(pde.domain(), this->integrator_,
                                                                             this->n_dofs_, this->dofs_);

            // streamline diffusion
            auto StreamDiff = streamline_diffusion<FEM>(b);
            stab_ = assembler.discretize_operator(StreamDiff); // stabilization matrix

            // strong staibilizer (GLS-SUPG-DW) [...]
            // auto StrongStab = SUPG<FEM, decltype(PDEparams.getData())>(PDEparams.getData());
            // stab_ = assembler.discretize_operator(StrongStab); // stabilization matrix
        }

        solver.compute(this->stiff_ + stab_);
        // stop if something was wrong
        if (solver.info() != Eigen::Success) {
            this->success = false;
            return;
        }

        // solve FEM linear system arising from the generalized Petrov Galerkin
        // streamline diffusion
        this->solution_ = solver.solve(this->force_);
        // strong stabilizers [...]

        this->success = true;
        return;
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __FEM_LINEAR_TRANSPORT_ELLIPTIC_SOLVER_H__
