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

        // retrieve PDE parameters here (how?) !!
        SVector<2> b;  b << 1., 1.;
        double mu = 1e-9;

        // add stabilization method IF IT IS NECESSARY
        // double h = 0.01;    // how can we get the measure of one element here?
        double h = this->basis_.get_element_size();
        double Pe = b.norm()*h/(2*mu);
        //todo: check Peclet number and decide whether to use stabilizer
        Assembler<FEM, DomainType, ReferenceBasis, Quadrature> assembler(pde.domain(), this->integrator_, this->n_dofs_, this->dofs_);

        // streamline diffusion
        auto StreamDiff = streamline_diffusion<FEM>(b);
        SpMatrix<double> stab_ = assembler.discretize_operator(StreamDiff); // stabilization matrix

        // strong staibilizer (GLS-SUPG-DW) [...]

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
