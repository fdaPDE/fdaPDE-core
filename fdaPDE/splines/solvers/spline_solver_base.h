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

#ifndef __SPLINE_SOLVER_BASE_H__
#define __SPLINE_SOLVER_BASE_H__

#include <exception>

#include "../../utils/integration/integrator.h"
#include "../../utils/symbols.h"
#include "../basis/spline_basis.h"
#include "../operators/reaction.h"   // for mass-matrix computation
#include "../spline_assembler.h"
#include "../spline_symbols.h"

namespace fdapde {
namespace core {

// base class for the definition of a general solver based on a B-Spline expansion of the solution
template <typename D, typename E, typename F, typename... Ts> class SplineSolverBase {
   public:
    typedef std::tuple<Ts...> SolverArgs;
    enum {
        spline_order = std::tuple_element<0, SolverArgs>::type::value
    };
    using DomainType = D;
    using FunctionalBasis = SplineBasis<spline_order>;
    using Quadrature = typename FunctionalBasis::Quadrature;
    // constructor
    SplineSolverBase() = default;
    SplineSolverBase(const DomainType& domain) : domain_(&domain), basis_(domain.nodes()) {};
    // getters
    const DMatrix<double>& solution() const { return solution_; }
    const DMatrix<double>& force() const { return force_; }
    const SpMatrix<double>& mass() const { return mass_; }
    const Quadrature& integrator() const { return integrator_; }
    const FunctionalBasis& basis() const { return basis_; }
    std::size_t n_dofs() const { return basis_.size(); }   // number of degrees of freedom (linear system's unknowns)
    DMatrix<double> dofs_coords() { return domain_->nodes(); };
    const SpMatrix<double>& stiff() const { return stiff_; }

    // flags
    bool is_init = false;   // notified true if initialization occurred with no errors
    bool success = false;   // notified true if problem solved with no errors

    template <typename PDE> void init(const PDE& pde) {
        fdapde_static_assert(is_pde<PDE>::value, THIS_METHOD_IS_FOR_PDE_ONLY);
        // assemble discretization matrix for given operator
        Assembler<SPLINE, DomainType, FunctionalBasis, Quadrature> assembler(pde.domain(), integrator_);
        stiff_ = assembler.discretize_operator(pde.differential_operator());
        stiff_.makeCompressed();

        // TODO: add discretization of forcing vector

        // compute mass matrix [R0]_{ij} = \int_{\Omega} \phi_i \phi_j
        mass_ = assembler.discretize_operator(Reaction<SPLINE, double>(1.0));
        is_init = true;
        return;
    }
    template <typename PDE> void set_dirichlet_bc(const PDE& pde) { return; }; // TODO
   protected:
    const DomainType* domain_;
    Quadrature integrator_ {};   // default to a quadrature rule which is exact for the considered spline order
    FunctionalBasis basis_ {};   // basis system defined over the domain of definition
    DMatrix<double> solution_;   // vector of coefficients of the approximate solution
    DMatrix<double> force_;      // discretized force [u]_i = \int_D f*\phi_i
    SpMatrix<double> stiff_;     // [R1_]_{ij} = a(\phi_i, \phi_j), being a(.,.) the bilinear form
    SpMatrix<double> mass_;      // mass matrix, [R0_]_{ij} = \int_D (\phi_i * \phi_j)
};

}   // namespace core
}   // namespace fdapde

#endif   // __SPLINE_SOLVER_BASE_H__
