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

#ifndef __FEM_SOLVER_SELECTOR_H__
#define __FEM_SOLVER_SELECTOR_H__

#include "../../utils/traits.h"
#include "../fem_symbols.h"
#include "fem_linear_elliptic_solver.h"
#include "fem_linear_parabolic_solver.h"
#include "fem_nonlinear_fixedpoint_solver.h"
#include "fem_nonlinear_broyden_solver.h"
#include "fem_nonlinear_newton_solver.h"
#include "fem_nonlinear_solver.h"
#include "fem_linear_transport_elliptic_solver.h"   //ADDED
#include "euler_semi_implicit.h"
#include "euler_implicit_fixedpoint.h"
#include "euler_implicit_newton.h"
#include "crank_nicolson_semi_implicit.h"

namespace fdapde {
namespace core {

// selects solver type depending on properties of operator E, carries domain D and forcing F to solver
template <typename D, typename E, typename F, typename... Ts> struct pde_solver_selector<FEM, D, E, F, Ts...> {
    using type = typename switch_type<
      switch_type_case< is_parabolic<E>::value && !is_nonlinear<E>::value, FEMLinearParabolicSolver<D, E, F, Ts...>>,
      // switch_type_case< is_parabolic<E>::value && is_nonlinear<E>::value, FEMEulerSemiImplicit<D, E, F, Ts...>>,
      // switch_type_case< is_parabolic<E>::value && is_nonlinear<E>::value, FEMEulerImplicitFixedPoint<D, E, F, Ts...>>,
      // switch_type_case< is_parabolic<E>::value && is_nonlinear<E>::value, FEMEulerImplicitNewton<D, E, F, Ts...>>,
      switch_type_case< is_parabolic<E>::value && is_nonlinear<E>::value, FEMCrankNicolsonSemiImplicit<D, E, F, Ts...>>,
      // switch_type_case<!is_parabolic<E>::value && is_nonlinear<E>::value, FEMNonLinearBroydenSolver<D, E, F, Ts...>>,
      // switch_type_case<!is_parabolic<E>::value && is_nonlinear<E>::value, FEMNonLinearFixedPointSolver<D, E, F, Ts...>>,
      switch_type_case<!is_parabolic<E>::value && is_nonlinear<E>::value, FEMNonLinearNewtonSolver<D, E, F, Ts...>>,
      // switch_type_case<!is_parabolic<E>::value && is_nonlinear<E>::value, FEMNonLinearSolver<D, E, F, Ts...>>,

      switch_type_case<!is_parabolic<E>::value && !is_nonlinear<E>::value, FEMLinearTransportEllipticSolver<D, E, F, Ts...>>
      // switch_type_case<!is_parabolic<E>::value && !is_nonlinear<E>::value, FEMLinearEllipticSolver<D, E, F, Ts...>>
      >::type;
};

}   // namespace core
}   // namespace fdapde

#endif   // __FEM_SOLVER_SELECTOR_H__
