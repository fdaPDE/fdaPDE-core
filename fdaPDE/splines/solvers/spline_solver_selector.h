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

#ifndef __SPLINE_SOLVER_SELECTOR_H__
#define __SPLINE_SOLVER_SELECTOR_H__

#include "../../utils/traits.h"
#include "../spline_symbols.h"
#include "spline_linear_elliptic_solver.h"

namespace fdapde {
namespace core {

// selects solver type depending on properties of operator E, carries domain D and forcing F to solver
template <typename D, typename E, typename F, typename... Ts> struct pde_solver_selector<SPLINE, D, E, F, Ts...> {
    using type = typename switch_type<
      switch_type_case<!is_parabolic<E>::value, SplineLinearEllipticSolver <D, E, F, Ts...>>
      >::type;
};

}   // namespace core
}   // namespace fdapde

#endif   // __SPLINE_SOLVER_SELECTOR_H__
