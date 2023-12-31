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

#ifndef __PDE_SYMBOLS_H__
#define __PDE_SYMBOLS_H__

#include "../splines/spline_symbols.h"
#include "../finite_elements/fem_symbols.h"
#include "../utils/traits.h"

namespace fdapde {
namespace core {

// a Partial Differential Equation Lf = u solved with strategy S
template <typename D, typename E, typename F, typename S, typename... Ts> class PDE;

// PDE-detection type trait.
template <typename T> struct is_pde {
    static constexpr bool value = fdapde::is_instance_of<T, PDE>::value;
};
  
// selects a proper solver for a given resolution strategy S and operator E.
template <typename S, typename D, typename E, typename F, typename... Ts> struct pde_solver_selector { };
  
// possible strategies for the evaluation of a functional basis over a physical domain
enum eval { pointwise, areal };
template <typename T> struct pointwise_evaluation;
template <typename T> struct areal_evaluation;
  
}   // namespace core
}   // namespace fdapde

#endif   // __PDE_SYMBOLS_H__
