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
#include "fem_linear_elliptic_solver.h"
#include "fem_linear_parabolic_solver.h"

namespace fdapde {
namespace core {

// selects solver type depending on operator properties
template <typename E> struct pde_fem_solver_selector {
    using type = typename switch_type<
      switch_type_case<!is_parabolic<E>::value, FEMLinearEllipticSolver>,
      switch_type_case< is_parabolic<E>::value, FEMLinearParabolicSolver> >::type;
};

}   // namespace core
}   // namespace fdapde

#endif   // __FEM_SOLVER_SELECTOR_H__
