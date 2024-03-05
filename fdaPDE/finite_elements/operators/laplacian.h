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

#ifndef __FEM_LAPLACIAN_H__
#define __FEM_LAPLACIAN_H__

#include <type_traits>

#include "../../pde/differential_expressions.h"
#include "../../pde/differential_operators.h"
#include "../fem_symbols.h"

namespace fdapde {
namespace core {

// laplacian operator (isotropic and stationary diffusion)
template <> struct Laplacian<FEM> : public DifferentialExpr<Laplacian<FEM>> {
    enum {
        is_space_varying = false,
        is_symmetric = true
    };

    // provides the operator's weak form
    template <typename... Args> auto integrate(const std::tuple<Args...>& mem_buffer) const {
        // unpack mem_buffer tuple
        auto nabla_psi_i = std::get<2>(mem_buffer);
        auto nabla_psi_j = std::get<3>(mem_buffer);
        auto invJ        = std::get<4>(mem_buffer);   // affine map to reference element
        // isotropic unitary diffusion: -(\Nabla psi_i).dot(\Nabla psi_j)
        return -(invJ * nabla_psi_i).dot(invJ * nabla_psi_j);
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __FEM_LAPLACIAN_H__
