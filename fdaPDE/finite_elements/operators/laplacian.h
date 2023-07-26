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

#ifndef __LAPLACIAN_H__
#define __LAPLACIAN_H__

#include <type_traits>

#include "../../utils/symbols.h"
#include "../../fields/matrix_field.h"
using fdapde::core::MatrixField;
#include "bilinear_form_expressions.h"
using fdapde::core::BilinearFormExpr;

namespace fdapde {
namespace core {

// laplacian operator (isotropic and stationary diffusion)
struct Laplacian : public BilinearFormExpr<Laplacian> {
    std::tuple<Laplacian> get_operator_type() const { return std::make_tuple(*this); }
    enum {
      is_space_varying = false,
      is_symmetric = true
    };
  
    // provides the operator's weak form
    template <typename... Args> auto integrate(const std::tuple<Args...>& mem_buffer) const {
        IMPORT_MEM_BUFFER_SYMBOLS(mem_buffer);
	// isotropic unitary diffusion: -(\Nabla psi_i).dot(\Nabla psi_j)
	return -(invJ * nabla_psi_i).dot(invJ * nabla_psi_j);
    }
};

// factory method
Laplacian laplacian() { return Laplacian(); }

}   // namespace core
}   // namespace fdapde

#endif   // __LAPLACIAN_H__
