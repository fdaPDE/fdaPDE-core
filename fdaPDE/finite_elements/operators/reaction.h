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

#ifndef __FEM_REACTION_H__
#define __FEM_REACTION_H__

#include <type_traits>

#include "../../fields/scalar_field.h"
#include "../../mesh/element.h"
#include "../../pde/differential_expressions.h"
#include "../../pde/differential_operators.h"
#include "../fem_symbols.h"

namespace fdapde {
namespace core {

// reaction operator
template <typename T> class Reaction<FEM, T> : public DifferentialExpr<Reaction<FEM, T>> {
    // perform compile-time sanity checks
    static_assert(
      std::is_base_of<ScalarBase, T>::value ||   // space-varying case
      std::is_floating_point<T>::value);         // constant coefficient case
   private:
    T c_;   // reaction term
   public:
    enum {
        is_space_varying = std::is_base_of<ScalarBase, T> ::value,
        is_symmetric = true
    };

    // constructors
    Reaction() = default;
    Reaction(const T& c) : c_(c) {};
    // provides the operator's weak form
    template <typename... Args> auto integrate(const std::tuple<Args...>& mem_buffer) const {
        IMPORT_FEM_MEM_BUFFER_SYMBOLS(mem_buffer);
	return c_ * psi_i * psi_j;   // c*\psi_i*\psi_j
    }
};
  
}   // namespace core
}   // namespace fdapde

#endif   // __FEM_REACTION_H__
