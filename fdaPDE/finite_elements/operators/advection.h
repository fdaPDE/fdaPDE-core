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

#ifndef __FEM_ADVECTION_H__
#define __FEM_ADVECTION_H__

#include <type_traits>

#include "../../utils/symbols.h"
#include "../../utils/compile_time.h"
#include "../../fields/vector_field.h"
#include "../../pde/differential_operators.h"
#include "../../pde/differential_expressions.h"
#include "../fem_symbols.h"

namespace fdapde {
namespace core {

// advection operator dot(b, grad()) (transport term)
template <typename T> class Advection<FEM, T> : public DifferentialExpr<Advection<FEM, T>> {
    // perform compile-time sanity checks
    static_assert(
      std::is_base_of<VectorBase, T>::value ||   // space-varying case
      is_eigen_vector<T>::value);                // constant coefficient case
   private:
    T b_;   // transport vector (either constant or space-varying)
   public:
    enum {
        is_space_varying = std::is_base_of<VectorBase, T> ::value,
        is_symmetric = false
    };

    // constructor
    Advection() = default;
    explicit Advection(const T& b) : b_(b) { }
    // provides the operator's weak form
    template <typename... Args> auto integrate(const std::tuple<Args...>& mem_buffer) const {
        IMPORT_FEM_MEM_BUFFER_SYMBOLS(mem_buffer);
        return psi_i * (invJ * nabla_psi_j).dot(b_);   // \psi_i*b.dot(\nabla \psi_j)
    }
};
  
}   // namespace core
}   // namespace fdapde

#endif   // __FEM_ADVECTION_H__
