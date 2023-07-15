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

#ifndef __GRADIENT_H__
#define __GRADIENT_H__

#include <type_traits>

#include "../../utils/symbols.h"
#include "../../utils/compile_time.h"
#include "../../fields/vector_field.h"
#include "bilinear_form_expressions.h"

namespace fdapde {
namespace core {

// gradient operator (transport term).
template <typename T> class Gradient : public BilinearFormExpr<Gradient<T>> {
    // perform compile-time sanity checks
    static_assert(
      std::is_base_of<VectorBase, T>::value ||   // space-varying case
      is_eigen_vector<T>());                     // constant coefficient case
   private:
    T b_;   // transport vector (either constant or space-varying)
   public:
    // constructor
    Gradient() = default;
    explicit Gradient(const T& b) : b_(b) { }

    std::tuple<Gradient<T>> get_operator_type() const { return std::make_tuple(*this); }
    enum {
      is_space_varying = std::is_base_of<VectorBase, T>::value,
      is_symmetric = false
    };

    // provides the operator's weak form
    template <typename... Args> auto integrate(const std::tuple<Args...>& mem_buffer) const {
        IMPORT_MEM_BUFFER_SYMBOLS(mem_buffer);
        return psi_i * (invJ * NablaPsi_j).dot(b_);   // \psi_i*b.dot(\nabla \psi_j)
    }
};
// template argument deduction rule
template <typename T> Gradient(const T&) -> Gradient<T>;
// factory method
template <typename T> Gradient<T> grad(const T& t) { return Gradient<T>(t); }
  
}   // namespace core
}   // namespace fdapde

#endif   // __GRADIENT_H__
