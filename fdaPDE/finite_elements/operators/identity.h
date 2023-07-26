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

#ifndef __IDENTITY_H__
#define __IDENTITY_H__

#include <type_traits>

#include "../../fields/scalar_field.h"
#include "../../mesh/element.h"
#include "bilinear_form_expressions.h"

namespace fdapde {
namespace core {

// Identity operator (reaction term)
template <typename T> class Identity : public BilinearFormExpr<Identity<T>> {
    // perform compile-time sanity checks
    static_assert(
      std::is_base_of<ScalarBase, T>::value ||   // space-varying case
      std::is_floating_point<T>::value);         // constant coefficient case
   private:
    T c_;   // reaction term
   public:
    // constructors
    Identity() = default;
    Identity(const T& c) : c_(c) {};

    std::tuple<Identity<T>> get_operator_type() const { return std::make_tuple(*this); }
    enum {
      is_space_varying = std::is_base_of<ScalarBase, T>::value,
      is_symmetric = true
    };

    // provides the operator's weak form
    template <typename... Args> auto integrate(const std::tuple<Args...>& mem_buffer) const {
        IMPORT_MEM_BUFFER_SYMBOLS(mem_buffer);
	return c_ * psi_i * psi_j;   // c*\psi_i*\psi_j
    }
};
  
// template argument deduction guide
template <typename T> Identity(const T&) -> Identity<T>;
// factory method
template <typename T> Identity<T> I(const T& t) { return Identity<T>(t); }  
  
}   // namespace core
}   // namespace fdapde

#endif   // __IDENTITY_H__
