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

#ifndef __FEM_DT_H__
#define __FEM_DT_H__

#include "../../pde/differential_expressions.h"
#include "../../pde/differential_operators.h"
#include "../fem_symbols.h"

namespace fdapde {
namespace core {

// time derivative operator.
template <> struct dT<FEM> : public DifferentialExpr<dT<FEM>> {
    enum {
        is_space_varying = false,
        is_symmetric = true
    };
    // return zero field
    template <typename... Args> auto integrate([[maybe_unused]] const std::tuple<Args...>& mem_buffer) const {
        return ScalarField<std::tuple_element_t<0, std::tuple<Args...>>::PtrType::input_space_dimension>::Zero();
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __FEM_DT_H__
