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

#ifndef __FEM_SUPG_RHS_H__
#define __FEM_SUPG_RHS_H__

#include <type_traits>

#include "../../utils/symbols.h"
#include "../../fields/matrix_field.h"
#include "../../pde/differential_expressions.h"
#include "../../pde/differential_operators.h"
#include "../fem_symbols.h"

namespace fdapde {
namespace core {
template <typename T> class SUPG_RHS<FEM, T> : public DifferentialExpr<SUPG_RHS<FEM, T>> {
    // todo: perform compile-time sanity checks
private:
    double delta_ = 0.25;
    T data_;    // tuple containing both transport and forcing data ( <b_, force_> )
public:
    enum {
        is_space_varying = std::is_base_of<VectorBase, std::decay_t<decltype(std::get<0>(data_))>>::value || std::is_base_of<VectorBase, std::decay_t<decltype(std::get<1>(data_))>>::value,
        is_symmetric = false
    };
    // constructor
    SUPG_RHS() = default;
    explicit SUPG_RHS(const T& data) : data_(data) { }

    // provides the operator's weak form
    template <typename... Args> auto integrate(const std::tuple<Args...>& mem_buffer) const {

        IMPORT_FEM_MEM_BUFFER_SYMBOLS(mem_buffer);

        auto b = std::get<0>(data_);
        auto force = std::get<1>(data_);

        return  delta_*(*h)/b.norm() * (0.5 * div(b) * force * psi_i + force * (invJ * nabla_psi_i).dot(b));
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __FEM_SUPG_RHS_H__
