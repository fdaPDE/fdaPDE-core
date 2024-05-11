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

#ifndef __FEM_SUPG_ADV_DIFF_REACT_H__
#define __FEM_SUPG_ADV_DIFF_REACT_H__

#include <type_traits>

#include "../../utils/symbols.h"
#include "../../fields/matrix_field.h"
#include "../../pde/differential_expressions.h"
#include "../../pde/differential_operators.h"
#include "../fem_symbols.h"

namespace fdapde {
namespace core {

template <typename T> class SUPG_ADV_DIFF_REACT<FEM, T> : public DifferentialExpr<SUPG_ADV_DIFF_REACT<FEM, T>> {
private:
    T data_;        // tuple containing the parameters <mu, b, c, normb, delta>
public:
    enum {
        is_space_varying = std::is_base_of<VectorBase, std::decay_t<decltype(std::get<0>(data_))>>::value || std::is_base_of<VectorBase, std::decay_t<decltype(std::get<1>(data_))>>::value,
        is_symmetric = false
    };
    
    SUPG_ADV_DIFF_REACT() = default;
    explicit SUPG_ADV_DIFF_REACT(const T& data) : data_(data) { }

    // provides the operator's weak form
    template <typename... Args> auto integrate(const std::tuple<Args...>& mem_buffer) const {

        IMPORT_FEM_MEM_BUFFER_SYMBOLS(mem_buffer);

        double constexpr dataSize = std::tuple_size<decltype(data_)>::value;

        auto mu = std::get<0>(data_);
        auto b = std::get<1>(data_);
        auto c = std::get<2>(data_);
        double normb = std::get<3>(data_);
        double delta = std::get<4>(data_);

        return  delta * 0.5 / normb * h * (
                div(mu * (invJ * nabla_psi_j)) * (invJ * nabla_psi_i).dot(b)    // SUPG diffusion
                + (invJ * nabla_psi_j).dot(b) * (invJ * nabla_psi_i).dot(b)     // SUPG transport
                + c * psi_j * (invJ * nabla_psi_i).dot(b)                       // SUPG reaction
                );
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __FEM_SUPG_ADV_DIFF_REACT_H__
