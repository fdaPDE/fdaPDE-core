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

#ifndef __FEM_SUPG_H__
#define __FEM_SUPG_H__

#include <type_traits>

#include "../../utils/symbols.h"
#include "../../fields/matrix_field.h"
#include "../../pde/differential_expressions.h"
#include "../../pde/differential_operators.h"
#include "../fem_symbols.h"

namespace fdapde {
namespace core {
// rho = {1, GLS}, {0, SUPG}, {-1, DW}
template <typename T> class SUPG<FEM, T> : public DifferentialExpr<SUPG<FEM, T>> {
    // todo: perform compile-time sanity checks
/*    static_assert(
      std::is_base_of<VectorBase, T>::value ||   // space-varying case
      is_eigen_vector<T>::value);                // constant coefficient case*/
private:
    double delta_ = 0.25;        // dim.less parameter for stabilization (not known a priori and should be tuned)
    T data_;        // tuple containing the parameters of the kind <mu_, b_, c_>
public:
    enum {
        is_space_varying = std::is_base_of<VectorBase, std::decay_t<decltype(std::get<0>(data_))>>::value || std::is_base_of<VectorBase, std::decay_t<decltype(std::get<1>(data_))>>::value,
        is_symmetric = false
    };
    // constructor
    SUPG() = default;
    explicit SUPG(const T& data) : data_(data) { }

    // provides the operator's weak form
    template <typename... Args> auto integrate(const std::tuple<Args...>& mem_buffer) const {

        IMPORT_FEM_MEM_BUFFER_SYMBOLS(mem_buffer);

        auto mu = std::get<0>(data_);
        auto b = std::get<1>(data_);

        return  delta_*(*h)/b.norm() * (
                -0.5 * div(b) * div(mu * (invJ * nabla_psi_j)) * psi_i - div(mu * (invJ * nabla_psi_j)) * (invJ * nabla_psi_i).dot(b)
                + 0.5 * div(b) * (invJ * nabla_psi_j).dot(b) * psi_i + (invJ * nabla_psi_j).dot(b) * (invJ * nabla_psi_i).dot(b)
                // missing reaction part
                );
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __FEM_SUPG_H__
