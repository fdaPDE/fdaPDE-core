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

#ifndef __SPLINE_BILAPLACIAN_H__
#define __SPLINE_BILAPLACIAN_H__

#include <type_traits>

#include "../../pde/differential_expressions.h"
#include "../../pde/differential_operators.h"
#include "../spline_symbols.h"

namespace fdapde {
namespace core {

// bilaplacian operator in one dimension (d^4/dt^4 f = d^2/dt^2(d^2/dt^2 f))
template <> struct BiLaplacian<SPLINE> : public DifferentialExpr<BiLaplacian<SPLINE>> {
    enum {
        is_space_varying = false,
        is_symmetric = true
    };

    // provides the operator's weak form: (\psi_i)_tt * (\psi_j)_tt
    template <typename... Args> auto integrate(const std::tuple<Args...>& mem_buffer) const {
        IMPORT_SPLINE_MEM_BUFFER_SYMBOLS(mem_buffer);
        return -(psi_i->template derive<2>() * psi_j->template derive<2>());
    }
};

}   // namespace core
}   // namespace fdapde

#endif   // __SPLINE_BILAPLACIAN_H__
