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

#ifndef __FEM_NON_LINEAR_OPERATOR_H__
#define __FEM_NON_LINEAR_OPERATOR_H__

#include <type_traits>

#include "../../utils/symbols.h"
#include "../../pde/differential_operators.h"
#include "../../pde/differential_expressions.h"
#include "../fem_symbols.h"

namespace fdapde {
namespace core {

// non-linear operator h(x, solution)
template <typename T> class NonLinearOp<FEM, T> : public DifferentialExpr<NonLinearOp<FEM, T>> {
    // perform compile-time sanity checks
    static_assert(
        std::is_invocable<T, std::shared_ptr<DVector<double>>>::value);
   private:
    T h_;
   public:
    enum {
        is_space_varying = false,
        is_symmetric = false
    };

    // constructor
    NonLinearOp() = default;
    explicit NonLinearOp(const T& h) : h_(h) { }

    // provides the operator's weak form
    template <typename... Args> auto integrate(const std::tuple<Args...>& mem_buffer) const {
        IMPORT_FEM_MEM_BUFFER_SYMBOLS(mem_buffer);
        return h_(f) * psi_i * psi_j;
    }
};
  
}   // namespace core
}   // namespace fdapde

#endif   // __FEM_NON_LINEAR_OPERATOR_H__
