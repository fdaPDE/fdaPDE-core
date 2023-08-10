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

#ifndef __FEM_DIFFUSION_H__
#define __FEM_DIFFUSION_H__

#include <type_traits>

#include "../../utils/symbols.h"
#include "../../fields/matrix_field.h"
#include "../../pde/differential_expressions.h"
#include "../../pde/differential_operators.h"
#include "../fem_symbols.h"

namespace fdapde {
namespace core {

// diffusion operator -div(K*grad(.)) (anisotropic and non-stationary diffusion)
template <typename T> class Diffusion<FEM, T> : public DifferentialExpr<Diffusion<FEM, T>> {
    // perform compile-time sanity checks
    static_assert(
      std::is_base_of<MatrixBase, T>::value ||            // space-varying case
      std::is_base_of<Eigen::MatrixBase<T>, T>::value);   // constant coefficient case
   private:
    T K_;   // diffusion tensor (either constant or space-varying)
   public:
    enum {
        is_space_varying = std::is_base_of<MatrixBase, T> ::value,
        is_symmetric = true
    };

    // constructor
    Diffusion() = default;
    explicit Diffusion(const T& K) : K_(K) { }
    // provides the operator's weak form
    template <typename... Args> auto integrate(const std::tuple<Args...>& mem_buffer) const {
        IMPORT_FEM_MEM_BUFFER_SYMBOLS(mem_buffer);
	// non unitary or anisotropic diffusion: (\Nabla psi_i)^T*K*(\Nabla \psi_j)
	return -(invJ * nabla_psi_i).dot(K_ * (invJ * nabla_psi_j));
    }
};
  
}   // namespace core
}   // namespace fdapde

#endif   // __FEM_DIFFUSION_H__
