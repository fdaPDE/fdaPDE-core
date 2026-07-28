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

#ifndef __FDAPDE_GFE_P1_FEM_CELL_QUADRATURE_H__
#define __FDAPDE_GFE_P1_FEM_CELL_QUADRATURE_H__

#include "header_check.h"

namespace fdapde {
namespace gfe {

template <std::size_t LocalDim_, std::size_t EmbedDim_, std::size_t QuadratureSize_> struct P1FEMCellQuadrature {
    static constexpr std::size_t local_dim = LocalDim_;
    static constexpr std::size_t embed_dim = EmbedDim_;
    static constexpr std::size_t quadrature_size = QuadratureSize_;
    static constexpr std::size_t node_count = local_dim + 1;

    std::array<std::size_t, node_count> dofs;
    std::array<std::array<double, node_count>, embed_dim> physical_weight_gradients;
    std::array<std::array<double, node_count>, quadrature_size> barycentric_weights;
    std::array<double, quadrature_size> integration_weights;
};

}   // namespace gfe
}   // namespace fdapde

#endif   // __FDAPDE_GFE_P1_FEM_CELL_QUADRATURE_H__
