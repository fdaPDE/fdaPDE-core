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

#include <fdaPDE/geometric_finite_elements_fem.h>

#ifndef __FDAPDE_GEOMETRIC_FINITE_ELEMENTS_MODULE_H__
#    error "The FEM bridge must import the geometric finite-elements module."
#endif

#ifndef __FDAPDE_FINITE_ELEMENTS_MODULE_H__
#    error "The FEM bridge must import the finite-elements module."
#endif

#include <array>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace {

using HeaderSpace = fdapde::FeSpace<fdapde::Triangulation<2, 3>, fdapde::FeP<1, 1>>;
using HeaderPacket =
  decltype(fdapde::gfe::p1_fem_cell_quadrature(std::declval<const HeaderSpace&>(), std::declval<std::size_t>()));

static_assert(std::is_same_v<HeaderPacket, fdapde::gfe::P1FEMCellQuadrature<2, 3, 3>>);
static_assert(std::is_same_v<decltype(HeaderPacket::dofs), std::array<std::size_t, 3>>);
static_assert(std::is_same_v<decltype(HeaderPacket::physical_weight_gradients), std::array<std::array<double, 3>, 3>>);
static_assert(std::is_same_v<decltype(HeaderPacket::barycentric_weights), std::array<std::array<double, 3>, 3>>);
static_assert(std::is_same_v<decltype(HeaderPacket::integration_weights), std::array<double, 3>>);

}   // namespace
