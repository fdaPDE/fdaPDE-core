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

#ifndef __FDAPDE_MANIFOLD_H__
#define __FDAPDE_MANIFOLD_H__

#include "header_check.h"

namespace fdapde {
namespace manifold {

template <typename Geometry> using point_t = typename std::remove_cvref_t<Geometry>::Point;
template <typename Geometry> using tangent_t = typename std::remove_cvref_t<Geometry>::Tangent;

template <typename Geometry>
concept GeometryTypes = requires {
    typename point_t<Geometry>;
    typename tangent_t<Geometry>;
} && std::copyable<point_t<Geometry>> && std::copyable<tangent_t<Geometry>>;

template <typename Geometry>
concept FirstOrderGeometry =
  GeometryTypes<Geometry> && requires(
                               const std::remove_cvref_t<Geometry>& geometry, const point_t<Geometry>& point,
                               const tangent_t<Geometry>& u, const tangent_t<Geometry>& v, double alpha, double beta) {
      { geometry.dimension() } -> std::same_as<std::size_t>;
      { geometry.inner_product(point, u, v) } -> std::convertible_to<double>;
      { geometry.norm(point, u) } -> std::convertible_to<double>;
      { geometry.project(point, u) } -> std::same_as<tangent_t<Geometry>>;
      { geometry.zero_tangent(point) } -> std::same_as<tangent_t<Geometry>>;
      { geometry.linear_combination(point, alpha, u, beta, v) } -> std::same_as<tangent_t<Geometry>>;
      { geometry.retract(point, u, alpha) } -> std::same_as<point_t<Geometry>>;
  };

template <typename Geometry>
concept VectorTransportGeometry =
  FirstOrderGeometry<Geometry> && requires(
                                    const std::remove_cvref_t<Geometry>& geometry, const point_t<Geometry>& from,
                                    const point_t<Geometry>& to, const tangent_t<Geometry>& tangent) {
      { geometry.transport(from, to, tangent) } -> std::same_as<tangent_t<Geometry>>;
  };

}   // namespace manifold
}   // namespace fdapde

#endif   // __FDAPDE_MANIFOLD_H__
