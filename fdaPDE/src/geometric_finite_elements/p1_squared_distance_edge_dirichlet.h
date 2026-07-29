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

#ifndef __FDAPDE_GFE_P1_SQUARED_DISTANCE_EDGE_DIRICHLET_H__
#define __FDAPDE_GFE_P1_SQUARED_DISTANCE_EDGE_DIRICHLET_H__

#include "header_check.h"

namespace fdapde {
namespace gfe {

namespace internals {

template <typename Geometry>
void p1_squared_distance_edge_dirichlet_validate(
  const Geometry& geometry, std::span<const typename Geometry::Point> nodal_values,
  const P1LumpedLaplacianStencil& stencil) {
    validate_p1_lumped_laplacian_stencil(stencil);
    if (nodal_values.size() != stencil.node_count()) {
        throw std::invalid_argument("P1 squared-distance edge Dirichlet node and stencil sizes must match");
    }
    for (const auto& node : nodal_values) {
        p1_objective_require_finite_shape(
          node, geometry.order(),
          "P1 squared-distance edge Dirichlet node has incompatible dimensions or nonfinite coefficients");
    }
    for (const auto& edge : stencil.edges) {
        if (edge.stiffness >= 0) {
            throw std::invalid_argument(
              "P1 squared-distance edge Dirichlet requires negative off-diagonal stiffnesses");
        }
    }
}

template <bool WithGradient, typename Geometry>
P1ObjectiveResult<WithGradient, typename Geometry::Tangent> p1_squared_distance_edge_dirichlet_impl(
  const Geometry& geometry, std::span<const typename Geometry::Point> nodal_values,
  const P1LumpedLaplacianStencil& stencil) {
    using Tangent = typename Geometry::Tangent;
    p1_squared_distance_edge_dirichlet_validate(geometry, nodal_values, stencil);

    P1ObjectiveResult<WithGradient, Tangent> result;
    if constexpr (WithGradient) { result.nodal_gradient = p1_objective_zero_gradient(geometry, nodal_values); }
    for (const auto& edge : stencil.edges) {
        const double distance = geometry.distance(nodal_values[edge.first], nodal_values[edge.second]);
        if (!std::isfinite(distance) || distance < 0) {
            throw std::domain_error("P1 squared-distance edge Dirichlet distance is invalid");
        }
        const double edge_weight = -edge.stiffness;
        result.value = std::fma(0.5 * edge_weight, distance * distance, result.value);
        if (!std::isfinite(result.value)) {
            throw std::domain_error("P1 squared-distance edge Dirichlet value is nonfinite");
        }

        if constexpr (WithGradient) {
            const Tangent first_logarithm = geometry.logarithm(nodal_values[edge.first], nodal_values[edge.second]);
            const Tangent second_logarithm = geometry.logarithm(nodal_values[edge.second], nodal_values[edge.first]);
            result.nodal_gradient[edge.first] = geometry.linear_combination(
              nodal_values[edge.first], 1, result.nodal_gradient[edge.first], edge.stiffness, first_logarithm);
            result.nodal_gradient[edge.second] = geometry.linear_combination(
              nodal_values[edge.second], 1, result.nodal_gradient[edge.second], edge.stiffness, second_logarithm);
        }
    }
    if constexpr (WithGradient) {
        for (const Tangent& gradient : result.nodal_gradient) {
            p1_objective_require_finite_shape<std::domain_error>(
              gradient, geometry.order(), "P1 squared-distance edge Dirichlet gradient is nonfinite");
        }
    }
    return result;
}

}   // namespace internals

template <typename Scalar, int Order>
P1ObjectiveValueResult p1_squared_distance_edge_dirichlet_value(
  const manifold::LogEuclideanSPDGeometry<Scalar, Order>& geometry,
  std::span<const typename manifold::LogEuclideanSPDGeometry<Scalar, Order>::Point> nodal_values,
  const P1LumpedLaplacianStencil& stencil) {
    return internals::p1_squared_distance_edge_dirichlet_impl<false>(geometry, nodal_values, stencil);
}

template <typename Scalar, int Order>
P1ObjectiveValueResult p1_squared_distance_edge_dirichlet_value(
  const manifold::AffineInvariantSPDGeometry<Scalar, Order>& geometry,
  std::span<const typename manifold::AffineInvariantSPDGeometry<Scalar, Order>::Point> nodal_values,
  const P1LumpedLaplacianStencil& stencil) {
    return internals::p1_squared_distance_edge_dirichlet_impl<false>(geometry, nodal_values, stencil);
}

// The returned gradient is global: entry i is based at nodal_values[i].
template <typename Scalar, int Order>
P1ObjectiveContributionResult<typename manifold::LogEuclideanSPDGeometry<Scalar, Order>::Tangent>
p1_squared_distance_edge_dirichlet_contribution(
  const manifold::LogEuclideanSPDGeometry<Scalar, Order>& geometry,
  std::span<const typename manifold::LogEuclideanSPDGeometry<Scalar, Order>::Point> nodal_values,
  const P1LumpedLaplacianStencil& stencil) {
    return internals::p1_squared_distance_edge_dirichlet_impl<true>(geometry, nodal_values, stencil);
}

template <typename Scalar, int Order>
P1ObjectiveContributionResult<typename manifold::AffineInvariantSPDGeometry<Scalar, Order>::Tangent>
p1_squared_distance_edge_dirichlet_contribution(
  const manifold::AffineInvariantSPDGeometry<Scalar, Order>& geometry,
  std::span<const typename manifold::AffineInvariantSPDGeometry<Scalar, Order>::Point> nodal_values,
  const P1LumpedLaplacianStencil& stencil) {
    return internals::p1_squared_distance_edge_dirichlet_impl<true>(geometry, nodal_values, stencil);
}

}   // namespace gfe
}   // namespace fdapde

#endif   // __FDAPDE_GFE_P1_SQUARED_DISTANCE_EDGE_DIRICHLET_H__
