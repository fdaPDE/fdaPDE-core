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

#ifndef __FDAPDE_GFE_P1_DISCRETE_TENSION_H__
#define __FDAPDE_GFE_P1_DISCRETE_TENSION_H__

#include "header_check.h"

namespace fdapde {
namespace gfe {

namespace internals {

inline double p1_discrete_tension_multiply_divide(double first, double second, double denominator) {
    if (first == 0 || second == 0) return 0;
    int first_exponent = 0;
    int second_exponent = 0;
    int denominator_exponent = 0;
    const double first_fraction = std::frexp(first, &first_exponent);
    const double second_fraction = std::frexp(second, &second_exponent);
    const double denominator_fraction = std::frexp(denominator, &denominator_exponent);
    return std::scalbn(
      first_fraction * second_fraction / denominator_fraction, first_exponent + second_exponent - denominator_exponent);
}

template <typename Geometry>
typename Geometry::Tangent p1_discrete_tension_scale_divide(
  const Geometry& geometry, const typename Geometry::Point& point, double multiplier,
  const typename Geometry::Tangent& tangent, double denominator, const char* message) {
    typename Geometry::Tangent result = geometry.zero_tangent(point);
    for (int row = 0; row < geometry.order(); ++row) {
        for (int col = 0; col <= row; ++col) {
            result(row, col) = static_cast<typename Geometry::Scalar>(
              p1_discrete_tension_multiply_divide(multiplier, static_cast<double>(tangent(row, col)), denominator));
        }
    }
    p1_objective_require_finite_shape<std::domain_error>(result, geometry.order(), message);
    return result;
}

template <typename Geometry>
void p1_discrete_tension_validate(
  const Geometry& geometry, std::span<const typename Geometry::Point> nodal_values,
  const P1LumpedLaplacianStencil& stencil) {
    validate_p1_lumped_laplacian_stencil(stencil);
    if (nodal_values.size() != stencil.node_count()) {
        throw std::invalid_argument("P1 discrete tension node and stencil sizes must match");
    }
    for (const auto& node : nodal_values) {
        p1_objective_require_finite_shape(
          node, geometry.order(), "P1 discrete tension node has incompatible dimensions or nonfinite coefficients");
    }
}

template <bool WithGradient, typename Scalar, int Order>
P1ObjectiveResult<WithGradient, typename manifold::LogEuclideanSPDGeometry<Scalar, Order>::Tangent>
p1_log_euclidean_discrete_tension_impl(
  const manifold::LogEuclideanSPDGeometry<Scalar, Order>& geometry,
  std::span<const typename manifold::LogEuclideanSPDGeometry<Scalar, Order>::Point> nodal_values,
  const P1LumpedLaplacianStencil& stencil) {
    using Geometry = manifold::LogEuclideanSPDGeometry<Scalar, Order>;
    using Tangent = typename Geometry::Tangent;
    p1_discrete_tension_validate(geometry, nodal_values, stencil);

    std::vector<Tangent> logarithms;
    logarithms.reserve(nodal_values.size());
    for (const auto& node : nodal_values) {
        logarithms.emplace_back(fdapde::linalg::matrix_log(node));
        p1_objective_require_finite_shape<std::domain_error>(
          logarithms.back(), geometry.order(), "P1 log-Euclidean discrete tension logarithm is nonfinite");
    }

    // q_i = sum_{j != i} K_ij (log(P_j) - log(P_i)).
    std::vector<Tangent> chart_residuals = p1_objective_zero_gradient(geometry, nodal_values);
    for (const auto& edge : stencil.edges) {
        const Tangent difference =
          geometry.linear_combination(nodal_values[edge.first], 1, logarithms[edge.second], -1, logarithms[edge.first]);
        chart_residuals[edge.first] = geometry.linear_combination(
          nodal_values[edge.first], 1, chart_residuals[edge.first], edge.stiffness, difference);
        chart_residuals[edge.second] = geometry.linear_combination(
          nodal_values[edge.second], 1, chart_residuals[edge.second], -edge.stiffness, difference);
    }

    P1ObjectiveResult<WithGradient, Tangent> result;
    for (std::size_t node = 0; node < nodal_values.size(); ++node) {
        p1_objective_require_finite_shape<std::domain_error>(
          chart_residuals[node], geometry.order(), "P1 log-Euclidean discrete tension residual is nonfinite");
        const double scaled_norm =
          static_cast<double>(chart_residuals[node].norm()) / std::sqrt(stencil.lumped_masses[node]);
        result.value = std::fma(0.5 * scaled_norm, scaled_norm, result.value);
        if (!std::isfinite(result.value)) {
            throw std::domain_error("P1 log-Euclidean discrete tension value is nonfinite");
        }
    }
    if constexpr (WithGradient) {
        std::vector<Tangent> chart_gradient = p1_objective_zero_gradient(geometry, nodal_values);
        for (const auto& edge : stencil.edges) {
            Tangent difference = geometry.zero_tangent(nodal_values[edge.first]);
            for (int row = 0; row < geometry.order(); ++row) {
                for (int col = 0; col <= row; ++col) {
                    const double first = p1_discrete_tension_multiply_divide(
                      edge.stiffness, static_cast<double>(chart_residuals[edge.first](row, col)),
                      stencil.lumped_masses[edge.first]);
                    const double second = p1_discrete_tension_multiply_divide(
                      edge.stiffness, static_cast<double>(chart_residuals[edge.second](row, col)),
                      stencil.lumped_masses[edge.second]);
                    difference(row, col) = static_cast<Scalar>(second - first);
                }
            }
            p1_objective_require_finite_shape<std::domain_error>(
              difference, geometry.order(), "P1 log-Euclidean inverse-mass edge gradient is nonfinite");
            chart_gradient[edge.first] =
              geometry.linear_combination(nodal_values[edge.first], 1, chart_gradient[edge.first], 1, difference);
            chart_gradient[edge.second] =
              geometry.linear_combination(nodal_values[edge.second], 1, chart_gradient[edge.second], -1, difference);
        }

        result.nodal_gradient.reserve(nodal_values.size());
        for (std::size_t node = 0; node < nodal_values.size(); ++node) {
            p1_objective_require_finite_shape<std::domain_error>(
              chart_gradient[node], geometry.order(), "P1 log-Euclidean discrete tension chart gradient is nonfinite");
            result.nodal_gradient.emplace_back(
              fdapde::linalg::matrix_exp_frechet(logarithms[node], chart_gradient[node]));
            p1_objective_require_finite_shape<std::domain_error>(
              result.nodal_gradient.back(), geometry.order(),
              "P1 log-Euclidean discrete tension gradient is nonfinite");
        }
    }
    return result;
}

template <bool WithGradient, typename Scalar, int Order>
P1ObjectiveResult<WithGradient, typename manifold::AffineInvariantSPDGeometry<Scalar, Order>::Tangent>
p1_affine_invariant_discrete_tension_impl(
  const manifold::AffineInvariantSPDGeometry<Scalar, Order>& geometry,
  std::span<const typename manifold::AffineInvariantSPDGeometry<Scalar, Order>::Point> nodal_values,
  const P1LumpedLaplacianStencil& stencil) {
    using Geometry = manifold::AffineInvariantSPDGeometry<Scalar, Order>;
    using Tangent = typename Geometry::Tangent;
    p1_discrete_tension_validate(geometry, nodal_values, stencil);

    // r_i = sum_{j != i} K_ij Log_{P_i}(P_j).
    std::vector<Tangent> residuals = p1_objective_zero_gradient(geometry, nodal_values);
    for (const auto& edge : stencil.edges) {
        const Tangent first_logarithm = geometry.logarithm(nodal_values[edge.first], nodal_values[edge.second]);
        const Tangent second_logarithm = geometry.logarithm(nodal_values[edge.second], nodal_values[edge.first]);
        residuals[edge.first] = geometry.linear_combination(
          nodal_values[edge.first], 1, residuals[edge.first], edge.stiffness, first_logarithm);
        residuals[edge.second] = geometry.linear_combination(
          nodal_values[edge.second], 1, residuals[edge.second], edge.stiffness, second_logarithm);
    }

    P1ObjectiveResult<WithGradient, Tangent> result;
    for (std::size_t node = 0; node < nodal_values.size(); ++node) {
        p1_objective_require_finite_shape<std::domain_error>(
          residuals[node], geometry.order(), "P1 affine-invariant discrete tension residual is nonfinite");
        const double scaled_norm =
          geometry.norm(nodal_values[node], residuals[node]) / std::sqrt(stencil.lumped_masses[node]);
        result.value = std::fma(0.5 * scaled_norm, scaled_norm, result.value);
        if (!std::isfinite(result.value)) {
            throw std::domain_error("P1 affine-invariant discrete tension value is nonfinite");
        }
    }
    if constexpr (WithGradient) {
        result.nodal_gradient = p1_objective_zero_gradient(geometry, nodal_values);
        auto accumulate_directed = [&](std::size_t base, std::size_t target, double stiffness) {
            const Tangent base_action =
              geometry.half_squared_distance_hessian_vector(nodal_values[base], nodal_values[target], residuals[base]);
            const Tangent target_action =
              geometry.logarithm_target_vjp(nodal_values[base], nodal_values[target], residuals[base]);
            const Tangent scaled_base = p1_discrete_tension_scale_divide(
              geometry, nodal_values[base], stiffness, base_action, stencil.lumped_masses[base],
              "P1 affine-invariant inverse-mass base gradient is nonfinite");
            const Tangent scaled_target = p1_discrete_tension_scale_divide(
              geometry, nodal_values[target], stiffness, target_action, stencil.lumped_masses[base],
              "P1 affine-invariant inverse-mass target gradient is nonfinite");
            result.nodal_gradient[base] =
              geometry.linear_combination(nodal_values[base], 1, result.nodal_gradient[base], -1, scaled_base);
            result.nodal_gradient[target] =
              geometry.linear_combination(nodal_values[target], 1, result.nodal_gradient[target], 1, scaled_target);
        };
        for (const auto& edge : stencil.edges) {
            accumulate_directed(edge.first, edge.second, edge.stiffness);
            accumulate_directed(edge.second, edge.first, edge.stiffness);
        }
        for (const Tangent& gradient : result.nodal_gradient) {
            p1_objective_require_finite_shape<std::domain_error>(
              gradient, geometry.order(), "P1 affine-invariant discrete tension gradient is nonfinite");
        }
    }
    return result;
}

}   // namespace internals

template <typename Scalar, int Order>
P1ObjectiveValueResult p1_discrete_tension_value(
  const manifold::LogEuclideanSPDGeometry<Scalar, Order>& geometry,
  std::span<const typename manifold::LogEuclideanSPDGeometry<Scalar, Order>::Point> nodal_values,
  const P1LumpedLaplacianStencil& stencil) {
    return internals::p1_log_euclidean_discrete_tension_impl<false>(geometry, nodal_values, stencil);
}

// The returned gradient is global: entry i is based at nodal_values[i].
template <typename Scalar, int Order>
P1ObjectiveContributionResult<typename manifold::LogEuclideanSPDGeometry<Scalar, Order>::Tangent>
p1_discrete_tension_contribution(
  const manifold::LogEuclideanSPDGeometry<Scalar, Order>& geometry,
  std::span<const typename manifold::LogEuclideanSPDGeometry<Scalar, Order>::Point> nodal_values,
  const P1LumpedLaplacianStencil& stencil) {
    return internals::p1_log_euclidean_discrete_tension_impl<true>(geometry, nodal_values, stencil);
}

template <typename Scalar, int Order>
P1ObjectiveValueResult p1_discrete_tension_value(
  const manifold::AffineInvariantSPDGeometry<Scalar, Order>& geometry,
  std::span<const typename manifold::AffineInvariantSPDGeometry<Scalar, Order>::Point> nodal_values,
  const P1LumpedLaplacianStencil& stencil) {
    return internals::p1_affine_invariant_discrete_tension_impl<false>(geometry, nodal_values, stencil);
}

// The returned gradient is global: entry i is based at nodal_values[i].
template <typename Scalar, int Order>
P1ObjectiveContributionResult<typename manifold::AffineInvariantSPDGeometry<Scalar, Order>::Tangent>
p1_discrete_tension_contribution(
  const manifold::AffineInvariantSPDGeometry<Scalar, Order>& geometry,
  std::span<const typename manifold::AffineInvariantSPDGeometry<Scalar, Order>::Point> nodal_values,
  const P1LumpedLaplacianStencil& stencil) {
    return internals::p1_affine_invariant_discrete_tension_impl<true>(geometry, nodal_values, stencil);
}

}   // namespace gfe
}   // namespace fdapde

#endif   // __FDAPDE_GFE_P1_DISCRETE_TENSION_H__
