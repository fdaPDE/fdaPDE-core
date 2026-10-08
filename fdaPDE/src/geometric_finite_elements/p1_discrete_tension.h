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

/// @brief forms a scaled coefficient while avoiding intermediate product overflow
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

/// @brief scales a tangent by stiffness over mass and rejects nonfinite results
template <typename Geometry>
typename Geometry::Tangent p1_discrete_tension_scale_divide(
  const Geometry& geometry, const auto& point, double multiplier, const typename Geometry::Tangent& tangent,
  double denominator, const char* message) {
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

/// @brief validates a nodal field against its lumped Laplacian stencil
template <typename Geometry, typename Nodes>
void p1_discrete_tension_validate(
  const Geometry& geometry, const Nodes& nodal_values, const P1LumpedLaplacianStencil& stencil) {
    validate_p1_lumped_laplacian_stencil(stencil);
    fdapde_strong_assert(
      nodal_values.size() == stencil.node_count(), std::invalid_argument,
      "P1 discrete tension node and stencil sizes must match");

    for (std::size_t i = 0; i < nodal_values.size(); ++i) {
        const auto& node = nodal_values[i];
        p1_objective_require_finite_shape(
          node, geometry.order(), "P1 discrete tension node has incompatible dimensions or nonfinite coefficients");
    }
}

/// @brief evaluates squared discrete tension in flat coordinates with optional gradients
template <bool WithGradient, typename Geometry, typename Nodes>
    requires is_flat_spd_geometry<Geometry>
P1ObjectiveResult<WithGradient, typename Geometry::Tangent> p1_flat_discrete_tension_impl(
  const Geometry& geometry, const Nodes& nodal_values, const P1LumpedLaplacianStencil& stencil) {
    using Scalar = typename Geometry::Scalar;
    using Tangent = typename Geometry::Tangent;
    p1_discrete_tension_validate(geometry, nodal_values, stencil);

    std::vector<typename FlatSPDChartFrame<Geometry>::type> frames;
    frames.reserve(nodal_values.size());
    for (std::size_t i = 0; i < nodal_values.size(); ++i) {
        const auto& node = nodal_values[i];
        frames.emplace_back(p1_flat_chart_frame(geometry, node));
        p1_objective_require_finite_shape<std::domain_error>(
          p1_flat_frame_coordinates<Geometry>(frames.back()), geometry.order(),
          "P1 flat-coordinate discrete tension chart is nonfinite");
    }

    // q_i = sum_{j != i} K_ij (chart(P_j) - chart(P_i))
    std::vector<Tangent> chart_residuals = p1_objective_zero_gradient(geometry, nodal_values);
    for (const auto& edge : stencil.edges) {
        const Tangent difference = geometry.linear_combination(
          nodal_values[edge.first], 1, p1_flat_frame_coordinates<Geometry>(frames[edge.second]), -1,
          p1_flat_frame_coordinates<Geometry>(frames[edge.first]));
        chart_residuals[edge.first] = geometry.linear_combination(
          nodal_values[edge.first], 1, chart_residuals[edge.first], edge.stiffness, difference);
        chart_residuals[edge.second] = geometry.linear_combination(
          nodal_values[edge.second], 1, chart_residuals[edge.second], -edge.stiffness, difference);
    }

    P1ObjectiveResult<WithGradient, Tangent> result;
    for (std::size_t node = 0; node < nodal_values.size(); ++node) {
        p1_objective_require_finite_shape<std::domain_error>(
          chart_residuals[node], geometry.order(), "P1 flat-coordinate discrete tension residual is nonfinite");
        const double scaled_norm =
          static_cast<double>(chart_residuals[node].norm()) / std::sqrt(stencil.lumped_masses[node]);
        result.value = std::fma(0.5 * scaled_norm, scaled_norm, result.value);
        fdapde_strong_assert(
          std::isfinite(result.value), std::domain_error, "P1 flat-coordinate discrete tension value is nonfinite");
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
              difference, geometry.order(), "P1 flat-coordinate inverse-mass edge gradient is nonfinite");
            chart_gradient[edge.first] =
              geometry.linear_combination(nodal_values[edge.first], 1, chart_gradient[edge.first], 1, difference);
            chart_gradient[edge.second] =
              geometry.linear_combination(nodal_values[edge.second], 1, chart_gradient[edge.second], -1, difference);
        }

        result.nodal_gradient.reserve(nodal_values.size());
        for (std::size_t node = 0; node < nodal_values.size(); ++node) {
            p1_objective_require_finite_shape<std::domain_error>(
              chart_gradient[node], geometry.order(),
              "P1 flat-coordinate discrete tension chart gradient is nonfinite");
            result.nodal_gradient.emplace_back(
              p1_flat_frame_inverse_chart_jvp(geometry, frames[node], chart_gradient[node]));
            p1_objective_require_finite_shape<std::domain_error>(
              result.nodal_gradient.back(), geometry.order(),
              "P1 flat-coordinate discrete tension gradient is nonfinite");
        }
    }
    return result;
}

/// @brief evaluates intrinsic squared tension while reusing directed relative frames
template <bool WithGradient, typename Geometry, typename Nodes>
P1ObjectiveResult<WithGradient, typename Geometry::Tangent> p1_intrinsic_discrete_tension_impl(
  const Geometry& geometry, const Nodes& nodal_values, const P1LumpedLaplacianStencil& stencil) {
    using Tangent = typename Geometry::Tangent;
    p1_discrete_tension_validate(geometry, nodal_values, stencil);

    // reuse both directed relative frames between the residual and its pullback
    std::vector<std::array<typename Geometry::RelativeFrame, 2>> frames;
    if constexpr (WithGradient) frames.reserve(stencil.edges.size());
    std::vector<Tangent> residuals = p1_objective_zero_gradient(geometry, nodal_values);
    for (const auto& edge : stencil.edges) {
        auto first = geometry.relative_frame(nodal_values[edge.first], nodal_values[edge.second]);
        auto second = geometry.relative_frame(nodal_values[edge.second], nodal_values[edge.first]);
        const Tangent first_logarithm = geometry.logarithm(first);
        const Tangent second_logarithm = geometry.logarithm(second);
        residuals[edge.first] = geometry.linear_combination(
          nodal_values[edge.first], 1, residuals[edge.first], edge.stiffness, first_logarithm);
        residuals[edge.second] = geometry.linear_combination(
          nodal_values[edge.second], 1, residuals[edge.second], edge.stiffness, second_logarithm);
        if constexpr (WithGradient) frames.push_back({std::move(first), std::move(second)});
    }

    P1ObjectiveResult<WithGradient, Tangent> result;
    for (std::size_t node = 0; node < nodal_values.size(); ++node) {
        p1_objective_require_finite_shape<std::domain_error>(
          residuals[node], geometry.order(), "P1 intrinsic discrete tension residual is nonfinite");
        const double scaled_norm =
          geometry.norm(nodal_values[node], residuals[node]) / std::sqrt(stencil.lumped_masses[node]);
        result.value = std::fma(0.5 * scaled_norm, scaled_norm, result.value);
        fdapde_strong_assert(
          std::isfinite(result.value), std::domain_error, "P1 intrinsic discrete tension value is nonfinite");
    }
    if constexpr (WithGradient) {
        result.nodal_gradient = p1_objective_zero_gradient(geometry, nodal_values);
        auto accumulate_directed = [&](std::size_t base, std::size_t target, double stiffness, const auto& frame) {
            const Tangent base_action = geometry.half_squared_distance_hessian_vector(frame, residuals[base]);
            const Tangent target_action = geometry.logarithm_target_vjp(frame, residuals[base]);
            const Tangent scaled_base = p1_discrete_tension_scale_divide(
              geometry, nodal_values[base], stiffness, base_action, stencil.lumped_masses[base],
              "P1 intrinsic inverse-mass base gradient is nonfinite");
            const Tangent scaled_target = p1_discrete_tension_scale_divide(
              geometry, nodal_values[target], stiffness, target_action, stencil.lumped_masses[base],
              "P1 intrinsic inverse-mass target gradient is nonfinite");
            result.nodal_gradient[base] =
              geometry.linear_combination(nodal_values[base], 1, result.nodal_gradient[base], -1, scaled_base);
            result.nodal_gradient[target] =
              geometry.linear_combination(nodal_values[target], 1, result.nodal_gradient[target], 1, scaled_target);
        };
        for (std::size_t i = 0; i < stencil.edges.size(); ++i) {
            const auto& edge = stencil.edges[i];
            accumulate_directed(edge.first, edge.second, edge.stiffness, frames[i][0]);
            accumulate_directed(edge.second, edge.first, edge.stiffness, frames[i][1]);
        }
        for (const Tangent& gradient : result.nodal_gradient) {
            p1_objective_require_finite_shape<std::domain_error>(
              gradient, geometry.order(), "P1 intrinsic discrete tension gradient is nonfinite");
        }
    }
    return result;
}

/// @brief lists each node's incident edges in the stencil's original accumulation order
inline std::vector<std::vector<std::size_t>> p1_discrete_tension_incidence(const P1LumpedLaplacianStencil& stencil) {
    std::vector<std::vector<std::size_t>> incident(stencil.node_count());
    for (std::size_t e = 0; e < stencil.edges.size(); ++e) {
        incident[stencil.edges[e].first].push_back(e);
        incident[stencil.edges[e].second].push_back(e);
    }
    return incident;
}

/// @brief evaluates flat-coordinate tension with independent edge work and ordered nodal gathers
template <bool WithGradient, typename Geometry, typename Nodes>
    requires is_flat_spd_geometry<Geometry>
P1ObjectiveResult<WithGradient, typename Geometry::Tangent> p1_flat_discrete_tension_parallel_impl(
  const Geometry& geometry, const Nodes& nodes, const P1LumpedLaplacianStencil& stencil) {
    using Scalar = typename Geometry::Scalar;
    using Tangent = typename Geometry::Tangent;
    using Frame = typename FlatSPDChartFrame<Geometry>::type;
    p1_discrete_tension_validate(geometry, nodes, stencil);
    const auto incident = p1_discrete_tension_incidence(stencil);
    std::vector<std::optional<Frame>> frames(nodes.size());
    for_each_point(nodes.size(), execution_par, [&](std::size_t i) {
        frames[i].emplace(p1_flat_chart_frame(geometry, nodes[i]));
        p1_objective_require_finite_shape<std::domain_error>(
          p1_flat_frame_coordinates<Geometry>(*frames[i]), geometry.order(),
          "P1 flat-coordinate discrete tension chart is nonfinite");
    });
    std::vector<Tangent> differences(stencil.edges.size());
    for_each_point(stencil.edges.size(), execution_par, [&](std::size_t e) {
        const auto& edge = stencil.edges[e];
        differences[e] = geometry.linear_combination(
          nodes[edge.first], 1, p1_flat_frame_coordinates<Geometry>(*frames[edge.second]), -1,
          p1_flat_frame_coordinates<Geometry>(*frames[edge.first]));
    });
    auto residuals = p1_objective_zero_gradient(geometry, nodes);
    std::vector<double> scaled_norms(nodes.size());
    for_each_point(nodes.size(), execution_par, [&](std::size_t i) {
        for (std::size_t e : incident[i]) {
            const auto& edge = stencil.edges[e];
            const double stiffness = edge.first == i ? edge.stiffness : -edge.stiffness;
            residuals[i] = geometry.linear_combination(nodes[i], 1, residuals[i], stiffness, differences[e]);
        }
        p1_objective_require_finite_shape<std::domain_error>(
          residuals[i], geometry.order(), "P1 flat-coordinate discrete tension residual is nonfinite");
        scaled_norms[i] = static_cast<double>(residuals[i].norm()) / std::sqrt(stencil.lumped_masses[i]);
    });
    P1ObjectiveResult<WithGradient, Tangent> result;
    for (double norm : scaled_norms) {
        result.value = std::fma(0.5 * norm, norm, result.value);
        fdapde_strong_assert(
          std::isfinite(result.value), std::domain_error, "P1 flat-coordinate discrete tension value is nonfinite");
    }
    if constexpr (WithGradient) {
        for_each_point(stencil.edges.size(), execution_par, [&](std::size_t e) {
            const auto& edge = stencil.edges[e];
            Tangent difference = geometry.zero_tangent(nodes[edge.first]);
            for (int row = 0; row < geometry.order(); ++row) {
                for (int col = 0; col <= row; ++col) {
                    const double first = p1_discrete_tension_multiply_divide(
                      edge.stiffness, static_cast<double>(residuals[edge.first](row, col)),
                      stencil.lumped_masses[edge.first]);
                    const double second = p1_discrete_tension_multiply_divide(
                      edge.stiffness, static_cast<double>(residuals[edge.second](row, col)),
                      stencil.lumped_masses[edge.second]);
                    difference(row, col) = static_cast<Scalar>(second - first);
                }
            }
            p1_objective_require_finite_shape<std::domain_error>(
              difference, geometry.order(), "P1 flat-coordinate inverse-mass edge gradient is nonfinite");
            differences[e] = std::move(difference);
        });
        result.nodal_gradient = p1_objective_zero_gradient(geometry, nodes);
        for_each_point(nodes.size(), execution_par, [&](std::size_t i) {
            Tangent gradient = geometry.zero_tangent(nodes[i]);
            for (std::size_t e : incident[i]) {
                const double sign = stencil.edges[e].first == i ? 1 : -1;
                gradient = geometry.linear_combination(nodes[i], 1, gradient, sign, differences[e]);
            }
            p1_objective_require_finite_shape<std::domain_error>(
              gradient, geometry.order(), "P1 flat-coordinate discrete tension chart gradient is nonfinite");
            result.nodal_gradient[i] = p1_flat_frame_inverse_chart_jvp(geometry, *frames[i], gradient);
            p1_objective_require_finite_shape<std::domain_error>(
              result.nodal_gradient[i], geometry.order(), "P1 flat-coordinate discrete tension gradient is nonfinite");
        });
    }
    return result;
}

/// @brief evaluates intrinsic tension with parallel relative frames and order-preserving nodal pullbacks
template <bool WithGradient, typename Geometry, typename Nodes>
P1ObjectiveResult<WithGradient, typename Geometry::Tangent> p1_intrinsic_discrete_tension_parallel_impl(
  const Geometry& geometry, const Nodes& nodes, const P1LumpedLaplacianStencil& stencil) {
    using Tangent = typename Geometry::Tangent;
    using Frame = typename Geometry::RelativeFrame;
    p1_discrete_tension_validate(geometry, nodes, stencil);
    const auto incident = p1_discrete_tension_incidence(stencil);
    std::vector<std::array<std::optional<Frame>, 2>> frames;
    if constexpr (WithGradient) frames.resize(stencil.edges.size());
    std::vector<std::array<Tangent, 2>> logarithms(stencil.edges.size());
    for_each_point(stencil.edges.size(), execution_par, [&](std::size_t e) {
        const auto& edge = stencil.edges[e];
        auto first = geometry.relative_frame(nodes[edge.first], nodes[edge.second]);
        auto second = geometry.relative_frame(nodes[edge.second], nodes[edge.first]);
        logarithms[e][0] = geometry.logarithm(first);
        logarithms[e][1] = geometry.logarithm(second);
        if constexpr (WithGradient) {
            frames[e][0].emplace(std::move(first));
            frames[e][1].emplace(std::move(second));
        }
    });
    auto residuals = p1_objective_zero_gradient(geometry, nodes);
    std::vector<double> scaled_norms(nodes.size());
    for_each_point(nodes.size(), execution_par, [&](std::size_t i) {
        for (std::size_t e : incident[i]) {
            const auto& edge = stencil.edges[e];
            const int direction = edge.first == i ? 0 : 1;
            residuals[i] =
              geometry.linear_combination(nodes[i], 1, residuals[i], edge.stiffness, logarithms[e][direction]);
        }
        p1_objective_require_finite_shape<std::domain_error>(
          residuals[i], geometry.order(), "P1 intrinsic discrete tension residual is nonfinite");
        scaled_norms[i] = geometry.norm(nodes[i], residuals[i]) / std::sqrt(stencil.lumped_masses[i]);
    });
    P1ObjectiveResult<WithGradient, Tangent> result;
    for (double norm : scaled_norms) {
        result.value = std::fma(0.5 * norm, norm, result.value);
        fdapde_strong_assert(
          std::isfinite(result.value), std::domain_error, "P1 intrinsic discrete tension value is nonfinite");
    }
    if constexpr (WithGradient) {
        std::vector<std::array<Tangent, 4>> actions(stencil.edges.size());
        for_each_point(stencil.edges.size(), execution_par, [&](std::size_t e) {
            const auto& edge = stencil.edges[e];
            for (int reverse = 0; reverse < 2; ++reverse) {
                const auto base = reverse ? edge.second : edge.first;
                const auto target = reverse ? edge.first : edge.second;
                const auto& frame = *frames[e][reverse];
                const Tangent base_action = geometry.half_squared_distance_hessian_vector(frame, residuals[base]);
                const Tangent target_action = geometry.logarithm_target_vjp(frame, residuals[base]);
                actions[e][2 * reverse] = p1_discrete_tension_scale_divide(
                  geometry, nodes[base], edge.stiffness, base_action, stencil.lumped_masses[base],
                  "P1 intrinsic inverse-mass base gradient is nonfinite");
                actions[e][2 * reverse + 1] = p1_discrete_tension_scale_divide(
                  geometry, nodes[target], edge.stiffness, target_action, stencil.lumped_masses[base],
                  "P1 intrinsic inverse-mass target gradient is nonfinite");
            }
        });
        result.nodal_gradient = p1_objective_zero_gradient(geometry, nodes);
        for_each_point(nodes.size(), execution_par, [&](std::size_t i) {
            for (std::size_t e : incident[i]) {
                if (stencil.edges[e].first == i) {
                    result.nodal_gradient[i] =
                      geometry.linear_combination(nodes[i], 1, result.nodal_gradient[i], -1, actions[e][0]);
                    result.nodal_gradient[i] =
                      geometry.linear_combination(nodes[i], 1, result.nodal_gradient[i], 1, actions[e][3]);
                } else {
                    result.nodal_gradient[i] =
                      geometry.linear_combination(nodes[i], 1, result.nodal_gradient[i], 1, actions[e][1]);
                    result.nodal_gradient[i] =
                      geometry.linear_combination(nodes[i], 1, result.nodal_gradient[i], -1, actions[e][2]);
                }
            }
            p1_objective_require_finite_shape<std::domain_error>(
              result.nodal_gradient[i], geometry.order(), "P1 intrinsic discrete tension gradient is nonfinite");
        });
    }
    return result;
}

}   // namespace internals

/// @brief evaluates half the mass-weighted squared discrete tension
template <typename Scalar, int Order, Usage Uses, typename Nodes, typename Point_>
P1ObjectiveValueResult p1_discrete_tension_value(
  const manifold::LogEuclideanSPDGeometry<Scalar, Order, Uses, Point_>& geometry, const Nodes& nodal_values,
  const P1LumpedLaplacianStencil& stencil) {
    return internals::p1_flat_discrete_tension_impl<false>(geometry, nodal_values, stencil);
}

/// @brief evaluates half the mass-weighted squared discrete tension
template <typename Geometry, typename Nodes>
    requires internals::is_flat_spd_geometry<Geometry> && (!internals::is_log_euclidean_spd_geometry<Geometry>)
P1ObjectiveValueResult p1_discrete_tension_value(
  const Geometry& geometry, const Nodes& nodal_values, const P1LumpedLaplacianStencil& stencil) {
    return internals::p1_flat_discrete_tension_impl<false>(geometry, nodal_values, stencil);
}

// the returned gradient is global: entry i is based at nodal_values[i]
/// @brief returns squared tension and global metric gradients based at each node
template <typename Scalar, int Order, Usage Uses, typename Nodes, typename Point_>
P1ObjectiveContributionResult<typename manifold::LogEuclideanSPDGeometry<Scalar, Order, Uses, Point_>::Tangent>
p1_discrete_tension_contribution(
  const manifold::LogEuclideanSPDGeometry<Scalar, Order, Uses, Point_>& geometry, const Nodes& nodal_values,
  const P1LumpedLaplacianStencil& stencil) {
    return internals::p1_flat_discrete_tension_impl<true>(geometry, nodal_values, stencil);
}

/// @brief returns squared tension and global metric gradients based at each node
template <typename Geometry, typename Nodes>
    requires internals::is_flat_spd_geometry<Geometry> && (!internals::is_log_euclidean_spd_geometry<Geometry>)
P1ObjectiveContributionResult<typename Geometry::Tangent> p1_discrete_tension_contribution(
  const Geometry& geometry, const Nodes& nodal_values, const P1LumpedLaplacianStencil& stencil) {
    return internals::p1_flat_discrete_tension_impl<true>(geometry, nodal_values, stencil);
}

/// @brief evaluates half the mass-weighted squared discrete tension
template <typename Scalar, int Order, Usage Uses, typename Nodes, typename Point_>
P1ObjectiveValueResult p1_discrete_tension_value(
  const manifold::AffineInvariantSPDGeometry<Scalar, Order, Uses, Point_>& geometry, const Nodes& nodal_values,
  const P1LumpedLaplacianStencil& stencil) {
    return internals::p1_intrinsic_discrete_tension_impl<false>(geometry, nodal_values, stencil);
}

// the returned gradient is global: entry i is based at nodal_values[i]
/// @brief returns squared tension and global metric gradients based at each node
template <typename Scalar, int Order, Usage Uses, typename Nodes, typename Point_>
P1ObjectiveContributionResult<typename manifold::AffineInvariantSPDGeometry<Scalar, Order, Uses, Point_>::Tangent>
p1_discrete_tension_contribution(
  const manifold::AffineInvariantSPDGeometry<Scalar, Order, Uses, Point_>& geometry, const Nodes& nodal_values,
  const P1LumpedLaplacianStencil& stencil) {
    return internals::p1_intrinsic_discrete_tension_impl<true>(geometry, nodal_values, stencil);
}

/// @brief evaluates half the mass-weighted squared BW discrete tension
template <typename Scalar, int Order, Usage Uses, typename Nodes, typename Point_>
P1ObjectiveValueResult p1_discrete_tension_value(
  const manifold::BuresWassersteinSPDGeometry<Scalar, Order, Uses, Point_>& geometry, const Nodes& nodal_values,
  const P1LumpedLaplacianStencil& stencil) {
    return internals::p1_intrinsic_discrete_tension_impl<false>(geometry, nodal_values, stencil);
}

/// @brief returns BW squared tension and its global metric gradients based at each node
template <typename Scalar, int Order, Usage Uses, typename Nodes, typename Point_>
P1ObjectiveContributionResult<typename manifold::BuresWassersteinSPDGeometry<Scalar, Order, Uses, Point_>::Tangent>
p1_discrete_tension_contribution(
  const manifold::BuresWassersteinSPDGeometry<Scalar, Order, Uses, Point_>& geometry, const Nodes& nodal_values,
  const P1LumpedLaplacianStencil& stencil) {
    return internals::p1_intrinsic_discrete_tension_impl<true>(geometry, nodal_values, stencil);
}

/// @brief evaluates squared tension using parallel edge work without changing nodal summation order
/// @details fewer than 32 nodes or one configured worker use the unchanged serial implementation
template <typename Geometry, typename Nodes, internals::PointExecutionPolicy Policy>
    requires requires(const Geometry& geometry, const Nodes& nodes, const P1LumpedLaplacianStencil& stencil) {
        p1_discrete_tension_value(geometry, nodes, stencil);
    }
P1ObjectiveValueResult p1_discrete_tension_value(
  const Geometry& geometry, const Nodes& nodes, const P1LumpedLaplacianStencil& stencil, Policy) {
    if constexpr (std::same_as<Policy, execution_seq_t>)
        return p1_discrete_tension_value(geometry, nodes, stencil);
    else {
        if (nodes.size() < 32 || parallel_get_num_threads() == 1)
            return p1_discrete_tension_value(geometry, nodes, stencil);
        if constexpr (internals::is_flat_spd_geometry<Geometry>)
            return internals::p1_flat_discrete_tension_parallel_impl<false>(geometry, nodes, stencil);
        else
            return internals::p1_intrinsic_discrete_tension_parallel_impl<false>(geometry, nodes, stencil);
    }
}

/// @brief returns tension and metric gradients using deterministic parallel edge and nodal work
/// @details fewer than 32 nodes or one configured worker use the unchanged serial implementation
template <typename Geometry, typename Nodes, internals::PointExecutionPolicy Policy>
    requires requires(const Geometry& geometry, const Nodes& nodes, const P1LumpedLaplacianStencil& stencil) {
        p1_discrete_tension_contribution(geometry, nodes, stencil);
    }
P1ObjectiveContributionResult<typename Geometry::Tangent> p1_discrete_tension_contribution(
  const Geometry& geometry, const Nodes& nodes, const P1LumpedLaplacianStencil& stencil, Policy) {
    if constexpr (std::same_as<Policy, execution_seq_t>)
        return p1_discrete_tension_contribution(geometry, nodes, stencil);
    else {
        if (nodes.size() < 32 || parallel_get_num_threads() == 1)
            return p1_discrete_tension_contribution(geometry, nodes, stencil);
        if constexpr (internals::is_flat_spd_geometry<Geometry>)
            return internals::p1_flat_discrete_tension_parallel_impl<true>(geometry, nodes, stencil);
        else
            return internals::p1_intrinsic_discrete_tension_parallel_impl<true>(geometry, nodes, stencil);
    }
}

}   // namespace gfe
}   // namespace fdapde

#endif   // __FDAPDE_GFE_P1_DISCRETE_TENSION_H__
