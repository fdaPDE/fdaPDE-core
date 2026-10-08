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

#ifndef __FDAPDE_GFE_P1_OBJECTIVE_CONTRIBUTIONS_H__
#define __FDAPDE_GFE_P1_OBJECTIVE_CONTRIBUTIONS_H__

#include "header_check.h"

namespace fdapde {
namespace gfe {

/// @brief identifies the failed interpolation or derivative stage
enum class P1ObjectiveStage {
    mean,
    data_pullback,
    dirichlet_spatial,
    dirichlet_mixed_output,
    dirichlet_mixed_pullback
};

/// @brief records the first failed mean or linear solve with its local site
struct P1ObjectiveFailure {
    P1ObjectiveStage stage = P1ObjectiveStage::mean;
    std::size_t site = 0;
    std::optional<std::size_t> axis;
    std::optional<manifold::BarycenterStopReason> barycenter_stop_reason;
    double stationarity_norm = std::numeric_limits<double>::quiet_NaN();
    std::optional<P1LinearSolveStatus> linear_solve;
};

/// @brief returns an objective value and optional failure diagnostics
struct P1ObjectiveValueResult {
    // on failure this is only the last available local candidate and must
    // not be consumed by an optimizer
    double value = 0;
    std::optional<P1ObjectiveFailure> first_failure;

    /// @brief reports whether the required mean and derivative solves succeeded
    bool converged() const noexcept { return !first_failure.has_value(); }
};

/// @brief returns an objective value, metric nodal gradients and failure diagnostics
template <typename Tangent> struct P1ObjectiveContributionResult {
    // on failure these are only the last available local candidates and
    // must not be consumed by an optimizer
    double value = 0;
    std::vector<Tangent> nodal_gradient;
    std::optional<P1ObjectiveFailure> first_failure;

    /// @brief reports whether the required mean and derivative solves succeeded
    bool converged() const noexcept { return !first_failure.has_value(); }
};

namespace internals {

/// @brief checks the order and finite stored coefficients of a symmetric matrix
template <typename Exception = std::invalid_argument, typename Matrix>
void p1_objective_require_finite_shape(const Matrix& matrix, int order, const char* message) {
    fdapde_strong_assert(matrix.rows() == order && matrix.cols() == order, Exception, message);

    for (int i = 0; i < order; ++i) {
        for (int j = 0; j <= i; ++j) {
            fdapde_strong_assert(std::isfinite(static_cast<double>(matrix(i, j))), Exception, message);
        }
    }
}

/// @brief validates a physical derivative of barycentric weights
inline void p1_objective_validate_spatial_direction(std::span<const double> direction) {
    long double sum = 0;
    long double correction = 0;
    long double absolute_sum = 0;
    for (const double coefficient : direction) {
        fdapde_strong_assert(
          std::isfinite(coefficient), std::invalid_argument, "P1 Dirichlet physical weight gradients must be finite");

        const long double value = static_cast<long double>(coefficient);
        const long double corrected = value - correction;
        const long double next = sum + corrected;
        correction = (next - sum) - corrected;
        sum = next;
        absolute_sum += std::abs(value);
    }
    const long double tolerance =
      static_cast<long double>(p1_weight_sum_tolerance(direction.size())) * std::max(1.0L, absolute_sum);
    fdapde_strong_assert(
      std::abs(sum) <= tolerance, std::invalid_argument, "P1 Dirichlet physical weight gradients must sum to zero");
}

/// @brief converts a failed barycenter into objective diagnostics
template <typename ValueResult>
P1ObjectiveFailure p1_objective_mean_failure(std::size_t site, const ValueResult& result) {
    return {P1ObjectiveStage::mean, site, std::nullopt, result.stop_reason, result.stationarity_norm, std::nullopt};
}

/// @brief converts a failed derivative solve into objective diagnostics
inline P1ObjectiveFailure p1_objective_solve_failure(
  P1ObjectiveStage stage, std::size_t site, std::optional<std::size_t> axis, const P1LinearSolveStatus& solve) {
    return {stage, site, axis, std::nullopt, std::numeric_limits<double>::quiet_NaN(), solve};
}

/// @brief allocates zero metric tangents in the supplied nodal order
template <typename Geometry, typename Nodes>
std::vector<typename Geometry::Tangent> p1_objective_zero_gradient(const Geometry& geometry, const Nodes& nodes) {
    std::vector<typename Geometry::Tangent> result;
    result.reserve(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto& node = nodes[i];
        result.push_back(geometry.zero_tangent(node));
    }
    return result;
}

/// @brief evaluates the Frobenius residual of a converged interpolated tensor
template <typename Geometry, typename ValueResult>
P1ObjectiveValueResult p1_frobenius_data_site_value_impl(
  const Geometry& geometry, const ValueResult& value_result, const typename Geometry::Tangent& observation) {
    using Tangent = typename Geometry::Tangent;
    p1_objective_require_finite_shape(
      observation, geometry.order(), "P1 Frobenius observation has incompatible dimensions or nonfinite coefficients");

    P1ObjectiveValueResult result;
    /// @brief reports whether the required mean and derivative solves succeeded
    if (!value_result.converged()) {
        result.first_failure = p1_objective_mean_failure(0, value_result);
        return result;
    }

    const Tangent ambient_gradient(value_result.value - observation);
    const double ambient_norm = static_cast<double>(ambient_gradient.norm());
    result.value = 0.5 * ambient_norm * ambient_norm;
    fdapde_strong_assert(std::isfinite(result.value), std::domain_error, "P1 Frobenius data contribution is nonfinite");

    return result;
}

/// @brief pulls a Frobenius residual through a prepared geodesic linearization
template <typename Geometry, typename Linearization, typename Nodes>
P1ObjectiveContributionResult<typename Geometry::Tangent> p1_frobenius_data_site_contribution_impl(
  const Geometry& geometry, const Nodes& nodes, Linearization linearization,
  const typename Geometry::Tangent& observation) {
    using Tangent = typename Geometry::Tangent;
    const auto& value_result = linearization.result();
    const auto value = p1_frobenius_data_site_value_impl(geometry, value_result, observation);

    P1ObjectiveContributionResult<Tangent> result;
    result.value = value.value;
    result.first_failure = value.first_failure;
    result.nodal_gradient = p1_objective_zero_gradient(geometry, nodes);
    /// @brief reports whether the required mean and derivative solves succeeded
    if (!result.converged()) return result;

    const Tangent ambient_gradient(value_result.value - observation);
    const Tangent value_gradient = geometry.euclidean_to_riemannian_gradient(value_result.value, ambient_gradient);
    p1_objective_require_finite_shape<std::domain_error>(
      value_gradient, geometry.order(), "P1 Frobenius metric gradient is nonfinite");

    if constexpr (is_flat_spd_geometry<Geometry>) {
        result.nodal_gradient = linearization.nodal_vjp(value_gradient);
    } else {
        auto pullback = linearization.nodal_vjp(value_gradient);
        result.nodal_gradient = std::move(pullback.derivative);
        if (!pullback.converged()) {
            result.first_failure = p1_objective_solve_failure(
              P1ObjectiveStage::data_pullback, 0, std::nullopt,
              {pullback.residual_norm, pullback.iterations, pullback.stop_reason});
            return result;
        }
    }
    for (const Tangent& gradient : result.nodal_gradient) {
        p1_objective_require_finite_shape<std::domain_error>(
          gradient, geometry.order(), "P1 Frobenius nodal gradient is nonfinite");
    }
    return result;
}

/// @brief evaluates a residual against an observation supplied in flat coordinates
template <typename Geometry, typename ValueResult>
P1ObjectiveValueResult p1_flat_coordinate_data_site_value_impl(
  const Geometry& geometry, const ValueResult& value_result, const typename Geometry::Tangent& observation_chart) {
    using Tangent = typename Geometry::Tangent;
    p1_objective_require_finite_shape(
      observation_chart, geometry.order(),
      "P1 flat-coordinate observation has incompatible dimensions or nonfinite coefficients");

    P1ObjectiveValueResult result;
    /// @brief reports whether the required mean and derivative solves succeeded
    if (!value_result.converged()) {
        result.first_failure = p1_objective_mean_failure(0, value_result);
        return result;
    }

    const Tangent value_chart = p1_flat_chart(geometry, value_result.value);
    const Tangent residual(value_chart - observation_chart);
    const double residual_norm = static_cast<double>(residual.norm());
    result.value = 0.5 * residual_norm * residual_norm;
    fdapde_strong_assert(
      std::isfinite(result.value), std::domain_error, "P1 flat-coordinate data contribution is nonfinite");

    return result;
}

/// @brief pulls a flat-coordinate residual back to metric nodal tangents
template <typename Geometry, typename Linearization, typename Nodes>
P1ObjectiveContributionResult<typename Geometry::Tangent> p1_flat_coordinate_data_site_contribution_impl(
  const Geometry& geometry, const Nodes& nodes, Linearization linearization,
  const typename Geometry::Tangent& observation_chart) {
    using Tangent = typename Geometry::Tangent;
    const auto& value_result = linearization.result();
    const auto value = p1_flat_coordinate_data_site_value_impl(geometry, value_result, observation_chart);

    P1ObjectiveContributionResult<Tangent> result;
    result.value = value.value;
    result.first_failure = value.first_failure;
    result.nodal_gradient = p1_objective_zero_gradient(geometry, nodes);
    /// @brief reports whether the required mean and derivative solves succeeded
    if (!result.converged()) return result;

    const Tangent value_chart = p1_flat_chart(geometry, value_result.value);
    const Tangent residual(value_chart - observation_chart);
    const Tangent value_gradient = p1_flat_inverse_chart_jvp(geometry, value_chart, residual);
    result.nodal_gradient = linearization.nodal_vjp(value_gradient);
    for (const Tangent& gradient : result.nodal_gradient) {
        p1_objective_require_finite_shape<std::domain_error>(
          gradient, geometry.order(), "P1 flat-coordinate nodal gradient is nonfinite");
    }
    return result;
}

template <bool WithGradient, typename Tangent>
using P1ObjectiveResult =
  std::conditional_t<WithGradient, P1ObjectiveContributionResult<Tangent>, P1ObjectiveValueResult>;

/// @brief evaluates arithmetic P1 interpolation with optional metric gradients
template <bool WithGradient, typename Geometry, typename Nodes>
P1ObjectiveResult<WithGradient, typename Geometry::Tangent> p1_ambient_frobenius_data_site_impl(
  const Geometry& geometry, const Nodes& nodes, std::span<const double> weights,
  const typename Geometry::Tangent& observation) {
    using Tangent = typename Geometry::Tangent;
    validate_p1_data(nodes.size(), weights);
    p1_objective_require_finite_shape(
      observation, geometry.order(),
      "P1 ambient Frobenius observation has incompatible dimensions or nonfinite coefficients");
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto& node = nodes[i];
        p1_objective_require_finite_shape(
          node, geometry.order(), "P1 ambient Frobenius node has incompatible dimensions or nonfinite coefficients");
    }

    long double weight_total = 0;
    long double weight_correction = 0;
    for (const double weight : weights) {
        const long double corrected = static_cast<long double>(weight) - weight_correction;
        const long double next = weight_total + corrected;
        weight_correction = (next - weight_total) - corrected;
        weight_total = next;
    }

    Tangent interpolated = geometry.zero_tangent(nodes[0]);
    for (int row = 0; row < geometry.order(); ++row) {
        for (int col = 0; col <= row; ++col) {
            long double sum = 0;
            long double correction = 0;
            for (std::size_t node = 0; node < nodes.size(); ++node) {
                const long double contribution =
                  static_cast<long double>(weights[node]) * static_cast<long double>(nodes[node](row, col));
                const long double corrected = contribution - correction;
                const long double next = sum + corrected;
                correction = (next - sum) - corrected;
                sum = next;
            }
            interpolated(row, col) = static_cast<typename Geometry::Scalar>(sum / weight_total);
        }
    }
    p1_objective_require_finite_shape<std::domain_error>(
      interpolated, geometry.order(), "P1 ambient Frobenius interpolation is nonfinite");

    const Tangent residual(interpolated - observation);
    const double residual_norm = static_cast<double>(residual.norm());
    P1ObjectiveResult<WithGradient, Tangent> result;
    result.value = 0.5 * residual_norm * residual_norm;
    fdapde_strong_assert(
      std::isfinite(result.value), std::domain_error, "P1 ambient Frobenius data contribution is nonfinite");

    if constexpr (WithGradient) {
        result.nodal_gradient.reserve(nodes.size());
        for (std::size_t node = 0; node < nodes.size(); ++node) {
            if (weights[node] == 0) {
                result.nodal_gradient.push_back(geometry.zero_tangent(nodes[node]));
                continue;
            }
            Tangent ambient_gradient = geometry.zero_tangent(nodes[node]);
            const long double coefficient = static_cast<long double>(weights[node]) / weight_total;
            for (int row = 0; row < geometry.order(); ++row) {
                for (int col = 0; col <= row; ++col) {
                    ambient_gradient(row, col) = static_cast<typename Geometry::Scalar>(
                      coefficient * static_cast<long double>(residual(row, col)));
                }
            }
            p1_objective_require_finite_shape<std::domain_error>(
              ambient_gradient, geometry.order(), "P1 ambient Frobenius Euclidean gradient is nonfinite");
            result.nodal_gradient.push_back(geometry.euclidean_to_riemannian_gradient(nodes[node], ambient_gradient));
            p1_objective_require_finite_shape<std::domain_error>(
              result.nodal_gradient.back(), geometry.order(), "P1 ambient Frobenius metric gradient is nonfinite");
        }
    }
    return result;
}

/// @brief integrates spatial interpolation energy with optional mixed pullbacks
template <
  bool WithGradient, typename Geometry, std::size_t LocalDim, std::size_t EmbedDim, std::size_t QuadratureSize,
  typename Builder, typename Nodes>
P1ObjectiveResult<WithGradient, typename Geometry::Tangent> p1_dirichlet_cell_objective_impl(
  const Geometry& geometry, const Nodes& nodal_values,
  const P1FEMCellQuadrature<LocalDim, EmbedDim, QuadratureSize>& packet, Builder&& build_linearization) {
    using Point = typename Geometry::Point;
    using Tangent = typename Geometry::Tangent;
    static_assert(LocalDim >= 1 && LocalDim <= 3);
    static_assert(EmbedDim >= LocalDim);
    static_assert(QuadratureSize > 0);

    for (const auto dof : packet.dofs)
        fdapde_strong_assert(dof < nodal_values.size(), std::out_of_range, "P1 cell DOF out of range");
    auto nodes = [&] {
        if constexpr (requires { nodal_values.select(packet.dofs); })
            return nodal_values.select(packet.dofs);
        else {
            MatrixBatch<Point> local(packet.node_count, geometry.order(), geometry.order());
            for (std::size_t i = 0; i < packet.node_count; ++i) local[i] = nodal_values[packet.dofs[i]];
            return local;
        }
    }();

    for (const auto& direction : packet.physical_weight_gradients) {
        p1_objective_validate_spatial_direction(std::span<const double>(direction));
    }
    for (std::size_t site = 0; site < packet.quadrature_size; ++site) {
        internals::validate_p1_data(packet.node_count, std::span<const double>(packet.barycentric_weights[site]));
        const double weight = packet.integration_weights[site];
        fdapde_strong_assert(
          std::isfinite(weight) && weight >= 0, std::invalid_argument,
          "P1 Dirichlet integration weights must be finite and non-negative");
    }

    P1ObjectiveResult<WithGradient, Tangent> result;
    if constexpr (WithGradient) { result.nodal_gradient = p1_objective_zero_gradient(geometry, nodes); }
    for (std::size_t site = 0; site < packet.quadrature_size; ++site) {
        const double integration_weight = packet.integration_weights[site];
        if (integration_weight == 0) continue;

        auto linearization = build_linearization(nodes, std::span<const double>(packet.barycentric_weights[site]));
        const auto& value_result = linearization.result();
        if (!value_result.converged()) {
            result.first_failure = p1_objective_mean_failure(site, value_result);
            return result;
        }

        double site_value = 0;
        std::vector<Tangent> site_gradient;
        if constexpr (WithGradient) { site_gradient = p1_objective_zero_gradient(geometry, nodes); }
        for (std::size_t axis = 0; axis < packet.embed_dim; ++axis) {
            const std::span<const double> direction(packet.physical_weight_gradients[axis]);
            if constexpr (is_flat_spd_geometry<Geometry>) {
                const Tangent spatial = linearization.weight_jvp(direction);
                const double squared_norm = geometry.inner_product(value_result.value, spatial, spatial);
                fdapde_strong_assert(
                  std::isfinite(squared_norm) && squared_norm >= 0, std::domain_error,
                  "P1 Dirichlet spatial energy is invalid");

                site_value = std::fma(0.5, squared_norm, site_value);
                if constexpr (WithGradient) {
                    auto pullback = linearization.covariant_mixed_nodal_vjp(direction, spatial);
                    for (std::size_t node = 0; node < packet.node_count; ++node) {
                        site_gradient[node] =
                          geometry.linear_combination(nodes[node], 1, site_gradient[node], 1, pullback[node]);
                    }
                }
            } else {
                auto spatial = linearization.weight_jvp(direction);
                if (!spatial.converged()) {
                    result.first_failure = p1_objective_solve_failure(
                      P1ObjectiveStage::dirichlet_spatial, site, axis,
                      {spatial.residual_norm, spatial.iterations, spatial.stop_reason});
                    return result;
                }

                const double squared_norm =
                  geometry.inner_product(value_result.value, spatial.derivative, spatial.derivative);
                fdapde_strong_assert(
                  std::isfinite(squared_norm) && squared_norm >= 0, std::domain_error,
                  "P1 Dirichlet spatial energy is invalid");

                site_value = std::fma(0.5, squared_norm, site_value);
                if constexpr (WithGradient) {
                    auto pullback = linearization.covariant_mixed_nodal_vjp(direction, spatial.derivative);
                    // status zero repeats the same deterministic weight solve
                    // certified above, so an unexpected failure is still spatial
                    constexpr std::array<P1ObjectiveStage, 3> stages {
                      P1ObjectiveStage::dirichlet_spatial, P1ObjectiveStage::dirichlet_mixed_output,
                      P1ObjectiveStage::dirichlet_mixed_pullback};
                    for (std::size_t solve = 0; solve < stages.size(); ++solve) {
                        if (!pullback.solve_statuses[solve].converged()) {
                            result.first_failure =
                              p1_objective_solve_failure(stages[solve], site, axis, pullback.solve_statuses[solve]);
                            return result;
                        }
                    }
                    for (std::size_t node = 0; node < packet.node_count; ++node) {
                        site_gradient[node] = geometry.linear_combination(
                          nodes[node], 1, site_gradient[node], 1, pullback.derivative[node]);
                    }
                }
            }
        }

        result.value = std::fma(integration_weight, site_value, result.value);
        fdapde_strong_assert(
          std::isfinite(result.value), std::domain_error, "P1 Dirichlet cell contribution is nonfinite");

        if constexpr (WithGradient) {
            for (std::size_t node = 0; node < packet.node_count; ++node) {
                result.nodal_gradient[node] = geometry.linear_combination(
                  nodes[node], 1, result.nodal_gradient[node], integration_weight, site_gradient[node]);
                p1_objective_require_finite_shape<std::domain_error>(
                  result.nodal_gradient[node], geometry.order(), "P1 Dirichlet nodal gradient is nonfinite");
            }
        }
    }
    return result;
}

}   // namespace internals

// the observation is log(D), not D. Computing and validating it once belongs
// to the data-ingest boundary
/// @brief evaluates the LE log-coordinate data loss against a precomputed observation log
template <typename Scalar_, int Order_, Usage Uses, typename Nodes, typename Point_>
P1ObjectiveValueResult p1_log_coordinate_data_site_value(
  const manifold::LogEuclideanSPDGeometry<Scalar_, Order_, Uses, Point_>& geometry, const Nodes& nodal_values,
  std::span<const double> barycentric_weights,
  const typename manifold::LogEuclideanSPDGeometry<Scalar_, Order_, Uses, Point_>::Tangent& observation_log) {
    const auto value_result = p1_geodesic_value(geometry, nodal_values, barycentric_weights);
    return internals::p1_flat_coordinate_data_site_value_impl(geometry, value_result, observation_log);
}

// the returned metric gradients follow nodal_values order. Observation
// weighting and global scattering belong to the caller
/// @brief returns the LE log-coordinate loss and metric gradients in nodal order
template <typename Scalar_, int Order_, Usage Uses, typename Nodes, typename Point_>
P1ObjectiveContributionResult<typename manifold::LogEuclideanSPDGeometry<Scalar_, Order_, Uses, Point_>::Tangent>
p1_log_coordinate_data_site_contribution(
  const manifold::LogEuclideanSPDGeometry<Scalar_, Order_, Uses, Point_>& geometry, const Nodes& nodal_values,
  std::span<const double> barycentric_weights,
  const typename manifold::LogEuclideanSPDGeometry<Scalar_, Order_, Uses, Point_>::Tangent& observation_log) {
    auto linearization = p1_geodesic_linearization(geometry, nodal_values, barycentric_weights);
    return internals::p1_flat_coordinate_data_site_contribution_impl(
      geometry, nodal_values, std::move(linearization), observation_log);
}

/// @brief evaluates flat-coordinate loss against a precomputed metric chart observation
template <typename Geometry, typename Nodes>
    requires internals::is_flat_spd_geometry<Geometry>
P1ObjectiveValueResult p1_flat_coordinate_data_site_value(
  const Geometry& geometry, const Nodes& nodes, std::span<const double> weights,
  const typename Geometry::Tangent& observation_chart) {
    const auto mean = p1_geodesic_value(geometry, nodes, weights);
    return internals::p1_flat_coordinate_data_site_value_impl(geometry, mean, observation_chart);
}

/// @brief returns flat-coordinate loss and its metric gradients in nodal order
template <typename Geometry, typename Nodes>
    requires internals::is_flat_spd_geometry<Geometry>
P1ObjectiveContributionResult<typename Geometry::Tangent> p1_flat_coordinate_data_site_contribution(
  const Geometry& geometry, const Nodes& nodes, std::span<const double> weights,
  const typename Geometry::Tangent& observation_chart) {
    auto linearization = p1_geodesic_linearization(geometry, nodes, weights);
    return internals::p1_flat_coordinate_data_site_contribution_impl(
      geometry, nodes, std::move(linearization), observation_chart);
}

/// @brief evaluates the Frobenius loss of arithmetic P1 interpolation
template <typename Scalar_, int Order_, Usage Uses, typename Nodes, typename Point_>
P1ObjectiveValueResult p1_ambient_frobenius_data_site_value(
  const manifold::LogEuclideanSPDGeometry<Scalar_, Order_, Uses, Point_>& geometry, const Nodes& nodal_values,
  std::span<const double> barycentric_weights,
  const typename manifold::LogEuclideanSPDGeometry<Scalar_, Order_, Uses, Point_>::Tangent& observation) {
    return internals::p1_ambient_frobenius_data_site_impl<false>(
      geometry, nodal_values, barycentric_weights, observation);
}

/// @brief evaluates the Frobenius loss of arithmetic P1 interpolation
template <typename Geometry, typename Nodes>
    requires internals::is_flat_spd_geometry<Geometry> && (!internals::is_log_euclidean_spd_geometry<Geometry>)
P1ObjectiveValueResult p1_ambient_frobenius_data_site_value(
  const Geometry& geometry, const Nodes& nodal_values, std::span<const double> barycentric_weights,
  const typename Geometry::Tangent& observation) {
    return internals::p1_ambient_frobenius_data_site_impl<false>(
      geometry, nodal_values, barycentric_weights, observation);
}

/// @brief evaluates the Frobenius loss of arithmetic P1 interpolation
template <typename Scalar_, int Order_, Usage Uses, typename Nodes, typename Point_>
P1ObjectiveValueResult p1_ambient_frobenius_data_site_value(
  const manifold::AffineInvariantSPDGeometry<Scalar_, Order_, Uses, Point_>& geometry, const Nodes& nodal_values,
  std::span<const double> barycentric_weights,
  const typename manifold::AffineInvariantSPDGeometry<Scalar_, Order_, Uses, Point_>::Tangent& observation) {
    return internals::p1_ambient_frobenius_data_site_impl<false>(
      geometry, nodal_values, barycentric_weights, observation);
}

// the returned metric gradients follow nodal_values order. This objective
// uses ordinary ambient P1 interpolation, not a Karcher mean
/// @brief returns the arithmetic P1 Frobenius loss and metric gradients in nodal order
template <typename Scalar_, int Order_, Usage Uses, typename Nodes, typename Point_>
P1ObjectiveContributionResult<typename manifold::LogEuclideanSPDGeometry<Scalar_, Order_, Uses, Point_>::Tangent>
p1_ambient_frobenius_data_site_contribution(
  const manifold::LogEuclideanSPDGeometry<Scalar_, Order_, Uses, Point_>& geometry, const Nodes& nodal_values,
  std::span<const double> barycentric_weights,
  const typename manifold::LogEuclideanSPDGeometry<Scalar_, Order_, Uses, Point_>::Tangent& observation) {
    return internals::p1_ambient_frobenius_data_site_impl<true>(
      geometry, nodal_values, barycentric_weights, observation);
}

/// @brief returns the arithmetic P1 Frobenius loss and metric gradients in nodal order
template <typename Geometry, typename Nodes>
    requires internals::is_flat_spd_geometry<Geometry> && (!internals::is_log_euclidean_spd_geometry<Geometry>)
P1ObjectiveContributionResult<typename Geometry::Tangent> p1_ambient_frobenius_data_site_contribution(
  const Geometry& geometry, const Nodes& nodal_values, std::span<const double> barycentric_weights,
  const typename Geometry::Tangent& observation) {
    return internals::p1_ambient_frobenius_data_site_impl<true>(
      geometry, nodal_values, barycentric_weights, observation);
}

/// @brief returns the arithmetic P1 Frobenius loss and metric gradients in nodal order
template <typename Scalar_, int Order_, Usage Uses, typename Nodes, typename Point_>
P1ObjectiveContributionResult<typename manifold::AffineInvariantSPDGeometry<Scalar_, Order_, Uses, Point_>::Tangent>
p1_ambient_frobenius_data_site_contribution(
  const manifold::AffineInvariantSPDGeometry<Scalar_, Order_, Uses, Point_>& geometry, const Nodes& nodal_values,
  std::span<const double> barycentric_weights,
  const typename manifold::AffineInvariantSPDGeometry<Scalar_, Order_, Uses, Point_>::Tangent& observation) {
    return internals::p1_ambient_frobenius_data_site_impl<true>(
      geometry, nodal_values, barycentric_weights, observation);
}

/// @brief evaluates the Frobenius loss of geodesic P1 interpolation
template <typename Scalar_, int Order_, Usage Uses, typename Nodes, typename Point_>
P1ObjectiveValueResult p1_frobenius_data_site_value(
  const manifold::LogEuclideanSPDGeometry<Scalar_, Order_, Uses, Point_>& geometry, const Nodes& nodal_values,
  std::span<const double> barycentric_weights,
  const typename manifold::LogEuclideanSPDGeometry<Scalar_, Order_, Uses, Point_>::Tangent& observation) {
    const auto value_result = p1_geodesic_value(geometry, nodal_values, barycentric_weights);
    return internals::p1_frobenius_data_site_value_impl(geometry, value_result, observation);
}

/// @brief evaluates the Frobenius loss of geodesic P1 interpolation
template <typename Geometry, typename Nodes>
    requires internals::is_flat_spd_geometry<Geometry> && (!internals::is_log_euclidean_spd_geometry<Geometry>)
P1ObjectiveValueResult p1_frobenius_data_site_value(
  const Geometry& geometry, const Nodes& nodal_values, std::span<const double> barycentric_weights,
  const typename Geometry::Tangent& observation) {
    const auto value_result = p1_geodesic_value(geometry, nodal_values, barycentric_weights);
    return internals::p1_frobenius_data_site_value_impl(geometry, value_result, observation);
}

/// @brief evaluates the Frobenius loss of geodesic P1 interpolation
template <typename Scalar_, int Order_, Usage Uses, typename Nodes, typename Point_>
P1ObjectiveValueResult p1_frobenius_data_site_value(
  const manifold::AffineInvariantSPDGeometry<Scalar_, Order_, Uses, Point_>& geometry, const Nodes& nodal_values,
  std::span<const double> barycentric_weights,
  const typename manifold::AffineInvariantSPDGeometry<Scalar_, Order_, Uses, Point_>::Tangent& observation,
  const manifold::WeightedKarcherMeanOptions& options = {}) {
    const auto value_result = p1_geodesic_value(geometry, nodal_values, barycentric_weights, options);
    return internals::p1_frobenius_data_site_value_impl(geometry, value_result, observation);
}

/// @brief returns the geodesic P1 Frobenius loss and metric gradients in nodal order
template <typename Scalar_, int Order_, Usage Uses, typename Nodes, typename Point_>
P1ObjectiveContributionResult<typename manifold::LogEuclideanSPDGeometry<Scalar_, Order_, Uses, Point_>::Tangent>
p1_frobenius_data_site_contribution(
  const manifold::LogEuclideanSPDGeometry<Scalar_, Order_, Uses, Point_>& geometry, const Nodes& nodal_values,
  std::span<const double> barycentric_weights,
  const typename manifold::LogEuclideanSPDGeometry<Scalar_, Order_, Uses, Point_>::Tangent& observation) {
    auto linearization = p1_geodesic_linearization(geometry, nodal_values, barycentric_weights);
    return internals::p1_frobenius_data_site_contribution_impl(
      geometry, nodal_values, std::move(linearization), observation);
}

/// @brief returns the geodesic P1 Frobenius loss and metric gradients in nodal order
template <typename Geometry, typename Nodes>
    requires internals::is_flat_spd_geometry<Geometry> && (!internals::is_log_euclidean_spd_geometry<Geometry>)
P1ObjectiveContributionResult<typename Geometry::Tangent> p1_frobenius_data_site_contribution(
  const Geometry& geometry, const Nodes& nodal_values, std::span<const double> barycentric_weights,
  const typename Geometry::Tangent& observation) {
    auto linearization = p1_geodesic_linearization(geometry, nodal_values, barycentric_weights);
    return internals::p1_frobenius_data_site_contribution_impl(
      geometry, nodal_values, std::move(linearization), observation);
}

/// @brief returns the geodesic P1 Frobenius loss and metric gradients in nodal order
template <typename Scalar_, int Order_, Usage Uses, typename Nodes, typename Point_>
P1ObjectiveContributionResult<typename manifold::AffineInvariantSPDGeometry<Scalar_, Order_, Uses, Point_>::Tangent>
p1_frobenius_data_site_contribution(
  const manifold::AffineInvariantSPDGeometry<Scalar_, Order_, Uses, Point_>& geometry, const Nodes& nodal_values,
  std::span<const double> barycentric_weights,
  const typename manifold::AffineInvariantSPDGeometry<Scalar_, Order_, Uses, Point_>::Tangent& observation,
  const P1GeodesicLinearizationOptions& options = {}) {
    auto linearization = p1_geodesic_linearization(geometry, nodal_values, barycentric_weights, options);
    return internals::p1_frobenius_data_site_contribution_impl(
      geometry, nodal_values, std::move(linearization), observation);
}

/// @brief evaluates the Frobenius loss of BW P1 interpolation
template <typename Scalar, int Order, Usage Uses, typename Nodes, typename Point_>
P1ObjectiveValueResult p1_frobenius_data_site_value(
  const manifold::BuresWassersteinSPDGeometry<Scalar, Order, Uses, Point_>& geometry, const Nodes& nodal_values,
  std::span<const double> barycentric_weights,
  const typename manifold::BuresWassersteinSPDGeometry<Scalar, Order, Uses, Point_>::Tangent& observation,
  const manifold::WeightedKarcherMeanOptions& options = {}) {
    const auto value_result = p1_geodesic_value(geometry, nodal_values, barycentric_weights, options);
    return internals::p1_frobenius_data_site_value_impl(geometry, value_result, observation);
}

/// @brief returns the BW P1 Frobenius loss and its metric gradients in nodal order
template <typename Scalar, int Order, Usage Uses, typename Nodes, typename Point_>
P1ObjectiveContributionResult<typename manifold::BuresWassersteinSPDGeometry<Scalar, Order, Uses, Point_>::Tangent>
p1_frobenius_data_site_contribution(
  const manifold::BuresWassersteinSPDGeometry<Scalar, Order, Uses, Point_>& geometry, const Nodes& nodal_values,
  std::span<const double> barycentric_weights,
  const typename manifold::BuresWassersteinSPDGeometry<Scalar, Order, Uses, Point_>::Tangent& observation,
  const P1GeodesicLinearizationOptions& options = {}) {
    auto linearization = p1_geodesic_linearization(geometry, nodal_values, barycentric_weights, options);
    return internals::p1_frobenius_data_site_contribution_impl(
      geometry, nodal_values, std::move(linearization), observation);
}

/// @brief integrates half the squared spatial metric derivative over one cell
template <
  typename Scalar_, int Order_, std::size_t LocalDim, std::size_t EmbedDim, std::size_t QuadratureSize, Usage Uses,
  typename Nodes, typename Point_>
P1ObjectiveValueResult p1_dirichlet_cell_value(
  const manifold::LogEuclideanSPDGeometry<Scalar_, Order_, Uses, Point_>& geometry, const Nodes& nodal_values,
  const P1FEMCellQuadrature<LocalDim, EmbedDim, QuadratureSize>& packet) {
    auto builder = [&geometry](const auto& nodes, auto weights) {
        return p1_geodesic_linearization(geometry, nodes, weights);
    };
    return internals::p1_dirichlet_cell_objective_impl<false>(geometry, nodal_values, packet, builder);
}

/// @brief integrates half the squared spatial metric derivative over one cell
template <typename Geometry, std::size_t LocalDim, std::size_t EmbedDim, std::size_t QuadratureSize, typename Nodes>
    requires internals::is_flat_spd_geometry<Geometry> && (!internals::is_log_euclidean_spd_geometry<Geometry>)
P1ObjectiveValueResult p1_dirichlet_cell_value(
  const Geometry& geometry, const Nodes& nodal_values,
  const P1FEMCellQuadrature<LocalDim, EmbedDim, QuadratureSize>& packet) {
    auto builder = [&geometry](const auto& nodes, auto weights) {
        return p1_geodesic_linearization(geometry, nodes, weights);
    };
    return internals::p1_dirichlet_cell_objective_impl<false>(geometry, nodal_values, packet, builder);
}

/// @brief integrates half the squared spatial metric derivative over one cell
template <
  typename Scalar_, int Order_, std::size_t LocalDim, std::size_t EmbedDim, std::size_t QuadratureSize, Usage Uses,
  typename Nodes, typename Point_>
P1ObjectiveValueResult p1_dirichlet_cell_value(
  const manifold::AffineInvariantSPDGeometry<Scalar_, Order_, Uses, Point_>& geometry, const Nodes& nodal_values,
  const P1FEMCellQuadrature<LocalDim, EmbedDim, QuadratureSize>& packet,
  const P1GeodesicLinearizationOptions& options = {}) {
    auto builder = [&geometry, &options](const auto& nodes, auto weights) {
        return p1_geodesic_linearization(geometry, nodes, weights, options);
    };
    return internals::p1_dirichlet_cell_objective_impl<false>(geometry, nodal_values, packet, builder);
}

// the returned gradient is packet-local: entry i is based at nodal_values[packet.dofs[i]]
// global scattering and lambda scaling belong to the caller
/// @brief returns cell Dirichlet energy and metric gradients in packet dof order
template <
  typename Scalar_, int Order_, std::size_t LocalDim, std::size_t EmbedDim, std::size_t QuadratureSize, Usage Uses,
  typename Nodes, typename Point_>
P1ObjectiveContributionResult<typename manifold::LogEuclideanSPDGeometry<Scalar_, Order_, Uses, Point_>::Tangent>
p1_dirichlet_cell_contribution(
  const manifold::LogEuclideanSPDGeometry<Scalar_, Order_, Uses, Point_>& geometry, const Nodes& nodal_values,
  const P1FEMCellQuadrature<LocalDim, EmbedDim, QuadratureSize>& packet) {
    auto builder = [&geometry](const auto& nodes, auto weights) {
        return p1_geodesic_linearization(geometry, nodes, weights);
    };
    return internals::p1_dirichlet_cell_objective_impl<true>(geometry, nodal_values, packet, builder);
}

/// @brief returns cell Dirichlet energy and metric gradients in packet dof order
template <typename Geometry, std::size_t LocalDim, std::size_t EmbedDim, std::size_t QuadratureSize, typename Nodes>
    requires internals::is_flat_spd_geometry<Geometry> && (!internals::is_log_euclidean_spd_geometry<Geometry>)
P1ObjectiveContributionResult<typename Geometry::Tangent> p1_dirichlet_cell_contribution(
  const Geometry& geometry, const Nodes& nodal_values,
  const P1FEMCellQuadrature<LocalDim, EmbedDim, QuadratureSize>& packet) {
    auto builder = [&geometry](const auto& nodes, auto weights) {
        return p1_geodesic_linearization(geometry, nodes, weights);
    };
    return internals::p1_dirichlet_cell_objective_impl<true>(geometry, nodal_values, packet, builder);
}

// the returned gradient follows the same packet-local contract as the
// log-Euclidean overload above
/// @brief returns cell Dirichlet energy and metric gradients in packet dof order
template <
  typename Scalar_, int Order_, std::size_t LocalDim, std::size_t EmbedDim, std::size_t QuadratureSize, Usage Uses,
  typename Nodes, typename Point_>
P1ObjectiveContributionResult<typename manifold::AffineInvariantSPDGeometry<Scalar_, Order_, Uses, Point_>::Tangent>
p1_dirichlet_cell_contribution(
  const manifold::AffineInvariantSPDGeometry<Scalar_, Order_, Uses, Point_>& geometry, const Nodes& nodal_values,
  const P1FEMCellQuadrature<LocalDim, EmbedDim, QuadratureSize>& packet,
  const P1GeodesicLinearizationOptions& options = {}) {
    auto builder = [&geometry, &options](const auto& nodes, auto weights) {
        return p1_geodesic_linearization(geometry, nodes, weights, options);
    };
    return internals::p1_dirichlet_cell_objective_impl<true>(geometry, nodal_values, packet, builder);
}

}   // namespace gfe
}   // namespace fdapde

#endif   // __FDAPDE_GFE_P1_OBJECTIVE_CONTRIBUTIONS_H__
