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

enum class P1ObjectiveStage {
    mean,
    data_pullback,
    dirichlet_spatial,
    dirichlet_mixed_output,
    dirichlet_mixed_pullback
};

struct P1ObjectiveFailure {
    P1ObjectiveStage stage = P1ObjectiveStage::mean;
    std::size_t site = 0;
    std::optional<std::size_t> axis;
    std::optional<manifold::BarycenterStopReason> barycenter_stop_reason;
    double stationarity_norm = std::numeric_limits<double>::quiet_NaN();
    std::optional<P1LinearSolveStatus> linear_solve;
};

struct P1ObjectiveValueResult {
    // On failure this is only the last available local candidate and must
    // not be consumed by an optimizer.
    double value = 0;
    std::optional<P1ObjectiveFailure> first_failure;

    bool converged() const noexcept { return !first_failure.has_value(); }
};

template <typename Tangent> struct P1ObjectiveContributionResult {
    // On failure these are only the last available local candidates and
    // must not be consumed by an optimizer.
    double value = 0;
    std::vector<Tangent> nodal_gradient;
    std::optional<P1ObjectiveFailure> first_failure;

    bool converged() const noexcept { return !first_failure.has_value(); }
};

namespace internals {

template <typename Exception = std::invalid_argument, typename Matrix>
void p1_objective_require_finite_shape(const Matrix& matrix, int order, const char* message) {
    if (matrix.rows() != order || matrix.cols() != order) { throw Exception(message); }
    for (int i = 0; i < order; ++i) {
        for (int j = 0; j <= i; ++j) {
            if (!std::isfinite(static_cast<double>(matrix(i, j)))) { throw Exception(message); }
        }
    }
}

inline void p1_objective_validate_spatial_direction(std::span<const double> direction) {
    long double sum = 0;
    long double correction = 0;
    long double absolute_sum = 0;
    for (const double coefficient : direction) {
        if (!std::isfinite(coefficient)) {
            throw std::invalid_argument("P1 Dirichlet physical weight gradients must be finite");
        }
        const long double value = static_cast<long double>(coefficient);
        const long double corrected = value - correction;
        const long double next = sum + corrected;
        correction = (next - sum) - corrected;
        sum = next;
        absolute_sum += std::abs(value);
    }
    const long double tolerance =
      static_cast<long double>(p1_weight_sum_tolerance(direction.size())) * std::max(1.0L, absolute_sum);
    if (std::abs(sum) > tolerance) {
        throw std::invalid_argument("P1 Dirichlet physical weight gradients must sum to zero");
    }
}

template <typename ValueResult>
P1ObjectiveFailure p1_objective_mean_failure(std::size_t site, const ValueResult& result) {
    return {P1ObjectiveStage::mean, site, std::nullopt, result.stop_reason, result.stationarity_norm, std::nullopt};
}

inline P1ObjectiveFailure p1_objective_solve_failure(
  P1ObjectiveStage stage, std::size_t site, std::optional<std::size_t> axis, const P1LinearSolveStatus& solve) {
    return {stage, site, axis, std::nullopt, std::numeric_limits<double>::quiet_NaN(), solve};
}

template <typename Geometry>
std::vector<typename Geometry::Tangent>
p1_objective_zero_gradient(const Geometry& geometry, std::span<const typename Geometry::Point> nodes) {
    std::vector<typename Geometry::Tangent> result;
    result.reserve(nodes.size());
    for (const auto& node : nodes) { result.push_back(geometry.zero_tangent(node)); }
    return result;
}

template <typename Geometry, typename ValueResult>
P1ObjectiveValueResult p1_frobenius_data_site_value_impl(
  const Geometry& geometry, const ValueResult& value_result, const typename Geometry::Tangent& observation) {
    using Tangent = typename Geometry::Tangent;
    p1_objective_require_finite_shape(
      observation, geometry.order(), "P1 Frobenius observation has incompatible dimensions or nonfinite coefficients");

    P1ObjectiveValueResult result;
    if (!value_result.converged()) {
        result.first_failure = p1_objective_mean_failure(0, value_result);
        return result;
    }

    const Tangent ambient_gradient(value_result.value - observation);
    const double ambient_norm = static_cast<double>(ambient_gradient.norm());
    result.value = 0.5 * ambient_norm * ambient_norm;
    if (!std::isfinite(result.value)) { throw std::domain_error("P1 Frobenius data contribution is nonfinite"); }
    return result;
}

template <typename Geometry, typename Linearization>
P1ObjectiveContributionResult<typename Geometry::Tangent> p1_frobenius_data_site_contribution_impl(
  const Geometry& geometry, std::span<const typename Geometry::Point> nodes, Linearization linearization,
  const typename Geometry::Tangent& observation) {
    using Tangent = typename Geometry::Tangent;
    const auto& value_result = linearization.result();
    const auto value = p1_frobenius_data_site_value_impl(geometry, value_result, observation);

    P1ObjectiveContributionResult<Tangent> result;
    result.value = value.value;
    result.first_failure = value.first_failure;
    result.nodal_gradient = p1_objective_zero_gradient(geometry, nodes);
    if (!result.converged()) return result;

    const Tangent ambient_gradient(value_result.value - observation);
    const Tangent value_gradient = geometry.euclidean_to_riemannian_gradient(value_result.value, ambient_gradient);
    p1_objective_require_finite_shape<std::domain_error>(
      value_gradient, geometry.order(), "P1 Frobenius metric gradient is nonfinite");

    if constexpr (is_log_euclidean_spd_geometry<Geometry>) {
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

template <bool WithGradient, typename Tangent>
using P1ObjectiveResult =
  std::conditional_t<WithGradient, P1ObjectiveContributionResult<Tangent>, P1ObjectiveValueResult>;

template <
  bool WithGradient, typename Geometry, std::size_t LocalDim, std::size_t EmbedDim, std::size_t QuadratureSize,
  typename Builder>
P1ObjectiveResult<WithGradient, typename Geometry::Tangent> p1_dirichlet_cell_objective_impl(
  const Geometry& geometry, std::span<const typename Geometry::Point> nodal_values,
  const P1FEMCellQuadrature<LocalDim, EmbedDim, QuadratureSize>& packet, Builder&& build_linearization) {
    using Point = typename Geometry::Point;
    using Tangent = typename Geometry::Tangent;
    static_assert(LocalDim >= 1 && LocalDim <= 3);
    static_assert(EmbedDim >= LocalDim);
    static_assert(QuadratureSize > 0);

    std::vector<Point> local_nodes;
    local_nodes.reserve(packet.node_count);
    for (const std::size_t dof : packet.dofs) {
        if (dof >= nodal_values.size()) {
            throw std::out_of_range("P1 Dirichlet cell degree of freedom is out of range");
        }
        p1_objective_require_finite_shape(
          nodal_values[dof], geometry.order(), "P1 Dirichlet nodal value has incompatible dimensions or is nonfinite");
        local_nodes.push_back(nodal_values[dof]);
    }
    const std::span<const Point> nodes(local_nodes);

    for (const auto& direction : packet.physical_weight_gradients) {
        p1_objective_validate_spatial_direction(std::span<const double>(direction));
    }
    for (std::size_t site = 0; site < packet.quadrature_size; ++site) {
        internals::validate_p1_data(packet.node_count, std::span<const double>(packet.barycentric_weights[site]));
        const double weight = packet.integration_weights[site];
        if (!std::isfinite(weight) || weight < 0) {
            throw std::invalid_argument("P1 Dirichlet integration weights must be finite and non-negative");
        }
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
            if constexpr (is_log_euclidean_spd_geometry<Geometry>) {
                const Tangent spatial = linearization.weight_jvp(direction);
                const double squared_norm = geometry.inner_product(value_result.value, spatial, spatial);
                if (!std::isfinite(squared_norm) || squared_norm < 0) {
                    throw std::domain_error("P1 Dirichlet spatial energy is invalid");
                }
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
                if (!std::isfinite(squared_norm) || squared_norm < 0) {
                    throw std::domain_error("P1 Dirichlet spatial energy is invalid");
                }
                site_value = std::fma(0.5, squared_norm, site_value);
                if constexpr (WithGradient) {
                    auto pullback = linearization.covariant_mixed_nodal_vjp(direction, spatial.derivative);
                    // Status zero repeats the same deterministic weight solve
                    // certified above, so an unexpected failure is still spatial.
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
        if (!std::isfinite(result.value)) { throw std::domain_error("P1 Dirichlet cell contribution is nonfinite"); }
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

template <typename Scalar_, int Order_>
P1ObjectiveValueResult p1_frobenius_data_site_value(
  const manifold::LogEuclideanSPDGeometry<Scalar_, Order_>& geometry,
  std::span<const typename manifold::LogEuclideanSPDGeometry<Scalar_, Order_>::Point> nodal_values,
  std::span<const double> barycentric_weights,
  const typename manifold::LogEuclideanSPDGeometry<Scalar_, Order_>::Tangent& observation) {
    const auto value_result = p1_geodesic_value(geometry, nodal_values, barycentric_weights);
    return internals::p1_frobenius_data_site_value_impl(geometry, value_result, observation);
}

template <typename Scalar_, int Order_>
P1ObjectiveValueResult p1_frobenius_data_site_value(
  const manifold::AffineInvariantSPDGeometry<Scalar_, Order_>& geometry,
  std::span<const typename manifold::AffineInvariantSPDGeometry<Scalar_, Order_>::Point> nodal_values,
  std::span<const double> barycentric_weights,
  const typename manifold::AffineInvariantSPDGeometry<Scalar_, Order_>::Tangent& observation,
  const manifold::WeightedKarcherMeanOptions& options = {}) {
    const auto value_result = p1_geodesic_value(geometry, nodal_values, barycentric_weights, options);
    return internals::p1_frobenius_data_site_value_impl(geometry, value_result, observation);
}

template <typename Scalar_, int Order_>
P1ObjectiveContributionResult<typename manifold::LogEuclideanSPDGeometry<Scalar_, Order_>::Tangent>
p1_frobenius_data_site_contribution(
  const manifold::LogEuclideanSPDGeometry<Scalar_, Order_>& geometry,
  std::span<const typename manifold::LogEuclideanSPDGeometry<Scalar_, Order_>::Point> nodal_values,
  std::span<const double> barycentric_weights,
  const typename manifold::LogEuclideanSPDGeometry<Scalar_, Order_>::Tangent& observation) {
    auto linearization = p1_geodesic_linearization(geometry, nodal_values, barycentric_weights);
    return internals::p1_frobenius_data_site_contribution_impl(
      geometry, nodal_values, std::move(linearization), observation);
}

template <typename Scalar_, int Order_>
P1ObjectiveContributionResult<typename manifold::AffineInvariantSPDGeometry<Scalar_, Order_>::Tangent>
p1_frobenius_data_site_contribution(
  const manifold::AffineInvariantSPDGeometry<Scalar_, Order_>& geometry,
  std::span<const typename manifold::AffineInvariantSPDGeometry<Scalar_, Order_>::Point> nodal_values,
  std::span<const double> barycentric_weights,
  const typename manifold::AffineInvariantSPDGeometry<Scalar_, Order_>::Tangent& observation,
  const P1GeodesicLinearizationOptions& options = {}) {
    auto linearization = p1_geodesic_linearization(geometry, nodal_values, barycentric_weights, options);
    return internals::p1_frobenius_data_site_contribution_impl(
      geometry, nodal_values, std::move(linearization), observation);
}

template <typename Scalar_, int Order_, std::size_t LocalDim, std::size_t EmbedDim, std::size_t QuadratureSize>
P1ObjectiveValueResult p1_dirichlet_cell_value(
  const manifold::LogEuclideanSPDGeometry<Scalar_, Order_>& geometry,
  std::span<const typename manifold::LogEuclideanSPDGeometry<Scalar_, Order_>::Point> nodal_values,
  const P1FEMCellQuadrature<LocalDim, EmbedDim, QuadratureSize>& packet) {
    auto builder = [&geometry](auto nodes, auto weights) {
        return p1_geodesic_linearization(geometry, nodes, weights);
    };
    return internals::p1_dirichlet_cell_objective_impl<false>(geometry, nodal_values, packet, builder);
}

template <typename Scalar_, int Order_, std::size_t LocalDim, std::size_t EmbedDim, std::size_t QuadratureSize>
P1ObjectiveValueResult p1_dirichlet_cell_value(
  const manifold::AffineInvariantSPDGeometry<Scalar_, Order_>& geometry,
  std::span<const typename manifold::AffineInvariantSPDGeometry<Scalar_, Order_>::Point> nodal_values,
  const P1FEMCellQuadrature<LocalDim, EmbedDim, QuadratureSize>& packet,
  const P1GeodesicLinearizationOptions& options = {}) {
    auto builder = [&geometry, &options](auto nodes, auto weights) {
        return p1_geodesic_linearization(geometry, nodes, weights, options);
    };
    return internals::p1_dirichlet_cell_objective_impl<false>(geometry, nodal_values, packet, builder);
}

// The returned gradient is packet-local: entry i is based at
// nodal_values[packet.dofs[i]]. Global scattering and lambda scaling belong
// to the caller.
template <typename Scalar_, int Order_, std::size_t LocalDim, std::size_t EmbedDim, std::size_t QuadratureSize>
P1ObjectiveContributionResult<typename manifold::LogEuclideanSPDGeometry<Scalar_, Order_>::Tangent>
p1_dirichlet_cell_contribution(
  const manifold::LogEuclideanSPDGeometry<Scalar_, Order_>& geometry,
  std::span<const typename manifold::LogEuclideanSPDGeometry<Scalar_, Order_>::Point> nodal_values,
  const P1FEMCellQuadrature<LocalDim, EmbedDim, QuadratureSize>& packet) {
    auto builder = [&geometry](auto nodes, auto weights) {
        return p1_geodesic_linearization(geometry, nodes, weights);
    };
    return internals::p1_dirichlet_cell_objective_impl<true>(geometry, nodal_values, packet, builder);
}

// The returned gradient follows the same packet-local contract as the
// log-Euclidean overload above.
template <typename Scalar_, int Order_, std::size_t LocalDim, std::size_t EmbedDim, std::size_t QuadratureSize>
P1ObjectiveContributionResult<typename manifold::AffineInvariantSPDGeometry<Scalar_, Order_>::Tangent>
p1_dirichlet_cell_contribution(
  const manifold::AffineInvariantSPDGeometry<Scalar_, Order_>& geometry,
  std::span<const typename manifold::AffineInvariantSPDGeometry<Scalar_, Order_>::Point> nodal_values,
  const P1FEMCellQuadrature<LocalDim, EmbedDim, QuadratureSize>& packet,
  const P1GeodesicLinearizationOptions& options = {}) {
    auto builder = [&geometry, &options](auto nodes, auto weights) {
        return p1_geodesic_linearization(geometry, nodes, weights, options);
    };
    return internals::p1_dirichlet_cell_objective_impl<true>(geometry, nodal_values, packet, builder);
}

}   // namespace gfe
}   // namespace fdapde

#endif   // __FDAPDE_GFE_P1_OBJECTIVE_CONTRIBUTIONS_H__
