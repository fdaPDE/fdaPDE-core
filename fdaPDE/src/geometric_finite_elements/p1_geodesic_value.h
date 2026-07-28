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

#ifndef __FDAPDE_GFE_P1_GEODESIC_VALUE_H__
#define __FDAPDE_GFE_P1_GEODESIC_VALUE_H__

#include "header_check.h"

namespace fdapde {
namespace gfe {

inline constexpr double p1_weight_sum_tolerance(std::size_t node_count) {
    return 64 * static_cast<double>(node_count) * std::numeric_limits<double>::epsilon();
}

template <typename Point> struct P1ValueResult {
    Point value;
    std::vector<double> normalized_weights;
    double stationarity_norm = std::numeric_limits<double>::quiet_NaN();
    manifold::BarycenterStopReason stop_reason = manifold::BarycenterStopReason::max_iterations;
    manifold::BarycenterUniqueness uniqueness = manifold::BarycenterUniqueness::not_certified;

    bool converged() const {
        return stop_reason == manifold::BarycenterStopReason::closed_form ||
               stop_reason == manifold::BarycenterStopReason::stationarity_tolerance;
    }
};

namespace internals {

inline std::optional<std::size_t> validate_p1_data(std::size_t node_count, std::span<const double> weights) {
    if (node_count == 0) throw std::invalid_argument("P1 geodesic evaluation requires at least one nodal value");
    if (node_count != weights.size())
        throw std::invalid_argument("P1 geodesic nodal-value and barycentric-weight counts must match");

    double sum = 0;
    double correction = 0;
    std::optional<std::size_t> one_hot_index;
    std::size_t positive_count = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        const double weight = weights[i];
        if (!std::isfinite(weight) || weight < 0)
            throw std::invalid_argument("P1 geodesic barycentric weights must be finite and non-negative");

        if (weight > 0) {
            ++positive_count;
            one_hot_index = i;
        }

        const double corrected = weight - correction;
        const double next = sum + corrected;
        correction = (next - sum) - corrected;
        sum = next;
    }
    if (!std::isfinite(sum) || std::abs(sum - 1) > p1_weight_sum_tolerance(weights.size()))
        throw std::invalid_argument("P1 geodesic barycentric weights must sum to one");

    return positive_count == 1 && one_hot_index ? one_hot_index : std::nullopt;
}

template <typename Point> P1ValueResult<Point> p1_value_result(manifold::WeightedKarcherMeanResult<Point> result) {
    return {
      std::move(result.point), std::move(result.normalized_weights), result.stationarity_norm, result.stop_reason,
      result.uniqueness};
}

template <typename Geometry> inline constexpr bool is_log_euclidean_spd_geometry = false;
template <typename Scalar_, int Order_>
inline constexpr bool is_log_euclidean_spd_geometry<manifold::LogEuclideanSPDGeometry<Scalar_, Order_>> = true;

template <manifold::GeodesicGeometry Geometry>
P1ValueResult<manifold::point_t<Geometry>> p1_vertex_result(
  const Geometry& geometry, std::span<const manifold::point_t<Geometry>> nodal_values, std::size_t vertex_index) {
    using Point = manifold::point_t<Geometry>;
    Point value = nodal_values[vertex_index];
    std::vector<double> weights(nodal_values.size(), 0);
    weights[vertex_index] = 1;

    const double distance = geometry.distance(value, value);
    const double cost = 0.5 * distance * distance;
    if (!std::isfinite(cost)) {
        return {
          std::move(value), std::move(weights), std::numeric_limits<double>::quiet_NaN(),
          manifold::BarycenterStopReason::non_finite_cost, manifold::BarycenterUniqueness::not_certified};
    }

    // A single-support barycenter has identically zero first-order residual.
    return {
      std::move(value), std::move(weights), 0, manifold::BarycenterStopReason::closed_form,
      manifold::BarycenterUniqueness::globally_unique};
}

}   // namespace internals

template <manifold::GeodesicGeometry Geometry>
    requires(!internals::is_log_euclidean_spd_geometry<Geometry>)
P1ValueResult<manifold::point_t<Geometry>> p1_geodesic_value(
  const Geometry& geometry, std::span<const manifold::point_t<Geometry>> nodal_values,
  std::span<const double> barycentric_weights, const manifold::point_t<Geometry>& initial,
  const manifold::WeightedKarcherMeanOptions& options = {}) {
    const auto vertex_index = internals::validate_p1_data(nodal_values.size(), barycentric_weights);
    // A vertex does not use the initial point or iterative options; only the selected nodal value is validated.
    if (vertex_index) return internals::p1_vertex_result(geometry, nodal_values, *vertex_index);
    return internals::p1_value_result(
      manifold::weighted_karcher_mean(geometry, nodal_values, barycentric_weights, initial, options));
}

template <typename Scalar_, int Order_>
P1ValueResult<typename manifold::LogEuclideanSPDGeometry<Scalar_, Order_>::Point> p1_geodesic_value(
  const manifold::LogEuclideanSPDGeometry<Scalar_, Order_>& geometry,
  std::span<const typename manifold::LogEuclideanSPDGeometry<Scalar_, Order_>::Point> nodal_values,
  std::span<const double> barycentric_weights) {
    const auto vertex_index = internals::validate_p1_data(nodal_values.size(), barycentric_weights);
    if (vertex_index) return internals::p1_vertex_result(geometry, nodal_values, *vertex_index);
    return internals::p1_value_result(manifold::weighted_karcher_mean(geometry, nodal_values, barycentric_weights));
}

}   // namespace gfe
}   // namespace fdapde

#endif   // __FDAPDE_GFE_P1_GEODESIC_VALUE_H__
