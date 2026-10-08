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

/// @brief bounds representational error in P1 barycentric weight sums
inline constexpr double p1_weight_sum_tolerance(std::size_t node_count) {
    return 64 * static_cast<double>(node_count) * std::numeric_limits<double>::epsilon();
}

/// @brief retains the interpolated candidate and its barycenter convergence certificate
template <typename Point> struct P1ValueResult {
    Point value;
    std::vector<double> normalized_weights;
    double stationarity_norm = std::numeric_limits<double>::quiet_NaN();
    manifold::BarycenterStopReason stop_reason = manifold::BarycenterStopReason::max_iterations;
    manifold::BarycenterUniqueness uniqueness = manifold::BarycenterUniqueness::not_certified;

    std::size_t iterations = 0;

    /// @brief reports whether the stored stopping certificate indicates convergence
    bool converged() const {
        return stop_reason == manifold::BarycenterStopReason::closed_form ||
               stop_reason == manifold::BarycenterStopReason::stationarity_tolerance;
    }
};

namespace internals {

/// @brief validates convex barycentric weights and identifies exact vertex support
inline std::optional<std::size_t> validate_p1_data(std::size_t node_count, std::span<const double> weights) {
    fdapde_strong_assert(
      node_count != 0, std::invalid_argument, "P1 geodesic evaluation requires at least one nodal value");
    fdapde_strong_assert(
      node_count == weights.size(), std::invalid_argument,
      "P1 geodesic nodal-value and barycentric-weight counts must match");
    double sum = 0;
    double correction = 0;
    std::optional<std::size_t> one_hot_index;
    std::size_t positive_count = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) {
        const double weight = weights[i];
        fdapde_strong_assert(
          std::isfinite(weight) && weight >= 0, std::invalid_argument,
          "P1 geodesic barycentric weights must be finite and non-negative");
        if (weight > 0) {
            ++positive_count;
            one_hot_index = i;
        }

        const double corrected = weight - correction;
        const double next = sum + corrected;
        correction = (next - sum) - corrected;
        sum = next;
    }
    fdapde_strong_assert(
      std::isfinite(sum) && std::abs(sum - 1) <= p1_weight_sum_tolerance(weights.size()), std::invalid_argument,
      "P1 geodesic barycentric weights must sum to one");
    return positive_count == 1 && one_hot_index ? one_hot_index : std::nullopt;
}

/// @brief preserves convergence diagnostics when adapting a mean to a P1 value
template <typename Point> P1ValueResult<Point> p1_value_result(manifold::WeightedKarcherMeanResult<Point> result) {
    return {std::move(result.point),  std::move(result.normalized_weights),
            result.stationarity_norm, result.stop_reason,
            result.uniqueness,        result.iterations};
}

template <typename Geometry> inline constexpr bool is_log_euclidean_spd_geometry = false;
template <typename Scalar_, int Order_, Usage Uses_, typename Point_>
inline constexpr bool is_log_euclidean_spd_geometry<manifold::LogEuclideanSPDGeometry<Scalar_, Order_, Uses_, Point_>> =
  true;

/// @brief identifies the SPD geometries with globally flat symmetric coordinates
template <typename Geometry> inline constexpr bool is_flat_spd_geometry = is_log_euclidean_spd_geometry<Geometry>;
template <typename Scalar, int Order, Usage Uses, typename Point_>
inline constexpr bool is_flat_spd_geometry<manifold::LogCholeskySPDGeometry<Scalar, Order, Uses, Point_>> = true;

/// @brief retains the compile-time order of a flat SPD coordinate chart
template <typename Geometry> inline constexpr int flat_spd_order = fdapde::Dynamic;
template <typename Scalar, int Order, Usage Uses, typename Point_>
inline constexpr int flat_spd_order<manifold::LogEuclideanSPDGeometry<Scalar, Order, Uses, Point_>> = Order;
template <typename Scalar, int Order, Usage Uses, typename Point_>
inline constexpr int flat_spd_order<manifold::LogCholeskySPDGeometry<Scalar, Order, Uses, Point_>> = Order;

/// @brief retains the chart representation required by one flat SPD geometry
template <typename Geometry> struct FlatSPDChartFrame {
    using type = typename Geometry::Tangent;
};
/// @brief retains the triangular factor alongside a log-Cholesky chart
template <typename Scalar, int Order, Usage Uses, typename Point_>
struct FlatSPDChartFrame<manifold::LogCholeskySPDGeometry<Scalar, Order, Uses, Point_>> {
    using type = typename manifold::LogCholeskySPDGeometry<Scalar, Order, Uses, Point_>::ChartFrame;
};

/// @brief prepares one reusable coordinate frame while preserving native log-Euclidean owner caches
template <typename Geometry, SPDLike Point>
    requires is_flat_spd_geometry<Geometry>
typename FlatSPDChartFrame<Geometry>::type p1_flat_chart_frame(const Geometry& geometry, const Point& point) {
    if constexpr (is_log_euclidean_spd_geometry<Geometry>)
        return typename Geometry::Tangent(fdapde::matrix_log(point));
    else
        return geometry.chart_frame(point);
}

/// @brief borrows symmetric coordinates from a retained flat geometry frame
template <typename Geometry>
    requires is_flat_spd_geometry<Geometry>
const typename Geometry::Tangent& p1_flat_frame_coordinates(const typename FlatSPDChartFrame<Geometry>::type& frame) {
    if constexpr (is_log_euclidean_spd_geometry<Geometry>)
        return frame;
    else
        return frame.coordinates;
}

/// @brief differentiates a flat chart using a retained triangular frame or the owner's cached log differential
template <typename Geometry, SPDLike Point>
    requires is_flat_spd_geometry<Geometry>
typename Geometry::Tangent p1_flat_frame_chart_jvp(
  const Geometry& geometry, const Point& point, const typename FlatSPDChartFrame<Geometry>::type& frame,
  const typename Geometry::Tangent& direction) {
    if constexpr (is_log_euclidean_spd_geometry<Geometry>)
        return typename Geometry::Tangent(fdapde::matrix_log_frechet(point, direction));
    else
        return geometry.chart_differential(frame, direction);
}

/// @brief maps a flat direction back to an ambient tangent while reusing the prepared factor
template <typename Geometry>
    requires is_flat_spd_geometry<Geometry>
typename Geometry::Tangent p1_flat_frame_inverse_chart_jvp(
  const Geometry& geometry, const typename FlatSPDChartFrame<Geometry>::type& frame,
  const typename Geometry::Tangent& direction) {
    if constexpr (is_log_euclidean_spd_geometry<Geometry>)
        return typename Geometry::Tangent(fdapde::matrix_exp_frechet(frame, direction));
    else
        return geometry.inverse_chart_jvp(frame, direction);
}

/// @brief reads flat coordinates while retaining the matrix owner's geometry cache
template <typename Geometry, SPDLike Point>
    requires is_flat_spd_geometry<Geometry>
typename Geometry::Tangent p1_flat_chart(const Geometry& geometry, const Point& point) {
    if constexpr (is_log_euclidean_spd_geometry<Geometry>)
        return typename Geometry::Tangent(fdapde::matrix_log(point));
    else
        return geometry.chart(point);
}

/// @brief converts a flat chart direction into the metric-dual ambient tangent
template <typename Geometry>
    requires is_flat_spd_geometry<Geometry>
typename Geometry::Tangent p1_flat_inverse_chart_jvp(
  const Geometry& geometry, const typename Geometry::Tangent& chart, const typename Geometry::Tangent& direction) {
    if constexpr (is_log_euclidean_spd_geometry<Geometry>)
        return typename Geometry::Tangent(fdapde::matrix_exp_frechet(chart, direction));
    else
        return geometry.inverse_chart_differential(chart, direction);
}

template <manifold::GeodesicGeometry Geometry, typename Nodes>
P1ValueResult<manifold::point_t<Geometry>>
p1_vertex_result(const Geometry& geometry, const Nodes& nodal_values, std::size_t vertex_index) {
    using Point = manifold::point_t<Geometry>;
    Point value(nodal_values[vertex_index]);
    std::vector<double> weights(nodal_values.size(), 0);
    weights[vertex_index] = 1;

    const double distance = geometry.distance(value, value);
    const double cost = 0.5 * distance * distance;
    if (!std::isfinite(cost)) {
        return {
          std::move(value), std::move(weights), std::numeric_limits<double>::quiet_NaN(),
          manifold::BarycenterStopReason::non_finite_cost, manifold::BarycenterUniqueness::not_certified};
    }

    /// @details a single-support barycenter has identically zero first-order residual
    return {
      std::move(value), std::move(weights), 0, manifold::BarycenterStopReason::closed_form,
      manifold::BarycenterUniqueness::globally_unique};
}

}   // namespace internals

template <manifold::GeodesicGeometry Geometry, typename Nodes>
    requires(!internals::is_flat_spd_geometry<Geometry>)
P1ValueResult<manifold::point_t<Geometry>> p1_geodesic_value(
  const Geometry& geometry, const Nodes& nodal_values, std::span<const double> barycentric_weights,
  const manifold::point_t<Geometry>& initial, const manifold::WeightedKarcherMeanOptions& options = {},
  manifold::internals::KarcherWorkspace<Geometry>* retained = nullptr) {
    const auto vertex_index = internals::validate_p1_data(nodal_values.size(), barycentric_weights);
    /// @details a vertex does not use the initial point or iterative options; only the selected nodal value is
    /// validated
    if (vertex_index) return internals::p1_vertex_result(geometry, nodal_values, *vertex_index);
    return internals::p1_value_result(
      manifold::weighted_karcher_mean(geometry, nodal_values, barycentric_weights, initial, options, retained));
}

/// @brief evaluates a closed-form P1 mean in the geometry's globally flat coordinates
template <typename Scalar_, int Order_, Usage Uses_, typename Nodes, typename Point_>
P1ValueResult<typename manifold::LogEuclideanSPDGeometry<Scalar_, Order_, Uses_, Point_>::Point> p1_geodesic_value(
  const manifold::LogEuclideanSPDGeometry<Scalar_, Order_, Uses_, Point_>& geometry, const Nodes& nodal_values,
  std::span<const double> barycentric_weights) {
    const auto vertex_index = internals::validate_p1_data(nodal_values.size(), barycentric_weights);
    if (vertex_index) return internals::p1_vertex_result(geometry, nodal_values, *vertex_index);
    return internals::p1_value_result(manifold::weighted_karcher_mean(geometry, nodal_values, barycentric_weights));
}

/// @brief evaluates a closed-form P1 mean in the geometry's globally flat coordinates
template <typename Geometry, typename Nodes>
    requires internals::is_flat_spd_geometry<Geometry> && (!internals::is_log_euclidean_spd_geometry<Geometry>)
P1ValueResult<typename Geometry::Point> p1_geodesic_value(
  const Geometry& geometry, const Nodes& nodal_values, std::span<const double> barycentric_weights) {
    const auto vertex_index = internals::validate_p1_data(nodal_values.size(), barycentric_weights);
    if (vertex_index) return internals::p1_vertex_result(geometry, nodal_values, *vertex_index);
    return internals::p1_value_result(manifold::weighted_karcher_mean(geometry, nodal_values, barycentric_weights));
}

template <typename Scalar_, int Order_, Usage Uses_, typename Nodes, typename Point_>
P1ValueResult<typename manifold::AffineInvariantSPDGeometry<Scalar_, Order_, Uses_, Point_>::Point> p1_geodesic_value(
  const manifold::AffineInvariantSPDGeometry<Scalar_, Order_, Uses_, Point_>& geometry, const Nodes& nodal_values,
  std::span<const double> barycentric_weights, const manifold::WeightedKarcherMeanOptions& options = {},
  manifold::internals::KarcherWorkspace<manifold::AffineInvariantSPDGeometry<Scalar_, Order_, Uses_, Point_>>*
    retained = nullptr) {
    const auto vertex_index = internals::validate_p1_data(nodal_values.size(), barycentric_weights);
    if (vertex_index) return internals::p1_vertex_result(geometry, nodal_values, *vertex_index);
    return internals::p1_value_result(
      manifold::weighted_karcher_mean(geometry, nodal_values, barycentric_weights, options, retained));
}

/// @brief evaluates a BW P1 mean while preserving its convergence diagnostics and cached frames
template <typename Scalar, int Order, Usage Uses, typename Nodes, typename Point_>
P1ValueResult<typename manifold::BuresWassersteinSPDGeometry<Scalar, Order, Uses, Point_>::Point> p1_geodesic_value(
  const manifold::BuresWassersteinSPDGeometry<Scalar, Order, Uses, Point_>& geometry, const Nodes& nodal_values,
  std::span<const double> barycentric_weights, const manifold::WeightedKarcherMeanOptions& options = {},
  manifold::internals::KarcherWorkspace<manifold::BuresWassersteinSPDGeometry<Scalar, Order, Uses, Point_>>* retained =
    nullptr) {
    const auto vertex_index = internals::validate_p1_data(nodal_values.size(), barycentric_weights);
    if (vertex_index) return internals::p1_vertex_result(geometry, nodal_values, *vertex_index);
    return internals::p1_value_result(
      manifold::weighted_karcher_mean(geometry, nodal_values, barycentric_weights, options, retained));
}

/// @brief initializes the rotation mean at the first node with maximal barycentric weight
template <typename S, int N, RotationUsage Uses, typename Nodes>
P1ValueResult<typename manifold::SOGeometry<S, N, Uses>::Point> p1_geodesic_value(
  const manifold::SOGeometry<S, N, Uses>& geometry, const Nodes& nodes, std::span<const double> weights,
  const manifold::WeightedKarcherMeanOptions& options = {},
  manifold::internals::KarcherWorkspace<manifold::SOGeometry<S, N, Uses>>* retained = nullptr) {
    const auto vertex = internals::validate_p1_data(nodes.size(), weights);
    if (vertex) return internals::p1_vertex_result(geometry, nodes, *vertex);
    const auto index = std::distance(weights.begin(), std::max_element(weights.begin(), weights.end()));
    const typename manifold::SOGeometry<S, N, Uses>::Point initial(nodes[index]);
    return p1_geodesic_value(geometry, nodes, weights, initial, options, retained);
}

}   // namespace gfe
}   // namespace fdapde

#endif   // __FDAPDE_GFE_P1_GEODESIC_VALUE_H__
