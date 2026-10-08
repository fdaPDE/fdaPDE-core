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

#ifndef __FDAPDE_MANIFOLD_LOG_EUCLIDEAN_SPD_H__
#define __FDAPDE_MANIFOLD_LOG_EUCLIDEAN_SPD_H__

#include "../../geometry_expr.h"
#include "../../header_check.h"
#include "../spd_geometry_common.h"

namespace fdapde {
namespace manifold {

/// @brief defines the log-Euclidean metric on checked SPD owners with ambient symmetric tangents
template <
  typename Scalar_, int Order_, Usage Uses_ = Usage::None,
  typename Point_ = fdapde::SPDMatrix<Scalar_, Order_, Cache::Policy<internals::log_euclidean_cache_flags(Uses_)>>>
class LogEuclideanSPDGeometry {
    fdapde_static_assert((static_cast<unsigned>(Uses_) & ~31u) == 0, SPD_GEOMETRY_USAGE_CONTAINS_UNKNOWN_FLAGS);
    fdapde_static_assert(
      std::is_floating_point_v<Scalar_> && !std::is_const_v<Scalar_> && !std::is_volatile_v<Scalar_>,
      SPD_GEOMETRIES_REQUIRE_AN_UNQUALIFIED_FLOATING_POINT_SCALAR);
    fdapde_static_assert(Order_ == fdapde::Dynamic || Order_ > 0, INVALID_SPD_GEOMETRY_ORDER);
    fdapde_static_assert(
      Order_ == fdapde::Dynamic || std::int64_t(Order_) * std::int64_t(Order_) <= std::numeric_limits<int>::max(),
      SPD_GEOMETRY_DENSE_WORKSPACE_SIZE_EXCEEDS_SUPPORTED_RANGE);
   public:
    using Scalar = Scalar_;
    using Point = Point_;
    using CachePolicy = typename Point::CachePolicy;
    using Tangent = fdapde::SymmetricMatrix<Scalar, Order_>;
    fdapde_static_assert(
      (std::same_as<Point, fdapde::SPDMatrix<Scalar, Order_, CachePolicy, Point::StorageOrder>>),
      LOG_EUCLIDEAN_GEOMETRY_REQUIRES_A_NATIVE_SPD_OWNER_WITH_MATCHING_SCALAR_AND_SHAPE);

    /// @brief constructs the fixed-order geometry using its positive compile-time matrix order
    LogEuclideanSPDGeometry()
        requires(Order_ != fdapde::Dynamic)
    = default;

    /// @brief constructs a dynamic geometry after checking positive order and the supported dense workspace bound
    explicit LogEuclideanSPDGeometry(int order)
        requires(Order_ == fdapde::Dynamic)
        : order_(order) {
        internals::validate_spd_geometry_order(order_);
    }

    /// @brief prepares spatial P1 evaluation on a copied simplex or borrowed mesh with immutable batch data
    /// @details include geometric_finite_elements.h for the definition; simplex data follow local vertex order and mesh
    /// data follow global node ids
    template <typename Element, typename Nodes>
        requires gfe::P1InterpolationBinding<Element, Nodes>
    auto interpolant(Element&& element, Nodes&& nodes) const;
    /// @brief prepares the same interpolant with explicit mean and linear-solve tolerances
    template <typename Element, typename Nodes>
        requires gfe::P1InterpolationBinding<Element, Nodes>
    auto interpolant(Element&& element, Nodes&& nodes, const gfe::P1GeodesicLinearizationOptions& options) const;

    /// @brief returns the matrix order
    int order() const { return order_; }
    /// @brief returns the number of independent tangent coefficients
    std::size_t dimension() const { return internals::spd_geometry_dimension(order_); }

    /// @brief pairs ambient tangents in the metric at point
    template <SPDLike PointPoint>
    double inner_product(const PointPoint& point, const Tangent& u, const Tangent& v) const {
        check_point_(point);
        check_tangent_(u);
        check_tangent_(v);
        const auto chart_u = fdapde::matrix_log_frechet(point, u);
        const auto chart_v = fdapde::matrix_log_frechet(point, v);
        return static_cast<double>(internals::frobenius_inner(chart_u, chart_v, order_));
    }

    /// @brief returns the metric norm of an ambient tangent
    template <SPDLike PointPoint> double norm(const PointPoint& point, const Tangent& tangent) const {
        check_point_(point);
        check_tangent_(tangent);
        return internals::frobenius_norm(fdapde::matrix_log_frechet(point, tangent), order_);
    }

    /// @brief copies an already symmetric ambient tangent into independent storage
    template <SPDLike PointPoint> Tangent project(const PointPoint& point, const Tangent& ambient) const {
        check_point_(point);
        check_tangent_(ambient);
        return ambient;
    }

    /// @brief returns an owning zero tangent of the point order
    template <SPDLike PointPoint> Tangent zero_tangent(const PointPoint& point) const {
        check_point_(point);
        auto result = internals::make_symmetric<Scalar, Order_>(order_);
        for (int i = 0; i < order_; ++i) {
            for (int j = 0; j <= i; ++j) { result(i, j) = Scalar(0); }
        }
        return result;
    }

    /// @brief combines ambient tangents with finite coefficients
    template <SPDLike PointPoint>
    Tangent
    linear_combination(const PointPoint& point, double alpha, const Tangent& u, double beta, const Tangent& v) const {
        check_point_(point);
        check_tangent_(u);
        check_tangent_(v);
        return internals::combine_symmetric<Scalar, Order_>(
          u, internals::geometry_coefficient<Scalar>(alpha), v, internals::geometry_coefficient<Scalar>(beta), order_);
    }

    /// @brief uses the exact exponential as a retraction
    template <SPDLike PointPoint> Point retract(const PointPoint& point, const Tangent& tangent, double step) const {
        return exponential(point, tangent, step);
    }

    /// @brief follows the geodesic with the supplied initial ambient tangent and finite step
    template <SPDLike PointPoint>
    Point exponential(const PointPoint& point, const Tangent& tangent, double step = 1.0) const {
        check_point_(point);
        check_tangent_(tangent);
        const auto chart = fdapde::matrix_log(point);
        const auto chart_tangent = fdapde::matrix_log_frechet(point, tangent);
        return fdapde::matrix_exp<CachePolicy>(internals::combine_symmetric<Scalar, Order_>(
          chart, Scalar(1), chart_tangent, internals::geometry_coefficient<Scalar>(step), order_));
    }

    /// @brief returns the initial ambient tangent of the geodesic from source to target
    template <SPDLike PointFrom, SPDLike PointTo> Tangent logarithm(const PointFrom& from, const PointTo& to) const {
        check_point_(from);
        check_point_(to);
        const auto from_chart = fdapde::matrix_log(from);
        const auto to_chart = fdapde::matrix_log(to);
        const auto chart_difference =
          internals::combine_symmetric<Scalar, Order_>(to_chart, Scalar(1), from_chart, Scalar(-1), order_);
        return internals::spd_exp_log_frechet(from, chart_difference);
    }

    /// @brief returns the geodesic distance between checked points
    template <SPDLike PointFrom, SPDLike PointTo> double distance(const PointFrom& from, const PointTo& to) const {
        check_point_(from);
        check_point_(to);
        const auto from_chart = fdapde::matrix_log(from);
        const auto to_chart = fdapde::matrix_log(to);
        const auto difference =
          internals::combine_symmetric<Scalar, Order_>(to_chart, Scalar(1), from_chart, Scalar(-1), order_);
        return internals::frobenius_norm(difference, order_);
    }

    /// @brief parallel-transports an ambient tangent along the source-to-target geodesic
    template <SPDLike PointFrom, SPDLike PointTo>
    Tangent transport(const PointFrom& from, const PointTo& to, const Tangent& tangent) const {
        check_point_(from);
        check_point_(to);
        check_tangent_(tangent);
        const auto chart_tangent = fdapde::matrix_log_frechet(from, tangent);
        return internals::spd_exp_log_frechet(to, chart_tangent);
    }

    /// @brief converts a symmetric Frobenius gradient to its Riemannian metric dual
    template <SPDLike PointPoint>
    Tangent euclidean_to_riemannian_gradient(const PointPoint& point, const Tangent& euclidean_gradient) const {
        check_point_(point);
        check_tangent_(euclidean_gradient);
        if constexpr (fdapde::internals::spd_cache_has_v<typename PointPoint::CachePolicy, Cache::Spectral>) {
            const auto first = internals::spd_exp_log_frechet(point, euclidean_gradient);
            return internals::spd_exp_log_frechet(point, first);
        } else {
            const auto chart = fdapde::matrix_log(point);
            return fdapde::matrix_exp_frechet(chart, fdapde::matrix_exp_frechet(chart, euclidean_gradient));
        }
    }

    /// @brief converts a Frobenius Hessian action using the flat logarithm chart and its differential
    /// @details euclidean_hessian is the ambient derivative of euclidean_gradient along direction
    template <SPDLike PointPoint>
    Tangent euclidean_to_riemannian_hessian(
      const PointPoint& point, const Tangent& euclidean_gradient, const Tangent& euclidean_hessian,
      const Tangent& direction) const {
        check_point_(point);
        check_tangent_(euclidean_gradient);
        check_tangent_(euclidean_hessian);
        check_tangent_(direction);
        const auto chart = fdapde::matrix_log(point);
        const auto chart_direction = fdapde::matrix_log_frechet(point, direction);
        const auto chart_curvature = fdapde::matrix_exp_second_frechet(chart, chart_direction, euclidean_gradient);
        const auto chart_hessian = internals::spd_exp_log_frechet(point, euclidean_hessian);
        const Tangent chart_sum(chart_curvature + chart_hessian);
        return internals::spd_exp_log_frechet(point, chart_sum);
    }

    /// @brief prepares an owning log-Euclidean geodesic snapshot from independently cached endpoints
    /// @details defers exp(log(from) + t * (log(to) - log(from))); SPD destinations certify evaluated coefficients
    template <SPDLike PointFrom, SPDLike PointTo> auto geodesic(const PointFrom& from, const PointTo& to) const {
        check_point_(from);
        check_point_(to);
        const Tangent first(fdapde::matrix_log(from));
        const Tangent last(fdapde::matrix_log(to));
        Tangent difference(last - first);
        return fdapde::internals::spd_geodesic<Scalar, Order_, false>(first, std::move(difference));
    }
    /// @brief returns uniformly spaced geodesic samples including both endpoints with the selected output cache policy
    /// @details defaults to the geometry point policy; prepares one curve and requires count >= 2
    /// sample i uses t = i / (count - 1); execution defaults to sequential and parallel calls join before returning
    template <
      typename OutputPolicy = CachePolicy, SPDLike From, SPDLike To,
      fdapde::internals::BatchExecutionPolicy ExecutionPolicy = execution_seq_t>
    auto interpolate(const From& from, const To& to, int count, ExecutionPolicy policy = {}) const {
        return internals::sample_spd_geodesic<OutputPolicy>(*this, from, to, count, policy);
    }

    /// @brief defers exp(sum_i weights[i] * log(points[i])) without normalizing finite real weights
    /// @details borrows persistent operands and retains temporary selection nodes; owning temporaries are rejected
    template <typename Points, typename Weights>
        requires(
          !fdapde::internals::is_owning_rvalue_expression_v<Points &&> &&
          !fdapde::internals::is_owning_rvalue_expression_v<Weights &&> &&
          SPDLike<typename std::remove_cvref_t<Points>::MatrixType>)
    auto weighted_mean(Points&& points, Weights&& weights) const& {
        return fdapde::internals::log_euclidean_mean_expr<
          LogEuclideanSPDGeometry, fdapde::internals::batch_nested_t<Points&&>,
          fdapde::internals::batch_nested_t<Weights&&>>(
          *this, std::forward<Points>(points), std::forward<Weights>(weights));
    }
    /// @brief prevents a deferred mean from borrowing a temporary geometry
    template <typename Points, typename Weights> void weighted_mean(Points&&, Weights&&) const&& = delete;
   private:
    /// @brief checks the point order and finite packed coefficients against this geometry
    template <SPDLike PointPoint> void check_point_(const PointPoint& point) const {
        internals::check_spd_geometry_shape(point, order_);
    }
    /// @brief checks the tangent order and finite packed coefficients against this geometry
    void check_tangent_(const Tangent& tangent) const { internals::check_spd_geometry_shape(tangent, order_); }

    int order_ = Order_ == fdapde::Dynamic ? 0 : Order_;
};

namespace internals {

/// @brief selects a single-point log-Euclidean geometry while preserving the exact native SPD owner
template <typename Point> struct log_euclidean_geometry_type {
    using type = LogEuclideanSPDGeometry<typename Point::Scalar, Point::Rows, Usage::None, Point>;
};

}   // namespace internals

/// @brief supplies the log-Euclidean metric for a native SPD owner with its exact cache policy
template <typename Point> using LogEuclideanGeometry = typename internals::log_euclidean_geometry_type<Point>::type;

/// @brief computes the weighted mean with explicit convergence diagnostics
template <typename Scalar_, int Order_, Usage Uses_, typename Samples, typename Point_>
WeightedKarcherMeanResult<typename LogEuclideanSPDGeometry<Scalar_, Order_, Uses_, Point_>::Point>
weighted_karcher_mean(
  const LogEuclideanSPDGeometry<Scalar_, Order_, Uses_, Point_>& geometry, const Samples& samples,
  std::span<const double> weights) {
    using Geometry = LogEuclideanSPDGeometry<Scalar_, Order_, Uses_, Point_>;
    using Point = typename Geometry::Point;
    using Tangent = typename Geometry::Tangent;
    using AccumulationScalar = std::common_type_t<Scalar_, double>;

    fdapde_strong_assert(
      samples.size() != 0, std::invalid_argument, "Weighted Karcher mean requires at least one sample");
    fdapde_strong_assert(
      samples.size() == weights.size(), std::invalid_argument,
      "Weighted Karcher mean sample and weight counts must match");
    auto normalized_weights = internals::normalize_karcher_weights(weights);
    auto weighted_log_sum = internals::make_symmetric<AccumulationScalar, Order_>(geometry.order());
    auto weighted_log_correction = internals::make_symmetric<AccumulationScalar, Order_>(geometry.order());
    for (int i = 0; i < geometry.order(); ++i) {
        for (int j = 0; j <= i; ++j) {
            weighted_log_sum(i, j) = 0;
            weighted_log_correction(i, j) = 0;
        }
    }

    std::vector<std::optional<Tangent>> sample_logs(samples.size());
    double normalized_total = 0;
    double normalized_total_correction = 0;
    for (std::size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
        const double weight = normalized_weights[sample_index];
        if (weight == 0) continue;
        internals::check_spd_geometry_shape(samples[sample_index], geometry.order());
        sample_logs[sample_index].emplace(fdapde::matrix_log(samples[sample_index]));

        const double corrected_weight = weight - normalized_total_correction;
        const double next_total = normalized_total + corrected_weight;
        normalized_total_correction = (next_total - normalized_total) - corrected_weight;
        normalized_total = next_total;

        for (int i = 0; i < geometry.order(); ++i) {
            for (int j = 0; j <= i; ++j) {
                const AccumulationScalar contribution =
                  static_cast<AccumulationScalar>(weight) *
                  static_cast<AccumulationScalar>((*sample_logs[sample_index])(i, j));
                const AccumulationScalar corrected = contribution - weighted_log_correction(i, j);
                const AccumulationScalar next = weighted_log_sum(i, j) + corrected;
                weighted_log_correction(i, j) = (next - weighted_log_sum(i, j)) - corrected;
                weighted_log_sum(i, j) = next;
            }
        }
    }

    auto mean_log = internals::make_symmetric<Scalar_, Order_>(geometry.order());
    // the stored normalized weights need not sum to exactly one after rounding
    // rescaling their chart sum keeps the returned point stationary for that represented objective
    for (int i = 0; i < geometry.order(); ++i) {
        for (int j = 0; j <= i; ++j) {
            mean_log(i, j) = static_cast<Scalar_>(weighted_log_sum(i, j) / normalized_total);
        }
    }
    Point point(fdapde::matrix_exp<typename Point::CachePolicy>(mean_log));
    const auto point_log = fdapde::matrix_log(point);
    auto residual = internals::make_symmetric<AccumulationScalar, Order_>(geometry.order());
    auto residual_correction = internals::make_symmetric<AccumulationScalar, Order_>(geometry.order());
    for (int i = 0; i < geometry.order(); ++i) {
        for (int j = 0; j <= i; ++j) {
            residual(i, j) = 0;
            residual_correction(i, j) = 0;
        }
    }

    double cost = 0;
    for (std::size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
        const double weight = normalized_weights[sample_index];
        if (weight == 0) continue;

        double distance = 0;
        for (int i = 0; i < geometry.order(); ++i) {
            for (int j = 0; j <= i; ++j) {
                const AccumulationScalar difference =
                  static_cast<AccumulationScalar>((*sample_logs[sample_index])(i, j)) -
                  static_cast<AccumulationScalar>(point_log(i, j));
                distance = std::hypot(distance, static_cast<double>(difference));
                if (i != j) distance = std::hypot(distance, static_cast<double>(difference));

                const AccumulationScalar contribution = static_cast<AccumulationScalar>(weight) * difference;
                const AccumulationScalar corrected = contribution - residual_correction(i, j);
                const AccumulationScalar next = residual(i, j) + corrected;
                residual_correction(i, j) = (next - residual(i, j)) - corrected;
                residual(i, j) = next;
            }
        }
        const double scaled_distance = std::sqrt(weight) * distance;
        cost = std::fma(0.5 * scaled_distance, scaled_distance, cost);
    }

    if (!std::isfinite(cost)) {
        return {
          std::move(point),
          std::move(normalized_weights),
          cost,
          std::numeric_limits<double>::quiet_NaN(),
          0,
          1,
          0,
          0,
          BarycenterStopReason::non_finite_cost,
          BarycenterUniqueness::globally_unique,
          ArmijoStatus::not_run};
    }

    const double stationarity_norm = internals::frobenius_norm(residual, geometry.order());
    return {
      std::move(point),
      std::move(normalized_weights),
      cost,
      stationarity_norm,
      0,
      1,
      1,
      0,
      std::isfinite(stationarity_norm) ? BarycenterStopReason::closed_form : BarycenterStopReason::non_finite_gradient,
      BarycenterUniqueness::globally_unique,
      ArmijoStatus::not_run};
}

}   // namespace manifold
}   // namespace fdapde

#endif   // __FDAPDE_MANIFOLD_LOG_EUCLIDEAN_SPD_H__
