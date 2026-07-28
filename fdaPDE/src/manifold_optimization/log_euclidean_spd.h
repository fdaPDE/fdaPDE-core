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

#include "header_check.h"
#include "spd_geometry_common.h"

namespace fdapde {
namespace manifold {

template <typename Scalar_, int Order_> class LogEuclideanSPDGeometry {
    fdapde_static_assert(
      std::is_floating_point_v<Scalar_> && !std::is_const_v<Scalar_> && !std::is_volatile_v<Scalar_>,
      SPD_GEOMETRIES_REQUIRE_AN_UNQUALIFIED_FLOATING_POINT_SCALAR);
    fdapde_static_assert(Order_ == fdapde::Dynamic || Order_ > 0, INVALID_SPD_GEOMETRY_ORDER);
    fdapde_static_assert(
      Order_ == fdapde::Dynamic || std::int64_t(Order_) * std::int64_t(Order_) <= std::numeric_limits<int>::max(),
      SPD_GEOMETRY_DENSE_WORKSPACE_SIZE_EXCEEDS_SUPPORTED_RANGE);
   public:
    using Scalar = Scalar_;
    using Point = fdapde::linalg::SPDMatrix<Scalar, Order_, Order_>;
    using Tangent = fdapde::linalg::SymmetricMatrix<Scalar, Order_, Order_>;

    LogEuclideanSPDGeometry()
        requires(Order_ != fdapde::Dynamic)
    = default;

    explicit LogEuclideanSPDGeometry(int order)
        requires(Order_ == fdapde::Dynamic)
        : order_(order) {
        internals::validate_spd_geometry_order(order_);
    }

    int order() const { return order_; }
    std::size_t dimension() const { return internals::spd_geometry_dimension(order_); }

    double inner_product(const Point& point, const Tangent& u, const Tangent& v) const {
        check_point_(point);
        check_tangent_(u);
        check_tangent_(v);
        const auto chart_u = fdapde::linalg::matrix_log_frechet(point, u);
        const auto chart_v = fdapde::linalg::matrix_log_frechet(point, v);
        return static_cast<double>(internals::frobenius_inner<Scalar>(chart_u, chart_v, order_));
    }

    double norm(const Point& point, const Tangent& tangent) const {
        check_point_(point);
        check_tangent_(tangent);
        return internals::frobenius_norm(fdapde::linalg::matrix_log_frechet(point, tangent), order_);
    }

    Tangent project(const Point& point, const Tangent& ambient) const {
        check_point_(point);
        check_tangent_(ambient);
        return internals::copy_symmetric<Scalar, Order_>(ambient, order_);
    }

    Tangent zero_tangent(const Point& point) const {
        check_point_(point);
        auto result = internals::make_symmetric<Scalar, Order_>(order_);
        for (int i = 0; i < order_; ++i) {
            for (int j = 0; j <= i; ++j) { result(i, j) = Scalar(0); }
        }
        return result;
    }

    Tangent
    linear_combination(const Point& point, double alpha, const Tangent& u, double beta, const Tangent& v) const {
        check_point_(point);
        check_tangent_(u);
        check_tangent_(v);
        return internals::combine_symmetric<Scalar, Order_>(
          u, static_cast<Scalar>(alpha), v, static_cast<Scalar>(beta), order_);
    }

    Point retract(const Point& point, const Tangent& tangent, double step) const {
        return exponential(point, tangent, step);
    }

    Point exponential(const Point& point, const Tangent& tangent, double step = 1.0) const {
        check_point_(point);
        check_tangent_(tangent);
        const auto chart = fdapde::linalg::matrix_log(point);
        const auto chart_tangent = fdapde::linalg::matrix_log_frechet(point, tangent);
        return fdapde::linalg::matrix_exp(
          internals::combine_symmetric<Scalar, Order_>(
            chart, Scalar(1), chart_tangent, static_cast<Scalar>(step), order_));
    }

    Tangent logarithm(const Point& from, const Point& to) const {
        check_point_(from);
        check_point_(to);
        const auto from_chart = fdapde::linalg::matrix_log(from);
        const auto to_chart = fdapde::linalg::matrix_log(to);
        const auto chart_difference =
          internals::combine_symmetric<Scalar, Order_>(to_chart, Scalar(1), from_chart, Scalar(-1), order_);
        return fdapde::linalg::matrix_exp_frechet(from_chart, chart_difference);
    }

    double distance(const Point& from, const Point& to) const {
        check_point_(from);
        check_point_(to);
        const auto from_chart = fdapde::linalg::matrix_log(from);
        const auto to_chart = fdapde::linalg::matrix_log(to);
        const auto difference =
          internals::combine_symmetric<Scalar, Order_>(to_chart, Scalar(1), from_chart, Scalar(-1), order_);
        return internals::frobenius_norm(difference, order_);
    }

    Tangent transport(const Point& from, const Point& to, const Tangent& tangent) const {
        check_point_(from);
        check_point_(to);
        check_tangent_(tangent);
        const auto chart_tangent = fdapde::linalg::matrix_log_frechet(from, tangent);
        return fdapde::linalg::matrix_exp_frechet(fdapde::linalg::matrix_log(to), chart_tangent);
    }

    Tangent euclidean_to_riemannian_gradient(const Point& point, const Tangent& euclidean_gradient) const {
        check_point_(point);
        check_tangent_(euclidean_gradient);
        const auto chart = fdapde::linalg::matrix_log(point);
        return fdapde::linalg::matrix_exp_frechet(chart, fdapde::linalg::matrix_exp_frechet(chart, euclidean_gradient));
    }
   private:
    void check_point_(const Point& point) const { internals::check_spd_geometry_shape(point, order_); }
    void check_tangent_(const Tangent& tangent) const { internals::check_spd_geometry_shape(tangent, order_); }

    int order_ = Order_ == fdapde::Dynamic ? 0 : Order_;
};

template <typename Scalar_, int Order_>
WeightedKarcherMeanResult<typename LogEuclideanSPDGeometry<Scalar_, Order_>::Point> weighted_karcher_mean(
  const LogEuclideanSPDGeometry<Scalar_, Order_>& geometry,
  std::span<const typename LogEuclideanSPDGeometry<Scalar_, Order_>::Point> samples, std::span<const double> weights) {
    using Geometry = LogEuclideanSPDGeometry<Scalar_, Order_>;
    using Point = typename Geometry::Point;
    using Tangent = typename Geometry::Tangent;
    using AccumulationScalar = std::common_type_t<Scalar_, double>;

    if (samples.empty()) throw std::invalid_argument("Weighted Karcher mean requires at least one sample");
    if (samples.size() != weights.size())
        throw std::invalid_argument("Weighted Karcher mean sample and weight counts must match");

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
        sample_logs[sample_index].emplace(fdapde::linalg::matrix_log(samples[sample_index]));

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
    // The stored normalized weights need not sum to exactly one after rounding.
    // Rescaling their chart sum keeps the returned point stationary for that represented objective.
    for (int i = 0; i < geometry.order(); ++i) {
        for (int j = 0; j <= i; ++j) {
            mean_log(i, j) = static_cast<Scalar_>(weighted_log_sum(i, j) / normalized_total);
        }
    }
    Point point = fdapde::linalg::matrix_exp(mean_log);
    const auto point_log = fdapde::linalg::matrix_log(point);
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
