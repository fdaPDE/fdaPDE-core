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

#ifndef __FDAPDE_MANIFOLD_LOG_EUCLIDEAN_MEAN_H__
#define __FDAPDE_MANIFOLD_LOG_EUCLIDEAN_MEAN_H__

#include "../header_check.h"

namespace fdapde::manifold {

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

}   // namespace fdapde::manifold

#endif   // __FDAPDE_MANIFOLD_LOG_EUCLIDEAN_MEAN_H__
