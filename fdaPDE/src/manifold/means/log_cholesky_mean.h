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

#ifndef __FDAPDE_MANIFOLD_LOG_CHOLESKY_MEAN_H__
#define __FDAPDE_MANIFOLD_LOG_CHOLESKY_MEAN_H__

#include "../header_check.h"

namespace fdapde::manifold {

/// @brief computes the globally unique closed-form barycenter and flat stationarity diagnostics
template <typename Scalar, int Order, Usage Uses, typename Samples, typename Point_>
WeightedKarcherMeanResult<typename LogCholeskySPDGeometry<Scalar, Order, Uses, Point_>::Point> weighted_karcher_mean(
  const LogCholeskySPDGeometry<Scalar, Order, Uses, Point_>& geometry, const Samples& samples,
  std::span<const double> weights) {
    using Geometry = LogCholeskySPDGeometry<Scalar, Order, Uses, Point_>;
    using Tangent = typename Geometry::Tangent;
    using Accumulation = std::common_type_t<Scalar, double>;
    fdapde_strong_assert(
      samples.size() != 0 && samples.size() == weights.size(), std::invalid_argument,
      "log-Cholesky mean requires matching nonempty samples and weights");
    auto normalized = internals::normalize_karcher_weights(weights);
    auto sum = internals::make_symmetric<Accumulation, Order>(geometry.order());
    auto correction = internals::make_symmetric<Accumulation, Order>(geometry.order());
    for (int i = 0; i < geometry.order(); ++i)
        for (int j = 0; j <= i; ++j) {
            sum(i, j) = 0;
            correction(i, j) = 0;
        }
    std::vector<std::optional<Tangent>> charts(samples.size());
    long double total = 0;
    for (std::size_t k = 0; k < samples.size(); ++k) {
        if (normalized[k] == 0) continue;
        charts[k].emplace(geometry.chart(samples[k]));
        total += normalized[k];
        for (int i = 0; i < geometry.order(); ++i)
            for (int j = 0; j <= i; ++j) {
                const Accumulation corrected = normalized[k] * (*charts[k])(i, j) - correction(i, j);
                const Accumulation next = sum(i, j) + corrected;
                correction(i, j) = (next - sum(i, j)) - corrected;
                sum(i, j) = next;
            }
    }
    auto mean_chart = internals::make_symmetric<Scalar, Order>(geometry.order());
    for (int i = 0; i < geometry.order(); ++i)
        for (int j = 0; j <= i; ++j) mean_chart(i, j) = static_cast<Scalar>(sum(i, j) / total);
    auto point = geometry.from_chart(mean_chart);
    const auto represented_chart = geometry.chart(point);
    auto residual = internals::make_symmetric<Accumulation, Order>(geometry.order());
    for (int i = 0; i < geometry.order(); ++i)
        for (int j = 0; j <= i; ++j) {
            residual(i, j) = 0;
            correction(i, j) = 0;
        }
    double cost = 0;
    for (std::size_t k = 0; k < samples.size(); ++k) {
        if (normalized[k] == 0) continue;
        double distance = 0;
        for (int i = 0; i < geometry.order(); ++i)
            for (int j = 0; j <= i; ++j) {
                const Accumulation difference = (*charts[k])(i, j) - represented_chart(i, j);
                const Accumulation corrected = normalized[k] * difference - correction(i, j);
                const Accumulation next = static_cast<Accumulation>(residual(i, j)) + corrected;
                correction(i, j) = (next - static_cast<Accumulation>(residual(i, j))) - corrected;
                residual(i, j) = next;
                distance = std::hypot(distance, static_cast<double>(difference));
                if (i != j) distance = std::hypot(distance, static_cast<double>(difference));
            }
        const double scaled = std::sqrt(normalized[k]) * distance;
        cost = std::fma(0.5 * scaled, scaled, cost);
    }
    const double stationarity = internals::frobenius_norm(residual, geometry.order());
    return {
      std::move(point),
      std::move(normalized),
      cost,
      stationarity,
      0,
      1,
      std::isfinite(cost) ? 1u : 0u,
      0,
      std::isfinite(cost) ? BarycenterStopReason::closed_form : BarycenterStopReason::non_finite_cost,
      BarycenterUniqueness::globally_unique,
      ArmijoStatus::not_run};
}

}   // namespace fdapde::manifold

#endif   // __FDAPDE_MANIFOLD_LOG_CHOLESKY_MEAN_H__
