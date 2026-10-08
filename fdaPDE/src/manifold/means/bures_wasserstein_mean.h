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

#ifndef __FDAPDE_MANIFOLD_BURES_WASSERSTEIN_MEAN_H__
#define __FDAPDE_MANIFOLD_BURES_WASSERSTEIN_MEAN_H__

#include "../header_check.h"

namespace fdapde::manifold {

/// @brief computes a BW barycenter with explicit diagnostics and retained implicit-derivative frames
template <typename Scalar, int Order, Usage Uses, typename Samples, typename Point_>
WeightedKarcherMeanResult<typename BuresWassersteinSPDGeometry<Scalar, Order, Uses, Point_>::Point>
weighted_karcher_mean(
  const BuresWassersteinSPDGeometry<Scalar, Order, Uses, Point_>& geometry, const Samples& samples,
  std::span<const double> weights,
  const typename BuresWassersteinSPDGeometry<Scalar, Order, Uses, Point_>::Point& initial,
  const WeightedKarcherMeanOptions& options = {},
  internals::KarcherWorkspace<BuresWassersteinSPDGeometry<Scalar, Order, Uses, Point_>>* retained = nullptr) {
    using Geometry = BuresWassersteinSPDGeometry<Scalar, Order, Uses, Point_>;
    fdapde_strong_assert(
      options.solver.line_search.initial_step <= 1, std::invalid_argument,
      "BW barycenter initial step must not exceed one to preserve the positive horizontal lift");
    internals::KarcherWorkspace<Geometry> workspace;
    auto result = weighted_karcher_mean<Geometry>(geometry, samples, weights, initial, options, &workspace);
    internals::polish_karcher_mean(geometry, samples, options, result, workspace);
    if (retained) *retained = std::move(workspace);
    result.uniqueness = BarycenterUniqueness::globally_unique;
    return result;
}

/// @brief starts BW barycenter descent at the deterministic weighted arithmetic SPD mean
template <typename Scalar, int Order, Usage Uses, typename Samples, typename Point_>
WeightedKarcherMeanResult<typename BuresWassersteinSPDGeometry<Scalar, Order, Uses, Point_>::Point>
weighted_karcher_mean(
  const BuresWassersteinSPDGeometry<Scalar, Order, Uses, Point_>& geometry, const Samples& samples,
  std::span<const double> weights, const WeightedKarcherMeanOptions& options = {},
  internals::KarcherWorkspace<BuresWassersteinSPDGeometry<Scalar, Order, Uses, Point_>>* retained = nullptr) {
    using Geometry = BuresWassersteinSPDGeometry<Scalar, Order, Uses, Point_>;
    fdapde_strong_assert(
      samples.size() != 0 && samples.size() == weights.size(), std::invalid_argument,
      "BW barycenter requires matching nonempty sample and weight counts");
    const auto normalized = internals::normalize_karcher_weights(weights);
    for (std::size_t k = 0; k < samples.size(); ++k)
        if (normalized[k] != 0) geometry.zero_tangent(samples[k]);
    auto initial = internals::make_symmetric<Scalar, Order>(geometry.order());
    long double total = 0;
    for (const double weight : normalized) total += weight;
    for (int i = 0; i < geometry.order(); ++i)
        for (int j = 0; j <= i; ++j) {
            long double value = 0;
            for (std::size_t k = 0; k < samples.size(); ++k)
                if (normalized[k] != 0) value += static_cast<long double>(normalized[k]) * samples[k](i, j);
            initial(i, j) = static_cast<Scalar>(value / total);
        }
    return weighted_karcher_mean(geometry, samples, weights, typename Geometry::Point(initial), options, retained);
}

}   // namespace fdapde::manifold

#endif   // __FDAPDE_MANIFOLD_BURES_WASSERSTEIN_MEAN_H__
