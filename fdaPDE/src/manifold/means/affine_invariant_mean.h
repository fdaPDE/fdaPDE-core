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

#ifndef __FDAPDE_MANIFOLD_AFFINE_INVARIANT_MEAN_H__
#define __FDAPDE_MANIFOLD_AFFINE_INVARIANT_MEAN_H__

#include "../header_check.h"

namespace fdapde::manifold {

/// @brief computes the weighted mean with explicit convergence diagnostics
template <typename Scalar_, int Order_, Usage Uses_, typename Samples, typename Point_>
WeightedKarcherMeanResult<typename AffineInvariantSPDGeometry<Scalar_, Order_, Uses_, Point_>::Point>
weighted_karcher_mean(
  const AffineInvariantSPDGeometry<Scalar_, Order_, Uses_, Point_>& geometry, const Samples& samples,
  std::span<const double> weights,
  const typename AffineInvariantSPDGeometry<Scalar_, Order_, Uses_, Point_>::Point& initial,
  const WeightedKarcherMeanOptions& options = {},
  internals::KarcherWorkspace<AffineInvariantSPDGeometry<Scalar_, Order_, Uses_, Point_>>* retained = nullptr) {
    using Geometry = AffineInvariantSPDGeometry<Scalar_, Order_, Uses_, Point_>;
    internals::KarcherWorkspace<Geometry> workspace;
    auto result = weighted_karcher_mean<Geometry>(geometry, samples, weights, initial, options, &workspace);
    internals::polish_karcher_mean(geometry, samples, options, result, workspace);
    if (retained) *retained = std::move(workspace);
    result.uniqueness = BarycenterUniqueness::globally_unique;
    return result;
}

/// @brief computes the weighted mean with explicit convergence diagnostics
template <typename Scalar_, int Order_, Usage Uses_, typename Samples, typename Point_>
WeightedKarcherMeanResult<typename AffineInvariantSPDGeometry<Scalar_, Order_, Uses_, Point_>::Point>
weighted_karcher_mean(
  const AffineInvariantSPDGeometry<Scalar_, Order_, Uses_, Point_>& geometry, const Samples& samples,
  std::span<const double> weights, const WeightedKarcherMeanOptions& options = {},
  internals::KarcherWorkspace<AffineInvariantSPDGeometry<Scalar_, Order_, Uses_, Point_>>* retained = nullptr) {
    auto log_geometry = [&]() {
        if constexpr (Order_ == fdapde::Dynamic) {
            return LogEuclideanSPDGeometry<Scalar_, Order_>(geometry.order());
        } else {
            return LogEuclideanSPDGeometry<Scalar_, Order_>();
        }
    }();
    /// @details use the exact log-Euclidean mean formula as a deterministic, sample-symmetric positive-definite
    /// initializer
    const auto initial = weighted_karcher_mean(log_geometry, samples, weights);
    return weighted_karcher_mean(
      geometry, samples, weights,
      typename AffineInvariantSPDGeometry<Scalar_, Order_, Uses_, Point_>::Point(initial.point), options, retained);
}

}   // namespace fdapde::manifold

#endif   // __FDAPDE_MANIFOLD_AFFINE_INVARIANT_MEAN_H__
