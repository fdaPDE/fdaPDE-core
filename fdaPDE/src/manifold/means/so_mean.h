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

#ifndef __FDAPDE_MANIFOLD_SO_MEAN_H__
#define __FDAPDE_MANIFOLD_SO_MEAN_H__

#include "../header_check.h"

namespace fdapde::manifold {

/// @brief solves a local rotation mean and reuses the local stationarity refinement when necessary
template <typename S, int N, RotationUsage Uses, typename Samples>
WeightedKarcherMeanResult<typename SOGeometry<S, N, Uses>::Point> weighted_karcher_mean(
  const SOGeometry<S, N, Uses>& geometry, const Samples& samples, std::span<const double> weights,
  const typename SOGeometry<S, N, Uses>::Point& initial, const WeightedKarcherMeanOptions& options = {},
  internals::KarcherWorkspace<SOGeometry<S, N, Uses>>* retained = nullptr) {
    using Geometry = SOGeometry<S, N, Uses>;
    internals::KarcherWorkspace<Geometry> workspace;
    auto result = weighted_karcher_mean<Geometry>(geometry, samples, weights, initial, options, &workspace);
    internals::polish_karcher_mean(geometry, samples, options, result, workspace);
    if (retained) *retained = std::move(workspace);
    return result;
}

}   // namespace fdapde::manifold

#endif   // __FDAPDE_MANIFOLD_SO_MEAN_H__
