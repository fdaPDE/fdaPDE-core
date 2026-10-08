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

#ifndef __FDAPDE_GFE_P1_GEODESIC_LINEARIZATION_H__
#define __FDAPDE_GFE_P1_GEODESIC_LINEARIZATION_H__

#include "header_check.h"

namespace fdapde {
namespace gfe {

/// @brief retains a mean and its differential data, borrowing immutable batches or owning span snapshots
template <typename Geometry, typename Nodes> class P1GeodesicLinearization;

/// @brief configures the mean and implicit differential solve independently
struct P1GeodesicLinearizationOptions {
    manifold::WeightedKarcherMeanOptions mean;
    manifold::PositiveDefiniteCGOptions linear_solve;
};

/// @brief reports a differential candidate certified only after its linear solve converges
template <typename Derivative> struct P1DerivativeResult {
    /// @details the differential candidate. On iterative paths it is derived from the
    /// @details last linear-solve iterate and is certified only when converged() is true
    Derivative derivative;
    double residual_norm = std::numeric_limits<double>::quiet_NaN();
    std::size_t iterations = 0;
    manifold::PositiveDefiniteCGStopReason stop_reason = manifold::PositiveDefiniteCGStopReason::max_iterations;

    /// @brief reports whether the stored stopping certificate indicates convergence
    bool converged() const { return stop_reason == manifold::PositiveDefiniteCGStopReason::residual_tolerance; }
};

/// @brief retains the residual and stop reason of one implicit differential solve
struct P1LinearSolveStatus {
    double residual_norm = std::numeric_limits<double>::quiet_NaN();
    std::size_t iterations = 0;
    manifold::PositiveDefiniteCGStopReason stop_reason = manifold::PositiveDefiniteCGStopReason::max_iterations;

    /// @brief reports whether the stored stopping certificate indicates convergence
    bool converged() const { return stop_reason == manifold::PositiveDefiniteCGStopReason::residual_tolerance; }
};

/// @brief reports the mixed candidate and all three dependent linear-solve certificates
template <typename Derivative> struct P1MixedDerivativeResult {
    /// @details the derivative candidate is assembled from the three last solver
    /// @details iterates and is certified only when converged() is true. Each method
    /// @details documents the dependency order represented by solve_statuses
    Derivative derivative;
    std::array<P1LinearSolveStatus, 3> solve_statuses;

    /// @brief reports whether the stored stopping certificate indicates convergence
    bool converged() const {
        for (const P1LinearSolveStatus& status : solve_statuses) {
            if (!status.converged()) return false;
        }
        return true;
    }
};

}   // namespace gfe
}   // namespace fdapde

#include "p1_geodesic_linearization_flat.h"
#include "p1_geodesic_linearization_intrinsic.h"

namespace fdapde {
namespace gfe {

/// @brief builds a derivative snapshot borrowing an immutable batch or retaining its selection binding
template <typename Geometry, typename Nodes, typename... Args>
    requires requires { std::remove_cvref_t<Nodes>::NestAsRef; } &&
             (std::is_lvalue_reference_v<Nodes &&> || std::remove_cvref_t<Nodes>::NestAsRef == 0)
auto p1_geodesic_linearization(
  const Geometry& geometry, Nodes&& nodes, std::span<const double> weights, Args&&... args) {
    using Stored = std::conditional_t<
      std::remove_cvref_t<Nodes>::NestAsRef == 0, std::remove_cvref_t<Nodes>, const std::remove_cvref_t<Nodes>&>;
    return P1GeodesicLinearization<Geometry, Stored>(
      geometry, std::forward<Nodes>(nodes), weights, std::forward<Args>(args)...);
}

/// @brief adapts contiguous legacy nodal values to an owning native batch snapshot
template <typename Geometry, typename Point, typename... Args>
    requires(SPDLike<Point> || RotationLike<Point>)
auto p1_geodesic_linearization(
  const Geometry& geometry, std::span<const Point> nodes, std::span<const double> weights, Args&&... args) {
    using Batch = MatrixBatch<Point>;
    return P1GeodesicLinearization<Geometry, Batch>(geometry, Batch(nodes), weights, std::forward<Args>(args)...);
}

}   // namespace gfe
}   // namespace fdapde

#endif   // __FDAPDE_GFE_P1_GEODESIC_LINEARIZATION_H__
