/// @details This file is part of fdaPDE, a C++ library for physics-informed
/// @details spatial and functional data analysis.
//
/// @details This program is free software: you can redistribute it and/or modify
/// @details it under the terms of the GNU General Public License as published by
/// @details the Free Software Foundation, either version 3 of the License, or
/// @details (at your option) any later version.
//
/// @details This program is distributed in the hope that it will be useful,
/// @details but WITHOUT ANY WARRANTY; without even the implied warranty of
/// @details MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
/// @details GNU General Public License for more details.
//
/// @details You should have received a copy of the GNU General Public License
// along with this program.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __FDAPDE_MANIFOLD_POSITIVE_DEFINITE_CONJUGATE_GRADIENT_H__
#define __FDAPDE_MANIFOLD_POSITIVE_DEFINITE_CONJUGATE_GRADIENT_H__

#include "header_check.h"

namespace fdapde {
namespace manifold {

/// @brief bounds tangent CG iterations and the relative residual tolerance
struct PositiveDefiniteCGOptions {
    std::size_t max_iterations = 100;
    double residual_tolerance = 1e-10;
};

/// @brief identifies residual convergence or the reason the tangent solve stopped
enum class PositiveDefiniteCGStopReason {
    residual_tolerance,
    non_positive_curvature,
    numerical_breakdown,
    max_iterations,
    non_finite
};

/// @brief reports the last tangent iterate and its linear-solve certificate
template <typename Tangent> struct PositiveDefiniteCGResult {
    Tangent solution;
    double residual_norm = std::numeric_limits<double>::quiet_NaN();
    std::size_t iterations = 0;
    PositiveDefiniteCGStopReason stop_reason = PositiveDefiniteCGStopReason::max_iterations;

    /// @brief reports whether the stored stopping certificate indicates convergence
    bool converged() const { return stop_reason == PositiveDefiniteCGStopReason::residual_tolerance; }
};

/// @brief solves a positive-definite self-adjoint operator in one fixed tangent metric
/// @details the callable must be self-adjoint and positive definite in the metric at the fixed point
/// unrepresentable squared metric products report numerical failure
class PositiveDefiniteConjugateGradient {
    PositiveDefiniteCGOptions options_;

    /// @brief rejects invalid user-supplied solver tolerances and iteration budgets
    static void validate(const PositiveDefiniteCGOptions& options) {
        fdapde_strong_assert(
          options.max_iterations != 0, std::invalid_argument, "PositiveDefiniteCG max_iterations must be positive");
        fdapde_strong_assert(
          std::isfinite(options.residual_tolerance) && options.residual_tolerance >= 0 &&
            options.residual_tolerance < 1,
          std::invalid_argument, "PositiveDefiniteCG residual_tolerance must be in [0, 1)");
    }

    /// @brief compares the residual norm with a relative tolerance without squaring norms
    static bool within_tolerance(double residual_norm, double right_hand_side_norm, double tolerance) {
        return residual_norm <= right_hand_side_norm && residual_norm / right_hand_side_norm <= tolerance;
    }
   public:
    /// @brief validates and stores the tangent linear-solve budget
    explicit PositiveDefiniteConjugateGradient(PositiveDefiniteCGOptions options = {}) : options_(options) {
        validate(options_);
    }

    /// @brief borrows the validated solver configuration
    const PositiveDefiniteCGOptions& options() const& { return options_; }
    /// @brief prevents borrowing storage from a temporary object
    const PositiveDefiniteCGOptions& options() const&& = delete;

    /// @brief runs tangent conjugate gradients with explicit curvature and numerical failure diagnostics
    template <typename LinearOperator, FirstOrderGeometry Geometry>
        requires requires(LinearOperator& apply, const tangent_t<Geometry>& direction) {
            { apply(direction) } -> std::same_as<tangent_t<Geometry>>;
        }
    PositiveDefiniteCGResult<tangent_t<Geometry>> solve(
      LinearOperator&& apply, const Geometry& geometry, const point_t<Geometry>& point,
      const tangent_t<Geometry>& right_hand_side) const {
        PositiveDefiniteCGResult<tangent_t<Geometry>> result {geometry.zero_tangent(point)};
        tangent_t<Geometry> residual = right_hand_side;
        const double right_hand_side_norm = geometry.norm(point, right_hand_side);
        result.residual_norm = right_hand_side_norm;
        if (!std::isfinite(right_hand_side_norm)) {
            result.stop_reason = PositiveDefiniteCGStopReason::non_finite;
            return result;
        }
        if (right_hand_side_norm == 0) {
            result.stop_reason = PositiveDefiniteCGStopReason::residual_tolerance;
            return result;
        }

        double residual_inner = geometry.inner_product(point, residual, residual);
        if (!std::isfinite(residual_inner)) {
            result.stop_reason = PositiveDefiniteCGStopReason::non_finite;
            return result;
        }
        if (residual_inner <= 0) {
            result.stop_reason = PositiveDefiniteCGStopReason::numerical_breakdown;
            return result;
        }
        tangent_t<Geometry> direction = residual;

        while (result.iterations < options_.max_iterations) {
            tangent_t<Geometry> image = apply(direction);
            ++result.iterations;
            const double curvature = geometry.inner_product(point, direction, image);
            if (!std::isfinite(curvature)) {
                result.stop_reason = PositiveDefiniteCGStopReason::non_finite;
                return result;
            }
            if (curvature <= 0) {
                result.stop_reason = PositiveDefiniteCGStopReason::non_positive_curvature;
                return result;
            }

            const double alpha = residual_inner / curvature;
            if (!std::isfinite(alpha)) {
                result.stop_reason = PositiveDefiniteCGStopReason::non_finite;
                return result;
            }
            if (alpha <= 0) {
                result.stop_reason = PositiveDefiniteCGStopReason::numerical_breakdown;
                return result;
            }

            tangent_t<Geometry> candidate = geometry.linear_combination(point, 1, result.solution, alpha, direction);
            const double candidate_norm = geometry.norm(point, candidate);
            if (!std::isfinite(candidate_norm)) {
                result.stop_reason = PositiveDefiniteCGStopReason::non_finite;
                return result;
            }
            tangent_t<Geometry> next_residual = geometry.linear_combination(point, 1, residual, -alpha, image);
            const double next_residual_norm = geometry.norm(point, next_residual);
            if (!std::isfinite(next_residual_norm)) {
                result.stop_reason = PositiveDefiniteCGStopReason::non_finite;
                return result;
            }

            result.solution = std::move(candidate);
            result.residual_norm = next_residual_norm;
            if (within_tolerance(next_residual_norm, right_hand_side_norm, options_.residual_tolerance)) {
                result.stop_reason = PositiveDefiniteCGStopReason::residual_tolerance;
                return result;
            }
            if (result.iterations == options_.max_iterations) break;

            const double next_residual_inner = geometry.inner_product(point, next_residual, next_residual);
            if (!std::isfinite(next_residual_inner)) {
                result.stop_reason = PositiveDefiniteCGStopReason::non_finite;
                return result;
            }
            if (next_residual_inner <= 0) {
                result.stop_reason = PositiveDefiniteCGStopReason::numerical_breakdown;
                return result;
            }
            const double beta = next_residual_inner / residual_inner;
            if (!std::isfinite(beta)) {
                result.stop_reason = PositiveDefiniteCGStopReason::non_finite;
                return result;
            }
            if (beta <= 0) {
                result.stop_reason = PositiveDefiniteCGStopReason::numerical_breakdown;
                return result;
            }

            tangent_t<Geometry> next_direction = geometry.linear_combination(point, 1, next_residual, beta, direction);
            const double next_direction_norm = geometry.norm(point, next_direction);
            if (!std::isfinite(next_direction_norm)) {
                result.stop_reason = PositiveDefiniteCGStopReason::non_finite;
                return result;
            }
            if (next_direction_norm <= 0) {
                result.stop_reason = PositiveDefiniteCGStopReason::numerical_breakdown;
                return result;
            }
            residual = std::move(next_residual);
            direction = std::move(next_direction);
            residual_inner = next_residual_inner;
        }

        result.stop_reason = PositiveDefiniteCGStopReason::max_iterations;
        return result;
    }
};

}   // namespace manifold
}   // namespace fdapde

#endif   // __FDAPDE_MANIFOLD_POSITIVE_DEFINITE_CONJUGATE_GRADIENT_H__
