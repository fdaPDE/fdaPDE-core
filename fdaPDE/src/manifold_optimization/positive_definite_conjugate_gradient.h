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

#ifndef __FDAPDE_MANIFOLD_POSITIVE_DEFINITE_CONJUGATE_GRADIENT_H__
#define __FDAPDE_MANIFOLD_POSITIVE_DEFINITE_CONJUGATE_GRADIENT_H__

#include "header_check.h"

namespace fdapde {
namespace manifold {

struct PositiveDefiniteCGOptions {
    std::size_t max_iterations = 100;
    double residual_tolerance = 1e-10;
};

enum class PositiveDefiniteCGStopReason {
    residual_tolerance,
    non_positive_curvature,
    numerical_breakdown,
    max_iterations,
    non_finite
};

template <typename Tangent> struct PositiveDefiniteCGResult {
    Tangent solution;
    double residual_norm = std::numeric_limits<double>::quiet_NaN();
    std::size_t iterations = 0;
    PositiveDefiniteCGStopReason stop_reason = PositiveDefiniteCGStopReason::max_iterations;

    bool converged() const { return stop_reason == PositiveDefiniteCGStopReason::residual_tolerance; }
};

// Unpreconditioned conjugate gradients for H solution = rhs in one fixed
// tangent space. The callable must be self-adjoint and positive definite in
// the geometry metric at point. Unrepresentable squared metric products are
// reported explicitly; the solver does not rescale through geometry-agnostic
// tangent coefficients.
class PositiveDefiniteConjugateGradient {
    PositiveDefiniteCGOptions options_;

    static void validate(const PositiveDefiniteCGOptions& options) {
        if (options.max_iterations == 0)
            throw std::invalid_argument("PositiveDefiniteCG max_iterations must be positive");
        if (
          !std::isfinite(options.residual_tolerance) || options.residual_tolerance < 0 ||
          options.residual_tolerance >= 1)
            throw std::invalid_argument("PositiveDefiniteCG residual_tolerance must be in [0, 1)");
    }

    static bool within_tolerance(double residual_norm, double right_hand_side_norm, double tolerance) {
        return residual_norm <= right_hand_side_norm && residual_norm / right_hand_side_norm <= tolerance;
    }
   public:
    explicit PositiveDefiniteConjugateGradient(PositiveDefiniteCGOptions options = {}) : options_(options) {
        validate(options_);
    }

    const PositiveDefiniteCGOptions& options() const& { return options_; }
    const PositiveDefiniteCGOptions& options() const&& = delete;

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
