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

#ifndef __FDAPDE_MANIFOLD_TRUNCATED_CONJUGATE_GRADIENT_H__
#define __FDAPDE_MANIFOLD_TRUNCATED_CONJUGATE_GRADIENT_H__

#include "header_check.h"

namespace fdapde {
namespace manifold {

/// @brief controls iteration and residual limits for the trust-region subproblem
struct TruncatedCGOptions {
    std::size_t max_iterations = 100;
    double residual_tolerance = 1e-10;
};

/// @brief identifies convergence, boundary truncation or subproblem failure
enum class TruncatedCGStopReason {
    residual_tolerance,
    boundary,
    negative_curvature,
    max_iterations,
    non_finite,
    not_run
};

/// @brief returns the truncated step, its Hessian action and solver diagnostics
template <typename Tangent> struct TruncatedCGResult {
    Tangent step;
    Tangent hessian_step;
    std::size_t iterations = 0;
    std::size_t hessian_evaluations = 0;
    TruncatedCGStopReason stop_reason = TruncatedCGStopReason::not_run;

    /// @brief reports truncation at the trust-region boundary
    bool hit_boundary() const {
        return stop_reason == TruncatedCGStopReason::boundary ||
               stop_reason == TruncatedCGStopReason::negative_curvature;
    }
};

// steihaug--Toint truncated conjugate gradients for trust-region subproblems;
// see Conn, Gould, and Toint, Trust Region Methods (2000)
/// @brief solves tangent-space trust-region subproblems by Steihaug conjugate gradients
class SteihaugTruncatedCG {
    TruncatedCGOptions options_;

    /// @brief checks that the radius admits a finite normal square
    static bool valid_radius(double radius) { return radius > 0 && std::isnormal(radius * radius); }

    /// @brief validates the public solver options
    static void validate(const TruncatedCGOptions& options) {
        fdapde_strong_assert(
          options.max_iterations != 0, std::invalid_argument, "TruncatedCG max_iterations must be positive");
        fdapde_strong_assert(
          std::isfinite(options.residual_tolerance) && options.residual_tolerance >= 0 &&
            options.residual_tolerance < 1,
          std::invalid_argument, "TruncatedCG residual_tolerance must be in [0, 1)");
    }

    /// @brief computes the forward intersection of a tangent ray with the trust-region boundary
    template <FirstOrderGeometry Geometry>
    static double boundary_step(
      const Geometry& geometry, const point_t<Geometry>& point, const tangent_t<Geometry>& step,
      const tangent_t<Geometry>& direction, double radius) {
        double direction_norm_sq = geometry.inner_product(point, direction, direction);
        double step_direction = geometry.inner_product(point, step, direction);
        double step_norm_sq = geometry.inner_product(point, step, step);
        if (
          !std::isfinite(direction_norm_sq) || direction_norm_sq <= 0 || !std::isfinite(step_direction) ||
          !std::isfinite(step_norm_sq) || step_norm_sq < 0)
            return std::numeric_limits<double>::quiet_NaN();
        double direction_norm = std::sqrt(direction_norm_sq);
        double normalized_step_norm_sq = step_norm_sq / (radius * radius);
        double normalized_projection = (step_direction / direction_norm) / radius;
        if (!std::isfinite(normalized_step_norm_sq) || !std::isfinite(normalized_projection))
            return std::numeric_limits<double>::quiet_NaN();
        double normalized_remaining_sq = std::max(0.0, 1 - normalized_step_norm_sq);
        double radical = std::hypot(normalized_projection, std::sqrt(normalized_remaining_sq));
        double normalized_step = normalized_projection > 0 ?
                                   normalized_remaining_sq / (radical + normalized_projection) :
                                   radical - normalized_projection;
        return normalized_step * (radius / direction_norm);
    }
   public:
    /// @brief constructs the subproblem solver with validated options
    explicit SteihaugTruncatedCG(TruncatedCGOptions options = {}) : options_(options) { validate(options_); }

    /// @brief exposes solver options only while the owning solver is alive
    const TruncatedCGOptions& options() const& { return options_; }
    /// @brief forbids option references from a temporary solver
    const TruncatedCGOptions& options() const&& = delete;

    /// @brief computes a truncated tangent step using the current evaluation workspace
    template <typename Problem, FirstOrderGeometry Geometry>
        requires RiemannianHessianProblem<Problem, Geometry>
    TruncatedCGResult<tangent_t<Geometry>> solve(
      Problem& problem, const Geometry& geometry, const point_t<Geometry>& point, const tangent_t<Geometry>& gradient,
      double radius, workspace_t<Problem>& workspace) const {
        fdapde_strong_assert(
          valid_radius(radius), std::invalid_argument,
          "TruncatedCG radius must be positive with a finite normal square");
        tangent_t<Geometry> step = geometry.zero_tangent(point);
        tangent_t<Geometry> hessian_step = geometry.zero_tangent(point);
        TruncatedCGResult<tangent_t<Geometry>> result {step, hessian_step};
        // residual is the gradient of the quadratic model at the current step
        tangent_t<Geometry> residual = gradient;
        double initial_residual_norm = geometry.norm(point, residual);
        if (!std::isfinite(initial_residual_norm)) {
            result.stop_reason = TruncatedCGStopReason::non_finite;
            return result;
        }
        if (initial_residual_norm == 0) {
            result.stop_reason = TruncatedCGStopReason::residual_tolerance;
            return result;
        }

        double residual_norm_sq = geometry.inner_product(point, residual, residual);
        if (!std::isfinite(residual_norm_sq) || residual_norm_sq <= 0) {
            result.stop_reason = TruncatedCGStopReason::non_finite;
            return result;
        }
        tangent_t<Geometry> direction =
          geometry.linear_combination(point, -1, residual, 0, geometry.zero_tangent(point));
        for (std::size_t iteration = 0; iteration < options_.max_iterations; ++iteration) {
            tangent_t<Geometry> hessian_direction = problem.hessian_vector(point, direction, workspace);
            ++result.hessian_evaluations;
            ++result.iterations;
            double curvature = geometry.inner_product(point, direction, hessian_direction);
            if (!std::isfinite(curvature) || !std::isfinite(residual_norm_sq) || residual_norm_sq <= 0) {
                result.stop_reason = TruncatedCGStopReason::non_finite;
                return result;
            }
            if (curvature <= 0) {
                // follow the search direction to the trust-region boundary
                double tau = boundary_step(geometry, point, step, direction, radius);
                if (!std::isfinite(tau)) {
                    result.stop_reason = TruncatedCGStopReason::non_finite;
                    return result;
                }
                result.step = geometry.linear_combination(point, 1, step, tau, direction);
                result.hessian_step = geometry.linear_combination(point, 1, hessian_step, tau, hessian_direction);
                result.stop_reason = TruncatedCGStopReason::negative_curvature;
                return result;
            }

            double alpha = residual_norm_sq / curvature;
            if (!std::isfinite(alpha)) {
                result.stop_reason = TruncatedCGStopReason::non_finite;
                return result;
            }
            tangent_t<Geometry> candidate = geometry.linear_combination(point, 1, step, alpha, direction);
            double candidate_norm = geometry.norm(point, candidate);
            if (!std::isfinite(candidate_norm)) {
                result.stop_reason = TruncatedCGStopReason::non_finite;
                return result;
            }
            if (candidate_norm >= radius) {
                // truncate the positive-curvature step at the boundary
                double tau = boundary_step(geometry, point, step, direction, radius);
                if (!std::isfinite(tau)) {
                    result.stop_reason = TruncatedCGStopReason::non_finite;
                    return result;
                }
                result.step = geometry.linear_combination(point, 1, step, tau, direction);
                result.hessian_step = geometry.linear_combination(point, 1, hessian_step, tau, hessian_direction);
                result.stop_reason = TruncatedCGStopReason::boundary;
                return result;
            }

            tangent_t<Geometry> candidate_hessian_step =
              geometry.linear_combination(point, 1, hessian_step, alpha, hessian_direction);
            step = std::move(candidate);
            hessian_step = std::move(candidate_hessian_step);
            tangent_t<Geometry> next_residual =
              geometry.linear_combination(point, 1, residual, alpha, hessian_direction);
            double next_residual_norm = geometry.norm(point, next_residual);
            if (!std::isfinite(next_residual_norm)) {
                result.step = std::move(step);
                result.hessian_step = std::move(hessian_step);
                result.stop_reason = TruncatedCGStopReason::non_finite;
                return result;
            }
            if (next_residual_norm <= options_.residual_tolerance * initial_residual_norm) {
                result.step = std::move(step);
                result.hessian_step = std::move(hessian_step);
                result.stop_reason = TruncatedCGStopReason::residual_tolerance;
                return result;
            }

            double next_residual_norm_sq = geometry.inner_product(point, next_residual, next_residual);
            if (!std::isfinite(next_residual_norm_sq) || next_residual_norm_sq < 0) {
                result.step = std::move(step);
                result.hessian_step = std::move(hessian_step);
                result.stop_reason = TruncatedCGStopReason::non_finite;
                return result;
            }
            double beta = next_residual_norm_sq / residual_norm_sq;
            if (!std::isfinite(beta)) {
                result.step = std::move(step);
                result.hessian_step = std::move(hessian_step);
                result.stop_reason = TruncatedCGStopReason::non_finite;
                return result;
            }
            direction = geometry.linear_combination(point, -1, next_residual, beta, direction);
            residual = std::move(next_residual);
            residual_norm_sq = next_residual_norm_sq;
        }
        result.step = std::move(step);
        result.hessian_step = std::move(hessian_step);
        result.stop_reason = TruncatedCGStopReason::max_iterations;
        return result;
    }
};

}   // namespace manifold
}   // namespace fdapde

#endif   // __FDAPDE_MANIFOLD_TRUNCATED_CONJUGATE_GRADIENT_H__
