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

#ifndef __FDAPDE_MANIFOLD_TRUST_REGION_H__
#define __FDAPDE_MANIFOLD_TRUST_REGION_H__

#include "header_check.h"

namespace fdapde {
namespace manifold {

struct TrustRegionOptions {
    std::size_t max_iterations = 100;
    double gradient_tolerance = 1e-6;
    double initial_radius = 1;
    double maximum_radius = 1000;
    double minimum_radius = 1e-12;
    double acceptance_threshold = 0.1;
    double shrink_threshold = 0.25;
    double expansion_threshold = 0.75;
    double shrink_factor = 0.25;
    double expansion_factor = 2;
    TruncatedCGOptions subproblem;
    double ratio_regularization = 1e3;
};

enum class TrustRegionStopReason {
    gradient_tolerance,
    max_iterations,
    radius_too_small,
    model_failure,
    non_finite_cost,
    non_finite_gradient
};

template <typename Point> struct TrustRegionResult {
    Point point;
    double cost = std::numeric_limits<double>::quiet_NaN();
    double gradient_norm = std::numeric_limits<double>::quiet_NaN();
    double radius = std::numeric_limits<double>::quiet_NaN();
    double last_ratio = std::numeric_limits<double>::quiet_NaN();
    std::size_t iterations = 0;
    std::size_t accepted_steps = 0;
    std::size_t rejected_steps = 0;
    std::size_t cost_evaluations = 0;
    std::size_t gradient_evaluations = 0;
    std::size_t hessian_evaluations = 0;
    TrustRegionStopReason stop_reason = TrustRegionStopReason::max_iterations;
    TruncatedCGStopReason subproblem_stop_reason = TruncatedCGStopReason::not_run;

    bool converged() const { return stop_reason == TrustRegionStopReason::gradient_tolerance; }
};

// Riemannian trust regions with a Steihaug subproblem solve; see
// Absil, Baker, and Gallivan, Trust-Region Methods on Riemannian Manifolds (2007).
class RiemannianTrustRegion {
    TrustRegionOptions options_;
    SteihaugTruncatedCG subproblem_;

    static bool valid_radius(double radius) { return radius > 0 && std::isnormal(radius * radius); }

    static void validate(const TrustRegionOptions& options) {
        if (options.max_iterations == 0) throw std::invalid_argument("TrustRegion max_iterations must be positive");
        if (!std::isfinite(options.gradient_tolerance) || options.gradient_tolerance < 0)
            throw std::invalid_argument("TrustRegion gradient_tolerance must be finite and non-negative");
        if (
          !valid_radius(options.initial_radius) || !valid_radius(options.maximum_radius) ||
          options.maximum_radius < options.initial_radius || !valid_radius(options.minimum_radius) ||
          options.minimum_radius > options.initial_radius)
            throw std::invalid_argument("TrustRegion radii must have finite normal squares and be ordered");
        if (
          !std::isfinite(options.acceptance_threshold) || !std::isfinite(options.shrink_threshold) ||
          !std::isfinite(options.expansion_threshold) || options.acceptance_threshold < 0 ||
          options.acceptance_threshold >= options.shrink_threshold ||
          options.shrink_threshold >= options.expansion_threshold || options.expansion_threshold >= 1)
            throw std::invalid_argument("TrustRegion thresholds must satisfy 0 <= acceptance < shrink < expansion < 1");
        if (
          !std::isfinite(options.shrink_factor) || options.shrink_factor <= 0 || options.shrink_factor >= 1 ||
          !std::isfinite(options.expansion_factor) || options.expansion_factor <= 1)
            throw std::invalid_argument("TrustRegion radius factors must satisfy shrink in (0, 1), expansion > 1");
        if (!std::isfinite(options.ratio_regularization) || options.ratio_regularization < 0)
            throw std::invalid_argument("TrustRegion ratio_regularization must be finite and non-negative");
    }
   public:
    explicit RiemannianTrustRegion(TrustRegionOptions options = {}) :
        options_(options), subproblem_(options.subproblem) {
        validate(options_);
    }

    const TrustRegionOptions& options() const& { return options_; }
    const TrustRegionOptions& options() const&& = delete;

    template <typename Problem, FirstOrderGeometry Geometry>
        requires RiemannianHessianProblem<Problem, Geometry>
    TrustRegionResult<point_t<Geometry>>
    optimize(Problem& problem, const Geometry& geometry, const point_t<Geometry>& initial_point) const {
        EvaluationContext<tangent_t<Geometry>, workspace_t<Problem>> context;
        TrustRegionResult<point_t<Geometry>> result {initial_point};
        result.radius = options_.initial_radius;

        if constexpr (CombinedCostGradientProblem<Problem, Geometry>) {
            evaluate_cost_gradient(problem, geometry, result.point, context.current());
            ++result.cost_evaluations;
            ++result.gradient_evaluations;
        } else {
            evaluate_cost(problem, geometry, result.point, context.current());
            ++result.cost_evaluations;
        }
        result.cost = *context.current().cost();
        if (!std::isfinite(result.cost)) {
            result.stop_reason = TrustRegionStopReason::non_finite_cost;
            return result;
        }
        if (!context.current().gradient()) {
            evaluate_gradient(problem, geometry, result.point, context.current());
            ++result.gradient_evaluations;
        }
        result.gradient_norm = geometry.norm(result.point, *context.current().gradient());
        if (!std::isfinite(result.gradient_norm)) {
            result.stop_reason = TrustRegionStopReason::non_finite_gradient;
            return result;
        }

        while (result.iterations < options_.max_iterations) {
            if (result.gradient_norm <= options_.gradient_tolerance) {
                result.stop_reason = TrustRegionStopReason::gradient_tolerance;
                return result;
            }
            const tangent_t<Geometry>& gradient = *context.current().gradient();
            auto subproblem_result = subproblem_.solve(
              problem, geometry, result.point, gradient, result.radius, context.current().workspace());
            result.hessian_evaluations += subproblem_result.hessian_evaluations;
            result.subproblem_stop_reason = subproblem_result.stop_reason;
            if (subproblem_result.stop_reason == TruncatedCGStopReason::non_finite) {
                result.stop_reason = TrustRegionStopReason::model_failure;
                return result;
            }

            // predicted reduction of the quadratic model at the proposed step
            double predicted_reduction =
              -(geometry.inner_product(result.point, gradient, subproblem_result.step) +
                0.5 * geometry.inner_product(result.point, subproblem_result.step, subproblem_result.hessian_step));
            if (!std::isfinite(predicted_reduction) || predicted_reduction <= 0) {
                result.stop_reason = TrustRegionStopReason::model_failure;
                return result;
            }

            point_t<Geometry> candidate = geometry.retract(result.point, subproblem_result.step, 1);
            context.reset_trial();
            double candidate_cost = evaluate_cost(problem, geometry, candidate, context.trial());
            ++result.cost_evaluations;
            // actual-to-predicted reduction controls acceptance and radius updates
            double actual_reduction = result.cost - candidate_cost;
            double ratio_shift = std::max(1.0, std::abs(result.cost)) * std::numeric_limits<double>::epsilon() *
                                 options_.ratio_regularization;
            result.last_ratio = (actual_reduction + ratio_shift) / (predicted_reduction + ratio_shift);
            if (!std::isfinite(candidate_cost) || !std::isfinite(actual_reduction) || !std::isfinite(result.last_ratio))
                result.last_ratio = -std::numeric_limits<double>::infinity();
            bool cost_increased = std::isfinite(candidate_cost) && candidate_cost > result.cost;

            if (cost_increased || result.last_ratio < options_.shrink_threshold) {
                result.radius *= options_.shrink_factor;
            } else if (result.last_ratio > options_.expansion_threshold && subproblem_result.hit_boundary()) {
                result.radius = std::min(options_.maximum_radius, options_.expansion_factor * result.radius);
            }

            ++result.iterations;
            if (
              std::isfinite(candidate_cost) && !cost_increased && result.last_ratio >= options_.acceptance_threshold) {
                result.point = std::move(candidate);
                result.cost = candidate_cost;
                context.promote_trial();
                ++result.accepted_steps;
                evaluate_gradient(problem, geometry, result.point, context.current());
                ++result.gradient_evaluations;
                result.gradient_norm = geometry.norm(result.point, *context.current().gradient());
                if (!std::isfinite(result.gradient_norm)) {
                    result.stop_reason = TrustRegionStopReason::non_finite_gradient;
                    return result;
                }
            } else {
                // rejected trials never replace the current point or its workspace
                context.reset_trial();
                ++result.rejected_steps;
            }
            if (result.gradient_norm <= options_.gradient_tolerance) {
                result.stop_reason = TrustRegionStopReason::gradient_tolerance;
                return result;
            }
            if (result.radius < options_.minimum_radius) {
                result.stop_reason = TrustRegionStopReason::radius_too_small;
                return result;
            }
        }
        result.stop_reason = TrustRegionStopReason::max_iterations;
        return result;
    }
};

}   // namespace manifold
}   // namespace fdapde

#endif   // __FDAPDE_MANIFOLD_TRUST_REGION_H__
