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

#ifndef __FDAPDE_MANIFOLD_STEEPEST_DESCENT_H__
#define __FDAPDE_MANIFOLD_STEEPEST_DESCENT_H__

#include "header_check.h"

namespace fdapde {
namespace manifold {

struct SteepestDescentOptions {
    std::size_t max_iterations = 500;
    double gradient_tolerance = 1e-6;
    ArmijoOptions line_search;
};

enum class SteepestDescentStopReason {
    gradient_tolerance,
    max_iterations,
    line_search_failed,
    non_finite_cost,
    non_finite_gradient
};

template <typename Point> struct SteepestDescentResult {
    Point point;
    double cost = std::numeric_limits<double>::quiet_NaN();
    double gradient_norm = std::numeric_limits<double>::quiet_NaN();
    std::size_t iterations = 0;
    std::size_t cost_evaluations = 0;
    std::size_t gradient_evaluations = 0;
    std::size_t rejected_trials = 0;
    SteepestDescentStopReason stop_reason = SteepestDescentStopReason::max_iterations;
    ArmijoStatus line_search_status = ArmijoStatus::not_run;

    bool converged() const { return stop_reason == SteepestDescentStopReason::gradient_tolerance; }
};

class RiemannianSteepestDescent {
    SteepestDescentOptions options_;
    ArmijoBacktracking line_search_;

    static void validate(const SteepestDescentOptions& options) {
        if (options.max_iterations == 0) throw std::invalid_argument("SteepestDescent max_iterations must be positive");
        if (!std::isfinite(options.gradient_tolerance) || options.gradient_tolerance < 0)
            throw std::invalid_argument("SteepestDescent gradient_tolerance must be finite and non-negative");
    }
   public:
    explicit RiemannianSteepestDescent(SteepestDescentOptions options = {}) :
        options_(options), line_search_(options.line_search) {
        validate(options_);
    }

    const SteepestDescentOptions& options() const& { return options_; }
    const SteepestDescentOptions& options() const&& = delete;

    template <typename Problem, FirstOrderGeometry Geometry>
        requires FirstOrderProblem<Problem, Geometry>
    SteepestDescentResult<point_t<Geometry>>
    optimize(Problem& problem, const Geometry& geometry, const point_t<Geometry>& initial_point) const {
        EvaluationContext<tangent_t<Geometry>, workspace_t<Problem>> context;
        SteepestDescentResult<point_t<Geometry>> result {initial_point};

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
            result.stop_reason = SteepestDescentStopReason::non_finite_cost;
            return result;
        }
        if (!context.current().gradient()) {
            evaluate_gradient(problem, geometry, result.point, context.current());
            ++result.gradient_evaluations;
        }
        result.gradient_norm = geometry.norm(result.point, *context.current().gradient());
        if (!std::isfinite(result.gradient_norm)) {
            result.stop_reason = SteepestDescentStopReason::non_finite_gradient;
            return result;
        }

        while (result.iterations < options_.max_iterations) {
            if (result.gradient_norm <= options_.gradient_tolerance) {
                result.stop_reason = SteepestDescentStopReason::gradient_tolerance;
                return result;
            }
            const tangent_t<Geometry>& gradient = *context.current().gradient();
            tangent_t<Geometry> direction =
              geometry.linear_combination(result.point, -1, gradient, 0, geometry.zero_tangent(result.point));
            double directional_derivative = geometry.inner_product(result.point, gradient, direction);
            auto line_result = line_search_.search(
              problem, geometry, result.point, direction, result.cost, directional_derivative, context);
            result.cost_evaluations += line_result.trials;
            result.rejected_trials += line_result.rejected_trials;
            result.line_search_status = line_result.status;
            if (!line_result.accepted()) {
                result.stop_reason = SteepestDescentStopReason::line_search_failed;
                return result;
            }

            result.point = std::move(*line_result.point);
            result.cost = line_result.cost;
            context.promote_trial();
            ++result.iterations;
            evaluate_gradient(problem, geometry, result.point, context.current());
            ++result.gradient_evaluations;
            result.gradient_norm = geometry.norm(result.point, *context.current().gradient());
            if (!std::isfinite(result.gradient_norm)) {
                result.stop_reason = SteepestDescentStopReason::non_finite_gradient;
                return result;
            }
        }
        result.stop_reason = result.gradient_norm <= options_.gradient_tolerance ?
                               SteepestDescentStopReason::gradient_tolerance :
                               SteepestDescentStopReason::max_iterations;
        return result;
    }
};

}   // namespace manifold
}   // namespace fdapde

#endif   // __FDAPDE_MANIFOLD_STEEPEST_DESCENT_H__
