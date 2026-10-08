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

#ifndef __FDAPDE_MANIFOLD_ARMIJO_H__
#define __FDAPDE_MANIFOLD_ARMIJO_H__

#include "header_check.h"

namespace fdapde {
namespace manifold {

/// @brief bounds step contraction and the strict sufficient-decrease acceptance test
struct ArmijoOptions {
    double initial_step = 1;
    double contraction = 0.5;
    double sufficient_decrease = 1e-4;
    double minimum_step = 1e-12;
    std::size_t max_trials = 25;
};

/// @brief identifies acceptance or the reason a bounded line search stopped
enum class ArmijoStatus {
    accepted,
    non_descent_direction,
    non_finite_current_cost,
    minimum_step,
    max_trials,
    not_run
};

/// @brief reports an accepted point or an explicit line-search failure
template <typename Point> struct ArmijoResult {
    std::optional<Point> point;
    double cost = std::numeric_limits<double>::quiet_NaN();
    double step = 0;
    std::size_t trials = 0;
    std::size_t rejected_trials = 0;
    ArmijoStatus status = ArmijoStatus::not_run;

    /// @brief reports whether the line search supplied an accepted point
    bool accepted() const { return status == ArmijoStatus::accepted; }
};

/// @brief searches a bounded sequence of retraction steps with strict Armijo decrease
class ArmijoBacktracking {
    ArmijoOptions options_;

    /// @brief rejects invalid user-supplied solver tolerances and iteration budgets
    static void validate(const ArmijoOptions& options) {
        fdapde_strong_assert(
          std::isfinite(options.initial_step) && options.initial_step > 0, std::invalid_argument,
          "Armijo initial_step must be finite and positive");
        fdapde_strong_assert(
          std::isfinite(options.contraction) && options.contraction > 0 && options.contraction < 1,
          std::invalid_argument, "Armijo contraction must be in (0, 1)");
        fdapde_strong_assert(
          std::isfinite(options.sufficient_decrease) && options.sufficient_decrease > 0 &&
            options.sufficient_decrease < 1,
          std::invalid_argument, "Armijo sufficient_decrease must be in (0, 1)");
        fdapde_strong_assert(
          std::isfinite(options.minimum_step) && options.minimum_step > 0 &&
            options.minimum_step <= options.initial_step,
          std::invalid_argument, "Armijo minimum_step must be finite, positive and no larger than initial_step");
        fdapde_strong_assert(options.max_trials > 0, std::invalid_argument, "Armijo max_trials must be positive");
    }
   public:
    /// @brief validates and stores the bounded line-search options
    explicit ArmijoBacktracking(ArmijoOptions options = {}) : options_(options) { validate(options_); }

    /// @brief borrows the validated solver configuration
    const ArmijoOptions& options() const& { return options_; }
    /// @brief prevents borrowing storage from a temporary object
    const ArmijoOptions& options() const&& = delete;

    /// @brief tests retraction steps while binding each trial to a fresh evaluation generation
    template <typename Problem, FirstOrderGeometry Geometry>
        requires FirstOrderProblem<Problem, Geometry>
    ArmijoResult<point_t<Geometry>> search(
      Problem& problem, const Geometry& geometry, const point_t<Geometry>& point, const tangent_t<Geometry>& direction,
      double current_cost, double directional_derivative, evaluation_context_t<Problem, Geometry>& context) const {
        ArmijoResult<point_t<Geometry>> result;
        context.reset_trial();
        if (!std::isfinite(current_cost)) {
            result.status = ArmijoStatus::non_finite_current_cost;
            return result;
        }
        if (!std::isfinite(directional_derivative) || directional_derivative >= 0) {
            result.status = ArmijoStatus::non_descent_direction;
            return result;
        }

        double step = options_.initial_step;
        for (std::size_t trial = 0; trial < options_.max_trials; ++trial) {
            if (step < options_.minimum_step) {
                context.reset_trial();
                result.status = ArmijoStatus::minimum_step;
                return result;
            }
            if (trial != 0) context.reset_trial();
            result.step = step;
            point_t<Geometry> candidate = geometry.retract(point, direction, step);
            double candidate_cost = evaluate_cost(problem, geometry, candidate, context.trial());
            ++result.trials;
            double scaled_step = step;
            double scaled_derivative = directional_derivative;
            if (step >= std::abs(directional_derivative)) {
                scaled_step *= options_.sufficient_decrease;
            } else {
                scaled_derivative *= options_.sufficient_decrease;
            }
            const double sufficient_decrease = std::fma(scaled_step, scaled_derivative, current_cost);
            if (
              std::isfinite(candidate_cost) && candidate_cost < current_cost && candidate_cost <= sufficient_decrease) {
                result.point.emplace(std::move(candidate));
                result.cost = candidate_cost;
                result.step = step;
                result.status = ArmijoStatus::accepted;
                return result;
            }
            ++result.rejected_trials;
            step *= options_.contraction;
        }
        context.reset_trial();
        result.status = ArmijoStatus::max_trials;
        return result;
    }
};

}   // namespace manifold
}   // namespace fdapde

#endif   // __FDAPDE_MANIFOLD_ARMIJO_H__
