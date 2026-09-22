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

#ifndef __FDAPDE_MANIFOLD_WEIGHTED_KARCHER_MEAN_H__
#define __FDAPDE_MANIFOLD_WEIGHTED_KARCHER_MEAN_H__

#include "header_check.h"

namespace fdapde {
namespace manifold {

/// @brief states whether the geometry certifies a globally unique mean
enum class BarycenterUniqueness {
    globally_unique,
    not_certified
};

/// @brief identifies a closed-form mean, convergence or an iterative failure
enum class BarycenterStopReason {
    stationarity_tolerance,
    max_iterations,
    line_search_failed,
    non_finite_cost,
    non_finite_gradient,
    closed_form
};

/// @brief configures the iterative weighted Riemannian mean
struct WeightedKarcherMeanOptions {
    SteepestDescentOptions solver;
};

/// @brief retains the mean candidate and explicit convergence and uniqueness diagnostics
template <typename Point> struct WeightedKarcherMeanResult {
    Point point;
    std::vector<double> normalized_weights;
    double cost = std::numeric_limits<double>::quiet_NaN();
    double stationarity_norm = std::numeric_limits<double>::quiet_NaN();
    std::size_t iterations = 0;
    std::size_t cost_evaluations = 0;
    std::size_t gradient_evaluations = 0;
    std::size_t rejected_trials = 0;
    BarycenterStopReason stop_reason = BarycenterStopReason::max_iterations;
    BarycenterUniqueness uniqueness = BarycenterUniqueness::not_certified;
    ArmijoStatus line_search_status = ArmijoStatus::not_run;

    /// @brief reports whether the stored stopping certificate indicates convergence
    bool converged() const {
        return stop_reason == BarycenterStopReason::closed_form ||
               stop_reason == BarycenterStopReason::stationarity_tolerance;
    }
};

namespace internals {

inline std::vector<double> normalize_karcher_weights(std::span<const double> weights) {
    double total = 0;
    for (double weight : weights) {
        fdapde_strong_assert(
          std::isfinite(weight) && weight >= 0, std::invalid_argument,
          "Weighted Karcher mean weights must be finite and non-negative");
        total += weight;
    }
    fdapde_strong_assert(
      std::isfinite(total) && total > 0, std::invalid_argument,
      "Weighted Karcher mean weights must have a finite positive total");
    std::vector<double> normalized_weights;
    normalized_weights.reserve(weights.size());
    for (double weight : weights) { normalized_weights.push_back(weight / total); }
    return normalized_weights;
}

inline BarycenterStopReason barycenter_stop_reason(SteepestDescentStopReason reason) {
    switch (reason) {
    case SteepestDescentStopReason::gradient_tolerance:
        return BarycenterStopReason::stationarity_tolerance;
    case SteepestDescentStopReason::max_iterations:
        return BarycenterStopReason::max_iterations;
    case SteepestDescentStopReason::line_search_failed:
        return BarycenterStopReason::line_search_failed;
    case SteepestDescentStopReason::non_finite_cost:
        return BarycenterStopReason::non_finite_cost;
    case SteepestDescentStopReason::non_finite_gradient:
        return BarycenterStopReason::non_finite_gradient;
    }
    throw std::logic_error("Unknown steepest-descent stop reason");
}

/// @brief supplies no storage to geometries without reusable relative frames
template <typename Geometry> struct KarcherWorkspace { };
/// @brief retains candidate-relative frames only within one evaluation generation
template <typename Geometry>
    requires requires { typename Geometry::RelativeFrame; }
struct KarcherWorkspace<Geometry> {
    std::vector<std::optional<typename Geometry::RelativeFrame>> frames;
};

/// @brief shares SPD base factors while keeping rotation bases in their native representation
template <typename Geometry> auto karcher_base(const point_t<Geometry>& point) {
    if constexpr (RotationLike<point_t<Geometry>>)
        return point;
    else
        return SPDMatrix<
          typename Geometry::Scalar, point_t<Geometry>::Rows, point_t<Geometry>::Cols,
          Cache::Union<Cache::Sqrt, Cache::InverseSqrt>>(point);
}

/// @brief evaluates a normalized squared-distance objective using indexed sample storage
template <GeodesicGeometry Geometry, typename Samples> class WeightedKarcherMeanProblem {
   public:
    using Point = point_t<Geometry>;
    using Tangent = tangent_t<Geometry>;
    using Workspace = KarcherWorkspace<Geometry>;

    /// @brief borrows samples and normalized weights for one mean solve
    WeightedKarcherMeanProblem(
      const Geometry& geometry, const Samples& samples, std::span<const double> normalized_weights) :
        geometry_(geometry), samples_(samples), normalized_weights_(normalized_weights) { }

    /// @brief evaluates the objective in the candidate-bound workspace
    double cost(const Point& point, Workspace& workspace) const {
        prepare(point, workspace);
        double result = 0;
        for (std::size_t i = 0; i < samples_.size(); ++i) {
            const double weight = normalized_weights_[i];
            if (weight == 0) continue;
            const double distance = [&] {
                if constexpr (requires { workspace.frames; })
                    return geometry_.distance(*workspace.frames[i]);
                else
                    return geometry_.distance(point, samples_[i]);
            }();
            const double scaled_distance = std::sqrt(weight) * distance;
            result = std::fma(0.5 * scaled_distance, scaled_distance, result);
        }
        return result;
    }

    /// @brief evaluates the Riemannian gradient in the candidate-bound workspace
    Tangent gradient(const Point& point, Workspace& workspace) const {
        prepare(point, workspace);
        Tangent result = geometry_.zero_tangent(point);
        for (std::size_t i = 0; i < samples_.size(); ++i) {
            const double weight = normalized_weights_[i];
            if (weight == 0) continue;
            const auto logarithm = [&] {
                if constexpr (requires { workspace.frames; })
                    return geometry_.logarithm(*workspace.frames[i]);
                else
                    return geometry_.logarithm(point, samples_[i]);
            }();
            result = geometry_.linear_combination(point, 1, result, -weight, logarithm);
        }
        return result;
    }
    /// @brief shares the relative preparation between objective and gradient evaluation
    std::pair<double, Tangent> cost_gradient(const Point& point, Workspace& workspace) const {
        const double value = cost(point, workspace);
        return {value, gradient(point, workspace)};
    }
    /// @brief prepares supported nodes once; the caller resets the workspace on every candidate change
    void prepare(const Point& point, Workspace& workspace) const {
        if constexpr (requires { workspace.frames; }) {
            if (!workspace.frames.empty()) return;
            const auto base = karcher_base<Geometry>(point);
            workspace.frames.resize(samples_.size());
            for (std::size_t i = 0; i < samples_.size(); ++i)
                if (normalized_weights_[i] != 0)
                    workspace.frames[i].emplace(geometry_.relative_frame(base, samples_[i]));
        }
    }
   private:
    const Geometry& geometry_;
    const Samples& samples_;
    std::span<const double> normalized_weights_;
};

/// @brief refines an already local mean when strict cost decrease is hidden by floating-point roundoff
template <typename Geometry, typename Samples>
void polish_karcher_mean(
  const Geometry& geometry, const Samples& samples, const WeightedKarcherMeanOptions& options,
  WeightedKarcherMeanResult<point_t<Geometry>>& result, KarcherWorkspace<Geometry>& workspace) {
    // polish local stationarity when strict Armijo decrease reaches the cost roundoff floor
    using Tangent = typename Geometry::Tangent;
    WeightedKarcherMeanProblem<Geometry, Samples> problem(geometry, samples, result.normalized_weights);
    for (int iteration = 0; iteration < 6 && result.stationarity_norm > options.solver.gradient_tolerance &&
                            result.stationarity_norm < 1e-5 && std::isfinite(result.cost);
         ++iteration) {
        if constexpr (requires { geometry.prepare_linearization(workspace.frames, result.normalized_weights); }) {
            try {
                geometry.prepare_linearization(workspace.frames, result.normalized_weights);
            } catch (const std::domain_error&) { break; }
        }
        const auto gradient = problem.gradient(result.point, workspace);
        auto hessian = [&](const Tangent& u) {
            Tangent h = geometry.zero_tangent(result.point);
            for (std::size_t i = 0; i < samples.size(); ++i)
                if (result.normalized_weights[i] > 0)
                    h = geometry.linear_combination(
                      result.point, 1, h, result.normalized_weights[i],
                      geometry.half_squared_distance_hessian_vector(*workspace.frames[i], u));
            return h;
        };
        const auto rhs = geometry.linear_combination(result.point, -1, gradient, 0, gradient);
        const auto step = PositiveDefiniteConjugateGradient().solve(hessian, geometry, result.point, rhs);
        if (!step.converged() || geometry.norm(result.point, step.solution) > .01) break;
        const auto next = geometry.exponential(result.point, step.solution);
        KarcherWorkspace<Geometry> trial;
        const double cost = problem.cost(next, trial), norm = geometry.norm(next, problem.gradient(next, trial));
        // near stationarity the cost decrease is below roundoff; require residual contraction instead
        if (!std::isfinite(cost) || !(norm <= .5 * result.stationarity_norm)) break;
        workspace = std::move(trial);
        result.point = next;
        result.cost = cost;
        result.stationarity_norm = norm;
        ++result.iterations;
        ++result.cost_evaluations;
        result.gradient_evaluations += 2;
    }
    if (result.stationarity_norm <= options.solver.gradient_tolerance)
        result.stop_reason = BarycenterStopReason::stationarity_tolerance;
}

}   // namespace internals

template <GeodesicGeometry Geometry, typename Samples>
WeightedKarcherMeanResult<point_t<Geometry>> weighted_karcher_mean(
  const Geometry& geometry, const Samples& samples, std::span<const double> weights, const point_t<Geometry>& initial,
  const WeightedKarcherMeanOptions& options = {}, internals::KarcherWorkspace<Geometry>* retained = nullptr) {
    fdapde_strong_assert(
      samples.size() != 0, std::invalid_argument, "Weighted Karcher mean requires at least one sample");
    fdapde_strong_assert(
      samples.size() == weights.size(), std::invalid_argument,
      "Weighted Karcher mean sample and weight counts must match");
    auto normalized_weights = internals::normalize_karcher_weights(weights);
    internals::WeightedKarcherMeanProblem<Geometry, Samples> problem(
      geometry, samples, std::span<const double>(normalized_weights));
    RiemannianSteepestDescent solver(options.solver);
    EvaluationContext<tangent_t<Geometry>, internals::KarcherWorkspace<Geometry>> context;
    auto solver_result = solver.optimize(problem, geometry, initial, context);
    if (retained) *retained = std::move(context.current().workspace());

    return {
      std::move(solver_result.point),
      std::move(normalized_weights),
      solver_result.cost,
      solver_result.gradient_norm,
      solver_result.iterations,
      solver_result.cost_evaluations,
      solver_result.gradient_evaluations,
      solver_result.rejected_trials,
      internals::barycenter_stop_reason(solver_result.stop_reason),
      BarycenterUniqueness::not_certified,
      solver_result.line_search_status};
}

}   // namespace manifold
}   // namespace fdapde

#endif   // __FDAPDE_MANIFOLD_WEIGHTED_KARCHER_MEAN_H__
