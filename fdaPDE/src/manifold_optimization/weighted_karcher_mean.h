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

enum class BarycenterUniqueness {
    globally_unique,
    not_certified
};

enum class BarycenterStopReason {
    stationarity_tolerance,
    max_iterations,
    line_search_failed,
    non_finite_cost,
    non_finite_gradient
};

struct WeightedKarcherMeanOptions {
    SteepestDescentOptions solver;
};

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

    bool converged() const { return stop_reason == BarycenterStopReason::stationarity_tolerance; }
};

namespace internals {

inline std::vector<double> normalize_karcher_weights(std::span<const double> weights) {
    double total = 0;
    for (double weight : weights) {
        if (!std::isfinite(weight) || weight < 0)
            throw std::invalid_argument("Weighted Karcher mean weights must be finite and non-negative");
        total += weight;
    }
    if (!std::isfinite(total) || total <= 0)
        throw std::invalid_argument("Weighted Karcher mean weights must have a finite positive total");

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

template <GeodesicGeometry Geometry> class WeightedKarcherMeanProblem {
   public:
    using Point = point_t<Geometry>;
    using Tangent = tangent_t<Geometry>;
    struct Workspace { };

    WeightedKarcherMeanProblem(
      const Geometry& geometry, std::span<const Point> samples, std::span<const double> normalized_weights) :
        geometry_(geometry), samples_(samples), normalized_weights_(normalized_weights) { }

    double cost(const Point& point, Workspace&) const {
        double result = 0;
        for (std::size_t i = 0; i < samples_.size(); ++i) {
            const double weight = normalized_weights_[i];
            if (weight == 0) continue;
            const double distance = geometry_.distance(point, samples_[i]);
            const double scaled_distance = std::sqrt(weight) * distance;
            result = std::fma(0.5 * scaled_distance, scaled_distance, result);
        }
        return result;
    }

    Tangent gradient(const Point& point, Workspace&) const {
        Tangent result = geometry_.zero_tangent(point);
        for (std::size_t i = 0; i < samples_.size(); ++i) {
            const double weight = normalized_weights_[i];
            if (weight == 0) continue;
            result = geometry_.linear_combination(point, 1, result, -weight, geometry_.logarithm(point, samples_[i]));
        }
        return result;
    }
   private:
    const Geometry& geometry_;
    std::span<const Point> samples_;
    std::span<const double> normalized_weights_;
};

}   // namespace internals

template <GeodesicGeometry Geometry>
WeightedKarcherMeanResult<point_t<Geometry>> weighted_karcher_mean(
  const Geometry& geometry, std::span<const point_t<Geometry>> samples, std::span<const double> weights,
  const point_t<Geometry>& initial, const WeightedKarcherMeanOptions& options = {}) {
    if (samples.empty()) throw std::invalid_argument("Weighted Karcher mean requires at least one sample");
    if (samples.size() != weights.size())
        throw std::invalid_argument("Weighted Karcher mean sample and weight counts must match");

    auto normalized_weights = internals::normalize_karcher_weights(weights);
    internals::WeightedKarcherMeanProblem<Geometry> problem(
      geometry, samples, std::span<const double>(normalized_weights));
    RiemannianSteepestDescent solver(options.solver);
    auto solver_result = solver.optimize(problem, geometry, initial);

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
