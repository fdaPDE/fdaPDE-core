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

#include <fdaPDE/manifold_optimization.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <vector>

namespace {

struct EuclideanGeodesic {
    using Point = double;
    using Tangent = double;

    std::size_t dimension() const { return 1; }
    double inner_product(const Point&, const Tangent& u, const Tangent& v) const { return u * v; }
    double norm(const Point&, const Tangent& tangent) const { return std::abs(tangent); }
    Tangent project(const Point&, const Tangent& tangent) const { return tangent; }
    Tangent zero_tangent(const Point&) const { return 0; }
    Tangent linear_combination(const Point&, double alpha, const Tangent& u, double beta, const Tangent& v) const {
        return alpha * u + beta * v;
    }
    Point retract(const Point& point, const Tangent& tangent, double step) const { return point + step * tangent; }
    Point exponential(const Point& point, const Tangent& tangent, double step) const { return point + step * tangent; }
    Tangent logarithm(const Point& from, const Point& to) const { return to - from; }
    double distance(const Point& from, const Point& to) const { return std::abs(to - from); }
};

struct NonFiniteDistanceGeodesic : EuclideanGeodesic {
    double distance(const Point&, const Point&) const { return std::numeric_limits<double>::infinity(); }
};

struct NonFiniteLogarithmGeodesic : EuclideanGeodesic {
    Tangent logarithm(const Point&, const Point&) const { return std::numeric_limits<double>::infinity(); }
};

template <typename Geometry, std::size_t SampleCount, std::size_t WeightCount>
auto mean(
  const Geometry& geometry, const std::array<double, SampleCount>& samples,
  const std::array<double, WeightCount>& weights, double initial,
  const fdapde::manifold::WeightedKarcherMeanOptions& options = {}) {
    return fdapde::manifold::weighted_karcher_mean(
      geometry, std::span<const double>(samples), std::span<const double>(weights), initial, options);
}

}   // namespace

TEST(WeightedKarcherMean, ComputesNormalizedEuclideanMean) {
    const EuclideanGeodesic geometry;
    const std::array<double, 3> samples {
      {-100, 1, 7}
    };
    const std::array<double, 3> weights {
      {0, 2, 6}
    };

    const auto result = mean(geometry, samples, weights, 0);

    EXPECT_TRUE(result.converged());
    EXPECT_EQ(result.stop_reason, fdapde::manifold::BarycenterStopReason::stationarity_tolerance);
    EXPECT_EQ(result.line_search_status, fdapde::manifold::ArmijoStatus::accepted);
    EXPECT_EQ(result.uniqueness, fdapde::manifold::BarycenterUniqueness::not_certified);
    EXPECT_EQ(result.normalized_weights, (std::vector<double> {0, 0.25, 0.75}));
    EXPECT_DOUBLE_EQ(result.point, 5.5);
    EXPECT_DOUBLE_EQ(result.cost, 3.375);
    EXPECT_DOUBLE_EQ(result.stationarity_norm, 0);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_EQ(result.cost_evaluations, 2);
    EXPECT_EQ(result.gradient_evaluations, 2);
    EXPECT_EQ(result.rejected_trials, 0);
}

TEST(WeightedKarcherMean, IsInvariantToWeightScalingAndZeroWeightRemoval) {
    const EuclideanGeodesic geometry;
    const std::array<double, 3> samples {
      {std::numeric_limits<double>::infinity(), 1, 7}
    };
    const std::array<double, 3> scaled_weights {
      {0, 20, 60}
    };
    const std::array<double, 2> reduced_samples {
      {1, 7}
    };
    const std::array<double, 2> reduced_weights {
      {1, 3}
    };

    const auto scaled = mean(geometry, samples, scaled_weights, 0);
    const auto reduced = mean(geometry, reduced_samples, reduced_weights, 0);

    EXPECT_DOUBLE_EQ(scaled.point, reduced.point);
    EXPECT_DOUBLE_EQ(scaled.cost, reduced.cost);
    EXPECT_DOUBLE_EQ(scaled.stationarity_norm, reduced.stationarity_norm);
    EXPECT_EQ(scaled.normalized_weights, (std::vector<double> {0, 0.25, 0.75}));
    EXPECT_EQ(reduced.normalized_weights, (std::vector<double> {0.25, 0.75}));
}

TEST(WeightedKarcherMean, PreservesRepresentableSmallCostContributions) {
    const EuclideanGeodesic geometry;
    const std::array<double, 2> samples {
      {1.0e154, 0}
    };
    const std::array<double, 2> weights {
      {std::numeric_limits<double>::denorm_min(), 1}
    };

    const auto result = mean(geometry, samples, weights, 0);
    const double scaled_distance = std::sqrt(weights[0]) * samples[0];
    const double expected_cost = 0.5 * scaled_distance * scaled_distance;

    EXPECT_TRUE(result.converged());
    EXPECT_GT(result.cost, 0);
    EXPECT_NEAR(result.cost, expected_cost, expected_cost * 1.0e-15);
    EXPECT_GT(result.stationarity_norm, 0);
    EXPECT_EQ(result.iterations, 0);
}

TEST(WeightedKarcherMean, StopsImmediatelyAtAConstantMean) {
    const EuclideanGeodesic geometry;
    const std::array<double, 3> samples {
      {4, 4, 4}
    };
    const std::array<double, 3> weights {
      {1, 2, 3}
    };

    const auto result = mean(geometry, samples, weights, 4);

    EXPECT_TRUE(result.converged());
    EXPECT_DOUBLE_EQ(result.point, 4);
    EXPECT_DOUBLE_EQ(result.cost, 0);
    EXPECT_DOUBLE_EQ(result.stationarity_norm, 0);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(result.line_search_status, fdapde::manifold::ArmijoStatus::not_run);
}

TEST(WeightedKarcherMean, RejectsInvalidInputsAndOptions) {
    const EuclideanGeodesic geometry;
    const std::array<double, 1> samples {{1}};
    const std::array<double, 1> valid_weights {{1}};
    const std::array<double, 2> extra_weights {
      {1, 1}
    };
    const std::array<double, 1> negative_weights {{-1}};
    const std::array<double, 1> nan_weights {{std::numeric_limits<double>::quiet_NaN()}};
    const std::array<double, 1> infinite_weights {{std::numeric_limits<double>::infinity()}};
    const std::array<double, 1> zero_weights {{0}};
    const std::array<double, 2> overflowing_weights {
      {std::numeric_limits<double>::max(), std::numeric_limits<double>::max()}
    };

    EXPECT_THROW(
      fdapde::manifold::weighted_karcher_mean(geometry, std::span<const double> {}, std::span<const double> {}, 0),
      std::invalid_argument);
    EXPECT_THROW(mean(geometry, samples, extra_weights, 0), std::invalid_argument);
    EXPECT_THROW(mean(geometry, samples, negative_weights, 0), std::invalid_argument);
    EXPECT_THROW(mean(geometry, samples, nan_weights, 0), std::invalid_argument);
    EXPECT_THROW(mean(geometry, samples, infinite_weights, 0), std::invalid_argument);
    EXPECT_THROW(mean(geometry, samples, zero_weights, 0), std::invalid_argument);
    EXPECT_THROW(
      mean(
        geometry,
        std::array<double, 2> {
          {1, 2}
    },
        overflowing_weights, 0),
      std::invalid_argument);

    fdapde::manifold::WeightedKarcherMeanOptions options;
    options.solver.max_iterations = 0;
    EXPECT_THROW(mean(geometry, samples, valid_weights, 0, options), std::invalid_argument);
}

TEST(WeightedKarcherMean, ReportsMaximumIterationsAndLineSearchFailure) {
    const EuclideanGeodesic geometry;
    const std::array<double, 2> samples {
      {0, 2}
    };
    const std::array<double, 2> weights {
      {1, 1}
    };

    fdapde::manifold::WeightedKarcherMeanOptions limited;
    limited.solver.max_iterations = 1;
    limited.solver.gradient_tolerance = 0;
    limited.solver.line_search.initial_step = 0.5;
    const auto maximum_iterations = mean(geometry, samples, weights, 0, limited);

    EXPECT_FALSE(maximum_iterations.converged());
    EXPECT_EQ(maximum_iterations.stop_reason, fdapde::manifold::BarycenterStopReason::max_iterations);
    EXPECT_EQ(maximum_iterations.uniqueness, fdapde::manifold::BarycenterUniqueness::not_certified);
    EXPECT_DOUBLE_EQ(maximum_iterations.point, 0.5);
    EXPECT_DOUBLE_EQ(maximum_iterations.cost, 0.625);
    EXPECT_DOUBLE_EQ(maximum_iterations.stationarity_norm, 0.5);
    EXPECT_EQ(maximum_iterations.iterations, 1);
    EXPECT_EQ(maximum_iterations.cost_evaluations, 2);
    EXPECT_EQ(maximum_iterations.gradient_evaluations, 2);
    EXPECT_EQ(maximum_iterations.rejected_trials, 0);

    fdapde::manifold::WeightedKarcherMeanOptions rejected;
    rejected.solver.line_search.initial_step = 4;
    rejected.solver.line_search.max_trials = 1;
    const auto line_search_failure = mean(geometry, samples, weights, 0, rejected);

    EXPECT_FALSE(line_search_failure.converged());
    EXPECT_EQ(line_search_failure.stop_reason, fdapde::manifold::BarycenterStopReason::line_search_failed);
    EXPECT_EQ(line_search_failure.line_search_status, fdapde::manifold::ArmijoStatus::max_trials);
    EXPECT_DOUBLE_EQ(line_search_failure.point, 0);
    EXPECT_DOUBLE_EQ(line_search_failure.cost, 1);
    EXPECT_DOUBLE_EQ(line_search_failure.stationarity_norm, 1);
    EXPECT_EQ(line_search_failure.iterations, 0);
    EXPECT_EQ(line_search_failure.cost_evaluations, 2);
    EXPECT_EQ(line_search_failure.gradient_evaluations, 1);
    EXPECT_EQ(line_search_failure.rejected_trials, 1);
}

TEST(WeightedKarcherMean, ReportsNonFiniteCostAndGradient) {
    const std::array<double, 1> samples {{1}};
    const std::array<double, 1> weights {{1}};

    const auto non_finite_cost = mean(NonFiniteDistanceGeodesic {}, samples, weights, 0);
    EXPECT_EQ(non_finite_cost.stop_reason, fdapde::manifold::BarycenterStopReason::non_finite_cost);
    EXPECT_FALSE(non_finite_cost.converged());

    const auto non_finite_gradient = mean(NonFiniteLogarithmGeodesic {}, samples, weights, 0);
    EXPECT_EQ(non_finite_gradient.stop_reason, fdapde::manifold::BarycenterStopReason::non_finite_gradient);
    EXPECT_FALSE(non_finite_gradient.converged());
}

TEST(WeightedKarcherMean, RepeatedRunsAreDeterministic) {
    const EuclideanGeodesic geometry;
    const std::array<double, 3> samples {
      {-100, 1, 7}
    };
    const std::array<double, 3> weights {
      {0, 2, 6}
    };

    const auto first = mean(geometry, samples, weights, 0);
    const auto second = mean(geometry, samples, weights, 0);

    EXPECT_DOUBLE_EQ(first.point, second.point);
    EXPECT_EQ(first.normalized_weights, second.normalized_weights);
    EXPECT_DOUBLE_EQ(first.cost, second.cost);
    EXPECT_DOUBLE_EQ(first.stationarity_norm, second.stationarity_norm);
    EXPECT_EQ(first.iterations, second.iterations);
    EXPECT_EQ(first.cost_evaluations, second.cost_evaluations);
    EXPECT_EQ(first.gradient_evaluations, second.gradient_evaluations);
    EXPECT_EQ(first.rejected_trials, second.rejected_trials);
    EXPECT_EQ(first.stop_reason, second.stop_reason);
    EXPECT_EQ(first.uniqueness, second.uniqueness);
    EXPECT_EQ(first.line_search_status, second.line_search_status);
}
