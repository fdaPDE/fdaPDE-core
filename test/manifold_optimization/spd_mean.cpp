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
#include <utility>
#include <vector>

namespace {

namespace native = fdapde::linalg;

constexpr std::array<double, 9> first_coefficients {4.0, 0.6, 0.2, 0.6, 2.5, -0.3, 0.2, -0.3, 1.7};
constexpr std::array<double, 9> second_coefficients {1.8, -0.25, 0.15, -0.25, 3.3, 0.4, 0.15, 0.4, 2.2};
constexpr std::array<double, 9> third_coefficients {2.6, 0.35, -0.2, 0.35, 1.4, 0.1, -0.2, 0.1, 4.1};

template <typename Geometry> typename Geometry::Point make_point(const std::array<double, 9>& coefficients) {
    native::Matrix<double, 3, 3> dense;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) { dense(i, j) = coefficients[static_cast<std::size_t>(3 * i + j)]; }
    }
    return typename Geometry::Point(dense, native::checked);
}

template <typename Geometry> typename Geometry::Point identity_point() {
    native::Matrix<double, 3, 3> dense;
    dense.set_zero();
    for (int i = 0; i < 3; ++i) { dense(i, i) = 1.0; }
    return typename Geometry::Point(dense, native::checked);
}

template <typename Lhs, typename Rhs> double matrix_difference_norm(const Lhs& lhs, const Rhs& rhs) {
    double result = 0;
    for (int i = 0; i < lhs.rows(); ++i) {
        for (int j = 0; j < lhs.cols(); ++j) {
            result = std::hypot(result, static_cast<double>(lhs(i, j)) - static_cast<double>(rhs(i, j)));
        }
    }
    return result;
}

template <typename Geometry> class FrechetMeanProblem {
   public:
    using Point = typename Geometry::Point;
    using Tangent = typename Geometry::Tangent;
    struct Workspace { };

    FrechetMeanProblem(const Geometry& geometry, std::vector<Point> samples) :
        geometry_(geometry), samples_(std::move(samples)) { }

    double cost(const Point& point, Workspace&) const {
        double result = 0;
        for (const auto& sample : samples_) {
            const double distance = geometry_.distance(point, sample);
            result += 0.5 * distance * distance;
        }
        return result;
    }

    Tangent gradient(const Point& point, Workspace&) const {
        auto result = geometry_.zero_tangent(point);
        for (const auto& sample : samples_) {
            result = geometry_.linear_combination(point, 1.0, result, -1.0, geometry_.logarithm(point, sample));
        }
        return result;
    }
   private:
    const Geometry& geometry_;
    std::vector<Point> samples_;
};

template <typename Geometry>
typename Geometry::Point
closed_form_log_mean(const Geometry& geometry, const std::vector<typename Geometry::Point>& samples) {
    auto mean_log = geometry.zero_tangent(samples.front());
    for (const auto& sample : samples) {
        const auto sample_log = native::matrix_log(sample);
        for (int i = 0; i < sample_log.rows(); ++i) {
            for (int j = 0; j <= i; ++j) {
                mean_log(i, j) = static_cast<double>(mean_log(i, j)) +
                                 static_cast<double>(sample_log(i, j)) / static_cast<double>(samples.size());
            }
        }
    }
    return native::matrix_exp(mean_log);
}

template <typename Geometry> void expect_log_mean_matches_closed_form(const Geometry& geometry) {
    using Problem = FrechetMeanProblem<Geometry>;
    std::vector<typename Geometry::Point> samples {
      make_point<Geometry>(first_coefficients), make_point<Geometry>(second_coefficients),
      make_point<Geometry>(third_coefficients)};
    Problem problem(geometry, samples);
    const auto expected = closed_form_log_mean(geometry, samples);
    const auto initial = identity_point<Geometry>();
    typename Problem::Workspace workspace;
    const double initial_cost = problem.cost(initial, workspace);

    fdapde::manifold::SteepestDescentOptions options;
    options.max_iterations = 20;
    options.gradient_tolerance = 1.0e-9;
    options.line_search.initial_step = 1.0 / static_cast<double>(samples.size());
    const fdapde::manifold::RiemannianSteepestDescent optimizer(options);
    const auto result = optimizer.optimize(problem, geometry, initial);

    EXPECT_TRUE(result.converged());
    EXPECT_EQ(result.stop_reason, fdapde::manifold::SteepestDescentStopReason::gradient_tolerance);
    EXPECT_EQ(result.line_search_status, fdapde::manifold::ArmijoStatus::accepted);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_EQ(result.cost_evaluations, 2);
    EXPECT_EQ(result.gradient_evaluations, 2);
    EXPECT_EQ(result.rejected_trials, 0);
    EXPECT_LT(result.gradient_norm, 1.0e-9);
    EXPECT_LT(result.cost, initial_cost);
    EXPECT_LT(geometry.distance(result.point, expected), 1.0e-9);
}

template <typename Geometry> typename Geometry::Point optimize_two_point_affine_mean(const Geometry& geometry) {
    using Problem = FrechetMeanProblem<Geometry>;
    const std::vector<typename Geometry::Point> samples {
      make_point<Geometry>(first_coefficients), make_point<Geometry>(second_coefficients)};
    Problem problem(geometry, samples);
    const auto expected = geometry.exponential(samples[0], geometry.logarithm(samples[0], samples[1]), 0.5);
    const auto initial = identity_point<Geometry>();

    fdapde::manifold::SteepestDescentOptions options;
    options.max_iterations = 100;
    options.gradient_tolerance = 1.0e-8;
    options.line_search.initial_step = 0.5;
    const fdapde::manifold::RiemannianSteepestDescent optimizer(options);
    const auto result = optimizer.optimize(problem, geometry, initial);

    EXPECT_TRUE(result.converged());
    EXPECT_EQ(result.stop_reason, fdapde::manifold::SteepestDescentStopReason::gradient_tolerance);
    EXPECT_EQ(result.line_search_status, fdapde::manifold::ArmijoStatus::accepted);
    EXPECT_LT(result.gradient_norm, 1.0e-8);
    EXPECT_LT(geometry.distance(result.point, expected), 1.0e-7);
    return result.point;
}

using FieldComponent = fdapde::manifold::LogEuclideanSPDGeometry<double, 3>;
using FieldGeometry = fdapde::manifold::PowerGeometry<FieldComponent>;

class FieldMeanProblem {
   public:
    struct Workspace { };

    FieldMeanProblem(const FieldComponent& component, std::vector<FieldGeometry::Point> samples) :
        component_(component), samples_(std::move(samples)) { }

    double cost(const FieldGeometry::Point& point, Workspace&) const {
        double result = 0;
        for (const auto& sample : samples_) {
            for (std::size_t i = 0; i < point.size(); ++i) {
                const double distance = component_.distance(point[i], sample[i]);
                result += 0.5 * distance * distance;
            }
        }
        return result;
    }

    FieldGeometry::Tangent gradient(const FieldGeometry::Point& point, Workspace&) const {
        FieldGeometry::Tangent result;
        result.reserve(point.size());
        for (std::size_t i = 0; i < point.size(); ++i) {
            auto component_gradient = component_.zero_tangent(point[i]);
            for (const auto& sample : samples_) {
                component_gradient = component_.linear_combination(
                  point[i], 1.0, component_gradient, -1.0, component_.logarithm(point[i], sample[i]));
            }
            result.push_back(std::move(component_gradient));
        }
        return result;
    }
   private:
    const FieldComponent& component_;
    std::vector<FieldGeometry::Point> samples_;
};

}   // namespace

TEST(LogEuclideanSPDMean, FixedAndDynamicThreeByThreeMeansMatchTheClosedFormOracle) {
    expect_log_mean_matches_closed_form(fdapde::manifold::LogEuclideanSPDGeometry<double, 3> {});
    expect_log_mean_matches_closed_form(fdapde::manifold::LogEuclideanSPDGeometry<double, fdapde::Dynamic> {3});
}

TEST(LogEuclideanSPDMean, PowerGeometryConvergesComponentwiseToClosedFormMeans) {
    const FieldComponent component;
    const FieldGeometry geometry(2, component);
    const FieldGeometry::Point first {
      make_point<FieldComponent>(first_coefficients), make_point<FieldComponent>(second_coefficients)};
    const FieldGeometry::Point second {
      make_point<FieldComponent>(third_coefficients), make_point<FieldComponent>(first_coefficients)};
    const std::vector<FieldGeometry::Point> samples {first, second};
    FieldMeanProblem problem(component, samples);
    const auto identity = identity_point<FieldComponent>();
    const FieldGeometry::Point initial(geometry.factor_count(), identity);
    const std::array<FieldComponent::Point, 2> expected {
      closed_form_log_mean(component, {first[0], second[0]}), closed_form_log_mean(component, {first[1], second[1]})};

    fdapde::manifold::SteepestDescentOptions options;
    options.max_iterations = 10;
    options.gradient_tolerance = 1.0e-9;
    options.line_search.initial_step = 0.5;
    const fdapde::manifold::RiemannianSteepestDescent optimizer(options);
    const auto result = optimizer.optimize(problem, geometry, initial);

    EXPECT_EQ(geometry.dimension(), 12);
    EXPECT_TRUE(result.converged());
    EXPECT_EQ(result.iterations, 1);
    EXPECT_EQ(result.cost_evaluations, 2);
    EXPECT_EQ(result.gradient_evaluations, 2);
    EXPECT_EQ(result.rejected_trials, 0);
    for (std::size_t i = 0; i < geometry.factor_count(); ++i) {
        EXPECT_LT(component.distance(result.point[i], expected[i]), 1.0e-9);
    }
}

TEST(AffineInvariantSPDMean, FixedAndDynamicThreeByThreeMeansMatchTheGeodesicMidpoint) {
    const fdapde::manifold::AffineInvariantSPDGeometry<double, 3> fixed_geometry;
    const fdapde::manifold::AffineInvariantSPDGeometry<double, fdapde::Dynamic> dynamic_geometry(3);
    const auto fixed_result = optimize_two_point_affine_mean(fixed_geometry);
    const auto dynamic_result = optimize_two_point_affine_mean(dynamic_geometry);

    EXPECT_LT(matrix_difference_norm(fixed_result, dynamic_result), 1.0e-10);
}
