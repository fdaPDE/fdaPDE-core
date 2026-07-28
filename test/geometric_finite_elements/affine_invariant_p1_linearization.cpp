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

#include <fdaPDE/geometric_finite_elements.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <span>
#include <vector>

namespace {

namespace manifold = fdapde::manifold;
namespace native = fdapde::linalg;

using FixedGeometry = manifold::AffineInvariantSPDGeometry<double, 3>;
using DynamicGeometry = manifold::AffineInvariantSPDGeometry<double, fdapde::Dynamic>;
using FixedLogGeometry = manifold::LogEuclideanSPDGeometry<double, 3>;
using DynamicLogGeometry = manifold::LogEuclideanSPDGeometry<double, fdapde::Dynamic>;

constexpr std::array<double, 9> first_coefficients {4.0, 0.6, 0.2, 0.6, 2.5, -0.3, 0.2, -0.3, 1.7};
constexpr std::array<double, 9> second_coefficients {1.8, -0.25, 0.15, -0.25, 3.3, 0.4, 0.15, 0.4, 2.2};
constexpr std::array<double, 9> third_coefficients {2.6, 0.35, -0.2, 0.35, 1.4, 0.1, -0.2, 0.1, 4.1};

template <typename Geometry> Geometry make_geometry() {
    if constexpr (Geometry::Point::Rows == fdapde::Dynamic) {
        return Geometry(3);
    } else {
        return Geometry {};
    }
}

template <typename Geometry> typename Geometry::Point make_point(const std::array<double, 9>& coefficients) {
    native::Matrix<double, 3, 3> dense;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) { dense(i, j) = coefficients[static_cast<std::size_t>(3 * i + j)]; }
    }
    return typename Geometry::Point(dense, native::checked);
}

template <typename Geometry> typename Geometry::Point make_diagonal_point(double a, double b, double c) {
    native::Matrix<double, 3, 3> dense;
    dense.set_zero();
    dense(0, 0) = a;
    dense(1, 1) = b;
    dense(2, 2) = c;
    return typename Geometry::Point(dense, native::checked);
}

template <typename Geometry> typename Geometry::Tangent make_tangent(const std::array<double, 6>& coefficients) {
    typename Geometry::Tangent tangent;
    if constexpr (Geometry::Tangent::Rows == fdapde::Dynamic) { tangent.resize(3, 3); }
    std::size_t index = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j <= i; ++j) { tangent(i, j) = coefficients[index++]; }
    }
    return tangent;
}

template <typename Geometry> std::vector<typename Geometry::Point> noncommuting_nodes() {
    return {
      make_point<Geometry>(first_coefficients), make_point<Geometry>(second_coefficients),
      make_point<Geometry>(third_coefficients)};
}

template <typename Geometry> std::vector<typename Geometry::Tangent> nodal_directions() {
    return {
      make_tangent<Geometry>({0.3, -0.2, 0.4, 0.1, 0.25, -0.15}),
      make_tangent<Geometry>({-0.1, 0.35, 0.2, -0.25, 0.05, 0.45}),
      make_tangent<Geometry>({0.2, 0.1, -0.3, 0.4, -0.15, 0.25})};
}

fdapde::gfe::P1GeodesicLinearizationOptions accurate_options() {
    fdapde::gfe::P1GeodesicLinearizationOptions options;
    options.mean.solver.max_iterations = 400;
    options.mean.solver.gradient_tolerance = 1.0e-8;
    options.linear_solve.max_iterations = 12;
    options.linear_solve.residual_tolerance = 1.0e-11;
    return options;
}

double compensated_sum(std::span<const double> values) {
    double sum = 0;
    double correction = 0;
    for (const double value : values) {
        const double corrected = value - correction;
        const double next = sum + corrected;
        correction = (next - sum) - corrected;
        sum = next;
    }
    return sum;
}

template <typename Geometry, typename Lhs, typename Rhs>
double relative_tangent_error(
  const Geometry& geometry, const typename Geometry::Point& point, const Lhs& lhs, const Rhs& rhs) {
    const auto difference = geometry.linear_combination(point, 1, lhs, -1, rhs);
    return geometry.norm(point, difference) / std::max({1.0, geometry.norm(point, lhs), geometry.norm(point, rhs)});
}

template <typename Geometry, typename Lhs, typename Rhs>
void expect_tangent_near(
  const Geometry& geometry, const typename Geometry::Point& point, const Lhs& lhs, const Rhs& rhs, double tolerance) {
    EXPECT_LE(relative_tangent_error(geometry, point, lhs, rhs), tolerance);
}

template <typename Lhs, typename Rhs> void expect_same_coefficients(const Lhs& lhs, const Rhs& rhs) {
    ASSERT_EQ(lhs.rows(), rhs.rows());
    ASSERT_EQ(lhs.cols(), rhs.cols());
    for (int i = 0; i < lhs.rows(); ++i) {
        for (int j = 0; j <= i; ++j) {
            EXPECT_DOUBLE_EQ(static_cast<double>(lhs(i, j)), static_cast<double>(rhs(i, j)));
        }
    }
}

template <typename Geometry>
typename Geometry::Tangent hessian_action(
  const Geometry& geometry, const typename Geometry::Point& mean, std::span<const typename Geometry::Point> nodes,
  std::span<const double> effective_weights, const typename Geometry::Tangent& direction) {
    auto result = geometry.zero_tangent(mean);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (effective_weights[i] == 0) continue;
        const auto component = geometry.half_squared_distance_hessian_vector(mean, nodes[i], direction);
        result = geometry.linear_combination(mean, 1, result, effective_weights[i], component);
    }
    return result;
}

template <typename Geometry>
typename Geometry::Tangent weight_right_hand_side(
  const Geometry& geometry, const typename Geometry::Point& mean, std::span<const typename Geometry::Point> nodes,
  std::span<const double> represented_weights, std::span<const double> direction) {
    const double total = compensated_sum(represented_weights);
    auto residual = geometry.zero_tangent(mean);
    std::vector<typename Geometry::Tangent> logs;
    logs.reserve(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        logs.push_back(geometry.logarithm(mean, nodes[i]));
        residual = geometry.linear_combination(mean, 1, residual, represented_weights[i] / total, logs.back());
    }

    auto result = geometry.zero_tangent(mean);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto centered = geometry.linear_combination(mean, 1, logs[i], -1, residual);
        result = geometry.linear_combination(mean, 1, result, direction[i] / total, centered);
    }
    return result;
}

template <typename Geometry>
typename Geometry::Tangent nodal_right_hand_side(
  const Geometry& geometry, const typename Geometry::Point& mean, std::span<const typename Geometry::Point> nodes,
  std::span<const double> represented_weights, std::span<const typename Geometry::Tangent> directions) {
    const double total = compensated_sum(represented_weights);
    auto result = geometry.zero_tangent(mean);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        if (represented_weights[i] == 0) continue;
        const auto target_action = geometry.logarithm_target_jvp(mean, nodes[i], directions[i]);
        result = geometry.linear_combination(mean, 1, result, represented_weights[i] / total, target_action);
    }
    return result;
}

template <typename Geometry>
typename Geometry::Tangent
congruence_direction(const native::Matrix<double, 3, 3>& generator, const typename Geometry::Point& point) {
    auto result = make_tangent<Geometry>({0, 0, 0, 0, 0, 0});
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j <= i; ++j) {
            double value = 0;
            for (int k = 0; k < 3; ++k) {
                value += generator(i, k) * static_cast<double>(point(k, j)) +
                         static_cast<double>(point(i, k)) * generator(j, k);
            }
            result(i, j) = value;
        }
    }
    return result;
}

auto owning_dynamic_linearization() {
    const DynamicGeometry geometry(3);
    auto nodes = noncommuting_nodes<DynamicGeometry>();
    std::vector<double> weights {0.25, 0.5, 0.25};
    auto linearization = fdapde::gfe::p1_geodesic_linearization(
      geometry, std::span<const DynamicGeometry::Point>(nodes), std::span<const double>(weights), accurate_options());
    nodes[0] = make_diagonal_point<DynamicGeometry>(7, 7, 7);
    weights = {1, 0, 0};
    return linearization;
}

}   // namespace

TEST(AffineInvariantP1Linearization, HasExactVertexAndSingleNodeActions) {
    const FixedGeometry geometry;
    const auto nodes = noncommuting_nodes<FixedGeometry>();
    const std::vector<double> weights {0, 1, 0};
    const std::vector<double> weight_direction {0.25, -0.5, 0.25};
    const auto directions = nodal_directions<FixedGeometry>();
    const auto output = make_tangent<FixedGeometry>({0.4, -0.15, 0.3, 0.2, -0.35, 0.1});
    const auto linearization = fdapde::gfe::p1_geodesic_linearization(
      geometry, std::span<const FixedGeometry::Point>(nodes), std::span<const double>(weights), accurate_options());

    const auto weight_action = linearization.weight_jvp(weight_direction);
    const auto nodal_action = linearization.nodal_jvp(directions);
    const auto pullback = linearization.nodal_vjp(output);
    ASSERT_TRUE(weight_action.converged());
    ASSERT_TRUE(nodal_action.converged());
    ASSERT_TRUE(pullback.converged());
    EXPECT_EQ(nodal_action.iterations, 0);
    EXPECT_DOUBLE_EQ(nodal_action.residual_norm, 0);
    EXPECT_EQ(pullback.iterations, 0);
    EXPECT_DOUBLE_EQ(pullback.residual_norm, 0);

    auto expected_weight = geometry.zero_tangent(nodes[1]);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        expected_weight = geometry.linear_combination(
          nodes[1], 1, expected_weight, weight_direction[i], geometry.logarithm(nodes[1], nodes[i]));
    }
    expect_tangent_near(geometry, nodes[1], weight_action.derivative, expected_weight, 2.0e-10);
    expect_tangent_near(geometry, nodes[1], nodal_action.derivative, directions[1], 2.0e-10);
    ASSERT_EQ(pullback.derivative.size(), nodes.size());
    expect_tangent_near(geometry, nodes[0], pullback.derivative[0], geometry.zero_tangent(nodes[0]), 0);
    expect_tangent_near(geometry, nodes[1], pullback.derivative[1], output, 2.0e-10);
    expect_tangent_near(geometry, nodes[2], pullback.derivative[2], geometry.zero_tangent(nodes[2]), 0);

    const std::vector<FixedGeometry::Point> one_node {nodes[1]};
    const std::vector<double> one_weight {1};
    const std::vector<double> zero_weight_direction {0};
    const std::vector<FixedGeometry::Tangent> one_direction {directions[1]};
    const auto one = fdapde::gfe::p1_geodesic_linearization(
      geometry, std::span<const FixedGeometry::Point>(one_node), std::span<const double>(one_weight),
      accurate_options());
    const auto zero_action = one.weight_jvp(zero_weight_direction);
    EXPECT_TRUE(zero_action.converged());
    EXPECT_EQ(zero_action.iterations, 0);
    EXPECT_DOUBLE_EQ(zero_action.residual_norm, 0);
    expect_tangent_near(geometry, nodes[1], zero_action.derivative, geometry.zero_tangent(nodes[1]), 0);
    const auto one_nodal = one.nodal_jvp(one_direction);
    const auto one_pullback = one.nodal_vjp(output);
    ASSERT_TRUE(one_nodal.converged());
    ASSERT_TRUE(one_pullback.converged());
    EXPECT_EQ(one_nodal.iterations, 0);
    EXPECT_EQ(one_pullback.iterations, 0);
    expect_tangent_near(geometry, nodes[1], one_nodal.derivative, directions[1], 2.0e-10);
    expect_tangent_near(geometry, nodes[1], one_pullback.derivative[0], output, 2.0e-10);
}

template <typename AffineGeometry, typename LogGeometry> void check_commuting_log_oracle() {
    const AffineGeometry affine_geometry = make_geometry<AffineGeometry>();
    const LogGeometry log_geometry = make_geometry<LogGeometry>();
    const std::vector<typename AffineGeometry::Point> affine_nodes {
      make_diagonal_point<AffineGeometry>(1.5, 3, 6), make_diagonal_point<AffineGeometry>(4, 2, 1.25),
      make_diagonal_point<AffineGeometry>(2.25, 5, 3.5)};
    const std::vector<typename LogGeometry::Point> log_nodes {
      make_diagonal_point<LogGeometry>(1.5, 3, 6), make_diagonal_point<LogGeometry>(4, 2, 1.25),
      make_diagonal_point<LogGeometry>(2.25, 5, 3.5)};
    const std::vector<double> weights {0.2, 0.3, 0.5};
    const std::vector<double> weight_direction {0.15, -0.25, 0.1};
    const std::vector<typename AffineGeometry::Tangent> affine_directions {
      make_tangent<AffineGeometry>({0.3, 0, -0.2, 0, 0, 0.4}),
      make_tangent<AffineGeometry>({-0.1, 0, 0.35, 0, 0, -0.25}),
      make_tangent<AffineGeometry>({0.2, 0, 0.1, 0, 0, 0.3})};
    const std::vector<typename LogGeometry::Tangent> log_directions {
      make_tangent<LogGeometry>({0.3, 0, -0.2, 0, 0, 0.4}), make_tangent<LogGeometry>({-0.1, 0, 0.35, 0, 0, -0.25}),
      make_tangent<LogGeometry>({0.2, 0, 0.1, 0, 0, 0.3})};
    const auto affine_output = make_tangent<AffineGeometry>({0.4, 0, -0.15, 0, 0, 0.25});
    const auto log_output = make_tangent<LogGeometry>({0.4, 0, -0.15, 0, 0, 0.25});

    const auto affine = fdapde::gfe::p1_geodesic_linearization(
      affine_geometry, std::span<const typename AffineGeometry::Point>(affine_nodes), std::span<const double>(weights),
      accurate_options());
    const auto logarithmic = fdapde::gfe::p1_geodesic_linearization(
      log_geometry, std::span<const typename LogGeometry::Point>(log_nodes), std::span<const double>(weights));
    ASSERT_TRUE(affine.result().converged());
    EXPECT_LT(affine_geometry.distance(affine.result().value, logarithmic.result().value), 1.0e-9);

    const auto affine_weight = affine.weight_jvp(weight_direction);
    const auto affine_nodal = affine.nodal_jvp(affine_directions);
    const auto affine_pullback = affine.nodal_vjp(affine_output);
    ASSERT_TRUE(affine_weight.converged());
    ASSERT_TRUE(affine_nodal.converged());
    ASSERT_TRUE(affine_pullback.converged());
    expect_tangent_near(
      affine_geometry, affine.result().value, affine_weight.derivative, logarithmic.weight_jvp(weight_direction),
      2.0e-9);
    expect_tangent_near(
      affine_geometry, affine.result().value, affine_nodal.derivative, logarithmic.nodal_jvp(log_directions), 2.0e-9);
    const auto logarithmic_pullback = logarithmic.nodal_vjp(log_output);
    for (std::size_t i = 0; i < affine_nodes.size(); ++i) {
        expect_tangent_near(
          affine_geometry, affine_nodes[i], affine_pullback.derivative[i], logarithmic_pullback[i], 2.0e-9);
    }
}

TEST(AffineInvariantP1Linearization, MatchesTheCommutingLogEuclideanOracleForFixedAndDynamicSPD) {
    check_commuting_log_oracle<FixedGeometry, FixedLogGeometry>();
    check_commuting_log_oracle<DynamicGeometry, DynamicLogGeometry>();
}

TEST(AffineInvariantP1Linearization, MatchesExactNoncommutingTwoNodeJacobiFields) {
    const FixedGeometry geometry;
    const std::vector<FixedGeometry::Point> nodes {
      make_point<FixedGeometry>(first_coefficients), make_point<FixedGeometry>(second_coefficients)};
    constexpr double parameter = 0.6;
    const std::vector<double> weights {1 - parameter, parameter};
    const std::vector<double> weight_direction {-1, 1};
    const auto geodesic_direction = geometry.logarithm(nodes[0], nodes[1]);
    const auto exact_mean = geometry.exponential(nodes[0], geodesic_direction, parameter);
    const auto linearization = fdapde::gfe::p1_geodesic_linearization(
      geometry, std::span<const FixedGeometry::Point>(nodes), std::span<const double>(weights), exact_mean,
      accurate_options());
    ASSERT_TRUE(linearization.result().converged());
    EXPECT_LT(geometry.distance(linearization.result().value, exact_mean), 2.0e-10);

    const auto weight_action = linearization.weight_jvp(weight_direction);
    ASSERT_TRUE(weight_action.converged());
    const auto expected_velocity = geometry.transport(nodes[0], exact_mean, geodesic_direction);
    expect_tangent_near(geometry, exact_mean, weight_action.derivative, expected_velocity, 2.0e-8);

    native::Matrix<double, 3, 3> generator;
    generator(0, 0) = 0.2;
    generator(0, 1) = -0.15;
    generator(0, 2) = 0.1;
    generator(1, 0) = 0.3;
    generator(1, 1) = -0.1;
    generator(1, 2) = 0.05;
    generator(2, 0) = -0.2;
    generator(2, 1) = 0.25;
    generator(2, 2) = 0.15;
    const std::vector<FixedGeometry::Tangent> directions {
      congruence_direction<FixedGeometry>(generator, nodes[0]),
      congruence_direction<FixedGeometry>(generator, nodes[1])};
    const auto nodal_action = linearization.nodal_jvp(directions);
    ASSERT_TRUE(nodal_action.converged());
    const auto expected_nodal = congruence_direction<FixedGeometry>(generator, exact_mean);
    expect_tangent_near(geometry, exact_mean, nodal_action.derivative, expected_nodal, 2.0e-8);
}

TEST(AffineInvariantP1Linearization, SatisfiesTheImplicitEquationsAndMetricAdjoint) {
    const FixedGeometry geometry;
    const auto nodes = noncommuting_nodes<FixedGeometry>();
    const std::vector<double> weights {0.25, 0.5, 0.25};
    const std::vector<double> weight_direction {0.125, -0.375, 0.25};
    const auto directions = nodal_directions<FixedGeometry>();
    const auto output = make_tangent<FixedGeometry>({0.4, -0.15, 0.3, 0.2, -0.35, 0.1});
    const auto linearization = fdapde::gfe::p1_geodesic_linearization(
      geometry, std::span<const FixedGeometry::Point>(nodes), std::span<const double>(weights), accurate_options());
    ASSERT_TRUE(linearization.result().converged());
    const auto& mean = linearization.result().value;
    const auto& represented_weights = linearization.result().normalized_weights;
    const double total = compensated_sum(represented_weights);
    std::vector<double> effective_weights(represented_weights);
    for (double& weight : effective_weights) { weight /= total; }

    const auto weight_action = linearization.weight_jvp(weight_direction);
    const auto nodal_action = linearization.nodal_jvp(directions);
    const auto pullback = linearization.nodal_vjp(output);
    ASSERT_TRUE(weight_action.converged());
    ASSERT_TRUE(nodal_action.converged());
    ASSERT_TRUE(pullback.converged());

    const auto expected_weight_rhs = weight_right_hand_side(
      geometry, mean, std::span<const FixedGeometry::Point>(nodes), represented_weights, weight_direction);
    const auto expected_nodal_rhs = nodal_right_hand_side(
      geometry, mean, std::span<const FixedGeometry::Point>(nodes), represented_weights,
      std::span<const FixedGeometry::Tangent>(directions));
    const auto applied_weight = hessian_action(
      geometry, mean, std::span<const FixedGeometry::Point>(nodes), effective_weights, weight_action.derivative);
    const auto applied_nodal = hessian_action(
      geometry, mean, std::span<const FixedGeometry::Point>(nodes), effective_weights, nodal_action.derivative);
    const auto weight_residual = geometry.linear_combination(mean, 1, expected_weight_rhs, -1, applied_weight);
    const auto nodal_residual = geometry.linear_combination(mean, 1, expected_nodal_rhs, -1, applied_nodal);
    const double weight_residual_norm = geometry.norm(mean, weight_residual);
    const double nodal_residual_norm = geometry.norm(mean, nodal_residual);
    EXPECT_LE(weight_residual_norm, 5.0e-9 * (1 + geometry.norm(mean, expected_weight_rhs)));
    EXPECT_LE(nodal_residual_norm, 5.0e-9 * (1 + geometry.norm(mean, expected_nodal_rhs)));
    EXPECT_NEAR(
      weight_action.residual_norm, weight_residual_norm, 5.0e-11 * (1 + geometry.norm(mean, expected_weight_rhs)));
    EXPECT_NEAR(
      nodal_action.residual_norm, nodal_residual_norm, 5.0e-11 * (1 + geometry.norm(mean, expected_nodal_rhs)));

    const double lhs = geometry.inner_product(mean, nodal_action.derivative, output);
    double rhs = 0;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        rhs += geometry.inner_product(nodes[i], directions[i], pullback.derivative[i]);
    }
    EXPECT_NEAR(lhs, rhs, 2.0e-8 * std::max({1.0, std::abs(lhs), std::abs(rhs)}));
}

TEST(AffineInvariantP1Linearization, MatchesCenteredGeometricFiniteDifferences) {
    const FixedGeometry geometry;
    const auto nodes = noncommuting_nodes<FixedGeometry>();
    const std::vector<double> weights {0.25, 0.5, 0.25};
    const std::vector<double> weight_direction {0.125, -0.375, 0.25};
    const auto directions = nodal_directions<FixedGeometry>();
    const auto output = make_tangent<FixedGeometry>({0.4, -0.15, 0.3, 0.2, -0.35, 0.1});
    auto options = accurate_options();
    options.mean.solver.max_iterations = 600;
    options.mean.solver.gradient_tolerance = 5.0e-9;
    const auto linearization = fdapde::gfe::p1_geodesic_linearization(
      geometry, std::span<const FixedGeometry::Point>(nodes), std::span<const double>(weights), options);
    ASSERT_TRUE(linearization.result().converged());
    const auto& mean = linearization.result().value;
    const auto weight_action = linearization.weight_jvp(weight_direction);
    const auto nodal_action = linearization.nodal_jvp(directions);
    const auto pullback = linearization.nodal_vjp(output);
    ASSERT_TRUE(weight_action.converged());
    ASSERT_TRUE(nodal_action.converged());
    ASSERT_TRUE(pullback.converged());

    double pullback_contraction = 0;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        pullback_contraction += geometry.inner_product(nodes[i], directions[i], pullback.derivative[i]);
    }

    double best_weight_error = std::numeric_limits<double>::infinity();
    double best_nodal_error = std::numeric_limits<double>::infinity();
    double best_pullback_error = std::numeric_limits<double>::infinity();
    for (const double step : {3.0e-3, 1.0e-3, 3.0e-4}) {
        std::vector<double> plus_weights(weights);
        std::vector<double> minus_weights(weights);
        std::vector<FixedGeometry::Point> plus_nodes;
        std::vector<FixedGeometry::Point> minus_nodes;
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            plus_weights[i] += step * weight_direction[i];
            minus_weights[i] -= step * weight_direction[i];
            plus_nodes.push_back(geometry.exponential(nodes[i], directions[i], step));
            minus_nodes.push_back(geometry.exponential(nodes[i], directions[i], -step));
        }
        const auto plus_weight_initial = geometry.exponential(mean, weight_action.derivative, step);
        const auto minus_weight_initial = geometry.exponential(mean, weight_action.derivative, -step);
        const auto plus_nodal_initial = geometry.exponential(mean, nodal_action.derivative, step);
        const auto minus_nodal_initial = geometry.exponential(mean, nodal_action.derivative, -step);

        const auto plus_weight = fdapde::gfe::p1_geodesic_value(
          geometry, std::span<const FixedGeometry::Point>(nodes), std::span<const double>(plus_weights),
          plus_weight_initial, options.mean);
        const auto minus_weight = fdapde::gfe::p1_geodesic_value(
          geometry, std::span<const FixedGeometry::Point>(nodes), std::span<const double>(minus_weights),
          minus_weight_initial, options.mean);
        const auto plus_nodal = fdapde::gfe::p1_geodesic_value(
          geometry, std::span<const FixedGeometry::Point>(plus_nodes), std::span<const double>(weights),
          plus_nodal_initial, options.mean);
        const auto minus_nodal = fdapde::gfe::p1_geodesic_value(
          geometry, std::span<const FixedGeometry::Point>(minus_nodes), std::span<const double>(weights),
          minus_nodal_initial, options.mean);
        ASSERT_TRUE(plus_weight.converged())
          << static_cast<int>(plus_weight.stop_reason) << " stationarity=" << plus_weight.stationarity_norm;
        ASSERT_TRUE(minus_weight.converged())
          << static_cast<int>(minus_weight.stop_reason) << " stationarity=" << minus_weight.stationarity_norm;
        ASSERT_TRUE(plus_nodal.converged())
          << static_cast<int>(plus_nodal.stop_reason) << " stationarity=" << plus_nodal.stationarity_norm;
        ASSERT_TRUE(minus_nodal.converged())
          << static_cast<int>(minus_nodal.stop_reason) << " stationarity=" << minus_nodal.stationarity_norm;

        const auto weight_plus_log = geometry.logarithm(mean, plus_weight.value);
        const auto weight_minus_log = geometry.logarithm(mean, minus_weight.value);
        const auto nodal_plus_log = geometry.logarithm(mean, plus_nodal.value);
        const auto nodal_minus_log = geometry.logarithm(mean, minus_nodal.value);
        const auto weight_difference =
          geometry.linear_combination(mean, 1 / (2 * step), weight_plus_log, -1 / (2 * step), weight_minus_log);
        const auto nodal_difference =
          geometry.linear_combination(mean, 1 / (2 * step), nodal_plus_log, -1 / (2 * step), nodal_minus_log);
        const double weight_error = relative_tangent_error(geometry, mean, weight_difference, weight_action.derivative);
        const double nodal_error = relative_tangent_error(geometry, mean, nodal_difference, nodal_action.derivative);
        EXPECT_LT(weight_error, 2.0e-4);
        EXPECT_LT(nodal_error, 2.0e-4);
        best_weight_error = std::min(best_weight_error, weight_error);
        best_nodal_error = std::min(best_nodal_error, nodal_error);

        const double plus_scalar = geometry.inner_product(mean, output, nodal_plus_log);
        const double minus_scalar = geometry.inner_product(mean, output, nodal_minus_log);
        const double finite_difference_pullback = (plus_scalar - minus_scalar) / (2 * step);
        const double pullback_error =
          std::abs(finite_difference_pullback - pullback_contraction) /
          std::max({1.0, std::abs(finite_difference_pullback), std::abs(pullback_contraction)});
        best_pullback_error = std::min(best_pullback_error, pullback_error);
    }
    EXPECT_LT(best_weight_error, 5.0e-6);
    EXPECT_LT(best_nodal_error, 5.0e-6);
    EXPECT_LT(best_pullback_error, 1.0e-5);
}

TEST(AffineInvariantP1Linearization, IsDeterministicPermutationCovariantAndOwnsDynamicSnapshots) {
    const FixedGeometry fixed_geometry;
    const auto fixed_nodes = noncommuting_nodes<FixedGeometry>();
    const std::vector<double> weights {0.25, 0.5, 0.25};
    const std::vector<double> weight_direction {0.125, -0.375, 0.25};
    const auto fixed_directions = nodal_directions<FixedGeometry>();
    const auto fixed_output = make_tangent<FixedGeometry>({0.4, -0.15, 0.3, 0.2, -0.35, 0.1});
    const auto fixed = fdapde::gfe::p1_geodesic_linearization(
      fixed_geometry, std::span<const FixedGeometry::Point>(fixed_nodes), std::span<const double>(weights),
      accurate_options());
    ASSERT_TRUE(fixed.result().converged());

    const auto first_weight = fixed.weight_jvp(weight_direction);
    const auto second_weight = fixed.weight_jvp(weight_direction);
    ASSERT_TRUE(first_weight.converged());
    ASSERT_TRUE(second_weight.converged());
    EXPECT_EQ(first_weight.stop_reason, second_weight.stop_reason);
    EXPECT_EQ(first_weight.iterations, second_weight.iterations);
    EXPECT_DOUBLE_EQ(first_weight.residual_norm, second_weight.residual_norm);
    expect_same_coefficients(first_weight.derivative, second_weight.derivative);
    const auto first_nodal = fixed.nodal_jvp(fixed_directions);
    const auto second_nodal = fixed.nodal_jvp(fixed_directions);
    ASSERT_TRUE(first_nodal.converged());
    ASSERT_TRUE(second_nodal.converged());
    EXPECT_EQ(first_nodal.stop_reason, second_nodal.stop_reason);
    EXPECT_EQ(first_nodal.iterations, second_nodal.iterations);
    EXPECT_DOUBLE_EQ(first_nodal.residual_norm, second_nodal.residual_norm);
    expect_same_coefficients(first_nodal.derivative, second_nodal.derivative);

    const auto dynamic = owning_dynamic_linearization();
    ASSERT_TRUE(dynamic.result().converged());
    const auto dynamic_directions = nodal_directions<DynamicGeometry>();
    const auto dynamic_output = make_tangent<DynamicGeometry>({0.4, -0.15, 0.3, 0.2, -0.35, 0.1});
    const auto dynamic_weight = dynamic.weight_jvp(weight_direction);
    const auto dynamic_nodal = dynamic.nodal_jvp(dynamic_directions);
    const auto dynamic_pullback = dynamic.nodal_vjp(dynamic_output);
    ASSERT_TRUE(dynamic_weight.converged());
    ASSERT_TRUE(dynamic_nodal.converged());
    ASSERT_TRUE(dynamic_pullback.converged());
    expect_tangent_near(
      fixed_geometry, fixed.result().value, dynamic_weight.derivative, first_weight.derivative, 2.0e-8);
    expect_tangent_near(fixed_geometry, fixed.result().value, dynamic_nodal.derivative, first_nodal.derivative, 2.0e-8);
    const auto fixed_pullback = fixed.nodal_vjp(fixed_output);
    ASSERT_TRUE(fixed_pullback.converged());
    for (std::size_t i = 0; i < fixed_nodes.size(); ++i) {
        expect_tangent_near(
          fixed_geometry, fixed_nodes[i], dynamic_pullback.derivative[i], fixed_pullback.derivative[i], 2.0e-8);
    }

    constexpr std::array<std::size_t, 3> permutation {2, 0, 1};
    std::vector<FixedGeometry::Point> permuted_nodes;
    std::vector<FixedGeometry::Tangent> permuted_directions;
    std::vector<double> permuted_weights;
    std::vector<double> permuted_weight_direction;
    for (const std::size_t index : permutation) {
        permuted_nodes.push_back(fixed_nodes[index]);
        permuted_directions.push_back(fixed_directions[index]);
        permuted_weights.push_back(weights[index]);
        permuted_weight_direction.push_back(weight_direction[index]);
    }
    const auto permuted = fdapde::gfe::p1_geodesic_linearization(
      fixed_geometry, std::span<const FixedGeometry::Point>(permuted_nodes), std::span<const double>(permuted_weights),
      accurate_options());
    ASSERT_TRUE(permuted.result().converged());
    EXPECT_LT(fixed_geometry.distance(fixed.result().value, permuted.result().value), 1.0e-7);
    const auto permuted_weight = permuted.weight_jvp(permuted_weight_direction);
    const auto permuted_nodal = permuted.nodal_jvp(permuted_directions);
    ASSERT_TRUE(permuted_weight.converged());
    ASSERT_TRUE(permuted_nodal.converged());
    expect_tangent_near(
      fixed_geometry, fixed.result().value, permuted_weight.derivative, first_weight.derivative, 5.0e-7);
    expect_tangent_near(
      fixed_geometry, fixed.result().value, permuted_nodal.derivative, first_nodal.derivative, 5.0e-7);
    const auto permuted_pullback = permuted.nodal_vjp(fixed_output);
    ASSERT_TRUE(permuted_pullback.converged());
    for (std::size_t i = 0; i < permutation.size(); ++i) {
        expect_tangent_near(
          fixed_geometry, permuted_nodes[i], permuted_pullback.derivative[i], fixed_pullback.derivative[permutation[i]],
          5.0e-7);
    }
}

TEST(AffineInvariantP1Linearization, SurfacesFailuresAndRejectsInvalidDirections) {
    const FixedGeometry geometry;
    auto nodes = noncommuting_nodes<FixedGeometry>();
    const std::vector<double> weights {0.25, 0.5, 0.25};
    const auto directions = nodal_directions<FixedGeometry>();
    const std::vector<double> weight_direction {0.125, -0.375, 0.25};

    auto failed_mean_options = accurate_options();
    failed_mean_options.mean.solver.gradient_tolerance = 0;
    failed_mean_options.mean.solver.line_search.initial_step = 1.0e6;
    failed_mean_options.mean.solver.line_search.max_trials = 1;
    const auto failed_mean = fdapde::gfe::p1_geodesic_linearization(
      geometry, std::span<const FixedGeometry::Point>(nodes), std::span<const double>(weights), failed_mean_options);
    EXPECT_FALSE(failed_mean.result().converged());
    EXPECT_THROW(failed_mean.weight_jvp(weight_direction), std::logic_error);

    auto one_step_options = accurate_options();
    one_step_options.linear_solve.max_iterations = 1;
    one_step_options.linear_solve.residual_tolerance = 0;
    const auto one_step = fdapde::gfe::p1_geodesic_linearization(
      geometry, std::span<const FixedGeometry::Point>(nodes), std::span<const double>(weights), one_step_options);
    const auto one_step_action = one_step.nodal_jvp(directions);
    EXPECT_FALSE(one_step_action.converged());
    EXPECT_EQ(one_step_action.stop_reason, manifold::PositiveDefiniteCGStopReason::max_iterations);
    EXPECT_EQ(one_step_action.iterations, 1);
    EXPECT_GT(one_step_action.residual_norm, 0);

    const auto linearization = fdapde::gfe::p1_geodesic_linearization(
      geometry, std::span<const FixedGeometry::Point>(nodes), std::span<const double>(weights), accurate_options());
    const std::vector<double> too_few_weights {0.5, -0.5};
    const std::vector<double> nonzero_sum {0.2, -0.1, 0};
    const std::vector<double> nonfinite_weight {std::numeric_limits<double>::quiet_NaN(), 0, 0};
    EXPECT_THROW(linearization.weight_jvp(too_few_weights), std::invalid_argument);
    EXPECT_THROW(linearization.weight_jvp(nonzero_sum), std::invalid_argument);
    EXPECT_THROW(linearization.weight_jvp(nonfinite_weight), std::invalid_argument);

    std::vector<FixedGeometry::Tangent> too_few_directions(directions.begin(), directions.begin() + 2);
    auto nonfinite_directions = directions;
    nonfinite_directions[0](0, 0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW(linearization.nodal_jvp(too_few_directions), std::invalid_argument);
    EXPECT_THROW(linearization.nodal_jvp(nonfinite_directions), std::invalid_argument);
    auto nonfinite_output = directions[0];
    nonfinite_output(1, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(linearization.nodal_vjp(nonfinite_output), std::invalid_argument);

    native::Matrix<double, 3, 3> indefinite;
    indefinite.set_zero();
    indefinite(0, 0) = 1;
    indefinite(1, 1) = 1;
    indefinite(2, 2) = -1;
    nodes[2] = FixedGeometry::Point(indefinite, native::unchecked);
    const std::vector<double> zero_last_weight {0.5, 0.5, 0};
    EXPECT_THROW(
      fdapde::gfe::p1_geodesic_linearization(
        geometry, std::span<const FixedGeometry::Point>(nodes), std::span<const double>(zero_last_weight),
        accurate_options()),
      std::domain_error);
    EXPECT_THROW(
      fdapde::gfe::p1_geodesic_linearization(
        geometry, std::span<const FixedGeometry::Point>(nodes), std::span<const double>(zero_last_weight),
        failed_mean_options),
      std::domain_error);

    nodes[2] = make_point<FixedGeometry>(third_coefficients);
    const auto inactive = fdapde::gfe::p1_geodesic_linearization(
      geometry, std::span<const FixedGeometry::Point>(nodes), std::span<const double>(zero_last_weight),
      accurate_options());
    auto inactive_directions = directions;
    inactive_directions[2](0, 0) = std::numeric_limits<double>::quiet_NaN();
    const auto inactive_action = inactive.nodal_jvp(inactive_directions);
    EXPECT_TRUE(inactive_action.converged());
    const auto inactive_pullback = inactive.nodal_vjp(directions[0]);
    ASSERT_TRUE(inactive_pullback.converged());
    expect_tangent_near(geometry, nodes[2], inactive_pullback.derivative[2], geometry.zero_tangent(nodes[2]), 0);

    const DynamicGeometry dynamic_geometry(3);
    const auto dynamic_nodes = noncommuting_nodes<DynamicGeometry>();
    const auto dynamic = fdapde::gfe::p1_geodesic_linearization(
      dynamic_geometry, std::span<const DynamicGeometry::Point>(dynamic_nodes), std::span<const double>(weights),
      accurate_options());
    auto wrong_shape_directions = nodal_directions<DynamicGeometry>();
    wrong_shape_directions[0].resize(2, 2);
    wrong_shape_directions[0](0, 0) = 1;
    wrong_shape_directions[0](1, 0) = 0;
    wrong_shape_directions[0](1, 1) = 1;
    EXPECT_THROW(dynamic.nodal_jvp(wrong_shape_directions), std::invalid_argument);

    auto invalid_options = accurate_options();
    invalid_options.linear_solve.max_iterations = 0;
    EXPECT_THROW(
      fdapde::gfe::p1_geodesic_linearization(
        geometry, std::span<const FixedGeometry::Point>(nodes), std::span<const double>(weights), invalid_options),
      std::invalid_argument);
}
