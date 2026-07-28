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
#include <stdexcept>
#include <vector>

namespace {

namespace native = fdapde::linalg;

using FixedGeometry = fdapde::manifold::LogEuclideanSPDGeometry<double, 3>;
using DynamicGeometry = fdapde::manifold::LogEuclideanSPDGeometry<double, fdapde::Dynamic>;
using FixedFloatGeometry = fdapde::manifold::LogEuclideanSPDGeometry<float, 3>;
using DynamicFloatGeometry = fdapde::manifold::LogEuclideanSPDGeometry<float, fdapde::Dynamic>;

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

template <typename Geometry> typename Geometry::Point make_diagonal_point(double first, double second, double third) {
    native::Matrix<double, 3, 3> dense;
    dense.set_zero();
    dense(0, 0) = first;
    dense(1, 1) = second;
    dense(2, 2) = third;
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

template <typename Geometry> typename Geometry::Tangent zero_tangent() {
    return make_tangent<Geometry>({0, 0, 0, 0, 0, 0});
}

template <typename Geometry, typename Lhs, typename Rhs>
typename Geometry::Tangent combine_tangents(double alpha, const Lhs& lhs, double beta, const Rhs& rhs) {
    auto result = zero_tangent<Geometry>();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j <= i; ++j) {
            result(i, j) = alpha * static_cast<double>(lhs(i, j)) + beta * static_cast<double>(rhs(i, j));
        }
    }
    return result;
}

template <typename MatrixType> double matrix_norm(const MatrixType& matrix) {
    double result = 0;
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) { result = std::hypot(result, static_cast<double>(matrix(i, j))); }
    }
    return result;
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

template <typename Lhs, typename Rhs> double relative_matrix_error(const Lhs& lhs, const Rhs& rhs) {
    return matrix_difference_norm(lhs, rhs) / std::max({1.0, matrix_norm(lhs), matrix_norm(rhs)});
}

template <typename Lhs, typename Rhs> void expect_matrix_near(const Lhs& lhs, const Rhs& rhs, double tolerance) {
    EXPECT_LE(relative_matrix_error(lhs, rhs), tolerance);
}

double compensated_sum(std::span<const double> values) {
    double sum = 0;
    double correction = 0;
    for (const double value : values) {
        if (value == 0) continue;
        const double corrected = value - correction;
        const double next = sum + corrected;
        correction = (next - sum) - corrected;
        sum = next;
    }
    return sum;
}

template <typename Geometry> std::vector<typename Geometry::Point> noncommuting_nodes() {
    return {
      make_point<Geometry>(first_coefficients), make_point<Geometry>(second_coefficients),
      make_point<Geometry>(third_coefficients)};
}

template <typename Geometry> std::vector<typename Geometry::Tangent> first_directions() {
    return {
      make_tangent<Geometry>({0.3, -0.2, 0.4, 0.1, 0.25, -0.15}),
      make_tangent<Geometry>({-0.1, 0.35, 0.2, -0.25, 0.05, 0.45}),
      make_tangent<Geometry>({0.2, 0.1, -0.3, 0.4, -0.15, 0.25})};
}

template <typename Geometry> std::vector<typename Geometry::Tangent> second_directions() {
    return {
      make_tangent<Geometry>({-0.2, 0.4, 0.1, -0.15, 0.3, 0.05}),
      make_tangent<Geometry>({0.25, -0.1, 0.35, 0.2, -0.4, 0.15}),
      make_tangent<Geometry>({0.1, -0.3, 0.2, 0.05, 0.45, -0.25})};
}

template <typename Geometry>
typename Geometry::Tangent
weighted_sum(std::span<const typename Geometry::Tangent> values, std::span<const double> coefficients) {
    auto result = zero_tangent<Geometry>();
    for (std::size_t k = 0; k < values.size(); ++k) {
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j <= i; ++j) {
                result(i, j) =
                  static_cast<double>(result(i, j)) + coefficients[k] * static_cast<double>(values[k](i, j));
            }
        }
    }
    return result;
}

template <typename Geometry>
typename Geometry::Tangent
expected_mean_chart(std::span<const typename Geometry::Point> nodes, std::span<const double> canonical_weights) {
    const double total = compensated_sum(canonical_weights);
    std::vector<typename Geometry::Tangent> logs;
    logs.reserve(nodes.size());
    for (const auto& node : nodes) { logs.emplace_back(native::matrix_log(node)); }
    std::vector<double> effective_weights(canonical_weights.begin(), canonical_weights.end());
    for (double& weight : effective_weights) { weight /= total; }
    return weighted_sum<Geometry>(logs, effective_weights);
}

template <typename Geometry>
typename Geometry::Tangent expected_weight_chart(
  std::span<const typename Geometry::Point> nodes, std::span<const double> canonical_weights,
  std::span<const double> direction) {
    const double total = compensated_sum(canonical_weights);
    const auto mean_chart = expected_mean_chart<Geometry>(nodes, canonical_weights);
    auto result = zero_tangent<Geometry>();
    for (std::size_t k = 0; k < nodes.size(); ++k) {
        const auto node_log = native::matrix_log(nodes[k]);
        for (int i = 0; i < 3; ++i) {
            for (int j = 0; j <= i; ++j) {
                result(i, j) = static_cast<double>(result(i, j)) +
                               direction[k] * (static_cast<double>(node_log(i, j)) - mean_chart(i, j)) / total;
            }
        }
    }
    return result;
}

template <typename Geometry>
typename Geometry::Tangent expected_nodal_chart(
  std::span<const typename Geometry::Point> nodes, std::span<const double> canonical_weights,
  std::span<const typename Geometry::Tangent> directions) {
    const double total = compensated_sum(canonical_weights);
    std::vector<typename Geometry::Tangent> chart_directions;
    chart_directions.reserve(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        chart_directions.emplace_back(native::matrix_log_frechet(nodes[i], directions[i]));
    }
    std::vector<double> effective_weights(canonical_weights.begin(), canonical_weights.end());
    for (double& weight : effective_weights) { weight /= total; }
    return weighted_sum<Geometry>(chart_directions, effective_weights);
}

template <typename Geometry>
typename Geometry::Point
perturb_point(const typename Geometry::Point& point, const typename Geometry::Tangent& direction, double step) {
    native::Matrix<double, 3, 3> dense;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            dense(i, j) = static_cast<double>(point(i, j)) + step * static_cast<double>(direction(i, j));
        }
    }
    return typename Geometry::Point(dense, native::checked);
}

template <typename Geometry>
typename Geometry::Tangent
centered_difference(const typename Geometry::Point& plus, const typename Geometry::Point& minus, double step) {
    auto result = zero_tangent<Geometry>();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j <= i; ++j) {
            result(i, j) = (static_cast<double>(plus(i, j)) - static_cast<double>(minus(i, j))) / (2 * step);
        }
    }
    return result;
}

auto owning_dynamic_linearization() {
    const DynamicGeometry geometry(3);
    auto nodes = noncommuting_nodes<DynamicGeometry>();
    std::vector<double> weights {0.25, 0.5, 0.25};
    auto linearization = fdapde::gfe::p1_geodesic_linearization(
      geometry, std::span<const DynamicGeometry::Point>(nodes), std::span<const double>(weights));

    nodes[0] = make_diagonal_point<DynamicGeometry>(7, 7, 7);
    weights = {1, 0, 0};
    return linearization;
}

}   // namespace

TEST(LogEuclideanP1Linearization, MatchesExactNoncommutingChartActionsAndPermutations) {
    const FixedGeometry geometry;
    const auto nodes = noncommuting_nodes<FixedGeometry>();
    const std::vector<double> weights {0.25, 0.5, 0.25};
    const std::vector<double> weight_direction {0.125, -0.375, 0.25};
    const auto nodal_directions = first_directions<FixedGeometry>();
    const auto output_gradient = make_tangent<FixedGeometry>({0.4, -0.15, 0.3, 0.2, -0.35, 0.1});
    const auto linearization = fdapde::gfe::p1_geodesic_linearization(
      geometry, std::span<const FixedGeometry::Point>(nodes), std::span<const double>(weights));

    const auto weight_jvp = linearization.weight_jvp(weight_direction);
    const auto nodal_jvp = linearization.nodal_jvp(nodal_directions);
    const auto nodal_vjp = linearization.nodal_vjp(output_gradient);
    const auto& result = linearization.result();
    const auto expected_weight =
      expected_weight_chart<FixedGeometry>(nodes, result.normalized_weights, weight_direction);
    const auto expected_nodal = expected_nodal_chart<FixedGeometry>(nodes, result.normalized_weights, nodal_directions);
    expect_matrix_near(native::matrix_log_frechet(result.value, weight_jvp), expected_weight, 2.0e-9);
    expect_matrix_near(native::matrix_log_frechet(result.value, nodal_jvp), expected_nodal, 2.0e-9);

    const double total = compensated_sum(result.normalized_weights);
    const auto output_chart = native::matrix_log_frechet(result.value, output_gradient);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        const auto expected =
          combine_tangents<FixedGeometry>(result.normalized_weights[i] / total, output_chart, 0, output_chart);
        expect_matrix_near(native::matrix_log_frechet(nodes[i], nodal_vjp[i]), expected, 2.0e-9);
    }

    expect_matrix_near(linearization.weight_jvp(weight_direction), weight_jvp, 0);
    expect_matrix_near(linearization.nodal_jvp(nodal_directions), nodal_jvp, 0);

    constexpr std::array<std::size_t, 3> permutation {2, 0, 1};
    std::vector<FixedGeometry::Point> permuted_nodes;
    std::vector<FixedGeometry::Tangent> permuted_directions;
    std::vector<double> permuted_weights;
    std::vector<double> permuted_weight_direction;
    for (const std::size_t index : permutation) {
        permuted_nodes.push_back(nodes[index]);
        permuted_directions.push_back(nodal_directions[index]);
        permuted_weights.push_back(weights[index]);
        permuted_weight_direction.push_back(weight_direction[index]);
    }
    const auto permuted = fdapde::gfe::p1_geodesic_linearization(
      geometry, std::span<const FixedGeometry::Point>(permuted_nodes), std::span<const double>(permuted_weights));
    expect_matrix_near(permuted.weight_jvp(permuted_weight_direction), weight_jvp, 5.0e-10);
    expect_matrix_near(permuted.nodal_jvp(permuted_directions), nodal_jvp, 5.0e-10);
    const auto permuted_vjp = permuted.nodal_vjp(output_gradient);
    for (std::size_t i = 0; i < permutation.size(); ++i) {
        expect_matrix_near(permuted_vjp[i], nodal_vjp[permutation[i]], 5.0e-10);
    }
}

TEST(LogEuclideanP1Linearization, HasExactVertexAndSingleNodeActions) {
    const FixedGeometry geometry;
    const auto nodes = noncommuting_nodes<FixedGeometry>();
    const std::vector<double> weights {0, 1, 0};
    const std::vector<double> weight_direction {0.25, -0.5, 0.25};
    const auto nodal_directions = first_directions<FixedGeometry>();
    const auto output_gradient = make_tangent<FixedGeometry>({0.4, -0.15, 0.3, 0.2, -0.35, 0.1});
    const auto linearization = fdapde::gfe::p1_geodesic_linearization(
      geometry, std::span<const FixedGeometry::Point>(nodes), std::span<const double>(weights));

    expect_matrix_near(linearization.nodal_jvp(nodal_directions), nodal_directions[1], 0);
    const auto pullback = linearization.nodal_vjp(output_gradient);
    expect_matrix_near(pullback[0], zero_tangent<FixedGeometry>(), 0);
    expect_matrix_near(pullback[1], output_gradient, 0);
    expect_matrix_near(pullback[2], zero_tangent<FixedGeometry>(), 0);
    const auto expected_weight =
      expected_weight_chart<FixedGeometry>(nodes, linearization.result().normalized_weights, weight_direction);
    expect_matrix_near(
      native::matrix_log_frechet(linearization.result().value, linearization.weight_jvp(weight_direction)),
      expected_weight, 2.0e-9);

    const std::vector<FixedGeometry::Point> one_node {nodes[1]};
    const std::vector<double> one_weight {1};
    const std::vector<double> zero_weight_direction {0};
    const std::vector<FixedGeometry::Tangent> one_direction {nodal_directions[1]};
    const auto one = fdapde::gfe::p1_geodesic_linearization(
      geometry, std::span<const FixedGeometry::Point>(one_node), std::span<const double>(one_weight));
    expect_matrix_near(one.weight_jvp(zero_weight_direction), zero_tangent<FixedGeometry>(), 0);
    expect_matrix_near(one.nodal_jvp(one_direction), nodal_directions[1], 0);
    expect_matrix_near(one.nodal_vjp(output_gradient)[0], output_gradient, 0);
}

TEST(LogEuclideanP1Linearization, IsLinearAndUsesTheRiemannianMetricAdjoint) {
    const FixedGeometry geometry;
    const auto nodes = noncommuting_nodes<FixedGeometry>();
    const std::vector<double> weights {0.25, 0.5, 0.25};
    const std::vector<double> first_weight_direction {0.2, -0.35, 0.15};
    const std::vector<double> second_weight_direction {-0.1, 0.4, -0.3};
    std::vector<double> combined_weight_direction(3);
    for (std::size_t i = 0; i < combined_weight_direction.size(); ++i) {
        combined_weight_direction[i] = 0.3 * first_weight_direction[i] - 0.7 * second_weight_direction[i];
    }
    const auto first_nodal = first_directions<FixedGeometry>();
    const auto second_nodal = second_directions<FixedGeometry>();
    std::vector<FixedGeometry::Tangent> combined_nodal;
    for (std::size_t i = 0; i < first_nodal.size(); ++i) {
        combined_nodal.push_back(combine_tangents<FixedGeometry>(0.3, first_nodal[i], -0.7, second_nodal[i]));
    }
    const auto first_output = make_tangent<FixedGeometry>({0.4, -0.15, 0.3, 0.2, -0.35, 0.1});
    const auto second_output = make_tangent<FixedGeometry>({-0.2, 0.25, 0.1, -0.3, 0.15, 0.45});
    const auto combined_output = combine_tangents<FixedGeometry>(0.3, first_output, -0.7, second_output);
    const auto linearization = fdapde::gfe::p1_geodesic_linearization(
      geometry, std::span<const FixedGeometry::Point>(nodes), std::span<const double>(weights));

    expect_matrix_near(
      linearization.weight_jvp(combined_weight_direction),
      combine_tangents<FixedGeometry>(
        0.3, linearization.weight_jvp(first_weight_direction), -0.7, linearization.weight_jvp(second_weight_direction)),
      5.0e-10);
    expect_matrix_near(
      linearization.nodal_jvp(combined_nodal),
      combine_tangents<FixedGeometry>(
        0.3, linearization.nodal_jvp(first_nodal), -0.7, linearization.nodal_jvp(second_nodal)),
      5.0e-10);

    const auto first_pullback = linearization.nodal_vjp(first_output);
    const auto second_pullback = linearization.nodal_vjp(second_output);
    const auto combined_pullback = linearization.nodal_vjp(combined_output);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        expect_matrix_near(
          combined_pullback[i], combine_tangents<FixedGeometry>(0.3, first_pullback[i], -0.7, second_pullback[i]),
          5.0e-10);
    }

    const auto nodal_action = linearization.nodal_jvp(first_nodal);
    const auto pullback = linearization.nodal_vjp(first_output);
    const double lhs = geometry.inner_product(linearization.result().value, nodal_action, first_output);
    double rhs = 0;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        rhs += geometry.inner_product(nodes[i], first_nodal[i], pullback[i]);
    }
    EXPECT_NEAR(lhs, rhs, 2.0e-9 * std::max({1.0, std::abs(lhs), std::abs(rhs)}));
}

TEST(LogEuclideanP1Linearization, FixedAndDynamicOwningSnapshotsAgree) {
    const FixedGeometry fixed_geometry;
    const auto fixed_nodes = noncommuting_nodes<FixedGeometry>();
    const std::vector<double> weights {0.25, 0.5, 0.25};
    const std::vector<double> weight_direction {0.125, -0.375, 0.25};
    const auto fixed_directions = first_directions<FixedGeometry>();
    const auto fixed_output = make_tangent<FixedGeometry>({0.4, -0.15, 0.3, 0.2, -0.35, 0.1});
    const auto fixed = fdapde::gfe::p1_geodesic_linearization(
      fixed_geometry, std::span<const FixedGeometry::Point>(fixed_nodes), std::span<const double>(weights));

    const auto dynamic = owning_dynamic_linearization();
    const auto dynamic_directions = first_directions<DynamicGeometry>();
    const auto dynamic_output = make_tangent<DynamicGeometry>({0.4, -0.15, 0.3, 0.2, -0.35, 0.1});
    expect_matrix_near(dynamic.result().value, fixed.result().value, 5.0e-10);
    expect_matrix_near(dynamic.weight_jvp(weight_direction), fixed.weight_jvp(weight_direction), 5.0e-10);
    expect_matrix_near(dynamic.nodal_jvp(dynamic_directions), fixed.nodal_jvp(fixed_directions), 5.0e-10);
    const auto dynamic_pullback = dynamic.nodal_vjp(dynamic_output);
    const auto fixed_pullback = fixed.nodal_vjp(fixed_output);
    for (std::size_t i = 0; i < fixed_pullback.size(); ++i) {
        expect_matrix_near(dynamic_pullback[i], fixed_pullback[i], 5.0e-10);
    }
}

TEST(LogEuclideanP1Linearization, UsesTheRepresentedCanonicalWeightTotal) {
    const FixedGeometry geometry;
    auto nodes = noncommuting_nodes<FixedGeometry>();
    nodes.push_back(make_diagonal_point<FixedGeometry>(3, 3, 3));
    const std::vector<double> weights {0.1, 0, 0.3, 0.6};
    const std::vector<double> weight_direction {0.2, 0.1, -0.15, -0.15};
    auto nodal_directions = first_directions<FixedGeometry>();
    nodal_directions.push_back(make_tangent<FixedGeometry>({0.1, -0.2, 0.3, -0.1, 0.2, -0.3}));
    const auto linearization = fdapde::gfe::p1_geodesic_linearization(
      geometry, std::span<const FixedGeometry::Point>(nodes), std::span<const double>(weights));

    const double represented_total = compensated_sum(linearization.result().normalized_weights);
    EXPECT_NE(represented_total, 1.0);
    const auto expected_weight =
      expected_weight_chart<FixedGeometry>(nodes, linearization.result().normalized_weights, weight_direction);
    const auto expected_nodal =
      expected_nodal_chart<FixedGeometry>(nodes, linearization.result().normalized_weights, nodal_directions);
    expect_matrix_near(
      native::matrix_log_frechet(linearization.result().value, linearization.weight_jvp(weight_direction)),
      expected_weight, 2.0e-9);
    expect_matrix_near(
      native::matrix_log_frechet(linearization.result().value, linearization.nodal_jvp(nodal_directions)),
      expected_nodal, 2.0e-9);
}

TEST(LogEuclideanP1Linearization, SupportsFixedAndDynamicFloatSnapshots) {
    const FixedFloatGeometry fixed_geometry;
    const DynamicFloatGeometry dynamic_geometry(3);
    const std::vector<FixedFloatGeometry::Point> fixed_nodes {
      make_diagonal_point<FixedFloatGeometry>(2, 2, 2), make_diagonal_point<FixedFloatGeometry>(5, 5, 5)};
    const std::vector<DynamicFloatGeometry::Point> dynamic_nodes {
      make_diagonal_point<DynamicFloatGeometry>(2, 2, 2), make_diagonal_point<DynamicFloatGeometry>(5, 5, 5)};
    const std::vector<double> weights {0.4, 0.6};
    const std::vector<double> weight_direction {0.25, -0.25};
    const std::vector<FixedFloatGeometry::Tangent> fixed_directions {
      make_tangent<FixedFloatGeometry>({0.3, -0.2, 0.4, 0.1, 0.25, -0.15}),
      make_tangent<FixedFloatGeometry>({-0.1, 0.35, 0.2, -0.25, 0.05, 0.45})};
    const std::vector<DynamicFloatGeometry::Tangent> dynamic_directions {
      make_tangent<DynamicFloatGeometry>({0.3, -0.2, 0.4, 0.1, 0.25, -0.15}),
      make_tangent<DynamicFloatGeometry>({-0.1, 0.35, 0.2, -0.25, 0.05, 0.45})};
    const auto fixed = fdapde::gfe::p1_geodesic_linearization(
      fixed_geometry, std::span<const FixedFloatGeometry::Point>(fixed_nodes), std::span<const double>(weights));
    const auto dynamic = fdapde::gfe::p1_geodesic_linearization(
      dynamic_geometry, std::span<const DynamicFloatGeometry::Point>(dynamic_nodes), std::span<const double>(weights));

    expect_matrix_near(fixed.result().value, dynamic.result().value, 2.0e-5);
    expect_matrix_near(fixed.weight_jvp(weight_direction), dynamic.weight_jvp(weight_direction), 2.0e-5);
    expect_matrix_near(fixed.nodal_jvp(fixed_directions), dynamic.nodal_jvp(dynamic_directions), 2.0e-5);
    const auto fixed_pullback = fixed.nodal_vjp(fixed_directions[0]);
    const auto dynamic_pullback = dynamic.nodal_vjp(dynamic_directions[0]);
    for (std::size_t i = 0; i < fixed_pullback.size(); ++i) {
        expect_matrix_near(fixed_pullback[i], dynamic_pullback[i], 2.0e-5);
    }
}

TEST(LogEuclideanP1Linearization, HandlesRepeatedAndCloseSpectra) {
    const FixedGeometry geometry;
    const std::vector<FixedGeometry::Point> repeated_nodes {
      make_diagonal_point<FixedGeometry>(2, 2, 2), make_diagonal_point<FixedGeometry>(5, 5, 5)};
    const std::vector<double> weights {0.4, 0.6};
    const std::vector<double> weight_direction {0.25, -0.25};
    const std::vector<FixedGeometry::Tangent> directions {
      make_tangent<FixedGeometry>({0.3, -0.2, 0.4, 0.1, 0.25, -0.15}),
      make_tangent<FixedGeometry>({-0.1, 0.35, 0.2, -0.25, 0.05, 0.45})};
    const auto repeated = fdapde::gfe::p1_geodesic_linearization(
      geometry, std::span<const FixedGeometry::Point>(repeated_nodes), std::span<const double>(weights));
    const double mean = std::pow(2.0, 0.4) * std::pow(5.0, 0.6);
    auto expected_nodal = zero_tangent<FixedGeometry>();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j <= i; ++j) {
            expected_nodal(i, j) = mean * (0.4 * static_cast<double>(directions[0](i, j)) / 2.0 +
                                           0.6 * static_cast<double>(directions[1](i, j)) / 5.0);
        }
    }
    expect_matrix_near(repeated.nodal_jvp(directions), expected_nodal, 2.0e-11);
    auto expected_weight = zero_tangent<FixedGeometry>();
    for (int i = 0; i < 3; ++i) { expected_weight(i, i) = 0.25 * mean * (std::log(2.0) - std::log(5.0)); }
    expect_matrix_near(repeated.weight_jvp(weight_direction), expected_weight, 2.0e-11);

    constexpr double gap = 1.0e-12;
    const std::vector<FixedGeometry::Point> close_nodes {
      make_diagonal_point<FixedGeometry>(2, 2 + gap, 5), make_diagonal_point<FixedGeometry>(3, 3 + gap, 4)};
    const auto off_diagonal = make_tangent<FixedGeometry>({0, 1, 0, 0, 0, 0});
    const std::vector<FixedGeometry::Tangent> close_directions {off_diagonal, off_diagonal};
    const auto close = fdapde::gfe::p1_geodesic_linearization(
      geometry, std::span<const FixedGeometry::Point>(close_nodes), std::span<const double>(weights));
    const double mean_log_first = 0.4 * std::log(2.0) + 0.6 * std::log(3.0);
    const double mean_log_second = 0.4 * std::log(2.0 + gap) + 0.6 * std::log(3.0 + gap);
    const double mean_gap = mean_log_second - mean_log_first;
    const double exp_divided_difference = std::exp(mean_log_first) * std::expm1(mean_gap) / mean_gap;
    const double chart_direction = 0.4 * std::log1p(gap / 2.0) / gap + 0.6 * std::log1p(gap / 3.0) / gap;
    const auto close_action = close.nodal_jvp(close_directions);
    EXPECT_NEAR(close_action(1, 0), exp_divided_difference * chart_direction, 2.0e-9);
}

TEST(LogEuclideanP1Linearization, MatchesCenteredFiniteDifferencesAndRejectsInvalidDirections) {
    const FixedGeometry geometry;
    const auto nodes = noncommuting_nodes<FixedGeometry>();
    const std::vector<double> weights {0.25, 0.5, 0.25};
    const std::vector<double> weight_direction {0.125, -0.375, 0.25};
    const auto nodal_directions = first_directions<FixedGeometry>();
    const auto linearization = fdapde::gfe::p1_geodesic_linearization(
      geometry, std::span<const FixedGeometry::Point>(nodes), std::span<const double>(weights));
    const auto exact_weight = linearization.weight_jvp(weight_direction);
    const auto exact_nodal = linearization.nodal_jvp(nodal_directions);

    double best_weight_error = std::numeric_limits<double>::infinity();
    double best_nodal_error = std::numeric_limits<double>::infinity();
    for (const double step : {1.0e-3, 3.0e-4, 1.0e-4}) {
        std::vector<double> plus_weights(weights);
        std::vector<double> minus_weights(weights);
        std::vector<FixedGeometry::Point> plus_nodes;
        std::vector<FixedGeometry::Point> minus_nodes;
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            plus_weights[i] += step * weight_direction[i];
            minus_weights[i] -= step * weight_direction[i];
            plus_nodes.push_back(perturb_point<FixedGeometry>(nodes[i], nodal_directions[i], step));
            minus_nodes.push_back(perturb_point<FixedGeometry>(nodes[i], nodal_directions[i], -step));
        }
        const auto plus_weight_value = fdapde::gfe::p1_geodesic_value(
          geometry, std::span<const FixedGeometry::Point>(nodes), std::span<const double>(plus_weights));
        const auto minus_weight_value = fdapde::gfe::p1_geodesic_value(
          geometry, std::span<const FixedGeometry::Point>(nodes), std::span<const double>(minus_weights));
        const auto plus_nodal_value = fdapde::gfe::p1_geodesic_value(
          geometry, std::span<const FixedGeometry::Point>(plus_nodes), std::span<const double>(weights));
        const auto minus_nodal_value = fdapde::gfe::p1_geodesic_value(
          geometry, std::span<const FixedGeometry::Point>(minus_nodes), std::span<const double>(weights));
        const auto weight_finite_difference =
          centered_difference<FixedGeometry>(plus_weight_value.value, minus_weight_value.value, step);
        const auto nodal_finite_difference =
          centered_difference<FixedGeometry>(plus_nodal_value.value, minus_nodal_value.value, step);
        const double weight_error = relative_matrix_error(weight_finite_difference, exact_weight);
        const double nodal_error = relative_matrix_error(nodal_finite_difference, exact_nodal);
        EXPECT_LT(weight_error, 2.0e-5);
        EXPECT_LT(nodal_error, 2.0e-5);
        best_weight_error = std::min(best_weight_error, weight_error);
        best_nodal_error = std::min(best_nodal_error, nodal_error);
    }
    EXPECT_LT(best_weight_error, 5.0e-7);
    EXPECT_LT(best_nodal_error, 5.0e-7);

    const std::vector<double> too_few_weights {0, 0};
    const std::vector<double> nonzero_sum {0.2, -0.1, 0};
    const std::vector<double> nonfinite_weight {std::numeric_limits<double>::quiet_NaN(), 0, 0};
    EXPECT_THROW(linearization.weight_jvp(too_few_weights), std::invalid_argument);
    EXPECT_THROW(linearization.weight_jvp(nonzero_sum), std::invalid_argument);
    EXPECT_THROW(linearization.weight_jvp(nonfinite_weight), std::invalid_argument);

    std::vector<FixedGeometry::Tangent> too_few_nodal(nodal_directions.begin(), nodal_directions.begin() + 2);
    auto nonfinite_tangent = nodal_directions;
    nonfinite_tangent[0](0, 0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW(linearization.nodal_jvp(too_few_nodal), std::invalid_argument);
    EXPECT_THROW(linearization.nodal_jvp(nonfinite_tangent), std::invalid_argument);
    auto nonfinite_output = nodal_directions[0];
    nonfinite_output(1, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(linearization.nodal_vjp(nonfinite_output), std::invalid_argument);

    const DynamicGeometry dynamic_geometry(3);
    const auto dynamic_nodes = noncommuting_nodes<DynamicGeometry>();
    const auto dynamic = fdapde::gfe::p1_geodesic_linearization(
      dynamic_geometry, std::span<const DynamicGeometry::Point>(dynamic_nodes), std::span<const double>(weights));
    auto dynamic_directions = first_directions<DynamicGeometry>();
    dynamic_directions[0].resize(2, 2);
    dynamic_directions[0](0, 0) = 1;
    dynamic_directions[0](1, 0) = 0;
    dynamic_directions[0](1, 1) = 1;
    EXPECT_THROW(dynamic.nodal_jvp(dynamic_directions), std::invalid_argument);
}

TEST(LogEuclideanP1Linearization, ValidatesEveryNodeWhileNodalActionsSkipZeroWeightDirections) {
    const FixedGeometry geometry;
    auto nodes = noncommuting_nodes<FixedGeometry>();
    native::Matrix<double, 3, 3> indefinite;
    indefinite.set_zero();
    indefinite(0, 0) = 1;
    indefinite(1, 1) = 1;
    indefinite(2, 2) = -1;
    nodes[2] = FixedGeometry::Point(indefinite, native::unchecked);
    const std::vector<double> weights {0.5, 0.5, 0};
    EXPECT_NO_THROW(
      static_cast<void>(fdapde::gfe::p1_geodesic_value(
        geometry, std::span<const FixedGeometry::Point>(nodes), std::span<const double>(weights))));
    EXPECT_THROW(
      fdapde::gfe::p1_geodesic_linearization(
        geometry, std::span<const FixedGeometry::Point>(nodes), std::span<const double>(weights)),
      std::domain_error);

    nodes[2] = make_point<FixedGeometry>(third_coefficients);
    const auto linearization = fdapde::gfe::p1_geodesic_linearization(
      geometry, std::span<const FixedGeometry::Point>(nodes), std::span<const double>(weights));
    auto nodal_directions = first_directions<FixedGeometry>();
    nodal_directions[2](0, 0) = std::numeric_limits<double>::quiet_NaN();
    EXPECT_NO_THROW(static_cast<void>(linearization.nodal_jvp(nodal_directions)));
    const auto pullback = linearization.nodal_vjp(nodal_directions[0]);
    expect_matrix_near(pullback[2], zero_tangent<FixedGeometry>(), 0);

    const std::vector<double> inactive_direction {0.1, -0.1, 0};
    EXPECT_NO_THROW(static_cast<void>(linearization.weight_jvp(inactive_direction)));
    const std::vector<double> activating_direction {0.1, -0.2, 0.1};
    EXPECT_NO_THROW(static_cast<void>(linearization.weight_jvp(activating_direction)));
}
