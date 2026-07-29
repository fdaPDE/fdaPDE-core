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
#include <cstddef>
#include <initializer_list>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace {

namespace gfe = fdapde::gfe;
namespace manifold = fdapde::manifold;
namespace native = fdapde::linalg;

using FixedLogGeometry2 = manifold::LogEuclideanSPDGeometry<double, 2>;
using FixedLogGeometry3 = manifold::LogEuclideanSPDGeometry<double, 3>;
using DynamicLogGeometry = manifold::LogEuclideanSPDGeometry<double, fdapde::Dynamic>;

template <typename Geometry> Geometry make_geometry(int order) {
    if constexpr (Geometry::Point::Rows == fdapde::Dynamic) {
        return Geometry(order);
    } else {
        if (order != Geometry::Point::Rows) { throw std::logic_error("fixed geometry order mismatch"); }
        return Geometry {};
    }
}

template <typename Geometry>
typename Geometry::Point make_point(const Geometry& geometry, std::initializer_list<double> coefficients) {
    if (coefficients.size() != static_cast<std::size_t>(geometry.order() * geometry.order())) {
        throw std::logic_error("point coefficient count mismatch");
    }
    native::Matrix<typename Geometry::Scalar, Geometry::Point::Rows, Geometry::Point::Cols> dense;
    if constexpr (Geometry::Point::Rows == fdapde::Dynamic) { dense.resize(geometry.order(), geometry.order()); }
    auto coefficient = coefficients.begin();
    for (int row = 0; row < geometry.order(); ++row) {
        for (int col = 0; col < geometry.order(); ++col) {
            dense(row, col) = static_cast<typename Geometry::Scalar>(*coefficient++);
        }
    }
    return typename Geometry::Point(dense, native::checked);
}

template <typename Geometry>
typename Geometry::Tangent make_tangent(const Geometry& geometry, std::initializer_list<double> coefficients) {
    if (coefficients.size() != static_cast<std::size_t>(geometry.order() * (geometry.order() + 1) / 2)) {
        throw std::logic_error("tangent coefficient count mismatch");
    }
    typename Geometry::Tangent tangent;
    if constexpr (Geometry::Tangent::Rows == fdapde::Dynamic) { tangent.resize(geometry.order(), geometry.order()); }
    auto coefficient = coefficients.begin();
    for (int row = 0; row < geometry.order(); ++row) {
        for (int col = 0; col <= row; ++col) {
            tangent(row, col) = static_cast<typename Geometry::Scalar>(*coefficient++);
        }
    }
    return tangent;
}

template <typename Geometry> std::vector<typename Geometry::Point> noncommuting_nodes(const Geometry& geometry) {
    if (geometry.order() == 2) {
        return {
          make_point(geometry, {4.0, 0.6, 0.6, 2.5}), make_point(geometry, {1.8, -0.25, -0.25, 3.3}),
          make_point(geometry, {2.6, 0.35, 0.35, 1.4})};
    }
    return {
      make_point(geometry, {4.0, 0.6, 0.2, 0.6, 2.5, -0.3, 0.2, -0.3, 1.7}),
      make_point(geometry, {1.8, -0.25, 0.15, -0.25, 3.3, 0.4, 0.15, 0.4, 2.2}),
      make_point(geometry, {2.6, 0.35, -0.2, 0.35, 1.4, 0.1, -0.2, 0.1, 4.1})};
}

template <typename Geometry> std::vector<typename Geometry::Tangent> nodal_directions(const Geometry& geometry) {
    if (geometry.order() == 2) {
        return {
          make_tangent(geometry, {0.3, -0.2, 0.4}), make_tangent(geometry, {-0.1, 0.35, 0.2}),
          make_tangent(geometry, {0.2, 0.1, -0.3})};
    }
    return {
      make_tangent(geometry, {0.3, -0.2, 0.4, 0.1, 0.25, -0.15}),
      make_tangent(geometry, {-0.1, 0.35, 0.2, -0.25, 0.05, 0.45}),
      make_tangent(geometry, {0.2, 0.1, -0.3, 0.4, -0.15, 0.25})};
}

gfe::P1LumpedLaplacianStencil three_node_stencil() {
    return {
      {0.7,          1.4,          0.9         },
      {{0, 1, -1.2}, {0, 2, 0.35}, {1, 2, -0.8}}
    };
}

template <typename Geometry>
void expect_tangent_near(
  const Geometry& geometry, const typename Geometry::Point& point, const typename Geometry::Tangent& actual,
  const typename Geometry::Tangent& expected, double tolerance) {
    const auto difference = geometry.linear_combination(point, 1, actual, -1, expected);
    const double scale = std::max({1.0, geometry.norm(point, actual), geometry.norm(point, expected)});
    EXPECT_LE(geometry.norm(point, difference), tolerance * scale);
}

template <typename Geometry>
std::vector<typename Geometry::Point> perturb_nodes(
  const Geometry& geometry, std::span<const typename Geometry::Point> nodes,
  std::span<const typename Geometry::Tangent> directions, double step) {
    std::vector<typename Geometry::Point> result;
    result.reserve(nodes.size());
    for (std::size_t node = 0; node < nodes.size(); ++node) {
        result.push_back(geometry.exponential(nodes[node], directions[node], step));
    }
    return result;
}

template <typename Geometry> void expect_independent_chart_oracle(int order) {
    const Geometry geometry = make_geometry<Geometry>(order);
    const auto nodes = noncommuting_nodes(geometry);
    const auto stencil = three_node_stencil();
    const auto value =
      gfe::p1_discrete_tension_value(geometry, std::span<const typename Geometry::Point>(nodes), stencil);
    const auto result =
      gfe::p1_discrete_tension_contribution(geometry, std::span<const typename Geometry::Point>(nodes), stencil);

    std::vector<typename Geometry::Tangent> logarithms;
    std::vector<typename Geometry::Tangent> residuals;
    std::vector<typename Geometry::Tangent> inverse_mass_residuals;
    std::vector<typename Geometry::Tangent> chart_gradient;
    for (const auto& node : nodes) {
        logarithms.emplace_back(native::matrix_log(node));
        residuals.push_back(geometry.zero_tangent(node));
        inverse_mass_residuals.push_back(geometry.zero_tangent(node));
        chart_gradient.push_back(geometry.zero_tangent(node));
    }
    for (const auto& edge : stencil.edges) {
        for (int row = 0; row < order; ++row) {
            for (int col = 0; col <= row; ++col) {
                const double difference =
                  static_cast<double>(logarithms[edge.second](row, col) - logarithms[edge.first](row, col));
                residuals[edge.first](row, col) =
                  static_cast<typename Geometry::Scalar>(residuals[edge.first](row, col) + edge.stiffness * difference);
                residuals[edge.second](row, col) = static_cast<typename Geometry::Scalar>(
                  residuals[edge.second](row, col) - edge.stiffness * difference);
            }
        }
    }

    double expected_value = 0;
    for (std::size_t node = 0; node < nodes.size(); ++node) {
        double squared_norm = 0;
        for (int row = 0; row < order; ++row) {
            for (int col = 0; col <= row; ++col) {
                const double coefficient = static_cast<double>(residuals[node](row, col));
                squared_norm += (row == col ? 1 : 2) * coefficient * coefficient;
                inverse_mass_residuals[node](row, col) =
                  static_cast<typename Geometry::Scalar>(coefficient / stencil.lumped_masses[node]);
            }
        }
        expected_value += 0.5 * squared_norm / stencil.lumped_masses[node];
    }
    for (const auto& edge : stencil.edges) {
        for (int row = 0; row < order; ++row) {
            for (int col = 0; col <= row; ++col) {
                const double difference = static_cast<double>(
                  inverse_mass_residuals[edge.second](row, col) - inverse_mass_residuals[edge.first](row, col));
                chart_gradient[edge.first](row, col) = static_cast<typename Geometry::Scalar>(
                  chart_gradient[edge.first](row, col) + edge.stiffness * difference);
                chart_gradient[edge.second](row, col) = static_cast<typename Geometry::Scalar>(
                  chart_gradient[edge.second](row, col) - edge.stiffness * difference);
            }
        }
    }

    ASSERT_TRUE(value.converged());
    ASSERT_TRUE(result.converged());
    ASSERT_EQ(result.nodal_gradient.size(), nodes.size());
    EXPECT_NEAR(value.value, expected_value, 2.0e-12 * std::max(1.0, expected_value));
    EXPECT_DOUBLE_EQ(result.value, value.value);
    for (std::size_t node = 0; node < nodes.size(); ++node) {
        const typename Geometry::Tangent expected = native::matrix_exp_frechet(logarithms[node], chart_gradient[node]);
        expect_tangent_near(geometry, nodes[node], result.nodal_gradient[node], expected, 3.0e-11);
    }
}

}   // namespace

TEST(P1DiscreteTension, LogEuclideanMatchesIndependentChartOracleForFixedAndDynamicSPD2AndSPD3) {
    expect_independent_chart_oracle<FixedLogGeometry2>(2);
    expect_independent_chart_oracle<DynamicLogGeometry>(2);
    expect_independent_chart_oracle<FixedLogGeometry3>(3);
    expect_independent_chart_oracle<DynamicLogGeometry>(3);
}

TEST(P1DiscreteTension, LogEuclideanGradientMatchesCenteredGeodesicDirectionalDifferences) {
    const FixedLogGeometry3 geometry;
    const auto nodes = noncommuting_nodes(geometry);
    const auto directions = nodal_directions(geometry);
    const auto stencil = three_node_stencil();
    const auto result =
      gfe::p1_discrete_tension_contribution(geometry, std::span<const FixedLogGeometry3::Point>(nodes), stencil);
    ASSERT_TRUE(result.converged());
    ASSERT_GT(result.value, 0);

    double exact = 0;
    for (std::size_t node = 0; node < nodes.size(); ++node) {
        exact += geometry.inner_product(nodes[node], result.nodal_gradient[node], directions[node]);
    }
    double best_error = std::numeric_limits<double>::infinity();
    for (const double step : {1.0e-4, 3.0e-5, 1.0e-5}) {
        const auto plus = perturb_nodes(
          geometry, std::span<const FixedLogGeometry3::Point>(nodes),
          std::span<const FixedLogGeometry3::Tangent>(directions), step);
        const auto minus = perturb_nodes(
          geometry, std::span<const FixedLogGeometry3::Point>(nodes),
          std::span<const FixedLogGeometry3::Tangent>(directions), -step);
        const double plus_value =
          gfe::p1_discrete_tension_value(geometry, std::span<const FixedLogGeometry3::Point>(plus), stencil).value;
        const double minus_value =
          gfe::p1_discrete_tension_value(geometry, std::span<const FixedLogGeometry3::Point>(minus), stencil).value;
        const double finite_difference = (plus_value - minus_value) / (2 * step);
        const double error =
          std::abs(finite_difference - exact) / std::max({1.0, std::abs(finite_difference), std::abs(exact)});
        best_error = std::min(best_error, error);
    }
    EXPECT_LT(best_error, 2.0e-7);
}

TEST(P1DiscreteTension, LogEuclideanHasTheConstantFieldNullspace) {
    const DynamicLogGeometry geometry(2);
    const auto point = make_point(geometry, {4.0, 0.6, 0.6, 2.5});
    const std::vector<DynamicLogGeometry::Point> nodes {point, point, point};
    const auto result = gfe::p1_discrete_tension_contribution(
      geometry, std::span<const DynamicLogGeometry::Point>(nodes), three_node_stencil());

    ASSERT_TRUE(result.converged());
    EXPECT_DOUBLE_EQ(result.value, 0);
    ASSERT_EQ(result.nodal_gradient.size(), nodes.size());
    for (std::size_t node = 0; node < nodes.size(); ++node) {
        EXPECT_DOUBLE_EQ(geometry.norm(nodes[node], result.nodal_gradient[node]), 0);
    }
}

TEST(P1DiscreteTension, LogEuclideanRemainsFiniteForSmallStiffnessAndMassScales) {
    const FixedLogGeometry2 geometry;
    const std::vector<FixedLogGeometry2::Point> nodes {
      make_point(geometry, {1, 0, 0, 1}), make_point(geometry, {std::exp(1.0), 0, 0, 1})};
    const double stiffness = 1.0e-160;
    const gfe::P1LumpedLaplacianStencil stencil {
      {1.0e-310, 1},
      {{0, 1, stiffness}}
    };
    const auto result =
      gfe::p1_discrete_tension_contribution(geometry, std::span<const FixedLogGeometry2::Point>(nodes), stencil);

    const double coefficient =
      (stiffness / stencil.lumped_masses[0]) * stiffness + (stiffness / stencil.lumped_masses[1]) * stiffness;
    const double expected_value =
      0.5 * coefficient * geometry.distance(nodes[0], nodes[1]) * geometry.distance(nodes[0], nodes[1]);
    ASSERT_TRUE(result.converged());
    ASSERT_TRUE(std::isfinite(result.value));
    EXPECT_NEAR(result.value, expected_value, 2.0e-13 * expected_value);
    const auto expected_first = geometry.linear_combination(
      nodes[0], -coefficient, geometry.logarithm(nodes[0], nodes[1]), 0, geometry.zero_tangent(nodes[0]));
    const auto expected_second = geometry.linear_combination(
      nodes[1], -coefficient, geometry.logarithm(nodes[1], nodes[0]), 0, geometry.zero_tangent(nodes[1]));
    expect_tangent_near(geometry, nodes[0], result.nodal_gradient[0], expected_first, 3.0e-13);
    expect_tangent_near(geometry, nodes[1], result.nodal_gradient[1], expected_second, 3.0e-13);
}

TEST(P1DiscreteTension, LogEuclideanAvoidsOverflowingInverseMassIntermediates) {
    const FixedLogGeometry2 geometry;
    const std::vector<FixedLogGeometry2::Point> nodes {
      make_point(geometry, {1, 0, 0, 1}), make_point(geometry, {std::exp(1.0), 0, 0, 1})};
    const double stiffness = 0.05;
    const gfe::P1LumpedLaplacianStencil stencil {
      {1.0e-310, 1.0e-310},
      {{0, 1, stiffness}}
    };
    const auto result =
      gfe::p1_discrete_tension_contribution(geometry, std::span<const FixedLogGeometry2::Point>(nodes), stencil);

    const double coefficient =
      stiffness * stiffness / stencil.lumped_masses[0] + stiffness * stiffness / stencil.lumped_masses[1];
    const double expected_value =
      0.5 * coefficient * geometry.distance(nodes[0], nodes[1]) * geometry.distance(nodes[0], nodes[1]);
    ASSERT_TRUE(result.converged());
    ASSERT_TRUE(std::isfinite(result.value));
    EXPECT_NEAR(result.value, expected_value, 3.0e-15 * expected_value);
    const auto expected_first = geometry.linear_combination(
      nodes[0], -coefficient, geometry.logarithm(nodes[0], nodes[1]), 0, geometry.zero_tangent(nodes[0]));
    const auto expected_second = geometry.linear_combination(
      nodes[1], -coefficient, geometry.logarithm(nodes[1], nodes[0]), 0, geometry.zero_tangent(nodes[1]));
    expect_tangent_near(geometry, nodes[0], result.nodal_gradient[0], expected_first, 3.0e-15);
    expect_tangent_near(geometry, nodes[1], result.nodal_gradient[1], expected_second, 3.0e-15);
}

TEST(P1DiscreteTension, LogEuclideanRejectsMalformedStencilsAndDynamicShapes) {
    const FixedLogGeometry2 geometry;
    const auto nodes = noncommuting_nodes(geometry);
    const auto valid = three_node_stencil();

    EXPECT_THROW(
      gfe::p1_discrete_tension_value(
        geometry, std::span<const FixedLogGeometry2::Point>(nodes.data(), nodes.size() - 1), valid),
      std::invalid_argument);
    auto malformed = valid;
    malformed.lumped_masses[0] = 0;
    EXPECT_THROW(
      gfe::p1_discrete_tension_value(geometry, std::span<const FixedLogGeometry2::Point>(nodes), malformed),
      std::invalid_argument);
    malformed = valid;
    malformed.lumped_masses[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(
      gfe::p1_discrete_tension_value(geometry, std::span<const FixedLogGeometry2::Point>(nodes), malformed),
      std::invalid_argument);
    malformed = valid;
    malformed.edges[0].first = malformed.edges[0].second;
    EXPECT_THROW(
      gfe::p1_discrete_tension_value(geometry, std::span<const FixedLogGeometry2::Point>(nodes), malformed),
      std::invalid_argument);
    malformed = valid;
    malformed.edges.push_back(malformed.edges.back());
    EXPECT_THROW(
      gfe::p1_discrete_tension_value(geometry, std::span<const FixedLogGeometry2::Point>(nodes), malformed),
      std::invalid_argument);

    const DynamicLogGeometry dynamic_geometry2(2);
    const DynamicLogGeometry dynamic_geometry3(3);
    const auto order_three_nodes = noncommuting_nodes(dynamic_geometry3);
    EXPECT_THROW(
      gfe::p1_discrete_tension_value(
        dynamic_geometry2, std::span<const DynamicLogGeometry::Point>(order_three_nodes), valid),
      std::invalid_argument);
}
