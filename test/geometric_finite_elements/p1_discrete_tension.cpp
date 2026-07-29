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
#include <type_traits>
#include <vector>

namespace {

namespace gfe = fdapde::gfe;
namespace manifold = fdapde::manifold;
namespace native = fdapde::linalg;

using FixedLogGeometry2 = manifold::LogEuclideanSPDGeometry<double, 2>;
using FixedLogGeometry3 = manifold::LogEuclideanSPDGeometry<double, 3>;
using DynamicLogGeometry = manifold::LogEuclideanSPDGeometry<double, fdapde::Dynamic>;
using FixedAffineGeometry2 = manifold::AffineInvariantSPDGeometry<double, 2>;
using FixedAffineGeometry3 = manifold::AffineInvariantSPDGeometry<double, 3>;
using DynamicAffineGeometry = manifold::AffineInvariantSPDGeometry<double, fdapde::Dynamic>;

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

gfe::P1LumpedLaplacianStencil squared_distance_edge_stencil() {
    return {
      {0.7,          1.4,           0.9         },
      {{0, 1, -1.2}, {0, 2, -0.35}, {1, 2, -0.8}}
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

template <typename Geometry> void expect_gradient_directional_difference(int order, double tolerance) {
    const Geometry geometry = make_geometry<Geometry>(order);
    const auto nodes = noncommuting_nodes(geometry);
    const auto directions = nodal_directions(geometry);
    const auto stencil = three_node_stencil();
    const auto result =
      gfe::p1_discrete_tension_contribution(geometry, std::span<const typename Geometry::Point>(nodes), stencil);
    ASSERT_TRUE(result.converged());
    ASSERT_GT(result.value, 0);

    double exact = 0;
    for (std::size_t node = 0; node < nodes.size(); ++node) {
        exact += geometry.inner_product(nodes[node], result.nodal_gradient[node], directions[node]);
    }
    double best_error = std::numeric_limits<double>::infinity();
    for (const double step : {1.0e-4, 3.0e-5, 1.0e-5}) {
        const auto plus = perturb_nodes(
          geometry, std::span<const typename Geometry::Point>(nodes),
          std::span<const typename Geometry::Tangent>(directions), step);
        const auto minus = perturb_nodes(
          geometry, std::span<const typename Geometry::Point>(nodes),
          std::span<const typename Geometry::Tangent>(directions), -step);
        const double plus_value =
          gfe::p1_discrete_tension_value(geometry, std::span<const typename Geometry::Point>(plus), stencil).value;
        const double minus_value =
          gfe::p1_discrete_tension_value(geometry, std::span<const typename Geometry::Point>(minus), stencil).value;
        const double finite_difference = (plus_value - minus_value) / (2 * step);
        const double error =
          std::abs(finite_difference - exact) / std::max({1.0, std::abs(finite_difference), std::abs(exact)});
        best_error = std::min(best_error, error);
    }
    EXPECT_LT(best_error, tolerance);
}

template <typename Geometry> void expect_affine_two_node_oracle(int order) {
    const Geometry geometry = make_geometry<Geometry>(order);
    const auto candidates = noncommuting_nodes(geometry);
    const std::vector<typename Geometry::Point> nodes {candidates[0], candidates[1]};
    constexpr double stiffness = -1.2;
    const gfe::P1LumpedLaplacianStencil stencil {
      {0.7, 1.4},
      {{0, 1, stiffness}}
    };
    const auto value =
      gfe::p1_discrete_tension_value(geometry, std::span<const typename Geometry::Point>(nodes), stencil);
    const auto result =
      gfe::p1_discrete_tension_contribution(geometry, std::span<const typename Geometry::Point>(nodes), stencil);
    const double coefficient = stiffness * stiffness * (1 / stencil.lumped_masses[0] + 1 / stencil.lumped_masses[1]);
    const double distance = geometry.distance(nodes[0], nodes[1]);
    const auto expected_first = geometry.linear_combination(
      nodes[0], -coefficient, geometry.logarithm(nodes[0], nodes[1]), 0, geometry.zero_tangent(nodes[0]));
    const auto expected_second = geometry.linear_combination(
      nodes[1], -coefficient, geometry.logarithm(nodes[1], nodes[0]), 0, geometry.zero_tangent(nodes[1]));

    ASSERT_TRUE(value.converged());
    ASSERT_TRUE(result.converged());
    EXPECT_NEAR(value.value, 0.5 * coefficient * distance * distance, 5.0e-11);
    EXPECT_DOUBLE_EQ(result.value, value.value);
    ASSERT_EQ(result.nodal_gradient.size(), nodes.size());
    expect_tangent_near(geometry, nodes[0], result.nodal_gradient[0], expected_first, 4.0e-10);
    expect_tangent_near(geometry, nodes[1], result.nodal_gradient[1], expected_second, 4.0e-10);
}

template <typename Geometry>
void expect_scaled_two_node_oracle(double stiffness, double first_mass, double second_mass, double tolerance) {
    const Geometry geometry = make_geometry<Geometry>(2);
    const std::vector<typename Geometry::Point> nodes {
      make_point(geometry, {1, 0, 0, 1}), make_point(geometry, {std::exp(1.0), 0, 0, 1})};
    const gfe::P1LumpedLaplacianStencil stencil {
      {first_mass, second_mass},
      {{0, 1, stiffness}}
    };
    const auto result =
      gfe::p1_discrete_tension_contribution(geometry, std::span<const typename Geometry::Point>(nodes), stencil);
    auto scaled_square = [stiffness](double mass) {
        const double quotient = stiffness / mass;
        return std::isfinite(quotient) ? quotient * stiffness : stiffness * stiffness / mass;
    };
    const double coefficient = scaled_square(first_mass) + scaled_square(second_mass);
    const double distance = geometry.distance(nodes[0], nodes[1]);
    const double expected_value = 0.5 * coefficient * distance * distance;
    const auto expected_first = geometry.linear_combination(
      nodes[0], -coefficient, geometry.logarithm(nodes[0], nodes[1]), 0, geometry.zero_tangent(nodes[0]));
    const auto expected_second = geometry.linear_combination(
      nodes[1], -coefficient, geometry.logarithm(nodes[1], nodes[0]), 0, geometry.zero_tangent(nodes[1]));

    ASSERT_TRUE(result.converged());
    ASSERT_TRUE(std::isfinite(result.value));
    EXPECT_NEAR(result.value, expected_value, tolerance * expected_value);
    expect_tangent_near(geometry, nodes[0], result.nodal_gradient[0], expected_first, tolerance);
    expect_tangent_near(geometry, nodes[1], result.nodal_gradient[1], expected_second, tolerance);
}

template <typename LogGeometry, typename AffineGeometry> void expect_commuting_geometries_agree(int order) {
    static_assert(std::is_same_v<typename LogGeometry::Point, typename AffineGeometry::Point>);
    const LogGeometry log_geometry = make_geometry<LogGeometry>(order);
    const AffineGeometry affine_geometry = make_geometry<AffineGeometry>(order);
    std::vector<typename LogGeometry::Point> nodes;
    if (order == 2) {
        nodes = {
          make_point(log_geometry, {2.0, 0, 0, 4.0}), make_point(log_geometry, {3.0, 0, 0, 1.5}),
          make_point(log_geometry, {1.2, 0, 0, 2.6})};
    } else {
        nodes = {
          make_point(log_geometry, {2.0, 0, 0, 0, 4.0, 0, 0, 0, 1.5}),
          make_point(log_geometry, {3.0, 0, 0, 0, 1.5, 0, 0, 0, 2.2}),
          make_point(log_geometry, {1.2, 0, 0, 0, 2.6, 0, 0, 0, 4.1})};
    }
    const auto stencil = three_node_stencil();
    const auto log_result =
      gfe::p1_discrete_tension_contribution(log_geometry, std::span<const typename LogGeometry::Point>(nodes), stencil);
    const auto affine_result = gfe::p1_discrete_tension_contribution(
      affine_geometry, std::span<const typename AffineGeometry::Point>(nodes), stencil);

    ASSERT_TRUE(log_result.converged());
    ASSERT_TRUE(affine_result.converged());
    EXPECT_NEAR(log_result.value, affine_result.value, 2.0e-11 * std::max(1.0, log_result.value));
    for (std::size_t node = 0; node < nodes.size(); ++node) {
        expect_tangent_near(
          log_geometry, nodes[node], affine_result.nodal_gradient[node], log_result.nodal_gradient[node], 3.0e-10);
    }
}

template <typename Outer, typename Middle>
native::SymmetricMatrix<double, 3, 3> congruence3(const Outer& outer, const Middle& middle) {
    native::SymmetricMatrix<double, 3, 3> result;
    for (int row = 0; row < 3; ++row) {
        for (int col = 0; col <= row; ++col) {
            double value = 0;
            for (int first = 0; first < 3; ++first) {
                for (int second = 0; second < 3; ++second) {
                    value += static_cast<double>(outer(row, first)) * static_cast<double>(middle(first, second)) *
                             static_cast<double>(outer(col, second));
                }
            }
            result(row, col) = value;
        }
    }
    return result;
}

template <typename Geometry> void expect_relabeling_covariance(const Geometry& geometry) {
    const auto nodes = noncommuting_nodes(geometry);
    const auto result = gfe::p1_discrete_tension_contribution(
      geometry, std::span<const typename Geometry::Point>(nodes), three_node_stencil());
    const std::vector<typename Geometry::Point> permuted_nodes {nodes[2], nodes[0], nodes[1]};
    const gfe::P1LumpedLaplacianStencil permuted_stencil {
      {0.9,          0.7,          1.4         },
      {{0, 1, 0.35}, {0, 2, -0.8}, {1, 2, -1.2}}
    };
    const auto permuted = gfe::p1_discrete_tension_contribution(
      geometry, std::span<const typename Geometry::Point>(permuted_nodes), permuted_stencil);

    ASSERT_TRUE(result.converged());
    ASSERT_TRUE(permuted.converged());
    EXPECT_NEAR(result.value, permuted.value, 3.0e-12 * std::max(1.0, result.value));
    constexpr std::array<std::size_t, 3> old_for_new {2, 0, 1};
    for (std::size_t node = 0; node < old_for_new.size(); ++node) {
        expect_tangent_near(
          geometry, permuted_nodes[node], permuted.nodal_gradient[node], result.nodal_gradient[old_for_new[node]],
          4.0e-10);
    }
}

template <typename Geometry> void expect_squared_distance_two_node_oracle(int order) {
    const Geometry geometry = make_geometry<Geometry>(order);
    const auto candidates = noncommuting_nodes(geometry);
    const std::vector<typename Geometry::Point> nodes {candidates[0], candidates[1]};
    constexpr double stiffness = -1.2;
    const gfe::P1LumpedLaplacianStencil stencil {
      {0.7, 1.4},
      {{0, 1, stiffness}}
    };
    const auto value = gfe::p1_squared_distance_edge_dirichlet_value(
      geometry, std::span<const typename Geometry::Point>(nodes), stencil);
    const auto result = gfe::p1_squared_distance_edge_dirichlet_contribution(
      geometry, std::span<const typename Geometry::Point>(nodes), stencil);
    const double distance = geometry.distance(nodes[0], nodes[1]);
    const auto expected_first = geometry.linear_combination(
      nodes[0], stiffness, geometry.logarithm(nodes[0], nodes[1]), 0, geometry.zero_tangent(nodes[0]));
    const auto expected_second = geometry.linear_combination(
      nodes[1], stiffness, geometry.logarithm(nodes[1], nodes[0]), 0, geometry.zero_tangent(nodes[1]));

    ASSERT_TRUE(value.converged());
    ASSERT_TRUE(result.converged());
    EXPECT_NEAR(value.value, 0.5 * (-stiffness) * distance * distance, 5.0e-12);
    EXPECT_DOUBLE_EQ(result.value, value.value);
    ASSERT_EQ(result.nodal_gradient.size(), nodes.size());
    expect_tangent_near(geometry, nodes[0], result.nodal_gradient[0], expected_first, 3.0e-11);
    expect_tangent_near(geometry, nodes[1], result.nodal_gradient[1], expected_second, 3.0e-11);
}

template <typename Geometry> void expect_squared_distance_log_chart_oracle(int order) {
    const Geometry geometry = make_geometry<Geometry>(order);
    const auto nodes = noncommuting_nodes(geometry);
    const auto stencil = squared_distance_edge_stencil();
    const auto value = gfe::p1_squared_distance_edge_dirichlet_value(
      geometry, std::span<const typename Geometry::Point>(nodes), stencil);
    const auto result = gfe::p1_squared_distance_edge_dirichlet_contribution(
      geometry, std::span<const typename Geometry::Point>(nodes), stencil);

    std::vector<typename Geometry::Tangent> logarithms;
    std::vector<typename Geometry::Tangent> chart_gradient;
    for (const auto& node : nodes) {
        logarithms.emplace_back(native::matrix_log(node));
        chart_gradient.push_back(geometry.zero_tangent(node));
    }
    double expected_value = 0;
    for (const auto& edge : stencil.edges) {
        const auto difference =
          geometry.linear_combination(nodes[edge.first], 1, logarithms[edge.second], -1, logarithms[edge.first]);
        double squared_norm = 0;
        for (int row = 0; row < order; ++row) {
            for (int col = 0; col <= row; ++col) {
                const double coefficient = static_cast<double>(difference(row, col));
                squared_norm += (row == col ? 1 : 2) * coefficient * coefficient;
            }
        }
        expected_value += 0.5 * (-edge.stiffness) * squared_norm;
        chart_gradient[edge.first] =
          geometry.linear_combination(nodes[edge.first], 1, chart_gradient[edge.first], edge.stiffness, difference);
        chart_gradient[edge.second] =
          geometry.linear_combination(nodes[edge.second], 1, chart_gradient[edge.second], -edge.stiffness, difference);
    }

    ASSERT_TRUE(value.converged());
    ASSERT_TRUE(result.converged());
    EXPECT_NEAR(value.value, expected_value, 3.0e-12 * std::max(1.0, expected_value));
    EXPECT_DOUBLE_EQ(result.value, value.value);
    for (std::size_t node = 0; node < nodes.size(); ++node) {
        const typename Geometry::Tangent expected = native::matrix_exp_frechet(logarithms[node], chart_gradient[node]);
        expect_tangent_near(geometry, nodes[node], result.nodal_gradient[node], expected, 4.0e-11);
    }
}

template <typename Geometry> void expect_squared_distance_directional_difference(int order, double tolerance) {
    const Geometry geometry = make_geometry<Geometry>(order);
    const auto nodes = noncommuting_nodes(geometry);
    const auto directions = nodal_directions(geometry);
    const auto stencil = squared_distance_edge_stencil();
    const auto result = gfe::p1_squared_distance_edge_dirichlet_contribution(
      geometry, std::span<const typename Geometry::Point>(nodes), stencil);
    ASSERT_TRUE(result.converged());

    double exact = 0;
    for (std::size_t node = 0; node < nodes.size(); ++node) {
        exact += geometry.inner_product(nodes[node], result.nodal_gradient[node], directions[node]);
    }
    double best_error = std::numeric_limits<double>::infinity();
    for (const double step : {1.0e-4, 3.0e-5, 1.0e-5}) {
        const auto plus = perturb_nodes(
          geometry, std::span<const typename Geometry::Point>(nodes),
          std::span<const typename Geometry::Tangent>(directions), step);
        const auto minus = perturb_nodes(
          geometry, std::span<const typename Geometry::Point>(nodes),
          std::span<const typename Geometry::Tangent>(directions), -step);
        const double plus_value = gfe::p1_squared_distance_edge_dirichlet_value(
                                    geometry, std::span<const typename Geometry::Point>(plus), stencil)
                                    .value;
        const double minus_value = gfe::p1_squared_distance_edge_dirichlet_value(
                                     geometry, std::span<const typename Geometry::Point>(minus), stencil)
                                     .value;
        const double finite_difference = (plus_value - minus_value) / (2 * step);
        const double error =
          std::abs(finite_difference - exact) / std::max({1.0, std::abs(finite_difference), std::abs(exact)});
        best_error = std::min(best_error, error);
    }
    EXPECT_LT(best_error, tolerance);
}

}   // namespace

TEST(P1DiscreteTension, LogEuclideanMatchesIndependentChartOracleForFixedAndDynamicSPD2AndSPD3) {
    expect_independent_chart_oracle<FixedLogGeometry2>(2);
    expect_independent_chart_oracle<DynamicLogGeometry>(2);
    expect_independent_chart_oracle<FixedLogGeometry3>(3);
    expect_independent_chart_oracle<DynamicLogGeometry>(3);
}

TEST(P1DiscreteTension, LogEuclideanGradientMatchesCenteredGeodesicDirectionalDifferences) {
    expect_gradient_directional_difference<FixedLogGeometry3>(3, 2.0e-7);
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
    expect_scaled_two_node_oracle<FixedLogGeometry2>(1.0e-160, 1.0e-310, 1, 3.0e-13);
}

TEST(P1DiscreteTension, LogEuclideanAvoidsOverflowingInverseMassIntermediates) {
    expect_scaled_two_node_oracle<FixedLogGeometry2>(0.05, 1.0e-310, 1.0e-310, 3.0e-15);
}

TEST(P1DiscreteTension, AffineInvariantMatchesExactTwoNodeOracleForFixedAndDynamicSPD2AndSPD3) {
    expect_affine_two_node_oracle<FixedAffineGeometry2>(2);
    expect_affine_two_node_oracle<DynamicAffineGeometry>(2);
    expect_affine_two_node_oracle<FixedAffineGeometry3>(3);
    expect_affine_two_node_oracle<DynamicAffineGeometry>(3);
}

TEST(P1DiscreteTension, AffineInvariantGradientMatchesCenteredNoncommutingDirectionalDifferences) {
    expect_gradient_directional_difference<DynamicAffineGeometry>(2, 2.0e-7);
    expect_gradient_directional_difference<FixedAffineGeometry3>(3, 2.0e-7);
}

TEST(P1DiscreteTension, AffineInvariantMatchesLogEuclideanForCommutingFields) {
    expect_commuting_geometries_agree<FixedLogGeometry2, FixedAffineGeometry2>(2);
    expect_commuting_geometries_agree<DynamicLogGeometry, DynamicAffineGeometry>(3);
}

TEST(P1DiscreteTension, AffineInvariantIsInvariantAndGradientEquivariantUnderGeneralCongruence) {
    const FixedAffineGeometry3 geometry;
    const auto nodes = noncommuting_nodes(geometry);
    const auto result = gfe::p1_discrete_tension_contribution(
      geometry, std::span<const FixedAffineGeometry3::Point>(nodes), three_node_stencil());
    const native::Matrix<double, 3, 3> basis({1.2, -0.2, 0.1, 0.3, 0.9, -0.15, -0.1, 0.25, 1.1});
    std::vector<FixedAffineGeometry3::Point> transformed_nodes;
    transformed_nodes.reserve(nodes.size());
    for (const auto& node : nodes) { transformed_nodes.emplace_back(congruence3(basis, node), native::checked); }
    const auto transformed = gfe::p1_discrete_tension_contribution(
      geometry, std::span<const FixedAffineGeometry3::Point>(transformed_nodes), three_node_stencil());

    ASSERT_TRUE(result.converged());
    ASSERT_TRUE(transformed.converged());
    EXPECT_NEAR(result.value, transformed.value, 3.0e-9 * std::max(1.0, result.value));
    for (std::size_t node = 0; node < nodes.size(); ++node) {
        expect_tangent_near(
          geometry, transformed_nodes[node], transformed.nodal_gradient[node],
          congruence3(basis, result.nodal_gradient[node]), 8.0e-9);
    }
}

TEST(P1DiscreteTension, IsCovariantUnderGlobalNodeRelabeling) {
    expect_relabeling_covariance(FixedLogGeometry3 {});
    expect_relabeling_covariance(FixedAffineGeometry3 {});
}

TEST(P1DiscreteTension, AffineInvariantHasTheConstantFieldNullspace) {
    const DynamicAffineGeometry geometry(2);
    const auto point = make_point(geometry, {4.0, 0.6, 0.6, 2.5});
    const std::vector<DynamicAffineGeometry::Point> nodes {point, point, point};
    const auto result = gfe::p1_discrete_tension_contribution(
      geometry, std::span<const DynamicAffineGeometry::Point>(nodes), three_node_stencil());

    ASSERT_TRUE(result.converged());
    EXPECT_LT(result.value, 1.0e-26);
    ASSERT_EQ(result.nodal_gradient.size(), nodes.size());
    for (std::size_t node = 0; node < nodes.size(); ++node) {
        EXPECT_LT(geometry.norm(nodes[node], result.nodal_gradient[node]), 2.0e-14);
    }
}

TEST(P1DiscreteTension, AffineInvariantKeepsFiniteRepresentableTwoNodeScaleLimits) {
    expect_scaled_two_node_oracle<FixedAffineGeometry2>(1.0e-160, 1.0e-310, 1, 2.0e-11);
    expect_scaled_two_node_oracle<FixedAffineGeometry2>(0.05, 1.0e-310, 1.0e-310, 2.0e-11);
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

TEST(P1SquaredDistanceEdgeDirichlet, MatchesTwoNodeOracleForFixedAndDynamicSPD2AndSPD3) {
    expect_squared_distance_two_node_oracle<FixedLogGeometry2>(2);
    expect_squared_distance_two_node_oracle<DynamicLogGeometry>(2);
    expect_squared_distance_two_node_oracle<FixedLogGeometry3>(3);
    expect_squared_distance_two_node_oracle<DynamicLogGeometry>(3);
    expect_squared_distance_two_node_oracle<FixedAffineGeometry2>(2);
    expect_squared_distance_two_node_oracle<DynamicAffineGeometry>(2);
    expect_squared_distance_two_node_oracle<FixedAffineGeometry3>(3);
    expect_squared_distance_two_node_oracle<DynamicAffineGeometry>(3);
}

TEST(P1SquaredDistanceEdgeDirichlet, LogEuclideanMatchesIndependentGraphChartOracle) {
    expect_squared_distance_log_chart_oracle<FixedLogGeometry2>(2);
    expect_squared_distance_log_chart_oracle<DynamicLogGeometry>(2);
    expect_squared_distance_log_chart_oracle<FixedLogGeometry3>(3);
    expect_squared_distance_log_chart_oracle<DynamicLogGeometry>(3);
}

TEST(P1SquaredDistanceEdgeDirichlet, GradientMatchesNoncommutingDirectionalDifferences) {
    expect_squared_distance_directional_difference<FixedLogGeometry3>(3, 2.0e-8);
    expect_squared_distance_directional_difference<DynamicAffineGeometry>(2, 2.0e-7);
    expect_squared_distance_directional_difference<FixedAffineGeometry3>(3, 2.0e-7);
}

TEST(P1SquaredDistanceEdgeDirichlet, AffineInvariantIsCongruenceInvariantAndGradientEquivariant) {
    const FixedAffineGeometry3 geometry;
    const auto nodes = noncommuting_nodes(geometry);
    const auto stencil = squared_distance_edge_stencil();
    const auto result = gfe::p1_squared_distance_edge_dirichlet_contribution(
      geometry, std::span<const FixedAffineGeometry3::Point>(nodes), stencil);
    const native::Matrix<double, 3, 3> basis({1.2, -0.2, 0.1, 0.3, 0.9, -0.15, -0.1, 0.25, 1.1});
    std::vector<FixedAffineGeometry3::Point> transformed_nodes;
    for (const auto& node : nodes) { transformed_nodes.emplace_back(congruence3(basis, node), native::checked); }
    const auto transformed = gfe::p1_squared_distance_edge_dirichlet_contribution(
      geometry, std::span<const FixedAffineGeometry3::Point>(transformed_nodes), stencil);

    ASSERT_TRUE(result.converged());
    ASSERT_TRUE(transformed.converged());
    EXPECT_NEAR(result.value, transformed.value, 3.0e-9 * std::max(1.0, result.value));
    for (std::size_t node = 0; node < nodes.size(); ++node) {
        expect_tangent_near(
          geometry, transformed_nodes[node], transformed.nodal_gradient[node],
          congruence3(basis, result.nodal_gradient[node]), 8.0e-9);
    }
}

TEST(P1SquaredDistanceEdgeDirichlet, IsRelabelingCovariantAndIndependentOfLumpedMasses) {
    const FixedLogGeometry3 geometry;
    const auto nodes = noncommuting_nodes(geometry);
    const auto stencil = squared_distance_edge_stencil();
    const auto result = gfe::p1_squared_distance_edge_dirichlet_contribution(
      geometry, std::span<const FixedLogGeometry3::Point>(nodes), stencil);

    auto changed_masses = stencil;
    changed_masses.lumped_masses = {8, 3, 5};
    const auto mass_changed = gfe::p1_squared_distance_edge_dirichlet_contribution(
      geometry, std::span<const FixedLogGeometry3::Point>(nodes), changed_masses);
    EXPECT_DOUBLE_EQ(mass_changed.value, result.value);
    for (std::size_t node = 0; node < nodes.size(); ++node) {
        expect_tangent_near(geometry, nodes[node], mass_changed.nodal_gradient[node], result.nodal_gradient[node], 0);
    }

    const std::vector<FixedLogGeometry3::Point> permuted_nodes {nodes[2], nodes[0], nodes[1]};
    const gfe::P1LumpedLaplacianStencil permuted_stencil {
      {0.9,           0.7,          1.4         },
      {{0, 1, -0.35}, {0, 2, -0.8}, {1, 2, -1.2}}
    };
    const auto permuted = gfe::p1_squared_distance_edge_dirichlet_contribution(
      geometry, std::span<const FixedLogGeometry3::Point>(permuted_nodes), permuted_stencil);
    constexpr std::array<std::size_t, 3> old_for_new {2, 0, 1};
    EXPECT_NEAR(permuted.value, result.value, 3.0e-12 * std::max(1.0, result.value));
    for (std::size_t node = 0; node < old_for_new.size(); ++node) {
        expect_tangent_near(
          geometry, permuted_nodes[node], permuted.nodal_gradient[node], result.nodal_gradient[old_for_new[node]],
          4.0e-10);
    }
}

TEST(P1SquaredDistanceEdgeDirichlet, HasConstantNullspaceAndRejectsSignedOrMalformedStencils) {
    const DynamicAffineGeometry geometry(2);
    const auto point = make_point(geometry, {4.0, 0.6, 0.6, 2.5});
    const std::vector<DynamicAffineGeometry::Point> constant_nodes {point, point, point};
    const auto constant = gfe::p1_squared_distance_edge_dirichlet_contribution(
      geometry, std::span<const DynamicAffineGeometry::Point>(constant_nodes), squared_distance_edge_stencil());
    EXPECT_LT(constant.value, 1.0e-26);
    for (std::size_t node = 0; node < constant_nodes.size(); ++node) {
        EXPECT_LT(geometry.norm(constant_nodes[node], constant.nodal_gradient[node]), 2.0e-14);
    }

    const auto nodes = noncommuting_nodes(geometry);
    const auto signed_stencil = three_node_stencil();
    EXPECT_THROW(
      gfe::p1_squared_distance_edge_dirichlet_value(
        geometry, std::span<const DynamicAffineGeometry::Point>(nodes), signed_stencil),
      std::invalid_argument);
    EXPECT_THROW(
      gfe::p1_squared_distance_edge_dirichlet_contribution(
        geometry, std::span<const DynamicAffineGeometry::Point>(nodes), signed_stencil),
      std::invalid_argument);
    EXPECT_THROW(
      gfe::p1_squared_distance_edge_dirichlet_value(
        geometry, std::span<const DynamicAffineGeometry::Point>(nodes.data(), nodes.size() - 1),
        squared_distance_edge_stencil()),
      std::invalid_argument);

    auto malformed = squared_distance_edge_stencil();
    malformed.lumped_masses[0] = 0;
    EXPECT_THROW(
      gfe::p1_squared_distance_edge_dirichlet_value(
        geometry, std::span<const DynamicAffineGeometry::Point>(nodes), malformed),
      std::invalid_argument);

    const DynamicAffineGeometry geometry3(3);
    const auto order_three_nodes = noncommuting_nodes(geometry3);
    EXPECT_THROW(
      gfe::p1_squared_distance_edge_dirichlet_value(
        geometry, std::span<const DynamicAffineGeometry::Point>(order_three_nodes), squared_distance_edge_stencil()),
      std::invalid_argument);
}
