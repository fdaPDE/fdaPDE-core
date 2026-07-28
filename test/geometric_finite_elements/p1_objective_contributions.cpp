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

namespace gfe = fdapde::gfe;
namespace manifold = fdapde::manifold;
namespace native = fdapde::linalg;

using FixedLogGeometry = manifold::LogEuclideanSPDGeometry<double, 3>;
using DynamicLogGeometry = manifold::LogEuclideanSPDGeometry<double, fdapde::Dynamic>;
using FixedFloatLogGeometry = manifold::LogEuclideanSPDGeometry<float, 3>;
using FixedAffineGeometry = manifold::AffineInvariantSPDGeometry<double, 3>;
using DynamicAffineGeometry = manifold::AffineInvariantSPDGeometry<double, fdapde::Dynamic>;
using FixedFloatAffineGeometry = manifold::AffineInvariantSPDGeometry<float, 3>;

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
    native::Matrix<typename Geometry::Scalar, 3, 3> dense;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            dense(i, j) = static_cast<typename Geometry::Scalar>(coefficients[static_cast<std::size_t>(3 * i + j)]);
        }
    }
    return typename Geometry::Point(dense, native::checked);
}

template <typename Geometry> typename Geometry::Point make_diagonal_point(double a, double b, double c) {
    native::Matrix<typename Geometry::Scalar, 3, 3> dense;
    dense.set_zero();
    dense(0, 0) = static_cast<typename Geometry::Scalar>(a);
    dense(1, 1) = static_cast<typename Geometry::Scalar>(b);
    dense(2, 2) = static_cast<typename Geometry::Scalar>(c);
    return typename Geometry::Point(dense, native::checked);
}

template <typename Geometry> typename Geometry::Tangent make_tangent(const std::array<double, 6>& coefficients) {
    typename Geometry::Tangent tangent;
    if constexpr (Geometry::Tangent::Rows == fdapde::Dynamic) { tangent.resize(3, 3); }
    std::size_t index = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j <= i; ++j) { tangent(i, j) = static_cast<typename Geometry::Scalar>(coefficients[index++]); }
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

gfe::P1GeodesicLinearizationOptions accurate_options() {
    gfe::P1GeodesicLinearizationOptions options;
    options.mean.solver.max_iterations = 600;
    options.mean.solver.gradient_tolerance = 1.0e-7;
    options.linear_solve.max_iterations = 24;
    options.linear_solve.residual_tolerance = 1.0e-9;
    return options;
}

template <typename Matrix> double frobenius_inner(const Matrix& lhs, const Matrix& rhs) {
    double result = 0;
    for (int i = 0; i < lhs.rows(); ++i) {
        for (int j = 0; j <= i; ++j) {
            const double contribution = static_cast<double>(lhs(i, j)) * static_cast<double>(rhs(i, j));
            result += i == j ? contribution : 2 * contribution;
        }
    }
    return result;
}

template <typename Geometry, typename Lhs, typename Rhs>
void expect_tangent_near(
  const Geometry& geometry, const typename Geometry::Point& point, const Lhs& lhs, const Rhs& rhs, double tolerance) {
    const auto difference = geometry.linear_combination(point, 1, lhs, -1, rhs);
    const double scale = std::max({1.0, geometry.norm(point, lhs), geometry.norm(point, rhs)});
    EXPECT_LE(geometry.norm(point, difference), tolerance * scale);
}

template <typename Geometry> gfe::P1FEMCellQuadrature<2, 2, 2> planar_packet() {
    return {
      {2, 0, 1},
      {{{-1, 1, 0}, {-1, 0, 1}}},
      {{{0.2, 0.3, 0.5}, {0.6, 0.1, 0.3}}},
      {0.2, 0.3}
    };
}

gfe::P1FEMCellQuadrature<2, 3, 2> surface_packet() {
    return {
      {0, 1, 2},
      {{{-0.5, 0.5, 0}, {-3.0 / 25.0, 0, 3.0 / 25.0}, {-4.0 / 25.0, 0, 4.0 / 25.0}}},
      {{{0.25, 0.5, 0.25}, {1, 0, 0}}},
      {2, 3}
    };
}

template <typename Geometry>
std::vector<typename Geometry::Point> perturb_nodes(
  const Geometry& geometry, std::span<const typename Geometry::Point> nodes,
  std::span<const typename Geometry::Tangent> directions, double step) {
    std::vector<typename Geometry::Point> result;
    result.reserve(nodes.size());
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        result.push_back(geometry.exponential(nodes[i], directions[i], step));
    }
    return result;
}

template <typename Geometry> void expect_affine_identity_objectives() {
    const Geometry geometry = make_geometry<Geometry>();
    const auto point = make_diagonal_point<Geometry>(1, 1, 1);
    const std::vector<typename Geometry::Point> nodes {point, point, point};
    const std::vector<double> weights {0.25, 0.5, 0.25};
    const auto observation = make_tangent<Geometry>({1, 0, 1, 0, 0, 1});
    const auto data = gfe::p1_frobenius_data_site_contribution(
      geometry, std::span<const typename Geometry::Point>(nodes), std::span<const double>(weights), observation,
      accurate_options());
    const auto dirichlet = gfe::p1_dirichlet_cell_contribution(
      geometry, std::span<const typename Geometry::Point>(nodes), surface_packet(), accurate_options());

    ASSERT_TRUE(data.converged());
    ASSERT_TRUE(dirichlet.converged());
    EXPECT_DOUBLE_EQ(data.value, 0);
    EXPECT_DOUBLE_EQ(dirichlet.value, 0);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        EXPECT_DOUBLE_EQ(geometry.norm(nodes[i], data.nodal_gradient[i]), 0);
        EXPECT_DOUBLE_EQ(geometry.norm(nodes[i], dirichlet.nodal_gradient[i]), 0);
    }
}

}   // namespace

TEST(P1ObjectiveContributions, LogEuclideanDataSiteMatchesTheAnalyticDiagonalOracle) {
    const FixedLogGeometry geometry;
    const std::vector<FixedLogGeometry::Point> nodes {
      make_diagonal_point<FixedLogGeometry>(2, 3, 4), make_diagonal_point<FixedLogGeometry>(5, 1.5, 2.5)};
    const std::vector<double> weights {0.4, 0.6};
    const auto observation = make_tangent<FixedLogGeometry>({2.2, 0, 2, 0, 0, 3.3});

    const auto result = gfe::p1_frobenius_data_site_contribution(
      geometry, std::span<const FixedLogGeometry::Point>(nodes), std::span<const double>(weights), observation);
    ASSERT_TRUE(result.converged());
    ASSERT_EQ(result.nodal_gradient.size(), nodes.size());
    EXPECT_FALSE(result.first_failure);

    const std::array<double, 3> first {2, 3, 4};
    const std::array<double, 3> second {5, 1.5, 2.5};
    const std::array<double, 3> observed {2.2, 2, 3.3};
    double expected_value = 0;
    std::array<FixedLogGeometry::Tangent, 2> expected {
      geometry.zero_tangent(nodes[0]), geometry.zero_tangent(nodes[1])};
    for (int axis = 0; axis < 3; ++axis) {
        const double mean = std::pow(first[static_cast<std::size_t>(axis)], weights[0]) *
                            std::pow(second[static_cast<std::size_t>(axis)], weights[1]);
        const double residual = mean - observed[static_cast<std::size_t>(axis)];
        expected_value += 0.5 * residual * residual;
        expected[0](axis, axis) = first[static_cast<std::size_t>(axis)] * weights[0] * mean * residual;
        expected[1](axis, axis) = second[static_cast<std::size_t>(axis)] * weights[1] * mean * residual;
    }
    EXPECT_NEAR(result.value, expected_value, 2.0e-12);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        expect_tangent_near(geometry, nodes[i], result.nodal_gradient[i], expected[i], 2.0e-11);
    }
}

TEST(P1ObjectiveContributions, LogEuclideanDataSiteSupportsDynamicAndFloatSPD3) {
    const FixedLogGeometry fixed_geometry;
    const DynamicLogGeometry dynamic_geometry(3);
    const FixedFloatLogGeometry float_geometry;
    const std::vector<FixedLogGeometry::Point> fixed_nodes {
      make_diagonal_point<FixedLogGeometry>(2, 3, 4), make_diagonal_point<FixedLogGeometry>(5, 1.5, 2.5)};
    const std::vector<DynamicLogGeometry::Point> dynamic_nodes {
      make_diagonal_point<DynamicLogGeometry>(2, 3, 4), make_diagonal_point<DynamicLogGeometry>(5, 1.5, 2.5)};
    const std::vector<FixedFloatLogGeometry::Point> float_nodes {
      make_diagonal_point<FixedFloatLogGeometry>(2, 3, 4), make_diagonal_point<FixedFloatLogGeometry>(5, 1.5, 2.5)};
    const std::vector<double> weights {0.4, 0.6};
    const auto fixed_observation = make_tangent<FixedLogGeometry>({2.2, 0, 2, 0, 0, 3.3});
    const auto dynamic_observation = make_tangent<DynamicLogGeometry>({2.2, 0, 2, 0, 0, 3.3});
    const auto float_observation = make_tangent<FixedFloatLogGeometry>({2.2, 0, 2, 0, 0, 3.3});

    const auto fixed = gfe::p1_frobenius_data_site_contribution(
      fixed_geometry, std::span<const FixedLogGeometry::Point>(fixed_nodes), std::span<const double>(weights),
      fixed_observation);
    const auto dynamic = gfe::p1_frobenius_data_site_contribution(
      dynamic_geometry, std::span<const DynamicLogGeometry::Point>(dynamic_nodes), std::span<const double>(weights),
      dynamic_observation);
    const auto float_result = gfe::p1_frobenius_data_site_contribution(
      float_geometry, std::span<const FixedFloatLogGeometry::Point>(float_nodes), std::span<const double>(weights),
      float_observation);
    ASSERT_TRUE(fixed.converged());
    ASSERT_TRUE(dynamic.converged());
    ASSERT_TRUE(float_result.converged());
    EXPECT_NEAR(dynamic.value, fixed.value, 2.0e-11);
    EXPECT_NEAR(float_result.value, fixed.value, 2.0e-4);
    for (std::size_t i = 0; i < fixed_nodes.size(); ++i) {
        for (int row = 0; row < 3; ++row) {
            for (int col = 0; col <= row; ++col) {
                EXPECT_NEAR(dynamic.nodal_gradient[i](row, col), fixed.nodal_gradient[i](row, col), 2.0e-10);
            }
        }
        EXPECT_TRUE(std::isfinite(float_geometry.norm(float_nodes[i], float_result.nodal_gradient[i])));
    }
}

TEST(P1ObjectiveContributions, AffineInvariantObjectivesSupportDynamicAndFloatSPD3) {
    expect_affine_identity_objectives<DynamicAffineGeometry>();
    expect_affine_identity_objectives<FixedFloatAffineGeometry>();
}

TEST(P1ObjectiveContributions, AffineInvariantDataSiteGradientMatchesNoncommutingDirectionalDifferences) {
    const FixedAffineGeometry geometry;
    const auto nodes = noncommuting_nodes<FixedAffineGeometry>();
    const auto directions = nodal_directions<FixedAffineGeometry>();
    const std::vector<double> weights {0.25, 0.5, 0.25};
    const auto observation = make_tangent<FixedAffineGeometry>({2.3, -0.1, 1.9, 0.2, 0.15, 3});
    const auto options = accurate_options();

    const auto result = gfe::p1_frobenius_data_site_contribution(
      geometry, std::span<const FixedAffineGeometry::Point>(nodes), std::span<const double>(weights), observation,
      options);
    ASSERT_TRUE(result.converged());
    double exact = 0;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        exact += geometry.inner_product(nodes[i], result.nodal_gradient[i], directions[i]);
    }

    double best_error = std::numeric_limits<double>::infinity();
    for (const double step : {1.0e-3, 3.0e-4, 1.0e-4}) {
        const auto plus_nodes = perturb_nodes(
          geometry, std::span<const FixedAffineGeometry::Point>(nodes),
          std::span<const FixedAffineGeometry::Tangent>(directions), step);
        const auto minus_nodes = perturb_nodes(
          geometry, std::span<const FixedAffineGeometry::Point>(nodes),
          std::span<const FixedAffineGeometry::Tangent>(directions), -step);
        const auto plus = gfe::p1_frobenius_data_site_contribution(
          geometry, std::span<const FixedAffineGeometry::Point>(plus_nodes), std::span<const double>(weights),
          observation, options);
        const auto minus = gfe::p1_frobenius_data_site_contribution(
          geometry, std::span<const FixedAffineGeometry::Point>(minus_nodes), std::span<const double>(weights),
          observation, options);
        ASSERT_TRUE(plus.converged());
        ASSERT_TRUE(minus.converged());
        const double finite_difference = (plus.value - minus.value) / (2 * step);
        const double error =
          std::abs(finite_difference - exact) / std::max({1.0, std::abs(finite_difference), std::abs(exact)});
        EXPECT_LT(error, 3.0e-4);
        best_error = std::min(best_error, error);
    }
    EXPECT_LT(best_error, 2.0e-5);
}

TEST(P1ObjectiveContributions, LogEuclideanCellMatchesTheExactChartOracleAndPermutations) {
    const FixedLogGeometry geometry;
    const auto global_nodes = noncommuting_nodes<FixedLogGeometry>();
    const auto packet = surface_packet();
    const auto result =
      gfe::p1_dirichlet_cell_contribution(geometry, std::span<const FixedLogGeometry::Point>(global_nodes), packet);
    ASSERT_TRUE(result.converged());
    ASSERT_EQ(result.nodal_gradient.size(), packet.node_count);
    EXPECT_GT(result.value, 0);

    std::array<FixedLogGeometry::Tangent, 3> logs;
    for (std::size_t i = 0; i < packet.node_count; ++i) { logs[i] = native::matrix_log(global_nodes[packet.dofs[i]]); }
    std::array<FixedLogGeometry::Tangent, 3> spatial {
      geometry.zero_tangent(global_nodes[0]), geometry.zero_tangent(global_nodes[0]),
      geometry.zero_tangent(global_nodes[0])};
    for (std::size_t axis = 0; axis < packet.embed_dim; ++axis) {
        for (std::size_t i = 0; i < packet.node_count; ++i) {
            spatial[axis] = geometry.linear_combination(
              global_nodes[0], 1, spatial[axis], packet.physical_weight_gradients[axis][i], logs[i]);
        }
    }
    const double measure = packet.integration_weights[0] + packet.integration_weights[1];
    double expected_value = 0;
    for (const auto& component : spatial) {
        expected_value = std::fma(0.5 * measure, frobenius_inner(component, component), expected_value);
    }
    EXPECT_NEAR(result.value, expected_value, 3.0e-11);

    for (std::size_t i = 0; i < packet.node_count; ++i) {
        auto chart_gradient = geometry.zero_tangent(global_nodes[0]);
        for (std::size_t axis = 0; axis < packet.embed_dim; ++axis) {
            chart_gradient = geometry.linear_combination(
              global_nodes[0], 1, chart_gradient, measure * packet.physical_weight_gradients[axis][i], spatial[axis]);
        }
        const FixedLogGeometry::Tangent expected = native::matrix_exp_frechet(logs[i], chart_gradient);
        expect_tangent_near(geometry, global_nodes[packet.dofs[i]], result.nodal_gradient[i], expected, 3.0e-9);
    }

    constexpr std::array<std::size_t, 3> permutation {2, 0, 1};
    auto permuted_packet = packet;
    for (std::size_t i = 0; i < packet.node_count; ++i) {
        permuted_packet.dofs[i] = packet.dofs[permutation[i]];
        for (std::size_t axis = 0; axis < packet.embed_dim; ++axis) {
            permuted_packet.physical_weight_gradients[axis][i] = packet.physical_weight_gradients[axis][permutation[i]];
        }
        for (std::size_t q = 0; q < packet.quadrature_size; ++q) {
            permuted_packet.barycentric_weights[q][i] = packet.barycentric_weights[q][permutation[i]];
        }
    }
    const auto permuted = gfe::p1_dirichlet_cell_contribution(
      geometry, std::span<const FixedLogGeometry::Point>(global_nodes), permuted_packet);
    ASSERT_TRUE(permuted.converged());
    EXPECT_NEAR(permuted.value, result.value, 2.0e-12);
    for (std::size_t i = 0; i < packet.node_count; ++i) {
        expect_tangent_near(
          geometry, global_nodes[permuted_packet.dofs[i]], permuted.nodal_gradient[i],
          result.nodal_gradient[permutation[i]], 3.0e-9);
    }
}

TEST(P1ObjectiveContributions, AffineInvariantCommutingCellMatchesTheLogEuclideanOracle) {
    const FixedLogGeometry log_geometry;
    const FixedAffineGeometry affine_geometry;
    const std::vector<FixedLogGeometry::Point> log_nodes {
      make_diagonal_point<FixedLogGeometry>(1.5, 3, 6), make_diagonal_point<FixedLogGeometry>(4, 2, 1.25),
      make_diagonal_point<FixedLogGeometry>(2.25, 5, 3.5)};
    const std::vector<FixedAffineGeometry::Point> affine_nodes {
      make_diagonal_point<FixedAffineGeometry>(1.5, 3, 6), make_diagonal_point<FixedAffineGeometry>(4, 2, 1.25),
      make_diagonal_point<FixedAffineGeometry>(2.25, 5, 3.5)};
    const auto packet = planar_packet<FixedLogGeometry>();
    const auto logarithmic =
      gfe::p1_dirichlet_cell_contribution(log_geometry, std::span<const FixedLogGeometry::Point>(log_nodes), packet);
    const auto affine = gfe::p1_dirichlet_cell_contribution(
      affine_geometry, std::span<const FixedAffineGeometry::Point>(affine_nodes), packet, accurate_options());
    ASSERT_TRUE(logarithmic.converged());
    ASSERT_TRUE(affine.converged());
    EXPECT_NEAR(affine.value, logarithmic.value, 3.0e-9);
    for (std::size_t i = 0; i < packet.node_count; ++i) {
        expect_tangent_near(
          affine_geometry, affine_nodes[packet.dofs[i]], affine.nodal_gradient[i], logarithmic.nodal_gradient[i],
          5.0e-8);
    }
}

TEST(P1ObjectiveContributions, AffineInvariantCellGradientMatchesNoncommutingDirectionalDifferences) {
    const FixedAffineGeometry geometry;
    const auto nodes = noncommuting_nodes<FixedAffineGeometry>();
    const auto directions = nodal_directions<FixedAffineGeometry>();
    const auto packet = surface_packet();
    const auto options = accurate_options();
    const auto result = gfe::p1_dirichlet_cell_contribution(
      geometry, std::span<const FixedAffineGeometry::Point>(nodes), packet, options);
    ASSERT_TRUE(result.converged());
    EXPECT_GT(result.value, 0);

    double exact = 0;
    for (std::size_t i = 0; i < packet.node_count; ++i) {
        exact += geometry.inner_product(nodes[packet.dofs[i]], result.nodal_gradient[i], directions[packet.dofs[i]]);
    }
    double best_error = std::numeric_limits<double>::infinity();
    for (const double step : {1.0e-3, 3.0e-4, 1.0e-4}) {
        const auto plus_nodes = perturb_nodes(
          geometry, std::span<const FixedAffineGeometry::Point>(nodes),
          std::span<const FixedAffineGeometry::Tangent>(directions), step);
        const auto minus_nodes = perturb_nodes(
          geometry, std::span<const FixedAffineGeometry::Point>(nodes),
          std::span<const FixedAffineGeometry::Tangent>(directions), -step);
        const auto plus = gfe::p1_dirichlet_cell_contribution(
          geometry, std::span<const FixedAffineGeometry::Point>(plus_nodes), packet, options);
        const auto minus = gfe::p1_dirichlet_cell_contribution(
          geometry, std::span<const FixedAffineGeometry::Point>(minus_nodes), packet, options);
        ASSERT_TRUE(plus.converged());
        ASSERT_TRUE(minus.converged());
        const double finite_difference = (plus.value - minus.value) / (2 * step);
        const double error =
          std::abs(finite_difference - exact) / std::max({1.0, std::abs(finite_difference), std::abs(exact)});
        EXPECT_LT(error, 1.0e-3);
        best_error = std::min(best_error, error);
    }
    EXPECT_LT(best_error, 8.0e-5);
}

TEST(P1ObjectiveContributions, CellContributionIsZeroForConstantsAndSkipsZeroIntegrationWeight) {
    const FixedLogGeometry geometry;
    const auto point = make_point<FixedLogGeometry>(first_coefficients);
    const std::vector<FixedLogGeometry::Point> nodes {point, point, point};
    auto packet = planar_packet<FixedLogGeometry>();
    packet.integration_weights[0] = 0;
    const auto result =
      gfe::p1_dirichlet_cell_contribution(geometry, std::span<const FixedLogGeometry::Point>(nodes), packet);
    ASSERT_TRUE(result.converged());
    EXPECT_DOUBLE_EQ(result.value, 0);
    for (std::size_t i = 0; i < packet.node_count; ++i) {
        EXPECT_DOUBLE_EQ(geometry.norm(nodes[packet.dofs[i]], result.nodal_gradient[i]), 0);
    }
}

TEST(P1ObjectiveContributions, ZeroIntegrationWeightSkipsAnAffineInvariantMeanFailure) {
    const FixedAffineGeometry geometry;
    const auto nodes = noncommuting_nodes<FixedAffineGeometry>();
    auto packet = surface_packet();
    packet.integration_weights = {0, 1};

    auto options = accurate_options();
    options.mean.solver.gradient_tolerance = 0;
    options.mean.solver.line_search.initial_step = 1.0e6;
    options.mean.solver.line_search.max_trials = 1;
    const auto result = gfe::p1_dirichlet_cell_contribution(
      geometry, std::span<const FixedAffineGeometry::Point>(nodes), packet, options);

    ASSERT_TRUE(result.converged());
    EXPECT_GT(result.value, 0);
}

TEST(P1ObjectiveContributions, SurfacesMeanAndLinearSolveFailuresWithExactStages) {
    const FixedAffineGeometry geometry;
    const auto nodes = noncommuting_nodes<FixedAffineGeometry>();
    const std::vector<double> weights {0.25, 0.5, 0.25};
    const auto observation = make_tangent<FixedAffineGeometry>({2.3, -0.1, 1.9, 0.2, 0.15, 3});

    auto failed_mean_options = accurate_options();
    failed_mean_options.mean.solver.gradient_tolerance = 0;
    failed_mean_options.mean.solver.line_search.initial_step = 1.0e6;
    failed_mean_options.mean.solver.line_search.max_trials = 1;
    const auto failed_mean = gfe::p1_frobenius_data_site_contribution(
      geometry, std::span<const FixedAffineGeometry::Point>(nodes), std::span<const double>(weights), observation,
      failed_mean_options);
    ASSERT_FALSE(failed_mean.converged());
    ASSERT_TRUE(failed_mean.first_failure);
    EXPECT_EQ(failed_mean.first_failure->stage, gfe::P1ObjectiveStage::mean);
    EXPECT_EQ(failed_mean.first_failure->site, 0);
    EXPECT_FALSE(failed_mean.first_failure->axis);
    ASSERT_TRUE(failed_mean.first_failure->barycenter_stop_reason);
    EXPECT_EQ(*failed_mean.first_failure->barycenter_stop_reason, manifold::BarycenterStopReason::line_search_failed);
    EXPECT_TRUE(std::isfinite(failed_mean.first_failure->stationarity_norm));

    auto failed_solve_options = accurate_options();
    failed_solve_options.linear_solve.max_iterations = 1;
    failed_solve_options.linear_solve.residual_tolerance = 0;
    const auto failed_data = gfe::p1_frobenius_data_site_contribution(
      geometry, std::span<const FixedAffineGeometry::Point>(nodes), std::span<const double>(weights), observation,
      failed_solve_options);
    ASSERT_FALSE(failed_data.converged());
    ASSERT_TRUE(failed_data.first_failure);
    EXPECT_EQ(failed_data.first_failure->stage, gfe::P1ObjectiveStage::data_pullback);
    ASSERT_TRUE(failed_data.first_failure->linear_solve);
    EXPECT_EQ(
      failed_data.first_failure->linear_solve->stop_reason, manifold::PositiveDefiniteCGStopReason::max_iterations);

    const auto packet = surface_packet();
    const auto failed_cell = gfe::p1_dirichlet_cell_contribution(
      geometry, std::span<const FixedAffineGeometry::Point>(nodes), packet, failed_solve_options);
    ASSERT_FALSE(failed_cell.converged());
    ASSERT_TRUE(failed_cell.first_failure);
    EXPECT_EQ(failed_cell.first_failure->stage, gfe::P1ObjectiveStage::dirichlet_spatial);
    EXPECT_EQ(failed_cell.first_failure->site, 0);
    ASSERT_TRUE(failed_cell.first_failure->axis);
    EXPECT_EQ(*failed_cell.first_failure->axis, 0);
    ASSERT_TRUE(failed_cell.first_failure->linear_solve);
    EXPECT_EQ(
      failed_cell.first_failure->linear_solve->stop_reason, manifold::PositiveDefiniteCGStopReason::max_iterations);
}

TEST(P1ObjectiveContributions, RejectsMalformedPacketsAndDynamicShapes) {
    const FixedLogGeometry geometry;
    const auto nodes = noncommuting_nodes<FixedLogGeometry>();
    auto packet = planar_packet<FixedLogGeometry>();

    packet.dofs[0] = nodes.size();
    EXPECT_THROW(
      gfe::p1_dirichlet_cell_contribution(geometry, std::span<const FixedLogGeometry::Point>(nodes), packet),
      std::out_of_range);
    packet = planar_packet<FixedLogGeometry>();
    packet.integration_weights[0] = std::numeric_limits<double>::quiet_NaN();
    EXPECT_THROW(
      gfe::p1_dirichlet_cell_contribution(geometry, std::span<const FixedLogGeometry::Point>(nodes), packet),
      std::invalid_argument);
    packet.integration_weights[0] = -1;
    EXPECT_THROW(
      gfe::p1_dirichlet_cell_contribution(geometry, std::span<const FixedLogGeometry::Point>(nodes), packet),
      std::invalid_argument);
    packet = planar_packet<FixedLogGeometry>();
    packet.physical_weight_gradients[0][0] = std::numeric_limits<double>::infinity();
    EXPECT_THROW(
      gfe::p1_dirichlet_cell_contribution(geometry, std::span<const FixedLogGeometry::Point>(nodes), packet),
      std::invalid_argument);
    packet = planar_packet<FixedLogGeometry>();
    packet.integration_weights[0] = 0;
    packet.barycentric_weights[0][0] += 0.1;
    EXPECT_THROW(
      gfe::p1_dirichlet_cell_contribution(geometry, std::span<const FixedLogGeometry::Point>(nodes), packet),
      std::invalid_argument);

    const DynamicLogGeometry dynamic_geometry(3);
    auto dynamic_nodes = noncommuting_nodes<DynamicLogGeometry>();
    native::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> wrong_dense(2, 2);
    wrong_dense.set_zero();
    wrong_dense(0, 0) = 1;
    wrong_dense(1, 1) = 1;
    dynamic_nodes[packet.dofs[0]] = DynamicLogGeometry::Point(wrong_dense, native::unchecked);
    EXPECT_THROW(
      gfe::p1_dirichlet_cell_contribution(
        dynamic_geometry, std::span<const DynamicLogGeometry::Point>(dynamic_nodes), packet),
      std::invalid_argument);

    auto wrong_observation = make_tangent<DynamicLogGeometry>({2.3, -0.1, 1.9, 0.2, 0.15, 3});
    wrong_observation.resize(2, 2);
    wrong_observation(0, 0) = 1;
    wrong_observation(1, 0) = 0;
    wrong_observation(1, 1) = 1;
    const std::vector<double> weights {0.25, 0.5, 0.25};
    const auto valid_dynamic_nodes = noncommuting_nodes<DynamicLogGeometry>();
    EXPECT_THROW(
      gfe::p1_frobenius_data_site_contribution(
        dynamic_geometry, std::span<const DynamicLogGeometry::Point>(valid_dynamic_nodes),
        std::span<const double>(weights), wrong_observation),
      std::invalid_argument);
}
