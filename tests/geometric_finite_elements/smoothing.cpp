// SPDX-License-Identifier: GPL-3.0-or-later
#include <fdaPDE/geometric_finite_elements.h>
#include <gtest/gtest.h>

namespace {
using namespace fdapde;
namespace gfe = fdapde::gfe;
/// @brief checks metric gradients by perturbing noncommuting nodes along their intrinsic geodesics
template <typename Geometry> void check_smoothing(const Geometry& geometry) {
    using Point = typename Geometry::Point;
    using Tangent = typename Geometry::Tangent;
    const int n = geometry.order();
    MatrixBatch<Point> data(3, n, n), plus(3, n, n), minus(3, n, n);
    std::vector<Tangent> directions;
    for (int i = 0; i < 3; ++i) {
        Tangent log, direction;
        if constexpr (Tangent::Rows == Dynamic) {
            log.resize(n, n);
            direction.resize(n, n);
        }
        for (int r = 0; r < n; ++r)
            for (int c = 0; c <= r; ++c) {
                log(r, c) = r == c ? .15 * (i + 1) * (r + 1) : .06 * (i - 1) * (r + c + 1);
                direction(r, c) = .13 * std::cos(1 + i + 2 * r + c);
            }
        data[i] = matrix_exp(log);
        directions.push_back(direction);
    }
    constexpr double h = 1e-5;
    for (int i = 0; i < 3; ++i) {
        plus[i] = geometry.exponential(data[i], directions[i], h);
        minus[i] = geometry.exponential(data[i], directions[i], -h);
    }
    Tangent response = geometry.zero_tangent(data[0]);
    for (int i = 0; i < n; ++i) response(i, i) = 1.1 + .2 * i;
    const std::array weights {.2, .3, .5};
    const gfe::P1LumpedLaplacianStencil stencil {
      {.7,           1.4,         .9         },
      {{0, 1, -1.2}, {0, 2, .35}, {1, 2, -.8}}
    };
    auto evaluate = [&](const auto& nodes, int kind) {
        if (kind == 0) return gfe::p1_discrete_tension_contribution(geometry, nodes, stencil);
        if (kind == 1) return gfe::p1_frobenius_data_site_contribution(geometry, nodes, weights, response);
        return gfe::p1_ambient_frobenius_data_site_contribution(geometry, nodes, weights, response);
    };
    for (int kind = 0; kind < 3; ++kind) {
        const auto result = evaluate(data, kind), p = evaluate(plus, kind), m = evaluate(minus, kind);
        // mean and adjoint solves must converge before their finite-difference oracle is meaningful
        ASSERT_TRUE(result.converged() && p.converged() && m.converged());
        double analytic = 0;
        for (int i = 0; i < 3; ++i)
            analytic += geometry.inner_product(data[i], result.nodal_gradient[i], directions[i]);
        const double numeric = (p.value - m.value) / (2 * h);
        // central geodesic perturbations independently check all nodal gradient contributions
        EXPECT_NEAR(analytic, numeric, 3e-6 * std::max(1., std::abs(numeric)));
    }
    const auto tension = gfe::p1_discrete_tension_contribution(geometry, data, stencil);
    // the value-only path must preserve the same normalization while skipping derivative preparation
    EXPECT_NEAR(tension.value, gfe::p1_discrete_tension_value(geometry, data, stencil).value, 1e-12);
    MatrixBatch<Point> constant(3, n, n);
    for (int i = 0; i < 3; ++i) constant[i] = data[0];
    // a constant field lies in the discrete Neumann tension nullspace
    EXPECT_NEAR(gfe::p1_discrete_tension_value(geometry, constant, stencil).value, 0, 1e-24);
    const auto selection = data.select(std::array {2, 0, 1});
    std::array<Point, 3> copy {Point(data[2]), Point(data[0]), Point(data[1])};
    const auto selected = gfe::p1_discrete_tension_contribution(geometry, selection, stencil);
    const auto legacy = gfe::p1_discrete_tension_contribution(geometry, std::span<const Point>(copy), stencil);
    // native selection borrowing and legacy spans agree under the same local ordering
    EXPECT_NEAR(selected.value, legacy.value, 1e-12);
    gfe::P1FEMCellQuadrature<2, 2, 1> packet {
      {0, 1, 2},
      {{{-1, 1, 0}, {-1, 0, 1}}},
      {{{.2, .3, .5}}},
      {{.5}}
    };
    const auto dirichlet = gfe::p1_dirichlet_cell_contribution(geometry, data, packet);
    const auto dp = gfe::p1_dirichlet_cell_value(geometry, plus, packet),
               dm = gfe::p1_dirichlet_cell_value(geometry, minus, packet);
    // mixed pullbacks and spatial derivatives must all report convergence
    ASSERT_TRUE(dirichlet.converged() && dp.converged() && dm.converged());
    double analytic = 0;
    for (int i = 0; i < 3; ++i) analytic += geometry.inner_product(data[i], dirichlet.nodal_gradient[i], directions[i]);
    // differentiating the integrated energy checks the existing mixed spatial-nodal pullbacks
    EXPECT_NEAR(analytic, (dp.value - dm.value) / (2 * h), 2e-6);
}
/// @brief exercises cached LE and AIRM kernels on fixed and dynamic SPD2 and SPD3 data
TEST(GeometricSmoothing, NativeMetricGradients) {
    check_smoothing(manifold::LogEuclideanSPDGeometry<double, 2, Usage::InterpolationNodes> {});
    check_smoothing(manifold::AffineInvariantSPDGeometry<double, 2, Usage::BasePointMaps> {});
    check_smoothing(manifold::LogEuclideanSPDGeometry<double, 3> {});
    check_smoothing(manifold::AffineInvariantSPDGeometry<double, Dynamic, Usage::BasePointMaps>(3));
}
/// @brief checks normalization, the independent log-coordinate Laplacian oracle and signed-edge validation
TEST(GeometricSmoothing, ScalarOracleAndContracts) {
    using Geometry = manifold::LogEuclideanSPDGeometry<double, 2>;
    Geometry geometry;
    MatrixBatch<Geometry::Point> data(3);
    for (int i = 0; i < 3; ++i) data[i] = Matrix<double, 2, 2>({std::exp(.2 * i), 0., 0., std::exp(-.1 * i)});
    const gfe::P1LumpedLaplacianStencil stencil {
      {1, 2, 1},
      {{0, 1, -1}, {1, 2, -1}}
    };
    const auto result = gfe::p1_discrete_tension_contribution(geometry, data, stencil);
    // K*(0,.2,.4) and K*(0,-.1,-.2) give two endpoint residuals and zero at the middle
    EXPECT_NEAR(result.value, .05, 1e-13);
    const auto edge = gfe::p1_squared_distance_edge_dirichlet_contribution(geometry, data, stencil);
    // two edges each contribute half the squared log-coordinate increment .2²+.1²
    EXPECT_NEAR(edge.value, .05, 1e-13);
    auto invalid = stencil;
    invalid.lumped_masses[1] = 0;
    // inverse lumped mass cannot accept unconnected or zero-measure DOFs
    EXPECT_THROW(gfe::p1_discrete_tension_value(geometry, data, invalid), std::invalid_argument);
    invalid = stencil;
    invalid.edges[0].stiffness = 1;
    // squared-distance edge energy requires nonpositive stiffness even though tension allows signed stencils
    EXPECT_THROW(gfe::p1_squared_distance_edge_dirichlet_value(geometry, data, invalid), std::invalid_argument);
    const auto observation = geometry.zero_tangent(data[0]);
    auto options = gfe::P1GeodesicLinearizationOptions {};
    options.mean.solver.max_iterations = 1;
    options.mean.solver.gradient_tolerance = 0;
    data[1] = Matrix<double, 2, 2>({2., .5, .5, 3.});
    const auto failed = gfe::p1_frobenius_data_site_contribution(
      manifold::AffineInvariantSPDGeometry<double, 2> {}, data, std::array {.2, .3, .5}, observation, options);
    // a failed inner mean is propagated to the outer objective instead of masquerading as a valid zero
    EXPECT_FALSE(failed.converged());
}
}   // namespace
