// SPDX-License-Identifier: GPL-3.0-or-later
#include <fdaPDE/execution.h>
#include <fdaPDE/geometric_finite_elements.h>
#include <fdaPDE/geometry.h>
#include <gtest/gtest.h>

namespace {
using namespace fdapde;
using LC = manifold::LogCholeskySPDGeometry<double, 2, Usage::InterpolationNodes>;

/// @brief computes a symmetric coordinate chart with a scalar Cholesky recurrence independent of the core cache
SymmetricMatrix<double, Dynamic> oracle_chart(const auto& point) {
    const int n = point.rows();
    Matrix<double, Dynamic, Dynamic> lower(n, n);
    lower.set_zero();
    SymmetricMatrix<double, Dynamic> result(n, n);
    for (int row = 0; row < n; ++row) {
        for (int col = 0; col <= row; ++col) {
            double entry = point(row, col);
            for (int k = 0; k < col; ++k) entry -= lower(row, k) * lower(col, k);
            lower(row, col) = row == col ? std::sqrt(entry) : entry / lower(col, col);
            result(row, col) = row == col ? std::log(lower(row, col)) : lower(row, col) / std::sqrt(2.);
        }
    }
    return result;
}

/// @brief reconstructs the complete SPD coefficients from an independent lower-triangular chart
Matrix<double, Dynamic, Dynamic> oracle_inverse(const auto& chart) {
    const int n = chart.rows();
    Matrix<double, Dynamic, Dynamic> lower(n, n), result(n, n);
    lower.set_zero();
    result.set_zero();
    for (int row = 0; row < n; ++row)
        for (int col = 0; col <= row; ++col)
            lower(row, col) = row == col ? std::exp(chart(row, col)) : std::sqrt(2.) * chart(row, col);
    for (int row = 0; row < n; ++row)
        for (int col = 0; col < n; ++col)
            for (int k = 0; k <= std::min(row, col); ++k) result(row, col) += lower(row, k) * lower(col, k);
    return result;
}

/// @brief differentiates the independent triangular reconstruction by scalar products
Matrix<double, Dynamic, Dynamic> oracle_inverse_jvp(const auto& chart, const auto& direction) {
    const int n = chart.rows();
    Matrix<double, Dynamic, Dynamic> lower(n, n), derivative(n, n), result(n, n);
    lower.set_zero();
    derivative.set_zero();
    result.set_zero();
    for (int row = 0; row < n; ++row)
        for (int col = 0; col <= row; ++col) {
            lower(row, col) = row == col ? std::exp(chart(row, col)) : std::sqrt(2.) * chart(row, col);
            derivative(row, col) =
              row == col ? lower(row, col) * direction(row, col) : std::sqrt(2.) * direction(row, col);
        }
    for (int row = 0; row < n; ++row)
        for (int col = 0; col < n; ++col)
            for (int k = 0; k <= std::min(row, col); ++k)
                result(row, col) += derivative(row, k) * lower(col, k) + lower(row, k) * derivative(col, k);
    return result;
}

/// @brief obtains a promoted zero symmetric matrix without depending on the geometry chart
SymmetricMatrix<double, Dynamic> zero_chart(int n) {
    SymmetricMatrix<double, Dynamic> result(n, n);
    for (int row = 0; row < n; ++row)
        for (int col = 0; col <= row; ++col) result(row, col) = 0;
    return result;
}

/// @brief measures error over both symmetric triangles without borrowing expression temporaries
double error(const auto& first, const auto& second) { return Matrix<double, Dynamic, Dynamic>(first - second).norm(); }

/// @brief builds noncommuting certified samples from independently prescribed lower-triangular charts
template <typename Geometry> MatrixBatch<typename Geometry::Point> data(const Geometry& geometry) {
    MatrixBatch<typename Geometry::Point> result(3, geometry.order(), geometry.order());
    for (int node = 0; node < 3; ++node) {
        auto chart = zero_chart(geometry.order());
        for (int row = 0; row < geometry.order(); ++row)
            for (int col = 0; col <= row; ++col)
                chart(row, col) = row == col ? .1 * (node + 1) * (row + 1) : .08 * (node - 1) * (1 + row + col);
        result[node] = oracle_inverse(chart);
    }
    return result;
}

/// @brief averages independently computed charts in the represented nodal order
SymmetricMatrix<double, Dynamic> mean_chart(const auto& nodes, std::span<const double> weights) {
    auto result = zero_chart(nodes[0].rows());
    double total = 0;
    for (std::size_t node = 0; node < nodes.size(); ++node) {
        const auto chart = oracle_chart(nodes[node]);
        total += weights[node];
        for (int row = 0; row < result.rows(); ++row)
            for (int col = 0; col <= row; ++col)
                result(row, col) = static_cast<double>(result(row, col)) + weights[node] * chart(row, col);
    }
    for (int row = 0; row < result.rows(); ++row)
        for (int col = 0; col <= row; ++col) result(row, col) = static_cast<double>(result(row, col)) / total;
    return result;
}

/// @brief checks closed P1 means and differential adjoints using scalar triangular and independent perturbation oracles
template <typename Geometry> void check_p1(const Geometry& geometry) {
    using Tangent = typename Geometry::Tangent;
    const auto nodes = data(geometry);
    const std::array weights {.2, .3, .5}, weight_direction {-.7, .2, .5};
    const auto chart = mean_chart(nodes, weights);
    const auto fit = gfe::p1_geodesic_linearization(geometry, nodes, weights);
    // a flat mean carries the exact closed-form certificate rather than an iterative stopping candidate
    ASSERT_EQ(fit.result().stop_reason, manifold::BarycenterStopReason::closed_form);
    // independent scalar Cholesky coordinates reproduce all noncommuting barycenter coefficients
    EXPECT_LT(error(fit.result().value, oracle_inverse(chart)), 3e-14);
    auto spatial_chart = zero_chart(geometry.order());
    for (int node = 0; node < 3; ++node) {
        const auto node_chart = oracle_chart(nodes[node]);
        for (int row = 0; row < geometry.order(); ++row)
            for (int col = 0; col <= row; ++col)
                spatial_chart(row, col) =
                  static_cast<double>(spatial_chart(row, col)) + weight_direction[node] * node_chart(row, col);
    }
    // the spatial differential uses the inverse triangular chart action on the independent weighted chart direction
    EXPECT_LT(error(fit.weight_jvp(weight_direction), oracle_inverse_jvp(chart, spatial_chart)), 5e-14);
    constexpr double step = 1e-5;
    auto chart_direction = zero_chart(geometry.order()), mixed_chart_direction = zero_chart(geometry.order());
    std::vector<Tangent> directions;
    for (int node = 0; node < 3; ++node) {
        Tangent direction = geometry.zero_tangent(nodes[node]);
        for (int row = 0; row < geometry.order(); ++row)
            for (int col = 0; col <= row; ++col) direction(row, col) = .13 * std::cos(1 + node + 2 * row + col);
        directions.push_back(direction);
        const typename Geometry::Point plus(nodes[node] + step * direction), minus(nodes[node] - step * direction);
        const auto p = oracle_chart(plus), m = oracle_chart(minus);
        for (int row = 0; row < geometry.order(); ++row)
            for (int col = 0; col <= row; ++col) {
                const double action = (p(row, col) - m(row, col)) / (2 * step);
                chart_direction(row, col) = static_cast<double>(chart_direction(row, col)) + weights[node] * action;
                mixed_chart_direction(row, col) =
                  static_cast<double>(mixed_chart_direction(row, col)) + weight_direction[node] * action;
            }
    }
    const auto nodal = fit.nodal_jvp(directions);
    // centered ambient perturbations of independent scalar Cholesky charts verify the nodal differential
    EXPECT_LT(error(nodal, oracle_inverse_jvp(chart, chart_direction)), 2e-10);
    const auto pullback = fit.nodal_vjp(directions[0]);
    double paired = 0;
    for (int node = 0; node < 3; ++node)
        paired += geometry.inner_product(nodes[node], directions[node], pullback[node]);
    // the nodal pullback pairs each input at its own metric base and the output at the mean
    EXPECT_NEAR(geometry.inner_product(fit.result().value, nodal, directions[0]), paired, 2e-14);
    const auto mixed = fit.covariant_mixed_nodal_jvp(weight_direction, directions);
    // flat covariant differentiation removes the connection term and differentiates only the nodal chart forcing
    EXPECT_LT(error(mixed, oracle_inverse_jvp(chart, mixed_chart_direction)), 2e-10);
    const auto mixed_pullback = fit.covariant_mixed_nodal_vjp(weight_direction, directions[0]);
    paired = 0;
    for (int node = 0; node < 3; ++node)
        paired += geometry.inner_product(nodes[node], directions[node], mixed_pullback[node]);
    // the mixed pullback is the metric adjoint of the covariant mixed nodal action
    EXPECT_NEAR(geometry.inner_product(fit.result().value, mixed, directions[0]), paired, 2e-14);
    const auto selected = nodes.select(std::array {2, 0, 1});
    const auto permuted = gfe::p1_geodesic_value(geometry, selected, std::array {.5, .2, .3});
    // borrowed batch selections preserve the same mean when weights follow their local ordering
    EXPECT_LT(error(permuted.value, fit.result().value), 2e-14);
}

/// @brief checks full data and smoothing gradients against independent ambient coefficient perturbations
template <typename Geometry> void check_objectives(const Geometry& geometry) {
    using Tangent = typename Geometry::Tangent;
    const auto nodes = data(geometry);
    auto plus = nodes, minus = nodes;
    std::vector<Tangent> directions;
    constexpr double step = 1e-5;
    for (int node = 0; node < 3; ++node) {
        Tangent direction = geometry.zero_tangent(nodes[node]);
        for (int row = 0; row < geometry.order(); ++row)
            for (int col = 0; col <= row; ++col) direction(row, col) = .1 * std::sin(1 + node + row + 2 * col);
        directions.push_back(direction);
        plus[node] = typename Geometry::Point(nodes[node] + step * direction);
        minus[node] = typename Geometry::Point(nodes[node] - step * direction);
    }
    const std::array weights {.2, .3, .5};
    const gfe::P1LumpedLaplacianStencil signed_stencil {
      {.7,           1.4,         .9         },
      {{0, 1, -1.2}, {0, 2, .35}, {1, 2, -.8}}
    };
    const gfe::P1LumpedLaplacianStencil edge_stencil {
      {.7,           1.4,          .9         },
      {{0, 1, -1.2}, {0, 2, -.35}, {1, 2, -.8}}
    };
    Tangent response = geometry.zero_tangent(nodes[0]);
    for (int row = 0; row < geometry.order(); ++row) response(row, row) = 1.2 + .2 * row;
    const Tangent response_chart(oracle_chart(typename Geometry::Point(response)));
    const gfe::P1FEMCellQuadrature<2, 2, 1> packet {
      {0, 1, 2},
      {{{-1, 1, 0}, {-1, 0, 1}}},
      {{{.2, .3, .5}}},
      {{.5}}
    };
    const auto evaluate = [&](const auto& sample, int kind) {
        if (kind == 0) return gfe::p1_discrete_tension_contribution(geometry, sample, signed_stencil);
        if (kind == 1) return gfe::p1_frobenius_data_site_contribution(geometry, sample, weights, response);
        if (kind == 2) return gfe::p1_ambient_frobenius_data_site_contribution(geometry, sample, weights, response);
        if (kind == 3) return gfe::p1_flat_coordinate_data_site_contribution(geometry, sample, weights, response_chart);
        if (kind == 4) return gfe::p1_squared_distance_edge_dirichlet_contribution(geometry, sample, edge_stencil);
        return gfe::p1_dirichlet_cell_contribution(geometry, sample, packet);
    };
    for (int kind = 0; kind < 6; ++kind) {
        const auto result = evaluate(nodes, kind), p = evaluate(plus, kind), m = evaluate(minus, kind);
        // all complete objective paths must return valid certificates before their gradients can be consumed
        ASSERT_TRUE(result.converged() && p.converged() && m.converged());
        double derivative = 0;
        for (int node = 0; node < 3; ++node)
            derivative += geometry.inner_product(nodes[node], result.nodal_gradient[node], directions[node]);
        const double finite = (p.value - m.value) / (2 * step);
        // central ambient node perturbations check each complete metric gradient including global and cell-local order
        EXPECT_NEAR(derivative, finite, 3e-9 * std::max(1., std::abs(finite)));
    }
    std::vector<SymmetricMatrix<double, Dynamic>> residuals(3, zero_chart(geometry.order()));
    for (const auto& edge : signed_stencil.edges) {
        const auto first = oracle_chart(nodes[edge.first]), second = oracle_chart(nodes[edge.second]);
        for (int row = 0; row < geometry.order(); ++row)
            for (int col = 0; col <= row; ++col) {
                const double difference = edge.stiffness * (second(row, col) - first(row, col));
                residuals[edge.first](row, col) = static_cast<double>(residuals[edge.first](row, col)) + difference;
                residuals[edge.second](row, col) = static_cast<double>(residuals[edge.second](row, col)) - difference;
            }
    }
    double oracle = 0;
    for (int node = 0; node < 3; ++node)
        oracle += .5 * std::pow(residuals[node].norm(), 2) / signed_stencil.lumped_masses[node];
    // scalar Cholesky charts and signed stiffness assembly verify the tension normalization independently
    EXPECT_NEAR(gfe::p1_discrete_tension_value(geometry, nodes, signed_stencil).value, oracle, 3e-14);
    const auto selected = nodes.select(std::array {2, 0, 1});
    const std::array<typename Geometry::Point, 3> copied {
      typename Geometry::Point(nodes[2]), typename Geometry::Point(nodes[0]), typename Geometry::Point(nodes[1])};
    // cached selected nodes agree with legacy owning snapshots under the same reordered stencil
    EXPECT_NEAR(
      gfe::p1_discrete_tension_value(geometry, selected, signed_stencil).value,
      gfe::p1_discrete_tension_value(geometry, std::span<const typename Geometry::Point>(copied), signed_stencil).value,
      3e-14);
}

// fixed and dynamic Log-Cholesky P1 means share exact flat derivatives on noncommuting SPD2 and SPD3 samples
TEST(LogCholeskyGFE, ClosedP1Differentials) {
    // fixed SPD2 exercises the cached-frame branch against scalar Cholesky means and perturbation differentials
    check_p1(LC {});
    // fixed SPD3 verifies that the same flat pullbacks retain all six independent tangent coefficients
    check_p1(manifold::LogCholeskySPDGeometry<double, 3> {});
    // runtime SPD3 checks resized workspaces against the same independent coordinate and metric-adjoint oracles
    check_p1(manifold::LogCholeskySPDGeometry<double, Dynamic>(3));
}

// every approved data and smoothing contribution retains its metric-gradient contract in the new flat geometry
TEST(LogCholeskyGFE, ObjectiveGradientsAndIndependentTension) {
    // fixed SPD2 checks complete objective gradients by central ambient refits and a scalar signed-stencil oracle
    check_objectives(LC {});
    // fixed SPD3 verifies the same global and packet-local gradients beyond the special two-dimensional kernels
    check_objectives(manifold::LogCholeskySPDGeometry<double, 3> {});
    // runtime SPD3 checks dynamic storage with the identical perturbation and independent tension mechanisms
    check_objectives(manifold::LogCholeskySPDGeometry<double, Dynamic>(3));
}

// spatial interpolation preserves exact vertices, edge restrictions, batch cache refresh and concurrent evaluation
TEST(LogCholeskyGFE, SpatialBindingAndCacheReplacement) {
    const LC geometry;
    auto nodes = data(geometry);
    const auto selection = nodes.select(std::array {2, 0, 1});
    const auto interpolation = geometry.interpolant(Simplex<2, 2>::Unit(), selection);
    const Eigen::Vector2d x(.2, .3);
    const std::array weights {.5, .2, .3};
    const LC::Point actual(interpolation(x));
    // spatial barycentric evaluation follows the reordered batch binding used by the independent triangular mean
    EXPECT_LT(error(actual, oracle_inverse(mean_chart(selection, weights))), 3e-14);
    // a vertex returns the exact selected nodal matrix without chart round trips
    EXPECT_EQ(error(LC::Point(interpolation(Eigen::Vector2d(0., 0.))), selection[0]), 0.);
    const LC::Point edge(interpolation(Eigen::Vector2d(.4, 0.)));
    // a prepared edge is the same globally flat two-node interpolation as its independent weighted chart
    EXPECT_LT(error(edge, oracle_inverse(mean_chart(selection, std::array {.6, .4, 0.}))), 3e-14);
    auto first = parallel_async([&] { return LC::Point(interpolation(x)); });
    auto second = parallel_async([&] { return LC::Point(interpolation(x)); });
    const LC::Point a = first.get(), b = second.get();
    parallel_join();
    // independent evaluation workspaces share immutable batch caches without changing the resulting coefficients
    EXPECT_EQ(error(a, b), 0.);
    const auto before = gfe::p1_geodesic_value(geometry, selection, weights);
    nodes[0] = Matrix<double, 2, 2>({3., -.2, -.2, 2.});
    const auto after = gfe::p1_geodesic_value(geometry, selection, weights);
    // a fresh mean reads updated owner coefficients through a live selection instead of retaining a stale chart
    EXPECT_LT(error(after.value, oracle_inverse(mean_chart(selection, weights))), 3e-14);
    // replacing an active selected coefficient has a detectable effect and cannot silently reuse its old chart
    EXPECT_GT(error(after.value, before.value), .05);
}

// existing log-Euclidean callers may explicitly supply scalar, order and cache-usage template arguments
TEST(LogCholeskyGFE, LogEuclideanExplicitTemplateCompatibility) {
    using LE = manifold::LogEuclideanSPDGeometry<double, 2, Usage::InterpolationNodes>;
    const LE geometry;
    const auto nodes = data(geometry);
    const std::array weights {.2, .3, .5};
    const auto original_call = gfe::p1_geodesic_value<double, 2, Usage::InterpolationNodes>(geometry, nodes, weights);
    const auto inferred_call = gfe::p1_geodesic_value(geometry, nodes, weights);
    // the original explicit public signature resolves the same closed mean as ordinary geometry deduction
    EXPECT_EQ(error(original_call.value, inferred_call.value), 0.);
}

// invalid weights, directions and signed-edge domains are rejected at the public interpolation boundary
TEST(LogCholeskyGFE, PublicBoundaryContracts) {
    const LC geometry;
    const auto nodes = data(geometry);
    // convex P1 interpolation rejects a negative barycentric coefficient before preparing a mean
    EXPECT_THROW(gfe::p1_geodesic_value(geometry, nodes, std::array {-.1, .5, .6}), std::invalid_argument);
    // a barycentric total outside the rounding tolerance cannot be normalized into an accepted P1 value
    EXPECT_THROW(gfe::p1_geodesic_value(geometry, nodes, std::array {.2, .3, .6}), std::invalid_argument);
    const auto fit = gfe::p1_geodesic_linearization(geometry, nodes, std::array {.2, .3, .5});
    // a weight differential must stay in the zero-sum barycentric tangent space
    EXPECT_THROW(fit.weight_jvp(std::array {.1, .2, .3}), std::invalid_argument);
    // a nodal differential must provide one ambient tangent for each retained coefficient
    EXPECT_THROW(fit.nodal_jvp(std::span<const LC::Tangent> {}), std::invalid_argument);
    const gfe::P1LumpedLaplacianStencil invalid {
      {1., 0., 1.},
      {{0, 1, -1.}, {1, 2, -1.}}
    };
    // tension cannot form an inverse lumped mass at a zero-measure degree of freedom
    EXPECT_THROW(gfe::p1_discrete_tension_value(geometry, nodes, invalid), std::invalid_argument);
    const gfe::P1LumpedLaplacianStencil signed_edge {
      {1., 1., 1.},
      {{0, 1, .1}, {1, 2, -1.}}
    };
    // edge Dirichlet energy retains its negative off-diagonal requirement independently of the flat geometry
    EXPECT_THROW(gfe::p1_squared_distance_edge_dirichlet_value(geometry, nodes, signed_edge), std::invalid_argument);
}
}   // namespace
