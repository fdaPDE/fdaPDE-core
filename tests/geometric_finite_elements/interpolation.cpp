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

namespace {
using namespace fdapde;
using Point = SPDMatrix<double, 2, 2>;
using Tangent = SymmetricMatrix<double, 2, 2>;
using Batch =
  MatrixBatch<SPDMatrix<double, 2, 2, Cache::Union<Cache::Log, Cache::Spectral, Cache::LogDividedDifferences>>>;
using LE = manifold::LogEuclideanSPDGeometry<double, 2, Usage::InterpolationNodes>;
using AIRM = manifold::AffineInvariantSPDGeometry<double, 2, Usage::BasePointMaps>;

/// @brief supplies noncommuting certified nodes with reusable logarithms and differentials
Batch nodes() {
    Batch result(3);
    result[0] = Matrix<double, 2, 2>({2., .5, .5, 3.});
    result[1] = Matrix<double, 2, 2>({5., -1., -1., 4.});
    result[2] = Matrix<double, 2, 2>({1.5, .2, .2, 2.5});
    return result;
}
/// @brief copies the lower triangle of a symmetric dense oracle into packed storage
Tangent tangent(const Matrix<double, 2, 2>& dense) { return Tangent(dense.as_symmetric<Lower>()); }
/// @brief measures complete coefficient error including both symmetric triangles
double error(const auto& a, const auto& b) { return Matrix<double, 2, 2>(a - b).norm(); }
/// @brief resolves the mean well below the finite-difference truncation error
gfe::P1GeodesicLinearizationOptions accurate() {
    gfe::P1GeodesicLinearizationOptions options;
    options.mean.solver.gradient_tolerance = 1e-11;
    options.linear_solve.residual_tolerance = 1e-12;
    return options;
}
/// @brief unwraps the exact LE action
const Tangent& action(const Tangent& value) { return value; }
/// @brief unwraps the iterative AIRM action after checking its residual certificate
const Tangent& action(const gfe::P1DerivativeResult<Tangent>& value) {
    // the solver certificate prevents comparing an unconverged derivative candidate
    EXPECT_TRUE(value.converged());
    return value.derivative;
}
/// @brief checks weight and nodal derivatives against centered differences and their metric adjoints
void check_derivatives(const auto& geometry) {
    const auto batch = nodes();
    const std::array<double, 3> weights {.2, .3, .5}, direction {-.7, .2, .5};
    const auto linearize = [&](const auto& data, const auto& w) {
        if constexpr (gfe::internals::is_log_euclidean_spd_geometry<std::remove_cvref_t<decltype(geometry)>>)
            return gfe::p1_geodesic_linearization(geometry, data, w);
        else
            return gfe::p1_geodesic_linearization(geometry, data, w, accurate());
    };
    const auto linearization = linearize(batch, weights);
    // the implicit derivative requires a converged mean at the evaluation point
    ASSERT_TRUE(linearization.result().converged());
    constexpr double step = 1e-4;
    auto plus_weights = weights, minus_weights = weights;
    for (int i = 0; i < 3; ++i) {
        plus_weights[i] += step * direction[i];
        minus_weights[i] -= step * direction[i];
    }
    const auto plus = linearize(batch, plus_weights), minus = linearize(batch, minus_weights);
    const Tangent finite_weight((plus.result().value - minus.result().value) / (2 * step));
    const auto weight_action = linearization.weight_jvp(direction);
    // centered spatial-weight differences independently check the chart or implicit differential
    EXPECT_LT(error(action(weight_action), finite_weight), 2e-7);
    std::array<Tangent, 3> directions {
      tangent(Matrix<double, 2, 2>({.3, -.1, -.1, .4})), tangent(Matrix<double, 2, 2>({-.2, .15, .15, .1})),
      tangent(Matrix<double, 2, 2>({.1, .2, .2, -.3}))};
    Batch plus_nodes(batch), minus_nodes(batch);
    for (int i = 0; i < 3; ++i) {
        plus_nodes[i] = geometry.exponential(batch[i], directions[i], step);
        minus_nodes[i] = geometry.exponential(batch[i], directions[i], -step);
    }
    const auto plus_nodal = linearize(plus_nodes, weights), minus_nodal = linearize(minus_nodes, weights);
    const Tangent finite_nodal((plus_nodal.result().value - minus_nodal.result().value) / (2 * step));
    const auto nodal_action = linearization.nodal_jvp(directions);
    // perturbing nodes along geodesics checks the ambient nodal JVP without reusing its formula
    EXPECT_LT(error(action(nodal_action), finite_nodal), 2e-7);
    const auto pullback = linearization.nodal_vjp(directions[0]);
    const auto& duals = [&]() -> const auto& {
        if constexpr (requires { pullback.derivative; })
            return pullback.derivative;
        else
            return pullback;
    }();
    double rhs = 0;
    for (int i = 0; i < 3; ++i) rhs += geometry.inner_product(batch[i], directions[i], duals[i]);
    // the adjoint equality uses each node's own Riemannian metric
    EXPECT_NEAR(geometry.inner_product(linearization.result().value, action(nodal_action), directions[0]), rhs, 2e-10);
    const auto mixed = linearization.covariant_mixed_nodal_jvp(direction, directions);
    const auto mixed_dual = linearization.covariant_mixed_nodal_vjp(direction, directions[0]);
    const auto& mixed_value = [&]() -> const auto& {
        if constexpr (requires { mixed.derivative; })
            return mixed.derivative;
        else
            return mixed;
    }();
    const auto& mixed_duals = [&]() -> const auto& {
        if constexpr (requires { mixed_dual.derivative; })
            return mixed_dual.derivative;
        else
            return mixed_dual;
    }();
    if constexpr (requires { mixed.converged(); }) {
        // all three dependent CG solves must converge before the mixed action is certified
        EXPECT_TRUE(mixed.converged());
        // the pullback has independent solves whose diagnostics must also indicate convergence
        EXPECT_TRUE(mixed_dual.converged());
    }
    rhs = 0;
    for (int i = 0; i < 3; ++i) rhs += geometry.inner_product(batch[i], directions[i], mixed_duals[i]);
    // the mixed metric adjoint exercises the second logarithm differential on noncommuting data
    EXPECT_NEAR(geometry.inner_product(linearization.result().value, mixed_value, directions[0]), rhs, 2e-9);
    const auto plus_weight = plus_nodal.weight_jvp(direction), minus_weight = minus_nodal.weight_jvp(direction);
    const auto transported_plus =
      geometry.transport(plus_nodal.result().value, linearization.result().value, action(plus_weight));
    const auto transported_minus =
      geometry.transport(minus_nodal.result().value, linearization.result().value, action(minus_weight));
    // parallel transport removes the connection term from the centered mixed finite difference
    EXPECT_LT(error(mixed_value, Tangent((transported_plus - transported_minus) / (2 * step))), 4e-7);
}

// cached noncommuting LE data retain exact first and mixed differential formulas
TEST(P1Interpolation, LogEuclideanDifferentials) { check_derivatives(LE {}); }
// candidate-relative AIRM caches preserve implicit first and mixed differentials
TEST(P1Interpolation, AffineInvariantDifferentials) { check_derivatives(AIRM {}); }

// candidate promotion preserves ready frames while reset prevents stale reuse at another point
TEST(P1Interpolation, RelativeWorkspaceGenerations) {
    const AIRM geometry;
    const auto batch = nodes();
    const std::array<double, 3> weights {.2, .3, .5};
    using Problem = manifold::internals::WeightedKarcherMeanProblem<AIRM, Batch>;
    Problem problem(geometry, batch, weights);
    manifold::EvaluationContext<Tangent, Problem::Workspace> context;
    const AIRM::Point base(batch[0]), next(batch[1]);
    const double cost = manifold::evaluate_cost(problem, geometry, base, context.trial());
    const auto* allocation = context.trial().workspace().frames.data();
    const auto generation = context.trial().generation();
    const auto gradient = manifold::evaluate_gradient(problem, geometry, base, context.trial());
    // cost and gradient share the same retained frame allocation for a fixed candidate
    EXPECT_EQ(allocation, context.trial().workspace().frames.data());
    context.promote_trial();
    // promotion moves the accepted candidate's frames without rebuilding the relative decompositions
    EXPECT_EQ(allocation, context.current().workspace().frames.data());
    // generation identity travels with the accepted point rather than being rebound to another point
    EXPECT_EQ(generation, context.current().generation());
    double oracle = 0;
    auto oracle_gradient = geometry.zero_tangent(base);
    for (int i = 0; i < 3; ++i) {
        const double distance = geometry.distance(base, batch[i]);
        oracle += .5 * weights[i] * distance * distance;
        oracle_gradient =
          geometry.linear_combination(base, 1, oracle_gradient, -weights[i], geometry.logarithm(base, batch[i]));
    }
    // the cached objective agrees with independently evaluated pairwise distances
    EXPECT_NEAR(cost, oracle, 2e-14);
    // the cached gradient agrees with independently evaluated logarithms
    EXPECT_LT(error(gradient, oracle_gradient), 2e-13);
    context.reset_current();
    // reset removes all frames before the caller binds the slot to another candidate
    EXPECT_TRUE(context.current().workspace().frames.empty());
    const auto next_cost = manifold::evaluate_cost(problem, geometry, next, context.current());
    // a fresh candidate must produce its own objective rather than the previous cached value
    EXPECT_GT(std::abs(cost - next_cost), .1);
}

// batch selections preserve vertex order and dynamic matrices match the fixed-size interpolant
TEST(P1Interpolation, SelectionDynamicAndCommutingOracle) {
    const auto batch = nodes();
    const std::array<double, 3> weights {.2, .3, .5}, permuted_weights {.5, .2, .3};
    const auto selection = batch.select(std::array {2, 0, 1});
    const auto result = gfe::p1_geodesic_value(AIRM {}, batch, weights, accurate().mean);
    const auto permuted = gfe::p1_geodesic_value(AIRM {}, selection, permuted_weights, accurate().mean);
    // simultaneously permuting local vertices and weights leaves the geometric value unchanged
    EXPECT_LT(error(result.value, permuted.value), 1e-10);
    MatrixBatch<SPDMatrix<double, Dynamic, Dynamic, Cache::Log>> dynamic(batch);
    const auto dynamic_result = gfe::p1_geodesic_value(
      manifold::AffineInvariantSPDGeometry<double, Dynamic>(2), dynamic, weights, accurate().mean);
    // runtime matrix order uses the same native algorithm as the fixed-order specialization
    EXPECT_LT(error(result.value, dynamic_result.value), 1e-10);
    Batch diagonal(3);
    for (int i = 0; i < 3; ++i) diagonal[i] = Matrix<double, 2, 2>({std::exp(double(i)), 0., 0., std::exp(-double(i))});
    const auto commuting = gfe::p1_geodesic_value(AIRM {}, diagonal, weights, accurate().mean);
    // commuting SPD data reduce to independent scalar geometric means
    EXPECT_LT(error(commuting.value, Matrix<double, 2, 2>({std::exp(1.3), 0., 0., std::exp(-1.3)})), 1e-12);
}

// invalid weights fail at the public boundary and failed means cannot be used as derivative certificates
TEST(P1Interpolation, InvalidDataAndFailedSolves) {
    const auto batch = nodes();
    const std::array<double, 3> negative {-.1, .5, .6}, bad_sum {.2, .3, .6}, weights {.2, .3, .5};
    // negative barycentric coordinates are outside the supported convex interpolation domain
    EXPECT_THROW(gfe::p1_geodesic_value(LE {}, batch, negative), std::invalid_argument);
    // P1 weights must sum to one before the mean normalizes representational rounding
    EXPECT_THROW(gfe::p1_geodesic_value(AIRM {}, batch, bad_sum), std::invalid_argument);
    auto options = accurate();
    options.mean.solver.max_iterations = 1;
    options.mean.solver.gradient_tolerance = 0;
    const auto linearization = gfe::p1_geodesic_linearization(AIRM {}, batch, weights, options);
    // one iteration cannot certify this noncommuting mean at zero tolerance
    EXPECT_FALSE(linearization.result().converged());
    // derivative evaluation rejects a failed mean instead of returning the last iterate as certified
    EXPECT_THROW(linearization.weight_jvp(std::array {-1., 1., 0.}), std::logic_error);
    const auto edge = gfe::p1_geodesic_linearization(AIRM {}, batch, std::array {.6, .4, 0.}, accurate());
    const auto transverse = edge.weight_jvp(std::array {-.5, -.5, 1.});
    // a transverse direction at an edge still includes the inactive third node in the full linearization
    EXPECT_TRUE(transverse.converged());
    // the inactive node's logarithm gives a nonzero transverse action
    EXPECT_GT((Matrix<double, 2, 2>(transverse.derivative).norm()), .1);
}

// second logarithm differentials remain finite and exact at a repeated spectrum
TEST(P1Interpolation, RepeatedSpectrumSecondFrechet) {
    const Point identity = Point::Identity();
    const Tangent a = tangent(Matrix<double, 2, 2>({.2, .4, .4, -.3})),
                  b = tangent(Matrix<double, 2, 2>({-.1, .2, .2, .5}));
    const auto actual = matrix_log_second_frechet(identity, a, b);
    const Tangent expected = tangent(Matrix<double, 2, 2>(-.5 * (a * b + b * a)));
    // the second logarithm differential at identity is the negative symmetrized product, independent of eigenbasis
    // choice
    EXPECT_LT(error(actual, expected), 1e-14);
}
}   // namespace
