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
using Geometry = manifold::SOGeometry<double, Dynamic>;
using Point = Geometry::Point;
using Tangent = Geometry::Tangent;
using Dense = Matrix<double, Dynamic, Dynamic>;
using Batch = MatrixBatch<RotationMatrix<double, Dynamic, Dynamic, RotationCache::Log>>;
/// @brief supplies deterministic noncommuting body tangents across arbitrary rotation orders
Tangent direction(int n, double seed) {
    Tangent result(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = i + 1; j < n; ++j) result(i, j) = .25 * std::sin(seed + 2 * i + j);
    return result;
}
/// @brief compares packed tangent actions through their complete dense coefficients
double error(const auto& a, const auto& b) { return Dense(a - b).norm(); }
/// @brief resolves the mean sufficiently for independent centered derivative checks
gfe::P1GeodesicLinearizationOptions options() {
    gfe::P1GeodesicLinearizationOptions result;
    result.mean.solver.gradient_tolerance = 1e-9;
    result.linear_solve.residual_tolerance = 1e-12;
    return result;
}
// relative geometry operators match endpoint perturbations, covariant transport and metric adjoints
TEST(SODifferential, AnalyticOperatorsAndAdjoints) {
    for (int n : {1, 2, 3, 4, 5}) {
        const Geometry g(n);
        const auto identity = Point::Identity(n);
        const auto q = g.exponential(identity, direction(n, .3));
        const auto r = g.exponential(identity, direction(n, 1.7));
        const auto frame = g.relative_frame(q, r);
        const auto u = direction(n, .8), v = direction(n, 2.1), w = direction(n, 1.1), z = direction(n, 3.1);
        constexpr double step = 1e-4;
        const auto qp = g.exponential(q, u, step), qm = g.exponential(q, u, -step);
        const auto rp = g.exponential(r, v, step), rm = g.exponential(r, v, -step);
        const auto j = g.logarithm_target_jvp(frame, v);
        const auto finite_j = g.linear_combination(q, .5 / step, g.logarithm(q, rp), -.5 / step, g.logarithm(q, rm));
        // moving only the target checks the logarithm differential independently of its analytic kernel
        EXPECT_LT(error(j, finite_j), 2e-8);
        const auto h = g.half_squared_distance_hessian_vector(frame, u);
        const auto lp = g.transport(qp, q, g.logarithm(qp, r));
        const auto lm = g.transport(qm, q, g.logarithm(qm, r));
        const auto finite_h = g.linear_combination(q, -.5 / step, lp, .5 / step, lm);
        // parallel transport converts differences of negative gradients into the covariant Hessian
        EXPECT_LT(error(h, finite_h), 2e-8);
        const auto j_dual = g.logarithm_target_vjp(frame, z);
        // the target pullback is the adjoint in the full-Frobenius body metric
        EXPECT_NEAR(g.inner_product(q, z, j), g.inner_product(r, j_dual, v), 2e-12);
        const auto mixed = g.half_squared_distance_hessian_covariant_jvp(frame, u, v, w);
        const auto hp =
          g.transport(qp, q, g.half_squared_distance_hessian_vector(g.relative_frame(qp, rp), g.transport(q, qp, w)));
        const auto hm =
          g.transport(qm, q, g.half_squared_distance_hessian_vector(g.relative_frame(qm, rm), g.transport(q, qm, w)));
        const auto finite_mixed = g.linear_combination(q, .5 / step, hp, -.5 / step, hm);
        // perturbing both endpoints and transporting both input and output checks the connection terms
        EXPECT_LT(error(mixed, finite_mixed), 3e-8);
        const auto dual = g.half_squared_distance_hessian_covariant_vjp(frame, w, z);
        // pairing the two endpoint pullbacks checks the reverse Frechet and commutator contractions
        EXPECT_NEAR(
          g.inner_product(q, z, mixed), g.inner_product(q, u, dual.first) + g.inner_product(r, v, dual.second), 3e-12);
    }
}
// the implicit mean derivatives agree with perturbed noncommuting nodal rotations and weight changes
TEST(SOInterpolation, ImplicitAndMixedDerivatives) {
    for (int n : {1, 2, 3, 4}) {
        const Geometry g(n);
        const auto identity = Point::Identity(n);
        Batch nodes(3, n, n);
        for (int i = 0; i < 3; ++i) nodes[i] = g.exponential(identity, direction(n, .6 + i));
        const std::array<double, 3> weights {.2, .3, .5}, dw {-.6, .2, .4};
        const auto lin = gfe::p1_geodesic_linearization(g, nodes, weights, options());
        // implicit differentiation requires a converged mean before evaluating any action
        ASSERT_TRUE(lin.result().converged()) << lin.result().stationarity_norm;
        // positive curvature of SO does not justify a global uniqueness certificate
        EXPECT_EQ(lin.result().uniqueness, manifold::BarycenterUniqueness::not_certified);
        const auto& mean = lin.result().value;
        constexpr double step = 2e-4;
        auto wp = weights, wm = weights;
        for (int i = 0; i < 3; ++i) {
            wp[i] += step * dw[i];
            wm[i] -= step * dw[i];
        }
        const auto mp = gfe::p1_geodesic_value(g, nodes, wp, options().mean);
        const auto mm = gfe::p1_geodesic_value(g, nodes, wm, options().mean);
        // both independently perturbed means must satisfy their stationarity tolerances
        ASSERT_TRUE(mp.converged() && mm.converged());
        const auto weight = lin.weight_jvp(dw);
        // the CG residual certificate is required before using the weight derivative
        ASSERT_TRUE(weight.converged());
        const auto finite_weight =
          g.linear_combination(mean, .5 / step, g.logarithm(mean, mp.value), -.5 / step, g.logarithm(mean, mm.value));
        // centered weight perturbations check the implicit Hessian solve
        EXPECT_LT(error(weight.derivative, finite_weight), 8e-6);
        const std::array<Tangent, 3> directions {direction(n, 1.1), direction(n, 2.2), direction(n, 3.3)};
        Batch np(nodes), nm(nodes);
        for (int i = 0; i < 3; ++i) {
            np[i] = g.exponential(nodes[i], directions[i], step);
            nm[i] = g.exponential(nodes[i], directions[i], -step);
        }
        const auto lp = gfe::p1_geodesic_linearization(g, np, weights, options());
        const auto lm = gfe::p1_geodesic_linearization(g, nm, weights, options());
        // both perturbed means must converge before they can serve as derivative oracles
        ASSERT_TRUE(lp.result().converged() && lm.result().converged());
        const auto nodal = lin.nodal_jvp(directions);
        const auto finite_nodal = g.linear_combination(
          mean, .5 / step, g.logarithm(mean, lp.result().value), -.5 / step, g.logarithm(mean, lm.result().value));
        // centered nodal geodesic perturbations check the target-logarithm forcing
        EXPECT_LT(error(nodal.derivative, finite_nodal), 8e-6);
        const auto z = direction(n, 1.4);
        const auto pullback = lin.nodal_vjp(z);
        // forward and reverse nodal actions must both satisfy their linear-solve certificates
        ASSERT_TRUE(nodal.converged() && pullback.converged());
        double pairing = 0;
        for (int i = 0; i < 3; ++i) pairing += g.inner_product(nodes[i], directions[i], pullback.derivative[i]);
        // the implicit nodal pullback uses the same self-adjoint mean Hessian
        EXPECT_NEAR(g.inner_product(mean, z, nodal.derivative), pairing, 2e-11);
        const auto mixed = lin.covariant_mixed_nodal_jvp(dw, directions);
        const auto mixed_dual = lin.covariant_mixed_nodal_vjp(dw, z);
        // all dependent systems must converge for both mixed directions
        ASSERT_TRUE(mixed.converged() && mixed_dual.converged());
        const auto plus = g.transport(lp.result().value, mean, lp.weight_jvp(dw).derivative);
        const auto minus = g.transport(lm.result().value, mean, lm.weight_jvp(dw).derivative);
        const auto finite_mixed = g.linear_combination(mean, .5 / step, plus, -.5 / step, minus);
        // transported weight derivatives independently check the mixed spatial-nodal action
        EXPECT_LT(error(mixed.derivative, finite_mixed), 8e-6);
        pairing = 0;
        for (int i = 0; i < 3; ++i) pairing += g.inner_product(nodes[i], directions[i], mixed_dual.derivative[i]);
        // the mixed pullback agrees with the independently evaluated forward action
        EXPECT_NEAR(g.inner_product(mean, z, mixed.derivative), pairing, 2e-11);
    }
}
// commuting rotations supply closed-form value and derivative oracles including fixed orders and cache selections
TEST(SOInterpolation, CommutingFixedOrderAndSelections) {
    using Fixed = manifold::SOGeometry<double, 2, RotationUsage::IdentityLog>;
    const Fixed g;
    const auto identity = Fixed::Point::Identity();
    MatrixBatch<Fixed::Point> nodes(4);
    const std::array<double, 4> angles {-.4, .6, .2, .8};
    for (int i = 0; i < 4; ++i) {
        Fixed::Tangent t;
        t(0, 1) = angles[i];
        nodes[i] = g.exponential(identity, t);
    }
    const std::array<int, 3> ids {2, 0, 3};
    const std::array<double, 3> weights {.2, .3, .5}, dw {-.5, .2, .3};
    const auto lin = gfe::p1_geodesic_linearization(g, nodes.select(ids), weights, options());
    Fixed::Tangent mean, derivative;
    for (int i = 0; i < 3; ++i) {
        mean(0, 1) = double(mean(0, 1)) + weights[i] * angles[ids[i]];
        derivative(0, 1) = double(derivative(0, 1)) + dw[i] * angles[ids[i]];
    }
    // rotations on the same regular plane reduce to the scalar weighted angle
    EXPECT_LT(error(lin.result().value, g.exponential(identity, mean)), 2e-12);
    // the implicit weight derivative reduces to the scalar weighted angle direction
    EXPECT_LT(error(lin.weight_jvp(dw).derivative, derivative), 2e-12);
    const std::array<Fixed::Point, 3> owned {Fixed::Point(nodes[2]), Fixed::Point(nodes[0]), Fixed::Point(nodes[3])};
    const auto snapshot = gfe::p1_geodesic_linearization(g, std::span<const Fixed::Point>(owned), weights, options());
    // the legacy span entry point owns the same rotation data as the selection-based path
    EXPECT_LT(error(snapshot.result().value, lin.result().value), 2e-12);
    const std::array<double, 3> vertex {0, 1, 0};
    const auto vertex_lin = gfe::p1_geodesic_linearization(g, nodes.select(ids), vertex, options());
    const std::array<Fixed::Tangent, 3> directions {mean, derivative, mean};
    // exact-vertex nodal differentiation returns the selected body tangent unchanged
    EXPECT_LT(error(vertex_lin.nodal_jvp(directions).derivative, derivative), 2e-12);
    const manifold::SOGeometry<float, 3> single;
    manifold::SOGeometry<float, 3>::Tangent t;
    t(0, 1) = .2f;
    const auto single_identity = decltype(single)::Point::Identity();
    const auto frame = single.relative_frame(single_identity, single.exponential(single_identity, t));
    // the fixed float instantiation preserves the commuting Hessian eigenvector
    EXPECT_LT(error(single.half_squared_distance_hessian_vector(frame, t), t), 2e-6);
}
// repeated logarithm frequencies and zero rotations have removable differential singularities
TEST(SODifferential, RepeatedAndSmallFrequencies) {
    const Geometry g(4);
    const auto identity = Point::Identity(4);
    const auto v = direction(4, 1.2), w = direction(4, 2.7);
    for (double angle : {0., 1e-7, .03, .9, 2.7}) {
        Tangent l(4, 4);
        l(0, 1) = angle;
        l(2, 3) = angle;
        const auto target = g.exponential(identity, l);
        const auto frame = g.relative_frame(identity, target);
        constexpr double step = 1e-5;
        const auto plus = g.exponential(target, v, step), minus = g.exponential(target, v, -step);
        const auto hp = g.half_squared_distance_hessian_vector(g.relative_frame(identity, plus), w);
        const auto hm = g.half_squared_distance_hessian_vector(g.relative_frame(identity, minus), w);
        const auto finite = g.linear_combination(identity, .5 / step, hp, -.5 / step, hm);
        const auto mixed = g.half_squared_distance_hessian_covariant_jvp(frame, g.zero_tangent(identity), v, w);
        // repeated spectral values use the analytic divided-difference limit checked by target perturbations
        EXPECT_LT(error(mixed, finite), 2e-7) << angle;
    }
}
// branch ambiguity, nonpositive mean curvature and invalid directions must not produce certified derivatives
TEST(SOInterpolation, BranchCurvatureAndFailureDiagnostics) {
    const Geometry g(4);
    const auto identity = Point::Identity(4);
    Tangent l(4, 4);
    l(0, 1) = 2.8;
    l(2, 3) = 2.8;
    Batch nodes(3, 4, 4);
    nodes[0] = identity;
    nodes[1] = g.exponential(identity, l);
    nodes[2] = g.exponential(identity, l, -1);
    const std::array<double, 3> weights {.4, .3, .3};
    // a stationary saddle with an indefinite mean Hessian cannot certify the implicit CG derivative
    EXPECT_THROW(gfe::p1_geodesic_linearization(g, nodes, weights, identity, options()), std::domain_error);
    l(0, 1) = std::acos(-1.) / 2;
    l(2, 3) = std::acos(-1.) / 2;
    nodes[1] = g.exponential(identity, l);
    nodes[2] = g.exponential(identity, l, -1);
    const std::array<double, 3> singular_weights {0, .5, .5};
    // cancelling Hessian eigenvalues at a stationary mean make its implicit derivative nonunique
    EXPECT_THROW(gfe::p1_geodesic_linearization(g, nodes, singular_weights, identity, options()), std::domain_error);
    l(0, 1) = std::acos(-1.);
    l(2, 3) = 0;
    nodes[1] = g.exponential(identity, l);
    // the highest-weight initializer encounters an ambiguous active relative log and must reject it
    EXPECT_THROW(gfe::p1_geodesic_value(g, nodes, weights), std::domain_error);
    l(0, 1) = std::acos(-1.) - 1e-10;
    const auto near = g.relative_frame(identity, g.exponential(identity, l));
    // value logarithms remain available on a uniquely resolved nearby branch
    EXPECT_NO_THROW(g.logarithm(near));
    // derivatives reject the numerically unstable near-cut branch explicitly
    EXPECT_THROW(g.half_squared_distance_hessian_vector(near, direction(4, .4)), std::domain_error);
    for (int i = 0; i < 3; ++i) nodes[i] = g.exponential(identity, direction(4, i + .5));
    auto short_options = options();
    short_options.mean.solver.max_iterations = 1;
    short_options.mean.solver.gradient_tolerance = 0;
    const auto failed = gfe::p1_geodesic_linearization(g, nodes, weights, short_options);
    // an exhausted mean solve retains failure diagnostics instead of marking the candidate converged
    EXPECT_FALSE(failed.result().converged());
    const std::array<double, 3> dw {-.5, .2, .3};
    // derivatives of an unconverged candidate are unavailable
    EXPECT_THROW(failed.weight_jvp(dw), std::logic_error);
    const auto lin = gfe::p1_geodesic_linearization(g, nodes, weights, options());
    const std::array<double, 3> invalid {1, 0, 0};
    // directions must preserve the normalized barycentric sum
    EXPECT_THROW(lin.weight_jvp(invalid), std::invalid_argument);
    auto cg_options = options();
    cg_options.linear_solve.max_iterations = 1;
    cg_options.linear_solve.residual_tolerance = 0;
    const auto limited = gfe::p1_geodesic_linearization(g, nodes, weights, cg_options);
    // a truncated Hessian solve returns its failure status instead of certifying a derivative candidate
    EXPECT_FALSE(limited.weight_jvp(dw).converged());
    auto nan = direction(4, 0);
    nan(0, 1) = std::numeric_limits<double>::quiet_NaN();
    // the body tangent boundary rejects nonfinite pullback inputs
    EXPECT_THROW(lin.nodal_vjp(nan), std::invalid_argument);
}
}   // namespace
