// SPDX-License-Identifier: GPL-3.0-or-later
#include <fdaPDE/geometric_finite_elements.h>
#include <gtest/gtest.h>

namespace {
using namespace fdapde;
using Geometry = manifold::BuresWassersteinSPDGeometry<double, 2>;
using Point = Geometry::Point;
using Tangent = Geometry::Tangent;
using Dense = Matrix<double, 2, 2>;
using Batch = MatrixBatch<SPDMatrix<double, 2, 2, Cache::Union<Cache::Spectral, Cache::Sqrt, Cache::InverseSqrt>>>;

/// @brief converts a dense symmetric oracle to independent packed tangent storage
Tangent tangent(const Dense& value) { return Tangent(value.as_symmetric<Lower>()); }
/// @brief measures complete coefficient error including both symmetric triangles
double error(const auto& first, const auto& second) { return Dense(first - second).norm(); }
/// @brief pairs symmetric matrices in the ambient Frobenius metric
double frobenius(const auto& first, const auto& second) {
    return first(0, 0) * second(0, 0) + 2 * first(1, 0) * second(1, 0) + first(1, 1) * second(1, 1);
}
/// @brief obtains an SPD2 square root from its determinant and trace without spectral kernels
Dense square_root(const auto& value) {
    const double root_det = std::sqrt(value(0, 0) * value(1, 1) - value(1, 0) * value(1, 0));
    const double divisor = std::sqrt(value(0, 0) + value(1, 1) + 2 * root_det);
    return Dense(
      {(value(0, 0) + root_det) / divisor, value(1, 0) / divisor, value(1, 0) / divisor,
       (value(1, 1) + root_det) / divisor});
}
/// @brief computes the inverse of a symmetric SPD2 oracle by cofactors
Dense inverse(const auto& value) {
    const double determinant = value(0, 0) * value(1, 1) - value(1, 0) * value(1, 0);
    return Dense(
      {value(1, 1) / determinant, -value(1, 0) / determinant, -value(1, 0) / determinant, value(0, 0) / determinant});
}
/// @brief evaluates the SPD2 BW squared distance using scalar invariants
double squared_distance(const auto& first, const auto& second) {
    const double det_first = first(0, 0) * first(1, 1) - first(1, 0) * first(1, 0);
    const double det_second = second(0, 0) * second(1, 1) - second(1, 0) * second(1, 0);
    return first(0, 0) + first(1, 1) + second(0, 0) + second(1, 1) -
           2 * std::sqrt(frobenius(first, second) + 2 * std::sqrt(det_first * det_second));
}
/// @brief solves the three scalar equations AX plus XA equals U without eigendecomposition
Dense lyapunov(const auto& point, const auto& direction) {
    const double a = point(0, 0), b = point(1, 0), c = point(1, 1);
    const double y = (direction(1, 0) - b * direction(0, 0) / (2 * a) - b * direction(1, 1) / (2 * c)) /
                     (a + c - b * b * (1 / a + 1 / c));
    return Dense({(direction(0, 0) - 2 * b * y) / (2 * a), y, y, (direction(1, 1) - 2 * b * y) / (2 * c)});
}
/// @brief supplies noncommuting certified samples with reusable endpoint factors
Batch nodes() {
    Batch result(3);
    result[0] = Dense({2., .5, .5, 3.});
    result[1] = Dense({5., -1., -1., 4.});
    result[2] = Dense({1.5, .2, .2, 2.5});
    return result;
}
/// @brief resolves stationarity and linear solves below the finite-difference truncation error
gfe::P1GeodesicLinearizationOptions accurate() {
    gfe::P1GeodesicLinearizationOptions result;
    result.mean.solver.gradient_tolerance = 1e-11;
    result.linear_solve.residual_tolerance = 1e-12;
    return result;
}
/// @brief checks noncommuting target maps in the fixed and runtime SPD3 kernel paths
void check_three_dimensional(const auto& geometry) {
    using G = std::remove_cvref_t<decltype(geometry)>;
    const typename G::Point base(Matrix<double, 3, 3>({3., .3, -.2, .3, 2., .4, -.2, .4, 4.})),
      target(Matrix<double, 3, 3>({2., -.2, .1, -.2, 3., .5, .1, .5, 1.5}));
    typename G::Tangent direction;
    if constexpr (G::Tangent::Rows == Dynamic) direction.resize(3, 3);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j <= i; ++j) direction(i, j) = .1 * (1 + i - 2 * j);
    constexpr double step = 1e-5;
    const typename G::Point plus(target + step * direction), minus(target - step * direction);
    const auto lp = geometry.logarithm(base, plus), lm = geometry.logarithm(base, minus);
    const auto frame = geometry.relative_frame(base, target);
    const auto action = geometry.logarithm_target_jvp(frame, direction);
    const typename G::Tangent finite((lp - lm) / (2 * step));
    // centered target perturbations check the larger noncommuting Sylvester system without sharing its derivative code
    EXPECT_LT((Matrix<double, 3, 3>(action - finite).norm()), 3e-9);
    // the larger target pullback still uses the two distinct endpoint metrics
    EXPECT_NEAR(
      geometry.inner_product(base, action, direction),
      geometry.inner_product(target, direction, geometry.logarithm_target_vjp(frame, direction)), 3e-13);
    const auto restored = geometry.exponential(base, geometry.logarithm(base, target));
    // the higher-order relative spectral reconstruction preserves the exp-log inverse on noncommuting SPD3
    EXPECT_LT((Matrix<double, 3, 3>(restored - target).norm()), 5e-13);
}

// the BW metric and retraction satisfy the first-order solver contract
static_assert(manifold::FirstOrderGeometry<Geometry>);
// the logarithm, exponential and distance provide the contract used by geodesic P1 interpolation
static_assert(manifold::GeodesicGeometry<Geometry>);

// scalar and commuting distances reduce to differences of square roots while noncommuting SPD2 has a trace oracle
TEST(BuresWasserstein, ClosedDistanceAndExponentialOracles) {
    const manifold::BuresWassersteinSPDGeometry<double, 1> scalar;
    Matrix<double, 1, 1> a1, b1;
    a1(0, 0) = 4;
    b1(0, 0) = 9;
    const decltype(scalar)::Point one(a1), two(b1);
    // scalar BW distance equals the difference between positive square roots
    EXPECT_NEAR(scalar.distance(one, two), 1, 1e-13);
    const Geometry geometry;
    const Point first(Dense({4., .7, .7, 2.})), second(Dense({1.5, -.3, -.3, 3.}));
    // the noncommuting distance agrees with the independent trace and determinant formula
    EXPECT_NEAR(std::pow(geometry.distance(first, second), 2), squared_distance(first, second), 2e-13);
    // following the logarithm for unit time recovers the target coefficients
    EXPECT_LT(error(geometry.exponential(first, geometry.logarithm(first, second)), second), 3e-13);
    // identical endpoints produce an exactly negligible logarithm even with a nontrivial eigenbasis
    EXPECT_LT(Dense(geometry.logarithm(first, first)).norm(), 2e-13);
    const manifold::BuresWassersteinSPDGeometry<double, 3> fixed;
    Matrix<double, 3, 3> a, b;
    a.set_zero();
    b.set_zero();
    for (int i = 0; i < 3; ++i) {
        a(i, i) = (i + 1) * (i + 1);
        b(i, i) = (i + 2) * (i + 2);
    }
    const decltype(fixed)::Point pa(a), pb(b);
    // commuting SPD3 is the Euclidean distance of three scalar square-root coordinates
    EXPECT_NEAR(fixed.distance(pa, pb), std::sqrt(3.), 2e-13);
    const manifold::BuresWassersteinSPDGeometry<double, Dynamic> dynamic(3);
    // runtime-order storage retains the same commuting scalar oracle
    EXPECT_NEAR(dynamic.distance(pa, pb), std::sqrt(3.), 2e-13);
    const manifold::BuresWassersteinSPDGeometry<float, 2> single;
    const decltype(single)::Point fa(Matrix<float, 2, 2>({4.f, .7f, .7f, 2.f})),
      fb(Matrix<float, 2, 2>({1.5f, -.3f, -.3f, 3.f}));
    // the closed noncommuting formula also checks float dispatch at its own precision
    EXPECT_NEAR(std::pow(single.distance(fa, fb), 2), squared_distance(fa, fb), 3e-6);
    const auto float_restored = single.exponential(fa, single.logarithm(fa, fb));
    // float principal factors retain the exp-log inverse at a tolerance scaled to single precision
    EXPECT_LT((Matrix<float, 2, 2>(float_restored - fb).norm()), 4e-5);
    for (double scale : {1e-200, 1e100, 1e200}) {
        const Point large(Dense({scale, 0., 0., scale})), larger(Dense({4 * scale, 0., 0., 4 * scale}));
        // endpoint normalization preserves negligible self-distance at finite extreme isotropic scales
        EXPECT_NEAR(geometry.distance(large, large) / std::sqrt(scale), 0, 2e-14);
        // BW homogeneity gives the same square-root-coordinate distance despite extreme relative products
        EXPECT_NEAR(geometry.distance(large, larger) / std::sqrt(scale), std::sqrt(2.), 3e-14);
    }
}

// the ambient gradient conversion and covariant Hessian agree with independent Sylvester and connection formulas
TEST(BuresWasserstein, MetricDualAndCovariantHessian) {
    const Geometry geometry;
    const Point point(Dense({4., .7, .7, 2.})), target(Dense({1.5, -.3, -.3, 3.}));
    const Tangent u = tangent(Dense({.2, .4, .4, -.1})), v = tangent(Dense({-.3, .1, .1, .5}));
    // the metric equals half the Frobenius pairing with an independently solved Lyapunov system
    EXPECT_NEAR(geometry.inner_product(point, u, v), .5 * frobenius(lyapunov(point, u), v), 2e-14);
    const auto gradient = geometry.euclidean_to_riemannian_gradient(point, v);
    // the Riemannian gradient represents the ambient differential under the BW metric
    EXPECT_NEAR(geometry.inner_product(point, gradient, u), frobenius(v, u), 3e-14);
    // the inverse metric conversion recovers the original ambient gradient
    EXPECT_LT(error(geometry.riemannian_to_euclidean_gradient(point, gradient), v), 3e-14);
    const auto logarithm = geometry.logarithm(point, target);
    constexpr double step = 1e-5;
    const Point plus(point + step * u), minus(point - step * u);
    // a central ambient perturbation verifies the sign and normalization of the half squared distance gradient
    EXPECT_NEAR(
      (squared_distance(plus, target) - squared_distance(minus, target)) / (4 * step),
      -geometry.inner_product(point, logarithm, u), 3e-10);
    const Dense p = lyapunov(point, u), q = lyapunov(point, logarithm);
    const Dense triple(p * point * q + q * point * p);
    const Tangent connection = tangent(Dense(-1. * triple));
    const auto log_plus = geometry.logarithm(plus, target), log_minus = geometry.logarithm(minus, target);
    const Tangent finite = tangent(Dense((-1. / (2 * step)) * (log_plus - log_minus) - connection));
    const auto frame = geometry.relative_frame(point, target);
    const auto hessian = geometry.half_squared_distance_hessian_vector(frame, u);
    // adding the independent Levi-Civita connection to the ordinary gradient derivative checks the covariant Hessian
    EXPECT_LT(error(hessian, finite), 2e-9);
    // the Hessian of a scalar distance objective must be self-adjoint in the metric at its base point
    EXPECT_NEAR(
      geometry.inner_product(point, hessian, v),
      geometry.inner_product(point, u, geometry.half_squared_distance_hessian_vector(frame, v)), 3e-13);
    // at coincident endpoints the half squared distance Hessian is the identity tangent map
    EXPECT_LT(error(geometry.half_squared_distance_hessian_vector(geometry.relative_frame(point, point), u), u), 3e-13);
}

// relative endpoint caches preserve target differentials and their metric adjoints across selection and assignment
TEST(BuresWasserstein, TargetDifferentialsAndCachedSelection) {
    const Geometry geometry;
    auto batch = nodes();
    const auto selection = batch.select(std::array {2, 0, 1});
    const Point base(batch[0]), target(batch[1]);
    const Tangent direction = tangent(Dense({.1, .2, .2, -.3})), dual = tangent(Dense({-.2, .3, .3, .4}));
    const auto frame = geometry.relative_frame(base, target);
    const auto action = geometry.logarithm_target_jvp(frame, direction);
    constexpr double step = 1e-5;
    const Point plus(target + step * direction), minus(target - step * direction);
    const auto log_plus = geometry.logarithm(base, plus), log_minus = geometry.logarithm(base, minus);
    const Tangent finite((log_plus - log_minus) / (2 * step));
    // central target perturbations check the logarithm JVP independently of the cached Sylvester solve
    EXPECT_LT(error(action, finite), 2e-9);
    // the target pullback is the metric adjoint with the target and base metrics evaluated at their own points
    EXPECT_NEAR(
      geometry.inner_product(base, action, dual),
      geometry.inner_product(target, direction, geometry.logarithm_target_vjp(frame, dual)), 3e-13);
    const auto selected_frame = geometry.relative_frame(selection[1], selection[2]);
    // reordered borrowed views retain the same endpoint values and cached logarithm action
    EXPECT_LT(error(geometry.logarithm_target_jvp(selected_frame, direction), action), 2e-13);
    batch[1] = Dense({3.5, -.4, -.4, 2.1});
    const auto refreshed = geometry.relative_frame(selection[1], selection[2]);
    const auto plain = geometry.relative_frame(Point(batch[0]), Point(batch[1]));
    // fresh frames after assignment use invalidated and regenerated caches through a live batch selection
    EXPECT_LT(error(geometry.logarithm(refreshed), geometry.logarithm(plain)), 2e-13);
    // the replacement endpoint changes the map enough to detect accidentally reused endpoint factors
    EXPECT_GT(error(geometry.logarithm(refreshed), geometry.logarithm(frame)), .1);
    // fixed SPD3 exercises the generic spectral route with a noncommuting central-difference oracle
    check_three_dimensional(manifold::BuresWassersteinSPDGeometry<double, 3> {});
    // runtime SPD3 checks that dynamic workspaces preserve the same differential and metric-adjoint contracts
    check_three_dimensional(manifold::BuresWassersteinSPDGeometry<double, Dynamic>(3));
}

// BW barycenters follow square-root scalar means and the two-point optimal-transport geodesic
TEST(BuresWasserstein, BarycenterValueOracles) {
    const Geometry geometry;
    const std::array weights {.2, .3, .5};
    Batch diagonal(3);
    for (int i = 0; i < 3; ++i) diagonal[i] = Dense({double((i + 1) * (i + 1)), 0., 0., double((i + 2) * (i + 2))});
    const auto commuting = gfe::p1_geodesic_value(geometry, diagonal, weights, accurate().mean);
    // the iterative mean must carry a stationarity certificate before its coefficients are treated as an oracle
    ASSERT_TRUE(commuting.converged());
    // each commuting eigenvalue is the square of its weighted mean square root
    EXPECT_LT(error(commuting.value, Dense({2.3 * 2.3, 0., 0., 3.3 * 3.3})), 2e-10);
    const auto batch = nodes();
    const auto pair = batch.select(std::array {0, 1});
    const Dense root = square_root(pair[0]), inverse_root = inverse(root);
    const Point target(pair[1]);
    const Dense relative(root * target * root), relative_root = square_root(relative);
    const Dense transport(inverse_root * relative_root * inverse_root);
    constexpr double time = .3;
    const Dense identity({1., 0., 0., 1.});
    const Dense lift((1 - time) * identity + time * transport);
    const Point base(pair[0]);
    const Dense oracle(lift * base * lift);
    const auto actual = gfe::p1_geodesic_value(geometry, pair, std::array {1 - time, time}, accurate().mean);
    // the two-point mean must converge independently of the closed transport construction
    ASSERT_TRUE(actual.converged());
    // closed SPD2 roots and cofactors give an independent noncommuting two-point barycenter
    EXPECT_LT(error(actual.value, oracle), 3e-10);
    const auto curve = geometry.geodesic(base, target);
    // the deferred native geodesic evaluates to the independently constructed transport interpolation
    EXPECT_LT(error(Point(curve(time)), oracle), 3e-13);
    // an expression produced by a temporary curve retains its owning transport snapshot until evaluation
    EXPECT_LT(error(Point(geometry.geodesic(base, target)(time)), oracle), 3e-13);
    const auto selected = batch.select(std::array {2, 0, 1});
    const auto mean = gfe::p1_geodesic_value(geometry, batch, weights, accurate().mean);
    const auto permuted = gfe::p1_geodesic_value(geometry, selected, std::array {.5, .2, .3}, accurate().mean);
    // both local orderings must resolve the same well-defined mean before comparing their values
    ASSERT_TRUE(mean.converged() && permuted.converged());
    // jointly permuting cached nodes and weights leaves the BW barycenter unchanged
    EXPECT_LT(error(mean.value, permuted.value), 3e-10);
}

// implicit BW nodal and weight derivatives agree with independent mean refits and metric pullbacks
TEST(BuresWasserstein, ImplicitP1Differentials) {
    const Geometry geometry;
    const auto batch = nodes();
    const std::array weights {.2, .3, .5}, weight_direction {-.7, .2, .5};
    const std::array directions {
      tangent(Dense({.3, -.1, -.1, .4})), tangent(Dense({-.2, .15, .15, .1})), tangent(Dense({.1, .2, .2, -.3}))};
    const auto linearization = gfe::p1_geodesic_linearization(geometry, batch, weights, accurate());
    // the implicit-function derivative requires the underlying mean's stationarity certificate
    ASSERT_TRUE(linearization.result().converged());
    constexpr double step = 1e-4;
    Batch plus(batch), minus(batch);
    auto plus_weights = weights, minus_weights = weights;
    for (int i = 0; i < 3; ++i) {
        plus[i] = geometry.exponential(batch[i], directions[i], step);
        minus[i] = geometry.exponential(batch[i], directions[i], -step);
        plus_weights[i] += step * weight_direction[i];
        minus_weights[i] -= step * weight_direction[i];
    }
    const auto p = gfe::p1_geodesic_value(geometry, plus, weights, accurate().mean),
               m = gfe::p1_geodesic_value(geometry, minus, weights, accurate().mean);
    const auto pw = gfe::p1_geodesic_value(geometry, batch, plus_weights, accurate().mean),
               mw = gfe::p1_geodesic_value(geometry, batch, minus_weights, accurate().mean);
    // the four independent refits must converge before a numerical derivative is a valid oracle
    ASSERT_TRUE(p.converged() && m.converged() && pw.converged() && mw.converged());
    const auto nodal = linearization.nodal_jvp(directions), spatial = linearization.weight_jvp(weight_direction);
    const auto pullback = linearization.nodal_vjp(directions[0]);
    // nodal, spatial and adjoint linear solves each provide their own residual certificate
    ASSERT_TRUE(nodal.converged() && spatial.converged() && pullback.converged());
    // centered geodesic node refits check the ambient implicit nodal differential
    EXPECT_LT(error(nodal.derivative, Tangent((p.value - m.value) / (2 * step))), 3e-7);
    // centered convex-weight refits check the independent spatial-weight differential
    EXPECT_LT(error(spatial.derivative, Tangent((pw.value - mw.value) / (2 * step))), 3e-7);
    double rhs = 0;
    for (int i = 0; i < 3; ++i) rhs += geometry.inner_product(batch[i], directions[i], pullback.derivative[i]);
    // each nodal metric participates in the pullback identity against the metric at the barycenter
    EXPECT_NEAR(geometry.inner_product(linearization.result().value, nodal.derivative, directions[0]), rhs, 3e-10);
}

// intrinsic BW tension and Frobenius data gradients preserve their metric differential under independent refits
TEST(BuresWasserstein, TensionAndDataGradients) {
    const Geometry geometry;
    const auto batch = nodes();
    const std::array directions {
      tangent(Dense({.3, -.1, -.1, .4})), tangent(Dense({-.2, .15, .15, .1})), tangent(Dense({.1, .2, .2, -.3}))};
    const gfe::P1LumpedLaplacianStencil stencil {
      {.7,           1.4,         .9         },
      {{0, 1, -1.2}, {0, 2, .35}, {1, 2, -.8}}
    };
    constexpr double step = 1e-5;
    Batch plus(batch), minus(batch);
    for (int i = 0; i < 3; ++i) {
        plus[i] = geometry.exponential(batch[i], directions[i], step);
        minus[i] = geometry.exponential(batch[i], directions[i], -step);
    }
    const auto evaluate = [&](const auto& data, int kind) {
        if (kind == 0) return gfe::p1_discrete_tension_contribution(geometry, data, stencil);
        return gfe::p1_frobenius_data_site_contribution(
          geometry, data, std::array {.2, .3, .5}, tangent(Dense({2.2, .1, .1, 1.7})), accurate());
    };
    for (int kind = 0; kind < 2; ++kind) {
        const auto result = evaluate(batch, kind), p = evaluate(plus, kind), m = evaluate(minus, kind);
        // all mean and differential certificates must converge before checking an objective's gradient
        ASSERT_TRUE(result.converged() && p.converged() && m.converged());
        double analytic = 0;
        for (int i = 0; i < 3; ++i)
            analytic += geometry.inner_product(batch[i], result.nodal_gradient[i], directions[i]);
        const double finite = (p.value - m.value) / (2 * step);
        // central geodesic node perturbations verify the complete intrinsic or data-site nodal differential
        EXPECT_NEAR(analytic, finite, 3e-6 * std::max(1., std::abs(finite)));
    }
    const auto result = gfe::p1_discrete_tension_contribution(geometry, batch, stencil);
    // the value-only tension path preserves the derivative path's half squared residual normalization
    EXPECT_NEAR(result.value, gfe::p1_discrete_tension_value(geometry, batch, stencil).value, 2e-12);
    Batch constant(3);
    for (int i = 0; i < 3; ++i) constant[i] = batch[0];
    // a constant SPD field has zero intrinsic Laplacian even on a signed stiffness stencil
    EXPECT_NEAR(gfe::p1_discrete_tension_value(geometry, constant, stencil).value, 0, 2e-24);
}

// public validation rejects invalid exponential lifts and incompatible shapes, weights or solver steps
TEST(BuresWasserstein, PublicBoundaryContracts) {
    const Geometry geometry;
    const Point identity = Point::Identity();
    const Tangent crossing = tangent(Dense({-4., 0., 0., -4.}));
    // a negative lift must fail even though its congruence after crossing the singularity would again be SPD
    EXPECT_THROW(geometry.exponential(identity, crossing), std::domain_error);
    // the retraction shares the same BW exponential domain rather than accepting a crossed lift
    EXPECT_THROW(geometry.retract(identity, crossing, 1), std::domain_error);
    const Point four(Dense({4., 0., 0., 4.}));
    const auto curve = geometry.geodesic(identity, four);
    // deferred curve extrapolation must enforce the same positive lift boundary when coefficients are evaluated
    EXPECT_THROW(Point(curve(-2)), std::domain_error);
    Tangent invalid = crossing;
    invalid(1, 0) = std::numeric_limits<double>::quiet_NaN();
    // nonfinite ambient tangent coefficients are rejected before a Lyapunov solve can obscure the input error
    EXPECT_THROW(geometry.exponential(identity, invalid), std::invalid_argument);
    const manifold::BuresWassersteinSPDGeometry<double, Dynamic> dynamic(3);
    // a runtime geometry rejects certified points with a different matrix order
    EXPECT_THROW(dynamic.distance(identity, identity), std::invalid_argument);
    const auto batch = nodes();
    // convex interpolation rejects negative weights before mean initialization
    EXPECT_THROW(gfe::p1_geodesic_value(geometry, batch, std::array {-.1, .5, .6}), std::invalid_argument);
    // barycentric weights must sum to one before normalized mean evaluation
    EXPECT_THROW(gfe::p1_geodesic_value(geometry, batch, std::array {.2, .3, .6}), std::invalid_argument);
    auto options = accurate().mean;
    options.solver.line_search.initial_step = 1.1;
    // a BW mean limits its initial step to the guaranteed positive convex lift domain
    EXPECT_THROW(gfe::p1_geodesic_value(geometry, batch, std::array {.2, .3, .5}, options), std::invalid_argument);
    const gfe::P1LumpedLaplacianStencil invalid_stencil {
      {.7, 0., .9},
      {{0, 1, -1.2}, {1, 2, -.8}}
    };
    // an inverse lumped mass cannot be formed from a zero-measure degree of freedom
    EXPECT_THROW(gfe::p1_discrete_tension_value(geometry, batch, invalid_stencil), std::invalid_argument);
}
}   // namespace
