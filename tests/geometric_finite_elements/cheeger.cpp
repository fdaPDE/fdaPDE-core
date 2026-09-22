// SPDX-License-Identifier: GPL-3.0-or-later
#include <fdaPDE/geometric_finite_elements.h>
#include <gtest/gtest.h>

namespace {
using namespace fdapde;
using Geometry = manifold::CheegerLogEuclideanSPDGeometry<double, 2, Usage::InterpolationNodes>;
using Point = Geometry::Point;
using Tangent = Geometry::Tangent;
using Batch = MatrixBatch<Point>;
/// @brief builds SPD2 data from principal log eigenvalues and the eigenframe angle
Point tensor(double s, double a, double theta) {
    return Geometry::from_chart({s, a * std::cos(2 * theta), a * std::sin(2 * theta)});
}
/// @brief measures both triangles in the ambient Frobenius norm
double error(const auto& a, const auto& b) { return Matrix<double, 2, 2>(a - b).norm(); }
/// @brief resolves the lifted solver below the centered-difference truncation error
gfe::P1GeodesicLinearizationOptions accurate() {
    gfe::P1GeodesicLinearizationOptions options;
    options.mean.solver.gradient_tolerance = 1e-11;
    options.linear_solve.residual_tolerance = 1e-11;
    return options;
}
/// @brief retains the noncommuting triple from the historical C-LE regression
Batch nodes() {
    Batch data(3);
    data[0] = tensor(.1, .7, .2);
    data[1] = tensor(-.2, 1, 1.1);
    data[2] = tensor(.3, .4, 2.1);
    return data;
}
/// @brief checks the imported metric, exact pair reduction and known multistart branch switch
TEST(CheegerInterpolation, GeometryAndHistoricalBranches) {
    // C-LE supplies intrinsic geodesics without inheriting the unrelated LE transport
    static_assert(manifold::GeodesicGeometry<Geometry> && !manifold::VectorTransportGeometry<Geometry>);
    const Geometry geometry;
    const auto data = nodes();
    const Point a(data[0]), b(data[1]);
    const auto tangent = geometry.logarithm(a, b);
    // distance equals the length of its minimizing initial tangent
    EXPECT_NEAR(geometry.norm(a, tangent), geometry.distance(a, b), 1e-10);
    // the native exponential reaches the other endpoint from the minimizing logarithm
    EXPECT_LT(error(geometry.exponential(a, tangent), b), 1e-10);
    const auto curve = geometry.geodesic(a, b);
    const auto edge = gfe::p1_geodesic_value(geometry, data, std::array {.3, .7, 0.});
    // exact two-node interpolation follows the independently prepared pair curve
    EXPECT_LT(error(edge.value, curve(.7)), 1e-10);
    const auto vertex = gfe::p1_geodesic_value(geometry, data, std::array {0., 1., 0.});
    // one-hot interpolation preserves the original native matrix coefficients exactly
    EXPECT_EQ(error(vertex.value, b), 0);
    const Point iso = tensor(.2, 0, 0);
    // the isotropic endpoint has no privileged eigenframe but its exponential still reaches the target
    EXPECT_LT(error(geometry.exponential(iso, geometry.logarithm(iso, a)), a), 1e-10);
    Batch aligned(3);
    aligned[0] = tensor(0, .7, .1);
    aligned[1] = tensor(.02, .6, .1);
    aligned[2] = tensor(-.01, .8, .1);
    const auto cle = gfe::p1_geodesic_value(geometry, aligned, std::array {.2, .3, .5});
    const auto flat =
      gfe::p1_geodesic_value(manifold::LogEuclideanSPDGeometry<double, 2> {}, aligned, std::array {.2, .3, .5});
    // aligned principal axes reduce the lifted problem to the ordinary log-Euclidean mean
    EXPECT_LT(error(cle.value, flat.value), 1e-10);
    Batch broad(3);
    const double pi = std::acos(-1.);
    broad[0] = tensor(-.18, 1.05, 8 * pi / 180);
    broad[1] = tensor(.26, .62, 72 * pi / 180);
    broad[2] = tensor(.02, .88, 138 * pi / 180);
    const auto fa = gfe::p1_geodesic_value(geometry, broad, std::array {5. / 18, 4. / 9, 5. / 18}, accurate().mean);
    const auto fb = gfe::p1_geodesic_value(geometry, broad, std::array {19. / 72, 4. / 9, 7. / 24}, accurate().mean);
    // the captured historical objective detects changes to branch selection
    EXPECT_NEAR(fa.objective, .45234295334608776, 1e-10);
    // the second captured objective lies across the known broad-data branch switch
    EXPECT_NEAR(fb.objective, .45553757984342447, 1e-10);
    const manifold::LogEuclideanSPDGeometry<double, 2> le;
    // retaining the discontinuity prevents silently claiming global smoothness
    EXPECT_GT(le.distance(fa.value, fb.value), 2);
    // multistart convergence alone leaves global uniqueness uncertified
    EXPECT_EQ(fa.uniqueness, manifold::BarycenterUniqueness::not_certified);
}
/// @brief validates analytic weight, local rho and nodal-log actions against independently refitted data
TEST(CheegerInterpolation, ImplicitDerivativesAndVariableRho) {
    const auto data = nodes();
    const std::array weights {.2, .3, .5}, dw {1., -1., 0.}, nodal_rho {.22, .31, .28};
    const auto options = accurate();
    constexpr double h = 1e-5;
    for (double epsilon : {.05, .25, .5, 1., 2., 4.}) {
        const Geometry geometry(epsilon);
        const auto fit = gfe::p1_geodesic_linearization(geometry, data, weights, options);
        // finite differences are meaningful only for a converged mean branch
        ASSERT_TRUE(fit.result().converged());
        auto wp = weights, wm = weights;
        for (int i = 0; i < 3; ++i) {
            wp[i] += h * dw[i];
            wm[i] -= h * dw[i];
        }
        const auto p = gfe::p1_geodesic_value(geometry, data, wp, options.mean);
        const auto m = gfe::p1_geodesic_value(geometry, data, wm, options.mean);
        const auto derivative = fit.weight_jvp(dw);
        // the cached linear solve must certify its residual before comparing derivatives
        ASSERT_TRUE(derivative.converged());
        const Tangent difference((p.value - m.value) / (2 * h));
        // centered refits cover weak and strong rotational penalties from the historical suite
        EXPECT_LT(error(derivative.derivative, difference), 1e-5);
    }
    const Geometry geometry;
    const auto fit =
      gfe::p1_geodesic_linearization(geometry, data, weights, options, std::span<const double>(nodal_rho));
    const double rho = .2 * .22 + .3 * .31 + .5 * .28;
    // the scalar field is interpolated with the exact same normalized nodal weights
    EXPECT_NEAR(fit.geometry().rho(), rho, 1e-15);
    const auto p = gfe::p1_geodesic_value(Geometry::from_rho(rho + h), data, weights, options.mean);
    const auto m = gfe::p1_geodesic_value(Geometry::from_rho(rho - h), data, weights, options.mean);
    const Tangent difference((p.value - m.value) / (2 * h));
    // changing rho directly independently checks differentiation of the rotation penalty
    EXPECT_LT(error(fit.rho_jvp().derivative, difference), 1e-6);
    auto wp = weights, wm = weights;
    for (int i = 0; i < 3; ++i) {
        wp[i] += h * dw[i];
        wm[i] -= h * dw[i];
    }
    const auto fp = gfe::p1_geodesic_linearization(geometry, data, wp, options, std::span<const double>(nodal_rho));
    const auto fm = gfe::p1_geodesic_linearization(geometry, data, wm, options, std::span<const double>(nodal_rho));
    const Tangent total_difference((fp.result().value - fm.result().value) / (2 * h));
    // weight derivatives include the chain-rule contribution from the spatially varying rho field
    EXPECT_LT(error(fit.weight_jvp(dw).derivative, total_difference), 1e-6);
    std::array<Tangent, 3> directions;
    auto plus = data, minus = data;
    for (int i = 0; i < 3; ++i) {
        directions[i] = manifold::internals::cheeger_matrix<double>({.1 * i, .2 - .1 * i, .05 * i});
        const auto log = matrix_log(data[i]);
        const Tangent lp(log + h * directions[i]), lm(log - h * directions[i]);
        plus[i] = matrix_exp(lp);
        minus[i] = matrix_exp(lm);
    }
    const auto np = gfe::p1_geodesic_value(fit.geometry(), plus, weights, options.mean);
    const auto nm = gfe::p1_geodesic_value(fit.geometry(), minus, weights, options.mean);
    const Tangent nodal_difference((np.value - nm.value) / (2 * h));
    // the imported smoothing adapter differentiates nodal logarithms at fixed local rho
    EXPECT_LT(error(fit.nodal_log_jvp(directions), nodal_difference), 1e-6);
    for (int i = 0; i < 3; ++i) {
        const auto log = matrix_log(data[i]);
        directions[i] = matrix_exp_frechet(log, directions[i]);
    }
    // ambient directions agree after converting the same logarithmic perturbations through Dexp
    EXPECT_LT(error(fit.nodal_jvp(directions).derivative, nodal_difference), 1e-6);
    // constant nodal rho perturbations reduce to a unit local scalar perturbation
    EXPECT_LT(error(fit.nodal_rho_jvp(std::array {1., 1., 1.}).derivative, fit.rho_jvp().derivative), 1e-12);
}
/// @brief rejects invalid parameters and derivative branches without hiding pair ambiguity
TEST(CheegerInterpolation, ParameterAndBranchContracts) {
    for (double rho : {0., -1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        // positive finite rho is required independently of debug assertion configuration
        EXPECT_THROW(Geometry::from_rho(rho), std::invalid_argument);
    }
    // squaring a finite epsilon must not overflow into an invalid metric
    EXPECT_THROW(Geometry {std::numeric_limits<double>::max()}, std::invalid_argument);
    const Geometry geometry;
    // the legacy epsilon default corresponds exactly to the target squared penalty
    EXPECT_EQ(geometry.rho(), .25);
    Batch pair(2);
    pair[0] = tensor(0, std::log(2.), 0);
    pair[1] = tensor(0, std::log(2.), std::acos(-1.) / 2);
    const auto fit = gfe::p1_geodesic_linearization(geometry, pair, std::array {.5, .5});
    // orthogonal principal directions expose both equal-cost minimizing alignments
    EXPECT_TRUE(fit.result().detected_ambiguity);
    // a selected value at a tie cannot provide a unique rho derivative
    EXPECT_THROW(fit.rho_jvp(), std::domain_error);
    // value materialization must not silently choose a detected ambiguous mean
    EXPECT_THROW(gfe::internals::require_converged(fit.result()), std::domain_error);
    const auto data = nodes();
    const auto edge = gfe::p1_geodesic_linearization(geometry, data, std::array {.5, .5, 0.});
    // the recovered implicit derivative supports interior weights only
    EXPECT_THROW(edge.rho_jvp(), std::domain_error);
    const auto interior = gfe::p1_geodesic_linearization(geometry, data, std::array {.2, .3, .5}, accurate());
    // non-tangent directions to the normalized P1 simplex are rejected
    EXPECT_THROW(interior.weight_jvp(std::array {1., 0., 0.}), std::invalid_argument);
}
/// @brief reproduces the TSPDE near-tie and the change in local stability after increasing rho
TEST(CheegerInterpolation, CapturedSmoothingBranchCrossing) {
    Batch data(3);
    data[0] = Matrix<double, 2, 2>({1.3819441956010734, 0.10407163589642791, 0.10407163589642791, 0.5205325482828601});
    data[1] =
      Matrix<double, 2, 2>({0.6999243111143584, 0.004026089043202127, 0.004026089043202127, 0.9893084925674775});
    data[2] = Matrix<double, 2, 2>({0.943327813860192, -0.09282478306313756, -0.09282478306313756, 0.7244188919242418});
    const std::array weights {0.480818702373651, 0.21001214021817097, 0.3091691574081781};
    auto options = accurate();
    options.mean.solver.gradient_tolerance = 1e-10;
    options.mean.solver.max_iterations = 500;
    const auto fit = gfe::p1_geodesic_linearization(Geometry {}, data, weights, options);
    // the captured training site has two stationary candidates within the historical tie threshold
    ASSERT_TRUE(fit.result().converged() && fit.result().detected_ambiguity);
    // even a positive local Hessian cannot resolve competing minimizing branches
    EXPECT_THROW(fit.rho_jvp(), std::domain_error);
    const auto shifted = gfe::p1_geodesic_linearization(Geometry::from_rho(.505 * .505), data, weights, options);
    // the TSPDE epsilon probe resolves this local near-tie after a small metric change
    ASSERT_TRUE(shifted.result().converged() && !shifted.result().detected_ambiguity);
    // resolving sampled candidates still does not certify uniqueness of the global minimizer
    EXPECT_EQ(shifted.result().uniqueness, manifold::BarycenterUniqueness::not_certified);
    const auto first = Geometry::chart(data[0]);
    std::array<Point, 2> predictions {fit.result().value, fit.result().value};
    std::array<Point, 2> regular {fit.result().value, fit.result().value};
    for (int i = 0; i < 2; ++i) {
        auto chart = first;
        chart.y += i == 0 ? -1e-8 : 1e-8;
        data[0] = Geometry::from_chart(chart);
        predictions[i] = gfe::p1_geodesic_value(Geometry {}, data, weights, options.mean).value;
        regular[i] = gfe::p1_geodesic_value(Geometry::from_rho(.505 * .505), data, weights, options.mean).value;
    }
    // the native implementation preserves the measured finite jump across the original branch boundary
    EXPECT_NEAR(error(predictions[0], predictions[1]), .11406128421730882, 1e-6);
    // the same perturbation is small at the shifted rho, without making a global uniqueness claim
    EXPECT_LT(error(regular[0], regular[1]), 1e-6);
}
}   // namespace
