// SPDX-License-Identifier: GPL-3.0-or-later
#include <fdaPDE/geometric_discretization.h>
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
/// @brief retains the fitted noncommuting nodes adjacent to the captured smoothing edge
Batch near_edge_nodes() {
    const std::array<std::array<double, 3>, 3> logs {
      {{0.73228411922199965, 4.0436828641257814e-06, 0.58784730643748428},
       {0.71294875312875017, 1.6309328873535968e-05, 0.59880399950094432},
       {0.73206110007738123, -0.0093906122717749702, 0.59887764716908209}}
    };
    Batch data(logs.size());
    for (std::size_t i = 0; i < logs.size(); ++i) {
        Tangent log;
        log(0, 0) = logs[i][0];
        log(0, 1) = logs[i][1];
        log(1, 1) = logs[i][2];
        data[i] = matrix_exp(log);
    }
    return data;
}
/// @brief preserves positive edge-limit weights and resolves their mean and rho derivative in every nodal order
TEST(CheegerInterpolation, NearEdgePositiveWeights) {
    const auto data = near_edge_nodes();
    const Geometry geometry;
    auto options = accurate();
    options.mean.solver.gradient_tolerance = 1e-10;
    options.mean.solver.max_iterations = 500;
    options.linear_solve.residual_tolerance = 1e-10;
    const auto edge = geometry.geodesic(data[0], data[1])(1. / 3);
    for (double tiny : {std::numeric_limits<double>::denorm_min(), 3.3306690738754696e-16, 1e-14, 1e-8, .02}) {
        const std::array weights {(1 - tiny) * 2 / 3, (1 - tiny) / 3, tiny};
        std::array<std::size_t, 3> permutation {0, 1, 2};
        do {
            std::array<double, 3> permuted_weights;
            for (int i = 0; i < 3; ++i) permuted_weights[i] = weights[permutation[i]];
            const auto fit =
              gfe::p1_geodesic_linearization(geometry, data.select(permutation), permuted_weights, options);
            // every supplied positive coordinate remains active, including the smallest representable weight
            EXPECT_TRUE(
              std::all_of(
                fit.result().normalized_weights.begin(), fit.result().normalized_weights.end(),
                [](double weight) { return weight > 0; }));
            // conditioning must resolve the original stationarity criterion without changing its tolerance
            ASSERT_TRUE(fit.result().converged());
            // the reported residual is still the unscaled lifted gradient norm
            EXPECT_LE(fit.result().stationarity_norm, options.mean.solver.gradient_tolerance);
            // the interior means approach the independently prepared two-node geodesic as the third weight vanishes
            EXPECT_LT(error(fit.result().value, edge), 1e-10 + tiny);
            const auto derivative = fit.rho_jvp();
            // the derivative factorization must remain usable under the same extreme weight ratios
            EXPECT_TRUE(derivative.converged());
        } while (std::next_permutation(permutation.begin(), permutation.end()));
    }
}
/// @brief checks boundary-limit implicit actions and their adjoint against independent refits
TEST(CheegerInterpolation, NearEdgeImplicitDerivatives) {
    const auto data = near_edge_nodes();
    const Geometry geometry;
    const std::array weights {2. / 3, 1. / 3, 3.3306690738754696e-16};
    auto options = accurate();
    options.mean.solver.gradient_tolerance = 1e-10;
    options.mean.solver.max_iterations = 500;
    options.linear_solve.residual_tolerance = 1e-10;
    const auto fit = gfe::p1_geodesic_linearization(geometry, data, weights, options);
    // implicit actions are checked only after the captured positive-weight mean has converged
    ASSERT_TRUE(fit.result().converged());
    constexpr double step = 1e-5;
    const auto rho_plus =
      gfe::p1_geodesic_value(Geometry::from_rho(geometry.rho() + step), data, weights, options.mean);
    const auto rho_minus =
      gfe::p1_geodesic_value(Geometry::from_rho(geometry.rho() - step), data, weights, options.mean);
    // the independently refitted metric perturbations remain stationary on an unambiguous branch
    ASSERT_TRUE(
      rho_plus.converged() && rho_minus.converged() && !rho_plus.detected_ambiguity && !rho_minus.detected_ambiguity);
    const auto rho_derivative = fit.rho_jvp();
    // the equilibrated solve is certified by the residual of the original unscaled system
    ASSERT_TRUE(rho_derivative.converged());
    // centered metric perturbations independently verify the original rho derivative
    EXPECT_LT(error(rho_derivative.derivative, Tangent((rho_plus.value - rho_minus.value) / (2 * step))), 1e-6);
    std::array<Tangent, 3> directions;
    auto plus = data, minus = data;
    for (int i = 0; i < 3; ++i) {
        directions[i] = manifold::internals::cheeger_matrix<double>({.02 * i, .03 - .01 * i, -.01 + .02 * i});
        const auto log = matrix_log(data[i]);
        plus[i] = matrix_exp(Tangent(log + step * directions[i]));
        minus[i] = matrix_exp(Tangent(log - step * directions[i]));
    }
    const auto nodal_plus = gfe::p1_geodesic_value(geometry, plus, weights, options.mean);
    const auto nodal_minus = gfe::p1_geodesic_value(geometry, minus, weights, options.mean);
    // stationary unambiguous nodal refits provide the finite-difference oracle
    ASSERT_TRUE(
      nodal_plus.converged() && nodal_minus.converged() && !nodal_plus.detected_ambiguity &&
      !nodal_minus.detected_ambiguity);
    const Tangent action = fit.nodal_log_jvp(directions);
    // centered nodal-log refits verify the forward action with the tiny weight retained
    EXPECT_LT(error(action, Tangent((nodal_plus.value - nodal_minus.value) / (2 * step))), 1e-6);
    const Tangent covector = manifold::internals::cheeger_matrix<double>({.2, -.1, .03});
    double rho_pullback = 0, nodal_pairing = 0;
    const auto pullback = fit.nodal_log_vjp(covector, &rho_pullback);
    for (int i = 0; i < 3; ++i)
        nodal_pairing += manifold::internals::cheeger_dot(
          manifold::internals::cheeger_chart(pullback[i]), manifold::internals::cheeger_chart(directions[i]));
    const auto pair = [&](const Tangent& value) {
        return manifold::internals::cheeger_dot(
          manifold::internals::cheeger_chart(covector), manifold::internals::cheeger_chart(value));
    };
    // the forward and reverse nodal actions must satisfy the same Frobenius adjoint identity
    EXPECT_NEAR(nodal_pairing, pair(action), 1e-12);
    // the reverse metric action must agree with the independently checked forward rho action
    EXPECT_NEAR(rho_pullback, pair(rho_derivative.derivative), 1e-12);
    const auto weight_derivative = fit.weight_jvp(std::array {-1., 0., 1.});
    // the weight action must satisfy the original implicit-system residual criterion
    ASSERT_TRUE(weight_derivative.converged());
    auto shifted = weights, twice_shifted = weights;
    shifted[0] -= step;
    shifted[2] += step;
    twice_shifted[0] -= 2 * step;
    twice_shifted[2] += 2 * step;
    const auto first = gfe::p1_geodesic_value(geometry, data, shifted, options.mean);
    const auto second = gfe::p1_geodesic_value(geometry, data, twice_shifted, options.mean);
    // both inward refits must converge on the same unambiguous branch as the boundary limit
    ASSERT_TRUE(first.converged() && second.converged() && !first.detected_ambiguity && !second.detected_ambiguity);
    // an inward direction admits a second-order one-sided oracle without introducing negative barycentric weights
    EXPECT_LT(
      error(
        weight_derivative.derivative, Tangent((-3 * fit.result().value + 4 * first.value - second.value) / (2 * step))),
      1e-6);
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
    const auto edge_derivative = edge.rho_jvp();
    // the active two-node Hessian certifies the rho derivative despite the inactive coordinate
    ASSERT_TRUE(edge_derivative.converged());
    constexpr double step = 1e-5;
    const auto edge_plus =
      gfe::p1_geodesic_value(Geometry::from_rho(geometry.rho() + step), data, std::array {.5, .5, 0.});
    const auto edge_minus =
      gfe::p1_geodesic_value(Geometry::from_rho(geometry.rho() - step), data, std::array {.5, .5, 0.});
    const Tangent edge_difference((edge_plus.value - edge_minus.value) / (2 * step));
    // centered refits independently check the edge rho derivative with a zero-weight node
    EXPECT_LT(error(edge_derivative.derivative, edge_difference), 1e-6);
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
/// @brief constructs coordered SPD3 logarithms in a shared nontrivial eigenframe
MatrixBatch<SPDMatrix<double, 3, Cache::Union<Cache::Log, Cache::Spectral>>> coordered_spd3() {
    using Cached = SPDMatrix<double, 3, Cache::Union<Cache::Log, Cache::Spectral>>;
    MatrixBatch<Cached> result(3);
    SkewSymmetricMatrix<double, 3> omega;
    omega(0, 1) = .2;
    omega(0, 2) = -.3;
    omega(1, 2) = .1;
    const auto rotation = rotation_exp(omega);
    for (int i = 0; i < 3; ++i) {
        SymmetricMatrix<double, 3> diagonal;
        for (int r = 0; r < 3; ++r)
            for (int c = r; c < 3; ++c) diagonal(r, c) = r == c ? -.7 + .8 * r + .05 * (r + 1) * i : 0.;
        const SymmetricMatrix<double, 3> log((rotation * diagonal * rotation.transpose()).as_symmetric<Lower>());
        result[i] = matrix_exp(log);
    }
    return result;
}
// the common weighted API resolves coordered SPD3 means without iterative starts and retains implicit derivatives
TEST(CheegerInterpolation, CoorderedSPD3ValueAndDerivatives) {
    const auto data = coordered_spd3();
    const manifold::CheegerLogEuclideanGeometry<SPDMatrix<double, 3, Cache::Union<Cache::Log, Cache::Spectral>>>
      geometry(.5);
    const std::array weights {.2, .3, .5}, dw {-.3, .1, .2};
    auto settings = accurate();
    settings.mean.solver.gradient_tolerance = 1e-10;
    const auto fit = gfe::p1_geodesic_linearization(geometry, data, weights, settings);
    const auto expected = gfe::p1_geodesic_value(manifold::LogEuclideanSPDGeometry<double, 3> {}, data, weights);
    // the spectral trace bound makes the identity lift optimal for the ordered common eigenframe
    ASSERT_TRUE(fit.result().converged());
    // the resolved mean follows the LE oracle without executing any local optimizer iterations
    EXPECT_EQ(fit.result().iterations, 0U);
    // comparison uses the full SPD matrix rather than only its determinant or spectrum
    EXPECT_LT((Matrix<double, 3, 3>(fit.result().value - expected.value)).norm(), 2e-12);
    // a uniquely minimizing identity lift cannot produce competing mean branches
    EXPECT_FALSE(fit.result().detected_ambiguity);
    const auto derivative = fit.weight_jvp(dw);
    // the shortcut must still prepare and certify the ordinary implicit derivative workspace
    ASSERT_TRUE(derivative.converged());
    constexpr double step = 1e-5;
    auto plus = weights, minus = weights;
    for (int i = 0; i < 3; ++i) {
        plus[i] += step * dw[i];
        minus[i] -= step * dw[i];
    }
    const auto a = gfe::p1_geodesic_value(geometry, data, plus, settings.mean);
    const auto b = gfe::p1_geodesic_value(geometry, data, minus, settings.mean);
    // independent refits verify weight derivatives on the same public API as every other geometry
    EXPECT_LT((Matrix<double, 3, 3>((a.value - b.value) / (2 * step) - derivative.derivative)).norm(), 2e-8);
    std::array<SymmetricMatrix<double, 3>, 3> direction;
    for (int i = 0; i < 3; ++i)
        for (int r = 0; r < 3; ++r)
            for (int c = r; c < 3; ++c) direction[i](r, c) = .1 * std::sin(i + 2 * r + c + 1.);
    const auto action = fit.nodal_log_jvp(direction);
    auto perturbed = data;
    std::array<std::remove_cvref_t<decltype(fit.result().value)>, 2> perturbed_values {
      fit.result().value, fit.result().value};
    for (int sign : {-1, 1}) {
        for (int i = 0; i < 3; ++i) {
            SymmetricMatrix<double, 3> log(matrix_log(data[i]));
            log = log + sign * step * direction[i];
            perturbed[i] = matrix_exp(log);
        }
        const auto result = gfe::p1_geodesic_value(geometry, perturbed, weights, settings.mean);
        // noncommuting perturbations use the original branch search and must meet the same tolerance
        ASSERT_TRUE(result.converged());
        perturbed_values[sign < 0 ? 0 : 1] = result.value;
    }
    // centered off-diagonal perturbations leave the fast path and check the full nodal implicit derivative
    EXPECT_LT((Matrix<double, 3, 3>((perturbed_values[1] - perturbed_values[0]) / (2 * step) - action)).norm(), 2e-6);
    const auto rho = fit.rho_jvp();
    // coordered means are independent of positive rho although their nodal derivative can depend on rho
    ASSERT_TRUE(rho.converged());
    // the metric parameter changes no value along this coordered family
    EXPECT_LT(rho.derivative.norm(), 2e-12);
}
// commuting matrices with inconsistent eigenvalue order must retain the nonconvex branch search
TEST(CheegerInterpolation, SPD3CrossedOrderingAndCacheInvalidation) {
    using Cached = SPDMatrix<double, 3, Cache::Union<Cache::Log, Cache::Spectral>>;
    using Geometry3 = manifold::CheegerLogEuclideanGeometry<Cached>;
    MatrixBatch<Cached> crossed(3);
    for (int i = 0; i < 3; ++i) {
        SymmetricMatrix<double, 3> diagonal;
        for (int r = 0; r < 3; ++r)
            for (int c = r; c < 3; ++c) diagonal(r, c) = r == c ? 2. * ((r + i) % 3 - 1) : 0.;
        crossed[i] = matrix_exp(diagonal);
    }
    auto settings = accurate();
    settings.mean.solver.gradient_tolerance = 1e-8;
    settings.mean.solver.max_iterations = 500;
    const std::array weights {.2, .3, .5};
    const auto fit = gfe::p1_geodesic_value(Geometry3(.5), crossed, weights, settings.mean);
    const auto le = gfe::p1_geodesic_value(manifold::LogEuclideanSPDGeometry<double, 3> {}, crossed, weights);
    // crossing spectra can favor nonidentity rotations despite exact pairwise commutation
    ASSERT_TRUE(fit.converged());
    // this configuration must not be accepted as the stationary but suboptimal identity lift
    EXPECT_GT(fit.iterations, 0U);
    // the nonconvex solution is observably different from the LE mean
    EXPECT_GT((Matrix<double, 3, 3>(fit.value - le.value)).norm(), .1);
    auto data = coordered_spd3();
    const auto before = gfe::p1_geodesic_value(Geometry3(.5), data, weights, settings.mean);
    SymmetricMatrix<double, 3> changed(matrix_log(data[0]));
    changed(0, 1) += .05;
    data[0] = matrix_exp(changed);
    const auto after = gfe::p1_geodesic_value(Geometry3(.5), data, weights, settings.mean);
    // replacing a batch entry invalidates its cached spectrum and prevents reuse of the old common-frame decision
    ASSERT_TRUE(after.converged());
    // the revised noncommuting field is solved rather than silently returning a cached value
    EXPECT_GT((Matrix<double, 3, 3>(after.value - before.value)).norm(), 1e-4);
    // the general local search remains active after the loss of coordering
    EXPECT_GT(after.iterations, 0U);
}
// fixed and runtime tensor orders share the weighted APIs across permutations, repeated spectra and sparse support
TEST(CheegerInterpolation, SPD3DynamicSelectionsAndRepeatedSpectrum) {
    const auto data = coordered_spd3();
    const manifold::CheegerLogEuclideanSPDGeometry<double, Dynamic> geometry(3, .5);
    const std::array<std::size_t, 3> permutation {2, 0, 1};
    auto settings = accurate();
    settings.mean.solver.gradient_tolerance = 1e-10;
    for (const auto weights : {
           std::array {.5,    .2, .3        },
            std::array {1e-14, .4, .6 - 1e-14},
            std::array {0.,    .4, .6        }
    }) {
        const auto selected = data.select(permutation);
        const auto fit = gfe::p1_geodesic_linearization(geometry, selected, weights, settings);
        const auto expected =
          gfe::p1_geodesic_value(manifold::LogEuclideanSPDGeometry<double, Dynamic>(3), selected, weights);
        // runtime order and noncontiguous selections preserve the same coordered stationary solution
        ASSERT_TRUE(fit.result().converged());
        // inactive and almost inactive nodes do not force a redundant rotational search
        EXPECT_EQ(fit.result().iterations, 0U);
        // a dynamic LE oracle independently materializes the expected full matrix
        EXPECT_LT((Matrix<double, Dynamic, Dynamic>(fit.result().value - expected.value)).norm(), 2e-12);
        const auto rho = fit.rho_jvp();
        // the retained positive active Hessian supports derivatives even on an open edge
        ASSERT_TRUE(rho.converged());
        // rho cannot change this coordered mean, including the positive-weight boundary limit
        EXPECT_LT(rho.derivative.norm(), 2e-12);
    }
    MatrixBatch<SPDMatrix<double, 3>> repeated(3);
    for (int i = 0; i < 3; ++i) {
        SymmetricMatrix<double, 3> log;
        for (int r = 0; r < 3; ++r)
            for (int c = r; c < 3; ++c) log(r, c) = r == c ? (r == 2 ? .8 : .2) + .1 * i : 0.;
        repeated[i] = matrix_exp(log);
    }
    const std::array weights {.2, .3, .5};
    const auto fit = gfe::p1_geodesic_linearization(geometry, repeated, weights, settings);
    // repeated eigenvalues and uncached batch owners are valid inputs to the same fast path
    ASSERT_TRUE(fit.result().converged());
    // positive rho fixes the rotations even when their eigenframes are not individually unique
    EXPECT_EQ(fit.result().iterations, 0U);
    // the derivative workspace remains nonsingular in the repeated-spectrum case
    EXPECT_TRUE(fit.rho_jvp().converged());
}
}   // namespace
