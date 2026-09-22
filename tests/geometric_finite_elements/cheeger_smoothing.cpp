// SPDX-License-Identifier: GPL-3.0-or-later
#include <fdaPDE/geometric_finite_elements.h>
#include <gtest/gtest.h>
namespace {
using namespace fdapde;
/// @brief compares joint matrix-log and rho gradients against independently refitted perturbations
template <int N> void check(int n = N) {
    using G = manifold::CheegerLogEuclideanSPDGeometry<double, N, Usage::InterpolationNodes>;
    G g = [&] {
        if constexpr (N == Dynamic)
            return G(n);
        else
            return G();
    }();
    using T = typename G::Tangent;
    MatrixBatch<typename G::Point> nodes(3, n, n), plus(3, n, n), minus(3, n, n);
    std::vector<T> dirs;
    for (int i = 0; i < 3; ++i) {
        T x, d;
        if constexpr (N == Dynamic) {
            x.resize(n, n);
            d.resize(n, n);
        }
        for (int r = 0; r < n; ++r)
            for (int c = 0; c <= r; ++c) {
                x(r, c) = r == c ? .17 * (i + 1) * (r + 1) : .08 * std::sin(1 + i + r + c);
                d(r, c) = .12 * std::cos(2 * i + r + 3 * c);
            }
        nodes[i] = matrix_exp(x);
        dirs.push_back(d);
        const T xp(x + 1e-5 * d), xm(x - 1e-5 * d);
        plus[i] = matrix_exp(xp);
        minus[i] = matrix_exp(xm);
    }
    std::array rho {.3, .8, .5}, rp = rho, rm = rho;
    std::array dr {.2, -.1, .3};
    for (int i = 0; i < 3; ++i) {
        rp[i] += 1e-5 * dr[i];
        rm[i] -= 1e-5 * dr[i];
    }
    gfe::P1LumpedLaplacianStencil s {
      {1.,          .8,         1.2        },
      {{0, 1, -.7}, {0, 2, .2}, {1, 2, -1.}}
    };
    auto f = gfe::p1_cheeger_discrete_tension_log_contribution(g, nodes, s, rho);
    auto p = gfe::p1_cheeger_discrete_tension_log_contribution(g, plus, s, rp);
    auto m = gfe::p1_cheeger_discrete_tension_log_contribution(g, minus, s, rm);
    double a = 0;
    for (int i = 0; i < 3; ++i)
        a += manifold::internals::cheeger_inner(f.nodal_gradient[i], dirs[i]) + f.rho_gradient[i] * dr[i];
    // perturbing tensors and rho together checks every term in the local-metric tension
    EXPECT_NEAR(a, (p.value - m.value) / 2e-5, 2e-6);
    gfe::P1GeodesicLinearizationOptions opt;
    opt.mean.solver.gradient_tolerance = 1e-10;
    opt.mean.solver.max_iterations = 300;
    const T observation(matrix_log(nodes[0]));
    f = gfe::p1_cheeger_frobenius_data_site_log_contribution(g, nodes, std::array {.2, .3, .5}, observation, opt, rho);
    p = gfe::p1_cheeger_frobenius_data_site_log_contribution(g, plus, std::array {.2, .3, .5}, observation, opt, rp);
    m = gfe::p1_cheeger_frobenius_data_site_log_contribution(g, minus, std::array {.2, .3, .5}, observation, opt, rm);
    a = 0;
    for (int i = 0; i < 3; ++i)
        a += manifold::internals::cheeger_inner(f.nodal_gradient[i], dirs[i]) + f.rho_gradient[i] * dr[i];
    // only stationary mean branches supply valid objective derivatives
    ASSERT_TRUE(f.converged() && p.converged() && m.converged());
    // independently refitting the mean checks the adjoint in tensor and rho directions
    EXPECT_NEAR(a, (p.value - m.value) / 2e-5, 2e-6);
    const auto constant = gfe::p1_cheeger_discrete_tension_log_contribution(g, nodes, s);
    const std::array equal_rho {g.rho(), g.rho(), g.rho()};
    const auto explicit_constant = gfe::p1_cheeger_discrete_tension_log_contribution(g, nodes, s, equal_rho);
    // omitted rho and an equal nodal field represent exactly the same discrete objective
    EXPECT_NEAR(constant.value, explicit_constant.value, 1e-13);
    if constexpr (N == 2) {
        using C = manifold::internals::CheegerChart;
        std::array<C, 3> x, r {};
        for (int i = 0; i < 3; ++i) x[i] = G::chart(nodes[i]);
        for (const auto& edge : s.edges)
            for (int reverse = 0; reverse < 2; ++reverse) {
                const auto i = reverse ? edge.second : edge.first, j = reverse ? edge.first : edge.second;
                const double beta = manifold::internals::cheeger_pair(x[i], x[j], g.rho()).rotations.front();
                const auto z = manifold::internals::cheeger_rotate(x[j], -beta);
                r[i].s += edge.stiffness * (z.s - x[i].s);
                r[i].x += edge.stiffness * (z.x - x[i].x - 2 * beta * x[i].y);
                r[i].y += edge.stiffness * (z.y - x[i].y + 2 * beta * x[i].x);
            }
        double legacy = 0;
        for (int i = 0; i < 3; ++i) {
            const double c = x[i].x * r[i].y - x[i].y * r[i].x;
            legacy += (r[i].s * r[i].s + r[i].x * r[i].x + r[i].y * r[i].y -
                       4 * c * c / (g.rho() + 4 * (x[i].x * x[i].x + x[i].y * x[i].y))) /
                      s.lumped_masses[i];
        }
        // the original TSPDE three-coordinate expression independently checks constant-rho recovery
        EXPECT_NEAR(constant.value, legacy, 1e-13);
    }
    const auto v = g.logarithm(typename G::Point(nodes[0]), typename G::Point(nodes[1]));
    const auto endpoint = g.exponential(typename G::Point(nodes[0]), v);
    // the general quotient exponential reaches the endpoint selected by its pair logarithm
    EXPECT_LT((Matrix<double, N, N>(endpoint - nodes[1]).norm()), 2e-8);
    // the metric norm of the initial velocity equals the length of the selected pair curve
    EXPECT_NEAR(
      g.norm(typename G::Point(nodes[0]), v), g.distance(typename G::Point(nodes[0]), typename G::Point(nodes[1])),
      2e-8);
    MatrixBatch<typename G::Point> same(3, n, n);
    for (int i = 0; i < 3; ++i) same[i] = nodes[0];
    const auto null = gfe::p1_cheeger_discrete_tension_log_contribution(g, same, s, rho);
    // a constant tensor field has zero discrete tension even when rho varies spatially
    EXPECT_NEAR(null.value, 0, 1e-23);
}
/// @brief covers planar specialization, three-dimensional tensors, general order and dynamic storage
TEST(CheegerSmoothing, JointGradientsAndGeometry) {
    check<2>();
    check<3>();
    check<4>();
    check<Dynamic>(2);
    check<Dynamic>(3);
}
/// @brief compares the specialized rotation differentials with the independent general SO implementation
TEST(CheegerSmoothing, SpecializedRotationDifferentials) {
    for (int n : {2, 3})
        for (double angle : {0., 1e-7, .2, 1.4, 3.}) {
            SkewSymmetricMatrix<double, Dynamic, Dynamic> omega(n, n), v(n, n);
            for (int i = 0; i < n; ++i)
                for (int j = i + 1; j < n; ++j) {
                    omega(i, j) = i == 0 && j == 1 ? angle : 0;
                    v(i, j) = .1 * (i + j + 1);
                }
            const manifold::internals::CheegerRotationDifferential<double, Dynamic> fast(omega);
            const manifold::internals::SOLogDifferential<double, Dynamic> reference(omega);
            const auto h = fast.hessian_action(v), ref = reference.hessian_action(v);
            // the closed two/three-dimensional formula agrees across zero, small angles and near-cut rotations
            EXPECT_LT((Matrix<double, Dynamic, Dynamic>(h - ref).norm()), 1e-12);
            const auto d = fast.target_action(v), rd = reference.target_action(v);
            // including the Lie bracket yields the same ordinary logarithm differential
            EXPECT_LT((Matrix<double, Dynamic, Dynamic>(d - rd).norm()), 1e-12);
        }
}
/// @brief checks active-support data losses and rejected scalar inputs at the public kernel boundary
TEST(CheegerSmoothing, VertexEdgeAndParameterContracts) {
    using G = manifold::CheegerLogEuclideanSPDGeometry<double, 2>;
    const G g;
    MatrixBatch<G::Point> nodes(3);
    nodes[0] = G::from_chart({.1, .2, .04});
    nodes[1] = G::from_chart({.2, .25, -.03});
    nodes[2] = G::from_chart({-.1, .1, .05});
    const G::Tangent observation = manifold::internals::cheeger_matrix<double>({.8, .1, .02});
    const std::array rho {.3, .6, .8};
    for (const std::array<double, 3> weights : {
           std::array {1., 0., 0.},
            std::array {.4, .6, 0.}
    }) {
        const auto f = gfe::p1_cheeger_frobenius_data_site_log_contribution(g, nodes, weights, observation, {}, rho);
        // vertices and open edges remain differentiable in their active nodal coefficients
        ASSERT_TRUE(f.converged());
        // zero-weight nodes have no data dependence on their tensor coefficients
        EXPECT_EQ(f.nodal_gradient[2].norm(), 0);
        // rho coefficients outside the active support have zero data derivative
        EXPECT_EQ(f.rho_gradient[2], 0);
    }
    auto invalid = rho;
    invalid[2] = 0;
    // invalid inactive rho still fails at the public field boundary
    EXPECT_THROW(
      gfe::p1_cheeger_frobenius_data_site_log_contribution(g, nodes, std::array {1., 0., 0.}, observation, {}, invalid),
      std::invalid_argument);
    const gfe::P1LumpedLaplacianStencil stencil {
      {1., 1., 1.},
      {{0, 1, -1.}, {1, 2, -1.}}
    };
    // tension rejects invalid rho before preparing any pair alignment
    EXPECT_THROW(gfe::p1_cheeger_discrete_tension_log_contribution(g, nodes, stencil, invalid), std::invalid_argument);
}
}   // namespace

namespace {
/// @brief recovers the exact nodal derivative regression from the TSPDE boundary probe
TEST(CheegerSmoothing, NodalVertexDerivatives) {
    using G = manifold::CheegerLogEuclideanSPDGeometry<double, 2>;
    const G geometry;
    MatrixBatch<G::Point> nodes(3);
    std::vector<G::Tangent> directions(3);
    for (int i = 0; i < 3; ++i) {
        nodes[i] = G::from_chart({.1 * i, .2, .1 * (i - 1)});
        directions[i] = manifold::internals::cheeger_matrix<double>({.1 + i, .2 - i, -.3 + i});
    }
    for (int active = 0; active < 3; ++active) {
        std::array<double, 3> weights {};
        weights[active] = 1;
        const auto linearization = gfe::p1_geodesic_linearization(geometry, nodes, weights);
        const auto analytic = linearization.nodal_log_jvp(directions);
        const G::Tangent log(matrix_log(nodes[active]));
        const auto plus = matrix_exp(G::Tangent(log + 1e-5 * directions[active]));
        const auto minus = matrix_exp(G::Tangent(log - 1e-5 * directions[active]));
        // inactive directions cannot affect a vertex value and central differences give its exponential derivative
        EXPECT_LT((Matrix<double, 2, 2>(analytic - (plus - minus) / 2e-5).norm()), 1e-8);
        // the exact nodal value is independent of rho
        EXPECT_EQ(linearization.rho_jvp().derivative.norm(), 0);
    }
}
/// @brief compares general spatial and scalar interpolation derivatives against refitted mean values
TEST(CheegerSmoothing, GeneralInterpolationDerivatives) {
    using G = manifold::CheegerLogEuclideanSPDGeometry<double, 3>;
    const G geometry;
    MatrixBatch<G::Point> nodes(3);
    for (int i = 0; i < 3; ++i) {
        G::Tangent x;
        for (int r = 0; r < 3; ++r)
            for (int c = 0; c <= r; ++c) x(r, c) = r == c ? .2 * (r + i) : .1 * std::sin(r + c + i);
        nodes[i] = matrix_exp(x);
    }
    std::array w {.2, .3, .5}, rho {.3, .6, .8}, dw {.1, -.2, .1}, dr {-.2, .3, .1};
    gfe::P1GeodesicLinearizationOptions options;
    options.mean.solver.gradient_tolerance = 1e-11;
    const auto fit = gfe::p1_geodesic_linearization(geometry, nodes, w, options, rho);
    for (bool weights : {false, true}) {
        auto wp = w, wm = w, rp = rho, rm = rho;
        for (int i = 0; i < 3; ++i) {
            if (weights) {
                wp[i] += 1e-5 * dw[i];
                wm[i] -= 1e-5 * dw[i];
            } else {
                rp[i] += 1e-5 * dr[i];
                rm[i] -= 1e-5 * dr[i];
            }
        }
        const auto p = gfe::p1_geodesic_linearization(geometry, nodes, wp, options, rp);
        const auto m = gfe::p1_geodesic_linearization(geometry, nodes, wm, options, rm);
        const auto analytic = weights ? fit.weight_jvp(dw) : fit.nodal_rho_jvp(dr);
        // refitted perturbations must remain stationary before serving as the difference oracle
        ASSERT_TRUE(p.result().converged() && m.result().converged() && analytic.converged());
        // weight directions include the induced local rho change whereas scalar directions hold weights fixed
        EXPECT_LT(
          (Matrix<double, 3, 3>(analytic.derivative - (p.result().value - m.result().value) / 2e-5).norm()), 1e-6);
    }
}
}   // namespace
