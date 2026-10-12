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
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <http://www.gnu.org/licenses/>.

#include <fdaPDE/geometric_discretization.h>
#include <gtest/gtest.h>

namespace {
using namespace fdapde;

/// @brief creates a noncommuting chart or direction with reproducible coefficients
template <int N> SymmetricMatrix<double, N> chart(int index, double scale) {
    SymmetricMatrix<double, N> value;
    for (int r = 0; r < N; ++r)
        for (int c = 0; c <= r; ++c)
            value(r, c) = scale * (r == c ? .2 * (index + 1) * (r + 1) : std::sin(index + 2 * r + c));
    return value;
}

/// @brief compares retained mean adjoint derivatives with independent perturbed branch solves
template <int N> void check_mean() {
    using G = manifold::CheegerLogEuclideanSPDGeometry<double, N>;
    using T = typename G::Tangent;
    constexpr double step = 2e-5, rho = .7, drho = .16;
    MatrixBatch<typename G::Point> nodes(3), plus(3), minus(3);
    std::array<T, 3> direction, second;
    for (int i = 0; i < 3; ++i) {
        const T x = chart<N>(i, .35);
        direction[i] = chart<N>(i + 2, .08);
        second[i] = chart<N>(i + 4, -.04);
        nodes[i] = matrix_exp(x);
        const T xp(x + step * direction[i]), xm(x - step * direction[i]);
        plus[i] = matrix_exp(xp);
        minus[i] = matrix_exp(xm);
    }
    gfe::P1GeodesicLinearizationOptions options;
    options.mean.solver.gradient_tolerance = 1e-10;
    options.mean.solver.max_iterations = 500;
    options.linear_solve.residual_tolerance = 1e-10;
    const T dual = chart<N>(5, .07), ddual = chart<N>(7, -.03);
    const T dual_plus(dual + step * ddual), dual_minus(dual - step * ddual);
    for (const std::array<double, 3> weights : {
           std::array {1.,    0., 0.        },
            std::array {.4,    .6, 0.        },
            std::array {.2,    .3, .5        },
           std::array {1e-13, .4, .6 - 1e-13}
    }) {
        const auto fit =
          gfe::p1_geodesic_linearization(G::from_rho(rho), nodes, std::span<const double>(weights), options);
        const auto p = gfe::p1_geodesic_linearization(
          G::from_rho(rho + step * drho), plus, std::span<const double>(weights), options);
        const auto m = gfe::p1_geodesic_linearization(
          G::from_rho(rho - step * drho), minus, std::span<const double>(weights), options);
        const T mean_direction = fit.nodal_log_rho_jvp(direction, drho);
        const T mean_difference((p.result().value - m.result().value) / (2 * step));
        // independently refitted endpoint means check the nodal-log and rho forward differential together
        EXPECT_LT((T(mean_direction - mean_difference).norm()), 3e-6);
        const auto [gradient, rho_gradient] = fit.nodal_log_rho_vjp_jvp(dual, ddual, direction, drho);
        const auto [gp, rp] = p.nodal_log_rho_vjp(dual_plus);
        const auto [gm, rm] = m.nodal_log_rho_vjp(dual_minus);
        for (int i = 0; i < 3; ++i) {
            const T reference((gp[i] - gm[i]) / (2 * step));
            // independent mean refits check the complete nodal adjoint derivative including inactive support
            EXPECT_LT((T(gradient[i] - reference).norm()), 5e-6);
        }
        // perturbing local rho checks the mixed scalar adjoint derivative without changing weights
        EXPECT_NEAR(rho_gradient, (rp - rm) / (2 * step), 5e-6);
        const T response = chart<N>(3, .11);
        const T residual(fit.result().value - response);
        const T second_mean = fit.nodal_log_rho_jvp(second, -.09);
        const auto [first_action, first_rho] = fit.nodal_log_rho_vjp_jvp(residual, mean_direction, direction, drho);
        const auto [second_action, second_rho] = fit.nodal_log_rho_vjp_jvp(residual, second_mean, second, -.09);
        double lhs = first_rho * -.09, rhs = second_rho * drho;
        for (int i = 0; i < 3; ++i) {
            lhs += manifold::internals::cheeger_inner(first_action[i], second[i]);
            rhs += manifold::internals::cheeger_inner(second_action[i], direction[i]);
        }
        // swapping two full tensor-rho directions checks the symmetry of the squared-loss Hessian
        EXPECT_NEAR(lhs, rhs, 2e-9);
    }
}

/// @brief compares selected pair logarithm and adjoint derivatives with independent perturbed pair searches
template <int N> void check_pair() {
    using Frame = manifold::internals::CheegerPairFrame<double, N>;
    using T = typename Frame::Sym;
    constexpr double step = 2e-5, rho = .7, drho = .16;
    const T x = chart<N>(0, .35), y = chart<N>(1, .35);
    const T dx = chart<N>(3, .08), dy = chart<N>(4, -.04);
    const T dual = chart<N>(5, .07), ddual = chart<N>(7, -.03);
    const T xp(x + step * dx), xm(x - step * dx), yp(y + step * dy), ym(y - step * dy);
    const T cp(dual + step * ddual), cm(dual - step * ddual);
    const Frame frame(x, y, rho), plus(xp, yp, rho + step * drho), minus(xm, ym, rho - step * drho);
    const auto variation = frame.linearize(dx, dy, drho);
    const auto lp = plus.logarithm(), lm = minus.logarithm();
    const T reference((lp - lm) / (2 * step));
    const auto logarithm_direction = frame.logarithm_jvp(variation);
    // a retained forward solve matches refitting both endpoint and rho perturbations
    EXPECT_LT((T(logarithm_direction - reference).norm()), 3e-6);
    const auto [base, target, parameter] = frame.pullback_jvp(dual, ddual, variation);
    const auto [bp, tp, rp] = plus.pullback(cp);
    const auto [bm, tm, rm] = minus.pullback(cm);
    const T base_reference((bp - bm) / (2 * step)), target_reference((tp - tm) / (2 * step));
    // differentiating the retained adjoint recovers the independently perturbed base covector
    EXPECT_LT((T(base - base_reference).norm()), 5e-6);
    // differentiating the moving target frame recovers the independently perturbed target covector
    EXPECT_LT((T(target - target_reference).norm()), 5e-6);
    // the scalar adjoint includes changes of both the implicit multiplier and the rotation logarithm
    EXPECT_NEAR(parameter, (rp - rm) / (2 * step), 5e-6);
}

/// @brief checks public curvature inputs and the requirement for a prepared pair factorization
template <int N> void check_contracts() {
    using G = manifold::CheegerLogEuclideanSPDGeometry<double, N>;
    using T = typename G::Tangent;
    const T value = chart<N>(0, .15), dual = chart<N>(1, .03);
    MatrixBatch<typename G::Point> nodes(3);
    for (int i = 0; i < 3; ++i) nodes[i] = matrix_exp(value);
    const auto fit = gfe::p1_geodesic_linearization(G(), nodes, std::array {.2, .3, .5});
    std::array<T, 3> directions {dual, dual, dual};
    const std::array<T, 2> short_directions {dual, dual};
    // the mean forward derivative rejects an incomplete nodal direction array
    EXPECT_THROW(static_cast<void>(fit.nodal_log_rho_jvp(short_directions)), std::invalid_argument);
    // the adjoint derivative rejects an incomplete nodal direction array before any implicit solve
    EXPECT_THROW(static_cast<void>(fit.nodal_log_rho_vjp_jvp(dual, dual, short_directions)), std::invalid_argument);
    // nonfinite local metric perturbations are rejected at the public curvature boundary
    EXPECT_THROW(
      static_cast<void>(fit.nodal_log_rho_vjp_jvp(dual, dual, directions, std::numeric_limits<double>::infinity())),
      std::invalid_argument);
    directions[0](0, 0) = std::numeric_limits<double>::quiet_NaN();
    // nonfinite nodal log directions cannot enter the retained lifted solve
    EXPECT_THROW(static_cast<void>(fit.nodal_log_rho_vjp_jvp(dual, dual, directions)), std::invalid_argument);
    const manifold::internals::CheegerPairFrame<double, N> pair(value, dual, .7, false);
    // a value-only pair search deliberately lacks the factorization required for derivative reuse
    EXPECT_THROW(static_cast<void>(pair.linearize(dual, dual)), std::logic_error);
}

// exact mean curvature preserves the planar scalar branch and the SPD3 multistart branch
TEST(CheegerCurvature, MeanAdjointDerivativesAndLossSymmetry) {
    check_mean<2>();
    check_mean<3>();
}

// exact pair curvature includes both endpoint directions and the local metric parameter
TEST(CheegerCurvature, PairAdjointDerivatives) {
    check_pair<2>();
    check_pair<3>();
}

// curvature rejects malformed directions and value-only frames in both supported tensor orders
TEST(CheegerCurvature, ValidatesPreparedBranchesAndDirections) {
    check_contracts<2>();
    check_contracts<3>();
}
}   // namespace
