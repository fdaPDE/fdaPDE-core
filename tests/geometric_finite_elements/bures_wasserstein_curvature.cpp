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
template <int N> using Dense = Matrix<double, N, N>;
template <int N> using Sym = SymmetricMatrix<double, N>;

/// @brief solves the full vectorized Sylvester system independently of spectral BW kernels
template <int N> Dense<N> sylvester(const Dense<N>& point, const Dense<N>& rhs) {
    Matrix<double, N * N, N * N> system;
    Vector<double, N * N> vector;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            vector[i * N + j] = rhs(i, j);
            for (int a = 0; a < N; ++a)
                for (int b = 0; b < N; ++b)
                    system(i * N + j, a * N + b) = (b == j ? point(i, a) : 0.) + (i == a ? point(b, j) : 0.);
        }
    const Vector<double, N * N> solution {PartialPivLU(system).solve(vector)};
    Dense<N> result;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) result(i, j) = solution[i * N + j];
    const Dense<N> residual(point * result + result * point - rhs);
    // the independently assembled system must solve AX plus XA equals rhs to a stricter error than the derivative
    // oracle
    EXPECT_LE(residual.norm(), 1e-11 * (1 + rhs.norm()));
    return result;
}

/// @brief evaluates the BW Levi-Civita connection through independent full Sylvester solves
template <int N> Dense<N> connection(const Dense<N>& point, const Dense<N>& first, const Dense<N>& second) {
    const auto a = sylvester(point, first), b = sylvester(point, second);
    return Dense<N>(-1. * (a * point * b + b * point * a));
}

/// @brief pairs tangent matrices using an independent BW metric solve
template <int N> double metric(const Dense<N>& point, const Dense<N>& first, const Dense<N>& second) {
    const auto lift = sylvester(point, first);
    double value = 0;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) value += .5 * lift(i, j) * second(i, j);
    return value;
}

/// @brief creates a symmetric owner without borrowing a temporary dense expression
template <int N> Sym<N> symmetric(const Dense<N>& value) {
    Sym<N> result;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j <= i; ++j) result(i, j) = .5 * (value(i, j) + value(j, i));
    return result;
}

/// @brief checks covariant third distance derivatives against endpoint perturbations and metric adjoints
template <int N> void check_curvature() {
    using Geometry = manifold::BuresWassersteinSPDGeometry<double, N>;
    using Point = typename Geometry::Point;
    const Geometry geometry;
    for (double scale : {.01, 1., 100.})
        for (int sample = 0; sample < 20; ++sample) {
            Dense<N> a, b, u, v, w, z;
            for (int i = 0; i < N; ++i)
                for (int j = 0; j < N; ++j) {
                    a(i, j) = std::sin(.31 * sample + 2 * i + j + .2);
                    b(i, j) = std::cos(.27 * sample + i + 3 * j + .7);
                    u(i, j) = scale * .1 * std::sin(.41 * sample + 2 * i + j + .3);
                    v(i, j) = scale * .1 * std::cos(.37 * sample + i + 3 * j + .6);
                    w(i, j) = scale * .1 * std::sin(.43 * sample + 3 * i + j + .9);
                    z(i, j) = scale * .1 * std::cos(.23 * sample + i + 2 * j + 1.2);
                }
            u = symmetric(u);
            v = symmetric(v);
            w = symmetric(w);
            z = symmetric(z);
            const IdentityMatrix<double, N, N> identity;
            Dense<N> p(scale * (a * a.transpose() + identity));
            Dense<N> q(scale * (b * b.transpose() + .6 * identity));
            if (sample % 5 == 0) q = p;
            if (sample % 5 == 1) q = p + scale * 1e-7 * identity;
            if (sample % 5 == 2) p = scale * identity;
            if (sample % 5 == 3) {
                p = scale * identity;
                p(0, 0) *= 1000.;
            }
            const Point base(symmetric(p)), target(symmetric(q));
            const auto base_direction = symmetric(u), target_direction = symmetric(v), action_direction = symmetric(w);
            const auto output_dual = symmetric(z);
            const auto frame = geometry.relative_frame(base, target);
            const Dense<N> analytic(geometry.half_squared_distance_hessian_covariant_jvp(
              frame, base_direction, target_direction, action_direction));
            const auto input_connection = connection(p, u, w);
            const Dense<N> parallel_direction(-1. * input_connection);
            const Dense<N> hessian(geometry.half_squared_distance_hessian_vector(frame, action_direction));
            double best_error = std::numeric_limits<double>::infinity();
            for (double step : {1e-3, 3e-4, 1e-4, 3e-5}) {
                const Dense<N> pp(p + step * u), pm(p - step * u), qp(q + step * v), qm(q - step * v);
                const Dense<N> wp(w + step * parallel_direction), wm(w - step * parallel_direction);
                const Point base_plus(symmetric(pp)), base_minus(symmetric(pm));
                const Point target_plus(symmetric(qp)), target_minus(symmetric(qm));
                const Dense<N> hp(geometry.half_squared_distance_hessian_vector(base_plus, target_plus, symmetric(wp)));
                const Dense<N> hm(
                  geometry.half_squared_distance_hessian_vector(base_minus, target_minus, symmetric(wm)));
                const auto output_connection = connection(p, u, hessian);
                const Dense<N> finite((hp - hm) / (2 * step) + output_connection);
                best_error = std::min(best_error, Dense<N>(finite - analytic).norm() / (scale + analytic.norm()));
            }
            // endpoint finite differences include independently computed input and output connection terms
            EXPECT_LT(best_error, 1e-7);
            const auto dual =
              geometry.half_squared_distance_hessian_covariant_vjp(frame, action_direction, output_dual);
            const double left = metric(p, z, analytic);
            const double right = metric(p, u, Dense<N>(dual.first)) + metric(q, v, Dense<N>(dual.second));
            // the complete adjoint pairs in two distinct endpoint metrics against the covariant JVP
            EXPECT_LT(std::abs(left - right) / (scale + std::abs(left)), 1e-11);
            const auto zero = geometry.zero_tangent(base);
            const Dense<N> target_action(
              geometry.half_squared_distance_hessian_covariant_jvp(frame, zero, target_direction, action_direction));
            const Dense<N> target_dual(
              geometry.half_squared_distance_hessian_covariant_target_vjp(frame, action_direction, output_dual));
            const double target_left = metric(p, z, target_action), target_right = metric(q, v, target_dual);
            // the specialized target adjoint agrees without the base endpoint contribution
            EXPECT_LT(std::abs(target_left - target_right) / (scale + std::abs(target_left)), 1e-11);
            if (sample % 5 == 0) {
                const Dense<N> same(geometry.half_squared_distance_hessian_covariant_jvp(
                  frame, base_direction, base_direction, action_direction));
                // moving coincident endpoints together preserves their identity distance Hessian
                EXPECT_LT(same.norm() / scale, 1e-10);
            }
        }
}

// BW third derivatives remain accurate at multiple scales, repeated spectra and unequal endpoint metrics
TEST(BuresWassersteinCurvature, CovariantEndpointDerivativesAndMetricAdjoints) {
    check_curvature<2>();
    check_curvature<3>();
}
}   // namespace
