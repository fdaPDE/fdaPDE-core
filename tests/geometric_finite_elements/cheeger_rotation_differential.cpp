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

#include <fdaPDE/geometric_discretization.h>
#include <gtest/gtest.h>

namespace {
using namespace fdapde;
namespace mi = fdapde::manifold::internals;
/// @brief creates deterministic noncommuting body tangents at the requested native order
template <int N> SkewSymmetricMatrix<double, N> body(int order, double phase) {
    SkewSymmetricMatrix<double, N> result;
    if constexpr (N == Dynamic) result.resize(order, order);
    for (int row = 0; row < order; ++row)
        for (int col = row + 1; col < order; ++col) result(row, col) = .3 * std::sin(phase + 2 * row + col);
    return result;
}
/// @brief compares complete skew coefficients without relying on compact storage multiplicities
double error(const auto& a, const auto& b) { return Matrix<double, Dynamic, Dynamic>(a - b).norm(); }
/// @brief checks ordinary logarithm-coefficient derivatives against independent general SO spectral operators
template <int N> void check_derivatives(int order) {
    using Skew = SkewSymmetricMatrix<double, N>;
    const auto direction = body<N>(order, .3), omega_direction = body<N>(order, 2.1), dual = body<N>(order, 1.2);
    const auto axis = body<N>(order, .7);
    constexpr double step = 1e-4;
    for (double angle : {0., 1e-12, 1e-6, .001, .1, 1., std::numbers::pi - 1e-6}) {
        const Skew omega(std::sqrt(2.) * angle / axis.norm() * axis);
        const mi::CheegerRotationDifferential<double, N> specialized(omega);
        const Skew plus(omega + step * omega_direction), minus(omega - step * omega_direction);
        const mi::SOLogDifferential<double, N> positive(plus), negative(minus), general(omega);
        // the specialized scalar Hessian agrees with the independent adjoint-spectrum implementation
        EXPECT_LT(error(specialized.hessian_action(direction), general.hessian_action(direction)), 2e-13)
          << order << " " << angle;
        const auto positive_hessian = positive.hessian_action(direction),
                   negative_hessian = negative.hessian_action(direction);
        const Skew finite_hessian((positive_hessian - negative_hessian) / (2 * step));
        const auto analytic_hessian = specialized.hessian_action_jvp(direction, omega_direction);
        // centered changes in the general spectral operator independently check the ordinary coefficient derivative
        EXPECT_LT(error(analytic_hessian, finite_hessian), 3e-9) << order << " " << angle;
        const auto positive_target = positive.target_action(direction),
                   negative_target = negative.target_action(direction);
        const Skew finite_target((positive_target - negative_target) / (2 * step));
        // centered general target-log actions check the differentiated commutator as well as Hessian curvature
        EXPECT_LT(error(specialized.target_action_jvp(direction, omega_direction), finite_target), 3e-9)
          << order << " " << angle;
        const auto dual_action = specialized.hessian_action_jvp(dual, omega_direction);
        // differentiating a self-adjoint Hessian preserves its full-Frobenius adjoint pairing
        EXPECT_NEAR(mi::cheeger_inner(dual, analytic_hessian), mi::cheeger_inner(direction, dual_action), 2e-15)
          << order << " " << angle;
    }
}

// static and runtime SO(2)/SO(3) derivative kernels retain the general operator's values and independent derivatives
TEST(CheegerRotationDifferential, AnalyticCoefficientDerivatives) {
    check_derivatives<2>(2);
    check_derivatives<3>(3);
    check_derivatives<Dynamic>(2);
    check_derivatives<Dynamic>(3);
}

// removable zero-frequency limits and unsupported higher-order derivatives have explicit behavior
TEST(CheegerRotationDifferential, ZeroFrequencyAndGeneralOrderContract) {
    const SkewSymmetricMatrix<double, 3> zero;
    const auto direction = body<3>(3, .4), omega_direction = body<3>(3, 1.6);
    const mi::CheegerRotationDifferential<double, 3> specialized(zero);
    // the Hessian is even in the logarithm, so its ordinary derivative vanishes exactly at identity
    EXPECT_EQ(specialized.hessian_action_jvp(direction, omega_direction).norm(), 0.);
    const SkewSymmetricMatrix<double, 3> commutator(
      (.5 * (omega_direction * direction - direction * omega_direction)).template as_skew_symmetric<Lower>());
    // the identity target-log derivative retains precisely the independently formed half-commutator
    EXPECT_LT(error(specialized.target_action_jvp(direction, omega_direction), commutator), 1e-16);
    const auto general_direction = body<4>(4, .4);
    const mi::CheegerRotationDifferential<double, 4> general(general_direction);
    // the new scalar derivative does not silently approximate an unsupported general-order operator
    EXPECT_THROW(general.hessian_action_jvp(general_direction, general_direction), std::logic_error);
    // the target derivative shares the same explicit higher-order restriction
    EXPECT_THROW(general.target_action_jvp(general_direction, general_direction), std::logic_error);
}
}   // namespace
