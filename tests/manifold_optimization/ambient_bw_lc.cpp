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

#include <fdaPDE/manifold_optimization.h>
#include <gtest/gtest.h>

namespace {
using namespace fdapde;
using namespace fdapde::manifold;

/// @brief contracts all matrix entries independently of packed symmetric storage
template <typename First, typename Second> double frobenius_pairing(const First& first, const Second& second) {
    double value = 0;
    for (int i = 0; i < first.rows(); ++i)
        for (int j = 0; j < first.rows(); ++j) value += first(i, j) * second(i, j);
    return value;
}

/// @brief supplies a smooth ambient objective with nonconstant Hessian and noncommuting matrix data
template <typename Sym> struct AmbientObjective {
    Sym target;
    Sym weights;

    /// @brief evaluates a Frobenius residual plus a fourth-power linear functional
    template <typename Point> double cost(const Point& point) const {
        const Sym residual(point - target);
        const double trace = frobenius_pairing(point, weights);
        return 0.5 * frobenius_pairing(residual, residual) + 0.25 * std::pow(trace, 4);
    }
    /// @brief returns the exact ambient gradient of the quadratic and quartic terms
    template <typename Point> Sym egrad(const Point& point) const {
        Sym result(point - target);
        const double coefficient = std::pow(frobenius_pairing(point, weights), 3);
        for (int i = 0; i < point.rows(); ++i)
            for (int j = 0; j <= i; ++j) result(i, j) = static_cast<double>(result(i, j)) + coefficient * weights(i, j);
        return result;
    }
    /// @brief applies the ambient Hessian including the quartic rank-one coupling
    template <typename Point> Sym ehess(const Point& point, const Sym& direction) const {
        Sym result(direction);
        const double trace = frobenius_pairing(point, weights);
        const double coefficient = 3 * trace * trace * frobenius_pairing(direction, weights);
        for (int i = 0; i < point.rows(); ++i)
            for (int j = 0; j <= i; ++j) result(i, j) = static_cast<double>(result(i, j)) + coefficient * weights(i, j);
        return result;
    }
};

/// @brief checks metric derivatives against independent geodesic differences of a smooth ambient objective
template <typename Geometry> void ambient_derivative_oracles(const Geometry& geometry) {
    using Point = typename Geometry::Point;
    using Sym = typename Geometry::Tangent;
    const int order = geometry.order();
    const Point identity = [&] {
        if constexpr (Point::Rows == Dynamic)
            return Point::Identity(order);
        else
            return Point::Identity();
    }();
    Sym base(identity), target(identity), weights(identity), first(identity), second(identity);
    for (int i = 0; i < order; ++i)
        for (int j = 0; j <= i; ++j) {
            base(i, j) = i == j ? 2 + 0.5 * i : 0.08 * (i + j + 1);
            target(i, j) = i == j ? 1.4 + 0.2 * i : -0.035 * (i + j + 1);
            weights(i, j) = i == j ? 0.08 - 0.03 * i : 0.07 - 0.015 * (i + j);
            first(i, j) = i == j ? 0.2 - 0.05 * i : 0.13 - 0.03 * (i + j);
            second(i, j) = i == j ? -0.15 + 0.07 * i : -0.11 + 0.025 * (i + j);
        }
    const Point point(base);
    const AmbientObjective<Sym> objective {target, weights};
    const auto gradient = objective.egrad(point);
    const auto riemannian = geometry.euclidean_to_riemannian_gradient(point, gradient);
    const auto first_hessian =
      geometry.euclidean_to_riemannian_hessian(point, gradient, objective.ehess(point, first), first);
    const auto second_hessian =
      geometry.euclidean_to_riemannian_hessian(point, gradient, objective.ehess(point, second), second);
    constexpr double gradient_step = 1e-5;
    const double finite_gradient = (objective.cost(geometry.exponential(point, first, gradient_step)) -
                                    objective.cost(geometry.exponential(point, first, -gradient_step))) /
                                   (2 * gradient_step);
    // the converted gradient recovers an independent geodesic directional derivative of the full ambient cost
    EXPECT_NEAR(geometry.inner_product(point, riemannian, first), finite_gradient, 2e-9);
    // both Hessian actions have the metric self-adjointness required of a smooth scalar objective
    EXPECT_NEAR(
      geometry.inner_product(point, first_hessian, second), geometry.inner_product(point, first, second_hessian),
      2e-12);
    const auto second_difference = [&](const Sym& direction, double step) {
        return (objective.cost(geometry.exponential(point, direction, step)) - 2 * objective.cost(point) +
                objective.cost(geometry.exponential(point, direction, -step))) /
               (step * step);
    };
    const auto directional_hessian = [&](const Sym& direction) {
        constexpr double step = 1e-3;
        return (4 * second_difference(direction, step / 2) - second_difference(direction, step)) / 3;
    };
    // Richardson extrapolation of geodesic cost curvature detects missing connection and quartic Hessian terms
    EXPECT_NEAR(geometry.inner_product(point, first_hessian, first), directional_hessian(first), 2e-8);
    const auto sum = geometry.linear_combination(point, 1, first, 1, second);
    const auto difference = geometry.linear_combination(point, 1, first, -1, second);
    const double mixed = (directional_hessian(sum) - directional_hessian(difference)) / 4;
    // polarization of independent cost curvatures validates the mixed Hessian component at a noncommuting point
    EXPECT_NEAR(geometry.inner_product(point, first_hessian, second), mixed, 2e-8);
    MatrixBatch<Point> batch(1, order, order);
    batch[0] = point;
    const auto view_hessian =
      geometry.euclidean_to_riemannian_hessian(std::as_const(batch)[0], gradient, objective.ehess(point, first), first);
    // borrowing a certified batch view preserves every coefficient of the owner-based conversion
    EXPECT_DOUBLE_EQ((view_hessian - first_hessian).norm(), 0);
}
}   // namespace

// the BW conversion matches independent nonlinear geodesic curvature for fixed and dynamic matrix orders
TEST(AmbientBuresWasserstein, ExactHessianConversionAcrossMatrixOrders) {
    ambient_derivative_oracles(BuresWassersteinSPDGeometry<double, 2> {});
    ambient_derivative_oracles(BuresWassersteinSPDGeometry<double, 3> {});
    ambient_derivative_oracles(BuresWassersteinSPDGeometry<double, Dynamic> {3});
}

// the log-Cholesky conversion differentiates the flat-chart pullback for noncommuting fixed and dynamic inputs
TEST(AmbientLogCholesky, ExactHessianConversionAcrossMatrixOrders) {
    ambient_derivative_oracles(LogCholeskySPDGeometry<double, 2> {});
    ambient_derivative_oracles(LogCholeskySPDGeometry<double, 3> {});
    ambient_derivative_oracles(LogCholeskySPDGeometry<double, Dynamic> {3});
}

// malformed public Hessian inputs fail before Lyapunov or triangular differential evaluation
TEST(AmbientSPDConversions, RejectsWrongShapesAndNonfiniteDerivatives) {
    const SPDMatrix<double, Dynamic> point = SPDMatrix<double, Dynamic>::Identity(3);
    SymmetricMatrix<double, Dynamic> correct(3, 3), wrong(2, 2);
    const BuresWassersteinSPDGeometry<double, Dynamic> bw(3);
    const LogCholeskySPDGeometry<double, Dynamic> lc(3);
    // a BW ambient Hessian action must have the point's matrix order
    EXPECT_THROW(bw.euclidean_to_riemannian_hessian(point, correct, wrong, correct), std::invalid_argument);
    // a chart-direction tangent must have the point's matrix order
    EXPECT_THROW(lc.euclidean_to_riemannian_hessian(point, correct, correct, wrong), std::invalid_argument);
    correct(1, 0) = std::numeric_limits<double>::infinity();
    // a nonfinite Euclidean gradient cannot enter the BW covariant conversion
    EXPECT_THROW(bw.euclidean_to_riemannian_hessian(point, correct, correct, correct), std::invalid_argument);
    // a nonfinite Euclidean gradient cannot enter the log-Cholesky chart pullback
    EXPECT_THROW(lc.euclidean_to_riemannian_hessian(point, correct, correct, correct), std::invalid_argument);
}
