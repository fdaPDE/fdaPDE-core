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

#include <cmath>
#include <limits>
#include <type_traits>

namespace {

using namespace fdapde;
using namespace fdapde::manifold;

using Sym = SymmetricMatrix<double, 2>;
using SPD = SPDMatrix<double, 2>;
using CachedSPD = SPDMatrix<double, 2, Cache::Union<Cache::Log, Cache::Spectral, Cache::LogDividedDifferences>>;
using DynamicSPD = SPDMatrix<double, Dynamic, typename CachedSPD::CachePolicy>;

/// @brief forms the independent two-by-two congruence oracle by explicit coefficient sums
template <typename Point> Sym ambient_congruence(const Sym& outer, const Point& middle) {
    Sym result;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j <= i; ++j) {
            double coefficient = 0;
            for (int k = 0; k < 2; ++k) {
                for (int l = 0; l < 2; ++l) coefficient += outer(i, k) * middle(k, l) * outer(l, j);
            }
            result(i, j) = coefficient;
        }
    }
    return result;
}

/// @brief pairs the full matrix coefficients so symmetric off-diagonal entries count twice
template <typename Lhs, typename Rhs> double ambient_inner(const Lhs& lhs, const Rhs& rhs) {
    double result = 0;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) result += lhs(i, j) * rhs(i, j);
    }
    return result;
}

/// @brief evaluates a quadratic whose noncommuting ambient Hessian is direction-dependent
template <typename Point> double ambient_cost(const Point& point) {
    const Sym weight(Vector<double, 3> {1.2, -0.3, 0.8});
    const Sym linear(Vector<double, 3> {0.2, 0.4, -0.1});
    const auto weighted = ambient_congruence(weight, point);
    return 0.5 * ambient_inner(point, weighted) - ambient_inner(linear, point);
}

/// @brief evaluates the Frobenius gradient of the weighted ambient quadratic independently of the metric
template <typename Point> Sym ambient_gradient(const Point& point) {
    const Sym weight(Vector<double, 3> {1.2, -0.3, 0.8});
    const Sym linear(Vector<double, 3> {0.2, 0.4, -0.1});
    const auto weighted = ambient_congruence(weight, point);
    return Sym(weighted - linear);
}

/// @brief evaluates the ambient Hessian action using the independent quadratic congruence oracle
Sym ambient_hessian(const Sym& direction) {
    const Sym weight(Vector<double, 3> {1.2, -0.3, 0.8});
    return ambient_congruence(weight, direction);
}

/// @brief compares a converted Hessian to transported gradients and scalar geodesic differences
template <typename Geometry, typename Point>
void check_noncommuting_derivatives(const Geometry& geometry, const Point& point) {
    const Sym direction(Vector<double, 3> {0.17, -0.11, 0.28});
    const Sym other(Vector<double, 3> {-0.23, 0.19, 0.08});
    const auto egrad = ambient_gradient(point);
    const auto ehess = ambient_hessian(direction);
    const auto gradient = geometry.euclidean_to_riemannian_gradient(point, egrad);
    const auto hessian = geometry.euclidean_to_riemannian_hessian(point, egrad, ehess, direction);
    const auto other_ehess = ambient_hessian(other);
    const auto other_hessian = geometry.euclidean_to_riemannian_hessian(point, egrad, other_ehess, other);

    // the converted gradient is the metric dual of the independently computed ambient directional derivative
    EXPECT_NEAR(geometry.inner_product(point, gradient, direction), ambient_inner(egrad, direction), 2.0e-12);
    // the connection-corrected Hessian remains self-adjoint for two noncommuting ambient directions
    EXPECT_NEAR(
      geometry.inner_product(point, other, hessian), geometry.inner_product(point, other_hessian, direction), 2.0e-12);

    const double step = 1.0e-4;
    const auto forward = geometry.exponential(point, direction, step);
    const auto backward = geometry.exponential(point, direction, -step);
    const auto forward_egrad = ambient_gradient(forward);
    const auto backward_egrad = ambient_gradient(backward);
    const auto forward_gradient = geometry.euclidean_to_riemannian_gradient(forward, forward_egrad);
    const auto backward_gradient = geometry.euclidean_to_riemannian_gradient(backward, backward_egrad);
    const auto forward_at_base = geometry.transport(forward, point, forward_gradient);
    const auto backward_at_base = geometry.transport(backward, point, backward_gradient);
    const Sym centered((forward_at_base - backward_at_base) * (0.5 / step));
    const Sym error(centered - hessian);
    // parallel transport removes changing tangent bases before differencing the gradient along the geodesic
    EXPECT_LT(std::sqrt(ambient_inner(error, error)), 2.0e-8);
    // the Hessian quadratic form equals the scalar second derivative of the cost along the same geodesic
    EXPECT_NEAR(
      geometry.inner_product(point, direction, hessian),
      (ambient_cost(forward) - 2 * ambient_cost(point) + ambient_cost(backward)) / (step * step), 2.0e-6);
}

/// @brief compares native Hessian coefficients against a prepared independent owner oracle
template <typename Actual, typename Expected> void expect_hessian_near(const Actual& actual, const Expected& expected) {
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j <= i; ++j) {
            // cached or dynamic operands yield the same coefficient as the uncached fixed-order reference
            EXPECT_NEAR(actual(i, j), expected(i, j), 2.0e-12);
        }
    }
}

}   // namespace

// the log-Euclidean conversion includes chart curvature for a quadratic with a nonidentity ambient Hessian
TEST(AmbientSPDDerivatives, LogEuclideanMatchesNoncommutingGeodesicDifferences) {
    const SPD point(Vector<double, 3> {2.1, 0.35, 1.4});
    check_noncommuting_derivatives(LogEuclideanGeometry<SPD>(), point);
}

// the affine-invariant conversion includes the Levi-Civita correction in independently transported differences
TEST(AmbientSPDDerivatives, AffineInvariantMatchesNoncommutingGeodesicDifferences) {
    const SPD point(Vector<double, 3> {2.1, 0.35, 1.4});
    check_noncommuting_derivatives(AffineInvariantGeometry<SPD>(), point);
}

// typed cache owners and borrowed batch views preserve the conversion coefficients at fixed and dynamic orders
TEST(AmbientSPDDerivatives, PreservesCachedOwnersAndBatchViews) {
    const SPD point(Vector<double, 3> {2.1, 0.35, 1.4});
    const CachedSPD cached(Vector<double, 3> {2.1, 0.35, 1.4});
    const DynamicSPD dynamic(Vector<double, 3> {2.1, 0.35, 1.4});
    MatrixBatch<CachedSPD> fixed_batch(1);
    fixed_batch[0] = cached;
    MatrixBatch<DynamicSPD> dynamic_batch(1, 2, 2);
    dynamic_batch[0] = dynamic;
    const auto& fixed_views = fixed_batch;
    const auto& dynamic_views = dynamic_batch;
    const Sym egrad = ambient_gradient(point);
    const Sym direction(Vector<double, 3> {0.17, -0.11, 0.28});
    const Sym ehess = ambient_hessian(direction);

    const auto expected_le =
      LogEuclideanGeometry<SPD>().euclidean_to_riemannian_hessian(point, egrad, ehess, direction);
    const auto expected_ai =
      AffineInvariantGeometry<SPD>().euclidean_to_riemannian_hessian(point, egrad, ehess, direction);
    const LogEuclideanGeometry<CachedSPD> cached_le;
    const AffineInvariantGeometry<CachedSPD> cached_ai;
    const LogEuclideanGeometry<DynamicSPD> dynamic_le(2);
    const AffineInvariantGeometry<DynamicSPD> dynamic_ai(2);
    const SymmetricMatrix<double, Dynamic> dynamic_gradient(egrad);
    const SymmetricMatrix<double, Dynamic> dynamic_hessian(ehess);
    const SymmetricMatrix<double, Dynamic> dynamic_direction(direction);

    // cached logarithm and spectral state produce the uncached chart Hessian coefficients
    expect_hessian_near(cached_le.euclidean_to_riemannian_hessian(cached, egrad, ehess, direction), expected_le);
    // borrowed cached SPD views provide the same log-Euclidean conversion as the owning point
    expect_hessian_near(
      cached_le.euclidean_to_riemannian_hessian(fixed_views[0], egrad, ehess, direction), expected_le);
    // the affine-invariant conversion accepts the exact selected cache owner without changing its coefficients
    expect_hessian_near(cached_ai.euclidean_to_riemannian_hessian(cached, egrad, ehess, direction), expected_ai);
    // borrowed cached views preserve the affine-invariant connection correction
    expect_hessian_near(
      cached_ai.euclidean_to_riemannian_hessian(fixed_views[0], egrad, ehess, direction), expected_ai);
    // runtime-order views and tangents reproduce the fixed-order log-Euclidean Hessian
    expect_hessian_near(
      dynamic_le.euclidean_to_riemannian_hessian(
        dynamic_views[0], dynamic_gradient, dynamic_hessian, dynamic_direction),
      expected_le);
    // runtime-order affine-invariant products preserve the same Hessian action as fixed-order storage
    expect_hessian_near(
      dynamic_ai.euclidean_to_riemannian_hessian(
        dynamic_views[0], dynamic_gradient, dynamic_hessian, dynamic_direction),
      expected_ai);
}

// in one dimension both metrics reduce to d(log x)^2 and have an analytic gradient-dependent Hessian correction
TEST(AmbientSPDDerivatives, ScalarConnectionUsesEuclideanGradient) {
    using Point = SPDMatrix<double, 1>;
    using Tangent = SymmetricMatrix<double, 1>;
    const Point point(Vector<double, 1> {2.3});
    Tangent egrad;
    Tangent ehess;
    Tangent direction;
    egrad(0, 0) = -0.7;
    ehess(0, 0) = 0.4;
    direction(0, 0) = 0.2;
    const double expected = 2.3 * 2.3 * 0.4 + 2.3 * 0.2 * -0.7;
    const auto le = LogEuclideanGeometry<Point>().euclidean_to_riemannian_hessian(point, egrad, ehess, direction);
    const auto ai = AffineInvariantGeometry<Point>().euclidean_to_riemannian_hessian(point, egrad, ehess, direction);
    // the scalar log chart contributes x times direction times the ambient gradient in addition to x squared ehess
    EXPECT_NEAR(le(0, 0), expected, 1.0e-12);
    // the affine-invariant scalar connection gives the same independent logarithmic-coordinate oracle
    EXPECT_NEAR(ai(0, 0), expected, 1.0e-12);
}

// public Hessian conversions reject every incompatible runtime operand and nonfinite ambient derivative
TEST(AmbientSPDDerivatives, ValidatesRuntimeShapesAndFiniteDerivatives) {
    const LogEuclideanGeometry<DynamicSPD> le(2);
    const AffineInvariantGeometry<DynamicSPD> ai(2);
    const DynamicSPD point(Vector<double, 3> {1, 0, 1});
    const DynamicSPD wrong_point(Vector<double, 6>({1, 0, 1, 0, 0, 1}));
    const SymmetricMatrix<double, Dynamic> zero(2, 2);
    const SymmetricMatrix<double, Dynamic> wrong(3, 3);
    auto nonfinite = zero;
    nonfinite(0, 1) = std::numeric_limits<double>::quiet_NaN();
    // a runtime point order mismatch is rejected before the log chart can be evaluated
    EXPECT_THROW(le.euclidean_to_riemannian_hessian(wrong_point, zero, zero, zero), std::invalid_argument);
    // the log-Euclidean conversion checks the ambient gradient order independently
    EXPECT_THROW(le.euclidean_to_riemannian_hessian(point, wrong, zero, zero), std::invalid_argument);
    // the log-Euclidean conversion checks the supplied Hessian action order independently
    EXPECT_THROW(le.euclidean_to_riemannian_hessian(point, zero, wrong, zero), std::invalid_argument);
    // the log-Euclidean conversion checks the tangent direction order independently
    EXPECT_THROW(le.euclidean_to_riemannian_hessian(point, zero, zero, wrong), std::invalid_argument);
    // the log-Euclidean conversion rejects nonfinite coefficients rather than forwarding them to spectral operations
    EXPECT_THROW(le.euclidean_to_riemannian_hessian(point, nonfinite, zero, zero), std::invalid_argument);
    // an affine-invariant point order mismatch is rejected before its congruence products are evaluated
    EXPECT_THROW(ai.euclidean_to_riemannian_hessian(wrong_point, zero, zero, zero), std::invalid_argument);
    // the affine-invariant conversion checks the ambient gradient order independently
    EXPECT_THROW(ai.euclidean_to_riemannian_hessian(point, wrong, zero, zero), std::invalid_argument);
    // the affine-invariant conversion checks the supplied Hessian action order independently
    EXPECT_THROW(ai.euclidean_to_riemannian_hessian(point, zero, wrong, zero), std::invalid_argument);
    // the affine-invariant conversion checks the tangent direction order independently
    EXPECT_THROW(ai.euclidean_to_riemannian_hessian(point, zero, zero, wrong), std::invalid_argument);
    // the affine-invariant conversion rejects a nonfinite Hessian action before forming any correction
    EXPECT_THROW(ai.euclidean_to_riemannian_hessian(point, zero, nonfinite, zero), std::invalid_argument);
}
