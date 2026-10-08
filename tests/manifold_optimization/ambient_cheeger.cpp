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

/// @brief constructs a noncommuting native symmetric matrix with bounded deterministic coefficients
template <typename Sym> Sym symmetric_fixture(int order, double phase) {
    Sym result;
    if constexpr (Sym::Rows == Dynamic) result.resize(order, order);
    for (int i = 0; i < order; ++i)
        for (int j = i; j < order; ++j)
            result(i, j) = .23 * std::sin(phase + 1.3 * i + .7 * j) + (i == j ? .15 * i : 0);
    return result;
}

/// @brief evaluates the Frobenius pairing independently of the geometry metric
template <typename A, typename B> double frobenius(const A& a, const B& b) {
    double value = 0;
    for (int i = 0; i < a.rows(); ++i)
        for (int j = 0; j < a.cols(); ++j) value += a(i, j) * b(i, j);
    return value;
}

/// @brief evaluates a quadratic ambient loss with an anisotropic rank-one Hessian term
template <typename Point, typename Target, typename Sym>
double ambient_cost(const Point& p, const Target& target, const Sym& coupling) {
    double value = .1 * std::pow(frobenius(p, coupling), 2);
    for (int i = 0; i < p.rows(); ++i)
        for (int j = 0; j < p.cols(); ++j) value += .5 * std::pow(p(i, j) - target(i, j), 2);
    return value;
}

/// @brief differentiates the scalar cost twice along the geometry's exact geodesic
template <typename Geometry, typename Point, typename Sym, typename Target>
double geodesic_second_derivative(
  const Geometry& geometry, const Point& point, const Sym& direction, const Target& target, const Sym& coupling) {
    constexpr double step = 2e-4;
    const double center = ambient_cost(point, target, coupling);
    auto second = [&](double h) {
        return (ambient_cost(geometry.exponential(point, direction, h), target, coupling) - 2 * center +
                ambient_cost(geometry.exponential(point, direction, -h), target, coupling)) /
               (h * h);
    };
    return (4 * second(step / 2) - second(step)) / 3;
}

/// @brief compares ambient conversions to gradient duality and independently differentiated geodesic costs
template <typename Geometry> void check_ambient_derivatives(const Geometry& geometry, bool repeated = false) {
    using Sym = typename Geometry::Tangent;
    using Point = typename Geometry::Point;
    const int n = geometry.order();
    Sym logs = symmetric_fixture<Sym>(n, .6);
    if (repeated) {
        for (int i = 0; i < n; ++i)
            for (int j = i; j < n; ++j) logs(i, j) = i == j ? .2 : 0;
    }
    const Point point(matrix_exp(logs));
    const Sym target_logs = symmetric_fixture<Sym>(n, 1.9);
    const Point target(matrix_exp(target_logs));
    const Sym coupling = symmetric_fixture<Sym>(n, 2.7);
    const Sym first = symmetric_fixture<Sym>(n, -.8), second = symmetric_fixture<Sym>(n, 3.6);
    const double dual = .2 * frobenius(point, coupling);
    const Sym egrad(point - target + dual * coupling);
    const Sym ehess(first + .2 * frobenius(first, coupling) * coupling);
    const Sym other_ehess(second + .2 * frobenius(second, coupling) * coupling);
    const auto grad = geometry.euclidean_to_riemannian_gradient(point, egrad);
    const auto hess = geometry.euclidean_to_riemannian_hessian(point, egrad, ehess, first);
    const auto other_hess = geometry.euclidean_to_riemannian_hessian(point, egrad, other_ehess, second);
    // inverse-metric conversion is characterized by equality with the ambient Frobenius directional pairing
    EXPECT_NEAR(geometry.inner_product(point, grad, first), frobenius(egrad, first), 2e-11);
    const double diagonal_oracle = geodesic_second_derivative(geometry, point, first, target, coupling);
    // a covariant Hessian paired with its direction equals the scalar second derivative along an exact geodesic
    EXPECT_NEAR(geometry.inner_product(point, hess, first), diagonal_oracle, 8e-7);
    const Sym sum(first + second), difference(first - second);
    const double mixed_oracle = (geodesic_second_derivative(geometry, point, sum, target, coupling) -
                                 geodesic_second_derivative(geometry, point, difference, target, coupling)) /
                                4;
    // polarization of geodesic second derivatives checks a genuinely mixed Hessian component
    EXPECT_NEAR(geometry.inner_product(point, hess, second), mixed_oracle, 8e-7);
    // exchanging independent tangent directions checks the metric self-adjointness of the converted Hessian
    EXPECT_NEAR(geometry.inner_product(point, hess, second), geometry.inner_product(point, other_hess, first), 2e-11);
    MatrixBatch<Point> batch(1, n, n);
    batch[0] = point;
    const auto viewed = geometry.euclidean_to_riemannian_hessian(batch[0], egrad, ehess, first);
    const Sym error(viewed - hess);
    // borrowed SPD batch points produce the same owning native tangent as the cached point owner
    EXPECT_LT(error.norm(), 2e-12);
}

// planar and higher-order geometries use the same metric connection for arbitrary ambient objectives
TEST(CheegerAmbientDerivatives, PlanarAndGeneralGeodesicHessians) {
    check_ambient_derivatives(manifold::CheegerLogEuclideanSPDGeometry<double, 2>(.7));
    check_ambient_derivatives(manifold::CheegerLogEuclideanSPDGeometry<double, 3>(.7));
    check_ambient_derivatives(manifold::CheegerLogEuclideanSPDGeometry<double, Dynamic>(4, .7));
}

// eigenvalue multiplicity and exact cache owner types do not require derivatives of individual eigenvectors
TEST(CheegerAmbientDerivatives, RepeatedEigenvaluesAndCachedOwners) {
    using Point = SPDMatrix<double, 3, Cache::Union<Cache::Log, Cache::Spectral, Cache::LogDividedDifferences>>;
    check_ambient_derivatives(manifold::CheegerLogEuclideanGeometry<Point>(.4), true);
    check_ambient_derivatives(manifold::CheegerLogEuclideanSPDGeometry<double, 2>(.4), true);
}

// suppressing the rotational deformation recovers the ordinary log-Euclidean derivative conversions
TEST(CheegerAmbientDerivatives, LargeRhoRecoversLogEuclideanMetric) {
    using Sym = SymmetricMatrix<double, 3>;
    using Point = SPDMatrix<double, 3>;
    const auto geometry = manifold::CheegerLogEuclideanGeometry<Point>::from_rho(1e10);
    const manifold::LogEuclideanGeometry<Point> euclidean;
    const Sym logs = symmetric_fixture<Sym>(3, .4);
    const Point point(matrix_exp(logs));
    const Sym gradient = symmetric_fixture<Sym>(3, .8), hessian = symmetric_fixture<Sym>(3, 1.5);
    const Sym direction = symmetric_fixture<Sym>(3, 2.1);
    const auto cheeger_gradient = geometry.euclidean_to_riemannian_gradient(point, gradient);
    const auto euclidean_gradient = euclidean.euclidean_to_riemannian_gradient(point, gradient);
    const Sym gradient_error(cheeger_gradient - euclidean_gradient);
    // the inverse-metric correction vanishes as rho grows with the fixed logarithmic spectrum
    EXPECT_LT(gradient_error.norm(), 2e-10);
    const auto cheeger_hessian = geometry.euclidean_to_riemannian_hessian(point, gradient, hessian, direction);
    const auto euclidean_hessian = euclidean.euclidean_to_riemannian_hessian(point, gradient, hessian, direction);
    const Sym hessian_error(cheeger_hessian - euclidean_hessian);
    // the C-LE connection correction vanishes in the same undeformed metric limit
    EXPECT_LT(hessian_error.norm(), 2e-10);
}

// public ambient conversion rejects incompatible dimensions and nonfinite derivatives before chart arithmetic
TEST(CheegerAmbientDerivatives, InvalidDerivativeInputs) {
    using Geometry = manifold::CheegerLogEuclideanSPDGeometry<double, Dynamic>;
    using Sym = Geometry::Tangent;
    const Geometry geometry(3);
    const auto point = Geometry::Point::Identity(3);
    const Sym valid(3, 3), wrong_shape(2, 2);
    // an ambient gradient with another matrix order cannot be converted by the bound geometry
    EXPECT_THROW(geometry.euclidean_to_riemannian_gradient(point, wrong_shape), std::invalid_argument);
    // an ambient Hessian action with another matrix order is rejected independently of the gradient shape
    EXPECT_THROW(geometry.euclidean_to_riemannian_hessian(point, valid, wrong_shape, valid), std::invalid_argument);
    Sym nonfinite(3, 3);
    nonfinite(0, 1) = std::numeric_limits<double>::quiet_NaN();
    // finite-coefficient validation rejects invalid ambient derivatives before logarithmic differential calls
    EXPECT_THROW(geometry.euclidean_to_riemannian_hessian(point, nonfinite, valid, valid), std::invalid_argument);
}

/// @brief compares isotropic C-LE conversion with LE when the scalar reciprocal of rho is unrepresentable
template <typename S, int N> void check_tiny_rho(double rho, double tolerance) {
    using Point = SPDMatrix<S, N>;
    using Sym = SymmetricMatrix<S, N>;
    const auto geometry = manifold::CheegerLogEuclideanGeometry<Point>::from_rho(rho);
    const manifold::LogEuclideanGeometry<Point> euclidean;
    const auto point = Point::Identity();
    const Sym gradient = symmetric_fixture<Sym>(N, .8), hessian = symmetric_fixture<Sym>(N, 1.5);
    const Sym direction = symmetric_fixture<Sym>(N, 2.1);
    const auto actual_gradient = geometry.euclidean_to_riemannian_gradient(point, gradient);
    const auto expected_gradient = euclidean.euclidean_to_riemannian_gradient(point, gradient);
    const Sym gradient_error(actual_gradient - expected_gradient);
    // scalar log points have zero commutators, so the gradient equals LE even for subnormal rho
    EXPECT_LT(gradient_error.norm(), tolerance);
    const auto actual_hessian = geometry.euclidean_to_riemannian_hessian(point, gradient, hessian, direction);
    const auto expected_hessian = euclidean.euclidean_to_riemannian_hessian(point, gradient, hessian, direction);
    const Sym hessian_error(actual_hessian - expected_hessian);
    // zero connection numerators divided by tiny rho remain zero rather than reciprocal-overflow NaNs
    EXPECT_LT(hessian_error.norm(), tolerance);
}

// isotropic points remain regular for every positive representable rho even when its reciprocal overflows
TEST(CheegerAmbientDerivatives, TinyRhoAtIsotropicPoints) {
    check_tiny_rho<double, 2>(1e-320, 2e-14);
    check_tiny_rho<double, 3>(1e-320, 2e-14);
    check_tiny_rho<float, 2>(1e-50, 2e-6);
    check_tiny_rho<float, 3>(1e-50, 2e-6);
}
}   // namespace
