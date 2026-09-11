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

// independent diagonal and off-diagonal oracles distinguish the two metrics
template <typename Scalar, int Order> void diagonal_oracles() {
    auto le = [] {
        if constexpr (Order == Dynamic)
            return LogEuclideanSPDGeometry<Scalar, Order>(2);
        else
            return LogEuclideanSPDGeometry<Scalar, Order>();
    }();
    auto airm = [] {
        if constexpr (Order == Dynamic)
            return AffineInvariantSPDGeometry<Scalar, Order>(2);
        else
            return AffineInvariantSPDGeometry<Scalar, Order>();
    }();
    using Point = typename decltype(le)::Point;
    using Tangent = typename decltype(le)::Tangent;
    Matrix<Scalar, 2, 2> dense;
    dense.set_zero();
    dense(0, 0) = 4;
    dense(1, 1) = 9;
    const Point point(dense);
    dense(0, 0) = 16;
    dense(1, 1) = 1;
    const Point target(dense);
    Tangent u;
    if constexpr (Order == Dynamic) u.resize(2, 2);
    u(0, 0) = 0;
    u(1, 0) = 1;
    u(1, 1) = 0;
    const double tolerance = std::is_same_v<Scalar, float> ? 2e-5 : 1e-11;
    const double divided_log = std::log(9.0 / 4.0) / 5;
    // the off-diagonal log-Euclidean metric uses twice the squared logarithmic divided difference
    EXPECT_NEAR(le.inner_product(point, u, u), 2 * divided_log * divided_log, tolerance);
    // the affine-invariant metric weights the two off-diagonal entries by the inverse diagonal product
    EXPECT_NEAR(airm.inner_product(point, u, u), 2.0 / 36, tolerance);
    // the two analytic metric values differ enough to detect accidentally sharing one metric formula
    EXPECT_GT(std::abs(le.inner_product(point, u, u) - airm.inner_product(point, u, u)), 1e-3);
    const double expected_distance = std::hypot(std::log(4.0), std::log(1.0 / 9.0));
    // commuting diagonal endpoints give the Euclidean norm of the log eigenvalue ratios
    EXPECT_NEAR(le.distance(point, target), expected_distance, tolerance);
    // for the same commuting endpoints the affine-invariant distance has the same scalar oracle
    EXPECT_NEAR(airm.distance(point, target), expected_distance, tolerance);
    for (const auto& tangent : {le.logarithm(point, target), airm.logarithm(point, target)}) {
        // the first diagonal logarithm coefficient is the source eigenvalue times its log ratio
        EXPECT_NEAR(tangent(0, 0), 4 * std::log(4.0), 10 * tolerance);
        // the second diagonal logarithm coefficient follows the same formula with the second eigenvalue
        EXPECT_NEAR(tangent(1, 1), 9 * std::log(1.0 / 9.0), 10 * tolerance);
        // commuting diagonal endpoints introduce no off-diagonal logarithm coefficient
        EXPECT_NEAR(tangent(1, 0), 0, tolerance);
    }
    u(0, 0) = 2;
    u(1, 0) = 0;
    u(1, 1) = -3;
    const auto retracted = airm.retract(point, u, 1);
    // the first retraction coefficient matches P plus U plus one half U squared divided by P
    EXPECT_NEAR(retracted(0, 0), 4 + 2 + 0.5 * 4 / 4, tolerance);
    // the second retraction coefficient matches the same scalar quadratic polynomial
    EXPECT_NEAR(retracted(1, 1), 9 - 3 + 0.5 * 9 / 9, tolerance);
    // the polynomial retraction differs from the exponential, detecting accidental substitution
    EXPECT_GT(std::abs(retracted(0, 0) - airm.exponential(point, u)(0, 0)), 0.05);
}

// the logarithm is the negative metric gradient of half the squared distance
template <typename Geometry> void distance_gradient() {
    Geometry geometry;
    Matrix<double, 2, 2> dense;
    dense(0, 0) = 4;
    dense(0, 1) = 0.8;
    dense(1, 0) = 0.8;
    dense(1, 1) = 2;
    const typename Geometry::Point point(dense);
    dense(0, 0) = 1.5;
    dense(0, 1) = -0.3;
    dense(1, 0) = -0.3;
    dense(1, 1) = 3;
    const typename Geometry::Point target(dense);
    typename Geometry::Tangent u;
    u(0, 0) = 0.2;
    u(1, 0) = 0.4;
    u(1, 1) = -0.1;
    constexpr double step = 1e-5;
    const double plus = geometry.distance(geometry.exponential(point, u, step), target);
    const double minus = geometry.distance(geometry.exponential(point, u, -step), target);
    // a central difference of half the squared distance equals pairing with the negative logarithm
    EXPECT_NEAR(
      (plus * plus - minus * minus) / (4 * step), -geometry.inner_product(point, geometry.logarithm(point, target), u),
      2e-9);
}

// the fixed log-Euclidean geometry satisfies the geodesic-map concept
static_assert(GeodesicGeometry<LogEuclideanSPDGeometry<double, 2>>);
// the dynamic affine-invariant geometry satisfies the same geodesic-map concept
static_assert(GeodesicGeometry<AffineInvariantSPDGeometry<double, Dynamic>>);
// the dynamic float log-Euclidean geometry satisfies the vector-transport concept
static_assert(VectorTransportGeometry<LogEuclideanSPDGeometry<float, Dynamic>>);
}   // namespace

// closed-form SPD(2) oracles distinguish the two metrics for float and double with fixed and dynamic storage
TEST(SPDGeometry, IndependentSPD2OraclesAcrossScalarAndStorageExtents) {
    // check closed-form metric, distance, logarithm and retraction values with fixed double storage
    diagonal_oracles<double, 2>();
    // check the same analytic values with dynamic double storage
    diagonal_oracles<double, Dynamic>();
    // check the analytic values with fixed float storage and the float tolerance
    diagonal_oracles<float, 2>();
    // check the same float oracles with dynamic storage
    diagonal_oracles<float, Dynamic>();
}

// finite differences on noncommuting SPD(2) inputs check the sign of the squared-distance gradient for both metrics
TEST(SPDGeometry, LogarithmHasTheGeodesicGradientSign) {
    // compare a geodesic central difference to the negative log-Euclidean logarithm pairing
    distance_gradient<LogEuclideanSPDGeometry<double, 2>>();
    // compare the same finite-difference oracle to the affine-invariant logarithm pairing
    distance_gradient<AffineInvariantSPDGeometry<double, 2>>();
}
