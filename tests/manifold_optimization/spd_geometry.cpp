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

// Independent diagonal and off-diagonal oracles distinguish the two metrics.
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
    EXPECT_NEAR(le.inner_product(point, u, u), 2 * divided_log * divided_log, tolerance);
    EXPECT_NEAR(airm.inner_product(point, u, u), 2.0 / 36, tolerance);
    EXPECT_GT(std::abs(le.inner_product(point, u, u) - airm.inner_product(point, u, u)), 1e-3);
    const double expected_distance = std::hypot(std::log(4.0), std::log(1.0 / 9.0));
    EXPECT_NEAR(le.distance(point, target), expected_distance, tolerance);
    EXPECT_NEAR(airm.distance(point, target), expected_distance, tolerance);
    for (const auto& tangent : {le.logarithm(point, target), airm.logarithm(point, target)}) {
        EXPECT_NEAR(tangent(0, 0), 4 * std::log(4.0), 10 * tolerance);
        EXPECT_NEAR(tangent(1, 1), 9 * std::log(1.0 / 9.0), 10 * tolerance);
        EXPECT_NEAR(tangent(1, 0), 0, tolerance);
    }
    u(0, 0) = 2;
    u(1, 0) = 0;
    u(1, 1) = -3;
    const auto retracted = airm.retract(point, u, 1);
    EXPECT_NEAR(retracted(0, 0), 4 + 2 + 0.5 * 4 / 4, tolerance);
    EXPECT_NEAR(retracted(1, 1), 9 - 3 + 0.5 * 9 / 9, tolerance);
    EXPECT_GT(std::abs(retracted(0, 0) - airm.exponential(point, u)(0, 0)), 0.05);
}

// The logarithm is the negative metric gradient of half the squared distance.
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
    EXPECT_NEAR(
      (plus * plus - minus * minus) / (4 * step), -geometry.inner_product(point, geometry.logarithm(point, target), u),
      2e-9);
}

static_assert(GeodesicGeometry<LogEuclideanSPDGeometry<double, 2>>);
static_assert(GeodesicGeometry<AffineInvariantSPDGeometry<double, Dynamic>>);
static_assert(VectorTransportGeometry<LogEuclideanSPDGeometry<float, Dynamic>>);
}   // namespace

TEST(SPDGeometry, IndependentSPD2OraclesAcrossScalarAndStorageExtents) {
    diagonal_oracles<double, 2>();
    diagonal_oracles<double, Dynamic>();
    diagonal_oracles<float, 2>();
    diagonal_oracles<float, Dynamic>();
}

TEST(SPDGeometry, LogarithmHasTheGeodesicGradientSign) {
    distance_gradient<LogEuclideanSPDGeometry<double, 2>>();
    distance_gradient<AffineInvariantSPDGeometry<double, 2>>();
}
