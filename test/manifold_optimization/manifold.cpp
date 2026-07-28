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

namespace {

struct ToyPoint {
    double value = 0;
};
struct ToyTangent {
    double value = 0;
};

struct ToyEuclideanGeometry {
    using Point = ToyPoint;
    using Tangent = ToyTangent;

    std::size_t dimension() const { return 1; }
    double inner_product(const Point&, const Tangent& u, const Tangent& v) const { return u.value * v.value; }
    double norm(const Point& point, const Tangent& u) const { return std::sqrt(inner_product(point, u, u)); }
    Tangent project(const Point&, const Tangent& u) const { return u; }
    Tangent zero_tangent(const Point&) const { return {}; }
    Tangent linear_combination(const Point&, double alpha, const Tangent& u, double beta, const Tangent& v) const {
        return {alpha * u.value + beta * v.value};
    }
    Point retract(const Point& point, const Tangent& u, double step) const { return {point.value + step * u.value}; }
};

struct TransportedToyEuclideanGeometry : ToyEuclideanGeometry {
    Tangent transport(const Point&, const Point&, const Tangent& u) const { return u; }
};

struct ToyGeodesicGeometry : ToyEuclideanGeometry {
    Point exponential(const Point& point, const Tangent& tangent, double step) const {
        return {point.value + step * tangent.value};
    }
    Tangent logarithm(const Point& from, const Point& to) const { return {to.value - from.value}; }
    double distance(const Point& from, const Point& to) const { return std::abs(to.value - from.value); }
};

struct SignedDimensionGeometry : ToyEuclideanGeometry {
    int dimension() const { return 1; }
};

struct FloatingDimensionGeometry : ToyEuclideanGeometry {
    double dimension() const { return 1; }
};

struct MissingRetractionGeometry {
    using Point = ToyPoint;
    using Tangent = ToyTangent;
    std::size_t dimension() const { return 1; }
    double inner_product(const Point&, const Tangent& u, const Tangent& v) const { return u.value * v.value; }
    double norm(const Point&, const Tangent& u) const { return std::abs(u.value); }
    Tangent project(const Point&, const Tangent& u) const { return u; }
    Tangent zero_tangent(const Point&) const { return {}; }
    Tangent linear_combination(const Point&, double alpha, const Tangent& u, double beta, const Tangent& v) const {
        return {alpha * u.value + beta * v.value};
    }
};

static_assert(fdapde::manifold::FirstOrderGeometry<ToyEuclideanGeometry>);
static_assert(!fdapde::manifold::GeodesicGeometry<ToyEuclideanGeometry>);
static_assert(fdapde::manifold::GeodesicGeometry<ToyGeodesicGeometry>);
static_assert(!fdapde::manifold::VectorTransportGeometry<ToyEuclideanGeometry>);
static_assert(fdapde::manifold::VectorTransportGeometry<TransportedToyEuclideanGeometry>);
static_assert(!fdapde::manifold::FirstOrderGeometry<SignedDimensionGeometry>);
static_assert(!fdapde::manifold::FirstOrderGeometry<FloatingDimensionGeometry>);
static_assert(!fdapde::manifold::FirstOrderGeometry<MissingRetractionGeometry>);
static_assert(!fdapde::manifold::FirstOrderGeometry<int>);

}   // namespace

TEST(ManifoldContracts, FirstOrderOperations) {
    ToyEuclideanGeometry geometry;
    ToyPoint point {2};
    ToyTangent u {3};
    ToyTangent v {-1};

    EXPECT_EQ(geometry.dimension(), 1);
    EXPECT_DOUBLE_EQ(geometry.inner_product(point, u, v), -3);
    EXPECT_DOUBLE_EQ(geometry.norm(point, u), 3);
    EXPECT_DOUBLE_EQ(geometry.project(point, u).value, 3);
    EXPECT_DOUBLE_EQ(geometry.zero_tangent(point).value, 0);
    EXPECT_DOUBLE_EQ(geometry.linear_combination(point, 2, u, 0.5, v).value, 5.5);
    EXPECT_DOUBLE_EQ(geometry.retract(point, u, 0.5).value, 3.5);
}
