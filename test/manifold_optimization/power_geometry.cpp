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
#include <utility>

namespace {

struct PowerPoint {
    double value = 0;
};
struct PowerTangent {
    double value = 0;
};

struct PowerToyGeometry {
    using Point = PowerPoint;
    using Tangent = PowerTangent;

    std::size_t dimension() const { return 1; }
    double inner_product(const Point&, const Tangent& u, const Tangent& v) const { return u.value * v.value; }
    double norm(const Point&, const Tangent& tangent) const { return std::abs(tangent.value); }
    Tangent project(const Point&, const Tangent& tangent) const { return tangent; }
    Tangent zero_tangent(const Point&) const { return {}; }
    Tangent linear_combination(const Point&, double alpha, const Tangent& u, double beta, const Tangent& v) const {
        return {alpha * u.value + beta * v.value};
    }
    Point retract(const Point& point, const Tangent& tangent, double step) const {
        return {point.value + step * tangent.value};
    }
    Tangent transport(const Point&, const Point&, const Tangent& tangent) const { return tangent; }
};

struct TwoDimensionalToyGeometry : PowerToyGeometry {
    std::size_t dimension() const { return 2; }
};

using PowerToy = fdapde::manifold::PowerGeometry<PowerToyGeometry>;

template <typename Geometry>
concept PermitsRvalueComponentAccess = requires(Geometry&& geometry) { std::move(geometry).component_geometry(); };

static_assert(fdapde::manifold::VectorTransportGeometry<PowerToy>);
static_assert(!PermitsRvalueComponentAccess<PowerToy>);

}   // namespace

TEST(ManifoldPowerGeometry, AppliesOperationsComponentwise) {
    PowerToy geometry(3);
    PowerToy::Point point {{1}, {2}, {3}};
    PowerToy::Tangent u {{2}, {-1}, {4}};
    PowerToy::Tangent v {{3}, {5}, {-2}};

    EXPECT_EQ(geometry.factor_count(), 3);
    EXPECT_EQ(geometry.dimension(), 3);
    EXPECT_DOUBLE_EQ(geometry.inner_product(point, u, v), -7);
    EXPECT_DOUBLE_EQ(geometry.norm(point, u), std::sqrt(21));

    const auto projected = geometry.project(point, u);
    const auto zero = geometry.zero_tangent(point);
    EXPECT_EQ(projected.size(), 3);
    EXPECT_EQ(zero.size(), 3);
    EXPECT_DOUBLE_EQ(projected[2].value, 4);
    EXPECT_DOUBLE_EQ(zero[1].value, 0);

    const auto combined = geometry.linear_combination(point, 2, u, -1, v);
    EXPECT_DOUBLE_EQ(combined[0].value, 1);
    EXPECT_DOUBLE_EQ(combined[1].value, -7);
    EXPECT_DOUBLE_EQ(combined[2].value, 10);

    const auto next = geometry.retract(point, u, 0.5);
    EXPECT_DOUBLE_EQ(next[0].value, 2);
    EXPECT_DOUBLE_EQ(next[1].value, 1.5);
    EXPECT_DOUBLE_EQ(next[2].value, 5);
    EXPECT_EQ(geometry.transport(point, next, u).size(), 3);
}

TEST(ManifoldPowerGeometry, RejectsEmptyMismatchedAndOverflowingProducts) {
    EXPECT_THROW(PowerToy(0), std::invalid_argument);
    PowerToy geometry(2);
    PowerToy::Point point {{1}, {2}};
    PowerToy::Tangent one {{1}};
    PowerToy::Tangent two {{1}, {2}};

    EXPECT_THROW(geometry.inner_product(point, one, two), std::invalid_argument);
    EXPECT_THROW(geometry.project(point, one), std::invalid_argument);
    EXPECT_THROW(geometry.zero_tangent(PowerToy::Point {{1}}), std::invalid_argument);
    EXPECT_THROW(geometry.retract(PowerToy::Point {{1}}, two, 1), std::invalid_argument);

    const std::size_t overflowing_count = std::numeric_limits<std::size_t>::max() / 2 + 1;
    const fdapde::manifold::PowerGeometry<TwoDimensionalToyGeometry> overflowing(overflowing_count);
    EXPECT_THROW(static_cast<void>(overflowing.dimension()), std::overflow_error);
}

TEST(ManifoldPowerGeometry, NormIsScaleSafeWhenTheSquaredMetricOverflowsOrUnderflows) {
    PowerToy geometry(2);
    PowerToy::Point point {{0}, {0}};

    for (const double scale : {1e308, 1e-308}) {
        PowerToy::Tangent tangent {{scale}, {scale}};
        EXPECT_DOUBLE_EQ(geometry.norm(point, tangent), std::hypot(scale, scale));
    }

    PowerToy::Tangent overflowing_squared_metric {{3e200}, {4e200}};
    EXPECT_TRUE(std::isinf(geometry.inner_product(point, overflowing_squared_metric, overflowing_squared_metric)));
}
