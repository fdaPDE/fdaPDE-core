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

#include <array>
#include <cmath>
#include <limits>

namespace {
using namespace fdapde;
using namespace fdapde::manifold;
using Geometry2 = LogCholeskySPDGeometry<double, 2>;

// fixed log-Cholesky exposes the same public transport contract as the other SPD geometries
static_assert(VectorTransportGeometry<Geometry2>);
// dynamic log-Cholesky retains the ambient-tangent geodesic contract
static_assert(GeodesicGeometry<LogCholeskySPDGeometry<double, Dynamic>>);

/// @brief constructs fixed or dynamic geometry with the supplied matching order
template <typename Scalar, int Order> auto geometry_for(int order) {
    if constexpr (Order == Dynamic)
        return LogCholeskySPDGeometry<Scalar, Order>(order);
    else
        return LogCholeskySPDGeometry<Scalar, Order>();
}
/// @brief forms an independent point oracle from deterministic lower factor entries
template <typename Geometry> typename Geometry::Point factor_point(const Geometry& geometry, double offset = 0) {
    using Scalar = typename Geometry::Scalar;
    Matrix<Scalar, Geometry::Point::Rows, Geometry::Point::Rows> factor;
    if constexpr (Geometry::Point::Rows == Dynamic) factor.resize(geometry.order(), geometry.order());
    factor.set_zero();
    for (int i = 0; i < geometry.order(); ++i)
        for (int j = 0; j <= i; ++j)
            factor(i, j) = static_cast<Scalar>(i == j ? 2 + i + offset : .2 * (i + 1) * (j + 1) + offset);
    const Matrix<Scalar, Geometry::Point::Rows, Geometry::Point::Rows> product(factor * factor.transpose());
    return typename Geometry::Point(product);
}
/// @brief constructs a finite symmetric direction without using a geometry differential
template <typename Geometry> typename Geometry::Tangent direction_for(const Geometry& geometry, double shift = 0) {
    auto result =
      fdapde::manifold::internals::make_symmetric<typename Geometry::Scalar, Geometry::Point::Rows>(geometry.order());
    for (int i = 0; i < geometry.order(); ++i)
        for (int j = 0; j <= i; ++j) result(i, j) = .12 * (i + 1) - .17 * j + shift;
    return result;
}
/// @brief evaluates the full Frobenius pairing without packed-storage shortcuts
template <typename First, typename Second> double inner(const First& first, const Second& second) {
    double result = 0;
    for (int i = 0; i < first.rows(); ++i)
        for (int j = 0; j < first.cols(); ++j) result += first(i, j) * second(i, j);
    return result;
}
/// @brief compares full matrix coefficients against an independent absolute-tolerance oracle
template <typename First, typename Second>
void expect_matrix(const First& first, const Second& second, double tolerance) {
    // equal row counts ensure the coefficient oracle compares the same shape
    ASSERT_EQ(first.rows(), second.rows());
    // equal column counts prevent a truncated matrix comparison
    ASSERT_EQ(first.cols(), second.cols());
    for (int i = 0; i < first.rows(); ++i)
        for (int j = 0; j < first.cols(); ++j) {
            // every full coefficient agrees with the supplied oracle, including mirrored entries
            EXPECT_NEAR(static_cast<double>(first(i, j)), static_cast<double>(second(i, j)), tolerance);
        }
}
/// @brief checks factor-derived chart roundtrips for one scalar and storage extent
template <typename Scalar, int Order> void chart_oracles(int order) {
    const auto geometry = geometry_for<Scalar, Order>(order);
    const auto point = factor_point(geometry);
    const auto coordinates = geometry.chart(point);
    const double tolerance = std::is_same_v<Scalar, float> ? 3e-5 : 2e-13;
    for (int i = 0; i < order; ++i) {
        // diagonal chart entries equal the logarithms of the independently supplied Cholesky pivots
        EXPECT_NEAR(coordinates(i, i), std::log(2.0 + i), tolerance);
        for (int j = 0; j < i; ++j) {
            // symmetric off-diagonal coordinates divide the independent factor by sqrt(2)
            EXPECT_NEAR(coordinates(i, j), .2 * (i + 1) * (j + 1) / std::sqrt(2.0), tolerance);
        }
    }
    // reconstructing the chart recovers the independent lower-factor product
    expect_matrix(geometry.from_chart(coordinates), point, tolerance);
    const auto tangent = direction_for(geometry);
    // the two exact differential maps compose to the ambient identity
    expect_matrix(geometry.inverse_chart_jvp(coordinates, geometry.chart_jvp(point, tangent)), tangent, tolerance);
}
/// @brief verifies analytic chart, second-map and adjoint actions against independent finite differences
template <int Order> void differential_oracles(int order) {
    const auto geometry = geometry_for<double, Order>(order);
    using Geometry = decltype(geometry);
    using Point = typename Geometry::Point;
    const auto point = factor_point(geometry);
    const auto target = factor_point(geometry, .35);
    const auto u = direction_for(geometry);
    const auto v = direction_for(geometry, -.27);
    const auto coordinates = geometry.chart(point);
    constexpr double h = 2e-5;
    const auto plus = geometry.linear_combination(point, 1, point, h, u);
    const auto minus = geometry.linear_combination(point, 1, point, -h, u);
    const auto finite_chart = geometry.linear_combination(
      point, 1 / (2 * h), geometry.chart(Point(plus)), -1 / (2 * h), geometry.chart(Point(minus)));
    // central ambient perturbations independently validate the Cholesky chart differential
    expect_matrix(geometry.chart_jvp(point, u), finite_chart, 2e-10);
    const auto chart_plus = geometry.linear_combination(point, 1, coordinates, h, u);
    const auto chart_minus = geometry.linear_combination(point, 1, coordinates, -h, u);
    const auto finite_inverse = geometry.linear_combination(
      point, 1 / (2 * h), geometry.from_chart(chart_plus), -1 / (2 * h), geometry.from_chart(chart_minus));
    // independent chart perturbations validate the inverse-map Jacobian
    expect_matrix(geometry.inverse_chart_jvp(coordinates, u), finite_inverse, 2e-8);
    const auto finite_second = geometry.linear_combination(
      point, 1 / (2 * h), geometry.inverse_chart_jvp(chart_plus, v), -1 / (2 * h),
      geometry.inverse_chart_jvp(chart_minus, v));
    // differentiating the first inverse action validates the analytic mixed second derivative
    expect_matrix(geometry.inverse_chart_second_jvp(coordinates, u, v), finite_second, 2e-9);
    const auto frame = geometry.chart_frame(point);
    // retaining the factor yields the same second chart differential as reconstructing it from coordinates
    expect_matrix(
      geometry.inverse_chart_second_differential(frame, u, v),
      geometry.inverse_chart_second_differential(coordinates, u, v), 3e-12);
    // the chart adjoint satisfies the full Frobenius duality independently of metric conversion
    EXPECT_NEAR(inner(geometry.chart_jvp(point, u), v), inner(u, geometry.chart_vjp(point, v)), 3e-13);
    // the inverse adjoint satisfies the corresponding Frobenius duality
    EXPECT_NEAR(
      inner(geometry.inverse_chart_jvp(coordinates, u), v),
      inner(u, geometry.inverse_chart_differential_adjoint(coordinates, v)), 3e-12);
    const auto riemannian = geometry.euclidean_to_riemannian_gradient(point, v);
    // the converted gradient realizes the same directional derivative in the log-Cholesky metric
    EXPECT_NEAR(geometry.inner_product(point, riemannian, u), inner(v, u), 3e-12);
    // the metric-to-Frobenius conversion inverts the gradient duality map
    expect_matrix(geometry.riemannian_to_euclidean_gradient(point, riemannian), v, 3e-12);
    const auto log_direction = geometry.logarithm_target_jvp(point, target, u);
    const auto target_plus = geometry.linear_combination(target, 1, target, h, u);
    const auto target_minus = geometry.linear_combination(target, 1, target, -h, u);
    const auto finite_log = geometry.linear_combination(
      point, 1 / (2 * h), geometry.logarithm(point, Point(target_plus)), -1 / (2 * h),
      geometry.logarithm(point, Point(target_minus)));
    // target perturbations validate the exact logarithm differential between ambient tangent spaces
    expect_matrix(log_direction, finite_log, 3e-10);
    // the logarithm target adjoint satisfies endpoint metric duality
    EXPECT_NEAR(
      geometry.inner_product(point, log_direction, v),
      geometry.inner_product(target, u, geometry.logarithm_target_vjp(point, target, v)), 3e-12);
    const double plus_distance = geometry.distance(geometry.exponential(point, u, h), target);
    const double minus_distance = geometry.distance(geometry.exponential(point, u, -h), target);
    // an independent distance difference recovers the negative logarithm as the metric gradient
    EXPECT_NEAR(
      (plus_distance * plus_distance - minus_distance * minus_distance) / (4 * h),
      -geometry.inner_product(point, geometry.logarithm(point, target), u), 2e-10);
    // global flatness gives the identity covariant Hessian of half the squared distance
    expect_matrix(geometry.half_squared_distance_hessian_vector(point, target, u), u, 0);
}
/// @brief computes an independent Lin SPD(2) distance directly from scalar Cholesky formulas
template <typename Point> double scalar_distance(const Point& first, const Point& second) {
    const auto chart = [](const Point& p) {
        const double diagonal = std::sqrt(static_cast<double>(p(0, 0)));
        const double lower = p(1, 0) / diagonal;
        return std::array<double, 3> {
          std::log(diagonal), lower, std::log(std::sqrt(static_cast<double>(p(1, 1)) - lower * lower))};
    };
    const auto a = chart(first);
    const auto b = chart(second);
    return std::hypot(std::hypot(a[0] - b[0], a[1] - b[1]), a[2] - b[2]);
}
}   // namespace

// independent factors exercise fixed and dynamic native points of both small matrix orders and scalar types
TEST(LogCholeskySPDGeometry, ChartRoundtripsAcrossScalarsAndStorageExtents) {
    // fixed double SPD(2) agrees with the independent lower-factor chart oracle
    chart_oracles<double, 2>(2);
    // fixed double SPD(3) agrees with the same scalar factor construction
    chart_oracles<double, 3>(3);
    // runtime SPD(2) preserves the fixed-order chart convention
    chart_oracles<double, Dynamic>(2);
    // runtime SPD(3) reconstructs the same independent factor product
    chart_oracles<double, Dynamic>(3);
    // fixed float SPD(2) preserves the metric normalization at float accuracy
    chart_oracles<float, 2>(2);
    // fixed float SPD(3) validates the native float chart and inverse action
    chart_oracles<float, 3>(3);
    // runtime float SPD(3) retains the same factor oracle without fixed workspaces
    chart_oracles<float, Dynamic>(3);
}

// Lin's scalar formulas expose the diagonal half factor and the strictly lower single-count normalization
TEST(LogCholeskySPDGeometry, IndependentSPD2MetricDistanceAndGeodesicOracles) {
    const Geometry2 geometry;
    Matrix<double, 2, 2> dense;
    dense(0, 0) = 4;
    dense(0, 1) = 2;
    dense(1, 0) = 2;
    dense(1, 1) = 10;
    const Geometry2::Point first(dense);
    dense(0, 0) = 16;
    dense(0, 1) = -8;
    dense(1, 0) = -8;
    dense(1, 1) = 5;
    const Geometry2::Point second(dense);
    const double expected = std::hypot(std::hypot(std::log(2.0), 3.0), std::log(1.0 / 3));
    // the metric distance matches Lin's raw lower-factor and log-diagonal Euclidean formula
    EXPECT_NEAR(geometry.distance(first, second), expected, 2e-13);
    // the independent scalar Cholesky calculation agrees with the documented distance formula
    EXPECT_NEAR(scalar_distance(first, second), expected, 2e-13);
    Geometry2::Tangent u;
    u(0, 0) = 0;
    u(1, 0) = 1;
    u(1, 1) = 0;
    const auto identity = Geometry2::Point::Identity();
    // the strictly lower Cholesky tangent contributes once rather than the ambient Frobenius factor two
    EXPECT_NEAR(geometry.inner_product(identity, u, u), 1, 2e-15);
    u(0, 0) = 2;
    u(1, 0) = 0;
    // differentiating log(sqrt(P_11)) gives one quarter of the squared diagonal ambient coefficient
    EXPECT_NEAR(geometry.inner_product(identity, u, u), 1, 2e-15);
    constexpr double t = .3;
    const Geometry2::Point interpolated(geometry.geodesic(first, second)(t));
    const double l11 = std::pow(2.0, 1 - t) * std::pow(4.0, t);
    const double l21 = 1 - 3 * t;
    const double l22 = std::pow(3.0, 1 - t);
    dense(0, 0) = l11 * l11;
    dense(0, 1) = l11 * l21;
    dense(1, 0) = l11 * l21;
    dense(1, 1) = l21 * l21 + l22 * l22;
    // the geodesic factor is linear below the diagonal and geometric on the positive diagonal
    expect_matrix(interpolated, dense, 2e-13);
    // the scalar SPD(2) determinant gives the expected affine log-determinant along the geodesic
    EXPECT_NEAR(
      std::log(interpolated(0, 0) * interpolated(1, 1) - interpolated(1, 0) * interpolated(1, 0)),
      (1 - t) * std::log(36.0) + t * std::log(16.0), 2e-13);
    // the exact exponential of the initial logarithm reconstructs the target coefficients
    expect_matrix(geometry.exponential(first, geometry.logarithm(first, second)), second, 2e-13);
    const auto transported = geometry.transport(first, second, u);
    // constant chart coordinates preserve the metric norm during parallel transport
    EXPECT_NEAR(geometry.norm(first, u), geometry.norm(second, transported), 2e-13);
    // transporting back along the inverse path recovers the initial ambient tangent
    expect_matrix(geometry.transport(second, first, transported), u, 2e-13);
}

// analytic differentials and adjoints are checked through perturbations of independent fixed and dynamic points
TEST(LogCholeskySPDGeometry, ExactFirstSecondAndAdjointDifferentials) {
    // fixed SPD(2) actions agree with independent finite differences and Frobenius duality
    differential_oracles<2>(2);
    // fixed SPD(3) exercises the third lower row in all triangular differential actions
    differential_oracles<3>(3);
    // dynamic SPD(3) uses runtime workspaces with the same independent derivative oracles
    differential_oracles<Dynamic>(3);
}

// the unique mean is the affine chart average and log-Cholesky retains its documented coordinate dependence
TEST(LogCholeskySPDGeometry, ClosedFormMeanAndCoordinateDependence) {
    const Geometry2 geometry;
    const std::array<Geometry2::Point, 2> points {factor_point(geometry), factor_point(geometry, .6)};
    const std::array<double, 2> weights {1, 3};
    const auto mean = weighted_karcher_mean(geometry, points, std::span<const double>(weights));
    const auto first_chart = geometry.chart(points[0]);
    const auto second_chart = geometry.chart(points[1]);
    const auto expected = geometry.linear_combination(points[0], .25, first_chart, .75, second_chart);
    // the unique closed-form point has the independently weighted affine chart coordinates
    expect_matrix(geometry.chart(mean.point), expected, 2e-13);
    // closed-form diagnostics report the explicit solver-free stop reason
    EXPECT_EQ(mean.stop_reason, BarycenterStopReason::closed_form);
    // flat Euclidean chart convexity establishes global uniqueness for positive weights
    EXPECT_EQ(mean.uniqueness, BarycenterUniqueness::globally_unique);
    // the returned represented mean is stationary for the normalized weighted chart objective
    EXPECT_LT(mean.stationarity_norm, 2e-13);
    Matrix<double, 2, 2> swap;
    swap(0, 0) = 0;
    swap(0, 1) = 1;
    swap(1, 0) = -1;
    swap(1, 1) = 0;
    const Geometry2::Point rotated_first(swap * points[0] * swap.transpose());
    const Geometry2::Point rotated_second(swap * points[1] * swap.transpose());
    // independent scalar Cholesky charts confirm the rotated distance value
    EXPECT_NEAR(
      geometry.distance(rotated_first, rotated_second), scalar_distance(rotated_first, rotated_second), 2e-13);
    // a common orthogonal change of coordinates generally changes the log-Cholesky distance
    EXPECT_GT(
      std::abs(geometry.distance(points[0], points[1]) - geometry.distance(rotated_first, rotated_second)), .01);
}

// malformed public shapes and unrepresentable chart diagonals are rejected before returning invalid points
TEST(LogCholeskySPDGeometry, RejectsInvalidOrdersCoordinatesAndWeights) {
    // runtime zero order is excluded by the geometry construction contract
    EXPECT_THROW((LogCholeskySPDGeometry<double, Dynamic>(0)), std::invalid_argument);
    const LogCholeskySPDGeometry<double, Dynamic> geometry(3);
    const SPDMatrix<double, 2, 2> wrong = SPDMatrix<double, 2, 2>::Identity();
    // a point of different order cannot enter a runtime chart
    EXPECT_THROW(geometry.chart(wrong), std::invalid_argument);
    const Geometry2 fixed;
    using DynamicGeometry = LogCholeskySPDGeometry<double, Dynamic>;
    const DynamicGeometry::Tangent short_chart(direction_for(fixed));
    // a dynamic curve rejects an inconsistent velocity order before deferred evaluation can index it
    EXPECT_THROW((DynamicGeometry::Curve(direction_for(geometry), short_chart)), std::invalid_argument);
    auto coordinates = direction_for(fixed);
    coordinates(0, 0) = std::numeric_limits<double>::infinity();
    // nonfinite chart inputs fail the public finite-coordinate check
    EXPECT_THROW(fixed.from_chart(coordinates), std::invalid_argument);
    coordinates(0, 0) = -1e4;
    // exponential underflow cannot create a zero Cholesky diagonal
    EXPECT_THROW(fixed.from_chart(coordinates), std::domain_error);
    const std::array<Geometry2::Point, 1> points {factor_point(fixed)};
    const std::array<double, 1> invalid {-1};
    // negative barycenter weights are rejected by the normalized positive-weight contract
    EXPECT_THROW(weighted_karcher_mean(fixed, points, std::span<const double>(invalid)), std::invalid_argument);
}

// cached LC frames reuse certified batch factors and agree exactly with independent plain-point preparation
TEST(LogCholeskySPDGeometry, CachedFramesPreserveMapsAndMetrics) {
    using CachedGeometry = LogCholeskySPDGeometry<double, 3, Usage::InterpolationNodes | Usage::LogExpDifferentials>;
    const CachedGeometry cached_geometry;
    const LogCholeskySPDGeometry<double, 3> plain_geometry;
    const auto point = factor_point(plain_geometry);
    const CachedGeometry::Point cached(point);
    const auto plain_frame = plain_geometry.chart_frame(point);
    const auto cached_frame = cached_geometry.chart_frame(cached);
    // a geometry requesting interpolation retains the native triangular factor
    static_assert(CachedGeometry::Point::CacheSlot::template Has<Cache::Cholesky>);
    // its matching flat coordinates share the same point-owned cache lifetime
    static_assert(CachedGeometry::Point::CacheSlot::template Has<Cache::LogCholesky>);
    // cached and plain preparation use the same extended-accumulation factor algorithm
    expect_matrix(cached_frame.factor, plain_frame.factor, 0);
    // the cached flat chart preserves each original scalar logarithm and normalization
    expect_matrix(cached_geometry.chart(cached), plain_frame.coordinates, 0);
    const auto direction = direction_for(plain_geometry);
    // borrowing a prepared frame leaves the analytic chart Jacobian unchanged
    expect_matrix(cached_geometry.chart_jvp(cached, direction), plain_geometry.chart_jvp(point, direction), 0);
    // both cached factors and coordinates preserve the complete metric-gradient pullback
    expect_matrix(
      cached_geometry.riemannian_to_euclidean_gradient(cached, direction),
      plain_geometry.riemannian_to_euclidean_gradient(point, direction), 0);
    const SPDMatrix<double, 3, 3, Cache::LogCholesky> chart_only(point);
    // the chart-only policy computes the coordinates directly from certified coefficients
    expect_matrix(cached_geometry.chart(chart_only), plain_frame.coordinates, 0);
}
