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
#include <utility>

namespace {

using namespace fdapde;
using namespace fdapde::manifold;

using point_cache = Cache::Union<Cache::Log, Cache::Spectral, Cache::LogDividedDifferences>;
using cached_point = SPDMatrix<double, 2, point_cache>;
using dynamic_point = SPDMatrix<double, Dynamic, point_cache>;
using scalar_point = SPDMatrix<double, 1, point_cache>;
using scalar_geometry = LogEuclideanGeometry<cached_point>;
using dynamic_geometry = LogEuclideanGeometry<dynamic_point>;
using batch_geometry = ProductGeometry<scalar_geometry>;
using dynamic_batch_geometry = ProductGeometry<dynamic_geometry>;
using scalar_batch_geometry = ProductGeometry<LogEuclideanGeometry<scalar_point>>;

// the point-typed scalar geometry preserves the selected cache owner exactly
static_assert(std::is_same_v<scalar_geometry::Point, cached_point>);
// the dynamic geometry preserves both dynamic extents and the selected cache owner
static_assert(std::is_same_v<dynamic_geometry::Point, dynamic_point>);
// point-typed geometries retain ordinary symmetric ambient tangents
static_assert(std::is_same_v<scalar_geometry::Tangent, SymmetricMatrix<double, 2>>);
// batching retains the requested cached SPD element type
static_assert(std::is_same_v<batch_geometry::Point, MatrixBatch<cached_point>>);
// batch directions use the native symmetric owner without SPD cache state
static_assert(std::is_same_v<batch_geometry::Tangent, MatrixBatch<SymmetricMatrix<double, 2>>>);
// dynamic batches retain the exact requested dynamic SPD owner
static_assert(std::is_same_v<dynamic_batch_geometry::Point, MatrixBatch<dynamic_point>>);
// fixed product geometries expose the full optimizer contract
static_assert(FirstOrderGeometry<batch_geometry>);
// dynamic product geometries expose the same geodesic contract
static_assert(GeodesicGeometry<dynamic_batch_geometry>);
// product geometries expose parallel transport with owning batch results
static_assert(VectorTransportGeometry<batch_geometry>);
// product geometries require their positive number of factors explicitly
static_assert(!std::is_default_constructible_v<batch_geometry>);
// fixed-order products pair an explicitly supplied element metric with their factor count
static_assert(std::is_constructible_v<batch_geometry, scalar_geometry, std::size_t>);
// dynamic products require a prepared element metric rather than accepting a count alone
static_assert(!std::is_constructible_v<dynamic_batch_geometry, std::size_t>);
// dynamic products accept the element metric that already retains its runtime matrix order
static_assert(std::is_constructible_v<dynamic_batch_geometry, dynamic_geometry, std::size_t>);
// scalar exponentials return the exact chosen SPD owner instead of a default cache type
static_assert(std::is_same_v<
              decltype(std::declval<const scalar_geometry&>().exponential(
                std::declval<const cached_point&>(), std::declval<const scalar_geometry::Tangent&>())),
              cached_point>);
// dynamic retractions retain the requested dynamic cached owner
static_assert(std::is_same_v<
              decltype(std::declval<const dynamic_geometry&>().retract(
                std::declval<const dynamic_point&>(), std::declval<const dynamic_geometry::Tangent&>(), 1.0)),
              dynamic_point>);
// product exponentials retain the exact cached batch owner
static_assert(std::is_same_v<
              decltype(std::declval<const batch_geometry&>().exponential(
                std::declval<const batch_geometry::Point&>(), std::declval<const batch_geometry::Tangent&>())),
              batch_geometry::Point>);

/// @brief compares complete symmetric matrices against an independent coefficient oracle
template <typename Actual, typename Expected>
void expect_matrix_near(const Actual& actual, const Expected& expected, double tolerance = 1.0e-12) {
    // matching row extents make the coefficient comparisons valid
    ASSERT_EQ(actual.rows(), expected.rows());
    // matching column extents make the coefficient comparisons valid
    ASSERT_EQ(actual.cols(), expected.cols());
    for (int i = 0; i < actual.rows(); ++i) {
        for (int j = 0; j < actual.cols(); ++j) {
            // the full matrix coefficient agrees with the source or analytic oracle
            EXPECT_NEAR(static_cast<double>(actual(i, j)), static_cast<double>(expected(i, j)), tolerance);
        }
    }
}

/// @brief compares every factor while retaining independent batch count validation
template <typename Actual, typename Expected>
void expect_batch_near(const Actual& actual, const Expected& expected, double tolerance = 1.0e-12) {
    // equal factor counts make corresponding element access valid
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t k = 0; k < actual.size(); ++k) {
        // each product factor agrees with its independent scalar-matrix oracle
        expect_matrix_near(actual[k], expected[k], tolerance);
    }
}

/// @brief supplies a coupled quadratic in two scalar logarithmic coordinates
struct CoupledLogQuadratic {
    using Point = scalar_batch_geometry::Point;
    using Tangent = scalar_batch_geometry::Tangent;

    /// @brief supplies the stateless workspace required by the optimization problem contract
    struct Workspace { };

    /// @brief evaluates the quadratic with Hessian entries two, one and three in the logarithmic chart
    double cost(const Point& point, Workspace&) const {
        const double first = std::log(point[0](0, 0));
        const double second = std::log(point[1](0, 0));
        return first * first + first * second + 1.5 * second * second - first + 2 * second;
    }

    /// @brief maps the chart gradient to ambient tangent coordinates by multiplying each SPD factor
    Tangent grad(const Point& point, Workspace&) const {
        const double first = std::log(point[0](0, 0));
        const double second = std::log(point[1](0, 0));
        Tangent result(2);
        result[0](0, 0) = point[0](0, 0) * (2 * first + second - 1);
        result[1](0, 0) = point[1](0, 0) * (first + 3 * second + 2);
        return result;
    }

    /// @brief applies the coupled chart Hessian after dividing ambient directions by their SPD factors
    Tangent hess(const Point& point, const Tangent& direction, Workspace&) const {
        const double first = direction[0](0, 0) / point[0](0, 0);
        const double second = direction[1](0, 0) / point[1](0, 0);
        Tangent result(2);
        result[0](0, 0) = point[0](0, 0) * (2 * first + second);
        result[1](0, 0) = point[1](0, 0) * (first + 3 * second);
        return result;
    }
};

// exact cache policies and dynamic matrix extents survive point-typed scalar exponentials and retractions
TEST(TypedLogEuclideanGeometry, PreservesCachedOwnersAndDynamicOrder) {
    const scalar_geometry fixed;
    const cached_point point(Matrix<double, 2, 2>({3, 1, 1, 2}));
    const scalar_geometry::Tangent tangent(Vector<double, 3> {0.2, -0.1, 0.3});
    const auto exponential = fixed.exponential(point, tangent, 0.4);
    const LogEuclideanSPDGeometry<double, 2> legacy;

    // the selected cached owner evaluates the same noncommuting exponential as the uncached geometry
    expect_matrix_near(exponential, legacy.exponential(point, tangent, 0.4));
    // retained logarithm data describe the returned point rather than the input cache
    expect_matrix_near(
      exponential.cache().template matrix<Cache::Log>(), matrix_log(legacy.exponential(point, tangent, 0.4)));

    const dynamic_geometry dynamic(2);
    const dynamic_point dynamic_input(point);
    const dynamic_geometry::Tangent dynamic_tangent(tangent);
    const auto retraction = dynamic.retract(dynamic_input, dynamic_tangent, 0.4);
    // the dynamic geometry preserves its explicit runtime order
    EXPECT_EQ(dynamic.order(), 2);
    // the dynamic cached retraction matches the independently evaluated fixed-order exponential
    expect_matrix_near(retraction, exponential);
}

// the product metric sums full Frobenius products and keeps independent storage for vector operations
TEST(TypedLogEuclideanGeometry, BatchMetricCountsOffDiagonalCoefficientsTwice) {
    const batch_geometry geometry(scalar_geometry(), 2);
    const batch_geometry::Point identity(2);
    batch_geometry::Tangent u(2);
    batch_geometry::Tangent v(2);
    u[0](0, 0) = 1;
    u[0](1, 0) = 2;
    u[0](1, 1) = -1;
    u[1](0, 0) = 0;
    u[1](1, 0) = 3;
    u[1](1, 1) = 2;
    v[0](0, 0) = 2;
    v[0](1, 0) = -1;
    v[0](1, 1) = 4;
    v[1](0, 0) = -2;
    v[1](1, 0) = 1;
    v[1](1, 1) = 1;

    // two symmetric two-by-two factors contribute three tangent dimensions each
    EXPECT_EQ(geometry.dimension(), std::size_t {6});
    // full Frobenius products contribute minus six and eight for the two identity factors
    EXPECT_NEAR(geometry.inner_product(identity, u, v), 2, 1.0e-12);
    // squared full Frobenius norms contribute ten and twenty-two for the two identity factors
    EXPECT_NEAR(geometry.norm(identity, u), std::sqrt(32.0), 1.0e-12);
    const auto zero = geometry.zero_tangent(identity);
    // the zero tangent has no displacement in any product factor
    EXPECT_DOUBLE_EQ(geometry.norm(identity, zero), 0);
    const auto combined = geometry.linear_combination(identity, 2, u, -0.5, v);
    for (std::size_t k = 0; k < u.size(); ++k) {
        for (int i = 0; i < u.rows(); ++i) {
            for (int j = 0; j <= i; ++j) {
                // each packed coefficient follows the same two-term ambient linear combination
                EXPECT_DOUBLE_EQ(combined[k](i, j), 2 * u[k](i, j) - 0.5 * v[k](i, j));
            }
        }
    }
    const auto projected = geometry.project(identity, u);
    // projection retains every symmetric ambient coefficient in the product tangent space
    expect_batch_near(projected, u);
    u[0](0, 0) = 9;
    // modifying the input after projection cannot alter its owning result
    EXPECT_DOUBLE_EQ(projected[0](0, 0), 1);
}

// scalar geometry oracles verify each nonlinear map and the product distance combines factor distances with hypot
TEST(TypedLogEuclideanGeometry, BatchMapsMatchScalarFactors) {
    const batch_geometry geometry(scalar_geometry(), 2);
    const scalar_geometry factor;
    batch_geometry::Point from(2);
    batch_geometry::Point to(2);
    from[0].assign(Matrix<double, 2, 2>({3, 1, 1, 2}));
    from[1].assign(Matrix<double, 2, 2>({2, -0.5, -0.5, 4}));
    to[0].assign(Matrix<double, 2, 2>({4, -0.25, -0.25, 2}));
    to[1].assign(Matrix<double, 2, 2>({3, 0.8, 0.8, 1.5}));
    const auto tangent = geometry.logarithm(from, to);
    const auto transported = geometry.transport(from, to, tangent);
    const auto converted = geometry.euclidean_to_riemannian_gradient(from, tangent);
    const auto retraction = geometry.retract(from, tangent, 0.35);
    for (std::size_t k = 0; k < from.size(); ++k) {
        const scalar_geometry::Tangent direction(tangent[k]);
        // each product logarithm equals the scalar initial ambient geodesic tangent
        expect_matrix_near(tangent[k], factor.logarithm(from[k], to[k]));
        // parallel transport applies the scalar logarithmic chart isometry to each factor
        expect_matrix_near(transported[k], factor.transport(from[k], to[k], direction));
        // gradient conversion applies the scalar inverse metric to each independent factor
        expect_matrix_near(converted[k], factor.euclidean_to_riemannian_gradient(from[k], direction));
        // finite product retractions apply the same scalar exponential step to each factor
        expect_matrix_near(retraction[k], factor.exponential(from[k], direction, 0.35));
    }
    // exponentiating the product logarithm recovers both noncommuting endpoint matrices
    expect_batch_near(geometry.exponential(from, tangent), to, 3.0e-12);
    // product distance is the Euclidean norm of the independently computed factor distances
    EXPECT_NEAR(
      geometry.distance(from, to), std::hypot(factor.distance(from[0], to[0]), factor.distance(from[1], to[1])),
      3.0e-12);
    // the initial product logarithm has norm equal to the full product geodesic distance
    EXPECT_NEAR(geometry.norm(from, tangent), geometry.distance(from, to), 3.0e-12);
    // product parallel transport preserves the sum metric along both endpoint geodesics
    EXPECT_NEAR(geometry.norm(to, transported), geometry.norm(from, tangent), 3.0e-12);
}

// factor norm accumulation avoids overflow and underflow from squaring independently representable directions
TEST(TypedLogEuclideanGeometry, BatchNormRemainsRepresentableAcrossScales) {
    const batch_geometry geometry(scalar_geometry(), 2);
    const batch_geometry::Point identity(2);
    for (double scale : {1.0e-300, 1.0e300}) {
        batch_geometry::Tangent direction(2);
        direction[0](0, 0) = 3 * scale;
        direction[1](1, 1) = 4 * scale;
        // hypot combines the factor norms into the analytic three-four-five displacement at either scale
        EXPECT_NEAR(geometry.norm(identity, direction) / (5 * scale), 1, 1.0e-14);
    }
}

// positive factor counts and dynamic orders are required and every operation rejects incompatible operands
TEST(TypedLogEuclideanGeometry, ValidatesBatchCountsAndDynamicShapes) {
    // an empty product has no optimization tangent space and cannot construct a fixed geometry
    EXPECT_THROW(batch_geometry(scalar_geometry(), 0), std::invalid_argument);
    // an empty dynamic product is rejected even when its element order is valid
    EXPECT_THROW(dynamic_batch_geometry(dynamic_geometry(3), 0), std::invalid_argument);
    // dynamic product elements require positive matrix order
    EXPECT_THROW(dynamic_batch_geometry(dynamic_geometry(0), 2), std::invalid_argument);
    // negative dynamic matrix order is rejected before allocation
    EXPECT_THROW(dynamic_batch_geometry(dynamic_geometry(-1), 2), std::invalid_argument);

    const dynamic_batch_geometry geometry(dynamic_geometry(3), 2);
    const dynamic_batch_geometry::Point point(2, 3, 3);
    const dynamic_batch_geometry::Point wrong_count(1, 3, 3);
    const dynamic_batch_geometry::Point wrong_order(2, 2, 2);
    const dynamic_batch_geometry::Tangent direction(2, 3, 3);
    const dynamic_batch_geometry::Tangent wrong_tangent_count(1, 3, 3);
    const dynamic_batch_geometry::Tangent wrong_tangent_order(2, 2, 2);
    // two symmetric three-by-three factors contribute six dimensions each
    EXPECT_EQ(geometry.dimension(), std::size_t {12});
    // dynamic product exponentials preserve the runtime element shape of both factors
    expect_batch_near(geometry.exponential(point, direction), point);
    // product distance rejects a target with a different number of factors
    EXPECT_THROW(geometry.distance(point, wrong_count), std::invalid_argument);
    // product distance rejects a target with a different matrix order
    EXPECT_THROW(geometry.distance(point, wrong_order), std::invalid_argument);
    // tangent metric evaluation rejects a direction with a different number of factors
    EXPECT_THROW(geometry.norm(point, wrong_tangent_count), std::invalid_argument);
    // tangent metric evaluation rejects a direction with a different matrix order
    EXPECT_THROW(geometry.norm(point, wrong_tangent_order), std::invalid_argument);
    // projection validates the source point count before returning any direction
    EXPECT_THROW(geometry.project(wrong_count, direction), std::invalid_argument);
    // finite-step product maps reject nonfinite public step coefficients
    EXPECT_THROW(geometry.retract(point, direction, std::numeric_limits<double>::infinity()), std::invalid_argument);
}

// the coupled oracle exposes cross-factor curvature and converges jointly to the analytic log-coordinate minimizer
TEST(TypedLogEuclideanGeometry, JointTrustRegionSolvesCoupledLogQuadratic) {
    const scalar_batch_geometry geometry(LogEuclideanGeometry<scalar_point>(), 2);
    scalar_batch_geometry::Point initial(2);
    initial[0].assign(Matrix<double, 1, 1>(std::exp(-1.5)));
    initial[1].assign(Matrix<double, 1, 1>(std::exp(1.2)));
    CoupledLogQuadratic problem;
    CoupledLogQuadratic::Workspace workspace;
    scalar_batch_geometry::Tangent direction(2);
    direction[0](0, 0) = initial[0](0, 0);
    const auto hessian = problem.hess(initial, direction, workspace);
    // a unit direction in the first logarithmic coordinate induces curvature two in that coordinate
    EXPECT_NEAR(hessian[0](0, 0) / initial[0](0, 0), 2, 1.0e-12);
    // the off-diagonal Hessian entry induces a nonzero action in the second factor
    EXPECT_NEAR(hessian[1](0, 0) / initial[1](0, 0), 1, 1.0e-12);

    const double epsilon = 1.0e-4;
    const auto forward = geometry.exponential(initial, direction, epsilon);
    const auto backward = geometry.exponential(initial, direction, -epsilon);
    const double forward_cost = problem.cost(forward, workspace);
    const double backward_cost = problem.cost(backward, workspace);
    const auto gradient = problem.grad(initial, workspace);
    // the ambient Riemannian gradient pairs to the centered directional derivative along the geodesic
    EXPECT_NEAR(
      geometry.inner_product(initial, gradient, direction), (forward_cost - backward_cost) / (2 * epsilon), 1.0e-8);
    // the ambient Riemannian Hessian pairs to the centered second derivative along the same geodesic
    EXPECT_NEAR(
      geometry.inner_product(initial, direction, hessian),
      (forward_cost - 2 * problem.cost(initial, workspace) + backward_cost) / (epsilon * epsilon), 1.0e-6);

    TrustRegionOptions options;
    options.initial_radius = 0.2;
    options.gradient_tolerance = 1.0e-10;
    options.subproblem.residual_tolerance = 1.0e-12;
    const auto result = RiemannianTrustRegion(options).optimize(problem, geometry, initial);
    // one product trust region reaches a stationary point of the coupled logarithmic quadratic
    ASSERT_TRUE(result.converged());
    // solving the two-by-two chart normal equations gives the first logarithmic coordinate one
    EXPECT_NEAR(std::log(result.point[0](0, 0)), 1, 1.0e-9);
    // the same coupled normal equations give the second logarithmic coordinate minus one
    EXPECT_NEAR(std::log(result.point[1](0, 0)), -1, 1.0e-9);
    // the independent analytic minimizer has objective value minus three halves
    EXPECT_NEAR(result.cost, -1.5, 1.0e-12);
    // joint convergence uses the norm of the complete product gradient
    EXPECT_LE(result.gradient_norm, options.gradient_tolerance);
}

}   // namespace
