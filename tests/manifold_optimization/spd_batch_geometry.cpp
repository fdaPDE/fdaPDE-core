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
#include <type_traits>
#include <utility>

namespace {

using namespace fdapde;
using namespace fdapde::manifold;

using log_geometry = LogEuclideanSPDGeometry<double, 2>;
using log_distance_geometry = LogEuclideanSPDGeometry<double, 2, Usage::Distance>;
using log_tangent_geometry = LogEuclideanSPDGeometry<double, 2, Usage::TangentMetric>;
using log_interpolation_geometry = LogEuclideanSPDGeometry<double, 2, Usage::InterpolationNodes>;
using log_differential_geometry = LogEuclideanSPDGeometry<double, 2, Usage::LogExpDifferentials>;
using log_base_point_geometry = LogEuclideanSPDGeometry<double, 2, Usage::BasePointMaps>;
using log_distance_tangent_geometry = LogEuclideanSPDGeometry<double, 2, Usage::Distance | Usage::TangentMetric>;
using airm_distance_geometry = AffineInvariantSPDGeometry<double, 2, Usage::Distance>;
using airm_tangent_geometry = AffineInvariantSPDGeometry<double, 2, Usage::TangentMetric>;
using airm_interpolation_geometry = AffineInvariantSPDGeometry<double, 2, Usage::InterpolationNodes>;
using airm_differential_geometry = AffineInvariantSPDGeometry<double, 2, Usage::LogExpDifferentials>;
using airm_base_geometry = AffineInvariantSPDGeometry<double, 2, Usage::BasePointMaps>;
using airm_distance_differential_geometry =
  AffineInvariantSPDGeometry<double, 2, Usage::Distance | Usage::LogExpDifferentials>;
using point_batch = MatrixBatch<log_geometry::Point>;
using weights_type = Matrix<double, 2, 1>;
using complete_cache =
  Cache::Union<Cache::Spectral, Cache::Log, Cache::Sqrt, Cache::InverseSqrt, Cache::LogDividedDifferences>;

template <typename Geometry, typename Points, typename Weights>
concept permits_weighted_mean = requires(Geometry&& geometry, Points&& points, Weights&& weights) {
    std::forward<Geometry>(geometry).weighted_mean(std::forward<Points>(points), std::forward<Weights>(weights));
};

// default geometry points request no retained spectral state
static_assert(log_geometry::CachePolicy::Flags == Cache::None::Flags);
// distance reuse retains only the logarithm for the log-Euclidean metric
static_assert(log_distance_geometry::CachePolicy::Flags == Cache::Log::Flags);
// tangent metric reuse retains one coherent spectral basis and logarithmic divided differences
static_assert(
  log_tangent_geometry::CachePolicy::Flags == (Cache::Spectral::Flags | Cache::LogDividedDifferences::Flags));
// interpolation-node reuse retains its logarithmic chart without unrelated spectral data
static_assert(log_interpolation_geometry::CachePolicy::Flags == Cache::Log::Flags);
// differential reuse retains the spectral basis and logarithmic divided differences needed by Frechet actions
static_assert(
  log_differential_geometry::CachePolicy::Flags == (Cache::Spectral::Flags | Cache::LogDividedDifferences::Flags));
// base-point maps combine the chart and Frechet cache requirements
static_assert(
  log_base_point_geometry::CachePolicy::Flags ==
  (Cache::Log::Flags | Cache::Spectral::Flags | Cache::LogDividedDifferences::Flags));
// independent log-Euclidean distance and tangent uses form the union of their retained quantities
static_assert(
  log_distance_tangent_geometry::CachePolicy::Flags ==
  (Cache::Log::Flags | Cache::Spectral::Flags | Cache::LogDividedDifferences::Flags));
// affine-invariant default geometry points retain no local factors
static_assert(AffineInvariantSPDGeometry<double, 2>::CachePolicy::Flags == Cache::None::Flags);
// affine-invariant distance reuse retains the inverse square-root factor
static_assert(airm_distance_geometry::CachePolicy::Flags == Cache::InverseSqrt::Flags);
// affine-invariant tangent metrics use the same inverse square-root factor as distance
static_assert(airm_tangent_geometry::CachePolicy::Flags == Cache::InverseSqrt::Flags);
// interpolation nodes need no affine-invariant base-point cache
static_assert(airm_interpolation_geometry::CachePolicy::Flags == Cache::None::Flags);
// affine-invariant differentials retain the shared spectral divided-difference representation
static_assert(
  airm_differential_geometry::CachePolicy::Flags == (Cache::Spectral::Flags | Cache::LogDividedDifferences::Flags));
// affine-invariant base-point maps retain both square-root factors
static_assert(airm_base_geometry::CachePolicy::Flags == (Cache::Sqrt::Flags | Cache::InverseSqrt::Flags));
// independent affine-invariant distance and differential uses retain the complete required union
static_assert(
  airm_distance_differential_geometry::CachePolicy::Flags ==
  (Cache::InverseSqrt::Flags | Cache::Spectral::Flags | Cache::LogDividedDifferences::Flags));
// a persistent geometry, batch and weights form a valid deferred operation
static_assert(permits_weighted_mean<log_geometry&, point_batch&, weights_type&>);
// an expiring owning batch cannot be borrowed by a deferred operation
static_assert(!permits_weighted_mean<log_geometry&, point_batch, weights_type&>);
// an expiring owning weight vector cannot be borrowed by a deferred operation
static_assert(!permits_weighted_mean<log_geometry&, point_batch&, weights_type>);
// an expiring geometry cannot be borrowed by a deferred operation
static_assert(!permits_weighted_mean<log_geometry, point_batch&, weights_type&>);

Matrix<double, 2, 2> diagonal(double first, double second) { return Matrix<double, 2, 2>({first, 0, 0, second}); }

template <typename Actual, typename Expected>
void expect_matrix_near(const Actual& actual, const Expected& expected, double tolerance = 1.0e-12) {
    // coefficient comparison requires equal row extents before indexing both operands
    ASSERT_EQ(actual.rows(), expected.rows());
    // coefficient comparison requires equal column extents before indexing both operands
    ASSERT_EQ(actual.cols(), expected.cols());
    for (int i = 0; i < actual.rows(); ++i) {
        for (int j = 0; j < actual.cols(); ++j) {
            // each materialized coefficient matches the independent analytic or source value
            EXPECT_NEAR(actual(i, j), expected(i, j), tolerance);
        }
    }
}

/// @brief counts the weight reads performed by one full geometry-expression materialization
struct CountingWeights {
    using Scalar = double;
    static constexpr int Rows = 2;
    static constexpr int Cols = 1;
    std::array<double, 2> values;
    mutable int reads = 0;

    /// @brief weights form a two-by-one vector compatible with a two-point batch
    int rows() const { return Rows; }
    /// @brief weights form a single-column vector compatible with a two-point batch
    int cols() const { return Cols; }
    /// @brief weights expose exactly one coefficient per batch point
    int size() const { return Rows; }
    /// @brief each indexed retrieval records one reduction operand read
    double operator[](int index) const {
        ++reads;
        return values[static_cast<std::size_t>(index)];
    }
};

Matrix<double, 2, 2> noncommuting_left() { return Matrix<double, 2, 2>({3, 1, 1, 2}); }

Matrix<double, 2, 2> noncommuting_right() { return Matrix<double, 2, 2>({2, -0.5, -0.5, 4}); }

SymmetricMatrix<double, 2> off_diagonal_tangent(double diagonal, double off_diagonal) {
    SymmetricMatrix<double, 2> tangent;
    tangent(0, 0) = diagonal;
    tangent(1, 0) = off_diagonal;
    tangent(1, 1) = -0.5 * diagonal;
    return tangent;
}

// verifies that the native exp and log Frechet actions invert each other at one SPD point
template <typename Point>
void expect_exp_log_frechet_inverse(const Point& point, const SymmetricMatrix<double, 2>& tangent, double tolerance) {
    const auto logarithm = matrix_log(point);
    const auto logarithmic_differential = matrix_log_frechet(point, tangent);
    const auto restored = matrix_exp_frechet(logarithm, logarithmic_differential);
    // differentiating exp after log recovers the original off-diagonal ambient tangent
    expect_matrix_near(restored, tangent, tolerance);
}

// compares every first-order geometry operation on equivalent cached and uncached SPD operands
template <typename Geometry, typename CachedFrom, typename CachedTo>
void expect_cached_geometry_operations(
  const Geometry& geometry, const CachedFrom& cached_from, const CachedTo& cached_to,
  const SPDMatrix<double, 2>& uncached_from, const SPDMatrix<double, 2>& uncached_to,
  const typename Geometry::Tangent& first_tangent, const typename Geometry::Tangent& second_tangent, double tolerance) {
    // cached distance agrees with an independently decomposed uncached point pair
    EXPECT_NEAR(geometry.distance(cached_from, cached_to), geometry.distance(uncached_from, uncached_to), tolerance);
    // cached metric products agree for noncommuting points and off-diagonal tangents
    EXPECT_NEAR(
      geometry.inner_product(cached_from, first_tangent, second_tangent),
      geometry.inner_product(uncached_from, first_tangent, second_tangent), tolerance);
    // cached metric norms agree for an off-diagonal tangent
    EXPECT_NEAR(geometry.norm(cached_from, first_tangent), geometry.norm(uncached_from, first_tangent), tolerance);
    // cached logarithms preserve the initial tangent of the same endpoint geodesic
    expect_matrix_near(
      geometry.logarithm(cached_from, cached_to), geometry.logarithm(uncached_from, uncached_to), tolerance);
    // cached transport preserves the native result between the same noncommuting endpoints
    expect_matrix_near(
      geometry.transport(cached_from, cached_to, first_tangent),
      geometry.transport(uncached_from, uncached_to, first_tangent), tolerance);
    // cached gradient conversion preserves its metric dual coordinates
    expect_matrix_near(
      geometry.euclidean_to_riemannian_gradient(cached_from, second_tangent),
      geometry.euclidean_to_riemannian_gradient(uncached_from, second_tangent), tolerance);
    // cached retraction preserves the geometry-specific finite-step point result
    expect_matrix_near(
      geometry.retract(cached_from, first_tangent, 0.35), geometry.retract(uncached_from, first_tangent, 0.35),
      tolerance);
    // cached exponential preserves the exact finite-step point result
    expect_matrix_near(
      geometry.exponential(cached_from, first_tangent, 0.35), geometry.exponential(uncached_from, first_tangent, 0.35),
      tolerance);
}

// finite negative weights retain a nonunit sum and implement the literal noncommuting log-Euclidean combination
TEST(SPDBatchGeometry, WeightedMeanAcceptsNegativeAndUnnormalizedWeights) {
    const log_geometry geometry;
    point_batch points(2);
    const SPDMatrix<double, 2> left(noncommuting_left());
    const SPDMatrix<double, 2> right(noncommuting_right());
    points[0].assign(left);
    points[1].assign(right);
    weights_type weights({1.5, -0.25});
    const auto expression = geometry.weighted_mean(points, weights);
    const log_geometry::Point result(expression);
    const auto left_chart = matrix_log(left);
    const auto right_chart = matrix_log(right);
    SymmetricMatrix<double, 2> expected_chart;
    for (int i = 0; i < expected_chart.rows(); ++i)
        for (int j = 0; j <= i; ++j) expected_chart(i, j) = 1.5 * left_chart(i, j) - 0.25 * right_chart(i, j);
    const auto expected = matrix_exp(expected_chart);

    // the supplied weights have a deliberately nonunit sum before reduction
    EXPECT_NEAR(weights[0] + weights[1], 1.25, 1.0e-12);
    // materialization equals the independent exp of the explicitly weighted noncommuting log charts
    expect_matrix_near(result, expected, 2.0e-12);
}

// empty and all-zero reductions preserve the explicit matrix order and return the exponential identity
TEST(SPDBatchGeometry, WeightedMeanHandlesZeroWeightsAndEmptyShapedBatches) {
    const log_geometry geometry;
    point_batch points(2);
    points[0].assign(diagonal(2, 2));
    points[1].assign(diagonal(8, 8));
    weights_type zeros({0, 0});
    const log_geometry::Point zero_mean(geometry.weighted_mean(points, zeros));
    point_batch empty(0);
    Matrix<double, Dynamic, 1> empty_weights(0);
    const log_geometry::Point empty_mean(geometry.weighted_mean(empty, empty_weights));
    using dynamic_geometry = LogEuclideanSPDGeometry<double, Dynamic>;
    dynamic_geometry dynamic(2);
    MatrixBatch<dynamic_geometry::Point> dynamic_empty(0, 2, 2);
    Matrix<double, Dynamic, 1> dynamic_empty_weights(0);
    const dynamic_geometry::Point dynamic_empty_mean(dynamic.weighted_mean(dynamic_empty, dynamic_empty_weights));

    // all-zero weights reduce to the zero logarithmic chart and therefore identity
    expect_matrix_near(zero_mean, diagonal(1, 1));
    // an empty fixed-shape batch retains the geometry order through materialization
    EXPECT_EQ(empty_mean.rows(), geometry.order());
    // an empty fixed-shape reduction also exponentiates the zero chart to identity
    expect_matrix_near(empty_mean, diagonal(1, 1));
    // an empty dynamic batch preserves its explicit runtime matrix order
    EXPECT_EQ(dynamic_empty_mean.rows(), dynamic.order());
    // an empty dynamic reduction also materializes the identity at that retained shape
    expect_matrix_near(dynamic_empty_mean, diagonal(1, 1));
}

// deferred expressions observe later source updates and each conversion evaluates the complete reduction once
TEST(SPDBatchGeometry, DeferredMeanUsesCurrentSourcesAndEvaluatesGlobally) {
    const log_geometry geometry;
    point_batch points(2);
    points[0].assign(diagonal(2, 2));
    points[1].assign(diagonal(8, 8));
    CountingWeights weights {
      {1, 0}
    };
    const auto expression = geometry.weighted_mean(points, weights);

    points[0].assign(diagonal(4, 4));
    weights.values = {0, 1};
    weights.reads = 0;
    const log_geometry::Point point_result(expression);
    // point materialization reads each current weight once for the full reduction
    EXPECT_EQ(weights.reads, 2);
    // the point conversion observes both post-expression source updates
    expect_matrix_near(point_result, diagonal(8, 8));

    weights.reads = 0;
    const Matrix<double, 2, 2> dense_result(expression);
    // dense conversion evaluates the geometric expression once rather than per coefficient
    EXPECT_EQ(weights.reads, 2);
    // dense conversion receives the same globally materialized SPD value
    expect_matrix_near(dense_result, diagonal(8, 8));

    const MatrixExpr<std::remove_cvref_t<decltype(expression)>>& expression_base = expression;
    weights.reads = 0;
    const log_geometry::Point base_point_result(expression_base);
    // point conversion through the MatrixExpr base still evaluates the complete expression once
    EXPECT_EQ(weights.reads, 2);
    // base-class conversion retains the destination point policy and current value
    expect_matrix_near(base_point_result, diagonal(8, 8));

    log_geometry::Point assigned = log_geometry::Point::Identity();
    weights.reads = 0;
    assigned.assign(expression);
    // checked SPD assignment performs exactly one global expression evaluation
    EXPECT_EQ(weights.reads, 2);
    // checked assignment commits the complete result after evaluation
    expect_matrix_near(assigned, diagonal(8, 8));

    auto view = assigned.view();
    weights.reads = 0;
    view.assign(expression_base);
    // view assignment through the MatrixExpr base also performs one global evaluation
    EXPECT_EQ(weights.reads, 2);
    // view assignment retains the evaluated value in its existing owner binding
    expect_matrix_near(view, diagonal(8, 8));
}

// selection nodes own their indices while borrowed points and destination policy remain independently controlled
TEST(SPDBatchGeometry, TemporarySelectionsAndDestinationPoliciesMaterializeSafely) {
    const log_distance_geometry geometry;
    MatrixBatch<log_distance_geometry::Point> points(2);
    points[0].assign(diagonal(2, 2));
    points[1].assign(diagonal(8, 8));
    const std::array<int, 2> reverse {1, 0};
    weights_type weights({1, 0});
    const auto expression = geometry.weighted_mean(points.select(reverse).select(reverse), weights);
    const SPDMatrix<double, 2, complete_cache> destination(expression);

    // nested temporary selections retain both copied index lists after the call returns
    expect_matrix_near(destination, diagonal(2, 2));
    // destination materialization selects its own requested complete cache policy
    static_assert(decltype(destination)::CachePolicy::Flags == complete_cache::Flags);
    // the destination cache contains spectral factors requested independently of the geometry source policy
    EXPECT_NEAR(destination.cache().eigenvalues()[0] * destination.cache().eigenvalues()[1], 4, 1.0e-12);
}

// invalid deferred inputs fail before assignment can replace an existing checked point
TEST(SPDBatchGeometry, WeightedMeanRejectsInvalidInputsWithStrongGuarantee) {
    const log_geometry geometry;
    point_batch points(2);
    points[0].assign(diagonal(2, 2));
    points[1].assign(diagonal(8, 8));
    Matrix<double, 1, 1> wrong_weights;
    wrong_weights(0, 0) = 1;
    const auto wrong_shape = geometry.weighted_mean(points, wrong_weights);
    log_geometry::Point destination(diagonal(4, 4));
    const auto snapshot = destination;
    weights_type underflowing_weights({-2000, 0});
    const auto underflow = geometry.weighted_mean(points, underflowing_weights);
    weights_type nonfinite_weights({std::numeric_limits<double>::infinity(), 0});
    const auto nonfinite = geometry.weighted_mean(points, nonfinite_weights);
    weights_type nan_weights({std::numeric_limits<double>::quiet_NaN(), 0});
    const auto nan = geometry.weighted_mean(points, nan_weights);

    // a missing weight is rejected before the logarithmic reduction begins
    EXPECT_THROW(destination.assign(wrong_shape), std::invalid_argument);
    // a failed shape check leaves the destination SPD value unchanged
    expect_matrix_near(destination, snapshot);
    // an underflowing exponential cannot publish a singular SPD result
    EXPECT_THROW({ const log_geometry::Point rejected {underflow}; }, std::domain_error);
    // a nonfinite public weight is rejected before it can contaminate the logarithmic chart
    EXPECT_THROW({ const log_geometry::Point rejected {nonfinite}; }, std::invalid_argument);
    // a nan public weight follows the same finite-input contract before chart materialization
    EXPECT_THROW({ const log_geometry::Point rejected {nan}; }, std::invalid_argument);
}

// dynamic geometries and dynamic cached batches retain runtime order through weighted materialization
TEST(SPDBatchGeometry, DynamicGeometryAndBatchRetainUniformRuntimeOrder) {
    using dynamic_geometry = LogEuclideanSPDGeometry<double, Dynamic, Usage::Distance>;
    dynamic_geometry geometry(2);
    MatrixBatch<dynamic_geometry::Point> points(2, 2, 2);
    points[0].assign(diagonal(2, 2));
    points[1].assign(diagonal(8, 8));
    weights_type weights({1, 0});
    const dynamic_geometry::Point result(geometry.weighted_mean(points, weights));

    // the dynamic geometry preserves its explicit runtime order
    EXPECT_EQ(result.rows(), 2);
    // the dynamic batch supplies uniformly shaped points to the reduction
    EXPECT_EQ(points.rows(), 2);
    // dynamic materialization returns the selected first point
    expect_matrix_near(result, diagonal(2, 2));
}

// affine-invariant retraction keeps its second-order polynomial distinct from the exact exponential
TEST(SPDBatchGeometry, AffineInvariantRetractionRemainsSecondOrderAndDistinctFromExponential) {
    const AffineInvariantSPDGeometry<double, 2> geometry;
    const auto point = AffineInvariantSPDGeometry<double, 2>::Point::Identity();
    AffineInvariantSPDGeometry<double, 2>::Tangent tangent;
    tangent(0, 0) = 1;
    tangent(1, 0) = 0;
    tangent(1, 1) = 1;
    const auto retraction = geometry.retract(point, tangent, 0.5);
    const auto exponential = geometry.exponential(point, tangent, 0.5);

    // the second-order polynomial evaluates to one plus one half plus one eighth
    EXPECT_NEAR(retraction(0, 0), 1.625, 1.0e-12);
    // the exact geodesic exponential evaluates to exp(one half)
    EXPECT_NEAR(exponential(0, 0), std::exp(0.5), 1.0e-12);
    // retaining the distinct formula prevents retract from silently becoming exponential
    EXPECT_GT(std::abs(retraction(0, 0) - exponential(0, 0)), 1.0e-3);
}

// log-Euclidean cache policies preserve every geometry operation on noncommuting SPD inputs
TEST(SPDBatchGeometry, LogEuclideanCachePoliciesMatchUncachedOperationsAndFrechetInverses) {
    using cached_geometry = LogEuclideanSPDGeometry<double, 2, Usage::Distance | Usage::TangentMetric>;
    using spectral_point = SPDMatrix<double, 2, Cache::Spectral>;
    using logarithm_point = SPDMatrix<double, 2, Cache::Log>;
    const cached_geometry geometry;
    const cached_geometry::Point cached_from(noncommuting_left());
    const cached_geometry::Point cached_to(noncommuting_right());
    const SPDMatrix<double, 2> uncached_from(noncommuting_left());
    const SPDMatrix<double, 2> uncached_to(noncommuting_right());
    const spectral_point spectral_from(noncommuting_left());
    const logarithm_point logarithm_from(noncommuting_left());
    const logarithm_point logarithm_to(noncommuting_right());
    const auto spectral_from_view = spectral_from.view();
    const auto logarithm_to_view = logarithm_to.view();
    const auto first_tangent = off_diagonal_tangent(0.25, -0.4);
    const auto second_tangent = off_diagonal_tangent(-0.35, 0.3);

    // log-Euclidean exponential returns the geometry's requested point cache policy
    static_assert(std::same_as<decltype(geometry.exponential(cached_from, first_tangent)), cached_geometry::Point>);
    // log-Euclidean retraction returns the same geometry point policy as exponential
    static_assert(std::same_as<decltype(geometry.retract(cached_from, first_tangent, 0.35)), cached_geometry::Point>);
    // log-Euclidean exponential canonicalizes a spectral input view to the geometry point policy
    static_assert(
      std::same_as<decltype(geometry.exponential(spectral_from_view, first_tangent)), cached_geometry::Point>);
    // log-Euclidean retraction canonicalizes a logarithm-only input view to the geometry point policy
    static_assert(
      std::same_as<decltype(geometry.retract(logarithm_to_view, first_tangent, 0.35)), cached_geometry::Point>);
    // cache and fallback paths agree for every geometry operation on the same values
    expect_cached_geometry_operations(
      geometry, cached_from, cached_to, uncached_from, uncached_to, first_tangent, second_tangent, 2.0e-10);
    // distinct spectral and logarithm cache views use the same generic geometry operations as uncached owners
    expect_cached_geometry_operations(
      geometry, spectral_from_view, logarithm_to_view, uncached_from, uncached_to, first_tangent, second_tangent,
      2.0e-10);
    // a logarithm-only point reuses its retained chart for the point logarithm
    expect_matrix_near(matrix_log(logarithm_from), matrix_log(uncached_from), 2.0e-12);
    // logarithm-only endpoints still produce the same cached geometry logarithm
    expect_matrix_near(
      geometry.logarithm(logarithm_from, logarithm_to), geometry.logarithm(uncached_from, uncached_to), 2.0e-10);
    // a spectral-only point uses its retained basis even when divided differences are recomputed
    expect_matrix_near(
      matrix_log_frechet(spectral_from, first_tangent), matrix_log_frechet(uncached_from, first_tangent), 2.0e-10);
    // a coherent spectral and divided-difference cache inverts the log Frechet action through exp
    expect_exp_log_frechet_inverse(cached_from, first_tangent, 2.0e-10);
}

// close spectra keep cached logarithmic divided differences coherent with the exp/log Frechet inverse
TEST(SPDBatchGeometry, LogEuclideanNearRepeatedCacheMatchesUncachedFrechetActions) {
    using cached_geometry = LogEuclideanSPDGeometry<double, 2, Usage::TangentMetric>;
    const cached_geometry geometry;
    const Matrix<double, 2, 2> close_spectrum({2, 1.0e-7, 1.0e-7, 2 + 2.0e-7});
    const cached_geometry::Point cached(close_spectrum);
    const SPDMatrix<double, 2> uncached(close_spectrum);
    const auto tangent = off_diagonal_tangent(0.4, -0.3);

    // cached near-repeated divided differences reproduce the uncached log derivative
    expect_matrix_near(matrix_log_frechet(cached, tangent), matrix_log_frechet(uncached, tangent), 2.0e-9);
    // cached close-spectrum factors preserve the reciprocal exp/log Frechet identity
    expect_exp_log_frechet_inverse(cached, tangent, 2.0e-9);
    // uncached close-spectrum factors provide the same reciprocal differential oracle
    expect_exp_log_frechet_inverse(uncached, tangent, 2.0e-9);
    // the tangent-metric cache does not alter the norm at a close spectrum
    EXPECT_NEAR(geometry.norm(cached, tangent), geometry.norm(uncached, tangent), 2.0e-9);
}

// affine-invariant square-root caches preserve all operations on noncommuting SPD inputs
TEST(SPDBatchGeometry, AffineInvariantCachePoliciesMatchUncachedOperations) {
    using sqrt_point = SPDMatrix<double, 2, Cache::Sqrt>;
    using inverse_sqrt_point = SPDMatrix<double, 2, Cache::InverseSqrt>;
    const airm_base_geometry geometry;
    const airm_base_geometry::Point cached_from(noncommuting_left());
    const airm_base_geometry::Point cached_to(noncommuting_right());
    const sqrt_point sqrt_from(noncommuting_left());
    const inverse_sqrt_point inverse_sqrt_to(noncommuting_right());
    const auto sqrt_from_view = sqrt_from.view();
    const auto inverse_sqrt_to_view = inverse_sqrt_to.view();
    const SPDMatrix<double, 2> uncached_from(noncommuting_left());
    const SPDMatrix<double, 2> uncached_to(noncommuting_right());
    const auto first_tangent = off_diagonal_tangent(0.25, -0.4);
    const auto second_tangent = off_diagonal_tangent(-0.35, 0.3);

    // affine-invariant exponential returns the geometry's square-root cache policy
    static_assert(
      std::same_as<decltype(geometry.exponential(sqrt_from_view, first_tangent)), airm_base_geometry::Point>);
    // affine-invariant polynomial retraction returns the same geometry point policy
    static_assert(
      std::same_as<decltype(geometry.retract(inverse_sqrt_to_view, first_tangent, 0.35)), airm_base_geometry::Point>);
    // a full base-point cache retains both factors while matching the uncached operation results
    expect_cached_geometry_operations(
      geometry, cached_from, cached_to, uncached_from, uncached_to, first_tangent, second_tangent, 3.0e-10);
    // cached square-root and inverse-square-root factors agree with uncached local factors everywhere
    expect_cached_geometry_operations(
      geometry, sqrt_from_view, inverse_sqrt_to_view, uncached_from, uncached_to, first_tangent, second_tangent,
      3.0e-10);
}

}   // namespace
