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
using Dense = Matrix<double, 2, 2>;
using Point = SPDMatrix<double, 2>;
using Full = Cache::Union<Cache::Spectral, Cache::Log, Cache::Sqrt, Cache::InverseSqrt, Cache::LogDividedDifferences>;

/// @brief bounds worker resources for the standalone geometry test process
class GeodesicWorkerEnvironment : public ::testing::Environment {
    /// @brief configures four threads before any geometry test invokes parallel sampling
    void SetUp() override { parallel_set_num_threads(4); }
    /// @brief drains the executor before the geometry test process exits
    void TearDown() override { parallel_join(); }
};
[[maybe_unused]] ::testing::Environment* const geodesic_execution_environment =
  ::testing::AddGlobalTestEnvironment(new GeodesicWorkerEnvironment);

template <typename Policy>
concept permits_sampling_policy =
  requires(const LogEuclideanGeometry<Point>& geometry, const Point& point, Policy policy) {
      geometry.geodesic(point, point, 3, policy);
  };

template <typename Actual, typename Expected>
void expect_point(const Actual& actual, const Expected& expected, double tolerance = 1e-10) {
    // row counts must agree before comparing the full matrices
    ASSERT_EQ(actual.rows(), expected.rows());
    // column counts must agree so no expected coefficient is skipped
    ASSERT_EQ(actual.cols(), expected.cols());
    for (int i = 0; i < actual.rows(); ++i)
        for (int j = 0; j < actual.cols(); ++j) {
            // each coefficient agrees with the supplied reference within the chosen tolerance
            EXPECT_NEAR(actual(i, j), expected(i, j), tolerance);
        }
}

template <typename Curve>
concept permits_const_temporary_curve = requires(const Curve& curve) { std::move(curve)(0.5); };

template <typename Geometry> void check_curve(const Geometry& geometry) {
    using S = typename Geometry::Scalar;
    constexpr int N = Geometry::Point::Rows;
    using Result = SPDMatrix<S, N, Full>;
    const SPDMatrix<S, N, Cache::Log> first(Matrix<S, 2, 2>({3, 1, 1, 2}));
    const SPDMatrix<S, N, Cache::Spectral> last(Matrix<S, 2, 2>({2, -0.5, -0.5, 4}));
    const auto curve = geometry.geodesic(first.view(), last.view());
    // a const temporary cannot leave a borrowed curve inside an escaping expression
    static_assert(!permits_const_temporary_curve<std::remove_cvref_t<decltype(curve)>>);
    const auto tangent = geometry.logarithm(first, last);
    const double tolerance = std::is_same_v<S, float> ? 2e-4 : 2e-10;
    for (double t : {-0.5, 0.0, 0.25, 1.0, 1.5}) {
        const auto expression = curve(t);
        // a deferred geodesic value is not trusted SPD storage before materialization
        static_assert(!SPDLike<decltype(expression)>);
        const Result result(expression);
        const SymmetricMatrix<S, N> coefficients(expression);
        // symmetric construction preserves the same coefficients without requiring an SPD result type
        expect_point(coefficients, result, tolerance);
        // ordinary materialization cannot silently return a certified SPD owner
        static_assert(!SPDLike<decltype(expression.eval_matrix())>);
        // interpolation and extrapolation match the public logarithm/exponential construction
        expect_point(result, geometry.exponential(first, tangent, t), tolerance);
        // the rounded prepared coefficients independently satisfy checked SPD construction
        EXPECT_NO_THROW((SPDMatrix<S, N>(Matrix<S, N, N>(result))));
        // prepared caches agree with the logarithm of independently checked result coefficients
        expect_point(result.cache().template matrix<Cache::Log>(), matrix_log(SPDMatrix<S, N>(result)), tolerance);
    }
    // the zero parameter reproduces the first endpoint
    expect_point(Result(curve(0)), first, tolerance);
    // the unit parameter reproduces the second endpoint
    expect_point(Result(curve(1)), last, tolerance);
}

// both metrics support noncommuting endpoints, mixed-policy views, destination caches and extrapolation
TEST(SPDGeodesic, MatchesPublicMapsForFixedDynamicAndFloatPoints) {
    // fixed AIRM points match the logarithm/exponential reference with mixed endpoint caches
    check_curve(AffineInvariantSPDGeometry<double, 2>());
    // fixed LE points match the logarithm/exponential reference with mixed endpoint caches
    check_curve(LogEuclideanSPDGeometry<double, 2>());
    // dynamic AIRM points satisfy the same formula and endpoint checks at runtime order two
    check_curve(AffineInvariantSPDGeometry<double, Dynamic>(2));
    // dynamic LE points satisfy the same formula and endpoint checks at runtime order two
    check_curve(LogEuclideanSPDGeometry<double, Dynamic>(2));
    // single-precision AIRM points match the public maps within the float tolerance
    check_curve(AffineInvariantSPDGeometry<float, 2>());
    // single-precision LE points match the public maps within the float tolerance
    check_curve(LogEuclideanSPDGeometry<float, 2>());
}

template <typename Geometry> void check_snapshot(const Geometry& geometry) {
    Point first(Dense({4, 0, 0, 9})), last(Dense({16, 0, 0, 1}));
    const auto curve = geometry.geodesic(first, last);
    double t = 0.5;
    const auto expression = curve(t);
    t = 0;
    first = Point::Identity();
    last = Point::Identity();
    const Dense midpoint({8, 0, 0, 3});
    // endpoint changes and later parameter changes do not alter the prepared snapshot or stored parameter
    expect_point(Point(expression), midpoint);
    const MatrixExpr<std::remove_cvref_t<decltype(expression)>>& base = expression;
    // dense conversion through the base materializes the same complete geometric value
    expect_point(Dense(base), midpoint);
    MatrixBatch<SPDMatrix<double, 2, Cache::Log>> points(1);
    points[0] = expression;
    // checked batch assignment preserves the analytic midpoint in the destination coefficients
    expect_point(points[0], midpoint);
    const auto owned_expression = Geometry().geodesic(Point(Dense({4, 0, 0, 9})), Point(Dense({16, 0, 0, 1})))(0.5);
    // temporary endpoints, geometry and curve may expire because the expression owns its prepared curve
    expect_point(Point(owned_expression), midpoint);
    const auto copied = curve;
    // a copied curve reproduces the analytic midpoint from the original snapshot
    expect_point(Point(copied(0.5)), midpoint);
}

// curves own endpoint snapshots while expressions borrow persistent curves and own temporary curves
TEST(SPDGeodesic, PreservesSnapshotsAndTemporaryLifetimes) {
    // the AIRM snapshots and temporary ownership preserve the analytic diagonal midpoint
    check_snapshot(AffineInvariantSPDGeometry<double, 2>());
    // the LE snapshots and temporary ownership preserve the same analytic diagonal midpoint
    check_snapshot(LogEuclideanSPDGeometry<double, 2>());
}

template <typename Geometry> void check_failures(const Geometry& geometry) {
    const Point first = Point::Identity();
    const Point last(Dense({2, 0, 0, 4}));
    const auto curve = geometry.geodesic(first, last);
    Point destination(Dense({3, 0, 0, 5}));
    const Point saved(destination);
    for (double t : {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        const auto expression = curve(t);
        // binding a nonfinite parameter is lazy; its materialization fails before replacing the destination
        EXPECT_THROW(destination.assign(expression), std::invalid_argument);
        // failed materialization preserves all previously verified destination coefficients
        expect_point(destination, saved);
    }
    for (double t : {-2000.0, 2000.0}) {
        // underflow or overflow cannot publish a singular or nonfinite geodesic point
        EXPECT_THROW(destination.assign(curve(t)), std::domain_error);
    }
    const auto identity_curve = geometry.geodesic(first, first);
    // a repeated unit spectrum remains identity even far outside the interpolation interval
    expect_point(Point(identity_curve(100)), first);
    const auto wrong = SPDMatrix<double, Dynamic>::Identity(3);
    // preparation rejects endpoints incompatible with the geometry order
    EXPECT_THROW(geometry.geodesic(first, wrong), std::invalid_argument);
}

// numerical failures and shape mismatches preserve the same checked public SPD boundary
TEST(SPDGeodesic, RejectsInvalidParametersResultsAndShapes) {
    // the AIRM rejects invalid parameters and shapes while preserving the saved destination
    check_failures(AffineInvariantSPDGeometry<double, 2>());
    // the LE rejects invalid parameters and shapes while preserving the saved destination
    check_failures(LogEuclideanSPDGeometry<double, 2>());
}

template <typename Geometry> void check_deferred_certification(const Geometry& geometry) {
    using Symmetric = SymmetricMatrix<double, 2>;
    const Point first = Point::Identity();
    const Point last(Dense({1, 0, 0, 2}));
    const auto curve = geometry.geodesic(first, last);
    const auto expression = curve(-100);
    const Dense expected({1, 0, 0, std::exp(-100 * std::log(2.0))});
    const MatrixExpr<std::remove_cvref_t<decltype(expression)>>& base = expression;
    const Symmetric coefficients(base);
    // the finite diagonal extrapolation can be stored even below the numerical SPD acceptance threshold
    expect_point(coefficients, expected, 1e-40);
    // dense construction through MatrixExpr must also defer numerical SPD certification
    expect_point(Dense(base), expected, 1e-40);
    MatrixBatch<Symmetric> unchecked(1);
    unchecked[0] = expression;
    // the same expression assigns to a symmetric batch without passing through an SPD owner
    expect_point(unchecked[0], expected, 1e-40);
    // explicitly promoting stored coefficients to SPD performs the deferred certification
    EXPECT_THROW((Point(coefficients)), std::domain_error);
    MatrixBatch<Point> checked(1);
    // an SPD batch certifies at assignment and rejects the same numerically ill-conditioned coefficients
    EXPECT_THROW(checked[0] = expression, std::domain_error);
    // failed certification leaves the batch's previously verified identity intact
    expect_point(checked[0], first);
    const Point shrinking(Dense({2, 0, 0, 4}));
    const auto underflow = geometry.geodesic(first, shrinking)(-2000);
    const Symmetric zero(underflow);
    // scalar underflow can be inspected as finite symmetric coefficients before SPD promotion
    expect_point(zero, Dense({0, 0, 0, 0}), 0);
    // those singular rounded coefficients cannot acquire verified SPD status
    EXPECT_THROW((Point(underflow)), std::domain_error);
}

// destination type decides certification for the same deferred expression in both metrics
TEST(SPDGeodesic, DefersCertificationUntilSPDDestination) {
    // the AIRM stores the finite extrapolation as symmetric data and rejects its SPD promotion
    check_deferred_certification(AffineInvariantSPDGeometry<double, 2>());
    // the LE stores the same finite extrapolation as symmetric data and rejects its SPD promotion
    check_deferred_certification(LogEuclideanSPDGeometry<double, 2>());
}

// runtime orders beyond the example's two-by-two case preserve shape and analytic diagonal values
TEST(SPDGeodesic, SupportsThreeDimensionalDynamicCurves) {
    const auto first = SPDMatrix<double, Dynamic>::Identity(3);
    const SPDMatrix<double, Dynamic> last(Matrix<double, 3, 3>({4, 0, 0, 0, 9, 0, 0, 0, 16}));
    const auto airm = AffineInvariantSPDGeometry<double, Dynamic>(3).geodesic(first, last);
    const auto le = LogEuclideanSPDGeometry<double, Dynamic>(3).geodesic(first, last);
    const Matrix<double, 3, 3> expected({2, 0, 0, 0, 3, 0, 0, 0, 4});
    // the AIRM midpoint takes the principal square root of the diagonal endpoint
    expect_point(SPDMatrix<double, Dynamic>(airm(0.5)), expected);
    // the LE midpoint has the same closed-form value on commuting diagonal endpoints
    expect_point(SPDMatrix<double, Dynamic>(le(0.5)), expected);
}

/// @brief checks default, sequential and parallel sampling against one curve with independent output cache policies
template <typename Geometry, SPDLike From, SPDLike To>
void check_sampled_curve(const Geometry& geometry, const From& from, const To& to, double tolerance = 2e-9) {
    using NativePoint = typename Geometry::Point;
    using UncachedPoint = SPDMatrix<typename NativePoint::Scalar, NativePoint::Rows>;
    const auto curve = geometry.geodesic(from, to);
    const auto points = geometry.geodesic(from, to, 10);
    const auto sequential = geometry.geodesic(from, to, 10, execution_seq);
    const auto parallel = geometry.geodesic(from, to, 10, execution_par);
    // batch elements retain the geometry's scalar, order and cache policy independently of endpoint types
    static_assert(std::same_as<std::remove_cvref_t<decltype(points)>, MatrixBatch<NativePoint>>);
    // the requested count includes both endpoints rather than adding them after sampling
    ASSERT_EQ(points.size(), 10);
    for (std::size_t i = 0; i < points.size(); ++i) {
        const UncachedPoint expected(curve(static_cast<double>(i) / (points.size() - 1)));
        // each coefficient matches the separately prepared curve at its uniformly spaced parameter
        expect_point(points[i], expected, tolerance);
        // an explicit sequential policy preserves the default sample value and index
        expect_point(sequential[i], expected, tolerance);
        // parallel completion order cannot permute the uniformly sampled output slots
        expect_point(parallel[i], expected, tolerance);
        if constexpr ((NativePoint::CachePolicy::Flags & Cache::Log::Flags) != 0) {
            // the retained log cache agrees with an independent uncached decomposition of the sampled point
            expect_point(points[i].cache().template matrix<Cache::Log>(), matrix_log(expected), tolerance);
        }
    }
    // the initial sample reconstructs the supplied first endpoint
    expect_point(points[0], from, tolerance);
    // the final sample reconstructs the supplied second endpoint
    expect_point(points[points.size() - 1], to, tolerance);
    using CachedResult = SPDMatrix<typename NativePoint::Scalar, NativePoint::Rows, Full>;
    const auto cached = geometry.template geodesic<Full>(from, to, 3);
    const auto uncached = geometry.template geodesic<Cache::None>(from, to, 3, execution_par);
    const auto parallel_cached = geometry.template geodesic<Full>(from, to, 3, execution_par);
    // an explicit cache policy changes the exact output owner without changing the geometry or endpoint policies
    static_assert(std::same_as<std::remove_cvref_t<decltype(cached)>, MatrixBatch<CachedResult>>);
    // explicitly removing caches keeps the geometry scalar and matrix order
    static_assert(std::same_as<std::remove_cvref_t<decltype(uncached)>, MatrixBatch<UncachedPoint>>);
    // selecting an output cache cannot change the number of inclusive samples
    ASSERT_EQ(cached.size(), 3);
    for (std::size_t i = 0; i < cached.size(); ++i) {
        const UncachedPoint expected(curve(static_cast<double>(i) / 2));
        // requested cached outputs follow the same prepared curve at both endpoints and the midpoint
        expect_point(cached[i], expected, tolerance);
        // parallel cached construction retains the same coefficients as the independently evaluated curve
        expect_point(parallel_cached[i], expected, tolerance);
        // workers populate each requested log cache from its own rounded sample
        expect_point(parallel_cached[i].cache().template matrix<Cache::Log>(), matrix_log(expected), tolerance);
        // disabling output caching leaves the geometric coefficients unchanged
        expect_point(uncached[i], expected, tolerance);
        // cached logarithms agree with an independent uncached decomposition of the rounded result
        expect_point(cached[i].cache().template matrix<Cache::Log>(), matrix_log(expected), tolerance);
        // retained square roots agree with independently decomposed sampled coefficients
        expect_point(cached[i].cache().template matrix<Cache::Sqrt>(), matrix_sqrt(expected), tolerance);
    }
    const auto endpoints = geometry.geodesic(from, to, 2);
    // the minimum supported count produces exactly the two requested endpoint samples
    ASSERT_EQ(endpoints.size(), 2);
    // the minimum batch starts at the same input endpoint as the larger batch
    expect_point(endpoints[0], from, tolerance);
    // the minimum batch ends at the second endpoint without dividing by a zero interval count
    expect_point(endpoints[1], to, tolerance);
    for (int count : {-1, 0, 1}) {
        // fewer than two samples cannot contain both endpoints and must fail at the public boundary
        EXPECT_THROW(geometry.geodesic(from, to, count), std::invalid_argument);
        // explicit output policy does not bypass the minimum sample-count contract
        EXPECT_THROW(geometry.template geodesic<Full>(from, to, count), std::invalid_argument);
        // parallel sampling applies count validation before allocating or submitting work
        EXPECT_THROW(geometry.template geodesic<Full>(from, to, count, execution_par), std::invalid_argument);
    }
}

// every SPD metric samples one prepared curve into owners with the geometry's exact cache policy
TEST(SPDGeodesic, BatchSamplesUseDeclaredPointTypeAndUniformParameters) {
    using CachedPoint = SPDMatrix<double, 2, Cache::Log>;
    const Point from(Dense({3, 1, 1, 2}));
    const SPDMatrix<double, 2, Cache::Spectral> to(Dense({2, -0.5, -0.5, 4}));
    // LE batch values agree with uniform curve samples despite mixed endpoint cache policies
    check_sampled_curve(LogEuclideanGeometry<CachedPoint>(), from.view(), to.view());
    // AIRM batch values retain cached owners and include both endpoints on the prepared curve
    check_sampled_curve(AffineInvariantGeometry<CachedPoint>(), from.view(), to.view());
    // BW batch values use the same positive horizontal lift as the prepared curve
    check_sampled_curve(BuresWassersteinGeometry<CachedPoint>(), from.view(), to.view());
    // LC batch values follow the prepared chart segment with matching endpoint coefficients
    check_sampled_curve(LogCholeskyGeometry<CachedPoint>(), from.view(), to.view());
    // the planar Cheeger specialization preserves its selected alignment throughout the sampled batch
    check_sampled_curve(CheegerLogEuclideanGeometry<CachedPoint>(.75), from.view(), to.view());
}

// batch sampling preserves runtime matrix order, float coefficients and both general Cheeger implementations
TEST(SPDGeodesic, BatchSamplesPreserveDynamicOrdersAndFloatScalars) {
    using FloatPoint = SPDMatrix<float, Dynamic, Cache::Spectral>;
    const FloatPoint float_from = FloatPoint::Identity(3);
    const FloatPoint float_to(Matrix<float, 3, 3>({2, 0, 0, 0, 3, 0, 0, 0, 4}));
    // dynamic float sampling retains order three and agrees with the curve within single-precision tolerance
    check_sampled_curve(LogEuclideanGeometry<FloatPoint>(3), float_from, float_to, 2e-4);
    const auto from = SPDMatrix<double, 3>::Identity();
    const SPDMatrix<double, 3> to(Matrix<double, 3, 3>({2, 0, 0, 0, 3, 0, 0, 0, 4}));
    // fixed orders beyond two use the general Cheeger curve with the same inclusive uniform sampling
    check_sampled_curve(CheegerLogEuclideanSPDGeometry<double, 3>(.75), from, to);
    // dynamic Cheeger output derives its order from the geometry rather than the fixed endpoint owner type
    check_sampled_curve(CheegerLogEuclideanSPDGeometry<double, Dynamic>(3, .75), from, to);
}

// returned batches own their coefficients after temporary curves, geometries and endpoints have expired
TEST(SPDGeodesic, BatchSamplesOwnTheirResults) {
    const auto points =
      LogEuclideanGeometry<Point>().geodesic(Point(Dense({4, 0, 0, 9})), Point(Dense({16, 0, 0, 1})), 3);
    // the middle element survives all input temporaries and equals the analytic commuting midpoint
    expect_point(points[1], Dense({8, 0, 0, 3}));
    Point from(Dense({4, 0, 0, 9})), to(Dense({16, 0, 0, 1}));
    const auto snapshot = AffineInvariantGeometry<Point>().geodesic(from, to, 3);
    from = Point::Identity();
    to = Point::Identity();
    // subsequent endpoint assignment cannot replace the owning batch's previously sampled midpoint
    expect_point(snapshot[1], Dense({8, 0, 0, 3}));
}
// worker-side cache failures reach the caller after joining and do not poison later geodesic calls
TEST(SPDGeodesic, ParallelSamplingPropagatesCacheFailureAndRecovers) {
    using ScalarPoint = SPDMatrix<double, 1>;
    const LogEuclideanGeometry<ScalarPoint> geometry;
    const ScalarPoint tiny(Vector<double, 1> {1e-310});
    const ScalarPoint unit = ScalarPoint::Identity();
    // the derivative of log at a tiny positive endpoint exceeds the scalar range in the requested output cache
    EXPECT_THROW(geometry.geodesic<Cache::LogDividedDifferences>(tiny, unit, 32, execution_par), std::domain_error);
    const auto recovered = geometry.geodesic<Cache::Log>(unit, unit, 10, execution_par);
    // a subsequent call completes all requested entries after the earlier parallel failure
    ASSERT_EQ(recovered.size(), 10);
    // the final slot and its cached logarithm match the independently known constant identity curve
    EXPECT_DOUBLE_EQ(recovered[9].cache().template matrix<Cache::Log>()(0, 0), 0.);
    // execution tags are accepted while unrelated argument types cannot select a sampling overload
    static_assert(
      permits_sampling_policy<execution_seq_t> && permits_sampling_policy<execution_par_t> &&
      !permits_sampling_policy<int>);
}

}   // namespace
