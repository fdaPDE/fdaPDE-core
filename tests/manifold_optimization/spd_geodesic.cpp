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
using Point = SPDMatrix<double, 2, 2>;
using Full = Cache::Union<Cache::Spectral, Cache::Log, Cache::Sqrt, Cache::InverseSqrt, Cache::LogDividedDifferences>;

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
    using Result = SPDMatrix<S, N, N, Full>;
    const SPDMatrix<S, N, N, Cache::Log> first(Matrix<S, 2, 2>({3, 1, 1, 2}));
    const SPDMatrix<S, N, N, Cache::Spectral> last(Matrix<S, 2, 2>({2, -0.5, -0.5, 4}));
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
        const SymmetricMatrix<S, N, N> coefficients(expression);
        // symmetric construction preserves the same coefficients without requiring an SPD result type
        expect_point(coefficients, result, tolerance);
        // ordinary materialization cannot silently return a certified SPD owner
        static_assert(!SPDLike<decltype(expression.eval_matrix())>);
        // interpolation and extrapolation match the public logarithm/exponential construction
        expect_point(result, geometry.exponential(first, tangent, t), tolerance);
        // the rounded prepared coefficients independently satisfy checked SPD construction
        EXPECT_NO_THROW((SPDMatrix<S, N, N>(Matrix<S, N, N>(result))));
        // prepared caches agree with the logarithm of independently checked result coefficients
        expect_point(result.cache().template matrix<Cache::Log>(), matrix_log(SPDMatrix<S, N, N>(result)), tolerance);
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
    MatrixBatch<SPDMatrix<double, 2, 2, Cache::Log>> points(1);
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
    const auto wrong = SPDMatrix<double, Dynamic, Dynamic>::Identity(3);
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
    using Symmetric = SymmetricMatrix<double, 2, 2>;
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
    const auto first = SPDMatrix<double, Dynamic, Dynamic>::Identity(3);
    const SPDMatrix<double, Dynamic, Dynamic> last(Matrix<double, 3, 3>({4, 0, 0, 0, 9, 0, 0, 0, 16}));
    const auto airm = AffineInvariantSPDGeometry<double, Dynamic>(3).geodesic(first, last);
    const auto le = LogEuclideanSPDGeometry<double, Dynamic>(3).geodesic(first, last);
    const Matrix<double, 3, 3> expected({2, 0, 0, 0, 3, 0, 0, 0, 4});
    // the AIRM midpoint takes the principal square root of the diagonal endpoint
    expect_point(SPDMatrix<double, Dynamic, Dynamic>(airm(0.5)), expected);
    // the LE midpoint has the same closed-form value on commuting diagonal endpoints
    expect_point(SPDMatrix<double, Dynamic, Dynamic>(le(0.5)), expected);
}
}   // namespace
