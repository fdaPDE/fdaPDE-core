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

namespace {

namespace native = fdapde;

using FixedGeometry = fdapde::manifold::AffineInvariantSPDGeometry<double, 3>;
using DynamicGeometry = fdapde::manifold::AffineInvariantSPDGeometry<double, fdapde::Dynamic>;

// the fixed geometry satisfies the operations required for vector transport
static_assert(fdapde::manifold::VectorTransportGeometry<FixedGeometry>);
// the dynamic geometry exposes the same vector-transport contract
static_assert(fdapde::manifold::VectorTransportGeometry<DynamicGeometry>);
// points use the native SPD owner with the geometry's fixed order
static_assert(std::is_same_v<FixedGeometry::Point, native::SPDMatrix<double, 3, 3>>);
// tangents use native symmetric storage rather than logarithmic-coordinate vectors
static_assert(std::is_same_v<FixedGeometry::Tangent, native::SymmetricMatrix<double, 3, 3>>);
// the fixed geometry obtains its order from the template arguments
static_assert(std::is_default_constructible_v<FixedGeometry>);
// a runtime order cannot override a fixed geometry's template order
static_assert(!std::is_constructible_v<FixedGeometry, int>);
// a dynamic geometry cannot be constructed without specifying its order
static_assert(!std::is_default_constructible_v<DynamicGeometry>);
// a dynamic geometry accepts an explicit runtime order
static_assert(std::is_constructible_v<DynamicGeometry, int>);

constexpr std::array<double, 9> point_x_coefficients {4.0, 0.6, 0.2, 0.6, 2.5, -0.3, 0.2, -0.3, 1.7};
constexpr std::array<double, 9> point_y_coefficients {1.8, -0.25, 0.15, -0.25, 3.3, 0.4, 0.15, 0.4, 2.2};

// materializes a checked SPD point from the supplied row-major three-by-three coefficients
template <typename Geometry> typename Geometry::Point make_point(const std::array<double, 9>& coefficients) {
    native::Matrix<double, 3, 3> dense;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) { dense(i, j) = coefficients[static_cast<std::size_t>(3 * i + j)]; }
    }
    return typename Geometry::Point(dense);
}

// constructs a scalar multiple of the three-by-three identity for analytic scale oracles
template <typename Geometry> typename Geometry::Point make_diagonal_point(double diagonal) {
    native::Matrix<double, 3, 3> dense;
    dense.set_zero();
    for (int i = 0; i < 3; ++i) { dense(i, i) = diagonal; }
    return typename Geometry::Point(dense);
}

// loads the six lower-triangular coefficients into fixed or dynamic symmetric tangent storage
template <typename Geometry> typename Geometry::Tangent make_tangent(const std::array<double, 6>& coefficients) {
    typename Geometry::Tangent tangent;
    if constexpr (Geometry::Tangent::Rows == fdapde::Dynamic) { tangent.resize(3, 3); }
    std::size_t index = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j <= i; ++j) { tangent(i, j) = coefficients[index++]; }
    }
    return tangent;
}

// computes a Frobenius norm with hypot to avoid squaring extreme coefficients
template <typename MatrixType> double matrix_norm(const MatrixType& matrix) {
    double result = 0;
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) { result = std::hypot(result, static_cast<double>(matrix(i, j))); }
    }
    return result;
}

// computes the Frobenius norm of coefficient differences using hypot accumulation
template <typename Lhs, typename Rhs> double matrix_difference_norm(const Lhs& lhs, const Rhs& rhs) {
    double result = 0;
    for (int i = 0; i < lhs.rows(); ++i) {
        for (int j = 0; j < lhs.cols(); ++j) {
            result = std::hypot(result, static_cast<double>(lhs(i, j)) - static_cast<double>(rhs(i, j)));
        }
    }
    return result;
}

// computes the full Frobenius product, including both symmetric off-diagonal entries
template <typename Lhs, typename Rhs> double full_inner(const Lhs& lhs, const Rhs& rhs) {
    double result = 0;
    for (int i = 0; i < lhs.rows(); ++i) {
        for (int j = 0; j < lhs.cols(); ++j) {
            result += static_cast<double>(lhs(i, j)) * static_cast<double>(rhs(i, j));
        }
    }
    return result;
}

// compares shapes and all coefficients against an absolute-tolerance matrix oracle
template <typename Lhs, typename Rhs> void expect_matrix_near(const Lhs& lhs, const Rhs& rhs, double tolerance) {
    // the coefficient comparison rejects a different row count before indexing
    ASSERT_EQ(lhs.rows(), rhs.rows());
    // the coefficient comparison rejects a different column count before indexing
    ASSERT_EQ(lhs.cols(), rhs.cols());
    for (int i = 0; i < lhs.rows(); ++i) {
        for (int j = 0; j < lhs.cols(); ++j) {
            // each coefficient matches the supplied matrix oracle within the requested tolerance
            EXPECT_NEAR(static_cast<double>(lhs(i, j)), static_cast<double>(rhs(i, j)), tolerance);
        }
    }
}

// bounds the Frobenius difference relative to the oracle norm with an absolute floor
template <typename Lhs, typename Rhs>
void expect_matrix_relative_near(const Lhs& lhs, const Rhs& rhs, double tolerance) {
    // the matrix difference is bounded relative to the oracle norm, with an absolute floor near zero
    EXPECT_LT(matrix_difference_norm(lhs, rhs), tolerance * (1.0 + matrix_norm(rhs)));
}

// computes a congruence oracle with explicit four-index sums independent of the geometry helpers
template <typename Middle>
native::SymmetricMatrix<double, 3, 3> congruence(const native::Matrix<double, 3, 3>& outer, const Middle& middle) {
    native::SymmetricMatrix<double, 3, 3> result;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j <= i; ++j) {
            double value = 0;
            for (int k = 0; k < 3; ++k) {
                for (int l = 0; l < 3; ++l) {
                    value += static_cast<double>(outer(i, k)) * static_cast<double>(middle(k, l)) *
                             static_cast<double>(outer(j, l));
                }
            }
            result(i, j) = value;
        }
    }
    return result;
}

}   // namespace

// identity-point Frobenius oracles and gradient duality check the ambient tangent metric and vector operations
TEST(AffineInvariantSPDGeometry, DefinesAmbientTangentMetricAndVectorOperations) {
    const FixedGeometry geometry;
    const auto identity = make_diagonal_point<FixedGeometry>(1.0);
    const auto u = make_tangent<FixedGeometry>({0.3, -0.2, 0.4, 0.1, 0.25, -0.15});
    const auto v = make_tangent<FixedGeometry>({-0.1, 0.35, 0.2, -0.25, 0.05, 0.45});

    // the fixed three-by-three geometry reports matrix order three
    EXPECT_EQ(geometry.order(), 3);
    // a symmetric three-by-three tangent has six independent coefficients
    EXPECT_EQ(geometry.dimension(), 6);
    // at the identity the metric product equals the full Frobenius product
    EXPECT_NEAR(geometry.inner_product(identity, u, v), full_inner(u, v), 1.0e-12);
    // at the identity the metric norm equals the scale-safe Frobenius oracle
    EXPECT_NEAR(geometry.norm(identity, u), matrix_norm(u), 1.0e-12);
    // projecting a symmetric tangent preserves all of its coefficients
    expect_matrix_near(geometry.project(identity, u), u, 0.0);
    // the zero tangent has zero Frobenius norm
    EXPECT_DOUBLE_EQ(matrix_norm(geometry.zero_tangent(identity)), 0.0);

    const auto combined = geometry.linear_combination(identity, 2.0, u, -0.5, v);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j <= i; ++j) {
            // each stored coefficient equals the explicit weighted sum of the two tangents
            EXPECT_NEAR(combined(i, j), 2.0 * static_cast<double>(u(i, j)) - 0.5 * static_cast<double>(v(i, j)), 0.0);
        }
    }

    const auto point = make_point<FixedGeometry>(point_x_coefficients);
    const auto euclidean_gradient = make_tangent<FixedGeometry>({0.7, -0.3, 0.2, 0.1, 0.25, -0.4});
    const auto direction = make_tangent<FixedGeometry>({0.15, 0.4, -0.2, -0.3, 0.1, 0.35});
    const auto riemannian_gradient = geometry.euclidean_to_riemannian_gradient(point, euclidean_gradient);
    // the converted gradient pairs with a direction exactly as the Frobenius gradient does
    EXPECT_NEAR(
      geometry.inner_product(point, riemannian_gradient, direction), full_inner(euclidean_gradient, direction),
      1.0e-10);
}

// geodesic inversion, distance and transport identities check mutually consistent maps on noncommuting matrices
TEST(AffineInvariantSPDGeometry, ExpLogDistanceAndTransportAreConsistent) {
    const FixedGeometry geometry;
    const auto point = make_point<FixedGeometry>(point_x_coefficients);
    const auto u = make_tangent<FixedGeometry>({0.3, -0.2, 0.4, 0.1, 0.25, -0.15});
    const auto v = make_tangent<FixedGeometry>({-0.1, 0.35, 0.2, -0.25, 0.05, 0.45});
    constexpr double step = 0.35;

    const auto next = geometry.exponential(point, v, step);
    const auto expected_tangent = geometry.linear_combination(point, step, v, 0.0, geometry.zero_tangent(point));
    // logarithm inverts the affine-invariant exponential and recovers the scaled tangent
    EXPECT_LT(matrix_difference_norm(geometry.logarithm(point, next), expected_tangent), 1.0e-10);
    // the geodesic distance equals the norm of the scaled initial tangent
    EXPECT_NEAR(geometry.distance(point, next), geometry.norm(point, expected_tangent), 1.0e-10);
    const auto target = make_point<FixedGeometry>(point_y_coefficients);
    // exponentiating the logarithm recovers the independently supplied target point
    expect_matrix_relative_near(geometry.exponential(point, geometry.logarithm(point, target)), target, 1.0e-10);

    const auto transported_u = geometry.transport(point, next, u);
    const auto transported_v = geometry.transport(point, next, v);
    // parallel transport preserves the metric product of the two tangents
    EXPECT_NEAR(
      geometry.inner_product(point, u, v), geometry.inner_product(next, transported_u, transported_v), 5.0e-10);
    // transporting back along the same geodesic recovers the original tangent
    expect_matrix_relative_near(geometry.transport(next, point, transported_u), u, 1.0e-10);
}

// step refinement and analytic scaled tangents check the quadratic retraction without replacing it by the exponential
TEST(AffineInvariantSPDGeometry, PolynomialRetractionIsSecondOrderAndScaleSafe) {
    const FixedGeometry geometry;
    const auto point = make_point<FixedGeometry>(point_x_coefficients);
    const auto tangent = make_tangent<FixedGeometry>({0.8, -0.5, 0.6, 0.25, -0.4, 0.7});
    constexpr double full_step = 0.8;
    constexpr double half_step = 0.5 * full_step;

    const double full_error = matrix_difference_norm(
      geometry.retract(point, tangent, full_step), geometry.exponential(point, tangent, full_step));
    const double half_error = matrix_difference_norm(
      geometry.retract(point, tangent, half_step), geometry.exponential(point, tangent, half_step));
    // the polynomial retraction differs measurably from the exact exponential at this step
    EXPECT_GT(full_error, 1.0e-8);
    // halving the step reduces the retraction error by more than six, consistent with cubic local error
    EXPECT_LT(half_error, full_error / 6.0);
    const auto large_step = geometry.retract(point, tangent, 100.0);
    // a large finite step still produces a result with a finite matrix norm
    EXPECT_TRUE(std::isfinite(matrix_norm(large_step)));

    const auto identity = make_diagonal_point<FixedGeometry>(1.0);
    const auto huge_tangent = make_tangent<FixedGeometry>({1.0e200, 0.0, 1.0e200, 0.0, 0.0, 1.0e200});
    const auto reciprocal_step = geometry.retract(identity, huge_tangent, 1.0e-200);
    const auto expected = make_diagonal_point<FixedGeometry>(2.5);
    // scaling the huge tangent by its reciprocal step gives the analytic retraction 2.5I
    expect_matrix_relative_near(reciprocal_step, expected, 1.0e-14);
    // a zero step returns the original identity despite the large tangent coefficients
    expect_matrix_near(geometry.retract(identity, huge_tangent, 0.0), identity, 0.0);
}

// an independently computed invertible congruence preserves the metric and commutes with maps and transport
TEST(AffineInvariantSPDGeometry, IsInvariantAndEquivariantUnderGeneralCongruence) {
    const FixedGeometry geometry;
    const auto point = make_point<FixedGeometry>(point_x_coefficients);
    const auto target = make_point<FixedGeometry>(point_y_coefficients);
    const auto u = make_tangent<FixedGeometry>({0.3, -0.2, 0.4, 0.1, 0.25, -0.15});
    const auto v = make_tangent<FixedGeometry>({-0.1, 0.35, 0.2, -0.25, 0.05, 0.45});
    const native::Matrix<double, 3, 3> basis({1.2, -0.2, 0.1, 0.3, 0.9, -0.15, -0.1, 0.25, 1.1});
    const FixedGeometry::Point transformed_point(congruence(basis, point));
    const FixedGeometry::Point transformed_target(congruence(basis, target));
    const auto transformed_u = congruence(basis, u);
    const auto transformed_v = congruence(basis, v);

    const double original_inner = geometry.inner_product(point, u, v);
    // a general invertible congruence preserves the metric product within relative tolerance
    EXPECT_LT(
      std::abs(geometry.inner_product(transformed_point, transformed_u, transformed_v) - original_inner),
      1.0e-9 * (1.0 + std::abs(original_inner)));
    // the same congruence preserves the distance between the transformed points
    EXPECT_NEAR(geometry.distance(transformed_point, transformed_target), geometry.distance(point, target), 1.0e-9);
    // exponentiation commutes with applying the congruence to the point and tangent
    expect_matrix_relative_near(
      geometry.exponential(transformed_point, transformed_u, 0.3),
      congruence(basis, geometry.exponential(point, u, 0.3)), 1.0e-9);
    // logarithm commutes with the congruence applied to both endpoint matrices
    expect_matrix_relative_near(
      geometry.logarithm(transformed_point, transformed_target), congruence(basis, geometry.logarithm(point, target)),
      1.0e-9);
    // the polynomial retraction commutes with the congruence transformation
    expect_matrix_relative_near(
      geometry.retract(transformed_point, transformed_u, 0.3), congruence(basis, geometry.retract(point, u, 0.3)),
      1.0e-9);
    // parallel transport commutes with the congruence on endpoints and tangent
    expect_matrix_relative_near(
      geometry.transport(transformed_point, transformed_target, transformed_u),
      congruence(basis, geometry.transport(point, target, u)), 1.0e-9);
}

// identical three-by-three inputs give matching geometry results for fixed and dynamic storage
TEST(AffineInvariantSPDGeometry, FixedAndDynamicThreeByThreeOperationsAgree) {
    const FixedGeometry fixed_geometry;
    const DynamicGeometry dynamic_geometry(3);
    const auto fixed_point = make_point<FixedGeometry>(point_x_coefficients);
    const auto dynamic_point = make_point<DynamicGeometry>(point_x_coefficients);
    const auto fixed_target = make_point<FixedGeometry>(point_y_coefficients);
    const auto dynamic_target = make_point<DynamicGeometry>(point_y_coefficients);
    const auto fixed_u = make_tangent<FixedGeometry>({0.3, -0.2, 0.4, 0.1, 0.25, -0.15});
    const auto dynamic_u = make_tangent<DynamicGeometry>({0.3, -0.2, 0.4, 0.1, 0.25, -0.15});
    const auto fixed_v = make_tangent<FixedGeometry>({-0.1, 0.35, 0.2, -0.25, 0.05, 0.45});
    const auto dynamic_v = make_tangent<DynamicGeometry>({-0.1, 0.35, 0.2, -0.25, 0.05, 0.45});

    // the dynamic geometry reports the same matrix order as the fixed geometry
    EXPECT_EQ(dynamic_geometry.order(), fixed_geometry.order());
    // both storage extents have the same number of independent tangent coefficients
    EXPECT_EQ(dynamic_geometry.dimension(), fixed_geometry.dimension());
    // the metric products agree for identical coefficients in fixed and dynamic storage
    EXPECT_NEAR(
      dynamic_geometry.inner_product(dynamic_point, dynamic_u, dynamic_v),
      fixed_geometry.inner_product(fixed_point, fixed_u, fixed_v), 1.0e-12);
    // the metric norms agree for identical fixed and dynamic tangents
    EXPECT_NEAR(dynamic_geometry.norm(dynamic_point, dynamic_u), fixed_geometry.norm(fixed_point, fixed_u), 1.0e-12);
    // dynamic projection preserves the same coefficients as the fixed tangent
    expect_matrix_near(dynamic_geometry.project(dynamic_point, dynamic_u), fixed_u, 1.0e-12);
    // both geometries produce the same zero tangent
    expect_matrix_near(dynamic_geometry.zero_tangent(dynamic_point), fixed_geometry.zero_tangent(fixed_point), 0.0);
    // the same weights and tangent coefficients give identical linear combinations
    expect_matrix_near(
      dynamic_geometry.linear_combination(dynamic_point, 0.3, dynamic_u, -0.7, dynamic_v),
      fixed_geometry.linear_combination(fixed_point, 0.3, fixed_u, -0.7, fixed_v), 1.0e-12);
    // the polynomial retractions agree across fixed and dynamic storage
    expect_matrix_near(
      dynamic_geometry.retract(dynamic_point, dynamic_u, 0.2), fixed_geometry.retract(fixed_point, fixed_u, 0.2),
      1.0e-12);
    // exponential maps give identical fixed and dynamic point coefficients
    expect_matrix_near(
      dynamic_geometry.exponential(dynamic_point, dynamic_u, 0.2),
      fixed_geometry.exponential(fixed_point, fixed_u, 0.2), 1.0e-12);
    // the logarithmic maps agree coefficient by coefficient across storage extents
    expect_matrix_near(
      dynamic_geometry.logarithm(dynamic_point, dynamic_target), fixed_geometry.logarithm(fixed_point, fixed_target),
      1.0e-12);
    // the distances agree for the same source and target matrices
    EXPECT_NEAR(
      dynamic_geometry.distance(dynamic_point, dynamic_target), fixed_geometry.distance(fixed_point, fixed_target),
      1.0e-12);
    // parallel transport yields identical fixed and dynamic tangent coefficients
    expect_matrix_near(
      dynamic_geometry.transport(dynamic_point, dynamic_target, dynamic_u),
      fixed_geometry.transport(fixed_point, fixed_target, fixed_u), 1.0e-12);
    // gradient conversion yields identical fixed and dynamic tangent coefficients
    expect_matrix_near(
      dynamic_geometry.euclidean_to_riemannian_gradient(dynamic_point, dynamic_v),
      fixed_geometry.euclidean_to_riemannian_gradient(fixed_point, fixed_v), 1.0e-12);
}

// invalid runtime orders and incompatible point or tangent shapes are rejected with specific exception types
TEST(AffineInvariantSPDGeometry, RejectsUnsupportedOrdersAndMismatchedShapes) {
    // order zero is rejected because the geometry requires a nonempty matrix
    EXPECT_THROW(DynamicGeometry(0), std::invalid_argument);
    // a negative runtime order is rejected before allocating any storage
    EXPECT_THROW(DynamicGeometry(-1), std::invalid_argument);
    // order 46341 is rejected because its square exceeds the supported int index range
    EXPECT_THROW(DynamicGeometry(46341), std::length_error);
    const int maximum_order = std::numeric_limits<int>::max();
    // the largest int order is rejected without overflowing the workspace-size calculation
    EXPECT_THROW(DynamicGeometry {maximum_order}, std::length_error);
    const DynamicGeometry largest_supported(46340);
    // order 46340 reports the analytically computed symmetric tangent dimension
    EXPECT_EQ(largest_supported.dimension(), std::size_t(1073720970));

    const DynamicGeometry geometry(3);
    const auto point = make_point<DynamicGeometry>(point_x_coefficients);
    native::Matrix<double, 2, 2> wrong_dense;
    wrong_dense.set_zero();
    wrong_dense(0, 0) = 1.0;
    wrong_dense(1, 1) = 1.0;
    const DynamicGeometry::Point wrong_point(wrong_dense);
    DynamicGeometry::Tangent wrong_tangent(2, 2);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j <= i; ++j) { wrong_tangent(i, j) = i == j ? 1.0 : 0.0; }
    }

    // a two-by-two target is rejected by a geometry of order three
    EXPECT_THROW(geometry.distance(point, wrong_point), std::invalid_argument);
    // a two-by-two tangent is rejected when computing its norm at an order-three point
    EXPECT_THROW(geometry.norm(point, wrong_tangent), std::invalid_argument);
    // projection rejects the same incompatible tangent dimensions
    EXPECT_THROW(geometry.project(point, wrong_tangent), std::invalid_argument);
}

// hypot and logarithmic scale-ratio oracles check norms and distances at very large and very small scales
TEST(AffineInvariantSPDGeometry, NormAndDistanceRemainRepresentableAcrossScales) {
    const FixedGeometry geometry;
    const auto identity = make_diagonal_point<FixedGeometry>(1.0);
    for (const double scale : {1.0e200, 1.0e-200}) {
        const auto tangent = make_tangent<FixedGeometry>({scale, 0.0, scale, 0.0, 0.0, scale});
        const double expected = std::hypot(std::hypot(scale, scale), scale);
        // at the identity the norm agrees with a hypot oracle at both extreme tangent scales
        EXPECT_NEAR(geometry.norm(identity, tangent) / expected, 1.0, 1.0e-14);
    }

    for (const double scale : {1.0e150, 1.0e-150}) {
        const auto point = make_diagonal_point<FixedGeometry>(scale);
        const auto tangent = make_tangent<FixedGeometry>({scale, 0.0, scale, 0.0, 0.0, scale});
        // scaling point and diagonal tangent together preserves the metric norm sqrt(3)
        EXPECT_NEAR(geometry.norm(point, tangent), std::sqrt(3.0), 1.0e-12);
    }

    const auto small = make_diagonal_point<FixedGeometry>(1.0e-150);
    const auto large = make_diagonal_point<FixedGeometry>(1.0e150);
    const double expected_distance = std::sqrt(3.0) * std::log(1.0e300);
    // distance between scalar identity multiples equals sqrt(3) times the log of their scale ratio
    EXPECT_NEAR(geometry.distance(small, large), expected_distance, 1.0e-10);
}
