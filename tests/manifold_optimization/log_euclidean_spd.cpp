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

using FixedGeometry = fdapde::manifold::LogEuclideanSPDGeometry<double, 3>;
using DynamicGeometry = fdapde::manifold::LogEuclideanSPDGeometry<double, fdapde::Dynamic>;

static_assert(fdapde::manifold::VectorTransportGeometry<FixedGeometry>);
static_assert(fdapde::manifold::VectorTransportGeometry<DynamicGeometry>);
static_assert(std::is_same_v<FixedGeometry::Point, native::SPDMatrix<double, 3, 3>>);
static_assert(std::is_same_v<FixedGeometry::Tangent, native::SymmetricMatrix<double, 3, 3>>);
static_assert(std::is_default_constructible_v<FixedGeometry>);
static_assert(!std::is_constructible_v<FixedGeometry, int>);
static_assert(!std::is_default_constructible_v<DynamicGeometry>);
static_assert(std::is_constructible_v<DynamicGeometry, int>);

constexpr std::array<double, 9> point_x_coefficients {4.0, 0.6, 0.2, 0.6, 2.5, -0.3, 0.2, -0.3, 1.7};
constexpr std::array<double, 9> point_y_coefficients {1.8, -0.25, 0.15, -0.25, 3.3, 0.4, 0.15, 0.4, 2.2};

template <typename Geometry> typename Geometry::Point make_point(const std::array<double, 9>& coefficients) {
    native::Matrix<double, 3, 3> dense;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) { dense(i, j) = coefficients[static_cast<std::size_t>(3 * i + j)]; }
    }
    return typename Geometry::Point(dense);
}

template <typename Geometry> typename Geometry::Point make_diagonal_point(double diagonal) {
    native::Matrix<double, 3, 3> dense;
    dense.set_zero();
    for (int i = 0; i < 3; ++i) { dense(i, i) = diagonal; }
    return typename Geometry::Point(dense);
}

template <typename Geometry> typename Geometry::Tangent make_tangent(const std::array<double, 6>& coefficients) {
    typename Geometry::Tangent tangent;
    if constexpr (Geometry::Tangent::Rows == fdapde::Dynamic) { tangent.resize(3, 3); }
    std::size_t index = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j <= i; ++j) { tangent(i, j) = coefficients[index++]; }
    }
    return tangent;
}

template <typename MatrixType> double matrix_norm(const MatrixType& matrix) {
    double result = 0;
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) { result = std::hypot(result, static_cast<double>(matrix(i, j))); }
    }
    return result;
}

template <typename Lhs, typename Rhs> double matrix_difference_norm(const Lhs& lhs, const Rhs& rhs) {
    double result = 0;
    for (int i = 0; i < lhs.rows(); ++i) {
        for (int j = 0; j < lhs.cols(); ++j) {
            result = std::hypot(result, static_cast<double>(lhs(i, j)) - static_cast<double>(rhs(i, j)));
        }
    }
    return result;
}

template <typename Lhs, typename Rhs> double full_inner(const Lhs& lhs, const Rhs& rhs) {
    double result = 0;
    for (int i = 0; i < lhs.rows(); ++i) {
        for (int j = 0; j < lhs.cols(); ++j) {
            result += static_cast<double>(lhs(i, j)) * static_cast<double>(rhs(i, j));
        }
    }
    return result;
}

template <typename Lhs, typename Rhs> void expect_matrix_near(const Lhs& lhs, const Rhs& rhs, double tolerance) {
    ASSERT_EQ(lhs.rows(), rhs.rows());
    ASSERT_EQ(lhs.cols(), rhs.cols());
    for (int i = 0; i < lhs.rows(); ++i) {
        for (int j = 0; j < lhs.cols(); ++j) {
            EXPECT_NEAR(static_cast<double>(lhs(i, j)), static_cast<double>(rhs(i, j)), tolerance);
        }
    }
}

}   // namespace

TEST(LogEuclideanSPDGeometry, DefinesAmbientTangentMetricAndVectorOperations) {
    const FixedGeometry geometry;
    const auto identity = make_diagonal_point<FixedGeometry>(1.0);
    const auto u = make_tangent<FixedGeometry>({0.3, -0.2, 0.4, 0.1, 0.25, -0.15});
    const auto v = make_tangent<FixedGeometry>({-0.1, 0.35, 0.2, -0.25, 0.05, 0.45});

    EXPECT_EQ(geometry.order(), 3);
    EXPECT_EQ(geometry.dimension(), 6);
    EXPECT_NEAR(geometry.inner_product(identity, u, v), full_inner(u, v), 1.0e-12);
    EXPECT_NEAR(geometry.norm(identity, u), matrix_norm(u), 1.0e-12);
    expect_matrix_near(geometry.project(identity, u), u, 0.0);
    EXPECT_DOUBLE_EQ(matrix_norm(geometry.zero_tangent(identity)), 0.0);

    const auto combined = geometry.linear_combination(identity, 2.0, u, -0.5, v);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j <= i; ++j) {
            EXPECT_NEAR(combined(i, j), 2.0 * static_cast<double>(u(i, j)) - 0.5 * static_cast<double>(v(i, j)), 0.0);
        }
    }

    const auto point = make_point<FixedGeometry>(point_x_coefficients);
    const auto euclidean_gradient = make_tangent<FixedGeometry>({0.7, -0.3, 0.2, 0.1, 0.25, -0.4});
    const auto direction = make_tangent<FixedGeometry>({0.15, 0.4, -0.2, -0.3, 0.1, 0.35});
    const auto riemannian_gradient = geometry.euclidean_to_riemannian_gradient(point, euclidean_gradient);
    EXPECT_NEAR(
      geometry.inner_product(point, riemannian_gradient, direction), full_inner(euclidean_gradient, direction),
      1.0e-10);
}

TEST(LogEuclideanSPDGeometry, ExpLogDistanceAndTransportAreConsistent) {
    const FixedGeometry geometry;
    const auto point = make_point<FixedGeometry>(point_x_coefficients);
    const auto u = make_tangent<FixedGeometry>({0.3, -0.2, 0.4, 0.1, 0.25, -0.15});
    const auto v = make_tangent<FixedGeometry>({-0.1, 0.35, 0.2, -0.25, 0.05, 0.45});
    constexpr double step = 0.35;

    const auto next = geometry.exponential(point, v, step);
    const auto expected_tangent = geometry.linear_combination(point, step, v, 0.0, geometry.zero_tangent(point));
    const auto recovered_tangent = geometry.logarithm(point, next);
    EXPECT_LT(matrix_difference_norm(recovered_tangent, expected_tangent), 1.0e-10);
    EXPECT_NEAR(geometry.distance(point, next), geometry.norm(point, expected_tangent), 1.0e-10);
    expect_matrix_near(geometry.retract(point, v, step), next, 1.0e-12);

    const auto transported_u = geometry.transport(point, next, u);
    const auto transported_v = geometry.transport(point, next, v);
    EXPECT_NEAR(
      geometry.inner_product(point, u, v), geometry.inner_product(next, transported_u, transported_v), 1.0e-10);
    const auto source_chart_tangent = native::matrix_log_frechet(point, u);
    const auto target_chart_tangent = native::matrix_log_frechet(next, transported_u);
    EXPECT_LT(matrix_difference_norm(source_chart_tangent, target_chart_tangent), 1.0e-10);
}

TEST(LogEuclideanSPDGeometry, FixedAndDynamicThreeByThreeOperationsAgree) {
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

    EXPECT_EQ(dynamic_geometry.order(), fixed_geometry.order());
    EXPECT_EQ(dynamic_geometry.dimension(), fixed_geometry.dimension());
    EXPECT_NEAR(
      dynamic_geometry.inner_product(dynamic_point, dynamic_u, dynamic_v),
      fixed_geometry.inner_product(fixed_point, fixed_u, fixed_v), 1.0e-12);
    EXPECT_NEAR(dynamic_geometry.norm(dynamic_point, dynamic_u), fixed_geometry.norm(fixed_point, fixed_u), 1.0e-12);
    expect_matrix_near(dynamic_geometry.project(dynamic_point, dynamic_u), fixed_u, 1.0e-12);
    expect_matrix_near(dynamic_geometry.zero_tangent(dynamic_point), fixed_geometry.zero_tangent(fixed_point), 0.0);
    expect_matrix_near(
      dynamic_geometry.linear_combination(dynamic_point, 0.3, dynamic_u, -0.7, dynamic_v),
      fixed_geometry.linear_combination(fixed_point, 0.3, fixed_u, -0.7, fixed_v), 1.0e-12);

    const auto fixed_next = fixed_geometry.exponential(fixed_point, fixed_u, 0.2);
    const auto dynamic_next = dynamic_geometry.exponential(dynamic_point, dynamic_u, 0.2);
    expect_matrix_near(dynamic_next, fixed_next, 1.0e-12);
    expect_matrix_near(
      dynamic_geometry.logarithm(dynamic_point, dynamic_target), fixed_geometry.logarithm(fixed_point, fixed_target),
      1.0e-12);
    EXPECT_NEAR(
      dynamic_geometry.distance(dynamic_point, dynamic_target), fixed_geometry.distance(fixed_point, fixed_target),
      1.0e-12);
    expect_matrix_near(
      dynamic_geometry.transport(dynamic_point, dynamic_target, dynamic_u),
      fixed_geometry.transport(fixed_point, fixed_target, fixed_u), 1.0e-12);
    expect_matrix_near(
      dynamic_geometry.euclidean_to_riemannian_gradient(dynamic_point, dynamic_v),
      fixed_geometry.euclidean_to_riemannian_gradient(fixed_point, fixed_v), 1.0e-12);
}

TEST(LogEuclideanSPDGeometry, RejectsUnsupportedOrdersAndMismatchedShapes) {
    EXPECT_THROW(DynamicGeometry(0), std::invalid_argument);
    EXPECT_THROW(DynamicGeometry(-1), std::invalid_argument);
    EXPECT_THROW(DynamicGeometry(46341), std::length_error);
    const int maximum_order = std::numeric_limits<int>::max();
    EXPECT_THROW(DynamicGeometry {maximum_order}, std::length_error);
    const DynamicGeometry largest_supported(46340);
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

    EXPECT_THROW(geometry.distance(point, wrong_point), std::invalid_argument);
    EXPECT_THROW(geometry.norm(point, wrong_tangent), std::invalid_argument);
    EXPECT_THROW(geometry.project(point, wrong_tangent), std::invalid_argument);
}

TEST(LogEuclideanSPDGeometry, NormAndDistanceRemainRepresentableAcrossScales) {
    const FixedGeometry geometry;
    const auto identity = make_diagonal_point<FixedGeometry>(1.0);
    for (const double scale : {1.0e200, 1.0e-200}) {
        const auto tangent = make_tangent<FixedGeometry>({scale, 0.0, scale, 0.0, 0.0, scale});
        const double expected = std::hypot(std::hypot(scale, scale), scale);
        EXPECT_NEAR(geometry.norm(identity, tangent) / expected, 1.0, 1.0e-14);
    }

    const auto small = make_diagonal_point<FixedGeometry>(1.0e-150);
    const auto large = make_diagonal_point<FixedGeometry>(1.0e150);
    const double expected_distance = std::sqrt(3.0) * std::log(1.0e300);
    EXPECT_NEAR(geometry.distance(small, large), expected_distance, 1.0e-10);
}
