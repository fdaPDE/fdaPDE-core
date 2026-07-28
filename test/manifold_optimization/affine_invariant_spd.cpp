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

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <type_traits>

namespace {

namespace native = fdapde::linalg;

using FixedGeometry = fdapde::manifold::AffineInvariantSPDGeometry<double, 3>;
using DynamicGeometry = fdapde::manifold::AffineInvariantSPDGeometry<double, fdapde::Dynamic>;

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
    return typename Geometry::Point(dense, native::checked);
}

template <typename Geometry> typename Geometry::Point make_diagonal_point(double diagonal) {
    native::Matrix<double, 3, 3> dense;
    dense.set_zero();
    for (int i = 0; i < 3; ++i) { dense(i, i) = diagonal; }
    return typename Geometry::Point(dense, native::checked);
}

template <typename Geometry> typename Geometry::Point make_diagonal_point(double first, double second, double third) {
    native::Matrix<double, 3, 3> dense;
    dense.set_zero();
    dense(0, 0) = first;
    dense(1, 1) = second;
    dense(2, 2) = third;
    return typename Geometry::Point(dense, native::checked);
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

template <typename Lhs, typename Rhs>
void expect_matrix_relative_near(const Lhs& lhs, const Rhs& rhs, double tolerance) {
    EXPECT_LT(matrix_difference_norm(lhs, rhs), tolerance * (1.0 + matrix_norm(rhs)));
}

template <typename Outer, typename Middle>
native::SymmetricMatrix<double, 3, 3> congruence(const Outer& outer, const Middle& middle) {
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

double log_divided_difference(double x, double y) {
    if (x == y) return 1.0 / x;
    const double delta = x - y;
    if (std::abs(delta) <= 0.5 * std::min(x, y)) {
        return 0.5 * (std::log1p(delta / y) / delta + std::log1p(-delta / x) / -delta);
    }
    return (std::log(x) - std::log(y)) / delta;
}

template <typename Geometry>
typename Geometry::Tangent diagonal_coefficient_action(
  const typename Geometry::Tangent& direction, const std::array<double, 3>& eigenvalues, const auto& coefficient) {
    auto result = make_tangent<Geometry>({0, 0, 0, 0, 0, 0});
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j <= i; ++j) {
            result(i, j) =
              coefficient(eigenvalues[static_cast<std::size_t>(i)], eigenvalues[static_cast<std::size_t>(j)]) *
              static_cast<double>(direction(i, j));
        }
    }
    return result;
}

template <typename Geometry>
typename Geometry::Tangent fixed_base_exp_jvp(
  const typename Geometry::Point& base, const typename Geometry::Tangent& exponent,
  const typename Geometry::Tangent& direction) {
    const auto base_sqrt = native::matrix_sqrt(base);
    const auto base_inverse_sqrt = native::matrix_inverse_sqrt(base);
    const auto chart_exponent = congruence(base_inverse_sqrt, exponent);
    native::SymmetricMatrix<double, fdapde::Dynamic, fdapde::Dynamic> chart_direction(3, 3);
    chart_direction = congruence(base_inverse_sqrt, direction);
    return
      typename Geometry::Tangent(congruence(base_sqrt, native::matrix_exp_frechet(chart_exponent, chart_direction)));
}

}   // namespace

TEST(AffineInvariantSPDGeometry, DefinesAmbientTangentMetricAndVectorOperations) {
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

TEST(AffineInvariantSPDGeometry, ExpLogDistanceAndTransportAreConsistent) {
    const FixedGeometry geometry;
    const auto point = make_point<FixedGeometry>(point_x_coefficients);
    const auto u = make_tangent<FixedGeometry>({0.3, -0.2, 0.4, 0.1, 0.25, -0.15});
    const auto v = make_tangent<FixedGeometry>({-0.1, 0.35, 0.2, -0.25, 0.05, 0.45});
    constexpr double step = 0.35;

    const auto next = geometry.exponential(point, v, step);
    const auto expected_tangent = geometry.linear_combination(point, step, v, 0.0, geometry.zero_tangent(point));
    EXPECT_LT(matrix_difference_norm(geometry.logarithm(point, next), expected_tangent), 1.0e-10);
    EXPECT_NEAR(geometry.distance(point, next), geometry.norm(point, expected_tangent), 1.0e-10);
    const auto target = make_point<FixedGeometry>(point_y_coefficients);
    expect_matrix_relative_near(geometry.exponential(point, geometry.logarithm(point, target)), target, 1.0e-10);

    const auto transported_u = geometry.transport(point, next, u);
    const auto transported_v = geometry.transport(point, next, v);
    EXPECT_NEAR(
      geometry.inner_product(point, u, v), geometry.inner_product(next, transported_u, transported_v), 5.0e-10);
    expect_matrix_relative_near(geometry.transport(next, point, transported_u), u, 1.0e-10);
}

TEST(AffineInvariantSPDGeometry, DifferentialActionsMatchScalarAndDiagonalClosedForms) {
    {
        using ScalarGeometry = fdapde::manifold::AffineInvariantSPDGeometry<double, 1>;
        const ScalarGeometry geometry;
        native::Matrix<double, 1, 1> from_dense;
        native::Matrix<double, 1, 1> to_dense;
        from_dense(0, 0) = 4;
        to_dense(0, 0) = 9;
        const ScalarGeometry::Point from(from_dense, native::checked);
        const ScalarGeometry::Point to(to_dense, native::checked);
        ScalarGeometry::Tangent to_direction;
        ScalarGeometry::Tangent from_dual;
        ScalarGeometry::Tangent base_direction;
        to_direction(0, 0) = 1.8;
        from_dual(0, 0) = -0.7;
        base_direction(0, 0) = 0.4;

        EXPECT_NEAR(geometry.logarithm_target_jvp(from, to, to_direction)(0, 0), 4.0 * 1.8 / 9.0, 1.0e-14);
        EXPECT_NEAR(geometry.logarithm_target_vjp(from, to, from_dual)(0, 0), 9.0 * -0.7 / 4.0, 1.0e-14);
        EXPECT_NEAR(
          geometry.half_squared_distance_hessian_vector(from, to, base_direction)(0, 0), base_direction(0, 0), 1.0e-14);
        EXPECT_NEAR(
          geometry
            .half_squared_distance_hessian_covariant_jvp(
              from, to, base_direction, to_direction, from_dual)(0, 0),
          0.0, 1.0e-14);
    }

    const FixedGeometry geometry;
    const auto identity = make_diagonal_point<FixedGeometry>(1);
    constexpr std::array<double, 3> eigenvalues {1.2, 2.3, 4.7};
    const auto target = make_diagonal_point<FixedGeometry>(eigenvalues[0], eigenvalues[1], eigenvalues[2]);
    const auto to_direction = make_tangent<FixedGeometry>({0.3, -0.2, 0.4, 0.1, 0.25, -0.15});
    const auto from_dual = make_tangent<FixedGeometry>({-0.1, 0.35, 0.2, -0.25, 0.05, 0.45});
    const auto base_direction = make_tangent<FixedGeometry>({0.15, 0.4, -0.2, -0.3, 0.1, 0.35});

    const auto expected_jvp = diagonal_coefficient_action<FixedGeometry>(
      to_direction, eigenvalues, [](double x, double y) { return log_divided_difference(x, y); });
    const auto expected_vjp = diagonal_coefficient_action<FixedGeometry>(
      from_dual, eigenvalues, [](double x, double y) { return x * y * log_divided_difference(x, y); });
    const auto expected_hessian = diagonal_coefficient_action<FixedGeometry>(
      base_direction, eigenvalues, [](double x, double y) { return 0.5 * (x + y) * log_divided_difference(x, y); });

    expect_matrix_relative_near(geometry.logarithm_target_jvp(identity, target, to_direction), expected_jvp, 2.0e-13);
    expect_matrix_relative_near(geometry.logarithm_target_vjp(identity, target, from_dual), expected_vjp, 2.0e-13);
    expect_matrix_relative_near(
      geometry.half_squared_distance_hessian_vector(identity, target, base_direction), expected_hessian, 2.0e-13);

    const auto diagonal_base_direction = make_tangent<FixedGeometry>({0.15, 0, -0.2, 0, 0, 0.35});
    const auto diagonal_target_direction = make_tangent<FixedGeometry>({-0.1, 0, 0.2, 0, 0, 0.45});
    const auto diagonal_action_direction = make_tangent<FixedGeometry>({0.3, 0, 0.4, 0, 0, -0.15});
    expect_matrix_near(
      geometry.half_squared_distance_hessian_covariant_jvp(
        identity, target, diagonal_base_direction, diagonal_target_direction, diagonal_action_direction),
      geometry.zero_tangent(identity), 2.0e-13);
}

TEST(AffineInvariantSPDGeometry, DifferentialActionsHaveTheExactMetricStructure) {
    const FixedGeometry geometry;
    const auto point = make_point<FixedGeometry>(point_x_coefficients);
    const auto target = make_point<FixedGeometry>(point_y_coefficients);
    const auto u = make_tangent<FixedGeometry>({0.3, -0.2, 0.4, 0.1, 0.25, -0.15});
    const auto v = make_tangent<FixedGeometry>({-0.1, 0.35, 0.2, -0.25, 0.05, 0.45});
    const auto z = make_tangent<FixedGeometry>({0.15, 0.4, -0.2, -0.3, 0.1, 0.35});
    const auto w = make_tangent<FixedGeometry>({-0.25, 0.1, 0.3, 0.2, -0.15, 0.05});

    const auto jvp = geometry.logarithm_target_jvp(point, target, v);
    const auto vjp = geometry.logarithm_target_vjp(point, target, z);
    const double lhs = geometry.inner_product(point, jvp, z);
    const double rhs = geometry.inner_product(target, v, vjp);
    EXPECT_NEAR(lhs, rhs, 2.0e-10 * std::max({1.0, std::abs(lhs), std::abs(rhs)}));

    const auto hessian_u = geometry.half_squared_distance_hessian_vector(point, target, u);
    const auto hessian_z = geometry.half_squared_distance_hessian_vector(point, target, z);
    const double self_adjoint_lhs = geometry.inner_product(point, u, hessian_z);
    const double self_adjoint_rhs = geometry.inner_product(point, hessian_u, z);
    EXPECT_NEAR(
      self_adjoint_lhs, self_adjoint_rhs,
      2.0e-10 * std::max({1.0, std::abs(self_adjoint_lhs), std::abs(self_adjoint_rhs)}));
    EXPECT_GE(geometry.inner_product(point, u, hessian_u), geometry.inner_product(point, u, u) * (1.0 - 2.0e-10));

    const auto covariant_jvp =
      geometry.half_squared_distance_hessian_covariant_jvp(point, target, u, v, z);
    const auto covariant_vjp =
      geometry.half_squared_distance_hessian_covariant_vjp(point, target, z, w);
    const double covariant_lhs = geometry.inner_product(point, covariant_jvp, w);
    const double covariant_rhs =
      geometry.inner_product(point, u, covariant_vjp.first) +
      geometry.inner_product(target, v, covariant_vjp.second);
    EXPECT_NEAR(
      covariant_lhs, covariant_rhs,
      3.0e-10 * std::max({1.0, std::abs(covariant_lhs), std::abs(covariant_rhs)}));

    const auto exponent = geometry.logarithm(point, target);
    expect_matrix_relative_near(
      geometry.logarithm_target_jvp(point, target, fixed_base_exp_jvp<FixedGeometry>(point, exponent, z)), z, 2.0e-9);
    expect_matrix_relative_near(
      fixed_base_exp_jvp<FixedGeometry>(point, exponent, geometry.logarithm_target_jvp(point, target, v)), v, 2.0e-9);
}

TEST(AffineInvariantSPDGeometry, HessianCovariantJvpVanishesAtSelfTarget) {
    const FixedGeometry geometry;
    const auto point = make_point<FixedGeometry>(point_x_coefficients);
    const auto base_direction = make_tangent<FixedGeometry>({0.3, -0.2, 0.4, 0.1, 0.25, -0.15});
    const auto target_direction = make_tangent<FixedGeometry>({-0.1, 0.35, 0.2, -0.25, 0.05, 0.45});
    const auto action_direction = make_tangent<FixedGeometry>({0.15, 0.4, -0.2, -0.3, 0.1, 0.35});

    expect_matrix_near(
      geometry.half_squared_distance_hessian_covariant_jvp(
        point, point, base_direction, target_direction, action_direction),
      geometry.zero_tangent(point), 2.0e-12);
    const auto pullback =
      geometry.half_squared_distance_hessian_covariant_vjp(point, point, action_direction, target_direction);
    expect_matrix_near(pullback.first, geometry.zero_tangent(point), 2.0e-12);
    expect_matrix_near(pullback.second, geometry.zero_tangent(point), 2.0e-12);
}

TEST(AffineInvariantSPDGeometry, HessianCovariantJvpIsLinearInVariationAndAction) {
    const FixedGeometry geometry;
    const auto point = make_point<FixedGeometry>(point_x_coefficients);
    const auto target = make_point<FixedGeometry>(point_y_coefficients);
    const auto first_base = make_tangent<FixedGeometry>({0.3, -0.2, 0.4, 0.1, 0.25, -0.15});
    const auto second_base = make_tangent<FixedGeometry>({-0.1, 0.35, 0.2, -0.25, 0.05, 0.45});
    const auto first_target = make_tangent<FixedGeometry>({0.15, 0.4, -0.2, -0.3, 0.1, 0.35});
    const auto second_target = make_tangent<FixedGeometry>({-0.25, 0.1, 0.3, 0.2, -0.15, 0.05});
    const auto first_action = make_tangent<FixedGeometry>({0.2, -0.1, 0.35, -0.25, 0.15, 0.4});
    const auto second_action = make_tangent<FixedGeometry>({-0.3, 0.25, 0.1, 0.2, -0.05, 0.15});
    constexpr double alpha = 0.35;
    constexpr double beta = -0.6;

    const auto first_variation = geometry.half_squared_distance_hessian_covariant_jvp(
      point, target, first_base, first_target, first_action);
    const auto second_variation = geometry.half_squared_distance_hessian_covariant_jvp(
      point, target, second_base, second_target, first_action);
    const auto combined_variation = geometry.half_squared_distance_hessian_covariant_jvp(
      point, target, geometry.linear_combination(point, alpha, first_base, beta, second_base),
      geometry.linear_combination(target, alpha, first_target, beta, second_target), first_action);
    expect_matrix_relative_near(
      combined_variation,
      geometry.linear_combination(point, alpha, first_variation, beta, second_variation), 2.0e-11);

    const auto second_action_result = geometry.half_squared_distance_hessian_covariant_jvp(
      point, target, first_base, first_target, second_action);
    const auto combined_action = geometry.half_squared_distance_hessian_covariant_jvp(
      point, target, first_base, first_target,
      geometry.linear_combination(point, alpha, first_action, beta, second_action));
    expect_matrix_relative_near(
      combined_action,
      geometry.linear_combination(point, alpha, first_variation, beta, second_action_result), 2.0e-11);
}

TEST(AffineInvariantSPDGeometry, DifferentialActionsMatchCenteredGeometricDifferences) {
    const FixedGeometry geometry;
    const auto point = make_point<FixedGeometry>(point_x_coefficients);
    const auto target = make_point<FixedGeometry>(point_y_coefficients);
    const auto base_direction = make_tangent<FixedGeometry>({0.03, -0.02, 0.04, 0.01, 0.025, -0.015});
    const auto target_direction = make_tangent<FixedGeometry>({-0.01, 0.035, 0.02, -0.025, 0.005, 0.045});
    const auto exact_target_jvp = geometry.logarithm_target_jvp(point, target, target_direction);
    const auto exact_hessian = geometry.half_squared_distance_hessian_vector(point, target, base_direction);

    double best_target_error = std::numeric_limits<double>::infinity();
    double best_hessian_error = std::numeric_limits<double>::infinity();
    for (const double step : {1.0e-3, 3.0e-4, 1.0e-4}) {
        const auto target_plus = geometry.exponential(target, target_direction, step);
        const auto target_minus = geometry.exponential(target, target_direction, -step);
        const auto target_difference = geometry.linear_combination(
          point, 0.5 / step, geometry.logarithm(point, target_plus), -0.5 / step,
          geometry.logarithm(point, target_minus));
        const double target_error =
          matrix_difference_norm(target_difference, exact_target_jvp) / (1.0 + matrix_norm(exact_target_jvp));
        EXPECT_LT(target_error, 2.0e-5);
        best_target_error = std::min(best_target_error, target_error);

        const auto point_plus = geometry.exponential(point, base_direction, step);
        const auto point_minus = geometry.exponential(point, base_direction, -step);
        const auto zero_plus = geometry.zero_tangent(point_plus);
        const auto zero_minus = geometry.zero_tangent(point_minus);
        const auto gradient_plus =
          geometry.linear_combination(point_plus, -1, geometry.logarithm(point_plus, target), 0, zero_plus);
        const auto gradient_minus =
          geometry.linear_combination(point_minus, -1, geometry.logarithm(point_minus, target), 0, zero_minus);
        const auto transported_plus = geometry.transport(point_plus, point, gradient_plus);
        const auto transported_minus = geometry.transport(point_minus, point, gradient_minus);
        const auto hessian_difference =
          geometry.linear_combination(point, 0.5 / step, transported_plus, -0.5 / step, transported_minus);
        const double hessian_error =
          matrix_difference_norm(hessian_difference, exact_hessian) / (1.0 + matrix_norm(exact_hessian));
        EXPECT_LT(hessian_error, 2.0e-5);
        best_hessian_error = std::min(best_hessian_error, hessian_error);
    }
    EXPECT_LT(best_target_error, 2.0e-8);
    EXPECT_LT(best_hessian_error, 2.0e-8);
}

TEST(AffineInvariantSPDGeometry, HessianCovariantJvpMatchesCenteredTransportedDifferences) {
    const FixedGeometry geometry;
    const auto point = make_point<FixedGeometry>(point_x_coefficients);
    const auto target = make_point<FixedGeometry>(point_y_coefficients);
    const auto base_direction = make_tangent<FixedGeometry>({0.03, -0.02, 0.04, 0.01, 0.025, -0.015});
    const auto target_direction = make_tangent<FixedGeometry>({-0.01, 0.035, 0.02, -0.025, 0.005, 0.045});
    const auto action_direction = make_tangent<FixedGeometry>({0.02, -0.015, 0.03, -0.01, 0.025, 0.035});
    const auto exact = geometry.half_squared_distance_hessian_covariant_jvp(
      point, target, base_direction, target_direction, action_direction);

    double best_error = std::numeric_limits<double>::infinity();
    for (const double step : {1.0e-3, 3.0e-4, 1.0e-4}) {
        const auto point_plus = geometry.exponential(point, base_direction, step);
        const auto point_minus = geometry.exponential(point, base_direction, -step);
        const auto target_plus = geometry.exponential(target, target_direction, step);
        const auto target_minus = geometry.exponential(target, target_direction, -step);
        const auto action_plus = geometry.transport(point, point_plus, action_direction);
        const auto action_minus = geometry.transport(point, point_minus, action_direction);
        const auto hessian_plus =
          geometry.half_squared_distance_hessian_vector(point_plus, target_plus, action_plus);
        const auto hessian_minus =
          geometry.half_squared_distance_hessian_vector(point_minus, target_minus, action_minus);
        const auto transported_plus = geometry.transport(point_plus, point, hessian_plus);
        const auto transported_minus = geometry.transport(point_minus, point, hessian_minus);
        const auto difference = geometry.linear_combination(
          point, 0.5 / step, transported_plus, -0.5 / step, transported_minus);
        const double error = matrix_difference_norm(difference, exact) / (1.0 + matrix_norm(exact));
        EXPECT_LT(error, 2.0e-6);
        best_error = std::min(best_error, error);
    }
    EXPECT_LT(best_error, 2.0e-8);
}

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
    EXPECT_GT(full_error, 1.0e-8);
    EXPECT_LT(half_error, full_error / 6.0);
    const auto large_step = geometry.retract(point, tangent, 100.0);
    EXPECT_TRUE(std::isfinite(matrix_norm(large_step)));

    const auto identity = make_diagonal_point<FixedGeometry>(1.0);
    const auto huge_tangent = make_tangent<FixedGeometry>({1.0e200, 0.0, 1.0e200, 0.0, 0.0, 1.0e200});
    const auto reciprocal_step = geometry.retract(identity, huge_tangent, 1.0e-200);
    const auto expected = make_diagonal_point<FixedGeometry>(2.5);
    expect_matrix_relative_near(reciprocal_step, expected, 1.0e-14);
    expect_matrix_near(geometry.retract(identity, huge_tangent, 0.0), identity, 0.0);
}

TEST(AffineInvariantSPDGeometry, IsInvariantAndEquivariantUnderGeneralCongruence) {
    const FixedGeometry geometry;
    const auto point = make_point<FixedGeometry>(point_x_coefficients);
    const auto target = make_point<FixedGeometry>(point_y_coefficients);
    const auto u = make_tangent<FixedGeometry>({0.3, -0.2, 0.4, 0.1, 0.25, -0.15});
    const auto v = make_tangent<FixedGeometry>({-0.1, 0.35, 0.2, -0.25, 0.05, 0.45});
    const auto z = make_tangent<FixedGeometry>({0.15, 0.4, -0.2, -0.3, 0.1, 0.35});
    const native::Matrix<double, 3, 3> basis({1.2, -0.2, 0.1, 0.3, 0.9, -0.15, -0.1, 0.25, 1.1});
    const FixedGeometry::Point transformed_point(congruence(basis, point), native::checked);
    const FixedGeometry::Point transformed_target(congruence(basis, target), native::checked);
    const auto transformed_u = congruence(basis, u);
    const auto transformed_v = congruence(basis, v);
    const auto transformed_z = congruence(basis, z);

    const double original_inner = geometry.inner_product(point, u, v);
    EXPECT_LT(
      std::abs(geometry.inner_product(transformed_point, transformed_u, transformed_v) - original_inner),
      1.0e-9 * (1.0 + std::abs(original_inner)));
    EXPECT_NEAR(geometry.distance(transformed_point, transformed_target), geometry.distance(point, target), 1.0e-9);
    expect_matrix_relative_near(
      geometry.exponential(transformed_point, transformed_u, 0.3),
      congruence(basis, geometry.exponential(point, u, 0.3)), 1.0e-9);
    expect_matrix_relative_near(
      geometry.logarithm(transformed_point, transformed_target), congruence(basis, geometry.logarithm(point, target)),
      1.0e-9);
    expect_matrix_relative_near(
      geometry.retract(transformed_point, transformed_u, 0.3), congruence(basis, geometry.retract(point, u, 0.3)),
      1.0e-9);
    expect_matrix_relative_near(
      geometry.transport(transformed_point, transformed_target, transformed_u),
      congruence(basis, geometry.transport(point, target, u)), 1.0e-9);
    expect_matrix_relative_near(
      geometry.logarithm_target_jvp(transformed_point, transformed_target, transformed_v),
      congruence(basis, geometry.logarithm_target_jvp(point, target, v)), 2.0e-9);
    expect_matrix_relative_near(
      geometry.logarithm_target_vjp(transformed_point, transformed_target, transformed_u),
      congruence(basis, geometry.logarithm_target_vjp(point, target, u)), 2.0e-9);
    expect_matrix_relative_near(
      geometry.half_squared_distance_hessian_vector(transformed_point, transformed_target, transformed_u),
      congruence(basis, geometry.half_squared_distance_hessian_vector(point, target, u)), 2.0e-9);
    expect_matrix_relative_near(
      geometry.half_squared_distance_hessian_covariant_jvp(
        transformed_point, transformed_target, transformed_u, transformed_v, transformed_z),
      congruence(
        basis, geometry.half_squared_distance_hessian_covariant_jvp(point, target, u, v, z)),
      3.0e-9);
    const auto pullback = geometry.half_squared_distance_hessian_covariant_vjp(point, target, z, u);
    const auto transformed_pullback = geometry.half_squared_distance_hessian_covariant_vjp(
      transformed_point, transformed_target, transformed_z, transformed_u);
    expect_matrix_relative_near(transformed_pullback.first, congruence(basis, pullback.first), 3.0e-9);
    expect_matrix_relative_near(transformed_pullback.second, congruence(basis, pullback.second), 3.0e-9);
}

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
    expect_matrix_near(
      dynamic_geometry.retract(dynamic_point, dynamic_u, 0.2), fixed_geometry.retract(fixed_point, fixed_u, 0.2),
      1.0e-12);
    expect_matrix_near(
      dynamic_geometry.exponential(dynamic_point, dynamic_u, 0.2),
      fixed_geometry.exponential(fixed_point, fixed_u, 0.2), 1.0e-12);
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
    expect_matrix_near(
      dynamic_geometry.logarithm_target_jvp(dynamic_point, dynamic_target, dynamic_v),
      fixed_geometry.logarithm_target_jvp(fixed_point, fixed_target, fixed_v), 1.0e-11);
    expect_matrix_near(
      dynamic_geometry.logarithm_target_vjp(dynamic_point, dynamic_target, dynamic_u),
      fixed_geometry.logarithm_target_vjp(fixed_point, fixed_target, fixed_u), 1.0e-11);
    expect_matrix_near(
      dynamic_geometry.half_squared_distance_hessian_vector(dynamic_point, dynamic_target, dynamic_u),
      fixed_geometry.half_squared_distance_hessian_vector(fixed_point, fixed_target, fixed_u), 1.0e-11);
    expect_matrix_near(
      dynamic_geometry.half_squared_distance_hessian_covariant_jvp(
        dynamic_point, dynamic_target, dynamic_u, dynamic_v, dynamic_u),
      fixed_geometry.half_squared_distance_hessian_covariant_jvp(
        fixed_point, fixed_target, fixed_u, fixed_v, fixed_u),
      2.0e-11);
    const auto dynamic_pullback = dynamic_geometry.half_squared_distance_hessian_covariant_vjp(
      dynamic_point, dynamic_target, dynamic_u, dynamic_v);
    const auto fixed_pullback = fixed_geometry.half_squared_distance_hessian_covariant_vjp(
      fixed_point, fixed_target, fixed_u, fixed_v);
    expect_matrix_near(dynamic_pullback.first, fixed_pullback.first, 2.0e-11);
    expect_matrix_near(dynamic_pullback.second, fixed_pullback.second, 2.0e-11);
}

TEST(AffineInvariantSPDGeometry, DifferentialActionsHandleRepeatedCloseAndScaledSpectra) {
    const FixedGeometry geometry;
    const auto point = make_point<FixedGeometry>(point_x_coefficients);
    const auto target =
      geometry.exponential(point, geometry.linear_combination(point, std::log(2.5), point.rep(), 0, point.rep()));
    const auto u = make_tangent<FixedGeometry>({0.3, -0.2, 0.4, 0.1, 0.25, -0.15});
    const auto z = make_tangent<FixedGeometry>({-0.1, 0.35, 0.2, -0.25, 0.05, 0.45});
    expect_matrix_relative_near(
      geometry.logarithm_target_jvp(point, target, u), geometry.linear_combination(point, 1.0 / 2.5, u, 0, u), 2.0e-10);
    expect_matrix_relative_near(
      geometry.logarithm_target_vjp(point, target, z), geometry.linear_combination(target, 2.5, z, 0, z), 2.0e-10);
    expect_matrix_relative_near(geometry.half_squared_distance_hessian_vector(point, target, u), u, 2.0e-10);

    const auto identity = make_diagonal_point<FixedGeometry>(1);
    constexpr double gap = 1.0e-12;
    constexpr std::array<double, 3> close_eigenvalues {2, 2 + gap, 5};
    const auto close_target =
      make_diagonal_point<FixedGeometry>(close_eigenvalues[0], close_eigenvalues[1], close_eigenvalues[2]);
    const auto off_diagonal = make_tangent<FixedGeometry>({0, 1, 0, 0, 0, 0});
    const double close_log_difference = std::log1p(gap / 2.0) / gap;
    EXPECT_NEAR(
      geometry.logarithm_target_jvp(identity, close_target, off_diagonal)(1, 0), close_log_difference, 2.0e-12);
    EXPECT_NEAR(
      geometry.logarithm_target_vjp(identity, close_target, off_diagonal)(1, 0),
      2.0 * (2.0 + gap) * close_log_difference, 2.0e-11);
    EXPECT_NEAR(
      geometry.half_squared_distance_hessian_vector(identity, close_target, off_diagonal)(1, 0),
      0.5 * (4.0 + gap) * close_log_difference, 2.0e-12);

    constexpr std::array<double, 3> conditioned_eigenvalues {1, 1.0e-6, 1.0e-12};
    const auto conditioned_target = make_diagonal_point<FixedGeometry>(
      conditioned_eigenvalues[0], conditioned_eigenvalues[1], conditioned_eigenvalues[2]);
    const auto expected_conditioned_jvp = diagonal_coefficient_action<FixedGeometry>(
      u, conditioned_eigenvalues, [](double x, double y) { return log_divided_difference(x, y); });
    const auto expected_conditioned_vjp = diagonal_coefficient_action<FixedGeometry>(
      z, conditioned_eigenvalues, [](double x, double y) { return x * y * log_divided_difference(x, y); });
    expect_matrix_relative_near(
      geometry.logarithm_target_jvp(identity, conditioned_target, u), expected_conditioned_jvp, 2.0e-9);
    expect_matrix_relative_near(
      geometry.logarithm_target_vjp(identity, conditioned_target, z), expected_conditioned_vjp, 2.0e-9);
    const auto small_axis_dual = make_tangent<FixedGeometry>({0, 0, 0, 0, 0, 1.0e12});
    EXPECT_NEAR(geometry.logarithm_target_vjp(identity, conditioned_target, small_axis_dual)(2, 2), 1.0, 2.0e-9);
    const auto conditioned_hessian = geometry.half_squared_distance_hessian_vector(identity, conditioned_target, u);
    EXPECT_TRUE(std::isfinite(matrix_norm(conditioned_hessian)));
    EXPECT_GE(
      geometry.inner_product(identity, u, conditioned_hessian),
      geometry.inner_product(identity, u, u) * (1.0 - 2.0e-10));

    for (const double scale : {1.0e150, 1.0e-150}) {
        const auto scaled_point = make_diagonal_point<FixedGeometry>(scale);
        const auto scaled_target = make_diagonal_point<FixedGeometry>(2 * scale);
        const auto scaled_direction =
          make_tangent<FixedGeometry>({scale, 0.2 * scale, scale, -0.1 * scale, 0.3 * scale, scale});
        expect_matrix_relative_near(
          geometry.logarithm_target_jvp(scaled_point, scaled_target, scaled_direction),
          geometry.linear_combination(scaled_point, 0.5, scaled_direction, 0, scaled_direction), 2.0e-12);
        expect_matrix_relative_near(
          geometry.logarithm_target_vjp(scaled_point, scaled_target, scaled_direction),
          geometry.linear_combination(scaled_target, 2, scaled_direction, 0, scaled_direction), 2.0e-12);
        expect_matrix_relative_near(
          geometry.half_squared_distance_hessian_vector(scaled_point, scaled_target, scaled_direction),
          scaled_direction, 2.0e-12);
    }

    const auto cross_scale_point = make_diagonal_point<FixedGeometry>(1.0e-307);
    const auto unit_target = make_diagonal_point<FixedGeometry>(1);
    const auto finite_target_direction = make_tangent<FixedGeometry>({20, 0, 20, 0, 0, 20});
    const auto small_base_dual = make_tangent<FixedGeometry>({2.0e-306, 0, 2.0e-306, 0, 0, 2.0e-306});
    const auto cross_scale_jvp = geometry.logarithm_target_jvp(cross_scale_point, unit_target, finite_target_direction);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j <= i; ++j) { EXPECT_NEAR(cross_scale_jvp(i, j) / 2.0e-306, i == j ? 1.0 : 0.0, 2.0e-12); }
    }
    expect_matrix_relative_near(
      geometry.logarithm_target_vjp(cross_scale_point, unit_target, small_base_dual), finite_target_direction, 2.0e-12);

    const auto scale_oracle_target = make_diagonal_point<FixedGeometry>(2, 3, 4);
    const auto scale_base_direction = make_tangent<FixedGeometry>({0.3, -0.2, 0.4, 0.1, 0.25, -0.15});
    const auto scale_target_direction = make_tangent<FixedGeometry>({-0.1, 0.35, 0.2, -0.25, 0.05, 0.45});
    const auto scale_action = make_tangent<FixedGeometry>({0.15, 0.4, -0.2, -0.3, 0.1, 0.35});
    const auto scale_output = make_tangent<FixedGeometry>({-0.25, 0.1, 0.3, 0.2, -0.15, 0.05});
    const auto oracle_covariant_jvp = geometry.half_squared_distance_hessian_covariant_jvp(
      identity, scale_oracle_target, scale_base_direction, scale_target_direction, scale_action);
    const auto oracle_covariant_vjp = geometry.half_squared_distance_hessian_covariant_vjp(
      identity, scale_oracle_target, scale_action, scale_output);
    for (const auto [base_scale, target_scale] :
         {std::pair {1.0e-150, 1.0e150}, std::pair {1.0e150, 1.0e-150}}) {
        const auto scaled_base = make_diagonal_point<FixedGeometry>(base_scale);
        const auto scaled_target =
          make_diagonal_point<FixedGeometry>(2 * target_scale, 3 * target_scale, 4 * target_scale);
        const auto scaled_base_direction = geometry.linear_combination(
          scaled_base, base_scale, scale_base_direction, 0, scale_base_direction);
        const auto scaled_target_direction = geometry.linear_combination(
          scaled_target, target_scale, scale_target_direction, 0, scale_target_direction);
        const auto scaled_action =
          geometry.linear_combination(scaled_base, base_scale, scale_action, 0, scale_action);
        const auto scaled_output =
          geometry.linear_combination(scaled_base, base_scale, scale_output, 0, scale_output);
        const auto scaled_covariant_jvp = geometry.half_squared_distance_hessian_covariant_jvp(
          scaled_base, scaled_target, scaled_base_direction, scaled_target_direction, scaled_action);
        expect_matrix_relative_near(
          geometry.linear_combination(
            scaled_base, 1.0 / base_scale, scaled_covariant_jvp, 0, scaled_covariant_jvp),
          oracle_covariant_jvp, 3.0e-9);

        const auto scaled_covariant_vjp = geometry.half_squared_distance_hessian_covariant_vjp(
          scaled_base, scaled_target, scaled_action, scaled_output);
        expect_matrix_relative_near(
          geometry.linear_combination(
            scaled_base, 1.0 / base_scale, scaled_covariant_vjp.first, 0, scaled_covariant_vjp.first),
          oracle_covariant_vjp.first, 3.0e-9);
        expect_matrix_relative_near(
          geometry.linear_combination(
            scaled_target, 1.0 / target_scale, scaled_covariant_vjp.second, 0, scaled_covariant_vjp.second),
          oracle_covariant_vjp.second, 3.0e-9);
    }
}

TEST(AffineInvariantSPDGeometry, DifferentialActionsSupportFixedAndDynamicFloat) {
    using ScalarFloatGeometry = fdapde::manifold::AffineInvariantSPDGeometry<float, 1>;
    using FixedFloatGeometry = fdapde::manifold::AffineInvariantSPDGeometry<float, 3>;
    using DynamicFloatGeometry = fdapde::manifold::AffineInvariantSPDGeometry<float, fdapde::Dynamic>;
    const ScalarFloatGeometry scalar_geometry;
    native::Matrix<float, 1, 1> scalar_from_dense;
    native::Matrix<float, 1, 1> scalar_to_dense;
    scalar_from_dense(0, 0) = 4;
    scalar_to_dense(0, 0) = 9;
    const ScalarFloatGeometry::Point scalar_from(scalar_from_dense, native::checked);
    const ScalarFloatGeometry::Point scalar_to(scalar_to_dense, native::checked);
    ScalarFloatGeometry::Tangent scalar_direction;
    scalar_direction(0, 0) = 1.8f;
    EXPECT_NEAR(
      scalar_geometry.logarithm_target_jvp(scalar_from, scalar_to, scalar_direction)(0, 0), 4.0f * 1.8f / 9.0f,
      2.0e-6f);
    EXPECT_NEAR(
      scalar_geometry.logarithm_target_vjp(scalar_from, scalar_to, scalar_direction)(0, 0), 9.0f * 1.8f / 4.0f,
      2.0e-6f);
    EXPECT_NEAR(
      scalar_geometry.half_squared_distance_hessian_vector(scalar_from, scalar_to, scalar_direction)(0, 0), 1.8f,
      2.0e-6f);
    EXPECT_NEAR(
      scalar_geometry
        .half_squared_distance_hessian_covariant_jvp(
          scalar_from, scalar_to, scalar_direction, scalar_direction, scalar_direction)(0, 0),
      0.0f, 2.0e-6f);
    const auto scalar_pullback = scalar_geometry.half_squared_distance_hessian_covariant_vjp(
      scalar_from, scalar_to, scalar_direction, scalar_direction);
    EXPECT_NEAR(scalar_pullback.first(0, 0), 0.0f, 2.0e-6f);
    EXPECT_NEAR(scalar_pullback.second(0, 0), 0.0f, 2.0e-6f);

    const FixedFloatGeometry fixed_geometry;
    const DynamicFloatGeometry dynamic_geometry(3);
    const auto fixed_point = make_point<FixedFloatGeometry>(point_x_coefficients);
    const auto fixed_target = make_point<FixedFloatGeometry>(point_y_coefficients);
    const auto fixed_u = make_tangent<FixedFloatGeometry>({0.3, -0.2, 0.4, 0.1, 0.25, -0.15});
    const auto dynamic_point = make_point<DynamicFloatGeometry>(point_x_coefficients);
    const auto dynamic_target = make_point<DynamicFloatGeometry>(point_y_coefficients);
    const auto dynamic_u = make_tangent<DynamicFloatGeometry>({0.3, -0.2, 0.4, 0.1, 0.25, -0.15});

    expect_matrix_relative_near(
      dynamic_geometry.logarithm_target_jvp(dynamic_point, dynamic_target, dynamic_u),
      fixed_geometry.logarithm_target_jvp(fixed_point, fixed_target, fixed_u), 2.0e-5);
    expect_matrix_relative_near(
      dynamic_geometry.logarithm_target_vjp(dynamic_point, dynamic_target, dynamic_u),
      fixed_geometry.logarithm_target_vjp(fixed_point, fixed_target, fixed_u), 2.0e-5);
    expect_matrix_relative_near(
      dynamic_geometry.half_squared_distance_hessian_vector(dynamic_point, dynamic_target, dynamic_u),
      fixed_geometry.half_squared_distance_hessian_vector(fixed_point, fixed_target, fixed_u), 2.0e-5);
    expect_matrix_relative_near(
      dynamic_geometry.half_squared_distance_hessian_covariant_jvp(
        dynamic_point, dynamic_target, dynamic_u, dynamic_u, dynamic_u),
      fixed_geometry.half_squared_distance_hessian_covariant_jvp(
        fixed_point, fixed_target, fixed_u, fixed_u, fixed_u),
      3.0e-5);
    const auto dynamic_pullback = dynamic_geometry.half_squared_distance_hessian_covariant_vjp(
      dynamic_point, dynamic_target, dynamic_u, dynamic_u);
    const auto fixed_pullback = fixed_geometry.half_squared_distance_hessian_covariant_vjp(
      fixed_point, fixed_target, fixed_u, fixed_u);
    expect_matrix_relative_near(dynamic_pullback.first, fixed_pullback.first, 3.0e-5);
    expect_matrix_relative_near(dynamic_pullback.second, fixed_pullback.second, 3.0e-5);
}

TEST(AffineInvariantSPDGeometry, RejectsUnsupportedOrdersAndMismatchedShapes) {
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
    const DynamicGeometry::Point wrong_point(wrong_dense, native::checked);
    DynamicGeometry::Tangent wrong_tangent(2, 2);
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j <= i; ++j) { wrong_tangent(i, j) = i == j ? 1.0 : 0.0; }
    }

    EXPECT_THROW(geometry.distance(point, wrong_point), std::invalid_argument);
    EXPECT_THROW(geometry.norm(point, wrong_tangent), std::invalid_argument);
    EXPECT_THROW(geometry.project(point, wrong_tangent), std::invalid_argument);
    EXPECT_THROW(geometry.logarithm_target_jvp(point, point, wrong_tangent), std::invalid_argument);
    EXPECT_THROW(
      geometry.logarithm_target_vjp(point, wrong_point, geometry.zero_tangent(point)), std::invalid_argument);
    EXPECT_THROW(
      geometry.half_squared_distance_hessian_vector(wrong_point, point, geometry.zero_tangent(point)),
      std::invalid_argument);
    const auto zero = geometry.zero_tangent(point);
    EXPECT_THROW(
      geometry.half_squared_distance_hessian_covariant_jvp(wrong_point, point, zero, zero, zero),
      std::invalid_argument);
    EXPECT_THROW(
      geometry.half_squared_distance_hessian_covariant_jvp(point, wrong_point, zero, zero, zero),
      std::invalid_argument);
    EXPECT_THROW(
      geometry.half_squared_distance_hessian_covariant_jvp(point, point, wrong_tangent, zero, zero),
      std::invalid_argument);
    EXPECT_THROW(
      geometry.half_squared_distance_hessian_covariant_jvp(point, point, zero, wrong_tangent, zero),
      std::invalid_argument);
    EXPECT_THROW(
      geometry.half_squared_distance_hessian_covariant_jvp(point, point, zero, zero, wrong_tangent),
      std::invalid_argument);
    EXPECT_THROW(
      geometry.half_squared_distance_hessian_covariant_vjp(wrong_point, point, zero, zero),
      std::invalid_argument);
    EXPECT_THROW(
      geometry.half_squared_distance_hessian_covariant_vjp(point, wrong_point, zero, zero),
      std::invalid_argument);
    EXPECT_THROW(
      geometry.half_squared_distance_hessian_covariant_vjp(point, point, wrong_tangent, zero),
      std::invalid_argument);
    EXPECT_THROW(
      geometry.half_squared_distance_hessian_covariant_vjp(point, point, zero, wrong_tangent),
      std::invalid_argument);

    auto nonfinite = geometry.zero_tangent(point);
    nonfinite(0, 0) = std::numeric_limits<double>::infinity();
    EXPECT_THROW(geometry.logarithm_target_jvp(point, point, nonfinite), std::invalid_argument);
    EXPECT_THROW(geometry.logarithm_target_vjp(point, point, nonfinite), std::invalid_argument);
    EXPECT_THROW(geometry.half_squared_distance_hessian_vector(point, point, nonfinite), std::invalid_argument);
    EXPECT_THROW(
      geometry.half_squared_distance_hessian_covariant_jvp(point, point, nonfinite, zero, zero),
      std::invalid_argument);
    EXPECT_THROW(
      geometry.half_squared_distance_hessian_covariant_jvp(point, point, zero, nonfinite, zero),
      std::invalid_argument);
    EXPECT_THROW(
      geometry.half_squared_distance_hessian_covariant_jvp(point, point, zero, zero, nonfinite),
      std::invalid_argument);
    EXPECT_THROW(
      geometry.half_squared_distance_hessian_covariant_vjp(point, point, nonfinite, zero),
      std::invalid_argument);
    EXPECT_THROW(
      geometry.half_squared_distance_hessian_covariant_vjp(point, point, zero, nonfinite),
      std::invalid_argument);

    native::Matrix<double, 3, 3> indefinite_dense;
    indefinite_dense.set_zero();
    indefinite_dense(0, 0) = 1;
    indefinite_dense(1, 1) = 1;
    indefinite_dense(2, 2) = -1;
    const DynamicGeometry::Point unchecked_indefinite(indefinite_dense, native::unchecked);
    EXPECT_THROW(
      geometry.logarithm_target_jvp(point, unchecked_indefinite, geometry.zero_tangent(point)), std::domain_error);
    EXPECT_THROW(
      geometry.half_squared_distance_hessian_covariant_jvp(
        point, unchecked_indefinite, zero, zero, zero),
      std::domain_error);
    EXPECT_THROW(
      geometry.half_squared_distance_hessian_covariant_vjp(
        point, unchecked_indefinite, zero, zero),
      std::domain_error);

    native::Matrix<double, 3, 3> nonfinite_point_dense;
    nonfinite_point_dense.set_zero();
    nonfinite_point_dense(0, 0) = 1;
    nonfinite_point_dense(1, 1) = 1;
    nonfinite_point_dense(2, 2) = std::numeric_limits<double>::infinity();
    const DynamicGeometry::Point unchecked_nonfinite(nonfinite_point_dense, native::unchecked);
    EXPECT_THROW(
      geometry.logarithm_target_jvp(point, unchecked_nonfinite, geometry.zero_tangent(point)), std::invalid_argument);
    EXPECT_THROW(
      geometry.half_squared_distance_hessian_covariant_jvp(
        point, unchecked_nonfinite, zero, zero, zero),
      std::invalid_argument);
    EXPECT_THROW(
      geometry.half_squared_distance_hessian_covariant_vjp(
        point, unchecked_nonfinite, zero, zero),
      std::invalid_argument);

    native::Matrix<double, 3, 3> below_threshold_dense;
    below_threshold_dense.set_zero();
    below_threshold_dense(0, 0) = 1;
    below_threshold_dense(1, 1) = 1;
    below_threshold_dense(2, 2) = 1.0e-16;
    const DynamicGeometry::Point unchecked_below_threshold(below_threshold_dense, native::unchecked);
    EXPECT_THROW(
      geometry.logarithm_target_jvp(point, unchecked_below_threshold, geometry.zero_tangent(point)), std::domain_error);
}

TEST(AffineInvariantSPDGeometry, NormAndDistanceRemainRepresentableAcrossScales) {
    const FixedGeometry geometry;
    const auto identity = make_diagonal_point<FixedGeometry>(1.0);
    for (const double scale : {1.0e200, 1.0e-200}) {
        const auto tangent = make_tangent<FixedGeometry>({scale, 0.0, scale, 0.0, 0.0, scale});
        const double expected = std::hypot(std::hypot(scale, scale), scale);
        EXPECT_NEAR(geometry.norm(identity, tangent) / expected, 1.0, 1.0e-14);
    }

    for (const double scale : {1.0e150, 1.0e-150}) {
        const auto point = make_diagonal_point<FixedGeometry>(scale);
        const auto tangent = make_tangent<FixedGeometry>({scale, 0.0, scale, 0.0, 0.0, scale});
        EXPECT_NEAR(geometry.norm(point, tangent), std::sqrt(3.0), 1.0e-12);
    }

    const auto small = make_diagonal_point<FixedGeometry>(1.0e-150);
    const auto large = make_diagonal_point<FixedGeometry>(1.0e150);
    const double expected_distance = std::sqrt(3.0) * std::log(1.0e300);
    EXPECT_NEAR(geometry.distance(small, large), expected_distance, 1.0e-10);
}
