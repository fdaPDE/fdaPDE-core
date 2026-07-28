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

#include <fdaPDE/geometric_finite_elements.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <span>
#include <vector>

namespace {

namespace native = fdapde::linalg;
namespace manifold = fdapde::manifold;

using FixedGeometry = manifold::AffineInvariantSPDGeometry<double, 3>;
using DynamicGeometry = manifold::AffineInvariantSPDGeometry<double, fdapde::Dynamic>;
using LogGeometry = manifold::LogEuclideanSPDGeometry<double, 3>;

constexpr std::array<double, 9> first_coefficients {4.0, 0.6, 0.2, 0.6, 2.5, -0.3, 0.2, -0.3, 1.7};
constexpr std::array<double, 9> second_coefficients {1.8, -0.25, 0.15, -0.25, 3.3, 0.4, 0.15, 0.4, 2.2};
constexpr std::array<double, 9> third_coefficients {2.6, 0.35, -0.2, 0.35, 1.4, 0.1, -0.2, 0.1, 4.1};
constexpr std::array<double, 9> first_diagonal {1.5, 0, 0, 0, 3, 0, 0, 0, 6};
constexpr std::array<double, 9> second_diagonal {4, 0, 0, 0, 2, 0, 0, 0, 1.25};
constexpr std::array<double, 9> third_diagonal {2.25, 0, 0, 0, 5, 0, 0, 0, 3.5};

template <typename Geometry> typename Geometry::Point make_point(const std::array<double, 9>& coefficients) {
    native::Matrix<double, 3, 3> dense;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) { dense(i, j) = coefficients[static_cast<std::size_t>(3 * i + j)]; }
    }
    return typename Geometry::Point(dense, native::checked);
}

template <typename Geometry> typename Geometry::Point make_scalar_point(double value) {
    native::Matrix<double, 1, 1> dense;
    dense(0, 0) = value;
    return typename Geometry::Point(dense, native::checked);
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

template <typename Lhs, typename Rhs> void expect_same_matrix(const Lhs& lhs, const Rhs& rhs) {
    ASSERT_EQ(lhs.rows(), rhs.rows());
    ASSERT_EQ(lhs.cols(), rhs.cols());
    for (int i = 0; i < lhs.rows(); ++i) {
        for (int j = 0; j < lhs.cols(); ++j) { EXPECT_DOUBLE_EQ(lhs(i, j), rhs(i, j)); }
    }
}

manifold::WeightedKarcherMeanOptions accurate_options() {
    manifold::WeightedKarcherMeanOptions options;
    options.solver.max_iterations = 200;
    options.solver.gradient_tolerance = 1.0e-8;
    return options;
}

template <typename Geometry>
auto p1_value(
  const Geometry& geometry, const std::vector<typename Geometry::Point>& nodes, std::span<const double> weights,
  const manifold::WeightedKarcherMeanOptions& options = accurate_options()) {
    return fdapde::gfe::p1_geodesic_value(geometry, std::span<const typename Geometry::Point>(nodes), weights, options);
}

template <typename Point> double log_determinant(const Point& point) {
    const auto point_log = native::matrix_log(point);
    double result = 0;
    for (int i = 0; i < point_log.rows(); ++i) { result += static_cast<double>(point_log(i, i)); }
    return result;
}

template <typename Geometry>
double stationarity_norm(
  const Geometry& geometry, const typename Geometry::Point& point, std::span<const typename Geometry::Point> nodes,
  std::span<const double> weights) {
    auto residual = geometry.zero_tangent(point);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        residual = geometry.linear_combination(point, 1, residual, weights[i], geometry.logarithm(point, nodes[i]));
    }
    return geometry.norm(point, residual);
}

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

template <typename Geometry> void expect_two_node_geodesic_values(const Geometry& geometry) {
    using Point = typename Geometry::Point;
    const std::vector<Point> nodes {
      make_point<Geometry>(first_coefficients), make_point<Geometry>(second_coefficients)};
    const std::array<double, 7> parameters {
      {0, 0.1, 0.25, 0.5, 0.75, 0.9, 1}
    };
    const auto direction = geometry.logarithm(nodes[0], nodes[1]);

    for (double parameter : parameters) {
        const std::array<double, 2> weights {
          {1 - parameter, parameter}
        };
        const auto result = p1_value(geometry, nodes, std::span<const double>(weights));

        EXPECT_TRUE(result.converged());
        EXPECT_EQ(result.uniqueness, manifold::BarycenterUniqueness::globally_unique);
        EXPECT_LE(result.stationarity_norm, 1.1e-8);
        EXPECT_GT(result.value.determinant(), 0);
        EXPECT_TRUE(std::isfinite(result.value.determinant()));
        if (parameter == 0) {
            EXPECT_EQ(result.stop_reason, manifold::BarycenterStopReason::closed_form);
            expect_same_matrix(result.value, nodes[0]);
        } else if (parameter == 1) {
            EXPECT_EQ(result.stop_reason, manifold::BarycenterStopReason::closed_form);
            expect_same_matrix(result.value, nodes[1]);
        } else {
            const Point expected = geometry.exponential(nodes[0], direction, parameter);
            EXPECT_EQ(result.stop_reason, manifold::BarycenterStopReason::stationarity_tolerance);
            EXPECT_LT(geometry.distance(result.value, expected), 2.0e-8);
        }
    }
}

}   // namespace

TEST(AffineInvariantP1Value, MatchesTheTwoNodeGeodesicIncludingExactEndpoints) {
    expect_two_node_geodesic_values(FixedGeometry {});
    expect_two_node_geodesic_values(DynamicGeometry {3});

    const FixedGeometry geometry;
    const std::vector<FixedGeometry::Point> nodes {
      make_point<FixedGeometry>(first_coefficients), make_point<FixedGeometry>(second_coefficients)};
    const std::array<double, 2> vertex_weights {
      {1, 0}
    };
    manifold::WeightedKarcherMeanOptions unused_invalid_options;
    unused_invalid_options.solver.max_iterations = 0;
    EXPECT_NO_THROW(p1_value(geometry, nodes, std::span<const double>(vertex_weights), unused_invalid_options));
}

TEST(AffineInvariantP1Value, MatchesLogEuclideanValuesForCommutingAndScalarData) {
    const FixedGeometry affine_geometry;
    const LogGeometry log_geometry;
    const std::vector<FixedGeometry::Point> affine_nodes {
      make_point<FixedGeometry>(first_diagonal), make_point<FixedGeometry>(second_diagonal),
      make_point<FixedGeometry>(third_diagonal)};
    const std::vector<LogGeometry::Point> log_nodes {
      make_point<LogGeometry>(first_diagonal), make_point<LogGeometry>(second_diagonal),
      make_point<LogGeometry>(third_diagonal)};
    const std::array<double, 3> weights {
      {0.2, 0.3, 0.5}
    };
    const auto affine = p1_value(affine_geometry, affine_nodes, std::span<const double>(weights));
    const auto logarithmic = fdapde::gfe::p1_geodesic_value(
      log_geometry, std::span<const LogGeometry::Point>(log_nodes), std::span<const double>(weights));
    const auto direct_mean = manifold::weighted_karcher_mean(
      affine_geometry, std::span<const FixedGeometry::Point>(affine_nodes), std::span<const double>(weights),
      accurate_options());
    const auto explicit_initial_mean = manifold::weighted_karcher_mean(
      affine_geometry, std::span<const FixedGeometry::Point>(affine_nodes), std::span<const double>(weights),
      affine_nodes[0], accurate_options());

    EXPECT_TRUE(affine.converged());
    EXPECT_TRUE(direct_mean.converged());
    EXPECT_TRUE(explicit_initial_mean.converged());
    EXPECT_EQ(direct_mean.uniqueness, manifold::BarycenterUniqueness::globally_unique);
    EXPECT_EQ(explicit_initial_mean.uniqueness, manifold::BarycenterUniqueness::globally_unique);
    EXPECT_LT(affine_geometry.distance(affine.value, logarithmic.value), 1.0e-10);
    EXPECT_LT(affine_geometry.distance(affine.value, direct_mean.point), 1.0e-12);
    EXPECT_LT(affine_geometry.distance(affine.value, explicit_initial_mean.point), 2.0e-8);

    using ScalarGeometry = manifold::AffineInvariantSPDGeometry<double, 1>;
    const ScalarGeometry scalar_geometry;
    const std::vector<ScalarGeometry::Point> scalar_nodes {
      make_scalar_point<ScalarGeometry>(4), make_scalar_point<ScalarGeometry>(9),
      make_scalar_point<ScalarGeometry>(25)};
    const auto scalar = p1_value(scalar_geometry, scalar_nodes, std::span<const double>(weights));
    const double expected = std::exp(0.2 * std::log(4.0) + 0.3 * std::log(9.0) + 0.5 * std::log(25.0));

    EXPECT_TRUE(scalar.converged());
    EXPECT_EQ(scalar.uniqueness, manifold::BarycenterUniqueness::globally_unique);
    EXPECT_NEAR(scalar.value(0, 0), expected, 1.0e-10);
}

TEST(AffineInvariantP1Value, HandlesConstantAndRepeatedSamples) {
    const FixedGeometry geometry;
    const auto first = make_point<FixedGeometry>(first_coefficients);
    const auto second = make_point<FixedGeometry>(second_coefficients);
    const std::array<double, 3> weights {
      {0.25, 0.5, 0.25}
    };
    const std::vector<FixedGeometry::Point> constant_nodes {first, first, first};
    const auto constant = p1_value(geometry, constant_nodes, std::span<const double>(weights));

    EXPECT_TRUE(constant.converged());
    EXPECT_EQ(constant.uniqueness, manifold::BarycenterUniqueness::globally_unique);
    EXPECT_LT(geometry.distance(constant.value, first), 1.0e-10);

    const std::vector<FixedGeometry::Point> repeated_nodes {first, second, first};
    const auto repeated = p1_value(geometry, repeated_nodes, std::span<const double>(weights));
    const auto midpoint = geometry.exponential(first, geometry.logarithm(first, second), 0.5);

    EXPECT_TRUE(repeated.converged());
    EXPECT_EQ(repeated.uniqueness, manifold::BarycenterUniqueness::globally_unique);
    EXPECT_LT(geometry.distance(repeated.value, midpoint), 2.0e-8);
}

TEST(AffineInvariantP1Value, IsCongruenceEquivariantAndPreservesTheLogDeterminantIdentity) {
    const FixedGeometry geometry;
    const std::vector<FixedGeometry::Point> nodes {
      make_point<FixedGeometry>(first_coefficients), make_point<FixedGeometry>(second_coefficients),
      make_point<FixedGeometry>(third_coefficients)};
    const std::array<double, 3> weights {
      {0.2, 0.3, 0.5}
    };
    native::Matrix<double, 3, 3> transform;
    transform(0, 0) = 1.2;
    transform(0, 1) = 0.2;
    transform(0, 2) = 0;
    transform(1, 0) = -0.1;
    transform(1, 1) = 0.9;
    transform(1, 2) = 0.15;
    transform(2, 0) = 0.05;
    transform(2, 1) = 0;
    transform(2, 2) = 1.1;

    std::vector<FixedGeometry::Point> transformed_nodes;
    transformed_nodes.reserve(nodes.size());
    for (const auto& node : nodes) { transformed_nodes.emplace_back(congruence(transform, node), native::checked); }

    const auto result = p1_value(geometry, nodes, std::span<const double>(weights));
    const auto transformed = p1_value(geometry, transformed_nodes, std::span<const double>(weights));
    const FixedGeometry::Point expected_transformed(congruence(transform, result.value), native::checked);

    EXPECT_TRUE(result.converged()) << static_cast<int>(result.stop_reason)
                                    << " stationarity=" << result.stationarity_norm;
    EXPECT_TRUE(transformed.converged()) << static_cast<int>(transformed.stop_reason)
                                         << " stationarity=" << transformed.stationarity_norm;
    EXPECT_LT(geometry.distance(transformed.value, expected_transformed), 5.0e-8);
    const double recomputed_stationarity = stationarity_norm(
      geometry, result.value, std::span<const FixedGeometry::Point>(nodes),
      std::span<const double>(result.normalized_weights));
    EXPECT_NEAR(recomputed_stationarity, result.stationarity_norm, 1.0e-12);
    EXPECT_LE(recomputed_stationarity, 1.1e-8);
    double expected_log_determinant = 0;
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        expected_log_determinant += result.normalized_weights[i] * log_determinant(nodes[i]);
    }
    EXPECT_NEAR(
      log_determinant(result.value), expected_log_determinant, std::sqrt(3.0) * recomputed_stationarity + 1.0e-12);
    EXPECT_GT(result.value.determinant(), 0);
    EXPECT_TRUE(std::isfinite(result.value.determinant()));
    const auto spectrum = result.value.evd();
    ASSERT_TRUE(spectrum.computed());
    for (double eigenvalue : spectrum.eigenvalues()) {
        EXPECT_TRUE(std::isfinite(eigenvalue));
        EXPECT_GT(eigenvalue, 0);
    }
}

TEST(AffineInvariantP1Value, IsDeterministicPermutationInvariantAndFixedDynamicConsistent) {
    const FixedGeometry fixed_geometry;
    const DynamicGeometry dynamic_geometry(3);
    const std::vector<FixedGeometry::Point> fixed_nodes {
      make_point<FixedGeometry>(first_coefficients), make_point<FixedGeometry>(second_coefficients),
      make_point<FixedGeometry>(third_coefficients)};
    const std::vector<FixedGeometry::Point> permuted_nodes {fixed_nodes[2], fixed_nodes[0], fixed_nodes[1]};
    const std::array<double, 3> weights {
      {0.2, 0.3, 0.5}
    };
    const std::array<double, 3> permuted_weights {
      {0.5, 0.2, 0.3}
    };

    const auto first = p1_value(fixed_geometry, fixed_nodes, std::span<const double>(weights));
    const auto second = p1_value(fixed_geometry, fixed_nodes, std::span<const double>(weights));
    const auto permuted = p1_value(fixed_geometry, permuted_nodes, std::span<const double>(permuted_weights));
    const auto defaults = fdapde::gfe::p1_geodesic_value(
      fixed_geometry, std::span<const FixedGeometry::Point>(fixed_nodes), std::span<const double>(weights));
    const auto direct_first = manifold::weighted_karcher_mean(
      fixed_geometry, std::span<const FixedGeometry::Point>(fixed_nodes), std::span<const double>(weights),
      accurate_options());
    const auto direct_second = manifold::weighted_karcher_mean(
      fixed_geometry, std::span<const FixedGeometry::Point>(fixed_nodes), std::span<const double>(weights),
      accurate_options());

    expect_same_matrix(first.value, second.value);
    EXPECT_DOUBLE_EQ(first.stationarity_norm, second.stationarity_norm);
    EXPECT_EQ(first.stop_reason, second.stop_reason);
    EXPECT_EQ(first.uniqueness, second.uniqueness);
    EXPECT_EQ(first.normalized_weights, (std::vector<double> {0.2, 0.3, 0.5}));
    EXPECT_TRUE(defaults.converged());
    EXPECT_EQ(defaults.uniqueness, manifold::BarycenterUniqueness::globally_unique);
    EXPECT_LE(defaults.stationarity_norm, 1.1e-6);
    EXPECT_TRUE(permuted.converged());
    EXPECT_EQ(permuted.uniqueness, manifold::BarycenterUniqueness::globally_unique);
    EXPECT_LT(fixed_geometry.distance(first.value, permuted.value), 5.0e-8);
    expect_same_matrix(direct_first.point, direct_second.point);
    EXPECT_DOUBLE_EQ(direct_first.cost, direct_second.cost);
    EXPECT_DOUBLE_EQ(direct_first.stationarity_norm, direct_second.stationarity_norm);
    EXPECT_EQ(direct_first.iterations, direct_second.iterations);
    EXPECT_EQ(direct_first.cost_evaluations, direct_second.cost_evaluations);
    EXPECT_EQ(direct_first.gradient_evaluations, direct_second.gradient_evaluations);
    EXPECT_EQ(direct_first.rejected_trials, direct_second.rejected_trials);
    EXPECT_EQ(direct_first.stop_reason, direct_second.stop_reason);
    EXPECT_EQ(direct_first.line_search_status, direct_second.line_search_status);

    const std::vector<DynamicGeometry::Point> dynamic_nodes {
      make_point<DynamicGeometry>(first_coefficients), make_point<DynamicGeometry>(second_coefficients),
      make_point<DynamicGeometry>(third_coefficients)};
    const auto dynamic = p1_value(dynamic_geometry, dynamic_nodes, std::span<const double>(weights));
    EXPECT_TRUE(dynamic.converged()) << static_cast<int>(dynamic.stop_reason)
                                     << " stationarity=" << dynamic.stationarity_norm;
    EXPECT_EQ(dynamic.uniqueness, manifold::BarycenterUniqueness::globally_unique);
    EXPECT_LT(matrix_difference_norm(first.value, dynamic.value), 1.0e-10);
}

TEST(AffineInvariantP1Value, SurfacesNonconvergenceAndUsesTheExactLogEuclideanInitializer) {
    const FixedGeometry geometry;
    const LogGeometry log_geometry;
    const std::vector<FixedGeometry::Point> nodes {
      make_point<FixedGeometry>(first_coefficients), make_point<FixedGeometry>(second_coefficients),
      make_point<FixedGeometry>(third_coefficients)};
    const std::vector<LogGeometry::Point> log_nodes {
      make_point<LogGeometry>(first_coefficients), make_point<LogGeometry>(second_coefficients),
      make_point<LogGeometry>(third_coefficients)};
    const std::array<double, 3> weights {
      {0.2, 0.3, 0.5}
    };
    manifold::WeightedKarcherMeanOptions forced_failure;
    forced_failure.solver.gradient_tolerance = 0;
    forced_failure.solver.line_search.initial_step = 1.0e6;
    forced_failure.solver.line_search.max_trials = 1;

    const auto result = p1_value(geometry, nodes, std::span<const double>(weights), forced_failure);
    const auto initializer = fdapde::gfe::p1_geodesic_value(
      log_geometry, std::span<const LogGeometry::Point>(log_nodes), std::span<const double>(weights));

    EXPECT_FALSE(result.converged());
    EXPECT_EQ(result.stop_reason, manifold::BarycenterStopReason::line_search_failed);
    EXPECT_EQ(result.uniqueness, manifold::BarycenterUniqueness::globally_unique);
    EXPECT_TRUE(std::isfinite(result.stationarity_norm));
    EXPECT_GT(result.stationarity_norm, 0);
    EXPECT_LT(matrix_difference_norm(result.value, initializer.value), 1.0e-14);
}

TEST(AffineInvariantP1Value, ValidatesOptionsAndSkipsZeroWeightShapeMismatches) {
    const FixedGeometry fixed_geometry;
    const std::vector<FixedGeometry::Point> fixed_nodes {
      make_point<FixedGeometry>(first_coefficients), make_point<FixedGeometry>(second_coefficients)};
    const std::array<double, 2> equal_weights {
      {0.5, 0.5}
    };
    manifold::WeightedKarcherMeanOptions invalid_options;
    invalid_options.solver.max_iterations = 0;
    EXPECT_THROW(
      p1_value(fixed_geometry, fixed_nodes, std::span<const double>(equal_weights), invalid_options),
      std::invalid_argument);

    native::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> two_by_two(2, 2);
    two_by_two.set_zero();
    two_by_two(0, 0) = 1;
    two_by_two(1, 1) = 1;
    const DynamicGeometry::Point wrong_shape(two_by_two, native::checked);
    const std::vector<DynamicGeometry::Point> dynamic_nodes {
      wrong_shape, make_point<DynamicGeometry>(first_coefficients), make_point<DynamicGeometry>(second_coefficients)};
    const std::array<double, 3> zero_wrong_shape_weight {
      {0, 0.5, 0.5}
    };
    const std::array<double, 3> positive_wrong_shape_weight {
      {0.1, 0.45, 0.45}
    };
    const DynamicGeometry dynamic_geometry(3);

    EXPECT_NO_THROW(p1_value(dynamic_geometry, dynamic_nodes, std::span<const double>(zero_wrong_shape_weight)));
    EXPECT_THROW(
      p1_value(dynamic_geometry, dynamic_nodes, std::span<const double>(positive_wrong_shape_weight)),
      std::invalid_argument);
}
