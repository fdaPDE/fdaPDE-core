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
#include <limits>
#include <span>
#include <type_traits>
#include <vector>

namespace {

struct EuclideanGeodesic {
    using Point = double;
    using Tangent = double;

    mutable std::size_t distance_calls = 0;
    mutable std::size_t logarithm_calls = 0;
    mutable std::size_t norm_calls = 0;
    mutable std::size_t retraction_calls = 0;

    std::size_t dimension() const { return 1; }
    double inner_product(const Point&, const Tangent& u, const Tangent& v) const { return u * v; }
    double norm(const Point&, const Tangent& tangent) const {
        ++norm_calls;
        return std::abs(tangent);
    }
    Tangent project(const Point&, const Tangent& tangent) const { return tangent; }
    Tangent zero_tangent(const Point&) const { return 0; }
    Tangent linear_combination(const Point&, double alpha, const Tangent& u, double beta, const Tangent& v) const {
        return alpha * u + beta * v;
    }
    Point retract(const Point& point, const Tangent& tangent, double step) const {
        ++retraction_calls;
        return point + step * tangent;
    }
    Point exponential(const Point& point, const Tangent& tangent, double step) const { return point + step * tangent; }
    Tangent logarithm(const Point& from, const Point& to) const {
        ++logarithm_calls;
        return to - from;
    }
    double distance(const Point& from, const Point& to) const {
        ++distance_calls;
        return std::abs(to - from);
    }
};

struct NonFiniteDistanceGeodesic : EuclideanGeodesic {
    double distance(const Point&, const Point&) const { return std::numeric_limits<double>::infinity(); }
};

struct NonFiniteLogarithmGeodesic : EuclideanGeodesic {
    Tangent logarithm(const Point&, const Point&) const { return std::numeric_limits<double>::infinity(); }
};

template <std::size_t NodeCount>
auto p1_value(
  const EuclideanGeodesic& geometry, const std::array<double, NodeCount>& nodes,
  const std::array<double, NodeCount>& weights, double initial,
  const fdapde::manifold::WeightedKarcherMeanOptions& options = {}) {
    return fdapde::gfe::p1_geodesic_value(
      geometry, std::span<const double>(nodes), std::span<const double>(weights), initial, options);
}

namespace native = fdapde::linalg;

constexpr std::array<double, 9> first_coefficients {4.0, 0.6, 0.2, 0.6, 2.5, -0.3, 0.2, -0.3, 1.7};
constexpr std::array<double, 9> second_coefficients {1.8, -0.25, 0.15, -0.25, 3.3, 0.4, 0.15, 0.4, 2.2};
constexpr std::array<double, 9> third_coefficients {2.6, 0.35, -0.2, 0.35, 1.4, 0.1, -0.2, 0.1, 4.1};

template <typename Geometry> typename Geometry::Point make_point(const std::array<double, 9>& coefficients) {
    native::Matrix<double, 3, 3> dense;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) { dense(i, j) = coefficients[static_cast<std::size_t>(3 * i + j)]; }
    }
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

template <typename Geometry> typename Geometry::Point expect_exact_log_euclidean_p1(const Geometry& geometry) {
    using Point = typename Geometry::Point;
    const std::vector<Point> nodes {
      make_point<Geometry>(first_coefficients), make_point<Geometry>(second_coefficients),
      make_point<Geometry>(third_coefficients)};
    const std::array<double, 3> weights {
      {0.25, 0.5, 0.25}
    };
    const auto result =
      fdapde::gfe::p1_geodesic_value(geometry, std::span<const Point>(nodes), std::span<const double>(weights));
    const auto oracle = fdapde::manifold::weighted_karcher_mean(
      geometry, std::span<const Point>(nodes), std::span<const double>(weights));

    static_assert(native::is_spd_matrix_v<decltype(result.value)>);
    EXPECT_TRUE(result.converged());
    EXPECT_EQ(result.stop_reason, fdapde::manifold::BarycenterStopReason::closed_form);
    EXPECT_EQ(result.uniqueness, fdapde::manifold::BarycenterUniqueness::globally_unique);
    EXPECT_EQ(result.normalized_weights, (std::vector<double> {0.25, 0.5, 0.25}));
    EXPECT_LT(result.stationarity_norm, 1.0e-9);
    EXPECT_LT(geometry.distance(result.value, oracle.point), 1.0e-10);
    return result.value;
}

}   // namespace

TEST(P1GeodesicValue, MatchesEuclideanP1OnEdgesAndInteriors) {
    const std::array<double, 3> nodes {
      {0, 2, 8}
    };
    const std::array<double, 3> edge_weights {
      {0.75, 0.25, 0}
    };
    const std::array<double, 3> interior_weights {
      {0.25, 0.5, 0.25}
    };

    const auto edge = p1_value(EuclideanGeodesic {}, nodes, edge_weights, 0);
    const auto interior = p1_value(EuclideanGeodesic {}, nodes, interior_weights, 0);
    const std::array<double, 4> tetrahedron_nodes {
      {0, 2, 8, 14}
    };
    const std::array<double, 4> tetrahedron_weights {
      {0.125, 0.25, 0.375, 0.25}
    };
    const auto tetrahedron = p1_value(EuclideanGeodesic {}, tetrahedron_nodes, tetrahedron_weights, 0);
    const std::array<double, 3> constant_nodes {
      {4, 4, 4}
    };
    const auto constant = p1_value(EuclideanGeodesic {}, constant_nodes, interior_weights, 0);

    EXPECT_TRUE(edge.converged());
    EXPECT_TRUE(interior.converged());
    EXPECT_TRUE(tetrahedron.converged());
    EXPECT_TRUE(constant.converged());
    EXPECT_DOUBLE_EQ(edge.value, 0.5);
    EXPECT_DOUBLE_EQ(interior.value, 3);
    EXPECT_DOUBLE_EQ(tetrahedron.value, 7);
    EXPECT_DOUBLE_EQ(constant.value, 4);
    EXPECT_EQ(edge.uniqueness, fdapde::manifold::BarycenterUniqueness::not_certified);
    EXPECT_EQ(interior.uniqueness, fdapde::manifold::BarycenterUniqueness::not_certified);
}

TEST(P1GeodesicValue, ShortCircuitsAnExactVertexAfterValidatingTheSelectedPoint) {
    EuclideanGeodesic geometry;
    const std::array<double, 3> nodes {
      {std::numeric_limits<double>::infinity(), 7, std::numeric_limits<double>::quiet_NaN()}
    };
    const std::array<double, 3> weights {
      {0, 1, 0}
    };

    const auto result = p1_value(geometry, nodes, weights, -100);

    EXPECT_DOUBLE_EQ(result.value, 7);
    EXPECT_EQ(result.normalized_weights, (std::vector<double> {0, 1, 0}));
    EXPECT_DOUBLE_EQ(result.stationarity_norm, 0);
    EXPECT_EQ(result.stop_reason, fdapde::manifold::BarycenterStopReason::closed_form);
    EXPECT_EQ(result.uniqueness, fdapde::manifold::BarycenterUniqueness::globally_unique);
    EXPECT_EQ(geometry.distance_calls, 1);
    EXPECT_EQ(geometry.logarithm_calls, 0);
    EXPECT_EQ(geometry.norm_calls, 0);
    EXPECT_EQ(geometry.retraction_calls, 0);

    EuclideanGeodesic near_vertex_geometry;
    const double tolerance = fdapde::gfe::p1_weight_sum_tolerance(nodes.size());
    const std::array<double, 3> near_vertex_weights {
      {0, 1 + 0.5 * tolerance, 0}
    };
    const auto near_vertex = p1_value(near_vertex_geometry, nodes, near_vertex_weights, -100);
    EXPECT_DOUBLE_EQ(near_vertex.value, 7);
    EXPECT_EQ(near_vertex.normalized_weights, (std::vector<double> {0, 1, 0}));
    EXPECT_EQ(near_vertex.stop_reason, fdapde::manifold::BarycenterStopReason::closed_form);
    EXPECT_EQ(near_vertex_geometry.retraction_calls, 0);

    fdapde::manifold::WeightedKarcherMeanOptions invalid_unused_options;
    invalid_unused_options.solver.max_iterations = 0;
    EXPECT_NO_THROW(p1_value(near_vertex_geometry, nodes, weights, -100, invalid_unused_options));

    const auto invalid_vertex = fdapde::gfe::p1_geodesic_value(
      NonFiniteDistanceGeodesic {}, std::span<const double>(nodes), std::span<const double>(weights), -100);
    EXPECT_FALSE(invalid_vertex.converged());
    EXPECT_EQ(invalid_vertex.stop_reason, fdapde::manifold::BarycenterStopReason::non_finite_cost);
    EXPECT_EQ(invalid_vertex.uniqueness, fdapde::manifold::BarycenterUniqueness::not_certified);
}

TEST(P1GeodesicValue, RejectsInvalidBarycentricDataAndCanonicalizesAcceptedWeights) {
    const EuclideanGeodesic geometry;
    const std::array<double, 2> nodes {
      {0, 1}
    };
    const std::array<double, 1> one_weight {{1}};
    const std::array<double, 2> nan_weights {
      {std::numeric_limits<double>::quiet_NaN(), 1}
    };
    const std::array<double, 2> infinite_weights {
      {std::numeric_limits<double>::infinity(), 0}
    };
    const std::array<double, 2> negative_weights {
      {-std::numeric_limits<double>::epsilon(), 1}
    };
    const std::array<double, 2> zero_weights {
      {0, 0}
    };
    const std::array<double, 2> low_sum {
      {0.25, 0.25}
    };
    const std::array<double, 2> high_sum {
      {0.5, 0.5000000001}
    };
    const std::array<double, 2> scalable_but_not_barycentric {
      {2, 2}
    };

    EXPECT_THROW(
      fdapde::gfe::p1_geodesic_value(geometry, std::span<const double> {}, std::span<const double> {}, 0),
      std::invalid_argument);
    EXPECT_THROW(
      fdapde::gfe::p1_geodesic_value(geometry, std::span<const double>(nodes), std::span<const double>(one_weight), 0),
      std::invalid_argument);
    EXPECT_THROW(p1_value(geometry, nodes, nan_weights, 0), std::invalid_argument);
    EXPECT_THROW(p1_value(geometry, nodes, infinite_weights, 0), std::invalid_argument);
    EXPECT_THROW(p1_value(geometry, nodes, negative_weights, 0), std::invalid_argument);
    EXPECT_THROW(p1_value(geometry, nodes, zero_weights, 0), std::invalid_argument);
    EXPECT_THROW(p1_value(geometry, nodes, low_sum, 0), std::invalid_argument);
    EXPECT_THROW(p1_value(geometry, nodes, high_sum, 0), std::invalid_argument);
    EXPECT_THROW(p1_value(geometry, nodes, scalable_but_not_barycentric, 0), std::invalid_argument);

    const double tolerance = fdapde::gfe::p1_weight_sum_tolerance(nodes.size());
    const std::array<double, 2> within_tolerance {
      {0.5, 0.5 + 0.5 * tolerance}
    };
    const auto canonicalized = p1_value(geometry, nodes, within_tolerance, 0);
    const double input_sum = within_tolerance[0] + within_tolerance[1];
    EXPECT_NEAR(canonicalized.normalized_weights[0], within_tolerance[0] / input_sum, 1.0e-16);
    EXPECT_NEAR(canonicalized.normalized_weights[1], within_tolerance[1] / input_sum, 1.0e-16);
    EXPECT_NEAR(canonicalized.value, within_tolerance[1] / input_sum, 1.0e-14);
}

TEST(P1GeodesicValue, PreservesGenericMeanOutcomesAndCertification) {
    const EuclideanGeodesic geometry;
    const std::array<double, 2> nodes {
      {0, 2}
    };
    const std::array<double, 2> weights {
      {0.5, 0.5}
    };

    fdapde::manifold::WeightedKarcherMeanOptions limited;
    limited.solver.max_iterations = 1;
    limited.solver.gradient_tolerance = 0;
    limited.solver.line_search.initial_step = 0.5;
    const auto maximum_iterations = p1_value(geometry, nodes, weights, 0, limited);
    EXPECT_FALSE(maximum_iterations.converged());
    EXPECT_EQ(maximum_iterations.stop_reason, fdapde::manifold::BarycenterStopReason::max_iterations);
    EXPECT_EQ(maximum_iterations.uniqueness, fdapde::manifold::BarycenterUniqueness::not_certified);
    EXPECT_DOUBLE_EQ(maximum_iterations.value, 0.5);
    EXPECT_DOUBLE_EQ(maximum_iterations.stationarity_norm, 0.5);

    fdapde::manifold::WeightedKarcherMeanOptions rejected;
    rejected.solver.line_search.initial_step = 4;
    rejected.solver.line_search.max_trials = 1;
    const auto line_search_failure = p1_value(geometry, nodes, weights, 0, rejected);
    EXPECT_FALSE(line_search_failure.converged());
    EXPECT_EQ(line_search_failure.stop_reason, fdapde::manifold::BarycenterStopReason::line_search_failed);
    EXPECT_DOUBLE_EQ(line_search_failure.value, 0);
    EXPECT_DOUBLE_EQ(line_search_failure.stationarity_norm, 1);

    const auto non_finite_cost = fdapde::gfe::p1_geodesic_value(
      NonFiniteDistanceGeodesic {}, std::span<const double>(nodes), std::span<const double>(weights), 0);
    EXPECT_FALSE(non_finite_cost.converged());
    EXPECT_EQ(non_finite_cost.stop_reason, fdapde::manifold::BarycenterStopReason::non_finite_cost);

    const auto non_finite_gradient = fdapde::gfe::p1_geodesic_value(
      NonFiniteLogarithmGeodesic {}, std::span<const double>(nodes), std::span<const double>(weights), 0);
    EXPECT_FALSE(non_finite_gradient.converged());
    EXPECT_EQ(non_finite_gradient.stop_reason, fdapde::manifold::BarycenterStopReason::non_finite_gradient);
}

TEST(P1GeodesicValue, FixedAndDynamicLogEuclideanValuesMatchTheExactMean) {
    const auto fixed = expect_exact_log_euclidean_p1(fdapde::manifold::LogEuclideanSPDGeometry<double, 3> {});
    const auto dynamic =
      expect_exact_log_euclidean_p1(fdapde::manifold::LogEuclideanSPDGeometry<double, fdapde::Dynamic> {3});

    EXPECT_LT(matrix_difference_norm(fixed, dynamic), 1.0e-10);
}

TEST(P1GeodesicValue, IsDeterministicAndPermutationInvariant) {
    const EuclideanGeodesic geometry;
    const std::array<double, 3> nodes {
      {0, 2, 8}
    };
    const std::array<double, 3> weights {
      {0.25, 0.5, 0.25}
    };
    const std::array<double, 3> permuted_nodes {
      {8, 0, 2}
    };
    const std::array<double, 3> permuted_weights {
      {0.25, 0.25, 0.5}
    };

    const auto first = p1_value(geometry, nodes, weights, 0);
    const auto second = p1_value(geometry, nodes, weights, 0);
    const auto permuted = p1_value(geometry, permuted_nodes, permuted_weights, 0);

    EXPECT_DOUBLE_EQ(first.value, second.value);
    EXPECT_DOUBLE_EQ(first.stationarity_norm, second.stationarity_norm);
    EXPECT_EQ(first.normalized_weights, second.normalized_weights);
    EXPECT_EQ(first.stop_reason, second.stop_reason);
    EXPECT_EQ(first.uniqueness, second.uniqueness);
    EXPECT_NEAR(first.value, permuted.value, 1.0e-14);
    EXPECT_NEAR(first.stationarity_norm, permuted.stationarity_norm, 1.0e-14);
    EXPECT_EQ(first.stop_reason, permuted.stop_reason);
    EXPECT_EQ(first.uniqueness, permuted.uniqueness);
}
