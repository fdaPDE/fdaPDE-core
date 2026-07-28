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
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

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

template <typename Geometry> typename Geometry::Point identity_point() {
    native::Matrix<double, 3, 3> dense;
    dense.set_zero();
    for (int i = 0; i < 3; ++i) { dense(i, i) = 1.0; }
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

template <typename Geometry> class FrechetMeanProblem {
   public:
    using Point = typename Geometry::Point;
    using Tangent = typename Geometry::Tangent;
    struct Workspace { };

    FrechetMeanProblem(const Geometry& geometry, std::vector<Point> samples) :
        geometry_(geometry), samples_(std::move(samples)) { }

    double cost(const Point& point, Workspace&) const {
        double result = 0;
        for (const auto& sample : samples_) {
            const double distance = geometry_.distance(point, sample);
            result += 0.5 * distance * distance;
        }
        return result;
    }

    Tangent gradient(const Point& point, Workspace&) const {
        auto result = geometry_.zero_tangent(point);
        for (const auto& sample : samples_) {
            result = geometry_.linear_combination(point, 1.0, result, -1.0, geometry_.logarithm(point, sample));
        }
        return result;
    }
   private:
    const Geometry& geometry_;
    std::vector<Point> samples_;
};

template <typename Geometry>
typename Geometry::Point
closed_form_log_mean(const Geometry& geometry, const std::vector<typename Geometry::Point>& samples) {
    auto mean_log = geometry.zero_tangent(samples.front());
    for (const auto& sample : samples) {
        const auto sample_log = native::matrix_log(sample);
        for (int i = 0; i < sample_log.rows(); ++i) {
            for (int j = 0; j <= i; ++j) {
                mean_log(i, j) = static_cast<double>(mean_log(i, j)) +
                                 static_cast<double>(sample_log(i, j)) / static_cast<double>(samples.size());
            }
        }
    }
    return native::matrix_exp(mean_log);
}

template <typename Geometry>
typename Geometry::Point closed_form_weighted_log_mean(
  const Geometry& geometry, const std::vector<typename Geometry::Point>& samples,
  std::span<const double> normalized_weights) {
    auto mean_log = geometry.zero_tangent(samples.front());
    for (std::size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
        const auto sample_log = native::matrix_log(samples[sample_index]);
        for (int i = 0; i < sample_log.rows(); ++i) {
            for (int j = 0; j <= i; ++j) {
                mean_log(i, j) = static_cast<double>(mean_log(i, j)) +
                                 normalized_weights[sample_index] * static_cast<double>(sample_log(i, j));
            }
        }
    }
    return native::matrix_exp(mean_log);
}

template <typename Geometry>
double closed_form_weighted_log_cost(
  const std::vector<typename Geometry::Point>& samples, std::span<const double> normalized_weights,
  const typename Geometry::Point& point) {
    const auto point_log = native::matrix_log(point);
    double result = 0;
    for (std::size_t sample_index = 0; sample_index < samples.size(); ++sample_index) {
        const auto sample_log = native::matrix_log(samples[sample_index]);
        double squared_distance = 0;
        for (int i = 0; i < sample_log.rows(); ++i) {
            for (int j = 0; j < sample_log.cols(); ++j) {
                const double difference = static_cast<double>(sample_log(i, j)) - static_cast<double>(point_log(i, j));
                squared_distance += difference * difference;
            }
        }
        result += 0.5 * normalized_weights[sample_index] * squared_distance;
    }
    return result;
}

template <typename Geometry>
auto exact_weighted_log_mean(
  const Geometry& geometry, const std::vector<typename Geometry::Point>& samples, std::span<const double> weights) {
    return fdapde::manifold::weighted_karcher_mean(
      geometry, std::span<const typename Geometry::Point>(samples), weights);
}

template <typename Geometry> typename Geometry::Point expect_exact_weighted_log_mean(const Geometry& geometry) {
    const std::vector<typename Geometry::Point> samples {
      make_point<Geometry>(first_coefficients), make_point<Geometry>(second_coefficients),
      make_point<Geometry>(third_coefficients)};
    const std::array<double, 3> weights {
      {1, 2, 5}
    };
    const std::array<double, 3> normalized_weights {
      {0.125, 0.25, 0.625}
    };
    const auto expected = closed_form_weighted_log_mean(geometry, samples, std::span<const double>(normalized_weights));
    const auto result = exact_weighted_log_mean(geometry, samples, std::span<const double>(weights));
    const std::vector<typename Geometry::Point> permuted_samples {samples[2], samples[0], samples[1]};
    const std::array<double, 3> permuted_weights {
      {5, 1, 2}
    };
    const auto permuted =
      exact_weighted_log_mean(geometry, permuted_samples, std::span<const double>(permuted_weights));

    static_assert(native::is_spd_matrix_v<decltype(result.point)>);
    EXPECT_TRUE(result.converged());
    EXPECT_EQ(result.stop_reason, fdapde::manifold::BarycenterStopReason::closed_form);
    EXPECT_EQ(result.line_search_status, fdapde::manifold::ArmijoStatus::not_run);
    EXPECT_EQ(result.uniqueness, fdapde::manifold::BarycenterUniqueness::globally_unique);
    EXPECT_EQ(result.normalized_weights, (std::vector<double> {0.125, 0.25, 0.625}));
    EXPECT_NEAR(
      result.cost,
      closed_form_weighted_log_cost<Geometry>(samples, std::span<const double>(normalized_weights), result.point),
      1.0e-12);
    EXPECT_LT(result.stationarity_norm, 1.0e-9);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(result.cost_evaluations, 1);
    EXPECT_EQ(result.gradient_evaluations, 1);
    EXPECT_EQ(result.rejected_trials, 0);
    EXPECT_LT(geometry.distance(result.point, expected), 1.0e-10);
    EXPECT_LT(geometry.distance(result.point, permuted.point), 1.0e-10);
    return result.point;
}

template <typename Geometry> void expect_exact_mean_handles_degenerate_weights(const Geometry& geometry) {
    using Point = typename Geometry::Point;
    const Point first = make_point<Geometry>(first_coefficients);
    const Point second = make_point<Geometry>(second_coefficients);
    const Point third = make_point<Geometry>(third_coefficients);
    const std::vector<Point> samples {first, second, third};
    const std::vector<Point> reduced_samples {second, third};
    const std::array<double, 3> scaled_weights {
      {0, 20, 60}
    };
    const std::array<double, 2> reduced_weights {
      {1, 3}
    };
    const auto scaled = exact_weighted_log_mean(geometry, samples, std::span<const double>(scaled_weights));
    const auto reduced = exact_weighted_log_mean(geometry, reduced_samples, std::span<const double>(reduced_weights));

    EXPECT_LT(geometry.distance(scaled.point, reduced.point), 1.0e-10);
    EXPECT_EQ(scaled.normalized_weights, (std::vector<double> {0, 0.25, 0.75}));

    const std::vector<Point> repeated {second, second, second};
    const std::array<double, 3> repeated_weights {
      {1, 7, 3}
    };
    const auto constant = exact_weighted_log_mean(geometry, repeated, std::span<const double>(repeated_weights));
    EXPECT_LT(geometry.distance(constant.point, second), 1.0e-10);

    const std::array<double, 3> one_hot_weights {
      {0, 1, 0}
    };
    const auto one_hot = exact_weighted_log_mean(geometry, samples, std::span<const double>(one_hot_weights));
    EXPECT_LT(geometry.distance(one_hot.point, second), 1.0e-10);
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

template <typename Geometry> void expect_log_mean_matches_closed_form(const Geometry& geometry) {
    using Problem = FrechetMeanProblem<Geometry>;
    std::vector<typename Geometry::Point> samples {
      make_point<Geometry>(first_coefficients), make_point<Geometry>(second_coefficients),
      make_point<Geometry>(third_coefficients)};
    Problem problem(geometry, samples);
    const auto expected = closed_form_log_mean(geometry, samples);
    const auto initial = identity_point<Geometry>();
    typename Problem::Workspace workspace;
    const double initial_cost = problem.cost(initial, workspace);

    fdapde::manifold::SteepestDescentOptions options;
    options.max_iterations = 20;
    options.gradient_tolerance = 1.0e-9;
    options.line_search.initial_step = 1.0 / static_cast<double>(samples.size());
    const fdapde::manifold::RiemannianSteepestDescent optimizer(options);
    const auto result = optimizer.optimize(problem, geometry, initial);

    EXPECT_TRUE(result.converged());
    EXPECT_EQ(result.stop_reason, fdapde::manifold::SteepestDescentStopReason::gradient_tolerance);
    EXPECT_EQ(result.line_search_status, fdapde::manifold::ArmijoStatus::accepted);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_EQ(result.cost_evaluations, 2);
    EXPECT_EQ(result.gradient_evaluations, 2);
    EXPECT_EQ(result.rejected_trials, 0);
    EXPECT_LT(result.gradient_norm, 1.0e-9);
    EXPECT_LT(result.cost, initial_cost);
    EXPECT_LT(geometry.distance(result.point, expected), 1.0e-9);
}

template <typename Geometry> typename Geometry::Point optimize_two_point_affine_mean(const Geometry& geometry) {
    using Problem = FrechetMeanProblem<Geometry>;
    const std::vector<typename Geometry::Point> samples {
      make_point<Geometry>(first_coefficients), make_point<Geometry>(second_coefficients)};
    Problem problem(geometry, samples);
    const auto expected = geometry.exponential(samples[0], geometry.logarithm(samples[0], samples[1]), 0.5);
    const auto initial = identity_point<Geometry>();

    fdapde::manifold::SteepestDescentOptions options;
    options.max_iterations = 100;
    options.gradient_tolerance = 1.0e-8;
    options.line_search.initial_step = 0.5;
    const fdapde::manifold::RiemannianSteepestDescent optimizer(options);
    const auto result = optimizer.optimize(problem, geometry, initial);

    EXPECT_TRUE(result.converged());
    EXPECT_EQ(result.stop_reason, fdapde::manifold::SteepestDescentStopReason::gradient_tolerance);
    EXPECT_EQ(result.line_search_status, fdapde::manifold::ArmijoStatus::accepted);
    EXPECT_LT(result.gradient_norm, 1.0e-8);
    EXPECT_LT(geometry.distance(result.point, expected), 1.0e-7);
    return result.point;
}

using FieldComponent = fdapde::manifold::LogEuclideanSPDGeometry<double, 3>;
using FieldGeometry = fdapde::manifold::PowerGeometry<FieldComponent>;

class FieldMeanProblem {
   public:
    struct Workspace { };

    FieldMeanProblem(const FieldComponent& component, std::vector<FieldGeometry::Point> samples) :
        component_(component), samples_(std::move(samples)) { }

    double cost(const FieldGeometry::Point& point, Workspace&) const {
        double result = 0;
        for (const auto& sample : samples_) {
            for (std::size_t i = 0; i < point.size(); ++i) {
                const double distance = component_.distance(point[i], sample[i]);
                result += 0.5 * distance * distance;
            }
        }
        return result;
    }

    FieldGeometry::Tangent gradient(const FieldGeometry::Point& point, Workspace&) const {
        FieldGeometry::Tangent result;
        result.reserve(point.size());
        for (std::size_t i = 0; i < point.size(); ++i) {
            auto component_gradient = component_.zero_tangent(point[i]);
            for (const auto& sample : samples_) {
                component_gradient = component_.linear_combination(
                  point[i], 1.0, component_gradient, -1.0, component_.logarithm(point[i], sample[i]));
            }
            result.push_back(std::move(component_gradient));
        }
        return result;
    }
   private:
    const FieldComponent& component_;
    std::vector<FieldGeometry::Point> samples_;
};

}   // namespace

TEST(LogEuclideanSPDMean, FixedAndDynamicThreeByThreeMeansMatchTheClosedFormOracle) {
    expect_log_mean_matches_closed_form(fdapde::manifold::LogEuclideanSPDGeometry<double, 3> {});
    expect_log_mean_matches_closed_form(fdapde::manifold::LogEuclideanSPDGeometry<double, fdapde::Dynamic> {3});
}

TEST(LogEuclideanSPDMean, ExactWeightedMeanMatchesClosedFormForFixedAndDynamic) {
    const auto fixed = expect_exact_weighted_log_mean(fdapde::manifold::LogEuclideanSPDGeometry<double, 3> {});
    const auto dynamic =
      expect_exact_weighted_log_mean(fdapde::manifold::LogEuclideanSPDGeometry<double, fdapde::Dynamic> {3});

    EXPECT_LT(matrix_difference_norm(fixed, dynamic), 1.0e-10);
}

TEST(LogEuclideanSPDMean, ExactWeightedMeanReducesToTheScalarGeometricMean) {
    using Geometry = fdapde::manifold::LogEuclideanSPDGeometry<double, 1>;
    const Geometry geometry;
    const auto scalar_point = [](double value) {
        native::Matrix<double, 1, 1> dense;
        dense(0, 0) = value;
        return Geometry::Point(dense, native::checked);
    };
    const std::vector<Geometry::Point> samples {scalar_point(4), scalar_point(9)};
    const std::array<double, 2> weights {
      {1, 3}
    };
    const auto result = exact_weighted_log_mean(geometry, samples, std::span<const double>(weights));

    EXPECT_NEAR(result.point(0, 0), std::exp(0.25 * std::log(4.0) + 0.75 * std::log(9.0)), 1.0e-12);

    const std::vector<Geometry::Point> reciprocal_scales {scalar_point(1.0e-300), scalar_point(1.0e300)};
    const std::array<double, 2> equal_weights {
      {1, 1}
    };
    const auto reciprocal =
      exact_weighted_log_mean(geometry, reciprocal_scales, std::span<const double>(equal_weights));
    EXPECT_NEAR(reciprocal.point(0, 0), 1, 1.0e-12);

    const std::vector<Geometry::Point> tiny_atom_samples {scalar_point(1.0e300), scalar_point(1)};
    const std::array<double, 2> tiny_atom_weights {
      {std::numeric_limits<double>::denorm_min(), 1}
    };
    const auto tiny_atom =
      exact_weighted_log_mean(geometry, tiny_atom_samples, std::span<const double>(tiny_atom_weights));
    EXPECT_TRUE(std::isfinite(tiny_atom.cost));
    EXPECT_GT(tiny_atom.cost, 0);
}

TEST(LogEuclideanSPDMean, ExactWeightedMeanHandlesRepeatedAndCloseSpectra) {
    using Geometry = fdapde::manifold::LogEuclideanSPDGeometry<double, 3>;
    const Geometry geometry;
    native::Matrix<double, 3, 3> first_dense;
    first_dense.set_zero();
    first_dense(0, 0) = 2;
    first_dense(1, 1) = 2;
    first_dense(2, 2) = 5;
    native::Matrix<double, 3, 3> second_dense(first_dense);
    second_dense(1, 1) += 1.0e-12;
    const std::vector<Geometry::Point> samples {
      Geometry::Point(first_dense, native::checked), Geometry::Point(second_dense, native::checked)};
    const std::array<double, 2> weights {
      {1, 1}
    };
    const std::array<double, 2> normalized_weights {
      {0.5, 0.5}
    };
    const auto expected = closed_form_weighted_log_mean(geometry, samples, std::span<const double>(normalized_weights));
    const auto result = exact_weighted_log_mean(geometry, samples, std::span<const double>(weights));

    EXPECT_LT(geometry.distance(result.point, expected), 1.0e-12);
    EXPECT_LT(result.stationarity_norm, 1.0e-12);
}

TEST(LogEuclideanSPDMean, ExactWeightedMeanIsInvariantToDegenerateWeightRepresentations) {
    expect_exact_mean_handles_degenerate_weights(fdapde::manifold::LogEuclideanSPDGeometry<double, 3> {});
    expect_exact_mean_handles_degenerate_weights(
      fdapde::manifold::LogEuclideanSPDGeometry<double, fdapde::Dynamic> {3});
}

TEST(LogEuclideanSPDMean, ExactWeightedMeanPreservesTheLogDeterminantIdentity) {
    using Geometry = fdapde::manifold::LogEuclideanSPDGeometry<double, 3>;
    const Geometry geometry;
    const std::vector<Geometry::Point> samples {
      make_point<Geometry>(first_coefficients), make_point<Geometry>(second_coefficients),
      make_point<Geometry>(third_coefficients)};
    const std::array<double, 3> weights {
      {1, 2, 5}
    };
    const auto result = exact_weighted_log_mean(geometry, samples, std::span<const double>(weights));

    double expected_log_determinant = 0;
    for (std::size_t i = 0; i < samples.size(); ++i) {
        const auto sample_log = native::matrix_log(samples[i]);
        for (int j = 0; j < sample_log.rows(); ++j) {
            expected_log_determinant += result.normalized_weights[i] * static_cast<double>(sample_log(j, j));
        }
    }
    const auto result_log = native::matrix_log(result.point);
    double result_log_determinant = 0;
    for (int i = 0; i < result_log.rows(); ++i) { result_log_determinant += static_cast<double>(result_log(i, i)); }
    EXPECT_GT(result.point.determinant(), 0);
    EXPECT_TRUE(std::isfinite(result.point.determinant()));
    EXPECT_NEAR(result_log_determinant, expected_log_determinant, 1.0e-10);
}

TEST(LogEuclideanSPDMean, ExactWeightedMeanIsOrthogonallyEquivariant) {
    using Geometry = fdapde::manifold::LogEuclideanSPDGeometry<double, 3>;
    const Geometry geometry;
    const std::vector<Geometry::Point> samples {
      make_point<Geometry>(first_coefficients), make_point<Geometry>(second_coefficients),
      make_point<Geometry>(third_coefficients)};
    const std::array<double, 3> weights {
      {1, 2, 5}
    };
    native::Matrix<double, 3, 3> rotation;
    rotation.set_zero();
    rotation(0, 0) = 0.8;
    rotation(0, 1) = -0.6;
    rotation(1, 0) = 0.6;
    rotation(1, 1) = 0.8;
    rotation(2, 2) = 1;

    std::vector<Geometry::Point> transformed_samples;
    transformed_samples.reserve(samples.size());
    for (const auto& sample : samples) {
        transformed_samples.emplace_back(congruence(rotation, sample), native::checked);
    }
    const auto mean = exact_weighted_log_mean(geometry, samples, std::span<const double>(weights));
    const auto transformed = exact_weighted_log_mean(geometry, transformed_samples, std::span<const double>(weights));
    const Geometry::Point expected(congruence(rotation, mean.point), native::checked);

    EXPECT_LT(geometry.distance(transformed.point, expected), 1.0e-10);
}

TEST(LogEuclideanSPDMean, ExactWeightedMeanRejectsInvalidPositiveWeightSamplesAndSkipsZeroWeightSamples) {
    using FixedGeometry = fdapde::manifold::LogEuclideanSPDGeometry<double, 3>;
    using DynamicGeometry = fdapde::manifold::LogEuclideanSPDGeometry<double, fdapde::Dynamic>;
    native::Matrix<double, 3, 3> indefinite;
    indefinite.set_zero();
    indefinite(0, 0) = -1;
    indefinite(1, 1) = 1;
    indefinite(2, 2) = 1;
    const FixedGeometry::Point unchecked_indefinite(indefinite, native::unchecked);
    const std::vector<FixedGeometry::Point> invalid_samples {
      unchecked_indefinite, make_point<FixedGeometry>(first_coefficients)};
    const std::array<double, 2> positive_weights {
      {1, 1}
    };
    const std::array<double, 2> zero_first_weight {
      {0, 1}
    };
    const std::array<double, 1> too_few_weights {{1}};
    const std::array<double, 2> negative_weights {
      {-1, 1}
    };

    EXPECT_THROW(
      fdapde::manifold::weighted_karcher_mean(
        FixedGeometry {}, std::span<const FixedGeometry::Point> {}, std::span<const double> {}),
      std::invalid_argument);
    EXPECT_THROW(
      exact_weighted_log_mean(FixedGeometry {}, invalid_samples, std::span<const double>(too_few_weights)),
      std::invalid_argument);
    EXPECT_THROW(
      exact_weighted_log_mean(FixedGeometry {}, invalid_samples, std::span<const double>(negative_weights)),
      std::invalid_argument);
    EXPECT_THROW(
      exact_weighted_log_mean(FixedGeometry {}, invalid_samples, std::span<const double>(positive_weights)),
      std::domain_error);
    EXPECT_NO_THROW(
      exact_weighted_log_mean(FixedGeometry {}, invalid_samples, std::span<const double>(zero_first_weight)));

    native::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> two_by_two(2, 2);
    two_by_two.set_zero();
    two_by_two(0, 0) = 1;
    two_by_two(1, 1) = 1;
    const DynamicGeometry::Point wrong_shape(two_by_two, native::checked);
    const std::vector<DynamicGeometry::Point> wrong_shape_samples {
      wrong_shape, make_point<DynamicGeometry>(first_coefficients)};
    const DynamicGeometry dynamic_geometry(3);
    EXPECT_THROW(
      exact_weighted_log_mean(dynamic_geometry, wrong_shape_samples, std::span<const double>(positive_weights)),
      std::invalid_argument);
    EXPECT_NO_THROW(
      exact_weighted_log_mean(dynamic_geometry, wrong_shape_samples, std::span<const double>(zero_first_weight)));
}

TEST(LogEuclideanSPDMean, PowerGeometryConvergesComponentwiseToClosedFormMeans) {
    const FieldComponent component;
    const FieldGeometry geometry(2, component);
    const FieldGeometry::Point first {
      make_point<FieldComponent>(first_coefficients), make_point<FieldComponent>(second_coefficients)};
    const FieldGeometry::Point second {
      make_point<FieldComponent>(third_coefficients), make_point<FieldComponent>(first_coefficients)};
    const std::vector<FieldGeometry::Point> samples {first, second};
    FieldMeanProblem problem(component, samples);
    const auto identity = identity_point<FieldComponent>();
    const FieldGeometry::Point initial(geometry.factor_count(), identity);
    const std::array<FieldComponent::Point, 2> expected {
      closed_form_log_mean(component, {first[0], second[0]}), closed_form_log_mean(component, {first[1], second[1]})};

    fdapde::manifold::SteepestDescentOptions options;
    options.max_iterations = 10;
    options.gradient_tolerance = 1.0e-9;
    options.line_search.initial_step = 0.5;
    const fdapde::manifold::RiemannianSteepestDescent optimizer(options);
    const auto result = optimizer.optimize(problem, geometry, initial);

    EXPECT_EQ(geometry.dimension(), 12);
    EXPECT_TRUE(result.converged());
    EXPECT_EQ(result.iterations, 1);
    EXPECT_EQ(result.cost_evaluations, 2);
    EXPECT_EQ(result.gradient_evaluations, 2);
    EXPECT_EQ(result.rejected_trials, 0);
    for (std::size_t i = 0; i < geometry.factor_count(); ++i) {
        EXPECT_LT(component.distance(result.point[i], expected[i]), 1.0e-9);
    }
}

TEST(AffineInvariantSPDMean, FixedAndDynamicThreeByThreeMeansMatchTheGeodesicMidpoint) {
    const fdapde::manifold::AffineInvariantSPDGeometry<double, 3> fixed_geometry;
    const fdapde::manifold::AffineInvariantSPDGeometry<double, fdapde::Dynamic> dynamic_geometry(3);
    const auto fixed_result = optimize_two_point_affine_mean(fixed_geometry);
    const auto dynamic_result = optimize_two_point_affine_mean(dynamic_geometry);

    EXPECT_LT(matrix_difference_norm(fixed_result, dynamic_result), 1.0e-10);
}
