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

namespace {

struct ScalarGeometry {
    using Point = double;
    using Tangent = double;

    std::size_t dimension() const { return 1; }
    double inner_product(const Point&, const Tangent& u, const Tangent& v) const { return u * v; }
    double norm(const Point&, const Tangent& tangent) const { return std::abs(tangent); }
    Tangent project(const Point&, const Tangent& tangent) const { return tangent; }
    Tangent zero_tangent(const Point&) const { return 0; }
    Tangent linear_combination(const Point&, double alpha, const Tangent& u, double beta, const Tangent& v) const {
        return alpha * u + beta * v;
    }
    Point retract(const Point& point, const Tangent& tangent, double step) const { return point + step * tangent; }
};

using ProductGeometry = fdapde::manifold::PowerGeometry<ScalarGeometry>;
using FixedGeometry = fdapde::manifold::AffineInvariantSPDGeometry<double, 3>;
using DynamicGeometry = fdapde::manifold::AffineInvariantSPDGeometry<double, fdapde::Dynamic>;

static_assert(fdapde::manifold::FirstOrderGeometry<ScalarGeometry>);
static_assert(fdapde::manifold::FirstOrderGeometry<ProductGeometry>);

template <typename Geometry> Geometry make_airm_geometry() {
    if constexpr (Geometry::Point::Rows == fdapde::Dynamic) {
        return Geometry(3);
    } else {
        return Geometry {};
    }
}

template <typename Geometry> typename Geometry::Point make_airm_point(const std::array<double, 9>& coefficients) {
    fdapde::linalg::Matrix<double, 3, 3> dense;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) { dense(i, j) = coefficients[static_cast<std::size_t>(3 * i + j)]; }
    }
    return typename Geometry::Point(dense, fdapde::linalg::checked);
}

template <typename Geometry> typename Geometry::Point make_airm_identity() {
    fdapde::linalg::Matrix<double, 3, 3> dense;
    dense.set_zero();
    for (int i = 0; i < 3; ++i) { dense(i, i) = 1; }
    return typename Geometry::Point(dense, fdapde::linalg::checked);
}

template <typename Geometry> typename Geometry::Tangent make_airm_tangent(const std::array<double, 6>& coefficients) {
    typename Geometry::Tangent tangent;
    if constexpr (Geometry::Tangent::Rows == fdapde::Dynamic) { tangent.resize(3, 3); }
    std::size_t index = 0;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j <= i; ++j) { tangent(i, j) = coefficients[index++]; }
    }
    return tangent;
}

template <typename Lhs, typename Rhs> void expect_matrix_near(const Lhs& lhs, const Rhs& rhs, double tolerance) {
    ASSERT_EQ(lhs.rows(), rhs.rows());
    ASSERT_EQ(lhs.cols(), rhs.cols());
    for (int i = 0; i < lhs.rows(); ++i) {
        for (int j = 0; j <= i; ++j) {
            EXPECT_NEAR(static_cast<double>(lhs(i, j)), static_cast<double>(rhs(i, j)), tolerance);
        }
    }
}

template <typename Geometry> typename Geometry::Tangent run_airm_manufactured_solve() {
    const Geometry geometry = make_airm_geometry<Geometry>();
    const auto base = make_airm_identity<Geometry>();
    const auto first_target = make_airm_point<Geometry>({2.2, 0.35, -0.1, 0.35, 1.4, 0.25, -0.1, 0.25, 3.1});
    const auto second_target = make_airm_point<Geometry>({1.3, -0.2, 0.3, -0.2, 2.7, 0.15, 0.3, 0.15, 1.9});
    const auto expected = make_airm_tangent<Geometry>({0.4, -0.2, 0.7, 0.15, -0.35, 0.5});
    auto hessian = [&](const typename Geometry::Tangent& direction) {
        const auto first = geometry.half_squared_distance_hessian_vector(base, first_target, direction);
        const auto second = geometry.half_squared_distance_hessian_vector(base, second_target, direction);
        return geometry.linear_combination(base, 0.35, first, 0.65, second);
    };
    const auto right_hand_side = hessian(expected);

    fdapde::manifold::PositiveDefiniteCGOptions options;
    options.max_iterations = 12;
    options.residual_tolerance = 1.0e-11;
    const fdapde::manifold::PositiveDefiniteConjugateGradient solver(options);
    std::size_t calls = 0;
    auto counting_hessian = [&](const typename Geometry::Tangent& direction) {
        ++calls;
        return hessian(direction);
    };
    const auto result = solver.solve(counting_hessian, geometry, base, right_hand_side);

    EXPECT_TRUE(result.converged());
    EXPECT_EQ(result.stop_reason, fdapde::manifold::PositiveDefiniteCGStopReason::residual_tolerance);
    EXPECT_EQ(calls, result.iterations);
    const auto solution_error = geometry.linear_combination(base, 1, result.solution, -1, expected);
    EXPECT_LT(geometry.norm(base, solution_error), 1.0e-8 * (1 + geometry.norm(base, expected)));

    const auto applied_solution = hessian(result.solution);
    const auto true_residual = geometry.linear_combination(base, 1, right_hand_side, -1, applied_solution);
    const double true_residual_norm = geometry.norm(base, true_residual);
    const double residual_tolerance = 5.0e-10 * std::max(1.0, geometry.norm(base, right_hand_side));
    EXPECT_LT(true_residual_norm, residual_tolerance);
    EXPECT_NEAR(result.residual_norm, true_residual_norm, residual_tolerance);

    calls = 0;
    const auto repeated = solver.solve(counting_hessian, geometry, base, right_hand_side);
    EXPECT_EQ(repeated.stop_reason, result.stop_reason);
    EXPECT_EQ(repeated.iterations, result.iterations);
    EXPECT_EQ(calls, repeated.iterations);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j <= i; ++j) { EXPECT_DOUBLE_EQ(repeated.solution(i, j), result.solution(i, j)); }
    }
    return result.solution;
}

}   // namespace

TEST(ManifoldPositiveDefiniteCG, SolvesScalarSystemWithThePositiveRightHandSideConvention) {
    const ScalarGeometry geometry;
    const fdapde::manifold::PositiveDefiniteConjugateGradient solver;
    std::size_t calls = 0;
    auto apply = [&](double direction) {
        ++calls;
        return 4 * direction;
    };
    const auto result = solver.solve(apply, geometry, 0.0, 6.0);

    EXPECT_TRUE(result.converged());
    EXPECT_EQ(result.stop_reason, fdapde::manifold::PositiveDefiniteCGStopReason::residual_tolerance);
    EXPECT_DOUBLE_EQ(result.solution, 1.5);
    EXPECT_DOUBLE_EQ(result.residual_norm, 0);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_EQ(calls, result.iterations);
}

TEST(ManifoldPositiveDefiniteCG, SolvesCoupledProductSystemAndMatchesTheTrueResidual) {
    const ProductGeometry geometry(2);
    const ProductGeometry::Point point {0, 0};
    const ProductGeometry::Tangent right_hand_side {6, 7};
    const fdapde::manifold::PositiveDefiniteConjugateGradient solver;
    std::size_t calls = 0;
    auto apply = [&](const ProductGeometry::Tangent& direction) {
        ++calls;
        return ProductGeometry::Tangent {4 * direction[0] + direction[1], direction[0] + 3 * direction[1]};
    };
    const auto result = solver.solve(apply, geometry, point, right_hand_side);

    EXPECT_TRUE(result.converged());
    ASSERT_EQ(result.solution.size(), 2);
    EXPECT_NEAR(result.solution[0], 1, 1.0e-14);
    EXPECT_NEAR(result.solution[1], 2, 1.0e-14);
    EXPECT_EQ(result.iterations, 2);
    EXPECT_EQ(calls, result.iterations);
    const auto applied = apply(result.solution);
    const auto true_residual = geometry.linear_combination(point, 1, right_hand_side, -1, applied);
    EXPECT_NEAR(result.residual_norm, geometry.norm(point, true_residual), 1.0e-14);
}

TEST(ManifoldPositiveDefiniteCG, ZeroRightHandSideSkipsTheOperator) {
    const ScalarGeometry geometry;
    const fdapde::manifold::PositiveDefiniteConjugateGradient solver;
    std::size_t calls = 0;
    auto apply = [&](double direction) {
        ++calls;
        return direction;
    };
    const auto result = solver.solve(apply, geometry, 0.0, 0.0);

    EXPECT_TRUE(result.converged());
    EXPECT_DOUBLE_EQ(result.solution, 0);
    EXPECT_DOUBLE_EQ(result.residual_norm, 0);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(calls, 0);
}

TEST(ManifoldPositiveDefiniteCG, HonorsTheIterationBudgetDeterministically) {
    const ProductGeometry geometry(2);
    const ProductGeometry::Point point {0, 0};
    const ProductGeometry::Tangent right_hand_side {2, 4};
    fdapde::manifold::PositiveDefiniteCGOptions options;
    options.max_iterations = 1;
    const fdapde::manifold::PositiveDefiniteConjugateGradient solver(options);
    auto apply = [](const ProductGeometry::Tangent& direction) {
        return ProductGeometry::Tangent {2 * direction[0], 4 * direction[1]};
    };

    const auto first = solver.solve(apply, geometry, point, right_hand_side);
    const auto second = solver.solve(apply, geometry, point, right_hand_side);
    EXPECT_FALSE(first.converged());
    EXPECT_EQ(first.stop_reason, fdapde::manifold::PositiveDefiniteCGStopReason::max_iterations);
    EXPECT_EQ(first.iterations, 1);
    ASSERT_EQ(first.solution.size(), 2);
    EXPECT_NEAR(first.solution[0], 5.0 / 9.0, 1.0e-15);
    EXPECT_NEAR(first.solution[1], 10.0 / 9.0, 1.0e-15);
    EXPECT_NEAR(first.residual_norm, 4 * std::sqrt(5.0) / 9, 1.0e-15);
    EXPECT_EQ(second.stop_reason, first.stop_reason);
    EXPECT_EQ(second.iterations, first.iterations);
    EXPECT_DOUBLE_EQ(second.solution[0], first.solution[0]);
    EXPECT_DOUBLE_EQ(second.solution[1], first.solution[1]);
    EXPECT_DOUBLE_EQ(second.residual_norm, first.residual_norm);
}

TEST(ManifoldPositiveDefiniteCG, ReportsCurvatureAndNonfiniteFailures) {
    const ScalarGeometry geometry;
    const fdapde::manifold::PositiveDefiniteConjugateGradient solver;

    auto negative = [](double direction) { return -direction; };
    const auto negative_result = solver.solve(negative, geometry, 0.0, 1.0);
    EXPECT_EQ(negative_result.stop_reason, fdapde::manifold::PositiveDefiniteCGStopReason::non_positive_curvature);
    EXPECT_EQ(negative_result.iterations, 1);
    EXPECT_DOUBLE_EQ(negative_result.solution, 0);

    auto zero = [](double) { return 0.0; };
    const auto zero_result = solver.solve(zero, geometry, 0.0, 1.0);
    EXPECT_EQ(zero_result.stop_reason, fdapde::manifold::PositiveDefiniteCGStopReason::non_positive_curvature);
    EXPECT_EQ(zero_result.iterations, 1);

    std::size_t calls = 0;
    auto identity = [&](double direction) {
        ++calls;
        return direction;
    };
    const auto nonfinite_rhs = solver.solve(identity, geometry, 0.0, std::numeric_limits<double>::infinity());
    EXPECT_EQ(nonfinite_rhs.stop_reason, fdapde::manifold::PositiveDefiniteCGStopReason::non_finite);
    EXPECT_EQ(nonfinite_rhs.iterations, 0);
    EXPECT_EQ(calls, 0);

    auto nonfinite = [](double) { return std::numeric_limits<double>::quiet_NaN(); };
    const auto nonfinite_operator = solver.solve(nonfinite, geometry, 0.0, 1.0);
    EXPECT_EQ(nonfinite_operator.stop_reason, fdapde::manifold::PositiveDefiniteCGStopReason::non_finite);
    EXPECT_EQ(nonfinite_operator.iterations, 1);

    auto tiny_curvature = [](double) { return 5.0e-155; };
    const auto nonfinite_alpha = solver.solve(tiny_curvature, geometry, 0.0, 1.3e154);
    EXPECT_EQ(nonfinite_alpha.stop_reason, fdapde::manifold::PositiveDefiniteCGStopReason::non_finite);
    EXPECT_EQ(nonfinite_alpha.iterations, 1);
}

TEST(ManifoldPositiveDefiniteCG, ReportsFiniteBreakdownAndRejectsInvalidInputs) {
    const ScalarGeometry geometry;
    const fdapde::manifold::PositiveDefiniteConjugateGradient solver;
    std::size_t calls = 0;
    auto identity = [&](double direction) {
        ++calls;
        return direction;
    };
    const auto underflow = solver.solve(identity, geometry, 0.0, 1.0e-200);
    EXPECT_EQ(underflow.stop_reason, fdapde::manifold::PositiveDefiniteCGStopReason::numerical_breakdown);
    EXPECT_EQ(underflow.iterations, 0);
    EXPECT_EQ(calls, 0);
    const auto overflowing_inner = solver.solve(identity, geometry, 0.0, 1.0e200);
    EXPECT_EQ(overflowing_inner.stop_reason, fdapde::manifold::PositiveDefiniteCGStopReason::non_finite);
    EXPECT_EQ(overflowing_inner.iterations, 0);
    EXPECT_EQ(calls, 0);

    fdapde::manifold::PositiveDefiniteCGOptions options;
    options.max_iterations = 0;
    EXPECT_THROW(fdapde::manifold::PositiveDefiniteConjugateGradient {options}, std::invalid_argument);
    options.max_iterations = 1;
    for (const double invalid :
         {-1.0, 1.0, std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        options.residual_tolerance = invalid;
        EXPECT_THROW(fdapde::manifold::PositiveDefiniteConjugateGradient {options}, std::invalid_argument);
    }
    options.residual_tolerance = 0;
    EXPECT_NO_THROW(fdapde::manifold::PositiveDefiniteConjugateGradient {options});

    const DynamicGeometry dynamic_geometry(3);
    const auto point = make_airm_identity<DynamicGeometry>();
    const auto right_hand_side = make_airm_tangent<DynamicGeometry>({1, 0, 1, 0, 0, 1});
    auto wrong_shape = [](const DynamicGeometry::Tangent&) {
        DynamicGeometry::Tangent result(2, 2);
        result(0, 0) = 1;
        result(1, 0) = 0;
        result(1, 1) = 1;
        return result;
    };
    EXPECT_THROW(solver.solve(wrong_shape, dynamic_geometry, point, right_hand_side), std::invalid_argument);
}

TEST(ManifoldPositiveDefiniteCG, SolvesManufacturedAIRMSystemsForFixedAndDynamicSPD) {
    const auto fixed = run_airm_manufactured_solve<FixedGeometry>();
    const auto dynamic = run_airm_manufactured_solve<DynamicGeometry>();
    expect_matrix_near(fixed, dynamic, 5.0e-9);
}
