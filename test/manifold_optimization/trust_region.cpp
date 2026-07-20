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
#include <stdexcept>

namespace {

struct TrustEuclideanGeometry {
    using Point = std::array<double, 2>;
    using Tangent = std::array<double, 2>;

    std::size_t dimension() const { return 2; }
    double inner_product(const Point&, const Tangent& u, const Tangent& v) const { return u[0] * v[0] + u[1] * v[1]; }
    double norm(const Point&, const Tangent& u) const { return std::hypot(u[0], u[1]); }
    Tangent project(const Point&, const Tangent& u) const { return u; }
    Tangent zero_tangent(const Point&) const { return {0, 0}; }
    Tangent linear_combination(const Point&, double alpha, const Tangent& u, double beta, const Tangent& v) const {
        return {alpha * u[0] + beta * v[0], alpha * u[1] + beta * v[1]};
    }
    Point retract(const Point& point, const Tangent& tangent, double step) const {
        return {point[0] + step * tangent[0], point[1] + step * tangent[1]};
    }
};

struct CheckedEuclideanGeometry : TrustEuclideanGeometry {
    Tangent
    linear_combination(const Point& point, double alpha, const Tangent& u, double beta, const Tangent& v) const {
        if (!std::isfinite(alpha) || !std::isfinite(beta))
            throw std::domain_error("linear combination requires finite coefficients");
        return TrustEuclideanGeometry::linear_combination(point, alpha, u, beta, v);
    }
};

struct TrustWorkspace {
    bool has_point = false;
    std::array<double, 2> point {0, 0};
};

struct ResetTrackingWorkspace {
    int* reset_count = nullptr;

    ResetTrackingWorkspace() = default;
    ResetTrackingWorkspace(ResetTrackingWorkspace&& other) noexcept : reset_count(other.reset_count) {
        other.reset_count = nullptr;
    }
    ResetTrackingWorkspace& operator=(ResetTrackingWorkspace&& other) noexcept {
        if (this != &other) {
            if (reset_count) ++*reset_count;
            reset_count = other.reset_count;
            other.reset_count = nullptr;
        }
        return *this;
    }
};

struct PositiveDefiniteProblem {
    using Workspace = TrustWorkspace;

    int cost_calls = 0;
    int gradient_calls = 0;
    int hessian_calls = 0;
    int workspace_reuses = 0;

    double cost(const std::array<double, 2>& point, Workspace& workspace) {
        ++cost_calls;
        workspace.has_point = true;
        workspace.point = point;
        return point[0] * point[0] + 2 * point[1] * point[1];
    }
    std::array<double, 2> gradient(const std::array<double, 2>& point, Workspace& workspace) {
        ++gradient_calls;
        if (workspace.has_point && workspace.point == point) ++workspace_reuses;
        return {2 * point[0], 4 * point[1]};
    }
    std::array<double, 2>
    hessian_vector(const std::array<double, 2>&, const std::array<double, 2>& tangent, Workspace&) {
        ++hessian_calls;
        return {2 * tangent[0], 4 * tangent[1]};
    }
};

struct CombinedPositiveDefiniteProblem {
    using Workspace = TrustWorkspace;

    int cost_calls = 0;
    int gradient_calls = 0;
    int combined_calls = 0;
    int hessian_calls = 0;

    double cost(const std::array<double, 2>& point, Workspace& workspace) {
        ++cost_calls;
        workspace.has_point = true;
        workspace.point = point;
        return point[0] * point[0] + 2 * point[1] * point[1];
    }
    std::array<double, 2> gradient(const std::array<double, 2>& point, Workspace&) {
        ++gradient_calls;
        return {2 * point[0], 4 * point[1]};
    }
    std::pair<double, std::array<double, 2>> cost_gradient(const std::array<double, 2>& point, Workspace& workspace) {
        ++combined_calls;
        workspace.has_point = true;
        workspace.point = point;
        return {
          point[0] * point[0] + 2 * point[1] * point[1], {2 * point[0], 4 * point[1]}
        };
    }
    std::array<double, 2>
    hessian_vector(const std::array<double, 2>&, const std::array<double, 2>& tangent, Workspace&) {
        ++hessian_calls;
        return {2 * tangent[0], 4 * tangent[1]};
    }
};

struct NegativeCurvatureProblem {
    using Workspace = TrustWorkspace;

    double cost(const std::array<double, 2>& point, Workspace&) { return -0.5 * point[0] * point[0]; }
    std::array<double, 2> gradient(const std::array<double, 2>& point, Workspace&) { return {-point[0], 0}; }
    std::array<double, 2>
    hessian_vector(const std::array<double, 2>&, const std::array<double, 2>& tangent, Workspace&) {
        return {-tangent[0], -tangent[1]};
    }
};

struct TinyCurvatureProblem {
    using Workspace = TrustWorkspace;

    double cost(const std::array<double, 2>& point, Workspace&) { return point[0]; }
    std::array<double, 2> gradient(const std::array<double, 2>&, Workspace&) { return {1, 0}; }
    std::array<double, 2>
    hessian_vector(const std::array<double, 2>&, const std::array<double, 2>& tangent, Workspace&) {
        const double tiny = std::numeric_limits<double>::denorm_min();
        return {tiny * tangent[0], tiny * tangent[1]};
    }
};

struct NonFiniteHessianProblem {
    using Workspace = TrustWorkspace;

    int cost_calls = 0;
    int gradient_calls = 0;
    int hessian_calls = 0;

    double cost(const std::array<double, 2>& point, Workspace&) {
        ++cost_calls;
        return 0.5 * (point[0] * point[0] + point[1] * point[1]);
    }
    std::array<double, 2> gradient(const std::array<double, 2>& point, Workspace&) {
        ++gradient_calls;
        return point;
    }
    std::array<double, 2> hessian_vector(const std::array<double, 2>&, const std::array<double, 2>&, Workspace&) {
        ++hessian_calls;
        return {std::numeric_limits<double>::quiet_NaN(), 0};
    }
};

struct NonFiniteInitialCostProblem {
    using Workspace = TrustWorkspace;

    int cost_calls = 0;
    int gradient_calls = 0;
    int hessian_calls = 0;

    double cost(const std::array<double, 2>&, Workspace&) {
        ++cost_calls;
        return std::numeric_limits<double>::infinity();
    }
    std::array<double, 2> gradient(const std::array<double, 2>&, Workspace&) {
        ++gradient_calls;
        return {1, 0};
    }
    std::array<double, 2>
    hessian_vector(const std::array<double, 2>&, const std::array<double, 2>& tangent, Workspace&) {
        ++hessian_calls;
        return tangent;
    }
};

struct NonFiniteInitialGradientProblem {
    using Workspace = TrustWorkspace;

    int cost_calls = 0;
    int gradient_calls = 0;
    int hessian_calls = 0;

    double cost(const std::array<double, 2>&, Workspace&) {
        ++cost_calls;
        return 1;
    }
    std::array<double, 2> gradient(const std::array<double, 2>&, Workspace&) {
        ++gradient_calls;
        return {std::numeric_limits<double>::infinity(), 0};
    }
    std::array<double, 2>
    hessian_vector(const std::array<double, 2>&, const std::array<double, 2>& tangent, Workspace&) {
        ++hessian_calls;
        return tangent;
    }
};

struct AcceptedNonFiniteGradientProblem {
    using Workspace = TrustWorkspace;

    int cost_calls = 0;
    int gradient_calls = 0;
    int hessian_calls = 0;

    double cost(const std::array<double, 2>& point, Workspace&) {
        ++cost_calls;
        return point[0] * point[0];
    }
    std::array<double, 2> gradient(const std::array<double, 2>& point, Workspace&) {
        ++gradient_calls;
        if (point[0] == 0) return {std::numeric_limits<double>::infinity(), 0};
        return {2 * point[0], 0};
    }
    std::array<double, 2>
    hessian_vector(const std::array<double, 2>&, const std::array<double, 2>& tangent, Workspace&) {
        ++hessian_calls;
        return {2 * tangent[0], 2 * tangent[1]};
    }
};

struct OffsetPositiveDefiniteProblem : PositiveDefiniteProblem {
    double cost(const std::array<double, 2>& point, Workspace& workspace) {
        ++cost_calls;
        workspace.has_point = true;
        workspace.point = point;
        return 1e20 + point[0] * point[0] + 2 * point[1] * point[1];
    }
};

struct OffsetIncreasingProblem : PositiveDefiniteProblem {
    double cost(const std::array<double, 2>& point, Workspace& workspace) {
        ++cost_calls;
        workspace.has_point = true;
        workspace.point = point;
        if (std::abs(point[0]) < 0.5 && std::abs(point[1]) < 0.5)
            return std::nextafter(1e20, std::numeric_limits<double>::infinity());
        return 1e20;
    }
};

struct RejectingProblem {
    using Workspace = ResetTrackingWorkspace;

    int cost_calls = 0;
    int gradient_calls = 0;
    int hessian_calls = 0;
    int workspace_resets = 0;

    double cost(const std::array<double, 2>& point, Workspace& workspace) {
        ++cost_calls;
        workspace.reset_count = &workspace_resets;
        return 0.5 * (point[0] * point[0] + point[1] * point[1]);
    }
    std::array<double, 2> gradient(const std::array<double, 2>& point, Workspace&) {
        ++gradient_calls;
        return {-point[0], -point[1]};
    }
    std::array<double, 2>
    hessian_vector(const std::array<double, 2>&, const std::array<double, 2>& tangent, Workspace&) {
        ++hessian_calls;
        return tangent;
    }
};

struct NonFiniteTrialProblem {
    using Workspace = ResetTrackingWorkspace;

    int cost_calls = 0;
    int gradient_calls = 0;
    int hessian_calls = 0;
    int workspace_resets = 0;

    double cost(const std::array<double, 2>& point, Workspace& workspace) {
        ++cost_calls;
        workspace.reset_count = &workspace_resets;
        if (point[0] == 0) return std::numeric_limits<double>::infinity();
        return point[0] * point[0];
    }
    std::array<double, 2> gradient(const std::array<double, 2>& point, Workspace&) {
        ++gradient_calls;
        return {2 * point[0], 0};
    }
    std::array<double, 2>
    hessian_vector(const std::array<double, 2>&, const std::array<double, 2>& tangent, Workspace&) {
        ++hessian_calls;
        return {2 * tangent[0], 2 * tangent[1]};
    }
};

static_assert(fdapde::manifold::RiemannianHessianProblem<PositiveDefiniteProblem, TrustEuclideanGeometry>);
static_assert(fdapde::manifold::CombinedCostGradientProblem<CombinedPositiveDefiniteProblem, TrustEuclideanGeometry>);

}   // namespace

TEST(ManifoldTruncatedCG, SolvesPositiveDefiniteQuadraticModel) {
    TrustEuclideanGeometry geometry;
    PositiveDefiniteProblem problem;
    TrustWorkspace workspace;
    const std::array<double, 2> point {1, 1};
    const std::array<double, 2> gradient {2, 4};

    auto result = fdapde::manifold::SteihaugTruncatedCG {}.solve(problem, geometry, point, gradient, 10, workspace);

    EXPECT_EQ(result.stop_reason, fdapde::manifold::TruncatedCGStopReason::residual_tolerance);
    EXPECT_NEAR(result.step[0], -1, 1e-12);
    EXPECT_NEAR(result.step[1], -1, 1e-12);
    EXPECT_NEAR(result.hessian_step[0], -2, 1e-12);
    EXPECT_NEAR(result.hessian_step[1], -4, 1e-12);
    EXPECT_EQ(result.iterations, 2);
    EXPECT_EQ(result.hessian_evaluations, 2);
}

TEST(ManifoldTruncatedCG, ZeroResidualStopsWithoutHessianEvaluation) {
    TrustEuclideanGeometry geometry;
    PositiveDefiniteProblem problem;
    TrustWorkspace workspace;
    const std::array<double, 2> point {1, 1};

    auto result = fdapde::manifold::SteihaugTruncatedCG {}.solve(problem, geometry, point, {0, 0}, 1, workspace);

    EXPECT_EQ(result.stop_reason, fdapde::manifold::TruncatedCGStopReason::residual_tolerance);
    EXPECT_EQ(result.step, (std::array<double, 2> {0, 0}));
    EXPECT_EQ(result.hessian_step, (std::array<double, 2> {0, 0}));
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(result.hessian_evaluations, 0);
}

TEST(ManifoldTruncatedCG, ReportsDeterministicIterationBudget) {
    TrustEuclideanGeometry geometry;
    PositiveDefiniteProblem problem;
    TrustWorkspace workspace;
    fdapde::manifold::TruncatedCGOptions options;
    options.max_iterations = 1;
    const std::array<double, 2> point {1, 1};

    auto result =
      fdapde::manifold::SteihaugTruncatedCG {options}.solve(problem, geometry, point, {2, 4}, 10, workspace);

    EXPECT_EQ(result.stop_reason, fdapde::manifold::TruncatedCGStopReason::max_iterations);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_EQ(result.hessian_evaluations, 1);
    EXPECT_NEAR(result.step[0], -5.0 / 9, 1e-14);
    EXPECT_NEAR(result.step[1], -10.0 / 9, 1e-14);
}

TEST(ManifoldTruncatedCG, NegativeCurvatureTerminatesOnBoundary) {
    TrustEuclideanGeometry geometry;
    NegativeCurvatureProblem problem;
    TrustWorkspace workspace;
    const std::array<double, 2> point {-1, 0};

    auto result = fdapde::manifold::SteihaugTruncatedCG {}.solve(problem, geometry, point, {1, 0}, 2, workspace);

    EXPECT_EQ(result.stop_reason, fdapde::manifold::TruncatedCGStopReason::negative_curvature);
    EXPECT_NEAR(result.step[0], -2, 1e-12);
    EXPECT_NEAR(result.step[1], 0, 1e-12);
    EXPECT_NEAR(result.hessian_step[0], 2, 1e-12);
    EXPECT_NEAR(result.hessian_step[1], 0, 1e-12);
    EXPECT_NEAR(geometry.norm(point, result.step), 2, 1e-12);
    EXPECT_EQ(result.iterations, 1);
}

TEST(ManifoldTruncatedCG, IntersectsBoundaryAfterAnAccumulatedStep) {
    TrustEuclideanGeometry geometry;
    PositiveDefiniteProblem problem;
    TrustWorkspace workspace;
    const std::array<double, 2> point {1, 1};
    const double first_step_norm = 5 * std::sqrt(5.0) / 9;
    const double radius = std::nextafter(first_step_norm, std::numeric_limits<double>::infinity());

    auto result = fdapde::manifold::SteihaugTruncatedCG {}.solve(problem, geometry, point, {2, 4}, radius, workspace);

    EXPECT_EQ(result.stop_reason, fdapde::manifold::TruncatedCGStopReason::boundary);
    EXPECT_EQ(result.iterations, 2);
    EXPECT_LT(result.step[0], -5.0 / 9);
    EXPECT_NEAR(geometry.norm(point, result.step), radius, 4 * std::numeric_limits<double>::epsilon());
    EXPECT_NEAR(result.hessian_step[0], 2 * result.step[0], 1e-14);
    EXPECT_NEAR(result.hessian_step[1], 4 * result.step[1], 1e-14);
}

TEST(ManifoldTruncatedCG, ValidatesAlphaBeforeCallingGeometry) {
    CheckedEuclideanGeometry geometry;
    TinyCurvatureProblem problem;
    TrustWorkspace workspace;
    const std::array<double, 2> point {0, 0};

    auto result = fdapde::manifold::SteihaugTruncatedCG {}.solve(problem, geometry, point, {1, 0}, 1, workspace);

    EXPECT_EQ(result.stop_reason, fdapde::manifold::TruncatedCGStopReason::non_finite);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_EQ(result.hessian_evaluations, 1);
}

TEST(ManifoldTrustRegion, ConvergesOnPositiveDefiniteQuadratic) {
    TrustEuclideanGeometry geometry;
    PositiveDefiniteProblem problem;
    fdapde::manifold::TrustRegionOptions options;
    options.initial_radius = 10;
    options.maximum_radius = 10;
    options.gradient_tolerance = 1e-12;

    auto result = fdapde::manifold::RiemannianTrustRegion {options}.optimize(problem, geometry, {1, 1});

    EXPECT_TRUE(result.converged());
    EXPECT_EQ(result.stop_reason, fdapde::manifold::TrustRegionStopReason::gradient_tolerance);
    EXPECT_NEAR(result.point[0], 0, 1e-12);
    EXPECT_NEAR(result.point[1], 0, 1e-12);
    EXPECT_NEAR(result.cost, 0, 1e-12);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_EQ(result.accepted_steps, 1);
    EXPECT_EQ(result.rejected_steps, 0);
    EXPECT_EQ(result.cost_evaluations, 2);
    EXPECT_EQ(result.gradient_evaluations, 2);
    EXPECT_EQ(result.hessian_evaluations, 2);
    EXPECT_EQ(problem.cost_calls, 2);
    EXPECT_EQ(problem.gradient_calls, 2);
    EXPECT_EQ(problem.hessian_calls, 2);
    EXPECT_EQ(problem.workspace_reuses, 2);
}

TEST(ManifoldTrustRegion, UsesCombinedInitialEvaluationWithLogicalCounters) {
    TrustEuclideanGeometry geometry;
    CombinedPositiveDefiniteProblem problem;
    fdapde::manifold::TrustRegionOptions options;
    options.initial_radius = 10;
    options.maximum_radius = 10;
    options.gradient_tolerance = 1e-12;

    auto result = fdapde::manifold::RiemannianTrustRegion {options}.optimize(problem, geometry, {1, 1});

    EXPECT_TRUE(result.converged());
    EXPECT_EQ(result.cost_evaluations, 2);
    EXPECT_EQ(result.gradient_evaluations, 2);
    EXPECT_EQ(result.hessian_evaluations, 2);
    EXPECT_EQ(problem.combined_calls, 1);
    EXPECT_EQ(problem.cost_calls, 1);
    EXPECT_EQ(problem.gradient_calls, 1);
    EXPECT_EQ(problem.hessian_calls, 2);
}

TEST(ManifoldTrustRegion, RejectedStepsShrinkRadiusUntilTermination) {
    TrustEuclideanGeometry geometry;
    RejectingProblem problem;
    fdapde::manifold::TrustRegionOptions options;
    options.max_iterations = 10;
    options.initial_radius = 1;
    options.maximum_radius = 1;
    options.minimum_radius = 0.1;
    options.shrink_factor = 0.25;
    const std::array<double, 2> initial {1, 0};

    auto result = fdapde::manifold::RiemannianTrustRegion {options}.optimize(problem, geometry, initial);

    EXPECT_EQ(result.stop_reason, fdapde::manifold::TrustRegionStopReason::radius_too_small);
    EXPECT_EQ(result.point, initial);
    EXPECT_EQ(result.iterations, 2);
    EXPECT_EQ(result.accepted_steps, 0);
    EXPECT_EQ(result.rejected_steps, 2);
    EXPECT_DOUBLE_EQ(result.radius, 0.0625);
    EXPECT_LT(result.last_ratio, 0);
    EXPECT_EQ(result.cost_evaluations, 3);
    EXPECT_EQ(result.gradient_evaluations, 1);
    EXPECT_EQ(problem.cost_calls, 3);
    EXPECT_EQ(problem.gradient_calls, 1);
    EXPECT_EQ(problem.hessian_calls, 2);
    EXPECT_EQ(problem.workspace_resets, 2);
}

TEST(ManifoldTrustRegion, RegularizesRatioForLargeOffsetCosts) {
    TrustEuclideanGeometry geometry;
    OffsetPositiveDefiniteProblem problem;
    fdapde::manifold::TrustRegionOptions options;
    options.initial_radius = 10;
    options.maximum_radius = 10;
    options.gradient_tolerance = 1e-12;

    auto result = fdapde::manifold::RiemannianTrustRegion {options}.optimize(problem, geometry, {1, 1});

    EXPECT_TRUE(result.converged());
    EXPECT_EQ(result.accepted_steps, 1);
    EXPECT_EQ(result.rejected_steps, 0);
    EXPECT_EQ(result.point, (std::array<double, 2> {0, 0}));
    EXPECT_DOUBLE_EQ(result.cost, 1e20);
    EXPECT_GT(result.last_ratio, options.expansion_threshold);
    EXPECT_EQ(problem.hessian_calls, 2);
}

TEST(ManifoldTrustRegion, RejectsObservedCostIncreaseDespiteRegularization) {
    TrustEuclideanGeometry geometry;
    OffsetIncreasingProblem problem;
    fdapde::manifold::TrustRegionOptions options;
    options.max_iterations = 1;
    options.initial_radius = 10;
    options.maximum_radius = 10;
    const std::array<double, 2> initial {1, 1};

    auto result = fdapde::manifold::RiemannianTrustRegion {options}.optimize(problem, geometry, initial);

    EXPECT_EQ(result.stop_reason, fdapde::manifold::TrustRegionStopReason::max_iterations);
    EXPECT_EQ(result.point, initial);
    EXPECT_EQ(result.accepted_steps, 0);
    EXPECT_EQ(result.rejected_steps, 1);
    EXPECT_GT(result.last_ratio, options.expansion_threshold);
    EXPECT_DOUBLE_EQ(result.radius, 2.5);
}

TEST(ManifoldTrustRegion, ReportsNonFiniteInitialCostWithoutRunningModel) {
    TrustEuclideanGeometry geometry;
    NonFiniteInitialCostProblem problem;

    auto result = fdapde::manifold::RiemannianTrustRegion {}.optimize(problem, geometry, {1, 0});

    EXPECT_EQ(result.stop_reason, fdapde::manifold::TrustRegionStopReason::non_finite_cost);
    EXPECT_EQ(result.subproblem_stop_reason, fdapde::manifold::TruncatedCGStopReason::not_run);
    EXPECT_EQ(result.cost_evaluations, 1);
    EXPECT_EQ(result.gradient_evaluations, 0);
    EXPECT_EQ(result.hessian_evaluations, 0);
    EXPECT_EQ(problem.cost_calls, 1);
    EXPECT_EQ(problem.gradient_calls, 0);
    EXPECT_EQ(problem.hessian_calls, 0);
}

TEST(ManifoldTrustRegion, ReportsNonFiniteInitialGradientWithoutRunningModel) {
    TrustEuclideanGeometry geometry;
    NonFiniteInitialGradientProblem problem;

    auto result = fdapde::manifold::RiemannianTrustRegion {}.optimize(problem, geometry, {1, 0});

    EXPECT_EQ(result.stop_reason, fdapde::manifold::TrustRegionStopReason::non_finite_gradient);
    EXPECT_EQ(result.subproblem_stop_reason, fdapde::manifold::TruncatedCGStopReason::not_run);
    EXPECT_EQ(result.cost_evaluations, 1);
    EXPECT_EQ(result.gradient_evaluations, 1);
    EXPECT_EQ(result.hessian_evaluations, 0);
    EXPECT_EQ(problem.cost_calls, 1);
    EXPECT_EQ(problem.gradient_calls, 1);
    EXPECT_EQ(problem.hessian_calls, 0);
}

TEST(ManifoldTrustRegion, ReportsNonFiniteGradientAfterAcceptedTrial) {
    TrustEuclideanGeometry geometry;
    AcceptedNonFiniteGradientProblem problem;
    fdapde::manifold::TrustRegionOptions options;
    options.initial_radius = 10;
    options.maximum_radius = 10;

    auto result = fdapde::manifold::RiemannianTrustRegion {options}.optimize(problem, geometry, {1, 0});

    EXPECT_EQ(result.stop_reason, fdapde::manifold::TrustRegionStopReason::non_finite_gradient);
    EXPECT_EQ(result.point, (std::array<double, 2> {0, 0}));
    EXPECT_EQ(result.iterations, 1);
    EXPECT_EQ(result.accepted_steps, 1);
    EXPECT_EQ(result.rejected_steps, 0);
    EXPECT_EQ(result.cost_evaluations, 2);
    EXPECT_EQ(result.gradient_evaluations, 2);
    EXPECT_EQ(result.hessian_evaluations, 1);
}

TEST(ManifoldTrustRegion, NonFiniteHessianReportsModelFailure) {
    TrustEuclideanGeometry geometry;
    NonFiniteHessianProblem problem;

    auto result = fdapde::manifold::RiemannianTrustRegion {}.optimize(problem, geometry, {1, 0});

    EXPECT_EQ(result.stop_reason, fdapde::manifold::TrustRegionStopReason::model_failure);
    EXPECT_EQ(result.subproblem_stop_reason, fdapde::manifold::TruncatedCGStopReason::non_finite);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(result.accepted_steps, 0);
    EXPECT_EQ(result.rejected_steps, 0);
    EXPECT_EQ(result.cost_evaluations, 1);
    EXPECT_EQ(result.gradient_evaluations, 1);
    EXPECT_EQ(result.hessian_evaluations, 1);
    EXPECT_EQ(problem.cost_calls, 1);
    EXPECT_EQ(problem.gradient_calls, 1);
    EXPECT_EQ(problem.hessian_calls, 1);
}

TEST(ManifoldTrustRegion, NonFiniteTrialIsRejectedWithoutReplacingCurrentCache) {
    TrustEuclideanGeometry geometry;
    NonFiniteTrialProblem problem;
    fdapde::manifold::TrustRegionOptions options;
    options.max_iterations = 1;
    options.initial_radius = 10;
    options.maximum_radius = 10;
    const std::array<double, 2> initial {1, 0};

    auto result = fdapde::manifold::RiemannianTrustRegion {options}.optimize(problem, geometry, initial);

    EXPECT_EQ(result.stop_reason, fdapde::manifold::TrustRegionStopReason::max_iterations);
    EXPECT_EQ(result.point, initial);
    EXPECT_DOUBLE_EQ(result.cost, 1);
    EXPECT_EQ(result.accepted_steps, 0);
    EXPECT_EQ(result.rejected_steps, 1);
    EXPECT_EQ(result.last_ratio, -std::numeric_limits<double>::infinity());
    EXPECT_DOUBLE_EQ(result.radius, 2.5);
    EXPECT_EQ(result.cost_evaluations, 2);
    EXPECT_EQ(result.gradient_evaluations, 1);
    EXPECT_EQ(result.hessian_evaluations, 1);
    EXPECT_EQ(problem.workspace_resets, 1);
}

TEST(ManifoldTrustRegion, ReportsSubproblemNotRunAtInitialSolution) {
    TrustEuclideanGeometry geometry;
    PositiveDefiniteProblem problem;

    auto result = fdapde::manifold::RiemannianTrustRegion {}.optimize(problem, geometry, {0, 0});
    fdapde::manifold::TruncatedCGResult<std::array<double, 2>> idle {
      {0, 0},
      {0, 0}
    };

    EXPECT_TRUE(result.converged());
    EXPECT_EQ(result.subproblem_stop_reason, fdapde::manifold::TruncatedCGStopReason::not_run);
    EXPECT_EQ(idle.stop_reason, fdapde::manifold::TruncatedCGStopReason::not_run);
    EXPECT_EQ(problem.hessian_calls, 0);
}

TEST(ManifoldTrustRegion, SuccessfulBoundaryStepExpandsRadius) {
    TrustEuclideanGeometry geometry;
    PositiveDefiniteProblem problem;
    fdapde::manifold::TrustRegionOptions options;
    options.max_iterations = 1;
    options.gradient_tolerance = 0;
    options.initial_radius = 0.25;
    options.maximum_radius = 1;
    options.ratio_regularization = 0;

    auto result = fdapde::manifold::RiemannianTrustRegion {options}.optimize(problem, geometry, {1, 0});

    EXPECT_EQ(result.stop_reason, fdapde::manifold::TrustRegionStopReason::max_iterations);
    EXPECT_EQ(result.subproblem_stop_reason, fdapde::manifold::TruncatedCGStopReason::boundary);
    EXPECT_EQ(result.accepted_steps, 1);
    EXPECT_EQ(result.rejected_steps, 0);
    EXPECT_EQ(result.hessian_evaluations, 1);
    EXPECT_NEAR(result.point[0], 0.75, 1e-12);
    EXPECT_DOUBLE_EQ(result.radius, 0.5);
    EXPECT_DOUBLE_EQ(result.last_ratio, 1);
}

TEST(ManifoldTrustRegion, InvalidOptionsAndRadiiAreRejected) {
    fdapde::manifold::TruncatedCGOptions cg_options;
    cg_options.max_iterations = 0;
    EXPECT_THROW(fdapde::manifold::SteihaugTruncatedCG {cg_options}, std::invalid_argument);
    cg_options.max_iterations = 1;
    cg_options.residual_tolerance = 1;
    EXPECT_THROW(fdapde::manifold::SteihaugTruncatedCG {cg_options}, std::invalid_argument);

    fdapde::manifold::TrustRegionOptions options;
    options.minimum_radius = 2;
    EXPECT_THROW(fdapde::manifold::RiemannianTrustRegion {options}, std::invalid_argument);
    options = {};
    options.maximum_radius = 1e200;
    EXPECT_THROW(fdapde::manifold::RiemannianTrustRegion {options}, std::invalid_argument);
    options = {};
    options.ratio_regularization = -1;
    EXPECT_THROW(fdapde::manifold::RiemannianTrustRegion {options}, std::invalid_argument);

    TrustEuclideanGeometry geometry;
    PositiveDefiniteProblem problem;
    TrustWorkspace workspace;
    const std::array<double, 2> point {1, 1};
    const std::array<double, 2> gradient {2, 4};
    fdapde::manifold::SteihaugTruncatedCG solver;
    EXPECT_THROW(solver.solve(problem, geometry, point, gradient, 1e200, workspace), std::invalid_argument);
    EXPECT_THROW(solver.solve(problem, geometry, point, gradient, 1e-200, workspace), std::invalid_argument);
}
