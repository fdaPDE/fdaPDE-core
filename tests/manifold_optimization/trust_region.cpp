// SPDX-License-Identifier: GPL-3.0-or-later
#include <fdaPDE/manifold_solvers.h>
#include <gtest/gtest.h>
namespace {
/// @brief supplies a scalar chart to isolate the recovered trust-region control flow
struct Chart {
    using Point = double;
    using Tangent = double;
    /// @brief returns the scalar tangent dimension
    std::size_t dimension() const { return 1; }
    /// @brief pairs chart directions with the Euclidean metric
    double inner_product(double, double u, double v) const { return u * v; }
    /// @brief measures the absolute chart displacement
    double norm(double, double u) const { return std::abs(u); }
    /// @brief retains every admissible scalar direction
    double project(double, double u) const { return u; }
    /// @brief creates the additive identity
    double zero_tangent(double) const { return 0; }
    /// @brief combines directions in the fixed chart
    double linear_combination(double, double a, double u, double b, double v) const { return a * u + b * v; }
    /// @brief takes an affine step in the scalar chart
    double retract(double x, double u, double step) const { return x + step * u; }
};
/// @brief supplies a quadratic objective with optional rejection of positive trial points
struct Quadratic {
    /// @brief holds an evaluation-local cache to detect cross-point reuse
    struct Workspace {
        std::optional<double> point;
    };
    bool reject = false;
    /// @brief evaluates the quadratic and remembers the bound point
    double cost(double x, Workspace& workspace) {
        workspace.point = x;
        return reject && x > 0 ? std::numeric_limits<double>::infinity() : .5 * (x - 1) * (x - 1);
    }
    /// @brief checks workspace ownership and returns the analytic gradient
    double grad(double x, Workspace& workspace) {
        if (workspace.point) {
            // accepted gradients must see the cache promoted from the same trial point
            EXPECT_EQ(*workspace.point, x);
        }
        return x - 1;
    }
    /// @brief applies the constant scalar curvature
    double hess(double, double v, Workspace&) { return v; }
};
/// @brief supplies a scalar Hessian to test each truncated-CG termination path
struct Curvature : Quadratic {
    double coefficient = 1;
    /// @brief applies the requested scalar model curvature
    double hess(double, double v, Workspace&) { return coefficient * v; }
};

/// @brief restricts affine steps to a local chart without changing the objective or metric
struct LocalChart : Chart {
    /// @brief rejects chart displacements larger than its supported local radius
    double retract(double x, double u, double step) const {
        fdapde_strong_assert(
          std::abs(step * u) <= .3, fdapde::manifold::RetractionDomainError, "trial leaves the local chart");
        return x + step * u;
    }
};

/// @brief models a geometric failure that has not opted into recoverable trial rejection
struct BrokenChart : Chart {
    /// @brief reports a permanent numerical error independently of the requested step
    double retract(double, double, double) const { throw std::domain_error("invalid geometry state"); }
};

/// @brief detects accidental recovery of exceptions raised by objective callbacks
struct ThrowingObjective : Quadratic {
    /// @brief permits the initial point and rejects later objective evaluations with the retraction marker
    double cost(double x, Workspace& workspace) {
        fdapde_strong_assert(
          x == -1, fdapde::manifold::RetractionDomainError, "objective callback error must propagate");
        return Quadratic::cost(x, workspace);
    }
};
/// @brief checks interior solutions, boundary truncation, negative curvature and invalid Hessian actions
TEST(SmoothingTrustRegion, TruncatedSubproblem) {
    using namespace fdapde::manifold;
    Curvature problem;
    Curvature::Workspace workspace;
    SteihaugTruncatedCG solver;
    problem.coefficient = 4;
    auto result = solver.solve(problem, Chart {}, 0., 2., 1., workspace);
    // the positive quadratic model has its exact minimizer at minus gradient over curvature
    EXPECT_NEAR(result.step, -.5, 1e-12);
    // the interior minimizer has zero residual without touching the radius
    EXPECT_EQ(result.stop_reason, TruncatedCGStopReason::residual_tolerance);
    result = solver.solve(problem, Chart {}, 0., 2., .25, workspace);
    // a smaller radius truncates the same descent ray at its analytic boundary intersection
    EXPECT_NEAR(result.step, -.25, 1e-12);
    // positive curvature with an infeasible Newton step reports ordinary boundary truncation
    EXPECT_EQ(result.stop_reason, TruncatedCGStopReason::boundary);
    problem.coefficient = -1;
    result = solver.solve(problem, Chart {}, 0., 2., 1., workspace);
    // negative curvature follows the descent ray all the way to the radius
    EXPECT_NEAR(result.step, -1., 1e-12);
    // negative curvature is distinct from convergence of a positive definite model
    EXPECT_EQ(result.stop_reason, TruncatedCGStopReason::negative_curvature);
    problem.coefficient = std::numeric_limits<double>::quiet_NaN();
    result = solver.solve(problem, Chart {}, 0., 2., 1., workspace);
    // invalid Hessian actions must stop the model before a trial point is accepted
    EXPECT_EQ(result.stop_reason, TruncatedCGStopReason::non_finite);
}
/// @brief exercises boundary steps, accepted workspace promotion and a refused infeasible minimizer
TEST(SmoothingTrustRegion, ConvergenceAndRejectedTrials) {
    using namespace fdapde::manifold;
    Quadratic problem;
    TrustRegionOptions options;
    options.initial_radius = .2;
    auto result = RiemannianTrustRegion(options).optimize(problem, Chart {}, -1.);
    // bounded subproblems and expanding radii reach the exact quadratic minimizer
    ASSERT_TRUE(result.converged());
    // the first-order certificate agrees with the independent closed-form optimum
    EXPECT_NEAR(result.point, 1., 1e-12);
    problem.reject = true;
    result = RiemannianTrustRegion(options).optimize(problem, Chart {}, -1.);
    // nonfinite trial costs are rejected rather than overwriting the valid current iterate
    EXPECT_GT(result.rejected_steps, 0u);
    // rejected evaluations do not result in a false convergence certificate
    EXPECT_FALSE(result.converged());
    // the retained iterate stays in the domain where the objective is finite
    EXPECT_LE(result.point, 0);
    options.initial_radius = 0;
    // invalid trust-region radii fail at the public constructor boundary
    EXPECT_THROW(RiemannianTrustRegion {options}, std::invalid_argument);
}

// a local retraction failure contracts the radius without evaluating or promoting an invalid candidate
TEST(SmoothingTrustRegion, RecoversLocalRetractionFailures) {
    using namespace fdapde::manifold;
    Quadratic problem;
    TrustRegionOptions options;
    options.initial_radius = 2;
    const auto result = RiemannianTrustRegion(options).optimize(problem, LocalChart {}, -1.);
    // repeated local radius contractions still reach the same exact scalar optimum
    ASSERT_TRUE(result.converged());
    // recovering a local chart exit preserves the independent closed-form solution
    EXPECT_NEAR(result.point, 1., 1e-12);
    // the initial Newton displacement exceeds the supported chart radius and must be refused
    EXPECT_GT(result.rejected_steps, 0u);
    // invalid retractions count as iterations but never as calls to the quadratic objective
    EXPECT_EQ(result.cost_evaluations + result.rejected_steps, result.iterations + 1);
    options.minimum_radius = 1;
    const auto bounded = RiemannianTrustRegion(options).optimize(problem, LocalChart {}, -1.);
    // an inadmissible first trial respects the configured lower radius bound
    EXPECT_EQ(bounded.stop_reason, TrustRegionStopReason::radius_too_small);
    // a refused retraction leaves the original point unchanged
    EXPECT_DOUBLE_EQ(bounded.point, -1.);
    // only the initial objective is evaluated when no retraction trial succeeds
    EXPECT_EQ(bounded.cost_evaluations, 1u);
}

// Armijo budgets count all retraction trials while evaluations count only candidates in the local chart
TEST(SmoothingTrustRegion, ArmijoRetractionBudgetsAndWorkspace) {
    using namespace fdapde::manifold;
    Quadratic problem;
    const LocalChart geometry;
    evaluation_context_t<Quadratic, LocalChart> context;
    const double current_cost = evaluate_cost(problem, geometry, -1., context.current());
    const auto result = ArmijoBacktracking().search(problem, geometry, -1., 2., current_cost, -4., context);
    // contractions from displacement two first enter the supported chart at step one eighth
    ASSERT_TRUE(result.accepted());
    // three failed retractions precede the first evaluable and decreasing candidate
    EXPECT_EQ(result.trials, 4u);
    // rejected retractions are reflected in the same bounded trial accounting as rejected costs
    EXPECT_EQ(result.rejected_trials, 3u);
    // only the admissible trial reaches the objective callback
    EXPECT_EQ(result.cost_evaluations, 1u);
    // the accepted displacement equals the analytically first admissible contracted step
    EXPECT_DOUBLE_EQ(*result.point, -.75);
    // searching never replaces the current workspace before the caller promotes an accepted candidate
    EXPECT_DOUBLE_EQ(*context.current().workspace().point, -1.);
    ArmijoOptions options;
    options.max_trials = 1;
    const auto bounded = ArmijoBacktracking(options).search(problem, geometry, -1., 2., current_cost, -4., context);
    // a failed retraction consumes the last trial and reports bounded line-search failure
    EXPECT_EQ(bounded.status, ArmijoStatus::max_trials);
    // exhausting the retraction budget cannot leave an accepted candidate behind
    EXPECT_FALSE(bounded.point);
    // exhausting the retraction budget never invokes an objective on an invalid candidate
    EXPECT_EQ(bounded.cost_evaluations, 0u);
    // the unsuccessful search clears the preceding trial workspace instead of retaining stale point data
    EXPECT_FALSE(context.trial().workspace().point);
}

// solvers recover only explicit retraction markers raised inside the geometry callback
TEST(SmoothingTrustRegion, PropagatesUnrelatedFailures) {
    using namespace fdapde::manifold;
    Quadratic problem;
    // ordinary geometry domain errors remain visible instead of being silently retried
    EXPECT_THROW(RiemannianTrustRegion().optimize(problem, BrokenChart {}, -1.), std::domain_error);
    // the line search preserves the same distinction for unmarked geometry failures
    EXPECT_THROW(RiemannianSteepestDescent().optimize(problem, BrokenChart {}, -1.), std::domain_error);
    ThrowingObjective throwing;
    // even the recoverable marker must propagate when it comes from a trust-region objective callback
    EXPECT_THROW(RiemannianTrustRegion().optimize(throwing, Chart {}, -1.), RetractionDomainError);
    // Armijo must also leave objective failures outside its retraction-only recovery scope
    EXPECT_THROW(RiemannianSteepestDescent().optimize(throwing, Chart {}, -1.), RetractionDomainError);
}
}   // namespace
