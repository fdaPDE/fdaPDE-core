// SPDX-License-Identifier: GPL-3.0-or-later
#include <fdaPDE/manifold_geometry.h>
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

/// @brief supplies a rescaled physical quadratic with a slightly inexact chart Hessian
struct PhysicalQuadratic : Quadratic {
    double scale = 1e6;
    std::size_t stationarity_evaluations = 0;
    /// @brief evaluates the physical quadratic while binding the chart workspace
    double cost(double x, Workspace& workspace) {
        workspace.point = x;
        return .5 * (scale * x - 1) * (scale * x - 1);
    }
    /// @brief differentiates the physical cost in chart coordinates
    double grad(double x, Workspace& workspace) {
        // gradient evaluation must use the current point's initial or promoted workspace
        EXPECT_EQ(*workspace.point, x);
        return scale * (scale * x - 1);
    }
    /// @brief leaves a small physical residual after the first accepted Newton trial
    double hess(double, double direction, Workspace&) { return scale * scale * (1 + 1e-4) * direction; }
    /// @brief converts the chart gradient back to the physical stopping units
    double stationarity_norm(double x, double gradient, Workspace& workspace) {
        // the stopping certificate must observe the same workspace as the accepted point
        EXPECT_EQ(*workspace.point, x);
        ++stationarity_evaluations;
        return std::abs(gradient) / scale;
    }
};

/// @brief exposes a misleading zero certificate only at trial points outside the objective domain
struct RejectedStationarity : Quadratic {
    std::size_t stationarity_evaluations = 0;
    /// @brief measures only accepted points and detects any leaked trial workspace
    double stationarity_norm(double x, double gradient, Workspace& workspace) {
        // a rejected trial cache must never replace the accepted point's certificate cache
        EXPECT_EQ(*workspace.point, x);
        ++stationarity_evaluations;
        return x > 0 ? 0 : std::abs(gradient);
    }
};

/// @brief provides an invalid certificate either initially or after one accepted step
struct InvalidStationarity : Quadratic {
    double invalid = std::numeric_limits<double>::quiet_NaN();
    bool initially_valid = false;
    /// @brief retains a valid initial norm when testing an invalid promoted certificate
    double stationarity_norm(double x, double gradient, Workspace&) {
        return initially_valid && x == -1 ? std::abs(gradient) : invalid;
    }
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
    // objectives without a custom certificate retain their original metric gradient stopping norm
    EXPECT_DOUBLE_EQ(result.stationarity_norm, result.gradient_norm);
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

// a physical certificate can stop a rescaled chart without changing its gradient or Hessian geometry
TEST(SmoothingTrustRegion, StopsAtPhysicalStationarity) {
    using namespace fdapde::manifold;
    PhysicalQuadratic problem;
    TrustRegionOptions options;
    const auto result = RiemannianTrustRegion(options).optimize(problem, Chart {}, (1 + 1e-3) / problem.scale);
    // the accepted trial reaches the original physical tolerance despite the deliberately inexact Hessian
    ASSERT_TRUE(result.converged());
    // the retained certificate is the independently recovered physical derivative
    EXPECT_LE(result.stationarity_norm, options.gradient_tolerance);
    // the chart norm remains large enough that the former metric-only test would continue iterating
    EXPECT_GT(result.gradient_norm, options.gradient_tolerance);
    // one accepted chart step is sufficient to satisfy the physical certificate
    EXPECT_EQ(result.accepted_steps, 1u);
    // certificate evaluation occurs only at the initial point and its accepted successor
    EXPECT_EQ(problem.stationarity_evaluations, 2u);
    problem.stationarity_evaluations = 0;
    const auto initial =
      RiemannianTrustRegion(options).optimize(problem, Chart {}, (1 + .5 * options.gradient_tolerance) / problem.scale);
    // an initial point already inside the physical tolerance needs no model or trial evaluation
    ASSERT_TRUE(initial.converged());
    // stopping before the first subproblem preserves the iteration counter
    EXPECT_EQ(initial.iterations, 0u);
    // the initial stopping certificate is evaluated exactly once
    EXPECT_EQ(problem.stationarity_evaluations, 1u);
}

// a rejected candidate's zero certificate cannot terminate or overwrite the accepted state
TEST(SmoothingTrustRegion, IgnoresRejectedTrialStationarity) {
    using namespace fdapde::manifold;
    RejectedStationarity problem;
    problem.reject = true;
    TrustRegionOptions options;
    options.initial_radius = 2;
    options.max_iterations = 1;
    const auto result = RiemannianTrustRegion(options).optimize(problem, Chart {}, -1.);
    // the rejected exact Newton candidate does not certify the original nonstationary point
    EXPECT_FALSE(result.converged());
    // the refusal consumes the one permitted outer iteration
    EXPECT_EQ(result.rejected_steps, 1u);
    // a rejected trial leaves the original point intact
    EXPECT_DOUBLE_EQ(result.point, -1.);
    // the stored certificate still belongs to the nonstationary accepted point
    EXPECT_DOUBLE_EQ(result.stationarity_norm, 2.);
    // rejected costs never trigger the optional stationarity callback
    EXPECT_EQ(problem.stationarity_evaluations, 1u);
}

// invalid stopping certificates cannot create convergence at either an initial or promoted point
TEST(SmoothingTrustRegion, RejectsInvalidStationarity) {
    using namespace fdapde::manifold;
    for (const double invalid :
         {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), -1.}) {
        InvalidStationarity problem;
        problem.invalid = invalid;
        const auto result = RiemannianTrustRegion().optimize(problem, Chart {}, -1.);
        // nonfinite and negative quantities are not valid stationarity norms
        EXPECT_FALSE(result.converged());
        // invalid certificates stop through the existing invalid-gradient diagnostic
        EXPECT_EQ(result.stop_reason, TrustRegionStopReason::non_finite_gradient);
        // the geometric gradient norm remains available independently of the invalid certificate
        EXPECT_DOUBLE_EQ(result.gradient_norm, 2.);
        // an invalid initial certificate prevents any trial step
        EXPECT_EQ(result.iterations, 0u);
    }
    InvalidStationarity promoted;
    promoted.initially_valid = true;
    TrustRegionOptions options;
    options.initial_radius = .2;
    const auto result = RiemannianTrustRegion(options).optimize(promoted, Chart {}, -1.);
    // an invalid certificate after promotion cannot turn an accepted nonstationary point into convergence
    EXPECT_FALSE(result.converged());
    // promoted invalid certificates use the same diagnostic as invalid initial certificates
    EXPECT_EQ(result.stop_reason, TrustRegionStopReason::non_finite_gradient);
    // the invalid certificate is checked immediately after the first accepted step
    EXPECT_EQ(result.accepted_steps, 1u);
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

/// @brief supplies an exact quadratic cost and gradient with an asymmetric Hessian approximation
struct InexactQuadratic {
    using Point = fdapde::Vector<double, 2>;
    /// @brief owns no candidate-specific state for the analytic oracle
    struct Workspace { };
    /// @brief evaluates the independent Euclidean quadratic oracle
    double cost(const Point& x, Workspace&) { return .5 * (x(0, 0) * x(0, 0) + x(1, 0) * x(1, 0)); }
    /// @brief returns the exact gradient of the quadratic cost
    Point grad(const Point& x, Workspace&) { return x; }
    /// @brief injects an antisymmetric error into the approximate Hessian action
    Point hess(const Point&, const Point& direction, Workspace&) {
        return Point(direction(0, 0) + 5 * direction(1, 0), direction(1, 0) - 5 * direction(0, 0));
    }
};

// a finite non-descent CG model recovers through a certified Cauchy trial without loosening stationarity
TEST(SmoothingTrustRegion, RecoversNonDescentInexactModel) {
    using namespace fdapde::manifold;
    InexactQuadratic problem;
    TrustRegionOptions options;
    options.initial_radius = options.maximum_radius = 10;
    options.gradient_tolerance = 1e-10;
    const auto result = RiemannianTrustRegion(options).optimize(
      problem, EuclideanGeometry<InexactQuadratic::Point> {}, InexactQuadratic::Point(1., 0.));
    // the exact quadratic reaches its original first-order convergence certificate
    ASSERT_TRUE(result.converged());
    // the stored norm meets the configured tolerance rather than a cost-decrease surrogate
    EXPECT_LE(result.gradient_norm, options.gradient_tolerance);
    // a spoiled CG model is replaced by an explicit Cauchy trial
    EXPECT_GT(result.cauchy_trials, 0u);
}
}   // namespace
