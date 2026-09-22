// SPDX-License-Identifier: GPL-3.0-or-later
#include <fdaPDE/manifold_optimization.h>
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
    double gradient(double x, Workspace& workspace) {
        if (workspace.point) {
            // accepted gradients must see the cache promoted from the same trial point
            EXPECT_EQ(*workspace.point, x);
        }
        return x - 1;
    }
    /// @brief applies the constant scalar curvature
    double hessian_vector(double, double v, Workspace&) { return v; }
};
/// @brief supplies a scalar Hessian to test each truncated-CG termination path
struct Curvature : Quadratic {
    double coefficient = 1;
    /// @brief applies the requested scalar model curvature
    double hessian_vector(double, double v, Workspace&) { return coefficient * v; }
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
}   // namespace
