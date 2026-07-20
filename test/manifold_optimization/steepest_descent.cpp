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

#include <cmath>
#include <limits>
#include <stdexcept>

namespace {

struct SolverEuclideanGeometry {
    using Point = double;
    using Tangent = double;

    std::size_t dimension() const { return 1; }
    double inner_product(const Point&, const Tangent& u, const Tangent& v) const { return u * v; }
    double norm(const Point&, const Tangent& u) const { return std::abs(u); }
    Tangent project(const Point&, const Tangent& u) const { return u; }
    Tangent zero_tangent(const Point&) const { return 0; }
    Tangent linear_combination(const Point&, double alpha, const Tangent& u, double beta, const Tangent& v) const {
        return alpha * u + beta * v;
    }
    Point retract(const Point& point, const Tangent& tangent, double step) const { return point + step * tangent; }
};

struct SolverWorkspace {
    bool has_point = false;
    double point = 0;
};

struct QuadraticProblem {
    using Workspace = SolverWorkspace;

    int cost_calls = 0;
    int gradient_calls = 0;
    int workspace_reuses = 0;

    double cost(double point, Workspace& workspace) {
        ++cost_calls;
        workspace.has_point = true;
        workspace.point = point;
        return 0.5 * point * point;
    }
    double gradient(double point, Workspace& workspace) {
        ++gradient_calls;
        if (workspace.has_point && workspace.point == point) ++workspace_reuses;
        return point;
    }
};

struct CombinedQuadraticProblem : QuadraticProblem {
    int combined_calls = 0;

    std::pair<double, double> cost_gradient(double point, Workspace& workspace) {
        ++combined_calls;
        workspace.has_point = true;
        workspace.point = point;
        return {0.5 * point * point, point};
    }
};

struct NonFiniteCostProblem : QuadraticProblem {
    double cost(double point, Workspace& workspace) {
        QuadraticProblem::cost(point, workspace);
        return std::numeric_limits<double>::infinity();
    }
};

struct NonFiniteGradientProblem : QuadraticProblem {
    double gradient(double point, Workspace& workspace) {
        QuadraticProblem::gradient(point, workspace);
        return std::numeric_limits<double>::infinity();
    }
};

struct PostStepNonFiniteGradientProblem : QuadraticProblem {
    double gradient(double point, Workspace& workspace) {
        QuadraticProblem::gradient(point, workspace);
        return point == 0 ? std::numeric_limits<double>::infinity() : point;
    }
};

struct ConstantCostProblem {
    using Workspace = SolverWorkspace;

    double value = 0;
    int cost_calls = 0;

    double cost(double point, Workspace& workspace) {
        ++cost_calls;
        workspace.has_point = true;
        workspace.point = point;
        return value;
    }
    double gradient(double, Workspace&) { return 0; }
};

struct NonFiniteTrialProblem : QuadraticProblem {
    double cost(double point, Workspace& workspace) {
        QuadraticProblem::cost(point, workspace);
        return point == 1 ? 0.5 : std::numeric_limits<double>::infinity();
    }
};

fdapde::manifold::ArmijoOptions rejecting_armijo_options() {
    fdapde::manifold::ArmijoOptions options;
    options.initial_step = 4;
    options.contraction = 0.5;
    options.sufficient_decrease = 1e-4;
    options.minimum_step = 1e-12;
    options.max_trials = 10;
    return options;
}

}   // namespace

TEST(ManifoldSteepestDescent, ConvergesAndPromotesAcceptedTrialCache) {
    SolverEuclideanGeometry geometry;
    QuadraticProblem problem;
    fdapde::manifold::SteepestDescentOptions options;
    options.max_iterations = 10;
    options.gradient_tolerance = 1e-12;
    options.line_search = rejecting_armijo_options();
    fdapde::manifold::RiemannianSteepestDescent optimizer(options);

    auto result = optimizer.optimize(problem, geometry, 1.0);

    EXPECT_TRUE(result.converged());
    EXPECT_EQ(result.stop_reason, fdapde::manifold::SteepestDescentStopReason::gradient_tolerance);
    EXPECT_EQ(result.line_search_status, fdapde::manifold::ArmijoStatus::accepted);
    EXPECT_DOUBLE_EQ(result.point, 0);
    EXPECT_DOUBLE_EQ(result.cost, 0);
    EXPECT_DOUBLE_EQ(result.gradient_norm, 0);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_EQ(result.cost_evaluations, 4);
    EXPECT_EQ(result.gradient_evaluations, 2);
    EXPECT_EQ(result.rejected_trials, 2);
    EXPECT_EQ(problem.cost_calls, 4);
    EXPECT_EQ(problem.gradient_calls, 2);
    EXPECT_EQ(problem.workspace_reuses, 2);
}

TEST(ManifoldSteepestDescent, UsesCombinedInitialEvaluationAndExactCounters) {
    SolverEuclideanGeometry geometry;
    CombinedQuadraticProblem problem;
    fdapde::manifold::RiemannianSteepestDescent optimizer;

    auto result = optimizer.optimize(problem, geometry, 1.0);

    EXPECT_TRUE(result.converged());
    EXPECT_DOUBLE_EQ(result.point, 0);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_EQ(result.cost_evaluations, 2);
    EXPECT_EQ(result.gradient_evaluations, 2);
    EXPECT_EQ(result.rejected_trials, 0);
    EXPECT_EQ(problem.combined_calls, 1);
    EXPECT_EQ(problem.cost_calls, 1);
    EXPECT_EQ(problem.gradient_calls, 1);
    EXPECT_EQ(problem.workspace_reuses, 1);
}

TEST(ManifoldSteepestDescent, ReportsLineSearchNotRunAtInitialSolution) {
    SolverEuclideanGeometry geometry;
    QuadraticProblem problem;
    fdapde::manifold::RiemannianSteepestDescent optimizer;

    auto result = optimizer.optimize(problem, geometry, 0.0);
    fdapde::manifold::ArmijoResult<double> idle;

    EXPECT_TRUE(result.converged());
    EXPECT_EQ(result.line_search_status, fdapde::manifold::ArmijoStatus::not_run);
    EXPECT_EQ(idle.status, fdapde::manifold::ArmijoStatus::not_run);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(result.cost_evaluations, 1);
    EXPECT_EQ(result.gradient_evaluations, 1);
}

TEST(ManifoldSteepestDescent, ReportsNonFiniteInitialCostAndGradient) {
    SolverEuclideanGeometry geometry;

    NonFiniteCostProblem cost_problem;
    fdapde::manifold::RiemannianSteepestDescent optimizer;
    auto cost_result = optimizer.optimize(cost_problem, geometry, 1.0);
    EXPECT_EQ(cost_result.stop_reason, fdapde::manifold::SteepestDescentStopReason::non_finite_cost);
    EXPECT_EQ(cost_result.line_search_status, fdapde::manifold::ArmijoStatus::not_run);
    EXPECT_EQ(cost_result.cost_evaluations, 1);
    EXPECT_EQ(cost_result.gradient_evaluations, 0);
    EXPECT_EQ(cost_problem.cost_calls, 1);
    EXPECT_EQ(cost_problem.gradient_calls, 0);

    NonFiniteGradientProblem gradient_problem;
    auto gradient_result = optimizer.optimize(gradient_problem, geometry, 1.0);
    EXPECT_EQ(gradient_result.stop_reason, fdapde::manifold::SteepestDescentStopReason::non_finite_gradient);
    EXPECT_EQ(gradient_result.line_search_status, fdapde::manifold::ArmijoStatus::not_run);
    EXPECT_EQ(gradient_result.cost_evaluations, 1);
    EXPECT_EQ(gradient_result.gradient_evaluations, 1);
    EXPECT_EQ(gradient_problem.cost_calls, 1);
    EXPECT_EQ(gradient_problem.gradient_calls, 1);
}

TEST(ManifoldSteepestDescent, ReportsNonFiniteGradientAfterAcceptedStep) {
    SolverEuclideanGeometry geometry;
    PostStepNonFiniteGradientProblem problem;
    fdapde::manifold::RiemannianSteepestDescent optimizer;

    auto result = optimizer.optimize(problem, geometry, 1.0);

    EXPECT_FALSE(result.converged());
    EXPECT_EQ(result.stop_reason, fdapde::manifold::SteepestDescentStopReason::non_finite_gradient);
    EXPECT_EQ(result.line_search_status, fdapde::manifold::ArmijoStatus::accepted);
    EXPECT_DOUBLE_EQ(result.point, 0);
    EXPECT_DOUBLE_EQ(result.cost, 0);
    EXPECT_TRUE(std::isinf(result.gradient_norm));
    EXPECT_EQ(result.iterations, 1);
    EXPECT_EQ(result.cost_evaluations, 2);
    EXPECT_EQ(result.gradient_evaluations, 2);
    EXPECT_EQ(problem.cost_calls, 2);
    EXPECT_EQ(problem.gradient_calls, 2);
}

TEST(ManifoldSteepestDescent, ReportsMaximumIterationsAfterAcceptedStep) {
    SolverEuclideanGeometry geometry;
    QuadraticProblem problem;
    fdapde::manifold::SteepestDescentOptions options;
    options.max_iterations = 1;
    options.gradient_tolerance = 0;
    options.line_search.initial_step = 0.5;
    fdapde::manifold::RiemannianSteepestDescent optimizer(options);

    auto result = optimizer.optimize(problem, geometry, 1.0);

    EXPECT_FALSE(result.converged());
    EXPECT_EQ(result.stop_reason, fdapde::manifold::SteepestDescentStopReason::max_iterations);
    EXPECT_EQ(result.line_search_status, fdapde::manifold::ArmijoStatus::accepted);
    EXPECT_DOUBLE_EQ(result.point, 0.5);
    EXPECT_DOUBLE_EQ(result.cost, 0.125);
    EXPECT_DOUBLE_EQ(result.gradient_norm, 0.5);
    EXPECT_EQ(result.iterations, 1);
    EXPECT_EQ(result.cost_evaluations, 2);
    EXPECT_EQ(result.gradient_evaluations, 2);
    EXPECT_EQ(result.rejected_trials, 0);
    EXPECT_EQ(problem.cost_calls, 2);
    EXPECT_EQ(problem.gradient_calls, 2);
}

TEST(ManifoldSteepestDescent, ReportsConvergenceReachedOnTheLastAllowedStep) {
    SolverEuclideanGeometry geometry;
    QuadraticProblem problem;
    fdapde::manifold::SteepestDescentOptions options;
    options.max_iterations = 1;
    options.gradient_tolerance = 0;
    fdapde::manifold::RiemannianSteepestDescent optimizer(options);

    auto result = optimizer.optimize(problem, geometry, 1.0);

    EXPECT_TRUE(result.converged());
    EXPECT_EQ(result.stop_reason, fdapde::manifold::SteepestDescentStopReason::gradient_tolerance);
    EXPECT_EQ(result.line_search_status, fdapde::manifold::ArmijoStatus::accepted);
    EXPECT_DOUBLE_EQ(result.point, 0);
    EXPECT_DOUBLE_EQ(result.gradient_norm, 0);
    EXPECT_EQ(result.iterations, 1);
}

TEST(ManifoldSteepestDescent, SurfacesBoundedLineSearchFailure) {
    SolverEuclideanGeometry geometry;
    QuadraticProblem problem;
    fdapde::manifold::SteepestDescentOptions options;
    options.line_search = rejecting_armijo_options();
    options.line_search.max_trials = 1;
    fdapde::manifold::RiemannianSteepestDescent optimizer(options);

    auto result = optimizer.optimize(problem, geometry, 1.0);

    EXPECT_FALSE(result.converged());
    EXPECT_EQ(result.stop_reason, fdapde::manifold::SteepestDescentStopReason::line_search_failed);
    EXPECT_EQ(result.line_search_status, fdapde::manifold::ArmijoStatus::max_trials);
    EXPECT_DOUBLE_EQ(result.point, 1);
    EXPECT_DOUBLE_EQ(result.cost, 0.5);
    EXPECT_DOUBLE_EQ(result.gradient_norm, 1);
    EXPECT_EQ(result.iterations, 0);
    EXPECT_EQ(result.cost_evaluations, 2);
    EXPECT_EQ(result.gradient_evaluations, 1);
    EXPECT_EQ(result.rejected_trials, 1);
    EXPECT_EQ(problem.cost_calls, 2);
    EXPECT_EQ(problem.gradient_calls, 1);
}

TEST(ManifoldArmijo, FailedSearchInvalidatesLastRejectedTrial) {
    SolverEuclideanGeometry geometry;
    NonFiniteTrialProblem problem;
    auto options = rejecting_armijo_options();
    options.max_trials = 2;
    fdapde::manifold::ArmijoBacktracking line_search(options);
    fdapde::manifold::EvaluationContext<double, SolverWorkspace> context;
    context.trial().set_cost(123);
    context.trial().workspace().has_point = true;
    const std::size_t stale_generation = context.trial().generation();

    auto result = line_search.search(problem, geometry, 1.0, -1.0, 0.5, -1.0, context);

    EXPECT_FALSE(result.accepted());
    EXPECT_EQ(result.status, fdapde::manifold::ArmijoStatus::max_trials);
    EXPECT_EQ(result.trials, 2);
    EXPECT_EQ(result.rejected_trials, 2);
    EXPECT_DOUBLE_EQ(result.step, 2);
    EXPECT_FALSE(result.point);
    EXPECT_TRUE(std::isnan(result.cost));
    EXPECT_EQ(problem.cost_calls, 2);
    EXPECT_GT(context.trial().generation(), stale_generation);
    EXPECT_FALSE(context.trial().cost());
    EXPECT_FALSE(context.trial().gradient());
    EXPECT_FALSE(context.trial().workspace().has_point);
}

TEST(ManifoldArmijo, MinimumStepReportsTheLastEvaluatedStepAndClearsTrial) {
    SolverEuclideanGeometry geometry;
    QuadraticProblem problem;
    auto options = rejecting_armijo_options();
    options.minimum_step = 3;
    fdapde::manifold::ArmijoBacktracking line_search(options);
    fdapde::manifold::EvaluationContext<double, SolverWorkspace> context;

    auto result = line_search.search(problem, geometry, 1.0, -1.0, 0.5, -1.0, context);

    EXPECT_FALSE(result.accepted());
    EXPECT_EQ(result.status, fdapde::manifold::ArmijoStatus::minimum_step);
    EXPECT_EQ(result.trials, 1);
    EXPECT_EQ(result.rejected_trials, 1);
    EXPECT_DOUBLE_EQ(result.step, 4);
    EXPECT_FALSE(result.point);
    EXPECT_TRUE(std::isnan(result.cost));
    EXPECT_EQ(problem.cost_calls, 1);
    EXPECT_FALSE(context.trial().cost());
    EXPECT_FALSE(context.trial().gradient());
    EXPECT_FALSE(context.trial().workspace().has_point);
}

TEST(ManifoldArmijo, InvalidCurrentCostAndDirectionAreRejectedWithoutEvaluation) {
    SolverEuclideanGeometry geometry;
    QuadraticProblem problem;
    fdapde::manifold::ArmijoBacktracking line_search;
    fdapde::manifold::EvaluationContext<double, SolverWorkspace> context;
    context.trial().set_cost(123);

    auto non_finite =
      line_search.search(problem, geometry, 1.0, -1.0, std::numeric_limits<double>::infinity(), -1.0, context);
    EXPECT_EQ(non_finite.status, fdapde::manifold::ArmijoStatus::non_finite_current_cost);
    EXPECT_EQ(non_finite.trials, 0);
    EXPECT_FALSE(non_finite.accepted());
    EXPECT_FALSE(non_finite.point);
    EXPECT_TRUE(std::isnan(non_finite.cost));
    EXPECT_FALSE(context.trial().cost());

    context.trial().set_cost(456);
    auto non_descent = line_search.search(problem, geometry, 1.0, 1.0, 0.5, 0.0, context);
    EXPECT_EQ(non_descent.status, fdapde::manifold::ArmijoStatus::non_descent_direction);
    EXPECT_EQ(non_descent.trials, 0);
    EXPECT_FALSE(non_descent.accepted());
    EXPECT_FALSE(non_descent.point);
    EXPECT_TRUE(std::isnan(non_descent.cost));
    EXPECT_FALSE(context.trial().cost());

    context.trial().set_cost(789);
    auto non_finite_derivative =
      line_search.search(problem, geometry, 1.0, -1.0, 0.5, std::numeric_limits<double>::quiet_NaN(), context);
    EXPECT_EQ(non_finite_derivative.status, fdapde::manifold::ArmijoStatus::non_descent_direction);
    EXPECT_EQ(non_finite_derivative.trials, 0);
    EXPECT_FALSE(non_finite_derivative.accepted());
    EXPECT_FALSE(non_finite_derivative.point);
    EXPECT_TRUE(std::isnan(non_finite_derivative.cost));
    EXPECT_FALSE(context.trial().cost());
    EXPECT_EQ(problem.cost_calls, 0);
}

TEST(ManifoldArmijo, FusedDecreaseBoundAvoidsIntermediateOverflow) {
    SolverEuclideanGeometry geometry;
    ConstantCostProblem problem;
    const double maximum = std::numeric_limits<double>::max();
    problem.value = -maximum;
    fdapde::manifold::ArmijoOptions options;
    options.initial_step = maximum;
    options.sufficient_decrease = 0.5;
    options.minimum_step = 1;
    options.max_trials = 1;
    fdapde::manifold::ArmijoBacktracking line_search(options);
    fdapde::manifold::EvaluationContext<double, SolverWorkspace> context;

    auto result = line_search.search(problem, geometry, 0.0, 0.0, maximum, -4.0, context);

    EXPECT_TRUE(result.accepted());
    EXPECT_EQ(result.status, fdapde::manifold::ArmijoStatus::accepted);
    EXPECT_DOUBLE_EQ(result.cost, -maximum);
    EXPECT_DOUBLE_EQ(result.step, maximum);
    EXPECT_EQ(result.trials, 1);
}

TEST(ManifoldArmijo, FusedDecreaseBoundDoesNotAcceptAnUnchangedSubnormalCost) {
    SolverEuclideanGeometry geometry;
    ConstantCostProblem problem;
    const double minimum = std::numeric_limits<double>::denorm_min();
    problem.value = minimum;
    fdapde::manifold::ArmijoOptions options;
    options.initial_step = minimum;
    options.sufficient_decrease = 0.5;
    options.minimum_step = minimum;
    options.max_trials = 1;
    fdapde::manifold::ArmijoBacktracking line_search(options);
    fdapde::manifold::EvaluationContext<double, SolverWorkspace> context;

    auto result = line_search.search(problem, geometry, 0.0, 0.0, minimum, -2.0, context);

    EXPECT_FALSE(result.accepted());
    EXPECT_EQ(result.status, fdapde::manifold::ArmijoStatus::max_trials);
    EXPECT_EQ(result.trials, 1);
    EXPECT_EQ(result.rejected_trials, 1);
}

TEST(ManifoldArmijo, RoundedDecreaseBoundDoesNotAcceptAnUnchangedQuadraticPoint) {
    SolverEuclideanGeometry geometry;
    QuadraticProblem problem;
    const double minimum = std::numeric_limits<double>::denorm_min();
    fdapde::manifold::ArmijoOptions options;
    options.initial_step = minimum;
    options.minimum_step = minimum;
    options.max_trials = 1;
    fdapde::manifold::ArmijoBacktracking line_search(options);
    fdapde::manifold::EvaluationContext<double, SolverWorkspace> context;

    auto result = line_search.search(problem, geometry, 1.0, -1.0, 0.5, -1.0, context);

    EXPECT_FALSE(result.accepted());
    EXPECT_EQ(result.status, fdapde::manifold::ArmijoStatus::max_trials);
    EXPECT_EQ(result.trials, 1);
    EXPECT_EQ(result.rejected_trials, 1);
    EXPECT_EQ(problem.cost_calls, 1);
    EXPECT_FALSE(context.trial().cost());
}

TEST(ManifoldSolverOptions, InvalidBoundsAndTolerancesAreRejected) {
    auto armijo = rejecting_armijo_options();
    armijo.initial_step = 0;
    EXPECT_THROW(fdapde::manifold::ArmijoBacktracking {armijo}, std::invalid_argument);

    armijo = rejecting_armijo_options();
    armijo.initial_step = std::numeric_limits<double>::infinity();
    EXPECT_THROW(fdapde::manifold::ArmijoBacktracking {armijo}, std::invalid_argument);

    armijo = rejecting_armijo_options();
    armijo.contraction = 0;
    EXPECT_THROW(fdapde::manifold::ArmijoBacktracking {armijo}, std::invalid_argument);

    armijo = rejecting_armijo_options();
    armijo.contraction = 1;
    EXPECT_THROW(fdapde::manifold::ArmijoBacktracking {armijo}, std::invalid_argument);

    armijo = rejecting_armijo_options();
    armijo.sufficient_decrease = 0;
    EXPECT_THROW(fdapde::manifold::ArmijoBacktracking {armijo}, std::invalid_argument);

    armijo = rejecting_armijo_options();
    armijo.sufficient_decrease = 1;
    EXPECT_THROW(fdapde::manifold::ArmijoBacktracking {armijo}, std::invalid_argument);

    armijo = rejecting_armijo_options();
    armijo.minimum_step = 0;
    EXPECT_THROW(fdapde::manifold::ArmijoBacktracking {armijo}, std::invalid_argument);

    armijo = rejecting_armijo_options();
    armijo.minimum_step = 5;
    EXPECT_THROW(fdapde::manifold::ArmijoBacktracking {armijo}, std::invalid_argument);

    armijo = rejecting_armijo_options();
    armijo.max_trials = 0;
    EXPECT_THROW(fdapde::manifold::ArmijoBacktracking {armijo}, std::invalid_argument);

    armijo = rejecting_armijo_options();
    armijo.minimum_step = armijo.initial_step;
    fdapde::manifold::ArmijoBacktracking valid_line_search(armijo);
    EXPECT_DOUBLE_EQ(valid_line_search.options().minimum_step, armijo.initial_step);

    fdapde::manifold::SteepestDescentOptions options;
    options.max_iterations = 0;
    EXPECT_THROW(fdapde::manifold::RiemannianSteepestDescent {options}, std::invalid_argument);

    options = {};
    options.gradient_tolerance = -1;
    EXPECT_THROW(fdapde::manifold::RiemannianSteepestDescent {options}, std::invalid_argument);

    options.gradient_tolerance = std::numeric_limits<double>::infinity();
    EXPECT_THROW(fdapde::manifold::RiemannianSteepestDescent {options}, std::invalid_argument);

    options.gradient_tolerance = 0;
    fdapde::manifold::RiemannianSteepestDescent valid_optimizer(options);
    EXPECT_DOUBLE_EQ(valid_optimizer.options().gradient_tolerance, 0);
}
