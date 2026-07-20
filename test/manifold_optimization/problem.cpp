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
#include <type_traits>
#include <utility>

namespace {

struct ProblemPoint {
    double value = 0;
};
struct ProblemTangent {
    double value = 0;
};

struct ProblemToyGeometry {
    using Point = ProblemPoint;
    using Tangent = ProblemTangent;

    std::size_t dimension() const { return 1; }
    double inner_product(const Point&, const Tangent& u, const Tangent& v) const { return u.value * v.value; }
    double norm(const Point& point, const Tangent& u) const { return std::sqrt(inner_product(point, u, u)); }
    Tangent project(const Point&, const Tangent& u) const { return u; }
    Tangent zero_tangent(const Point&) const { return {}; }
    Tangent linear_combination(const Point&, double alpha, const Tangent& u, double beta, const Tangent& v) const {
        return {alpha * u.value + beta * v.value};
    }
    Point retract(const Point& point, const Tangent& u, double step) const { return {point.value + step * u.value}; }
};

struct ProblemWorkspace {
    int evaluations = 0;
    double point = 0;
};

struct DirectToyProblem {
    using Workspace = ProblemWorkspace;
    int cost_calls = 0;
    int gradient_calls = 0;

    double cost(const ProblemPoint& point, Workspace& workspace) {
        ++cost_calls;
        ++workspace.evaluations;
        workspace.point = point.value;
        return 0.5 * point.value * point.value;
    }
    ProblemTangent gradient(const ProblemPoint& point, Workspace& workspace) {
        ++gradient_calls;
        ++workspace.evaluations;
        workspace.point = point.value;
        return {point.value};
    }
};

struct CombinedToyProblem : DirectToyProblem {
    int combined_calls = 0;
    std::pair<double, ProblemTangent> cost_gradient(const ProblemPoint& point, Workspace& workspace) {
        ++combined_calls;
        ++workspace.evaluations;
        workspace.point = point.value;
        return {0.5 * point.value * point.value, {point.value}};
    }
};

struct HessianToyProblem : DirectToyProblem {
    ProblemTangent hessian_vector(const ProblemPoint&, const ProblemTangent& tangent, Workspace&) { return tangent; }
};

struct MissingGradientProblem {
    using Workspace = ProblemWorkspace;
    double cost(const ProblemPoint& point, Workspace&) { return point.value; }
};

struct ConstToyProblem {
    using Workspace = ProblemWorkspace;

    double cost(const ProblemPoint& point, Workspace& workspace) const {
        workspace.point = point.value;
        return 0.5 * point.value * point.value;
    }
    ProblemTangent gradient(const ProblemPoint& point, Workspace&) const { return {point.value}; }
};

struct ThrowingMoveState {
    ThrowingMoveState() noexcept = default;
    ThrowingMoveState(const ThrowingMoveState&) = delete;
    ThrowingMoveState& operator=(const ThrowingMoveState&) = delete;
    ThrowingMoveState(ThrowingMoveState&&) noexcept(false) { }
    ThrowingMoveState& operator=(ThrowingMoveState&&) noexcept(false) { return *this; }
};

struct NothrowMoveOnlyState {
    NothrowMoveOnlyState() noexcept = default;
    NothrowMoveOnlyState(const NothrowMoveOnlyState&) = delete;
    NothrowMoveOnlyState& operator=(const NothrowMoveOnlyState&) = delete;
    NothrowMoveOnlyState(NothrowMoveOnlyState&&) noexcept = default;
    NothrowMoveOnlyState& operator=(NothrowMoveOnlyState&&) noexcept = default;
};

template <typename Gradient, typename Workspace>
concept SupportsEvaluationContext = requires { typename fdapde::manifold::EvaluationContext<Gradient, Workspace>; };

template <typename Context>
concept PermitsRvalueCurrent = requires(Context&& context) { std::move(context).current(); };

template <typename Evaluation>
concept PermitsRvalueCost = requires(Evaluation&& evaluation) { std::move(evaluation).cost(); };

template <typename Evaluation>
concept PermitsRvalueGradient = requires(Evaluation&& evaluation) { std::move(evaluation).gradient(); };

template <typename Evaluation>
concept PermitsRvalueWorkspace = requires(Evaluation&& evaluation) { std::move(evaluation).workspace(); };

using MoveOnlyContext = fdapde::manifold::EvaluationContext<NothrowMoveOnlyState, NothrowMoveOnlyState>;
using MoveOnlyEvaluation = typename MoveOnlyContext::Evaluation;

static_assert(fdapde::manifold::FirstOrderProblem<DirectToyProblem, ProblemToyGeometry>);
static_assert(!fdapde::manifold::FirstOrderProblem<const DirectToyProblem, ProblemToyGeometry>);
static_assert(fdapde::manifold::FirstOrderProblem<const ConstToyProblem, ProblemToyGeometry>);
static_assert(!fdapde::manifold::CombinedCostGradientProblem<DirectToyProblem, ProblemToyGeometry>);
static_assert(fdapde::manifold::CombinedCostGradientProblem<CombinedToyProblem, ProblemToyGeometry>);
static_assert(fdapde::manifold::RiemannianHessianProblem<HessianToyProblem, ProblemToyGeometry>);
static_assert(!fdapde::manifold::RiemannianHessianProblem<DirectToyProblem, ProblemToyGeometry>);
static_assert(!fdapde::manifold::FirstOrderProblem<MissingGradientProblem, ProblemToyGeometry>);
static_assert(SupportsEvaluationContext<ThrowingMoveState, NothrowMoveOnlyState>);
static_assert(SupportsEvaluationContext<NothrowMoveOnlyState, ThrowingMoveState>);
static_assert(std::is_default_constructible_v<MoveOnlyContext>);
static_assert(!PermitsRvalueCurrent<MoveOnlyContext>);
static_assert(!PermitsRvalueCost<MoveOnlyEvaluation>);
static_assert(!PermitsRvalueGradient<MoveOnlyEvaluation>);
static_assert(!PermitsRvalueWorkspace<MoveOnlyEvaluation>);

}   // namespace

TEST(ManifoldProblem, DirectEvaluationsAreCachedOnce) {
    ProblemToyGeometry geometry;
    DirectToyProblem problem;
    ProblemPoint point {3};
    fdapde::manifold::EvaluationContext<ProblemTangent, ProblemWorkspace> context;

    EXPECT_DOUBLE_EQ(fdapde::manifold::evaluate_cost(problem, geometry, point, context.current()), 4.5);
    EXPECT_DOUBLE_EQ(fdapde::manifold::evaluate_cost(problem, geometry, point, context.current()), 4.5);
    EXPECT_DOUBLE_EQ(fdapde::manifold::evaluate_gradient(problem, geometry, point, context.current()).value, 3);
    EXPECT_DOUBLE_EQ(fdapde::manifold::evaluate_gradient(problem, geometry, point, context.current()).value, 3);
    EXPECT_EQ(problem.cost_calls, 1);
    EXPECT_EQ(problem.gradient_calls, 1);
    EXPECT_EQ(context.current().workspace().evaluations, 2);
}

TEST(ManifoldProblem, CombinedEvaluationSharesCostAndGradient) {
    ProblemToyGeometry geometry;
    CombinedToyProblem problem;
    ProblemPoint point {4};
    fdapde::manifold::EvaluationContext<ProblemTangent, ProblemWorkspace> context;

    const auto& evaluation = fdapde::manifold::evaluate_cost_gradient(problem, geometry, point, context.current());

    ASSERT_TRUE(evaluation.cost());
    ASSERT_TRUE(evaluation.gradient());
    EXPECT_DOUBLE_EQ(*evaluation.cost(), 8);
    EXPECT_DOUBLE_EQ(evaluation.gradient()->value, 4);
    EXPECT_DOUBLE_EQ(fdapde::manifold::evaluate_gradient(problem, geometry, point, context.current()).value, 4);
    EXPECT_DOUBLE_EQ(fdapde::manifold::evaluate_cost(problem, geometry, point, context.current()), 8);
    EXPECT_EQ(problem.combined_calls, 1);
    EXPECT_EQ(problem.cost_calls, 0);
    EXPECT_EQ(problem.gradient_calls, 0);
    EXPECT_EQ(context.current().workspace().evaluations, 1);
}

TEST(ManifoldProblem, AcceptedTrialIsPromotedWithItsWorkspace) {
    ProblemToyGeometry geometry;
    CombinedToyProblem problem;
    ProblemPoint point {5};
    fdapde::manifold::EvaluationContext<ProblemTangent, ProblemWorkspace> context;
    const std::size_t trial_generation = context.trial().generation();

    fdapde::manifold::evaluate_cost_gradient(problem, geometry, point, context.trial());
    context.promote_trial();

    EXPECT_EQ(context.current().generation(), trial_generation);
    ASSERT_TRUE(context.current().cost());
    ASSERT_TRUE(context.current().gradient());
    EXPECT_DOUBLE_EQ(*context.current().cost(), 12.5);
    EXPECT_DOUBLE_EQ(context.current().gradient()->value, 5);
    EXPECT_EQ(context.current().workspace().point, 5);
    EXPECT_GT(context.trial().generation(), trial_generation);
    EXPECT_FALSE(context.trial().cost());
    EXPECT_FALSE(context.trial().gradient());
    EXPECT_EQ(context.trial().workspace().evaluations, 0);
}

TEST(ManifoldProblem, RejectedTrialResetInvalidatesCachedData) {
    ProblemToyGeometry geometry;
    CombinedToyProblem problem;
    ProblemPoint current_point {2};
    ProblemPoint trial_point {7};
    fdapde::manifold::EvaluationContext<ProblemTangent, ProblemWorkspace> context;

    fdapde::manifold::evaluate_cost(problem, geometry, current_point, context.current());
    fdapde::manifold::evaluate_cost(problem, geometry, trial_point, context.trial());
    const std::size_t rejected_generation = context.trial().generation();
    context.reset_trial();

    EXPECT_GT(context.trial().generation(), rejected_generation);
    EXPECT_FALSE(context.trial().cost());
    EXPECT_FALSE(context.trial().gradient());
    EXPECT_EQ(context.trial().workspace().evaluations, 0);
    ASSERT_TRUE(context.current().cost());
    EXPECT_DOUBLE_EQ(*context.current().cost(), 2);
    EXPECT_EQ(context.current().workspace().point, 2);
    EXPECT_EQ(problem.cost_calls, 2);
    EXPECT_EQ(problem.gradient_calls, 0);
    EXPECT_EQ(problem.combined_calls, 0);

    const std::size_t current_generation = context.current().generation();
    context.reset_current();
    EXPECT_GT(context.current().generation(), current_generation);
    EXPECT_FALSE(context.current().cost());
    EXPECT_FALSE(context.current().gradient());
    EXPECT_EQ(context.current().workspace().evaluations, 0);
}

TEST(ManifoldProblem, ResettingAGenerationRebindsItToANewPoint) {
    ProblemToyGeometry geometry;
    DirectToyProblem problem;
    fdapde::manifold::EvaluationContext<ProblemTangent, ProblemWorkspace> context;

    EXPECT_DOUBLE_EQ(fdapde::manifold::evaluate_cost(problem, geometry, ProblemPoint {2}, context.current()), 2);
    const std::size_t first_generation = context.current().generation();
    context.reset_current();
    EXPECT_DOUBLE_EQ(fdapde::manifold::evaluate_cost(problem, geometry, ProblemPoint {6}, context.current()), 18);

    EXPECT_GT(context.current().generation(), first_generation);
    EXPECT_EQ(context.current().workspace().point, 6);
    EXPECT_EQ(problem.cost_calls, 2);
}

TEST(ManifoldProblem, ConstCallabilityIsPreservedByConceptsAndEvaluation) {
    ProblemToyGeometry geometry;
    const ConstToyProblem problem;
    ProblemPoint point {3};
    fdapde::manifold::EvaluationContext<ProblemTangent, ProblemWorkspace> context;

    EXPECT_DOUBLE_EQ(fdapde::manifold::evaluate_cost(problem, geometry, point, context.current()), 4.5);
    EXPECT_DOUBLE_EQ(fdapde::manifold::evaluate_gradient(problem, geometry, point, context.current()).value, 3);
    EXPECT_EQ(context.current().workspace().point, 3);
}
