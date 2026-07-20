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

#ifndef __FDAPDE_MANIFOLD_PROBLEM_H__
#define __FDAPDE_MANIFOLD_PROBLEM_H__

#include "header_check.h"

namespace fdapde {
namespace manifold {

template <typename Problem> using workspace_t = typename std::remove_cvref_t<Problem>::Workspace;

template <typename Problem>
concept ProblemWorkspace = requires { typename workspace_t<Problem>; } &&
                           std::default_initializable<workspace_t<Problem>> && std::movable<workspace_t<Problem>>;

template <typename Problem, typename Geometry>
concept RiemannianCostProblem =
  FirstOrderGeometry<Geometry> && ProblemWorkspace<Problem> &&
  requires(Problem& problem, const point_t<Geometry>& point, workspace_t<Problem>& workspace) {
      { problem.cost(point, workspace) } -> std::convertible_to<double>;
  };

template <typename Problem, typename Geometry>
concept RiemannianGradientProblem =
  FirstOrderGeometry<Geometry> && ProblemWorkspace<Problem> &&
  requires(Problem& problem, const point_t<Geometry>& point, workspace_t<Problem>& workspace) {
      { problem.gradient(point, workspace) } -> std::same_as<tangent_t<Geometry>>;
  };

template <typename Problem, typename Geometry>
concept FirstOrderProblem = RiemannianCostProblem<Problem, Geometry> && RiemannianGradientProblem<Problem, Geometry>;

template <typename Problem, typename Geometry>
concept CombinedCostGradientProblem =
  FirstOrderProblem<Problem, Geometry> &&
  requires(Problem& problem, const point_t<Geometry>& point, workspace_t<Problem>& workspace) {
      { problem.cost_gradient(point, workspace) } -> std::same_as<std::pair<double, tangent_t<Geometry>>>;
  };

template <typename Problem, typename Geometry>
concept RiemannianHessianProblem =
  FirstOrderProblem<Problem, Geometry> && requires(
                                            Problem& problem, const point_t<Geometry>& point,
                                            const tangent_t<Geometry>& tangent, workspace_t<Problem>& workspace) {
      { problem.hessian_vector(point, tangent, workspace) } -> std::same_as<tangent_t<Geometry>>;
  };

// evaluation is a solver-owned cache slot already bound to point for its
// complete generation; these helpers never compare or hash manifold points
template <typename Problem, FirstOrderGeometry Geometry>
    requires CombinedCostGradientProblem<Problem, Geometry>
const typename EvaluationContext<tangent_t<Geometry>, workspace_t<Problem>>::Evaluation& evaluate_cost_gradient(
  Problem& problem, const Geometry&, const point_t<Geometry>& point,
  typename EvaluationContext<tangent_t<Geometry>, workspace_t<Problem>>::Evaluation& evaluation) {
    if (!evaluation.cost() || !evaluation.gradient()) {
        auto result = problem.cost_gradient(point, evaluation.workspace());
        evaluation.set_cost(result.first);
        evaluation.set_gradient(std::move(result.second));
    }
    return evaluation;
}

template <typename Problem, FirstOrderGeometry Geometry>
    requires FirstOrderProblem<Problem, Geometry>
double evaluate_cost(
  Problem& problem, const Geometry&, const point_t<Geometry>& point,
  typename EvaluationContext<tangent_t<Geometry>, workspace_t<Problem>>::Evaluation& evaluation) {
    if (!evaluation.cost()) { evaluation.set_cost(static_cast<double>(problem.cost(point, evaluation.workspace()))); }
    return *evaluation.cost();
}

template <typename Problem, FirstOrderGeometry Geometry>
    requires FirstOrderProblem<Problem, Geometry>
const tangent_t<Geometry>& evaluate_gradient(
  Problem& problem, const Geometry&, const point_t<Geometry>& point,
  typename EvaluationContext<tangent_t<Geometry>, workspace_t<Problem>>::Evaluation& evaluation) {
    if (!evaluation.gradient()) { evaluation.set_gradient(problem.gradient(point, evaluation.workspace())); }
    return *evaluation.gradient();
}

}   // namespace manifold
}   // namespace fdapde

#endif   // __FDAPDE_MANIFOLD_PROBLEM_H__
