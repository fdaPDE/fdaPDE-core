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

/// @brief accepts an explicit intrinsic gradient through grad
template <typename Problem, typename Geometry>
concept RiemannianGradientProblem =
  FirstOrderGeometry<Geometry> && ProblemWorkspace<Problem> &&
  requires(Problem& problem, const point_t<Geometry>& point, workspace_t<Problem>& workspace) {
      { problem.grad(point, workspace) } -> std::same_as<tangent_t<Geometry>>;
  };

/// @brief accepts a joint cost and intrinsic gradient without requiring a separate gradient method
template <typename Problem, typename Geometry>
concept CombinedCostGradientProblem =
  RiemannianCostProblem<Problem, Geometry> &&
  requires(Problem& problem, const point_t<Geometry>& point, workspace_t<Problem>& workspace) {
      { problem.cost_gradient(point, workspace) } -> std::same_as<std::pair<double, tangent_t<Geometry>>>;
  };

/// @brief requires an owning ambient gradient and a geometry conversion to its tangent representation
template <typename Problem, typename Geometry>
concept EuclideanGradientProblem =
  FirstOrderGeometry<Geometry> && ProblemWorkspace<Problem> &&
  requires(
    Problem& problem, const Geometry& geometry, const point_t<Geometry>& point, workspace_t<Problem>& workspace) {
      requires std::movable<decltype(problem.egrad(point, workspace))>;
      {
          geometry.euclidean_to_riemannian_gradient(point, problem.egrad(point, workspace))
      } -> std::same_as<tangent_t<Geometry>>;
  };

/// @brief accepts intrinsic derivatives or an ambient gradient convertible by the geometry
template <typename Problem, typename Geometry>
concept FirstOrderProblem =
  RiemannianCostProblem<Problem, Geometry> &&
  (RiemannianGradientProblem<Problem, Geometry> || CombinedCostGradientProblem<Problem, Geometry> ||
   EuclideanGradientProblem<Problem, Geometry>);

/// @brief accepts an explicit intrinsic Hessian action through hess
template <typename Problem, typename Geometry>
concept RiemannianHessianProblem =
  FirstOrderProblem<Problem, Geometry> && requires(
                                            Problem& problem, const point_t<Geometry>& point,
                                            const tangent_t<Geometry>& tangent, workspace_t<Problem>& workspace) {
      { problem.hess(point, tangent, workspace) } -> std::same_as<tangent_t<Geometry>>;
  };

/// @brief represents a solver direction in the ambient space used by Euclidean derivatives
template <FirstOrderGeometry Geometry>
decltype(auto)
ambient_direction(const Geometry& geometry, const point_t<Geometry>& point, const tangent_t<Geometry>& tangent) {
    if constexpr (requires { geometry.to_ambient(point, tangent); })
        return geometry.to_ambient(point, tangent);
    else
        return (tangent);
}

/// @brief requires both ambient derivatives and the geometry's connection-aware Hessian conversion
template <typename Problem, typename Geometry>
concept EuclideanHessianProblem =
  FirstOrderProblem<Problem, Geometry> && EuclideanGradientProblem<Problem, Geometry> &&
  requires(
    Problem& problem, const Geometry& geometry, const point_t<Geometry>& point, const tangent_t<Geometry>& tangent,
    workspace_t<Problem>& workspace) {
      {
          geometry.euclidean_to_riemannian_hessian(
            point, problem.egrad(point, workspace),
            problem.ehess(point, ambient_direction(geometry, point, tangent), workspace), tangent)
      } -> std::same_as<tangent_t<Geometry>>;
  };

/// @brief admits second-order solvers when either Hessian representation is available
template <typename Problem, typename Geometry>
concept SecondOrderProblem = FirstOrderProblem<Problem, Geometry> && (RiemannianHessianProblem<Problem, Geometry> ||
                                                                      EuclideanHessianProblem<Problem, Geometry>);

namespace internals {
/// @brief preserves the existing cache type when no ambient gradient is supplied
template <typename Problem, typename Geometry, bool = EuclideanGradientProblem<Problem, Geometry>>
struct problem_euclidean_gradient {
    using type = tangent_t<Geometry>;
};
/// @brief retains the objective's owning ambient gradient type independently of tangent coordinates
template <typename Problem, typename Geometry> struct problem_euclidean_gradient<Problem, Geometry, true> {
    using type = std::remove_cvref_t<decltype(std::declval<Problem&>().egrad(
      std::declval<const point_t<Geometry>&>(), std::declval<workspace_t<Problem>&>()))>;
};
}   // namespace internals

template <typename Problem, typename Geometry>
using euclidean_gradient_t = typename internals::problem_euclidean_gradient<Problem, Geometry>::type;
template <typename Problem, typename Geometry>
using evaluation_context_t =
  EvaluationContext<tangent_t<Geometry>, workspace_t<Problem>, euclidean_gradient_t<Problem, Geometry>>;

// each evaluation slot belongs to one complete candidate generation
// resetting or promoting a slot invalidates or retains both gradient representations together
/// @brief caches joint intrinsic evaluations without overriding an explicit gradient method
template <typename Problem, FirstOrderGeometry Geometry>
    requires CombinedCostGradientProblem<Problem, Geometry>
const typename evaluation_context_t<Problem, Geometry>::Evaluation& evaluate_cost_gradient(
  Problem& problem, const Geometry&, const point_t<Geometry>& point,
  typename evaluation_context_t<Problem, Geometry>::Evaluation& evaluation) {
    if (!evaluation.cost() || !evaluation.gradient()) {
        auto result = problem.cost_gradient(point, evaluation.workspace());
        evaluation.set_cost(result.first);
        if constexpr (RiemannianGradientProblem<Problem, Geometry>)
            evaluation.set_gradient(problem.grad(point, evaluation.workspace()));
        else
            evaluation.set_gradient(std::move(result.second));
    }
    return evaluation;
}

/// @brief evaluates the scalar objective once per candidate generation
template <typename Problem, FirstOrderGeometry Geometry>
    requires FirstOrderProblem<Problem, Geometry>
double evaluate_cost(
  Problem& problem, const Geometry&, const point_t<Geometry>& point,
  typename evaluation_context_t<Problem, Geometry>::Evaluation& evaluation) {
    if (!evaluation.cost()) evaluation.set_cost(static_cast<double>(problem.cost(point, evaluation.workspace())));
    return *evaluation.cost();
}

/// @brief reuses an ambient gradient across Riemannian conversions and Hessian directions
template <typename Problem, FirstOrderGeometry Geometry>
    requires EuclideanGradientProblem<Problem, Geometry>
const euclidean_gradient_t<Problem, Geometry>& evaluate_euclidean_gradient(
  Problem& problem, const Geometry&, const point_t<Geometry>& point,
  typename evaluation_context_t<Problem, Geometry>::Evaluation& evaluation) {
    if (!evaluation.euclidean_gradient())
        evaluation.set_euclidean_gradient(problem.egrad(point, evaluation.workspace()));
    return *evaluation.euclidean_gradient();
}

/// @brief prefers an explicit intrinsic gradient and otherwise converts the cached ambient gradient
template <typename Problem, FirstOrderGeometry Geometry>
    requires FirstOrderProblem<Problem, Geometry>
const tangent_t<Geometry>& evaluate_gradient(
  Problem& problem, const Geometry& geometry, const point_t<Geometry>& point,
  typename evaluation_context_t<Problem, Geometry>::Evaluation& evaluation) {
    if (!evaluation.gradient()) {
        if constexpr (RiemannianGradientProblem<Problem, Geometry>)
            evaluation.set_gradient(problem.grad(point, evaluation.workspace()));
        else if constexpr (CombinedCostGradientProblem<Problem, Geometry>)
            evaluate_cost_gradient(problem, geometry, point, evaluation);
        else
            evaluation.set_gradient(geometry.euclidean_to_riemannian_gradient(
              point, evaluate_euclidean_gradient(problem, geometry, point, evaluation)));
    }
    return *evaluation.gradient();
}

/// @brief prefers an intrinsic Hessian action and otherwise converts ambient derivatives along this direction
template <typename Problem, FirstOrderGeometry Geometry>
    requires SecondOrderProblem<Problem, Geometry>
tangent_t<Geometry> evaluate_hessian_vector(
  Problem& problem, const Geometry& geometry, const point_t<Geometry>& point, const tangent_t<Geometry>& direction,
  workspace_t<Problem>& workspace, const euclidean_gradient_t<Problem, Geometry>* euclidean_gradient = nullptr) {
    if constexpr (RiemannianHessianProblem<Problem, Geometry>)
        return problem.hess(point, direction, workspace);
    else {
        decltype(auto) ambient = ambient_direction(geometry, point, direction);
        const auto ehess = problem.ehess(point, ambient, workspace);
        if (euclidean_gradient)
            return geometry.euclidean_to_riemannian_hessian(point, *euclidean_gradient, ehess, direction);
        const auto egrad = problem.egrad(point, workspace);
        return geometry.euclidean_to_riemannian_hessian(point, egrad, ehess, direction);
    }
}

}   // namespace manifold
}   // namespace fdapde

#endif   // __FDAPDE_MANIFOLD_PROBLEM_H__
