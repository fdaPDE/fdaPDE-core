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

// compile from the repository root with c++ -std=c++20 -O2 -pthread -I.
// examples/manifold/so_product_optimization.cpp
#include <fdaPDE/manifold_geometry.h>
#include <fdaPDE/manifold_solvers.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace fdapde;
using namespace fdapde::manifold;

namespace {
using Factor = SOGeometry<double, 3>;
using Geometry = ProductGeometry<Factor>;
using Point = Geometry::Point;
using Dense = Matrix<double, 3, 3>;
using Ambient = MatrixBatch<Dense>;

/// @brief fits two rotations with a shared residual difference and an attainable zero-cost target
struct CoupledObjective {
    /// @brief provides independent candidate-local storage for callbacks without retained intermediates
    struct Workspace { };
    const Point& target;
    static constexpr double coupling = .5;

    /// @brief evaluates one half of the residual norm plus a coupled residual-difference penalty
    double cost(const Point& points, Workspace&) const {
        double value = 0;
        for (std::size_t i = 0; i < points.size(); ++i) value += .5 * (points[i] - target[i]).squared_norm();
        const auto difference = (points[0] - target[0]) - (points[1] - target[1]);
        return value + .5 * coupling * difference.squared_norm();
    }
    /// @brief differentiates the full coupled objective with respect to ambient rotation coefficients
    Ambient egrad(const Point& points, Workspace&) const {
        Ambient result(points.size());
        for (std::size_t i = 0; i < points.size(); ++i) result[i] = points[i] - target[i];
        const Dense difference(result[0] - result[1]);
        result[0] += coupling * difference;
        result[1] -= coupling * difference;
        return result;
    }
    /// @brief applies diagonal and mixed-factor ambient Hessian blocks to full matrix velocities
    Ambient ehess(const Point&, const Ambient& direction, Workspace&) const {
        Ambient result(direction);
        const Dense difference(direction[0] - direction[1]);
        result[0] += coupling * difference;
        result[1] -= coupling * difference;
        return result;
    }
};

/// @brief checks the known coupled minimizer and reports comparable solver certificates
void report_solution(const char* name, const auto& result, const Point& target) {
    // the solver must produce a finite first-order certificate rather than merely exhaust its iteration budget
    fdapde_strong_assert(
      result.converged() && std::isfinite(result.cost) && std::isfinite(result.gradient_norm), std::runtime_error,
      "joint rotation optimization did not converge");
    double error = 0;
    for (std::size_t i = 0; i < target.size(); ++i) error = std::max(error, (result.point[i] - target[i]).norm());
    // the independently prescribed target makes every nonnegative residual term vanish simultaneously
    fdapde_strong_assert(
      result.cost < 1e-15 && error < 1e-7, std::runtime_error, "joint solution differs from the known target");
    std::cout << name << ": iterations " << result.iterations << ", cost evaluations " << result.cost_evaluations
              << ", cost " << result.cost << ", gradient norm " << result.gradient_norm << ", maximum target error "
              << error << '\n';
}
}   // namespace

// solves one coupled objective using body skew tangents and ambient derivatives through both native optimizers
int main() {
    const Geometry geometry(Factor {}, 2);
    const Point initial(2);
    Point target(2);
    constexpr double first_angle = .35, second_angle = -.25;
    target[0] = Dense(
      {std::cos(first_angle), -std::sin(first_angle), 0., std::sin(first_angle), std::cos(first_angle), 0., 0., 0.,
       1.});
    target[1] = Dense(
      {std::cos(second_angle), 0., std::sin(second_angle), 0., 1., 0., -std::sin(second_angle), 0.,
       std::cos(second_angle)});
    CoupledObjective objective {target};
    CoupledObjective::Workspace workspace;

    // a product gradient is a packed skew batch even though egrad returns full ambient matrices
    const auto gradient = geometry.euclidean_to_riemannian_gradient(initial, objective.egrad(initial, workspace));
    const auto zero = geometry.zero_tangent(initial);
    const auto direction = geometry.linear_combination(initial, -1., gradient, 0., zero);
    const auto velocity = geometry.to_ambient(initial, direction);
    const double body_norm = geometry.norm(initial, direction);
    const double ambient_norm =
      velocity.redux(0., [](double value, const auto& matrix) { return std::hypot(value, matrix.norm()); });
    // the product metric counts all ambient entries even though each body tangent stores only three coefficients
    fdapde_strong_assert(
      std::isfinite(body_norm) && std::abs(body_norm - ambient_norm) < 1e-13, std::runtime_error,
      "product body and ambient norms disagree");

    // changing only the first factor still changes the second component of this objective's Hessian action
    auto probe = geometry.zero_tangent(initial);
    probe[0](0, 1) = .2;
    const auto ambient_probe = geometry.to_ambient(initial, probe);
    const auto action = objective.ehess(initial, ambient_probe, workspace);
    // the nonzero off-diagonal Hessian block is exactly minus the coupling weight times the first velocity
    fdapde_strong_assert(
      action[1].norm() > 0 && (action[1] + CoupledObjective::coupling * ambient_probe[0]).norm() < 1e-14,
      std::runtime_error, "cross-factor Hessian term was lost");

    std::cout << std::scientific << std::setprecision(6)
              << "two SO(3) factors, product dimension: " << geometry.dimension() << '\n'
              << "stored body coefficients per factor: " << direction.coefficient_stride() << '\n'
              << "objective: 0.5 sum ||Q_i - T_i||_F^2 + 0.25 ||(Q_0-T_0)-(Q_1-T_1)||_F^2\n"
              << "initial cost: " << objective.cost(initial, workspace) << '\n'
              << "descent body norm: " << body_norm << ", ambient velocity norm: " << ambient_norm << '\n'
              << "cross-factor Hessian action norm: " << action[1].norm() << '\n';

    SteepestDescentOptions descent_options;
    descent_options.gradient_tolerance = 1e-9;
    // the ambient quadratic has largest curvature two, so a half step avoids nearly oscillatory unit updates
    descent_options.line_search.initial_step = .5;
    const auto descent = RiemannianSteepestDescent(descent_options).optimize(objective, geometry, initial);
    // Armijo descent must reach the known simultaneous zero of both coupled residuals
    report_solution("steepest descent", descent, target);

    TrustRegionOptions trust_options;
    trust_options.gradient_tolerance = 1e-10;
    const auto trust = RiemannianTrustRegion(trust_options).optimize(objective, geometry, initial);
    // trust regions use the same complete ambient objective including its mixed Hessian blocks
    report_solution("trust region", trust, target);
}
