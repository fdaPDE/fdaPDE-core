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
// examples/geometric_finite_elements/cheeger_p1_interpolation.cpp
#include <fdaPDE/geometric_finite_elements.h>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace fdapde;

// interpolates SPD matrices with a nodal C-LE parameter and differentiates its local scalar field
int main() {
    using Geometry = manifold::CheegerLogEuclideanSPDGeometry<double, 2, Usage::InterpolationNodes>;
    using Sym = Geometry::Tangent;
    const Geometry geometry = Geometry::from_rho(.25);

    // 1. generate three nearby noncommuting SPD matrices through their symmetric logarithms
    MatrixBatch<Geometry::Point> nodes(3);
    nodes[0] = matrix_exp(Sym(Vector<double, 3> {.8, .12, -.1}));
    nodes[1] = matrix_exp(Sym(Vector<double, 3> {.4, -.08, .6}));
    nodes[2] = matrix_exp(Sym(Vector<double, 3> {.6, .18, .2}));
    const std::array<double, 3> weights {.2, .3, .5}, rho {.25, .35, .45};

    // 2. bind rho in the same local node order; its value is interpolated with the scalar P1 weights
    gfe::P1GeodesicLinearizationOptions options;
    options.mean.solver.gradient_tolerance = 1e-11;
    options.linear_solve.residual_tolerance = 1e-11;
    const auto linearization = gfe::p1_geodesic_linearization(geometry, nodes, weights, options, rho);
    // convergence and the absence of detected ties are required to consume this selected local mean
    fdapde_strong_assert(
      linearization.result().converged() && !linearization.result().detected_ambiguity, std::runtime_error,
      "C-LE interpolation did not find a regular mean");
    double local_rho = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) local_rho += weights[i] * rho[i];
    // binding nodal rho must produce its weighted scalar value at the interpolation point
    fdapde_strong_assert(
      std::abs(linearization.geometry().rho() - local_rho) < 1e-14, std::runtime_error,
      "interpolated rho differs from the P1 formula");

    // 3. change nodal rho while keeping tensors and weights fixed; its local change is another P1 average
    const std::array<double, 3> rho_direction {.1, -.2, .3};
    const auto local_action = linearization.rho_jvp();
    const auto nodal_action = linearization.nodal_rho_jvp(rho_direction);
    // analytic actions must satisfy their derivative-solve certificates before use
    fdapde_strong_assert(
      local_action.converged() && nodal_action.converged(), std::runtime_error, "rho derivatives did not converge");
    double local_direction = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) local_direction += weights[i] * rho_direction[i];
    // nodal scalar sensitivity follows the chain rule through the interpolated local rho
    fdapde_strong_assert(
      (nodal_action.derivative - local_direction * local_action.derivative).norm() < 1e-12, std::runtime_error,
      "nodal rho derivative violates the P1 chain rule");

    // 4. refit nearby values only to verify the analytic derivative independently
    constexpr double step = 1e-5;
    const auto plus =
      gfe::p1_geodesic_value(Geometry::from_rho(local_rho + step * local_direction), nodes, weights, options.mean);
    const auto minus =
      gfe::p1_geodesic_value(Geometry::from_rho(local_rho - step * local_direction), nodes, weights, options.mean);
    // centered differences require both perturbed means to remain converged and free of detected ties
    fdapde_strong_assert(
      plus.converged() && minus.converged() && !plus.detected_ambiguity && !minus.detected_ambiguity,
      std::runtime_error, "perturbed C-LE means are not regular");
    const Sym difference((plus.value - minus.value) / (2 * step));
    const double derivative_error = (difference - nodal_action.derivative).norm();
    // the centered value difference checks the analytic rho action in ambient matrix coordinates
    fdapde_strong_assert(derivative_error < 1e-6, std::runtime_error, "rho derivative differs from centered values");

    // convergence describes this selected local branch and does not certify global uniqueness
    std::cout << std::scientific << std::setprecision(6) << "interpolated rho: " << local_rho << '\n'
              << "C-LE interpolated tensor:\n"
              << linearization.result().value << '\n'
              << "nodal rho derivative norm: " << nodal_action.derivative.norm() << '\n'
              << "rho derivative centered-difference error: " << derivative_error << '\n';
}
