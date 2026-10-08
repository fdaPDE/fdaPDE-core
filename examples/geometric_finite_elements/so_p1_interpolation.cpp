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
// examples/geometric_finite_elements/so_p1_interpolation.cpp
#include <fdaPDE/geometric_discretization.h>

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace fdapde;

// interpolates rotations on one triangle and differentiates its local nodal data and barycentric weights
int main() {
    using Geometry = manifold::SOGeometry<double, 3>;
    using Rotation = Geometry::Point;
    using Skew = Geometry::Tangent;
    const Geometry geometry;
    const Rotation identity = Rotation::Identity();

    // 1. create nearby rotations about the z axis; their mean angle has a closed-form oracle
    const std::array<double, 3> angles {-.2, .3, .5};
    MatrixBatch<Rotation> nodes(3);
    for (std::size_t i = 0; i < nodes.size(); ++i) {
        Skew generator;
        generator(0, 1) = -angles[i];
        nodes[i] = geometry.exponential(identity, generator);
    }

    // 2. supply the scalar P1 weights at an interior point, without requiring a mesh or FEM dependency
    const std::array<double, 3> weights {.2, .3, .5};
    const auto value = gfe::p1_geodesic_value(geometry, nodes, weights);
    // the mean solver must certify stationarity before its matrix can be consumed
    fdapde_strong_assert(value.converged(), std::runtime_error, "rotation mean did not converge");
    double expected_angle = 0;
    for (std::size_t i = 0; i < weights.size(); ++i) expected_angle += weights[i] * angles[i];
    Skew expected_generator;
    expected_generator(0, 1) = -expected_angle;
    const Rotation expected = geometry.exponential(identity, expected_generator);
    // commuting rotations in this regular neighborhood average their signed angles
    fdapde_strong_assert(
      (value.value - expected).norm() < 1e-11, std::runtime_error, "rotation mean differs from the angle oracle");

    // 3. retain the linearization while nodes stay alive and unchanged; repeated derivative calls reuse it
    const auto linearization = gfe::p1_geodesic_linearization(geometry, nodes, weights);
    const std::array<double, 3> weight_direction {-1., 1., 0.};
    const auto spatial = linearization.weight_jvp(weight_direction);
    // weight derivatives return a candidate together with a linear-solve certificate
    fdapde_strong_assert(spatial.converged(), std::runtime_error, "weight derivative did not converge");
    Skew expected_spatial;
    expected_spatial(0, 1) = -(angles[1] - angles[0]);
    // shifting weight from node zero to node one changes the mean angle by their angle difference
    fdapde_strong_assert(
      (spatial.derivative - expected_spatial).norm() < 1e-11, std::runtime_error,
      "weight derivative differs from the angle oracle");

    // 4. body tangents Omega represent ambient velocities Q * Omega, including for the output derivative
    // perturbations around different axes exercise noncommuting derivatives of the same-axis mean
    std::array<Skew, 3> directions;
    for (std::size_t i = 0; i < directions.size(); ++i) {
        directions[i](0, 1) = -.1 * (i + 1);
        directions[i](0, 2) = .08 * (i + 1);
        directions[i](1, 2) = .06 * (2. - i);
    }
    const auto nodal = linearization.nodal_jvp(directions);
    Skew dual = expected_spatial;
    dual(0, 2) = .3;
    dual(1, 2) = -.2;
    const auto pullback = linearization.nodal_vjp(dual);
    // both solves must succeed before the forward action and metric pullback can be paired
    fdapde_strong_assert(
      nodal.converged() && pullback.converged(), std::runtime_error, "nodal derivatives did not converge");
    double pairing = 0;
    for (std::size_t i = 0; i < nodes.size(); ++i)
        pairing += geometry.inner_product(nodes[i], directions[i], pullback.derivative[i]);
    const double adjoint_error =
      std::abs(pairing - geometry.inner_product(linearization.result().value, nodal.derivative, dual));
    // the full-Frobenius metric pairs the nodal pullback with the same variation as the forward derivative
    fdapde_strong_assert(adjoint_error < 1e-11, std::runtime_error, "rotation differential and pullback disagree");

    // 5. differentiate the weight action with respect to the nodes using the manifold connection
    // the mixed pullback sends its output dual back to the three body-coordinate nodal tangents
    const auto mixed = linearization.covariant_mixed_nodal_jvp(weight_direction, directions);
    const auto mixed_pullback = linearization.covariant_mixed_nodal_vjp(weight_direction, dual);
    // each mixed result certifies all three dependent linear solves before exposing a usable derivative
    fdapde_strong_assert(
      mixed.converged() && mixed_pullback.converged(), std::runtime_error, "mixed derivatives did not converge");
    // the chosen off-axis perturbations must generate a nontrivial variation of the spatial weight action
    fdapde_strong_assert(
      std::hypot(mixed.derivative(0, 2), mixed.derivative(1, 2)) > 1e-3, std::runtime_error,
      "off-axis mixed derivative unexpectedly vanished");
    double mixed_pairing = 0;
    for (std::size_t i = 0; i < nodes.size(); ++i)
        mixed_pairing += geometry.inner_product(nodes[i], directions[i], mixed_pullback.derivative[i]);
    const double mixed_adjoint_error =
      std::abs(mixed_pairing - geometry.inner_product(linearization.result().value, mixed.derivative, dual));
    // the mixed metric pullback must reproduce the independently computed forward mixed pairing
    fdapde_strong_assert(
      mixed_adjoint_error < 1e-11, std::runtime_error, "mixed rotation differential and pullback disagree");

    std::cout << std::scientific << std::setprecision(6)
              << "interpolated angle: " << std::atan2(value.value(1, 0), value.value(0, 0)) << '\n'
              << "mean rotation:\n"
              << value.value << '\n'
              << "weight derivative body norm: " << spatial.derivative.norm() << '\n'
              << "nodal adjoint residual: " << adjoint_error << '\n'
              << "mixed derivative body norm: " << mixed.derivative.norm() << '\n'
              << "mixed adjoint residual: " << mixed_adjoint_error << '\n';
}
