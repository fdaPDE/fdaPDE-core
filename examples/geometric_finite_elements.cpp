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

// compile from the core root with c++ -std=c++20 -O2 -pthread -I. -I/path/to/eigen3
// examples/geometric_finite_elements.cpp
#include <fdaPDE/geometric_finite_elements_fem.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <utility>

using namespace fdapde;
using namespace fdapde::manifold;

// follows the construction, evaluation, update and differentiation of one SPD-valued P1 field
int main() {
    // these methods support multithreading; use four threads for calls with execution_par
    parallel_set_num_threads(4);

    // 1. choose the matrix type and interpolation metric; caches retain logs and factors reused by derivatives
    // log-Euclidean interpolation is the exponential of the P1-weighted nodal logarithms
    using Policy = Cache::Union<Cache::Spectral, Cache::Log, Cache::LogDividedDifferences>;
    using SPD = SPDMatrix<double, 2, Policy>;
    using Sym = SymmetricMatrix<double, 2>;
    using Log = SymmetricMatrix<double, 2, Cache::Spectral>;
    using Geometry = LogEuclideanGeometry<SPD>;
    const Geometry geometry;

    // 2. define the spatial mesh and scalar P1 basis; each DOF will hold one complete SPD matrix
    // the mesh must outlive the space, which must outlive its fields and prepared evaluations
    const auto mesh = Triangulation<2, 2>::UnitSquare(5);
    const GeometricFeSpace space(mesh, P1<1>, geometry);

    // 3. prescribe nodal tensors through their symmetric logarithms, then construct the field
    // the affine log field L(x,y) = [[0.2+x, 0.1(x+y)], [0.1(x+y), 0.4+y]] gives a known example
    const auto nodes = space.dof_handler().dofs_coords();
    MatrixBatch<Log> logs(space.n_dofs());
    for (int i = 0; i < space.n_dofs(); ++i) {
        const double x = nodes(i, 0), y = nodes(i, 1);
        logs[i] = Log(Vector<double, 3> {0.2 + x, 0.1 * (x + y), 0.4 + y});
    }
    // MatrixBatch<SPD>
    auto coefficients = logs.exp<Policy>(execution_par);
    // the field takes ownership; pass an lvalue batch instead to keep a separate copy
    GeometricFeFunction field(space, std::move(coefficients));

    // 4. evaluate a set of locations once, for example to predict or plot the field
    // eval_at prepares the spatial data and returns owning results in the supplied location order
    constexpr int side = 11;
    MatrixBatch<Vector<double, 2>> grid(side * side);
    for (int j = 0; j < side; ++j)
        for (int i = 0; i < side; ++i)
            grid[j * side + i] = Vector<double, 2> {double(i) / (side - 1), double(j) / (side - 1)};
    // MatrixBatch<SPD>: one-off sequential evaluation
    const auto once = field.eval_at(grid);

    // 5. retain Psi when evaluating these same locations repeatedly, for example during fitting
    // preparation keeps locations, cell IDs and P1 weights; its execution policy is independent of evaluation
    const auto Psi = space.prepare_evaluation(grid, execution_par);
    // MatrixBatch<SPD>: evaluation defaults to sequential execution
    const auto sequential = Psi(field);
    // MatrixBatch<SPD>: every location is evaluated using the parallel policy
    const auto parallel = Psi(field, execution_par);

    // 6. replace nodal data after an optimization step or a new observation, then reuse Psi
    // adding log(2) I to the logs doubles every tensor; previous result batches retain their original values
    auto replacement_logs = field.coeff().log<Cache::Spectral>(execution_par);
    const Sym shift(Vector<double, 3> {std::log(2.), 0., std::log(2.)});
    replacement_logs.assign(replacement_logs + shift, execution_par);
    // set_coeff invalidates old borrowed expressions and linearizations; obtain fresh ones after every update
    field.set_coeff(replacement_logs.exp<Policy>(execution_par));
    // MatrixBatch<SPD>: the same spatial preparation now evaluates the new coefficients
    const auto updated = Psi(field, execution_par);

    // 7. evaluate one interior point and inspect interpolation success before using its matrix value
    // result(x) includes the value, normalized weights, convergence, iterations and stationarity residual
    const Eigen::Vector2d x = mesh.cell(0).barycenter();
    // gfe::P1ValueResult<SPD>
    const auto result = field.result(x);
    fdapde_strong_assert(result.converged(), std::runtime_error, "point interpolation did not converge");
    const SPD& value = result.value;
    // use const SPD value(field(x)) instead when only the interpolated matrix is needed

    // 8. obtain cell-local data for quadrature and derivatives without losing global coefficient ordering
    // packet.dofs selects the three nodal matrices; the packet also contains physical P1 gradients and weights
    const auto packet = gfe::p1_fem_cell_quadrature(space, 0, QS2DP4);
    const auto local_nodes = field.coeff().select(packet.dofs);
    // local interpolation is useful inside cell calculations when the containing cell is already known
    const auto local_field = geometry.interpolant(space.dof_handler().cell(0), local_nodes);
    const SPD local_value(local_field(x));

    // 9. differentiate the field in space, for example to measure spatial variation or build a penalty
    // linearization(x) prepares local derivatives; weight_jvp applies a change of P1 weights with zero sum
    const auto linearization = field.linearization(x);
    // Sym: dU/dx, obtained by differentiating the scalar shape weights in the physical x direction
    const auto dx = linearization.weight_jvp(packet.physical_weight_gradients[0]);

    // 10. measure how a perturbation of the local nodal tensors changes the interpolated value
    // nodal_jvp applies the derivative to three directions in packet.dofs order, one tangent per node
    std::array<Sym, 3> directions;
    // nodal_jvp accepts a contiguous span of tangent matrices
    // dP_i/dt = P_i is a concrete perturbation: all local tensors are scaled by exp(t)
    for (std::size_t i = 0; i < directions.size(); ++i) directions[i] = Sym(local_nodes[i]);
    // Sym: dU/dt for the chosen nodal perturbation
    const auto nodal_action = linearization.nodal_jvp(directions);

    // 11. propagate a data-loss gradient from U(x) back to the nodal coefficients, as needed by an optimizer
    // use the Frobenius loss F = 0.5 ||U(x)-D||_F^2, with the identity tensor as observation D
    const SPD observation = SPD::Identity();
    const Sym residual(value - observation);
    const double data_loss = 0.5 * residual.squared_norm();
    // Sym: the geometry converts the Frobenius residual to the Riemannian gradient at U(x)
    const auto value_gradient = geometry.euclidean_to_riemannian_gradient(value, residual);
    // std::vector<Sym>: nodal_vjp returns the three local Riemannian gradients in packet.dofs order
    const auto local_gradients = linearization.nodal_vjp(value_gradient);
    // scatter cell-local gradients into the global DOF ordering used by a batch optimizer
    MatrixBatch<Sym> nodal_gradient(space.n_dofs());
    for (std::size_t i = 0; i < local_gradients.size(); ++i) nodal_gradient[packet.dofs[i]] += local_gradients[i];

    // 12. integrate spatial variation over a cell, for example as a smoothing regularizer
    // this Dirichlet energy is 0.5 integral_cell ||grad U||_geometry^2; packet.dofs selects global coefficients
    // gfe::P1ObjectiveValueResult
    const auto energy = gfe::p1_dirichlet_cell_value(geometry, field.coeff(), packet);
    fdapde_strong_assert(energy.converged(), std::runtime_error, "cell energy evaluation did not converge");

    // verify the tutorial's known formulas and the agreement of its different evaluation paths
    const auto maximum_norm = [](double maximum, const auto& difference) {
        return std::max(maximum, difference.norm());
    };
    const double evaluation_error =
      std::max((sequential - parallel).redux(0., maximum_norm), (sequential - once).redux(0., maximum_norm));
    const double update_error = (updated - 2. * sequential).redux(0., maximum_norm);
    // the three interfaces must evaluate the original field at the same points in the same order
    fdapde_strong_assert(evaluation_error < 1e-12, std::runtime_error, "grid evaluation paths disagree");
    // the updated field must be twice the original at every retained location
    fdapde_strong_assert(update_error < 1e-12, std::runtime_error, "coefficient update differs from scaling oracle");
    // local interpolation and the global spatial interface must describe the same updated field
    fdapde_strong_assert(
      (value - local_value).norm() < 1e-12, std::runtime_error, "local and global interpolation disagree");
    const Sym expected_dx = matrix_exp_frechet(value.log(), Sym(Vector<double, 3> {1., 0.1, 0.}));
    // dL/dx = [[1, 0.1], [0.1, 0]] is unchanged by the constant log(2) I shift
    fdapde_strong_assert(
      (dx - expected_dx).norm() < 1e-12, std::runtime_error, "spatial derivative differs from the affine oracle");
    // common nodal scaling commutes with interpolation, giving dU/dt = U(x)
    fdapde_strong_assert(
      (nodal_action - value).norm() < 1e-12, std::runtime_error, "nodal derivative differs from scaling oracle");
    double adjoint_pairing = 0;
    for (std::size_t i = 0; i < directions.size(); ++i)
        adjoint_pairing += geometry.inner_product(local_nodes[i], directions[i], nodal_gradient[packet.dofs[i]]);
    const double adjoint_error =
      std::abs(geometry.inner_product(value, nodal_action, value_gradient) - adjoint_pairing);
    // the scattered loss gradient must satisfy the metric differential-adjoint identity
    fdapde_strong_assert(adjoint_error < 1e-12, std::runtime_error, "nodal differential and adjoint disagree");
    // constant log-chart gradients give energy density 1.02 on a triangle of area 1/32
    fdapde_strong_assert(
      std::abs(energy.value - 1.02 / 32) < 1e-12, std::runtime_error, "cell energy differs from the affine oracle");

    std::cout << std::scientific << std::setprecision(6) << "DOFs: " << space.n_dofs()
              << "; grid points: " << Psi.size() << "; prepared spatial cells: " << Psi.prepared_cells() << '\n'
              << "updated U(x):\n"
              << value << '\n'
              << "spatial variation ||dU/dx||_F: " << dx.norm() << '\n'
              << "data loss against identity: " << data_loss << '\n'
              << "cell Dirichlet energy: " << energy.value << '\n'
              << "evaluation paths maximum difference: " << evaluation_error << '\n'
              << "coefficient update maximum error: " << update_error << '\n'
              << "nodal loss-gradient adjoint residual: " << adjoint_error << '\n';
}
