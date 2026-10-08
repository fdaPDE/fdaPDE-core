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
// examples/geometric_finite_elements/spd_discrete_tension.cpp
#include <fdaPDE/geometric_finite_elements.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace fdapde;

// reuses one spatial stencil for SPD smoothing values and gradients under two metrics
int main() {
    parallel_set_num_threads(4);
    using Geometry = manifold::LogEuclideanSPDGeometry<double, 2, Usage::InterpolationNodes>;
    using SPD = Geometry::Point;
    using Sym = Geometry::Tangent;
    const Geometry geometry;

    // 1. build a deterministic tensor field on a chain with enough nodes to use the parallel tension kernel
    constexpr std::size_t count = 33;
    MatrixBatch<SPD> nodes(count);
    for (std::size_t i = 0; i < count; ++i) {
        const double t = double(i) / (count - 1);
        const Sym logarithm(Vector<double, 3> {.2 + t, .08 * std::sin(2 * t), .5 + .3 * t * t});
        nodes[i] = matrix_exp(logarithm);
    }

    // 2. unit-length P1 segments have off-diagonal stiffness -1 and endpoint lumped mass 1/2
    gfe::P1LumpedLaplacianStencil stencil;
    stencil.lumped_masses.assign(count, 1.);
    stencil.lumped_masses.front() = stencil.lumped_masses.back() = .5;
    for (std::size_t i = 1; i < count; ++i) stencil.edges.push_back({i - 1, i, -1.});
    // prepare once outside repeated objective evaluations; this owns masses, edges and their incidence
    const gfe::PreparedP1LumpedLaplacianStencil prepared(stencil);

    // 3. value-only calls avoid gradient work; contribution gradients follow the global node order
    const auto value = gfe::p1_discrete_tension_value(geometry, nodes, prepared, execution_par);
    const auto sequential = gfe::p1_discrete_tension_contribution(geometry, nodes, prepared, execution_seq);
    const auto parallel = gfe::p1_discrete_tension_contribution(geometry, nodes, prepared, execution_par);
    // objective diagnostics must succeed before consuming values or nodal gradients
    fdapde_strong_assert(
      value.converged() && sequential.converged() && parallel.converged(), std::runtime_error,
      "LE tension evaluation did not converge");
    double policy_error =
      std::max(std::abs(value.value - sequential.value), std::abs(parallel.value - sequential.value));
    for (std::size_t i = 0; i < count; ++i)
        policy_error = std::max(policy_error, (parallel.nodal_gradient[i] - sequential.nodal_gradient[i]).norm());
    // the value-only path and both execution policies must preserve the same objective and gradient
    fdapde_strong_assert(policy_error < 1e-12, std::runtime_error, "LE tension evaluation paths disagree");

    // 4. LE contributions are Riemannian gradients; a small negative exponential step gives a new candidate
    MatrixBatch<SPD> candidate(count);
    for (std::size_t i = 0; i < count; ++i)
        candidate[i] = geometry.exponential(nodes[i], parallel.nodal_gradient[i], -.01);
    const auto updated = gfe::p1_discrete_tension_value(geometry, candidate, prepared, execution_par);
    // this fixed small step must decrease the deterministic example's quadratic log-coordinate penalty
    fdapde_strong_assert(
      updated.converged() && updated.value < value.value, std::runtime_error, "LE descent step did not lower tension");

    // 5. C-LE accepts the same stencil and one positive rho coefficient at each base node
    const manifold::CheegerLogEuclideanSPDGeometry<double, 2, Usage::InterpolationNodes> cheeger;
    std::vector<double> rho(count);
    for (std::size_t i = 0; i < count; ++i) rho[i] = .3 + .2 * double(i) / (count - 1);
    const auto cle_seq =
      gfe::p1_cheeger_discrete_tension_log_contribution(cheeger, nodes, prepared, rho, execution_seq);
    const auto cle_par =
      gfe::p1_cheeger_discrete_tension_log_contribution(cheeger, nodes, prepared, rho, execution_par);
    // the C-LE contribution carries the same explicit success contract as the other objective kernels
    fdapde_strong_assert(
      cle_seq.converged() && cle_par.converged(), std::runtime_error, "C-LE tension evaluation did not converge");
    double cle_error = std::abs(cle_seq.value - cle_par.value), shared_rho_derivative = 0;
    for (std::size_t i = 0; i < count; ++i) {
        cle_error = std::max(cle_error, (cle_seq.nodal_gradient[i] - cle_par.nodal_gradient[i]).norm());
        cle_error = std::max(cle_error, std::abs(cle_seq.rho_gradient[i] - cle_par.rho_gradient[i]));
        shared_rho_derivative += cle_par.rho_gradient[i];
    }
    // serial and parallel C-LE must agree for the value, log-coordinate covectors and scalar partial derivatives
    fdapde_strong_assert(cle_error < 1e-12, std::runtime_error, "C-LE tension execution policies disagree");
    // the C-LE nodal_gradient pairs with d(log P) in the Frobenius metric; rho_gradient pairs with scalar drho
    // unlike the LE gradients above, these log-coordinate covectors are not ambient Riemannian tangents

    std::cout << std::scientific << std::setprecision(6) << "nodes: " << count
              << "; prepared edges: " << prepared.stencil().edges.size() << '\n'
              << "LE tension before/after one step: " << value.value << " / " << updated.value << '\n'
              << "LE evaluation paths maximum difference: " << policy_error << '\n'
              << "C-LE tension: " << cle_par.value << '\n'
              << "C-LE derivative for increasing every rho: " << shared_rho_derivative << '\n'
              << "C-LE execution policies maximum difference: " << cle_error << '\n';
}
