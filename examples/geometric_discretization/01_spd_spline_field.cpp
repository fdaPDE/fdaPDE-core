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

// compile with C++20, pthread, the core include path and Eigen3
#include <fdaPDE/geometric_splines.h>

using namespace fdapde;
using namespace fdapde::manifold;

// constructs, evaluates, differentiates and updates an SPD-valued B-spline field
int main() {
    parallel_set_num_threads(4);
    using Policy = Cache::Union<Cache::Spectral, Cache::Log, Cache::LogDividedDifferences>;
    using SPD = SPDMatrix<double, 2, Policy>;
    using Sym = SymmetricMatrix<double, 2>;
    using Log = SymmetricMatrix<double, 2, Cache::Spectral>;

    // 1. retain the interval and scalar cubic spline space for the lifetime of the geometric field
    const Triangulation<1, 1> interval(0., 1., 7);
    const BsSpace scalar_space(interval, 3);
    const LogEuclideanGeometry<SPD> geometry;
    const GeometricSpace space(scalar_space, geometry);

    // 2. define geometric control coefficients, which are not generally field values at knots
    // clamped endpoint weights select the first and last controls; interior values blend active controls
    MatrixBatch<Log> logarithms(space.n_dofs());
    for (int i = 0; i < space.n_dofs(); ++i) {
        const double position = double(i) / (space.n_dofs() - 1);
        logarithms[i] = Log(Vector<double, 3> {.2 + position, .1 * position, .4 - .2 * position});
    }
    // MatrixBatch<SPD>
    auto controls = logarithms.exp<Policy>(execution_par);
    GeometricFunction field(space, std::move(controls));

    // 3. evaluate one physical coordinate through the same interface used by scalar P1 fields
    const Eigen::Matrix<double, 1, 1> x = Eigen::Matrix<double, 1, 1>::Constant(.4);
    // SPD
    const SPD value(field(x));
    // gfe::P1ValueResult<SPD>: the existing weighted mean result retains convergence diagnostics
    const auto result = field.result(x);

    // 4. retain basis support and weights when the same locations are evaluated repeatedly during fitting
    constexpr int count = 101;
    MatrixBatch<Vector<double, 1>> grid(count);
    for (int i = 0; i < count; ++i) grid[i] = Vector<double, 1> {double(i) / (count - 1)};
    const auto Psi = space.prepare_evaluation(grid, execution_par);
    // MatrixBatch<SPD>
    const auto values = Psi(field, execution_par);
    // MatrixBatch<SPD>: a one-off evaluation prepares its own spatial data
    const auto once = field.eval_at(grid, execution_par);

    // 5. compose physical scalar basis derivatives with the geometric derivative with respect to weights
    const int span = scalar_space.triangulation().locate(x);
    const auto active_dofs = scalar_space.dof_handler().active_dofs(span);
    std::vector<double> weight_derivatives;
    for (int id : active_dofs) weight_derivatives.push_back(scalar_space.physical_basis()[id].gradient(1)(x[0]));
    const auto linearization = field.linearization(x);
    // Sym: dU/dx in physical coordinates, using the active control ordering
    [[maybe_unused]] const auto spatial_derivative = linearization.weight_jvp(weight_derivatives);

    // 6. replace control values and reuse Psi; old owning value batches retain their original matrices
    // adding log(2) I to every control logarithm doubles the complete log-Euclidean field
    auto updated_logs = field.coeff().log<Cache::Spectral>(execution_par);
    const Sym shift(Vector<double, 3> {std::log(2.), 0., std::log(2.)});
    updated_logs.assign(updated_logs + shift, execution_par);
    field.set_coeff(updated_logs.exp<Policy>(execution_par));
    // MatrixBatch<SPD>
    const auto updated = Psi(field, execution_par);
    // obtain new borrowed expressions and derivative workspaces after replacing coefficients
}
