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

// compile from the core root with c++ -std=c++20 -O2 -pthread -I. -I/path/to/eigen3 examples/spd_field_estimation.cpp
#include <fdaPDE/geometric_finite_elements_fem.h>
#include <fdaPDE/manifold_optimization.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>

using namespace fdapde;
using namespace fdapde::manifold;

namespace {
using Policy = Cache::Union<Cache::Spectral, Cache::Log>;
using SPD = SPDMatrix<double, 2, Policy>;
using SPDBatch = MatrixBatch<SPD>;
using Sym = SymmetricMatrix<double, 2>;
using SymBatch = MatrixBatch<Sym>;
using Log = SymmetricMatrix<double, 2, Cache::Spectral>;
using LogBatch = MatrixBatch<Log>;

/// @brief fits SPD targets through symmetric logarithms with the product Frobenius metric
struct SymmetricObjective {
    /// @brief supplies an independent workspace for each optimizer candidate
    struct Workspace { };
    const SPDBatch& target;

    /// @brief evaluates one half of the squared Frobenius residual after exponentiation
    double cost(const LogBatch& logs, Workspace&) {
        double value = 0;
        for (std::size_t i = 0; i < logs.size(); ++i) {
            const auto point = logs[i].exp();
            value += 0.5 * (point - target[i]).squared_norm();
        }
        return value;
    }

    /// @brief pulls residuals back through the self-adjoint exponential differential
    LogBatch grad(const LogBatch& logs, Workspace&) {
        LogBatch result(logs.size());
        for (std::size_t i = 0; i < logs.size(); ++i) {
            const auto point = logs[i].exp();
            const Sym residual(point - target[i]);
            result[i] = matrix_exp_frechet(logs[i], residual);
        }
        return result;
    }

    /// @brief applies the full logarithmic Hessian including the second exponential differential
    LogBatch hess(const LogBatch& logs, const LogBatch& direction, Workspace&) {
        LogBatch result(logs.size());
        for (std::size_t i = 0; i < logs.size(); ++i) {
            const auto point = logs[i].exp();
            const Sym residual(point - target[i]);
            const auto point_direction = matrix_exp_frechet(logs[i], direction[i]);
            const auto curvature = matrix_exp_second_frechet(logs[i], direction[i], residual);
            const auto normal = matrix_exp_frechet(logs[i], point_direction);
            result[i] = curvature + normal;
        }
        return result;
    }
};

/// @brief fits SPD targets with ambient derivatives independent of the optimizer metric
struct SPDObjective {
    /// @brief supplies an independent workspace for each optimizer candidate
    struct Workspace { };
    const SPDBatch& target;

    /// @brief evaluates the squared Frobenius loss directly on the SPD coefficients
    double cost(const SPDBatch& points, Workspace&) {
        double value = 0;
        for (std::size_t i = 0; i < points.size(); ++i) value += 0.5 * (points[i] - target[i]).squared_norm();
        return value;
    }

    /// @brief returns the ambient symmetric residual for every nodal coefficient
    SymBatch egrad(const SPDBatch& points, Workspace&) {
        SymBatch result(points.size());
        for (std::size_t i = 0; i < points.size(); ++i) result[i] = points[i] - target[i];
        return result;
    }

    /// @brief applies the identity ambient Hessian of the quadratic loss
    SymBatch ehess(const SPDBatch&, const SymBatch& direction, Workspace&) { return direction; }
};
}   // namespace

// estimates the same SPD field in logarithmic and SPD coordinates and evaluates it on a shared grid
int main() {
    parallel_set_num_threads(4);

    // Triangulation<2, 2>: 121 nodes and 200 triangles
    const auto mesh = Triangulation<2, 2>::UnitSquare(11);

    // define geometry
    const LogEuclideanGeometry<SPD> geometry;
    // alternative element metrics: AffineInvariantGeometry<SPD>, BuresWassersteinGeometry<SPD>,
    // LogCholeskyGeometry<SPD> or CheegerLogEuclideanGeometry<SPD>(0.1)

    // define geometric finite elements space
    const GeometricFeSpace space(mesh, P1<1>, geometry);
    using Space = std::remove_cvref_t<decltype(space)>;

    // define target field
    const auto nodes = space.dof_handler().dofs_coords();
    LogBatch target_logs(space.n_dofs());
    for (int i = 0; i < space.n_dofs(); ++i) {
        const double x = nodes(i, 0), y = nodes(i, 1);
        target_logs[i] = Log(Vector<double, 3> {0.2 + x, 0.1 * (x + y), 0.4 + y});
    }
    const auto target = target_logs.exp<Policy>(execution_par);

    // define optimizer
    TrustRegionOptions options;
    options.max_iterations = 100;
    options.gradient_tolerance = 1e-8;
    options.subproblem.max_iterations = 100;
    options.subproblem.residual_tolerance = 1e-3;
    const RiemannianTrustRegion optimizer(options);

    // define evaluation grid
    constexpr int side = 31;
    MatrixBatch<Vector<double, 2>> grid(side * side);
    for (int j = 0; j < side; ++j)
        for (int i = 0; i < side; ++i)
            grid[j * side + i] = Vector<double, 2> {double(i) / (side - 1), double(j) / (side - 1)};

    // GeometricFeEvaluation<Space, Vector<double, 2>>
    const auto Psi = space.prepare_evaluation(grid, execution_par);

    SPDBatch symmetric_estimates, spd_estimates;

    // zero symmetric logarithms represent the identity SPD initialization
    {
        const LogBatch initial(space.n_dofs());
        const EuclideanGeometry<Log> symmetric_geometry;
        const ProductGeometry optimization_geometry(symmetric_geometry, initial.size());
        SymmetricObjective objective {target};
        // TrustRegionResult<LogBatch>
        const auto solution = optimizer.optimize(objective, optimization_geometry, initial);
        fdapde_strong_assert(solution.converged(), std::runtime_error, "symmetric trust region did not converge");
        // SPDBatch
        const auto coefficients = solution.point.exp<Policy>(execution_par);
        const GeometricFeFunction<const Space, SPD> tensor_field(space, coefficients);
        symmetric_estimates = Psi(tensor_field, execution_par);
    }

    // SPD batches start at identity; the product metric converts the ambient derivatives
    {
        const SPDBatch initial(space.n_dofs());
        const ProductGeometry optimization_geometry(geometry, initial.size());
        SPDObjective objective {target};
        // TrustRegionResult<SPDBatch>
        const auto solution = optimizer.optimize(objective, optimization_geometry, initial);
        fdapde_strong_assert(solution.converged(), std::runtime_error, "SPD trust region did not converge");
        const GeometricFeFunction<const Space, SPD> tensor_field(space, solution.point);
        spd_estimates = Psi(tensor_field, execution_par);
    }

    // compare both fields at the same grid points using the full Frobenius norm
    double max_difference = 0, difference_norm = 0, spd_norm = 0;
    for (std::size_t i = 0; i < grid.size(); ++i) {
        const double difference = (symmetric_estimates[i] - spd_estimates[i]).norm();
        max_difference = std::max(max_difference, difference);
        difference_norm = std::hypot(difference_norm, difference);
        spd_norm = std::hypot(spd_norm, spd_estimates[i].norm());
    }
    std::cout << std::scientific << std::setprecision(6) << "grid points: " << grid.size() << '\n'
              << "maximum Frobenius difference: " << max_difference << '\n'
              << "relative Frobenius difference (SPD reference): " << difference_norm / spd_norm << '\n';
}
