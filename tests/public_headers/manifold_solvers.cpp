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

#include <fdaPDE/manifold_solvers.h>

// generic solvers must compile without importing any concrete geometry or barycenter module
#if defined(__FDAPDE_MANIFOLD_GEOMETRY_MODULE_H__) || defined(__FDAPDE_MANIFOLD_MEANS_MODULE_H__)
#    error "The solver aggregate must depend only on shared manifold contracts"
#endif

namespace {
/// @brief supplies an application-owned scalar geometry independently of the native geometry module
struct ScalarGeometry {
    using Point = double;
    using Tangent = double;
    /// @brief counts the single independent scalar coordinate
    std::size_t dimension() const { return 1; }
    /// @brief pairs scalar tangent velocities with the Euclidean metric
    double inner_product(double, double u, double v) const { return u * v; }
    /// @brief measures a scalar velocity
    double norm(double, double u) const { return std::abs(u); }
    /// @brief retains a velocity already expressed in tangent coordinates
    double project(double, double u) const { return u; }
    /// @brief returns the additive tangent identity
    double zero_tangent(double) const { return 0; }
    /// @brief combines scalar tangent velocities
    double linear_combination(double, double a, double u, double b, double v) const { return a * u + b * v; }
    /// @brief translates the scalar point along its velocity
    double retract(double point, double u, double step) const { return point + step * u; }
};
/// @brief defines a quadratic objective using only the generic solver callback contract
struct Quadratic {
    /// @brief supplies candidate-local storage without objective intermediates
    struct Workspace { };
    /// @brief evaluates the squared residual from the known minimizer
    double cost(double x, Workspace&) const { return .5 * (x - 2) * (x - 2); }
    /// @brief differentiates the scalar residual
    double grad(double x, Workspace&) const { return x - 2; }
    /// @brief applies the identity scalar Hessian
    double hess(double, double u, Workspace&) const { return u; }
};
// the custom scalar geometry satisfies the solver contract without native point or tangent owners
static_assert(fdapde::manifold::FirstOrderGeometry<ScalarGeometry>);
}   // namespace

/// @brief instantiates both solvers with an application-defined geometry before any concrete geometry include
void instantiate_manifold_solvers() {
    Quadratic objective;
    (void)fdapde::manifold::RiemannianSteepestDescent().optimize(objective, ScalarGeometry {}, 0.);
    (void)fdapde::manifold::RiemannianTrustRegion().optimize(objective, ScalarGeometry {}, 0.);
}

#include <fdaPDE/manifold_geometry.h>
#include <fdaPDE/manifold_means.h>
