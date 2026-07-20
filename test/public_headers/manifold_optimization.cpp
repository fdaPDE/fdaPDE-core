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

#include <fdaPDE/manifold_optimization.h>

#include <type_traits>
#include <utility>

namespace {

struct HeaderPoint {
    double value = 0;
};
struct HeaderTangent {
    double value = 0;
};

struct HeaderGeometry {
    using Point = HeaderPoint;
    using Tangent = HeaderTangent;

    std::size_t dimension() const { return 1; }
    double inner_product(const Point&, const Tangent& u, const Tangent& v) const { return u.value * v.value; }
    double norm(const Point&, const Tangent& tangent) const { return tangent.value; }
    Tangent project(const Point&, const Tangent& tangent) const { return tangent; }
    Tangent zero_tangent(const Point&) const { return {}; }
    Tangent linear_combination(const Point&, double alpha, const Tangent& u, double beta, const Tangent& v) const {
        return {alpha * u.value + beta * v.value};
    }
    Point retract(const Point& point, const Tangent& tangent, double step) const {
        return {point.value + step * tangent.value};
    }
};

struct HeaderWorkspace { };

struct HeaderProblem {
    using Workspace = HeaderWorkspace;
    double cost(const HeaderPoint& point, Workspace&) { return point.value; }
    HeaderTangent gradient(const HeaderPoint& point, Workspace&) { return {point.value}; }
};

struct HeaderConstProblem {
    using Workspace = HeaderWorkspace;
    double cost(const HeaderPoint& point, Workspace&) const { return point.value; }
    HeaderTangent gradient(const HeaderPoint& point, Workspace&) const { return {point.value}; }
};

using HeaderPowerGeometry = fdapde::manifold::PowerGeometry<HeaderGeometry>;
using HeaderContext = fdapde::manifold::EvaluationContext<HeaderTangent, HeaderWorkspace>;
using HeaderSymmetricTangent = fdapde::linalg::SymmetricMatrix<double, 2, 2>;
using HeaderSymmetricContext = fdapde::manifold::EvaluationContext<HeaderSymmetricTangent, HeaderWorkspace>;
using HeaderLogGeometry = fdapde::manifold::LogEuclideanSPDGeometry<double, 3>;
using HeaderDynamicLogGeometry = fdapde::manifold::LogEuclideanSPDGeometry<double, fdapde::Dynamic>;
using HeaderLogPoint = fdapde::manifold::point_t<HeaderLogGeometry>;
using HeaderLogTangent = fdapde::manifold::tangent_t<HeaderLogGeometry>;

struct HeaderLogProblem {
    using Workspace = HeaderWorkspace;
    double cost(const HeaderLogPoint&, Workspace&) const { return 0; }
    HeaderLogTangent gradient(const HeaderLogPoint& point, Workspace&) const {
        return HeaderLogGeometry {}.zero_tangent(point);
    }
};

template <typename Context>
concept HeaderPermitsRvalueCurrent = requires(Context&& context) { std::move(context).current(); };

template <typename Geometry>
concept HeaderPermitsRvalueComponentAccess =
  requires(Geometry&& geometry) { std::move(geometry).component_geometry(); };

template <typename Evaluation>
concept HeaderPermitsRvalueEvaluationAccess = requires(Evaluation&& evaluation) {
    std::move(evaluation).cost();
    std::move(evaluation).gradient();
    std::move(evaluation).workspace();
};

template <typename Solver>
concept HeaderPermitsRvalueOptions = requires(Solver&& solver) { std::move(solver).options(); };

using HeaderArmijoResult = decltype(std::declval<const fdapde::manifold::ArmijoBacktracking&>().search(
  std::declval<HeaderProblem&>(), std::declval<const HeaderGeometry&>(), std::declval<const HeaderPoint&>(),
  std::declval<const HeaderTangent&>(), 0.0, -1.0, std::declval<HeaderContext&>()));
using HeaderSteepestDescentResult =
  decltype(std::declval<const fdapde::manifold::RiemannianSteepestDescent&>().optimize(
    std::declval<HeaderProblem&>(), std::declval<const HeaderGeometry&>(), std::declval<const HeaderPoint&>()));

static_assert(fdapde::manifold::FirstOrderGeometry<HeaderGeometry>);
static_assert(fdapde::manifold::FirstOrderGeometry<HeaderPowerGeometry>);
static_assert(fdapde::manifold::VectorTransportGeometry<HeaderLogGeometry>);
static_assert(fdapde::manifold::VectorTransportGeometry<HeaderDynamicLogGeometry>);
static_assert(fdapde::manifold::FirstOrderProblem<HeaderProblem, HeaderGeometry>);
static_assert(!fdapde::manifold::FirstOrderProblem<const HeaderProblem, HeaderGeometry>);
static_assert(fdapde::manifold::FirstOrderProblem<const HeaderConstProblem, HeaderGeometry>);
static_assert(std::is_default_constructible_v<HeaderContext>);
static_assert(std::is_default_constructible_v<HeaderSymmetricContext>);
static_assert(std::is_default_constructible_v<HeaderLogGeometry>);
static_assert(!std::is_constructible_v<HeaderLogGeometry, int>);
static_assert(!std::is_default_constructible_v<HeaderDynamicLogGeometry>);
static_assert(std::is_constructible_v<HeaderDynamicLogGeometry, int>);
static_assert(std::is_same_v<HeaderLogPoint, fdapde::linalg::SPDMatrix<double, 3, 3>>);
static_assert(std::is_same_v<HeaderLogTangent, fdapde::linalg::SymmetricMatrix<double, 3, 3>>);
static_assert(fdapde::manifold::FirstOrderProblem<const HeaderLogProblem, HeaderLogGeometry>);
static_assert(!HeaderPermitsRvalueCurrent<HeaderContext>);
static_assert(!HeaderPermitsRvalueEvaluationAccess<typename HeaderContext::Evaluation>);
static_assert(!HeaderPermitsRvalueComponentAccess<HeaderPowerGeometry>);
static_assert(!HeaderPermitsRvalueOptions<fdapde::manifold::ArmijoBacktracking>);
static_assert(!HeaderPermitsRvalueOptions<fdapde::manifold::RiemannianSteepestDescent>);
static_assert(std::is_same_v<fdapde::manifold::point_t<HeaderPowerGeometry>, std::vector<HeaderPoint>>);
static_assert(std::is_same_v<fdapde::manifold::tangent_t<HeaderPowerGeometry>, std::vector<HeaderTangent>>);
static_assert(std::is_same_v<HeaderArmijoResult, fdapde::manifold::ArmijoResult<HeaderPoint>>);
static_assert(std::is_same_v<HeaderSteepestDescentResult, fdapde::manifold::SteepestDescentResult<HeaderPoint>>);

[[maybe_unused]] void instantiate_const_problem_solver() {
    HeaderGeometry geometry;
    const HeaderConstProblem problem;
    const auto result = fdapde::manifold::RiemannianSteepestDescent {}.optimize(problem, geometry, HeaderPoint {});
    static_cast<void>(result);
}

[[maybe_unused]] void instantiate_log_geometry_solver() {
    fdapde::linalg::Matrix<double, 3, 3> identity;
    identity.set_zero();
    for (int i = 0; i < 3; ++i) { identity(i, i) = 1; }
    const HeaderLogPoint point(identity, fdapde::linalg::checked);
    const HeaderLogGeometry geometry;
    const HeaderLogProblem problem;
    const auto result = fdapde::manifold::RiemannianSteepestDescent {}.optimize(problem, geometry, point);
    static_cast<void>(result);
}

}   // namespace
