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

static_assert(fdapde::manifold::FirstOrderGeometry<HeaderGeometry>);
static_assert(fdapde::manifold::FirstOrderGeometry<HeaderPowerGeometry>);
static_assert(fdapde::manifold::FirstOrderProblem<HeaderProblem, HeaderGeometry>);
static_assert(!fdapde::manifold::FirstOrderProblem<const HeaderProblem, HeaderGeometry>);
static_assert(fdapde::manifold::FirstOrderProblem<const HeaderConstProblem, HeaderGeometry>);
static_assert(std::is_default_constructible_v<HeaderContext>);
static_assert(std::is_default_constructible_v<HeaderSymmetricContext>);
static_assert(!HeaderPermitsRvalueCurrent<HeaderContext>);
static_assert(!HeaderPermitsRvalueEvaluationAccess<typename HeaderContext::Evaluation>);
static_assert(!HeaderPermitsRvalueComponentAccess<HeaderPowerGeometry>);
static_assert(std::is_same_v<fdapde::manifold::point_t<HeaderPowerGeometry>, std::vector<HeaderPoint>>);
static_assert(std::is_same_v<fdapde::manifold::tangent_t<HeaderPowerGeometry>, std::vector<HeaderTangent>>);

}   // namespace
