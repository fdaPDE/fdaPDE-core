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

#include <array>
#include <span>
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
    double norm(const Point&, const Tangent& tangent) const { return std::abs(tangent.value); }
    Tangent project(const Point&, const Tangent& tangent) const { return tangent; }
    Tangent zero_tangent(const Point&) const { return {}; }
    Tangent linear_combination(const Point&, double alpha, const Tangent& u, double beta, const Tangent& v) const {
        return {alpha * u.value + beta * v.value};
    }
    Point retract(const Point& point, const Tangent& tangent, double step) const {
        return {point.value + step * tangent.value};
    }
    Point exponential(const Point& point, const Tangent& tangent, double step) const {
        return {point.value + step * tangent.value};
    }
    Tangent logarithm(const Point& from, const Point& to) const { return {to.value - from.value}; }
    double distance(const Point& from, const Point& to) const { return std::abs(to.value - from.value); }
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

struct HeaderHessianProblem : HeaderProblem {
    HeaderTangent hessian_vector(const HeaderPoint&, const HeaderTangent& tangent, Workspace&) { return tangent; }
};

using HeaderPowerGeometry = fdapde::manifold::PowerGeometry<HeaderGeometry>;
using HeaderContext = fdapde::manifold::EvaluationContext<HeaderTangent, HeaderWorkspace>;
using HeaderSymmetricTangent = fdapde::linalg::SymmetricMatrix<double, 2, 2>;
using HeaderSymmetricContext = fdapde::manifold::EvaluationContext<HeaderSymmetricTangent, HeaderWorkspace>;
using HeaderLogGeometry = fdapde::manifold::LogEuclideanSPDGeometry<double, 3>;
using HeaderDynamicLogGeometry = fdapde::manifold::LogEuclideanSPDGeometry<double, fdapde::Dynamic>;
using HeaderLogPoint = fdapde::manifold::point_t<HeaderLogGeometry>;
using HeaderLogTangent = fdapde::manifold::tangent_t<HeaderLogGeometry>;
using HeaderDynamicLogPoint = fdapde::manifold::point_t<HeaderDynamicLogGeometry>;
using HeaderAffineGeometry = fdapde::manifold::AffineInvariantSPDGeometry<double, 3>;
using HeaderDynamicAffineGeometry = fdapde::manifold::AffineInvariantSPDGeometry<double, fdapde::Dynamic>;
using HeaderAffinePoint = fdapde::manifold::point_t<HeaderAffineGeometry>;
using HeaderAffineTangent = fdapde::manifold::tangent_t<HeaderAffineGeometry>;

struct HeaderLogProblem {
    using Workspace = HeaderWorkspace;
    double cost(const HeaderLogPoint&, Workspace&) const { return 0; }
    HeaderLogTangent gradient(const HeaderLogPoint& point, Workspace&) const {
        return HeaderLogGeometry {}.zero_tangent(point);
    }
};

struct HeaderSPDHessianProblem {
    using Workspace = HeaderWorkspace;
    double cost(const HeaderAffinePoint&, Workspace&) const { return 0; }
    HeaderAffineTangent gradient(const HeaderAffinePoint& point, Workspace&) const {
        return HeaderAffineGeometry {}.zero_tangent(point);
    }
    HeaderAffineTangent hessian_vector(const HeaderAffinePoint&, const HeaderAffineTangent& tangent, Workspace&) const {
        return tangent;
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

template <typename Geometry>
concept HeaderPermitsMeanWithoutInitial = requires(
  const Geometry& geometry, std::span<const fdapde::manifold::point_t<Geometry>> samples,
  std::span<const double> weights) { fdapde::manifold::weighted_karcher_mean(geometry, samples, weights); };

using HeaderArmijoResult = decltype(std::declval<const fdapde::manifold::ArmijoBacktracking&>().search(
  std::declval<HeaderProblem&>(), std::declval<const HeaderGeometry&>(), std::declval<const HeaderPoint&>(),
  std::declval<const HeaderTangent&>(), 0.0, -1.0, std::declval<HeaderContext&>()));
using HeaderSteepestDescentResult =
  decltype(std::declval<const fdapde::manifold::RiemannianSteepestDescent&>().optimize(
    std::declval<HeaderProblem&>(), std::declval<const HeaderGeometry&>(), std::declval<const HeaderPoint&>()));
using HeaderTruncatedCGResult = decltype(std::declval<const fdapde::manifold::SteihaugTruncatedCG&>().solve(
  std::declval<const HeaderSPDHessianProblem&>(), std::declval<const HeaderAffineGeometry&>(),
  std::declval<const HeaderAffinePoint&>(), std::declval<const HeaderAffineTangent&>(), 1.0,
  std::declval<HeaderWorkspace&>()));
using HeaderTrustRegionResult = decltype(std::declval<const fdapde::manifold::RiemannianTrustRegion&>().optimize(
  std::declval<const HeaderSPDHessianProblem&>(), std::declval<const HeaderAffineGeometry&>(),
  std::declval<const HeaderAffinePoint&>()));
using HeaderMeanResult = decltype(fdapde::manifold::weighted_karcher_mean(
  std::declval<const HeaderGeometry&>(), std::declval<std::span<const HeaderPoint>>(),
  std::declval<std::span<const double>>(), std::declval<const HeaderPoint&>()));
using HeaderExactLogMeanResult = decltype(fdapde::manifold::weighted_karcher_mean(
  std::declval<const HeaderLogGeometry&>(), std::declval<std::span<const HeaderLogPoint>>(),
  std::declval<std::span<const double>>()));
using HeaderExactDynamicLogMeanResult = decltype(fdapde::manifold::weighted_karcher_mean(
  std::declval<const HeaderDynamicLogGeometry&>(), std::declval<std::span<const HeaderDynamicLogPoint>>(),
  std::declval<std::span<const double>>()));

static_assert(fdapde::manifold::FirstOrderGeometry<HeaderGeometry>);
static_assert(fdapde::manifold::GeodesicGeometry<HeaderGeometry>);
static_assert(fdapde::manifold::FirstOrderGeometry<HeaderPowerGeometry>);
static_assert(fdapde::manifold::VectorTransportGeometry<HeaderLogGeometry>);
static_assert(fdapde::manifold::GeodesicGeometry<HeaderLogGeometry>);
static_assert(fdapde::manifold::VectorTransportGeometry<HeaderDynamicLogGeometry>);
static_assert(fdapde::manifold::GeodesicGeometry<HeaderDynamicLogGeometry>);
static_assert(fdapde::manifold::VectorTransportGeometry<HeaderAffineGeometry>);
static_assert(fdapde::manifold::GeodesicGeometry<HeaderAffineGeometry>);
static_assert(fdapde::manifold::VectorTransportGeometry<HeaderDynamicAffineGeometry>);
static_assert(fdapde::manifold::GeodesicGeometry<HeaderDynamicAffineGeometry>);
static_assert(fdapde::manifold::FirstOrderProblem<HeaderProblem, HeaderGeometry>);
static_assert(!fdapde::manifold::FirstOrderProblem<const HeaderProblem, HeaderGeometry>);
static_assert(fdapde::manifold::FirstOrderProblem<const HeaderConstProblem, HeaderGeometry>);
static_assert(fdapde::manifold::RiemannianHessianProblem<HeaderHessianProblem, HeaderGeometry>);
static_assert(!fdapde::manifold::RiemannianHessianProblem<const HeaderHessianProblem, HeaderGeometry>);
static_assert(std::is_default_constructible_v<HeaderContext>);
static_assert(std::is_default_constructible_v<HeaderSymmetricContext>);
static_assert(std::is_default_constructible_v<HeaderLogGeometry>);
static_assert(!std::is_constructible_v<HeaderLogGeometry, int>);
static_assert(!std::is_default_constructible_v<HeaderDynamicLogGeometry>);
static_assert(std::is_constructible_v<HeaderDynamicLogGeometry, int>);
static_assert(std::is_same_v<HeaderLogPoint, fdapde::linalg::SPDMatrix<double, 3, 3>>);
static_assert(std::is_same_v<HeaderLogTangent, fdapde::linalg::SymmetricMatrix<double, 3, 3>>);
static_assert(fdapde::manifold::FirstOrderProblem<const HeaderLogProblem, HeaderLogGeometry>);
static_assert(std::is_default_constructible_v<HeaderAffineGeometry>);
static_assert(!std::is_constructible_v<HeaderAffineGeometry, int>);
static_assert(!std::is_default_constructible_v<HeaderDynamicAffineGeometry>);
static_assert(std::is_constructible_v<HeaderDynamicAffineGeometry, int>);
static_assert(std::is_same_v<HeaderAffinePoint, fdapde::linalg::SPDMatrix<double, 3, 3>>);
static_assert(std::is_same_v<HeaderAffineTangent, fdapde::linalg::SymmetricMatrix<double, 3, 3>>);
static_assert(fdapde::manifold::FirstOrderProblem<const HeaderLogProblem, HeaderAffineGeometry>);
static_assert(fdapde::manifold::RiemannianHessianProblem<const HeaderSPDHessianProblem, HeaderAffineGeometry>);
static_assert(!HeaderPermitsRvalueCurrent<HeaderContext>);
static_assert(!HeaderPermitsRvalueEvaluationAccess<typename HeaderContext::Evaluation>);
static_assert(!HeaderPermitsRvalueComponentAccess<HeaderPowerGeometry>);
static_assert(!HeaderPermitsRvalueOptions<fdapde::manifold::ArmijoBacktracking>);
static_assert(!HeaderPermitsRvalueOptions<fdapde::manifold::RiemannianSteepestDescent>);
static_assert(!HeaderPermitsRvalueOptions<fdapde::manifold::SteihaugTruncatedCG>);
static_assert(!HeaderPermitsRvalueOptions<fdapde::manifold::RiemannianTrustRegion>);
static_assert(std::is_same_v<fdapde::manifold::point_t<HeaderPowerGeometry>, std::vector<HeaderPoint>>);
static_assert(std::is_same_v<fdapde::manifold::tangent_t<HeaderPowerGeometry>, std::vector<HeaderTangent>>);
static_assert(std::is_same_v<HeaderArmijoResult, fdapde::manifold::ArmijoResult<HeaderPoint>>);
static_assert(std::is_same_v<HeaderSteepestDescentResult, fdapde::manifold::SteepestDescentResult<HeaderPoint>>);
static_assert(std::is_same_v<HeaderTruncatedCGResult, fdapde::manifold::TruncatedCGResult<HeaderAffineTangent>>);
static_assert(std::is_same_v<HeaderTrustRegionResult, fdapde::manifold::TrustRegionResult<HeaderAffinePoint>>);
static_assert(std::is_same_v<HeaderMeanResult, fdapde::manifold::WeightedKarcherMeanResult<HeaderPoint>>);
static_assert(std::is_same_v<HeaderExactLogMeanResult, fdapde::manifold::WeightedKarcherMeanResult<HeaderLogPoint>>);
static_assert(
  std::is_same_v<HeaderExactDynamicLogMeanResult, fdapde::manifold::WeightedKarcherMeanResult<HeaderDynamicLogPoint>>);
static_assert(!HeaderPermitsMeanWithoutInitial<HeaderGeometry>);
static_assert(HeaderPermitsMeanWithoutInitial<HeaderLogGeometry>);
static_assert(HeaderPermitsMeanWithoutInitial<HeaderDynamicLogGeometry>);
static_assert(!HeaderPermitsMeanWithoutInitial<HeaderAffineGeometry>);

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

[[maybe_unused]] void instantiate_affine_geometry_solver() {
    fdapde::linalg::Matrix<double, 3, 3> identity;
    identity.set_zero();
    for (int i = 0; i < 3; ++i) { identity(i, i) = 1; }
    const HeaderAffinePoint point(identity, fdapde::linalg::checked);
    const HeaderAffineGeometry geometry;
    const HeaderLogProblem problem;
    const auto result = fdapde::manifold::RiemannianSteepestDescent {}.optimize(problem, geometry, point);
    static_cast<void>(result);
}

[[maybe_unused]] void instantiate_affine_geometry_trust_region_solvers() {
    fdapde::linalg::Matrix<double, 3, 3> identity;
    identity.set_zero();
    for (int i = 0; i < 3; ++i) { identity(i, i) = 1; }
    const HeaderAffinePoint point(identity, fdapde::linalg::checked);
    const HeaderAffineGeometry geometry;
    const HeaderSPDHessianProblem problem;
    HeaderWorkspace workspace;
    const HeaderAffineTangent gradient = problem.gradient(point, workspace);
    const auto subproblem =
      fdapde::manifold::SteihaugTruncatedCG {}.solve(problem, geometry, point, gradient, 1, workspace);
    const auto result = fdapde::manifold::RiemannianTrustRegion {}.optimize(problem, geometry, point);
    static_cast<void>(subproblem);
    static_cast<void>(result);
}

[[maybe_unused]] void instantiate_weighted_karcher_mean() {
    const HeaderGeometry geometry;
    const std::array<HeaderPoint, 2> samples {
      {{0}, {2}}
    };
    const std::array<double, 2> weights {
      {1, 1}
    };
    const auto result = fdapde::manifold::weighted_karcher_mean(
      geometry, std::span<const HeaderPoint>(samples), std::span<const double>(weights), HeaderPoint {});
    static_cast<void>(result);
}

[[maybe_unused]] void instantiate_exact_log_euclidean_weighted_mean() {
    fdapde::linalg::Matrix<double, 3, 3> identity;
    identity.set_zero();
    for (int i = 0; i < 3; ++i) { identity(i, i) = 1; }
    const std::array<double, 1> weights {{1}};

    const HeaderLogGeometry fixed_geometry;
    const std::array<HeaderLogPoint, 1> fixed_samples {{HeaderLogPoint(identity, fdapde::linalg::checked)}};
    const auto fixed_result = fdapde::manifold::weighted_karcher_mean(
      fixed_geometry, std::span<const HeaderLogPoint>(fixed_samples), std::span<const double>(weights));

    const HeaderDynamicLogGeometry dynamic_geometry(3);
    const std::array<HeaderDynamicLogPoint, 1> dynamic_samples {
      {HeaderDynamicLogPoint(identity, fdapde::linalg::checked)}};
    const auto dynamic_result = fdapde::manifold::weighted_karcher_mean(
      dynamic_geometry, std::span<const HeaderDynamicLogPoint>(dynamic_samples), std::span<const double>(weights));
    static_cast<void>(fixed_result);
    static_cast<void>(dynamic_result);
}

}   // namespace
