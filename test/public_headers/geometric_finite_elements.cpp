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

#include <fdaPDE/geometric_finite_elements.h>

#ifdef __FDAPDE_FINITE_ELEMENTS_MODULE_H__
#    error "The opt-in geometric finite-element header must not import the legacy finite-elements module."
#endif

#include <array>
#include <span>
#include <type_traits>
#include <utility>

namespace {

struct HeaderGeometry {
    using Point = double;
    using Tangent = double;

    std::size_t dimension() const { return 1; }
    double inner_product(const Point&, const Tangent& u, const Tangent& v) const { return u * v; }
    double norm(const Point&, const Tangent& tangent) const { return std::abs(tangent); }
    Tangent project(const Point&, const Tangent& tangent) const { return tangent; }
    Tangent zero_tangent(const Point&) const { return 0; }
    Tangent linear_combination(const Point&, double alpha, const Tangent& u, double beta, const Tangent& v) const {
        return alpha * u + beta * v;
    }
    Point retract(const Point& point, const Tangent& tangent, double step) const { return point + step * tangent; }
    Point exponential(const Point& point, const Tangent& tangent, double step) const { return point + step * tangent; }
    Tangent logarithm(const Point& from, const Point& to) const { return to - from; }
    double distance(const Point& from, const Point& to) const { return std::abs(to - from); }
};

using HeaderLogGeometry = fdapde::manifold::LogEuclideanSPDGeometry<double, 3>;
using HeaderDynamicLogGeometry = fdapde::manifold::LogEuclideanSPDGeometry<double, fdapde::Dynamic>;
using HeaderAffineGeometry = fdapde::manifold::AffineInvariantSPDGeometry<double, 3>;
using HeaderLogPoint = fdapde::manifold::point_t<HeaderLogGeometry>;
using HeaderDynamicLogPoint = fdapde::manifold::point_t<HeaderDynamicLogGeometry>;

template <typename Geometry>
concept HeaderPermitsP1WithoutInitial = requires(
  const Geometry& geometry, std::span<const fdapde::manifold::point_t<Geometry>> nodal_values,
  std::span<const double> weights) { fdapde::gfe::p1_geodesic_value(geometry, nodal_values, weights); };

template <typename Geometry>
concept HeaderPermitsP1WithInitial = requires(
  const Geometry& geometry, std::span<const fdapde::manifold::point_t<Geometry>> nodal_values,
  std::span<const double> weights, const fdapde::manifold::point_t<Geometry>& initial) {
    fdapde::gfe::p1_geodesic_value(geometry, nodal_values, weights, initial);
};

using HeaderGenericResult = decltype(fdapde::gfe::p1_geodesic_value(
  std::declval<const HeaderGeometry&>(), std::declval<std::span<const double>>(),
  std::declval<std::span<const double>>(), 0.0));
using HeaderLogResult = decltype(fdapde::gfe::p1_geodesic_value(
  std::declval<const HeaderLogGeometry&>(), std::declval<std::span<const HeaderLogPoint>>(),
  std::declval<std::span<const double>>()));
using HeaderDynamicLogResult = decltype(fdapde::gfe::p1_geodesic_value(
  std::declval<const HeaderDynamicLogGeometry&>(), std::declval<std::span<const HeaderDynamicLogPoint>>(),
  std::declval<std::span<const double>>()));

static_assert(std::is_same_v<HeaderGenericResult, fdapde::gfe::P1ValueResult<double>>);
static_assert(std::is_same_v<HeaderLogResult, fdapde::gfe::P1ValueResult<HeaderLogPoint>>);
static_assert(std::is_same_v<HeaderDynamicLogResult, fdapde::gfe::P1ValueResult<HeaderDynamicLogPoint>>);
static_assert(!HeaderPermitsP1WithoutInitial<HeaderGeometry>);
static_assert(HeaderPermitsP1WithoutInitial<HeaderLogGeometry>);
static_assert(HeaderPermitsP1WithoutInitial<HeaderDynamicLogGeometry>);
static_assert(!HeaderPermitsP1WithoutInitial<HeaderAffineGeometry>);
static_assert(HeaderPermitsP1WithInitial<HeaderGeometry>);
static_assert(!HeaderPermitsP1WithInitial<HeaderLogGeometry>);
static_assert(!HeaderPermitsP1WithInitial<HeaderDynamicLogGeometry>);
static_assert(HeaderPermitsP1WithInitial<HeaderAffineGeometry>);

[[maybe_unused]] void instantiate_generic_p1_value() {
    const HeaderGeometry geometry;
    const std::array<double, 2> nodal_values {
      {0, 1}
    };
    const std::array<double, 2> weights {
      {0.5, 0.5}
    };
    const auto result = fdapde::gfe::p1_geodesic_value(
      geometry, std::span<const double>(nodal_values), std::span<const double>(weights), 0.0);
    static_cast<void>(result);
}

[[maybe_unused]] void instantiate_log_euclidean_p1_values() {
    fdapde::linalg::Matrix<double, 3, 3> identity;
    identity.set_zero();
    for (int i = 0; i < 3; ++i) { identity(i, i) = 1; }
    const std::array<double, 1> weights {{1}};

    const HeaderLogGeometry fixed_geometry;
    const std::array<HeaderLogPoint, 1> fixed_values {{HeaderLogPoint(identity, fdapde::linalg::checked)}};
    const auto fixed_result = fdapde::gfe::p1_geodesic_value(
      fixed_geometry, std::span<const HeaderLogPoint>(fixed_values), std::span<const double>(weights));

    const HeaderDynamicLogGeometry dynamic_geometry(3);
    const std::array<HeaderDynamicLogPoint, 1> dynamic_values {
      {HeaderDynamicLogPoint(identity, fdapde::linalg::checked)}};
    const auto dynamic_result = fdapde::gfe::p1_geodesic_value(
      dynamic_geometry, std::span<const HeaderDynamicLogPoint>(dynamic_values), std::span<const double>(weights));
    static_cast<void>(fixed_result);
    static_cast<void>(dynamic_result);
}

}   // namespace
