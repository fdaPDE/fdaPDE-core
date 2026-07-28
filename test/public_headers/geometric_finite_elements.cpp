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
#include <concepts>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

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
using HeaderDynamicAffineGeometry = fdapde::manifold::AffineInvariantSPDGeometry<double, fdapde::Dynamic>;
using HeaderLogPoint = fdapde::manifold::point_t<HeaderLogGeometry>;
using HeaderDynamicLogPoint = fdapde::manifold::point_t<HeaderDynamicLogGeometry>;
using HeaderAffinePoint = fdapde::manifold::point_t<HeaderAffineGeometry>;
using HeaderDynamicAffinePoint = fdapde::manifold::point_t<HeaderDynamicAffineGeometry>;
using HeaderLogTangent = fdapde::manifold::tangent_t<HeaderLogGeometry>;
using HeaderDynamicLogTangent = fdapde::manifold::tangent_t<HeaderDynamicLogGeometry>;
using HeaderAffineTangent = fdapde::manifold::tangent_t<HeaderAffineGeometry>;
using HeaderDynamicAffineTangent = fdapde::manifold::tangent_t<HeaderDynamicAffineGeometry>;

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

template <typename Geometry>
concept HeaderPermitsP1WithOptions = requires(
  const Geometry& geometry, std::span<const fdapde::manifold::point_t<Geometry>> nodal_values,
  std::span<const double> weights, const fdapde::manifold::WeightedKarcherMeanOptions& options) {
    fdapde::gfe::p1_geodesic_value(geometry, nodal_values, weights, options);
};

template <typename Geometry>
concept HeaderPermitsP1Linearization = requires(
  const Geometry& geometry, std::span<const fdapde::manifold::point_t<Geometry>> nodal_values,
  std::span<const double> weights) { fdapde::gfe::p1_geodesic_linearization(geometry, nodal_values, weights); };

template <typename Geometry>
concept HeaderPermitsP1LinearizationWithInitial = requires(
  const Geometry& geometry, std::span<const fdapde::manifold::point_t<Geometry>> nodal_values,
  std::span<const double> weights, const fdapde::manifold::point_t<Geometry>& initial,
  const fdapde::gfe::P1GeodesicLinearizationOptions& options) {
    fdapde::gfe::p1_geodesic_linearization(geometry, nodal_values, weights, initial, options);
};

template <typename Linearization>
concept HeaderPermitsRvalueLinearizationResult =
  requires(Linearization&& linearization) { std::move(linearization).result(); };

template <typename Linearization, typename Tangent>
concept HeaderP1LinearizationActions = requires(
  const Linearization& linearization, std::span<const double> weight_direction,
  std::span<const Tangent> nodal_directions, const Tangent& output_direction) {
    { linearization.weight_jvp(weight_direction) } -> std::same_as<Tangent>;
    { linearization.nodal_jvp(nodal_directions) } -> std::same_as<Tangent>;
    { linearization.nodal_vjp(output_direction) } -> std::same_as<std::vector<Tangent>>;
};

template <typename Linearization, typename Tangent>
concept HeaderAffineP1LinearizationActions = requires(
  const Linearization& linearization, std::span<const double> weight_direction,
  std::span<const Tangent> nodal_directions, const Tangent& output_direction) {
    { linearization.weight_jvp(weight_direction) } -> std::same_as<fdapde::gfe::P1DerivativeResult<Tangent>>;
    { linearization.nodal_jvp(nodal_directions) } -> std::same_as<fdapde::gfe::P1DerivativeResult<Tangent>>;
    {
        linearization.nodal_vjp(output_direction)
    } -> std::same_as<fdapde::gfe::P1DerivativeResult<std::vector<Tangent>>>;
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
using HeaderAffineResult = decltype(fdapde::gfe::p1_geodesic_value(
  std::declval<const HeaderAffineGeometry&>(), std::declval<std::span<const HeaderAffinePoint>>(),
  std::declval<std::span<const double>>()));
using HeaderDynamicAffineResult = decltype(fdapde::gfe::p1_geodesic_value(
  std::declval<const HeaderDynamicAffineGeometry&>(), std::declval<std::span<const HeaderDynamicAffinePoint>>(),
  std::declval<std::span<const double>>()));
using HeaderLogLinearization = decltype(fdapde::gfe::p1_geodesic_linearization(
  std::declval<const HeaderLogGeometry&>(), std::declval<std::span<const HeaderLogPoint>>(),
  std::declval<std::span<const double>>()));
using HeaderDynamicLogLinearization = decltype(fdapde::gfe::p1_geodesic_linearization(
  std::declval<const HeaderDynamicLogGeometry&>(), std::declval<std::span<const HeaderDynamicLogPoint>>(),
  std::declval<std::span<const double>>()));
using HeaderAffineLinearization = decltype(fdapde::gfe::p1_geodesic_linearization(
  std::declval<const HeaderAffineGeometry&>(), std::declval<std::span<const HeaderAffinePoint>>(),
  std::declval<std::span<const double>>()));
using HeaderDynamicAffineLinearization = decltype(fdapde::gfe::p1_geodesic_linearization(
  std::declval<const HeaderDynamicAffineGeometry&>(), std::declval<std::span<const HeaderDynamicAffinePoint>>(),
  std::declval<std::span<const double>>()));

static_assert(std::is_same_v<HeaderGenericResult, fdapde::gfe::P1ValueResult<double>>);
static_assert(std::is_same_v<HeaderLogResult, fdapde::gfe::P1ValueResult<HeaderLogPoint>>);
static_assert(std::is_same_v<HeaderDynamicLogResult, fdapde::gfe::P1ValueResult<HeaderDynamicLogPoint>>);
static_assert(std::is_same_v<HeaderAffineResult, fdapde::gfe::P1ValueResult<HeaderAffinePoint>>);
static_assert(std::is_same_v<HeaderDynamicAffineResult, fdapde::gfe::P1ValueResult<HeaderDynamicAffinePoint>>);
static_assert(!HeaderPermitsP1WithoutInitial<HeaderGeometry>);
static_assert(HeaderPermitsP1WithoutInitial<HeaderLogGeometry>);
static_assert(HeaderPermitsP1WithoutInitial<HeaderDynamicLogGeometry>);
static_assert(HeaderPermitsP1WithoutInitial<HeaderAffineGeometry>);
static_assert(HeaderPermitsP1WithoutInitial<HeaderDynamicAffineGeometry>);
static_assert(HeaderPermitsP1WithInitial<HeaderGeometry>);
static_assert(!HeaderPermitsP1WithInitial<HeaderLogGeometry>);
static_assert(!HeaderPermitsP1WithInitial<HeaderDynamicLogGeometry>);
static_assert(HeaderPermitsP1WithInitial<HeaderAffineGeometry>);
static_assert(HeaderPermitsP1WithInitial<HeaderDynamicAffineGeometry>);
static_assert(!HeaderPermitsP1WithOptions<HeaderGeometry>);
static_assert(!HeaderPermitsP1WithOptions<HeaderLogGeometry>);
static_assert(!HeaderPermitsP1WithOptions<HeaderDynamicLogGeometry>);
static_assert(HeaderPermitsP1WithOptions<HeaderAffineGeometry>);
static_assert(HeaderPermitsP1WithOptions<HeaderDynamicAffineGeometry>);
static_assert(!HeaderPermitsP1Linearization<HeaderGeometry>);
static_assert(HeaderPermitsP1Linearization<HeaderLogGeometry>);
static_assert(HeaderPermitsP1Linearization<HeaderDynamicLogGeometry>);
static_assert(HeaderPermitsP1Linearization<HeaderAffineGeometry>);
static_assert(HeaderPermitsP1Linearization<HeaderDynamicAffineGeometry>);
static_assert(HeaderPermitsP1LinearizationWithInitial<HeaderAffineGeometry>);
static_assert(HeaderPermitsP1LinearizationWithInitial<HeaderDynamicAffineGeometry>);
static_assert(std::is_same_v<HeaderLogLinearization, fdapde::gfe::P1GeodesicLinearization<HeaderLogGeometry>>);
static_assert(
  std::is_same_v<HeaderDynamicLogLinearization, fdapde::gfe::P1GeodesicLinearization<HeaderDynamicLogGeometry>>);
static_assert(std::is_same_v<HeaderAffineLinearization, fdapde::gfe::P1GeodesicLinearization<HeaderAffineGeometry>>);
static_assert(
  std::is_same_v<HeaderDynamicAffineLinearization, fdapde::gfe::P1GeodesicLinearization<HeaderDynamicAffineGeometry>>);
static_assert(std::is_same_v<decltype(std::declval<const HeaderLogLinearization&>().result()), const HeaderLogResult&>);
static_assert(std::is_same_v<
              decltype(std::declval<const HeaderDynamicLogLinearization&>().result()), const HeaderDynamicLogResult&>);
static_assert(
  std::is_same_v<decltype(std::declval<const HeaderAffineLinearization&>().result()), const HeaderAffineResult&>);
static_assert(
  std::is_same_v<
    decltype(std::declval<const HeaderDynamicAffineLinearization&>().result()), const HeaderDynamicAffineResult&>);
static_assert(!HeaderPermitsRvalueLinearizationResult<HeaderLogLinearization>);
static_assert(!HeaderPermitsRvalueLinearizationResult<HeaderDynamicLogLinearization>);
static_assert(!HeaderPermitsRvalueLinearizationResult<HeaderAffineLinearization>);
static_assert(!HeaderPermitsRvalueLinearizationResult<HeaderDynamicAffineLinearization>);
static_assert(HeaderP1LinearizationActions<HeaderLogLinearization, HeaderLogTangent>);
static_assert(HeaderP1LinearizationActions<HeaderDynamicLogLinearization, HeaderDynamicLogTangent>);
static_assert(HeaderAffineP1LinearizationActions<HeaderAffineLinearization, HeaderAffineTangent>);
static_assert(HeaderAffineP1LinearizationActions<HeaderDynamicAffineLinearization, HeaderDynamicAffineTangent>);

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
    const auto fixed_linearization = fdapde::gfe::p1_geodesic_linearization(
      fixed_geometry, std::span<const HeaderLogPoint>(fixed_values), std::span<const double>(weights));
    const std::array<double, 1> weight_direction {{0}};
    const std::array<HeaderLogTangent, 1> fixed_nodal_directions {{fixed_geometry.zero_tangent(fixed_values[0])}};
    const auto fixed_weight_jvp = fixed_linearization.weight_jvp(std::span<const double>(weight_direction));
    const auto fixed_nodal_jvp =
      fixed_linearization.nodal_jvp(std::span<const HeaderLogTangent>(fixed_nodal_directions));
    const auto fixed_nodal_vjp = fixed_linearization.nodal_vjp(fixed_nodal_directions[0]);

    const HeaderDynamicLogGeometry dynamic_geometry(3);
    const std::array<HeaderDynamicLogPoint, 1> dynamic_values {
      {HeaderDynamicLogPoint(identity, fdapde::linalg::checked)}};
    const auto dynamic_result = fdapde::gfe::p1_geodesic_value(
      dynamic_geometry, std::span<const HeaderDynamicLogPoint>(dynamic_values), std::span<const double>(weights));
    const auto dynamic_linearization = fdapde::gfe::p1_geodesic_linearization(
      dynamic_geometry, std::span<const HeaderDynamicLogPoint>(dynamic_values), std::span<const double>(weights));
    const std::array<HeaderDynamicLogTangent, 1> dynamic_nodal_directions {
      {dynamic_geometry.zero_tangent(dynamic_values[0])}};
    const auto dynamic_weight_jvp = dynamic_linearization.weight_jvp(std::span<const double>(weight_direction));
    const auto dynamic_nodal_jvp =
      dynamic_linearization.nodal_jvp(std::span<const HeaderDynamicLogTangent>(dynamic_nodal_directions));
    const auto dynamic_nodal_vjp = dynamic_linearization.nodal_vjp(dynamic_nodal_directions[0]);
    static_cast<void>(fixed_result);
    static_cast<void>(fixed_linearization.result());
    static_cast<void>(fixed_weight_jvp);
    static_cast<void>(fixed_nodal_jvp);
    static_cast<void>(fixed_nodal_vjp);
    static_cast<void>(dynamic_result);
    static_cast<void>(dynamic_linearization.result());
    static_cast<void>(dynamic_weight_jvp);
    static_cast<void>(dynamic_nodal_jvp);
    static_cast<void>(dynamic_nodal_vjp);
}

[[maybe_unused]] void instantiate_affine_invariant_p1_values() {
    fdapde::linalg::Matrix<double, 3, 3> identity;
    identity.set_zero();
    for (int i = 0; i < 3; ++i) { identity(i, i) = 1; }
    const std::array<double, 1> weights {{1}};
    const fdapde::manifold::WeightedKarcherMeanOptions options;
    const fdapde::gfe::P1GeodesicLinearizationOptions linearization_options;
    const std::array<double, 1> weight_direction {{0}};

    const HeaderAffineGeometry fixed_geometry;
    const std::array<HeaderAffinePoint, 1> fixed_values {{HeaderAffinePoint(identity, fdapde::linalg::checked)}};
    const auto fixed_default = fdapde::gfe::p1_geodesic_value(
      fixed_geometry, std::span<const HeaderAffinePoint>(fixed_values), std::span<const double>(weights));
    const auto fixed_options = fdapde::gfe::p1_geodesic_value(
      fixed_geometry, std::span<const HeaderAffinePoint>(fixed_values), std::span<const double>(weights), options);
    const auto fixed_initial = fdapde::gfe::p1_geodesic_value(
      fixed_geometry, std::span<const HeaderAffinePoint>(fixed_values), std::span<const double>(weights),
      fixed_values[0], options);
    const auto fixed_linearization = fdapde::gfe::p1_geodesic_linearization(
      fixed_geometry, std::span<const HeaderAffinePoint>(fixed_values), std::span<const double>(weights));
    const auto fixed_initial_linearization = fdapde::gfe::p1_geodesic_linearization(
      fixed_geometry, std::span<const HeaderAffinePoint>(fixed_values), std::span<const double>(weights),
      fixed_values[0], linearization_options);
    const std::array<HeaderAffineTangent, 1> fixed_directions {{fixed_geometry.zero_tangent(fixed_values[0])}};
    const auto fixed_weight_jvp = fixed_linearization.weight_jvp(std::span<const double>(weight_direction));
    const auto fixed_nodal_jvp =
      fixed_initial_linearization.nodal_jvp(std::span<const HeaderAffineTangent>(fixed_directions));
    const auto fixed_nodal_vjp = fixed_initial_linearization.nodal_vjp(fixed_directions[0]);

    const HeaderDynamicAffineGeometry dynamic_geometry(3);
    const std::array<HeaderDynamicAffinePoint, 1> dynamic_values {
      {HeaderDynamicAffinePoint(identity, fdapde::linalg::checked)}};
    const auto dynamic_default = fdapde::gfe::p1_geodesic_value(
      dynamic_geometry, std::span<const HeaderDynamicAffinePoint>(dynamic_values), std::span<const double>(weights));
    const auto dynamic_options = fdapde::gfe::p1_geodesic_value(
      dynamic_geometry, std::span<const HeaderDynamicAffinePoint>(dynamic_values), std::span<const double>(weights),
      options);
    const auto dynamic_initial = fdapde::gfe::p1_geodesic_value(
      dynamic_geometry, std::span<const HeaderDynamicAffinePoint>(dynamic_values), std::span<const double>(weights),
      dynamic_values[0], options);
    const auto dynamic_linearization = fdapde::gfe::p1_geodesic_linearization(
      dynamic_geometry, std::span<const HeaderDynamicAffinePoint>(dynamic_values), std::span<const double>(weights));
    const auto dynamic_initial_linearization = fdapde::gfe::p1_geodesic_linearization(
      dynamic_geometry, std::span<const HeaderDynamicAffinePoint>(dynamic_values), std::span<const double>(weights),
      dynamic_values[0], linearization_options);
    const std::array<HeaderDynamicAffineTangent, 1> dynamic_directions {
      {dynamic_geometry.zero_tangent(dynamic_values[0])}};
    const auto dynamic_weight_jvp = dynamic_linearization.weight_jvp(std::span<const double>(weight_direction));
    const auto dynamic_nodal_jvp =
      dynamic_initial_linearization.nodal_jvp(std::span<const HeaderDynamicAffineTangent>(dynamic_directions));
    const auto dynamic_nodal_vjp = dynamic_initial_linearization.nodal_vjp(dynamic_directions[0]);
    static_cast<void>(fixed_default);
    static_cast<void>(fixed_options);
    static_cast<void>(fixed_initial);
    static_cast<void>(fixed_linearization.result());
    static_cast<void>(fixed_weight_jvp);
    static_cast<void>(fixed_nodal_jvp);
    static_cast<void>(fixed_nodal_vjp);
    static_cast<void>(dynamic_default);
    static_cast<void>(dynamic_options);
    static_cast<void>(dynamic_initial);
    static_cast<void>(dynamic_linearization.result());
    static_cast<void>(dynamic_weight_jvp);
    static_cast<void>(dynamic_nodal_jvp);
    static_cast<void>(dynamic_nodal_vjp);
}

}   // namespace
