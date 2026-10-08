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

#ifndef __FDAPDE_MANIFOLD_AFFINE_INVARIANT_SPD_H__
#define __FDAPDE_MANIFOLD_AFFINE_INVARIANT_SPD_H__

#include "../../geometry_expr.h"
#include "../../header_check.h"
#include "../spd_geometry_common.h"

namespace fdapde {
namespace manifold {

/// @brief defines the affine-invariant metric on checked SPD owners with ambient symmetric tangents
template <
  typename Scalar_, int Order_, Usage Uses_ = Usage::None,
  typename Point_ = fdapde::SPDMatrix<Scalar_, Order_, Cache::Policy<internals::affine_invariant_cache_flags(Uses_)>>>
class AffineInvariantSPDGeometry {
    fdapde_static_assert((static_cast<unsigned>(Uses_) & ~31u) == 0, SPD_GEOMETRY_USAGE_CONTAINS_UNKNOWN_FLAGS);
    fdapde_static_assert(
      std::is_floating_point_v<Scalar_> && !std::is_const_v<Scalar_> && !std::is_volatile_v<Scalar_>,
      SPD_GEOMETRIES_REQUIRE_AN_UNQUALIFIED_FLOATING_POINT_SCALAR);
    fdapde_static_assert(Order_ == fdapde::Dynamic || Order_ > 0, INVALID_SPD_GEOMETRY_ORDER);
    fdapde_static_assert(
      Order_ == fdapde::Dynamic || std::int64_t(Order_) * std::int64_t(Order_) <= std::numeric_limits<int>::max(),
      SPD_GEOMETRY_DENSE_WORKSPACE_SIZE_EXCEEDS_SUPPORTED_RANGE);
   public:
    using Scalar = Scalar_;
    using Point = Point_;
    using CachePolicy = typename Point::CachePolicy;
    using Tangent = fdapde::SymmetricMatrix<Scalar, Order_>;
    fdapde_static_assert(
      (std::same_as<Point, fdapde::SPDMatrix<Scalar, Order_, CachePolicy, Point::StorageOrder>>),
      AFFINE_INVARIANT_GEOMETRY_REQUIRES_A_NATIVE_SPD_OWNER_WITH_MATCHING_SCALAR_AND_SHAPE);

    /// @brief retains certified base factors and the spectral data of one scaled relative SPD point
    struct RelativeFrame {
        SPDMatrix<Scalar, Order_> from_sqrt;
        SPDMatrix<Scalar, Order_> from_inverse_sqrt;
        SPDMatrix<Scalar, Order_, Cache::Union<Cache::Spectral, Cache::Log, Cache::LogDividedDifferences>>
          scaled_relative;
        Scalar relative_scale;
    };

    /// @brief constructs the fixed-order geometry using its positive compile-time matrix order
    AffineInvariantSPDGeometry()
        requires(Order_ != fdapde::Dynamic)
    = default;

    /// @brief constructs a dynamic geometry after checking positive order and the supported dense workspace bound
    explicit AffineInvariantSPDGeometry(int order)
        requires(Order_ == fdapde::Dynamic)
        : order_(order) {
        internals::validate_spd_geometry_order(order_);
    }

    /// @brief prepares spatial P1 evaluation on a copied simplex or borrowed mesh with immutable batch data
    /// @details include geometric_finite_elements.h for the definition; simplex data follow local vertex order and mesh
    /// data follow global node ids
    template <typename Element, typename Nodes>
        requires gfe::P1InterpolationBinding<Element, Nodes>
    auto interpolant(Element&& element, Nodes&& nodes) const;
    /// @brief prepares the same interpolant with explicit mean and linear-solve tolerances
    template <typename Element, typename Nodes>
        requires gfe::P1InterpolationBinding<Element, Nodes>
    auto interpolant(Element&& element, Nodes&& nodes, const gfe::P1GeodesicLinearizationOptions& options) const;

    /// @brief returns the matrix order
    int order() const { return order_; }
    /// @brief returns the number of independent tangent coefficients
    std::size_t dimension() const { return internals::spd_geometry_dimension(order_); }

    /// @brief pairs ambient tangents in the metric at point
    template <SPDLike PointPoint>
    double inner_product(const PointPoint& point, const Tangent& u, const Tangent& v) const {
        check_point_(point);
        check_tangent_(u);
        check_tangent_(v);
        const auto inv_sqrt = internals::spd_inv_sqrt_factor(point);
        const auto whitened_u = internals::symmetric_congruence<Scalar, Order_>(inv_sqrt, u, order_);
        const auto whitened_v = internals::symmetric_congruence<Scalar, Order_>(inv_sqrt, v, order_);
        return static_cast<double>(internals::frobenius_inner(whitened_u, whitened_v, order_));
    }

    /// @brief returns the metric norm of an ambient tangent
    template <SPDLike PointPoint> double norm(const PointPoint& point, const Tangent& tangent) const {
        check_point_(point);
        check_tangent_(tangent);
        const auto inv_sqrt = internals::spd_inv_sqrt_factor(point);
        const auto whitened = internals::symmetric_congruence<Scalar, Order_>(inv_sqrt, tangent, order_);
        return internals::frobenius_norm(whitened, order_);
    }

    /// @brief copies an already symmetric ambient tangent into independent storage
    template <SPDLike PointPoint> Tangent project(const PointPoint& point, const Tangent& ambient) const {
        check_point_(point);
        check_tangent_(ambient);
        return ambient;
    }

    /// @brief returns an owning zero tangent of the point order
    template <SPDLike PointPoint> Tangent zero_tangent(const PointPoint& point) const {
        check_point_(point);
        auto result = internals::make_symmetric<Scalar, Order_>(order_);
        for (int i = 0; i < order_; ++i) {
            for (int j = 0; j <= i; ++j) { result(i, j) = Scalar(0); }
        }
        return result;
    }

    /// @brief combines ambient tangents with finite coefficients
    template <SPDLike PointPoint>
    Tangent
    linear_combination(const PointPoint& point, double alpha, const Tangent& u, double beta, const Tangent& v) const {
        check_point_(point);
        check_tangent_(u);
        check_tangent_(v);
        return internals::combine_symmetric<Scalar, Order_>(
          u, internals::geometry_coefficient<Scalar>(alpha), v, internals::geometry_coefficient<Scalar>(beta), order_);
    }

    /// @brief uses P^(1/2) * (I + W + W^2 / 2) * P^(1/2), W = step * P^(-1/2) * U * P^(-1/2)
    template <SPDLike PointPoint> Point retract(const PointPoint& point, const Tangent& tangent, double step) const {
        check_point_(point);
        check_tangent_(tangent);
        const auto point_sqrt = internals::spd_sqrt_factor(point);
        const auto point_inverse_sqrt = internals::spd_inv_sqrt_factor(point);
        const auto whitened = internals::symmetric_congruence<Scalar, Order_>(point_inverse_sqrt, tangent, order_);
        const auto scaled = internals::combine_symmetric<Scalar, Order_>(
          whitened, internals::geometry_coefficient<Scalar>(step), whitened, Scalar(0), order_);
        const auto squared = internals::symmetric_square<Scalar, Order_>(scaled, order_);

        auto polynomial = internals::make_symmetric<Scalar, Order_>(order_);
        for (int i = 0; i < order_; ++i) {
            for (int j = 0; j <= i; ++j) {
                polynomial(i, j) = (i == j ? Scalar(1) : Scalar(0)) + scaled(i, j) + Scalar(0.5) * squared(i, j);
            }
        }
        return Point(internals::symmetric_congruence<Scalar, Order_>(point_sqrt, polynomial, order_));
    }

    /// @brief follows the geodesic with the supplied initial ambient tangent and finite step
    template <SPDLike PointPoint>
    Point exponential(const PointPoint& point, const Tangent& tangent, double step = 1.0) const {
        check_point_(point);
        check_tangent_(tangent);
        const auto point_sqrt = internals::spd_sqrt_factor(point);
        const auto point_inverse_sqrt = internals::spd_inv_sqrt_factor(point);
        const auto whitened = internals::symmetric_congruence<Scalar, Order_>(point_inverse_sqrt, tangent, order_);
        const auto scaled = internals::combine_symmetric<Scalar, Order_>(
          whitened, internals::geometry_coefficient<Scalar>(step), whitened, Scalar(0), order_);
        const auto chart_exponential = fdapde::matrix_exp(scaled);
        return Point(internals::symmetric_congruence<Scalar, Order_>(point_sqrt, chart_exponential, order_));
    }

    /// @brief returns the initial ambient tangent of the geodesic from source to target
    template <SPDLike PointFrom, SPDLike PointTo> Tangent logarithm(const PointFrom& from, const PointTo& to) const {
        check_point_(from);
        check_point_(to);
        const auto from_sqrt = internals::spd_sqrt_factor(from);
        const auto from_inverse_sqrt = internals::spd_inv_sqrt_factor(from);
        const fdapde::SPDMatrix<Scalar, Order_> relative(
          internals::symmetric_congruence<Scalar, Order_>(from_inverse_sqrt, to, order_));
        return internals::symmetric_congruence<Scalar, Order_>(from_sqrt, fdapde::matrix_log(relative), order_);
    }

    /// @brief prepares an owning geodesic snapshot from two verified endpoints with independent cache policies
    /// @details the returned curve produces deferred expressions; SPD destinations certify their evaluated coefficients
    template <SPDLike PointFrom, SPDLike PointTo> auto geodesic(const PointFrom& from, const PointTo& to) const {
        check_point_(from);
        check_point_(to);
        const auto root = internals::spd_sqrt_factor(from);
        const auto inverse_root = internals::spd_inv_sqrt_factor(from);
        const fdapde::SPDMatrix<Scalar, Order_, Cache::Spectral> relative(
          internals::symmetric_congruence<Scalar, Order_>(inverse_root, to, order_));
        fdapde::Matrix<Scalar, Order_, Order_> factors(root * relative.cache().eigenvectors());
        fdapde::Vector<Scalar, Order_> logarithms;
        if constexpr (Order_ == fdapde::Dynamic) logarithms.resize(order_);
        for (int k = 0; k < order_; ++k) logarithms[k] = std::log(relative.cache().eigenvalues()[k]);
        return fdapde::internals::spd_geodesic<Scalar, Order_, true>(std::move(factors), std::move(logarithms));
    }
    /// @brief returns uniformly spaced geodesic samples including both endpoints with the selected output cache policy
    /// @details defaults to the geometry point policy; prepares one curve and requires count >= 2
    /// sample i uses t = i / (count - 1); execution defaults to sequential and parallel calls join before returning
    template <
      typename OutputPolicy = CachePolicy, SPDLike From, SPDLike To,
      fdapde::internals::BatchExecutionPolicy ExecutionPolicy = execution_seq_t>
    auto interpolate(const From& from, const To& to, int count, ExecutionPolicy policy = {}) const {
        return internals::sample_spd_geodesic<OutputPolicy>(*this, from, to, count, policy);
    }

    /// @brief applies the target differential of the affine-invariant logarithm
    /// @details exact target differential of Log_from(to)
    /// @details with R = c S, L_log(c S, V) = L_log(S, V) / c. The JVP absorbs
    /// @details 1/c into its whitening congruence, the metric VJP absorbs c into
    /// @details its unwhitening congruence, and c cancels from the Hessian action
    template <SPDLike From, SPDLike To>
    Tangent logarithm_target_jvp(const From& from, const To& to, const Tangent& to_direction) const {
        return logarithm_target_jvp(relative_frame(from, to), to_direction);
    }

    /// @brief evaluates the differential using a retained relative spectral frame
    Tangent logarithm_target_jvp(const RelativeFrame& frame, const Tangent& to_direction) const {
        check_tangent_(to_direction);

        const Scalar inverse_sqrt_scale = Scalar(1) / std::sqrt(frame.relative_scale);
        const auto scaled_inverse_sqrt = internals::combine_symmetric<Scalar, Order_>(
          frame.from_inverse_sqrt, inverse_sqrt_scale, frame.from_inverse_sqrt, Scalar(0), order_);
        const auto scaled_direction =
          internals::symmetric_congruence<Scalar, Order_>(scaled_inverse_sqrt, to_direction, order_);
        const auto chart_direction = logarithm_frechet_(frame.scaled_relative, scaled_direction);
        return checked_tangent_result_(
          internals::symmetric_congruence<Scalar, Order_>(frame.from_sqrt, chart_direction, order_));
    }

    /// @brief applies the target differential adjoint in the affine-invariant metric
    /// @details affine-invariant metric adjoint of logarithm_target_jvp. The argument and result
    /// @details are metric-dual tangent representations at from and to, respectively
    template <SPDLike From, SPDLike To>
    Tangent logarithm_target_vjp(const From& from, const To& to, const Tangent& from_metric_dual) const {
        return logarithm_target_vjp(relative_frame(from, to), from_metric_dual);
    }

    /// @brief evaluates the differential using a retained relative spectral frame
    Tangent logarithm_target_vjp(const RelativeFrame& frame, const Tangent& from_metric_dual) const {
        check_tangent_(from_metric_dual);

        const auto whitened_dual =
          internals::symmetric_congruence<Scalar, Order_>(frame.from_inverse_sqrt, from_metric_dual, order_);
        const auto chart_dual = logarithm_frechet_(frame.scaled_relative, whitened_dual);
        const auto relative_dual =
          internals::symmetric_congruence<Scalar, Order_>(frame.scaled_relative, chart_dual, order_);
        const Scalar sqrt_scale = std::sqrt(frame.relative_scale);
        const auto scaled_from_sqrt =
          internals::combine_symmetric<Scalar, Order_>(frame.from_sqrt, sqrt_scale, frame.from_sqrt, Scalar(0), order_);
        return checked_tangent_result_(
          internals::symmetric_congruence<Scalar, Order_>(scaled_from_sqrt, relative_dual, order_));
    }

    /// @brief applies the covariant base Hessian of half the squared distance
    /// @details covariant Hessian action at base of one half the squared distance to
    /// @details target. This is the negative covariant base differential of Log_base(target)
    template <SPDLike From, SPDLike To>
    Tangent
    half_squared_distance_hessian_vector(const From& base, const To& target, const Tangent& base_direction) const {
        return half_squared_distance_hessian_vector(relative_frame(base, target), base_direction);
    }

    /// @brief evaluates the differential using a retained relative spectral frame
    Tangent half_squared_distance_hessian_vector(const RelativeFrame& frame, const Tangent& base_direction) const {
        check_tangent_(base_direction);

        const auto whitened_direction =
          internals::symmetric_congruence<Scalar, Order_>(frame.from_inverse_sqrt, base_direction, order_);
        const auto log_direction = logarithm_frechet_(frame.scaled_relative, whitened_direction);
        // jordan_S and L_log(S, .) commute because they share S's spectral basis
        const auto chart_result = jordan_product_(frame.scaled_relative, log_direction);
        return checked_tangent_result_(
          internals::symmetric_congruence<Scalar, Order_>(frame.from_sqrt, chart_result, order_));
    }

    /// @brief differentiates the distance Hessian along simultaneous base and target variation
    /// @details covariant derivative of half_squared_distance_hessian_vector along
    /// @details simultaneous base/target variation. The action direction is continued
    /// @details parallelly along the base variation
    template <SPDLike From, SPDLike To>
    Tangent half_squared_distance_hessian_covariant_jvp(
      const From& base, const To& target, const Tangent& base_direction, const Tangent& target_direction,
      const Tangent& action_direction) const {
        return half_squared_distance_hessian_covariant_jvp(
          relative_frame(base, target), base_direction, target_direction, action_direction);
    }

    /// @brief evaluates the differential using a retained relative spectral frame
    Tangent half_squared_distance_hessian_covariant_jvp(
      const RelativeFrame& frame, const Tangent& base_direction, const Tangent& target_direction,
      const Tangent& action_direction) const {
        check_tangent_(base_direction);
        check_tangent_(target_direction);
        check_tangent_(action_direction);

        const auto whitened_base_direction =
          internals::symmetric_congruence<Scalar, Order_>(frame.from_inverse_sqrt, base_direction, order_);
        const auto whitened_action_direction =
          internals::symmetric_congruence<Scalar, Order_>(frame.from_inverse_sqrt, action_direction, order_);

        // scaling R = c S makes the relative variation E/c - Jordan(X, S)
        // this cancels both powers of c in the second logarithm differential
        const Scalar inverse_sqrt_scale = Scalar(1) / std::sqrt(frame.relative_scale);
        const auto scaled_inverse_sqrt = internals::combine_symmetric<Scalar, Order_>(
          frame.from_inverse_sqrt, inverse_sqrt_scale, frame.from_inverse_sqrt, Scalar(0), order_);
        const auto scaled_target_direction =
          internals::symmetric_congruence<Scalar, Order_>(scaled_inverse_sqrt, target_direction, order_);
        const auto base_relative_change = jordan_product_(whitened_base_direction, frame.scaled_relative);
        const auto relative_direction = internals::combine_symmetric<Scalar, Order_>(
          scaled_target_direction, Scalar(1), base_relative_change, Scalar(-1), order_);

        const auto log_action = logarithm_frechet_(frame.scaled_relative, whitened_action_direction);
        const auto log_second =
          fdapde::matrix_log_second_frechet(frame.scaled_relative, relative_direction, whitened_action_direction);
        const auto chart_result = internals::combine_symmetric<Scalar, Order_>(
          jordan_product_(relative_direction, log_action), Scalar(1),
          jordan_product_(frame.scaled_relative, log_second), Scalar(1), order_);
        return checked_tangent_result_(
          internals::symmetric_congruence<Scalar, Order_>(frame.from_sqrt, chart_result, order_));
    }

    /// @brief returns base and target metric adjoints of the Hessian variation
    /// @details metric adjoint of the simultaneous base/target variation in
    /// @details half_squared_distance_hessian_covariant_jvp, for a fixed parallel
    /// @details action direction. The pair contains base and target metric-dual tangents
    template <SPDLike From, SPDLike To>
    std::pair<Tangent, Tangent> half_squared_distance_hessian_covariant_vjp(
      const From& base, const To& target, const Tangent& action_direction, const Tangent& output_metric_dual) const {
        return half_squared_distance_hessian_covariant_vjp(
          relative_frame(base, target), action_direction, output_metric_dual);
    }

    /// @brief evaluates the differential using a retained relative spectral frame
    std::pair<Tangent, Tangent> half_squared_distance_hessian_covariant_vjp(
      const RelativeFrame& frame, const Tangent& action_direction, const Tangent& output_metric_dual) const {
        check_tangent_(action_direction);
        check_tangent_(output_metric_dual);

        const auto whitened_action =
          internals::symmetric_congruence<Scalar, Order_>(frame.from_inverse_sqrt, action_direction, order_);
        const auto whitened_output =
          internals::symmetric_congruence<Scalar, Order_>(frame.from_inverse_sqrt, output_metric_dual, order_);
        const auto log_action = logarithm_frechet_(frame.scaled_relative, whitened_action);
        const auto relative_output = jordan_product_(frame.scaled_relative, whitened_output);
        const auto log_second =
          fdapde::matrix_log_second_frechet(frame.scaled_relative, relative_output, whitened_action);
        const auto variation_dual = internals::combine_symmetric<Scalar, Order_>(
          jordan_product_(whitened_output, log_action), Scalar(1), log_second, Scalar(1), order_);

        auto base_chart_dual = jordan_product_(frame.scaled_relative, variation_dual);
        for (int i = 0; i < order_; ++i) {
            for (int j = 0; j <= i; ++j) { base_chart_dual(i, j) = -base_chart_dual(i, j); }
        }
        Tangent base_dual = checked_tangent_result_(
          internals::symmetric_congruence<Scalar, Order_>(frame.from_sqrt, base_chart_dual, order_));

        const auto target_chart_dual =
          internals::symmetric_congruence<Scalar, Order_>(frame.scaled_relative, variation_dual, order_);
        const Scalar sqrt_scale = std::sqrt(frame.relative_scale);
        const auto scaled_from_sqrt =
          internals::combine_symmetric<Scalar, Order_>(frame.from_sqrt, sqrt_scale, frame.from_sqrt, Scalar(0), order_);
        Tangent target_dual = checked_tangent_result_(
          internals::symmetric_congruence<Scalar, Order_>(scaled_from_sqrt, target_chart_dual, order_));
        return {std::move(base_dual), std::move(target_dual)};
    }

    /// @brief returns the geodesic distance between checked points
    template <SPDLike PointFrom, SPDLike PointTo> double distance(const PointFrom& from, const PointTo& to) const {
        check_point_(from);
        check_point_(to);
        const auto from_inverse_sqrt = internals::spd_inv_sqrt_factor(from);
        const fdapde::SPDMatrix<Scalar, Order_> relative(
          internals::symmetric_congruence<Scalar, Order_>(from_inverse_sqrt, to, order_));
        return internals::frobenius_norm(fdapde::matrix_log(relative), order_);
    }

    /// @brief parallel-transports an ambient tangent along the source-to-target geodesic
    template <SPDLike PointFrom, SPDLike PointTo>
    Tangent transport(const PointFrom& from, const PointTo& to, const Tangent& tangent) const {
        check_point_(from);
        check_point_(to);
        check_tangent_(tangent);
        const auto from_sqrt = internals::spd_sqrt_factor(from);
        const auto from_inverse_sqrt = internals::spd_inv_sqrt_factor(from);
        const fdapde::SPDMatrix<Scalar, Order_> relative(
          internals::symmetric_congruence<Scalar, Order_>(from_inverse_sqrt, to, order_));
        const auto relative_sqrt = internals::spd_sqrt_factor(relative);
        const auto whitened = internals::symmetric_congruence<Scalar, Order_>(from_inverse_sqrt, tangent, order_);
        const auto transported_whitened =
          internals::symmetric_congruence<Scalar, Order_>(relative_sqrt, whitened, order_);
        return internals::symmetric_congruence<Scalar, Order_>(from_sqrt, transported_whitened, order_);
    }

    /// @brief converts a symmetric Frobenius gradient to its Riemannian metric dual
    template <SPDLike PointPoint>
    Tangent euclidean_to_riemannian_gradient(const PointPoint& point, const Tangent& euclidean_gradient) const {
        check_point_(point);
        check_tangent_(euclidean_gradient);
        return internals::symmetric_congruence<Scalar, Order_>(point, euclidean_gradient, order_);
    }


    /// @brief prepares certified base roots and the cached scaled relative SPD spectrum
    template <SPDLike From, SPDLike To> RelativeFrame relative_frame(const From& from, const To& to) const {
        check_point_(from);
        check_point_(to);
        for (int i = 0; i < order_; ++i) {
            for (int j = 0; j <= i; ++j) {
                fdapde_strong_assert(
                  std::isfinite(static_cast<Scalar>(to(i, j))), std::invalid_argument,
                  "Affine-invariant SPD differential point coefficients must be finite");
            }
        }
        auto from_sqrt = fdapde::matrix_sqrt(from);
        auto from_inverse_sqrt = fdapde::matrix_inv_sqrt(from);
        const auto relative = internals::symmetric_congruence<Scalar, Order_>(from_inverse_sqrt, to, order_);

        Scalar scale = 0;
        for (int i = 0; i < order_; ++i) {
            for (int j = 0; j <= i; ++j) {
                const Scalar coefficient = static_cast<Scalar>(relative(i, j));
                fdapde_strong_assert(
                  std::isfinite(coefficient), std::domain_error,
                  "Affine-invariant SPD differential produced a nonfinite relative point");
                scale = std::max(scale, std::abs(coefficient));
            }
        }
        fdapde_strong_assert(
          (scale > Scalar(0)) && std::isfinite(scale), std::domain_error,
          "Affine-invariant SPD differential has an invalid relative scale");

        auto scaled_relative = internals::make_symmetric<Scalar, Order_>(order_);
        for (int i = 0; i < order_; ++i) {
            for (int j = 0; j <= i; ++j) { scaled_relative(i, j) = static_cast<Scalar>(relative(i, j)) / scale; }
        }
        return {
          decltype(RelativeFrame::from_sqrt)(from_sqrt), decltype(RelativeFrame::from_inverse_sqrt)(from_inverse_sqrt),
          decltype(RelativeFrame::scaled_relative)(scaled_relative), scale};
    }

    /// @brief reconstructs the logarithm at the base from the cached scaled relative chart
    Tangent logarithm(const RelativeFrame& frame) const {
        Tangent chart(fdapde::matrix_log(frame.scaled_relative));
        for (int i = 0; i < order_; ++i) chart(i, i) = Scalar(chart(i, i)) + std::log(frame.relative_scale);
        return internals::symmetric_congruence<Scalar, Order_>(frame.from_sqrt, chart, order_);
    }
    /// @brief computes distance using the retained relative chart and scale
    double distance(const RelativeFrame& frame) const {
        Tangent chart(fdapde::matrix_log(frame.scaled_relative));
        for (int i = 0; i < order_; ++i) chart(i, i) = Scalar(chart(i, i)) + std::log(frame.relative_scale);
        return internals::frobenius_norm(chart, order_);
    }
   private:
    /// @brief reuses the relative spectrum and logarithm divided differences
    template <SPDLike Relative> Tangent logarithm_frechet_(const Relative& point, const Tangent& direction) const {
        return fdapde::matrix_log_frechet(point, direction);
    }

    /// @brief forms the symmetric Jordan product without a dense temporary
    template <typename LhsType_, typename RhsType_>
    Tangent jordan_product_(const LhsType_& lhs, const RhsType_& rhs) const {
        auto result = internals::make_symmetric<Scalar, Order_>(order_);
        for (int i = 0; i < order_; ++i) {
            for (int j = 0; j <= i; ++j) {
                Scalar value = 0;
                for (int k = 0; k < order_; ++k) {
                    value += Scalar(0.5) * (static_cast<Scalar>(lhs(i, k)) * static_cast<Scalar>(rhs(k, j)) +
                                            static_cast<Scalar>(rhs(i, k)) * static_cast<Scalar>(lhs(k, j)));
                }
                result(i, j) = value;
            }
        }
        return result;
    }

    /// @brief rejects nonfinite differential coefficients before returning a tangent
    Tangent checked_tangent_result_(Tangent result) const {
        for (int i = 0; i < order_; ++i) {
            for (int j = 0; j <= i; ++j) {
                fdapde_strong_assert(
                  std::isfinite(static_cast<Scalar>(result(i, j))), std::domain_error,
                  "Affine-invariant SPD differential produced a nonfinite tangent");
            }
        }
        return result;
    }

    /// @brief checks the point order and finite packed coefficients against this geometry
    template <SPDLike PointPoint> void check_point_(const PointPoint& point) const {
        internals::check_spd_geometry_shape(point, order_);
    }
    /// @brief checks the tangent order and finite packed coefficients against this geometry
    void check_tangent_(const Tangent& tangent) const { internals::check_spd_geometry_shape(tangent, order_); }

    int order_ = Order_ == fdapde::Dynamic ? 0 : Order_;
};

namespace internals {

/// @brief selects the affine-invariant metric while preserving the exact native SPD owner
template <typename Point> struct affine_invariant_geometry_type {
    using type = AffineInvariantSPDGeometry<typename Point::Scalar, Point::Rows, Usage::None, Point>;
};

}   // namespace internals

/// @brief supplies the affine-invariant metric for a native SPD owner with its cache policy
template <typename Point>
using AffineInvariantGeometry = typename internals::affine_invariant_geometry_type<Point>::type;

/// @brief computes the weighted mean with explicit convergence diagnostics
template <typename Scalar_, int Order_, Usage Uses_, typename Samples, typename Point_>
WeightedKarcherMeanResult<typename AffineInvariantSPDGeometry<Scalar_, Order_, Uses_, Point_>::Point>
weighted_karcher_mean(
  const AffineInvariantSPDGeometry<Scalar_, Order_, Uses_, Point_>& geometry, const Samples& samples,
  std::span<const double> weights,
  const typename AffineInvariantSPDGeometry<Scalar_, Order_, Uses_, Point_>::Point& initial,
  const WeightedKarcherMeanOptions& options = {},
  internals::KarcherWorkspace<AffineInvariantSPDGeometry<Scalar_, Order_, Uses_, Point_>>* retained = nullptr) {
    using Geometry = AffineInvariantSPDGeometry<Scalar_, Order_, Uses_, Point_>;
    internals::KarcherWorkspace<Geometry> workspace;
    auto result = weighted_karcher_mean<Geometry>(geometry, samples, weights, initial, options, &workspace);
    internals::polish_karcher_mean(geometry, samples, options, result, workspace);
    if (retained) *retained = std::move(workspace);
    result.uniqueness = BarycenterUniqueness::globally_unique;
    return result;
}

/// @brief computes the weighted mean with explicit convergence diagnostics
template <typename Scalar_, int Order_, Usage Uses_, typename Samples, typename Point_>
WeightedKarcherMeanResult<typename AffineInvariantSPDGeometry<Scalar_, Order_, Uses_, Point_>::Point>
weighted_karcher_mean(
  const AffineInvariantSPDGeometry<Scalar_, Order_, Uses_, Point_>& geometry, const Samples& samples,
  std::span<const double> weights, const WeightedKarcherMeanOptions& options = {},
  internals::KarcherWorkspace<AffineInvariantSPDGeometry<Scalar_, Order_, Uses_, Point_>>* retained = nullptr) {
    auto log_geometry = [&]() {
        if constexpr (Order_ == fdapde::Dynamic) {
            return LogEuclideanSPDGeometry<Scalar_, Order_>(geometry.order());
        } else {
            return LogEuclideanSPDGeometry<Scalar_, Order_>();
        }
    }();
    /// @details use the exact log-Euclidean mean formula as a deterministic, sample-symmetric positive-definite
    /// initializer
    const auto initial = weighted_karcher_mean(log_geometry, samples, weights);
    return weighted_karcher_mean(
      geometry, samples, weights,
      typename AffineInvariantSPDGeometry<Scalar_, Order_, Uses_, Point_>::Point(initial.point), options, retained);
}

}   // namespace manifold
}   // namespace fdapde

#endif   // __FDAPDE_MANIFOLD_AFFINE_INVARIANT_SPD_H__
