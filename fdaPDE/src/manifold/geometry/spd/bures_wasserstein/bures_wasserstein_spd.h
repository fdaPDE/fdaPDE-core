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

#ifndef __FDAPDE_MANIFOLD_BURES_WASSERSTEIN_SPD_H__
#define __FDAPDE_MANIFOLD_BURES_WASSERSTEIN_SPD_H__

#include "../../../header_check.h"
#include "../../geometry_expr.h"
#include "../spd_geometry_common.h"

namespace fdapde {
namespace manifold {
namespace internals {

/// @brief retains base factors, tangent spectra or log-chart differentials requested by BW operations
constexpr unsigned bures_wasserstein_cache_flags(Usage uses) {
    return (has_spd_usage(uses, Usage::Distance | Usage::BasePointMaps) ?
              Cache::Sqrt::Flags | Cache::InverseSqrt::Flags :
              0u) |
           (has_spd_usage(
              uses, Usage::Distance | Usage::TangentMetric | Usage::BasePointMaps | Usage::LogExpDifferentials) ?
              Cache::Spectral::Flags :
              0u) |
           (has_spd_usage(uses, Usage::LogExpDifferentials) ? Cache::LogDividedDifferences::Flags : 0u);
}

}   // namespace internals

/// @brief defines the Bures-Wasserstein metric with exact first derivatives on native SPD values
/// @details tangents use ambient symmetric coordinates; the metric is one half the Frobenius pairing with the
/// solution of A X + X A = U, and exponential steps stay on the positive-definite horizontal lift
template <
  typename Scalar_, int Order_, Usage Uses_ = Usage::None,
  typename Point_ = fdapde::SPDMatrix<Scalar_, Order_, Cache::Policy<internals::bures_wasserstein_cache_flags(Uses_)>>>
class BuresWassersteinSPDGeometry {
    fdapde_static_assert((static_cast<unsigned>(Uses_) & ~31u) == 0, SPD_GEOMETRY_USAGE_CONTAINS_UNKNOWN_FLAGS);
    fdapde_static_assert(
      std::is_floating_point_v<Scalar_> && !std::is_const_v<Scalar_> && !std::is_volatile_v<Scalar_>,
      SPD_GEOMETRIES_REQUIRE_AN_UNQUALIFIED_FLOATING_POINT_SCALAR);
    fdapde_static_assert(Order_ == fdapde::Dynamic || Order_ > 0, INVALID_SPD_GEOMETRY_ORDER);
    static constexpr bool small_transport = Order_ == 2 || Order_ == 3;
    static constexpr int transport_coefficients = small_transport ? Order_ * (Order_ + 1) / 2 : 1;
    using TransportSystem = Matrix<Scalar_, transport_coefficients, transport_coefficients>;
    using TransportFactor = std::conditional_t<small_transport, PartialPivLU<TransportSystem>, std::tuple<>>;
    fdapde_static_assert(
      Order_ == fdapde::Dynamic || std::int64_t(Order_) * std::int64_t(Order_) <= std::numeric_limits<int>::max(),
      SPD_GEOMETRY_DENSE_WORKSPACE_SIZE_EXCEEDS_SUPPORTED_RANGE);
   public:
    using Scalar = Scalar_;
    using Point = Point_;
    using CachePolicy = typename Point::CachePolicy;
    using Tangent = fdapde::SymmetricMatrix<Scalar, Order_>;
    using MeanCachePolicy = Cache::Union<Cache::Spectral, Cache::Sqrt, Cache::InverseSqrt>;
    fdapde_static_assert(
      (std::same_as<Point, fdapde::SPDMatrix<Scalar, Order_, CachePolicy, Point::StorageOrder>>),
      BURES_WASSERSTEIN_GEOMETRY_REQUIRES_A_NATIVE_SPD_OWNER_WITH_MATCHING_SCALAR_AND_SHAPE);

    /// @brief owns the base and transport residual of a deferred BW geodesic independently of its endpoints
    class Curve {
       public:
        using Scalar = Scalar_;
        static constexpr int Rows = Order_;

        /// @brief snapshots a verified base and its prepared symmetric transport residual
        Curve(SPDMatrix<Scalar, Rows> from, Tangent residual) : from_(std::move(from)), residual_(std::move(residual)) {
            internals::check_spd_geometry_shape(residual_, from_.rows());
        }
        /// @brief returns the matrix order of the stored base
        int rows() const { return from_.rows(); }
        /// @brief borrows persistent curve data and retains the parameter in a deferred matrix expression
        auto operator()(double parameter) const& {
            return fdapde::internals::spd_geodesic_expr<const Curve&>(*this, parameter);
        }
        /// @brief keeps a temporary curve alive inside the deferred matrix expression
        auto operator()(double parameter) && {
            return fdapde::internals::spd_geodesic_expr<Curve>(std::move(*this), parameter);
        }
        /// @brief prevents an expression from borrowing a const temporary curve
        void operator()(double) const&& = delete;
        /// @brief reconstructs one geodesic point while rejecting extrapolation outside the positive lift
        Tangent eval(double parameter) const {
            const Scalar coefficient = internals::geometry_coefficient<Scalar>(parameter);
            Tangent lift(residual_);
            for (int i = 0; i < rows(); ++i)
                for (int j = 0; j <= i; ++j)
                    lift(i, j) = internals::checked_geometry_result(
                      coefficient * static_cast<Scalar>(lift(i, j)) + (i == j ? Scalar(1) : Scalar(0)));
            const SPDMatrix<Scalar, Rows> positive_lift(lift);
            return internals::symmetric_congruence<Scalar, Rows>(positive_lift, from_, rows());
        }
       private:
        SPDMatrix<Scalar, Rows> from_;
        Tangent residual_;
    };

    /// @brief retains base factors and the relative square-root spectrum shared by transport and its implicit
    /// derivatives
    struct RelativeFrame {
        SPDMatrix<Scalar, Order_, Cache::Spectral> from;
        SPDMatrix<Scalar, Order_> to;
        Tangent from_sqrt;
        Tangent from_inverse_sqrt;
        SPDMatrix<Scalar, Order_, Cache::Spectral> relative_sqrt;
        Tangent transport;
        Matrix<Scalar, Order_, Order_> lift_residual;
        Scalar base_sqrt_scale;
        Scalar relative_scale;
        [[no_unique_address]] TransportFactor transport_factor;
        [[no_unique_address]] TransportFactor transport_adjoint_factor;
    };

    /// @brief constructs a geometry with its positive compile-time order
    BuresWassersteinSPDGeometry()
        requires(Order_ != fdapde::Dynamic)
    = default;
    /// @brief validates the runtime order of a dynamic geometry
    explicit BuresWassersteinSPDGeometry(int order)
        requires(Order_ == fdapde::Dynamic)
        : order_(order) {
        internals::validate_spd_geometry_order(order_);
    }

    /// @brief prepares a simplex or mesh interpolant retaining immutable batch bindings
    template <typename Element, typename Nodes>
        requires gfe::P1InterpolationBinding<Element, Nodes>
    auto interpolant(Element&& element, Nodes&& nodes) const;
    /// @brief prepares an interpolant with explicit mean and linear-solve tolerances
    template <typename Element, typename Nodes>
        requires gfe::P1InterpolationBinding<Element, Nodes>
    auto interpolant(Element&& element, Nodes&& nodes, const gfe::P1GeodesicLinearizationOptions& options) const;

    /// @brief returns the uniform matrix order
    int order() const { return order_; }
    /// @brief returns the dimension of the symmetric tangent space
    std::size_t dimension() const { return internals::spd_geometry_dimension(order_); }

    /// @brief pairs ambient tangents through the positive Lyapunov inverse at the point
    template <SPDLike PointPoint>
    double inner_product(const PointPoint& point, const Tangent& u, const Tangent& v) const {
        check_point_(point);
        check_tangent_(u);
        check_tangent_(v);
        return Scalar(0.5) * internals::frobenius_inner(lyapunov_(point, u), v, order_);
    }
    /// @brief returns the metric norm of an ambient tangent
    template <SPDLike PointPoint> double norm(const PointPoint& point, const Tangent& tangent) const {
        return std::sqrt(inner_product(point, tangent, tangent));
    }
    /// @brief copies an already symmetric ambient tangent
    template <SPDLike PointPoint> Tangent project(const PointPoint& point, const Tangent& ambient) const {
        check_point_(point);
        check_tangent_(ambient);
        return ambient;
    }
    /// @brief returns an owning zero tangent of the point order
    template <SPDLike PointPoint> Tangent zero_tangent(const PointPoint& point) const {
        check_point_(point);
        auto result = internals::make_symmetric<Scalar, Order_>(order_);
        for (int i = 0; i < order_; ++i)
            for (int j = 0; j <= i; ++j) result(i, j) = Scalar(0);
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

    /// @brief follows the BW exponential while its horizontal lift remains positive definite
    template <SPDLike PointPoint>
    Point exponential(const PointPoint& point, const Tangent& tangent, double step = 1.0) const {
        check_point_(point);
        check_tangent_(tangent);
        Tangent lift = lyapunov_(point, tangent);
        const Scalar coefficient = internals::geometry_coefficient<Scalar>(step);
        for (int i = 0; i < order_; ++i)
            for (int j = 0; j <= i; ++j)
                lift(i, j) = internals::checked_geometry_result(
                  coefficient * static_cast<Scalar>(lift(i, j)) + (i == j ? Scalar(1) : Scalar(0)));
        const SPDMatrix<Scalar, Order_> positive_lift(lift);
        return Point(internals::symmetric_congruence<Scalar, Order_>(positive_lift, point, order_));
    }
    /// @brief uses the exact BW exponential and marks numerical domain exits as recoverable trial failures
    template <SPDLike PointPoint> Point retract(const PointPoint& point, const Tangent& tangent, double step) const {
        try {
            return exponential(point, tangent, step);
        } catch (const std::domain_error& error) { throw RetractionDomainError(error.what()); }
    }

    /// @brief prepares an immutable relative frame without discarding compatible endpoint caches
    template <SPDLike From, SPDLike To> RelativeFrame relative_frame(const From& from, const To& to) const {
        check_point_(from);
        check_point_(to);
        const SPDMatrix<Scalar, Order_, MeanCachePolicy> base(from);
        const Scalar base_sqrt_scale = std::sqrt(point_scale_(from));
        const Scalar target_sqrt_scale = std::sqrt(point_scale_(to));
        const Scalar relative_scale = base_sqrt_scale * target_sqrt_scale;
        Tangent root(internals::spd_sqrt_factor(base));
        Tangent inverse_root(internals::spd_inv_sqrt_factor(base));
        const SPDMatrix<Scalar, Order_, MeanCachePolicy> target(to);
        Tangent target_root(internals::spd_sqrt_factor(target));
        for (int i = 0; i < order_; ++i)
            for (int j = 0; j <= i; ++j) {
                root(i, j) = static_cast<Scalar>(root(i, j)) / base_sqrt_scale;
                inverse_root(i, j) = static_cast<Scalar>(inverse_root(i, j)) * base_sqrt_scale;
                target_root(i, j) = static_cast<Scalar>(target_root(i, j)) / target_sqrt_scale;
            }
        // factor the square-root product without squaring its condition number
        Matrix<Scalar, Order_, Order_> product(root * target_root);
        Matrix<Scalar, Order_, Order_> right;
        if constexpr (Order_ == fdapde::Dynamic) right.resize(order_, order_);
        for (int i = 0; i < order_; ++i)
            for (int j = 0; j < order_; ++j) right(i, j) = i == j ? Scalar(1) : Scalar(0);
        orthogonalize_columns_(product, right);
        auto relative_root = internals::make_symmetric<Scalar, Order_>(order_);
        Matrix<Scalar, Order_, Order_> alignment;
        if constexpr (Order_ == fdapde::Dynamic) alignment.resize(order_, order_);
        std::vector<Scalar> singular_values(order_);
        for (int j = 0; j < order_; ++j) {
            Scalar norm = 0;
            for (int i = 0; i < order_; ++i) norm = std::hypot(norm, product(i, j));
            singular_values[j] = norm;
            for (int i = 0; i < order_; ++i) product(i, j) /= norm;
        }
        for (int i = 0; i < order_; ++i)
            for (int j = 0; j < order_; ++j) {
                Scalar value = 0;
                for (int k = 0; k < order_; ++k) value += right(i, k) * product(j, k);
                alignment(i, j) = value;
            }
        const Matrix<Scalar, Order_, Order_> aligned_root(target_root * alignment);
        const Matrix<Scalar, Order_, Order_> map(aligned_root * inverse_root);
        auto transport = internals::make_symmetric<Scalar, Order_>(order_);
        for (int i = 0; i < order_; ++i)
            for (int j = 0; j <= i; ++j) {
                Scalar value = 0;
                for (int k = 0; k < order_; ++k) value += product(i, k) * singular_values[k] * product(j, k);
                relative_root(i, j) = value;
                transport(i, j) = Scalar(.5) * (map(i, j) + map(j, i));
            }
        const decltype(RelativeFrame::relative_sqrt) relative(relative_root);
        const Scalar transport_scale = target_sqrt_scale / base_sqrt_scale;
        Matrix<Scalar, Order_, Order_> lift_residual(aligned_root);
        for (int i = 0; i < order_; ++i)
            for (int j = 0; j < order_; ++j)
                lift_residual(i, j) =
                  internals::checked_geometry_result(transport_scale * aligned_root(i, j) - root(i, j));
        for (int i = 0; i < order_; ++i)
            for (int j = 0; j <= i; ++j)
                transport(i, j) =
                  internals::checked_geometry_result(static_cast<Scalar>(transport(i, j)) * transport_scale);
        RelativeFrame result {
          decltype(RelativeFrame::from)(base),
          decltype(RelativeFrame::to)(to),
          root,
          inverse_root,
          relative,
          transport,
          lift_residual,
          base_sqrt_scale,
          relative_scale,
          {},
          {}};
        if constexpr (small_transport) {
            const Matrix<Scalar, Order_, Order_> coefficient(transport * from);
            bool representable = true;
            for (int row = 0; row < order_; ++row)
                for (int col = 0; col < order_; ++col)
                    representable = representable && std::isfinite(coefficient(row, col)) &&
                                    std::abs(coefficient(row, col)) <= std::numeric_limits<Scalar>::max() / Scalar(2);
            // optional factors must not narrow the endpoint range when a Sylvester coefficient sum would overflow
            if (representable) {
                result.transport_factor.compute(transport_system_(coefficient));
                const Matrix<Scalar, Order_, Order_> adjoint(coefficient.transpose());
                result.transport_adjoint_factor.compute(transport_system_(adjoint));
            }
        }
        return result;
    }
    /// @brief returns the initial tangent of the shortest BW geodesic to the target
    template <SPDLike From, SPDLike To> Tangent logarithm(const From& from, const To& to) const {
        return logarithm(relative_frame(from, to));
    }
    /// @brief reconstructs the logarithm directly from the aligned lift residual
    Tangent logarithm(const RelativeFrame& frame) const {
        auto result = internals::make_symmetric<Scalar, Order_>(order_);
        for (int i = 0; i < order_; ++i)
            for (int j = 0; j <= i; ++j) {
                Scalar value = 0;
                for (int k = 0; k < order_; ++k)
                    value += frame.lift_residual(i, k) * frame.from_sqrt(j, k) +
                             frame.from_sqrt(i, k) * frame.lift_residual(j, k);
                result(i, j) =
                  internals::checked_geometry_result((value * frame.base_sqrt_scale) * frame.base_sqrt_scale);
            }
        return result;
    }
    /// @brief snapshots the shortest BW geodesic as a native deferred curve
    template <SPDLike From, SPDLike To> Curve geodesic(const From& from, const To& to) const {
        const auto frame = relative_frame(from, to);
        return Curve(SPDMatrix<Scalar, Order_>(frame.from), transport_residual_(frame));
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
    /// @brief returns the BW distance between two verified endpoints
    template <SPDLike From, SPDLike To> double distance(const From& from, const To& to) const {
        return distance(relative_frame(from, to));
    }
    /// @brief evaluates the distance as a lift norm to avoid cancellation of nearly equal traces
    double distance(const RelativeFrame& frame) const {
        double result = 0;
        for (int i = 0; i < order_; ++i)
            for (int j = 0; j < order_; ++j)
                result = std::hypot(result, static_cast<double>(frame.lift_residual(i, j) * frame.base_sqrt_scale));
        return internals::checked_geometry_result(result);
    }

    /// @brief maps an ambient Frobenius gradient to its metric-dual tangent
    template <SPDLike PointPoint>
    Tangent euclidean_to_riemannian_gradient(const PointPoint& point, const Tangent& gradient) const {
        check_point_(point);
        check_tangent_(gradient);
        const auto result = jordan_sum_(point, gradient);
        return internals::combine_symmetric<Scalar, Order_>(result, Scalar(2), result, Scalar(0), order_);
    }
    /// @brief converts an ambient Hessian action through the BW metric and its Levi-Civita connection
    template <SPDLike P>
    Tangent euclidean_to_riemannian_hessian(
      const P& point, const Tangent& gradient, const Tangent& hessian, const Tangent& direction) const {
        check_point_(point);
        check_tangent_(gradient);
        check_tangent_(hessian);
        check_tangent_(direction);
        const auto ordinary = jordan_sum_(direction, gradient);
        const auto metric_action = jordan_sum_(point, hessian);
        const auto connection = symmetric_triple_sum_(lyapunov_(point, direction), point, gradient);
        const auto derivative =
          internals::combine_symmetric<Scalar, Order_>(ordinary, Scalar(2), metric_action, Scalar(2), order_);
        return internals::combine_symmetric<Scalar, Order_>(derivative, Scalar(1), connection, Scalar(-2), order_);
    }
    /// @brief maps a metric-dual tangent to the ambient Frobenius covector
    template <SPDLike PointPoint>
    Tangent riemannian_to_euclidean_gradient(const PointPoint& point, const Tangent& gradient) const {
        check_point_(point);
        check_tangent_(gradient);
        const auto result = lyapunov_(point, gradient);
        return internals::combine_symmetric<Scalar, Order_>(result, Scalar(0.5), result, Scalar(0), order_);
    }

    /// @brief applies the exact target differential of the BW logarithm
    template <SPDLike From, SPDLike To>
    Tangent logarithm_target_jvp(const From& from, const To& to, const Tangent& direction) const {
        return logarithm_target_jvp(relative_frame(from, to), direction);
    }
    /// @brief differentiates the implicit transport equation using its retained relative spectrum
    Tangent logarithm_target_jvp(const RelativeFrame& frame, const Tangent& direction) const {
        check_tangent_(direction);
        return jordan_sum_(frame.from, transport_solve_(frame, direction));
    }
    /// @brief returns the target differential adjoint in the endpoint BW metrics
    template <SPDLike From, SPDLike To>
    Tangent logarithm_target_vjp(const From& from, const To& to, const Tangent& metric_dual) const {
        return logarithm_target_vjp(relative_frame(from, to), metric_dual);
    }
    /// @brief reverses the transport differential through the self-adjoint relative Lyapunov solve
    Tangent logarithm_target_vjp(const RelativeFrame& frame, const Tangent& metric_dual) const {
        check_tangent_(metric_dual);
        const auto rhs = internals::symmetric_congruence<Scalar, Order_>(frame.from_inverse_sqrt, metric_dual, order_);
        const auto relative_dual = relative_lyapunov_(frame, rhs);
        const auto dual = internals::symmetric_congruence<Scalar, Order_>(frame.from_sqrt, relative_dual, order_);
        return jordan_sum_(frame.to, dual);
    }

    /// @brief applies the covariant base Hessian of one half the squared BW distance
    template <SPDLike From, SPDLike To>
    Tangent half_squared_distance_hessian_vector(const From& from, const To& to, const Tangent& direction) const {
        return half_squared_distance_hessian_vector(relative_frame(from, to), direction);
    }
    /// @brief differentiates transport analytically and includes the BW Levi-Civita connection
    Tangent half_squared_distance_hessian_vector(const RelativeFrame& frame, const Tangent& direction) const {
        check_tangent_(direction);
        const auto residual = transport_residual_(frame);
        const auto rhs = internals::symmetric_congruence<Scalar, Order_>(frame.transport, direction, order_);
        const auto transport_action = transport_solve_(frame, rhs);
        const auto base_action = jordan_sum_(direction, residual);
        const auto transported_action = jordan_sum_(frame.from, transport_action);
        const auto connection = symmetric_triple_sum_(lyapunov_(frame.from, direction), frame.from, residual);
        const auto ordinary =
          internals::combine_symmetric<Scalar, Order_>(transported_action, Scalar(1), base_action, Scalar(-1), order_);
        return internals::combine_symmetric<Scalar, Order_>(ordinary, Scalar(1), connection, Scalar(1), order_);
    }

    /// @brief differentiates the BW distance Hessian along both endpoints with a parallel action tangent
    Tangent half_squared_distance_hessian_covariant_jvp(
      const RelativeFrame& frame, const Tangent& base_direction, const Tangent& target_direction,
      const Tangent& action_direction) const {
        check_tangent_(base_direction);
        check_tangent_(target_direction);
        check_tangent_(action_direction);
        const auto base_lift = lyapunov_(frame.from, base_direction);
        const auto action_lift = lyapunov_(frame.from, action_direction);
        const auto action_change = symmetric_triple_sum_(base_lift, frame.from, action_lift);
        const auto base_transport =
          internals::symmetric_congruence<Scalar, Order_>(frame.transport, base_direction, order_);
        const auto transport_rhs =
          internals::combine_symmetric<Scalar, Order_>(target_direction, Scalar(1), base_transport, Scalar(-1), order_);
        const auto transport_change = transport_solve_(frame, transport_rhs);
        const auto residual = transport_residual_(frame);
        const auto transported_action =
          internals::symmetric_congruence<Scalar, Order_>(frame.transport, action_direction, order_);
        const auto transport_action = transport_solve_(frame, transported_action);

        // differentiate the retained transport equation and the horizontal action lift
        Tangent action_rhs = symmetric_triple_sum_(transport_change, action_direction, frame.transport);
        const auto moved_action =
          internals::symmetric_congruence<Scalar, Order_>(frame.transport, action_change, order_);
        action_rhs =
          internals::combine_symmetric<Scalar, Order_>(action_rhs, Scalar(1), moved_action, Scalar(1), order_);
        const auto transport_coefficient = symmetric_triple_sum_(transport_change, frame.from, transport_action);
        action_rhs = internals::combine_symmetric<Scalar, Order_>(
          action_rhs, Scalar(1), transport_coefficient, Scalar(-1), order_);
        const auto base_coefficient = symmetric_triple_sum_(frame.transport, base_direction, transport_action);
        action_rhs =
          internals::combine_symmetric<Scalar, Order_>(action_rhs, Scalar(1), base_coefficient, Scalar(-1), order_);
        const auto transported_change = transport_solve_(frame, action_rhs);
        const auto lift_coefficient = jordan_sum_(base_direction, action_lift);
        const auto lift_rhs =
          internals::combine_symmetric<Scalar, Order_>(action_change, Scalar(1), lift_coefficient, Scalar(-1), order_);
        const auto lift_change = lyapunov_(frame.from, lift_rhs);

        Tangent derivative = jordan_sum_(base_direction, transport_action);
        auto accumulate = [&](const Tangent& term, Scalar scale) {
            derivative = internals::combine_symmetric<Scalar, Order_>(derivative, Scalar(1), term, scale, order_);
        };
        accumulate(jordan_sum_(frame.from, transported_change), Scalar(1));
        accumulate(jordan_sum_(action_change, residual), Scalar(-1));
        accumulate(jordan_sum_(action_direction, transport_change), Scalar(-1));
        accumulate(symmetric_triple_sum_(lift_change, frame.from, residual), Scalar(1));
        accumulate(symmetric_triple_sum_(action_lift, base_direction, residual), Scalar(1));
        accumulate(symmetric_triple_sum_(action_lift, frame.from, transport_change), Scalar(1));
        const auto hessian = half_squared_distance_hessian_vector(frame, action_direction);
        accumulate(symmetric_triple_sum_(base_lift, frame.from, lyapunov_(frame.from, hessian)), Scalar(-1));
        return derivative;
    }

    /// @brief differentiates the BW distance Hessian after retaining the endpoint transport frame
    template <SPDLike From, SPDLike To>
    Tangent half_squared_distance_hessian_covariant_jvp(
      const From& from, const To& to, const Tangent& base_direction, const Tangent& target_direction,
      const Tangent& action_direction) const {
        return half_squared_distance_hessian_covariant_jvp(
          relative_frame(from, to), base_direction, target_direction, action_direction);
    }

    /// @brief reverses target variation of the distance Hessian through two retained transport solves
    Tangent half_squared_distance_hessian_covariant_target_vjp(
      const RelativeFrame& frame, const Tangent& action_direction, const Tangent& output_metric_dual) const {
        check_tangent_(action_direction);
        check_tangent_(output_metric_dual);
        const auto output = riemannian_to_euclidean_gradient(frame.from, output_metric_dual);
        const auto lift = lyapunov_(frame.from, action_direction);
        const auto transported =
          internals::symmetric_congruence<Scalar, Order_>(frame.transport, action_direction, order_);
        const auto transport_action = transport_solve_(frame, transported);
        const auto action_dual = transport_adjoint_solve_(frame, jordan_sum_(frame.from, output));
        const Matrix<Scalar, Order_, Order_> first(action_direction * frame.transport);
        const Matrix<Scalar, Order_, Order_> second(frame.from * transport_action);
        const Matrix<Scalar, Order_, Order_> coefficient(first - second);
        const Matrix<Scalar, Order_, Order_> product(coefficient * action_dual);
        Tangent transport_dual = zero_tangent(frame.from);
        for (int row = 0; row < order_; ++row)
            for (int col = 0; col <= row; ++col)
                transport_dual(row, col) = internals::checked_geometry_result(product(row, col) + product(col, row));
        const auto ordinary = jordan_sum_(action_direction, output);
        transport_dual =
          internals::combine_symmetric<Scalar, Order_>(transport_dual, Scalar(1), ordinary, Scalar(-1), order_);
        const auto connection = symmetric_triple_sum_(output, lift, frame.from);
        transport_dual =
          internals::combine_symmetric<Scalar, Order_>(transport_dual, Scalar(1), connection, Scalar(1), order_);
        return euclidean_to_riemannian_gradient(frame.to, transport_adjoint_solve_(frame, transport_dual));
    }

    /// @brief returns both metric adjoints of the exact BW Hessian variation
    std::pair<Tangent, Tangent> half_squared_distance_hessian_covariant_vjp(
      const RelativeFrame& frame, const Tangent& action_direction, const Tangent& output_metric_dual) const {
        check_tangent_(action_direction);
        check_tangent_(output_metric_dual);
        Tangent base_covector = zero_tangent(frame.from);
        const Tangent zero = zero_tangent(frame.from);
        for (int row = 0; row < order_; ++row)
            for (int col = 0; col <= row; ++col) {
                Tangent basis = zero;
                basis(row, col) = Scalar(1);
                const auto base_action =
                  half_squared_distance_hessian_covariant_jvp(frame, basis, zero, action_direction);
                const Scalar multiplicity = row == col ? Scalar(1) : Scalar(2);
                base_covector(row, col) = inner_product(frame.from, output_metric_dual, base_action) / multiplicity;
            }
        return {
          euclidean_to_riemannian_gradient(frame.from, base_covector),
          half_squared_distance_hessian_covariant_target_vjp(frame, action_direction, output_metric_dual)};
    }

    /// @brief returns BW distance third-derivative adjoints after retaining the endpoint transport frame
    template <SPDLike From, SPDLike To>
    std::pair<Tangent, Tangent> half_squared_distance_hessian_covariant_vjp(
      const From& from, const To& to, const Tangent& action_direction, const Tangent& output_metric_dual) const {
        return half_squared_distance_hessian_covariant_vjp(
          relative_frame(from, to), action_direction, output_metric_dual);
    }
   private:
    /// @brief obtains the singular vectors by one-sided Jacobi rotations without forming normal equations
    void orthogonalize_columns_(Matrix<Scalar, Order_, Order_>& left, Matrix<Scalar, Order_, Order_>& right) const {
        const Scalar tolerance = Scalar(8) * std::numeric_limits<Scalar>::epsilon();
        for (int sweep = 0; sweep < 64; ++sweep) {
            bool converged = true;
            for (int p = 0; p < order_; ++p)
                for (int q = p + 1; q < order_; ++q) {
                    Scalar alpha = 0, beta = 0, gamma = 0;
                    for (int i = 0; i < order_; ++i) {
                        alpha += left(i, p) * left(i, p);
                        beta += left(i, q) * left(i, q);
                        gamma += left(i, p) * left(i, q);
                    }
                    if (std::abs(gamma) <= tolerance * std::sqrt(alpha) * std::sqrt(beta)) continue;
                    converged = false;
                    const Scalar zeta = (beta - alpha) / (Scalar(2) * gamma);
                    const Scalar t = std::copysign(Scalar(1), zeta) / (std::abs(zeta) + std::hypot(Scalar(1), zeta));
                    const Scalar cosine = Scalar(1) / std::hypot(Scalar(1), t), sine = cosine * t;
                    for (int i = 0; i < order_; ++i) {
                        const Scalar lp = left(i, p), lq = left(i, q), rp = right(i, p), rq = right(i, q);
                        left(i, p) = cosine * lp - sine * lq;
                        left(i, q) = sine * lp + cosine * lq;
                        right(i, p) = cosine * rp - sine * rq;
                        right(i, q) = sine * rp + cosine * rq;
                    }
                }
            if (converged) return;
        }
        fdapde_strong_assert(false, std::domain_error, "BW square-root product factorization did not converge");
    }
    /// @brief solves the positive Lyapunov equation through compatible cached native eigenpairs
    template <SPDLike Base> Tangent lyapunov_(const Base& base, const Tangent& rhs) const {
        return fdapde::internals::with_spd_spectral(base, [&](const auto& spectral) {
            return fdapde::internals::frechet_symmetric(spectral, order_, rhs, [](auto x, auto y) {
                return decltype(x)(0.5) / (decltype(x)(0.5) * x + decltype(x)(0.5) * y);
            });
        });
    }
    /// @brief solves at the relative square root using its retained eigenpairs
    Tangent relative_lyapunov_(const RelativeFrame& frame, const Tangent& rhs) const {
        Tangent scaled_rhs(rhs);
        for (int i = 0; i < order_; ++i)
            for (int j = 0; j <= i; ++j)
                scaled_rhs(i, j) = static_cast<Scalar>(scaled_rhs(i, j)) / frame.relative_scale;
        return fdapde::internals::frechet_symmetric(
          frame.relative_sqrt.cache(), order_, scaled_rhs, [](auto x, auto y) { return decltype(x)(1) / (x + y); });
    }
    /// @brief solves dT A T + T A dT = rhs without differentiating either base square-root factor
    Tangent transport_solve_(const RelativeFrame& frame, const Tangent& rhs) const {
        if constexpr (small_transport) {
            if (frame.transport_factor.info() == 0) return small_transport_solve_(frame.transport_factor, rhs);
        }
        const auto relative_rhs = internals::symmetric_congruence<Scalar, Order_>(frame.from_sqrt, rhs, order_);
        const auto action = relative_lyapunov_(frame, relative_rhs);
        return internals::symmetric_congruence<Scalar, Order_>(frame.from_inverse_sqrt, action, order_);
    }
    /// @brief reverses the transport differential through the transposed congruence factors
    Tangent transport_adjoint_solve_(const RelativeFrame& frame, const Tangent& rhs) const {
        if constexpr (small_transport) {
            if (frame.transport_adjoint_factor.info() == 0)
                return small_transport_solve_(frame.transport_adjoint_factor, rhs);
        }
        const auto relative_rhs = internals::symmetric_congruence<Scalar, Order_>(frame.from_inverse_sqrt, rhs, order_);
        const auto action = relative_lyapunov_(frame, relative_rhs);
        return internals::symmetric_congruence<Scalar, Order_>(frame.from_sqrt, action, order_);
    }
    /// @brief assembles a small general Sylvester operator without inverse square-root congruences
    TransportSystem transport_system_(const Matrix<Scalar, Order_, Order_>& coefficient) const
        requires(small_transport)
    {
        TransportSystem result;
        for (int first = 0, component = 0; first < order_; ++first)
            for (int second = first; second < order_; ++second, ++component) {
                auto basis = zero_tangent_basis_(first, second);
                const Matrix<Scalar, Order_, Order_> image(coefficient * basis + basis * coefficient.transpose());
                for (int row = 0, output = 0; row < order_; ++row)
                    for (int col = row; col < order_; ++col, ++output) result(output, component) = image(row, col);
            }
        return result;
    }
    /// @brief creates one raw symmetric coordinate direction for the small transport equation
    Tangent zero_tangent_basis_(int first, int second) const
        requires(small_transport)
    {
        Tangent result;
        for (int row = 0; row < order_; ++row)
            for (int col = row; col < order_; ++col) result(row, col) = Scalar(0);
        result(first, second) = Scalar(1);
        return result;
    }
    /// @brief applies a retained packed Sylvester factor to one symmetric right-hand side
    Tangent small_transport_solve_(const TransportFactor& factor, const Tangent& rhs) const
        requires(small_transport)
    {
        Vector<Scalar, transport_coefficients> coordinates;
        for (int row = 0, component = 0; row < order_; ++row)
            for (int col = row; col < order_; ++col, ++component) coordinates[component] = rhs(row, col);
        const Vector<Scalar, transport_coefficients> solution(factor.solve(coordinates));
        Tangent result;
        for (int row = 0, component = 0; row < order_; ++row)
            for (int col = row; col < order_; ++col, ++component)
                result(row, col) = internals::checked_geometry_result(solution[component]);
        return result;
    }
    /// @brief subtracts the identity from the retained optimal transport map
    Tangent transport_residual_(const RelativeFrame& frame) const {
        Tangent result(frame.transport);
        for (int i = 0; i < order_; ++i) result(i, i) = static_cast<Scalar>(result(i, i)) - Scalar(1);
        return result;
    }
    /// @brief forms A B + B A in packed symmetric storage
    template <typename A, typename B> Tangent jordan_sum_(const A& lhs, const B& rhs) const {
        auto result = internals::make_symmetric<Scalar, Order_>(order_);
        for (int i = 0; i < order_; ++i)
            for (int j = 0; j <= i; ++j) {
                Scalar value = 0;
                for (int k = 0; k < order_; ++k) value += lhs(i, k) * rhs(k, j) + rhs(i, k) * lhs(k, j);
                result(i, j) = internals::checked_geometry_result(value);
            }
        return result;
    }
    /// @brief forms X A Y + Y A X with one dense native product workspace
    template <typename X, typename A, typename Y>
    Tangent symmetric_triple_sum_(const X& lhs, const A& middle, const Y& rhs) const {
        const Matrix<Scalar, Order_, Order_> product(lhs * middle);
        auto result = internals::make_symmetric<Scalar, Order_>(order_);
        for (int i = 0; i < order_; ++i)
            for (int j = 0; j <= i; ++j) {
                Scalar value = 0;
                for (int k = 0; k < order_; ++k) value += product(i, k) * rhs(k, j) + rhs(i, k) * product(j, k);
                result(i, j) = internals::checked_geometry_result(value);
            }
        return result;
    }
    /// @brief selects a positive diagonal scale before forming the relative product
    template <SPDLike P> Scalar point_scale_(const P& point) const {
        Scalar result = 0;
        for (int i = 0; i < order_; ++i) result = std::max(result, static_cast<Scalar>(point(i, i)));
        return result;
    }
    /// @brief validates an endpoint against this geometry's order and finite-coordinate contract
    template <SPDLike P> void check_point_(const P& point) const { internals::check_spd_geometry_shape(point, order_); }
    /// @brief validates an ambient tangent against this geometry's order and finite-coordinate contract
    void check_tangent_(const Tangent& tangent) const { internals::check_spd_geometry_shape(tangent, order_); }

    int order_ = Order_ == fdapde::Dynamic ? 0 : Order_;
};

namespace internals {

/// @brief selects the Bures-Wasserstein metric while preserving the exact native SPD owner
template <typename Point> struct bures_wasserstein_geometry_type {
    using type = BuresWassersteinSPDGeometry<typename Point::Scalar, Point::Rows, Usage::None, Point>;
};

}   // namespace internals

/// @brief supplies the Bures-Wasserstein metric for a native SPD owner with its cache policy
template <typename Point>
using BuresWassersteinGeometry = typename internals::bures_wasserstein_geometry_type<Point>::type;

}   // namespace manifold
}   // namespace fdapde

#endif   // __FDAPDE_MANIFOLD_BURES_WASSERSTEIN_SPD_H__
