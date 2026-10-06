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

#ifndef __FDAPDE_MANIFOLD_LOG_CHOLESKY_SPD_H__
#define __FDAPDE_MANIFOLD_LOG_CHOLESKY_SPD_H__

#include "../../geometry_expr.h"
#include "../../header_check.h"
#include "../spd_geometry_common.h"

namespace fdapde {
namespace manifold {

/// @brief defines Lin's flat log-Cholesky metric with ambient symmetric tangent coordinates
/// @details the symmetric chart stores log(L_ii) on the diagonal and L_ij / sqrt(2) off the diagonal,
/// so its full Frobenius product counts each strictly lower Cholesky coefficient once
/// @see https://doi.org/10.1137/18M1221084
template <typename Scalar_, int Order_, Usage Uses_ = Usage::None> class LogCholeskySPDGeometry {
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
    using CachePolicy = Cache::Policy<
      (Uses_ != Usage::None ? Cache::Cholesky::Flags | Cache::LogCholesky::Flags : 0u) |
      (internals::has_spd_usage(Uses_, Usage::LogExpDifferentials) ?
         Cache::Spectral::Flags | Cache::LogDividedDifferences::Flags :
         0u)>;
    using Point = SPDMatrix<Scalar, Order_, Order_, CachePolicy>;
    using Tangent = SymmetricMatrix<Scalar, Order_, Order_>;
    using Factor = Matrix<Scalar, Order_, Order_>;

    /// @brief owns a prepared Cholesky factor and its isometric symmetric chart
    struct ChartFrame {
        Factor factor;
        Tangent coordinates;
    };
    /// @brief owns endpoint factors shared by exact logarithm and distance derivatives
    struct RelativeFrame {
        ChartFrame from;
        ChartFrame to;
    };
    /// @brief owns chart endpoints independently of the geometry and point lifetimes
    class Curve {
       public:
        using Scalar = Scalar_;
        static constexpr int Rows = Order_;
        /// @brief validates and retains a chart origin and the constant chart velocity
        Curve(Tangent origin, Tangent difference) : origin_(std::move(origin)), difference_(std::move(difference)) {
            internals::validate_spd_geometry_order(origin_.rows());
            internals::check_spd_geometry_shape(origin_, origin_.rows());
            internals::check_spd_geometry_shape(difference_, origin_.rows());
        }
        /// @brief returns the stored matrix order
        int rows() const { return origin_.rows(); }
        /// @brief borrows a persistent curve in a deferred geometric expression
        auto operator()(double parameter) const& {
            return fdapde::internals::spd_geodesic_expr<const Curve&>(*this, parameter);
        }
        /// @brief retains a temporary curve inside the deferred geometric expression
        auto operator()(double parameter) && {
            return fdapde::internals::spd_geodesic_expr<Curve>(std::move(*this), parameter);
        }
        /// @brief prevents an expression from borrowing a const temporary
        void operator()(double) const&& = delete;
        /// @brief reconstructs the linear chart path as finite symmetric coefficients
        Tangent eval(double parameter) const {
            const auto geometry = [&] {
                if constexpr (Order_ == Dynamic)
                    return LogCholeskySPDGeometry(rows());
                else
                    return LogCholeskySPDGeometry();
            }();
            const auto value = internals::combine_symmetric<Scalar, Order_>(
              origin_, Scalar(1), difference_, internals::geometry_coefficient<Scalar>(parameter), rows());
            return geometry.factor_product_(geometry.factor_from_chart_(value));
        }
       private:
        Tangent origin_;
        Tangent difference_;
    };

    /// @brief constructs a fixed-order geometry
    LogCholeskySPDGeometry()
        requires(Order_ != Dynamic)
    = default;
    /// @brief validates the runtime matrix order
    explicit LogCholeskySPDGeometry(int order)
        requires(Order_ == Dynamic)
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
    /// @brief returns the matrix order
    int order() const { return order_; }
    /// @brief returns the independent tangent dimension
    std::size_t dimension() const { return internals::spd_geometry_dimension(order_); }

    /// @brief prepares native triangular factors once for repeated analytic chart operations
    template <SPDLike P> ChartFrame chart_frame(const P& point) const {
        check_point_(point);
        auto factor = make_factor_();
        if constexpr (fdapde::internals::spd_cache_has_v<typename P::CachePolicy, Cache::Cholesky>) {
            const auto retained = point.cache().cholesky();
            for (int i = 0; i < order_; ++i)
                for (int j = 0; j < order_; ++j) factor(i, j) = retained(i, j);
        } else {
            // positive pivots use extended accumulation before narrowing the verified factor
            for (int i = 0; i < order_; ++i)
                for (int j = 0; j <= i; ++j) {
                    long double value = static_cast<Scalar>(point(i, j));
                    for (int k = 0; k < j; ++k) value -= static_cast<long double>(factor(i, k)) * factor(j, k);
                    if (i == j) {
                        fdapde_strong_assert(
                          value > 0 && std::isfinite(value), std::domain_error,
                          "log-Cholesky: Cholesky pivot must be positive and finite");
                        factor(i, j) = internals::checked_geometry_result(static_cast<Scalar>(std::sqrt(value)));
                    } else
                        factor(i, j) = internals::checked_geometry_result(static_cast<Scalar>(value / factor(j, j)));
                }
        }
        auto coordinates = internals::make_symmetric<Scalar, Order_>(order_);
        if constexpr (fdapde::internals::spd_cache_has_v<typename P::CachePolicy, Cache::LogCholesky>) {
            coordinates = point.cache().template matrix<Cache::LogCholesky>();
        } else {
            const Scalar inverse_root_two = Scalar(1) / std::sqrt(Scalar(2));
            for (int i = 0; i < order_; ++i)
                for (int j = 0; j <= i; ++j)
                    coordinates(i, j) = i == j ? std::log(factor(i, i)) : factor(i, j) * inverse_root_two;
        }
        return {std::move(factor), std::move(coordinates)};
    }
    /// @brief maps a verified SPD point to its global isometric coordinates
    template <SPDLike P> Tangent chart(const P& point) const {
        if constexpr (fdapde::internals::spd_cache_has_v<typename P::CachePolicy, Cache::LogCholesky>) {
            check_point_(point);
            return Tangent(point.cache().template matrix<Cache::LogCholesky>());
        } else
            return chart_frame(point).coordinates;
    }
    /// @brief reconstructs a verified SPD point from finite global chart coordinates
    Point from_chart(const Tangent& coordinates) const {
        check_tangent_(coordinates);
        return Point(factor_product_(factor_from_chart_(coordinates)));
    }
    /// @brief reconstructs a chart point under the inverse-map naming used by flat finite elements
    Point inverse_chart(const Tangent& coordinates) const { return from_chart(coordinates); }

    /// @brief pushes an ambient tangent into the flat chart through triangular solves
    template <SPDLike P> Tangent chart_differential(const P& point, const Tangent& direction) const {
        return chart_differential(chart_frame(point), direction);
    }
    /// @brief reuses a prepared Cholesky factor for the exact chart differential
    Tangent chart_differential(const ChartFrame& frame, const Tangent& direction) const {
        check_tangent_(direction);
        const auto relative = inverse_congruence_(frame.factor, direction, false);
        auto result = internals::make_symmetric<Scalar, Order_>(order_);
        const Scalar inverse_root_two = Scalar(1) / std::sqrt(Scalar(2));
        for (int i = 0; i < order_; ++i)
            for (int j = 0; j <= i; ++j) {
                Scalar value = 0;
                for (int k = j; k <= i; ++k)
                    value += frame.factor(i, k) * (k == j ? Scalar(0.5) : Scalar(1)) * relative(k, j);
                result(i, j) =
                  internals::checked_geometry_result(i == j ? Scalar(0.5) * relative(i, i) : value * inverse_root_two);
            }
        return result;
    }
    /// @brief applies the ambient-to-chart Jacobian used by flat finite elements
    template <typename P> Tangent chart_jvp(const P& point, const Tangent& direction) const {
        return chart_differential(point, direction);
    }
    /// @brief pulls a chart tangent back to ambient symmetric coordinates
    Tangent inverse_chart_differential(const Tangent& coordinates, const Tangent& direction) const {
        check_tangent_(coordinates);
        check_tangent_(direction);
        return inverse_differential_(factor_from_chart_(coordinates), direction);
    }
    /// @brief applies the inverse-chart Jacobian used by flat finite elements
    Tangent inverse_chart_jvp(const Tangent& coordinates, const Tangent& direction) const {
        return inverse_chart_differential(coordinates, direction);
    }
    /// @brief applies the inverse-chart Jacobian while reusing a prepared factor
    Tangent inverse_chart_jvp(const ChartFrame& frame, const Tangent& direction) const {
        check_tangent_(direction);
        return inverse_differential_(frame.factor, direction);
    }
    /// @brief applies the analytic symmetric second differential of the inverse chart
    Tangent
    inverse_chart_second_differential(const Tangent& coordinates, const Tangent& first, const Tangent& second) const {
        check_tangent_(coordinates);
        check_tangent_(first);
        check_tangent_(second);
        return inverse_second_differential_(factor_from_chart_(coordinates), first, second);
    }
    /// @brief applies the inverse-chart second derivative with an already prepared Cholesky factor
    Tangent
    inverse_chart_second_differential(const ChartFrame& frame, const Tangent& first, const Tangent& second) const {
        check_tangent_(first);
        check_tangent_(second);
        return inverse_second_differential_(frame.factor, first, second);
    }
    /// @brief applies the inverse-chart second derivative used by flat finite elements
    Tangent inverse_chart_second_jvp(const Tangent& coordinates, const Tangent& first, const Tangent& second) const {
        return inverse_chart_second_differential(coordinates, first, second);
    }
    /// @brief reverses a chart differential in the full Frobenius pairings
    template <SPDLike P> Tangent chart_differential_adjoint(const P& point, const Tangent& cotangent) const {
        return chart_differential_adjoint(chart_frame(point), cotangent);
    }
    /// @brief reverses the triangular chart differential using retained endpoint factors
    Tangent chart_differential_adjoint(const ChartFrame& frame, const Tangent& cotangent) const {
        check_tangent_(cotangent);
        auto lower_dual = make_factor_();
        const Scalar root_two = std::sqrt(Scalar(2));
        for (int i = 0; i < order_; ++i)
            for (int j = 0; j <= i; ++j)
                lower_dual(i, j) = i == j ? cotangent(i, i) / frame.factor(i, i) : root_two * cotangent(i, j);
        auto relative_dual = internals::make_symmetric<Scalar, Order_>(order_);
        for (int i = 0; i < order_; ++i)
            for (int j = 0; j <= i; ++j) {
                Scalar value = 0;
                for (int k = i; k < order_; ++k) value += frame.factor(k, i) * lower_dual(k, j);
                relative_dual(i, j) = Scalar(0.5) * value;
            }
        return inverse_congruence_(frame.factor, relative_dual, true);
    }
    /// @brief applies the ambient-to-chart Frobenius adjoint used by flat finite elements
    template <typename P> Tangent chart_vjp(const P& point, const Tangent& cotangent) const {
        return chart_differential_adjoint(point, cotangent);
    }
    /// @brief reverses an inverse-chart differential in the full Frobenius pairings
    Tangent inverse_chart_differential_adjoint(const Tangent& coordinates, const Tangent& cotangent) const {
        check_tangent_(coordinates);
        check_tangent_(cotangent);
        return inverse_adjoint_(factor_from_chart_(coordinates), cotangent);
    }

    /// @brief pairs ambient tangents through their flat chart coordinates
    template <SPDLike P> double inner_product(const P& point, const Tangent& first, const Tangent& second) const {
        const auto frame = chart_frame(point);
        return internals::frobenius_inner(chart_differential(frame, first), chart_differential(frame, second), order_);
    }
    /// @brief returns the metric norm of an ambient tangent
    template <SPDLike P> double norm(const P& point, const Tangent& tangent) const {
        return internals::frobenius_norm(chart_differential(point, tangent), order_);
    }
    /// @brief copies an already symmetric ambient tangent
    template <SPDLike P> Tangent project(const P& point, const Tangent& ambient) const {
        check_point_(point);
        check_tangent_(ambient);
        return ambient;
    }
    /// @brief creates a zero ambient tangent of the geometry order
    template <SPDLike P> Tangent zero_tangent(const P& point) const {
        check_point_(point);
        auto result = internals::make_symmetric<Scalar, Order_>(order_);
        for (int i = 0; i < order_; ++i)
            for (int j = 0; j <= i; ++j) result(i, j) = Scalar(0);
        return result;
    }
    /// @brief combines ambient tangents with finite coefficients
    template <SPDLike P>
    Tangent
    linear_combination(const P& point, double alpha, const Tangent& first, double beta, const Tangent& second) const {
        check_point_(point);
        check_tangent_(first);
        check_tangent_(second);
        return internals::combine_symmetric<Scalar, Order_>(
          first, internals::geometry_coefficient<Scalar>(alpha), second, internals::geometry_coefficient<Scalar>(beta),
          order_);
    }
    /// @brief follows the globally defined exponential through the affine chart path
    template <SPDLike P> Point exponential(const P& point, const Tangent& tangent, double step = 1) const {
        const auto frame = chart_frame(point);
        return from_chart(
          internals::combine_symmetric<Scalar, Order_>(
            frame.coordinates, Scalar(1), chart_differential(frame, tangent),
            internals::geometry_coefficient<Scalar>(step), order_));
    }
    /// @brief uses the exact exponential as a retraction
    template <SPDLike P> Point retract(const P& point, const Tangent& tangent, double step) const {
        return exponential(point, tangent, step);
    }
    /// @brief retains both endpoint factors for repeated logarithm and differential operations
    template <SPDLike From, SPDLike To> RelativeFrame relative_frame(const From& from, const To& to) const {
        return {chart_frame(from), chart_frame(to)};
    }
    /// @brief returns the initial ambient tangent of the unique minimizing geodesic
    template <SPDLike From, SPDLike To> Tangent logarithm(const From& from, const To& to) const {
        return logarithm(relative_frame(from, to));
    }
    /// @brief reconstructs the logarithm from retained flat endpoint coordinates
    Tangent logarithm(const RelativeFrame& frame) const {
        const auto difference = internals::combine_symmetric<Scalar, Order_>(
          frame.to.coordinates, Scalar(1), frame.from.coordinates, Scalar(-1), order_);
        return inverse_differential_(frame.from.factor, difference);
    }
    /// @brief computes the distance between two verified endpoints
    template <SPDLike From, SPDLike To> double distance(const From& from, const To& to) const {
        return distance(relative_frame(from, to));
    }
    /// @brief evaluates a scale-safe chart distance from retained endpoint coordinates
    double distance(const RelativeFrame& frame) const {
        return internals::frobenius_norm(
          internals::combine_symmetric<Scalar, Order_>(
            frame.to.coordinates, Scalar(1), frame.from.coordinates, Scalar(-1), order_),
          order_);
    }
    /// @brief owns the unique geodesic as a deferred affine chart curve
    template <SPDLike From, SPDLike To> Curve geodesic(const From& from, const To& to) const {
        const auto frame = relative_frame(from, to);
        return Curve(
          frame.from.coordinates, internals::combine_symmetric<Scalar, Order_>(
                                    frame.to.coordinates, Scalar(1), frame.from.coordinates, Scalar(-1), order_));
    }
    /// @brief transports an ambient tangent by keeping its chart coordinates constant
    template <SPDLike From, SPDLike To>
    Tangent transport(const From& from, const To& to, const Tangent& tangent) const {
        const auto frame = relative_frame(from, to);
        return inverse_differential_(frame.to.factor, chart_differential(frame.from, tangent));
    }
    /// @brief maps a Frobenius gradient to its metric-dual tangent
    template <SPDLike P> Tangent euclidean_to_riemannian_gradient(const P& point, const Tangent& gradient) const {
        const auto frame = chart_frame(point);
        check_tangent_(gradient);
        return inverse_differential_(frame.factor, inverse_adjoint_(frame.factor, gradient));
    }
    /// @brief maps a metric-dual tangent to its Frobenius covector
    template <SPDLike P> Tangent riemannian_to_euclidean_gradient(const P& point, const Tangent& gradient) const {
        const auto frame = chart_frame(point);
        return chart_differential_adjoint(frame, chart_differential(frame, gradient));
    }
    /// @brief applies the target differential of the logarithm between ambient tangent spaces
    template <SPDLike From, SPDLike To>
    Tangent logarithm_target_jvp(const From& from, const To& to, const Tangent& direction) const {
        return logarithm_target_jvp(relative_frame(from, to), direction);
    }
    /// @brief reuses endpoint factors for the flat logarithm target differential
    Tangent logarithm_target_jvp(const RelativeFrame& frame, const Tangent& direction) const {
        return inverse_differential_(frame.from.factor, chart_differential(frame.to, direction));
    }
    /// @brief reverses the logarithm target differential in the endpoint metrics
    template <SPDLike From, SPDLike To>
    Tangent logarithm_target_vjp(const From& from, const To& to, const Tangent& metric_dual) const {
        return logarithm_target_vjp(relative_frame(from, to), metric_dual);
    }
    /// @brief reuses endpoint factors for the inverse parallel transport adjoint
    Tangent logarithm_target_vjp(const RelativeFrame& frame, const Tangent& metric_dual) const {
        return inverse_differential_(frame.to.factor, chart_differential(frame.from, metric_dual));
    }
    /// @brief applies the base Hessian of one half the squared distance in the flat metric
    template <SPDLike From, SPDLike To>
    Tangent half_squared_distance_hessian_vector(const From& from, const To& to, const Tangent& direction) const {
        return half_squared_distance_hessian_vector(relative_frame(from, to), direction);
    }
    /// @brief returns the identity Hessian under the globally isometric chart
    Tangent half_squared_distance_hessian_vector(const RelativeFrame&, const Tangent& direction) const {
        check_tangent_(direction);
        return direction;
    }
   private:
    /// @brief allocates a zero native dense triangular workspace
    Factor make_factor_() const {
        Factor result;
        if constexpr (Order_ == Dynamic) result.resize(order_, order_);
        for (int i = 0; i < order_; ++i)
            for (int j = 0; j < order_; ++j) result(i, j) = Scalar(0);
        return result;
    }
    /// @brief reconstructs a lower Cholesky factor from the finite symmetric chart
    Factor factor_from_chart_(const Tangent& coordinates) const {
        auto factor = make_factor_();
        const Scalar root_two = std::sqrt(Scalar(2));
        for (int i = 0; i < order_; ++i)
            for (int j = 0; j <= i; ++j) {
                factor(i, j) = internals::checked_geometry_result(
                  i == j ? std::exp(static_cast<Scalar>(coordinates(i, i))) : root_two * coordinates(i, j));
                fdapde_strong_assert(
                  i != j || factor(i, i) > 0, std::domain_error,
                  "log-Cholesky: diagonal exponential is not representably positive");
            }
        return factor;
    }
    /// @brief forms the symmetric product of a finite lower factor
    Tangent factor_product_(const Factor& factor) const {
        auto result = internals::make_symmetric<Scalar, Order_>(order_);
        for (int i = 0; i < order_; ++i)
            for (int j = 0; j <= i; ++j) {
                Scalar value = 0;
                for (int k = 0; k <= j; ++k) value += factor(i, k) * factor(j, k);
                result(i, j) = internals::checked_geometry_result(value);
            }
        return result;
    }
    /// @brief differentiates a Cholesky factor in global chart coordinates
    Factor factor_differential_(const Factor& factor, const Tangent& direction) const {
        auto result = make_factor_();
        const Scalar root_two = std::sqrt(Scalar(2));
        for (int i = 0; i < order_; ++i)
            for (int j = 0; j <= i; ++j)
                result(i, j) = i == j ? factor(i, i) * direction(i, i) : root_two * direction(i, j);
        return result;
    }
    /// @brief applies the inverse-chart Jacobian with a prepared factor
    Tangent inverse_differential_(const Factor& factor, const Tangent& direction) const {
        const auto derivative = factor_differential_(factor, direction);
        auto result = internals::make_symmetric<Scalar, Order_>(order_);
        for (int i = 0; i < order_; ++i)
            for (int j = 0; j <= i; ++j) {
                Scalar value = 0;
                for (int k = 0; k <= j; ++k) value += derivative(i, k) * factor(j, k) + factor(i, k) * derivative(j, k);
                result(i, j) = internals::checked_geometry_result(value);
            }
        return result;
    }
    /// @brief applies the second inverse-chart differential through factor product derivatives
    Tangent inverse_second_differential_(const Factor& factor, const Tangent& first, const Tangent& second) const {
        const auto first_factor = factor_differential_(factor, first);
        const auto second_factor = factor_differential_(factor, second);
        auto result = internals::make_symmetric<Scalar, Order_>(order_);
        for (int i = 0; i < order_; ++i)
            for (int j = 0; j <= i; ++j) {
                Scalar value = 0;
                for (int k = 0; k <= j; ++k) {
                    value += first_factor(i, k) * second_factor(j, k) + second_factor(i, k) * first_factor(j, k);
                    if (i == k) value += factor(i, i) * first(i, i) * second(i, i) * factor(j, k);
                    if (j == k) value += factor(i, k) * factor(j, j) * first(j, j) * second(j, j);
                }
                result(i, j) = internals::checked_geometry_result(value);
            }
        return result;
    }
    /// @brief applies the inverse-chart Frobenius adjoint without dense Jacobian assembly
    Tangent inverse_adjoint_(const Factor& factor, const Tangent& cotangent) const {
        auto result = internals::make_symmetric<Scalar, Order_>(order_);
        const Scalar root_two = std::sqrt(Scalar(2));
        for (int i = 0; i < order_; ++i)
            for (int j = 0; j <= i; ++j) {
                Scalar value = 0;
                for (int k = j; k < order_; ++k) value += cotangent(i, k) * factor(k, j);
                result(i, j) =
                  internals::checked_geometry_result(i == j ? Scalar(2) * value * factor(i, i) : root_two * value);
            }
        return result;
    }
    /// @brief applies inverse triangular congruences in forward or transposed backward order
    Tangent inverse_congruence_(const Factor& factor, const Tangent& rhs, bool transpose) const {
        auto first = make_factor_();
        auto second = make_factor_();
        for (int step = 0; step < order_; ++step) {
            const int i = transpose ? order_ - 1 - step : step;
            for (int j = 0; j < order_; ++j) {
                Scalar value = rhs(i, j);
                for (int previous = 0; previous < step; ++previous) {
                    const int k = transpose ? order_ - 1 - previous : previous;
                    value -= (transpose ? factor(k, i) : factor(i, k)) * first(k, j);
                }
                first(i, j) = value / factor(i, i);
            }
        }
        for (int step = 0; step < order_; ++step) {
            const int j = transpose ? order_ - 1 - step : step;
            for (int i = 0; i < order_; ++i) {
                Scalar value = first(i, j);
                for (int previous = 0; previous < step; ++previous) {
                    const int k = transpose ? order_ - 1 - previous : previous;
                    value -= second(i, k) * (transpose ? factor(k, j) : factor(j, k));
                }
                second(i, j) = value / factor(j, j);
            }
        }
        auto result = internals::make_symmetric<Scalar, Order_>(order_);
        for (int i = 0; i < order_; ++i)
            for (int j = 0; j <= i; ++j) result(i, j) = internals::checked_geometry_result(second(i, j));
        return result;
    }
    /// @brief validates a public endpoint against the geometry order
    template <SPDLike P> void check_point_(const P& point) const { internals::check_spd_geometry_shape(point, order_); }
    /// @brief validates finite ambient or chart tangent coefficients
    void check_tangent_(const Tangent& tangent) const { internals::check_spd_geometry_shape(tangent, order_); }
    int order_ = Order_ == Dynamic ? 0 : Order_;
};

/// @brief computes the globally unique closed-form barycenter and flat stationarity diagnostics
template <typename Scalar, int Order, Usage Uses, typename Samples>
WeightedKarcherMeanResult<typename LogCholeskySPDGeometry<Scalar, Order, Uses>::Point> weighted_karcher_mean(
  const LogCholeskySPDGeometry<Scalar, Order, Uses>& geometry, const Samples& samples,
  std::span<const double> weights) {
    using Geometry = LogCholeskySPDGeometry<Scalar, Order, Uses>;
    using Tangent = typename Geometry::Tangent;
    using Accumulation = std::common_type_t<Scalar, double>;
    fdapde_strong_assert(
      samples.size() != 0 && samples.size() == weights.size(), std::invalid_argument,
      "log-Cholesky mean requires matching nonempty samples and weights");
    auto normalized = internals::normalize_karcher_weights(weights);
    auto sum = internals::make_symmetric<Accumulation, Order>(geometry.order());
    auto correction = internals::make_symmetric<Accumulation, Order>(geometry.order());
    for (int i = 0; i < geometry.order(); ++i)
        for (int j = 0; j <= i; ++j) {
            sum(i, j) = 0;
            correction(i, j) = 0;
        }
    std::vector<std::optional<Tangent>> charts(samples.size());
    long double total = 0;
    for (std::size_t k = 0; k < samples.size(); ++k) {
        if (normalized[k] == 0) continue;
        charts[k].emplace(geometry.chart(samples[k]));
        total += normalized[k];
        for (int i = 0; i < geometry.order(); ++i)
            for (int j = 0; j <= i; ++j) {
                const Accumulation corrected = normalized[k] * (*charts[k])(i, j) - correction(i, j);
                const Accumulation next = sum(i, j) + corrected;
                correction(i, j) = (next - sum(i, j)) - corrected;
                sum(i, j) = next;
            }
    }
    auto mean_chart = internals::make_symmetric<Scalar, Order>(geometry.order());
    for (int i = 0; i < geometry.order(); ++i)
        for (int j = 0; j <= i; ++j) mean_chart(i, j) = static_cast<Scalar>(sum(i, j) / total);
    auto point = geometry.from_chart(mean_chart);
    const auto represented_chart = geometry.chart(point);
    auto residual = internals::make_symmetric<Accumulation, Order>(geometry.order());
    for (int i = 0; i < geometry.order(); ++i)
        for (int j = 0; j <= i; ++j) {
            residual(i, j) = 0;
            correction(i, j) = 0;
        }
    double cost = 0;
    for (std::size_t k = 0; k < samples.size(); ++k) {
        if (normalized[k] == 0) continue;
        double distance = 0;
        for (int i = 0; i < geometry.order(); ++i)
            for (int j = 0; j <= i; ++j) {
                const Accumulation difference = (*charts[k])(i, j) - represented_chart(i, j);
                const Accumulation corrected = normalized[k] * difference - correction(i, j);
                const Accumulation next = static_cast<Accumulation>(residual(i, j)) + corrected;
                correction(i, j) = (next - static_cast<Accumulation>(residual(i, j))) - corrected;
                residual(i, j) = next;
                distance = std::hypot(distance, static_cast<double>(difference));
                if (i != j) distance = std::hypot(distance, static_cast<double>(difference));
            }
        const double scaled = std::sqrt(normalized[k]) * distance;
        cost = std::fma(0.5 * scaled, scaled, cost);
    }
    const double stationarity = internals::frobenius_norm(residual, geometry.order());
    return {
      std::move(point),
      std::move(normalized),
      cost,
      stationarity,
      0,
      1,
      std::isfinite(cost) ? 1u : 0u,
      0,
      std::isfinite(cost) ? BarycenterStopReason::closed_form : BarycenterStopReason::non_finite_cost,
      BarycenterUniqueness::globally_unique,
      ArmijoStatus::not_run};
}

}   // namespace manifold
}   // namespace fdapde

#endif   // __FDAPDE_MANIFOLD_LOG_CHOLESKY_SPD_H__
