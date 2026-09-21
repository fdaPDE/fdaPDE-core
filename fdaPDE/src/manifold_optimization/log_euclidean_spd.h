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

#ifndef __FDAPDE_MANIFOLD_LOG_EUCLIDEAN_SPD_H__
#define __FDAPDE_MANIFOLD_LOG_EUCLIDEAN_SPD_H__

#include "geometry_expr.h"
#include "header_check.h"
#include "spd_geometry_common.h"

namespace fdapde {
namespace manifold {

/// @brief defines the log-Euclidean metric on checked SPD owners with ambient symmetric tangents
template <typename Scalar_, int Order_, Usage Uses_ = Usage::None> class LogEuclideanSPDGeometry {
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
    using CachePolicy = Cache::Policy<internals::log_euclidean_cache_flags(Uses_)>;
    using Point = fdapde::SPDMatrix<Scalar, Order_, Order_, CachePolicy>;
    using Tangent = fdapde::SymmetricMatrix<Scalar, Order_, Order_>;

    /// @brief constructs the fixed-order geometry using its positive compile-time matrix order
    LogEuclideanSPDGeometry()
        requires(Order_ != fdapde::Dynamic)
    = default;

    /// @brief constructs a dynamic geometry after checking positive order and the supported dense workspace bound
    explicit LogEuclideanSPDGeometry(int order)
        requires(Order_ == fdapde::Dynamic)
        : order_(order) {
        internals::validate_spd_geometry_order(order_);
    }

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
        const auto chart_u = fdapde::matrix_log_frechet(point, u);
        const auto chart_v = fdapde::matrix_log_frechet(point, v);
        return static_cast<double>(internals::frobenius_inner(chart_u, chart_v, order_));
    }

    /// @brief returns the metric norm of an ambient tangent
    template <SPDLike PointPoint> double norm(const PointPoint& point, const Tangent& tangent) const {
        check_point_(point);
        check_tangent_(tangent);
        return internals::frobenius_norm(fdapde::matrix_log_frechet(point, tangent), order_);
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

    /// @brief uses the exact exponential as a retraction
    template <SPDLike PointPoint> Point retract(const PointPoint& point, const Tangent& tangent, double step) const {
        return exponential(point, tangent, step);
    }

    /// @brief follows the geodesic with the supplied initial ambient tangent and finite step
    template <SPDLike PointPoint>
    Point exponential(const PointPoint& point, const Tangent& tangent, double step = 1.0) const {
        check_point_(point);
        check_tangent_(tangent);
        const auto chart = fdapde::matrix_log(point);
        const auto chart_tangent = fdapde::matrix_log_frechet(point, tangent);
        return fdapde::matrix_exp<CachePolicy>(internals::combine_symmetric<Scalar, Order_>(
          chart, Scalar(1), chart_tangent, internals::geometry_coefficient<Scalar>(step), order_));
    }

    /// @brief returns the initial ambient tangent of the geodesic from source to target
    template <SPDLike PointFrom, SPDLike PointTo> Tangent logarithm(const PointFrom& from, const PointTo& to) const {
        check_point_(from);
        check_point_(to);
        const auto from_chart = fdapde::matrix_log(from);
        const auto to_chart = fdapde::matrix_log(to);
        const auto chart_difference =
          internals::combine_symmetric<Scalar, Order_>(to_chart, Scalar(1), from_chart, Scalar(-1), order_);
        return internals::spd_exp_log_frechet(from, chart_difference);
    }

    /// @brief returns the geodesic distance between checked points
    template <SPDLike PointFrom, SPDLike PointTo> double distance(const PointFrom& from, const PointTo& to) const {
        check_point_(from);
        check_point_(to);
        const auto from_chart = fdapde::matrix_log(from);
        const auto to_chart = fdapde::matrix_log(to);
        const auto difference =
          internals::combine_symmetric<Scalar, Order_>(to_chart, Scalar(1), from_chart, Scalar(-1), order_);
        return internals::frobenius_norm(difference, order_);
    }

    /// @brief parallel-transports an ambient tangent along the source-to-target geodesic
    template <SPDLike PointFrom, SPDLike PointTo>
    Tangent transport(const PointFrom& from, const PointTo& to, const Tangent& tangent) const {
        check_point_(from);
        check_point_(to);
        check_tangent_(tangent);
        const auto chart_tangent = fdapde::matrix_log_frechet(from, tangent);
        return internals::spd_exp_log_frechet(to, chart_tangent);
    }

    /// @brief converts a symmetric Frobenius gradient to its Riemannian metric dual
    template <SPDLike PointPoint>
    Tangent euclidean_to_riemannian_gradient(const PointPoint& point, const Tangent& euclidean_gradient) const {
        check_point_(point);
        check_tangent_(euclidean_gradient);
        if constexpr (fdapde::internals::spd_cache_has_v<typename PointPoint::CachePolicy, Cache::Spectral>) {
            const auto first = internals::spd_exp_log_frechet(point, euclidean_gradient);
            return internals::spd_exp_log_frechet(point, first);
        } else {
            const auto chart = fdapde::matrix_log(point);
            return fdapde::matrix_exp_frechet(chart, fdapde::matrix_exp_frechet(chart, euclidean_gradient));
        }
    }

    /// @brief prepares an owning log-Euclidean geodesic snapshot from independently cached endpoints
    /// @details defers exp(log(from) + t * (log(to) - log(from))); SPD destinations certify evaluated coefficients
    template <SPDLike PointFrom, SPDLike PointTo> auto geodesic(const PointFrom& from, const PointTo& to) const {
        check_point_(from);
        check_point_(to);
        const Tangent first(fdapde::matrix_log(from));
        const Tangent last(fdapde::matrix_log(to));
        Tangent difference(last - first);
        return fdapde::internals::spd_geodesic<Scalar, Order_, false>(first, std::move(difference));
    }

    /// @brief defers exp(sum_i weights[i] * log(points[i])) without normalizing finite real weights
    /// @details borrows persistent operands and retains temporary selection nodes; owning temporaries are rejected
    template <typename Points, typename Weights>
        requires(
          !fdapde::internals::is_owning_rvalue_expression_v<Points &&> &&
          !fdapde::internals::is_owning_rvalue_expression_v<Weights &&> &&
          SPDLike<typename std::remove_cvref_t<Points>::MatrixType>)
    auto weighted_mean(Points&& points, Weights&& weights) const& {
        return fdapde::internals::log_euclidean_mean_expr<
          LogEuclideanSPDGeometry, fdapde::internals::batch_nested_t<Points&&>,
          fdapde::internals::batch_nested_t<Weights&&>>(
          *this, std::forward<Points>(points), std::forward<Weights>(weights));
    }
    /// @brief prevents a deferred mean from borrowing a temporary geometry
    template <typename Points, typename Weights> void weighted_mean(Points&&, Weights&&) const&& = delete;
   private:
    /// @brief checks the point order and finite packed coefficients against this geometry
    template <SPDLike PointPoint> void check_point_(const PointPoint& point) const {
        internals::check_spd_geometry_shape(point, order_);
    }
    /// @brief checks the tangent order and finite packed coefficients against this geometry
    void check_tangent_(const Tangent& tangent) const { internals::check_spd_geometry_shape(tangent, order_); }

    int order_ = Order_ == fdapde::Dynamic ? 0 : Order_;
};

}   // namespace manifold
}   // namespace fdapde

#endif   // __FDAPDE_MANIFOLD_LOG_EUCLIDEAN_SPD_H__
