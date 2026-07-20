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

#include "header_check.h"
#include "spd_geometry_common.h"

namespace fdapde {
namespace manifold {

template <typename Scalar_, int Order_> class LogEuclideanSPDGeometry {
    fdapde_static_assert(
      std::is_floating_point_v<Scalar_> && !std::is_const_v<Scalar_> && !std::is_volatile_v<Scalar_>,
      SPD_GEOMETRIES_REQUIRE_AN_UNQUALIFIED_FLOATING_POINT_SCALAR);
    fdapde_static_assert(Order_ == fdapde::Dynamic || Order_ > 0, INVALID_SPD_GEOMETRY_ORDER);
    fdapde_static_assert(
      Order_ == fdapde::Dynamic || std::int64_t(Order_) * std::int64_t(Order_) <= std::numeric_limits<int>::max(),
      SPD_GEOMETRY_DENSE_WORKSPACE_SIZE_EXCEEDS_SUPPORTED_RANGE);
   public:
    using Scalar = Scalar_;
    using Point = fdapde::linalg::SPDMatrix<Scalar, Order_, Order_>;
    using Tangent = fdapde::linalg::SymmetricMatrix<Scalar, Order_, Order_>;

    LogEuclideanSPDGeometry()
        requires(Order_ != fdapde::Dynamic)
    = default;

    explicit LogEuclideanSPDGeometry(int order)
        requires(Order_ == fdapde::Dynamic)
        : order_(order) {
        internals::validate_spd_geometry_order(order_);
    }

    int order() const { return order_; }
    std::size_t dimension() const { return internals::spd_geometry_dimension(order_); }

    double inner_product(const Point& point, const Tangent& u, const Tangent& v) const {
        check_point_(point);
        check_tangent_(u);
        check_tangent_(v);
        const auto chart_u = fdapde::linalg::matrix_log_frechet(point, u);
        const auto chart_v = fdapde::linalg::matrix_log_frechet(point, v);
        return static_cast<double>(internals::frobenius_inner<Scalar>(chart_u, chart_v, order_));
    }

    double norm(const Point& point, const Tangent& tangent) const {
        check_point_(point);
        check_tangent_(tangent);
        return internals::frobenius_norm(fdapde::linalg::matrix_log_frechet(point, tangent), order_);
    }

    Tangent project(const Point& point, const Tangent& ambient) const {
        check_point_(point);
        check_tangent_(ambient);
        return internals::copy_symmetric<Scalar, Order_>(ambient, order_);
    }

    Tangent zero_tangent(const Point& point) const {
        check_point_(point);
        auto result = internals::make_symmetric<Scalar, Order_>(order_);
        for (int i = 0; i < order_; ++i) {
            for (int j = 0; j <= i; ++j) { result(i, j) = Scalar(0); }
        }
        return result;
    }

    Tangent
    linear_combination(const Point& point, double alpha, const Tangent& u, double beta, const Tangent& v) const {
        check_point_(point);
        check_tangent_(u);
        check_tangent_(v);
        return internals::combine_symmetric<Scalar, Order_>(
          u, static_cast<Scalar>(alpha), v, static_cast<Scalar>(beta), order_);
    }

    Point retract(const Point& point, const Tangent& tangent, double step) const {
        return exponential(point, tangent, step);
    }

    Point exponential(const Point& point, const Tangent& tangent, double step = 1.0) const {
        check_point_(point);
        check_tangent_(tangent);
        const auto chart = fdapde::linalg::matrix_log(point);
        const auto chart_tangent = fdapde::linalg::matrix_log_frechet(point, tangent);
        return fdapde::linalg::matrix_exp(
          internals::combine_symmetric<Scalar, Order_>(
            chart, Scalar(1), chart_tangent, static_cast<Scalar>(step), order_));
    }

    Tangent logarithm(const Point& from, const Point& to) const {
        check_point_(from);
        check_point_(to);
        const auto from_chart = fdapde::linalg::matrix_log(from);
        const auto to_chart = fdapde::linalg::matrix_log(to);
        const auto chart_difference =
          internals::combine_symmetric<Scalar, Order_>(to_chart, Scalar(1), from_chart, Scalar(-1), order_);
        return fdapde::linalg::matrix_exp_frechet(from_chart, chart_difference);
    }

    double distance(const Point& from, const Point& to) const {
        check_point_(from);
        check_point_(to);
        const auto from_chart = fdapde::linalg::matrix_log(from);
        const auto to_chart = fdapde::linalg::matrix_log(to);
        const auto difference =
          internals::combine_symmetric<Scalar, Order_>(to_chart, Scalar(1), from_chart, Scalar(-1), order_);
        return internals::frobenius_norm(difference, order_);
    }

    Tangent transport(const Point& from, const Point& to, const Tangent& tangent) const {
        check_point_(from);
        check_point_(to);
        check_tangent_(tangent);
        const auto chart_tangent = fdapde::linalg::matrix_log_frechet(from, tangent);
        return fdapde::linalg::matrix_exp_frechet(fdapde::linalg::matrix_log(to), chart_tangent);
    }

    Tangent euclidean_to_riemannian_gradient(const Point& point, const Tangent& euclidean_gradient) const {
        check_point_(point);
        check_tangent_(euclidean_gradient);
        const auto chart = fdapde::linalg::matrix_log(point);
        return fdapde::linalg::matrix_exp_frechet(chart, fdapde::linalg::matrix_exp_frechet(chart, euclidean_gradient));
    }
   private:
    void check_point_(const Point& point) const { internals::check_spd_geometry_shape(point, order_); }
    void check_tangent_(const Tangent& tangent) const { internals::check_spd_geometry_shape(tangent, order_); }

    int order_ = Order_ == fdapde::Dynamic ? 0 : Order_;
};

}   // namespace manifold
}   // namespace fdapde

#endif   // __FDAPDE_MANIFOLD_LOG_EUCLIDEAN_SPD_H__
