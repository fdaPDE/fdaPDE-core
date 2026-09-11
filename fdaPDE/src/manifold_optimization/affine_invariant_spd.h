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

#include "header_check.h"
#include "spd_geometry_common.h"

namespace fdapde {
namespace manifold {

/// @brief defines the affine-invariant metric on checked SPD owners with ambient symmetric tangents
template <typename Scalar_, int Order_> class AffineInvariantSPDGeometry {
    fdapde_static_assert(
      std::is_floating_point_v<Scalar_> && !std::is_const_v<Scalar_> && !std::is_volatile_v<Scalar_>,
      SPD_GEOMETRIES_REQUIRE_AN_UNQUALIFIED_FLOATING_POINT_SCALAR);
    fdapde_static_assert(Order_ == fdapde::Dynamic || Order_ > 0, INVALID_SPD_GEOMETRY_ORDER);
    fdapde_static_assert(
      Order_ == fdapde::Dynamic || std::int64_t(Order_) * std::int64_t(Order_) <= std::numeric_limits<int>::max(),
      SPD_GEOMETRY_DENSE_WORKSPACE_SIZE_EXCEEDS_SUPPORTED_RANGE);
   public:
    using Scalar = Scalar_;
    using Point = fdapde::SPDMatrix<Scalar, Order_, Order_>;
    using Tangent = fdapde::SymmetricMatrix<Scalar, Order_, Order_>;

    AffineInvariantSPDGeometry()
        requires(Order_ != fdapde::Dynamic)
    = default;

    explicit AffineInvariantSPDGeometry(int order)
        requires(Order_ == fdapde::Dynamic)
        : order_(order) {
        internals::validate_spd_geometry_order(order_);
    }

    /// @brief returns the matrix order
    int order() const { return order_; }
    /// @brief returns the number of independent tangent coefficients
    std::size_t dimension() const { return internals::spd_geometry_dimension(order_); }

    /// @brief pairs ambient tangents in the metric at point
    double inner_product(const Point& point, const Tangent& u, const Tangent& v) const {
        check_point_(point);
        check_tangent_(u);
        check_tangent_(v);
        const auto inverse_sqrt = fdapde::matrix_inverse_sqrt(point);
        const auto whitened_u = internals::symmetric_congruence<Scalar, Order_>(inverse_sqrt, u, order_);
        const auto whitened_v = internals::symmetric_congruence<Scalar, Order_>(inverse_sqrt, v, order_);
        return static_cast<double>(internals::frobenius_inner(whitened_u, whitened_v, order_));
    }

    /// @brief returns the metric norm of an ambient tangent
    double norm(const Point& point, const Tangent& tangent) const {
        check_point_(point);
        check_tangent_(tangent);
        const auto inverse_sqrt = fdapde::matrix_inverse_sqrt(point);
        const auto whitened = internals::symmetric_congruence<Scalar, Order_>(inverse_sqrt, tangent, order_);
        return internals::frobenius_norm(whitened, order_);
    }

    /// @brief copies an already symmetric ambient tangent into independent storage
    Tangent project(const Point& point, const Tangent& ambient) const {
        check_point_(point);
        check_tangent_(ambient);
        return ambient;
    }

    /// @brief returns an owning zero tangent of the point order
    Tangent zero_tangent(const Point& point) const {
        check_point_(point);
        auto result = internals::make_symmetric<Scalar, Order_>(order_);
        for (int i = 0; i < order_; ++i) {
            for (int j = 0; j <= i; ++j) { result(i, j) = Scalar(0); }
        }
        return result;
    }

    /// @brief combines ambient tangents with finite coefficients
    Tangent
    linear_combination(const Point& point, double alpha, const Tangent& u, double beta, const Tangent& v) const {
        check_point_(point);
        check_tangent_(u);
        check_tangent_(v);
        return internals::combine_symmetric<Scalar, Order_>(
          u, internals::geometry_coefficient<Scalar>(alpha), v, internals::geometry_coefficient<Scalar>(beta), order_);
    }

    /// @brief uses P^(1/2) * (I + W + W^2 / 2) * P^(1/2), W = step * P^(-1/2) * U * P^(-1/2)
    Point retract(const Point& point, const Tangent& tangent, double step) const {
        check_point_(point);
        check_tangent_(tangent);
        const auto point_sqrt = fdapde::matrix_sqrt(point);
        const auto point_inverse_sqrt = fdapde::matrix_inverse_sqrt(point);
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
    Point exponential(const Point& point, const Tangent& tangent, double step = 1.0) const {
        check_point_(point);
        check_tangent_(tangent);
        const auto point_sqrt = fdapde::matrix_sqrt(point);
        const auto point_inverse_sqrt = fdapde::matrix_inverse_sqrt(point);
        const auto whitened = internals::symmetric_congruence<Scalar, Order_>(point_inverse_sqrt, tangent, order_);
        const auto scaled = internals::combine_symmetric<Scalar, Order_>(
          whitened, internals::geometry_coefficient<Scalar>(step), whitened, Scalar(0), order_);
        const auto chart_exponential = fdapde::matrix_exp(scaled);
        return Point(internals::symmetric_congruence<Scalar, Order_>(point_sqrt, chart_exponential, order_));
    }

    /// @brief returns the initial ambient tangent of the geodesic from source to target
    Tangent logarithm(const Point& from, const Point& to) const {
        check_point_(from);
        check_point_(to);
        const auto from_sqrt = fdapde::matrix_sqrt(from);
        const auto from_inverse_sqrt = fdapde::matrix_inverse_sqrt(from);
        const Point relative(internals::symmetric_congruence<Scalar, Order_>(from_inverse_sqrt, to, order_));
        return internals::symmetric_congruence<Scalar, Order_>(from_sqrt, fdapde::matrix_log(relative), order_);
    }

    /// @brief returns the geodesic distance between checked points
    double distance(const Point& from, const Point& to) const {
        check_point_(from);
        check_point_(to);
        const auto from_inverse_sqrt = fdapde::matrix_inverse_sqrt(from);
        const Point relative(internals::symmetric_congruence<Scalar, Order_>(from_inverse_sqrt, to, order_));
        return internals::frobenius_norm(fdapde::matrix_log(relative), order_);
    }

    /// @brief parallel-transports an ambient tangent along the source-to-target geodesic
    Tangent transport(const Point& from, const Point& to, const Tangent& tangent) const {
        check_point_(from);
        check_point_(to);
        check_tangent_(tangent);
        const auto from_sqrt = fdapde::matrix_sqrt(from);
        const auto from_inverse_sqrt = fdapde::matrix_inverse_sqrt(from);
        const Point relative(internals::symmetric_congruence<Scalar, Order_>(from_inverse_sqrt, to, order_));
        const auto relative_sqrt = fdapde::matrix_sqrt(relative);
        const auto whitened = internals::symmetric_congruence<Scalar, Order_>(from_inverse_sqrt, tangent, order_);
        const auto transported_whitened =
          internals::symmetric_congruence<Scalar, Order_>(relative_sqrt, whitened, order_);
        return internals::symmetric_congruence<Scalar, Order_>(from_sqrt, transported_whitened, order_);
    }

    /// @brief converts a symmetric Frobenius gradient to its Riemannian metric dual
    Tangent euclidean_to_riemannian_gradient(const Point& point, const Tangent& euclidean_gradient) const {
        check_point_(point);
        check_tangent_(euclidean_gradient);
        return internals::symmetric_congruence<Scalar, Order_>(point, euclidean_gradient, order_);
    }
   private:
    void check_point_(const Point& point) const { internals::check_spd_geometry_shape(point, order_); }
    void check_tangent_(const Tangent& tangent) const { internals::check_spd_geometry_shape(tangent, order_); }

    int order_ = Order_ == fdapde::Dynamic ? 0 : Order_;
};

}   // namespace manifold
}   // namespace fdapde

#endif   // __FDAPDE_MANIFOLD_AFFINE_INVARIANT_SPD_H__
