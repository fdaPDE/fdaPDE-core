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
    using Point = fdapde::linalg::SPDMatrix<Scalar, Order_, Order_>;
    using Tangent = fdapde::linalg::SymmetricMatrix<Scalar, Order_, Order_>;

    AffineInvariantSPDGeometry()
        requires(Order_ != fdapde::Dynamic)
    = default;

    explicit AffineInvariantSPDGeometry(int order)
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
        const auto inverse_sqrt = fdapde::linalg::matrix_inverse_sqrt(point);
        const auto whitened_u = internals::symmetric_congruence<Scalar, Order_>(inverse_sqrt, u, order_);
        const auto whitened_v = internals::symmetric_congruence<Scalar, Order_>(inverse_sqrt, v, order_);
        return static_cast<double>(internals::frobenius_inner<Scalar>(whitened_u, whitened_v, order_));
    }

    double norm(const Point& point, const Tangent& tangent) const {
        check_point_(point);
        check_tangent_(tangent);
        const auto inverse_sqrt = fdapde::linalg::matrix_inverse_sqrt(point);
        const auto whitened = internals::symmetric_congruence<Scalar, Order_>(inverse_sqrt, tangent, order_);
        return internals::frobenius_norm(whitened, order_);
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
        check_point_(point);
        check_tangent_(tangent);
        const auto point_sqrt = fdapde::linalg::matrix_sqrt(point);
        const auto point_inverse_sqrt = fdapde::linalg::matrix_inverse_sqrt(point);
        const auto whitened = internals::symmetric_congruence<Scalar, Order_>(point_inverse_sqrt, tangent, order_);
        const auto scaled = internals::combine_symmetric<Scalar, Order_>(
          whitened, static_cast<Scalar>(step), whitened, Scalar(0), order_);
        const auto squared = internals::symmetric_square<Scalar, Order_>(scaled, order_);

        auto polynomial = internals::make_symmetric<Scalar, Order_>(order_);
        for (int i = 0; i < order_; ++i) {
            for (int j = 0; j <= i; ++j) {
                polynomial(i, j) = (i == j ? Scalar(1) : Scalar(0)) + scaled(i, j) + Scalar(0.5) * squared(i, j);
            }
        }
        return Point(
          internals::symmetric_congruence<Scalar, Order_>(point_sqrt, polynomial, order_), fdapde::linalg::checked);
    }

    Point exponential(const Point& point, const Tangent& tangent, double step = 1.0) const {
        check_point_(point);
        check_tangent_(tangent);
        const auto point_sqrt = fdapde::linalg::matrix_sqrt(point);
        const auto point_inverse_sqrt = fdapde::linalg::matrix_inverse_sqrt(point);
        const auto whitened = internals::symmetric_congruence<Scalar, Order_>(point_inverse_sqrt, tangent, order_);
        const auto scaled = internals::combine_symmetric<Scalar, Order_>(
          whitened, static_cast<Scalar>(step), whitened, Scalar(0), order_);
        const auto chart_exponential = fdapde::linalg::matrix_exp(scaled);
        return Point(
          internals::symmetric_congruence<Scalar, Order_>(point_sqrt, chart_exponential, order_),
          fdapde::linalg::checked);
    }

    Tangent logarithm(const Point& from, const Point& to) const {
        check_point_(from);
        check_point_(to);
        const auto from_sqrt = fdapde::linalg::matrix_sqrt(from);
        const auto from_inverse_sqrt = fdapde::linalg::matrix_inverse_sqrt(from);
        const Point relative(
          internals::symmetric_congruence<Scalar, Order_>(from_inverse_sqrt, to, order_), fdapde::linalg::checked);
        return internals::symmetric_congruence<Scalar, Order_>(from_sqrt, fdapde::linalg::matrix_log(relative), order_);
    }

    double distance(const Point& from, const Point& to) const {
        check_point_(from);
        check_point_(to);
        const auto from_inverse_sqrt = fdapde::linalg::matrix_inverse_sqrt(from);
        const Point relative(
          internals::symmetric_congruence<Scalar, Order_>(from_inverse_sqrt, to, order_), fdapde::linalg::checked);
        return internals::frobenius_norm(fdapde::linalg::matrix_log(relative), order_);
    }

    Tangent transport(const Point& from, const Point& to, const Tangent& tangent) const {
        check_point_(from);
        check_point_(to);
        check_tangent_(tangent);
        const auto from_sqrt = fdapde::linalg::matrix_sqrt(from);
        const auto from_inverse_sqrt = fdapde::linalg::matrix_inverse_sqrt(from);
        const Point relative(
          internals::symmetric_congruence<Scalar, Order_>(from_inverse_sqrt, to, order_), fdapde::linalg::checked);
        const auto relative_sqrt = fdapde::linalg::matrix_sqrt(relative);
        const auto whitened = internals::symmetric_congruence<Scalar, Order_>(from_inverse_sqrt, tangent, order_);
        const auto transported_whitened =
          internals::symmetric_congruence<Scalar, Order_>(relative_sqrt, whitened, order_);
        return internals::symmetric_congruence<Scalar, Order_>(from_sqrt, transported_whitened, order_);
    }

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

template <typename Scalar_, int Order_>
WeightedKarcherMeanResult<typename AffineInvariantSPDGeometry<Scalar_, Order_>::Point> weighted_karcher_mean(
  const AffineInvariantSPDGeometry<Scalar_, Order_>& geometry,
  std::span<const typename AffineInvariantSPDGeometry<Scalar_, Order_>::Point> samples, std::span<const double> weights,
  const typename AffineInvariantSPDGeometry<Scalar_, Order_>::Point& initial,
  const WeightedKarcherMeanOptions& options = {}) {
    using Geometry = AffineInvariantSPDGeometry<Scalar_, Order_>;
    auto result = weighted_karcher_mean<Geometry>(geometry, samples, weights, initial, options);
    result.uniqueness = BarycenterUniqueness::globally_unique;
    return result;
}

template <typename Scalar_, int Order_>
WeightedKarcherMeanResult<typename AffineInvariantSPDGeometry<Scalar_, Order_>::Point> weighted_karcher_mean(
  const AffineInvariantSPDGeometry<Scalar_, Order_>& geometry,
  std::span<const typename AffineInvariantSPDGeometry<Scalar_, Order_>::Point> samples, std::span<const double> weights,
  const WeightedKarcherMeanOptions& options = {}) {
    auto log_geometry = [&]() {
        if constexpr (Order_ == fdapde::Dynamic) {
            return LogEuclideanSPDGeometry<Scalar_, Order_>(geometry.order());
        } else {
            return LogEuclideanSPDGeometry<Scalar_, Order_>();
        }
    }();
    // use the exact log-Euclidean mean formula as a deterministic, sample-symmetric positive-definite initializer
    const auto initial = weighted_karcher_mean(log_geometry, samples, weights);
    return weighted_karcher_mean(geometry, samples, weights, initial.point, options);
}

}   // namespace manifold
}   // namespace fdapde

#endif   // __FDAPDE_MANIFOLD_AFFINE_INVARIANT_SPD_H__
