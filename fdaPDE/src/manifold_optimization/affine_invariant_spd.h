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

    // Exact target differential of Log_from(to).
    // With R = c S, L_log(c S, V) = L_log(S, V) / c. The JVP absorbs
    // 1/c into its whitening congruence, the metric VJP absorbs c into
    // its unwhitening congruence, and c cancels from the Hessian action.
    Tangent logarithm_target_jvp(const Point& from, const Point& to, const Tangent& to_direction) const {
        check_tangent_(to_direction);
        const auto frame = relative_frame_(from, to);
        const Scalar inverse_sqrt_scale = Scalar(1) / std::sqrt(frame.relative_scale);
        const auto scaled_inverse_sqrt = internals::combine_symmetric<Scalar, Order_>(
          frame.from_inverse_sqrt, inverse_sqrt_scale, frame.from_inverse_sqrt, Scalar(0), order_);
        const auto scaled_direction =
          internals::symmetric_congruence<Scalar, Order_>(scaled_inverse_sqrt, to_direction, order_);
        const auto chart_direction = logarithm_frechet_(frame.scaled_relative, scaled_direction);
        return checked_tangent_result_(
          internals::symmetric_congruence<Scalar, Order_>(frame.from_sqrt, chart_direction, order_));
    }

    // AIRM-metric adjoint of logarithm_target_jvp. The argument and result
    // are metric-dual tangent representations at from and to, respectively.
    Tangent logarithm_target_vjp(const Point& from, const Point& to, const Tangent& from_metric_dual) const {
        check_tangent_(from_metric_dual);
        const auto frame = relative_frame_(from, to);
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

    // Covariant Hessian action at base of one half the squared distance to
    // target. This is the negative covariant base differential of Log_base(target).
    Tangent
    half_squared_distance_hessian_vector(const Point& base, const Point& target, const Tangent& base_direction) const {
        check_tangent_(base_direction);
        const auto frame = relative_frame_(base, target);
        const auto whitened_direction =
          internals::symmetric_congruence<Scalar, Order_>(frame.from_inverse_sqrt, base_direction, order_);
        const auto log_direction = logarithm_frechet_(frame.scaled_relative, whitened_direction);
        // Jordan_S and L_log(S, .) commute because they share S's spectral basis.
        const auto chart_result = jordan_product_(frame.scaled_relative, log_direction);
        return checked_tangent_result_(
          internals::symmetric_congruence<Scalar, Order_>(frame.from_sqrt, chart_result, order_));
    }

    // Covariant derivative of half_squared_distance_hessian_vector along
    // simultaneous base/target variation. The action direction is continued
    // parallelly along the base variation.
    Tangent half_squared_distance_hessian_covariant_jvp(
      const Point& base, const Point& target, const Tangent& base_direction, const Tangent& target_direction,
      const Tangent& action_direction) const {
        check_tangent_(base_direction);
        check_tangent_(target_direction);
        check_tangent_(action_direction);
        const auto frame = relative_frame_(base, target);
        const auto whitened_base_direction =
          internals::symmetric_congruence<Scalar, Order_>(frame.from_inverse_sqrt, base_direction, order_);
        const auto whitened_action_direction =
          internals::symmetric_congruence<Scalar, Order_>(frame.from_inverse_sqrt, action_direction, order_);

        // For the unscaled relative point R = c S, its whitened variation
        // divided by c is E/c - Jordan(X, S). Expressing the formula with this
        // scaled variation cancels both powers of c in D2 log(c S).
        const Scalar inverse_sqrt_scale = Scalar(1) / std::sqrt(frame.relative_scale);
        const auto scaled_inverse_sqrt = internals::combine_symmetric<Scalar, Order_>(
          frame.from_inverse_sqrt, inverse_sqrt_scale, frame.from_inverse_sqrt, Scalar(0), order_);
        const auto scaled_target_direction =
          internals::symmetric_congruence<Scalar, Order_>(scaled_inverse_sqrt, target_direction, order_);
        const auto base_relative_change = jordan_product_(whitened_base_direction, frame.scaled_relative);
        const auto relative_direction = internals::combine_symmetric<Scalar, Order_>(
          scaled_target_direction, Scalar(1), base_relative_change, Scalar(-1), order_);

        const auto log_action = logarithm_frechet_(frame.scaled_relative, whitened_action_direction);
        const auto log_second = fdapde::linalg::matrix_log_second_frechet(
          frame.scaled_relative, relative_direction, whitened_action_direction);
        const auto chart_result = internals::combine_symmetric<Scalar, Order_>(
          jordan_product_(relative_direction, log_action), Scalar(1),
          jordan_product_(frame.scaled_relative, log_second), Scalar(1), order_);
        return checked_tangent_result_(
          internals::symmetric_congruence<Scalar, Order_>(frame.from_sqrt, chart_result, order_));
    }

    // Metric adjoint of the simultaneous base/target variation in
    // half_squared_distance_hessian_covariant_jvp, for a fixed parallel
    // action direction. The pair contains base and target metric-dual tangents.
    std::pair<Tangent, Tangent> half_squared_distance_hessian_covariant_vjp(
      const Point& base, const Point& target, const Tangent& action_direction,
      const Tangent& output_metric_dual) const {
        check_tangent_(action_direction);
        check_tangent_(output_metric_dual);
        const auto frame = relative_frame_(base, target);
        const auto whitened_action =
          internals::symmetric_congruence<Scalar, Order_>(frame.from_inverse_sqrt, action_direction, order_);
        const auto whitened_output =
          internals::symmetric_congruence<Scalar, Order_>(frame.from_inverse_sqrt, output_metric_dual, order_);
        const auto log_action = logarithm_frechet_(frame.scaled_relative, whitened_action);
        const auto relative_output = jordan_product_(frame.scaled_relative, whitened_output);
        const auto log_second = fdapde::linalg::matrix_log_second_frechet(
          frame.scaled_relative, relative_output, whitened_action);
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
          internals::combine_symmetric<Scalar, Order_>(
            frame.from_sqrt, sqrt_scale, frame.from_sqrt, Scalar(0), order_);
        Tangent target_dual = checked_tangent_result_(
          internals::symmetric_congruence<Scalar, Order_>(scaled_from_sqrt, target_chart_dual, order_));
        return {std::move(base_dual), std::move(target_dual)};
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
    struct RelativeFrame {
        Point from_sqrt;
        Point from_inverse_sqrt;
        Point scaled_relative;
        Scalar relative_scale;
    };

    RelativeFrame relative_frame_(const Point& from, const Point& to) const {
        check_point_(from);
        check_point_(to);
        for (int i = 0; i < order_; ++i) {
            for (int j = 0; j <= i; ++j) {
                if (!std::isfinite(static_cast<Scalar>(to(i, j)))) {
                    throw std::invalid_argument("Affine-invariant SPD differential point coefficients must be finite");
                }
            }
        }
        auto from_sqrt = fdapde::linalg::matrix_sqrt(from);
        auto from_inverse_sqrt = fdapde::linalg::matrix_inverse_sqrt(from);
        const auto relative = internals::symmetric_congruence<Scalar, Order_>(from_inverse_sqrt, to, order_);

        Scalar scale = 0;
        for (int i = 0; i < order_; ++i) {
            for (int j = 0; j <= i; ++j) {
                const Scalar coefficient = static_cast<Scalar>(relative(i, j));
                if (!std::isfinite(coefficient)) {
                    throw std::domain_error("Affine-invariant SPD differential produced a nonfinite relative point");
                }
                scale = std::max(scale, std::abs(coefficient));
            }
        }
        if (!(scale > Scalar(0)) || !std::isfinite(scale)) {
            throw std::domain_error("Affine-invariant SPD differential has an invalid relative scale");
        }

        auto scaled_relative = internals::make_symmetric<Scalar, Order_>(order_);
        for (int i = 0; i < order_; ++i) {
            for (int j = 0; j <= i; ++j) { scaled_relative(i, j) = static_cast<Scalar>(relative(i, j)) / scale; }
        }
        return {
          std::move(from_sqrt), std::move(from_inverse_sqrt), Point(scaled_relative, fdapde::linalg::checked), scale};
    }

    Tangent logarithm_frechet_(const Point& point, const Tangent& direction) const {
        if constexpr (Order_ == fdapde::Dynamic) {
            return fdapde::linalg::matrix_log_frechet(point, direction);
        } else {
            // A dynamic read operand avoids a false GCC bounds diagnostic when
            // the packed fixed-size proxy is inlined into the spectral loop.
            fdapde::linalg::SymmetricMatrix<Scalar, fdapde::Dynamic, fdapde::Dynamic> dynamic_direction(order_, order_);
            dynamic_direction = direction;
            return fdapde::linalg::matrix_log_frechet(point, dynamic_direction);
        }
    }

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

    Tangent checked_tangent_result_(Tangent result) const {
        for (int i = 0; i < order_; ++i) {
            for (int j = 0; j <= i; ++j) {
                if (!std::isfinite(static_cast<Scalar>(result(i, j)))) {
                    throw std::domain_error("Affine-invariant SPD differential produced a nonfinite tangent");
                }
            }
        }
        return result;
    }

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
