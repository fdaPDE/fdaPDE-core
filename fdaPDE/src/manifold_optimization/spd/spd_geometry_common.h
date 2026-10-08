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

#ifndef __FDAPDE_MANIFOLD_SPD_GEOMETRY_COMMON_H__
#define __FDAPDE_MANIFOLD_SPD_GEOMETRY_COMMON_H__

#include "../header_check.h"

namespace fdapde {
namespace manifold {
namespace internals {

/// @brief materializes uniform samples of one prepared curve using the geometry's scalar and order with an explicit
/// output cache
template <
  typename OutputPolicy, typename Geometry, SPDLike From, SPDLike To,
  fdapde::internals::BatchExecutionPolicy ExecutionPolicy>
auto sample_spd_geodesic(const Geometry& geometry, const From& from, const To& to, int count, ExecutionPolicy policy) {
    fdapde_strong_assert(count >= 2, std::invalid_argument, "geodesic: at least two samples are required");
    const auto curve = geometry.geodesic(from, to);
    using Point = typename Geometry::Point;
    using Result = SPDMatrix<typename Point::Scalar, Point::Rows, OutputPolicy>;
    return fdapde::internals::generate_matrix_batch<Result>(
      static_cast<std::size_t>(count), geometry.order(), geometry.order(), policy, [&](std::size_t i) {
          const double t = static_cast<double>(i) / (count - 1);
          if constexpr (requires { curve.template operator()<OutputPolicy>(t); })
              return curve.template operator()<OutputPolicy>(t);
          else
              return curve(t);
      });
}

/// @brief applies the exponential differential at log(point), reusing its spectral basis when retained
template <SPDLike Point, typename Direction> auto spd_exp_log_frechet(const Point& point, const Direction& direction) {
    using Policy = typename Point::CachePolicy;
    if constexpr (fdapde::internals::spd_cache_has_v<Policy, Cache::Spectral>) {
        if constexpr (fdapde::internals::spd_cache_has_v<Policy, Cache::LogDividedDifferences>) {
            const auto differences = point.cache().log_divided_differences();
            return fdapde::internals::frechet_symmetric(
              point.cache(), point.rows(), direction,
              [&](auto, auto, int i, int j) { return typename Point::Scalar(1) / differences(i, j); });
        } else
            return fdapde::internals::frechet_symmetric(point.cache(), point.rows(), direction, [](auto x, auto y) {
                return decltype(x)(1) / fdapde::internals::log_divided_difference(x, y);
            });
    } else {
        const auto chart = fdapde::matrix_log(point);
        return fdapde::matrix_exp_frechet(chart, direction);
    }
}

/// @brief tests whether a compile-time geometry usage requests any of the given operations
constexpr bool has_spd_usage(Usage uses, Usage requested) {
    return (static_cast<unsigned>(uses) & static_cast<unsigned>(requested)) != 0;
}
/// @brief maps log-Euclidean operations to the union of reusable algebraic quantities
constexpr unsigned log_euclidean_cache_flags(Usage uses) {
    return (has_spd_usage(uses, Usage::Distance | Usage::InterpolationNodes | Usage::BasePointMaps) ?
              Cache::Log::Flags :
              0u) |
           (has_spd_usage(uses, Usage::TangentMetric | Usage::LogExpDifferentials | Usage::BasePointMaps) ?
              Cache::Spectral::Flags | Cache::LogDividedDifferences::Flags :
              0u);
}
/// @brief maps affine-invariant operations to factors on the base point rather than interpolation-node caches
constexpr unsigned affine_invariant_cache_flags(Usage uses) {
    return (has_spd_usage(uses, Usage::Distance | Usage::TangentMetric | Usage::BasePointMaps) ?
              Cache::InverseSqrt::Flags :
              0u) |
           (has_spd_usage(uses, Usage::BasePointMaps) ? Cache::Sqrt::Flags : 0u) |
           (has_spd_usage(uses, Usage::LogExpDifferentials) ?
              Cache::Spectral::Flags | Cache::LogDividedDifferences::Flags :
              0u);
}
/// @brief borrows a retained square-root factor or computes an independent local factor
/// @details the cached symmetric factor is an internal intermediate; public SPD results remain certified
template <SPDLike Point> auto spd_sqrt_factor(const Point& point) {
    if constexpr (fdapde::internals::spd_cache_has_v<typename Point::CachePolicy, Cache::Sqrt>)
        return point.cache().template matrix<Cache::Sqrt>();
    else
        return fdapde::matrix_sqrt(point);
}
/// @brief borrows a retained inverse-square-root factor or computes an independent local factor
template <SPDLike Point> auto spd_inv_sqrt_factor(const Point& point) {
    if constexpr (fdapde::internals::spd_cache_has_v<typename Point::CachePolicy, Cache::InverseSqrt>)
        return point.cache().template matrix<Cache::InverseSqrt>();
    else
        return fdapde::matrix_inv_sqrt(point);
}

/// @brief checks positive order and the dense workspace bound
inline void validate_spd_geometry_order(int order) {
    if (order <= 0) { throw std::invalid_argument("SPD geometry order must be positive."); }
    const std::int64_t dimension = order;
    if (dimension * dimension > std::numeric_limits<int>::max()) {
        throw std::length_error("SPD geometry dense workspace size exceeds supported range.");
    }
}

/// @brief returns the dimension of the symmetric tangent space
inline std::size_t spd_geometry_dimension(int order) {
    const std::size_t dimension = static_cast<std::size_t>(order);
    return dimension * (dimension + 1) / 2;
}

/// @brief checks the geometry order and finite coefficients
template <typename MatrixType_> void check_spd_geometry_shape(const MatrixType_& matrix, int order) {
    if (matrix.rows() != order || matrix.cols() != order) {
        throw std::invalid_argument("SPD geometry matrix dimension mismatch.");
    }
    for (int i = 0; i < order; ++i) {
        for (int j = 0; j <= i; ++j) {
            if (!std::isfinite(static_cast<typename MatrixType_::Scalar>(matrix(i, j)))) {
                throw std::invalid_argument("SPD geometry coefficients must be finite.");
            }
        }
    }
}

/// @brief rejects nonfinite arithmetic results before returning an owning value
template <typename Scalar_> Scalar_ checked_geometry_result(Scalar_ value) {
    if (!std::isfinite(value)) throw std::domain_error("SPD geometry produced a nonfinite result.");
    return value;
}

/// @brief checks a public double coefficient before narrowing to the geometry scalar
template <typename Scalar_> Scalar_ geometry_coefficient(double value) {
    if (!std::isfinite(value) || std::abs(static_cast<long double>(value)) > std::numeric_limits<Scalar_>::max()) {
        throw std::invalid_argument("SPD geometry coefficient is not finite or representable.");
    }
    return static_cast<Scalar_>(value);
}

/// @brief allocates an owning symmetric result with the geometry order
template <typename Scalar_, int Order_> fdapde::SymmetricMatrix<Scalar_, Order_> make_symmetric(int order) {
    fdapde::SymmetricMatrix<Scalar_, Order_> result;
    if constexpr (Order_ == fdapde::Dynamic) { result.resize(order, order); }
    return result;
}

/// @brief forms a finite linear combination in owning symmetric storage
template <typename Scalar_, int Order_, typename LhsType_, typename RhsType_>
fdapde::SymmetricMatrix<Scalar_, Order_>
combine_symmetric(const LhsType_& lhs, Scalar_ alpha, const RhsType_& rhs, Scalar_ beta, int order) {
    if (!std::isfinite(alpha) || !std::isfinite(beta)) {
        throw std::invalid_argument("SPD geometry coefficients must be finite.");
    }
    auto result = make_symmetric<Scalar_, Order_>(order);
    for (int i = 0; i < order; ++i) {
        for (int j = 0; j <= i; ++j) {
            result(i, j) =
              checked_geometry_result(alpha * static_cast<Scalar_>(lhs(i, j)) + beta * static_cast<Scalar_>(rhs(i, j)));
        }
    }
    return result;
}

/// @brief accumulates the full Frobenius product, counting packed off-diagonal entries twice
template <typename LhsType_, typename RhsType_>
double frobenius_inner(const LhsType_& lhs, const RhsType_& rhs, int order) {
    long double result = 0;
    for (int i = 0; i < order; ++i) {
        result += static_cast<long double>(lhs(i, i)) * static_cast<long double>(rhs(i, i));
        for (int j = 0; j < i; ++j) {
            result += 2 * static_cast<long double>(lhs(i, j)) * static_cast<long double>(rhs(i, j));
        }
    }
    return checked_geometry_result(static_cast<double>(result));
}

/// @brief computes the full Frobenius norm with scale-safe hypot accumulation
template <typename MatrixType_> double frobenius_norm(const MatrixType_& matrix, int order) {
    double result = 0;
    for (int i = 0; i < order; ++i) {
        result = std::hypot(result, static_cast<double>(matrix(i, i)));
        for (int j = 0; j < i; ++j) {
            const double coefficient = static_cast<double>(matrix(i, j));
            result = std::hypot(result, coefficient);
            result = std::hypot(result, coefficient);
        }
    }
    return checked_geometry_result(result);
}

/// @brief forms outer * middle * outer transpose with checked native workspaces
template <typename Scalar_, int Order_, typename OuterType_, typename MiddleType_>
fdapde::SymmetricMatrix<Scalar_, Order_>
symmetric_congruence(const OuterType_& outer, const MiddleType_& middle, int order) {
    fdapde::Matrix<Scalar_, Order_, Order_> product;
    if constexpr (Order_ == fdapde::Dynamic) { product.resize(order, order); }
    for (int i = 0; i < order; ++i) {
        for (int j = 0; j < order; ++j) {
            Scalar_ value = 0;
            for (int k = 0; k < order; ++k) {
                value += static_cast<Scalar_>(outer(i, k)) * static_cast<Scalar_>(middle(k, j));
            }
            product(i, j) = checked_geometry_result(value);
        }
    }

    auto result = make_symmetric<Scalar_, Order_>(order);
    for (int i = 0; i < order; ++i) {
        for (int j = 0; j <= i; ++j) {
            Scalar_ value = 0;
            for (int k = 0; k < order; ++k) { value += product(i, k) * static_cast<Scalar_>(outer(j, k)); }
            result(i, j) = checked_geometry_result(value);
        }
    }
    return result;
}

/// @brief computes the square of a symmetric matrix in owning storage
template <typename Scalar_, int Order_, typename MatrixType_>
fdapde::SymmetricMatrix<Scalar_, Order_> symmetric_square(const MatrixType_& matrix, int order) {
    auto result = make_symmetric<Scalar_, Order_>(order);
    for (int i = 0; i < order; ++i) {
        for (int j = 0; j <= i; ++j) {
            Scalar_ value = 0;
            for (int k = 0; k < order; ++k) {
                value += static_cast<Scalar_>(matrix(i, k)) * static_cast<Scalar_>(matrix(k, j));
            }
            result(i, j) = checked_geometry_result(value);
        }
    }
    return result;
}

}   // namespace internals
}   // namespace manifold
}   // namespace fdapde

#endif   // __FDAPDE_MANIFOLD_SPD_GEOMETRY_COMMON_H__
