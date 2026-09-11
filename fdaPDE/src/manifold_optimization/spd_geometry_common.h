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

#include "header_check.h"

namespace fdapde {
namespace manifold {
namespace internals {

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
template <typename Scalar_, int Order_> fdapde::SymmetricMatrix<Scalar_, Order_, Order_> make_symmetric(int order) {
    fdapde::SymmetricMatrix<Scalar_, Order_, Order_> result;
    if constexpr (Order_ == fdapde::Dynamic) { result.resize(order, order); }
    return result;
}

/// @brief forms a finite linear combination in owning symmetric storage
template <typename Scalar_, int Order_, typename LhsType_, typename RhsType_>
fdapde::SymmetricMatrix<Scalar_, Order_, Order_>
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
fdapde::SymmetricMatrix<Scalar_, Order_, Order_>
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
fdapde::SymmetricMatrix<Scalar_, Order_, Order_> symmetric_square(const MatrixType_& matrix, int order) {
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
