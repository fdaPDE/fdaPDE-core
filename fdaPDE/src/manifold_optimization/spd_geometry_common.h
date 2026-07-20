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

inline void validate_spd_geometry_order(int order) {
    if (order <= 0) { throw std::invalid_argument("SPD geometry order must be positive."); }
    const std::int64_t dimension = order;
    if (dimension * dimension > std::numeric_limits<int>::max()) {
        throw std::length_error("SPD geometry dense workspace size exceeds supported range.");
    }
}

inline std::size_t spd_geometry_dimension(int order) {
    const std::size_t dimension = static_cast<std::size_t>(order);
    return dimension * (dimension + 1) / 2;
}

template <typename MatrixType_> void check_spd_geometry_shape(const MatrixType_& matrix, int order) {
    if (matrix.rows() != order || matrix.cols() != order) {
        throw std::invalid_argument("SPD geometry matrix dimension mismatch.");
    }
}

template <typename Scalar_, int Order_>
fdapde::linalg::SymmetricMatrix<Scalar_, Order_, Order_> make_symmetric(int order) {
    fdapde::linalg::SymmetricMatrix<Scalar_, Order_, Order_> result;
    if constexpr (Order_ == fdapde::Dynamic) { result.resize(order, order); }
    return result;
}

template <typename Scalar_, int Order_, typename MatrixType_>
fdapde::linalg::SymmetricMatrix<Scalar_, Order_, Order_> copy_symmetric(const MatrixType_& matrix, int order) {
    auto result = make_symmetric<Scalar_, Order_>(order);
    for (int i = 0; i < order; ++i) {
        for (int j = 0; j <= i; ++j) { result(i, j) = static_cast<Scalar_>(matrix(i, j)); }
    }
    return result;
}

template <typename Scalar_, int Order_, typename LhsType_, typename RhsType_>
fdapde::linalg::SymmetricMatrix<Scalar_, Order_, Order_>
combine_symmetric(const LhsType_& lhs, Scalar_ alpha, const RhsType_& rhs, Scalar_ beta, int order) {
    auto result = make_symmetric<Scalar_, Order_>(order);
    for (int i = 0; i < order; ++i) {
        for (int j = 0; j <= i; ++j) {
            result(i, j) = alpha * static_cast<Scalar_>(lhs(i, j)) + beta * static_cast<Scalar_>(rhs(i, j));
        }
    }
    return result;
}

template <typename Scalar_, typename LhsType_, typename RhsType_>
Scalar_ frobenius_inner(const LhsType_& lhs, const RhsType_& rhs, int order) {
    Scalar_ result = 0;
    for (int i = 0; i < order; ++i) {
        result += static_cast<Scalar_>(lhs(i, i)) * static_cast<Scalar_>(rhs(i, i));
        for (int j = 0; j < i; ++j) {
            result += Scalar_(2) * static_cast<Scalar_>(lhs(i, j)) * static_cast<Scalar_>(rhs(i, j));
        }
    }
    return result;
}

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
    return result;
}

}   // namespace internals
}   // namespace manifold
}   // namespace fdapde

#endif   // __FDAPDE_MANIFOLD_SPD_GEOMETRY_COMMON_H__
