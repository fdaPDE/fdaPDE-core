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

#ifndef __FDAPDE_LINALG_KERNELS_SO3_H__
#define __FDAPDE_LINALG_KERNELS_SO3_H__

#include <array>
#include <cmath>
#include <limits>

#include "../header_check.h"

namespace fdapde {
namespace internals {

/// @brief normalizes finite quaternion coefficients without squaring their original scale
/// @details nonzero scalar multiples represent the same orientation, including coefficients near overflow
template <typename S> std::array<S, 4> normalized_quaternion(std::array<S, 4> value) {
    S scale = 0;
    for (S coefficient : value) {
        fdapde_strong_assert(
          std::isfinite(coefficient), std::invalid_argument, "quaternion coefficients must be finite");
        scale = std::max(scale, std::abs(coefficient));
    }
    fdapde_strong_assert(scale > S(0), std::domain_error, "quaternion norm must be positive");
    S norm = 0;
    for (S& coefficient : value) {
        coefficient /= scale;
        norm = std::hypot(norm, coefficient);
    }
    for (S& coefficient : value) coefficient /= norm;
    return value;
}

/// @brief extracts a normalized quaternion using the largest stable trace or diagonal pivot
/// @details the nonnegative scalar convention selects the principal hemisphere without dividing by a small sine
template <typename Q> auto so3_quaternion(const Q& rotation) {
    using S = std::remove_cv_t<typename Q::Scalar>;
    std::array<S, 4> value {};
    const S trace = rotation(0, 0) + rotation(1, 1) + rotation(2, 2);
    if (trace > S(0)) {
        const S denominator = S(2) * std::sqrt(S(1) + trace);
        value[0] = denominator / S(4);
        value[1] = (rotation(2, 1) - rotation(1, 2)) / denominator;
        value[2] = (rotation(0, 2) - rotation(2, 0)) / denominator;
        value[3] = (rotation(1, 0) - rotation(0, 1)) / denominator;
    } else {
        int i = 0;
        if (rotation(1, 1) > rotation(i, i)) i = 1;
        if (rotation(2, 2) > rotation(i, i)) i = 2;
        const int j = (i + 1) % 3, k = (i + 2) % 3;
        const S denominator = S(2) * std::sqrt(S(1) + rotation(i, i) - rotation(j, j) - rotation(k, k));
        value[i + 1] = denominator / S(4);
        value[j + 1] = (rotation(j, i) + rotation(i, j)) / denominator;
        value[k + 1] = (rotation(k, i) + rotation(i, k)) / denominator;
        value[0] = (rotation(k, j) - rotation(j, k)) / denominator;
    }
    value = normalized_quaternion(value);
    if (value[0] < S(0))
        for (S& coefficient : value) coefficient = -coefficient;
    return value;
}

/// @brief maps a finite rotation vector to a unit quaternion with a removable small-angle limit
template <typename S> auto so3_quaternion_exp(const std::array<S, 3>& vector) {
    S angle = 0;
    for (S coefficient : vector) {
        fdapde_strong_assert(std::isfinite(coefficient), std::domain_error, "nonfinite rotation tangent");
        angle = std::hypot(angle, coefficient);
    }
    fdapde_strong_assert(std::isfinite(angle), std::domain_error, "rotation tangent norm overflow");
    S real, factor;
    if (angle < std::sqrt(std::numeric_limits<S>::epsilon())) {
        const S squared = angle * angle;
        real = S(1) - squared / S(8);
        factor = S(0.5) - squared / S(48);
    } else {
        real = std::cos(angle / S(2));
        factor = std::sin(angle / S(2)) / angle;
    }
    return std::array<S, 4> {real, factor * vector[0], factor * vector[1], factor * vector[2]};
}

/// @brief evaluates quaternion rotation coefficients with scalar fixed-size arithmetic
template <typename S> auto so3_quaternion_matrix(const std::array<S, 4>& value) {
    const S w = value[0], x = value[1], y = value[2], z = value[3];
    Matrix<S, 3, 3> result;
    result(0, 0) = S(1) - S(2) * (y * y + z * z);
    result(0, 1) = S(2) * (x * y - w * z);
    result(0, 2) = S(2) * (x * z + w * y);
    result(1, 0) = S(2) * (x * y + w * z);
    result(1, 1) = S(1) - S(2) * (x * x + z * z);
    result(1, 2) = S(2) * (y * z - w * x);
    result(2, 0) = S(2) * (x * z - w * y);
    result(2, 1) = S(2) * (y * z + w * x);
    result(2, 2) = S(1) - S(2) * (x * x + y * y);
    return result;
}
}   // namespace internals
}   // namespace fdapde
#endif
