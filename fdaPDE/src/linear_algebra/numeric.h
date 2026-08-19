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

#ifndef __FDAPDE_LINALG_NUMERIC_H__
#define __FDAPDE_LINALG_NUMERIC_H__

#include "header_check.h"

#include <cmath>
#include <concepts>
#include <limits>
#include <type_traits>

namespace fdapde {
namespace internals {

template <std::floating_point Scalar> constexpr Scalar scale_safe_sqrt(Scalar value) {
    if (!std::is_constant_evaluated()) { return std::sqrt(value); }

    const Scalar infinity = std::numeric_limits<Scalar>::infinity();
    if (value == Scalar(0) || value == infinity) { return value; }
    if (!(value > Scalar(0))) { return std::numeric_limits<Scalar>::quiet_NaN(); }

    int scale = 0;
    while (value >= Scalar(4)) {
        value *= Scalar(0.25);
        ++scale;
    }
    while (value < Scalar(1)) {
        value *= Scalar(4);
        --scale;
    }

    Scalar result = fdapde::sqrt(value);
    while (scale > 0) {
        result *= Scalar(2);
        --scale;
    }
    while (scale < 0) {
        result *= Scalar(0.5);
        ++scale;
    }
    return result;
}

template <std::floating_point Scalar> constexpr Scalar scale_safe_hypot(Scalar lhs, Scalar rhs) {
    if (!std::is_constant_evaluated()) { return std::hypot(lhs, rhs); }

    Scalar large = lhs < Scalar(0) ? -lhs : lhs;
    Scalar small = rhs < Scalar(0) ? -rhs : rhs;
    const Scalar infinity = std::numeric_limits<Scalar>::infinity();
    if (large == infinity || small == infinity) { return infinity; }
    if (large != large || small != small) { return std::numeric_limits<Scalar>::quiet_NaN(); }
    if (large < small) {
        const Scalar tmp = large;
        large = small;
        small = tmp;
    }
    if (large == Scalar(0)) { return Scalar(0); }

    const Scalar ratio = small / large;
    const Scalar factor = scale_safe_sqrt(Scalar(1) + ratio * ratio);
    const Scalar half_max = std::numeric_limits<Scalar>::max() * Scalar(0.5);
    if (large <= half_max) { return large * factor; }

    const Scalar half_result = (large * Scalar(0.5)) * factor;
    if (half_result > half_max) { return infinity; }
    return half_result * Scalar(2);
}

}   // namespace internals
}   // namespace fdapde

#endif   // __FDAPDE_LINALG_NUMERIC_H__
