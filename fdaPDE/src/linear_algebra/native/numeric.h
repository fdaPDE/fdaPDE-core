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

#ifndef __FDAPDE_LINALG_NATIVE_NUMERIC_H__
#define __FDAPDE_LINALG_NATIVE_NUMERIC_H__

#include "header_check.h"

#include <cmath>
#include <limits>
#include <type_traits>

namespace fdapde::linalg::internals {

constexpr double scale_by_power_of_two(double value, int exponent) {
    if (!std::is_constant_evaluated()) return std::ldexp(value, exponent);
    if (value == 0.0 || value != value || value == std::numeric_limits<double>::infinity() ||
        value == -std::numeric_limits<double>::infinity()) {
        return value;
    }

    constexpr double overflow_limit = std::numeric_limits<double>::max() / 2.0;
    while (exponent > 0) {
        if (value > overflow_limit) return std::numeric_limits<double>::infinity();
        if (value < -overflow_limit) return -std::numeric_limits<double>::infinity();
        value *= 2.0;
        --exponent;
    }
    while (exponent < 0 && value != 0.0) {
        value *= 0.5;
        ++exponent;
    }
    return value;
}

constexpr double exp(double value) {
    if (!std::is_constant_evaluated()) return std::exp(value);

    constexpr double ln2 = 0.69314718055994530941723212145817656;
    constexpr double inverse_ln2 = 1.44269504088896340735992468100189214;
    if (value != value) return std::numeric_limits<double>::quiet_NaN();
    if (value > 709.782712893384) return std::numeric_limits<double>::infinity();
    if (value < -745.133219101941) return 0.0;

    const int exponent = static_cast<int>(value * inverse_ln2 + (value >= 0.0 ? 0.5 : -0.5));
    const double remainder = value - exponent * ln2;
    const double polynomial =
      ((((((1.38888888888888894189e-3 * remainder + 8.33333333333333321769e-3) * remainder +
            4.16666666666666643537e-2) *
             remainder +
           1.66666666666666657415e-1) *
            remainder +
          0.5) *
           remainder +
         1.0) *
          remainder +
        1.0);
    return scale_by_power_of_two(polynomial, exponent);
}

constexpr double normalized_fraction(double value, int& exponent) {
    if (!std::is_constant_evaluated()) return std::frexp(value, &exponent);
    if (value == 0.0) {
        exponent = 0;
        return 0.0;
    }
    const bool negative = value < 0.0;
    double fraction = negative ? -value : value;
    exponent = 0;
    while (fraction >= 1.0) {
        fraction *= 0.5;
        ++exponent;
    }
    while (fraction < 0.5) {
        fraction *= 2.0;
        --exponent;
    }
    return negative ? -fraction : fraction;
}

constexpr double log(double value) {
    if (!std::is_constant_evaluated()) return std::log(value);
    if (value < 0.0 || value != value) return std::numeric_limits<double>::quiet_NaN();
    if (value == 0.0) return -std::numeric_limits<double>::infinity();
    if (value == std::numeric_limits<double>::infinity()) return value;

    constexpr double ln2_high = 6.93147180369123816490e-01;
    constexpr double ln2_low = 1.90821492927058770002e-10;
    int exponent = 0;
    const double fraction = normalized_fraction(value, exponent);
    const double reduced = fraction - 1.0;
    const double scaled = reduced / (2.0 + reduced);
    const double square = scaled * scaled;
    const double remainder = square *
      (6.666666666666735130e-01 +
       square *
         (3.999999999940941908e-01 +
          square *
            (2.857142874366239149e-01 +
             square *
               (2.222219843214978396e-01 +
                square *
                  (1.818357216161805012e-01 +
                   square * (1.531383769920937332e-01 + square * 1.479819860511658591e-01))))));
    const double half_square = 0.5 * reduced * reduced;
    return exponent * ln2_high -
           ((half_square - (scaled * (half_square + remainder) + exponent * ln2_low)) - reduced);
}

}   // namespace fdapde::linalg::internals

#endif   // __FDAPDE_LINALG_NATIVE_NUMERIC_H__
