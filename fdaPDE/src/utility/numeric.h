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

#ifndef __FDAPDE_NUMERIC_H__
#define __FDAPDE_NUMERIC_H__

#include "header_check.h"

namespace fdapde {

/// @brief factorial of n
constexpr int factorial(const int n) {
    fdapde_assert(n >= 0, std::domain_error, "factorial is undefined for negative integers");
    int factorial_ = 1;
    if (n == 0) return factorial_;
    int m = n;
    while (m > 0) {
        factorial_ = m * factorial_;
        m--;
    }
    return factorial_;
}
/// @brief binomial coefficient n over m
constexpr int binomial_coefficient(const int n, const int m) {
    if (m == 0 || n == m) return 1;
    return factorial(n) / (factorial(m) * factorial(n - m));
}
/// @brief binomial_coefficient(n, k) x k matrix of combinations of k elements from a set of n
constexpr std::vector<int> combinations(int k, int n) {
    std::vector<bool> bitmask(k, 1);
    bitmask.resize(n, 0);
    std::vector<int> result(binomial_coefficient(n, k) * k);
    int j = 0;
    do {
        int l = 0;
        for (int i = 0; i < n; ++i) {
            if (bitmask[i]) {
                result[j * k + l] = i;
                l++;
            }
        }
        j++;
    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
    return result;
}

/// @brief integer division with round up
template <typename T1, typename T2>
    requires(internals::is_integer_v<T1> && internals::is_integer_v<T2>)
constexpr std::common_type_t<T1, T2> int_ceil(T1 a, T2 b) {
    return ((a ^ b) >= 0) ? a / b + (a % b != 0) : a / b;
}
/// @brief integer division with round down
template <typename T1, typename T2>
    requires(internals::is_integer_v<T1> && internals::is_integer_v<T2>)
constexpr std::common_type_t<T1, T2> int_floor(T1 a, T2 b) {
    return ((a ^ b) < 0 && a % b != 0) ? a / b - 1 : a / b;
}

/// @brief min function with common type conversion
template <typename T1, typename T2>
    requires requires(T1 a, T2 b) {
        (static_cast<std::common_type_t<T1, T2>>(a) <= static_cast<std::common_type_t<T1, T2>>(b)) ? 0 : 0;
    }
constexpr std::common_type_t<T1, T2> min(T1 a, T2 b) {
    using T = std::common_type_t<T1, T2>;
    return (static_cast<T>(a) <= static_cast<T>(b)) ? static_cast<T>(a) : static_cast<T>(b);
}
/// @brief max function with common type conversion
template <typename T1, typename T2>
    requires requires(T1 a, T2 b) {
        (static_cast<std::common_type_t<T1, T2>>(a) >= static_cast<std::common_type_t<T1, T2>>(b)) ? 0 : 0;
    }
constexpr std::common_type_t<T1, T2> max(T1 a, T2 b) {
    using T = std::common_type_t<T1, T2>;
    return (static_cast<T>(a) >= static_cast<T>(b)) ? static_cast<T>(a) : static_cast<T>(b);
}

/// @brief constexpr absolute value
template <typename T>
    requires(std::is_arithmetic_v<T>)
constexpr T abs(T x) {
    return x < 0 ? -x : x;
}
/// @brief returns the absolute floating-point value
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr T fabs(T x) {
    return x < 0 ? -x : x;
}

/// @brief test for floating point equality
[[maybe_unused]] constexpr double double_tolerance = 1e-10;
[[maybe_unused]] constexpr double machine_epsilon = 50 * std::numeric_limits<double>::epsilon();   // approx 10^-14
/// @brief compares values with absolute and relative tolerance
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr bool almost_equal(T a, T b, T epsilon) {
    return fdapde::fabs(a - b) < epsilon ||
           fdapde::fabs(a - b) < ((fdapde::fabs(a) < fdapde::fabs(b) ? fdapde::fabs(b) : fdapde::fabs(a)) * epsilon);
}
/// @brief compares values with absolute and relative tolerance
template <typename T> constexpr bool almost_equal(T a, T b) { return almost_equal(a, b, double_tolerance); }
/// @brief checks a non-strict difference against relative tolerance
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr bool greater_equal(T a, T b, T epsilon) {
    return (a - b) >= ((fdapde::fabs(a) < fdapde::fabs(b) ? fdapde::fabs(b) : fdapde::fabs(a)) * epsilon);
}
/// @brief checks a non-strict difference against relative tolerance
template <typename T> constexpr bool greater_equal(T a, T b) { return greater_equal(a, b, double_tolerance); }
/// @brief checks a reversed non-strict difference against relative tolerance
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr bool less_equal(T a, T b, T epsilon) {
    return (b - a) >= ((fdapde::fabs(a) < fdapde::fabs(b) ? fdapde::fabs(b) : fdapde::fabs(a)) * epsilon);
}
/// @brief checks a reversed non-strict difference against relative tolerance
template <typename T> constexpr bool less_equal(T a, T b) { return less_equal(a, b, double_tolerance); }
/// @brief checks a strict difference against relative tolerance
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr bool greater_than(T a, T b, T epsilon) {
    return (a - b) > ((fdapde::fabs(a) < fdapde::fabs(b) ? fdapde::fabs(b) : fdapde::fabs(a)) * epsilon);
}
/// @brief checks a strict difference against relative tolerance
template <typename T> constexpr bool greater_than(T a, T b) { return greater_than(a, b, double_tolerance); }
/// @brief checks a reversed strict difference against relative tolerance
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr bool less_than(T a, T b, T epsilon) {
    return (b - a) > ((fdapde::fabs(a) < fdapde::fabs(b) ? fdapde::fabs(b) : fdapde::fabs(a)) * epsilon);
}
/// @brief checks a reversed strict difference against relative tolerance
template <typename T> constexpr bool less_than(T a, T b) { return less_than(a, b, double_tolerance); }
/// @brief checks absolute magnitude against a tolerance
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr bool almost_zero(T a, T epsilon) {
    return fdapde::fabs(a) < epsilon;
}
/// @brief checks absolute magnitude against a tolerance
template <typename T> constexpr bool almost_zero(T a) { return almost_zero(a, machine_epsilon); }

/// @brief constexpr square root
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr T sqrt(T x) {
    if (x == T(0) || x == std::numeric_limits<T>::infinity()) return x;
    if (!(x > T(0))) return std::numeric_limits<T>::quiet_NaN();
    if (!std::is_constant_evaluated()) return std::sqrt(x);

    // scale into [1,4) so Heron's iteration is safe at both ends of the floating-point range
    T scale = T(1);
    while (x < T(1)) {
        x *= T(4);
        scale *= T(0.5);
    }
    while (x >= T(4)) {
        x *= T(0.25);
        scale *= T(2);
    }
    T curr = x, prev = T(0);
    while (fdapde::abs(curr - prev) > std::numeric_limits<T>::epsilon() * curr) {
        prev = curr;
        curr = T(0.5) * (curr + x / curr);
    }
    return curr * scale;
}

/// @brief constexpr ceil
template <typename T> constexpr std::conditional_t<std::is_floating_point_v<T>, T, double> ceil(T x) {
    long int int_part = static_cast<long int>(x);
    return (x > 0.0 && x != static_cast<T>(int_part)) ? int_part + 1.0 : int_part;
}
/// @brief constexpr floor
template <typename T> constexpr std::conditional_t<std::is_floating_point_v<T>, T, double> floor(T x) {
    long int int_part = static_cast<long int>(x);
    return (x < 0.0 && x != static_cast<T>(int_part)) ? int_part - 1.0 : int_part;
}

/// @brief returns one for nonnegative inputs and zero otherwise
template <typename T> constexpr int sign(T x) { return x >= 0 ? 1 : 0; }

/// @brief constexpr pow, only integer exponent support
template <typename BaseT, typename ExpT>
    requires(std::is_floating_point_v<BaseT> && internals::is_integer_v<ExpT>)
constexpr BaseT pow(BaseT base, ExpT exp) {
    if (exp == 0) return BaseT {1};
    using UnsignedExp = std::make_unsigned_t<std::remove_cvref_t<ExpT>>;
    const UnsignedExp unsigned_exp = static_cast<UnsignedExp>(exp);
    UnsignedExp abs_exp = exp < 0 ? UnsignedExp {0} - unsigned_exp : unsigned_exp;
    BaseT result = BaseT {1};
    while (abs_exp > 0) {
        if (abs_exp % 2 == 1) result *= base;
        base *= base;
        abs_exp /= 2;
    }
    return exp < 0 ? BaseT {1} / result : result;
}

/// @brief constexpr ldexp, computes num * 2^exp
template <typename BaseT, typename ExpT>
    requires(std::is_floating_point_v<BaseT> && internals::is_integer_v<ExpT>)
constexpr BaseT ldexp(BaseT num, ExpT exp) {
    using Limits = std::numeric_limits<BaseT>;
    if (num == BaseT(0) || num != num || fdapde::abs(num) == Limits::infinity()) return num;
    constexpr int exponent_bound = Limits::max_exponent - Limits::min_exponent + Limits::digits;
    if (std::cmp_greater(+exp, exponent_bound)) return num < 0 ? -Limits::infinity() : Limits::infinity();
    if (std::cmp_less(+exp, -exponent_bound)) return num < 0 ? -BaseT(0) : BaseT(0);
    int exponent = static_cast<int>(exp);
    if (!std::is_constant_evaluated()) return std::ldexp(num, exponent);

    // normalize exactly before scaling to avoid intermediate overflow and repeated subnormal rounding
    while (fdapde::abs(num) >= BaseT(1)) {
        num *= BaseT(0.5);
        ++exponent;
    }
    while (fdapde::abs(num) < BaseT(0.5)) {
        num *= BaseT(2);
        --exponent;
    }
    if (exponent > Limits::max_exponent) return num < 0 ? -Limits::infinity() : Limits::infinity();
    if (exponent < Limits::min_exponent - Limits::digits) return num < 0 ? -BaseT(0) : BaseT(0);
    BaseT scale = BaseT(1);
    if (exponent < Limits::min_exponent) {
        // form subnormal results with a single final rounding at the minimum normal scale
        for (int i = exponent; i < Limits::min_exponent - 1; ++i) num *= BaseT(0.5);
        return num * Limits::min();
    }
    if (exponent > 0) {
        for (int i = 1; i < exponent; ++i) scale *= BaseT(2);
        return (num * BaseT(2)) * scale;
    }
    for (int i = 0; i > exponent; --i) scale *= BaseT(0.5);
    return num * scale;
}

/// @brief approximates the exponential with a reduced Taylor polynomial
constexpr double exp(double x) {
    constexpr double ln2 = 0.69314718055994530941723212145817656;      // ln(2)
    constexpr double invln2 = 1.44269504088896340735992468100189214;   // 1/ln(2)
    // polynomial coefficients for the reduced exponential Taylor expansion
    constexpr double C1 = 1.0;
    constexpr double C2 = 1.0;                         // 1
    constexpr double C3 = 0.5;                         // 1/2
    constexpr double C4 = 1.66666666666666657415e-1;   // 1/6
    constexpr double C5 = 4.16666666666666643537e-2;   // 1/24
    constexpr double C6 = 8.33333333333333321769e-3;   // 1/120
    constexpr double C7 = 1.38888888888888894189e-3;   // 1/720

    if (x != x) return std::numeric_limits<double>::quiet_NaN();
    if (x > 709.782712893384) return std::numeric_limits<double>::infinity();
    if (x < -745.133219101941) return 0.0;
    // reduction
    int k = static_cast<int>(x * invln2 + sign(x) * 0.5);
    double r = x - k * ln2;
    // evaluate the Taylor polynomial using Horner's method
    double R = ((((((C7 * r + C6) * r + C5) * r + C4) * r + C3) * r + C2) * r + C1);
    return ldexp(R, k);
};

/// @brief constexpr ilogb: integer log2 of |x|: floor(\log_2(|x|))
constexpr int ilogb(double x) {
    if (x == 0.0) return std::numeric_limits<int>::min();   // fP_ILOGB0
    if (x != x) return std::numeric_limits<int>::max();     // fP_ILOGBNAN

    std::uint64_t bits = std::bit_cast<std::uint64_t>(x < 0 ? -x : x);   // recover bit expression
    int exp = static_cast<int>((bits >> 52) & 0x7FF);
    if (exp == 0) {   // subnormal
        int shift = 0;
        while ((bits & (1ULL << 52)) == 0) {
            bits <<= 1;
            ++shift;
        }
        return -1022 - shift + 1;
    }
    return exp - 1023;   // remove IEEE bias
}

/// @brief constexpr frexp: x = fraction * 2^out, with |fraction| in [0.5,1) for finite nonzero x
constexpr double frexp(double x, int& out) {
    out = 0;
    if (x == 0.0 || x != x || fdapde::abs(x) == std::numeric_limits<double>::infinity()) return x;
    if (!std::is_constant_evaluated()) return std::frexp(x, &out);
    while (fdapde::abs(x) >= 1.0) {
        x *= 0.5;
        ++out;
    }
    while (fdapde::abs(x) < 0.5) {
        x *= 2.0;
        --out;
    }
    return x;
}

/// @brief approximates the natural logarithm with a polynomial inspired by fdlibm
constexpr double log(double x) {
    // constants split for accuracy
    constexpr double ln2_hi = 6.93147180369123816490e-01;
    constexpr double ln2_lo = 1.90821492927058770002e-10;
    // polynomial coefficients (credits: fdlibm)
    constexpr double Lg1 = 6.666666666666735130e-01;
    constexpr double Lg2 = 3.999999999940941908e-01;
    constexpr double Lg3 = 2.857142874366239149e-01;
    constexpr double Lg4 = 2.222219843214978396e-01;
    constexpr double Lg5 = 1.818357216161805012e-01;
    constexpr double Lg6 = 1.531383769920937332e-01;
    constexpr double Lg7 = 1.479819860511658591e-01;

    if (x < 0.0 || x != x) { return std::numeric_limits<double>::quiet_NaN(); }
    if (x == 0.0) { return -std::numeric_limits<double>::infinity(); }
    if (x == std::numeric_limits<double>::infinity()) { return x; }
    // decompose
    int k;
    double m = fdapde::frexp(x, k);   // x = m * 2^k, m in [0.5,1)
    // range reduction
    double f = m - 1.0;   // in [-0.5,0)
    double s = f / (2.0 + f);
    double z = s * s;
    // evaluate the reduced logarithm polynomial using Horner's method
    double R = z * (Lg1 + z * (Lg2 + z * (Lg3 + z * (Lg4 + z * (Lg5 + z * (Lg6 + z * Lg7))))));
    double hfsq = 0.5 * f * f;
    return k * ln2_hi - ((hfsq - (s * (hfsq + R) + k * ln2_lo)) - f);
}

/// @brief constexpr log1p
constexpr double log1p(double x) {
    if (x == 0.0) { return 0.0; }
    if (x == -1.0) { return -std::numeric_limits<double>::infinity(); }   // log(0)
    if (x < -1.0) { return std::numeric_limits<double>::quiet_NaN(); }    // log of negative value

    // for small x, use series expansion
    if (x > -1e-8 && x < 1e-8) {
        double x2 = x * x;
        double x3 = x2 * x;
        double x4 = x3 * x;
        return x - 0.5 * x2 + x3 / 3.0 - x4 / 4.0;
    }
    return fdapde::log(1.0 + x);
}

/// @brief computes log(1 + exp(x)) with range-dependent approximations to avoid overflow
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr T log1pexp(T x) {
    if (x <= -37.0) return fdapde::exp(x);
    if (x <= 18.0) return fdapde::log1p(fdapde::exp(x));
    if (x > 33.3) return x;
    return x + fdapde::exp(-x);
}

}   // namespace fdapde

#endif   // __FDAPDE_NUMERIC_H__
