// this file is part of fdapde, a c++ library for physics-informed
// spatial and functional data analysis
//
// this program is free software: you can redistribute it and/or modify
// it under the terms of the gnu general public license as published by
// the free software foundation, either version 3 of the license, or
// (at your option) any later version
//
// this program is distributed in the hope that it will be useful,
// but without any warranty; without even the implied warranty of
// merchantability or fitness for a particular purpose.  see the
// gnu general public license for more details
//
// you should have received a copy of the gnu general public license
// along with this program.  if not, see <http://www.gnu.org/licenses/>

#ifndef __FDAPDE_NUMERIC_H__
#define __FDAPDE_NUMERIC_H__

#include "header_check.h"

namespace fdapde {

[[maybe_unused]] constexpr float  flt_tolerance = 1e-5f;
[[maybe_unused]] constexpr double dbl_tolerance = 1e-10;

namespace internals {

// ieee binary32 and binary64 representation
template <typename T>
    requires(std::is_floating_point_v<T>)
struct floating_point_bits {
    using uint_t = std::conditional_t<std::is_same_v<T, float>, std::uint32_t, std::uint64_t>;
    static_assert(std::numeric_limits<T>::is_iec559 && sizeof(T) == sizeof(uint_t));
    static constexpr int fraction_bits    = std::numeric_limits<T>::digits - 1;
    static constexpr int exponent_bias    = std::numeric_limits<T>::max_exponent - 1;
    static constexpr int exponent_mask    = 2 * exponent_bias + 1;
    static constexpr int leading_bits     = std::numeric_limits<uint_t>::digits - fraction_bits - 1;
    static constexpr uint_t sign_mask     = uint_t(1) << (std::numeric_limits<uint_t>::digits - 1);
    static constexpr uint_t fraction_mask = (uint_t(1) << fraction_bits) - 1;
};

// selects a tolerance in the operand type
template <typename T>
inline constexpr T numeric_tolerance = [] { return std::is_same_v<T, float> ? flt_tolerance : dbl_tolerance; }();

}   // namespace internals

// computes n factorial using unsigned 64-bit arithmetic
constexpr std::uint64_t fact(const int n) {
    fdapde_assert(n >= 0, std::domain_error, "factorial not defined for negative arguments");
    std::uint64_t result = 1;
    for (int i = n; i > 1; --i) { result *= static_cast<std::uint64_t>(i); }
    return result;
}
// computes the binomial coefficient n over m assuming the result fits in uint64_t
constexpr std::uint64_t binom(const int n, const int m) {
    fdapde_assert(n >= 0 && m >= 0 && m <= n, std::domain_error, "binom(n, m) defined for 0 <= m <= n");
    const int k = m < n - m ? m : n - m;
    std::uint64_t result = 1;
    for (int i = 1; i <= k; ++i) {
        const std::uint64_t divisor = std::gcd(result, static_cast<std::uint64_t>(i));
        const std::uint64_t reduced = result / divisor;
        const std::uint64_t numerator = static_cast<std::uint64_t>(n - k + i) / (static_cast<std::uint64_t>(i) / divisor);
        result = reduced * numerator;
    }
    return result;
}
// lists the k-element combinations of n elements in a flattened row-major array
constexpr std::vector<std::uint64_t> comb(int k, int n) {
    fdapde_assert(n >= 0 && k >= 0 && k <= n, std::domain_error, "comb(k, n) require 0 <= k <= n");
    std::vector<bool> bitmask(k, true);
    bitmask.resize(n, false);
    const auto count = static_cast<std::size_t>(binom(n, k));
    std::vector<std::uint64_t> result(count * static_cast<std::size_t>(k));
    std::size_t offset = 0;
    do {
        for (int i = 0; i < n; ++i) {
            if (bitmask[i]) result[offset++] = static_cast<std::uint64_t>(i);
        }
    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
    return result;
}

// integer division with round up
template <typename T1, typename T2>
    requires(internals::is_integer_v<T1> && internals::is_integer_v<T2>)
constexpr std::common_type_t<T1, T2> idiv_ceil(T1 a, T2 b) {
    return ((a ^ b) >= 0) ? a / b + (a % b != 0) : a / b;
}
// integer division with round down
template <typename T1, typename T2>
    requires(internals::is_integer_v<T1> && internals::is_integer_v<T2>)
constexpr std::common_type_t<T1, T2> idiv_floor(T1 a, T2 b) {
    return ((a ^ b) < 0 && a % b != 0) ? a / b - 1 : a / b;
}

// min function with common type conversion
template <typename T1, typename T2>
    requires requires(T1 a, T2 b) {
        (static_cast<std::common_type_t<T1, T2>>(a) <= static_cast<std::common_type_t<T1, T2>>(b)) ? 0 : 0;
    }
constexpr std::common_type_t<T1, T2> min(T1 a, T2 b) {
    using T = std::common_type_t<T1, T2>;
    return (static_cast<T>(a) <= static_cast<T>(b)) ? static_cast<T>(a) : static_cast<T>(b);
}
// max function with common type conversion
template <typename T1, typename T2>
    requires requires(T1 a, T2 b) {
        (static_cast<std::common_type_t<T1, T2>>(a) >= static_cast<std::common_type_t<T1, T2>>(b)) ? 0 : 0;
    }
constexpr std::common_type_t<T1, T2> max(T1 a, T2 b) {
    using T = std::common_type_t<T1, T2>;
    return (static_cast<T>(a) >= static_cast<T>(b)) ? static_cast<T>(a) : static_cast<T>(b);
}

// computes the floating-point absolute value
template <typename T>
    requires(std::is_same_v<T, float> || std::is_same_v<T, double>)
constexpr T fabs(T x) {
    if (!std::is_constant_evaluated()) return std::fabs(x);
    using bits_t = internals::floating_point_bits<T>;
    using uint_t = typename bits_t::uint_t;
    return std::bit_cast<T>(std::bit_cast<uint_t>(x) & ~bits_t::sign_mask);
}

// computes the absolute value when it is representable in the result type
template <typename T>
    requires(std::is_arithmetic_v<T>)
constexpr T abs(T x) {
    if constexpr (std::is_floating_point_v<T>) {
        return fdapde::fabs(x);
    } else {
        return x < 0 ? -x : x;
    }
}

// compares a and b using absolute and relative tolerance, including exact equality
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr bool almost_equal(T a, T b, T epsilon) {
    fdapde_assert(epsilon >= 0, std::invalid_argument, "tolerance must be nonnegative");
    if (a == b) return true;
    const T aa = fdapde::fabs(a), bb = fdapde::fabs(b);
    if (a != a || b != b || aa == std::numeric_limits<T>::infinity() || bb == std::numeric_limits<T>::infinity())
        return false;
    const T scale = aa > bb ? aa : bb;
    // normalization avoids overflow when subtracting large values of opposite signs
    if ((a < T(0)) != (b < T(0))) {
        return (aa < epsilon && bb < epsilon - aa) || (scale != T(0) && aa / scale + bb / scale < epsilon);
    }
    const T difference = fdapde::fabs(a - b);
    return difference < epsilon || (scale != T(0) && difference / scale < epsilon);
}
// compares a and b using the tolerance for their scalar type
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr bool almost_equal(T a, T b) {
    return almost_equal(a, b, internals::numeric_tolerance<T>);
}
// tests whether a is greater than b or equal within tolerance
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr bool greater_equal(T a, T b, T epsilon) {
    fdapde_assert(epsilon >= 0, std::invalid_argument, "tolerance must be nonnegative");
    return a > b || almost_equal(a, b, epsilon);
}
// tests whether a is greater than b or equal within the default tolerance
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr bool greater_equal(T a, T b) {
    return greater_equal(a, b, internals::numeric_tolerance<T>);
}
// tests whether a is less than b or equal within tolerance
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr bool less_equal(T a, T b, T epsilon) {
    fdapde_assert(epsilon >= 0, std::invalid_argument, "tolerance must be nonnegative");
    return a < b || almost_equal(a, b, epsilon);
}
// tests whether a is less than b or equal within the default tolerance
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr bool less_equal(T a, T b) { return less_equal(a, b, internals::numeric_tolerance<T>); }
// tests whether a exceeds b beyond the tolerance
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr bool greater_than(T a, T b, T epsilon) {
    fdapde_assert(epsilon >= T(0), std::invalid_argument, "tolerance must be nonnegative");
    return a > b && !almost_equal(a, b, epsilon);
}
// tests whether a exceeds b beyond the default tolerance
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr bool greater_than(T a, T b) {
    return greater_than(a, b, internals::numeric_tolerance<T>);
}
// tests whether a is below b beyond the tolerance
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr bool less_than(T a, T b, T epsilon) {
    fdapde_assert(epsilon >= 0, std::invalid_argument, "tolerance must be nonnegative");
    return a < b && !almost_equal(a, b, epsilon);
}
// tests whether a is below b beyond the default tolerance
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr bool less_than(T a, T b) { return less_than(a, b, internals::numeric_tolerance<T>); }
// tests whether the magnitude of a is below the tolerance, including zero
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr bool almost_zero(T a, T epsilon) {
    fdapde_assert(epsilon >= 0, std::invalid_argument, "tolerance must be nonnegative");
    return a == T(0) || fdapde::fabs(a) < epsilon;
}
// tests whether a is near zero using a tolerance based on its scalar precision
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr bool almost_zero(T a) { return almost_zero(a, 50 * std::numeric_limits<T>::epsilon()); }

// computes the square root of x
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr T sqrt(T x) {
    fdapde_assert(!(x < 0), std::domain_error, "sqrt not defined for negative arguments");
    if (!std::is_constant_evaluated()) return std::sqrt(x);
    if (x < 0 || x != x) return std::numeric_limits<T>::quiet_NaN();
    if (x == 0 || x == std::numeric_limits<T>::infinity()) return x;
    // scale into [1,4) so heron's iteration is safe at both ends of the floating-point range
    T scale = 1;
    while (x < 1) {
        x *= 4;
        scale *= 0.5;
    }
    while (x >= 4) {
        x *= 0.25;
        scale *= 2;
    }
    T curr = x, prev = 0;
    while (fdapde::abs(curr - prev) > std::numeric_limits<T>::epsilon() * curr) {
        prev = curr;
        curr = 0.5 * (curr + x / curr);
    }
    return curr * scale;
}

// rounds x toward positive infinity when its integral part fits in long
template <typename T> constexpr std::conditional_t<std::is_floating_point_v<T>, T, double> ceil(T x) {
    long int int_part = static_cast<long int>(x);
    return (x > 0.0 && x != static_cast<T>(int_part)) ? int_part + 1.0 : int_part;
}
// rounds x toward negative infinity when its integral part fits in long
template <typename T> constexpr std::conditional_t<std::is_floating_point_v<T>, T, double> floor(T x) {
    long int int_part = static_cast<long int>(x);
    return (x < 0.0 && x != static_cast<T>(int_part)) ? int_part - 1.0 : int_part;
}

// returns -1, 0, or 1 according to the sign of x
template <typename T> constexpr int sign(T x) { return (T(0) < x) - (x < T(0)); }

// computes base raised to an integer exponent without reciprocating an overflowing positive power
template <typename BaseT, typename ExpT>
    requires(std::is_floating_point_v<BaseT> && internals::is_integer_v<ExpT>)
constexpr BaseT pow(BaseT base, ExpT exp) {
    if (exp == 0) return 1;
    using uexp_t = std::make_unsigned_t<std::remove_cvref_t<ExpT>>;
    const uexp_t uexp = static_cast<uexp_t>(exp);
    uexp_t abs_exp = exp < 0 ? uexp_t(0) - uexp : uexp;
    if (exp < 0) base = BaseT(1) / base;
    BaseT result = 1;
    while (abs_exp > 0) {
        if (abs_exp & 1) result *= base;
        abs_exp >>= 1;
        if (abs_exp != 0) base *= base;
    }
    return result;
}

// computes num * 2^exp
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr T ldexp(T num, int exp) {
    if (!std::is_constant_evaluated()) return std::ldexp(num, exp);
    using bits_t = internals::floating_point_bits<T>;
    using uint_t = typename bits_t::uint_t;
    const auto representation = std::bit_cast<uint_t>(num);
    const auto sign = representation & bits_t::sign_mask;
    auto mantissa = representation & bits_t::fraction_mask;
    int exponent = static_cast<int>((representation >> bits_t::fraction_bits) & bits_t::exponent_mask);
    if (exponent == bits_t::exponent_mask || (exponent == 0 && mantissa == 0)) return num;
    if (exponent == 0) {
        const int shift = std::countl_zero(mantissa) - bits_t::leading_bits;
        mantissa <<= shift;
        exponent = 1 - shift;
    } else {
        mantissa |= uint_t(1) << bits_t::fraction_bits;
    }
    // compare before adding so even int_min and int_max exponents cannot overflow
    if (exp > bits_t::exponent_mask - 1 - exponent)
        return std::bit_cast<T>(sign | (uint_t(bits_t::exponent_mask) << bits_t::fraction_bits));
    if (exp < -bits_t::fraction_bits - exponent) return std::bit_cast<T>(sign);
    exponent += exp;
    if (exponent > 0) {
        return std::bit_cast<T>(sign | (uint_t(exponent) << bits_t::fraction_bits) | (mantissa & bits_t::fraction_mask));
    }
    // shift into the subnormal range once, retaining discarded bits for correct rounding
    const int shift = 1 - exponent;
    const auto remainder = mantissa & ((uint_t(1) << shift) - 1);
    const auto halfway = uint_t(1) << (shift - 1);
    mantissa >>= shift;
    if (remainder > halfway || (remainder == halfway && (mantissa & 1))) ++mantissa;
    // rounding can carry into the smallest normal number, whose encoding is also valid here
    return std::bit_cast<T>(sign | mantissa);
}

// computes the exponential of x
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr T exp(T x) {
    if (!std::is_constant_evaluated()) return std::exp(x);
    if (x != x) return x;
    constexpr T overflow_limit  = std::is_same_v<T, float> ? T( 88.72283905206835) : T( 709.782712893384);
    constexpr T underflow_limit = std::is_same_v<T, float> ? T(-103.9720770839918) : T(-745.133219101941);
    if (x > overflow_limit ) return std::numeric_limits<T>::infinity();
    if (x < underflow_limit) return 0;
    constexpr T ln2_hi = std::is_same_v<T, float> ? T(6.9313812255859375e-01) : T(6.93147180369123816490e-01);
    constexpr T ln2_lo = std::is_same_v<T, float> ? T(9.0580006145387890e-06) : T(1.90821492927058770002e-10);
    const int k = static_cast<int>(x * T(1.44269504088896340736) + T(sign(x)) * T(0.5));
    const T r = (x - T(k) * ln2_hi) - T(k) * ln2_lo;
    T result = 1, term = 1;
    for (int i = 1; i <= 18; ++i) {
        term *= r / T(i);
        result += term;
    }
    return fdapde::ldexp(result, k);
}

// computes floor(log2(|x|))
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr int ilogb(T x) {
    if (!std::is_constant_evaluated()) return std::ilogb(x);
    using bits_t = internals::floating_point_bits<T>;
    using uint_t = typename bits_t::uint_t;
    const auto bits = std::bit_cast<uint_t>(x);
    const auto mantissa = bits & bits_t::fraction_mask;
    const int exponent = static_cast<int>((bits >> bits_t::fraction_bits) & bits_t::exponent_mask);
    if (exponent == bits_t::exponent_mask) return mantissa == 0 ? std::numeric_limits<int>::max() : FP_ILOGBNAN;
    if (exponent == 0) {
        if (mantissa == 0) return FP_ILOGB0;
        const int shift = std::countl_zero(mantissa) - bits_t::leading_bits;
        return 1 - bits_t::exponent_bias - shift;
    }
    return exponent - bits_t::exponent_bias;
}

// decomposes num into a signed fraction and a binary exponent
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr T frexp(T num, int& out) {
    if (!std::is_constant_evaluated()) return std::frexp(num, &out);
    using bits_t = internals::floating_point_bits<T>;
    using uint_t = typename bits_t::uint_t;
    const auto bits = std::bit_cast<uint_t>(num);
    auto mantissa = bits & bits_t::fraction_mask;
    int exponent = static_cast<int>((bits >> bits_t::fraction_bits) & bits_t::exponent_mask);
    out = 0;
    if (exponent == bits_t::exponent_mask || (exponent == 0 && mantissa == 0)) return num;
    if (exponent == 0) {
        const int shift = std::countl_zero(mantissa) - bits_t::leading_bits;
        mantissa = (mantissa << shift) & bits_t::fraction_mask;
        exponent = 1 - shift;
    }
    out = exponent - (bits_t::exponent_bias - 1);
    return std::bit_cast<T>(
      (bits & bits_t::sign_mask) | (uint_t(bits_t::exponent_bias - 1) << bits_t::fraction_bits) | mantissa);
}

// computes the natural logarithm of x
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr T log(T x) {
    fdapde_assert(!(x < 0), std::domain_error, "logarithm not defined for negative arguments");
    if (!std::is_constant_evaluated()) return std::log(x);

    if (x < 0 || x != x) return std::numeric_limits<T>::quiet_NaN();
    if (x == 0) return -std::numeric_limits<T>::infinity();
    if (x == std::numeric_limits<T>::infinity()) return x;
    int k = 0;
    T m = fdapde::frexp(x, k);
    if (m < T(0.70710678118654752440)) {
        m *= T(2);
        --k;
    }
    const T z = (m - T(1)) / (m + T(1)), z2 = z * z;
    T term = z, sum = z;
    for (int i = 3; i <= 49; i += 2) {
        term *= z2;
        sum += term / T(i);
    }
    constexpr T ln2_hi = std::is_same_v<T, float> ? T(6.9313812255859375e-01) : T(6.93147180369123816490e-01);
    constexpr T ln2_lo = std::is_same_v<T, float> ? T(9.0580006145387890e-06) : T(1.90821492927058770002e-10);
    return T(k) * ln2_hi + (T(2) * sum + T(k) * ln2_lo);
}

// computes the natural logarithm of 1 + x without cancellation near zero
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr T log1p(T x) {
    fdapde_assert(x >= -1, std::domain_error, "log1p(x) requires x >= -1");
    if (!std::is_constant_evaluated()) return std::log1p(x);

    if (x == 0 || x != x) return x;
    if (fdapde::fabs(x) < std::numeric_limits<T>::epsilon()) return x;
    if (x < -1) return std::numeric_limits<T>::quiet_NaN();
    if (x == -1) return -std::numeric_limits<T>::infinity();
    if (x > -0.5 && x < 0.5) {
        const T z = x / (2 + x), z2 = z * z;
        T term = z, sum = z;
        for (int i = 3; i <= 69; i += 2) {
            term *= z2;
            sum += term / T(i);
        }
        return 2 * sum;
    }
    const T u = 1 + x;
    if (u == std::numeric_limits<T>::infinity()) return u;
    return fdapde::log(u) - ((u - 1) - x) / u;
}

// computes log(1 + exp(x)) using range-dependent approximations
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr T log1pexp(T x) {
    if (x <= -37.0) return fdapde::exp(x);
    if (x <=  18.0) return fdapde::log1p(fdapde::exp(x));
    if (x >   33.3) return x;
    return x + fdapde::exp(-x);
}

}   // namespace fdapde

#endif   // __fdapde_numeric_h__
