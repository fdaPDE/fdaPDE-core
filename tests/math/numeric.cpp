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

#include <fdaPDE/math.h>
#include <gtest/gtest.h>

#include <array>
#include <climits>
#include <type_traits>

namespace {

template <typename T> void check_numeric_scalar() {
    using limits = std::numeric_limits<T>;
    using uint_t = std::conditional_t<std::is_same_v<T, float>, std::uint32_t, std::uint64_t>;
    const auto bits = [](T x) { return std::bit_cast<uint_t>(x); };
    constexpr T infinity = limits::infinity();
    constexpr T nan = limits::quiet_NaN();
    constexpr T smallest = limits::denorm_min();
    constexpr int smallest_exponent = limits::min_exponent - limits::digits;

    // transcendental functions preserve the scalar type and ilogb returns int
    static_assert(std::is_same_v<decltype(fdapde::exp(T(1))), T>);
    static_assert(std::is_same_v<decltype(fdapde::log(T(1))), T>);
    static_assert(std::is_same_v<decltype(fdapde::log1p(T(1))), T>);
    static_assert(std::is_same_v<decltype(fdapde::ilogb(T(1))), int>);
    // absolute value clears the sign of negative values and negative zero
    static_assert(fdapde::fabs(T(-2)) == T(2));
    static_assert(std::bit_cast<uint_t>(fdapde::fabs(-T(0))) == 0);
    // square root accepts zero and preserves positive infinity
    static_assert(fdapde::sqrt(T(0)) == T(0));
    static_assert(fdapde::sqrt(infinity) == infinity);
    // integer powers handle negative and wide exponents, subnormal results, and zero to zero
    static_assert(fdapde::pow(T(2), -3) == T(0.125));
    static_assert(fdapde::pow(T(2), smallest_exponent) == smallest);
    static_assert(fdapde::pow(T(-1), LLONG_MIN) == T(1));
    static_assert(fdapde::pow(T(1), ULLONG_MAX) == T(1));
    static_assert(fdapde::pow(T(0), 0) == T(1));
    // exponential handles zero and both infinities during constant evaluation
    static_assert(fdapde::exp(T(0)) == T(1));
    static_assert(fdapde::exp(-infinity) == T(0));
    static_assert(fdapde::exp(infinity) == infinity);
    // logarithm handles one, zero, and positive infinity during constant evaluation
    static_assert(fdapde::log(T(1)) == T(0));
    static_assert(fdapde::log(T(0)) == -infinity);
    static_assert(fdapde::log(infinity) == infinity);
    // log1p preserves tiny inputs and returns negative infinity at its domain boundary
    static_assert(fdapde::log1p(T(-1)) == -infinity);
    static_assert(fdapde::log1p(smallest) == smallest);
    // ilogb handles zero, nan, infinity, subnormal inputs, and negative values
    static_assert(fdapde::ilogb(T(0)) == FP_ILOGB0);
    static_assert(fdapde::ilogb(nan) == FP_ILOGBNAN);
    static_assert(fdapde::ilogb(infinity) == INT_MAX);
    static_assert(fdapde::ilogb(smallest) == smallest_exponent);
    static_assert(fdapde::ilogb(T(-8)) == 3);
    // log1p preserves the sign of negative zero
    static_assert(std::bit_cast<uint_t>(fdapde::log1p(-T(0))) == std::bit_cast<uint_t>(-T(0)));

    // these arrays force the custom paths to execute during constant evaluation
    constexpr std::array<T, 12> positive = {smallest, limits::min(), T(1e-20), T(0.125), T(0.5),  T(0.707),
                                            T(1),     T(1.001),      T(2),     T(10),    T(1e20), limits::max()};
    constexpr auto roots = [=] {
        std::array<T, positive.size()> result {};
        for (std::size_t i = 0; i < positive.size(); ++i) result[i] = fdapde::sqrt(positive[i]);
        return result;
    }();
    constexpr auto logarithms = [=] {
        std::array<T, positive.size()> result {};
        for (std::size_t i = 0; i < positive.size(); ++i) result[i] = fdapde::log(positive[i]);
        return result;
    }();
    // constexpr roots and logarithms agree with std across the representable range
    for (std::size_t i = 0; i < positive.size(); ++i) {
        const T root = std::sqrt(positive[i]);
        EXPECT_NEAR(roots[i], root, T(8) * limits::epsilon() * root);
        const T logarithm = std::log(positive[i]);
        EXPECT_NEAR(logarithms[i], logarithm, T(8) * limits::epsilon() * std::max(T(1), std::fabs(logarithm)));
        // volatile inputs exercise runtime dispatch for sqrt, log, and ilogb
        volatile T runtime = positive[i];
        EXPECT_EQ(fdapde::sqrt(T(runtime)), root);
        EXPECT_EQ(fdapde::log(T(runtime)), logarithm);
        EXPECT_EQ(fdapde::ilogb(T(runtime)), std::ilogb(T(runtime)));
    }

    // constexpr and runtime exponentials agree with std for small and large arguments
    constexpr std::array<T, 10> exponent_inputs = {T(-103),    T(-37), T(-1), T(-0.34657), T(0),
                                                   T(0.34657), T(1),   T(18), T(33.3),     T(88)};
    constexpr auto exponentials = [=] {
        std::array<T, exponent_inputs.size()> result {};
        for (std::size_t i = 0; i < exponent_inputs.size(); ++i) result[i] = fdapde::exp(exponent_inputs[i]);
        return result;
    }();
    for (std::size_t i = 0; i < exponent_inputs.size(); ++i) {
        const T expected = std::exp(exponent_inputs[i]);
        EXPECT_NEAR(exponentials[i], expected, std::max(smallest, T(8) * limits::epsilon() * expected));
        volatile T runtime = exponent_inputs[i];
        EXPECT_EQ(fdapde::exp(T(runtime)), expected);
    }
    // log1p agrees with std near zero, near minus one, and for large inputs
    constexpr std::array<T, 10> log1p_inputs = {T(-0.999), T(-0.5), T(-0.125), T(-1e-8), -T(0),
                                                smallest,  T(1e-8), T(0.125),  T(0.5),   limits::max()};
    constexpr auto log1ps = [=] {
        std::array<T, log1p_inputs.size()> result {};
        for (std::size_t i = 0; i < log1p_inputs.size(); ++i) result[i] = fdapde::log1p(log1p_inputs[i]);
        return result;
    }();
    for (std::size_t i = 0; i < log1p_inputs.size(); ++i) {
        const T expected = std::log1p(log1p_inputs[i]);
        EXPECT_NEAR(log1ps[i], expected, T(8) * limits::epsilon() * std::max(T(1), std::fabs(expected)));
        volatile T runtime = log1p_inputs[i];
        EXPECT_EQ(bits(fdapde::log1p(T(runtime))), bits(expected));
    }

    // scaling handles subnormal underflow, ties-to-even rounding, and carry into the normal range
    static_assert(fdapde::ldexp(T(1), smallest_exponent) == smallest);
    static_assert(fdapde::ldexp(T(1), smallest_exponent - 1) == T(0));
    static_assert(fdapde::ldexp(T(1.5), smallest_exponent) == T(2) * smallest);
    static_assert(fdapde::ldexp(T(2.5), smallest_exponent) == T(2) * smallest);
    static_assert(fdapde::ldexp(T(3.5), smallest_exponent) == T(4) * smallest);
    static_assert(fdapde::ldexp(T(2) - limits::epsilon(), limits::min_exponent - 2) == limits::min());
    // extreme exponents saturate to infinity or zero while preserving signed zero
    static_assert(fdapde::ldexp(T(1), INT_MAX) == infinity);
    static_assert(fdapde::ldexp(T(-1), INT_MIN) == -T(0));
    static_assert(std::bit_cast<uint_t>(fdapde::ldexp(-T(0), INT_MAX)) == std::bit_cast<uint_t>(-T(0)));
    // constexpr decomposition normalizes the smallest subnormal and reports its exponent
    static_assert([] {
        int exponent = 0;
        return fdapde::frexp(smallest, exponent) == T(0.5) && exponent == smallest_exponent + 1;
    }());
    // constexpr decomposition preserves the sign of the fraction
    static_assert([] {
        int exponent = 0;
        return fdapde::frexp(T(-8), exponent) == T(-0.5) && exponent == 4;
    }());
    // constexpr decomposition preserves negative zero and resets the exponent
    static_assert([] {
        int exponent = 9;
        const T fraction = fdapde::frexp(-T(0), exponent);
        return exponent == 0 && std::bit_cast<uint_t>(fraction) == std::bit_cast<uint_t>(-T(0));
    }());
    // constexpr scaling matches std bit for bit at representative exponent boundaries
    constexpr auto scales = [] {
        constexpr std::array<int, 7> shifts = {INT_MIN, smallest_exponent - 1, smallest_exponent, -1, 0, 1, INT_MAX};
        std::array<T, shifts.size()> result {};
        for (std::size_t i = 0; i < shifts.size(); ++i) result[i] = fdapde::ldexp(T(1.5), shifts[i]);
        return result;
    }();
    const std::array<int, 7> shifts = {INT_MIN, smallest_exponent - 1, smallest_exponent, -1, 0, 1, INT_MAX};
    for (std::size_t i = 0; i < shifts.size(); ++i) EXPECT_EQ(bits(scales[i]), bits(std::ldexp(T(1.5), shifts[i])));
    // runtime decomposition and absolute value match std including signs and extreme finite values
    for (T x : {T(0), -T(0), smallest, -smallest, limits::min(), limits::max(), T(-8)}) {
        int actual = 0, expected = 0;
        EXPECT_EQ(bits(fdapde::frexp(x, actual)), bits(std::frexp(x, &expected)));
        EXPECT_EQ(actual, expected);
        EXPECT_EQ(bits(fdapde::fabs(x)), bits(std::fabs(x)));
    }
    // runtime exponential and scaling propagate nan and infinity
    EXPECT_TRUE(std::isnan(fdapde::exp(nan)));
    EXPECT_TRUE(std::isnan(fdapde::ldexp(nan, 1)));
    EXPECT_EQ(fdapde::ldexp(infinity, -1), infinity);

    // exact equality satisfies non-strict comparisons but never strict comparisons
    static_assert(fdapde::almost_equal(T(1), T(1)));
    static_assert(fdapde::almost_equal(T(1), T(1), T(0)));
    static_assert(fdapde::greater_equal(T(1), T(1)));
    static_assert(fdapde::less_equal(T(1), T(1)));
    static_assert(!fdapde::greater_than(T(1), T(1)));
    static_assert(!fdapde::less_than(T(1), T(1)));
    static_assert(fdapde::almost_zero(T(0)));
    static_assert(fdapde::almost_equal(infinity, infinity));
    // default tolerances distinguish nearby values and avoid overflow for opposite extreme values
    const T tolerance = fdapde::internals::numeric_tolerance<T>;
    EXPECT_TRUE(fdapde::almost_equal(T(1), T(1) + tolerance / 2));
    EXPECT_FALSE(fdapde::almost_equal(T(1), T(1) + tolerance * 4));
    EXPECT_FALSE(fdapde::almost_equal(-limits::max(), limits::max()));
    EXPECT_TRUE(fdapde::greater_equal(T(1), T(1) + tolerance / 2));
    EXPECT_TRUE(fdapde::less_equal(T(1), T(1) - tolerance / 2));
    EXPECT_TRUE(fdapde::greater_than(T(2), T(1)));
    EXPECT_TRUE(fdapde::less_than(T(1), T(2)));
    EXPECT_FALSE(fdapde::almost_equal(nan, nan));
    // log1pexp agrees with a stable reference across its range-dependent branches
    for (T x : {T(-40), T(-1), T(0), T(20), T(40)}) {
        const T expected = std::max(x, T(0)) + std::log1p(std::exp(-std::fabs(x)));
        EXPECT_NEAR(fdapde::log1pexp(x), expected, T(8) * limits::epsilon() * std::max(T(1), expected));
    }
}

}   // namespace

TEST(math, numeric) {
    // exercise the same scalar contracts for float and double
    check_numeric_scalar<float>();
    check_numeric_scalar<double>();

    // factorial returns uint64_t and is exact through 20 factorial
    static_assert(std::is_same_v<decltype(fdapde::fact(0)), std::uint64_t>);
    static_assert(fdapde::fact(0) == 1 && fdapde::fact(13) == UINT64_C(6227020800));
    static_assert(fdapde::fact(20) == UINT64_C(2432902008176640000));
    // binomial coefficients remain exact above the 32-bit range and near the uint64_t limit
    static_assert(std::is_same_v<decltype(fdapde::binom(0, 0)), std::uint64_t>);
    static_assert(fdapde::binom(13, 1) == 13 && fdapde::binom(33, 16) == UINT64_C(1166803110));
    static_assert(fdapde::binom(67, 33) == UINT64_C(14226520737620288370));
    static_assert(fdapde::binom(67, 34) == fdapde::binom(67, 33));
    static_assert(fdapde::binom(INT_MAX, 1) == INT_MAX);
    static_assert(fdapde::binom(5, 0) == 1 && fdapde::binom(5, 5) == 1);
    // combinations have the expected ordering and zero-length selections produce an empty array
    EXPECT_EQ(fdapde::comb(2, 4), (std::vector<std::uint64_t> {0, 1, 0, 2, 0, 3, 1, 2, 1, 3, 2, 3}));
    static_assert(std::is_same_v<decltype(fdapde::comb(2, 4)), std::vector<std::uint64_t>>);
    EXPECT_TRUE(fdapde::comb(0, 4).empty());
    // integer ceiling and floor division round correctly for both operand signs
    for (int a : {-5, -4, 0, 4, 5}) {
        for (int b : {-2, 2}) {
            EXPECT_EQ(fdapde::idiv_ceil (a, b), std::ceil (double(a) / b));
            EXPECT_EQ(fdapde::idiv_floor(a, b), std::floor(double(a) / b));
        }
    }
    // min and max promote mixed scalar types to their common type
    static_assert(fdapde::min(2, 1.5) == 1.5 && fdapde::max(2, 1.5) == 2.0);
    // integer absolute value and sign distinguish negative, zero, and positive inputs
    static_assert(fdapde::abs(-3) == 3);
    static_assert(fdapde::sign(-1) == -1 && fdapde::sign(0) == 0 && fdapde::sign(1u) == 1);
    // simple ceiling and floor round both signs and retain the accepted positive-zero result
    static_assert(fdapde::ceil(-1.25) == -1 && fdapde::floor(-1.25) == -2);
    static_assert(fdapde::ceil(1.25) == 2 && fdapde::floor(1.25) == 1);
    EXPECT_FALSE(std::signbit(fdapde::ceil(-0.25)));
}
