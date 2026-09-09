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

#include <fdaPDE/utility.h>
#include <gtest/gtest.h>

namespace {
/// @brief models an expression that retains its owner by reference
struct owner_expression {
    static constexpr int NestAsRef = 1;
};
/// @brief models an expression node copied into its parent
struct value_expression {
    static constexpr int NestAsRef = 0;
};
}   // namespace

// verifies storage selection preserves constness and the declared nesting policy
TEST(NumericFoundation, ExpressionNestingTraits) {
    using namespace fdapde::internals;
    // verifies mutable owners are stored by reference
    static_assert(std::is_same_v<ref_select_t<owner_expression>, owner_expression&>);
    // verifies const owners remain const when nested
    static_assert(std::is_same_v<ref_select_t<const owner_expression>, const owner_expression&>);
    // verifies value nodes are copied with their cv qualification
    static_assert(std::is_same_v<ref_select_t<const value_expression>, const value_expression>);
    // verifies types without a nesting flag retain their original reference category
    static_assert(std::is_same_v<ref_select_t<int&>, int&>);
    // verifies value nodes supplied as references keep the existing no-flag-lookup behavior
    static_assert(std::is_same_v<ref_select_t<value_expression&>, value_expression&>);
}

// verifies scalar helpers remain usable in constant expressions with mixed arithmetic types
TEST(NumericFoundation, ScalarHelpers) {
    // verifies min promotes both arguments before selecting the smaller value
    static_assert(fdapde::min(2, 1.5) == 1.5);
    // verifies max preserves a promoted floating-point result
    static_assert(fdapde::max(2, 1.5) == 2.0);
    // verifies positive integer division rounds upward
    static_assert(fdapde::int_ceil(7, 3) == 3);
    // verifies negative integer division rounds downward
    static_assert(fdapde::int_floor(-7, 3) == -3);
    // verifies absolute value accepts unsigned coefficients
    static_assert(fdapde::abs(3u) == 3u);
    // verifies strict stable comparison remains false at equality
    static_assert(!fdapde::greater_than(2.0, 2.0) && !fdapde::less_than(2.0, 2.0));
    // verifies the existing tolerance-based non-strict comparison convention
    static_assert(!fdapde::greater_equal(2.0, 2.0) && !fdapde::less_equal(2.0, 2.0));
    // verifies the nonnegative sign convention remains unchanged
    static_assert(fdapde::sign(-2.0) == 0 && fdapde::sign(0.0) == 1);
    // verifies the minimum signed exponent is handled without signed negation overflow
    static_assert(fdapde::pow(1.0, std::numeric_limits<int>::min()) == 1.0);
    // verifies wide exponents are not truncated to unsigned int
    static_assert(fdapde::pow(-1.0, (1LL << 32) + 1) == -1.0);
    // verifies negative powers evaluate the reciprocal
    EXPECT_DOUBLE_EQ(fdapde::pow(2.0, -3), 0.125);
    // compares a representative exponential with its standard library reference
    EXPECT_NEAR(fdapde::exp(1.0), std::exp(1.0), 2e-6);
    // checks stable softplus avoids overflow for large positive inputs
    EXPECT_DOUBLE_EQ(fdapde::log1pexp(1000.0), 1000.0);
}

// verifies binary scaling preserves subnormals and supports every declared floating-point type
TEST(NumericFoundation, BinaryScaling) {
    // verifies the smallest positive double is representable during constant evaluation
    static_assert(fdapde::ldexp(1.0, -1074) == std::numeric_limits<double>::denorm_min());
    // verifies scaling a negative subnormal retains its sign and magnitude
    static_assert(
      fdapde::ldexp(-std::numeric_limits<double>::denorm_min(), 1) == -2 * std::numeric_limits<double>::denorm_min());
    // verifies character-sized integer exponents retain the declared integer support
    static_assert(fdapde::ldexp(0.75, char(3)) == 6.0);
    // verifies float scaling does not depend on a binary64 bit_cast
    static_assert(fdapde::ldexp(0.75f, 3) == 6.0f);
    // verifies long double scaling uses its own format and precision
    static_assert(fdapde::ldexp(0.75L, 3) == 6.0L);
    // verifies maximum exponents are bounded before integer conversion
    static_assert(
      fdapde::ldexp(1.0, std::numeric_limits<unsigned long long>::max()) == std::numeric_limits<double>::infinity());
    // verifies minimum exponents underflow without integer overflow
    static_assert(fdapde::ldexp(1.0, std::numeric_limits<long long>::min()) == 0.0);
    constexpr std::array inputs {0.5, 0.75, 1.0, 1.5, -1.5, std::numeric_limits<double>::max()};
    constexpr std::array exponents {-1075, -1074, -1023, -3, 0, 3, 1023};
    constexpr auto constant_results = [inputs, exponents] {
        std::array<double, inputs.size() * exponents.size()> results {};
        int i = 0;
        for (double value : inputs)
            for (int exponent : exponents) results[i++] = fdapde::ldexp(value, exponent);
        return results;
    }();
    int i = 0;
    for (double value : inputs) {
        for (int exponent : exponents) {
            SCOPED_TRACE(value);
            SCOPED_TRACE(exponent);
            // compares constant scaling with std::ldexp including ties at underflow
            EXPECT_EQ(constant_results[i++], std::ldexp(value, exponent));
            // compares runtime scaling with std::ldexp at the same boundaries
            EXPECT_EQ(fdapde::ldexp(value, exponent), std::ldexp(value, exponent));
        }
    }
    // verifies runtime scaling preserves negative zero
    EXPECT_TRUE(std::signbit(fdapde::ldexp(-0.0, 8)));
}
