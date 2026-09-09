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

#include <array>
#include <cmath>
#include <limits>

namespace {
constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double not_a_number = std::numeric_limits<double>::quiet_NaN();
// bounds the constexpr root near the analytic value 1e-10
static_assert(fdapde::sqrt(1e-20) > 0.99999999999999e-10 && fdapde::sqrt(1e-20) < 1.00000000000001e-10);
// preserves the positive infinite square root
static_assert(fdapde::sqrt(infinity) == infinity);
// returns NaN for the root of a negative finite value
static_assert(fdapde::sqrt(-1.0) != fdapde::sqrt(-1.0));
// returns NaN for the root of negative infinity
static_assert(fdapde::sqrt(-infinity) != fdapde::sqrt(-infinity));
// propagates a NaN input through the constexpr root
static_assert(fdapde::sqrt(not_a_number) != fdapde::sqrt(not_a_number));
// compares bits to verify the root retains negative zero
static_assert(std::bit_cast<std::uint64_t>(fdapde::sqrt(-0.0)) == std::bit_cast<std::uint64_t>(-0.0));

/// @brief returns the signed fraction and binary exponent of a value
constexpr auto decompose(double value) {
    int exponent = 0;
    const double fraction = fdapde::frexp(value, exponent);
    return std::pair {fraction, exponent};
}
// decomposes negative eight into a negative fraction and exponent four
static_assert(decompose(-8.0) == std::pair {-0.5, 4});
// checks the fraction and exponent of the smallest positive subnormal
static_assert(
  decompose(std::numeric_limits<double>::denorm_min()) ==
  std::pair {0.5, std::numeric_limits<double>::min_exponent - std::numeric_limits<double>::digits + 1});
// preserves positive infinity with the documented zero exponent
static_assert(decompose(infinity) == std::pair {infinity, 0});
// preserves negative infinity with the documented zero exponent
static_assert(decompose(-infinity) == std::pair {-infinity, 0});
// propagates NaN in the decomposed fraction
static_assert(decompose(not_a_number).first != decompose(not_a_number).first);
// sets the exponent of a NaN input to zero
static_assert(decompose(not_a_number).second == 0);
// compares fraction bits to verify decomposition retains negative zero
static_assert(std::bit_cast<std::uint64_t>(decompose(-0.0).first) == std::bit_cast<std::uint64_t>(-0.0));

// preserves the positive infinite logarithmic limit
static_assert(fdapde::log(infinity) == infinity);
// maps both signed zeros to the negative infinite logarithmic limit
static_assert(fdapde::log(0.0) == -infinity && fdapde::log(-0.0) == -infinity);
// returns NaN for the logarithm of a negative finite value
static_assert(fdapde::log(-1.0) != fdapde::log(-1.0));
// returns NaN for the logarithm of negative infinity
static_assert(fdapde::log(-infinity) != fdapde::log(-infinity));
// propagates a NaN input through the constexpr logarithm
static_assert(fdapde::log(not_a_number) != fdapde::log(not_a_number));
// preserves the positive infinite limit through log1p
static_assert(fdapde::log1p(infinity) == infinity);

/// @brief compares runtime and constexpr roots against the standard library
template <typename Scalar> void check_square_roots() {
    constexpr std::array inputs {
      Scalar(0),
      Scalar(1),
      Scalar(4),
      Scalar(1e-20),
      Scalar(1e20),
      std::numeric_limits<Scalar>::min(),
      std::numeric_limits<Scalar>::denorm_min(),
      std::numeric_limits<Scalar>::max()};
    constexpr auto constant_results = [inputs] {
        auto results = inputs;
        for (auto& value : results) value = fdapde::sqrt(value);
        return results;
    }();
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        SCOPED_TRACE(inputs[i]);
        const Scalar expected = std::sqrt(inputs[i]);
        if (inputs[i] == Scalar(0)) {
            // checks that constant evaluation preserves exact zero
            EXPECT_EQ(constant_results[i], expected);
            // checks that runtime evaluation preserves exact zero
            EXPECT_EQ(fdapde::sqrt(inputs[i]), expected);
        } else {
            const Scalar tolerance = Scalar(8) * std::numeric_limits<Scalar>::epsilon();
            // bounds the relative error of the constexpr root across the floating-point range
            EXPECT_LE(std::abs(constant_results[i] / expected - Scalar(1)), tolerance);
            // bounds the runtime root error relative to the standard library result
            EXPECT_LE(std::abs(fdapde::sqrt(inputs[i]) / expected - Scalar(1)), tolerance);
        }
    }
    // checks that the runtime root retains negative zero
    EXPECT_TRUE(std::signbit(fdapde::sqrt(Scalar(-0.0))));
    // checks the positive infinite root
    EXPECT_EQ(fdapde::sqrt(std::numeric_limits<Scalar>::infinity()), std::numeric_limits<Scalar>::infinity());
    // checks that a negative finite argument produces NaN
    EXPECT_TRUE(std::isnan(fdapde::sqrt(Scalar(-1))));
    // checks that negative infinity is outside the real domain
    EXPECT_TRUE(std::isnan(fdapde::sqrt(-std::numeric_limits<Scalar>::infinity())));
    // checks NaN propagation through the runtime root
    EXPECT_TRUE(std::isnan(fdapde::sqrt(std::numeric_limits<Scalar>::quiet_NaN())));
}
}   // namespace

// verifies root scaling and special values for every supported floating-point type
TEST(utility, square_root_preserves_scale_and_domain_in_runtime_and_constant_evaluation) {
    check_square_roots<float>();
    check_square_roots<double>();
    check_square_roots<long double>();
}

// verifies signed decomposition at zero, finite extremes and nonfinite values
TEST(utility, frexp_preserves_signed_values_and_exponents) {
    constexpr std::array inputs {
      0.0,
      -0.0,
      1.0,
      -8.0,
      0.75,
      -0.75,
      std::numeric_limits<double>::min(),
      -std::numeric_limits<double>::min(),
      std::numeric_limits<double>::denorm_min(),
      -std::numeric_limits<double>::denorm_min(),
      std::numeric_limits<double>::max(),
      -std::numeric_limits<double>::max()};
    constexpr auto constant_results = [inputs] {
        std::array<std::pair<double, int>, inputs.size()> results {};
        for (std::size_t i = 0; i < inputs.size(); ++i) results[i] = decompose(inputs[i]);
        return results;
    }();
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        SCOPED_TRACE(inputs[i]);
        int expected_exponent = 0;
        const double expected_fraction = std::frexp(inputs[i], &expected_exponent);
        for (const auto& result : {constant_results[i], decompose(inputs[i])}) {
            // compares the signed fraction with std::frexp
            EXPECT_EQ(result.first, expected_fraction);
            // compares the exponent with std::frexp including subnormal inputs
            EXPECT_EQ(result.second, expected_exponent);
            // checks the fraction sign separately so negative zero is covered
            EXPECT_EQ(std::signbit(result.first), std::signbit(expected_fraction));
            // reconstructs the original input from the returned fraction and exponent
            EXPECT_EQ(std::ldexp(result.first, result.second), inputs[i]);
        }
    }
    for (const double value : {infinity, -infinity, not_a_number}) {
        const auto result = decompose(value);
        // checks the documented zero exponent for nonfinite inputs
        EXPECT_EQ(result.second, 0);
        if (std::isnan(value)) {
            // verifies a NaN fraction remains NaN
            EXPECT_TRUE(std::isnan(result.first));
        } else {
            // verifies the sign of infinity is retained
            EXPECT_EQ(result.first, value);
        }
    }
}

// verifies logarithm limits and approximation error at representative finite scales
TEST(utility, logarithm_preserves_limits_and_finite_values) {
    constexpr std::array inputs {
      0.5,
      1.0,
      1.00000001,
      2.0,
      1e-20,
      1e20,
      std::numeric_limits<double>::min(),
      std::numeric_limits<double>::denorm_min(),
      std::numeric_limits<double>::max()};
    constexpr auto constant_results = [inputs] {
        auto results = inputs;
        for (auto& value : results) value = fdapde::log(value);
        return results;
    }();
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        SCOPED_TRACE(inputs[i]);
        const double expected = std::log(inputs[i]);
        // the logarithm polynomial is checked with an absolute approximation tolerance
        constexpr double tolerance = 1e-9;
        // bounds constexpr logarithm approximation error against std::log
        EXPECT_NEAR(constant_results[i], expected, tolerance);
        // bounds runtime logarithm approximation error against std::log
        EXPECT_NEAR(fdapde::log(inputs[i]), expected, tolerance);
    }
    // checks that logarithm maps positive infinity to positive infinity
    EXPECT_EQ(fdapde::log(infinity), infinity);
    // checks that log1p inherits the positive infinite limit
    EXPECT_EQ(fdapde::log1p(infinity), infinity);
    // checks the right-hand logarithmic limit at zero
    EXPECT_EQ(fdapde::log(0.0), -infinity);
    // checks that signed zero has the same logarithmic limit
    EXPECT_EQ(fdapde::log(-0.0), -infinity);
    // rejects negative finite logarithm inputs with NaN
    EXPECT_TRUE(std::isnan(fdapde::log(-1.0)));
    // rejects negative infinity with NaN
    EXPECT_TRUE(std::isnan(fdapde::log(-infinity)));
    // checks NaN propagation through the logarithm
    EXPECT_TRUE(std::isnan(fdapde::log(not_a_number)));
}
