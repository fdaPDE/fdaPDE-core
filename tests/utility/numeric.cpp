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
using lookup_types = std::tuple<int, double, char>;
static_assert(fdapde::internals::index_of<int, lookup_types>::value == 0);
static_assert(fdapde::internals::index_of<double, lookup_types>::value == 1);
static_assert(fdapde::internals::index_of<char, lookup_types>::value == 2);

constexpr double infinity = std::numeric_limits<double>::infinity();
constexpr double not_a_number = std::numeric_limits<double>::quiet_NaN();
static_assert(fdapde::sqrt(1e-20) > 0.99999999999999e-10 && fdapde::sqrt(1e-20) < 1.00000000000001e-10);
static_assert(fdapde::sqrt(infinity) == infinity);
static_assert(fdapde::sqrt(-1.0) != fdapde::sqrt(-1.0));
static_assert(fdapde::sqrt(-infinity) != fdapde::sqrt(-infinity));
static_assert(fdapde::sqrt(not_a_number) != fdapde::sqrt(not_a_number));
static_assert(std::bit_cast<std::uint64_t>(fdapde::sqrt(-0.0)) == std::bit_cast<std::uint64_t>(-0.0));

constexpr auto decompose(double value) {
    int exponent = 0;
    const double fraction = fdapde::frexp(value, exponent);
    return std::pair {fraction, exponent};
}
static_assert(decompose(-8.0) == std::pair {-0.5, 4});
static_assert(decompose(std::numeric_limits<double>::denorm_min()) ==
              std::pair {0.5, std::numeric_limits<double>::min_exponent - std::numeric_limits<double>::digits + 1});
static_assert(decompose(infinity) == std::pair {infinity, 0});
static_assert(decompose(-infinity) == std::pair {-infinity, 0});
static_assert(decompose(not_a_number).first != decompose(not_a_number).first);
static_assert(decompose(not_a_number).second == 0);
static_assert(std::bit_cast<std::uint64_t>(decompose(-0.0).first) == std::bit_cast<std::uint64_t>(-0.0));

static_assert(fdapde::log(infinity) == infinity);
static_assert(fdapde::log(0.0) == -infinity && fdapde::log(-0.0) == -infinity);
static_assert(fdapde::log(-1.0) != fdapde::log(-1.0));
static_assert(fdapde::log(-infinity) != fdapde::log(-infinity));
static_assert(fdapde::log(not_a_number) != fdapde::log(not_a_number));
static_assert(fdapde::log1p(infinity) == infinity);

template <typename Scalar> void check_square_roots() {
    constexpr std::array inputs {
      Scalar(0), Scalar(1), Scalar(4), Scalar(1e-20), Scalar(1e20),
      std::numeric_limits<Scalar>::min(), std::numeric_limits<Scalar>::denorm_min(),
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
            EXPECT_EQ(constant_results[i], expected);
            EXPECT_EQ(fdapde::sqrt(inputs[i]), expected);
        } else {
            const Scalar tolerance = Scalar(8) * std::numeric_limits<Scalar>::epsilon();
            EXPECT_LE(std::abs(constant_results[i] / expected - Scalar(1)), tolerance);
            EXPECT_LE(std::abs(fdapde::sqrt(inputs[i]) / expected - Scalar(1)), tolerance);
        }
    }
    EXPECT_TRUE(std::signbit(fdapde::sqrt(Scalar(-0.0))));
    EXPECT_EQ(fdapde::sqrt(std::numeric_limits<Scalar>::infinity()), std::numeric_limits<Scalar>::infinity());
    EXPECT_TRUE(std::isnan(fdapde::sqrt(Scalar(-1))));
    EXPECT_TRUE(std::isnan(fdapde::sqrt(-std::numeric_limits<Scalar>::infinity())));
    EXPECT_TRUE(std::isnan(fdapde::sqrt(std::numeric_limits<Scalar>::quiet_NaN())));
}
}   // namespace

TEST(utility, square_root_preserves_scale_and_domain_in_runtime_and_constant_evaluation) {
    check_square_roots<float>();
    check_square_roots<double>();
    check_square_roots<long double>();
}

TEST(utility, frexp_preserves_signed_values_and_exponents) {
    constexpr std::array inputs {
      0.0, -0.0, 1.0, -8.0, 0.75, -0.75,
      std::numeric_limits<double>::min(), -std::numeric_limits<double>::min(),
      std::numeric_limits<double>::denorm_min(), -std::numeric_limits<double>::denorm_min(),
      std::numeric_limits<double>::max(), -std::numeric_limits<double>::max()};
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
            EXPECT_EQ(result.first, expected_fraction);
            EXPECT_EQ(result.second, expected_exponent);
            EXPECT_EQ(std::signbit(result.first), std::signbit(expected_fraction));
            EXPECT_EQ(std::ldexp(result.first, result.second), inputs[i]);
        }
    }
    for (const double value : {infinity, -infinity, not_a_number}) {
        const auto result = decompose(value);
        EXPECT_EQ(result.second, 0);
        if (std::isnan(value)) EXPECT_TRUE(std::isnan(result.first));
        else EXPECT_EQ(result.first, value);
    }
}

TEST(utility, logarithm_preserves_limits_and_finite_values) {
    constexpr std::array inputs {
      0.5, 1.0, 1.00000001, 2.0, 1e-20, 1e20, std::numeric_limits<double>::min(),
      std::numeric_limits<double>::denorm_min(), std::numeric_limits<double>::max()};
    constexpr auto constant_results = [inputs] {
        auto results = inputs;
        for (auto& value : results) value = fdapde::log(value);
        return results;
    }();
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        SCOPED_TRACE(inputs[i]);
        const double expected = std::log(inputs[i]);
        // The retained finite logarithm is a polynomial approximation, not a correctly rounded implementation.
        constexpr double tolerance = 1e-9;
        EXPECT_NEAR(constant_results[i], expected, tolerance);
        EXPECT_NEAR(fdapde::log(inputs[i]), expected, tolerance);
    }
    EXPECT_EQ(fdapde::log(infinity), infinity);
    EXPECT_EQ(fdapde::log1p(infinity), infinity);
    EXPECT_EQ(fdapde::log(0.0), -infinity);
    EXPECT_EQ(fdapde::log(-0.0), -infinity);
    EXPECT_TRUE(std::isnan(fdapde::log(-1.0)));
    EXPECT_TRUE(std::isnan(fdapde::log(-infinity)));
    EXPECT_TRUE(std::isnan(fdapde::log(not_a_number)));
}
