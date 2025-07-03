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

// factorial of n
constexpr int factorial(const int n) {
    fdapde_constexpr_assert(n >= 0);
    int factorial_ = 1;
    if (n == 0) return factorial_;
    int m = n;
    while (m > 0) {
        factorial_ = m * factorial_;
        m--;
    }
    return factorial_;
}
// binomial coefficient n over m
constexpr int binomial_coefficient(const int n, const int m) {
    if (m == 0 || n == m) return 1;
    return factorial(n) / (factorial(m) * factorial(n - m));
}
// binomial_coefficient(n, k) x k matrix of combinations of k elements from a set of n
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

// integer division with round up
template <typename T1, typename T2>
    requires(internals::is_integer_v<T1> && internals::is_integer_v<T2>)
constexpr std::common_type_t<T1, T2> int_ceil(T1 a, T2 b) {
    return ((a ^ b) >= 0) ? a / b + (a % b != 0) : a / b;
}
// integer division with round down
template <typename T1, typename T2>
    requires(internals::is_integer_v<T1> && internals::is_integer_v<T2>)
constexpr std::common_type_t<T1, T2> int_floor(T1 a, T2 b) {
    return ((a ^ b) < 0 && a % b != 0) ? a / b - 1 : a / b;
}

// min function with common type conversion
template <typename T1, typename T2> std::common_type_t<T1, T2> min(T1 a, T2 b) {
    using T = std::common_type_t<T1, T2>;
    return std::min<T>(static_cast<T>(a), static_cast<T>(b));
}
// max function with common type conversion
template <typename T1, typename T2> std::common_type_t<T1, T2> max(T1 a, T2 b) {
    using T = std::common_type_t<T1, T2>;
    return std::max<T>(static_cast<T>(a), static_cast<T>(b));
}

// test for floating point equality
[[maybe_unused]] constexpr double double_tolerance = 1e-10;
[[maybe_unused]] constexpr double machine_epsilon  = 50 * std::numeric_limits<double>::epsilon();   // approx 10^-14
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr bool almost_equal(T a, T b, T epsilon) {
    return std::fabs(a - b) < epsilon ||
           std::fabs(a - b) < ((std::fabs(a) < std::fabs(b) ? std::fabs(b) : std::fabs(a)) * epsilon);
}
template <typename T> constexpr bool almost_equal(T a, T b) { return almost_equal(a, b, double_tolerance); }
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr bool greater_than(T a, T b, T epsilon) {
    return (a - b) > ((std::fabs(a) < std::fabs(b) ? std::fabs(b) : std::fabs(a)) * epsilon);
}
template <typename T> constexpr bool greater_than(T a, T b) { return greater_than(a, b, double_tolerance); }
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr bool less_than(T a, T b, T epsilon) {
    return (b - a) > ((std::fabs(a) < std::fabs(b) ? std::fabs(b) : std::fabs(a)) * epsilon);
}
template <typename T> constexpr bool less_than(T a, T b) { return less_than(a, b, double_tolerance); }
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr bool almost_zero(T a, T epsilon) {
    return std::fabs(a) < epsilon;
}
template <typename T> constexpr bool almost_zero(T a) { return almost_zero(a, machine_epsilon); }

// numerical stable log(1 + exp(x)) computation (see "Machler, M. (2012). Accurately computing log(1-exp(-|a|))")
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr T log1pexp(T x) {
    if (x <= -37.0) return std::exp(x);
    if (x <=  18.0) return std::log1p(std::exp(x));
    if (x >   33.3) return x;
    return x + std::exp(-x);
}

// constexpr absoulte value
template <typename T> requires(std::is_signed_v<T>) constexpr T abs(T x) { return x < 0 ? -x : x; }

// constexpr square root
template <typename T>
    requires(std::is_floating_point_v<T>)
constexpr T sqrt(T x) {
    auto heron_method = [](T x) {
        T curr = x, prev = 0;
        while (fdapde::abs(curr - prev) > machine_epsilon) {
            prev = curr;
            curr = 0.5 * (curr + x / curr);
        }
        return curr;
    };
    return x >= 0 && x < std::numeric_limits<T>::infinity() ? heron_method(x) : std::numeric_limits<T>::quiet_NaN();
}

// constexpr ceil
template <typename T> constexpr std::conditional_t<std::is_floating_point_v<T>, T, double> ceil(T x) {
    long int int_part = static_cast<long int>(x);
    return (x > 0.0 && x != static_cast<T>(int_part)) ? int_part + 1.0 : int_part;
}
// constexpr floor
template <typename T> constexpr std::conditional_t<std::is_floating_point_v<T>, T, double> floor(T x) {
    long int int_part = static_cast<long int>(x);
    return (x < 0.0 && x != static_cast<T>(int_part)) ? int_part - 1.0 : int_part;
}

// constexpr pow, only integer exponent support
template <typename BaseT, typename ExpT>
    requires(std::is_floating_point_v<BaseT> && internals::is_integer_v<ExpT>)
constexpr BaseT pow(BaseT base, ExpT exp) {
    if (exp == 0) return BaseT {1};
    unsigned int abs_exp = static_cast<unsigned int>((exp < 0) ? -exp : exp);
    BaseT result = BaseT {1};
    while (abs_exp > 0) {
        if (abs_exp % 2 == 1) result *= base;
        base *= base;
        abs_exp /= 2;
    }
    return exp < 0 ? BaseT {1} / result : result;
}

}   // namespace fdapde

#endif   // __FDAPDE_NUMERIC_H__
