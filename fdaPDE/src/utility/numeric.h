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

// factorial function n!
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
// binomial coefficient function n over m
constexpr int binomial_coefficient(const int n, const int m) {
    if (m == 0 || n == m) return 1;
    return factorial(n) / (factorial(m) * factorial(n - m));
}
// linearized binomial_coefficient(n, k) x k matrix of combinations of k elements from a set of n
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

// unsigned integer division with round up
constexpr int int_ceil(unsigned int a, unsigned int b) { return a / b + (a % b != 0); }

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

}   // namespace fdapde

#endif   // __FDAPDE_NUMERIC_H__
