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

#ifndef __CONSTEXPR_H__
#define __CONSTEXPR_H__

#include <array>

#include "../linear_algebra/constexpr_matrix.h"

namespace fdapde {
namespace cexpr {

// factorial function n!
constexpr int factorial(const int n) {
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
    if (m == 0) return 1;
    return factorial(n) / (factorial(m) * factorial(n - m));
}

// all combinations of k elements from a set of n elements
template <int K, int N> constexpr Matrix<int, binomial_coefficient(N, K), K> combinations() {
    std::vector<bool> bitmask(K, 1);
    bitmask.resize(N, 0);
    Matrix<int, binomial_coefficient(N, K), K> result;
    int j = 0;
    do {
        int k = 0;
        for (int i = 0; i < N; ++i) {
            if (bitmask[i]) result(j, k++) = i;
        }
        j++;
    } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
    return result;
}

template <typename Scalar, int Size>
constexpr typename std::enable_if<std::is_arithmetic_v<Scalar>, Scalar>::type
array_sum(std::array<Scalar, Size> array) {
    Scalar result = 0;
    for (int j = 0; j < Size; ++j) result += array[j];
    return result;
}

}   // namespace cexpr
}   // namespace fdapde

#endif   // _CONSTEXPR_H__
