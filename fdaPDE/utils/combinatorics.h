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

#ifndef __COMBINATORICS_H__
#define __COMBINATORICS_H__

#include "compile_time.h"
#include "symbols.h"

namespace fdapde {
namespace core {

// a set of utilities for combinatoric calculus

// compile time evaluation of the factorial function
constexpr int ct_factorial(const int n) { return n ? (n * ct_factorial(n - 1)) : 1; }

// compile time evaluation of the binomial coefficient N over M
constexpr int ct_binomial_coefficient(const int N, const int M) {
    return ct_factorial(N) / (ct_factorial(M) * ct_factorial(N - M));
}

// all combinations of k elements from a set of n
template <int K, int N> SMatrix<ct_binomial_coefficient(N, K), K, int> combinations() {
    std::vector<bool> bitmask(K, 1);
    bitmask.resize(N, 0);
    SMatrix<ct_binomial_coefficient(N, K), K, int> result;
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

}   // namespace core
}   // namespace fdapde

#endif   // _COMBINATORICS_H__
