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

#ifndef __FDAPDE_COMPILE_TIME_H__
#define __FDAPDE_COMPILE_TIME_H__

#include <array>

namespace fdapde {
namespace core {

// compile time evaluation of the factorial function
constexpr std::size_t ct_factorial(const std::size_t n) { return n ? (n * ct_factorial(n - 1)) : 1; }

// compile time evaluation of binomial coefficient
constexpr std::size_t ct_binomial_coefficient(const std::size_t N, const std::size_t M) {
    return ct_factorial(N) / (ct_factorial(M) * ct_factorial(N - M));
}

// compile time computation of the sum of elements in array
template <typename T, std::size_t N>
constexpr typename std::enable_if<std::is_arithmetic<T>::value, T>::type ct_array_sum(std::array<T, N> A) {
    T result = 0;
    for (int j = 0; j < N; ++j) result += A[j];
    return result;
}

// trait to detect if T is an Eigen vector
template <typename T> struct is_eigen_vector {
    // check if T is an eigen matrix and if it has exactly one column, otherwise return false
    static constexpr bool value = std::is_base_of<Eigen::MatrixBase<T>, T>::value ? T::ColsAtCompileTime == 1 : false;
};

}   // namespace core
}   // namespace fdapde

#endif   // __FDAPDE_COMPILE_TIME_H__
