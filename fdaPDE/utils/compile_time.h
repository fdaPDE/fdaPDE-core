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

// compile time computation of the sum of elements in array
template <typename T, int N>
constexpr typename std::enable_if<std::is_arithmetic<T>::value, T>::type ct_array_sum(std::array<T, N> A) {
    T result = 0;
    for (int j = 0; j < N; ++j) result += A[j];
    return result;
}

// trait to detect if T is an Eigen vector
template <typename T> class is_eigen_vector {
   private:
    static constexpr bool check_() {
        if constexpr (std::is_base_of<Eigen::MatrixBase<T>, T>::value) {
            if constexpr (T::ColsAtCompileTime == 1) return true;
            return false;
        }
        return false;
    }
   public:
    // check if T is an eigen matrix and if it has exactly one column, otherwise return false
    static constexpr bool value = check_();
};

}   // namespace core
}   // namespace fdapde

#endif   // __FDAPDE_COMPILE_TIME_H__
