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

#include <fdaPDE/linear_algebra.h>

/// @brief supplies zero coefficients without allocating matrix storage
struct zero_functor {
    /// @brief returns zero for any requested coefficient
    constexpr double operator()(int, int) const { return 0.0; }
};

using long_column = fdapde::ProceduralMatrix<zero_functor, 50000, 1>;
using long_row = fdapde::ProceduralMatrix<zero_functor, 1, 50000>;
using oversized_outer_product = decltype(std::declval<long_column&>() * std::declval<long_row&>());
using oversized_block = fdapde::MatrixBlock<50000, 50000, oversized_outer_product>;

int main() { return sizeof(oversized_block); }
