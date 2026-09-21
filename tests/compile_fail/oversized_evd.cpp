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

#include <fdaPDE/dense_linear_algebra.h>

/// @brief supplies static dimensions whose squared workspace size exceeds the supported integer range
struct oversized_evd_expression {
    using Scalar = double;
    static constexpr int Rows = 46341;
    static constexpr int Cols = 46341;
};

using oversized_evd = fdapde::EVD<oversized_evd_expression>;

int main() { return sizeof(oversized_evd); }
