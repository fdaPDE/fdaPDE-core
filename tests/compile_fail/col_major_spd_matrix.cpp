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

// column-major packed SPD storage must fail with its dedicated storage-order diagnostic
using col_major_spd_matrix = fdapde::SPDMatrix<double, 3, 3, fdapde::ColMajor>;

// sizeof instantiates the invalid type without also requiring a deleted default constructor
int main() { return sizeof(col_major_spd_matrix); }
