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

#include <fdaPDE/geometry.h>

int public_header_link_other();

int main() {
    fdapde::Matrix<double, fdapde::Dynamic, fdapde::Dynamic> point(1, 2);
    point(0, 0) = 0.0;
    point(0, 1) = 0.0;
    const auto boundary = fdapde::hexagonal_lattice_boundary(point, 1.0);
    const auto simplified = fdapde::simplify_polygon_domain(fdapde::PlanarDomain {.outer = boundary}, 0.0);
    return boundary.rows() == 6 && simplified.outer.rows() == 6 && public_header_link_other() > 0 ? 0 : 1;
}
