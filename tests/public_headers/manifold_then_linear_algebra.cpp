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
#include <fdaPDE/manifold_optimization.h>

static_assert(fdapde::internals::is_vector_like_v<Eigen::VectorXd>);
static_assert(!fdapde::internals::is_vector_like_v<Eigen::MatrixXd>);
static_assert(fdapde::manifold::GeodesicGeometry<fdapde::manifold::LogEuclideanSPDGeometry<double, 2>>);
int geometry_header_order();
int main() { return geometry_header_order() == 3 ? 0 : 1; }
