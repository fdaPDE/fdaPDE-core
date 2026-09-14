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

// the Eigen dynamic vectors retain their vector classification with this inclusion order
static_assert(fdapde::internals::is_vector_like_v<Eigen::VectorXd>);
// the Eigen dynamic matrices are not misclassified as vectors with this inclusion order
static_assert(!fdapde::internals::is_vector_like_v<Eigen::MatrixXd>);
// the log-Euclidean geometry retains its geodesic interface with this inclusion order
static_assert(fdapde::manifold::GeodesicGeometry<fdapde::manifold::LogEuclideanSPDGeometry<double, 2>>);
// supplies the SPD(2) tangent dimension from a separate translation unit for the link check
int geometry_header_order() {
    return static_cast<int>(fdapde::manifold::AffineInvariantSPDGeometry<double, 2>().dimension());
}
