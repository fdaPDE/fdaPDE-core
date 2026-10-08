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

#include <fdaPDE/manifold_means.h>

/// @brief instantiates closed-form and iterative mean overloads through their dedicated public aggregate
void instantiate_manifold_means() {
    using namespace fdapde;
    using namespace fdapde::manifold;
    const std::array<double, 2> weights {.25, .75};
    MatrixBatch<SPDMatrix<double, 2>> points(2);
    const auto identity = SPDMatrix<double, 2>::Identity();
    (void)weighted_karcher_mean(LogEuclideanSPDGeometry<double, 2> {}, points, weights);
    (void)weighted_karcher_mean(LogCholeskySPDGeometry<double, 2> {}, points, weights);
    (void)weighted_karcher_mean(AffineInvariantSPDGeometry<double, 2> {}, points, weights);
    (void)weighted_karcher_mean(BuresWassersteinSPDGeometry<double, 2> {}, points, weights);
    (void)weighted_karcher_mean(CheegerLogEuclideanSPDGeometry<double, 2> {}, points, weights, identity);
    const MatrixBatch<RotationMatrix<double, 3, 3>> rotations(2);
    (void)weighted_karcher_mean(SOGeometry<double, 3> {}, rotations, weights, RotationMatrix<double, 3, 3>::Identity());
}

#include <fdaPDE/manifold_geometry.h>
#include <fdaPDE/manifold_solvers.h>
