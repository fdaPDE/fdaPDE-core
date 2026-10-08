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

#include <fdaPDE/manifold_geometry.h>

// geometric operations must not import barycenter result types or iterative mean implementations
#ifdef __FDAPDE_MANIFOLD_MEANS_MODULE_H__
#    error "The geometry aggregate must not depend on the means aggregate"
#endif

// SO and general-order Cheeger remain available through the geometry-only public entry point
static_assert(fdapde::manifold::GeodesicGeometry<fdapde::manifold::SOGeometry<double, 3>>);
// the general-order quotient geometry retains its solver-backed implementation
static_assert(fdapde::manifold::GeodesicGeometry<fdapde::manifold::CheegerLogEuclideanSPDGeometry<double, 3>>);

/// @brief instantiates independent SO and Cheeger operations before any mean include
void instantiate_manifold_geometries() {
    const fdapde::manifold::SOGeometry<double, 3> rotation;
    const auto identity = decltype(rotation)::Point::Identity();
    (void)rotation.retract(identity, rotation.zero_tangent(identity), .5);
    const fdapde::manifold::CheegerLogEuclideanSPDGeometry<double, 3> cheeger;
    const auto point = decltype(cheeger)::Point::Identity();
    (void)cheeger.distance(point, point);
}

#include <fdaPDE/manifold_means.h>
