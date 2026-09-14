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

#include <fdaPDE/manifold_optimization.h>

// force log-Euclidean instantiation to check rejection of
// a const-qualified scalar instead of an unqualified floating-point type
static_assert(sizeof(fdapde::manifold::LogEuclideanSPDGeometry<const double, 2>) > 0);
// force affine-invariant instantiation to check rejection of
// a const-qualified scalar instead of an unqualified floating-point type
static_assert(sizeof(fdapde::manifold::AffineInvariantSPDGeometry<const double, 2>) > 0);
int main() { }
