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

#ifndef __FDAPDE_GEOMETRY_MODULE_H__
#define __FDAPDE_GEOMETRY_MODULE_H__

// clang-format off

// include required modules
#include "linear_algebra.h"    // pull Eigen first
#include "utility.h"
#include "io.h"

// minimal geometric entites
#include "src/geometry/utility.h"
#include "src/geometry/primitives.h"
#include "src/geometry/hyperplane.h"
#include "src/geometry/simplex.h"
#include "src/geometry/segment.h"
#include "src/geometry/triangle.h"
#include "src/geometry/tetrahedron.h"
// algorithms
#include "src/geometry/kd_tree.h"
#include "src/geometry/tree_search.h"
#include "src/geometry/walk_search.h"
#include "src/geometry/projection.h"
// data structures
#include "src/geometry/triangulation.h"
#include "src/geometry/interval.h"
#include "src/geometry/linear_network.h"
#include "src/geometry/dcel.h"
#include "src/geometry/polygon.h"





// clang-format on

#endif   // __FDAPDE_GEOMETRY_MODULE_H__
