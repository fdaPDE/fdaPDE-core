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

#ifndef __FDAPDE_GEOMETRIC_FINITE_ELEMENTS_MODULE_H__
#define __FDAPDE_GEOMETRIC_FINITE_ELEMENTS_MODULE_H__

// clang-format off

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

#include "manifold_optimization.h"

#include "src/geometric_finite_elements/p1_fem_cell_quadrature.h"
#include "src/geometric_finite_elements/p1_geodesic_value.h"
#include "src/geometric_finite_elements/p1_lumped_laplacian_stencil.h"
#include "src/geometric_finite_elements/p1_geodesic_linearization.h"
#include "src/geometric_finite_elements/p1_objective_contributions.h"

// clang-format on

#endif   // __FDAPDE_GEOMETRIC_FINITE_ELEMENTS_MODULE_H__
