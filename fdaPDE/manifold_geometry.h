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

#ifndef __FDAPDE_MANIFOLD_GEOMETRY_MODULE_H__
#define __FDAPDE_MANIFOLD_GEOMETRY_MODULE_H__

// clang-format off

#include "dense_linear_algebra.h"
#include <concepts>
#include <cstddef>
#include <optional>
#include <ranges>
#include <span>

#include "src/manifold/concepts.h"

#include "src/manifold/geometry/geometry_expr.h"
#include "src/manifold/geometry/euclidean.h"
#include "src/manifold/geometry/product_geometry.h"
#include "src/manifold/geometry/spd/log_euclidean/log_euclidean_spd.h"
#include "src/manifold/geometry/spd/log_cholesky/log_cholesky_spd.h"
#include "src/manifold/geometry/spd/affine_invariant/affine_invariant_spd.h"
#include "src/manifold/geometry/spd/bures_wasserstein/bures_wasserstein_spd.h"
#include "src/manifold/geometry/so/so.h"

// general-order Cheeger alignment uses generic solvers, which depend only on the shared contracts
#include "manifold_solvers.h"
#include "src/manifold/geometry/spd/cheeger_log_euclidean/cheeger_log_euclidean_spd.h"
#include "src/manifold/geometry/spd/cheeger_log_euclidean/cheeger_lift.h"
#include "src/manifold/geometry/spd/cheeger_log_euclidean/cheeger_spd_general.h"

// clang-format on

#endif   // __FDAPDE_MANIFOLD_GEOMETRY_MODULE_H__
