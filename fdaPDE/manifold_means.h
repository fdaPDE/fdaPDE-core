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

#ifndef __FDAPDE_MANIFOLD_MEANS_MODULE_H__
#define __FDAPDE_MANIFOLD_MEANS_MODULE_H__

// clang-format off

#include "manifold_geometry.h"
#include "manifold_solvers.h"

#include "src/manifold/means/weighted_karcher_mean.h"
#include "src/manifold/means/log_euclidean_mean.h"
#include "src/manifold/means/log_cholesky_mean.h"
#include "src/manifold/means/affine_invariant_mean.h"
#include "src/manifold/means/bures_wasserstein_mean.h"
#include "src/manifold/means/so_mean.h"

// clang-format on

#endif   // __FDAPDE_MANIFOLD_MEANS_MODULE_H__
