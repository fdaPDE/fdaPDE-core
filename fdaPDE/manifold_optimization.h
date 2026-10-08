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

#ifndef __FDAPDE_MANIFOLD_OPTIMIZATION_MODULE_H__
#define __FDAPDE_MANIFOLD_OPTIMIZATION_MODULE_H__

// clang-format off

#include "dense_linear_algebra.h"
#include <concepts>
#include <cstddef>
#include <optional>
#include <ranges>
#include <span>

#include "src/manifold_optimization/manifold.h"
#include "src/manifold_optimization/euclidean.h"
#include "src/manifold_optimization/product_geometry.h"
#include "src/manifold_optimization/evaluation_context.h"
#include "src/manifold_optimization/problem.h"
#include "src/manifold_optimization/armijo.h"
#include "src/manifold_optimization/steepest_descent.h"
#include "src/manifold_optimization/positive_definite_conjugate_gradient.h"
#include "src/manifold_optimization/truncated_conjugate_gradient.h"
#include "src/manifold_optimization/trust_region.h"
#include "src/manifold_optimization/weighted_karcher_mean.h"
#include "src/manifold_optimization/spd/log_euclidean/log_euclidean_spd.h"
#include "src/manifold_optimization/spd/log_cholesky/log_cholesky_spd.h"
#include "src/manifold_optimization/spd/affine_invariant/affine_invariant_spd.h"
#include "src/manifold_optimization/spd/bures_wasserstein/bures_wasserstein_spd.h"
#include "src/manifold_optimization/so.h"
#include "src/manifold_optimization/spd/cheeger_log_euclidean/cheeger_log_euclidean_spd.h"
#include "src/manifold_optimization/spd/cheeger_log_euclidean/cheeger_lift.h"
#include "src/manifold_optimization/spd/cheeger_log_euclidean/cheeger_spd_general.h"

// clang-format on

#endif   // __FDAPDE_MANIFOLD_OPTIMIZATION_MODULE_H__
