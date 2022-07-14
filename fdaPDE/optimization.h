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

#ifndef __FDAPDE_OPTIMIZATION_MODULE_H__
#define __FDAPDE_OPTIMIZATION_MODULE_H__

// clang-format off

// include required modules
#include "linear_algebra.h"    // pull Eigen first
#include "utility.h"
#include "fields.h"

// callbacks
#include "src/optimization/callbacks.h"
#include "src/optimization/backtracking_line_search.h"
#include "src/optimization/wolfe_line_search.h"
// algorithms
#include "src/optimization/grid.h"
#include "src/optimization/newton.h"
#include "src/optimization/gradient_descent.h"
#include "src/optimization/bfgs.h"

// clang-format on

#endif   // __FDAPDE_OPTIMIZATION_MODULE_H__
