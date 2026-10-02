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

#ifndef __FDAPDE_FIELDS_MODULE_H__
#define __FDAPDE_FIELDS_MODULE_H__

// clang-format off

#include "linear_algebra.h"

// include required modules
#include "utility.h"

#include "src/fields/xpr_helper.h"
// import scalar fields logic first, as matrix field will depend on it
#include "src/fields/scalar_field.h"
#include "src/fields/divergence.h"
#include "src/fields/dot.h"
#include "src/fields/laplacian.h"
#include "src/fields/norm.h"
#include "src/fields/space_time_field.h"
// matrix field logic
#include "src/fields/jacobian.h"
#include "src/fields/matrix_field.h"
#include "src/fields/gradient.h"
#include "src/fields/hessian.h"

#include "src/fields/polynomial.h"
#include "src/fields/spline.h"

// clang-format on

#endif   // __FDAPDE_FIELDS_MODULE_H__
