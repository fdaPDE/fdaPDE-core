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

#ifndef __FDAPDE_SPLINES_MODULE_H__
#define __FDAPDE_SPLINES_MODULE_H__

// clang-format off

// include required modules
#include "linear_algebra.h"    // pull Eigen first
#include "utility.h"
#include "fields.h"
#include "geometry.h"

namespace fdapde{

struct spline_tag { };

}   // namespace fdapde

// dof management logic
#include "src/splines/dof_handler.h"
// quadrature rules
#include "src/splines/sp_integration.h"
// assembly logic
#include "src/assembly.h"
#include "src/splines/sp_assembler_base.h"
#include "src/splines/sp_bilinear_form_assembler.h"
#include "src/splines/sp_linear_form_assembler.h"
// spline spaces
#include "src/splines/bspline_basis.h"
#include "src/splines/bs_space.h"
// weak forms
#include "src/splines/sp_objects.h"

// clang-format on

#endif   // __FDAPDE_SPLINES_MODULE_H__
