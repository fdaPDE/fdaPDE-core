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

#ifndef __FDAPDE_FINITE_ELEMENTS_MODULE_H__
#define __FDAPDE_FINITE_ELEMENTS_MODULE_H__

// clang-format off

// include required modules
#include "linear_algebra.h"    // pull Eigen first
#include "utility.h"
#include "fields.h"
#include "geometry.h"

namespace fdapde{

struct finite_element_tag { };

}   // namespace fdapde

// dof management logic
#include "src/finite_elements/dof_segment.h"
#include "src/finite_elements/dof_tetrahedron.h"
#include "src/finite_elements/dof_triangle.h"
#include "src/finite_elements/dof_constraints.h"
#include "src/finite_elements/dof_handler.h"
// quadrature rules
#include "src/finite_elements/fe_integration.h"
// assembly logic
#include "src/assembly.h"
#include "src/finite_elements/fe_assembler_base.h"
#include "src/finite_elements/fe_bilinear_form_assembler.h"
#include "src/finite_elements/fe_linear_form_assembler.h"
#include "src/finite_elements/fe_mass_assembler.h"
#include "src/finite_elements/fe_evaluator.h"
// finite element spaces
#include "src/finite_elements/lagrange_basis.h"
#include "src/finite_elements/fe_p.h"
#include "src/finite_elements/fe_space.h"
// weak forms
#include "src/finite_elements/fe_objects.h"
// tensor product spaces
#include "src/tp_space.h"

// clang-format on

#endif   // __FDAPDE_FINITE_ELEMENTS_MODULE_H__
