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

#ifndef __FDAPDE_PDE_MODULE_H__
#define __FDAPDE_PDE_MODULE_H__

#include "pde/pde.h"
#include "pde/differential_operators.h"
#include "pde/differential_expressions.h"

#include "utils/integration/integrator.h"
#include "utils/integration/integrator_tables.h"

#endif

#include "splines/spline_symbols.h"
#include "splines/spline_assembler.h"
#include "splines/basis/spline.h"
#include "splines/basis/spline_basis.h"
#include "splines/solvers/spline_solver_base.h"
#include "splines/solvers/spline_solver_selector.h"
#include "splines/solvers/spline_linear_elliptic_solver.h"
#include "splines/operators/reaction.h"
#include "splines/operators/bilaplacian.h"

#endif   // __FDAPDE_SPLINES_MODULE_H__
