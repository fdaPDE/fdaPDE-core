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

#ifndef __FDAPDE_PDE_MODULE_H__
#define __FDAPDE_PDE_MODULE_H__

#include "pde/pde.h"
#include "pde/differential_operators.h"
#include "pde/differential_expressions.h"

#include "utils/integration/integrator.h"
#include "utils/integration/integrator_tables.h"

#endif

#include "finite_elements/fem_symbols.h"
#include "finite_elements/fem_assembler.h"
#include "finite_elements/basis/multivariate_polynomial.h"
#include "finite_elements/basis/lagrangian_basis.h"
#include "finite_elements/solvers/fem_solver_base.h"
#include "finite_elements/solvers/fem_solver_selector.h"
#include "finite_elements/solvers/fem_linear_elliptic_solver.h"
#include "finite_elements/solvers/fem_linear_parabolic_solver.h"
#include "finite_elements/operators/laplacian.h"
#include "finite_elements/operators/diffusion.h"
#include "finite_elements/operators/advection.h"
#include "finite_elements/operators/reaction.h"
#include "finite_elements/operators/dt.h"

#endif   // __FDAPDE_FINITE_ELEMENTS_MODULE_H__
