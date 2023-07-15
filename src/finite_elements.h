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

#ifndef __FDAPDE_FINITE_ELEMENTS_MOUDLE_H__
#define __FDAPDE_FINITE_ELEMENTS_MOUDLE_H__

#include "finite_elements/pde.h"
#include "finite_elements/assembler.h"
#include "finite_elements/basis/multivariate_polynomial.h"
#include "finite_elements/basis/lagrangian_basis.h"
#include "finite_elements/basis/basis_cache.h"
#include "finite_elements/integration/integrator.h"
#include "finite_elements/integration/integrator_tables.h"
#include "finite_elements/solvers/fem_solver_base.h"
#include "finite_elements/solvers/fem_standard_space_solver.h"
#include "finite_elements/solvers/fem_standard_spacetime_solver.h"
#include "finite_elements/operators/bilinear_form_expressions.h"
#include "finite_elements/operators/bilinear_form_traits.h"
#include "finite_elements/operators/laplacian.h"
#include "finite_elements/operators/divergence.h"
#include "finite_elements/operators/gradient.h"
#include "finite_elements/operators/identity.h"
#include "finite_elements/operators/dt.h"

#endif   // __FDAPDE_FINITE_ELEMENTS_MOUDLE_H__
