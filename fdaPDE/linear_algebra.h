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

#ifndef __FDAPDE_LINEAR_ALGEBRA_MODULE_H__
#define __FDAPDE_LINEAR_ALGEBRA_MODULE_H__

// clang-format off

// include Eigen linear algebra library
#include <Eigen/Eigen>
#define __FDAPDE_HAS_EIGEN

// include required modules
#include "utility.h"

#include "src/linear_algebra/eigen_helper.h"
#include "src/linear_algebra/fspai.h"
#include "src/linear_algebra/kronecker.h"
#include "src/linear_algebra/lumping.h"
#include "src/linear_algebra/sparse_block_matrix.h"
#include "src/linear_algebra/woodbury.h"
//#include "src/linear_algebra/mumps.h"

// clang-format on

#endif   // __FDAPDE_LINEAR_ALGEBRA_MODULE_H__
