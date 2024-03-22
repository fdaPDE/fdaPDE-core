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

#include <Eigen/Core>
#include <Eigen/Sparse>

#include "linear_algebra/binary_matrix.h"
#include "linear_algebra/constexpr_matrix.h"
#include "linear_algebra/kronecker.h"
#include "linear_algebra/lumping.h"
#include "linear_algebra/smw.h"
#include "linear_algebra/sparse_block_matrix.h"
#include "linear_algebra/fspai.h"
#include "linear_algebra/lumping.h"

#include "fdaPDE/linear_algebra/randomized_algorithms/rand_range_finder.h"
#include "fdaPDE/linear_algebra/randomized_algorithms/randomized_svd.h"
#include "fdaPDE/linear_algebra/randomized_algorithms/rand_nys_approximation.h"


#endif   // __FDAPDE_LINEAR_ALGEBRA_MODULE_H__
