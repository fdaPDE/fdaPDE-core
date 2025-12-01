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

#include "math.h"

namespace fdapde {

// forward declaration
template <typename XprType> struct MatrixExpr;
template <typename XprType> struct MatrixCoeffWiseExpr;

// storage orders
[[maybe_unused]] constexpr int RowMajor = 0;
[[maybe_unused]] constexpr int ColMajor = 1;
// triangular views
[[maybe_unused]] constexpr int Upper = 0;       // lower triangular view of matrix
[[maybe_unused]] constexpr int Lower = 1;       // upper triangular view of matrix

[[maybe_unused]] static constexpr int LhsMode = 0;
[[maybe_unused]] static constexpr int RhsMode = 1;

namespace internals {

// tag types to enable/disable costly data integrity checks
struct checked_t { };
struct unchecked_t { };

}   // namespace internals

[[maybe_unused]] inline constexpr internals::checked_t   checked   {};
[[maybe_unused]] inline constexpr internals::unchecked_t unchecked {};
  
}   // namespace fdapde

#include "src/linear_algebra/traits.h"
#include "src/linear_algebra/matrix.h"
#include "src/linear_algebra/binary_op.h"
#include "src/linear_algebra/block.h"
#include "src/linear_algebra/unary_op.h"
#include "src/linear_algebra/ternary_op.h"
#include "src/linear_algebra/vectorwise.h"
#include "src/linear_algebra/diagonal.h"
#include "src/linear_algebra/orthogonal.h"
#include "src/linear_algebra/triangular.h"
#include "src/linear_algebra/bool.h"
#include "src/linear_algebra/coeffwise.h"
#include "src/linear_algebra/symmetric.h"
// #include "src/linear_algebra/skew.h"
#include "src/linear_algebra/permutation.h"
#include "src/linear_algebra/spd.h"

#include "src/linear_algebra/partial_piv_lu.h"
#include "src/linear_algebra/evd.h"
// #include "src/linear_algebra/qr.h"

// expression template system
#include "src/linear_algebra/xpr.h"


// clang-format on

#endif   // __FDAPDE_LINEAR_ALGEBRA_MODULE_H__
