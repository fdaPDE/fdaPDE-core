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
template <int Rows, int Cols, typename XprType> struct MatrixExpr;

// storage orders
[[maybe_unused]] constexpr int RowMajor = 0;
[[maybe_unused]] constexpr int ColMajor = 1;
// triangular views
[[maybe_unused]] constexpr int Upper = 0;       // lower triangular view of matrix
[[maybe_unused]] constexpr int Lower = 1;       // upper triangular view of matrix
[[maybe_unused]] constexpr int UnitUpper = 2;   // lower triangular view of matrix with ones on the diagonal
[[maybe_unused]] constexpr int UnitLower = 3;   // upper triangular view of matrix with ones on the diagonal

[[maybe_unused]] static constexpr int LhsMode = 0;
[[maybe_unused]] static constexpr int RhsMode = 1;

namespace internals {

// detects whether XprType represents a static sized or dynamic sized expression
template <typename XprType> struct is_dynamic_sized {
    static constexpr bool value = std::decay_t<XprType>::Rows == Dynamic || std::decay_t<XprType>::Cols == Dynamic;
};
template <typename XprType> static constexpr bool is_dynamic_sized_v = is_dynamic_sized<XprType>::value;
template <typename XprType> struct is_static_sized {
    static constexpr bool value = !is_dynamic_sized_v<XprType>;
};
template <typename XprType> static constexpr bool is_static_sized_v = is_static_sized<XprType>::value;
template <typename LhsXprType, typename RhsXprType> struct same_static_shape {
   private:
    using LhsXprTypeClean = std::decay_t<LhsXprType>;
    using RhsXprTypeClean = std::decay_t<RhsXprType>;
   public:
    static constexpr bool value =
      !is_dynamic_sized_v<LhsXprTypeClean> && !is_dynamic_sized_v<RhsXprTypeClean> &&
      (LhsXprTypeClean::Rows == RhsXprTypeClean::Rows && LhsXprTypeClean::Cols == RhsXprTypeClean::Cols);
};
template <typename LhsXprType, typename RhsXprType>
static constexpr bool same_static_shape_v = same_static_shape<LhsXprType, RhsXprType>::value;

}   // namespace internals
}   // namespace fdapde

#include "src/linear_algebra/matrix.h"
#include "src/linear_algebra/binary_op.h"
#include "src/linear_algebra/block.h"
#include "src/linear_algebra/unary_op.h"
#include "src/linear_algebra/ternary_op.h"
#include "src/linear_algebra/vectorwise_op.h"
#include "src/linear_algebra/diagonal.h"
#include "src/linear_algebra/orthogonal.h"
#include "src/linear_algebra/triangular.h"
#include "src/linear_algebra/symmetric.h"
#include "src/linear_algebra/bool.h"

// #include "src/linear_algebra/skew.h"
// #include "src/linear_algebra/permutation.h"
// #include "src/linear_algebra/spd.h"

// expression template system
#include "src/linear_algebra/xpr.h"

// algorithms
#include "src/linear_algebra/evd.h"
// #include "src/linear_algebra/partial_piv_lu.h"
// #include "src/linear_algebra/qr.h"

// clang-format on

#endif   // __FDAPDE_LINEAR_ALGEBRA_MODULE_H__
