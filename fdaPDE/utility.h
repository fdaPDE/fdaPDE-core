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

#ifndef __FDAPDE_UTILITY_MODULE_H__
#define __FDAPDE_UTILITY_MODULE_H__

// clang-format off

// STL includes
#include <utility>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iostream>
#include <numeric>
#include <limits>
#include <memory>   // for std::shared_ptr
#include <optional>
#include <random>
#include <type_traits>
#include <typeindex>
#include <string>
#include <numbers>
#include <sstream>
#include <iomanip>
// common STL containers
#include <array>
#include <queue>
#include <stack>
#include <set>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <tuple>

// utils include
#include "src/utility/assert.h"
#include "src/utility/meta.h"
#include "src/utility/misc.h"

// define basic symbols
namespace fdapde {

[[maybe_unused]] constexpr int Dynamic     = -1;
[[maybe_unused]] constexpr int random_seed = -1;

// algorithm computation policies
[[maybe_unused]] static struct tag_exact     { } Exact;
[[maybe_unused]] static struct tag_not_exact { } NotExact;

}   // namespace fdapde

#include "src/utility/numeric.h"
// #include "src/utility/matrix/square_matrix_base.h"

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
   private:
    using XprTypeClean = std::decay_t<XprType>;
   public:
    static constexpr bool value = XprTypeClean::Rows == Dynamic || XprTypeClean::Cols == Dynamic;
};
template <typename XprType> static constexpr bool is_dynamic_sized_v = is_dynamic_sized<XprType>::value;

// if XprType has its NestAsRef bit set, sets type member type to XprType&, otherwise just repeats XprType
template <typename XprType, bool has_ref_bit> struct ref_select_impl;
template <typename XprType> struct ref_select_impl<XprType, true> {
   private:
    using XprTypeClean = std::decay_t<XprType>;
   public:
    using type = std::conditional_t<
      XprTypeClean::NestAsRef == 0, std::remove_reference_t<XprType>, std::add_lvalue_reference_t<XprType>>;
};
template <typename XprType> struct ref_select_impl<XprType, false> : std::type_identity<XprType> { };
template <typename XprType> struct ref_select {
    using type = ref_select_impl<XprType, requires(XprType) { XprType::NestAsRef; }>::type;
};
template <typename XprType> using ref_select_t = typename ref_select<XprType>::type;

}   // namespace internals
}

#include "src/utility/matrix/matrix.h"
// #include "src/utility/permutation_matrix.h"
#include "src/utility/matrix/diagonal.h"
#include "src/utility/matrix/triangular.h"

// special matrices
#include "src/utility/matrix/orthogonal.h"
#include "src/utility/matrix/symmetric.h"
#include "src/utility/matrix/skew.h"

#include "src/utility/matrix/matrix_expr.h"

#include "src/utility/matrix/evd.h"

#include "src/utility/binary.h"
#include "src/utility/mdarray.h"
#include "src/utility/binary_tree.h"
// #include "src/utility/positive_symmetric_matrix.h"

// clang-format on

#endif   // __FDAPDE_UTILITY_MODULE_H__
