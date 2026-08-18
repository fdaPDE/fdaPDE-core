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

#ifndef __FDAPDE_LINALG_TRAITS_H__
#define __FDAPDE_LINALG_TRAITS_H__

#include "header_check.h"

#include <concepts>

namespace fdapde {
namespace internals {

template <typename T> using xpr_clean_t = std::remove_cvref_t<T>;

template <typename T>
concept matrix_expression = std::derived_from<xpr_clean_t<T>, MatrixExpr<xpr_clean_t<T>>>;

template <typename Arg, bool HasNestAsRef = requires { xpr_clean_t<Arg>::NestAsRef; }>
struct is_owning_rvalue_expression : std::false_type { };

template <typename Arg> struct is_owning_rvalue_expression<Arg, true> :
    std::bool_constant<!std::is_lvalue_reference_v<Arg> && (xpr_clean_t<Arg>::NestAsRef != 0)> { };

template <typename Arg>
inline constexpr bool is_owning_rvalue_expression_v = is_owning_rvalue_expression<Arg>::value;

template <typename Nested, typename Arg>
concept safely_nestable =
  std::is_constructible_v<Nested, Arg> &&
  (!std::is_reference_v<Nested> ||
   (std::is_lvalue_reference_v<Arg> && std::same_as<std::remove_cvref_t<Nested>, std::remove_cvref_t<Arg>>));

// sizing traits
template <typename XprType_> struct is_dynamic_sized {
    using XprType = std::decay_t<XprType_>;
    static constexpr bool value = XprType::Rows == Dynamic || XprType::Cols == Dynamic;
};
template <typename XprType> static constexpr bool is_dynamic_sized_v = is_dynamic_sized<XprType>::value;
template <typename XprType> struct is_static_sized {
    static constexpr bool value = !is_dynamic_sized_v<XprType>;
};
template <typename XprType> static constexpr bool is_static_sized_v = is_static_sized<XprType>::value;
// true if is possible to determine at compile time whether Lhs and Rhs have the same size, regardless of their shape
template <typename LhsXprType_, typename RhsXprType_> struct same_static_size {
   private:
    using Lhs = std::decay_t<LhsXprType_>;
    using Rhs = std::decay_t<RhsXprType_>;
   public:
    static constexpr bool value =
      is_static_sized_v<Lhs> && is_static_sized_v<Rhs> && (Lhs::Rows * Lhs::Cols == Rhs::Rows * Rhs::Cols);
};
template <typename LhsXprType, typename RhsXprType>
static constexpr bool same_static_size_v = same_static_size<LhsXprType, RhsXprType>::value;
// true if Lhs and Rhs might have the same size, regardless of their shape
template <typename LhsXprType, typename RhsXprType> struct same_static_size_weak {
    static constexpr bool value =
      is_dynamic_sized_v<LhsXprType> || is_dynamic_sized_v<RhsXprType> || same_static_size_v<LhsXprType, RhsXprType>;
};
template <typename LhsXprType, typename RhsXprType>
static constexpr bool same_static_size_weak_v = same_static_size_weak<LhsXprType, RhsXprType>::value;
 
// shaping traits
// true if is possibile to determine at compile time whether Lhs and Rhs have the same shape
template <typename LhsXprType_, typename RhsXprType_> struct same_static_shape {
   private:
    using Lhs = std::decay_t<LhsXprType_>;
    using Rhs = std::decay_t<RhsXprType_>;
   public:
    static constexpr bool value =
      !is_dynamic_sized_v<Lhs> && !is_dynamic_sized_v<Rhs> && (Lhs::Rows == Rhs::Rows && Lhs::Cols == Rhs::Cols);
};
template <typename LhsXprType, typename RhsXprType>
static constexpr bool same_static_shape_v = same_static_shape<LhsXprType, RhsXprType>::value;
// true if Lhs and Rhs might have the same shape (allows for Dynamic sized matrices)
template <typename LhsXprType, typename RhsXprType> struct same_static_shape_weak {
    static constexpr bool value =
      is_dynamic_sized_v<LhsXprType> || is_dynamic_sized_v<RhsXprType> || same_static_shape_v<LhsXprType, RhsXprType>;
};
template <typename LhsXprType, typename RhsXprType>
static constexpr bool same_static_shape_weak_v = same_static_shape_weak<LhsXprType, RhsXprType>::value;
// true if Xpr represents a vector expression
template <typename XprType_> struct is_vector_shaped {
    using XprType = std::decay_t<XprType_>;
    static constexpr bool value =
      (XprType::Rows == 1 || XprType::Cols == 1) && !(XprType::Rows == 1 && XprType::Cols == 1);
};
template <typename XprType> static constexpr bool is_vector_shaped_v = is_vector_shaped<XprType>::value;

// storage order promotion, defaults to RowMajor if no consensus
template <int LhsStorageOrder, int RhsStorageOrder> struct promote_storage_order {
    static constexpr int value = (LhsStorageOrder == RhsStorageOrder) ? LhsStorageOrder : RowMajor;
};
template <int LhsStorageOrder, int RhsStorageOrder>
static constexpr int promote_storage_order_v = promote_storage_order<LhsStorageOrder, RhsStorageOrder>::value;

// infer assignment loop
struct deleted_assignment_executor { };
template <typename XprType, typename = void> struct assignment_executor_of {
    using type = deleted_assignment_executor;   // if XprType does not defines an assignment loop, delete it
};
template <typename XprType>
struct assignment_executor_of<XprType, std::void_t<typename std::decay_t<XprType>::assignment_executor>> {
    using type = typename std::decay_t<XprType>::assignment_executor;
};
template <typename XprType> using assignment_executor_of_t = typename assignment_executor_of<XprType>::type;

}   // namespace internals
}   // namespace fdapde

#endif   //  __FDAPDE_LINALG_TRAITS_H__
