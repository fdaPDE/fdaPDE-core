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

#ifndef __FDAPDE_LINALG_TERNARY_OP_H__
#define __FDAPDE_LINALG_TERNARY_OP_H__

#include "header_check.h"

namespace fdapde {

// expression of a matrix ternary operator, i.e., A(i, j) ? B(i, j) : C(i, j)
/// @brief represents ternary op
template <typename ConditionXprType, typename LhsXprType, typename RhsXprType>
class TernaryOp : public MatrixExpr<TernaryOp<ConditionXprType, LhsXprType, RhsXprType>> {
    fdapde_static_assert(
      (internals::is_dynamic_sized_v<ConditionXprType> || internals::is_dynamic_sized_v<LhsXprType> ||
       internals::is_dynamic_sized_v<RhsXprType> ||
       (internals::same_static_shape_v<ConditionXprType, LhsXprType> &&
        internals::same_static_shape_v<ConditionXprType, RhsXprType>)),
      INVALID_TERNARY_OPERATION__MATRICES_OF_DIFFERENT_STATIC_SIZE);
   private:
    using Base = MatrixExpr<TernaryOp<ConditionXprType, LhsXprType, RhsXprType>>;
    using ConditionXprTypeNested = internals::ref_select_t<const ConditionXprType>;
    using LhsXprTypeNested = internals::ref_select_t<const LhsXprType>;
    using RhsXprTypeNested = internals::ref_select_t<const RhsXprType>;
   public:
    using Scalar = std::common_type_t<typename LhsXprType::Scalar, typename RhsXprType::Scalar>;
    static constexpr int Rows =
      (LhsXprType::Rows == Dynamic || RhsXprType::Rows == Dynamic) ? Dynamic : LhsXprType::Rows;
    static constexpr int Cols =
      (LhsXprType::Cols == Dynamic || RhsXprType::Cols == Dynamic) ? Dynamic : LhsXprType::Cols;
    static constexpr int StorageOrder = internals::promote_storage_order_v<
      ConditionXprType::StorageOrder,
      internals::promote_storage_order_v<LhsXprType::StorageOrder, RhsXprType::StorageOrder>>;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    /// @brief constructs ternary op from the supplied state
    template <typename ConditionXprType_, typename LhsXprType_, typename RhsXprType_>
        requires(internals::safely_nestable<ConditionXprTypeNested, ConditionXprType_> &&
                 internals::safely_nestable<LhsXprTypeNested, LhsXprType_> &&
                 internals::safely_nestable<RhsXprTypeNested, RhsXprType_>)
    constexpr TernaryOp(ConditionXprType_&& cond, LhsXprType_&& lhs, RhsXprType_&& rhs) :
        cond_(std::forward<ConditionXprType_>(cond)),
        lhs_(std::forward<LhsXprType_>(lhs)),
        rhs_(std::forward<RhsXprType_>(rhs)) {
        if constexpr (
          internals::is_dynamic_sized_v<ConditionXprType> || internals::is_dynamic_sized_v<LhsXprType> ||
          internals::is_dynamic_sized_v<RhsXprType>) {
            fdapde_assert(
              !(!std::cmp_equal(cond_.rows(), lhs_.rows()) || !std::cmp_equal(cond_.cols(), lhs_.cols()) ||
                !std::cmp_equal(cond_.rows(), rhs_.rows()) || !std::cmp_equal(cond_.cols(), rhs_.cols())),
              std::invalid_argument, "matrix ternary operation requires matching dimensions");
        }
    }
    /// @brief accesses or evaluates the requested coefficient
    constexpr Scalar operator()(int i, int j) const { return cond_(i, j) ? lhs_(i, j) : rhs_(i, j); }
    /// @brief accesses the requested vector coefficient
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(
          (ConditionXprType::Cols == 1 && LhsXprType::Cols == 1 && RhsXprType::Cols == 1) ||
            (ConditionXprType::Rows == 1 && LhsXprType::Rows == 1 && RhsXprType::Rows == 1),
          THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return cond_[i] ? lhs_[i] : rhs_[i];
    }
    /// @brief returns the row count
    constexpr int rows() const { return Rows != Dynamic ? Rows : cond_.rows(); }
    /// @brief returns the column count
    constexpr int cols() const { return Cols != Dynamic ? Cols : cond_.cols(); }
   private:
    ConditionXprTypeNested cond_;
    LhsXprTypeNested lhs_;
    RhsXprTypeNested rhs_;
};

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_TERNARY_OP_H__
