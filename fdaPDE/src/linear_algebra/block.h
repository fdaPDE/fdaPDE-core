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

#ifndef __FDAPDE_LINALG_BLOCK_H__
#define __FDAPDE_LINALG_BLOCK_H__

#include "header_check.h"

namespace fdapde {

// expression representing a dense sub-block (static or dynamic) of a MatrixExpr operand.
// Supports general blocks as well as row/column vector views.

template <int BlockRows_, int BlockCols_, typename XprType>
class MatrixBlock : public MatrixExpr<BlockRows_, BlockCols_, MatrixBlock<BlockRows_, BlockCols_, XprType>> {
    fdapde_static_assert(
      internals::is_dynamic_sized_v<XprType> ||
        ((BlockRows_ == Dynamic || (BlockRows_ > 0 && BlockRows_ <= XprType::Rows)) &&
         (BlockCols_ == Dynamic || (BlockCols_ > 0 && BlockCols_ <= XprType::Cols))),
      INVALID_STATIC_SIZED_BLOCK);
   public:
    using Base = MatrixExpr<BlockRows_, BlockCols_, MatrixBlock<BlockRows_, BlockCols_, XprType>>;
    using XprTypeNested = internals::ref_select_t<XprType>;
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = BlockRows_;
    static constexpr int Cols = BlockCols_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = XprType::ReadOnly;
    using assignment_executor = internals::generic_assignment_executor;

    // row/column constructor
    template <typename XprType_>
        requires(std::is_constructible_v<XprTypeNested, XprType_>)
    constexpr MatrixBlock(XprType_&& xpr, int i) :
        start_row_(BlockRows_ == 1 ? i : 0),
        start_col_(BlockCols_ == 1 ? i : 0),
        block_rows_(BlockRows_ == 1 ? 1 : xpr.rows()),
        block_cols_(BlockCols_ == 1 ? 1 : xpr.cols()),
        xpr_(std::forward<XprType_>(xpr)) {
        fdapde_static_assert(BlockRows_ == 1 || BlockCols_ == 1, THIS_METHOD_IS_FOR_ROW_AND_COLUMN_BLOCKS_ONLY);
        fdapde_assert(
          i >= 0 && ((BlockRows_ == 1 && i < xpr_.rows()) || (BlockCols_ == 1 && i < xpr_.cols())));
    }
    template <typename XprType_>
        requires(std::is_constructible_v<XprTypeNested, XprType_>)
    constexpr MatrixBlock(XprType_&& xpr, int start_row, int start_col) :
        start_row_(start_row),
        start_col_(start_col),
        block_rows_(BlockRows_),
        block_cols_(BlockCols_),
        xpr_(std::forward<XprType_>(xpr)) {
        fdapde_static_assert(
          BlockRows_ != Dynamic && BlockCols_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_BLOCKS_ONLY);
        fdapde_assert(
          start_row >= 0 && start_row + block_rows_ <= xpr_.rows() && start_col >= 0 &&
          start_col + block_cols_ <= xpr.cols());
    }
    template <typename XprType_>
        requires(std::is_constructible_v<XprTypeNested, XprType_>)
    constexpr MatrixBlock(XprType_&& xpr, int start_row, int start_col, int block_rows, int block_cols) :
        start_row_(start_row),
        start_col_(start_col),
        block_rows_(block_rows),
        block_cols_(block_cols),
        xpr_(std::forward<XprType_>(xpr)) {
        fdapde_static_assert(
          BlockRows_ == Dynamic && BlockCols_ == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_BLOCKS_ONLY);
        fdapde_assert(
          start_row >= 0 && start_row + block_rows_ <= xpr_.rows() && start_col >= 0 &&
          start_col + block_cols_ <= xpr.cols());
    }

    constexpr int rows() const { return Rows != Dynamic ? Rows : block_rows_; }
    constexpr int cols() const { return Cols != Dynamic ? Cols : block_cols_; }
    constexpr int size() const { return rows() * cols(); }
    constexpr Scalar operator()(int i, int j) const { return xpr_(start_row_ + i, start_col_ + j); }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(BlockRows_ == 1 || BlockCols_ == 1, THIS_METHOD_IS_FOR_ROW_AND_COLUMN_BLOCKS_ONLY);
        if constexpr (Rows == 1) return xpr_(start_row_, start_col_ + i);
        if constexpr (Cols == 1) return xpr_(start_row_ + i, start_col_);
    }
    constexpr Scalar& operator()(int i, int j) {
        fdapde_static_assert(XprType::ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION);
        return xpr_(start_row_ + i, start_col_ + j);
    }
    constexpr Scalar& operator[](int i) {
        fdapde_static_assert(BlockRows_ == 1 || BlockCols_ == 1, THIS_METHOD_IS_FOR_ROW_AND_COLUMN_BLOCKS_ONLY);
        fdapde_static_assert(XprType::ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION);
        if constexpr (Rows == 1) return xpr_(start_row_, start_col_ + i);
        if constexpr (Cols == 1) return xpr_(start_row_ + i, start_col_);
    }
    // inherit standard assignment operator
    using Base::operator=;
   private:
    int start_row_ = 0, start_col_ = 0;
    int block_rows_ = 0, block_cols_ = 0;
    XprTypeNested xpr_;
};
  
}   // namespace fdapde

#endif // __FDAPDE_LINALG_BLOCK_H__
