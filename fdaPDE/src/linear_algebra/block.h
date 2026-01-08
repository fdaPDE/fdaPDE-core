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

template <int BlockRows_, int BlockCols_, typename XprType_>
class MatrixBlock : public MatrixExpr<MatrixBlock<BlockRows_, BlockCols_, XprType_>> {
   private:
    using Base = MatrixExpr<MatrixBlock<BlockRows_, BlockCols_, XprType_>>;
    using XprType = std::decay_t<XprType_>;
    fdapde_static_assert(
      internals::is_dynamic_sized_v<XprType> ||
        ((BlockRows_ == Dynamic || (BlockRows_ > 0 && BlockRows_ <= XprType::Rows)) &&
         (BlockCols_ == Dynamic || (BlockCols_ > 0 && BlockCols_ <= XprType::Cols))),
      INVALID_BLOCK__STATIC_SIZES_DONT_FIT_WRAPPED_EXPRESSION);
    using XprTypeNested = internals::ref_select_t<XprType_>;   // derive constness from wrapped expression
   public:
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = BlockRows_;
    static constexpr int Cols = BlockCols_;
    static constexpr int StorageOrder = XprType::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = XprType::ReadOnly;
    using assignment_executor = internals::generic_assignment_executor;
    // iterator support (only for vector blocks)
    struct iterator {
        fdapde_static_assert(Rows == 1 || Cols == 1, ITERATOR_SUPPORT_IS_FOR_VECTOR_SHAPED_EXPRESSIONS_ONLY);
       public:
        using value_type = decltype(std::declval<MatrixBlock>().operator[](std::declval<int>()));
        using pointer = std::add_pointer_t<value_type>;
        using reference = std::add_lvalue_reference_t<value_type>;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using iterator_category = std::bidirectional_iterator_tag;
      
        constexpr iterator() : blk_(nullptr), i_(0) { }
        constexpr iterator(const MatrixBlock* blk) : blk_(blk), i_(0) { }
        constexpr iterator(const MatrixBlock* blk, int i) : blk_(blk), i_(i) { }
        reference operator*() { return blk_->operator[](i_); }
        const reference operator*() const { return blk_->operator[](i_); }
        pointer operator->() { return std::addressof(blk_->operator[](i_)); }
        const pointer operator->() const { return std::addressof(blk_->operator[](i_)); }
        iterator& operator++() {
            i_++;
            return *this;
        }
        iterator& operator--() {
            i_--;
            return *this;
        }
        friend constexpr bool operator==(const iterator& lhs, const iterator& rhs) { return lhs.i_ == rhs.i_; }
        friend constexpr bool operator!=(const iterator& lhs, const iterator& rhs) { return lhs.i_ != rhs.i_; }
       private:
        const MatrixBlock* blk_;
        int i_;
    };

    // row/column constructor
    template <typename XprType__>
        requires(std::is_constructible_v<XprTypeNested, XprType__>)
    constexpr MatrixBlock(XprType__&& xpr, int i) :
        start_row_(BlockRows_ == 1 ? min(i, xpr.rows() - 1) : 0),
        start_col_(BlockCols_ == 1 ? min(i, xpr.cols() - 1) : 0),
        block_rows_(BlockRows_ == 1 ? 1 : xpr.rows()),
        block_cols_(BlockCols_ == 1 ? 1 : xpr.cols()),
        xpr_(std::forward<XprType__>(xpr)) {
        fdapde_static_assert(BlockRows_ == 1 || BlockCols_ == 1, THIS_METHOD_IS_FOR_ROW_AND_COLUMN_BLOCKS_ONLY);
        fdapde_assert(
          i >= 0 && ((BlockRows_ == 1 && i < xpr_.rows()) || (BlockCols_ == 1 && i < xpr_.cols())));
    }
    template <typename XprType__>
        requires(std::is_constructible_v<XprTypeNested, XprType__>)
    constexpr MatrixBlock(XprType__&& xpr, int start_row, int start_col) :
        start_row_(start_row),
        start_col_(start_col),
        block_rows_(BlockRows_),
        block_cols_(BlockCols_),
        xpr_(std::forward<XprType__>(xpr)) {
        fdapde_static_assert(
          BlockRows_ != Dynamic && BlockCols_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_BLOCKS_ONLY);
        fdapde_assert(
          start_row >= 0 && start_row + block_rows_ <= xpr_.rows() && start_col >= 0 &&
          start_col + block_cols_ <= xpr.cols());
    }
    template <typename XprType__>
        requires(std::is_constructible_v<XprTypeNested, XprType__>)
    constexpr MatrixBlock(XprType__&& xpr, int start_row, int start_col, int block_rows, int block_cols) :
        start_row_(start_row),
        start_col_(start_col),
        block_rows_(block_rows),
        block_cols_(block_cols),
        xpr_(std::forward<XprType__>(xpr)) {
        fdapde_static_assert(
          BlockRows_ == Dynamic && BlockCols_ == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_BLOCKS_ONLY);
        fdapde_assert(
          start_row >= 0 && start_row + block_rows_ <= xpr_.rows() && start_col >= 0 &&
          start_col + block_cols_ <= xpr.cols());
    }

    constexpr int rows() const { return Rows != Dynamic ? Rows : block_rows_; }
    constexpr int cols() const { return Cols != Dynamic ? Cols : block_cols_; }
    constexpr int size() const { return rows() * cols(); }
    constexpr decltype(auto) operator()(int i, int j) const { return xpr_(start_row_ + i, start_col_ + j); }
    constexpr decltype(auto) operator[](int i) const {
        fdapde_static_assert(BlockRows_ == 1 || BlockCols_ == 1, THIS_METHOD_IS_FOR_ROW_AND_COLUMN_BLOCKS_ONLY);
        if constexpr (Rows == 1) return xpr_(start_row_, start_col_ + i);
        if constexpr (Cols == 1) return xpr_(start_row_ + i, start_col_);
    }
    constexpr decltype(auto) operator()(int i, int j) { return xpr_(start_row_ + i, start_col_ + j); }
    constexpr decltype(auto) operator[](int i) {
        fdapde_static_assert(BlockRows_ == 1 || BlockCols_ == 1, THIS_METHOD_IS_FOR_ROW_AND_COLUMN_BLOCKS_ONLY);
        if constexpr (Rows == 1) return xpr_(start_row_, start_col_ + i);
        if constexpr (Cols == 1) return xpr_(start_row_ + i, start_col_);
    }
    // inherit standard assignment operator
    using Base::operator=;
    template <typename Scalar_>
        requires(std::is_constructible_v<Scalar, Scalar_>)
    constexpr MatrixBlock& operator=(const std::initializer_list<Scalar_>& data) {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        fdapde_assert(std::cmp_equal(this->size() FDAPDE_COMMA data.size()));
        int i = 0;
        for (const auto& v : data) { operator[](i++) = v; }
        return *this;
    }
    // iterators
    iterator begin() const { return iterator(this, 0); }
    iterator end() const { return iterator(this, block_rows_); }
   private:
    int start_row_ = 0, start_col_ = 0;
    int block_rows_ = 0, block_cols_ = 0;
    XprTypeNested xpr_;
};
  
}   // namespace fdapde

#endif // __FDAPDE_LINALG_BLOCK_H__
