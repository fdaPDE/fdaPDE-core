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

#include <iterator>

namespace fdapde::linalg {

// expression representing a dense sub-block (static or dynamic) of a MatrixExpr operand.
// Supports general blocks as well as row/column vector views.

template <int BlockRows_, int BlockCols_, typename XprType_>
class MatrixBlock : public MatrixExpr<MatrixBlock<BlockRows_, BlockCols_, XprType_>> {
   private:
    using Base = MatrixExpr<MatrixBlock<BlockRows_, BlockCols_, XprType_>>;
    using XprType = std::remove_reference_t<XprType_>;
    using XprTypeClean = std::remove_cv_t<XprType>;
    fdapde_static_assert(
      (BlockRows_ == Dynamic || BlockRows_ > 0) && (BlockCols_ == Dynamic || BlockCols_ > 0),
      INVALID_BLOCK_DIMENSIONS);
    fdapde_static_assert(
      internals::is_dynamic_sized_v<XprTypeClean> ||
        ((BlockRows_ == Dynamic || BlockRows_ <= XprTypeClean::Rows) &&
         (BlockCols_ == Dynamic || BlockCols_ <= XprTypeClean::Cols)),
      INVALID_BLOCK__STATIC_SIZES_DONT_FIT_WRAPPED_EXPRESSION);
    using XprTypeNested = fdapde::internals::ref_select_t<XprType_>;   // derive constness from wrapped expression
   public:
    using Scalar = typename XprTypeClean::Scalar;
    static constexpr int Rows = BlockRows_;
    static constexpr int Cols = BlockCols_;
    static constexpr int StorageOrder = XprTypeClean::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = std::is_const_v<XprType> || XprTypeClean::ReadOnly;
    using assignment_executor = internals::generic_assignment_executor;
    // iterator support (only for vector blocks)
    template <bool IsConst> struct block_iterator {
        using BlockType = std::conditional_t<IsConst, const MatrixBlock, MatrixBlock>;
       public:
        using reference = decltype(std::declval<BlockType&>()[0]);
        using value_type = std::remove_cv_t<std::remove_reference_t<reference>>;
        using pointer = std::conditional_t<
          std::is_reference_v<reference>, std::add_pointer_t<std::remove_reference_t<reference>>, void>;
        using size_type = std::size_t;
        using difference_type = std::ptrdiff_t;
        using iterator_concept = std::bidirectional_iterator_tag;
        using iterator_category = std::bidirectional_iterator_tag;

        constexpr block_iterator() : blk_(nullptr), i_(0) { }
        constexpr block_iterator(BlockType* blk, int i) : blk_(blk), i_(i) { }
        constexpr decltype(auto) operator*() const { return blk_->operator[](i_); }
        constexpr auto operator->() const requires(std::is_reference_v<reference>) {
            return std::addressof(blk_->operator[](i_));
        }
        constexpr block_iterator& operator++() {
            i_++;
            return *this;
        }
        constexpr block_iterator operator++(int) {
            block_iterator previous = *this;
            ++(*this);
            return previous;
        }
        constexpr block_iterator& operator--() {
            i_--;
            return *this;
        }
        constexpr block_iterator operator--(int) {
            block_iterator previous = *this;
            --(*this);
            return previous;
        }
        friend constexpr bool operator==(const block_iterator& lhs, const block_iterator& rhs) {
            return lhs.i_ == rhs.i_;
        }
       private:
        BlockType* blk_;
        int i_;
    };
    using iterator = block_iterator<false>;
    using const_iterator = block_iterator<true>;

    // row/column constructor
    constexpr MatrixBlock(const MatrixBlock&) = default;
    template <typename XprType__>
        requires(internals::safely_nestable<XprTypeNested, XprType__>)
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
        requires(internals::safely_nestable<XprTypeNested, XprType__>)
    constexpr MatrixBlock(XprType__&& xpr, int start_row, int start_col) :
        start_row_(start_row),
        start_col_(start_col),
        block_rows_(BlockRows_),
        block_cols_(BlockCols_),
        xpr_(std::forward<XprType__>(xpr)) {
        fdapde_static_assert(
          BlockRows_ != Dynamic && BlockCols_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_BLOCKS_ONLY);
        fdapde_assert(
          block_rows_ > 0 && block_cols_ > 0 && start_row >= 0 &&
          start_row + block_rows_ <= xpr_.rows() && start_col >= 0 &&
          start_col + block_cols_ <= xpr_.cols());
    }
    template <typename XprType__>
        requires(internals::safely_nestable<XprTypeNested, XprType__>)
    constexpr MatrixBlock(XprType__&& xpr, int start_row, int start_col, int block_rows, int block_cols) :
        start_row_(start_row),
        start_col_(start_col),
        block_rows_(block_rows),
        block_cols_(block_cols),
        xpr_(std::forward<XprType__>(xpr)) {
        fdapde_static_assert(
          BlockRows_ == Dynamic && BlockCols_ == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_BLOCKS_ONLY);
        const bool valid = block_rows_ > 0 && block_cols_ > 0 && start_row >= 0 &&
          start_row + block_rows_ <= xpr_.rows() && start_col >= 0 && start_col + block_cols_ <= xpr_.cols();
        fdapde_assert(valid);
        if (!valid) {
            start_row_ = 0;
            start_col_ = 0;
            block_rows_ = 0;
            block_cols_ = 0;
        }
    }

    constexpr int rows() const { return Rows != Dynamic ? Rows : block_rows_; }
    constexpr int cols() const { return Cols != Dynamic ? Cols : block_cols_; }
    constexpr int size() const { return rows() * cols(); }
    constexpr decltype(auto) operator()(int i, int j) const {
        fdapde_assert(i >= 0 && i < rows() && j >= 0 && j < cols());
        return std::as_const(xpr_)(start_row_ + i, start_col_ + j);
    }
    constexpr decltype(auto) operator[](int i) const {
        fdapde_static_assert(BlockRows_ == 1 || BlockCols_ == 1, THIS_METHOD_IS_FOR_ROW_AND_COLUMN_BLOCKS_ONLY);
        fdapde_assert(i >= 0 && i < size());
        if constexpr (Rows == 1) return std::as_const(xpr_)(start_row_, start_col_ + i);
        if constexpr (Cols == 1) return std::as_const(xpr_)(start_row_ + i, start_col_);
    }
    constexpr decltype(auto) operator()(int i, int j) {
        fdapde_assert(i >= 0 && i < rows() && j >= 0 && j < cols());
        return xpr_(start_row_ + i, start_col_ + j);
    }
    constexpr decltype(auto) operator[](int i) {
        fdapde_static_assert(BlockRows_ == 1 || BlockCols_ == 1, THIS_METHOD_IS_FOR_ROW_AND_COLUMN_BLOCKS_ONLY);
        fdapde_assert(i >= 0 && i < size());
        if constexpr (Rows == 1) return xpr_(start_row_, start_col_ + i);
        if constexpr (Cols == 1) return xpr_(start_row_ + i, start_col_);
    }
    // inherit standard assignment operator
    using Base::operator=;
    constexpr MatrixBlock& operator=(const MatrixBlock& rhs) & requires(ReadOnly == 0) {
        Base::template operator=<MatrixBlock>(rhs);
        return *this;
    }
    constexpr MatrixBlock operator=(const MatrixBlock& rhs) && requires(ReadOnly == 0) {
        static_cast<Base&>(*this).template operator=<MatrixBlock>(rhs);
        return *this;
    }
    template <typename Scalar_>
        requires(ReadOnly == 0 && std::is_constructible_v<Scalar, Scalar_>)
    constexpr MatrixBlock& operator=(const std::initializer_list<Scalar_>& data) & {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        const bool compatible = std::cmp_equal(this->size() FDAPDE_COMMA data.size());
        fdapde_assert(compatible);
        if (!compatible) return *this;
        int i = 0;
        for (const auto& v : data) { operator[](i++) = v; }
        return *this;
    }
    template <typename Scalar_>
        requires(ReadOnly == 0 && std::is_constructible_v<Scalar, Scalar_>)
    constexpr MatrixBlock operator=(const std::initializer_list<Scalar_>& data) && {
        static_cast<MatrixBlock&>(*this).operator=(data);
        return *this;
    }
    // iterators
    constexpr iterator begin() & requires(Rows == 1 || Cols == 1) { return iterator(this, 0); }
    constexpr iterator end() & requires(Rows == 1 || Cols == 1) { return iterator(this, size()); }
    constexpr const_iterator begin() const & requires(Rows == 1 || Cols == 1) { return const_iterator(this, 0); }
    constexpr const_iterator end() const & requires(Rows == 1 || Cols == 1) { return const_iterator(this, size()); }
    constexpr void begin() && = delete;
    constexpr void end() && = delete;
    constexpr void begin() const && = delete;
    constexpr void end() const && = delete;
   private:
    int start_row_ = 0, start_col_ = 0;
    int block_rows_ = 0, block_cols_ = 0;
    XprTypeNested xpr_;
};

}   // namespace fdapde::linalg

#endif // __FDAPDE_LINALG_BLOCK_H__
