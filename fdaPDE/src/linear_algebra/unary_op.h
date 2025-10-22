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

#ifndef __FDAPDE_LINALG_UNARY_OP_H__
#define __FDAPDE_LINALG_UNARY_OP_H__

#include "header_check.h"

namespace fdapde {

// this file contains all the expression nodes involving an operation applied on a single MatrixExpr operand

// expression of the transpose of a MatrixExpr operand
template <typename XprType> struct TransposeOp : public MatrixExpr<TransposeOp<XprType>> {
    using Base = MatrixExpr<TransposeOp<XprType>>;
    using XprTypeNested = internals::ref_select_t<const XprType>;
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = XprType::Cols;
    static constexpr int Cols = XprType::Rows;
    static constexpr int StorageOrder = XprType::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = std::is_const_v<std::remove_reference_t<XprType>>;

    template <typename XprType_>
        requires(std::is_constructible_v<XprTypeNested, XprType_>)
    explicit constexpr TransposeOp(XprType_&& xpr) : xpr_(std::forward<XprType_>(xpr)) { }
    constexpr Scalar operator()(int i, int j) const { return xpr_(j, i); }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(XprType::Cols == 1 || XprType::Rows == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return xpr_[i];
    }
    constexpr int rows() const { return Rows != Dynamic ? Rows : xpr_.cols(); }
    constexpr int cols() const { return Cols != Dynamic ? Cols : xpr_.rows(); }
   private:
    XprTypeNested xpr_;
};

// expression of a reshaped MatrixExpr operand. Reshaping modifes the expression dimensions without reallocating memory
template <int Rows_, int Cols_, typename XprType>
struct ReshapeOp : public MatrixExpr<ReshapeOp<Rows_, Cols_, XprType>> {
    using Base = MatrixExpr<ReshapeOp<Rows_, Cols_, XprType>>;
    using XprTypeNested = internals::ref_select_t<XprType>;
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = XprType::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = XprType::ReadOnly;

    template <typename XprType_>
        requires(std::is_constructible_v<XprTypeNested, XprType_>)
    constexpr explicit ReshapeOp(XprType_&& xpr) : rows_(Rows), cols_(Cols), xpr_(std::forward<XprType_>(xpr)) {
        fdapde_static_assert(Rows_ != Dynamic && Cols != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        fdapde_assert(rows_ * cols_ == xpr.size());
    }
    template <typename XprType_>
        requires(std::is_constructible_v<XprTypeNested, XprType_>)
    constexpr ReshapeOp(XprType_&& xpr, int rows, int cols) :
        rows_(Rows == Dynamic ? rows : Rows), cols_(Cols == Dynamic ? cols : Cols), xpr_(std::forward<XprType_>(xpr)) {
        fdapde_assert(rows_ * cols_ == xpr.size());
    }
    template <typename XprType_>
        requires(std::is_constructible_v<XprTypeNested, XprType_>)
    constexpr ReshapeOp(XprType_&& xpr, int rows) : ReshapeOp(xpr, rows, 1) {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_ONLY_FOR_ROW_OR_COLUMN_VECTORS);
    }
    // access
    constexpr decltype(auto) operator()(int i, int j) const {
        const auto [row, col] = reshaped_(i, j);
        return xpr_(row, col);
    }
    constexpr decltype(auto) operator[](int i) const {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_ONLY_FOR_ROW_OR_COLUMN_VECTORS);
        return Rows == 1 ? operator()(i, 0) : operator()(0, i);
    }
    constexpr decltype(auto) operator()(int i, int j) {
        fdapde_static_assert(ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION);
        const auto [row, col] = reshaped_(i, j);
        return xpr_(row, col);
    }
    constexpr decltype(auto) operator[](int i) {
        fdapde_static_assert(ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION);
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_ONLY_FOR_ROW_OR_COLUMN_VECTORS);
        return Rows == 1 ? operator()(i, 0) : operator()(0, i);
    }
    // observers
    constexpr int rows() const { return rows_; }
    constexpr int cols() const { return cols_; }
   private:
    std::pair<int, int> reshaped_(int i, int j) const {
        const int k = i * cols_ + j;
        return std::make_pair(k / xpr_.cols(), k % xpr_.cols());
    }
    int rows_, cols_;
    XprTypeNested xpr_;
};
  
// redux suppport. Reductions are unary operations which collapse a MatrixExpr operand into a single scalar
namespace internals {

// linear reduction loop on matrix expressions
struct matrix_redux_linear_executor {
    template <typename XprType_, typename Scalar, typename Functor>
    static constexpr auto run(XprType_&& xpr, Scalar init, Functor f) {
        using XprType = std::decay_t<XprType_>;
        fdapde_assert(xpr.size() > 0);
        Scalar res = init;
        const int rows_ = xpr.rows();
        const int cols_ = xpr.cols();
        // exploit cache-locality depending on storage order of target expression
        if constexpr (XprType::StorageOrder == RowMajor) {
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) { res = f(res, xpr(i, j)); }
            }
        } else {   // ColMajor
            for (int j = 0; j < cols_; ++j) {
                for (int i = 0; i < rows_; ++i) { res = f(res, xpr(i, j)); }
            }
        }
        return res;
    }
};

// boolean linear reduction loop on matrix expression
struct boolean_redux_linear_executor {
    // returns b at the first true occurence of f, otherwise returns !b
    template <typename XprType_, typename Functor>
    static constexpr auto run(XprType_&& xpr, bool b, Functor f) {
        using XprType = std::decay_t<XprType_>;
        fdapde_assert(xpr.size() > 0);
        const int rows_ = xpr.rows();
        const int cols_ = xpr.cols();
        // exploit cache-locality depending on storage order of target expression
        if constexpr (XprType::StorageOrder == RowMajor) {
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) {
                    if (bool(f(xpr(i, j)))) { return b; }
                }
            }
        } else {   // ColMajor
            for (int j = 0; j < cols_; ++j) {
                for (int i = 0; i < rows_; ++i) {
                    if (bool(f(xpr(i, j)))) { return b; }
                }
            }
        }
        return !b;
    }
};
  
}   // namespace internals
}   // namespace fdapde

#endif // __FDAPDE_LINALG_UNARY_OP_H__
