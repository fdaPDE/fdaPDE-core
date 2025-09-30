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
template <typename XprType> struct TransposeOp : public MatrixExpr<XprType::Cols, XprType::Rows, TransposeOp<XprType>> {
    using Base = MatrixExpr<XprType::Cols, XprType::Rows, TransposeOp<XprType>>;
    using XprTypeNested = internals::ref_select_t<const XprType>;
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = XprType::Cols;
    static constexpr int Cols = XprType::Rows;
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
template <int Rows_, int Cols_, int StorageOrder_, typename XprType>
struct ReshapeOp : public MatrixExpr<Rows_, Cols_, ReshapeOp<Rows_, Cols_, StorageOrder_, XprType>> {
    using Base = MatrixExpr<Rows_, Cols_, ReshapeOp<Rows_, Cols_, StorageOrder_, XprType>>;
    using XprTypeNested = internals::ref_select_t<XprType>;
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = XprType::ReadOnly;

    template <typename XprType_>
        requires(std::is_constructible_v<XprTypeNested, XprType_>)
    constexpr explicit ReshapeOp(XprType_&& xpr) : rows_(Rows), cols_(Cols), xpr_(std::forward<XprType_>(xpr)) {
        fdapde_static_assert(Rows_ != Dynamic && Cols != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        fdapde_constexpr_assert(rows_ * cols_ == xpr.size());
    }
    template <typename XprType_>
        requires(std::is_constructible_v<XprTypeNested, XprType_>)
    constexpr ReshapeOp(XprType_&& xpr, int rows, int cols) :
        rows_(Rows == Dynamic ? rows : Rows), cols_(Cols == Dynamic ? cols : Cols), xpr_(std::forward<XprType_>(xpr)) {
        fdapde_constexpr_assert(rows_ * cols_ == xpr.size());
    }
    template <typename XprType_>
        requires(std::is_constructible_v<XprTypeNested, XprType_>)
    constexpr ReshapeOp(XprType_&& xpr, int rows) : ReshapeOp(xpr, rows, 1) {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_ONLY_FOR_ROW_OR_COLUMN_VECTORS);
    }
    // access
    constexpr Scalar operator()(int i, int j) const {
        const auto& [row, col] = reshaped_(i, j);
        return xpr_(row, col);
    }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_ONLY_FOR_ROW_OR_COLUMN_VECTORS);
        return operator()(i, 0);
    }
    constexpr Scalar& operator()(int i, int j) {
        const auto& [row, col] = reshaped_(i, j);
        return xpr_(row, col);
    }
    constexpr Scalar& operator[](int i) {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_ONLY_FOR_ROW_OR_COLUMN_VECTORS);
        return operator()(i, 0);
    }
    // observers
    constexpr int rows() const { return rows_; }
    constexpr int cols() const { return cols_; }
   private:
    std::pair<int, int> reshaped_(int i, int j) const {
        int k = i * cols_ + j * rows_;
        if constexpr (StorageOrder_ == RowMajor) return std::make_pair(k / xpr_.rows(), k % xpr_.rows());
        if constexpr (StorageOrder_ == ColMajor) return std::make_pair(k % xpr_.rows(), k / xpr_.rows());
    }
    int rows_, cols_;
    XprTypeNested xpr_;
};

// redux suppport. Reductions are unary operations which collapse a MatrixExpr operand into a single scalar
namespace internals {

// linear reduction loop on matrix expressions
template <typename XprType, typename Functor> struct matrix_linear_redux_executor {
    using XprTypeClean = std::decay_t<XprType>;
    using Scalar = decltype(std::declval<Functor>().operator()(
      std::declval<typename XprTypeClean::Scalar>(), std::declval<typename XprTypeClean::Scalar>()));

    static constexpr Scalar run(const XprType& xpr, Scalar init, Functor f) {
        fdapde_constexpr_assert(xpr.size() > 0);
        Scalar res = init;
        const int rows_ = xpr.rows();
	const int cols_ = xpr.cols();
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) { res = f(res, xpr(i, j)); }
        }
        return res;
    }
};

}   // namespace internals

template <typename XprType, typename Executor> struct MatrixReduxOp {
    using ExecutorReturnType = typename Executor::Scalar;
    using Scalar = typename XprType::Scalar;
    using XprTypeNested = internals::ref_select_t<const XprType>;

    template <typename XprType_>
        requires(std::is_constructible_v<XprTypeNested, XprType_>)
    constexpr explicit MatrixReduxOp(XprType_&& xpr) : xpr_(std::forward<XprType_>(xpr)) {
        fdapde_static_assert(
          std::is_convertible_v<ExecutorReturnType FDAPDE_COMMA Scalar>, INVALID_EXECUTIR_RETURN_TYPE);
    }
    template <typename ReduxOp> constexpr auto run(Scalar init, ReduxOp op) { return Executor::run(xpr_, init, op); }
   private:
    XprTypeNested xpr_;
};
  
}

#endif // __FDAPDE_LINALG_UNARY_OP_H__
