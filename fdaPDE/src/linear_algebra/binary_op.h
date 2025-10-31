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

#ifndef __FDAPDE_LINALG_BINARY_OP_H__
#define __FDAPDE_LINALG_BINARY_OP_H__

#include "header_check.h"

namespace fdapde {

// this file contains all the expression nodes involving an operation applied on two generic MatrixExpr operands

// expression representing elementwise application of a binary operation to two MatrixExpr operands.
template <typename LhsXprType_, typename RhsXprType_, typename BinaryOp>
struct MatrixBinOp : public MatrixExpr<MatrixBinOp<LhsXprType_, RhsXprType_, BinaryOp>> {
   private:
    using LhsXprType = std::decay_t<LhsXprType_>;
    using RhsXprType = std::decay_t<RhsXprType_>;
    fdapde_static_assert(
      internals::same_static_shape_weak_v<LhsXprType_ FDAPDE_COMMA RhsXprType_>,
      INVALID_BINARY_OPERATION__OPERANDS_OF_DIFFERENT_STATIC_SIZE);
    using LhsXprTypeNested = internals::ref_select_t<const LhsXprType>;
    using RhsXprTypeNested = internals::ref_select_t<const RhsXprType>;
   public:
    using Scalar = promote_type_t<typename LhsXprType::Scalar, typename RhsXprType::Scalar>;
    static constexpr int Rows =
      (LhsXprType::Rows == Dynamic || RhsXprType::Rows == Dynamic) ? Dynamic : LhsXprType::Rows;
    static constexpr int Cols =
      (LhsXprType::Cols == Dynamic || RhsXprType::Cols == Dynamic) ? Dynamic : LhsXprType::Cols;
    static constexpr int StorageOrder =
      internals::promote_storage_order_v<LhsXprType::StorageOrder, RhsXprType::StorageOrder>;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename LhsXprType__, typename RhsXprType__>
        requires(std::is_constructible_v<LhsXprTypeNested, LhsXprType__> &&
                 std::is_constructible_v<RhsXprTypeNested, RhsXprType__>)
    constexpr MatrixBinOp(LhsXprType__&& lhs, RhsXprType__&& rhs, BinaryOp op) :
        lhs_(std::forward<LhsXprType__>(lhs)), rhs_(std::forward<RhsXprType__>(rhs)), op_(op) {
        if constexpr (internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType>) {
            fdapde_assert(
              std::cmp_equal(lhs_.rows() FDAPDE_COMMA rhs_.rows()) &&
              std::cmp_equal(lhs_.cols() FDAPDE_COMMA rhs_.cols()));
        }
    }
    constexpr Scalar operator()(int i, int j) const { return op_(lhs_(i, j), rhs_(i, j)); }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(
          (LhsXprType::Cols == 1 && RhsXprType::Cols == 1) || (LhsXprType::Rows == 1 && RhsXprType::Rows == 1),
          THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return op_(lhs_[i], rhs_[i]);
    }
    constexpr int rows() const { return Rows != Dynamic ? Rows : lhs_.rows(); }
    constexpr int cols() const { return Cols != Dynamic ? Cols : lhs_.cols(); }
   private:
    LhsXprTypeNested lhs_;
    RhsXprTypeNested rhs_;
    BinaryOp op_;
};

// definition of the linear vector-space structure of the set of M x N matrices
  
// matrix linear structure
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator+(const MatrixExpr<LhsXprType>& lhs, const MatrixExpr<RhsXprType>& rhs) {
    return MatrixBinOp<LhsXprType, RhsXprType, std::plus<>>(lhs.derived(), rhs.derived(), std::plus<>());
}
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator-(const MatrixExpr<LhsXprType>& lhs, const MatrixExpr<RhsXprType>& rhs) {
    return MatrixBinOp<LhsXprType, RhsXprType, std::minus<>>(lhs.derived(), rhs.derived(), std::minus<>());
}

// expression of the scalar-matrix multiplication between a scalar and a MatrixExpr
template <typename XprType_, typename ScalarType>
struct MatrixScalarMultiplicationOp : public MatrixExpr<MatrixScalarMultiplicationOp<XprType_, ScalarType>> {
   private:
    using XprType = std::decay_t<XprType_>;
    using XprTypeNested = internals::ref_select_t<const XprType>;
   public:
    using Scalar = promote_type_t<typename XprType::Scalar, ScalarType>;
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    static constexpr int StorageOrder = XprType::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename XprType__>
        requires(std::is_constructible_v<XprType, XprType__>)
    constexpr MatrixScalarMultiplicationOp(XprType__&& xpr, ScalarType s) :
        xpr_(std::forward<XprType__>(xpr)), s_(s) { }
    constexpr Scalar operator()(int i, int j) const { return xpr_(i, j) * s_; }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(XprType::Rows == 1 || XprType::Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return xpr_[i] * s_;
    }
    constexpr int rows() const { return Rows != Dynamic ? Rows : xpr_.rows(); }
    constexpr int cols() const { return Cols != Dynamic ? Cols : xpr_.cols(); }
   protected:
    XprTypeNested xpr_;
    ScalarType s_;
};

// multiplication by scalar
template <typename XprType, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator*(const MatrixExpr<XprType>& lhs, ScalarType rhs) {
    return MatrixScalarMultiplicationOp<XprType, ScalarType>(lhs.derived(), rhs);
}
template <typename XprType, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator*(ScalarType lhs, const MatrixExpr<XprType>& rhs) {
    return rhs * lhs;
}
template <typename XprType, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator/(const MatrixExpr<XprType>& lhs, ScalarType rhs) {
    return MatrixScalarMultiplicationOp<XprType, ScalarType>(lhs.derived(), ScalarType(1) / rhs);
}

// expression of the matrix-product of two MatrixExpr operands
template <typename LhsXprType_, typename RhsXprType_, typename Executor>
struct MatrixMultiplicationOp : public MatrixExpr<MatrixMultiplicationOp<LhsXprType_, RhsXprType_, Executor>> {
   private:
    using LhsXprType = std::decay_t<LhsXprType_>;
    using RhsXprType = std::decay_t<RhsXprType_>;
    fdapde_static_assert(
      (internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType> ||
       LhsXprType::Cols == RhsXprType::Rows),
      INVALID_PRODUCT__OPERANDS_HAVE_INCOMPATIBLE_STATIC_SIZE);
    using LhsXprTypeNested = internals::ref_select_t<const LhsXprType>;
    using RhsXprTypeNested = internals::ref_select_t<const RhsXprType>;
   public:
    using Scalar = promote_type_t<typename LhsXprType::Scalar, typename RhsXprType::Scalar>;
    static constexpr int Rows = LhsXprType::Rows;
    static constexpr int Cols = RhsXprType::Cols;
    static constexpr int StrageOrder =
      internals::promote_storage_order_v<LhsXprType::StorageOrder, RhsXprType::StorageOrder>;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename LhsXprType__, typename RhsXprType__>
        requires(std::is_constructible_v<LhsXprTypeNested, LhsXprType__> &&
                 std::is_constructible_v<RhsXprTypeNested, RhsXprType__>)
    constexpr MatrixMultiplicationOp(LhsXprType__&& lhs, RhsXprType__&& rhs) :
        lhs_(std::forward<LhsXprType__>(lhs)), rhs_(std::forward<RhsXprType__>(rhs)) {
        if constexpr (internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType>) {
            fdapde_assert(std::cmp_equal(lhs_.cols() FDAPDE_COMMA rhs_.rows()));
        }
    }
    constexpr Scalar operator()(int i, int j) const { return Executor::run(i, j, lhs_, rhs_); }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(
          LhsXprType::Rows == 1 || RhsXprType::Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return Executor::run(LhsXprType::Rows == 1 ? 0 : i, RhsXprType::Cols == 1 ? 0 : i, lhs_, rhs_);
    }
    constexpr int rows() const { return Rows != Dynamic ? Rows : lhs_.rows(); }
    constexpr int cols() const { return Cols != Dynamic ? Cols : rhs_.cols(); }
   protected:
    LhsXprTypeNested lhs_;
    RhsXprTypeNested rhs_;
};

// definition of generic product executors
namespace internals {

// general dense matrix-matrix product loop
// specialization of this template induce matrix-specific product loops
struct generic_matrix_product_executor {
    template <typename LhsXprType_, typename RhsXprType_>
    static constexpr auto run(int i, int j, const LhsXprType_& lhs, const RhsXprType_& rhs) {
        using LhsXprType = std::decay_t<LhsXprType_>;
        using RhsXprType = std::decay_t<RhsXprType_>;
        using Scalar = promote_type_t<typename LhsXprType::Scalar, typename RhsXprType::Scalar>;
        Scalar prod = 0;
        for (int k = 0, size = lhs.cols(); k < size; ++k) { prod += lhs(i, k) * rhs(k, j); }
        return prod;
    }
};

// outer product v * v^\top executor
struct outer_product_executor {
    template <typename LhsXprType_, typename RhsXprType_>
    static constexpr auto run(int i, int j, const LhsXprType_& lhs, const RhsXprType_& rhs) {
        using LhsXprType = std::decay_t<LhsXprType_>;
        using RhsXprType = std::decay_t<RhsXprType_>;
        fdapde_static_assert(
          LhsXprType::Cols == 1 && RhsXprType::Rows == 1, INVALID_OUTER_PRODUCT__OPERANDS_HAVE_INCOMPATIBLE_SHAPES);
        return lhs[i] * rhs[j];
    }
};
  
}   // namespace internals

// generic matrix-matrix product
template <typename LhsXprType_, typename RhsXprType_>
constexpr auto operator*(const MatrixExpr<LhsXprType_>& lhs, const MatrixExpr<RhsXprType_>& rhs) {
    using LhsXprType = std::decay_t<LhsXprType_>;
    using RhsXprType = std::decay_t<RhsXprType_>;
    if constexpr (LhsXprType::Cols == 1 && RhsXprType::Rows == 1) {   // outer product v * u^\top
        return MatrixMultiplicationOp<LhsXprType, RhsXprType, internals::outer_product_executor> {
          lhs.derived(), rhs.derived()};
    } else {
        return MatrixMultiplicationOp<LhsXprType, RhsXprType, internals::generic_matrix_product_executor> {
          lhs.derived(), rhs.derived()};
    }
}

// specialized binary operations

// expression of the dense kronecker tensor product between two MatrixExpr operands
template <typename LhsXprType_, typename RhsXprType_>
struct MatrixKroneckerProductOp : public MatrixExpr<MatrixKroneckerProductOp<LhsXprType_, RhsXprType_>> {
   private:
    using LhsXprType = std::decay_t<LhsXprType_>;
    using RhsXprType = std::decay_t<RhsXprType_>;
    using LhsXprTypeNested = internals::ref_select_t<const LhsXprType>;
    using RhsXprTypeNested = internals::ref_select_t<const RhsXprType>;
   public:
    using Scalar = promote_type_t<typename LhsXprType::Scalar, typename RhsXprType::Scalar>;
    static constexpr int Rows =
      LhsXprType::Rows == Dynamic || RhsXprType::Rows == Dynamic ? Dynamic : LhsXprType::Rows * RhsXprType::Rows;
    static constexpr int Cols =
      LhsXprType::Cols == Dynamic || RhsXprType::Cols == Dynamic ? Dynamic : LhsXprType::Cols * RhsXprType::Cols;
    static constexpr int StrageOrder =
      internals::promote_storage_order_v<LhsXprType::StorageOrder, RhsXprType::StorageOrder>;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename LhsXprType__, typename RhsXprType__>
        requires(std::is_constructible_v<LhsXprType, LhsXprType__> && std::is_constructible_v<RhsXprType, RhsXprType__>)
    constexpr MatrixKroneckerProductOp(LhsXprType__&& lhs, RhsXprType__&& rhs) :
        lhs_(std::forward<LhsXprType__>(lhs)), rhs_(std::forward<RhsXprType__>(rhs)) { }
    constexpr Scalar operator()(int i, int j) const {
        const int h = RhsXprType::Rows != Dynamic ? RhsXprType::Rows : rhs_.rows();
        const int k = RhsXprType::Cols != Dynamic ? RhsXprType::Cols : rhs_.cols();
        // compute offsets in operand matrices
        int col_lhs = j / k, row_lhs = i / h;
        int col_rhs = j % k, row_rhs = i % h;
        return lhs_(row_lhs, col_lhs) * rhs_(row_rhs, col_rhs);
    }
    constexpr int rows() const {
        return (LhsXprType::Rows != Dynamic ? LhsXprType::Rows : lhs_.rows()) *
               (RhsXprType::Rows != Dynamic ? RhsXprType::Rows : rhs_.rows());
    }
    constexpr int cols() const {
        return (LhsXprType::Cols != Dynamic ? LhsXprType::Cols : lhs_.cols()) *
               (RhsXprType::Cols != Dynamic ? RhsXprType::Cols : rhs_.cols());
    }
   private:
    LhsXprTypeNested lhs_;
    RhsXprTypeNested rhs_;
};
template <typename LhsXprType, typename RhsXprType>
constexpr MatrixKroneckerProductOp<LhsXprType, RhsXprType>
kron(const MatrixExpr<LhsXprType>& op1, const MatrixExpr<RhsXprType>& op2) {
    return MatrixKroneckerProductOp<LhsXprType, RhsXprType> {op1.derived(), op2.derived()};
}

// expression of the cross product between two vector expressions
template <typename LhsXprType_, typename RhsXprType_>
struct MatrixCrossProductOp : public MatrixExpr<MatrixCrossProductOp<LhsXprType_, RhsXprType_>> {
   private:
    using LhsXprType = std::decay_t<LhsXprType_>;
    using RhsXprType = std::decay_t<RhsXprType_>;
    fdapde_static_assert(
      (internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType> ||
       (LhsXprType::Rows == 3 && RhsXprType::Rows == 3 && LhsXprType::Cols == 1 && RhsXprType::Cols == 1)),
      THIS_CLASS_IS_FOR_THREE_DIMENSIONAL_VECTORS_ONLY);
    using LhsXprTypeNested = internals::ref_select_t<const LhsXprType>;
    using RhsXprTypeNested = internals::ref_select_t<const RhsXprType>;
   public:
    using Scalar = promote_type_t<typename LhsXprType::Scalar, typename RhsXprType::Scalar>;
    static constexpr int Rows = 3;
    static constexpr int Cols = 1;
    static constexpr int StrageOrder =
      internals::promote_storage_order_v<LhsXprType::StorageOrder, RhsXprType::StorageOrder>;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename LhsXprType__, typename RhsXprType__>
        requires(std::is_constructible_v<LhsXprType, LhsXprType__> && std::is_constructible_v<RhsXprType, RhsXprType__>)
    constexpr MatrixCrossProductOp(LhsXprType__&& lhs, RhsXprType__&& rhs) :
        lhs_(std::forward<LhsXprType__>(lhs)), rhs_(std::forward<RhsXprType__>(rhs)) {
        if constexpr (internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType>) {
            fdapde_assert(lhs_.rows() == 3 && rhs_.rows() == 3 && lhs_.cols() == 1 && rhs_.cols() == 1);
        }
    }
    constexpr Scalar operator()(int i, [[maybe_unused]] int j) const {
        fdapde_assert(i >= 0 && i < rows() && j >= 0 && j < cols());
        return operator[](i);
    }
    constexpr Scalar operator[](int i) const {
        fdapde_assert(i >= 0 && i < rows());
        if (i == 0) { return lhs_[1] * rhs_[2] - lhs_[2] * rhs_[1]; }
        if (i == 1) { return lhs_[2] * rhs_[0] - lhs_[0] * rhs_[2]; }
        return lhs_[0] * rhs_[1] - lhs_[1] * rhs_[0];
    }
    constexpr int rows() const { return 3; }
    constexpr int cols() const { return 1; }
   private:
    LhsXprTypeNested lhs_;
    RhsXprTypeNested rhs_;
};

}   // namespace fdapde

#endif // __FDAPDE_LINALG_BINARY_OP_H__
