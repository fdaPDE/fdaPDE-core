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
template <typename LhsXprType, typename RhsXprType, typename BinaryOperation>
struct MatrixBinOp :
    public MatrixExpr<LhsXprType::Rows, LhsXprType::Cols, MatrixBinOp<LhsXprType, RhsXprType, BinaryOperation>> {
    fdapde_static_assert(
      internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType> ||
        (LhsXprType::Rows == RhsXprType::Rows && LhsXprType::Cols == RhsXprType::Cols),
      INVALID_BINARY_OPERATION__MATRICES_OF_DIFFERENT_STATIC_SIZE);
    using Base = MatrixExpr<LhsXprType::Rows, LhsXprType::Cols, MatrixBinOp<LhsXprType, RhsXprType, BinaryOperation>>;
    using LhsXprTypeNested = internals::ref_select_t<const LhsXprType>;
    using RhsXprTypeNested = internals::ref_select_t<const RhsXprType>;
    using Scalar = decltype(std::declval<BinaryOperation>().operator()(
      std::declval<typename LhsXprType::Scalar>(), std::declval<typename RhsXprType::Scalar>()));
    static constexpr int Rows =
      (LhsXprType::Rows == Dynamic || RhsXprType::Rows == Dynamic) ? Dynamic : LhsXprType::Rows;
    static constexpr int Cols =
      (LhsXprType::Cols == Dynamic || RhsXprType::Cols == Dynamic) ? Dynamic : LhsXprType::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename LhsXprType_, typename RhsXprType_>
        requires(std::is_constructible_v<LhsXprTypeNested, LhsXprType_> &&
                 std::is_constructible_v<RhsXprTypeNested, RhsXprType_>)
    constexpr MatrixBinOp(LhsXprType_&& lhs, RhsXprType_&& rhs, BinaryOperation op) :
        lhs_(std::forward<LhsXprType_>(lhs)), rhs_(std::forward<RhsXprType_>(rhs)), op_(op) {
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
    BinaryOperation op_;
};

// definition of the linear vector-space structure of the set of M x N matrices
  
// matrix linear structure
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator+(
  const MatrixExpr<LhsXprType::Rows, LhsXprType::Cols, LhsXprType>& lhs,
  const MatrixExpr<RhsXprType::Rows, RhsXprType::Cols, RhsXprType>& rhs) {
    return MatrixBinOp<LhsXprType, RhsXprType, std::plus<>>(lhs.derived(), rhs.derived(), std::plus<>());
}
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator-(
  const MatrixExpr<LhsXprType::Rows, LhsXprType::Cols, LhsXprType>& lhs,
  const MatrixExpr<RhsXprType::Rows, RhsXprType::Cols, RhsXprType>& rhs) {
    return MatrixBinOp<LhsXprType, RhsXprType, std::minus<>>(lhs.derived(), rhs.derived(), std::minus<>());
}

// expression of the scalar-matrix multiplication between a scalar and a MatrixExpr
template <typename XprType, typename ScalarType>
struct MatrixScalarMultiplicationOp :
    public MatrixExpr<XprType::Rows, XprType::Cols, MatrixScalarMultiplicationOp<XprType, ScalarType>> {
    using Base = MatrixExpr<XprType::Rows, XprType::Cols, MatrixScalarMultiplicationOp<XprType, ScalarType>>;
    using XprTypeNested = internals::ref_select_t<const XprType>;
    using Scalar = decltype(std::declval<typename XprType::Scalar>() * std::declval<ScalarType>());
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename XprType_>
        requires(std::is_constructible_v<XprType, XprType_>)
    constexpr MatrixScalarMultiplicationOp(XprType_ && xpr, ScalarType s) : xpr_(std::forward<XprType_>(xpr)), s_(s) { }
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
constexpr auto operator*(const MatrixExpr<XprType::Rows, XprType::Cols, XprType>& lhs, ScalarType rhs) {
    return MatrixScalarMultiplicationOp<XprType, ScalarType>(lhs.derived(), rhs);
}
template <typename XprType, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator*(ScalarType lhs, const MatrixExpr<XprType::Rows, XprType::Cols, XprType>& rhs) {
    return rhs * lhs;
}
template <typename XprType, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator/(const MatrixExpr<XprType::Rows, XprType::Cols, XprType>& lhs, ScalarType rhs) {
    return MatrixScalarMultiplicationOp<XprType, ScalarType>(lhs.derived(), ScalarType(1) / rhs);
}

// expression of the matrix-product of two MatrixExpr operands
template <typename LhsXprType, typename RhsXprType, typename Executor>
struct MatrixMultiplicationOp :
    public MatrixExpr<LhsXprType::Rows, RhsXprType::Cols, MatrixMultiplicationOp<LhsXprType, RhsXprType, Executor>> {
    fdapde_static_assert(
      internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType> ||
        LhsXprType::Cols == RhsXprType::Rows,
      INVALID_STATIC_SIZED_OPERANDS_FOR_MATRIX_MATRIX_PRODUCT);
    using Base =
      MatrixExpr<LhsXprType::Rows, RhsXprType::Cols, MatrixMultiplicationOp<LhsXprType, RhsXprType, Executor>>;
    using LhsXprTypeNested = internals::ref_select_t<const LhsXprType>;
    using RhsXprTypeNested = internals::ref_select_t<const RhsXprType>;
    using Scalar = decltype(std::declval<typename LhsXprType::Scalar>() * std::declval<typename RhsXprType::Scalar>());
    static constexpr int Rows = LhsXprType::Rows;
    static constexpr int Cols = RhsXprType::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename LhsXprType_, typename RhsXprType_>
        requires(std::is_constructible_v<LhsXprType, LhsXprType_> && std::is_constructible_v<RhsXprType, RhsXprType_>)
    constexpr MatrixMultiplicationOp(LhsXprType_&& lhs, RhsXprType_&& rhs) :
        lhs_(std::forward<LhsXprType_>(lhs)), rhs_(std::forward<RhsXprType_>(rhs)) {
        if constexpr (internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType>) {
            fdapde_assert(std::cmp_equal(lhs_.cols() FDAPDE_COMMA rhs_.rows()));
        }
    }
    constexpr Scalar operator()(int i, int j) const { return Executor::run(i, j, lhs_, rhs_); }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(
          LhsXprType::Rows == 1 || RhsXprType::Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return Executor::run(LhsXprType::Rows == 1 ? 0 : i, RhsXprType::Cols == 1 ? i : 0, lhs_, rhs_);
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
template <typename LhsXprType, typename RhsXprType> struct generic_matrix_product_executor {
    using LhsXprTypeClean = std::decay_t<LhsXprType>;
    using RhsXprTypeClean = std::decay_t<RhsXprType>;
    using Scalar =
      decltype(std::declval<typename LhsXprTypeClean::Scalar>() * std::declval<typename RhsXprTypeClean::Scalar>());

    static constexpr auto run(int i, int j, const LhsXprType& lhs, const RhsXprType& rhs) {
        Scalar prod = 0;
        const int size = lhs.cols();
        for (int k = 0; k < size; ++k) { prod += lhs(i, k) * rhs(k, j); }
        return prod;
    }
};

// outer product v * v^\top executor
template <typename LhsXprType, typename RhsXprType> struct outer_product_executor {
    using LhsXprTypeClean = std::decay_t<LhsXprType>;
    using RhsXprTypeClean = std::decay_t<RhsXprType>;
    using Scalar =
      decltype(std::declval<typename LhsXprTypeClean::Scalar>() * std::declval<typename RhsXprTypeClean::Scalar>());
    static_assert(LhsXprTypeClean::Cols == 1 && RhsXprTypeClean::Rows == 1);

    static constexpr auto run(int i, int j, const LhsXprType& lhs, const RhsXprType& rhs) { return lhs[i] * rhs[j]; }
};

  
}   // namespace internals

// generic matrix-matrix product
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator*(
  const MatrixExpr<LhsXprType::Rows, LhsXprType::Cols, LhsXprType>& lhs,
  const MatrixExpr<RhsXprType::Rows, RhsXprType::Cols, RhsXprType>& rhs) {
    if constexpr (LhsXprType::Cols == 1 && RhsXprType::Rows == 1) {   // outer product v * u^\top
        return MatrixMultiplicationOp<
          LhsXprType, RhsXprType, internals::outer_product_executor<LhsXprType, RhsXprType>> {
          lhs.derived(), rhs.derived()};
    } else {
        return MatrixMultiplicationOp<
          LhsXprType, RhsXprType, internals::generic_matrix_product_executor<LhsXprType, RhsXprType>> {
          lhs.derived(), rhs.derived()};
    }
}

// specialized binary operations

// expression of the dense kronecker tensor product between two MatrixExpr operands
template <typename LhsXprType, typename RhsXprType>
struct MatrixKroneckerProductOp :
    public MatrixExpr<
      LhsXprType::Rows == Dynamic || RhsXprType::Rows == Dynamic ? Dynamic : LhsXprType::Rows * RhsXprType::Rows,
      LhsXprType::Cols == Dynamic || RhsXprType::Cols == Dynamic ? Dynamic : LhsXprType::Cols * RhsXprType::Cols,
      MatrixKroneckerProductOp<LhsXprType, RhsXprType>> {
    using LhsXprTypeNested = internals::ref_select_t<const LhsXprType>;
    using RhsXprTypeNested = internals::ref_select_t<const RhsXprType>;
    using Scalar = decltype(std::declval<typename LhsXprType::Scalar>() * std::declval<typename RhsXprType::Scalar>());
    static constexpr int Rows =
      LhsXprType::Rows == Dynamic || RhsXprType::Rows == Dynamic ? Dynamic : LhsXprType::Rows * RhsXprType::Rows;
    static constexpr int Cols =
      LhsXprType::Cols == Dynamic || RhsXprType::Cols == Dynamic ? Dynamic : LhsXprType::Cols * RhsXprType::Cols;
    using Base = MatrixExpr<Rows, Cols, MatrixKroneckerProductOp<LhsXprType, RhsXprType>>;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename LhsXprType_, typename RhsXprType_>
        requires(std::is_constructible_v<LhsXprType, LhsXprType_> && std::is_constructible_v<RhsXprType, RhsXprType_>)
    constexpr MatrixKroneckerProductOp(LhsXprType_&& lhs, RhsXprType_&& rhs) :
        lhs_(std::forward<LhsXprType_>(lhs)), rhs_(std::forward<RhsXprType_>(rhs)) { }
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
constexpr MatrixKroneckerProductOp<LhsXprType, RhsXprType> kron(
  const MatrixExpr<LhsXprType::Rows, LhsXprType::Cols, LhsXprType>& op1,
  const MatrixExpr<RhsXprType::Rows, RhsXprType::Cols, RhsXprType>& op2) {
    return MatrixKroneckerProductOp<LhsXprType, RhsXprType> {op1.derived(), op2.derived()};
}

// expression of the cross product between two vector expressions
template <typename LhsXprType, typename RhsXprType>
struct MatrixCrossProductOp : public MatrixExpr<LhsXprType::Rows, 1, MatrixCrossProductOp<LhsXprType, RhsXprType>> {
    fdapde_static_assert(
      internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType> ||
        (LhsXprType::Rows == 3 && RhsXprType::Rows == 3 && LhsXprType::Cols == 1 && RhsXprType::Cols == 1),
      THIS_CLASS_IS_FOR_THREE_DIMENSIONAL_VECTORS_ONLY);
    using LhsXprTypeNested = internals::ref_select_t<const LhsXprType>;
    using RhsXprTypeNested = internals::ref_select_t<const RhsXprType>;
    using Scalar = decltype(std::declval<typename LhsXprType::Scalar>() * std::declval<typename RhsXprType::Scalar>());
    static constexpr int Rows = 3;
    static constexpr int Cols = 1;
    using Base = MatrixExpr<LhsXprType::Rows, 1, MatrixCrossProductOp<LhsXprType, RhsXprType>>;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename LhsXprType_, typename RhsXprType_>
        requires(std::is_constructible_v<LhsXprType, LhsXprType_> && std::is_constructible_v<RhsXprType, RhsXprType_>)
    constexpr MatrixCrossProductOp(LhsXprType_&& lhs, RhsXprType_&& rhs) :
        lhs_(std::forward<LhsXprType_>(lhs)), rhs_(std::forward<RhsXprType_>(rhs)) {
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
