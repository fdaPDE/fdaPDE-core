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
      YOU_MIXED_MATRICES_OF_DIFFERENT_STATIC_SIZE);
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
// expresion representing the coefficient-wise application of a unary operation to a MatrixExpr operand
template <typename XprType, typename CoeffOperation>
struct MatrixCoeffWiseOp : public MatrixExpr<XprType::Rows, XprType::Cols, MatrixCoeffWiseOp<XprType, CoeffOperation>> {
   public:
    using Base = MatrixExpr<XprType::Rows, XprType::Cols, MatrixCoeffWiseOp<XprType, CoeffOperation>>;
    using XprTypeNested = internals::ref_select_t<const XprType>;
    using XprTypeClean = std::decay_t<XprType>;
    using Scalar = decltype(std::declval<CoeffOperation>().operator()(std::declval<typename XprTypeClean::Scalar>()));
    static constexpr int Rows = XprTypeClean::Rows;
    static constexpr int Cols = XprTypeClean::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename XprType_>
        requires(std::is_constructible_v<XprTypeNested, XprType_>)
    constexpr MatrixCoeffWiseOp(XprType_&& xpr, CoeffOperation op) : xpr_(std::forward<XprType_>(xpr)), op_(op) { }
    constexpr Scalar operator()(int i, int j) const { return op_(xpr_(i, j)); }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return op_(xpr_[i]);
    }
    constexpr int rows() const { return Rows != Dynamic ? Rows : xpr_.rows(); }
    constexpr int cols() const { return Cols != Dynamic ? Cols : xpr_.cols(); }
   private:
    XprTypeNested xpr_;
    CoeffOperation op_;
};

// definition of the linear vector-space structure of the set of M x N matrices
  
namespace internals {

template <typename Scalar> struct matrix_coeff_mult_t {
    constexpr explicit matrix_coeff_mult_t(Scalar x) noexcept : x_(x) { }
    template <typename Scalar_> constexpr auto operator()(const Scalar_& y) const { return x_ * y; }
   private:
    Scalar x_;
};

}   // namespace internals

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
// scalar multiplication
template <typename XprType, typename CoeffType>
    requires(std::is_arithmetic_v<CoeffType>)
constexpr auto operator*(const MatrixExpr<XprType::Rows, XprType::Cols, XprType>& lhs, CoeffType rhs) {
    return MatrixCoeffWiseOp<XprType, internals::matrix_coeff_mult_t<CoeffType>>(
      lhs.derived(), internals::matrix_coeff_mult_t(rhs));
}
template <typename XprType, typename CoeffType>
    requires(std::is_arithmetic_v<CoeffType>)
constexpr auto operator*(CoeffType lhs, const MatrixExpr<XprType::Rows, XprType::Cols, XprType>& rhs) {
    return rhs * lhs;
}
template <typename XprType, typename CoeffType>
    requires(std::is_arithmetic_v<CoeffType>)
constexpr auto operator/(const MatrixExpr<XprType::Rows, XprType::Cols, XprType>& lhs, CoeffType rhs) {
    return MatrixCoeffWiseOp<XprType, internals::matrix_coeff_mult_t<CoeffType>>(
      lhs.derived(), internals::matrix_coeff_mult_t(CoeffType(1) / rhs));
}

// expression of the matrix-product of two MatrixExpr operands
template <typename LhsXprType, typename RhsXprType, typename Executor>
struct MatrixProductOp :
    public MatrixExpr<LhsXprType::Rows, RhsXprType::Cols, MatrixProductOp<LhsXprType, RhsXprType, Executor>> {
    fdapde_static_assert(
      internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType> ||
        LhsXprType::Cols == RhsXprType::Rows,
      INVALID_STATIC_SIZED_OPERANDS_FOR_MATRIX_MATRIX_PRODUCT);
    using Base = MatrixExpr<LhsXprType::Rows, RhsXprType::Cols, MatrixProductOp<LhsXprType, RhsXprType, Executor>>;
    using LhsXprTypeNested = internals::ref_select_t<const LhsXprType>;
    using RhsXprTypeNested = internals::ref_select_t<const RhsXprType>;
    using Scalar = decltype(std::declval<typename LhsXprType::Scalar>() * std::declval<typename RhsXprType::Scalar>());
    static constexpr int Rows = LhsXprType::Rows;
    static constexpr int Cols = RhsXprType::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename LhsXprType_, typename RhsXprType_>
        requires(std::is_constructible_v<LhsXprType, LhsXprType_> && std::is_constructible_v<RhsXprType, RhsXprType_>)
    constexpr MatrixProductOp(LhsXprType_&& lhs, RhsXprType_&& rhs) :
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
        return MatrixProductOp<LhsXprType, RhsXprType, internals::outer_product_executor<LhsXprType, RhsXprType>> {
          lhs.derived(), rhs.derived()};
    } else {
        return MatrixProductOp<
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

}   // namespace fdapde

#endif // __FDAPDE_LINALG_BINARY_OP_H__
