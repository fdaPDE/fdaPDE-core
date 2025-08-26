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

#ifndef __FDAPDE_MATRIX_BASE_H__
#define __FDAPDE_MATRIX_BASE_H__

#include <iomanip>

#include "../header_check.h"

namespace fdapde {

// matrix transpose
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

// matrix binary operation
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
            fdapde_constexpr_assert(
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
// matrix addition
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator+(
  const MatrixExpr<LhsXprType::Rows, LhsXprType::Cols, LhsXprType>& lhs,
  const MatrixExpr<RhsXprType::Rows, RhsXprType::Cols, RhsXprType>& rhs) {
    return MatrixBinOp<LhsXprType, RhsXprType, std::plus<>>(lhs.derived(), rhs.derived(), std::plus<>());
}
// matrix subtraction
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator-(
  const MatrixExpr<LhsXprType::Rows, LhsXprType::Cols, LhsXprType>& lhs,
  const MatrixExpr<RhsXprType::Rows, RhsXprType::Cols, RhsXprType>& rhs) {
    return MatrixBinOp<LhsXprType, RhsXprType, std::minus<>>(lhs.derived(), rhs.derived(), std::minus<>());
}

// matrix coefficient-wise operation
template <typename XprType, typename UnaryOperation>
struct MatrixCoeffWiseOp : public MatrixExpr<XprType::Rows, XprType::Cols, MatrixCoeffWiseOp<XprType, UnaryOperation>> {
   public:
    using Base = MatrixExpr<XprType::Rows, XprType::Cols, MatrixCoeffWiseOp<XprType, UnaryOperation>>;
    using XprTypeNested = internals::ref_select_t<const XprType>;
    using XprTypeClean = std::decay_t<XprType>;
    using Scalar = decltype(std::declval<UnaryOperation>().operator()(std::declval<typename XprTypeClean::Scalar>()));
    static constexpr int Rows = XprTypeClean::Rows;
    static constexpr int Cols = XprTypeClean::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename XprType_>
        requires(std::is_constructible_v<XprTypeNested, XprType_>)
    constexpr MatrixCoeffWiseOp(XprType_&& xpr, UnaryOperation op) : xpr_(std::forward<XprType_>(xpr)), op_(op) { }
    constexpr Scalar operator()(int i, int j) const { return op_(xpr_(i, j)); }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return op_(xpr_[i]);
    }
    constexpr int rows() const { return Rows != Dynamic ? Rows : xpr_.rows(); }
    constexpr int cols() const { return Cols != Dynamic ? Cols : xpr_.cols(); }
   private:
    XprTypeNested xpr_;
    UnaryOperation op_;
};
template <typename XprType, typename CoeffType>
    requires(std::is_arithmetic_v<CoeffType>)
constexpr auto operator*(const MatrixExpr<XprType::Rows, XprType::Cols, XprType>& lhs, CoeffType rhs) {
    auto op_ = [rhs](const typename XprType::Scalar& x) { return x * rhs; };
    return MatrixCoeffWiseOp<XprType, decltype(op_)>(lhs.derived(), op_);
}
template <typename XprType, typename CoeffType>
    requires(std::is_arithmetic_v<CoeffType>)
constexpr auto operator*(CoeffType lhs, const MatrixExpr<XprType::Rows, XprType::Cols, XprType>& rhs) {
    return rhs * lhs;
}
template <typename XprType, typename CoeffType>
    requires(std::is_arithmetic_v<CoeffType>)
constexpr auto operator/(const MatrixExpr<XprType::Rows, XprType::Cols, XprType>& lhs, CoeffType rhs) {
    auto op_ = [rhs](const typename XprType::Scalar& x) { return x / rhs; };
    return MatrixCoeffWiseOp<XprType, decltype(op_)>(lhs.derived(), op_);
}

// matrix vector-wise operation
namespace internals {

template <typename XprType, typename ReductionOp, int ByRow>
struct partial_matrix_redux_op :
    public MatrixExpr<
      ByRow ? 1 : XprType::Rows, ByRow ? XprType::Cols : 1, partial_matrix_redux_op<XprType, ReductionOp, ByRow>> {
    static constexpr int Rows = ByRow ? 1 : XprType::Rows;
    static constexpr int Cols = ByRow ? XprType::Cols : 1;
    using Base = MatrixExpr<Rows, Cols, partial_matrix_redux_op<XprType, ReductionOp, ByRow>>;
    using XprTypeNested = internals::ref_select_t<std::conditional_t<XprType::ReadOnly, const XprType, XprType>>;
    using Scalar = typename XprType::Scalar;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = XprType::ReadOnly;

    template <typename XprType_>
        requires(std::is_constructible_v<XprTypeNested, XprType_>)
    partial_matrix_redux_op(XprType_&& xpr, Scalar init, ReductionOp op) :
        xpr_(std::forward<XprType_>(xpr)), init_(init), op_(op) { }

    constexpr Scalar operator()(int i, int j) const {
        fdapde_constexpr_assert(i >= 0 && i < rows() && j >= 0 && j < cols());
        Scalar res(init_);
        const int k = ByRow ? j : i;
        for (int h = 0, size_ = ByRow ? xpr_.rows() : xpr_.cols(); h < size_; ++h) {
            res = op_(res, ByRow ? xpr_(h, k) : xpr_(k, h));
        }
        return res;
    }
    constexpr Scalar operator[](int i) const { return operator()(ByRow ? 0 : i, ByRow ? i : 0); }
    // observers
    constexpr int rows() const { return ByRow ? Rows : xpr_.rows(); }
    constexpr int cols() const { return ByRow ? xpr_.cols() : Cols; }
   private:
    XprTypeNested xpr_;
    Scalar init_;
    ReductionOp op_;
};

}   // namespace internals

template <typename XprType, int ByRow>
struct MatrixVectorWiseOp :
    public MatrixExpr<ByRow ? 1 : XprType::Rows, ByRow ? XprType::Cols : 1, MatrixVectorWiseOp<XprType, ByRow>> {
    static constexpr int Rows = ByRow ? 1 : XprType::Rows;
    static constexpr int Cols = ByRow ? XprType::Cols : 1;
    using Base = MatrixExpr<Rows, Cols, MatrixVectorWiseOp<XprType, ByRow>>;
    using XprTypeNested = internals::ref_select_t<std::conditional_t<XprType::ReadOnly, const XprType, XprType>>;
    using Scalar = typename XprType::Scalar;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = XprType::ReadOnly;

    template <typename XprType_>
        requires(std::is_constructible_v<XprTypeNested, XprType_>)
    MatrixVectorWiseOp(XprType_&& xpr) : xpr_(std::forward<XprType_>(xpr)) { }

    // generic redux operator
    template <typename XprType_, typename Scalar_, typename ReductionOp>
        requires(requires(ReductionOp op, Scalar a, Scalar b) {
            { op(a, b) } -> std::convertible_to<Scalar>;
        })
    constexpr auto redux(XprType_&& xpr, Scalar_ init, ReductionOp&& op) const {
        fdapde_static_assert(
          std::is_convertible_v<Scalar_ FDAPDE_COMMA Scalar>, INVALID_SCALAR_INIT_TYPE_IN_REDUX_OPERATION);
        return internals::partial_matrix_redux_op<XprType_, ReductionOp, ByRow>(xpr, init, op);
    }
    // standard reductions
    constexpr auto sum() const {
        return redux(xpr_, Scalar(0), [](Scalar tmp, Scalar x) { return tmp + x; });
    }
    constexpr auto prod() const {
        return redux(xpr_, Scalar(1), [](Scalar tmp, Scalar x) { return tmp * x; });
    }
    constexpr auto mean() const {
        int size_ = ByRow ? xpr_.rows() : xpr_.cols();
        return sum() / size_;
    }
    constexpr auto max() const {
        return redux(xpr_, std::numeric_limits<Scalar>::min(), [](Scalar tmp, Scalar x) { return tmp > x ? tmp : x; });
    }
    constexpr auto min() const {
        return redux(xpr_, std::numeric_limits<Scalar>::max(), [](Scalar tmp, Scalar x) { return tmp < x ? tmp : x; });
    }
    // L^2 squared norm
    constexpr auto squared_norm() const {
        return redux(xpr_, Scalar(0), [](Scalar tmp, Scalar x) { return tmp + x * x; });
    }
    constexpr auto norm() const { return squared_norm().cwise_sqrt(); }
    // L^\infty norm
    constexpr auto inf_norm() const {
        return redux(
          xpr_.cwise_abs(), std::numeric_limits<Scalar>::min(), [](Scalar tmp, Scalar x) { return tmp > x ? tmp : x; });
    }

    // vector-wise assignment
    template <int XprRows_, int XprCols_, typename XprType_>
    constexpr MatrixVectorWiseOp& operator=(const MatrixExpr<XprRows_, XprCols_, XprType_>& rhs) {
        partial_redux_inplace_loop_(rhs, [](Scalar& a, Scalar b) { a = b; });
        return *this;
    }
    template <int XprRows_, int XprCols_, typename XprType_>
    constexpr MatrixVectorWiseOp& operator+=(const MatrixExpr<XprRows_, XprCols_, XprType_>& rhs) {
        partial_redux_inplace_loop_(rhs, [](Scalar& a, Scalar b) { a += b; });
        return *this;
    }
    template <int XprRows_, int XprCols_, typename XprType_>
    constexpr MatrixVectorWiseOp& operator-=(const MatrixExpr<XprRows_, XprCols_, XprType_>& rhs) {
        partial_redux_inplace_loop_(rhs, [](Scalar& a, Scalar b) { a -= b; });
        return *this;
    }

    // observers
    constexpr int rows() const { return ByRow ? Rows : xpr_.rows(); }
    constexpr int cols() const { return ByRow ? xpr_.cols() : Cols; }
   private:
    // internals
    template <int XprRows_, int XprCols_, typename XprType_, typename Operator_>
    constexpr void partial_redux_inplace_loop_(const MatrixExpr<XprRows_, XprCols_, XprType_>& rhs, Operator_&& op) {
        fdapde_static_assert(
          (Rows == 1 && XprRows_ == 1) || (Cols == 1 & XprCols_ == 1), NO_MATCHING_SIZES_IN_VECTOR_WISE_ASSIGNMENT);
        fdapde_static_assert(XprType::ReadOnly == 0, ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION);
        fdapde_constexpr_assert(Rows == 1 && xpr_.cols() == rhs.cols() || Cols == 1 && xpr_.rows() == rhs.rows());

        int inner_size_ = ByRow ? xpr_.rows() : xpr_.cols();
        int outer_size_ = ByRow ? xpr_.cols() : xpr_.rows();
        for (int i = 0; i < outer_size_; ++i) {
            for (int j = 0; j < inner_size_; ++j) {
                if constexpr (ByRow == 0) op(xpr_(i, j), rhs.derived()[i]);
                if constexpr (ByRow == 1) op(xpr_(j, i), rhs.derived()[i]);
            }
        }
	return;
    }
  
    XprTypeNested xpr_;
};
// row-wise matrix reduction expression
template <typename XprType> struct MatrixRowWiseOp : public MatrixVectorWiseOp<XprType, 1> {
    using Base = MatrixVectorWiseOp<XprType, 1>;
    template <typename XprType_> MatrixRowWiseOp(XprType_&& xpr) : Base(xpr) { }
    using Base::operator=;
};
// col-wise matrix reduction expression
template <typename XprType> struct MatrixColWiseOp : public MatrixVectorWiseOp<XprType, 0> {
    using Base = MatrixVectorWiseOp<XprType, 0>;
    template <typename XprType_> MatrixColWiseOp(XprType_&& xpr) : Base(xpr) { }
    using Base::operator=;
};

// matrix product operation
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
        const int size = LhsXprTypeClean::Rows == 1 ? lhs.cols() : lhs.rows();
        for (int k = 0; k < size; ++k) { prod += lhs(i, k) * rhs(k, j); }
        return prod;
    }
};

}   // namespace internals

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
            fdapde_constexpr_assert(std::cmp_equal(lhs_.cols() FDAPDE_COMMA rhs_.rows()));
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
// generic matrix-matrix product
template <typename LhsXprType, typename RhsXprType>
constexpr MatrixProductOp<LhsXprType, RhsXprType, internals::generic_matrix_product_executor<LhsXprType, RhsXprType>>
operator*(
  const MatrixExpr<LhsXprType::Rows, LhsXprType::Cols, LhsXprType>& lhs,
  const MatrixExpr<RhsXprType::Rows, RhsXprType::Cols, RhsXprType>& rhs) {
    return MatrixProductOp<LhsXprType, RhsXprType, internals::generic_matrix_product_executor<LhsXprType, RhsXprType>> {
      lhs.derived(), rhs.derived()};
}

// kronecker tensor product operation
template <typename LhsXprType, typename RhsXprType>
struct MatrixKroneckerProductOp :
    public MatrixExpr<
      LhsXprType::Rows == Dynamic || RhsXprType::Rows == Dynamic ? Dynamic : LhsXprType::Rows * RhsXprType::Rows,
      LhsXprType::Cols == Dynamic || RhsXprType::Cols == Dynamic ? Dynamic : LhsXprType::Cols * RhsXprType::Cols,
      MatrixKroneckerProductOp<LhsXprType, RhsXprType>> {
    using Base = MatrixExpr<
      LhsXprType::Rows * RhsXprType::Rows, LhsXprType::Cols * RhsXprType::Cols,
      MatrixKroneckerProductOp<LhsXprType, RhsXprType>>;
    using LhsXprTypeNested = internals::ref_select_t<const LhsXprType>;
    using RhsXprTypeNested = internals::ref_select_t<const RhsXprType>;
    using Scalar = decltype(std::declval<typename LhsXprType::Scalar>() * std::declval<typename RhsXprType::Scalar>());
    static constexpr int Rows =
      LhsXprType::Rows == Dynamic || RhsXprType::Rows == Dynamic ? Dynamic : LhsXprType::Rows * RhsXprType::Rows;
    static constexpr int Cols =
      LhsXprType::Cols == Dynamic || RhsXprType::Cols == Dynamic ? Dynamic : LhsXprType::Cols * RhsXprType::Cols;
    static constexpr int NestAsReaf = 0;
    static constexpr int ReadOnly = 1;

    template <typename LhsXprType_, typename RhsXprType_>
        requires(std::is_constructible_v<LhsXprType, LhsXprType_> && std::is_constructible_v<RhsXprType, RhsXprType_>)
    constexpr MatrixKroneckerProductOp(LhsXprType_&& lhs, RhsXprType_&& rhs) :
        lhs_(std::forward<LhsXprType_>(lhs)), rhs_(std::forward<RhsXprType_>(rhs)) { }
    constexpr Scalar operator()(int i, int j) const {
        const int rows = Rows != Dynamic ? Rows : rhs_.rows();
        const int cols = Cols != Dynamic ? Cols : rhs_.cols();
        // compute offsets in operand matrices
        int col_lhs = j / cols, row_lhs = i / rows;
        int col_rhs = j % cols, row_rhs = i % rows;
        return lhs_(row_lhs, col_lhs) * rhs_(row_rhs, col_rhs);
    }
    constexpr int rows() const {
        return (Rows != Dynamic ? Rows : lhs_.rows()) * (Rows != Dynamic ? Rows : rhs_.rows());
    }
    constexpr int cols() const {
        return (Cols != Dynamic ? Cols : lhs_.cols()) * (Cols != Dynamic ? Cols : rhs_.cols());
    }
   private:
    LhsXprTypeNested lhs_;
    RhsXprTypeNested rhs_;
};
template <typename LhsXprType, typename RhsXprType>
constexpr MatrixKroneckerProductOp<LhsXprType, RhsXprType> kronecker(
  const MatrixExpr<LhsXprType::Rows, LhsXprType::Cols, LhsXprType>& op1,
  const MatrixExpr<RhsXprType::Rows, RhsXprType::Cols, RhsXprType>& op2) {
    return MatrixKroneckerProductOp<LhsXprType, RhsXprType> {op1.derived(), op2.derived()};
}
  
template <int BlockRows_, int BlockCols_, typename XprType>
class MatrixBlock : public MatrixExpr<BlockRows_, BlockCols_, MatrixBlock<BlockRows_, BlockCols_, XprType>> {
    fdapde_static_assert(
      (BlockRows_ == Dynamic || (BlockRows_ > 0 && BlockRows_ <= XprType::Rows)) &&
        (BlockCols_ == Dynamic || (BlockCols_ > 0 && BlockCols_ <= XprType::Cols)),
      INVALID_STATIC_BLOCK_SIZE);
   public:
    using Base = MatrixExpr<BlockRows_, BlockCols_, MatrixBlock<BlockRows_, BlockCols_, XprType>>;
    using XprTypeNested = internals::ref_select_t<XprType>;
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = BlockRows_;
    static constexpr int Cols = BlockCols_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = XprType::ReadOnly;

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
        fdapde_constexpr_assert(
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
        fdapde_constexpr_assert(
          start_row >= 0 && start_row + block_rows_ < xpr_.rows() && start_col >= 0 &&
          start_col + block_cols_ < xpr.cols());
    }
    template <typename XprType_>
        requires(std::is_constructible_v<XprTypeNested, XprType_>)
    constexpr MatrixBlock(XprType_&& xpr, int start_row, int start_col, int block_rows, int block_cols) :
        start_row_(start_row),
        start_col_(start_col),
        block_rows_(block_rows_),
        block_cols_(block_cols_),
        xpr_(std::forward<XprType_>(xpr)) {
        fdapde_static_assert(
          BlockRows_ == Dynamic && BlockCols_ == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_BLOCKS_ONLY);
        fdapde_constexpr_assert(
          start_row >= 0 && start_row + block_rows_ < xpr_.rows() && start_col >= 0 &&
          start_col + block_cols_ < xpr.cols());
    }

    constexpr int rows() const { return Rows != Dynamic ? Rows : block_rows_; }
    constexpr int cols() const { return Cols != Dynamic ? Cols : block_cols_; }
    constexpr int size() const { return rows() * cols(); }
    constexpr const Scalar& operator()(int i, int j) const { return xpr_(start_row_ + i, start_col_ + j); }
    constexpr const Scalar& operator[](int i) const {
        fdapde_static_assert(BlockRows_ == 1 || BlockCols_ == 1, THIS_METHOD_IS_FOR_ROW_AND_COLUMN_BLOCKS_ONLY);
        if constexpr (Rows == 1) return xpr_(start_row_, start_col_ + i);
        if constexpr (Cols == 1) return xpr_(start_row_ + i, start_col_);
    }
    constexpr Scalar& operator()(int i, int j) {
        fdapde_static_assert(XprType::ReadOnly == 0, ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION);
        return xpr_(start_row_ + i, start_col_ + j);
    }
    constexpr Scalar& operator[](int i) {
        fdapde_static_assert(BlockRows_ == 1 || BlockCols_ == 1, THIS_METHOD_IS_FOR_ROW_AND_COLUMN_BLOCKS_ONLY);
        fdapde_static_assert(XprType::ReadOnly == 0, ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION);
        if constexpr (Rows == 1) return xpr_(start_row_, start_col_ + i);
        if constexpr (Cols == 1) return xpr_(start_row_ + i, start_col_);
    }
    // block assignment
    constexpr MatrixBlock& operator=(const MatrixBlock& other) {
        fdapde_static_assert(XprType::ReadOnly == 0, ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION);
        for (int i = 0; i < rows(); ++i) {
            for (int j = 0; j < cols(); ++j) { xpr_(start_row_ + i, start_col_ + j) = other(i, j); }
        }
        return *this;
    }
    template <int RhsRows_, int RhsCols_, typename RhsXprType>
    constexpr MatrixBlock<BlockRows_, BlockCols_, XprType>&
    operator=(const MatrixExpr<RhsRows_, RhsCols_, RhsXprType>& rhs) {
        fdapde_static_assert(XprType::ReadOnly == 0, BLOCK_ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION_IS_INVALID);
        using RhsXprTypeClean = std::decay_t<RhsXprType>;
        fdapde_static_assert(
          Cols == Dynamic || Rows == Dynamic || internals::is_dynamic_sized_v<RhsXprTypeClean> ||
            (RhsRows_ == Rows && RhsCols_ == Cols),
          INVALID_ASSIGNMENT__LHS_AND_RHS_STATIC_SIZES_DOES_NOT_MATCH);
        if constexpr (Cols == Dynamic || Rows == Dynamic || internals::is_dynamic_sized_v<RhsXprTypeClean>) {
            fdapde_constexpr_assert(block_rows_ == rhs.rows() && block_cols_ == rhs.cols());
        }
        for (int i = 0; i < rows(); ++i) {
            for (int j = 0; j < cols(); ++j) { xpr_(start_row_ + i, start_col_ + j) = rhs.derived()(i, j); }
        }
        return *this;
    }
   private:
    int start_row_ = 0, start_col_ = 0;
    int block_rows_ = 0, block_cols_ = 0;
    XprTypeNested xpr_;
};

namespace internals {

// linear reduction loop on matrix expressions
template <typename XprType, typename Functor> struct linear_matrix_redux_op {
    using Scalar = typename XprType::Scalar;

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

  // an expression of a reshaped operation. Reshaping modifes the expression dimensions without reallocating memory
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

// matrix base
template <int Rows, int Cols, typename XprType> struct MatrixExpr {
    MatrixExpr() = default;

    // assignment
    template <int RhsRows_, int RhsCols_, typename RhsXprType_>
    constexpr XprType& operator=(const MatrixExpr<RhsRows_, RhsCols_, RhsXprType_>& rhs) {
        fdapde_static_assert(
          (Rows == Dynamic || Rows == RhsRows_) && (Cols == Dynamic || Cols == RhsCols_),
          INVALID_ASSIGNMENT__LHS_AND_RHS_STATIC_SIZES_DOES_NOT_MATCH);
        if constexpr (Rows == Dynamic || Cols == Dynamic) {
            fdapde_constexpr_assert(derived().rows() == rhs.rows() && derived().cols() == rhs.cols());
        }
        using executor = typename XprType::assignment_executor;
        executor::run(derived(), rhs.derived());
        return derived();
    }

    // observers
    constexpr int size() const {
        return (Rows != Dynamic && Cols != Dynamic) ? Rows * Cols : derived().rows() * derived().cols();
    }
    constexpr int rows() const { return Rows == Dynamic ? derived().rows() : Rows; }
    constexpr int cols() const { return Cols == Dynamic ? derived().cols() : Cols; }
    constexpr const XprType& derived() const { return static_cast<const XprType&>(*this); }
    constexpr XprType& derived() { return static_cast<XprType&>(*this); }
    // ostream
    friend std::ostream& operator<<(std::ostream& os, const MatrixExpr& m) {
        const int rows = m.derived().rows();
        const int cols = m.derived().cols();
        // compute max width per column
        size_t width = 0;
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                std::ostringstream ss;
                ss << m.derived()(i, j);
                width = std::max(width, ss.str().size());
            }
        }
        // print values with alignment
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) { os << std::setw(int(width)) << m.derived()(i, j) << " "; }
            if (i != rows - 1) os << "\n";
        }
        return os;
    }

    // coeffwise operators
    // general coefficient wise executor
    template <typename CoeffOp> constexpr auto cwise(CoeffOp&& op) const {
        using CoeffOpReturnType =
          decltype(std::declval<CoeffOp>().operator()(std::declval<typename XprType::Scalar>()));
        using Scalar = typename XprType::Scalar;
        fdapde_static_assert(std::is_convertible_v<CoeffOpReturnType FDAPDE_COMMA Scalar>, INVALID_COEFFWISE_OPERATOR);
        fdapde_constexpr_assert(derived().rows() > 0 && derived().cols() > 0);

        return MatrixCoeffWiseOp<XprType, CoeffOp>(derived(), op);
    }
    constexpr auto cwise_abs() const {
        using Scalar = typename XprType::Scalar;
        return cwise([](Scalar x) { return fdapde::abs(x); });
    }
    constexpr auto cwise_pow(int i) const {
        using Scalar = typename XprType::Scalar;
        return cwise([i](Scalar x) { return fdapde::pow(x, i); });
    }
    constexpr auto cwise_pow2() const { return cwise_pow(2); }
    constexpr auto cwise_sqrt() const {
        using Scalar = typename XprType::Scalar;
        fdapde_static_assert(std::is_floating_point_v<Scalar>, THIS_METHOD_IS_FOR_FLOATING_POINT_MATRICES_ONLY);
        return cwise([](Scalar x) { return fdapde::sqrt(x); });
    }
    constexpr auto cwise_inv() const {
        using Scalar = typename XprType::Scalar;
        fdapde_static_assert(std::is_floating_point_v<Scalar>, THIS_METHOD_IS_FOR_FLOATING_POINT_MATRICES_ONLY);
        return cwise([](Scalar x) { return 1.0 / x; });
    }
    constexpr auto cwise_exp() const {
        using Scalar = typename XprType::Scalar;
        fdapde_static_assert(std::is_floating_point_v<Scalar>, THIS_METHOD_IS_FOR_FLOATING_POINT_MATRICES_ONLY);
        return cwise([](Scalar x) { return std::exp(x); });
    }
    constexpr auto cwise_log() const {
        using Scalar = typename XprType::Scalar;
        fdapde_static_assert(std::is_floating_point_v<Scalar>, THIS_METHOD_IS_FOR_FLOATING_POINT_MATRICES_ONLY);
        return cwise([](Scalar x) { return std::log(x); });
    }

    // redux operators
    // frobenius norm (squared L^2 norm)
    constexpr auto squared_norm() const {
        typename XprType::Scalar norm_ = 0;
        for (int i = 0; i < derived().rows(); ++i) {
            for (int j = 0; j < derived().cols(); ++j) { norm_ += fdapde::pow(derived().operator()(i, j), 2); }
        }
        return norm_;
    }
    constexpr auto norm() const { return fdapde::sqrt(squared_norm()); }
    // maximum norm (L^\infty norm)
    constexpr auto inf_norm() const {
        using Scalar = typename XprType::Scalar;
        Scalar norm_ = std::numeric_limits<Scalar>::min();
        for (int i = 0; i < derived().rows(); ++i) {
            for (int j = 0; j < derived().cols(); ++j) {
                Scalar tmp = fdapde::abs(derived().operator()(i, j));
                if (tmp > norm_) norm_ = tmp;
            }
        }
        return norm_;
    }
    // general redux executor
    template <typename Scalar_, typename ReduxOp> constexpr auto redux(Scalar_ init, ReduxOp&& op) const {
        using ReduxOpReturnType = decltype(std::declval<ReduxOp>().operator()(
          std::declval<typename XprType::Scalar>(), std::declval<typename XprType::Scalar>()));
        using Scalar = typename XprType::Scalar;
        fdapde_static_assert(std::is_convertible_v<ReduxOpReturnType FDAPDE_COMMA Scalar>, INVALID_REDUX_OPERATOR);
        fdapde_constexpr_assert(derived().rows() > 0 && derived().cols() > 0);

        return internals::linear_matrix_redux_op<XprType, ReduxOp>::run(derived(), init, op);
    }
    constexpr auto sum() const {
        using Scalar = typename XprType::Scalar;
        if (Rows == 0 || Cols == 0) return Scalar(0);
        return redux(Scalar(0), [](Scalar tmp, Scalar x) { return tmp + x; });
    }
    constexpr auto prod() const {
        using Scalar = typename XprType::Scalar;
        if (Rows == 0 || Cols == 0) return Scalar(1);
        return redux(Scalar(1), [](Scalar tmp, Scalar x) { return tmp * x; });
    }
    constexpr auto mean() const { return derived().sum() / derived().size(); }
    constexpr auto max() const {
        using Scalar = typename XprType::Scalar;
        return redux(std::numeric_limits<Scalar>::min(), [](Scalar tmp, Scalar x) { return tmp > x ? tmp : x; });
    }
    constexpr auto min() const {
        using Scalar = typename XprType::Scalar;
        return redux(std::numeric_limits<Scalar>::max(), [](Scalar tmp, Scalar x) { return tmp < x ? tmp : x; });
    }
    // vector-wise redux operators
    MatrixRowWiseOp<XprType> rowwise() { return MatrixRowWiseOp<XprType>(derived()); }
    MatrixRowWiseOp<const XprType> rowwise() const { return MatrixRowWiseOp<const XprType>(derived()); }
    MatrixColWiseOp<XprType> colwise() { return MatrixColWiseOp<XprType>(derived()); }
    MatrixColWiseOp<const XprType> colwise() const { return MatrixColWiseOp<const XprType>(derived()); }

    // unary operators
    constexpr TransposeOp<XprType> transpose() const { return TransposeOp<XprType>(derived()); }
    constexpr Diagonal<Rows, 1, const XprType> diagonal() const { return Diagonal<Rows, 1, const XprType>(derived()); }
    constexpr Diagonal<Rows, 1, XprType> diagonal() { return Diagonal<Rows, 1, XprType>(derived()); }
    constexpr internals::diagonal_wrapper<Rows == 1 ? Cols : Rows, Cols == 1 ? Rows : Cols, XprType> as_diagonal() {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return internals::diagonal_wrapper < Rows == 1 ? Cols : Rows, Cols == 1 ? Rows : Cols, XprType > (derived());
    }
    constexpr internals::diagonal_wrapper<Rows, Cols, const XprType> as_diagonal() const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return internals::diagonal_wrapper<Rows, Cols, const XprType>(derived());
    }
    template <int ViewMode> auto as_triangular() {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return internals::triangular_wrapper < Rows == 1 ? Cols : Rows, Cols == 1 ? Rows : Cols, ViewMode,
               XprType > (derived());
    }
    template <int ViewMode> auto as_triangular() const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return internals::triangular_wrapper < Rows == 1 ? Cols : Rows, Cols == 1 ? Rows : Cols, ViewMode,
               const XprType > (derived());
    }

    // block accessors
    // static-sized block
    template <int BlockRows, int BlockCols> constexpr MatrixBlock<BlockRows, BlockCols, XprType> block(int i, int j) {
        return MatrixBlock<BlockRows, BlockCols, XprType>(derived(), i, j);
    }
    template <int BlockRows, int BlockCols>
    constexpr MatrixBlock<BlockRows, BlockCols, const XprType> block(int i, int j) const {
        return MatrixBlock<BlockRows, BlockCols, const XprType>(derived(), i, j);
    }
    // dynamic-sized block
    constexpr MatrixBlock<Dynamic, Dynamic, XprType> block(int i, int j, int rows, int cols) {
        return MatrixBlock<Dynamic, Dynamic, XprType>(derived(), i, j, rows, cols);
    }
    constexpr MatrixBlock<Dynamic, Dynamic, const XprType> block(int i, int j, int rows, int cols) const {
        return MatrixBlock<Dynamic, Dynamic, const XprType>(derived(), i, j, rows, cols);
    }
    // row/col accessors
    constexpr MatrixBlock<Rows, 1, XprType> col(int i) { return MatrixBlock<Rows, 1, XprType>(derived(), i); }
    constexpr MatrixBlock<Rows, 1, const XprType> col(int i) const {
        return MatrixBlock<Rows, 1, const XprType>(derived(), i);
    }
    constexpr MatrixBlock<1, Cols, XprType> row(int i) { return MatrixBlock<1, Cols, XprType>(derived(), i); }
    constexpr MatrixBlock<1, Cols, const XprType> row(int i) const {
        return MatrixBlock<1, Cols, const XprType>(derived(), i);
    }
    // other block-type accessors
    template <int BlockRows> constexpr auto topRows() { return block<BlockRows, Cols>(0, 0); }
    template <int BlockRows> constexpr auto topRows() const { return block<BlockRows, Cols>(0, 0); }
    constexpr auto topRows(int rows) { return block(0, 0, rows, derived().cols()); }
    constexpr auto topRows(int rows) const { return block(0, 0, rows, derived().cols()); }

    template <int BlockRows> constexpr auto bottomRows() { return block<BlockRows, Cols>(Rows - BlockRows, 0); }
    template <int BlockRows> constexpr auto bottomRows() const { return block<BlockRows, Cols>(Rows - BlockRows, 0); }
    constexpr auto bottomRows(int rows) { return block(derived().rows() - rows, 0, rows, derived().cols()); }
    constexpr auto bottomRows(int rows) const { return block(derived().rows() - rows, 0, rows, derived().cols()); }

    template <int BlockCols> constexpr auto leftCols() { return block<Rows, BlockCols>(0, 0); }
    template <int BlockCols> constexpr auto leftCols() const { return block<Rows, BlockCols>(0, 0); }
    constexpr auto leftCols(int cols) { return block(0, 0, derived().rows(), cols); }
    constexpr auto leftCols(int cols) const { return block(0, 0, derived().rows(), cols); }

    template <int BlockCols> constexpr auto rightCols() { return block<Rows, BlockCols>(0, Cols - BlockCols); }
    template <int BlockCols> constexpr auto rightCols() const { return block<Rows, BlockCols>(0, Cols - BlockCols); }
    constexpr auto rightCols(int cols) { return block(0, derived().cols() - cols, derived().rows(), cols); }
    constexpr auto rightCols(int cols) const { return block(0, derived().cols() - cols, derived().rows(), cols); }

    // dot product
    template <int RhsRows, int RhsCols, typename RhsXprType>
    constexpr auto dot(const MatrixExpr<RhsRows, RhsCols, RhsXprType>& rhs) const {
        fdapde_static_assert(
          (RhsRows == 1 || RhsCols == 1) && (Rows == 1 || Cols == 1), THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        fdapde_constexpr_assert(
          fdapde::min(rows() FDAPDE_COMMA cols()) == 1 &&
          fdapde::max(rows() FDAPDE_COMMA cols()) == fdapde::max(rhs.rows() FDAPDE_COMMA rhs.cols()) &&
          fdapde::min(rhs.rows() FDAPDE_COMMA rhs.cols()) == 1);
        typename XprType::Scalar dot_ = 0;
        for (int i = 0, n = fdapde::max(rows(), cols()); i < n; ++i) {
            dot_ += derived().operator[](i) * rhs.derived().operator[](i);
        }
        return dot_;
    }

    // arithmetic operators
    template <int OtherRows, int OtherCols, typename OtherXprType>
    constexpr XprType& operator+=(const MatrixExpr<OtherRows, OtherCols, OtherXprType>& other) {
        fdapde_static_assert(Rows == OtherRows && Cols == OtherCols, YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES);
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) { derived().operator()(i, j) += other.derived()(i, j); }
        }
        return derived();
    }
    template <int OtherRows, int OtherCols, typename OtherXprType>
    constexpr XprType& operator-=(const MatrixExpr<OtherRows, OtherCols, OtherXprType>& other) {
        fdapde_static_assert(Rows == OtherRows && Cols == OtherCols, YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES);
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) { derived().operator()(i, j) -= other.derived()(i, j); }
        }
        return derived();
    }

    // reshaping
    // static-sized
    template <int Rows_, int Cols_, int StorageOrder_ = RowMajor> constexpr auto reshape() {
        return ReshapeOp<Rows_, Cols_, StorageOrder_, XprType>(derived());
    }
    template <int Rows_, int Cols_, int StorageOrder_ = RowMajor> constexpr auto reshape() const {
        return ReshapeOp<Rows_, Cols_, StorageOrder_, const XprType>(derived());
    }
    template <int Rows_, int StorageOrder_ = RowMajor> constexpr auto reshape() {
        return ReshapeOp<Rows_, 1, RowMajor, XprType>(derived());
    }
    template <int Rows_, int StorageOrder_ = RowMajor> constexpr auto reshape() const {
        return ReshapeOp<Rows_, 1, RowMajor, const XprType>(derived());
    }
    // dynamic-sized
    constexpr auto reshape(int rows, int cols) {
        return ReshapeOp<Dynamic, Dynamic, RowMajor, XprType>(derived(), rows, cols);
    }
    constexpr auto reshape(int rows, int cols) const {
        return ReshapeOp<Dynamic, Dynamic, RowMajor, const XprType>(derived(), rows, cols);
    }
    constexpr auto reshape(int rows) { return ReshapeOp<Dynamic, 1, RowMajor, XprType>(derived(), rows); }
    constexpr auto reshape(int rows) const { return ReshapeOp<Dynamic, 1, RowMajor, const XprType>(derived(), rows); }
    // dynamic-sized with storage order control
    template <int StorageOrder_> constexpr auto reshape(int rows, int cols) {
        return ReshapeOp<Dynamic, Dynamic, StorageOrder_, XprType>(derived(), rows, cols);
    }
    template <int StorageOrder_> constexpr auto reshape(int rows, int cols) const {
        return ReshapeOp<Dynamic, Dynamic, StorageOrder_, const XprType>(derived(), rows, cols);
    }

    // square matrix methods

    constexpr auto symm_part() const { return 0.5 * (derived() + derived().transpose()); }   // symmetric part
    constexpr auto skew_part() const { return 0.5 * (derived() - derived().transpose()); }   // skew-symmetric part
    // triangular block accessors
    template <int BlockMode> constexpr TriangularBlock<const XprType, BlockMode> triangular_block() const {
        return TriangularBlock<const XprType, BlockMode>(derived());
    }
    template <int BlockMode> constexpr TriangularBlock<XprType, BlockMode> triangular_block() {
        return TriangularBlock<XprType, BlockMode>(derived());
    }
};

// comparison operators
template <int Rows1, int Cols1, typename XprType1, int Rows2, int Cols2, typename XprType2>
constexpr bool
operator==(const MatrixExpr<Rows1, Cols1, XprType1>& op1, const MatrixExpr<Rows2, Cols2, XprType2>& op2) {
    fdapde_static_assert(Rows1 == Rows2 && Cols1 == Cols2, YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES);
    for (int i = 0; i < Rows1; ++i) {
        for (int j = 0; j < Cols1; ++j) {
            if (op1.derived()(i, j) != op2.derived()(i, j)) return false;
        }
    }
    return true;
}
template <int Rows1, int Cols1, typename XprType1, int Rows2, int Cols2, typename XprType2>
constexpr bool
operator!=(const MatrixExpr<Rows1, Cols1, XprType1>& op1, const MatrixExpr<Rows2, Cols2, XprType2>& op2) {
    fdapde_static_assert(Rows1 == Rows2 && Cols1 == Cols2, YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES);
    for (int i = 0; i < Rows1; ++i) {
        for (int j = 0; j < Cols1; ++j) {
            if (op1.derived()(i, j) == op2.derived()(i, j)) return false;
        }
    }
    return true;
}
template <int Rows1, int Cols1, typename XprType1, int Rows2, int Cols2, typename XprType2>
constexpr bool almost_equal(
  const MatrixExpr<Rows1, Cols1, XprType1>& op1, const MatrixExpr<Rows2, Cols2, XprType2>& op2, double epsilon = 1e-7) {
    fdapde_static_assert(Rows1 == Rows2 && Cols1 == Cols2, YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES);
    fdapde_static_assert(
      std::is_same_v<typename XprType1::Scalar FDAPDE_COMMA typename XprType2::Scalar>,
      YOU_MIXED_MATRICES_OF_DIFFERENT_SCALAR_TYPES);
    using Scalar_ = typename XprType1::Scalar;
    for (int i = 0; i < Rows1; ++i) {
        for (int j = 0; j < Cols1; ++j) {
            Scalar_ a = op1.derived()(i, j);
            Scalar_ b = op2.derived()(i, j);
            if (!(std::fabs(a - b) < epsilon ||
                  std::fabs(a - b) < ((std::fabs(a) < std::fabs(b) ? std::fabs(b) : std::fabs(a)) * epsilon))) {
                return false;
            }
        }
    }
    return true;
}

}   // namespace fdapde

#endif   // _FDAPDE_MATRIX_BASE_H__


// #ifdef __FDAPDE_HAS_EIGEN__

// template <typename LhsXprType, typename RhsXprType>
// constexpr MatrixProduct<LhsXprType, internals::eigen_xpr_wrap<RhsXprType>> operator*(
//   const MatrixExpr<LhsXprType::Rows, LhsXprType::Cols, LhsXprType>& lhs, const Eigen::MatrixExpr<RhsXprType>& rhs) {
//     return MatrixProduct<LhsXprType, internals::eigen_xpr_wrap<RhsXprType>> {lhs.derived(), rhs.derived()};
// }
// template <typename LhsXprType, typename RhsXprType>
// constexpr MatrixProduct<internals::eigen_xpr_wrap<LhsXprType>, RhsXprType> operator*(
//   const Eigen::MatrixExpr<LhsXprType>& lhs, const MatrixExpr<RhsXprType::Rows, RhsXprType::Cols, RhsXprType>& rhs) {
//     return MatrixProduct<internals::eigen_xpr_wrap<LhsXprType>, RhsXprType> {lhs.derived(), rhs.derived()};
// }

// #endif

// #ifdef __FDAPDE_HAS_EIGEN__

// template <typename Derived> struct eigen_xpr_wrap : public Derived {
//     using Derived::Derived;   // inherits Derived constructors
//     eigen_xpr_wrap(const Derived& xpr) : Derived(xpr) { }
//     eigen_xpr_wrap& operator=(const Derived& xpr) {
//         Derived::operator=(xpr);
//         return *this;
//     }
//     eigen_xpr_wrap(Derived&& xpr) : Derived(xpr) { }
//     eigen_xpr_wrap& operator=(Derived&& xpr) {
//         Derived::operator=(xpr);
//         return *this;
//     }
//     // injected additional constants
//     static constexpr int Rows = Derived::RowsAtCompileTime;
//     static constexpr int Cols = Derived::ColsAtCompileTime;
// };

// #endif


// #ifdef __FDAPDE_HAS_EIGEN__

// template <typename LhsXprType, typename RhsXprType>
// constexpr MatrixBinOp<LhsXprType, internals::eigen_xpr_wrap<RhsXprType>, std::plus<>> operator+(
//   const MatrixExpr<LhsXprType::Rows, LhsXprType::Cols, LhsXprType>& lhs, const Eigen::MatrixExpr<RhsXprType>& rhs) {
//     return MatrixBinOp<LhsXprType, internals::eigen_xpr_wrap<RhsXprType>, std::plus<>> {
//       lhs.derived(), rhs.derived(), std::plus<>()};
// }
// template <typename LhsXprType, typename RhsXprType>
// constexpr MatrixBinOp<internals::eigen_xpr_wrap<LhsXprType>, RhsXprType, std::plus<>> operator+(
//   const Eigen::MatrixExpr<LhsXprType>& lhs, const MatrixExpr<RhsXprType::Rows, RhsXprType::Cols, RhsXprType>& rhs) {
//     return MatrixBinOp<internals::eigen_xpr_wrap<LhsXprType>, RhsXprType, std::plus<>> {
//       lhs.derived(), rhs.derived(), std::plus<>()};
// }
// template <typename LhsXprType, typename RhsXprType>
// constexpr MatrixBinOp<LhsXprType, internals::eigen_xpr_wrap<RhsXprType>, std::minus<>> operator-(
//   const MatrixExpr<LhsXprType::Rows, LhsXprType::Cols, LhsXprType>& lhs, const Eigen::MatrixExpr<RhsXprType>& rhs) {
//     return MatrixBinOp<LhsXprType, internals::eigen_xpr_wrap<RhsXprType>, std::minus<>> {
//       lhs.derived(), rhs.derived(), std::minus<>()};
// }
// template <typename LhsXprType, typename RhsXprType>
// constexpr MatrixBinOp<internals::eigen_xpr_wrap<LhsXprType>, RhsXprType, std::minus<>> operator-(
//   const Eigen::MatrixExpr<LhsXprType>& lhs, const MatrixExpr<RhsXprType::Rows, RhsXprType::Cols, RhsXprType>& rhs) {
//     return MatrixBinOp<internals::eigen_xpr_wrap<LhsXprType>, RhsXprType, std::minus<>> {
//       lhs.derived(), rhs.derived(), std::minus<>()};
// }

// #endif

    // #ifdef __FDAPDE_HAS_EIGEN__
    //     // conversion to Eigen matrix
    //     auto as_eigen_matrix() const {
    //         Eigen::Matrix<typename Derived::Scalar, Rows, Cols> m;
    //         for (int i = 0; i < Rows; ++i) {
    //             for (int j = 0; j < Cols; ++j) { m(i, j) = derived().operator()(i, j); }
    //         }
    //         return m;
    //     }
    // #endif
