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

#ifndef __FDAPDE_LINALG_COEFFWISE_H__
#define __FDAPDE_LINALG_COEFFWISE_H__

#include "header_check.h"

#include <concepts>

namespace fdapde {

// coefficient-wise type system
template <typename XprType_> struct MatrixCoeffWiseExpr;

namespace internals {

// CoeffOp used by MatrixExpr::cwise() to enter the coeffwise type system
struct identity_op {
    template <typename ValueType> constexpr ValueType operator()(ValueType&& v) const noexcept { return v; }
    template <typename ValueType> constexpr ValueType& operator()(ValueType& v) noexcept { return v; }   // lvalue
};

template <typename Operand, bool IsScalar = std::is_arithmetic_v<std::decay_t<Operand>>>
struct cwise_operand_traits;

template <typename Operand> struct cwise_operand_traits<Operand, false> {
    using Type = std::decay_t<Operand>;
    using Scalar = typename Type::Scalar;
    static constexpr int Rows = Type::Rows;
    static constexpr int Cols = Type::Cols;
    static constexpr int StorageOrder = Type::StorageOrder;
};

template <typename Operand> struct cwise_operand_traits<Operand, true> {
    using Scalar = std::decay_t<Operand>;
    static constexpr int Rows = Dynamic;
    static constexpr int Cols = Dynamic;
    static constexpr int StorageOrder = RowMajor;
};

template <
  typename Lhs, typename Rhs,
  bool HasScalar = std::is_arithmetic_v<std::decay_t<Lhs>> || std::is_arithmetic_v<std::decay_t<Rhs>>>
struct cwise_shape_compatible : std::bool_constant<same_static_shape_weak_v<Lhs, Rhs>> { };

template <typename Lhs, typename Rhs>
struct cwise_shape_compatible<Lhs, Rhs, true> : std::true_type { };

template <typename Operand>
constexpr decltype(auto) cwise_access(const Operand& operand, int i, int j) {
    if constexpr (std::is_arithmetic_v<std::decay_t<Operand>>) {
        return operand;
    } else {
        return operand(i, j);
    }
}

template <typename Operand>
constexpr decltype(auto) cwise_access(const Operand& operand, int i) {
    if constexpr (std::is_arithmetic_v<std::decay_t<Operand>>) {
        return operand;
    } else {
        return operand[i];
    }
}

// assignment executor having one trivial scalar operand
struct scalar_cwise_assignment_executor {
    template <typename DstMatrixType, typename ScalarType, typename AssignmentOp>
    static constexpr void run(DstMatrixType& dst, const ScalarType& src, AssignmentOp&& op) {
        fdapde_static_assert(DstMatrixType::ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION);
        using assignment_executor = typename DstMatrixType::assignment_executor;
        assignment_executor::run(dst.xpr(), src, [op](auto&& l, const auto& r) { op(l, r); });
        return;
    }
};

// internal type for reinterpreting a coefficient-wise expression as a matrix expression
template <typename XprType_> struct mwise_wrapper : public MatrixExpr<mwise_wrapper<XprType_>> {
   private:
    using XprType = std::decay_t<XprType_>;
    using XprTypeNested = internals::ref_select_t<XprType_>;
   public:
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    static constexpr int StorageOrder = XprType::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = std::is_const_v<std::remove_reference_t<XprType_>> || XprType::ReadOnly;

    constexpr mwise_wrapper(const mwise_wrapper&) = default;
    template <typename XprType__>
        requires(
          !std::same_as<std::remove_cvref_t<XprType__>, mwise_wrapper> &&
          internals::safely_nestable<XprTypeNested, XprType__>)
    constexpr explicit mwise_wrapper(XprType__&& xpr) : xpr_(std::forward<XprType__>(xpr)) { }
    // access
    constexpr decltype(auto) operator()(int i, int j) const {
        if (i < 0 || i >= rows() || j < 0 || j >= cols()) {
            throw std::out_of_range("coefficient-wise matrix index out of range");
        }
        return std::as_const(xpr_)(i, j);
    }
    constexpr decltype(auto) operator()(int i, int j) requires(ReadOnly == 0) {
        if (i < 0 || i >= rows() || j < 0 || j >= cols()) {
            throw std::out_of_range("coefficient-wise matrix index out of range");
        }
        return xpr_(i, j);
    }
    constexpr decltype(auto) operator[](int i) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        if (i < 0 || i >= rows() * cols()) {
            throw std::out_of_range("coefficient-wise vector index out of range");
        }
        return std::as_const(xpr_)[i];
    }
    constexpr decltype(auto) operator[](int i) requires(ReadOnly == 0) {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        if (i < 0 || i >= rows() * cols()) {
            throw std::out_of_range("coefficient-wise vector index out of range");
        }
        return xpr_[i];
    }
    // observers
    constexpr int rows() const { return xpr_.rows(); }
    constexpr int cols() const { return xpr_.cols(); }
   private:
    XprTypeNested xpr_;
};

}   // namespace internals
  
// expression node for the coefficient wise application of CoeffOp on a matrix expression
template <typename XprType_, typename CoeffOp = internals::identity_op>
struct MatrixCoeffWiseOp : public MatrixCoeffWiseExpr<MatrixCoeffWiseOp<XprType_, CoeffOp>> {
   private:
    using XprType = std::decay_t<XprType_>;
    using XprTypeNested = internals::ref_select_t<XprType_>;
   public:
    using InputScalar = typename XprType::Scalar;
    using Scalar = std::remove_cvref_t<std::invoke_result_t<const CoeffOp&, const InputScalar&>>;
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    static constexpr int StorageOrder = XprType::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr bool Writable = XprType::ReadOnly == 0 &&
      requires(CoeffOp& op, XprType& xpr, InputScalar value, int i, int j) {
          op(xpr(i, j)) = value;
      };
    static constexpr int ReadOnly =
      std::is_const_v<std::remove_reference_t<XprType_>> || XprType::ReadOnly || !Writable;
    static constexpr bool BulkWritable = ReadOnly == 0 && std::same_as<CoeffOp, internals::identity_op>;
    using assignment_executor = std::conditional_t<
      ReadOnly, internals::deleted_assignment_executor, internals::assignment_executor_of_t<XprType>>;

    constexpr MatrixCoeffWiseOp(const MatrixCoeffWiseOp&) = default;
    template <typename XprType__>
        requires(
          !std::same_as<std::remove_cvref_t<XprType__>, MatrixCoeffWiseOp> &&
          internals::safely_nestable<XprTypeNested, XprType__>)
    constexpr MatrixCoeffWiseOp(XprType__&& xpr, CoeffOp op) :
        xpr_(std::forward<XprType__>(xpr)), op_(std::move(op)) { }
    // scalar assignment
    template <typename Scalar_>
        requires(BulkWritable && std::is_convertible_v<Scalar_, Scalar>)
    constexpr MatrixCoeffWiseOp& operator=(Scalar_ rhs) & {
        internals::scalar_cwise_assignment_executor::run(*this, rhs, [](auto& l, const Scalar_& r) { l = r; });
        return *this;
    }
    template <typename Scalar_>
        requires(BulkWritable && std::is_convertible_v<Scalar_, Scalar>)
    constexpr MatrixCoeffWiseOp operator=(Scalar_ rhs) && {
        static_cast<MatrixCoeffWiseOp&>(*this).operator=(rhs);
        return *this;
    }
    // access
    constexpr Scalar operator()(int i, int j) const {
        if (i < 0 || i >= rows() || j < 0 || j >= cols()) {
            throw std::out_of_range("coefficient-wise matrix index out of range");
        }
        return op_(std::as_const(xpr_)(i, j));
    }
    constexpr decltype(auto) operator()(int i, int j) requires(ReadOnly == 0) {   // write-access
        if (i < 0 || i >= rows() || j < 0 || j >= cols()) {
            throw std::out_of_range("coefficient-wise matrix index out of range");
        }
        return op_(xpr_(i, j));
    }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        if (i < 0 || i >= rows() * cols()) {
            throw std::out_of_range("coefficient-wise vector index out of range");
        }
        return op_(std::as_const(xpr_)[i]);
    }
    constexpr decltype(auto) operator[](int i) requires(ReadOnly == 0) {   // write-access
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        if (i < 0 || i >= rows() * cols()) {
            throw std::out_of_range("coefficient-wise vector index out of range");
        }
        return op_(xpr_[i]);
    }
    // observers
    constexpr int rows() const { return Rows != Dynamic ? Rows : xpr_.rows(); }
    constexpr int cols() const { return Cols != Dynamic ? Cols : xpr_.cols(); }
    constexpr const XprType& xpr() const & { return std::as_const(xpr_); }
    constexpr XprType& xpr() & requires(ReadOnly == 0) { return xpr_; }
    constexpr void xpr() const && = delete;
    constexpr void xpr() && = delete;
   private:
    XprTypeNested xpr_;
    CoeffOp op_;
};
  
// expression node for a binary coefficient wise operation
template <typename LhsXprType_, typename RhsXprType_, typename BinaryOp>
struct MatrixCoeffWiseBinOp : public MatrixCoeffWiseExpr<MatrixCoeffWiseBinOp<LhsXprType_, RhsXprType_, BinaryOp>> {
   private:
    using LhsXprType = std::decay_t<LhsXprType_>;
    using RhsXprType = std::decay_t<RhsXprType_>;
    fdapde_static_assert(
      internals::cwise_shape_compatible<LhsXprType_ FDAPDE_COMMA RhsXprType_>::value,
      INVALID_BINARY_OPERATION__MATRICES_OF_DIFFERENT_STATIC_SIZE);
    using LhsXprTypeNested = internals::ref_select_t<LhsXprType_>;
    using RhsXprTypeNested = internals::ref_select_t<RhsXprType_>;
    static constexpr int infer_static_shape_(int lhs_dim, int rhs_dim) {
        return std::is_arithmetic_v<LhsXprType> ?
                 rhs_dim :
                 (std::is_arithmetic_v<RhsXprType> ? lhs_dim :
                                                     ((lhs_dim == Dynamic || rhs_dim == Dynamic) ? Dynamic : lhs_dim));
    }  
   public:
    using LhsTraits = internals::cwise_operand_traits<LhsXprType>;
    using RhsTraits = internals::cwise_operand_traits<RhsXprType>;
    using Scalar = std::remove_cvref_t<
      std::invoke_result_t<const BinaryOp&, typename LhsTraits::Scalar, typename RhsTraits::Scalar>>;
    static constexpr int Rows = infer_static_shape_(LhsTraits::Rows, RhsTraits::Rows);
    static constexpr int Cols = infer_static_shape_(LhsTraits::Cols, RhsTraits::Cols);
    static constexpr int StorageOrder = [] {
        if constexpr (std::is_arithmetic_v<LhsXprType>) {
            return RhsTraits::StorageOrder;
        } else if constexpr (std::is_arithmetic_v<RhsXprType>) {
            return LhsTraits::StorageOrder;
        } else {
            return internals::promote_storage_order_v<LhsTraits::StorageOrder, RhsTraits::StorageOrder>;
        }
    }();
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;
    static constexpr bool BulkWritable = false;
  
    template <typename LhsXprType__, typename RhsXprType__>
        requires(internals::safely_nestable<LhsXprTypeNested, LhsXprType__> &&
                 internals::safely_nestable<RhsXprTypeNested, RhsXprType__>)
    constexpr MatrixCoeffWiseBinOp(LhsXprType__&& lhs, RhsXprType__&& rhs, BinaryOp op) :
        lhs_(std::forward<LhsXprType__>(lhs)), rhs_(std::forward<RhsXprType__>(rhs)), op_(std::move(op)) {
        if constexpr (!std::is_arithmetic_v<LhsXprType> && !std::is_arithmetic_v<RhsXprType>) {
            if constexpr (internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType>) {
                if (!std::cmp_equal(lhs_.rows(), rhs_.rows()) || !std::cmp_equal(lhs_.cols(), rhs_.cols())) {
                    throw std::invalid_argument("coefficient-wise binary operation requires matching shapes");
                }
            }
        }
    }
    constexpr Scalar operator()(int i, int j) const {
        if (i < 0 || i >= rows() || j < 0 || j >= cols()) {
            throw std::out_of_range("coefficient-wise matrix index out of range");
        }
        return op_(internals::cwise_access(lhs_, i, j), internals::cwise_access(rhs_, i, j));
    }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        if (i < 0 || i >= rows() * cols()) {
            throw std::out_of_range("coefficient-wise vector index out of range");
        }
        return op_(internals::cwise_access(lhs_, i), internals::cwise_access(rhs_, i));
    }
    constexpr int rows() const {
        if constexpr (!std::is_arithmetic_v<LhsXprType>) {
            return lhs_.rows();
        } else {
            return rhs_.rows();
        }
    }
    constexpr int cols() const {
        if constexpr (!std::is_arithmetic_v<LhsXprType>) {
            return lhs_.cols();
        } else {
            return rhs_.cols();
        }
    }
   private:
    LhsXprTypeNested lhs_;
    RhsXprTypeNested rhs_;
    BinaryOp op_;
};

namespace internals {

// internal utility to factor unary operators
template <typename CoeffOp, typename XprType_>
constexpr auto make_cwise_op(const MatrixCoeffWiseExpr<XprType_>& xpr, CoeffOp&& op) {
    using StoredOp = std::decay_t<CoeffOp>;
    return MatrixCoeffWiseOp<XprType_, StoredOp>(xpr.derived(), std::forward<CoeffOp>(op));
}
// internal utilities to factor binary operators
template <typename BinaryOp, typename LhsXprType_, typename RhsXprType_>
constexpr auto make_cwise_op(const MatrixCoeffWiseExpr<LhsXprType_>& lhs, const MatrixCoeffWiseExpr<RhsXprType_>& rhs) {
    return MatrixCoeffWiseBinOp<LhsXprType_, RhsXprType_, BinaryOp>(lhs.derived(), rhs.derived(), BinaryOp());
}
template <typename BinaryOp, typename Scalar_, typename XprType_>
    requires(std::is_arithmetic_v<Scalar_> && requires(typename XprType_::Scalar x, Scalar_ s) { BinaryOp {}(x, s); })
constexpr auto make_cwise_op(const MatrixCoeffWiseExpr<XprType_>& lhs, const Scalar_& rhs) {
    return MatrixCoeffWiseBinOp<XprType_, Scalar_, BinaryOp>(lhs.derived(), rhs, BinaryOp());
}
template <typename BinaryOp, typename Scalar_, typename XprType_>
    requires(std::is_arithmetic_v<Scalar_> && requires(typename XprType_::Scalar x, Scalar_ s) { BinaryOp {}(x, s); })
constexpr auto make_cwise_op(const Scalar_& lhs, const MatrixCoeffWiseExpr<XprType_>& rhs) {
    return MatrixCoeffWiseBinOp<Scalar_, XprType_, BinaryOp>(lhs, rhs.derived(), BinaryOp());
}

}   // namespace internals

// base class for coefficient wise arithmetic
template <typename XprType_> struct MatrixCoeffWiseExpr {
   private:
    using XprType = std::decay_t<XprType_>;
   public:
    constexpr const XprType& derived() const & { return static_cast<const XprType&>(*this); }
    constexpr XprType& derived() & { return static_cast<XprType&>(*this); }
    constexpr void derived() const && = delete;
    constexpr void derived() && = delete;
    // generic coeffwise executor
    template <typename CoeffOp_> constexpr auto apply(CoeffOp_&& op) & {
        using StoredOp = std::decay_t<CoeffOp_>;
        return MatrixCoeffWiseOp<XprType, StoredOp>(derived(), std::forward<CoeffOp_>(op));
    }
    template <typename CoeffOp_> constexpr auto apply(CoeffOp_&& op) const & {
        using StoredOp = std::decay_t<CoeffOp_>;
        return MatrixCoeffWiseOp<const XprType, StoredOp>(derived(), std::forward<CoeffOp_>(op));
    }
    template <typename CoeffOp_> constexpr auto apply(CoeffOp_&& op) && {
        using StoredOp = std::decay_t<CoeffOp_>;
        return MatrixCoeffWiseOp<XprType, StoredOp>(
          static_cast<XprType&&>(*this), std::forward<CoeffOp_>(op));
    }
    template <typename CoeffOp_> constexpr auto apply(CoeffOp_&& op) const && {
        using StoredOp = std::decay_t<CoeffOp_>;
        return MatrixCoeffWiseOp<const XprType, StoredOp>(
          static_cast<const XprType&&>(*this), std::forward<CoeffOp_>(op));
    }
    // catalogue of coeficient wise operations
    constexpr auto pow(int i) const {
        return internals::make_cwise_op(derived(), [i](const auto& x) { return fdapde::pow(x, i); });
    }
    constexpr auto pow2() const { return pow(2); }
    constexpr auto abs() const {
        return internals::make_cwise_op(derived(), [](const auto& x) { return fdapde::abs(x); });
    }
    constexpr decltype(auto) sqrt() const
        requires(std::floating_point<std::remove_cv_t<typename XprType::Scalar>>)
    {
        using Scalar = std::remove_cv_t<typename XprType::Scalar>;
        return internals::make_cwise_op(derived(), [](const auto& x) -> Scalar {
            return internals::scale_safe_sqrt(static_cast<Scalar>(x));
        });
    }
    constexpr auto inv() const {
        return internals::make_cwise_op(derived(), [](const auto& x) { return 1. / x; });
    }
    constexpr auto exp() const {
        return internals::make_cwise_op(derived(), [](const auto& x) { return fdapde::exp(x); });
    }
    constexpr auto log() const {
        return internals::make_cwise_op(derived(), [](const auto& x) { return fdapde::log(x); });
    }
    // unary negation
    constexpr auto operator-() const {
        return internals::make_cwise_op(derived(), [](const auto& x) { return -x; });
    }
    // compound coeffwise arithmetic
    template <typename Scalar_>
        requires(
          XprType::BulkWritable && std::is_arithmetic_v<Scalar_> &&
          requires(typename XprType::Scalar x, Scalar_ s) { x + s; })
    constexpr XprType& operator+=(Scalar_ rhs) & {
        internals::scalar_cwise_assignment_executor::run(derived(), rhs, [](auto& l, const Scalar_& r) { l += r; });
        return derived();
    }
    template <typename Scalar_>
        requires(
          XprType::BulkWritable && std::is_arithmetic_v<Scalar_> &&
          requires(typename XprType::Scalar x, Scalar_ s) { x + s; })
    constexpr XprType operator+=(Scalar_ rhs) && {
        static_cast<MatrixCoeffWiseExpr&>(*this).operator+=(rhs);
        return static_cast<XprType&>(*this);
    }
    template <typename RhsXprType_>
        requires(XprType::ReadOnly == 0)
    constexpr XprType& operator+=(const MatrixCoeffWiseExpr<RhsXprType_>& rhs) & {
        using executor = typename XprType::assignment_executor;
        using RhsXprType = std::decay_t<RhsXprType_>;
        using RhsScalar = std::remove_cv_t<typename RhsXprType::Scalar>;
        Matrix<RhsScalar, RhsXprType::Rows, RhsXprType::Cols, RhsXprType::StorageOrder> tmp(rhs.mwise());
        executor::run(derived(), tmp, [](auto& l, const auto& r) { l += r; });
        return derived();
    }
    template <typename RhsXprType_>
        requires(XprType::ReadOnly == 0)
    constexpr XprType operator+=(const MatrixCoeffWiseExpr<RhsXprType_>& rhs) && {
        static_cast<MatrixCoeffWiseExpr&>(*this).operator+=(rhs);
        return static_cast<XprType&>(*this);
    }
    template <typename Scalar_>
        requires(
          XprType::BulkWritable && std::is_arithmetic_v<Scalar_> &&
          requires(typename XprType::Scalar x, Scalar_ s) { x - s; })
    constexpr XprType& operator-=(Scalar_ rhs) & {
        internals::scalar_cwise_assignment_executor::run(derived(), rhs, [](auto& l, const Scalar_& r) { l -= r; });
        return derived();
    }
    template <typename Scalar_>
        requires(
          XprType::BulkWritable && std::is_arithmetic_v<Scalar_> &&
          requires(typename XprType::Scalar x, Scalar_ s) { x - s; })
    constexpr XprType operator-=(Scalar_ rhs) && {
        static_cast<MatrixCoeffWiseExpr&>(*this).operator-=(rhs);
        return static_cast<XprType&>(*this);
    }
    template <typename RhsXprType_>
        requires(XprType::ReadOnly == 0)
    constexpr XprType& operator-=(const MatrixCoeffWiseExpr<RhsXprType_>& rhs) & {
        using executor = typename XprType::assignment_executor;
        using RhsXprType = std::decay_t<RhsXprType_>;
        using RhsScalar = std::remove_cv_t<typename RhsXprType::Scalar>;
        Matrix<RhsScalar, RhsXprType::Rows, RhsXprType::Cols, RhsXprType::StorageOrder> tmp(rhs.mwise());
        executor::run(derived(), tmp, [](auto& l, const auto& r) { l -= r; });
        return derived();
    }
    template <typename RhsXprType_>
        requires(XprType::ReadOnly == 0)
    constexpr XprType operator-=(const MatrixCoeffWiseExpr<RhsXprType_>& rhs) && {
        static_cast<MatrixCoeffWiseExpr&>(*this).operator-=(rhs);
        return static_cast<XprType&>(*this);
    }
    template <typename Scalar_>
        requires(
          XprType::BulkWritable && std::is_arithmetic_v<Scalar_> &&
          requires(typename XprType::Scalar x, Scalar_ s) { x * s; })
    constexpr XprType& operator*=(Scalar_ rhs) & {
        internals::scalar_cwise_assignment_executor::run(derived(), rhs, [](auto& l, const Scalar_& r) { l *= r; });
        return derived();
    }
    template <typename Scalar_>
        requires(
          XprType::BulkWritable && std::is_arithmetic_v<Scalar_> &&
          requires(typename XprType::Scalar x, Scalar_ s) { x * s; })
    constexpr XprType operator*=(Scalar_ rhs) && {
        static_cast<MatrixCoeffWiseExpr&>(*this).operator*=(rhs);
        return static_cast<XprType&>(*this);
    }
    template <typename RhsXprType_>
        requires(XprType::ReadOnly == 0)
    constexpr XprType& operator*=(const MatrixCoeffWiseExpr<RhsXprType_>& rhs) & {
        using executor = typename XprType::assignment_executor;
        using RhsXprType = std::decay_t<RhsXprType_>;
        using RhsScalar = std::remove_cv_t<typename RhsXprType::Scalar>;
        Matrix<RhsScalar, RhsXprType::Rows, RhsXprType::Cols, RhsXprType::StorageOrder> tmp(rhs.mwise());
        executor::run(derived(), tmp, [](auto& l, const auto& r) { l *= r; });
        return derived();
    }
    template <typename RhsXprType_>
        requires(XprType::ReadOnly == 0)
    constexpr XprType operator*=(const MatrixCoeffWiseExpr<RhsXprType_>& rhs) && {
        static_cast<MatrixCoeffWiseExpr&>(*this).operator*=(rhs);
        return static_cast<XprType&>(*this);
    }
    template <typename Scalar_>
        requires(
          XprType::BulkWritable && std::is_arithmetic_v<Scalar_> &&
          requires(typename XprType::Scalar x, Scalar_ s) { x / s; })
    constexpr XprType& operator/=(Scalar_ rhs) & {
        internals::scalar_cwise_assignment_executor::run(derived(), rhs, [](auto& l, const Scalar_& r) { l /= r; });
        return derived();
    }
    template <typename Scalar_>
        requires(
          XprType::BulkWritable && std::is_arithmetic_v<Scalar_> &&
          requires(typename XprType::Scalar x, Scalar_ s) { x / s; })
    constexpr XprType operator/=(Scalar_ rhs) && {
        static_cast<MatrixCoeffWiseExpr&>(*this).operator/=(rhs);
        return static_cast<XprType&>(*this);
    }
    template <typename RhsXprType_>
        requires(XprType::ReadOnly == 0)
    constexpr XprType& operator/=(const MatrixCoeffWiseExpr<RhsXprType_>& rhs) & {
        using executor = typename XprType::assignment_executor;
        using RhsXprType = std::decay_t<RhsXprType_>;
        using RhsScalar = std::remove_cv_t<typename RhsXprType::Scalar>;
        Matrix<RhsScalar, RhsXprType::Rows, RhsXprType::Cols, RhsXprType::StorageOrder> tmp(rhs.mwise());
        executor::run(derived(), tmp, [](auto& l, const auto& r) { l /= r; });
        return derived();
    }
    template <typename RhsXprType_>
        requires(XprType::ReadOnly == 0)
    constexpr XprType operator/=(const MatrixCoeffWiseExpr<RhsXprType_>& rhs) && {
        static_cast<MatrixCoeffWiseExpr&>(*this).operator/=(rhs);
        return static_cast<XprType&>(*this);
    }
    // reinrepret cwise expression as matrix expression
    constexpr auto mwise() & { return internals::mwise_wrapper<XprType>(derived()); }
    constexpr auto mwise() const & { return internals::mwise_wrapper<const XprType>(derived()); }
    constexpr auto mwise() && {
        return internals::mwise_wrapper<XprType>(static_cast<XprType&&>(*this));
    }
    constexpr auto mwise() const && {
        return internals::mwise_wrapper<const XprType>(static_cast<const XprType&&>(*this));
    }
    // ostream
    friend std::ostream& operator<<(std::ostream& os, const MatrixCoeffWiseExpr& m) {
        os << m.mwise();
        return os;
    }
    // reductions
    constexpr bool any() const { return mwise().any(); }
    constexpr bool all() const { return mwise().all(); }
    constexpr int count() const { return mwise().count(); }
};

// coeffwise arithmetic
// operations are always interpreted on the single scalar coefficients
// coeffwise addition
template <typename LhsXprType_, typename RhsXprType_>
constexpr auto operator+(const MatrixCoeffWiseExpr<LhsXprType_>& lhs, const MatrixCoeffWiseExpr<RhsXprType_>& rhs) {
    return internals::make_cwise_op<std::plus<>>(lhs, rhs);
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator+(const MatrixCoeffWiseExpr<XprType_>& lhs, const ScalarType& rhs) {
    return internals::make_cwise_op<std::plus<>>(lhs, rhs);
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator+(const ScalarType& lhs, const MatrixCoeffWiseExpr<XprType_>& rhs) {
    return internals::make_cwise_op<std::plus<>>(lhs, rhs);
}
// coeffwise difference
template <typename LhsXprType_, typename RhsXprType_>
constexpr auto operator-(const MatrixCoeffWiseExpr<LhsXprType_>& lhs, const MatrixCoeffWiseExpr<RhsXprType_>& rhs) {
    return internals::make_cwise_op<std::minus<>>(lhs, rhs);
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator-(const MatrixCoeffWiseExpr<XprType_>& lhs, const ScalarType& rhs) {
    return internals::make_cwise_op<std::minus<>>(lhs, rhs);
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator-(const ScalarType& lhs, const MatrixCoeffWiseExpr<XprType_>& rhs) {
    return internals::make_cwise_op<std::minus<>>(lhs, rhs);
}
// coeffwise multiplication (hadamard product)
template <typename LhsXprType_, typename RhsXprType_>
constexpr auto operator*(const MatrixCoeffWiseExpr<LhsXprType_>& lhs, const MatrixCoeffWiseExpr<RhsXprType_>& rhs) {
    return internals::make_cwise_op<std::multiplies<>>(lhs, rhs);
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator*(const MatrixCoeffWiseExpr<XprType_>& lhs, const ScalarType& rhs) {
    return internals::make_cwise_op<std::multiplies<>>(lhs, rhs);
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator*(const ScalarType& lhs, const MatrixCoeffWiseExpr<XprType_>& rhs) {
    return internals::make_cwise_op<std::multiplies<>>(lhs, rhs);
}
// coeffwise division (hadamard division)
template <typename LhsXprType_, typename RhsXprType_>
constexpr auto operator/(const MatrixCoeffWiseExpr<LhsXprType_>& lhs, const MatrixCoeffWiseExpr<RhsXprType_>& rhs) {
    return internals::make_cwise_op<std::divides<>>(lhs, rhs);
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator/(const MatrixCoeffWiseExpr<XprType_>& lhs, const ScalarType& rhs) {
    return internals::make_cwise_op<std::divides<>>(lhs, rhs);
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator/(const ScalarType& lhs, const MatrixCoeffWiseExpr<XprType_>& rhs) {
    return internals::make_cwise_op<std::divides<>>(lhs, rhs);
}

// coeffwise comparisons
// differently from the matrix-domain, coeffwise comparisons produce a boolean matrix of elementwise comparisons
// strict comparison
template <typename LhsXprType_, typename RhsXprType_>
constexpr auto operator<(const MatrixCoeffWiseExpr<LhsXprType_>& lhs, const MatrixCoeffWiseExpr<RhsXprType_>& rhs) {
    return internals::make_cwise_op<std::less<>>(lhs, rhs);
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator<(const MatrixCoeffWiseExpr<XprType_>& lhs, const ScalarType& rhs) {
    return internals::make_cwise_op<std::less<>>(lhs, rhs);
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator<(const ScalarType& lhs, const MatrixCoeffWiseExpr<XprType_>& rhs) {
    return internals::make_cwise_op<std::less<>>(lhs, rhs);
}
template <typename LhsXprType_, typename RhsXprType_>
constexpr auto operator>(const MatrixCoeffWiseExpr<LhsXprType_>& lhs, const MatrixCoeffWiseExpr<RhsXprType_>& rhs) {
    return internals::make_cwise_op<std::greater<>>(lhs, rhs);
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator>(const MatrixCoeffWiseExpr<XprType_>& lhs, const ScalarType& rhs) {
    return internals::make_cwise_op<std::greater<>>(lhs, rhs);
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator>(const ScalarType& lhs, const MatrixCoeffWiseExpr<XprType_>& rhs) {
    return internals::make_cwise_op<std::greater<>>(lhs, rhs);
}
// weak comparison
template <typename LhsXprType_, typename RhsXprType_>
constexpr auto operator<=(const MatrixCoeffWiseExpr<LhsXprType_>& lhs, const MatrixCoeffWiseExpr<RhsXprType_>& rhs) {
    return internals::make_cwise_op<std::less_equal<>>(lhs, rhs);
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator<=(const MatrixCoeffWiseExpr<XprType_>& lhs, const ScalarType& rhs) {
    return internals::make_cwise_op<std::less_equal<>>(lhs, rhs);
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator<=(const ScalarType& lhs, const MatrixCoeffWiseExpr<XprType_>& rhs) {
    return internals::make_cwise_op<std::less_equal<>>(lhs, rhs);
}
template <typename LhsXprType_, typename RhsXprType_>
constexpr auto operator>=(const MatrixCoeffWiseExpr<LhsXprType_>& lhs, const MatrixCoeffWiseExpr<RhsXprType_>& rhs) {
    return internals::make_cwise_op<std::greater_equal<>>(lhs, rhs);
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator>=(const MatrixCoeffWiseExpr<XprType_>& lhs, const ScalarType& rhs) {
    return internals::make_cwise_op<std::greater_equal<>>(lhs, rhs);
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator>=(const ScalarType& lhs, const MatrixCoeffWiseExpr<XprType_>& rhs) {
    return internals::make_cwise_op<std::greater_equal<>>(lhs, rhs);
}
// equality comparison
template <typename LhsXprType_, typename RhsXprType_>
constexpr auto operator==(const MatrixCoeffWiseExpr<LhsXprType_>& lhs, const MatrixCoeffWiseExpr<RhsXprType_>& rhs) {
    return internals::make_cwise_op<std::equal_to<>>(lhs, rhs);
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator==(const MatrixCoeffWiseExpr<XprType_>& lhs, const ScalarType& rhs) {
    return internals::make_cwise_op<std::equal_to<>>(lhs, rhs);
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator==(const ScalarType& lhs, const MatrixCoeffWiseExpr<XprType_>& rhs) {
    return internals::make_cwise_op<std::equal_to<>>(lhs, rhs);
}
template <typename LhsXprType_, typename RhsXprType_>
constexpr auto operator!=(const MatrixCoeffWiseExpr<LhsXprType_>& lhs, const MatrixCoeffWiseExpr<RhsXprType_>& rhs) {
    return internals::make_cwise_op<std::not_equal_to<>>(lhs, rhs);
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator!=(const MatrixCoeffWiseExpr<XprType_>& lhs, const ScalarType& rhs) {
    return internals::make_cwise_op<std::not_equal_to<>>(lhs, rhs);
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator!=(const ScalarType& lhs, const MatrixCoeffWiseExpr<XprType_>& rhs) {
    return internals::make_cwise_op<std::not_equal_to<>>(lhs, rhs);
}

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_COEFFWISE_H__
