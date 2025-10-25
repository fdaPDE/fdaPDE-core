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

namespace fdapde {

template <typename XprType_> struct MatrixCoeffWiseExpr;

namespace internals {

// CoeffOp used by MatrixExpr::cwise() to enter the coeffwise type system
struct identity_op {
    template <typename ValueType> constexpr ValueType operator()(ValueType&& v) const noexcept { return v; }
    template <typename ValueType> constexpr ValueType& operator()(ValueType& v) noexcept { return v; }   // lvalue
};

// internal utility to wrap a scalar coefficient into an indexable type
template <typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
struct scalar_wrap {
    using Scalar = ScalarType;
    static constexpr int Rows = Adapted;   // inferred from context
    static constexpr int Cols = Adapted;   // inferred from context
    static constexpr int StorageOrder = RowMajor;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;
  
    constexpr explicit scalar_wrap(ScalarType scalar) noexcept : scalar_(scalar) { }
    constexpr decltype(auto) operator()([[maybe_unused]] int i, [[maybe_unused]] int j) const { return scalar_; }
    constexpr decltype(auto) operator[]([[maybe_unused]] int i) const { return scalar_; }
    constexpr const scalar_wrap& derived() const { return *this; }
   private:
    ScalarType scalar_;
};
  
// assignment executor having one trivial scalar operand
struct scalar_cwise_assignment_executor {
    template <typename DstMatrixType, typename ScalarType, typename AssignmentOp>
        requires(requires(AssignmentOp op, typename DstMatrixType::Scalar& l, const ScalarType& r) {
            { op(l, r) } -> std::same_as<void>;
        })
    static constexpr void run(DstMatrixType& dst, const ScalarType& src, AssignmentOp&& op) {
        fdapde_static_assert(DstMatrixType::ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION);
        const int rows_ = dst.rows();
        const int cols_ = dst.cols();
        // exploit cache-locality depending on StorageOrder of destination
        if constexpr (DstMatrixType::StorageOrder == RowMajor) {
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) { op(dst(i, j), src); }
            }
        } else {   // ColMajor
            for (int j = 0; j < cols_; ++j) {
                for (int i = 0; i < rows_; ++i) { op(dst(i, j), src); }
            }
        }
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
    static constexpr int ReadOnly = XprType::ReadOnly;

    template <typename XprType__>
        requires(std::is_constructible_v<XprTypeNested, XprType__>)
    constexpr explicit mwise_wrapper(XprType__&& xpr) : xpr_(std::forward<XprType__>(xpr)) { }
    // access
    constexpr decltype(auto) operator()(int i, int j) const noexcept {
        fdapde_assert(i >= 0 && i < rows() && j >= 0 && j < cols());
        return xpr_(i, j);
    }
    constexpr decltype(auto) operator()(int i, int j) noexcept {
        fdapde_assert(i >= 0 && i < rows() && j >= 0 && j < cols());
        return xpr_(i, j);
    }
    constexpr decltype(auto) operator[](int i) const noexcept {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        fdapde_assert(i >= 0 && i < rows());
        return xpr_[i];
    }
    constexpr decltype(auto) operator[](int i) noexcept {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        fdapde_assert(i >= 0 && i < rows());
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
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    static constexpr int StorageOrder = XprType::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = XprType::ReadOnly;
    using assignment_executor = internals::assignment_executor_of_t<XprType>;

    template <typename XprType__>
        requires(std::is_constructible_v<XprTypeNested, XprType__>)
    constexpr MatrixCoeffWiseOp(XprType__&& xpr, CoeffOp op) : xpr_(std::forward<XprType__>(xpr)), op_(op) { }
    // access
    constexpr decltype(auto) operator()(int i, int j) const {
        fdapde_assert(i >= 0 && i < rows() && j >= 0 && j < cols());
        return op_(xpr_(i, j));
    }
    constexpr decltype(auto) operator()(int i, int j) {   // write-access
        fdapde_assert(i >= 0 && i < rows() && j >= 0 && j < cols());
        return op_(xpr_(i, j));
    }
    constexpr decltype(auto) operator[](int i) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        fdapde_assert(i >= 0 && i < rows());
        return op_(xpr_[i]);
    }
    constexpr decltype(auto) operator[](int i) {   // write-access
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        fdapde_assert(i >= 0 && i < rows());
        return op_(xpr_[i]);
    }
    // observers
    constexpr int rows() const { return Rows != Dynamic ? Rows : xpr_.rows(); }
    constexpr int cols() const { return Cols != Dynamic ? Cols : xpr_.cols(); }
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
      internals::same_static_shape_weak_v<LhsXprType_ FDAPDE_COMMA RhsXprType_>,
      INVALID_BINARY_OPERATION__MATRICES_OF_DIFFERENT_STATIC_SIZE);
    fdapde_static_assert(
      ((LhsXprType::Rows != Adapted || RhsXprType::Rows != Adapted) &&
       (LhsXprType::Cols != Adapted || RhsXprType::Cols != Adapted)),
      INVALID_BINARY_OPERATION__CANNOT_INFER_EXPRESSION_SHAPE_FROM_CONTEXT);
    using LhsXprTypeNested = internals::ref_select_t<LhsXprType_>;
    using RhsXprTypeNested = internals::ref_select_t<RhsXprType_>;
    static constexpr int infer_static_shape_(int lhs_dim, int rhs_dim) {
        return lhs_dim == Adapted ?
                 rhs_dim :
                 (rhs_dim == Adapted ? lhs_dim : ((lhs_dim == Dynamic || rhs_dim == Dynamic) ? Dynamic : lhs_dim));
    }  
   public:
    using Scalar = promote_type_t<typename LhsXprType::Scalar, typename RhsXprType::Scalar>;
    static constexpr int Rows = infer_static_shape_(LhsXprType::Rows, RhsXprType::Rows);
    static constexpr int Cols = infer_static_shape_(LhsXprType::Cols, RhsXprType::Cols);
    static constexpr int StorageOrder =
      internals::promote_storage_order_v<LhsXprType::StorageOrder, RhsXprType::StorageOrder>;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;
  
    template <typename LhsXprType__, typename RhsXprType__>
        requires(std::is_constructible_v<LhsXprTypeNested, LhsXprType__> &&
                 std::is_constructible_v<RhsXprTypeNested, RhsXprType__>)
    constexpr MatrixCoeffWiseBinOp(LhsXprType__&& lhs, RhsXprType__&& rhs, BinaryOp op) :
        lhs_(std::forward<LhsXprType__>(lhs)), rhs_(std::forward<RhsXprType__>(rhs)), op_(op) {
        if constexpr (
          !internals::is_adapted_sized_v<LhsXprType> && !internals::is_adapted_sized_v<RhsXprType> &&
          (internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType>)) {
            fdapde_assert(
              std::cmp_equal(lhs_.rows() FDAPDE_COMMA rhs_.rows()) &&
              std::cmp_equal(lhs_.cols() FDAPDE_COMMA rhs_.cols()));
        }
    }
    constexpr decltype(auto) operator()(int i, int j) const {
        fdapde_assert(i >= 0 && i < rows() && j >= 0 && j < cols());
        return op_(lhs_(i, j), rhs_(i, j));
    }
    constexpr decltype(auto) operator[](int i) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        fdapde_assert(i >= 0 && i < rows());
        return op_(lhs_[i], rhs_[i]);
    }
    constexpr int rows() const { return Rows != Dynamic ? Rows : lhs_.rows(); }
    constexpr int cols() const { return Cols != Dynamic ? Cols : lhs_.cols(); }
   private:
    LhsXprTypeNested lhs_;
    RhsXprTypeNested rhs_;
    BinaryOp op_;
};

namespace internals {

// internal utility to factor unary operators
template <typename CoeffOp, typename XprType_>
constexpr auto make_cwise_op(const MatrixCoeffWiseExpr<XprType_>& xpr, CoeffOp&& op) {
    return MatrixCoeffWiseOp<XprType_, CoeffOp>(xpr.derived(), op);
}
// internal utilities to factor binary operators
template <typename BinaryOp, typename LhsXprType_, typename RhsXprType_>
constexpr auto make_cwise_op(const MatrixCoeffWiseExpr<LhsXprType_>& lhs, const MatrixCoeffWiseExpr<RhsXprType_>& rhs) {
    return MatrixCoeffWiseBinOp<LhsXprType_, RhsXprType_, BinaryOp>(lhs.derived(), rhs.derived(), BinaryOp());
}
template <typename BinaryOp, typename Scalar_, typename XprType_>
    requires(std::is_arithmetic_v<Scalar_> && requires(typename XprType_::Scalar x, Scalar_ s) { BinaryOp {}(x, s); })
constexpr auto make_cwise_op(const MatrixCoeffWiseExpr<XprType_>& lhs, const Scalar_& rhs) {
    return MatrixCoeffWiseBinOp<XprType_, internals::scalar_wrap<Scalar_>, BinaryOp>(
      lhs.derived(), internals::scalar_wrap(rhs), BinaryOp());
}
template <typename BinaryOp, typename Scalar_, typename XprType_>
    requires(std::is_arithmetic_v<Scalar_> && requires(typename XprType_::Scalar x, Scalar_ s) { BinaryOp {}(x, s); })
constexpr auto make_cwise_op(const Scalar_& lhs, const MatrixCoeffWiseExpr<XprType_>& rhs) {
    return MatrixCoeffWiseBinOp<internals::scalar_wrap<Scalar_>, XprType_, BinaryOp>(
      internals::scalar_wrap(lhs), rhs.derived(), BinaryOp());
}

}   // namespace internals

// base class for coefficient wise arithmetic
template <typename XprType_> struct MatrixCoeffWiseExpr {
   private:
    using XprType = std::decay_t<XprType_>;
   public:
    constexpr const XprType& derived() const { return static_cast<const XprType&>(*this); }
    constexpr XprType& derived() { return static_cast<XprType&>(*this); }
    // generic coeffwise executor
    template <typename CoeffOp_> constexpr auto apply(CoeffOp_&& op) const {
        return MatrixCoeffWiseOp<XprType_, CoeffOp_>(derived(), std::forward<CoeffOp_>(op));
    }
    // catalogue of coeficient wise operations
    constexpr auto pow(int i) const {
        return internals::make_cwise_op(derived(), [i](const auto& x) { return fdapde::pow(x, i); });
    }
    constexpr auto pow2() const { return pow(2); }
    constexpr auto abs() const {
        return internals::make_cwise_op(derived(), [](const auto& x) { return fdapde::abs(x); });
    }
    constexpr decltype(auto) sqrt() const {
        return internals::make_cwise_op(derived(), [](const auto& x) { return fdapde::sqrt(x); });
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
        requires(std::is_arithmetic_v<Scalar_> && requires(typename XprType::Scalar x, Scalar_ s) { x + s; })
    constexpr XprType& operator+=(Scalar_ rhs) {
        internals::scalar_cwise_assignment_executor::run(derived(), rhs, [](auto& l, const Scalar_& r) { l += r; });
        return derived();
    }
    template <typename RhsXprType_> constexpr XprType& operator+=(const MatrixCoeffWiseExpr<RhsXprType_>& rhs) {
        using executor = typename XprType::assignment_executor;
        executor::run(derived(), rhs.derived(), [](auto& l, const auto& r) { l += r; });
        return derived();
    }
    template <typename Scalar_>
        requires(std::is_arithmetic_v<Scalar_> && requires(typename XprType::Scalar x, Scalar_ s) { x - s; })
    constexpr XprType& operator-=(Scalar_ rhs) {
        internals::scalar_cwise_assignment_executor::run(derived(), rhs, [](auto& l, const Scalar_& r) { l -= r; });
        return derived();
    }
    template <typename RhsXprType_> constexpr XprType& operator-=(const MatrixCoeffWiseExpr<RhsXprType_>& rhs) {
        using executor = typename XprType::assignment_executor;
        executor::run(derived(), rhs.derived(), [](auto& l, const auto& r) { l -= r; });
        return derived();
    }
    template <typename Scalar_>
        requires(std::is_arithmetic_v<Scalar_> && requires(typename XprType::Scalar x, Scalar_ s) { x * s; })
    constexpr XprType& operator*=(Scalar_ rhs) {
        internals::scalar_cwise_assignment_executor::run(derived(), rhs, [](auto& l, const Scalar_& r) { l *= r; });
        return derived();
    }
    template <typename RhsXprType_> constexpr XprType& operator*=(const MatrixCoeffWiseExpr<RhsXprType_>& rhs) {
        using executor = typename XprType::assignment_executor;
        executor::run(derived(), rhs.derived(), [](auto& l, const auto& r) { l *= r; });
        return derived();
    }
    template <typename Scalar_>
        requires(std::is_arithmetic_v<Scalar_> && requires(typename XprType::Scalar x, Scalar_ s) { x / s; })
    constexpr XprType& operator/=(Scalar_ rhs) {
        internals::scalar_cwise_assignment_executor::run(derived(), rhs, [](auto& l, const Scalar_& r) { l /= r; });
        return derived();
    }
    template <typename RhsXprType_> constexpr XprType& operator/=(const MatrixCoeffWiseExpr<RhsXprType_>& rhs) {
        using executor = typename XprType::assignment_executor;
        executor::run(derived(), rhs.derived(), [](auto& l, const auto& r) { l /= r; });
        return derived();
    }
    // reinrepret cwise expression as matrix expression
    constexpr auto mwise() { return internals::mwise_wrapper<XprType>(derived()); }
    constexpr auto mwise() const { return internals::mwise_wrapper<const XprType>(derived()); }
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
