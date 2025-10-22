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

#ifndef __FDAPDE_LINALG_COEFFWISE_OP_H__
#define __FDAPDE_LINALG_COEFFWISE_OP_H__

#include "header_check.h"

namespace fdapde {

template <typename XprType_> struct MatrixCoeffWiseExpr;

// expression node for the coefficient wise operation between an expression and a scalar value
template <typename XprType_, typename CoeffOp>
struct MatrixCoeffWiseOp : public MatrixCoeffWiseExpr<MatrixCoeffWiseOp<XprType_, CoeffOp>> {
   private:
    using XprType = std::decay_t<XprType_>;
    using XprTypeNested = internals::ref_select_t<const XprType>;
   public:
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    static constexpr int StorageOrder = XprType::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename XprType__>
        requires(std::is_constructible_v<XprTypeNested, XprType__>)
    constexpr MatrixCoeffWiseOp(XprType__&& xpr, CoeffOp op) : xpr_(std::forward<XprType__>(xpr)), op_(op) { }
    constexpr decltype(auto) operator()(int i, int j) const { return op_(xpr_(i, j)); }
    constexpr decltype(auto) operator[](int i) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return op_(xpr_[i]);
    }
    constexpr int rows() const { return Rows != Dynamic ? Rows : xpr_.rows(); }
    constexpr int cols() const { return Cols != Dynamic ? Cols : xpr_.cols(); }
    // coeffwise executor, enable nested coeffwise operations
    template <typename CoeffOp_> constexpr auto apply(CoeffOp_&& op) const {
        return MatrixCoeffWiseOp<MatrixCoeffWiseOp<XprType_, CoeffOp>, CoeffOp_>(*this, op);
    }
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
    constexpr MatrixCoeffWiseBinOp(LhsXprType__&& lhs, RhsXprType__&& rhs, BinaryOp op) :
        lhs_(std::forward<LhsXprType__>(lhs)), rhs_(std::forward<RhsXprType__>(rhs)), op_(op) {
        if constexpr (internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType>) {
            fdapde_assert(
              std::cmp_equal(lhs_.rows() FDAPDE_COMMA rhs_.rows()) &&
              std::cmp_equal(lhs_.cols() FDAPDE_COMMA rhs_.cols()));
        }
    }
    constexpr decltype(auto) operator()(int i, int j) const { return op_(lhs_(i, j), rhs_(i, j)); }
    constexpr decltype(auto) operator[](int i) const {
        fdapde_static_assert(
          (LhsXprType::Cols == 1 && RhsXprType::Cols == 1) || (LhsXprType::Rows == 1 && RhsXprType::Rows == 1),
          THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return op_(lhs_[i], rhs_[i]);
    }
    constexpr int rows() const { return Rows != Dynamic ? Rows : lhs_.rows(); }
    constexpr int cols() const { return Cols != Dynamic ? Cols : lhs_.cols(); }
    // coeffwise executor, enable nested coeffwise operations
    template <typename CoeffOp_> constexpr auto apply(CoeffOp_&& op) const {
        return MatrixCoeffWiseOp<MatrixCoeffWiseBinOp<LhsXprType_, RhsXprType_, BinaryOp>, CoeffOp_>(*this, op);
    }
   private:
    LhsXprTypeNested lhs_;
    RhsXprTypeNested rhs_;
    BinaryOp op_;
};

// return type of MatrixExpr::cwise(), represents the entry point for coefficient wise algebra
template <typename XprType_> struct MatrixCoeffWiseProxy : public MatrixCoeffWiseExpr<MatrixCoeffWiseProxy<XprType_>> {
   private:
    using XprType = std::decay_t<XprType_>;
    using XprTypeNested = internals::ref_select_t<const XprType>;
   public:
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    static constexpr int StorageOrder = XprType::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename XprType__>
        requires(std::is_constructible_v<XprTypeNested, XprType__>)
    constexpr MatrixCoeffWiseProxy(XprType__&& xpr) : xpr_(std::forward<XprType__>(xpr)) { }
    constexpr decltype(auto) operator()(int i, int j) const { return xpr_(i, j); }
    constexpr decltype(auto) operator[](int i) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return xpr_[i];
    }
    constexpr int rows() const { return Rows != Dynamic ? Rows : xpr_.rows(); }
    constexpr int cols() const { return Cols != Dynamic ? Cols : xpr_.cols(); }
    // coeffwise executor
    template <typename CoeffOp> constexpr auto apply(CoeffOp&& op) const {
        return MatrixCoeffWiseOp<XprType_, CoeffOp>(xpr_, op);
    }
   private:
    XprTypeNested xpr_;
};

template <typename XprType_> struct MatrixCoeffWiseExpr : public MatrixExpr<XprType_> {
    using XprType = std::decay_t<XprType_>;

    // observers
    constexpr const XprType& derived() const { return static_cast<const XprType&>(*this); }
    constexpr XprType& derived() { return static_cast<XprType&>(*this); }
    // catalogue of coeficient wise operations
    constexpr auto abs() const {
        using Scalar = typename XprType::Scalar;
        return derived().apply([](Scalar x) { return fdapde::abs(x); });
    }
    constexpr auto pow(int i) const {
        using Scalar = typename XprType::Scalar;
        return derived().apply([i](Scalar x) { return fdapde::pow(x, i); });
    }
    constexpr auto pow2() const { return pow(2); }
    constexpr auto sqrt() const {
        using Scalar = typename XprType::Scalar;
        fdapde_static_assert(std::is_floating_point_v<Scalar>, THIS_METHOD_IS_FOR_FLOATING_POINT_MATRICES_ONLY);
        return derived().apply([](Scalar x) { return fdapde::sqrt(x); });
    }
    constexpr auto inv() const {
        using Scalar = typename XprType::Scalar;
        fdapde_static_assert(std::is_floating_point_v<Scalar>, THIS_METHOD_IS_FOR_FLOATING_POINT_MATRICES_ONLY);
        return derived().apply([](Scalar x) { return 1.0 / x; });
    }
    constexpr auto exp() const {
        using Scalar = typename XprType::Scalar;
        fdapde_static_assert(std::is_floating_point_v<Scalar>, THIS_METHOD_IS_FOR_FLOATING_POINT_MATRICES_ONLY);
        return derived().apply([](Scalar x) { return fdapde::exp(x); });
    }
    constexpr auto log() const {
        using Scalar = typename XprType::Scalar;
        fdapde_static_assert(std::is_floating_point_v<Scalar>, THIS_METHOD_IS_FOR_FLOATING_POINT_MATRICES_ONLY);
        return derived().apply([](Scalar x) { return fdapde::log(x); });
    }
};

// coeffwise arithmetic

// coeffwise addition
template <typename LhsXprType_, typename RhsXprType_>
constexpr auto operator+(const MatrixCoeffWiseExpr<LhsXprType_>& lhs, const MatrixCoeffWiseExpr<RhsXprType_>& rhs) {
    return MatrixCoeffWiseBinOp<LhsXprType_, RhsXprType_, std::plus<>>(lhs.derived(), rhs.derived(), std::plus<>());
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType> && requires(typename XprType_::Scalar x, ScalarType s) { x + s; })
constexpr auto operator+(const MatrixCoeffWiseExpr<XprType_>& lhs, const ScalarType& rhs) {
    return lhs.derived().apply([rhs](auto x) { return x + rhs; });
}
template <typename XprType_, typename ScalarType>
constexpr auto operator+(const ScalarType& lhs, const MatrixCoeffWiseExpr<XprType_>& rhs) {
    return rhs + lhs;
}

// coeffwise difference
template <typename LhsXprType_, typename RhsXprType_>
constexpr auto operator-(const MatrixCoeffWiseExpr<LhsXprType_>& lhs, const MatrixCoeffWiseExpr<RhsXprType_>& rhs) {
    return MatrixCoeffWiseBinOp<LhsXprType_, RhsXprType_, std::minus<>>(lhs.derived(), rhs.derived(), std::minus<>());
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType> && requires(typename XprType_::Scalar x, ScalarType s) { x - s; })
constexpr auto operator-(const MatrixCoeffWiseExpr<XprType_>& lhs, const ScalarType& rhs) {
    return lhs.derived().apply([rhs](auto x) { return x - rhs; });
}
template <typename XprType_, typename ScalarType>
constexpr auto operator-(const ScalarType& lhs, const MatrixCoeffWiseExpr<XprType_>& rhs) {
    return rhs - lhs;
}

// coeffwise multiplication
template <typename LhsXprType_, typename RhsXprType_>
constexpr auto operator*(const MatrixCoeffWiseExpr<LhsXprType_>& lhs, const MatrixCoeffWiseExpr<RhsXprType_>& rhs) {
    return MatrixCoeffWiseBinOp<LhsXprType_, RhsXprType_, std::multiplies<>>(
      lhs.derived(), rhs.derived(), std::multiplies<>());
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType> && requires(typename XprType_::Scalar x, ScalarType s) { x * s; })
constexpr auto operator*(const MatrixCoeffWiseExpr<XprType_>& lhs, const ScalarType& rhs) {
    return lhs.derived().apply([rhs](auto x) { return x * rhs; });
}
template <typename XprType_, typename ScalarType>
constexpr auto operator*(const ScalarType& lhs, const MatrixCoeffWiseExpr<XprType_>& rhs) {
    return rhs * lhs;
}

// coeffwise division
template <typename LhsXprType_, typename RhsXprType_>
constexpr auto operator/(const MatrixCoeffWiseExpr<LhsXprType_>& lhs, const MatrixCoeffWiseExpr<RhsXprType_>& rhs) {
    return MatrixCoeffWiseBinOp<LhsXprType_, RhsXprType_, std::divides<>>(
      lhs.derived(), rhs.derived(), std::divides<>());
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType> && requires(typename XprType_::Scalar x, ScalarType s) { x / s; })
constexpr auto operator/(const MatrixCoeffWiseExpr<XprType_>& lhs, const ScalarType& rhs) {
    return lhs.derived().apply([rhs](auto x) { return x / rhs; });
}

// coeffwise comparison
template <typename XprType_, typename ComparisonOp>
struct MatrixCoeffWiseComparisonOp : public MatrixExpr<MatrixCoeffWiseComparisonOp<XprType_, ComparisonOp>> {
   private:
    using XprType = std::decay_t<XprType_>;
    using XprTypeNested = internals::ref_select_t<const XprType>;
   public:
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    static constexpr int StorageOrder = XprType::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    MatrixCoeffWiseComparisonOp(const XprType_& xpr, ComparisonOp op) : xpr_(xpr), op_(op) { }
    constexpr decltype(auto) operator()(int i, int j) const {
        fdapde_assert(i >= 0 && i < xpr_.rows() && j >= 0 && j < xpr_.cols());
        return op_(xpr_(i, j));
    }
    constexpr decltype(auto) operator[](int i) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return op_(xpr_[i]);
    }
    constexpr int rows() const { return Rows != Dynamic ? Rows : xpr_.rows(); }
    constexpr int cols() const { return Cols != Dynamic ? Cols : xpr_.cols(); }
   private:
    XprTypeNested xpr_;
    ComparisonOp op_;
};
// strict comparison
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType> && requires(typename XprType_::Scalar x, ScalarType s) { x < s; })
constexpr auto operator<(const MatrixCoeffWiseExpr<XprType_>& lhs, const ScalarType& rhs) {
    return MatrixCoeffWiseComparisonOp(lhs.derived(), [rhs](const auto& x) { return x < rhs; });
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType> && requires(typename XprType_::Scalar x, ScalarType s) { x > s; })
constexpr auto operator>(const MatrixCoeffWiseExpr<XprType_>& lhs, const ScalarType& rhs) {
    return MatrixCoeffWiseComparisonOp(lhs.derived(), [rhs](const auto& x) { return x > rhs; });
}
template <typename XprType_, typename ScalarType>
constexpr auto operator<(const ScalarType& lhs, const MatrixCoeffWiseExpr<XprType_>& rhs) {
    return rhs > lhs;
}
template <typename XprType_, typename ScalarType>
constexpr auto operator>(const ScalarType& lhs, const MatrixCoeffWiseExpr<XprType_>& rhs) {
    return rhs < lhs;
}
// weak comparison
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType> && requires(typename XprType_::Scalar x, ScalarType s) { x <= s; })
constexpr auto operator<=(const MatrixCoeffWiseExpr<XprType_>& lhs, const ScalarType& rhs) {
    return MatrixCoeffWiseComparisonOp(lhs.derived(), [rhs](const auto& x) { return x <= rhs; });
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType> && requires(typename XprType_::Scalar x, ScalarType s) { x >= s; })
constexpr auto operator>=(const MatrixCoeffWiseExpr<XprType_>& lhs, const ScalarType& rhs) {
    return MatrixCoeffWiseComparisonOp(lhs.derived(), [rhs](const auto& x) { return x >= rhs; });
}
template <typename XprType_, typename ScalarType>
constexpr auto operator<=(const ScalarType& lhs, const MatrixCoeffWiseExpr<XprType_>& rhs) {
    return rhs >= lhs;
}
template <typename XprType_, typename ScalarType>
constexpr auto operator>=(const ScalarType& lhs, const MatrixCoeffWiseExpr<XprType_>& rhs) {
    return rhs <= lhs;
}
// equality comparison
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType> && requires(typename XprType_::Scalar x, ScalarType s) { x == s; })
constexpr auto operator==(const MatrixCoeffWiseExpr<XprType_>& lhs, const ScalarType& rhs) {
    return MatrixCoeffWiseComparisonOp(lhs.derived(), [rhs](const auto& x) { return x == rhs; });
}
template <typename XprType_, typename ScalarType>
constexpr auto operator==(const ScalarType& lhs, const MatrixCoeffWiseExpr<XprType_>& rhs) {
    return rhs == lhs;
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType> && requires(typename XprType_::Scalar x, ScalarType s) { x != s; })
constexpr auto operator!=(const MatrixCoeffWiseExpr<XprType_>& lhs, const ScalarType& rhs) {
    return MatrixCoeffWiseComparisonOp(lhs.derived(), [rhs](const auto& x) { return x != rhs; });
}
template <typename XprType_, typename ScalarType>
constexpr auto operator!=(const ScalarType& lhs, const MatrixCoeffWiseExpr<XprType_>& rhs) {
    return rhs != lhs;
}

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_COEFFWISE_OP_H__
