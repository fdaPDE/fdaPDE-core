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

template <typename XprType, typename CoeffOp>
struct MatrixCoeffWiseOp : public MatrixCoeffWiseExpr<MatrixCoeffWiseOp<XprType, CoeffOp>> {
   private:
    using Base = MatrixCoeffWiseExpr<MatrixCoeffWiseOp<XprType, CoeffOp>>;
    using XprTypeNested = internals::ref_select_t<const XprType>;
    using XprTypeClean = std::decay_t<XprType>;
   public:
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = XprTypeClean::Rows;
    static constexpr int Cols = XprTypeClean::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename XprType_>
        requires(std::is_constructible_v<XprTypeNested, XprType_>)
    constexpr MatrixCoeffWiseOp(XprType_&& xpr, CoeffOp op) : xpr_(std::forward<XprType_>(xpr)), op_(op) { }
    constexpr Scalar operator()(int i, int j) const { return op_(xpr_(i, j)); }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return op_(xpr_[i]);
    }
    constexpr int rows() const { return Rows != Dynamic ? Rows : xpr_.rows(); }
    constexpr int cols() const { return Cols != Dynamic ? Cols : xpr_.cols(); }
    // coeffwise executor, enable nested coeffwise operations
    template <typename CoeffOp_> constexpr auto apply(CoeffOp_&& op) const {
        return MatrixCoeffWiseOp<MatrixCoeffWiseOp<XprType, CoeffOp>, CoeffOp_>(*this, op);
    }
   private:
    XprTypeNested xpr_;
    CoeffOp op_;
};

// return type of MatrixExpr::cwise(), represents the entry point for coefficient wise algebra
template <typename XprType> struct MatrixCoeffWiseProxy : public MatrixCoeffWiseExpr<MatrixCoeffWiseProxy<XprType>> {
   private:
    using Base = MatrixCoeffWiseExpr<MatrixCoeffWiseProxy<XprType>>;
    using XprTypeNested = internals::ref_select_t<const XprType>;
   public:
    using Scalar = typename XprType::Scalar;

    template <typename XprType_>
        requires(std::is_constructible_v<XprTypeNested, XprType_>)
    constexpr MatrixCoeffWiseProxy(XprType_&& xpr) : xpr_(std::forward<XprType_>(xpr)) { }
    // coeffwise executor
    template <typename CoeffOp> constexpr auto apply(CoeffOp&& op) const {
        return MatrixCoeffWiseOp<XprType, CoeffOp>(xpr_, op);
    }
   private:
    XprTypeNested xpr_;
};

template <typename XprType_> struct MatrixCoeffWiseExpr : public MatrixExpr<XprType_> {
    using XprType = XprType_;

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
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator+(const MatrixCoeffWiseExpr<XprType_>& lhs, ScalarType rhs) {
    return lhs.derived().apply([rhs](auto x) { return x + rhs; });
}
template <typename XprType_, typename ScalarType>
constexpr auto operator+(ScalarType lhs, const MatrixCoeffWiseExpr<XprType_>& rhs) {
    return rhs + lhs;
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator-(const MatrixCoeffWiseExpr<XprType_>& lhs, ScalarType rhs) {
    return lhs.derived().apply([rhs](auto x) { return x - rhs; });
}
template <typename XprType_, typename ScalarType>
constexpr auto operator-(ScalarType lhs, const MatrixCoeffWiseExpr<XprType_>& rhs) {
    return rhs - lhs;
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator*(const MatrixCoeffWiseExpr<XprType_>& lhs, ScalarType rhs) {
    return lhs.derived().apply([rhs](auto x) { return x * rhs; });
}
template <typename XprType_, typename ScalarType>
constexpr auto operator*(ScalarType lhs, const MatrixCoeffWiseExpr<XprType_>& rhs) {
    return rhs * lhs;
}
template <typename XprType_, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator/(const MatrixCoeffWiseExpr<XprType_>& lhs, ScalarType rhs) {
    return lhs.derived().apply([rhs](auto x) { return x / rhs; });
}

// template <typename LhsXprType_, typename RhsXprType_>
// struct MatrixCwiseComparisonOp :
//     public BoolMatrixExpr<
//       (LhsXprType_::Rows == Dynamic || RhsXprType_::Cols == Dynamic) ? Dynamic : LhsXprType_::Rows,
//       (LhsXprType_::Cols == Dynamic || RhsXprType_::Cols == Dynamic) ? Dynamic : LhsXprType_::Cols,
//       MatrixCwiseComparisonOp<LhsXprType_, RhsXprType_>> {
//     fdapde_static_assert(
//       internals::is_dynamic_sized_v<LhsXprType_> || internals::is_dynamic_sized_v<RhsXprType_> ||
//         internals::same_static_shape_v<LhsXprType_ FDAPDE_COMMA RhsXprType_>,
//       INVALID_BINARY_OPERATION__MATRICES_OF_DIFFERENT_STATIC_SIZE);
//     static constexpr int Rows = 
//       (LhsXprType_::Rows == Dynamic || RhsXprType_::Cols == Dynamic) ? Dynamic : LhsXprType_::Rows;
//     static constexpr int Cols =
//       (LhsXprType_::Cols == Dynamic || RhsXprType_::Cols == Dynamic) ? Dynamic : LhsXprType_::Cols;
//     using Base = BoolMatrixExpr<Rows_, Cols_, MatrixCwiseComparisonOp<Rows_, Cols_, LhsXprType_>>;
//     using LhsXprTypeNested = internals::ref_select_t<const LhsXprType_>;
//     using Scalar = typename Base::Scalar;
//     static constexpr int NestAsRef = 0;
//     static constexpr int Rows = Rows_;
//     static constexpr int Cols = Cols_;
//     static constexpr int ReadOnly = 1;

//     template <typename LhsXprType>
//         requires(std::is_constructible_v<LhsXprTypeNested, LhsXprType>)
//     constexpr MatrixCwiseComparisonOp(LhsXprType&& xpr) : xpr_(std::forward<LhsXprType>(xpr)) { }
//    private:
//     LhsXprTypeNested xpr_;
// };

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_COEFFWISE_OP_H__
