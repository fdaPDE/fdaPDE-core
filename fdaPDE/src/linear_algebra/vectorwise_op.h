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

#ifndef __FDAPDE_LINALG_VECTORWISE_OP_H__
#define __FDAPDE_LINALG_VECTORWISE_OP_H__

#include "header_check.h"

namespace fdapde {

// definition of matrix vector-wise operations: row- or column-wise reductions and assignments
  
namespace internals {

// Lazy expression representing a row-wise or column-wise reduction with a custom operator
template <typename XprType, typename ReductionOp, int ByRow>
struct partial_matrix_redux_op : public MatrixExpr<partial_matrix_redux_op<XprType, ReductionOp, ByRow>> {
   private:
    using Base = MatrixExpr<partial_matrix_redux_op<XprType, ReductionOp, ByRow>>;
    using XprTypeNested = internals::ref_select_t<std::conditional_t<XprType::ReadOnly, const XprType, XprType>>;
   public:
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = ByRow ? 1 : XprType::Rows;
    static constexpr int Cols = ByRow ? XprType::Cols : 1;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = XprType::ReadOnly;

    template <typename XprType_>
        requires(std::is_constructible_v<XprTypeNested, XprType_>)
    constexpr partial_matrix_redux_op(XprType_&& xpr, Scalar init, ReductionOp op) noexcept :
        xpr_(std::forward<XprType_>(xpr)), init_(init), op_(op) { }

    constexpr Scalar operator()(int i, int j) const {
        fdapde_assert(i >= 0 && i < rows() && j >= 0 && j < cols());
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
struct MatrixVectorWiseOp : public MatrixExpr<MatrixVectorWiseOp<XprType, ByRow>> {
   private:
    using Base = MatrixExpr<MatrixVectorWiseOp<XprType, ByRow>>;
    using XprTypeNested = internals::ref_select_t<std::conditional_t<XprType::ReadOnly, const XprType, XprType>>;
   public:
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = ByRow ? 1 : XprType::Rows;
    static constexpr int Cols = ByRow ? XprType::Cols : 1;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = XprType::ReadOnly;

    template <typename XprType_>
        requires(std::is_constructible_v<XprTypeNested, XprType_>)
    constexpr MatrixVectorWiseOp(XprType_&& xpr) noexcept : xpr_(std::forward<XprType_>(xpr)) { }

    // generic redux operator
    template <typename XprType_, typename Scalar_, typename ReductionOp>
        requires(requires(ReductionOp op, Scalar a, Scalar b) {
            { op(a, b) } -> std::convertible_to<Scalar>;
        })
    constexpr auto redux(XprType_&& xpr, Scalar_ init, ReductionOp&& op) const {
        fdapde_static_assert(
          std::is_convertible_v<Scalar_ FDAPDE_COMMA Scalar>, INVALID_SCALAR_INIT_TYPE_IN_REDUX_OPERATION);
        return internals::partial_matrix_redux_op<XprType, ReductionOp, ByRow>(std::forward<XprType_>(xpr), init, op);
    }
    // standard reductions
    constexpr auto sum() const {
        return redux(xpr_, Scalar(0), [](Scalar tmp, Scalar x) { return tmp + x; });
    }
    constexpr auto prod() const {
        return redux(xpr_, Scalar(1), [](Scalar tmp, Scalar x) { return tmp * x; });
    }
    constexpr auto mean() const {
        Scalar size_ = ByRow ? xpr_.rows() : xpr_.cols();
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
    constexpr auto norm() const { return squared_norm().cwise().sqrt(); }
    // L^\infty norm
    constexpr auto inf_norm() const {
        return redux(xpr_, std::numeric_limits<Scalar>::min(), [](Scalar tmp, Scalar x) {
            Scalar x_abs = fdapde::abs(x);
            return tmp > x_abs ? tmp : x_abs;
        });
    }

    // vector-wise assignment
    template <typename XprType_> constexpr MatrixVectorWiseOp& operator=(const MatrixExpr<XprType_>& rhs) {
        partial_redux_inplace_loop_(rhs, [](Scalar& a, Scalar b) { a = b; });
        return *this;
    }
    template <typename XprType_> constexpr MatrixVectorWiseOp& operator+=(const MatrixExpr<XprType_>& rhs) {
        partial_redux_inplace_loop_(rhs, [](Scalar& a, Scalar b) { a += b; });
        return *this;
    }
    template <typename XprType_> constexpr MatrixVectorWiseOp& operator-=(const MatrixExpr<XprType_>& rhs) {
        partial_redux_inplace_loop_(rhs, [](Scalar& a, Scalar b) { a -= b; });
        return *this;
    }

    // vectorwise comparison
    template <typename XprType_>
    friend constexpr bool operator==(const MatrixVectorWiseOp& lhs, const MatrixExpr<XprType_>& rhs) {
        fdapde_static_assert(
          (internals::is_dynamic_sized_v<MatrixVectorWiseOp> || internals::is_dynamic_sized_v<XprType_> ||
           internals::same_static_shape_v<MatrixVectorWiseOp FDAPDE_COMMA XprType_>),
          INVALID_VECTORWISE_COMPARISON__OPERANDS_HAVE_DIFFERENT_SIZES);
        if constexpr (internals::is_dynamic_sized_v<MatrixVectorWiseOp> || internals::is_dynamic_sized_v<XprType_>) {
            fdapde_assert(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols());
        }
        const auto& d2 = rhs.derived();
        for (int i = 0, n = lhs.xpr_.rows(); i < n; ++i) {
            for (int j = 0, m = lhs.xpr_.cols(); j < m; ++j) {
                if (lhs.xpr_(i, j) != d2[ByRow ? j : i]) { return false; }
            }
        }
        return true;
    }
    template <typename XprType_>
    friend constexpr bool operator==(const MatrixExpr<XprType_>& lhs, const MatrixVectorWiseOp& rhs) {
        return rhs == lhs;
    }
    template <typename XprType_>
    friend constexpr bool operator!=(const MatrixVectorWiseOp& lhs, const MatrixExpr<XprType_>& rhs) {
        return !(lhs == rhs);
    }
    template <typename XprType_>
    friend constexpr bool operator!=(const MatrixExpr<XprType_>& lhs, const MatrixVectorWiseOp& rhs) {
        return !(rhs == lhs);
    }

    // observers
    constexpr int rows() const { return ByRow ? Rows : xpr_.rows(); }
    constexpr int cols() const { return ByRow ? xpr_.cols() : Cols; }
   private:
    // internals
    template <typename XprType_, typename Operator_>
    constexpr void partial_redux_inplace_loop_(const MatrixExpr<XprType_>& rhs, Operator_&& op) {
        constexpr int XprRows = XprType_::Rows, XprCols = XprType_::Cols;
        fdapde_static_assert(
          (Rows == 1 && XprRows == 1) || (Cols == 1 & XprCols == 1), NO_MATCHING_SIZES_IN_VECTORWISE_ASSIGNMENT);
        fdapde_static_assert(XprType::ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION);
        fdapde_assert(Rows == 1 && xpr_.cols() == rhs.cols() || Cols == 1 && xpr_.rows() == rhs.rows());

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
template <typename XprType> struct MatrixRowWiseOp : public MatrixVectorWiseOp<XprType, 0> {
    using Base = MatrixVectorWiseOp<XprType, 0>;
    template <typename XprType_> constexpr explicit MatrixRowWiseOp(XprType_&& xpr) noexcept : Base(xpr) { }
    using Base::operator=;
};
// col-wise matrix reduction expression
template <typename XprType> struct MatrixColWiseOp : public MatrixVectorWiseOp<XprType, 1> {
    using Base = MatrixVectorWiseOp<XprType, 1>;
    template <typename XprType_> constexpr explicit MatrixColWiseOp(XprType_&& xpr) noexcept : Base(xpr) { }
    using Base::operator=;
};  
  
}   // namespace fdapde

#endif // __FDAPDE_LINALG_VECTORWISE_OP_H__
