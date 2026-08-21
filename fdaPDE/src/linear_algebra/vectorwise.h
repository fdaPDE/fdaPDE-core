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

#ifndef __FDAPDE_LINALG_VECTORWISE_H__
#define __FDAPDE_LINALG_VECTORWISE_H__

#include "header_check.h"

namespace fdapde {

// definition of matrix vector-wise operations: row- or column-wise reductions and assignments
  
namespace internals {

// Lazy expression representing a row-wise or column-wise reduction with a custom operator
template <typename XprType_, typename ReductionOp, int ByRow>
struct partial_matrix_redux_op : public MatrixExpr<partial_matrix_redux_op<XprType_, ReductionOp, ByRow>> {
   private:
    using Base = MatrixExpr<partial_matrix_redux_op<XprType_, ReductionOp, ByRow>>;
    using XprType = std::decay_t<XprType_>;
    using XprTypeNested = internals::ref_select_t<XprType_>;
   public:
    using Scalar = std::remove_cv_t<typename XprType::Scalar>;
    static constexpr int Rows = ByRow ? 1 : XprType::Rows;
    static constexpr int Cols = ByRow ? XprType::Cols : 1;
    static constexpr int StorageOrder = XprType::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename XprType__>
        requires(
          !std::same_as<std::remove_cvref_t<XprType__>, partial_matrix_redux_op> &&
          internals::safely_nestable<XprTypeNested, XprType__>)
    constexpr partial_matrix_redux_op(XprType__&& xpr, Scalar init, ReductionOp op) :
        xpr_(std::forward<XprType__>(xpr)), init_(init), op_(std::move(op)) { }

    constexpr Scalar operator()(int i, int j) const {
        if (i < 0 || i >= rows() || j < 0 || j >= cols()) {
            throw std::out_of_range("vector-wise reduction index out of range");
        }
        Scalar res(init_);
        const int k = ByRow ? j : i;
        const auto& xpr = std::as_const(xpr_);
        for (int h = 0, size_ = ByRow ? xpr.rows() : xpr.cols(); h < size_; ++h) {
            res = op_(res, ByRow ? xpr(h, k) : xpr(k, h));
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

template <typename XprType_, int ByRow>
struct MatrixVectorWiseOp : public MatrixExpr<MatrixVectorWiseOp<XprType_, ByRow>> {
   private:
    using Base = MatrixExpr<MatrixVectorWiseOp<XprType_, ByRow>>;
    using XprType = std::decay_t<XprType_>;
    using XprTypeNested = internals::ref_select_t<XprType_>;

    static constexpr bool static_axis_compatible_(int lhs, int rhs) {
        return lhs == Dynamic || rhs == Dynamic || lhs == rhs;
    }
    template <typename RhsXprType_>
    static constexpr bool statically_compatible_rhs_ = [] {
        using RhsXprType = std::decay_t<RhsXprType_>;
        if constexpr (ByRow == 0) {
            return static_axis_compatible_(XprType::Rows, RhsXprType::Rows) &&
              static_axis_compatible_(1, RhsXprType::Cols);
        } else {
            return static_axis_compatible_(1, RhsXprType::Rows) &&
              static_axis_compatible_(XprType::Cols, RhsXprType::Cols);
        }
    }();
   public:
    using Scalar = std::remove_cv_t<typename XprType::Scalar>;
    static constexpr int Rows = ByRow ? 1 : XprType::Rows;
    static constexpr int Cols = ByRow ? XprType::Cols : 1;
    static constexpr int StorageOrder = XprType::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly =
      std::is_const_v<std::remove_reference_t<XprType_>> || XprType::ReadOnly;

    template <typename XprType__>
        requires(
          !std::same_as<std::remove_cvref_t<XprType__>, MatrixVectorWiseOp> &&
          internals::safely_nestable<XprTypeNested, XprType__>)
    constexpr MatrixVectorWiseOp(XprType__&& xpr) : xpr_(std::forward<XprType__>(xpr)) { }

    // generic redux operator
    template <typename XprType__, typename Scalar_, typename ReductionOp>
        requires(requires(const std::decay_t<ReductionOp>& op, Scalar& accumulated, const Scalar& value) {
            { op(accumulated, value) } -> std::convertible_to<Scalar>;
        })
    constexpr auto redux(XprType__&& xpr, Scalar_ init, ReductionOp&& op) const {
        fdapde_static_assert(
          std::is_convertible_v<Scalar_ FDAPDE_COMMA Scalar>, INVALID_SCALAR_INIT_TYPE_IN_REDUX_OPERATION);
        using StoredOp = std::decay_t<ReductionOp>;
        return internals::partial_matrix_redux_op<XprType_, StoredOp, ByRow>(
          std::forward<XprType__>(xpr), static_cast<Scalar>(init), std::forward<ReductionOp>(op));
    }
    // standard reductions
    constexpr auto sum() const {
        return redux(xpr_, Scalar(0), [](Scalar tmp, Scalar x) { return tmp + x; });
    }
    constexpr auto prod() const {
        return redux(xpr_, Scalar(1), [](Scalar tmp, Scalar x) { return tmp * x; });
    }
    constexpr auto mean() const {
        const int size = reduced_size_();
        require_defined_reduction_(size);
        return sum()
          .cwise()
          .apply([size](Scalar total) -> Scalar { return static_cast<Scalar>(total / size); })
          .mwise();
    }
    constexpr auto max() const {
        require_defined_reduction_(reduced_size_());
        return redux(
          xpr_, std::numeric_limits<Scalar>::lowest(), [](Scalar tmp, Scalar x) { return tmp > x ? tmp : x; });
    }
    constexpr auto min() const {
        require_defined_reduction_(reduced_size_());
        return redux(xpr_, std::numeric_limits<Scalar>::max(), [](Scalar tmp, Scalar x) { return tmp < x ? tmp : x; });
    }
    // L^2 squared norm
    constexpr auto squared_norm() const {
        return redux(xpr_, Scalar(0), [](Scalar tmp, Scalar x) { return tmp + x * x; });
    }
    constexpr auto norm() const requires(std::floating_point<Scalar>) {
        return redux(xpr_, Scalar(0), [](Scalar norm, Scalar value) {
            return internals::scale_safe_hypot(norm, value);
        });
    }
    // L^\infty norm
    constexpr auto inf_norm() const {
        return redux(xpr_, Scalar(0), [](Scalar tmp, Scalar x) {
            Scalar x_abs = fdapde::abs(x);
            return tmp > x_abs ? tmp : x_abs;
        });
    }

    // vector-wise assignment
    template <typename RhsXprType_>
        requires(ReadOnly == 0 && statically_compatible_rhs_<RhsXprType_>)
    constexpr MatrixVectorWiseOp& operator=(const MatrixExpr<RhsXprType_>& rhs) & {
        require_compatible_shape_(rhs);
        using RhsXprType = std::decay_t<RhsXprType_>;
        using RhsScalar = std::remove_cv_t<typename RhsXprType::Scalar>;
        Matrix<RhsScalar, RhsXprType::Rows, RhsXprType::Cols, RhsXprType::StorageOrder> tmp(rhs);
        partial_redux_inplace_loop_(tmp, [](Scalar& a, Scalar b) { a = b; });
        return *this;
    }
    template <typename RhsXprType_>
        requires(ReadOnly == 0 && statically_compatible_rhs_<RhsXprType_>)
    constexpr MatrixVectorWiseOp operator=(const MatrixExpr<RhsXprType_>& rhs) && {
        static_cast<MatrixVectorWiseOp&>(*this).operator=(rhs);
        return *this;
    }
    template <typename RhsXprType_>
        requires(ReadOnly == 0 && statically_compatible_rhs_<RhsXprType_>)
    constexpr MatrixVectorWiseOp& operator+=(const MatrixExpr<RhsXprType_>& rhs) & {
        require_compatible_shape_(rhs);
        using RhsXprType = std::decay_t<RhsXprType_>;
        using RhsScalar = std::remove_cv_t<typename RhsXprType::Scalar>;
        Matrix<RhsScalar, RhsXprType::Rows, RhsXprType::Cols, RhsXprType::StorageOrder> tmp(rhs);
        partial_redux_inplace_loop_(tmp, [](Scalar& a, Scalar b) { a += b; });
        return *this;
    }
    template <typename RhsXprType_>
        requires(ReadOnly == 0 && statically_compatible_rhs_<RhsXprType_>)
    constexpr MatrixVectorWiseOp operator+=(const MatrixExpr<RhsXprType_>& rhs) && {
        static_cast<MatrixVectorWiseOp&>(*this).operator+=(rhs);
        return *this;
    }
    template <typename RhsXprType_>
        requires(ReadOnly == 0 && statically_compatible_rhs_<RhsXprType_>)
    constexpr MatrixVectorWiseOp& operator-=(const MatrixExpr<RhsXprType_>& rhs) & {
        require_compatible_shape_(rhs);
        using RhsXprType = std::decay_t<RhsXprType_>;
        using RhsScalar = std::remove_cv_t<typename RhsXprType::Scalar>;
        Matrix<RhsScalar, RhsXprType::Rows, RhsXprType::Cols, RhsXprType::StorageOrder> tmp(rhs);
        partial_redux_inplace_loop_(tmp, [](Scalar& a, Scalar b) { a -= b; });
        return *this;
    }
    template <typename RhsXprType_>
        requires(ReadOnly == 0 && statically_compatible_rhs_<RhsXprType_>)
    constexpr MatrixVectorWiseOp operator-=(const MatrixExpr<RhsXprType_>& rhs) && {
        static_cast<MatrixVectorWiseOp&>(*this).operator-=(rhs);
        return *this;
    }

    // vectorwise comparison
    template <typename RhsXprType_>
        requires(statically_compatible_rhs_<RhsXprType_>)
    friend constexpr bool operator==(const MatrixVectorWiseOp& lhs, const MatrixExpr<RhsXprType_>& rhs) {
        lhs.require_compatible_shape_(rhs);
        const auto& d2 = rhs.derived();
        const auto& xpr = std::as_const(lhs.xpr_);
        for (int i = 0, n = xpr.rows(); i < n; ++i) {
            for (int j = 0, m = xpr.cols(); j < m; ++j) {
                if (xpr(i, j) != (ByRow ? d2(0, j) : d2(i, 0))) { return false; }
            }
        }
        return true;
    }
    template <typename LhsXprType_>
        requires(statically_compatible_rhs_<LhsXprType_>)
    friend constexpr bool operator==(const MatrixExpr<LhsXprType_>& lhs, const MatrixVectorWiseOp& rhs) {
        return rhs == lhs;
    }
    template <typename RhsXprType_>
        requires(statically_compatible_rhs_<RhsXprType_>)
    friend constexpr bool operator!=(const MatrixVectorWiseOp& lhs, const MatrixExpr<RhsXprType_>& rhs) {
        return !(lhs == rhs);
    }
    template <typename LhsXprType_>
        requires(statically_compatible_rhs_<LhsXprType_>)
    friend constexpr bool operator!=(const MatrixExpr<LhsXprType_>& lhs, const MatrixVectorWiseOp& rhs) {
        return !(rhs == lhs);
    }
    template <typename RhsXprType_>
        requires(!statically_compatible_rhs_<RhsXprType_>)
    friend constexpr bool operator==(const MatrixVectorWiseOp&, const MatrixExpr<RhsXprType_>&) = delete;
    template <typename LhsXprType_>
        requires(!statically_compatible_rhs_<LhsXprType_>)
    friend constexpr bool operator==(const MatrixExpr<LhsXprType_>&, const MatrixVectorWiseOp&) = delete;
    template <typename RhsXprType_>
        requires(!statically_compatible_rhs_<RhsXprType_>)
    friend constexpr bool operator!=(const MatrixVectorWiseOp&, const MatrixExpr<RhsXprType_>&) = delete;
    template <typename LhsXprType_>
        requires(!statically_compatible_rhs_<LhsXprType_>)
    friend constexpr bool operator!=(const MatrixExpr<LhsXprType_>&, const MatrixVectorWiseOp&) = delete;

    // observers
    constexpr int rows() const { return ByRow ? Rows : xpr_.rows(); }
    constexpr int cols() const { return ByRow ? xpr_.cols() : Cols; }
   private:
    // internals
    constexpr int reduced_size_() const { return ByRow ? xpr_.rows() : xpr_.cols(); }
    constexpr void require_defined_reduction_(int reduced_size) const {
        const int output_size = ByRow ? xpr_.cols() : xpr_.rows();
        if (reduced_size == 0 && output_size > 0) {
            throw std::domain_error("vector-wise reduction requires a nonempty reduced axis");
        }
    }
    template <typename RhsXprType_>
    constexpr void require_compatible_shape_(const MatrixExpr<RhsXprType_>& rhs) const {
        const bool compatible = ByRow == 0 ?
          (rhs.rows() == xpr_.rows() && rhs.cols() == 1) :
          (rhs.rows() == 1 && rhs.cols() == xpr_.cols());
        if (!compatible) {
            throw std::invalid_argument("vector-wise operation requires a matching broadcast vector");
        }
    }
    template <typename RhsXprType_, typename Operator_>
    constexpr void partial_redux_inplace_loop_(const MatrixExpr<RhsXprType_>& rhs, Operator_&& op) {
        int inner_size_ = ByRow ? xpr_.rows() : xpr_.cols();
        int outer_size_ = ByRow ? xpr_.cols() : xpr_.rows();
        const auto& rhs_derived = rhs.derived();
        for (int i = 0; i < outer_size_; ++i) {
            for (int j = 0; j < inner_size_; ++j) {
                if constexpr (ByRow == 0) op(xpr_(i, j), rhs_derived(i, 0));
                if constexpr (ByRow == 1) op(xpr_(j, i), rhs_derived(0, i));
            }
        }
        return;
    }
  
    XprTypeNested xpr_;
};

// row-wise matrix reduction expression
template <typename XprType> struct MatrixRowWiseOp : public MatrixVectorWiseOp<XprType, 0> {
    using Base = MatrixVectorWiseOp<XprType, 0>;
    using Base::Base;
    using Base::operator=;
};
// col-wise matrix reduction expression
template <typename XprType> struct MatrixColWiseOp : public MatrixVectorWiseOp<XprType, 1> {
    using Base = MatrixVectorWiseOp<XprType, 1>;
    using Base::Base;
    using Base::operator=;
};
  
}   // namespace fdapde

#endif // __FDAPDE_LINALG_VECTORWISE_H__
