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

#ifndef __FDAPDE_LINALG_TRIANGULAR_H__
#define __FDAPDE_LINALG_TRIANGULAR_H__

#include "header_check.h"

namespace fdapde {

// triangular matrix type system
template <typename XprType> struct TriangularMatrixExpr;
template <typename Scalar_, int Rows_, int Cols_, int ViewMode_, int StorageOrder_ = RowMajor> struct TriangularMatrix;

namespace internals {

struct triangular_assignment_executor {
    template <typename DstXprType, typename SrcXprType, typename AssignmentOp>
        requires(requires(AssignmentOp op, typename DstXprType::Scalar& l, const typename SrcXprType::Scalar& r) {
            { op(l, r) } -> std::same_as<void>;
        })
    static constexpr void run(DstXprType& dst, const SrcXprType& src, AssignmentOp&& op) {
        fdapde_static_assert(DstXprType::ReadOnly == 0, ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION);
        constexpr int ViewMode = DstXprType::ViewMode;
        fdapde_static_assert(
          ViewMode == Upper || ViewMode == Lower, TRIANGULAR_BLOCK_ASSIGNMENT_REQUIRES_EITHER_UPPER_OR_LOWER_VIEW);
        int row = 0, col = 0;
        for (int i = 0, n = dst.rows(); i < n; ++i) {
            for (int j = 0; j < i; ++j) {
                row = ViewMode == Lower ? i : j;
                col = ViewMode == Lower ? j : i;
                op(dst(row, col), src(row, col));
            }
        }
        // assign diagonal
        for (int i = 0, n = dst.rows(); i < n; ++i) { dst(i, i) = src(i, i); }
    }
};

// class wrapping a linear vector to the expression of a triangular matrix. internal usage only
template <int ViewMode_, typename TriangularXprType_>
struct triangular_wrapper : TriangularMatrixExpr<triangular_wrapper<ViewMode_, TriangularXprType_>> {
   private:
    using Base = TriangularMatrixExpr<triangular_wrapper<ViewMode_, TriangularXprType_>>;
    using XprType = std::decay_t<TriangularXprType_>;
    using XprTypeNested = internals::ref_select_t<TriangularXprType_>;
   public:
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = XprType::Rows == 1 ? XprType::Cols : XprType::Rows;
    static constexpr int Cols = XprType::Cols == 1 ? XprType::Rows : XprType::Cols;
    static constexpr int ViewMode = ViewMode_;
    static constexpr int StorageOrder = XprType::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename XprType__>
        requires(std::is_constructible_v<XprTypeNested, XprType__>)
    constexpr triangular_wrapper(XprType__&& xpr) : xpr_(std::forward<XprType__>(xpr)) { }
    constexpr Scalar operator()(int i, int j) const {
        fdapde_assert(i >= 0 && i < xpr_.rows() && j >= 0 && j < xpr_.cols());
        if constexpr (ViewMode == Upper) { return i > j ? Scalar(0) : xpr_(i, j); }
        if constexpr (ViewMode == Lower) { return i < j ? Scalar(0) : xpr_(i, j); }
        if constexpr (ViewMode == UnitUpper) { return i > j ? Scalar(0) : (i == j ? Scalar(1) : xpr_(i, j)); }
        if constexpr (ViewMode == UnitLower) { return i < j ? Scalar(0) : (i == j ? Scalar(1) : xpr_(i, j)); }
    }
    const XprTypeNested& data() const { return xpr_; }
   private:
    XprTypeNested xpr_;
};

// helper cast function
template <int ViewMode_, typename XprType_> auto triangular_cast(XprType_&& xpr) {
    return triangular_wrapper<ViewMode_, XprType_>(xpr);
}

}   // namespace internals

template <typename XprType_> struct TriangularMatrixExpr : public MatrixExpr<XprType_> {
    using XprType = std::decay_t<XprType_>;
    // make derived() point to innermost type
    constexpr const XprType& derived() const { return static_cast<const XprType&>(*this); }
    constexpr XprType& derived() { return static_cast<XprType&>(*this); }
    // inherit assignment from base
    using MatrixExpr<XprType_>::operator=;

    constexpr auto inverse() const {
        using Scalar = typename XprType::Scalar;
        constexpr int Rows = XprType::Rows;
	constexpr int Cols = XprType::Cols;
        constexpr int ViewMode = XprType::ViewMode;

        TriangularMatrix<Scalar, Rows, Cols, ViewMode> inverse_;
        if constexpr (Rows == 1) {
            inverse_(0, 0) = 1. / derived()(0, 0);
        } else if constexpr (Rows == 2) {
            Scalar a = derived()(0, 0);
            Scalar b = derived()(1, 1);
            Vector<Scalar, 3> v;
            v[0] = 1. / a;
            v[2] = 1. / b;
            if constexpr (ViewMode == Lower || ViewMode == UnitLower) { v[1] = -derived()(1, 0) / (a * b); }
            if constexpr (ViewMode == Upper || ViewMode == UnitUpper) { v[1] = -derived()(0, 1) / (a * b); }
            inverse_ = internals::triangular_cast<ViewMode>(v);
        } else if constexpr (Rows == 3) {
            Scalar a = derived()(0, 0);
            Scalar b = derived()(1, 1);
            Scalar c = derived()(2, 2);
            Vector<Scalar, 6> v;
            v[0] = 1. / a;

            v[5] = 1. / c;
            if constexpr (ViewMode == Lower || ViewMode == UnitLower) {
                v[1] = -derived()(1, 0) / (a * b);
                v[2] = 1. / b;
                v[3] = derived()(1, 0) * derived()(2, 1) / (a * b * c) - derived()(2, 0) / (a * c);
                v[4] = -derived()(2, 1) / (b * c);
            }
            if constexpr (ViewMode == Upper || ViewMode == UnitUpper) {
                v[1] = -derived()(0, 1) / (a * b);
                v[2] = derived()(0, 1) * derived()(1, 2) / (a * b * c) - derived()(0, 2) / (a * c);
                v[3] = 1. / b;
                v[4] = -derived()(1, 2) / (b * c);
            }
            inverse_ = internals::triangular_cast<ViewMode>(v);
        } else {
            // general inversion solves linear system A * X = I
            Matrix<Scalar, Rows, Rows> X;
            Vector<Scalar, Rows> b;
            if constexpr (Rows == Dynamic) {
                const int rows = derived().rows();
                int size = rows * (rows + 1) / 2;
                X.resize(size, size);
                b.resize(rows);
            }
            for (int i = 0, n = derived().rows(); i < n; ++i) {
                b[i] = 1;
                X.col(i) = solve(b);
                b[i] = 0;
            }
            inverse_ = X;
        }
        return inverse_;
    }
    // linear system solver Ax = b
    template <typename RhsXprType> constexpr auto solve(const RhsXprType& b) const {
        constexpr int ViewMode = XprType::ViewMode;
        if constexpr (ViewMode == Lower || ViewMode == UnitLower) { return fwd_sub_(b); }
        if constexpr (ViewMode == Upper || ViewMode == UnitUpper) { return bwd_sub_(b); }
    }
    constexpr double determinant() const { return derived().diagonal().prod(); }
    constexpr const auto& data() const { return derived().data(); }
    constexpr auto& data() { return derived().data(); }
   private:
    // forward substitution for lower-triangular matrix, vector rhs
    template <typename RhsXprType>
        requires(RhsXprType::Cols == 1)
    constexpr auto fwd_sub_(const MatrixExpr<RhsXprType>& b) const {
        constexpr int ViewMode = XprType::ViewMode;
        fdapde_static_assert(
          ViewMode == Lower || ViewMode == UnitLower, THIS_METHOD_IS_FOR_LOWER_TRIANGULAR_MATRICES_ONLY);
        const RhsXprType& b_ = b.derived();
        fdapde_assert(b_.rows() == derived().rows() && b_.cols() == 1);
        using Scalar = typename XprType::Scalar;
        constexpr int Rows = XprType::Rows, RhsRows = RhsXprType::Rows;
        Vector<Scalar, Rows> x;
        if constexpr (RhsRows == Dynamic) { x.resize(b_.rows()); }
        x[0] = b_[0] / derived()(0, 0);
        int rows = b_.rows();
        for (int i = 1; i < rows; ++i) {
            Scalar sum = 0;
            for (int j = 0; j < i; ++j) sum += derived()(i, j) * x[j];
            if constexpr (ViewMode != UnitLower) { x[i] = (b_[i] - sum) / derived()(i, i); }
        }
        return x;
    }
    // forward substitution for lower-triangular matrix, matrix rhs (cache-friendly approach)
    template <typename RhsXprType>
        requires(RhsXprType::Cols > 1 || RhsXprType::Cols == Dynamic)
    constexpr auto fwd_sub_(const MatrixExpr<RhsXprType>& B) const {
        constexpr int ViewMode = XprType::ViewMode;
        fdapde_static_assert(
          ViewMode == Lower || ViewMode == UnitLower, THIS_METHOD_IS_FOR_LOWER_TRIANGULAR_MATRICES_ONLY);
        const RhsXprType& B_ = B.derived();
        fdapde_assert(B_.rows() == derived().rows() && B_.cols() > 1);
        using Scalar = typename XprType::Scalar;
        constexpr int RhsRows = RhsXprType::Rows, RhsCols = RhsXprType::Cols;
        Matrix<Scalar, RhsRows, RhsCols> X;
        if constexpr (RhsRows == Dynamic || RhsCols == Dynamic) { X.resize(B_.rows(), B_.cols()); }
        const int rows = B_.rows();
        const int cols = B_.cols();
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) { X(i, j) = B_(i, j); }
            for (int k = 0; k < i; ++k) {
                for (int j = 0; j < cols; ++j) { X(i, j) -= derived()(i, k) * B_(k, j); }
            }
            if constexpr (ViewMode != UnitLower) {
                for (int j = 0; j < cols; ++j) { X(i, j) = X(i, j) / derived()(i, i); }
            }
        }
        return X;
    }
    // backward substitution for upper-triangular matrix, vector rhs
    template <typename RhsXprType>
        requires(RhsXprType::Cols == 1)
    constexpr auto bwd_sub_(const MatrixExpr<RhsXprType>& b) const {
        constexpr int ViewMode = XprType::ViewMode;
        fdapde_static_assert(
          ViewMode == Upper || ViewMode == UnitUpper, THIS_METHOD_IS_FOR_UPPER_TRIANGULAR_MATRICES_ONLY);
        const RhsXprType& b_ = b.derived();
        fdapde_assert(b_.rows() == derived().rows() && b_.cols() == 1);
        using Scalar = typename XprType::Scalar;
        constexpr int Rows = XprType::Rows;
        Vector<Scalar, Rows> x;
        if constexpr (Rows == Dynamic) { x.resize(b_.rows()); }
        int rows = b_.rows();
        x[rows - 1] = b_[rows - 1] / derived()(rows - 1, rows - 1);
        for (int i = rows - 2; i >= 0; --i) {
            Scalar sum = 0;
            for (int j = i + 1; j < rows; ++j) sum += derived()(i, j) * x[j];
            if constexpr (ViewMode != UnitUpper) { x[i] = (b_[i] - sum) / derived()(i, i); }
        }
        return x;
    }
    // backward substitution for upper-triangular matrix, matrix rhs (cache-friendly approach)
    template <typename RhsXprType>
        requires(RhsXprType::Cols > 1 || RhsXprType::Cols == Dynamic)
    constexpr auto bwd_sub_(const MatrixExpr<RhsXprType>& B) const {
        constexpr int ViewMode = XprType::ViewMode;
        fdapde_static_assert(
          ViewMode == Upper || ViewMode == UnitUpper, THIS_METHOD_IS_FOR_UPPER_TRIANGULAR_MATRICES_ONLY);
        const RhsXprType& B_ = B.derived();
        fdapde_assert(B_.rows() == derived().rows() && B_.cols() > 1);
        using Scalar = typename XprType::Scalar;
        constexpr int RhsRows = XprType::Rows, RhsCols = XprType::Cols;
        Matrix<Scalar, RhsRows, RhsCols> X;
        if constexpr (RhsRows == Dynamic || RhsCols == Dynamic) { X.resize(B_.rows(), B_.cols()); }
        const int rows = B_.rows();
        const int cols = B_.cols();
        for (int i = rows - 1; i >= 0; --i) {
            for (int j = 0; j < cols; ++j) { X(i, j) = B_(i, j); }
            for (int k = i + 1; k < rows; ++k) {
                for (int j = 0; j < cols; ++j) { X(i, j) -= derived()(i, k) * X(k, j); }
            }
            if constexpr (ViewMode != UnitUpper) {
                for (int j = 0; j < cols; ++j) { X(i, j) = X(i, j) / derived()(i, i); }
            }
        }
        return X;
    }
};

// expression of the triangular part of a matrix
template <int ViewMode_, typename XprType_>
struct Triangular : public TriangularMatrixExpr<Triangular<ViewMode_, XprType_>> {
   private:
    using Base = TriangularMatrixExpr<Triangular<ViewMode_, XprType_>>;
    using XprType = std::decay_t<XprType_>;
    using XprTypeNested = internals::ref_select_t<XprType_>;
   public:
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    static constexpr int StorageOrder = XprType::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ViewMode = ViewMode_;
    static constexpr int ReadOnly = XprType::ReadOnly || (ViewMode == UnitLower || ViewMode == UnitUpper);
    using assignment_executor = internals::triangular_assignment_executor;

    // constructor
    template <typename XprType__>
        requires(std::is_constructible_v<XprTypeNested, XprType__>)
    constexpr explicit Triangular(XprType__&& xpr) : xpr_(std::forward<XprType__>(xpr)) {
        fdapde_static_assert(
          Rows == Dynamic || Cols == Dynamic || Rows == Cols, THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
        if constexpr (Rows == Dynamic || Cols == Dynamic) { fdapde_assert(xpr.rows() == xpr.cols()); }
    }
    // access
    constexpr Scalar operator()(int i, int j) const {
        fdapde_assert(i >= 0 && i < xpr_.rows() && j >= 0 && j < xpr_.cols());
        if constexpr (ViewMode == Upper) { return i > j ? Scalar(0) : xpr_(i, j); }
        if constexpr (ViewMode == Lower) { return i < j ? Scalar(0) : xpr_(i, j); }
        if constexpr (ViewMode == UnitUpper) { return i > j ? Scalar(0) : (i == j ? Scalar(1) : xpr_(i, j)); }
        if constexpr (ViewMode == UnitLower) { return i < j ? Scalar(0) : (i == j ? Scalar(1) : xpr_(i, j)); }
    }
    constexpr Scalar& operator()(int i, int j) {
        fdapde_static_assert(ViewMode == Upper || ViewMode == Lower, WRITE_ACCESS_TO_READ_ONLY_EXPRESSION);
        fdapde_assert(i >= 0 && i < xpr_.rows() && j >= 0 && j < xpr_.cols());
        if constexpr (ViewMode == Upper) { return i > j ? Scalar(0) : xpr_(i, j); }
        if constexpr (ViewMode == Lower) { return i < j ? Scalar(0) : xpr_(i, j); }
    }
    // observers
    constexpr int rows() const { return xpr_.rows(); }
    constexpr int cols() const { return xpr_.cols(); }
    const XprTypeNested& data() const { return xpr_; }
   private:
    XprTypeNested xpr_;
};

// base class for triangular matrices
template <typename Scalar_, int Rows_, int Cols_, int ViewMode_, int StorageOrder_, typename TriangularMatrixType>
class TriangularMatrixBase : public TriangularMatrixExpr<TriangularMatrixType> {
   protected:
    using Base = TriangularMatrixExpr<TriangularMatrixType>;
    using Base::derived;
   public:
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int ViewMode = ViewMode_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int ReadOnly = std::is_const_v<Scalar_>;

    template <int ViewMode__, typename Scalar__>
        requires(std::is_same_v<Scalar, std::decay_t<Scalar__>> && (ViewMode__ == Lower || ViewMode__ == Upper))
    struct triangular_proxy {
        using Scalar = Scalar__;

        constexpr triangular_proxy(Scalar__* data, int i, int j, int size) :
            data_(data), index_(compute_linear_index_(i, j, size)), b_(ViewMode__ == Upper ? (i <= j) : (i >= j)) { }
        template <typename T>
            requires(std::is_convertible_v<T, Scalar>)
        constexpr triangular_proxy& operator=(T value) {
            if constexpr (ViewMode__ == Upper) {
                if (b_) data_[index_] = value;   // do nothing otherwise
            }
            if constexpr (ViewMode__ == Lower) {
                if (b_) data_[index_] = value;   // do nothing otherwise
            }
            return *this;
        }
        constexpr operator Scalar() const {
            if constexpr (ViewMode__ == Lower) { return b_ ? data_[index_] : Scalar(0); }
            if constexpr (ViewMode__ == Upper) { return b_ ? data_[index_] : Scalar(0); }
        }
        constexpr operator Scalar() {
            if constexpr (ViewMode__ == Lower) { return b_ ? data_[index_] : Scalar(0); }
            if constexpr (ViewMode__ == Upper) { return b_ ? data_[index_] : Scalar(0); }
        }
       private:
        constexpr int compute_linear_index_(int i, int j, int size) const {
            if constexpr (StorageOrder == RowMajor) {
                return ViewMode == Upper ? i * (2 * size - i + 1) / 2 + (j - i) : i * (i + 1) / 2 + j;
            }
            if constexpr (StorageOrder == ColMajor) {
                return ViewMode == Lower ? j * (2 * size - j + 1) / 2 + (i - j) : j * (j + 1) / 2 + i;
            }
        }
        Scalar* data_;
        int index_;
        bool b_;
    };
    using reference = triangular_proxy<ViewMode, Scalar>;
    using const_reference = triangular_proxy<ViewMode, const Scalar>;

    constexpr TriangularMatrixBase() : rows_(Rows == Dynamic ? 0 : Rows), cols_(Cols == Dynamic ? 0 : Cols) { }
    constexpr TriangularMatrixBase(double rows, double cols) : rows_(rows), cols_(cols) {
        // size can be a floating point value as a result of calling triangular_cast() on a vector which cannot map to a
        // triangular matrix. This checks guarantees that "there are enought values" to make a triangular matrix
        fdapde_assert(rows == fdapde::floor(rows) && cols == fdapde::floor(cols) && rows == cols);
    }
    // access
    constexpr const_reference operator()(int i, int j) const {
        fdapde_assert(i >= 0 && i < rows_ && j >= 0 && j < cols_);
        return const_reference(derived().data(), i, j, rows_);
    }
    constexpr reference operator()(int i, int j) {
        fdapde_static_assert(ReadOnly == 0, WRITE_ACCESS_TO_READ_ONLY_LOCATION);
        fdapde_assert(i >= 0 && i < rows_ && j >= 0 && j < cols_);
        return reference(derived().data(), i, j, rows_);
    }
    // observers
    constexpr int rows() const { return rows_; }
    constexpr int cols() const { return cols_; }
   protected:
    int rows_, cols_;
};

// triangular matrix subalgebra of the associative algebra of square matrices
template <typename LhsXprType, typename RhsXprType>
    requires(LhsXprType::ViewMode == RhsXprType::ViewMode)   // if summing different ViewMode, exit from triangular TS
constexpr auto operator+(const TriangularMatrixExpr<LhsXprType>& lhs, const TriangularMatrixExpr<RhsXprType>& rhs) {
    return internals::triangular_cast<LhsXprType::ViewMode>(lhs.data() + rhs.data());
}
template <typename LhsXprType, typename RhsXprType>
    requires(LhsXprType::ViewMode == RhsXprType::ViewMode)   // if summing different ViewMode, exit from triangular TS
constexpr auto operator-(const TriangularMatrixExpr<LhsXprType>& lhs, const TriangularMatrixExpr<RhsXprType>& rhs) {
    return internals::triangular_cast<LhsXprType::ViewMode>(lhs.data() - rhs.data());
}
template <typename XprType, typename CoeffType>
    requires(std::is_arithmetic_v<CoeffType>)
constexpr auto operator*(const TriangularMatrixExpr<XprType>& lhs, CoeffType rhs) {
    return internals::triangular_cast<XprType::ViewMode>(lhs.data() * rhs);
}
template <typename XprType, typename CoeffType>
    requires(std::is_arithmetic_v<CoeffType>)
constexpr auto operator*(CoeffType lhs, const TriangularMatrixExpr<XprType>& rhs) {
    return internals::triangular_cast<XprType::ViewMode>(lhs * rhs.data());
}
template <typename XprType, typename CoeffType>
    requires(std::is_arithmetic_v<CoeffType>)
constexpr auto operator/(const TriangularMatrixExpr<XprType>& lhs, CoeffType rhs) {
    return internals::triangular_cast<XprType::ViewMode>(lhs.data() / rhs);
}

// specialized products
namespace internals {

// expressions of the (i,j)-th entry of the product between a triangular and a dense matrix expression
template <int ProductMode> struct triangular_matrix_product_executor {
    template <typename LhsXprType_, typename RhsXprType_>
    static constexpr auto run(int i, int j, const LhsXprType_& lhs, const RhsXprType_& rhs) {
        using TriXprType = std::decay_t<std::conditional_t<ProductMode == LhsMode, LhsXprType_, RhsXprType_>>;
        using MtxXprType = std::decay_t<std::conditional_t<ProductMode == LhsMode, RhsXprType_, LhsXprType_>>;
        using Scalar = promote_type_t<typename TriXprType::Scalar, typename MtxXprType::Scalar>;
        constexpr int ViewMode = TriXprType::ViewMode;
        constexpr int is_lower = ViewMode == Lower || ViewMode == UnitLower;
        Scalar prod = 0;
        const int h = (ProductMode == LhsMode ? (is_lower ? 0 : i) : (is_lower ? j : 0));
        const int n =
          (ProductMode == LhsMode ? (is_lower ? (i + 1) : (lhs.rows() - i)) : (is_lower ? (lhs.cols() - j) : (j + 1)));
        for (int k = 0; k < n; ++k) { prod += lhs(i, h + k) * rhs(h + k, j); }
        return prod;
    }
};

// expressions of the (i,j)-th entry of the product between triangular expressions
struct triangular_triangular_product_executor {
    template <typename LhsXprType_, typename RhsXprType_>
    static constexpr auto run(int i, int j, const LhsXprType_& lhs, const RhsXprType_& rhs) {
        using LhsXprType = std::decay_t<LhsXprType_>;
        using RhsXprType = std::decay_t<RhsXprType_>;
        using Scalar = promote_type_t<typename LhsXprType::Scalar, typename RhsXprType::Scalar>;
        constexpr int LhsViewMode = LhsXprType::ViewMode;
        constexpr int RhsViewMode = RhsXprType::ViewMode;
        if constexpr (LhsViewMode != RhsViewMode) {   // just multiply the diagonals
            return i == j ? lhs(i, i) * rhs(i, i) : Scalar(0);
        } else {
            // operands have same ViewMode
            constexpr int is_lower = LhsViewMode == Lower || LhsViewMode == UnitLower;
            Scalar prod = 0;
            const int h = is_lower ? (i + 1) : (j + 1);
            if ((is_lower && (i < j)) || (!is_lower && i > j)) { return Scalar(0); }
            for (int k = 0; k < h; ++k) prod += lhs(i, k) * rhs(k, j);
            return prod;
        }
    }
};

}   // namespace internals

// Triangular * M
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator*(const TriangularMatrixExpr<LhsXprType>& lhs, const MatrixExpr<RhsXprType>& rhs) {
    return MatrixMultiplicationOp<LhsXprType, RhsXprType, internals::triangular_matrix_product_executor<LhsMode>> {
      lhs.derived(), rhs.derived()};
}
// M * Triangular
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator*(const MatrixExpr<LhsXprType>& lhs, const TriangularMatrixExpr<RhsXprType>& rhs) {
    return MatrixMultiplicationOp<LhsXprType, RhsXprType, internals::triangular_matrix_product_executor<RhsMode>> {
      lhs.derived(), rhs.derived()};
}
// Triangular * Triangular
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator*(const TriangularMatrixExpr<LhsXprType>& lhs, const TriangularMatrixExpr<RhsXprType>& rhs) {
    return MatrixMultiplicationOp<LhsXprType, RhsXprType, internals::triangular_triangular_product_executor> {
      lhs.derived(), rhs.derived()}
      .template triangular_block<LhsXprType::ViewMode>();   // close wrt triangular subalgebra
}

// owning storage diagonal matrix
template <typename Scalar_, int Rows_, int Cols_, int ViewMode_, int StorageOrder_>
struct TriangularMatrix :
    public TriangularMatrixBase<
      Scalar_, Rows_, Cols_, ViewMode_, StorageOrder_,
      TriangularMatrix<Scalar_, Rows_, Cols_, ViewMode_, StorageOrder_>> {
    fdapde_static_assert(
      Rows_ == Dynamic || Cols_ == Dynamic || Rows_ == Cols_, THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
    fdapde_static_assert(ViewMode_ == Lower || ViewMode_ == Upper, THIS_CLASS_IS_FOR_LOWER_OR_UPPER_VIEW_MODE_ONLY);
   private:
    using Base = TriangularMatrixBase<
      Scalar_, Rows_, Cols_, ViewMode_, StorageOrder_,
      TriangularMatrix<Scalar_, Rows_, Cols_, ViewMode_, StorageOrder_>>;
    static constexpr int StorageSize = Rows_ == Dynamic ? Dynamic : (Rows_ * (Rows_ + 1) / 2);
    using StorageType = Vector<Scalar_, StorageSize>;
   public:
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int NestAsRef = 0;
    static constexpr int StorageOrder = StorageOrder_;
    using assignment_executor = internals::triangular_assignment_executor;

    constexpr TriangularMatrix() : Base() { }
    // copy semantic
    constexpr TriangularMatrix(const TriangularMatrix& rhs) : Base(rhs.rows(), rhs.cols()) { clone_(rhs); }
    constexpr TriangularMatrix& operator=(const TriangularMatrix& rhs) {
        clone_(rhs);
        return *this;
    }
    constexpr explicit TriangularMatrix(int rows, int cols) :
        Base(), data_() {   // initialize with no sizes, resize will set them
        fdapde_static_assert(Rows == Dynamic || Cols == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_MATRICES_ONLY);
        fdapde_assert(rows == cols);
        resize(rows, cols);
    }
    template <typename RhsXprType_>
    constexpr TriangularMatrix(const MatrixExpr<RhsXprType_>& rhs) : Base(rhs.rows(), rhs.cols()) {
        fdapde_assert(StorageSize == Dynamic || rhs.rows() == rhs.cols());
        clone_(rhs.derived());
    }
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr explicit TriangularMatrix(const std::vector<Scalar__>& data) :
        Base(compute_shape_(data.size()), compute_shape_(data.size())), data_() {
        if constexpr (Rows == Dynamic || Cols == Dynamic) { data_.resize(data.size()); }
        fdapde_assert(data_.size() == data.size());
        for (int i = 0, n = data_.size(); i < n; ++i) { data_[i] = data[i]; }
    }
    template <typename Scalar__, std::size_t Size>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr explicit TriangularMatrix(const Scalar__ (&data)[Size]) :
        Base(compute_shape_(Size), compute_shape_(Size)), data_() {
        fdapde_static_assert(
          Rows_ != Dynamic && Cols_ != Dynamic && StorageSize == Size, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        for (int i = 0, n = data_.size(); i < n; ++i) { data_[i] = data[i]; }
    }
    // modifiers
    void resize(int rows, int cols) {
        fdapde_static_assert(Rows == Dynamic || Cols == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_MATRICES_ONLY);
        if (std::cmp_equal(this->rows_, rows) && std::cmp_equal(this->cols_, cols)) return;
        // update and reallocate memory
        this->rows_ = rows;
        this->cols_ = cols;
        data_.resize(this->rows_ * (this->cols_ + 1) / 2);
        return;
    }
    // data pointers
    constexpr const Scalar_* data() const { return data_.data(); }
    constexpr Scalar_* data() { return data_.data(); }
   private:
    constexpr int compute_shape_(int i) {   // given x : x = 0.5 * (n * (n + 1)), computes n
        return (fdapde::sqrt(static_cast<double>(1 + 8 * i)) - 1) / 2;
    }
    template <typename RhsXprType> constexpr void clone_(const RhsXprType& rhs) {
        if constexpr (Rows == Dynamic || Cols == Dynamic) { resize(rhs.rows(), rhs.cols()); }
        assignment_executor::run(*this, rhs, [](auto&& l, const auto& r) { l = r; });
        return;
    }
    StorageType data_;
};

// triangular view of an existing block of data
template <typename Scalar_, int Rows_, int Cols_, int ViewMode_, int StorageOrder_>
class TriangularMatrixView :
    public TriangularMatrixBase<
      Scalar_, Rows_, Cols_, ViewMode_, StorageOrder_,
      TriangularMatrixView<Scalar_, Rows_, Cols_, ViewMode_, StorageOrder_>> {
    fdapde_static_assert(
      Rows_ == Dynamic || Cols_ == Dynamic || Rows_ == Cols_, THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
    fdapde_static_assert(ViewMode_ == Lower || ViewMode_ == Upper, THIS_CLASS_IS_FOR_LOWER_OR_UPPER_VIEW_MODE_ONLY);
    using Base = TriangularMatrixBase<
      Scalar_, Rows_, Cols_, ViewMode_, StorageOrder_,
      TriangularMatrixView<Scalar_, Rows_, Cols_, ViewMode_, StorageOrder_>>;
    using StorageType = std::add_pointer_t<Scalar_>;
   public:
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int NestAsRef = 1;
    static constexpr int ReadOnly = std::is_const_v<Scalar_>;
    using assignment_executor = internals::triangular_assignment_executor;

    // constructors
    constexpr TriangularMatrixView() : Base(), data_(nullptr) { }
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr explicit TriangularMatrixView(Scalar__* data) : Base(), data_(data) {
        fdapde_static_assert(Rows != Dynamic && Cols != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
    }
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr TriangularMatrixView(Scalar__* data, int rows, int cols) : Base(rows, cols), data_(data) {
        fdapde_assert(rows > 0 && cols > 0 && rows == cols);
    }
    // data pointers
    constexpr const StorageType& data() const { return data_; }
    constexpr StorageType& data() { return data_; }
   private:
    StorageType data_;
};

// type aliases
template <typename Scalar, int Rows, int Cols, int StorageOrder = RowMajor>
using UpperTriangularMatrix = TriangularMatrix<Scalar, Rows, Cols, Upper, StorageOrder>;
template <typename Scalar, int Rows, int Cols, int StorageOrder = RowMajor>
using LowerTriangularMatrix = TriangularMatrix<Scalar, Rows, Cols, Lower, StorageOrder>;
template <typename Scalar, int Rows, int Cols, int StorageOrder = RowMajor>
using UpperTriangularMatrixView = TriangularMatrixView<Scalar, Rows, Cols, Upper, StorageOrder>;
template <typename Scalar, int Rows, int Cols, int StorageOrder = RowMajor>
using LowerTriangularMatrixView = TriangularMatrixView<Scalar, Rows, Cols, Lower, StorageOrder>;

// detection trait
template <typename XprType> struct is_triangular_matrix {
    static constexpr bool value = std::is_base_of_v<TriangularMatrixExpr<std::decay_t<XprType>>, XprType>;
};
template <typename XprType> static constexpr bool is_triangular_matrix_v = is_triangular_matrix<XprType>::value;
  
}   // namespace fdapde

#endif   // __FDAPDE_LINALG_TRIANGULAR_H__
