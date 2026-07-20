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

namespace fdapde::linalg {

// triangular matrix type system
template <typename XprType> struct TriangularMatrixExpr;
template <typename Scalar_, int Rows_, int Cols_, int ViewMode_, int StorageOrder_ = RowMajor> struct TriangularMatrix;

namespace internals {

// given a linear memory block of size x representing a flattened n x n triangular matrix, computes n, i.e. finds the
// integer solution n to the quadratic equation x = 0.5 * (n * (n + 1))
constexpr int compute_triangular_shape(int x) {
    if (x < 0) return -1;
    int n = 0;
    while (static_cast<long long>(n) * (n + 1) / 2 < x) ++n;
    return static_cast<long long>(n) * (n + 1) / 2 == x ? n : -1;
}

// if xpr is a vector-expression, rescales the pair (i, j) to perform a triangular-like access on an n x n matrix,
// otherwise forwards (i, j) to xpr
template <int ViewMode, int StorageOrder, typename XprType>
constexpr decltype(auto) triangular_access(XprType& xpr, int i, int j, [[maybe_unused]] int n) {
    if constexpr (std::is_arithmetic_v<XprType>) {
        return xpr;
    } else {
        if constexpr (std::is_pointer_v<std::remove_reference_t<XprType>>) {
            if constexpr (StorageOrder == RowMajor) {
                return xpr[ViewMode == Upper ? i * (2 * n - i + 1) / 2 + (j - i) : i * (i + 1) / 2 + j];
            } else {
                return xpr[ViewMode == Lower ? j * (2 * n - j + 1) / 2 + (i - j) : j * (j + 1) / 2 + i];
            }
        } else if constexpr (is_vector_shaped_v<XprType>) {
            if constexpr (StorageOrder == RowMajor) {
                return xpr[ViewMode == Upper ? i * (2 * n - i + 1) / 2 + (j - i) : i * (i + 1) / 2 + j];
            }
            if constexpr (StorageOrder == ColMajor) {
                return xpr[ViewMode == Lower ? j * (2 * n - j + 1) / 2 + (i - j) : j * (j + 1) / 2 + i];
            }
        } else {
            return xpr(i, j);
        }
    }
}

struct triangular_assignment_executor {
    template <typename DstMatrixType, typename SrcXprType, typename AssignmentOp>
    static constexpr void run(DstMatrixType& dst, const SrcXprType& src, AssignmentOp&& op) {
        fdapde_static_assert(DstMatrixType::ReadOnly == 0, ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION);
        constexpr int ViewMode = DstMatrixType::ViewMode;
        fdapde_static_assert(
          ViewMode == Upper || ViewMode == Lower, TRIANGULAR_BLOCK_ASSIGNMENT_REQUIRES_EITHER_UPPER_OR_LOWER_VIEW);
        if constexpr (!std::is_arithmetic_v<SrcXprType>) {
            fdapde_static_assert(
              internals::is_dynamic_sized_v<DstMatrixType> || internals::is_dynamic_sized_v<SrcXprType> ||
                internals::same_static_shape_v<DstMatrixType FDAPDE_COMMA SrcXprType> ||
                internals::is_vector_shaped_v<SrcXprType>,
              INVALID_ASSIGNMENT__DIFFERENT_LHS_AND_RHS_STATIC_SHAPES);
            const bool valid_shape = internals::is_vector_shaped_v<SrcXprType> ?
                                       compute_triangular_shape(src.size()) == dst.rows() :
                                       (src.rows() == dst.rows() && src.cols() == dst.cols());
            if (!valid_shape) {
                fdapde_assert(valid_shape);
                return;
            }
        }

        constexpr int DstStorageOrder = DstMatrixType::StorageOrder;
        constexpr int SrcStorageOrder = []() {
            if constexpr (std::is_arithmetic_v<SrcXprType>) {
                return DstMatrixType::StorageOrder;
            } else {
                return SrcXprType::StorageOrder;
            }
        }();
        decltype(auto) dst_rep = dst.rep();
        int row = 0, col = 0;
        for (int i = 0, n = dst.rows(); i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                row = ViewMode == Lower ? i : j;
                col = ViewMode == Lower ? j : i;
                decltype(auto) target = triangular_access<ViewMode, DstStorageOrder>(dst_rep, row, col, n);
                op(target, triangular_access<ViewMode, SrcStorageOrder>(src, row, col, n));
            }
        }
    }
};

// class wrapping a matrix expression to the expression of a triangular matrix. internal usage only
template <int ViewMode_, typename TriangularXprType_>
struct triangular_wrapper : TriangularMatrixExpr<triangular_wrapper<ViewMode_, TriangularXprType_>> {
   private:
    using Base = TriangularMatrixExpr<triangular_wrapper<ViewMode_, TriangularXprType_>>;
    using XprType = std::decay_t<TriangularXprType_>;
    using XprTypeNested = fdapde::internals::ref_select_t<TriangularXprType_>;
   public:
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = is_vector_shaped_v<XprType> ?
                                  compute_triangular_shape(XprType::Rows == 1 ? XprType::Cols : XprType::Rows) :
                                  XprType::Rows;
    static constexpr int Cols = Rows;
    static constexpr int ViewMode = ViewMode_;
    static constexpr int StorageOrder = XprType::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    constexpr triangular_wrapper(const triangular_wrapper&) = default;
    template <typename XprType__>
        requires(
          !std::same_as<std::remove_cvref_t<XprType__>, triangular_wrapper> &&
          std::is_constructible_v<XprTypeNested, XprType__>)
    constexpr triangular_wrapper(XprType__&& xpr) :
        xpr_(std::forward<XprType__>(xpr)),
        rows_(is_vector_shaped_v<XprType> ? compute_triangular_shape(xpr.size()) : xpr.rows()),
        cols_(is_vector_shaped_v<XprType> ? compute_triangular_shape(xpr.size()) : xpr.cols()) {
        fdapde_assert(
          xpr.rows() > 0 && xpr.cols() > 0 && (xpr.rows() == 1 || xpr.cols() == 1 || xpr.rows() == xpr.cols()));
        if (rows_ < 0 || cols_ < 0) {
            fdapde_assert(rows_ >= 0 && cols_ >= 0);
            rows_ = 0;
            cols_ = 0;
        }
    }
    constexpr Scalar operator()(int i, int j) const {
        fdapde_assert(i >= 0 && i < rows_ && j >= 0 && j < cols_);
        if constexpr (ViewMode == Upper) {
            return i > j ? Scalar(0) : triangular_access<ViewMode, StorageOrder>(xpr_, i, j, rows_);
        }
        if constexpr (ViewMode == Lower) {
            return i < j ? Scalar(0) : triangular_access<ViewMode, StorageOrder>(xpr_, i, j, rows_);
        }
    }
    // observers
    constexpr int rows() const { return rows_; }
    constexpr int cols() const { return cols_; }
    constexpr const XprTypeNested& rep() const { return xpr_; }
   private:
    XprTypeNested xpr_;
    int rows_, cols_;
};

// helper cast function
template <int ViewMode_, typename XprType_> auto triangular_cast(XprType_&& xpr) {
    return triangular_wrapper<ViewMode_, XprType_>(std::forward<XprType_>(xpr));
}

}   // namespace internals

template <typename XprType_> struct TriangularMatrixExpr : public MatrixExpr<XprType_> {
    using XprType = std::decay_t<XprType_>;
    using MatrixExpr<XprType_>::derived;
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
            if constexpr (ViewMode == Lower) { v[1] = -derived()(1, 0) / (a * b); }
            if constexpr (ViewMode == Upper) { v[1] = -derived()(0, 1) / (a * b); }
            inverse_ = internals::triangular_cast<ViewMode>(v);
        } else if constexpr (Rows == 3) {
            Scalar a = derived()(0, 0);
            Scalar b = derived()(1, 1);
            Scalar c = derived()(2, 2);
            Vector<Scalar, 6> v;
            v[0] = 1. / a;

            v[5] = 1. / c;
            if constexpr (ViewMode == Lower) {
                v[1] = -derived()(1, 0) / (a * b);
                v[2] = 1. / b;
                v[3] = derived()(1, 0) * derived()(2, 1) / (a * b * c) - derived()(2, 0) / (a * c);
                v[4] = -derived()(2, 1) / (b * c);
            }
            if constexpr (ViewMode == Upper) {
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
                X.resize(rows, rows);
                b.resize(rows);
            }
            for (int i = 0, n = derived().rows(); i < n; ++i) {
                for (int j = 0; j < n; ++j) b[j] = 0;
                b[i] = 1;
                X.col(i) = solve(b);
            }
            inverse_ = X;
        }
        return inverse_;
    }
    // linear system solver Ax = b
    template <typename RhsXprType> constexpr auto solve(const RhsXprType& b) const {
        constexpr int ViewMode = XprType::ViewMode;
        if constexpr (ViewMode == Lower) { return fwd_sub_(b); }
        if constexpr (ViewMode == Upper) { return bwd_sub_(b); }
    }
    constexpr auto determinant() const { return derived().diagonal().prod(); }
    constexpr decltype(auto) rep() const { return derived().rep(); }
    constexpr decltype(auto) rep() { return derived().rep(); }
    constexpr auto data() const { return derived().data(); }
    constexpr auto data() { return derived().data(); }
   private:
    // forward substitution for lower-triangular matrix, vector rhs
    template <typename RhsXprType>
        requires(RhsXprType::Cols == 1)
    constexpr auto fwd_sub_(const MatrixExpr<RhsXprType>& b) const {
        constexpr int ViewMode = XprType::ViewMode;
        fdapde_static_assert(ViewMode == Lower, THIS_METHOD_IS_FOR_LOWER_TRIANGULAR_MATRICES_ONLY);
        const RhsXprType& b_ = b.derived();
        fdapde_assert(b_.rows() == derived().rows() && b_.cols() == 1);
        using Scalar = typename XprType::Scalar;
        constexpr int Rows = XprType::Rows;
        Vector<Scalar, Rows> x;
        if constexpr (Rows == Dynamic) { x.resize(b_.rows()); }
        if (b_.rows() != derived().rows() || b_.cols() != 1 || b_.rows() == 0) {
            fdapde_assert(b_.rows() == derived().rows() && b_.cols() == 1 && b_.rows() > 0);
            return x;
        }
        x[0] = b_[0] / derived()(0, 0);
        int rows = b_.rows();
        for (int i = 1; i < rows; ++i) {
            Scalar sum = 0;
            for (int j = 0; j < i; ++j) sum += derived()(i, j) * x[j];
            x[i] = (b_[i] - sum) / derived()(i, i);
        }
        return x;
    }
    // forward substitution for lower-triangular matrix, matrix rhs (cache-friendly approach)
    template <typename RhsXprType>
        requires(RhsXprType::Cols > 1 || RhsXprType::Cols == Dynamic)
    constexpr auto fwd_sub_(const MatrixExpr<RhsXprType>& B) const {
        constexpr int ViewMode = XprType::ViewMode;
        fdapde_static_assert(ViewMode == Lower, THIS_METHOD_IS_FOR_LOWER_TRIANGULAR_MATRICES_ONLY);
        const RhsXprType& B_ = B.derived();
        fdapde_assert(B_.rows() == derived().rows() && B_.cols() > 1);
        using Scalar = typename XprType::Scalar;
        constexpr int RhsRows = RhsXprType::Rows, RhsCols = RhsXprType::Cols;
        Matrix<Scalar, RhsRows, RhsCols> X;
        if constexpr (RhsRows == Dynamic || RhsCols == Dynamic) { X.resize(B_.rows(), B_.cols()); }
        if (B_.rows() != derived().rows() || B_.cols() <= 1 || B_.rows() == 0) {
            fdapde_assert(B_.rows() == derived().rows() && B_.cols() > 1 && B_.rows() > 0);
            return X;
        }
        const int rows = B_.rows();
        const int cols = B_.cols();
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) { X(i, j) = B_(i, j); }
            for (int k = 0; k < i; ++k) {
                for (int j = 0; j < cols; ++j) { X(i, j) -= derived()(i, k) * X(k, j); }
            }
            for (int j = 0; j < cols; ++j) { X(i, j) = X(i, j) / derived()(i, i); }
        }
        return X;
    }
    // backward substitution for upper-triangular matrix, vector rhs
    template <typename RhsXprType>
        requires(RhsXprType::Cols == 1)
    constexpr auto bwd_sub_(const MatrixExpr<RhsXprType>& b) const {
        constexpr int ViewMode = XprType::ViewMode;
        fdapde_static_assert(ViewMode == Upper, THIS_METHOD_IS_FOR_UPPER_TRIANGULAR_MATRICES_ONLY);
        const RhsXprType& b_ = b.derived();
        fdapde_assert(b_.rows() == derived().rows() && b_.cols() == 1);
        using Scalar = typename XprType::Scalar;
        constexpr int Rows = XprType::Rows;
        Vector<Scalar, Rows> x;
        if constexpr (Rows == Dynamic) { x.resize(b_.rows()); }
        if (b_.rows() != derived().rows() || b_.cols() != 1 || b_.rows() == 0) {
            fdapde_assert(b_.rows() == derived().rows() && b_.cols() == 1 && b_.rows() > 0);
            return x;
        }
        int rows = b_.rows();
        x[rows - 1] = b_[rows - 1] / derived()(rows - 1, rows - 1);
        for (int i = rows - 2; i >= 0; --i) {
            Scalar sum = 0;
            for (int j = i + 1; j < rows; ++j) sum += derived()(i, j) * x[j];
            x[i] = (b_[i] - sum) / derived()(i, i);
        }
        return x;
    }
    // backward substitution for upper-triangular matrix, matrix rhs (cache-friendly approach)
    template <typename RhsXprType>
        requires(RhsXprType::Cols > 1 || RhsXprType::Cols == Dynamic)
    constexpr auto bwd_sub_(const MatrixExpr<RhsXprType>& B) const {
        constexpr int ViewMode = XprType::ViewMode;
        fdapde_static_assert(ViewMode == Upper, THIS_METHOD_IS_FOR_UPPER_TRIANGULAR_MATRICES_ONLY);
        const RhsXprType& B_ = B.derived();
        fdapde_assert(B_.rows() == derived().rows() && B_.cols() >= 1);
        using Scalar = typename XprType::Scalar;
        constexpr int RhsRows = RhsXprType::Rows, RhsCols = RhsXprType::Cols;
        Matrix<Scalar, RhsRows, RhsCols> X;
        if constexpr (RhsRows == Dynamic || RhsCols == Dynamic) { X.resize(B_.rows(), B_.cols()); }
        if (B_.rows() != derived().rows() || B_.cols() <= 1 || B_.rows() == 0) {
            fdapde_assert(B_.rows() == derived().rows() && B_.cols() > 1 && B_.rows() > 0);
            return X;
        }
        const int rows = B_.rows();
        const int cols = B_.cols();
        for (int i = rows - 1; i >= 0; --i) {
            for (int j = 0; j < cols; ++j) { X(i, j) = B_(i, j); }
            for (int k = i + 1; k < rows; ++k) {
                for (int j = 0; j < cols; ++j) { X(i, j) -= derived()(i, k) * X(k, j); }
            }
            for (int j = 0; j < cols; ++j) { X(i, j) = X(i, j) / derived()(i, i); }
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
    using XprTypeNested = fdapde::internals::ref_select_t<XprType_>;
   public:
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    static constexpr int StorageOrder = XprType::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ViewMode = ViewMode_;
    static constexpr int ReadOnly = std::is_const_v<std::remove_reference_t<XprType_>> || XprType::ReadOnly;
    using assignment_executor = internals::triangular_assignment_executor;

    template <int ViewMode__, typename XprType__> struct triangular_proxy {
        using Scalar = typename XprType__::Scalar;

        constexpr triangular_proxy(XprType__& xpr, int i, int j) :
            xpr_(xpr), i_(i), j_(j), b_(ViewMode__ == Upper ? (i <= j) : (i >= j)) { }
        template <typename T>
            requires(std::is_convertible_v<T, Scalar> && !std::is_const_v<XprType__>)
        constexpr triangular_proxy& operator=(T value) {
            if (b_) xpr_(i_, j_) = value;   // do nothing otherwise
            return *this;
        }
        constexpr operator Scalar() const { return b_ ? xpr_(i_, j_) : Scalar(0); }
       private:
        XprType__& xpr_;
        int i_, j_;
        bool b_;
    };
    using reference = triangular_proxy<ViewMode, XprType>;
    using const_reference = triangular_proxy<ViewMode, const XprType>;

    // constructor
    constexpr Triangular(const Triangular& other) : xpr_(other.xpr_) { }
    template <typename XprType__>
        requires(
          !std::same_as<std::remove_cvref_t<XprType__>, Triangular> &&
          std::is_constructible_v<XprTypeNested, XprType__>)
    constexpr explicit Triangular(XprType__&& xpr) : xpr_(std::forward<XprType__>(xpr)) {
        fdapde_static_assert(
          Rows == Dynamic || Cols == Dynamic || Rows == Cols, THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
        if constexpr (Rows == Dynamic || Cols == Dynamic) { fdapde_assert(xpr.rows() == xpr.cols()); }
    }
    // access
    constexpr const_reference operator()(int i, int j) const {
        fdapde_assert(i >= 0 && i < xpr_.rows() && j >= 0 && j < xpr_.cols());
        return const_reference(xpr_, i, j);
    }
    constexpr reference operator()(int i, int j) requires(ReadOnly == 0) {
        fdapde_assert(i >= 0 && i < xpr_.rows() && j >= 0 && j < xpr_.cols());
        return reference(xpr_, i, j);
    }
    // observers
    constexpr int rows() const { return xpr_.rows(); }
    constexpr int cols() const { return xpr_.cols(); }
    constexpr const XprTypeNested& rep() const { return xpr_; }
    constexpr XprTypeNested& rep() requires(ReadOnly == 0) { return xpr_; }
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
    using Base::operator=;

    template <int ViewMode__, typename Scalar__>
        requires(
          std::is_same_v<std::remove_cv_t<Scalar>, std::remove_cv_t<Scalar__>> &&
          (ViewMode__ == Lower || ViewMode__ == Upper))
    struct triangular_proxy {
        using Scalar = Scalar__;

        constexpr triangular_proxy(Scalar__* data, int i, int j, int size) :
            data_(data),
            index_(triangular_access_(i, j, size)),
            b_(ViewMode__ == Upper ? (i <= j) : (i >= j)) { }
        template <typename T>
            requires(std::is_convertible_v<T, Scalar__> && !std::is_const_v<Scalar__>)
        constexpr triangular_proxy& operator=(T value) {
            if (b_) data_[index_] = value;   // do nothing otherwise
            return *this;
        }
        constexpr operator Scalar() const { return b_ ? data_[index_] : Scalar(0); }
       private:
        // internally, acess is always performed in RowMajor-format
        constexpr int triangular_access_(int i, int j, int size) const {
            return ViewMode == Upper ? i * (2 * size - i + 1) / 2 + (j - i) : i * (i + 1) / 2 + j;
        }
        Scalar* data_;
        int index_;
        bool b_;
    };
    using reference = triangular_proxy<ViewMode, Scalar>;
    using const_reference = triangular_proxy<ViewMode, const Scalar>;

    constexpr TriangularMatrixBase() : rows_(default_shape_()), cols_(default_shape_()) { }
    constexpr TriangularMatrixBase(int rows, int cols) : rows_(rows), cols_(cols) {
        if (rows < 0 || cols < 0 || rows != cols || (Rows != Dynamic && rows != Rows) ||
            (Cols != Dynamic && cols != Cols)) {
            fdapde_assert(
              rows >= 0 && cols >= 0 && rows == cols && (Rows == Dynamic || rows == Rows) &&
              (Cols == Dynamic || cols == Cols));
            rows_ = default_shape_();
            cols_ = default_shape_();
        }
    }
    // access
    constexpr const_reference operator()(int i, int j) const {
        fdapde_assert(i >= 0 && i < rows_ && j >= 0 && j < cols_);
        return const_reference(derived().data(), i, j, rows_);
    }
    constexpr reference operator()(int i, int j) requires(ReadOnly == 0) {
        fdapde_assert(i >= 0 && i < rows_ && j >= 0 && j < cols_);
        return reference(derived().data(), i, j, rows_);
    }
    // observers
    constexpr int rows() const { return rows_; }
    constexpr int cols() const { return cols_; }
   protected:
    static constexpr int default_shape_() {
        if constexpr (Rows != Dynamic) return Rows;
        if constexpr (Cols != Dynamic) return Cols;
        return 0;
    }
    int rows_, cols_;
};

// triangular matrix subalgebra of the associative algebra of square matrices
template <typename LhsXprType, typename RhsXprType>
    requires(LhsXprType::ViewMode == RhsXprType::ViewMode)   // if summing different ViewMode, exit from triangular TS
constexpr auto operator+(const TriangularMatrixExpr<LhsXprType>& lhs, const TriangularMatrixExpr<RhsXprType>& rhs) {
    return internals::triangular_cast<LhsXprType::ViewMode>(
      MatrixBinOp<LhsXprType, RhsXprType, std::plus<>>(lhs.derived(), rhs.derived(), std::plus<>()));
}
template <typename LhsXprType, typename RhsXprType>
    requires(LhsXprType::ViewMode == RhsXprType::ViewMode)   // if summing different ViewMode, exit from triangular TS
constexpr auto operator-(const TriangularMatrixExpr<LhsXprType>& lhs, const TriangularMatrixExpr<RhsXprType>& rhs) {
    return internals::triangular_cast<LhsXprType::ViewMode>(
      MatrixBinOp<LhsXprType, RhsXprType, std::minus<>>(lhs.derived(), rhs.derived(), std::minus<>()));
}
template <typename XprType, typename CoeffType>
    requires(std::is_arithmetic_v<CoeffType>)
constexpr auto operator*(const TriangularMatrixExpr<XprType>& lhs, CoeffType rhs) {
    return internals::triangular_cast<XprType::ViewMode>(lhs.rep() * rhs);
}
template <typename XprType, typename CoeffType>
    requires(std::is_arithmetic_v<CoeffType>)
constexpr auto operator*(CoeffType lhs, const TriangularMatrixExpr<XprType>& rhs) {
    return internals::triangular_cast<XprType::ViewMode>(lhs * rhs.rep());
}
template <typename XprType, typename CoeffType>
    requires(std::is_arithmetic_v<CoeffType>)
constexpr auto operator/(const TriangularMatrixExpr<XprType>& lhs, CoeffType rhs) {
    return internals::triangular_cast<XprType::ViewMode>(lhs.rep() / rhs);
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
        constexpr int is_lower = ViewMode == Lower;
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
            constexpr int is_lower = LhsViewMode == Lower;
            Scalar prod = 0;
            const int h = is_lower ? (i + 1) : (j + 1);
            if ((is_lower && (i < j)) || (!is_lower && i > j)) { return Scalar(0); }
            for (int k = 0; k < h; ++k) prod += lhs(i, k) * rhs(k, j);
            return prod;
        }
    }
};

}   // namespace internals

// diagonal scaling preserves triangular structure
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator*(const DiagonalMatrixExpr<LhsXprType>& lhs, const TriangularMatrixExpr<RhsXprType>& rhs) {
    return MatrixMultiplicationOp<
      LhsXprType, RhsXprType, internals::diagonal_matrix_product_executor<LhsMode>> {lhs.derived(), rhs.derived()}
      .template triangular_block<RhsXprType::ViewMode>();
}
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator*(const TriangularMatrixExpr<LhsXprType>& lhs, const DiagonalMatrixExpr<RhsXprType>& rhs) {
    return MatrixMultiplicationOp<
      LhsXprType, RhsXprType, internals::diagonal_matrix_product_executor<RhsMode>> {lhs.derived(), rhs.derived()}
      .template triangular_block<LhsXprType::ViewMode>();
}

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
    if constexpr (LhsXprType::ViewMode == RhsXprType::ViewMode) {
        return MatrixMultiplicationOp<LhsXprType, RhsXprType, internals::triangular_triangular_product_executor> {
          lhs.derived(), rhs.derived()}
          .template triangular_block<LhsXprType::ViewMode>();
    } else {
        return MatrixMultiplicationOp<LhsXprType, RhsXprType, internals::generic_matrix_product_executor> {
          lhs.derived(), rhs.derived()};
    }
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
    fdapde_static_assert(StorageOrder_ == RowMajor, PACKED_COL_MAJOR_STRUCTURED_STORAGE_IS_NOT_SUPPORTED);
   private:
    using Base = TriangularMatrixBase<
      Scalar_, Rows_, Cols_, ViewMode_, StorageOrder_,
      TriangularMatrix<Scalar_, Rows_, Cols_, ViewMode_, StorageOrder_>>;
    static constexpr int StaticSize = Rows_ != Dynamic ? Rows_ : Cols_;
    static constexpr int StorageSize = StaticSize == Dynamic ? Dynamic : (StaticSize * (StaticSize + 1) / 2);
    // a vector with StorageOrder-induced access pattern
    using StorageType = Vector<Scalar_, StorageSize>;
   public:
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int NestAsRef = 1;
    static constexpr int StorageOrder = StorageOrder_;
    using assignment_executor = internals::triangular_assignment_executor;

    constexpr TriangularMatrix() : Base() { }
    // copy semantic
    constexpr TriangularMatrix(const TriangularMatrix& rhs) : Base(rhs.rows(), rhs.cols()) { clone_(rhs); }
    constexpr TriangularMatrix& operator=(const TriangularMatrix& rhs) & {
        clone_(rhs);
        return *this;
    }
    constexpr void operator=(const TriangularMatrix&) && = delete;
    using Base::operator=;
    constexpr explicit TriangularMatrix(int rows, int cols) :
        Base(rows, cols), data_() {
        fdapde_static_assert(Rows == Dynamic || Cols == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_MATRICES_ONLY);
        if (this->rows_ != rows || this->cols_ != cols) return;
        resize(rows, cols);
    }
    template <typename RhsXprType_>
    constexpr TriangularMatrix(const MatrixExpr<RhsXprType_>& rhs) : Base(rhs.rows(), rhs.cols()) {
        if (this->rows_ != rhs.rows() || this->cols_ != rhs.cols()) return;
        clone_(rhs.derived());
    }
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr explicit TriangularMatrix(const std::vector<Scalar__>& data) :
        Base(internals::compute_triangular_shape(data.size()), internals::compute_triangular_shape(data.size())),
        data_() {
        if (internals::compute_triangular_shape(data.size()) < 0 ||
            !std::cmp_equal(data.size(), this->rows_ * (this->rows_ + 1) / 2)) {
            fdapde_assert(false);
            return;
        }
        if constexpr (Rows == Dynamic || Cols == Dynamic) { data_.resize(data.size()); }
        if (!std::cmp_equal(data_.size(), data.size())) {
            fdapde_assert(std::cmp_equal(data_.size() FDAPDE_COMMA data.size()));
            return;
        }
        assignment_executor::run(
          *this, VectorView<const Scalar__, Dynamic>(data.data(), data.size()), [](auto& l, const auto& r) { l = r; });
    }
    template <typename Scalar__, std::size_t Size>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr explicit TriangularMatrix(const Scalar__ (&data)[Size]) :
        Base(internals::compute_triangular_shape(Size), internals::compute_triangular_shape(Size)), data_() {
        fdapde_static_assert(
          Rows_ != Dynamic && Cols_ != Dynamic && StorageSize == Size, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        assignment_executor::run(*this, VectorView<const Scalar__, Size>(data), [](auto& l, const auto& r) { l = r; });
    }
    // modifiers
    void resize(int rows, int cols) {
        fdapde_static_assert(Rows == Dynamic || Cols == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_MATRICES_ONLY);
        const bool valid = rows >= 0 && cols >= 0 && rows == cols && (Rows == Dynamic || rows == Rows) &&
                           (Cols == Dynamic || cols == Cols);
        if (!valid) {
            fdapde_assert(valid);
            return;
        }
        const int storage_size = rows * (cols + 1) / 2;
        if (std::cmp_equal(this->rows_, rows) && std::cmp_equal(this->cols_, cols) &&
            std::cmp_equal(data_.size(), storage_size))
            return;
        // update and reallocate memory
        this->rows_ = rows;
        this->cols_ = cols;
        if constexpr (StorageSize == Dynamic) { data_.resize(storage_size); }
        return;
    }
    // internal vector representation
    constexpr const StorageType& rep() const { return data_; }
    constexpr StorageType& rep() { return data_; }
    // data pointers
    constexpr const Scalar_* data() const { return data_.data(); }
    constexpr Scalar_* data() { return data_.data(); }
   private:
    template <typename RhsXprType> constexpr void clone_(const RhsXprType& rhs) {
        if constexpr (Rows == Dynamic || Cols == Dynamic) { resize(rhs.rows(), rhs.cols()); }
        assignment_executor::run(*this, rhs, [](auto& l, const auto& r) { l = r; });
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
    fdapde_static_assert(StorageOrder_ == RowMajor, PACKED_COL_MAJOR_STRUCTURED_STORAGE_IS_NOT_SUPPORTED);
    using Base = TriangularMatrixBase<
      Scalar_, Rows_, Cols_, ViewMode_, StorageOrder_,
      TriangularMatrixView<Scalar_, Rows_, Cols_, ViewMode_, StorageOrder_>>;
    using StorageType = std::add_pointer_t<Scalar_>;
   public:
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = std::is_const_v<Scalar_>;
    static constexpr int StaticSize = Rows_ != Dynamic ? Rows_ : Cols_;
    static constexpr int StorageSize = StaticSize == Dynamic ? Dynamic : StaticSize * (StaticSize + 1) / 2;
    using assignment_executor = internals::triangular_assignment_executor;

    // constructors
    constexpr TriangularMatrixView(const TriangularMatrixView&) = default;
    constexpr TriangularMatrixView() requires(Rows_ == Dynamic && Cols_ == Dynamic) : Base(), data_(nullptr) { }
    constexpr TriangularMatrixView() requires(Rows_ != Dynamic || Cols_ != Dynamic) = delete;
    template <typename Scalar__>
        requires(std::is_convertible_v<Scalar__*, Scalar_*>)
    constexpr explicit TriangularMatrixView(Scalar__* data) : Base(), data_(data) {
        fdapde_static_assert(Rows != Dynamic && Cols != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        if (data == nullptr) fdapde_assert(data != nullptr);
    }
    template <typename Scalar__>
        requires(std::is_convertible_v<Scalar__*, Scalar_*>)
    constexpr TriangularMatrixView(Scalar__* data, int rows, int cols) : Base(rows, cols), data_(data) {
        if (rows < 0 || cols < 0 || (rows > 0 && data == nullptr) || this->rows_ != rows || this->cols_ != cols) {
            fdapde_assert(rows >= 0 && cols >= 0 && (rows == 0 || data != nullptr));
            this->rows_ = 0;
            this->cols_ = 0;
            data_ = nullptr;
        }
    }
    using Base::operator=;
    constexpr TriangularMatrixView& operator=(const TriangularMatrixView& other) & {
        static_cast<MatrixExpr<TriangularMatrixView>&>(*this).template operator=<TriangularMatrixView>(other);
        return *this;
    }
    constexpr TriangularMatrixView operator=(const TriangularMatrixView& other) && {
        static_cast<MatrixExpr<TriangularMatrixView>&>(*this).template operator=<TriangularMatrixView>(other);
        return *this;
    }
    // internal vector representation
    constexpr auto rep() const {
        using View = VectorView<const std::remove_const_t<Scalar_>, StorageSize>;
        if constexpr (StorageSize == Dynamic) {
            if (this->rows_ == 0) return View();
            return View(data_, this->rows_ * (this->rows_ + 1) / 2);
        } else {
            return View(data_);
        }
    }
    constexpr auto rep() requires(!std::is_const_v<Scalar_>) {
        using View = VectorView<Scalar_, StorageSize>;
        if constexpr (StorageSize == Dynamic) {
            if (this->rows_ == 0) return View();
            return View(data_, this->rows_ * (this->rows_ + 1) / 2);
        } else {
            return View(data_);
        }
    }
    // data pointers
    constexpr const std::remove_const_t<Scalar_>* data() const { return data_; }
    constexpr StorageType data() { return data_; }
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
    using Type = std::remove_cvref_t<XprType>;
    static constexpr bool value = std::is_base_of_v<TriangularMatrixExpr<Type>, Type>;
};
template <typename XprType> static constexpr bool is_triangular_matrix_v = is_triangular_matrix<XprType>::value;

}   // namespace fdapde::linalg

#endif   // __FDAPDE_LINALG_TRIANGULAR_H__
