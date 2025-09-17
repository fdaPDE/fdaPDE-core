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

#ifndef __FDAPDE_TRIANGULAR_MATRIX_H__
#define __FDAPDE_TRIANGULAR_MATRIX_H__

#include "../header_check.h"

namespace fdapde {

// triangular matrix type system
template <typename Scalar_, int Rows_, int ViewMode_> struct TriangularMatrix;

namespace internals {

// class wrapping a linear vector to the expression of a triangular matrix. internal usage only
template <
  int Rows_, int Cols_, int ViewMode_, typename TriangularXprType,
  int Size_ = (Rows_ == Dynamic || Cols_ == Dynamic) ? Dynamic : int((fdapde::sqrt(double(1 + 8 * Rows_)) - 1) / 2)>
struct triangular_wrapper :
    TriangularMatrixBase<
      typename TriangularXprType::Scalar, Size_, ViewMode_,
      triangular_wrapper<Rows_, Cols_, ViewMode_, TriangularXprType, Size_>> {
    using Base = TriangularMatrixBase<
      typename TriangularXprType::Scalar, Size_, ViewMode_,
      triangular_wrapper<Rows_, Cols_, ViewMode_, TriangularXprType, Size_>>;
    using TriangularXprTypeNested = internals::ref_select_t<const TriangularXprType>;

    template <typename XprType>
        requires(std::is_constructible_v<TriangularXprTypeNested, XprType>)
    constexpr triangular_wrapper(XprType&& xpr) :
        Base((fdapde::sqrt(static_cast<double>(1 + 8 * xpr.rows())) - 1) / 2), xpr_(std::forward<XprType>(xpr)) { }
    const TriangularXprTypeNested& data() const { return xpr_; }
   private:
    TriangularXprTypeNested xpr_;
};

// helper cast function
template <int ViewMode, typename XprType> auto triangular_cast(XprType&& xpr) {
    using XprTypeClean = std::decay_t<XprType>;
    static constexpr int Rows = XprTypeClean::Rows;
    static constexpr int Cols = XprTypeClean::Cols;
    fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
    return triangular_wrapper<Rows == 1 ? Cols : Rows, Cols == 1 ? Rows : Cols, ViewMode, XprType>(xpr);
}

}   // namespace internals
  
template <int Rows_, int Cols_, int ViewMode_, typename TriangularXprType>
struct TriangularMatrixExpr : public MatrixExpr<Rows_, Cols_, TriangularXprType> {
    using Base = MatrixExpr<Rows_, Cols_, TriangularXprType>;
    using Base::derived;
    static constexpr int Rows = TriangularXprType::Rows;
    static constexpr int Cols = TriangularXprType::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int ViewMode = ViewMode_;
    static constexpr int ReadOnly = TriangularXprType::ReadOnly || (ViewMode == UnitLower || ViewMode == UnitUpper);
    struct assignment_executor {
        template <typename SrcXprType> static constexpr void run(TriangularXprType& dst, const SrcXprType& src) {
            fdapde_static_assert(TriangularXprType::ReadOnly == 0, ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION);
            fdapde_static_assert(
              ViewMode == Upper || ViewMode == Lower, TRIANGULAR_BLOCK_ASSIGNMENT_REQUIRES_EITHER_UPPER_OR_LOWER_VIEW);
            int row = 0, col = 0;
            for (int i = 0, n = dst.rows(); i < n; ++i) {
                for (int j = 0; j < i; ++j) {
                    row = ViewMode == Lower ? i : j;
                    col = ViewMode == Lower ? j : i;
                    dst(row, col) = src(row, col);
                }
            }
            // assign diagonal
            for (int i = 0, n = dst.rows(); i < n; ++i) { dst(i, i) = src(i, i); }
        }
    };
    // inherit assignment from base
    using Base::operator=;
    // copy assignment
    constexpr TriangularXprType& operator=(const TriangularXprType& other) {
        fdapde_static_assert(ReadOnly == 0, ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION);
	assignment_executor::run(derived(), other);
        return derived();
    }
    constexpr auto inverse() const {
        using Scalar = typename TriangularXprType::Scalar;
        TriangularMatrix<Scalar, Rows, ViewMode> inverse_;
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
        fdapde_static_assert(
          ViewMode == Lower || ViewMode == Upper, THIS_METHOD_IS_FOR_LOWER_OR_UPPER_TRIANGULAR_MATRICES_ONLY);
        if constexpr (ViewMode == Lower || ViewMode == UnitLower) return forward_sub(b);
        if constexpr (ViewMode == Upper || ViewMode == UnitLower) return backward_sub(b);
    }
    constexpr double determinant() const { return derived().diagonal().prod(); }

    constexpr const auto& data() const { return derived().data(); }
    constexpr auto& data() { return derived().data(); }
   private:
    // forward substitution for lower-triangular matrix
    template <typename RhsXprType> constexpr auto forward_sub(const RhsXprType& b) const {
        fdapde_static_assert(ViewMode == Lower, THIS_METHOD_IS_FOR_LOWER_TRIANGULAR_MATRICES_ONLY);
        using Scalar = typename TriangularXprType::Scalar;
        Vector<Scalar, Rows> x;
        if constexpr (Rows == Dynamic) { x.resize(derived().rows()); }
        x[0] = b[0] / derived()(0, 0);
	int rows = derived().rows();
        for (int i = 1; i < rows; ++i) {
            Scalar sum = 0;
            for (int j = 0; j < i; ++j) sum += derived()(i, j) * x[j];
            x[i] = (b[i] - sum) / derived()(i, i);
        }
        return x;
    }
    // backward substitution for upper-triangular matrix
    template <typename RhsXprType> constexpr auto backward_sub(const RhsXprType& b) const {
        fdapde_static_assert(ViewMode == Upper, THIS_METHOD_IS_FOR_UPPER_TRIANGULAR_MATRICES_ONLY);
        using Scalar = typename TriangularXprType::Scalar;
        Vector<Scalar, Rows> x;
        if constexpr (Rows == Dynamic) { x.resize(derived().rows()); }
        int rows = derived().rows();
        x[rows - 1] = b[rows - 1] / derived()(rows - 1, rows - 1);
        for (int i = rows - 2; i >= 0; --i) {
            Scalar sum = 0;
            for (int j = i + 1; j < rows; ++j) sum += derived()(i, j) * x[j];
            x[i] = (b[i] - sum) / derived()(i, i);
        }
        return x;
    }
};

// expression of the triangular part of a matrix
template <typename XprType, int ViewMode_>
struct TriangularBlock :
    public TriangularMatrixExpr<XprType::Rows, XprType::Cols, ViewMode_, TriangularBlock<XprType, ViewMode_>> {
    using Base = TriangularMatrixExpr<XprType::Rows, XprType::Cols, ViewMode_, TriangularBlock<XprType, ViewMode_>>;
    using XprTypeNested = internals::ref_select_t<XprType>;
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int ViewMode = ViewMode_;
    static constexpr int ReadOnly = XprType::ReadOnly || (ViewMode == UnitLower || ViewMode == UnitUpper);

    // constructor
    template <typename XprType_>
        requires(std::is_constructible_v<XprTypeNested, XprType_>)
    constexpr explicit TriangularBlock(XprType_&& xpr) : Base(), xpr_(std::forward<XprType_>(xpr)) {
        fdapde_static_assert(
          XprType::Rows == Dynamic || XprType::Cols == Dynamic || XprType::Rows == XprType::Cols,
          THIS_EXPRESSION_IS_FOR_SQUARE_MATRICES_ONLY);
        if constexpr (XprType::Rows == Dynamic || XprType::Cols == Dynamic) {
            fdapde_constexpr_assert(xpr.rows() == xpr.cols());
        }
    }
    // inherit assignment from base
    using Base::operator=;
    // access
    constexpr Scalar operator()(int i, int j) const {
        fdapde_constexpr_assert(i >= 0 && i < xpr_.rows() && j >= 0 && j < xpr_.cols());
        if constexpr (ViewMode == Upper) return i > j ? 0 : xpr_(i, j);
        if constexpr (ViewMode == Lower) return i < j ? 0 : xpr_(i, j);
        if constexpr (ViewMode == UnitUpper) return i > j ? 0 : (i == j ? Scalar(1) : xpr_(i, j));
        if constexpr (ViewMode == UnitLower) return i < j ? 0 : (i == j ? Scalar(1) : xpr_(i, j));      
    }
    constexpr Scalar& operator()(int i, int j) {
        fdapde_static_assert(ViewMode == Upper || ViewMode == Lower, WRITE_ACCESS_TO_READ_ONLY_EXPRESSION);
        fdapde_constexpr_assert(i >= 0 && i < xpr_.rows() && j >= 0 && j < xpr_.cols());
        if constexpr (ViewMode == Upper) return i > j ? 0 : xpr_(i, j);
        if constexpr (ViewMode == Lower) return i < j ? 0 : xpr_(i, j);
    }
    // observers
    constexpr int rows() const { return xpr_.rows(); }
    constexpr int cols() const { return xpr_.cols(); }
    constexpr int size() const { return xpr_.size(); }
   private:
    XprTypeNested xpr_;
};

// base class for triangular matrices
template <typename Scalar_, int Rows_, int ViewMode_, typename TriangularMatrixType>
class TriangularMatrixBase : public TriangularMatrixExpr<Rows_, Rows_, ViewMode_, TriangularMatrixType> {
   public:
    using Base = TriangularMatrixExpr<Rows_, Rows_, ViewMode_, TriangularMatrixType>;
    using Base::derived;
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Rows_;
    static constexpr int ViewMode = ViewMode_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = std::is_const_v<Scalar_> ? 1 : 0;

    constexpr TriangularMatrixBase() : Base(), size_(Rows == Dynamic || Cols == Dynamic ? 0 : Rows) { }
    constexpr TriangularMatrixBase(double size) : Base(), size_(size) {
        // size can be a floating point value as a result of calling triangular_cast() on a vector which cannot map to a
        // triangular matrix. This checks guarantees that "there are enought values" to make a triangular matrix
        fdapde_constexpr_assert(size == fdapde::floor(size));
    }
    // inherit assignment from base
    using Base::operator=;
    // access
    constexpr Scalar operator()(int i, int j) const {
        fdapde_constexpr_assert(i >= 0 && i < size_ && j >= 0 && j < size_);
        if constexpr (ViewMode == Upper) return i > j ? 0 : derived().data()[index_(i, j)];
        if constexpr (ViewMode == Lower) return i < j ? 0 : derived().data()[index_(i, j)];
    }
    constexpr Scalar& operator()(int i, int j) {
        fdapde_static_assert(ReadOnly == 0, WRITE_ACCESS_TO_READ_ONLY_LOCATION);
        fdapde_constexpr_assert(
          i >= 0 && i < size_ && j >= 0 && j < size_ &&
          ((ViewMode == Upper && i <= j) || (ViewMode == Lower && i >= j)));
        if constexpr (ViewMode == Upper) return derived().data()[index_(i, j)];
        if constexpr (ViewMode == Lower) return derived().data()[index_(i, j)];
    }
    // observers
    constexpr int rows() const { return size_; }
    constexpr int cols() const { return size_; }
   protected:
    constexpr int index_(int i, int j) const {
        return ViewMode == Upper ? i * (2 * size_ - i + 1) / 2 + (j - i) : i * (i + 1) / 2 + j;
    }
    int size_;
};

// triangular matrix subalgebra of the associative algebra of square matrices
template <typename LhsXprType, typename RhsXprType, int ViewMode>
constexpr auto operator+(
  const TriangularMatrixExpr<LhsXprType::Rows, LhsXprType::Cols, ViewMode, LhsXprType>& lhs,
  const TriangularMatrixExpr<RhsXprType::Rows, RhsXprType::Cols, ViewMode, RhsXprType>& rhs) {
    return internals::triangular_cast<ViewMode>(lhs.data() + rhs.data());
}
template <typename LhsXprType, typename RhsXprType, int ViewMode>
constexpr auto operator-(
  const TriangularMatrixExpr<LhsXprType::Rows, LhsXprType::Cols, ViewMode, LhsXprType>& lhs,
  const TriangularMatrixExpr<RhsXprType::Rows, RhsXprType::Cols, ViewMode, RhsXprType>& rhs) {
    return internals::triangular_cast<ViewMode>(lhs.data() - rhs.data());
}
template <typename XprType, typename CoeffType, int ViewMode>
    requires(std::is_arithmetic_v<CoeffType>)
constexpr auto
operator*(const TriangularMatrixExpr<XprType::Rows, XprType::Cols, ViewMode, XprType>& lhs, CoeffType rhs) {
    return internals::triangular_cast<ViewMode>(rhs.data() * lhs);
}
template <typename XprType, typename CoeffType, int ViewMode>
    requires(std::is_arithmetic_v<CoeffType>)
constexpr auto
operator*(CoeffType lhs, const TriangularMatrixExpr<XprType::Rows, XprType::Cols, ViewMode, XprType>& rhs) {
    return rhs * lhs;
}
template <typename XprType, typename CoeffType, int ViewMode>
    requires(std::is_arithmetic_v<CoeffType>)
constexpr auto
operator/(const TriangularMatrixExpr<XprType::Rows, XprType::Cols, ViewMode, XprType>& lhs, CoeffType rhs) {
    return internals::triangular_cast<ViewMode>(lhs.data() / rhs);
}

// specialized products
namespace internals {

// expressions of the (i,j)-th entry of the product between a triangular and a dense matrix expression
template <typename LhsXprType, typename RhsXprType, int ProductMode> struct triangular_matrix_product_executor {
    using TriXprTypeClean = std::decay_t<std::conditional_t<ProductMode == LhsMode, LhsXprType, RhsXprType>>;
    using MtxXprTypeClean = std::decay_t<std::conditional_t<ProductMode == LhsMode, RhsXprType, LhsXprType>>;
    using Scalar =
      decltype(std::declval<typename TriXprTypeClean::Scalar>() * std::declval<typename MtxXprTypeClean::Scalar>());
    static constexpr int ViewMode = TriXprTypeClean::ViewMode;
    static constexpr int is_lower = ViewMode == Lower || ViewMode == UnitLower;

    static constexpr auto run(int i, int j, const LhsXprType& lhs, const RhsXprType& rhs) {
        Scalar prod = 0;
        const int h = (ProductMode == LhsMode ? (is_lower ? 0 : i) : (is_lower ? j : 0));
        const int n =
          (ProductMode == LhsMode ? (is_lower ? (i + 1) : (lhs.rows() - i)) : (is_lower ? (lhs.cols() - j) : (j + 1)));
        for (int k = 0; k < n; ++k) { prod += lhs(i, h + k) * rhs(h + k, j); }
        return prod;
    }
};

// expressions of the (i,j)-th entry of the product between triangular expressions
template <typename LhsXprType, typename RhsXprType> struct triangular_triangular_product_executor {
    using LhsXprTypeClean = std::decay_t<LhsXprType>;
    using RhsXprTypeClean = std::decay_t<RhsXprType>;
    using Scalar =
      decltype(std::declval<typename LhsXprTypeClean::Scalar>() * std::declval<typename RhsXprTypeClean::Scalar>());
    static constexpr int LhsViewMode = LhsXprTypeClean::ViewMode;
    static constexpr int RhsViewMode = RhsXprTypeClean::ViewMode;

    static constexpr auto run(int i, int j, const LhsXprType& lhs, const RhsXprType& rhs) {
        if constexpr (LhsViewMode != RhsViewMode) {   // just multiply the diagonals
            return i == j ? lhs(i, i) * rhs(i, i) : Scalar(0);
        } else {
            // operands have same ViewMode
            constexpr int is_lower = LhsViewMode == Lower || LhsViewMode == UnitLower;
            Scalar prod = 0;
            const int h = is_lower ? (i + 1) : (j + 1);
            if ((is_lower && (i > j)) || (!is_lower && j < i)) { return Scalar(0); }
            for (int k = 0; k < h; ++k) prod += lhs(i, k) * rhs(k, j);
            return prod;
        }
    }
};

}   // namespace internals

template <typename LhsXprType, typename RhsXprType, typename Executor> struct MatrixProductOp;
// Triangular * M
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator*(
  const TriangularMatrixExpr<LhsXprType::Rows, LhsXprType::Cols, LhsXprType::ViewMode, LhsXprType>& lhs,
  const MatrixExpr<RhsXprType::Rows, RhsXprType::Cols, RhsXprType>& rhs) {
    return MatrixProductOp<
      LhsXprType, RhsXprType, internals::triangular_matrix_product_executor<LhsXprType, RhsXprType, LhsMode>> {
      lhs.derived(), rhs.derived()};
}
// M * Triangular
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator*(
  const MatrixExpr<LhsXprType::Rows, LhsXprType::Cols, LhsXprType>& lhs,
  const TriangularMatrixExpr<RhsXprType::Rows, RhsXprType::Cols, RhsXprType::ViewMode, RhsXprType>& rhs) {
    return MatrixProductOp<
      LhsXprType, RhsXprType, internals::triangular_matrix_product_executor<LhsXprType, RhsXprType, RhsMode>> {
      lhs.derived(), rhs.derived()};
}
// Triangular * Triangular
template <typename LhsXprType, typename RhsXprType, int ViewMode>
constexpr auto operator*(
  const TriangularMatrixExpr<LhsXprType::Rows, LhsXprType::Cols, ViewMode, LhsXprType>& lhs,
  const TriangularMatrixExpr<RhsXprType::Rows, RhsXprType::Cols, ViewMode, RhsXprType>& rhs) {
    return MatrixProductOp<
             LhsXprType, RhsXprType, internals::triangular_triangular_product_executor<LhsXprType, RhsXprType>> {
      lhs.derived(), rhs.derived()}
      .template triangular_block<ViewMode>();   // close wrt triangular subalgebra
}

// owning storage diagonal matrix
template <typename Scalar_, int Rows_, int ViewMode_>
struct TriangularMatrix :
    public TriangularMatrixBase<Scalar_, Rows_, ViewMode_, TriangularMatrix<Scalar_, Rows_, ViewMode_>> {
    fdapde_static_assert(
      ViewMode_ == Lower || ViewMode_ == Upper, TRIANGULAR_MATRICES_CAN_BE_IN_LOWER_OR_UPPER_MODE_ONLY);
    using Base = TriangularMatrixBase<Scalar_, Rows_, ViewMode_, TriangularMatrix<Scalar_, Rows_, ViewMode_>>;
    using Scalar = Scalar_;
    static constexpr int StorageSize = Rows_ == Dynamic ? Dynamic : (Rows_ * (Rows_ + 1) / 2);
    using StorageType = Vector<Scalar, StorageSize>;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Rows_;
    static constexpr int ViewMode = ViewMode_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = std::is_const_v<Scalar_> ? 1 : 0;

    constexpr TriangularMatrix() : Base() { }
    constexpr explicit TriangularMatrix(int size) : Base(size), data_() {
        fdapde_static_assert(
          Rows == Dynamic || Cols == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_TRIANGULAR_MATRICES_ONLY);
        data_.resize(size);
    }
    template <int RhsRows_, int RhsCols_, typename RhsXprType_>
    constexpr TriangularMatrix(const MatrixExpr<RhsRows_, RhsCols_, RhsXprType_>& rhs) : Base(rhs.rows()) {
        fdapde_constexpr_assert(StorageSize == Dynamic || rhs.rows() == rhs.cols());
        if constexpr (Rows == Dynamic || Cols == Dynamic) { resize(rhs.rows()); }
        using assignment = typename Base::assignment_executor;
        assignment::run(*this, rhs.derived());
    }
    template <typename DataT>
        requires(internals::is_vector_like_v<DataT> && !internals::is_matrix_like_v<DataT>)
    constexpr explicit TriangularMatrix(DataT&& data) : Base(size_(data.size())), data_() {
        if constexpr (Rows == Dynamic || Cols == Dynamic) { data_.resize(data.size()); }
        fdapde_constexpr_assert(data_.size() == data.size());
        for (int i = 0, n = data_.size(); i < n; ++i) { data_[i] = data[i]; }
    }
    template <std::size_t RhsSize>
    constexpr explicit TriangularMatrix(const Scalar (&data)[RhsSize]) : Base(size_(RhsSize)), data_() {
        fdapde_static_assert(Rows != Dynamic && Cols != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        fdapde_static_assert(StorageSize == RhsSize, INVALID_DATA_SIZE);
        for (int i = 0, n = data_.size(); i < n; ++i) { data_[i] = data[i]; }
    }
    // static named constructors
    static constexpr TriangularMatrix Ones() { return OnesMatrix<Rows, Cols>(); }
    static constexpr TriangularMatrix Ones(int size) { return OnesMatrix<Rows, Cols>(size, size); }
    static constexpr TriangularMatrix Zero() { return ZeroMatrix<Rows, Cols>(); }
    static constexpr TriangularMatrix Zero(int size) { return ZeroMatrix<Rows, Cols>(size, size); }
    static constexpr TriangularMatrix Identity() { return IdentityMatrix<Rows, Cols>(); }
    static constexpr TriangularMatrix Identity(int size) { return IdentityMatrix<Rows, Cols>(size, size); }
    // inherit assignment from base
    using Base::operator=;
    // modifiers
    void resize(int size) {
        fdapde_static_assert(Rows == Dynamic || Cols == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_MATRICES_ONLY);
        Base::size_ = StorageSize == Dynamic ? size : StorageSize;
        if (std::cmp_equal(Base::size_, data_.size())) return;
        // update and reallocate memory
        data_.resize(Base::size_ * (Base::size_ + 1) / 2);
        return;
    }
    // data pointers
    constexpr const StorageType& data() const { return data_; }
    constexpr StorageType& data() { return data_; }
   private:
    constexpr int size_(int i) {   // given x : x = 0.5 * (n * (n + 1)), computes n
        return (fdapde::sqrt(static_cast<double>(1 + 8 * i)) - 1) / 2;
    }
    StorageType data_;
};

// triangular view of an existing block of data
template <typename Scalar_, int Rows_, int ViewMode_>
class TriangularMatrixView :
    public TriangularMatrixBase<Scalar_, Rows_, ViewMode_, TriangularMatrixView<Scalar_, Rows_, ViewMode_>> {
   public:
    using Base = TriangularMatrixBase<Scalar_, Rows_, ViewMode_, TriangularMatrixView<Scalar_, Rows_, ViewMode_>>;
    using Scalar = Scalar_;
    using StorageType = std::add_pointer_t<Scalar>;
    static constexpr int ReadOnly = std::is_const_v<Scalar_> ? 1 : 0;
    static constexpr int NestAsRef = 1;
  
    // constructors
    constexpr TriangularMatrixView() : Base(), data_(nullptr) { }
    constexpr explicit TriangularMatrixView(Scalar* data) : Base(), data_(data) {
        fdapde_static_assert(Rows_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
    }
    constexpr TriangularMatrixView(Scalar* data, int size) : Base(size), data_(data) {
	fdapde_constexpr_assert(size > 0);
    }
    // data pointers
    constexpr const StorageType& data() const { return data_; }
    constexpr StorageType& data() { return data_; }
   private:
    StorageType data_;
};

// type aliases
template <typename Scalar, int Size> using UpperTriangularMatrix = TriangularMatrix<Scalar, Size, Upper>;
template <typename Scalar, int Size> using UpperTriangularMatrixView = TriangularMatrixView<Scalar, Size, Upper>;
template <typename Scalar, int Size> using LowerTriangularMatrix = TriangularMatrix<Scalar, Size, Lower>;  
template <typename Scalar, int Size> using LowerTriangularMatrixView = TriangularMatrixView<Scalar, Size, Lower>;

}   // namespace fdapde

#endif // __FDAPDE_TRIANGULAR_MATRIX_H__
