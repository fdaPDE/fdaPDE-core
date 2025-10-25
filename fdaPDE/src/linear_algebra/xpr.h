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

#ifndef __FDAPDE_LINALG_XPR_H__
#define __FDAPDE_LINALG_XPR_H__

#include "header_check.h"

namespace fdapde {

// MatrixExpr type-system base class
template <typename XprType_> struct MatrixExpr {
    using XprType = XprType_;

    // assignment
    template <typename RhsXprType_> constexpr XprType& operator=(const MatrixExpr<RhsXprType_>& rhs) {
        using executor = typename XprType::assignment_executor;
        constexpr int Rows = XprType::Rows;
        constexpr int Cols = XprType::Cols;
        if constexpr (requires(XprType_ xpr, int i, int j) {
                          xpr.resize(i, j);
                      } && (Rows == Dynamic || Cols == Dynamic)) {
            if (derived().rows() != rhs.rows() || derived().cols() != rhs.cols()) {
                derived().resize(rhs.rows(), rhs.cols());
            }
        }
        executor::run(derived(), rhs.derived(), [](auto& l, const auto& r) { l = r; });
        return derived();
    }
    template <typename RhsXprType_> constexpr XprType& operator=(const MatrixCoeffWiseExpr<RhsXprType_>& rhs) {
        operator=(rhs.mwise());
        return derived();
    }
    // compound linear algebra
    template <typename RhsXprType_> constexpr XprType& operator+=(const MatrixExpr<RhsXprType_>& rhs) {
        using executor = typename XprType::assignment_executor;
        executor::run(derived(), rhs.derived(), [](auto& l, const auto& r) { l += r; });
        return derived();
    }
    template <typename RhsXprType_> constexpr XprType& operator-=(const MatrixExpr<RhsXprType_>& rhs) {
        using executor = typename XprType::assignment_executor;
        executor::run(derived(), rhs.derived(), [](auto& l, const auto& r) { l -= r; });
        return derived();
    }
    template <typename Scalar_>
        requires(std::is_arithmetic_v<Scalar_>)
    constexpr XprType& operator*=(Scalar_ rhs) {
        using executor = typename XprType::assignment_executor;
        executor::run(derived(), derived(), [rhs](auto& l, const auto& r) { l = rhs * r; });
        return derived();
    }
    template <typename Scalar_>
        requires(std::is_arithmetic_v<Scalar_>)
    constexpr XprType& operator/=(Scalar_ rhs) {
        using executor = typename XprType::assignment_executor;
        executor::run(derived(), derived(), [rhs](auto& l, const auto& r) { l = r / rhs; });
        return derived();
    }
    // compound matrix multiplication
    template <typename RhsXprType_> constexpr XprType& operator*=(const MatrixExpr<RhsXprType_>& rhs) {
        using executor = typename XprType::assignment_executor;
        // avoid aliasing by evaluating the product in a temporary
        using Scalar = typename XprType::Scalar;
	constexpr int Rows = XprType::Rows;
	constexpr int Cols = XprType::Cols;
        Matrix<Scalar, Rows, Cols> tmp = derived() * rhs;
        // assign
        executor::run(derived(), tmp, [](auto& l, const auto& r) { l = r; });
        return derived();
    }

    // observers
    constexpr int rows() const { return XprType::Rows == Dynamic ? derived().rows() : XprType::Rows; }
    constexpr int cols() const { return XprType::Cols == Dynamic ? derived().cols() : XprType::Cols; }
    constexpr int size() const {
        constexpr int Rows = XprType::Rows;
        constexpr int Cols = XprType::Cols;
        return (Rows != Dynamic && Cols != Dynamic) ? Rows * Cols : derived().rows() * derived().cols();
    }
    constexpr const XprType& derived() const { return static_cast<const XprType&>(*this); }
    constexpr XprType& derived() { return static_cast<XprType&>(*this); }
    // ostream
    friend std::ostream& operator<<(std::ostream& os, const MatrixExpr& m) {
        const int rows = m.derived().rows();
        const int cols = m.derived().cols();
        const auto& d = m.derived();
        // compute max width per column
        size_t width = 0;
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                std::ostringstream ss;
                ss << d(i, j);
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
    // coeffwise access
    constexpr auto cwise() const {
        return MatrixCoeffWiseOp<const XprType, internals::identity_op>(derived(), internals::identity_op());
    }
    constexpr auto cwise() {
        return MatrixCoeffWiseOp<XprType, internals::identity_op>(derived(), internals::identity_op());
    }

    // redux operators
    // frobenius norm (squared L^2 norm)
    constexpr auto squared_norm() const {
        typename XprType::Scalar norm_ = 0;
        for (int i = 0; i < derived().rows(); ++i) {
            for (int j = 0; j < derived().cols(); ++j) { norm_ += fdapde::pow(derived()(i, j), 2); }
        }
        return norm_;
    }
    constexpr auto norm() const { return fdapde::sqrt(squared_norm()); }
    // maximum norm (L^\infty norm)
    constexpr auto inf_norm() const {
        using Scalar = typename XprType::Scalar;
        Scalar norm_ = std::numeric_limits<Scalar>::min();
        const int rows = derived().rows();
        const int cols = derived().cols();
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                Scalar tmp = fdapde::abs(derived()(i, j));
                if (tmp > norm_) norm_ = tmp;
            }
        }
        return norm_;
    }
    // general redux executor
    template <typename Scalar_, typename ReduxOp> constexpr auto redux(Scalar_ init, ReduxOp&& op) const {
        fdapde_assert(derived().rows() > 0 && derived().cols() > 0);
        return internals::matrix_redux_linear_executor::run(derived(), init, op);
    }
    constexpr auto sum() const {
        using Scalar = typename XprType::Scalar;
        if (derived().size() == 0) return Scalar(0);
        return redux(Scalar(0), [](const Scalar& tmp, const Scalar& x) { return tmp + x; });
    }
    constexpr auto prod() const {
        using Scalar = typename XprType::Scalar;
        if (derived().size() == 0) return Scalar(1);
        return redux(Scalar(1), [](const Scalar& tmp, const Scalar& x) { return tmp * x; });
    }
    constexpr auto mean() const { return derived().sum() / derived().size(); }
    constexpr auto max() const {
        using Scalar = typename XprType::Scalar;
        return redux(
          std::numeric_limits<Scalar>::min(), [](const Scalar& tmp, const Scalar& x) { return tmp > x ? tmp : x; });
    }
    constexpr auto min() const {
        using Scalar = typename XprType::Scalar;
        return redux(
          std::numeric_limits<Scalar>::max(), [](const Scalar& tmp, const Scalar& x) { return tmp < x ? tmp : x; });
    }
    // boolean reductions
    // true if at least one of the coefficients of the expression evalutes true
    constexpr bool any() const {
        return internals::boolean_redux_linear_executor::run(derived(), 1, [](const auto& x) { return  bool(x); });
    }
    // true if none of the coefficients of the expression evaluates false
    constexpr bool all() const {
        return internals::boolean_redux_linear_executor::run(derived(), 0, [](const auto& x) { return !bool(x); });
    }
    // number of coefficients evaluating true in the expression
    constexpr int count() const {
        if (derived().size() == 0) return int(0);
        return redux(int(0), [](int cnt, auto x) { return cnt + (bool(x) ? 1 : 0); });
    }
    // vector-wise redux operators
    constexpr MatrixRowWiseOp<XprType> rowwise() { return MatrixRowWiseOp<XprType>(derived()); }
    constexpr MatrixRowWiseOp<const XprType> rowwise() const { return MatrixRowWiseOp<const XprType>(derived()); }
    constexpr MatrixColWiseOp<XprType> colwise() { return MatrixColWiseOp<XprType>(derived()); }
    constexpr MatrixColWiseOp<const XprType> colwise() const { return MatrixColWiseOp<const XprType>(derived()); }

    // unary operators
    constexpr TransposeOp<XprType> transpose() const { return TransposeOp<XprType>(derived()); }
    constexpr auto diagonal() const { return Diagonal<const XprType>(derived()); }
    constexpr auto diagonal() { return Diagonal<XprType>(derived()); }
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
    constexpr auto col(int i) { return MatrixBlock<XprType::Rows, 1, XprType>(derived(), i); }
    constexpr auto col(int i) const { return MatrixBlock<XprType::Rows, 1, const XprType>(derived(), i); }
    constexpr auto row(int i) { return MatrixBlock<1, XprType::Cols, XprType>(derived(), i); }
    constexpr auto row(int i) const { return MatrixBlock<1, XprType::Cols, const XprType>(derived(), i); }
    // other block-type accessors
    template <int BlockRows> constexpr auto top_rows() { return block<BlockRows, XprType::Cols>(0, 0); }
    template <int BlockRows> constexpr auto top_rows() const { return block<BlockRows, XprType::Cols>(0, 0); }
    constexpr auto top_rows(int rows) { return block(0, 0, rows, derived().cols()); }
    constexpr auto top_rows(int rows) const { return block(0, 0, rows, derived().cols()); }

    template <int BlockRows> constexpr auto bottom_rows() {
        return block<BlockRows, XprType::Cols>(derived().rows() - BlockRows, 0);
    }
    template <int BlockRows> constexpr auto bottom_rows() const {
        return block<BlockRows, XprType::Cols>(derived().rows() - BlockRows, 0);
    }
    constexpr auto bottom_rows(int rows) { return block(derived().rows() - rows, 0, rows, derived().cols()); }
    constexpr auto bottom_rows(int rows) const { return block(derived().rows() - rows, 0, rows, derived().cols()); }

    template <int BlockCols> constexpr auto left_cols() { return block<XprType::Rows, BlockCols>(0, 0); }
    template <int BlockCols> constexpr auto left_cols() const { return block<XprType::Rows, BlockCols>(0, 0); }
    constexpr auto left_cols(int cols) { return block(0, 0, derived().rows(), cols); }
    constexpr auto left_cols(int cols) const { return block(0, 0, derived().rows(), cols); }

    template <int BlockCols> constexpr auto right_cols() {
        return block<XprType::Rows, BlockCols>(0, derived().cols() - BlockCols);
    }
    template <int BlockCols> constexpr auto right_cols() const {
        return block<XprType::Rows, BlockCols>(0, derived().cols() - BlockCols);
    }
    constexpr auto right_cols(int cols) { return block(0, derived().cols() - cols, derived().rows(), cols); }
    constexpr auto right_cols(int cols) const { return block(0, derived().cols() - cols, derived().rows(), cols); }

    // // dot product
    template <typename RhsXprType> constexpr auto dot(const MatrixExpr<RhsXprType>& rhs) const {
        constexpr int RhsRows = RhsXprType::Rows, Rows = XprType::Rows;
        constexpr int RhsCols = RhsXprType::Cols, Cols = XprType::Cols;
        fdapde_static_assert(
          (RhsRows == 1 || RhsCols == 1) && (Rows == 1 || Cols == 1), THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        fdapde_assert(derived().size() == rhs.derived().size());
        using Scalar = std::common_type_t<typename XprType::Scalar, typename RhsXprType::Scalar>;
        Scalar dot_ = 0;
        for (int i = 0, n = fdapde::max(rows(), cols()); i < n; ++i) {
            dot_ += derived().operator[](i) * rhs.derived().operator[](i);
        }
        return dot_;
    }
    // cross product
    template <typename RhsXprType> constexpr auto cross(const MatrixExpr<RhsXprType>& rhs) const {
        return MatrixCrossProductOp<XprType, RhsXprType>(derived(), rhs.derived());
    }

    // reshaping
    // static-sized
    template <int ReshapedRows_, int ReshapedCols_> constexpr auto reshape() {
        return ReshapeOp<ReshapedRows_, ReshapedCols_, XprType>(derived());
    }
    template <int ReshapedRows_, int ReshapedCols_> constexpr auto reshape() const {
        return ReshapeOp<ReshapedRows_, ReshapedCols_, const XprType>(derived());
    }
    template <int ReshapedRows_> constexpr auto reshape() { return ReshapeOp<ReshapedRows_, 1, XprType>(derived()); }
    template <int ReshapedRows_> constexpr auto reshape() const {
        return ReshapeOp<ReshapedRows_, 1, const XprType>(derived());
    }
    // dynamic-sized
    constexpr auto reshape(int rows, int cols) { return ReshapeOp<Dynamic, Dynamic, XprType>(derived(), rows, cols); }
    constexpr auto reshape(int rows, int cols) const {
        return ReshapeOp<Dynamic, Dynamic, const XprType>(derived(), rows, cols);
    }
    constexpr auto reshape(int rows) { return ReshapeOp<Dynamic, 1, XprType>(derived(), rows); }
    constexpr auto reshape(int rows) const { return ReshapeOp<Dynamic, 1, const XprType>(derived(), rows); }

    // square matrix methods
    constexpr auto symm_part() const {
        constexpr int Rows = XprType::Rows, Cols = XprType::Cols;
        fdapde_static_assert(
          Rows == Dynamic || Cols == Dynamic || Rows == Cols, THIS_METHODS_IS_FOR_SQUARE_MATRICES_ONLY);
        if constexpr (Rows == Dynamic || Cols == Dynamic) {
            fdapde_assert(derived().rows() == derived().cols());
        }
        return 0.5 * (derived() + derived().transpose());   // symmetric part
    }
    constexpr auto skew_part() const {
        constexpr int Rows = XprType::Rows, Cols = XprType::Cols;
        fdapde_static_assert(
          Rows == Dynamic || Cols == Dynamic || Rows == Cols, THIS_METHODS_IS_FOR_SQUARE_MATRICES_ONLY);
        if constexpr (Rows == Dynamic || Cols == Dynamic) {
            fdapde_assert(derived().rows() == derived().cols());
        }
        return 0.5 * (derived() - derived().transpose());   // skew-symmetric part
    }
    constexpr auto inverse() const {
        using Scalar = typename XprType::Scalar;
        constexpr int Rows = XprType::Rows, Cols = XprType::Cols;
        fdapde_static_assert(
          Rows == Dynamic || Cols == Dynamic || Rows == Cols, THIS_METHODS_IS_FOR_SQUARE_MATRICES_ONLY);
        if constexpr (Rows == Dynamic || Cols == Dynamic) { fdapde_assert(derived().rows() == derived().cols()); }
        Matrix<Scalar, Rows, Cols> inverse_;
        const XprType& m = derived();
        const int rows_ = m.rows(), cols_ = m.cols();
        fdapde_assert(rows_ == cols_);
        if constexpr (Rows == Dynamic || Cols == Dynamic) { inverse_.resize(rows_, cols_); }
        // inverse computation
        if (rows_ == 1) {
            inverse_(0, 0) = Scalar(1) / m(0, 0);
            return inverse_;
        }
        if (rows_ == 2) {
            const Scalar a00 = m(0, 0), a01 = m(0, 1), a10 = m(1, 0), a11 = m(1, 1);
            const Scalar det = a00 * a11 - a01 * a10;
            const Scalar inv_det = Scalar(1) / det;
            inverse_(0, 0) =  a11 * inv_det;
            inverse_(0, 1) = -a01 * inv_det;
            inverse_(1, 0) = -a10 * inv_det;
            inverse_(1, 1) =  a00 * inv_det;
            return inverse_;
        }
        if (rows_ == 3) {
            const Scalar a00 = m(0, 0), a01 = m(0, 1), a02 = m(0, 2);
            const Scalar a10 = m(1, 0), a11 = m(1, 1), a12 = m(1, 2);
            const Scalar a20 = m(2, 0), a21 = m(2, 1), a22 = m(2, 2);
            // cache shared cofactors
            const Scalar c00 = a11 * a22 - a12 * a21;
            const Scalar c10 = a12 * a20 - a10 * a22;
            const Scalar c20 = a10 * a21 - a11 * a20;
            // compute determinant and assemble inverse
            const Scalar det = a00 * c00 + a01 * c10 + a02 * c20;
            const Scalar inv_det = Scalar(1) / det;
            inverse_(0, 0) = c00 * inv_det;
            inverse_(1, 0) = c10 * inv_det;
            inverse_(2, 0) = c20 * inv_det;
            inverse_(0, 1) = -(a01 * a22 - a02 * a21) * inv_det;
            inverse_(1, 1) =  (a00 * a22 - a02 * a20) * inv_det;
            inverse_(2, 1) = -(a00 * a21 - a01 * a20) * inv_det;
            inverse_(0, 2) =  (a01 * a12 - a02 * a11) * inv_det;
            inverse_(1, 2) = -(a00 * a12 - a02 * a10) * inv_det;
            inverse_(2, 2) =  (a00 * a11 - a01 * a10) * inv_det;
            return inverse_;
        }
        // general fallback (compute M*X = I by LU factorizatoin)
        PartialPivLU<Matrix<Scalar, Rows, Cols>> lu(m);
        Matrix<Scalar, Rows, Cols> I;
        if constexpr (Rows == Dynamic || Cols == Dynamic) { I.resize(rows_, cols_); }
        for (int i = 0; i < rows_; ++i) { I(i, i) = Scalar(1); }
        return lu.solve(I);
    }
    auto determinant() const {
        using Scalar = typename XprType::Scalar;
        constexpr int Rows = XprType::Rows, Cols = XprType::Cols;
        fdapde_static_assert(
          Rows == Dynamic || Cols == Dynamic || Rows == Cols, THIS_METHODS_IS_FOR_SQUARE_MATRICES_ONLY);
        if constexpr (Rows == Dynamic || Cols == Dynamic) {
            fdapde_assert(derived().rows() == derived().cols());
        }
        const XprType& m = derived();
        const int rows_ = m.rows(), cols_ = m.cols();
        fdapde_assert(rows_ == cols_);
	// determinant computation
        if (rows_ == 1) { return m(0, 0); }
        if (rows_ == 2) {
            const Scalar a00 = m(0, 0), a01 = m(0, 1), a10 = m(1, 0), a11 = m(1, 1);
            return a00 * a11 - a01 * a10;
        }
        if (rows_ == 3) {
            const Scalar a00 = m(0, 0), a01 = m(0, 1), a02 = m(0, 2);
            const Scalar a10 = m(1, 0), a11 = m(1, 1), a12 = m(1, 2);
            const Scalar a20 = m(2, 0), a21 = m(2, 1), a22 = m(2, 2);
            return a00 * (a11 * a22 - a12 * a21) + a01 * (a12 * a20 - a10 * a22) + a02 * (a10 * a21 - a11 * a20);
        }
        // general fallback (factorize and extract determinant)
        PartialPivLU<XprType> lu(m);
        return m.determinant();
    }

    // triangular block accessors
    template <int BlockMode> constexpr TriangularBlock<const XprType, BlockMode> triangular_block() const {
        return TriangularBlock<const XprType, BlockMode>(derived());
    }
    template <int BlockMode> constexpr TriangularBlock<XprType, BlockMode> triangular_block() {
        return TriangularBlock<XprType, BlockMode>(derived());
    }
    // cast
    // template <int ViewMode> constexpr auto as_symmetric() const {
    //     return internals::symmetric_cast<ViewMode>(derived());
    // }
    // template <int ViewMode> constexpr auto as_symmetric() { return internals::symmetric_cast<ViewMode>(derived()); }
    constexpr auto as_diagonal() const {
        fdapde_static_assert(XprType::Rows == 1 || XprType::Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return internals::diagonal_cast(derived());
    }
    constexpr auto as_diagonal() {
        fdapde_static_assert(XprType::Rows == 1 || XprType::Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return internals::diagonal_cast(derived());
    }
};

// comparison operators
template <typename LhsXprType, typename RhsXprType>
constexpr bool
operator==(const MatrixExpr<LhsXprType>& lhs, const MatrixExpr<RhsXprType>& rhs) {
    fdapde_static_assert(
      (internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType> ||
       internals::same_static_shape_v<LhsXprType FDAPDE_COMMA RhsXprType>),
      INVALID_COMPARISON__MATRICES_OF_DIFFERENT_STATIC_SIZE);
    if constexpr (internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType>) {
        fdapde_assert(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols());
    }
    const auto& d1 = lhs.derived();
    const auto& d2 = rhs.derived();
    for (int i = 0, n = d1.rows(); i < n; ++i) {
        for (int j = 0, m = d1.cols(); j < m; ++j) {
            if (d1(i, j) != d2(i, j)) { return false; }
        }
    }
    return true;
}
template <typename LhsXprType, typename RhsXprType>
constexpr bool operator!=(const MatrixExpr<LhsXprType>& op1, const MatrixExpr<RhsXprType>& op2) {
    return !(op1 == op2);
}
template <typename LhsXprType, typename RhsXprType>
constexpr bool
almost_equal(const MatrixExpr<LhsXprType>& lhs, const MatrixExpr<RhsXprType>& rhs, double epsilon = 1e-7) {
    fdapde_static_assert(
      (internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType> ||
       internals::same_static_shape_v<LhsXprType FDAPDE_COMMA RhsXprType>),
      INVALID_COMPARISON__MATRICES_OF_DIFFERENT_STATIC_SIZE);
    if constexpr (internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType>) {
        fdapde_assert(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols());
    }
    using Scalar_ = std::common_type_t<typename LhsXprType::Scalar, typename RhsXprType::Scalar>;
    const auto& d1 = lhs.derived();
    const auto& d2 = rhs.derived();
    for (int i = 0, n = d1.rows(); i < n; ++i) {
        for (int j = 0, m = d1.cols(); j < m; ++j) {
            Scalar_ a = d1(i, j);
            Scalar_ b = d2(i, j);
            if (!(fdapde::fabs(a - b) < epsilon ||
                  fdapde::fabs(a - b) <
                    ((fdapde::fabs(a) < fdapde::fabs(b) ? fdapde::fabs(b) : fdapde::fabs(a)) * epsilon))) {
                return false;
            }
        }
    }
    return true;
}

}   // namespace fdapde

#endif // __FDAPDE_LINALG_XPR_H__
