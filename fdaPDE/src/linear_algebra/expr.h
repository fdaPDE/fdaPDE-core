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

#ifndef __FDAPDE_LINALG_EXPR_H__
#define __FDAPDE_LINALG_EXPR_H__

#include "header_check.h"

namespace fdapde {

// MatrixExpr type-system base class
template <int Rows_, int Cols_, typename XprType_> struct MatrixExpr {
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    using XprType = XprType_;

    // assignment
    template <int RhsRows_, int RhsCols_, typename RhsXprType_>
    constexpr XprType& operator=(const MatrixExpr<RhsRows_, RhsCols_, RhsXprType_>& rhs) {
        using executor = typename XprType::assignment_executor;
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) {   // resize to rhs size, if lhs is Dynamic
            if (derived().rows() != rhs.rows() || derived().cols() != rhs.cols()) {
                derived().resize(rhs.rows(), rhs.cols());
            }
        }
        executor::run(derived(), rhs.derived(), [](auto& l, const auto& r) { l = r; });
        return derived();
    }
    // compound linear algebra
    template <int RhsXprRows_, int RhsXprCols_, typename RhsXprType_>
    constexpr XprType& operator+=(const MatrixExpr<RhsXprRows_, RhsXprCols_, RhsXprType_>& other) {
	using executor = typename XprType::assignment_executor;
        executor::run(derived(), other.derived(), [](auto& l, const auto& r) { l += r; });
        return derived();
    }
    template <int RhsXprRows_, int RhsXprCols_, typename RhsXprType_>
    constexpr XprType& operator-=(const MatrixExpr<RhsXprRows_, RhsXprCols_, RhsXprType_>& other) {
	using executor = typename XprType::assignment_executor;
        executor::run(derived(), other.derived(), [](auto& l, const auto& r) { l -= r; });
        return derived();
    }
    template <typename Scalar_>
        requires(std::is_arithmetic_v<Scalar_>)
    constexpr XprType& operator*=(Scalar_ coeff) {
        using executor = typename XprType::assignment_executor;
        executor::run(derived(), derived(), [coeff](auto& l, const auto& r) { l = coeff * r; });
        return derived();
    }
    template <typename Scalar_>
        requires(std::is_arithmetic_v<Scalar_>)
    constexpr XprType& operator/=(Scalar_ coeff) {
        using executor = typename XprType::assignment_executor;
        executor::run(derived(), derived(), [coeff](auto& l, const auto& r) { l = r / coeff; });
        return derived();
    }
    // compound matrix multiplication
    template <int RhsXprRows_, int RhsXprCols_, typename RhsXprType_>
    constexpr XprType& operator*=(const MatrixExpr<RhsXprRows_, RhsXprCols_, RhsXprType_>& other) {
	using executor = typename XprType::assignment_executor;
	// avoid aliasing by evaluating the product in a temporary
	using Scalar = typename XprType::Scalar;
	Matrix<Scalar, Rows, Cols> tmp = derived() * other;
	// assign
        executor::run(derived(), tmp, [](auto& l, const auto& r) { l = r; });
        return derived();
    }

    // observers
    constexpr int size() const {
        return (Rows != Dynamic && Cols != Dynamic) ? Rows * Cols : derived().rows() * derived().cols();
    }
    constexpr int rows() const { return Rows == Dynamic ? derived().rows() : Rows; }
    constexpr int cols() const { return Cols == Dynamic ? derived().cols() : Cols; }
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

    // coeffwise operators
    // general coefficient wise executor
    template <typename CoeffOp> constexpr auto cwise(CoeffOp&& op) const {
        using Scalar = typename XprType::Scalar;
        using CoeffOpReturnType = std::invoke_result_t<CoeffOp, Scalar>;
        fdapde_static_assert(
          std::is_convertible_v<CoeffOpReturnType FDAPDE_COMMA Scalar>, INVALID_COEFFWISE_OPERATOR_RETURN_TYPE);
        fdapde_assert(derived().rows() > 0 && derived().cols() > 0);
        return MatrixCoeffWiseOp<XprType, CoeffOp>(derived(), op);
    }
    constexpr auto cwise_abs() const {
        using Scalar = typename XprType::Scalar;
        return cwise([](Scalar x) { return fdapde::abs(x); });
    }
    constexpr auto cwise_pow(int i) const {
        using Scalar = typename XprType::Scalar;
        return cwise([i](Scalar x) { return fdapde::pow(x, i); });
    }
    constexpr auto cwise_pow2() const { return cwise_pow(2); }
    constexpr auto cwise_sqrt() const {
        using Scalar = typename XprType::Scalar;
        fdapde_static_assert(std::is_floating_point_v<Scalar>, THIS_METHOD_IS_FOR_FLOATING_POINT_MATRICES_ONLY);
        return cwise([](Scalar x) { return fdapde::sqrt(x); });
    }
    constexpr auto cwise_inv() const {
        using Scalar = typename XprType::Scalar;
        fdapde_static_assert(std::is_floating_point_v<Scalar>, THIS_METHOD_IS_FOR_FLOATING_POINT_MATRICES_ONLY);
        return cwise([](Scalar x) { return 1.0 / x; });
    }
    constexpr auto cwise_exp() const {
        using Scalar = typename XprType::Scalar;
        fdapde_static_assert(std::is_floating_point_v<Scalar>, THIS_METHOD_IS_FOR_FLOATING_POINT_MATRICES_ONLY);
        return cwise([](Scalar x) { return fdapde::exp(x); });
    }
    constexpr auto cwise_log() const {
        using Scalar = typename XprType::Scalar;
        fdapde_static_assert(std::is_floating_point_v<Scalar>, THIS_METHOD_IS_FOR_FLOATING_POINT_MATRICES_ONLY);
        return cwise([](Scalar x) { return fdapde::log(x); });
    }

    // redux operators
    // frobenius norm (squared L^2 norm)
    constexpr auto squared_norm() const {
        typename XprType::Scalar norm_ = 0;
        for (int i = 0; i < derived().rows(); ++i) {
            for (int j = 0; j < derived().cols(); ++j) { norm_ += fdapde::pow(derived().operator()(i, j), 2); }
        }
        return norm_;
    }
    constexpr auto norm() const { return fdapde::sqrt(squared_norm()); }
    // maximum norm (L^\infty norm)
    constexpr auto inf_norm() const {
        using Scalar = typename XprType::Scalar;
        Scalar norm_ = std::numeric_limits<Scalar>::min();
        for (int i = 0; i < derived().rows(); ++i) {
            for (int j = 0; j < derived().cols(); ++j) {
                Scalar tmp = fdapde::abs(derived().operator()(i, j));
                if (tmp > norm_) norm_ = tmp;
            }
        }
        return norm_;
    }
    // general redux executor
    template <typename Scalar_, typename ReduxOp> constexpr auto redux(Scalar_ init, ReduxOp&& op) const {
        fdapde_assert(derived().rows() > 0 && derived().cols() > 0);
        return MatrixReduxOp<XprType, internals::matrix_linear_redux_executor<XprType, ReduxOp>>(derived()).run(
          init, op);
    }
    constexpr auto sum() const {
        using Scalar = typename XprType::Scalar;
        if (derived().size() == 0) return Scalar(0);
        return redux(Scalar(0), [](Scalar tmp, Scalar x) { return tmp + x; });
    }
    constexpr auto prod() const {
        using Scalar = typename XprType::Scalar;
        if (derived().size() == 0) return Scalar(1);
        return redux(Scalar(1), [](Scalar tmp, Scalar x) { return tmp * x; });
    }
    constexpr auto mean() const { return derived().sum() / derived().size(); }
    constexpr auto max() const {
        using Scalar = typename XprType::Scalar;
        return redux(std::numeric_limits<Scalar>::min(), [](Scalar tmp, Scalar x) { return tmp > x ? tmp : x; });
    }
    constexpr auto min() const {
        using Scalar = typename XprType::Scalar;
        return redux(std::numeric_limits<Scalar>::max(), [](Scalar tmp, Scalar x) { return tmp < x ? tmp : x; });
    }
    // vector-wise redux operators
    constexpr MatrixRowWiseOp<XprType> rowwise() { return MatrixRowWiseOp<XprType>(derived()); }
    constexpr MatrixRowWiseOp<const XprType> rowwise() const { return MatrixRowWiseOp<const XprType>(derived()); }
    constexpr MatrixColWiseOp<XprType> colwise() { return MatrixColWiseOp<XprType>(derived()); }
    constexpr MatrixColWiseOp<const XprType> colwise() const { return MatrixColWiseOp<const XprType>(derived()); }

    // unary operators
    constexpr TransposeOp<XprType> transpose() const { return TransposeOp<XprType>(derived()); }
    constexpr Diagonal<Rows, 1, const XprType> diagonal() const { return Diagonal<Rows, 1, const XprType>(derived()); }
    constexpr Diagonal<Rows, 1, XprType> diagonal() { return Diagonal<Rows, 1, XprType>(derived()); }
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
    constexpr MatrixBlock<Rows, 1, XprType> col(int i) { return MatrixBlock<Rows, 1, XprType>(derived(), i); }
    constexpr MatrixBlock<Rows, 1, const XprType> col(int i) const {
        return MatrixBlock<Rows, 1, const XprType>(derived(), i);
    }
    constexpr MatrixBlock<1, Cols, XprType> row(int i) { return MatrixBlock<1, Cols, XprType>(derived(), i); }
    constexpr MatrixBlock<1, Cols, const XprType> row(int i) const {
        return MatrixBlock<1, Cols, const XprType>(derived(), i);
    }
    // other block-type accessors
    template <int BlockRows> constexpr auto top_rows() { return block<BlockRows, Cols>(0, 0); }
    template <int BlockRows> constexpr auto top_rows() const { return block<BlockRows, Cols>(0, 0); }
    constexpr auto top_rows(int rows) { return block(0, 0, rows, derived().cols()); }
    constexpr auto top_rows(int rows) const { return block(0, 0, rows, derived().cols()); }

    template <int BlockRows> constexpr auto bottom_rows() { return block<BlockRows, Cols>(Rows - BlockRows, 0); }
    template <int BlockRows> constexpr auto bottom_rows() const { return block<BlockRows, Cols>(Rows - BlockRows, 0); }
    constexpr auto bottom_rows(int rows) { return block(derived().rows() - rows, 0, rows, derived().cols()); }
    constexpr auto bottom_rows(int rows) const { return block(derived().rows() - rows, 0, rows, derived().cols()); }

    template <int BlockCols> constexpr auto left_cols() { return block<Rows, BlockCols>(0, 0); }
    template <int BlockCols> constexpr auto left_cols() const { return block<Rows, BlockCols>(0, 0); }
    constexpr auto left_cols(int cols) { return block(0, 0, derived().rows(), cols); }
    constexpr auto left_cols(int cols) const { return block(0, 0, derived().rows(), cols); }

    template <int BlockCols> constexpr auto right_cols() { return block<Rows, BlockCols>(0, Cols - BlockCols); }
    template <int BlockCols> constexpr auto right_cols() const { return block<Rows, BlockCols>(0, Cols - BlockCols); }
    constexpr auto right_cols(int cols) { return block(0, derived().cols() - cols, derived().rows(), cols); }
    constexpr auto right_cols(int cols) const { return block(0, derived().cols() - cols, derived().rows(), cols); }

    // dot product
    template <int RhsRows, int RhsCols, typename RhsXprType>
    constexpr auto dot(const MatrixExpr<RhsRows, RhsCols, RhsXprType>& rhs) const {
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
  
    // reshaping
    // static-sized
    template <int ReshapedRows_, int ReshapedCols_, int StorageOrder_ = RowMajor> constexpr auto reshape() {
        return ReshapeOp<ReshapedRows_, ReshapedCols_, StorageOrder_, XprType>(derived());
    }
    template <int ReshapedRows_, int ReshapedCols_, int StorageOrder_ = RowMajor> constexpr auto reshape() const {
        return ReshapeOp<ReshapedRows_, ReshapedCols_, StorageOrder_, const XprType>(derived());
    }
    template <int ReshapedRows_, int StorageOrder_ = RowMajor> constexpr auto reshape() {
        return ReshapeOp<ReshapedRows_, 1, RowMajor, XprType>(derived());
    }
    template <int ReshapedRows_, int StorageOrder_ = RowMajor> constexpr auto reshape() const {
        return ReshapeOp<ReshapedRows_, 1, RowMajor, const XprType>(derived());
    }
    // dynamic-sized
    constexpr auto reshape(int rows, int cols) {
        return ReshapeOp<Dynamic, Dynamic, RowMajor, XprType>(derived(), rows, cols);
    }
    constexpr auto reshape(int rows, int cols) const {
        return ReshapeOp<Dynamic, Dynamic, RowMajor, const XprType>(derived(), rows, cols);
    }
    constexpr auto reshape(int rows) { return ReshapeOp<Dynamic, 1, RowMajor, XprType>(derived(), rows); }
    constexpr auto reshape(int rows) const { return ReshapeOp<Dynamic, 1, RowMajor, const XprType>(derived(), rows); }
    // dynamic-sized with storage order control
    template <int StorageOrder_> constexpr auto reshape(int rows, int cols) {
        return ReshapeOp<Dynamic, Dynamic, StorageOrder_, XprType>(derived(), rows, cols);
    }
    template <int StorageOrder_> constexpr auto reshape(int rows, int cols) const {
        return ReshapeOp<Dynamic, Dynamic, StorageOrder_, const XprType>(derived(), rows, cols);
    }

    // square matrix methods
    constexpr auto symm_part() const {
        fdapde_static_assert(
          Rows == Dynamic || Cols == Dynamic || Rows == Cols, THIS_METHODS_IS_FOR_SQUARE_MATRICES_ONLY);
        if constexpr (Rows == Dynamic || Cols == Dynamic) {
            fdapde_assert(derived().rows() == derived().cols());
        }
        return 0.5 * (derived() + derived().transpose());   // symmetric part
    }
    constexpr auto skew_part() const {
        fdapde_static_assert(
          Rows == Dynamic || Cols == Dynamic || Rows == Cols, THIS_METHODS_IS_FOR_SQUARE_MATRICES_ONLY);
        if constexpr (Rows == Dynamic || Cols == Dynamic) {
            fdapde_assert(derived().rows() == derived().cols());
        }
        return 0.5 * (derived() - derived().transpose());   // skew-symmetric part
    }
    // triangular block accessors
    template <int BlockMode> constexpr TriangularBlock<const XprType, BlockMode> triangular_block() const {
        return TriangularBlock<const XprType, BlockMode>(derived());
    }
    template <int BlockMode> constexpr TriangularBlock<XprType, BlockMode> triangular_block() {
        return TriangularBlock<XprType, BlockMode>(derived());
    }
    // cast
    template <int ViewMode> constexpr auto as_symmetric() const {
        return internals::symmetric_cast<ViewMode>(derived());
    }
    template <int ViewMode> constexpr auto as_symmetric() { return internals::symmetric_cast<ViewMode>(derived()); }
    constexpr auto as_diagonal() const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return internals::diagonal_cast(derived());
    }
    constexpr auto as_diagonal() {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return internals::diagonal_cast(derived());
    }
};

// comparison operators
template <int Rows1, int Cols1, typename XprType1, int Rows2, int Cols2, typename XprType2>
constexpr bool
operator==(const MatrixExpr<Rows1, Cols1, XprType1>& op1, const MatrixExpr<Rows2, Cols2, XprType2>& op2) {
    fdapde_static_assert(
      (internals::is_dynamic_sized_v<XprType1> || internals::is_dynamic_sized_v<XprType2> ||
       (Rows1 == Rows2 && Cols1 == Cols2)),
      INVALID_COMPARISON__OPERANDS_HAVE_DIFFERENT_SIZES);
    if constexpr (internals::is_dynamic_sized_v<XprType1> || internals::is_dynamic_sized_v<XprType2>) {
        fdapde_assert(op1.rows() == op2.rows() && op1.cols() == op2.cols());
    }
    const auto& d1 = op1.derived();
    const auto& d2 = op2.derived();
    for (int i = 0, n = d1.rows(); i < n; ++i) {
        for (int j = 0, m = d1.cols(); j < m; ++j) {
            if (d1(i, j) != d2(i, j)) { return false; }
        }
    }
    return true;
}
template <int Rows1, int Cols1, typename XprType1, int Rows2, int Cols2, typename XprType2>
constexpr bool
operator!=(const MatrixExpr<Rows1, Cols1, XprType1>& op1, const MatrixExpr<Rows2, Cols2, XprType2>& op2) {
    return !(op1 == op2);
}
template <int Rows1, int Cols1, typename XprType1, int Rows2, int Cols2, typename XprType2>
constexpr bool almost_equal(
  const MatrixExpr<Rows1, Cols1, XprType1>& op1, const MatrixExpr<Rows2, Cols2, XprType2>& op2, double epsilon = 1e-7) {
    fdapde_static_assert(Rows1 == Rows2 && Cols1 == Cols2, YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES);
    using Scalar_ = typename XprType1::Scalar;
    const auto& d1 = op1.derived();
    const auto& d2 = op2.derived();
    for (int i = 0; i < Rows1; ++i) {
        for (int j = 0; j < Cols1; ++j) {
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

#endif // __FDAPDE_LINALG_EXPR_H__
