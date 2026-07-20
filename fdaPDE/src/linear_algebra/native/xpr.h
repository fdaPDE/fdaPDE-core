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

#include <iomanip>
#include <sstream>

namespace fdapde::linalg {

template <typename XprType> struct Diagonal;
template <int ViewMode, typename XprType> struct Triangular;

namespace internals {

template <typename XprType> constexpr auto diagonal_cast(XprType&& xpr);
template <int ViewMode, typename XprType> auto symmetric_cast(XprType&& xpr);

}   // namespace internals

// MatrixExpr type-system base class
template <typename XprType_> struct MatrixExpr {
    using XprType = XprType_;

    // assignment
    template <typename RhsXprType_>
        requires(XprType::ReadOnly == 0)
    constexpr XprType& operator=(const MatrixExpr<RhsXprType_>& rhs) & {
        using executor = typename XprType::assignment_executor;
        using RhsXprType = std::decay_t<RhsXprType_>;
        Matrix<typename RhsXprType::Scalar, RhsXprType::Rows, RhsXprType::Cols, RhsXprType::StorageOrder> tmp(rhs);
        constexpr int Rows = XprType::Rows;
        constexpr int Cols = XprType::Cols;
        if constexpr (requires(XprType_ xpr, int i, int j) {
                          xpr.resize(i, j);
                      } && (Rows == Dynamic || Cols == Dynamic)) {
            if constexpr (Rows == 1 || Cols == 1) {
                if (derived().size() != tmp.size()) derived().resize(tmp.size());
            } else if (derived().rows() != tmp.rows() || derived().cols() != tmp.cols()) {
                derived().resize(tmp.rows(), tmp.cols());
            }
        }
        executor::run(derived(), tmp, [](auto& l, const auto& r) { l = r; });
        return derived();
    }
    template <typename RhsXprType_>
    constexpr XprType operator=(const MatrixExpr<RhsXprType_>& rhs) &&
        requires(XprType::NestAsRef == 0 && XprType::ReadOnly == 0)
    {
        static_cast<MatrixExpr&>(*this).operator=(rhs);
        return derived();
    }
    template <typename RhsXprType_>
    constexpr void operator=(const MatrixExpr<RhsXprType_>&) &&
        requires(XprType::NestAsRef != 0)
      = delete;
    template <typename RhsXprType_>
    constexpr XprType& operator=(const MatrixCoeffWiseExpr<RhsXprType_>& rhs) &
        requires(XprType::ReadOnly == 0)
    {
        operator=(rhs.mwise());
        return derived();
    }
    template <typename RhsXprType_>
    constexpr XprType operator=(const MatrixCoeffWiseExpr<RhsXprType_>& rhs) &&
        requires(XprType::NestAsRef == 0 && XprType::ReadOnly == 0)
    {
        static_cast<MatrixExpr&>(*this).operator=(rhs);
        return derived();
    }
    template <typename RhsXprType_>
    constexpr void operator=(const MatrixCoeffWiseExpr<RhsXprType_>&) &&
        requires(XprType::NestAsRef != 0)
      = delete;
    // compound algebra
    template <typename RhsXprType_>
        requires(XprType::ReadOnly == 0)
    constexpr XprType& operator+=(const MatrixExpr<RhsXprType_>& rhs) & {
        using executor = typename XprType::assignment_executor;
        using RhsXprType = std::decay_t<RhsXprType_>;
        Matrix<typename RhsXprType::Scalar, RhsXprType::Rows, RhsXprType::Cols, RhsXprType::StorageOrder> tmp(rhs);
        executor::run(derived(), tmp, [](auto& l, const auto& r) { l += r; });
        return derived();
    }
    template <typename RhsXprType_>
    constexpr XprType operator+=(const MatrixExpr<RhsXprType_>& rhs) &&
        requires(XprType::NestAsRef == 0 && XprType::ReadOnly == 0)
    {
        static_cast<MatrixExpr&>(*this).operator+=(rhs);
        return derived();
    }
    template <typename RhsXprType_>
    constexpr void operator+=(const MatrixExpr<RhsXprType_>&) &&
        requires(XprType::NestAsRef != 0)
      = delete;
    template <typename RhsXprType_>
        requires(XprType::ReadOnly == 0)
    constexpr XprType& operator-=(const MatrixExpr<RhsXprType_>& rhs) & {
        using executor = typename XprType::assignment_executor;
        using RhsXprType = std::decay_t<RhsXprType_>;
        Matrix<typename RhsXprType::Scalar, RhsXprType::Rows, RhsXprType::Cols, RhsXprType::StorageOrder> tmp(rhs);
        executor::run(derived(), tmp, [](auto& l, const auto& r) { l -= r; });
        return derived();
    }
    template <typename RhsXprType_>
    constexpr XprType operator-=(const MatrixExpr<RhsXprType_>& rhs) &&
        requires(XprType::NestAsRef == 0 && XprType::ReadOnly == 0)
    {
        static_cast<MatrixExpr&>(*this).operator-=(rhs);
        return derived();
    }
    template <typename RhsXprType_>
    constexpr void operator-=(const MatrixExpr<RhsXprType_>&) &&
        requires(XprType::NestAsRef != 0)
      = delete;
    template <typename Scalar_>
        requires(std::is_arithmetic_v<Scalar_> && XprType::ReadOnly == 0)
    constexpr XprType& operator*=(Scalar_ rhs) & {
        using executor = typename XprType::assignment_executor;
        executor::run(derived(), derived(), [rhs](auto& l, const auto& r) { l = rhs * r; });
        return derived();
    }
    template <typename Scalar_>
        requires(std::is_arithmetic_v<Scalar_> && XprType::NestAsRef == 0 && XprType::ReadOnly == 0)
    constexpr XprType operator*=(Scalar_ rhs) && {
        static_cast<MatrixExpr&>(*this).operator*=(rhs);
        return derived();
    }
    template <typename Scalar_>
        requires(std::is_arithmetic_v<Scalar_> && XprType::NestAsRef != 0)
    constexpr void operator*=(Scalar_) && = delete;
    template <typename Scalar_>
        requires(std::is_arithmetic_v<Scalar_> && XprType::ReadOnly == 0)
    constexpr XprType& operator/=(Scalar_ rhs) & {
        using executor = typename XprType::assignment_executor;
        executor::run(derived(), derived(), [rhs](auto& l, const auto& r) { l = r / rhs; });
        return derived();
    }
    template <typename Scalar_>
        requires(std::is_arithmetic_v<Scalar_> && XprType::NestAsRef == 0 && XprType::ReadOnly == 0)
    constexpr XprType operator/=(Scalar_ rhs) && {
        static_cast<MatrixExpr&>(*this).operator/=(rhs);
        return derived();
    }
    template <typename Scalar_>
        requires(std::is_arithmetic_v<Scalar_> && XprType::NestAsRef != 0)
    constexpr void operator/=(Scalar_) && = delete;
    // compound matrix multiplication
    template <typename RhsXprType_>
        requires(XprType::ReadOnly == 0)
    constexpr XprType& operator*=(const MatrixExpr<RhsXprType_>& rhs) & {
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
    template <typename RhsXprType_>
    constexpr XprType operator*=(const MatrixExpr<RhsXprType_>& rhs) &&
        requires(XprType::NestAsRef == 0 && XprType::ReadOnly == 0)
    {
        static_cast<MatrixExpr&>(*this).operator*=(rhs);
        return derived();
    }
    template <typename RhsXprType_>
    constexpr void operator*=(const MatrixExpr<RhsXprType_>&) &&
        requires(XprType::NestAsRef != 0)
      = delete;
    // observers
    constexpr int rows() const { return XprType::Rows == Dynamic ? derived().rows() : XprType::Rows; }
    constexpr int cols() const { return XprType::Cols == Dynamic ? derived().cols() : XprType::Cols; }
    constexpr int size() const {
        constexpr int Rows = XprType::Rows;
        constexpr int Cols = XprType::Cols;
        return (Rows != Dynamic && Cols != Dynamic) ? Rows * Cols : derived().rows() * derived().cols();
    }
    constexpr const XprType& derived() const & { return static_cast<const XprType&>(*this); }
    constexpr XprType& derived() & { return static_cast<XprType&>(*this); }
    constexpr void derived() const && = delete;
    constexpr void derived() && = delete;
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
    constexpr auto cwise() const & {
        return MatrixCoeffWiseOp<const XprType, internals::identity_op>(derived(), internals::identity_op());
    }
    constexpr auto cwise() & {
        return MatrixCoeffWiseOp<XprType, internals::identity_op>(derived(), internals::identity_op());
    }
    constexpr auto cwise() const && requires(XprType::NestAsRef == 0) {
        return MatrixCoeffWiseOp<const XprType, internals::identity_op>(derived(), internals::identity_op());
    }
    constexpr auto cwise() && requires(XprType::NestAsRef == 0) {
        return MatrixCoeffWiseOp<XprType, internals::identity_op>(derived(), internals::identity_op());
    }
    constexpr void cwise() const && requires(XprType::NestAsRef != 0) = delete;
    constexpr void cwise() && requires(XprType::NestAsRef != 0) = delete;

    // redux operators
    // frobenius norm (squared L^2 norm)
    constexpr auto squared_norm() const {
        using Scalar = typename XprType::Scalar;
        Scalar norm_ = 0;
        for (int i = 0; i < derived().rows(); ++i) {
            for (int j = 0; j < derived().cols(); ++j) {
                const Scalar value = static_cast<Scalar>(derived()(i, j));
                norm_ += value * value;
            }
        }
        return norm_;
    }
    constexpr auto norm() const { return fdapde::sqrt(squared_norm()); }
    // maximum norm (L^\infty norm)
    constexpr auto inf_norm() const {
        using Scalar = typename XprType::Scalar;
        Scalar norm_ = Scalar(0);
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
          std::numeric_limits<Scalar>::lowest(), [](const Scalar& tmp, const Scalar& x) { return tmp > x ? tmp : x; });
    }
    constexpr auto min() const {
        using Scalar = typename XprType::Scalar;
        return redux(
          std::numeric_limits<Scalar>::max(), [](const Scalar& tmp, const Scalar& x) { return tmp < x ? tmp : x; });
    }
    // boolean reductions
    // true if at least one of the coefficients of the expression evalutes true
    constexpr bool any() const {
        if (derived().size() == 0) return false;
        return internals::boolean_redux_linear_executor::run(derived(), 1, [](const auto& x) { return  bool(x); });
    }
    // true if none of the coefficients of the expression evaluates false
    constexpr bool all() const {
        if (derived().size() == 0) return true;
        return internals::boolean_redux_linear_executor::run(derived(), 0, [](const auto& x) { return !bool(x); });
    }
    // number of coefficients evaluating true in the expression
    constexpr int count() const {
        if (derived().size() == 0) return int(0);
        return redux(int(0), [](int cnt, auto x) { return cnt + (bool(x) ? 1 : 0); });
    }
    // vector-wise redux operators
    constexpr MatrixRowWiseOp<XprType> rowwise() & { return MatrixRowWiseOp<XprType>(derived()); }
    constexpr MatrixRowWiseOp<const XprType> rowwise() const & { return MatrixRowWiseOp<const XprType>(derived()); }
    constexpr MatrixColWiseOp<XprType> colwise() & { return MatrixColWiseOp<XprType>(derived()); }
    constexpr MatrixColWiseOp<const XprType> colwise() const & { return MatrixColWiseOp<const XprType>(derived()); }
    constexpr void rowwise() const && = delete;
    constexpr void colwise() const && = delete;

    // unary operators
    constexpr TransposeOp<XprType> transpose() const & { return TransposeOp<XprType>(derived()); }
    constexpr TransposeOp<XprType> transpose() const && requires(XprType::NestAsRef == 0) {
        return TransposeOp<XprType>(derived());
    }
    constexpr void transpose() const && requires(XprType::NestAsRef != 0) = delete;
    constexpr Diagonal<XprType> diagonal() & { return Diagonal<XprType>(derived()); }
    constexpr Diagonal<const XprType> diagonal() const & { return Diagonal<const XprType>(derived()); }
    constexpr Diagonal<XprType> diagonal() && requires(XprType::NestAsRef == 0) {
        return Diagonal<XprType>(std::move(derived()));
    }
    constexpr void diagonal() && requires(XprType::NestAsRef != 0) = delete;
    // block accessors
    // static-sized block
    template <int BlockRows, int BlockCols>
    constexpr MatrixBlock<BlockRows, BlockCols, XprType> block(int i, int j) & {
        return MatrixBlock<BlockRows, BlockCols, XprType>(derived(), i, j);
    }
    template <int BlockRows, int BlockCols>
    constexpr MatrixBlock<BlockRows, BlockCols, const XprType> block(int i, int j) const & {
        return MatrixBlock<BlockRows, BlockCols, const XprType>(derived(), i, j);
    }
    // dynamic-sized block
    constexpr MatrixBlock<Dynamic, Dynamic, XprType> block(int i, int j, int rows, int cols) & {
        return MatrixBlock<Dynamic, Dynamic, XprType>(derived(), i, j, rows, cols);
    }
    constexpr MatrixBlock<Dynamic, Dynamic, const XprType> block(int i, int j, int rows, int cols) const & {
        return MatrixBlock<Dynamic, Dynamic, const XprType>(derived(), i, j, rows, cols);
    }
    template <int BlockRows, int BlockCols> constexpr void block(int, int) const && = delete;
    constexpr void block(int, int, int, int) const && = delete;
    // row/col accessors
    constexpr auto col(int i) & { return MatrixBlock<XprType::Rows, 1, XprType>(derived(), i); }
    constexpr auto col(int i) const & { return MatrixBlock<XprType::Rows, 1, const XprType>(derived(), i); }
    constexpr auto row(int i) & { return MatrixBlock<1, XprType::Cols, XprType>(derived(), i); }
    constexpr auto row(int i) const & { return MatrixBlock<1, XprType::Cols, const XprType>(derived(), i); }
    constexpr void col(int) const && = delete;
    constexpr void row(int) const && = delete;
    // other block-type accessors
    template <int BlockRows> constexpr auto top_rows() & { return block<BlockRows, XprType::Cols>(0, 0); }
    template <int BlockRows> constexpr auto top_rows() const & { return block<BlockRows, XprType::Cols>(0, 0); }
    constexpr auto top_rows(int rows) & { return block(0, 0, rows, derived().cols()); }
    constexpr auto top_rows(int rows) const & { return block(0, 0, rows, derived().cols()); }
    template <int BlockRows> constexpr void top_rows() const && = delete;
    constexpr void top_rows(int) const && = delete;

    template <int BlockRows> constexpr auto bottom_rows() & {
        return block<BlockRows, XprType::Cols>(derived().rows() - BlockRows, 0);
    }
    template <int BlockRows> constexpr auto bottom_rows() const & {
        return block<BlockRows, XprType::Cols>(derived().rows() - BlockRows, 0);
    }
    constexpr auto bottom_rows(int rows) & { return block(derived().rows() - rows, 0, rows, derived().cols()); }
    constexpr auto bottom_rows(int rows) const & {
        return block(derived().rows() - rows, 0, rows, derived().cols());
    }
    template <int BlockRows> constexpr void bottom_rows() const && = delete;
    constexpr void bottom_rows(int) const && = delete;

    template <int BlockCols> constexpr auto left_cols() & { return block<XprType::Rows, BlockCols>(0, 0); }
    template <int BlockCols> constexpr auto left_cols() const & { return block<XprType::Rows, BlockCols>(0, 0); }
    constexpr auto left_cols(int cols) & { return block(0, 0, derived().rows(), cols); }
    constexpr auto left_cols(int cols) const & { return block(0, 0, derived().rows(), cols); }
    template <int BlockCols> constexpr void left_cols() const && = delete;
    constexpr void left_cols(int) const && = delete;

    template <int BlockCols> constexpr auto right_cols() & {
        return block<XprType::Rows, BlockCols>(0, derived().cols() - BlockCols);
    }
    template <int BlockCols> constexpr auto right_cols() const & {
        return block<XprType::Rows, BlockCols>(0, derived().cols() - BlockCols);
    }
    constexpr auto right_cols(int cols) & { return block(0, derived().cols() - cols, derived().rows(), cols); }
    constexpr auto right_cols(int cols) const & {
        return block(0, derived().cols() - cols, derived().rows(), cols);
    }
    template <int BlockCols> constexpr void right_cols() const && = delete;
    constexpr void right_cols(int) const && = delete;

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
    template <typename RhsXprType> constexpr auto cross(const MatrixExpr<RhsXprType>& rhs) const & {
        return MatrixCrossProductOp<XprType, RhsXprType>(derived(), rhs.derived());
    }
    template <internals::matrix_expression RhsXprType>
        requires(internals::is_owning_rvalue_expression_v<RhsXprType&&>)
    constexpr void cross(RhsXprType&&) const & = delete;
    template <typename RhsXprType> constexpr void cross(const MatrixExpr<RhsXprType>&) const && = delete;

    // reshaping
    // static-sized
    template <int ReshapedRows_, int ReshapedCols_> constexpr auto reshape() & {
        return ReshapeOp<ReshapedRows_, ReshapedCols_, XprType>(derived());
    }
    template <int ReshapedRows_, int ReshapedCols_> constexpr auto reshape() const & {
        return ReshapeOp<ReshapedRows_, ReshapedCols_, const XprType>(derived());
    }
    template <int ReshapedRows_> constexpr auto reshape() & { return ReshapeOp<ReshapedRows_, 1, XprType>(derived()); }
    template <int ReshapedRows_> constexpr auto reshape() const & {
        return ReshapeOp<ReshapedRows_, 1, const XprType>(derived());
    }
    // dynamic-sized
    constexpr auto reshape(int rows, int cols) & {
        return ReshapeOp<Dynamic, Dynamic, XprType>(derived(), rows, cols);
    }
    constexpr auto reshape(int rows, int cols) const & {
        return ReshapeOp<Dynamic, Dynamic, const XprType>(derived(), rows, cols);
    }
    constexpr auto reshape(int rows) & { return ReshapeOp<Dynamic, 1, XprType>(derived(), rows); }
    constexpr auto reshape(int rows) const & { return ReshapeOp<Dynamic, 1, const XprType>(derived(), rows); }
    template <int ReshapedRows_, int ReshapedCols_> constexpr void reshape() const && = delete;
    template <int ReshapedRows_> constexpr void reshape() const && = delete;
    constexpr void reshape(int, int) const && = delete;
    constexpr void reshape(int) const && = delete;

    // structured views and casts
    template <int ViewMode> constexpr Triangular<ViewMode, XprType> triangular_block() & {
        return Triangular<ViewMode, XprType>(derived());
    }
    template <int ViewMode> constexpr Triangular<ViewMode, const XprType> triangular_block() const & {
        return Triangular<ViewMode, const XprType>(derived());
    }
    template <int ViewMode>
    constexpr Triangular<ViewMode, XprType> triangular_block() && requires(XprType::NestAsRef == 0) {
        return Triangular<ViewMode, XprType>(std::move(derived()));
    }
    template <int ViewMode> constexpr void triangular_block() && requires(XprType::NestAsRef != 0) = delete;

    template <int ViewMode> constexpr auto as_symmetric() & {
        return internals::symmetric_cast<ViewMode>(derived());
    }
    template <int ViewMode> constexpr auto as_symmetric() const & {
        return internals::symmetric_cast<ViewMode>(derived());
    }
    template <int ViewMode> constexpr auto as_symmetric() && requires(XprType::NestAsRef == 0) {
        return internals::symmetric_cast<ViewMode>(std::move(derived()));
    }
    template <int ViewMode> constexpr void as_symmetric() && requires(XprType::NestAsRef != 0) = delete;

    constexpr auto as_diagonal() & {
        fdapde_static_assert(XprType::Rows == 1 || XprType::Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return internals::diagonal_cast(derived());
    }
    constexpr auto as_diagonal() const & {
        fdapde_static_assert(XprType::Rows == 1 || XprType::Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return internals::diagonal_cast(derived());
    }
    constexpr auto as_diagonal() && requires(XprType::NestAsRef == 0) {
        fdapde_static_assert(XprType::Rows == 1 || XprType::Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return internals::diagonal_cast(std::move(derived()));
    }
    constexpr void as_diagonal() && requires(XprType::NestAsRef != 0) = delete;

    // square matrix methods
    constexpr auto symm_part() const & {
        constexpr int Rows = XprType::Rows, Cols = XprType::Cols;
        fdapde_static_assert(
          Rows == Dynamic || Cols == Dynamic || Rows == Cols, THIS_METHODS_IS_FOR_SQUARE_MATRICES_ONLY);
        if constexpr (Rows == Dynamic || Cols == Dynamic) {
            fdapde_assert(derived().rows() == derived().cols());
        }
        return 0.5 * (derived() + derived().transpose());   // symmetric part
    }
    constexpr auto skew_part() const & {
        constexpr int Rows = XprType::Rows, Cols = XprType::Cols;
        fdapde_static_assert(
          Rows == Dynamic || Cols == Dynamic || Rows == Cols, THIS_METHODS_IS_FOR_SQUARE_MATRICES_ONLY);
        if constexpr (Rows == Dynamic || Cols == Dynamic) {
            fdapde_assert(derived().rows() == derived().cols());
        }
        return 0.5 * (derived() - derived().transpose());   // skew-symmetric part
    }
    constexpr void symm_part() const && = delete;
    constexpr void skew_part() const && = delete;
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
            if (!(fdapde::abs(a - b) < epsilon ||
                  fdapde::abs(a - b) <
                    ((fdapde::abs(a) < fdapde::abs(b) ? fdapde::abs(b) : fdapde::abs(a)) * epsilon))) {
                return false;
            }
        }
    }
    return true;
}

// detection trait
template <typename XprType> struct is_matrix {
    using CleanXprType = std::remove_cvref_t<XprType>;
    static constexpr bool value = std::is_base_of_v<MatrixExpr<CleanXprType>, CleanXprType>;
};
template <typename XprType> static constexpr bool is_matrix_v = is_matrix<XprType>::value;
template <typename XprType> struct is_vector {
    using CleanXprType = std::remove_cvref_t<XprType>;
    static constexpr bool value = [] {
        if constexpr (is_matrix_v<CleanXprType>) {
            return CleanXprType::Cols == 1 || CleanXprType::Rows == 1;
        } else {
            return false;
        }
    }();
};
template <typename XprType> static constexpr bool is_vector_v = is_vector<XprType>::value;

}   // namespace fdapde::linalg

#endif // __FDAPDE_LINALG_XPR_H__
