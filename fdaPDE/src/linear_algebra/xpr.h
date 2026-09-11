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

// matrixExpr type-system base class
/// @brief provides the dense matrix expression interface
template <typename XprType_> struct MatrixExpr {
    using XprType = XprType_;

    // assignment
    /// @brief materializes the source before assignment, resizing dynamic owners and preserving aliases
    template <typename RhsXprType_>
        requires(XprType::ReadOnly == 0 || internals::is_mutable_matrix_view_v<XprType>)
    constexpr XprType& operator=(const MatrixExpr<RhsXprType_>& rhs) & {
        using RhsXprType = std::decay_t<RhsXprType_>;
        using RhsScalar = std::remove_cv_t<typename RhsXprType::Scalar>;
        Matrix<RhsScalar, RhsXprType::Rows, RhsXprType::Cols, RhsXprType::StorageOrder> tmp(rhs);
        using executor = typename XprType::assignment_executor;
        constexpr int Rows = XprType::Rows;
        constexpr int Cols = XprType::Cols;
        if constexpr (requires(XprType_ xpr, int i, int j) {
                          xpr.resize(i, j);
                      } && (Rows == Dynamic || Cols == Dynamic)) {
            if (derived().rows() != tmp.rows() || derived().cols() != tmp.cols()) {
                derived().resize(tmp.rows(), tmp.cols());
            }
        }
        executor::run(derived(), tmp, [](auto& l, const auto& r) { l = r; });
        return derived();
    }
    /// @brief assigns a source snapshot and returns the temporary view or expression by value
    template <typename RhsXprType_>
      constexpr XprType operator=(const MatrixExpr<RhsXprType_>& rhs) &&
      requires((XprType::NestAsRef == 0 && XprType::ReadOnly == 0) || internals::is_mutable_matrix_view_v<XprType>) {
          static_cast<MatrixExpr&>(*this).operator=(rhs);
          return derived();
      }
      /// @brief rejects assignment to a temporary owner
      template <typename RhsXprType_>
      constexpr void operator=(const MatrixExpr<RhsXprType_>&) && requires(XprType::NestAsRef != 0) = delete;
    /// @brief evaluates coefficientwise values with snapshot-based matrix assignment
    template <typename RhsXprType_>
    constexpr XprType& operator=(const MatrixCoeffWiseExpr<RhsXprType_>& rhs) &
        requires(XprType::ReadOnly == 0 || internals::is_mutable_matrix_view_v<XprType>)
    {
        operator=(rhs.mwise());
        return derived();
    }
    /// @brief assigns coefficientwise values and returns the temporary view or expression by value
    template <typename RhsXprType_>
      constexpr XprType operator=(const MatrixCoeffWiseExpr<RhsXprType_>& rhs) &&
      requires((XprType::NestAsRef == 0 && XprType::ReadOnly == 0) || internals::is_mutable_matrix_view_v<XprType>) {
          static_cast<MatrixExpr&>(*this).operator=(rhs);
          return derived();
      }
      /// @brief rejects coefficientwise assignment to a temporary owner
      template <typename RhsXprType_>
      constexpr void operator=(const MatrixCoeffWiseExpr<RhsXprType_>&) && requires(XprType::NestAsRef != 0) = delete;
    // compound algebra
    /// @brief adds a source snapshot to corresponding destination coefficients
    template <typename RhsXprType_>
        requires(XprType::ReadOnly == 0 || internals::is_mutable_matrix_view_v<XprType>)
    constexpr XprType& operator+=(const MatrixExpr<RhsXprType_>& rhs) & {
        using RhsXprType = std::decay_t<RhsXprType_>;
        using RhsScalar = std::remove_cv_t<typename RhsXprType::Scalar>;
        Matrix<RhsScalar, RhsXprType::Rows, RhsXprType::Cols, RhsXprType::StorageOrder> tmp(rhs);
        using executor = typename XprType::assignment_executor;
        executor::run(derived(), tmp, [](auto& l, const auto& r) { l += r; });
        return derived();
    }
    /// @brief adds a source snapshot to corresponding destination coefficients and returns the temporary adaptor by
    /// value
    template <typename RhsXprType_>
      constexpr XprType operator+=(const MatrixExpr<RhsXprType_>& rhs) &&
      requires((XprType::NestAsRef == 0 && XprType::ReadOnly == 0) || internals::is_mutable_matrix_view_v<XprType>) {
          static_cast<MatrixExpr&>(*this).operator+=(rhs);
          return derived();
      }
      /// @brief rejects compound assignment to a temporary owner
      template <typename RhsXprType_>
      constexpr void operator+=(const MatrixExpr<RhsXprType_>&) && requires(XprType::NestAsRef != 0) = delete;
    /// @brief subtracts a source snapshot from corresponding destination coefficients
    template <typename RhsXprType_>
        requires(XprType::ReadOnly == 0 || internals::is_mutable_matrix_view_v<XprType>)
    constexpr XprType& operator-=(const MatrixExpr<RhsXprType_>& rhs) & {
        using RhsXprType = std::decay_t<RhsXprType_>;
        using RhsScalar = std::remove_cv_t<typename RhsXprType::Scalar>;
        Matrix<RhsScalar, RhsXprType::Rows, RhsXprType::Cols, RhsXprType::StorageOrder> tmp(rhs);
        using executor = typename XprType::assignment_executor;
        executor::run(derived(), tmp, [](auto& l, const auto& r) { l -= r; });
        return derived();
    }
    /// @brief subtracts a source snapshot from corresponding destination coefficients and returns the temporary adaptor
    /// by value
    template <typename RhsXprType_>
      constexpr XprType operator-=(const MatrixExpr<RhsXprType_>& rhs) &&
      requires((XprType::NestAsRef == 0 && XprType::ReadOnly == 0) || internals::is_mutable_matrix_view_v<XprType>) {
          static_cast<MatrixExpr&>(*this).operator-=(rhs);
          return derived();
      }
      /// @brief rejects compound assignment to a temporary owner
      template <typename RhsXprType_>
      constexpr void operator-=(const MatrixExpr<RhsXprType_>&) && requires(XprType::NestAsRef != 0) = delete;
    /// @brief multiplies every writable coefficient by the scalar
    template <typename Scalar_>
        requires(
          std::is_arithmetic_v<Scalar_> && (XprType::ReadOnly == 0 || internals::is_mutable_matrix_view_v<XprType>))
    constexpr XprType& operator*=(Scalar_ rhs) & {
        using executor = typename XprType::assignment_executor;
        executor::run(derived(), derived(), [rhs](auto& l, const auto& r) { l = rhs * r; });
        return derived();
    }
    /// @brief multiplies every writable coefficient by the scalar and returns the temporary adaptor by value
    template <typename Scalar_>
        requires(
          std::is_arithmetic_v<Scalar_> &&
          ((XprType::NestAsRef == 0 && XprType::ReadOnly == 0) || internals::is_mutable_matrix_view_v<XprType>))
    constexpr XprType operator*=(Scalar_ rhs) && {
        static_cast<MatrixExpr&>(*this).operator*=(rhs);
        return derived();
    }
    /// @brief rejects compound assignment to a temporary owner
    template <typename Scalar_>
        requires(std::is_arithmetic_v<Scalar_> && XprType::NestAsRef != 0)
    constexpr void operator*=(Scalar_) && = delete;
    /// @brief divides every writable coefficient by the scalar
    template <typename Scalar_>
        requires(
          std::is_arithmetic_v<Scalar_> && (XprType::ReadOnly == 0 || internals::is_mutable_matrix_view_v<XprType>))
    constexpr XprType& operator/=(Scalar_ rhs) & {
        using executor = typename XprType::assignment_executor;
        executor::run(derived(), derived(), [rhs](auto& l, const auto& r) { l = r / rhs; });
        return derived();
    }
    /// @brief divides every writable coefficient by the scalar and returns the temporary adaptor by value
    template <typename Scalar_>
        requires(
          std::is_arithmetic_v<Scalar_> &&
          ((XprType::NestAsRef == 0 && XprType::ReadOnly == 0) || internals::is_mutable_matrix_view_v<XprType>))
    constexpr XprType operator/=(Scalar_ rhs) && {
        static_cast<MatrixExpr&>(*this).operator/=(rhs);
        return derived();
    }
    /// @brief rejects compound assignment to a temporary owner
    template <typename Scalar_>
        requires(std::is_arithmetic_v<Scalar_> && XprType::NestAsRef != 0)
    constexpr void operator/=(Scalar_) && = delete;
    // compound matrix multiplication
    /// @brief replaces the destination with its matrix product after materializing aliased operands
    template <typename RhsXprType_>
        requires(XprType::ReadOnly == 0 || internals::is_mutable_matrix_view_v<XprType>)
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
    /// @brief replaces the destination with its matrix product after materializing aliased operands and returns the
    /// temporary adaptor by value
    template <typename RhsXprType_>
      constexpr XprType operator*=(const MatrixExpr<RhsXprType_>& rhs) &&
      requires((XprType::NestAsRef == 0 && XprType::ReadOnly == 0) || internals::is_mutable_matrix_view_v<XprType>) {
          static_cast<MatrixExpr&>(*this).operator*=(rhs);
          return derived();
      }
      /// @brief rejects compound assignment to a temporary owner
      template <typename RhsXprType_>
      constexpr void operator*=(const MatrixExpr<RhsXprType_>&) && requires(XprType::NestAsRef != 0) = delete;

    // observers
    /// @brief returns the row count
    constexpr int rows() const { return XprType::Rows == Dynamic ? derived().rows() : XprType::Rows; }
    /// @brief returns the column count
    constexpr int cols() const { return XprType::Cols == Dynamic ? derived().cols() : XprType::Cols; }
    /// @brief returns the coefficient count
    constexpr int size() const {
        constexpr int Rows = XprType::Rows;
        constexpr int Cols = XprType::Cols;
        return (Rows != Dynamic && Cols != Dynamic) ? Rows * Cols : derived().rows() * derived().cols();
    }
    /// @brief returns the concrete expression
    constexpr const XprType& derived() const& { return static_cast<const XprType&>(*this); }
    /// @brief returns the concrete expression
    constexpr XprType& derived() & { return static_cast<XprType&>(*this); }
    /// @brief rejects returning a reference to state held by a temporary expression
    constexpr void derived() const&& = delete;
    /// @brief rejects returning a reference to state held by a temporary expression
    constexpr void derived() && = delete;
    // ostream
    /// @brief writes logical matrix rows separated by newlines without a trailing newline
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
    /// @brief returns the coefficient-wise algebra adaptor
    constexpr auto cwise() const& {
        return MatrixCoeffWiseOp<const XprType, internals::identity_op>(derived(), internals::identity_op());
    }
    /// @brief returns the coefficient-wise algebra adaptor
    constexpr auto cwise() & {
        return MatrixCoeffWiseOp<XprType, internals::identity_op>(derived(), internals::identity_op());
    }
    /// @brief returns the coefficient-wise algebra adaptor
    constexpr auto cwise() const&&
        requires(XprType::NestAsRef == 0)
    {
        return MatrixCoeffWiseOp<const XprType, internals::identity_op>(
          static_cast<const XprType&&>(*this), internals::identity_op());
    }
    /// @brief returns the coefficient-wise algebra adaptor
    constexpr auto cwise() &&
      requires(XprType::NestAsRef == 0) {
          return MatrixCoeffWiseOp<XprType, internals::identity_op>(
            static_cast<XprType &&>(*this), internals::identity_op());
      }
      /// @brief rejects borrowing a coefficientwise adaptor from a temporary owner
      constexpr void cwise() const&&
          requires(XprType::NestAsRef != 0)
      = delete;
    /// @brief rejects borrowing a coefficientwise adaptor from a temporary owner
    constexpr void cwise() && requires(XprType::NestAsRef != 0) = delete;

    // redux operators
    // frobenius norm (squared L^2 norm)
    /// @brief returns the sum of squared coefficients
    constexpr auto squared_norm() const {
        typename XprType::Scalar norm_ = 0;
        for (int i = 0; i < derived().rows(); ++i) {
            for (int j = 0; j < derived().cols(); ++j) { norm_ += fdapde::pow(derived()(i, j), 2); }
        }
        return norm_;
    }
    /// @brief returns the Euclidean or Frobenius norm
    constexpr auto norm() const
        requires(std::floating_point<std::remove_cv_t<typename XprType::Scalar>>)
    {
        using Scalar = std::remove_cv_t<typename XprType::Scalar>;
        Scalar norm_ = Scalar(0);
        for (int i = 0; i < derived().rows(); ++i) {
            for (int j = 0; j < derived().cols(); ++j) {
                norm_ = internals::scale_safe_hypot(norm_, static_cast<Scalar>(derived()(i, j)));
            }
        }
        return norm_;
    }
    // maximum norm (L^\infty norm)
    /// @brief returns the maximum absolute coefficient
    constexpr auto inf_norm() const {
        using Scalar = std::remove_cv_t<typename XprType::Scalar>;
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
    /// @brief reduces coefficients using the supplied callable
    template <typename Scalar_, typename ReduxOp> constexpr auto redux(Scalar_ init, ReduxOp&& op) const {
        fdapde_assert(
          derived().rows() > 0 && derived().cols() > 0, std::invalid_argument, "reduction requires a nonempty matrix");
        return internals::matrix_redux_linear_executor::run(derived(), init, op);
    }
    /// @brief returns the sum of coefficients
    constexpr auto sum() const {
        using Scalar = typename XprType::Scalar;
        if (derived().size() == 0) return Scalar(0);
        return redux(Scalar(0), [](const Scalar& tmp, const Scalar& x) { return tmp + x; });
    }
    /// @brief returns the product of coefficients
    constexpr auto prod() const {
        using Scalar = typename XprType::Scalar;
        if (derived().size() == 0) return Scalar(1);
        return redux(Scalar(1), [](const Scalar& tmp, const Scalar& x) { return tmp * x; });
    }
    /// @brief returns the arithmetic mean of coefficients
    constexpr auto mean() const {
        fdapde_assert(!(derived().size() == 0), std::domain_error, "reduction requires a nonempty matrix");
        return derived().sum() / derived().size();
    }
    /// @brief returns the maximum coefficient
    constexpr auto max() const {
        using Scalar = typename XprType::Scalar;
        fdapde_assert(!(derived().size() == 0), std::domain_error, "max requires a nonempty matrix");
        return redux(
          std::numeric_limits<Scalar>::lowest(), [](const Scalar& tmp, const Scalar& x) { return tmp > x ? tmp : x; });
    }
    /// @brief returns the minimum coefficient
    constexpr auto min() const {
        using Scalar = typename XprType::Scalar;
        fdapde_assert(!(derived().size() == 0), std::domain_error, "min requires a nonempty matrix");
        return redux(
          std::numeric_limits<Scalar>::max(), [](const Scalar& tmp, const Scalar& x) { return tmp < x ? tmp : x; });
    }
    // boolean reductions
    // true if at least one of the coefficients of the expression evalutes true
    /// @brief reports whether any coefficient is true
    constexpr bool any() const {
        return internals::boolean_redux_linear_executor::run(derived(), 1, [](const auto& x) { return bool(x); });
    }
    // true if none of the coefficients of the expression evaluates false
    /// @brief reports whether every coefficient is true
    constexpr bool all() const {
        return internals::boolean_redux_linear_executor::run(derived(), 0, [](const auto& x) { return !bool(x); });
    }
    // number of coefficients evaluating true in the expression
    /// @brief counts the true coefficients
    constexpr int count() const {
        if (derived().size() == 0) return int(0);
        return redux(int(0), [](int cnt, auto x) { return cnt + (bool(x) ? 1 : 0); });
    }
    // vector-wise redux operators
    /// @brief returns the row-wise reduction adaptor
    constexpr MatrixRowWiseOp<XprType> rowwise() & { return MatrixRowWiseOp<XprType>(derived()); }
    /// @brief returns the row-wise reduction adaptor
    constexpr MatrixRowWiseOp<const XprType> rowwise() const& { return MatrixRowWiseOp<const XprType>(derived()); }
    /// @brief returns the row-wise reduction adaptor
    constexpr MatrixRowWiseOp<XprType> rowwise() &&
      requires(XprType::NestAsRef == 0) { return MatrixRowWiseOp<XprType>(static_cast<XprType &&>(*this)); }
      /// @brief returns the row-wise reduction adaptor
      constexpr MatrixRowWiseOp<const XprType> rowwise() const&&
          requires(XprType::NestAsRef == 0)
    {
        return MatrixRowWiseOp<const XprType>(static_cast<const XprType&&>(*this));
    }
    /// @brief rejects borrowing a rowwise adaptor from a temporary owner
    constexpr void rowwise() && requires(XprType::NestAsRef != 0) = delete;
    /// @brief rejects borrowing a rowwise adaptor from a temporary owner
    constexpr void rowwise() const&&
        requires(XprType::NestAsRef != 0)
    = delete;
    /// @brief returns the column-wise reduction adaptor
    constexpr MatrixColWiseOp<XprType> colwise() & { return MatrixColWiseOp<XprType>(derived()); }
    /// @brief returns the column-wise reduction adaptor
    constexpr MatrixColWiseOp<const XprType> colwise() const& { return MatrixColWiseOp<const XprType>(derived()); }
    /// @brief returns the column-wise reduction adaptor
    constexpr MatrixColWiseOp<XprType> colwise() &&
      requires(XprType::NestAsRef == 0) { return MatrixColWiseOp<XprType>(static_cast<XprType &&>(*this)); }
      /// @brief returns the column-wise reduction adaptor
      constexpr MatrixColWiseOp<const XprType> colwise() const&&
          requires(XprType::NestAsRef == 0)
    {
        return MatrixColWiseOp<const XprType>(static_cast<const XprType&&>(*this));
    }
    /// @brief rejects borrowing a columnwise adaptor from a temporary owner
    constexpr void colwise() && requires(XprType::NestAsRef != 0) = delete;
    /// @brief rejects borrowing a columnwise adaptor from a temporary owner
    constexpr void colwise() const&&
        requires(XprType::NestAsRef != 0)
    = delete;

    // unary operators
    /// @brief returns the transposed expression
    constexpr TransposeOp<XprType> transpose() const& { return TransposeOp<XprType>(derived()); }
    /// @brief returns the transposed expression
    constexpr TransposeOp<XprType> transpose() const&&
        requires(XprType::NestAsRef == 0)
    {
        return TransposeOp<XprType>(derived());
    }
    /// @brief rejects borrowing a transpose from a temporary owner
    constexpr void transpose() const&&
        requires(XprType::NestAsRef != 0)
    = delete;
    /// @brief returns the main diagonal representation
    constexpr Diagonal<XprType> diagonal() & { return Diagonal<XprType>(derived()); }
    /// @brief returns the main diagonal representation
    constexpr Diagonal<const XprType> diagonal() const& { return Diagonal<const XprType>(derived()); }
    /// @brief returns the main diagonal representation
    constexpr Diagonal<XprType> diagonal() &&
      requires(XprType::NestAsRef == 0) { return Diagonal<XprType>(std::move(derived())); }
      /// @brief returns the main diagonal representation
      constexpr Diagonal<const XprType> diagonal() const&&
          requires(XprType::NestAsRef == 0)
    {
        return Diagonal<const XprType>(std::move(derived()));
    }
    /// @brief rejects borrowing a diagonal view from a temporary owner
    constexpr void diagonal() && requires(XprType::NestAsRef != 0) = delete;
    /// @brief rejects borrowing a diagonal view from a temporary owner
    constexpr void diagonal() const&&
        requires(XprType::NestAsRef != 0)
    = delete;
    // block accessors
    // static-sized block
    /// @brief returns a view of the requested rectangular region
    template <int BlockRows, int BlockCols> constexpr MatrixBlock<BlockRows, BlockCols, XprType> block(int i, int j) & {
        return MatrixBlock<BlockRows, BlockCols, XprType>(derived(), i, j);
    }
    /// @brief returns a view of the requested rectangular region
    template <int BlockRows, int BlockCols>
    constexpr MatrixBlock<BlockRows, BlockCols, const XprType> block(int i, int j) const& {
        return MatrixBlock<BlockRows, BlockCols, const XprType>(derived(), i, j);
    }
    /// @brief returns a view of the requested rectangular region
    template <int BlockRows, int BlockCols>
      constexpr MatrixBlock<BlockRows, BlockCols, XprType> block(int i, int j) &&
      requires(XprType::NestAsRef == 0) {
          return MatrixBlock<BlockRows, BlockCols, XprType>(static_cast<XprType&>(*this), i, j);
      }
      /// @brief returns a view of the requested rectangular region
      template <int BlockRows, int BlockCols>
      constexpr MatrixBlock<BlockRows, BlockCols, const XprType> block(int i, int j) const&&
          requires(XprType::NestAsRef == 0)
    {
        return MatrixBlock<BlockRows, BlockCols, const XprType>(static_cast<const XprType&>(*this), i, j);
    }
    /// @brief rejects borrowing a block from a temporary owner
    template <int BlockRows, int BlockCols>
      constexpr void block(int, int) && requires(XprType::NestAsRef != 0) = delete;
    /// @brief rejects borrowing a block from a temporary owner
    template <int BlockRows, int BlockCols>
    constexpr void block(int, int) const&&
        requires(XprType::NestAsRef != 0)
    = delete;
    // dynamic-sized block
    /// @brief returns a view of the requested rectangular region
    constexpr MatrixBlock<Dynamic, Dynamic, XprType> block(int i, int j, int rows, int cols) & {
        return MatrixBlock<Dynamic, Dynamic, XprType>(derived(), i, j, rows, cols);
    }
    /// @brief returns a view of the requested rectangular region
    constexpr MatrixBlock<Dynamic, Dynamic, const XprType> block(int i, int j, int rows, int cols) const& {
        return MatrixBlock<Dynamic, Dynamic, const XprType>(derived(), i, j, rows, cols);
    }
    /// @brief returns a view of the requested rectangular region
    constexpr MatrixBlock<Dynamic, Dynamic, XprType> block(int i, int j, int rows, int cols) &&
      requires(XprType::NestAsRef == 0) {
          return MatrixBlock<Dynamic, Dynamic, XprType>(static_cast<XprType&>(*this), i, j, rows, cols);
      }
      /// @brief returns a view of the requested rectangular region
      constexpr MatrixBlock<Dynamic, Dynamic, const XprType> block(int i, int j, int rows, int cols) const&&
          requires(XprType::NestAsRef == 0)
    {
        return MatrixBlock<Dynamic, Dynamic, const XprType>(static_cast<const XprType&>(*this), i, j, rows, cols);
    }
    /// @brief rejects borrowing a block from a temporary owner
    constexpr void block(int, int, int, int) && requires(XprType::NestAsRef != 0) = delete;
    /// @brief rejects borrowing a block from a temporary owner
    constexpr void block(int, int, int, int) const&&
        requires(XprType::NestAsRef != 0)
    = delete;
    // row/col accessors
    /// @brief returns a view of the requested column
    constexpr auto col(int i) & { return MatrixBlock<XprType::Rows, 1, XprType>(derived(), i); }
    /// @brief returns a view of the requested column
    constexpr auto col(int i) const& { return MatrixBlock<XprType::Rows, 1, const XprType>(derived(), i); }
    /// @brief returns a view of the requested column
    constexpr auto col(int i) &&
      requires(XprType::NestAsRef == 0) {
          return MatrixBlock<XprType::Rows, 1, XprType>(static_cast<XprType&>(*this), i);
      }
      /// @brief returns a view of the requested column
      constexpr auto col(int i) const&&
          requires(XprType::NestAsRef == 0)
    {
        return MatrixBlock<XprType::Rows, 1, const XprType>(static_cast<const XprType&>(*this), i);
    }
    /// @brief rejects borrowing a column view from a temporary owner
    constexpr void col(int) && requires(XprType::NestAsRef != 0) = delete;
    /// @brief rejects borrowing a column view from a temporary owner
    constexpr void col(int) const&&
        requires(XprType::NestAsRef != 0)
    = delete;
    /// @brief returns a view of the requested row
    constexpr auto row(int i) & { return MatrixBlock<1, XprType::Cols, XprType>(derived(), i); }
    /// @brief returns a view of the requested row
    constexpr auto row(int i) const& { return MatrixBlock<1, XprType::Cols, const XprType>(derived(), i); }
    /// @brief returns a view of the requested row
    constexpr auto row(int i) &&
      requires(XprType::NestAsRef == 0) {
          return MatrixBlock<1, XprType::Cols, XprType>(static_cast<XprType&>(*this), i);
      }
      /// @brief returns a view of the requested row
      constexpr auto row(int i) const&&
          requires(XprType::NestAsRef == 0)
    {
        return MatrixBlock<1, XprType::Cols, const XprType>(static_cast<const XprType&>(*this), i);
    }
    /// @brief rejects borrowing a row view from a temporary owner
    constexpr void row(int) && requires(XprType::NestAsRef != 0) = delete;
    /// @brief rejects borrowing a row view from a temporary owner
    constexpr void row(int) const&&
        requires(XprType::NestAsRef != 0)
    = delete;
    // other block-type accessors
    /// @brief returns the requested top rows view
    template <int BlockRows> constexpr auto top_rows() & { return block<BlockRows, XprType::Cols>(0, 0); }
    /// @brief returns the requested top rows view
    template <int BlockRows> constexpr auto top_rows() const& { return block<BlockRows, XprType::Cols>(0, 0); }
    /// @brief returns the requested top rows view
    template <int BlockRows>
      constexpr auto top_rows() &&
      requires(XprType::NestAsRef == 0) {
          return MatrixBlock<BlockRows, XprType::Cols, XprType>(static_cast<XprType&>(*this), 0, 0);
      }
      /// @brief returns the requested top rows view
      template <int BlockRows>
      constexpr auto top_rows() const&&
          requires(XprType::NestAsRef == 0)
    {
        return MatrixBlock<BlockRows, XprType::Cols, const XprType>(static_cast<const XprType&>(*this), 0, 0);
    }
    /// @brief rejects borrowing a top-row view from a temporary owner
    template <int BlockRows> constexpr void top_rows() && requires(XprType::NestAsRef != 0) = delete;
    /// @brief rejects borrowing a top-row view from a temporary owner
    template <int BlockRows>
    constexpr void top_rows() const&&
        requires(XprType::NestAsRef != 0)
    = delete;
    /// @brief returns the requested top rows view
    constexpr auto top_rows(int rows) & { return block(0, 0, rows, derived().cols()); }
    /// @brief returns the requested top rows view
    constexpr auto top_rows(int rows) const& { return block(0, 0, rows, derived().cols()); }
    /// @brief returns the requested top rows view
    constexpr auto top_rows(int rows) &&
      requires(XprType::NestAsRef == 0) {
          auto& xpr = static_cast<XprType&>(*this);
          return MatrixBlock<Dynamic, Dynamic, XprType>(xpr, 0, 0, rows, xpr.cols());
      }
      /// @brief returns the requested top rows view
      constexpr auto top_rows(int rows) const&&
          requires(XprType::NestAsRef == 0)
    {
        const auto& xpr = static_cast<const XprType&>(*this);
        return MatrixBlock<Dynamic, Dynamic, const XprType>(xpr, 0, 0, rows, xpr.cols());
    }
    /// @brief rejects borrowing a top-row view from a temporary owner
    constexpr void top_rows(int) && requires(XprType::NestAsRef != 0) = delete;
    /// @brief rejects borrowing a top-row view from a temporary owner
    constexpr void top_rows(int) const&&
        requires(XprType::NestAsRef != 0)
    = delete;

    /// @brief returns the requested bottom rows view
    template <int BlockRows> constexpr auto bottom_rows() & {
        return block<BlockRows, XprType::Cols>(derived().rows() - BlockRows, 0);
    }
    /// @brief returns the requested bottom rows view
    template <int BlockRows> constexpr auto bottom_rows() const& {
        return block<BlockRows, XprType::Cols>(derived().rows() - BlockRows, 0);
    }
    /// @brief returns the requested bottom rows view
    template <int BlockRows>
      constexpr auto bottom_rows() &&
      requires(XprType::NestAsRef == 0) {
          auto& xpr = static_cast<XprType&>(*this);
          return MatrixBlock<BlockRows, XprType::Cols, XprType>(xpr, xpr.rows() - BlockRows, 0);
      }
      /// @brief returns the requested bottom rows view
      template <int BlockRows>
      constexpr auto bottom_rows() const&&
          requires(XprType::NestAsRef == 0)
    {
        const auto& xpr = static_cast<const XprType&>(*this);
        return MatrixBlock<BlockRows, XprType::Cols, const XprType>(xpr, xpr.rows() - BlockRows, 0);
    }
    /// @brief rejects borrowing a bottom-row view from a temporary owner
    template <int BlockRows> constexpr void bottom_rows() && requires(XprType::NestAsRef != 0) = delete;
    /// @brief rejects borrowing a bottom-row view from a temporary owner
    template <int BlockRows>
    constexpr void bottom_rows() const&&
        requires(XprType::NestAsRef != 0)
    = delete;
    /// @brief returns the requested bottom rows view
    constexpr auto bottom_rows(int rows) & {
        const int xpr_rows = derived().rows();
        fdapde_assert(!(rows <= 0), std::invalid_argument, "bottom row count must be positive");
        fdapde_assert(!(rows > xpr_rows), std::out_of_range, "bottom rows exceed expression bounds");
        return block(xpr_rows - rows, 0, rows, derived().cols());
    }
    /// @brief returns the requested bottom rows view
    constexpr auto bottom_rows(int rows) const& {
        const int xpr_rows = derived().rows();
        fdapde_assert(!(rows <= 0), std::invalid_argument, "bottom row count must be positive");
        fdapde_assert(!(rows > xpr_rows), std::out_of_range, "bottom rows exceed expression bounds");
        return block(xpr_rows - rows, 0, rows, derived().cols());
    }
    /// @brief returns the requested bottom rows view
    constexpr auto bottom_rows(int rows) &&
      requires(XprType::NestAsRef == 0) {
          auto& xpr = static_cast<XprType&>(*this);
          const int xpr_rows = xpr.rows();
          fdapde_assert(!(rows <= 0), std::invalid_argument, "bottom row count must be positive");
          fdapde_assert(!(rows > xpr_rows), std::out_of_range, "bottom rows exceed expression bounds");
          return MatrixBlock<Dynamic, Dynamic, XprType>(xpr, xpr_rows - rows, 0, rows, xpr.cols());
      }
      /// @brief returns the requested bottom rows view
      constexpr auto bottom_rows(int rows) const&&
          requires(XprType::NestAsRef == 0)
    {
        const auto& xpr = static_cast<const XprType&>(*this);
        const int xpr_rows = xpr.rows();
        fdapde_assert(!(rows <= 0), std::invalid_argument, "bottom row count must be positive");
        fdapde_assert(!(rows > xpr_rows), std::out_of_range, "bottom rows exceed expression bounds");
        return MatrixBlock<Dynamic, Dynamic, const XprType>(xpr, xpr_rows - rows, 0, rows, xpr.cols());
    }
    /// @brief rejects borrowing a bottom-row view from a temporary owner
    constexpr void bottom_rows(int) && requires(XprType::NestAsRef != 0) = delete;
    /// @brief rejects borrowing a bottom-row view from a temporary owner
    constexpr void bottom_rows(int) const&&
        requires(XprType::NestAsRef != 0)
    = delete;

    /// @brief returns the requested left cols view
    template <int BlockCols> constexpr auto left_cols() & { return block<XprType::Rows, BlockCols>(0, 0); }
    /// @brief returns the requested left cols view
    template <int BlockCols> constexpr auto left_cols() const& { return block<XprType::Rows, BlockCols>(0, 0); }
    /// @brief returns the requested left cols view
    template <int BlockCols>
      constexpr auto left_cols() &&
      requires(XprType::NestAsRef == 0) {
          return MatrixBlock<XprType::Rows, BlockCols, XprType>(static_cast<XprType&>(*this), 0, 0);
      }
      /// @brief returns the requested left cols view
      template <int BlockCols>
      constexpr auto left_cols() const&&
          requires(XprType::NestAsRef == 0)
    {
        return MatrixBlock<XprType::Rows, BlockCols, const XprType>(static_cast<const XprType&>(*this), 0, 0);
    }
    /// @brief rejects borrowing a left-column view from a temporary owner
    template <int BlockCols> constexpr void left_cols() && requires(XprType::NestAsRef != 0) = delete;
    /// @brief rejects borrowing a left-column view from a temporary owner
    template <int BlockCols>
    constexpr void left_cols() const&&
        requires(XprType::NestAsRef != 0)
    = delete;
    /// @brief returns the requested left cols view
    constexpr auto left_cols(int cols) & { return block(0, 0, derived().rows(), cols); }
    /// @brief returns the requested left cols view
    constexpr auto left_cols(int cols) const& { return block(0, 0, derived().rows(), cols); }
    /// @brief returns the requested left cols view
    constexpr auto left_cols(int cols) &&
      requires(XprType::NestAsRef == 0) {
          auto& xpr = static_cast<XprType&>(*this);
          return MatrixBlock<Dynamic, Dynamic, XprType>(xpr, 0, 0, xpr.rows(), cols);
      }
      /// @brief returns the requested left cols view
      constexpr auto left_cols(int cols) const&&
          requires(XprType::NestAsRef == 0)
    {
        const auto& xpr = static_cast<const XprType&>(*this);
        return MatrixBlock<Dynamic, Dynamic, const XprType>(xpr, 0, 0, xpr.rows(), cols);
    }
    /// @brief rejects borrowing a left-column view from a temporary owner
    constexpr void left_cols(int) && requires(XprType::NestAsRef != 0) = delete;
    /// @brief rejects borrowing a left-column view from a temporary owner
    constexpr void left_cols(int) const&&
        requires(XprType::NestAsRef != 0)
    = delete;

    /// @brief returns the requested right cols view
    template <int BlockCols> constexpr auto right_cols() & {
        return block<XprType::Rows, BlockCols>(0, derived().cols() - BlockCols);
    }
    /// @brief returns the requested right cols view
    template <int BlockCols> constexpr auto right_cols() const& {
        return block<XprType::Rows, BlockCols>(0, derived().cols() - BlockCols);
    }
    /// @brief returns the requested right cols view
    template <int BlockCols>
      constexpr auto right_cols() &&
      requires(XprType::NestAsRef == 0) {
          auto& xpr = static_cast<XprType&>(*this);
          return MatrixBlock<XprType::Rows, BlockCols, XprType>(xpr, 0, xpr.cols() - BlockCols);
      }
      /// @brief returns the requested right cols view
      template <int BlockCols>
      constexpr auto right_cols() const&&
          requires(XprType::NestAsRef == 0)
    {
        const auto& xpr = static_cast<const XprType&>(*this);
        return MatrixBlock<XprType::Rows, BlockCols, const XprType>(xpr, 0, xpr.cols() - BlockCols);
    }
    /// @brief rejects borrowing a right-column view from a temporary owner
    template <int BlockCols> constexpr void right_cols() && requires(XprType::NestAsRef != 0) = delete;
    /// @brief rejects borrowing a right-column view from a temporary owner
    template <int BlockCols>
    constexpr void right_cols() const&&
        requires(XprType::NestAsRef != 0)
    = delete;
    /// @brief returns the requested right cols view
    constexpr auto right_cols(int cols) & {
        const int xpr_cols = derived().cols();
        fdapde_assert(!(cols <= 0), std::invalid_argument, "right column count must be positive");
        fdapde_assert(!(cols > xpr_cols), std::out_of_range, "right columns exceed expression bounds");
        return block(0, xpr_cols - cols, derived().rows(), cols);
    }
    /// @brief returns the requested right cols view
    constexpr auto right_cols(int cols) const& {
        const int xpr_cols = derived().cols();
        fdapde_assert(!(cols <= 0), std::invalid_argument, "right column count must be positive");
        fdapde_assert(!(cols > xpr_cols), std::out_of_range, "right columns exceed expression bounds");
        return block(0, xpr_cols - cols, derived().rows(), cols);
    }
    /// @brief returns the requested right cols view
    constexpr auto right_cols(int cols) &&
      requires(XprType::NestAsRef == 0) {
          auto& xpr = static_cast<XprType&>(*this);
          const int xpr_cols = xpr.cols();
          fdapde_assert(!(cols <= 0), std::invalid_argument, "right column count must be positive");
          fdapde_assert(!(cols > xpr_cols), std::out_of_range, "right columns exceed expression bounds");
          return MatrixBlock<Dynamic, Dynamic, XprType>(xpr, 0, xpr_cols - cols, xpr.rows(), cols);
      }
      /// @brief returns the requested right cols view
      constexpr auto right_cols(int cols) const&&
          requires(XprType::NestAsRef == 0)
    {
        const auto& xpr = static_cast<const XprType&>(*this);
        const int xpr_cols = xpr.cols();
        fdapde_assert(!(cols <= 0), std::invalid_argument, "right column count must be positive");
        fdapde_assert(!(cols > xpr_cols), std::out_of_range, "right columns exceed expression bounds");
        return MatrixBlock<Dynamic, Dynamic, const XprType>(xpr, 0, xpr_cols - cols, xpr.rows(), cols);
    }
    /// @brief rejects borrowing a right-column view from a temporary owner
    constexpr void right_cols(int) && requires(XprType::NestAsRef != 0) = delete;
    /// @brief rejects borrowing a right-column view from a temporary owner
    constexpr void right_cols(int) const&&
        requires(XprType::NestAsRef != 0)
    = delete;

    // // dot product
    /// @brief returns the vector inner product
    template <typename RhsXprType> constexpr auto dot(const MatrixExpr<RhsXprType>& rhs) const {
        constexpr int RhsRows = RhsXprType::Rows, Rows = XprType::Rows;
        constexpr int RhsCols = RhsXprType::Cols, Cols = XprType::Cols;
        fdapde_static_assert(
          (RhsRows == 1 || RhsCols == 1) && (Rows == 1 || Cols == 1), THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        fdapde_assert(
          derived().size() == rhs.derived().size(), std::invalid_argument, "matrix dimensions are incompatible");
        using Scalar = std::common_type_t<typename XprType::Scalar, typename RhsXprType::Scalar>;
        Scalar dot_ = 0;
        for (int i = 0, n = fdapde::max(rows(), cols()); i < n; ++i) {
            dot_ += derived().operator[](i) * rhs.derived().operator[](i);
        }
        return dot_;
    }
    // cross product
    /// @brief returns the three-dimensional vector cross product
    template <typename RhsXprType> constexpr auto cross(const MatrixExpr<RhsXprType>& rhs) const& {
        return MatrixCrossProductOp<XprType, RhsXprType>(derived(), rhs.derived());
    }
    /// @brief rejects a cross product that would borrow a temporary right operand
    template <internals::matrix_expression RhsXprType>
        requires(internals::is_owning_rvalue_expression_v<RhsXprType &&>)
    constexpr void cross(RhsXprType&&) const& = delete;
    /// @brief returns the three-dimensional vector cross product
    template <typename RhsXprType>
    constexpr auto cross(const MatrixExpr<RhsXprType>& rhs) const&&
        requires(XprType::NestAsRef == 0)
    {
        return MatrixCrossProductOp<XprType, RhsXprType>(derived(), rhs.derived());
    }
    /// @brief rejects a cross product that would borrow a temporary right operand
    template <internals::matrix_expression RhsXprType>
        requires(internals::is_owning_rvalue_expression_v<RhsXprType &&>)
    constexpr void cross(RhsXprType&&) const&& = delete;
    /// @brief rejects a cross product that would borrow a temporary left owner
    template <typename RhsXprType>
    constexpr void cross(const MatrixExpr<RhsXprType>&) const&&
        requires(XprType::NestAsRef != 0)
    = delete;

    // reshaping
    // static-sized
    /// @brief reinterprets the expression with the requested dimensions
    template <int ReshapedRows_, int ReshapedCols_> constexpr auto reshape() & {
        return ReshapeOp<ReshapedRows_, ReshapedCols_, XprType>(derived());
    }
    /// @brief reinterprets the expression with the requested dimensions
    template <int ReshapedRows_, int ReshapedCols_> constexpr auto reshape() const& {
        return ReshapeOp<ReshapedRows_, ReshapedCols_, const XprType>(derived());
    }
    /// @brief reinterprets the expression with the requested dimensions
    template <int ReshapedRows_, int ReshapedCols_>
      constexpr auto reshape() &&
      requires(XprType::NestAsRef == 0) {
          return ReshapeOp<ReshapedRows_, ReshapedCols_, XprType>(static_cast<XprType&>(*this));
      }
      /// @brief reinterprets the expression with the requested dimensions
      template <int ReshapedRows_, int ReshapedCols_>
      constexpr auto reshape() const&&
          requires(XprType::NestAsRef == 0)
    {
        return ReshapeOp<ReshapedRows_, ReshapedCols_, const XprType>(static_cast<const XprType&>(*this));
    }
    /// @brief rejects borrowing a reshape from a temporary owner
    template <int ReshapedRows_, int ReshapedCols_>
      constexpr void reshape() && requires(XprType::NestAsRef != 0) = delete;
    /// @brief rejects borrowing a reshape from a temporary owner
    template <int ReshapedRows_, int ReshapedCols_>
    constexpr void reshape() const&&
        requires(XprType::NestAsRef != 0)
    = delete;
    /// @brief reinterprets the expression with the requested dimensions
    template <int ReshapedRows_> constexpr auto reshape() & { return ReshapeOp<ReshapedRows_, 1, XprType>(derived()); }
    /// @brief reinterprets the expression with the requested dimensions
    template <int ReshapedRows_> constexpr auto reshape() const& {
        return ReshapeOp<ReshapedRows_, 1, const XprType>(derived());
    }
    /// @brief reinterprets the expression with the requested dimensions
    template <int ReshapedRows_>
      constexpr auto reshape() &&
      requires(XprType::NestAsRef == 0) { return ReshapeOp<ReshapedRows_, 1, XprType>(static_cast<XprType&>(*this)); }
      /// @brief reinterprets the expression with the requested dimensions
      template <int ReshapedRows_>
      constexpr auto reshape() const&&
          requires(XprType::NestAsRef == 0)
    {
        return ReshapeOp<ReshapedRows_, 1, const XprType>(static_cast<const XprType&>(*this));
    }
    /// @brief rejects borrowing a reshape from a temporary owner
    template <int ReshapedRows_> constexpr void reshape() && requires(XprType::NestAsRef != 0) = delete;
    /// @brief rejects borrowing a reshape from a temporary owner
    template <int ReshapedRows_>
    constexpr void reshape() const&&
        requires(XprType::NestAsRef != 0)
    = delete;
    // dynamic-sized
    /// @brief reinterprets the expression with the requested dimensions
    constexpr auto reshape(int rows, int cols) & { return ReshapeOp<Dynamic, Dynamic, XprType>(derived(), rows, cols); }
    /// @brief reinterprets the expression with the requested dimensions
    constexpr auto reshape(int rows, int cols) const& {
        return ReshapeOp<Dynamic, Dynamic, const XprType>(derived(), rows, cols);
    }
    /// @brief reinterprets the expression with the requested dimensions
    constexpr auto reshape(int rows, int cols) &&
      requires(XprType::NestAsRef == 0) {
          return ReshapeOp<Dynamic, Dynamic, XprType>(static_cast<XprType&>(*this), rows, cols);
      }
      /// @brief reinterprets the expression with the requested dimensions
      constexpr auto reshape(int rows, int cols) const&&
          requires(XprType::NestAsRef == 0)
    {
        return ReshapeOp<Dynamic, Dynamic, const XprType>(static_cast<const XprType&>(*this), rows, cols);
    }
    /// @brief rejects borrowing a reshape from a temporary owner
    constexpr void reshape(int, int) && requires(XprType::NestAsRef != 0) = delete;
    /// @brief rejects borrowing a reshape from a temporary owner
    constexpr void reshape(int, int) const&&
        requires(XprType::NestAsRef != 0)
    = delete;
    /// @brief reinterprets the expression with the requested dimensions
    constexpr auto reshape(int rows) & { return ReshapeOp<Dynamic, 1, XprType>(derived(), rows); }
    /// @brief reinterprets the expression with the requested dimensions
    constexpr auto reshape(int rows) const& { return ReshapeOp<Dynamic, 1, const XprType>(derived(), rows); }
    /// @brief reinterprets the expression with the requested dimensions
    constexpr auto reshape(int rows) &&
      requires(XprType::NestAsRef == 0) { return ReshapeOp<Dynamic, 1, XprType>(static_cast<XprType&>(*this), rows); }
      /// @brief reinterprets the expression with the requested dimensions
      constexpr auto reshape(int rows) const&&
          requires(XprType::NestAsRef == 0)
    {
        return ReshapeOp<Dynamic, 1, const XprType>(static_cast<const XprType&>(*this), rows);
    }
    /// @brief rejects borrowing a reshape from a temporary owner
    constexpr void reshape(int) && requires(XprType::NestAsRef != 0) = delete;
    /// @brief rejects borrowing a reshape from a temporary owner
    constexpr void reshape(int) const&&
        requires(XprType::NestAsRef != 0)
    = delete;

    // square matrix methods
    /// @brief returns the symmetric part of the matrix
    constexpr auto symm_part() const& {
        constexpr int Rows = XprType::Rows, Cols = XprType::Cols;
        fdapde_static_assert(
          Rows == Dynamic || Cols == Dynamic || Rows == Cols, THIS_METHODS_IS_FOR_SQUARE_MATRICES_ONLY);
        if constexpr (Rows == Dynamic || Cols == Dynamic) {
            fdapde_assert(
              !(derived().rows() != derived().cols()), std::invalid_argument,
              "symmetric part requires a square matrix");
        }
        return 0.5 * (derived() + derived().transpose());   // symmetric part
    }
    /// @brief returns the skew-symmetric part of the matrix
    constexpr auto skew_part() const& {
        constexpr int Rows = XprType::Rows, Cols = XprType::Cols;
        fdapde_static_assert(
          Rows == Dynamic || Cols == Dynamic || Rows == Cols, THIS_METHODS_IS_FOR_SQUARE_MATRICES_ONLY);
        if constexpr (Rows == Dynamic || Cols == Dynamic) {
            fdapde_assert(
              !(derived().rows() != derived().cols()), std::invalid_argument,
              "skew-symmetric part requires a square matrix");
        }
        return 0.5 * (derived() - derived().transpose());   // skew-symmetric part
    }
    /// @brief returns the symmetric part of the matrix
    constexpr auto symm_part() const&&
        requires(XprType::NestAsRef == 0)
    {
        return static_cast<const MatrixExpr&>(*this).symm_part();
    }
    /// @brief returns the skew-symmetric part of the matrix
    constexpr auto skew_part() const&&
        requires(XprType::NestAsRef == 0)
    {
        return static_cast<const MatrixExpr&>(*this).skew_part();
    }
    /// @brief rejects borrowing a symmetric-part expression from a temporary owner
    constexpr void symm_part() const&&
        requires(XprType::NestAsRef != 0)
    = delete;
    /// @brief rejects borrowing a skew-part expression from a temporary owner
    constexpr void skew_part() const&&
        requires(XprType::NestAsRef != 0)
    = delete;
    /// @brief returns the inverse matrix expression
    constexpr auto inverse() const
        requires(std::is_floating_point_v<std::remove_cv_t<typename XprType::Scalar>>)
    {
        using Scalar = std::remove_cv_t<typename XprType::Scalar>;
        constexpr int Rows = XprType::Rows, Cols = XprType::Cols;
        fdapde_static_assert(
          Rows == Dynamic || Cols == Dynamic || Rows == Cols, THIS_METHODS_IS_FOR_SQUARE_MATRICES_ONLY);
        const XprType& m = derived();
        const int rows_ = m.rows(), cols_ = m.cols();
        fdapde_assert(
          !(rows_ <= 0 || rows_ != cols_), std::invalid_argument, "inverse requires a nonempty square matrix");
        Matrix<Scalar, Rows, Cols> matrix(m);
        PartialPivLU<Matrix<Scalar, Rows, Cols>> factorization(matrix);
        fdapde_assert(!(factorization.info() != 0), std::domain_error, "inverse requires a nonsingular matrix");
        Matrix<Scalar, Rows, Cols> identity;
        if constexpr (Rows == Dynamic || Cols == Dynamic) { identity.resize(rows_, cols_); }
        for (int row = 0; row < rows_; ++row) {
            for (int col = 0; col < cols_; ++col) identity(row, col) = row == col ? Scalar(1) : Scalar(0);
        }
        return factorization.solve(identity);
    }
    /// @brief returns the matrix determinant
    constexpr auto determinant() const {
        using Scalar = std::remove_cv_t<typename XprType::Scalar>;
        constexpr int Rows = XprType::Rows, Cols = XprType::Cols;
        fdapde_static_assert(
          Rows == Dynamic || Cols == Dynamic || Rows == Cols, THIS_METHODS_IS_FOR_SQUARE_MATRICES_ONLY);
        const XprType& m = derived();
        const int rows_ = m.rows(), cols_ = m.cols();
        fdapde_assert(
          !(rows_ <= 0 || rows_ != cols_), std::invalid_argument, "determinant requires a nonempty square matrix");
        if constexpr (std::is_floating_point_v<Scalar>) {
            const PartialPivLU<XprType> factorization(m);
            return factorization.determinant();
        } else {
            fdapde_static_assert(
              Rows != Dynamic && Cols != Dynamic && Rows <= 3,
              DETERMINANTS_REQUIRE_FLOATING_POINT_SCALARS_ABOVE_FIXED_THREE_BY_THREE);
            if constexpr (Rows == 1) return Scalar(m(0, 0));
            if constexpr (Rows == 2) return Scalar(m(0, 0) * m(1, 1) - m(0, 1) * m(1, 0));
            const Scalar a00 = m(0, 0), a01 = m(0, 1), a02 = m(0, 2);
            const Scalar a10 = m(1, 0), a11 = m(1, 1), a12 = m(1, 2);
            const Scalar a20 = m(2, 0), a21 = m(2, 1), a22 = m(2, 2);
            return a00 * (a11 * a22 - a12 * a21) + a01 * (a12 * a20 - a10 * a22) + a02 * (a10 * a21 - a11 * a20);
        }
    }

    // triangular block accessors
    /// @brief returns a view of the selected triangular region
    template <int BlockMode> constexpr Triangular<BlockMode, const XprType> triangular_block() const& {
        return Triangular<BlockMode, const XprType>(derived());
    }
    /// @brief returns a view of the selected triangular region
    template <int BlockMode> constexpr Triangular<BlockMode, XprType> triangular_block() & {
        return Triangular<BlockMode, XprType>(derived());
    }
    /// @brief returns a view of the selected triangular region
    template <int BlockMode>
      constexpr Triangular<BlockMode, XprType> triangular_block() &&
      requires(XprType::NestAsRef == 0) { return Triangular<BlockMode, XprType>(std::move(derived())); }
      /// @brief returns a view of the selected triangular region
      template <int BlockMode>
      constexpr Triangular<BlockMode, const XprType> triangular_block() const&&
          requires(XprType::NestAsRef == 0)
    {
        return Triangular<BlockMode, const XprType>(std::move(derived()));
    }
    /// @brief rejects borrowing a triangular block from a temporary owner
    template <int BlockMode> constexpr void triangular_block() && requires(XprType::NestAsRef != 0) = delete;
    /// @brief rejects borrowing a triangular block from a temporary owner
    template <int BlockMode>
    constexpr void triangular_block() const&&
        requires(XprType::NestAsRef != 0)
    = delete;
    // cast
    /// @brief returns a symmetric matrix adaptor
    template <int ViewMode> constexpr auto as_symmetric() & { return internals::symmetric_cast<ViewMode>(derived()); }
    /// @brief returns a symmetric matrix adaptor
    template <int ViewMode> constexpr auto as_symmetric() const& {
        return internals::symmetric_cast<ViewMode>(derived());
    }
    /// @brief returns a symmetric matrix adaptor
    template <int ViewMode>
      constexpr auto as_symmetric() &&
      requires(XprType::NestAsRef == 0) { return internals::symmetric_cast<ViewMode>(std::move(derived())); }
      /// @brief returns a symmetric matrix adaptor
      template <int ViewMode>
      constexpr auto as_symmetric() const&&
          requires(XprType::NestAsRef == 0)
    {
        return internals::symmetric_cast<ViewMode>(std::move(derived()));
    }
    /// @brief rejects borrowing a symmetric wrapper from a temporary owner
    template <int ViewMode> constexpr void as_symmetric() && requires(XprType::NestAsRef != 0) = delete;
    /// @brief rejects borrowing a symmetric wrapper from a temporary owner
    template <int ViewMode>
    constexpr void as_symmetric() const&&
        requires(XprType::NestAsRef != 0)
    = delete;
    /// @brief returns a skew-symmetric matrix adaptor
    template <int ViewMode> constexpr auto as_skew_symmetric() & {
        return internals::skew_symmetric_cast<ViewMode>(derived());
    }
    /// @brief returns a skew-symmetric matrix adaptor
    template <int ViewMode> constexpr auto as_skew_symmetric() const& {
        return internals::skew_symmetric_cast<ViewMode>(derived());
    }
    /// @brief returns a skew-symmetric matrix adaptor
    template <int ViewMode>
      constexpr auto as_skew_symmetric() &&
      requires(XprType::NestAsRef == 0) { return internals::skew_symmetric_cast<ViewMode>(std::move(derived())); }
      /// @brief returns a skew-symmetric matrix adaptor
      template <int ViewMode>
      constexpr auto as_skew_symmetric() const&&
          requires(XprType::NestAsRef == 0)
    {
        return internals::skew_symmetric_cast<ViewMode>(std::move(derived()));
    }
    /// @brief rejects borrowing a skew-symmetric wrapper from a temporary owner
    template <int ViewMode> constexpr void as_skew_symmetric() && requires(XprType::NestAsRef != 0) = delete;
    /// @brief rejects borrowing a skew-symmetric wrapper from a temporary owner
    template <int ViewMode>
    constexpr void as_skew_symmetric() const&&
        requires(XprType::NestAsRef != 0)
    = delete;
    /// @brief returns a diagonal matrix adaptor
    constexpr auto as_diagonal() & {
        fdapde_static_assert(XprType::Rows == 1 || XprType::Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return internals::diagonal_cast(derived());
    }
    /// @brief returns a diagonal matrix adaptor
    constexpr auto as_diagonal() const& {
        fdapde_static_assert(XprType::Rows == 1 || XprType::Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return internals::diagonal_cast(derived());
    }
    /// @brief returns a diagonal matrix adaptor
    constexpr auto as_diagonal() &&
      requires(XprType::NestAsRef == 0) {
          fdapde_static_assert(XprType::Rows == 1 || XprType::Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
          return internals::diagonal_cast(std::move(derived()));
      }
      /// @brief returns a diagonal matrix adaptor
      constexpr auto as_diagonal() const&&
          requires(XprType::NestAsRef == 0)
    {
        fdapde_static_assert(XprType::Rows == 1 || XprType::Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return internals::diagonal_cast(std::move(derived()));
    }
    /// @brief rejects borrowing a diagonal wrapper from a temporary owner
    constexpr void as_diagonal() && requires(XprType::NestAsRef != 0) = delete;
    /// @brief rejects borrowing a diagonal wrapper from a temporary owner
    constexpr void as_diagonal() const&&
        requires(XprType::NestAsRef != 0)
    = delete;
};

// comparison operators
/// @brief tests equality of every logical coefficient after validating matching shapes
template <typename LhsXprType, typename RhsXprType>
constexpr bool operator==(const MatrixExpr<LhsXprType>& lhs, const MatrixExpr<RhsXprType>& rhs) {
    fdapde_static_assert(
      (internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType> ||
       internals::same_static_shape_v<LhsXprType FDAPDE_COMMA RhsXprType>),
      INVALID_COMPARISON__MATRICES_OF_DIFFERENT_STATIC_SIZE);
    if constexpr (internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType>) {
        fdapde_assert(
          lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols(), std::invalid_argument,
          "matrix dimensions are incompatible");
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
/// @brief tests whether equally shaped matrices differ at any logical coefficient
template <typename LhsXprType, typename RhsXprType>
constexpr bool operator!=(const MatrixExpr<LhsXprType>& op1, const MatrixExpr<RhsXprType>& op2) {
    return !(op1 == op2);
}
/// @brief compares coefficients with relative tolerance
template <typename LhsXprType, typename RhsXprType>
constexpr bool
almost_equal(const MatrixExpr<LhsXprType>& lhs, const MatrixExpr<RhsXprType>& rhs, double epsilon = 1e-7) {
    fdapde_static_assert(
      (internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType> ||
       internals::same_static_shape_v<LhsXprType FDAPDE_COMMA RhsXprType>),
      INVALID_COMPARISON__MATRICES_OF_DIFFERENT_STATIC_SIZE);
    if constexpr (internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType>) {
        fdapde_assert(
          lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols(), std::invalid_argument,
          "matrix dimensions are incompatible");
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

// detection trait
/// @brief identifies matrix expressions after removing cv and reference qualifiers
template <typename XprType> struct is_matrix {
    static constexpr bool value = std::is_base_of_v<MatrixExpr<std::decay_t<XprType>>, XprType>;
};
template <typename XprType> static constexpr bool is_matrix_v = is_matrix<XprType>::value;
/// @brief identifies vector expressions after removing cv and reference qualifiers
template <typename XprType> struct is_vector {
    static constexpr bool value =
      is_matrix_v<XprType> && (std::decay_t<XprType>::Cols == 1 || std::decay_t<XprType>::Rows == 1);
};
template <typename XprType> static constexpr bool is_vector_v = is_vector<XprType>::value;

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_XPR_H__
