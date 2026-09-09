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

#ifndef __FDAPDE_LINALG_UNARY_OP_H__
#define __FDAPDE_LINALG_UNARY_OP_H__

#include "header_check.h"

namespace fdapde {

// this file contains all the expression nodes involving an operation applied on a single MatrixExpr operand

// expression of the transpose of a MatrixExpr operand
/// @brief represents transpose op
template <typename XprType> struct TransposeOp : public MatrixExpr<TransposeOp<XprType>> {
    using Base = MatrixExpr<TransposeOp<XprType>>;
    using XprTypeNested = internals::ref_select_t<const XprType>;
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = XprType::Cols;
    static constexpr int Cols = XprType::Rows;
    static constexpr int StorageOrder = XprType::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    /// @brief constructs transpose op from the supplied state
    template <typename XprType_>
        requires(
          !std::same_as<std::remove_cvref_t<XprType_>, TransposeOp> &&
          internals::safely_nestable<XprTypeNested, XprType_>)
    explicit constexpr TransposeOp(XprType_&& xpr) : xpr_(std::forward<XprType_>(xpr)) { }
    /// @brief accesses or evaluates the requested coefficient
    constexpr Scalar operator()(int i, int j) const { return xpr_(j, i); }
    /// @brief accesses the requested vector coefficient
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(XprType::Cols == 1 || XprType::Rows == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return xpr_[i];
    }
    /// @brief returns the row count
    constexpr int rows() const { return Rows != Dynamic ? Rows : xpr_.cols(); }
    /// @brief returns the column count
    constexpr int cols() const { return Cols != Dynamic ? Cols : xpr_.rows(); }
   private:
    XprTypeNested xpr_;
};

// expression of a reshaped MatrixExpr operand. Reshaping modifes the expression dimensions without reallocating memory
/// @brief represents reshape op
template <int Rows_, int Cols_, typename XprType_>
struct ReshapeOp : public MatrixExpr<ReshapeOp<Rows_, Cols_, XprType_>> {
   private:
    using Base = MatrixExpr<ReshapeOp<Rows_, Cols_, XprType_>>;
    using XprType = std::remove_reference_t<XprType_>;
    using XprTypeClean = std::remove_cv_t<XprType>;
    using XprTypeNested = internals::ref_select_t<XprType_>;
    fdapde_static_assert(
      (Rows_ == Dynamic || Rows_ > 0) && (Cols_ == Dynamic || Cols_ > 0), INVALID_RESHAPE_DIMENSIONS);
    fdapde_static_assert(
      Rows_ == Dynamic || Cols_ == Dynamic ||
        static_cast<std::uint64_t>(Rows_) * static_cast<std::uint64_t>(Cols_) <=
          static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
      MATRIX_SIZE_EXCEEDS_SUPPORTED_RANGE);
    fdapde_static_assert(
      Rows_ == Dynamic || Cols_ == Dynamic || XprTypeClean::Rows == Dynamic || XprTypeClean::Cols == Dynamic ||
        static_cast<std::uint64_t>(Rows_) * static_cast<std::uint64_t>(Cols_) ==
          static_cast<std::uint64_t>(XprTypeClean::Rows) * static_cast<std::uint64_t>(XprTypeClean::Cols),
      INVALID_RESHAPE__DIFFERENT_STATIC_SIZE);
   public:
    using Scalar = typename XprTypeClean::Scalar;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = XprTypeClean::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = std::is_const_v<XprType> || XprTypeClean::ReadOnly;
    using assignment_executor = std::conditional_t<
      (Rows_ == 1 || Cols_ == 1) && !(Rows_ == 1 && Cols_ == 1), internals::vector_assignment_executor,
      internals::generic_assignment_executor>;

    /// @brief constructs reshape op from the supplied state
    constexpr ReshapeOp(const ReshapeOp&) = default;
    /// @brief constructs reshape op from the supplied state
    template <typename XprType__>
        requires(!std::same_as<std::remove_cvref_t<XprType__>, ReshapeOp> &&
                 internals::safely_nestable<XprTypeNested, XprType__>)
    constexpr explicit ReshapeOp(XprType__&& xpr) : rows_(Rows), cols_(Cols), xpr_(std::forward<XprType__>(xpr)) {
        fdapde_static_assert(Rows_ != Dynamic && Cols_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        validate_();
    }
    /// @brief constructs reshape op from the supplied state
    template <typename XprType__>
        requires(internals::safely_nestable<XprTypeNested, XprType__>)
    constexpr ReshapeOp(XprType__&& xpr, int rows, int cols) :
        rows_(Rows == Dynamic ? rows : Rows), cols_(Cols == Dynamic ? cols : Cols), xpr_(std::forward<XprType__>(xpr)) {
        validate_((Rows == Dynamic || rows == Rows) && (Cols == Dynamic || cols == Cols));
    }
    /// @brief constructs reshape op from the supplied state
    template <typename XprType__>
        requires(internals::safely_nestable<XprTypeNested, XprType__>)
    constexpr ReshapeOp(XprType__&& xpr, int rows) :
        ReshapeOp(
          std::forward<XprType__>(xpr), Rows_ == 1 && Cols_ != 1 ? 1 : rows, Rows_ == 1 && Cols_ != 1 ? rows : 1) {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_ONLY_FOR_ROW_OR_COLUMN_VECTORS);
    }
    using Base::operator=;
    /// @brief assigns the supplied coefficients
    constexpr ReshapeOp& operator=(const ReshapeOp& rhs) &
        requires(ReadOnly == 0)
    {
        static_cast<Base&>(*this).template operator= <ReshapeOp>(rhs);
        return *this;
    }
    /// @brief assigns the supplied coefficients
    constexpr ReshapeOp operator=(const ReshapeOp& rhs) &&
      requires(ReadOnly == 0) {
          static_cast<Base&>(*this).template operator= <ReshapeOp>(rhs);
          return *this;
      }
      /// @brief assigns the supplied coefficients
      constexpr ReshapeOp& operator=(const ReshapeOp&) &
          requires(ReadOnly != 0)
      = delete;
    /// @brief assigns the supplied coefficients
    constexpr ReshapeOp operator=(const ReshapeOp&) && requires(ReadOnly != 0) = delete;
    // access
    /// @brief accesses or evaluates the requested coefficient
    constexpr decltype(auto) operator()(int i, int j) const {
        fdapde_assert(!(i < 0 || i >= rows_ || j < 0 || j >= cols_), std::out_of_range, "reshape index out of range");
        const auto [row, col] = reshaped_(i, j);
        return std::as_const(xpr_)(row, col);
    }
    /// @brief accesses the requested vector coefficient
    constexpr decltype(auto) operator[](int i) const {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_ONLY_FOR_ROW_OR_COLUMN_VECTORS);
        fdapde_assert(!(i < 0 || i >= rows_ * cols_), std::out_of_range, "reshape index out of range");
        return Rows == 1 ? operator()(0, i) : operator()(i, 0);
    }
    /// @brief accesses or evaluates the requested coefficient
    constexpr decltype(auto) operator()(int i, int j)
        requires(ReadOnly == 0)
    {
        fdapde_assert(!(i < 0 || i >= rows_ || j < 0 || j >= cols_), std::out_of_range, "reshape index out of range");
        const auto [row, col] = reshaped_(i, j);
        return xpr_(row, col);
    }
    /// @brief accesses the requested vector coefficient
    constexpr decltype(auto) operator[](int i)
        requires(ReadOnly == 0)
    {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_ONLY_FOR_ROW_OR_COLUMN_VECTORS);
        fdapde_assert(!(i < 0 || i >= rows_ * cols_), std::out_of_range, "reshape index out of range");
        return Rows == 1 ? operator()(0, i) : operator()(i, 0);
    }
    // observers
    /// @brief returns the row count
    constexpr int rows() const { return rows_; }
    /// @brief returns the column count
    constexpr int cols() const { return cols_; }
    /// @brief accesses the requested packed word
    constexpr decltype(auto) bitpack(int i) const
        requires requires(const XprTypeClean& xpr) { xpr.bitpack(i); }
    {
        return std::as_const(xpr_).bitpack(i);
    }
    /// @brief returns the number of occupied storage words
    constexpr int bitpacks() const
        requires requires(const XprTypeClean& xpr) { xpr.bitpacks(); }
    {
        return std::as_const(xpr_).bitpacks();
    }
   private:
    /// @brief checks the type requirements
    constexpr void validate_(bool target_matches = true) const {
        fdapde_assert(!(!target_matches), std::invalid_argument, "reshape arguments do not match its static shape");
        const int target_size = internals::checked_matrix_size(rows_, cols_);
        const int source_size = internals::checked_matrix_size(xpr_.rows(), xpr_.cols());
        fdapde_assert(
          !(target_size != source_size), std::invalid_argument, "reshape requires matching source and target sizes");
    }
    /// @brief returns the nested expression with its new dimensions
    constexpr std::pair<int, int> reshaped_(int i, int j) const {
        const int k = StorageOrder == RowMajor ? i * cols_ + j : j * rows_ + i;
        if constexpr (StorageOrder == RowMajor) {
            return std::make_pair(k / xpr_.cols(), k % xpr_.cols());
        } else {
            return std::make_pair(k % xpr_.rows(), k / xpr_.rows());
        }
    }
    int rows_, cols_;
    XprTypeNested xpr_;
};

// redux suppport. Reductions are unary operations which collapse a MatrixExpr operand into a single scalar
namespace internals {

// linear reduction loop on matrix expressions
/// @brief represents matrix redux linear executor
struct matrix_redux_linear_executor {
    /// @brief executes the coefficient operation over the supplied expressions
    template <typename XprType_, typename Scalar, typename Functor>
    static constexpr auto run(XprType_&& xpr, Scalar init, Functor f) {
        using XprType = std::decay_t<XprType_>;
        fdapde_assert(xpr.size() > 0, std::invalid_argument, "reshape requires a nonempty expression");
        Scalar res = init;
        const int rows_ = xpr.rows();
        const int cols_ = xpr.cols();
        // exploit cache-locality depending on storage order of target expression
        if constexpr (XprType::StorageOrder == RowMajor) {
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) { res = f(res, xpr(i, j)); }
            }
        } else {   // ColMajor
            for (int j = 0; j < cols_; ++j) {
                for (int i = 0; i < rows_; ++i) { res = f(res, xpr(i, j)); }
            }
        }
        return res;
    }
};

// boolean linear reduction loop on matrix expression
/// @brief represents boolean redux linear executor
struct boolean_redux_linear_executor {
    // returns b at the first true occurence of f, otherwise returns !b
    /// @brief executes the coefficient operation over the supplied expressions
    template <typename XprType_, typename Functor> static constexpr auto run(XprType_&& xpr, bool b, Functor f) {
        using XprType = std::decay_t<XprType_>;
        fdapde_assert(xpr.size() > 0, std::invalid_argument, "reshape requires a nonempty expression");
        const int rows_ = xpr.rows();
        const int cols_ = xpr.cols();
        // exploit cache-locality depending on storage order of target expression
        if constexpr (XprType::StorageOrder == RowMajor) {
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) {
                    if (bool(f(xpr(i, j)))) { return b; }
                }
            }
        } else {   // ColMajor
            for (int j = 0; j < cols_; ++j) {
                for (int i = 0; i < rows_; ++i) {
                    if (bool(f(xpr(i, j)))) { return b; }
                }
            }
        }
        return !b;
    }
};

}   // namespace internals
}   // namespace fdapde

#endif   // __FDAPDE_LINALG_UNARY_OP_H__
