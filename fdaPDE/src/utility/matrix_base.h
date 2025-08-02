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

#ifndef __FDAPDE_MATRIX_BASE_H__
#define __FDAPDE_MATRIX_BASE_H__

#include "header_check.h"

namespace fdapde {

// TODO: visitors, rowwise, colwwise, coeffwise iterators ... support Dynamic

template <int Rows, int Cols, typename Derived> struct MatrixBase;

[[maybe_unused]] constexpr int RowMajor = 0;
[[maybe_unused]] constexpr int ColMajor = 1;

namespace internals {

#ifdef __FDAPDE_HAS_EIGEN__

template <typename Derived> struct eigen_xpr_wrap : public Derived {
    using Derived::Derived;   // inherits Derived constructors
    eigen_xpr_wrap(const Derived& xpr) : Derived(xpr) { }
    eigen_xpr_wrap& operator=(const Derived& xpr) {
        Derived::operator=(xpr);
        return *this;
    }
    eigen_xpr_wrap(Derived&& xpr) : Derived(xpr) { }
    eigen_xpr_wrap& operator=(Derived&& xpr) {
        Derived::operator=(xpr);
        return *this;
    }
    // injected additional constants
    static constexpr int Rows = Derived::RowsAtCompileTime;
    static constexpr int Cols = Derived::ColsAtCompileTime;
};

#endif

}   // namespace internals

template <typename Derived>
struct Transpose : public MatrixBase<Derived::Cols, Derived::Rows, Transpose<Derived>> {
    using Base = MatrixBase<Derived::Cols, Derived::Rows, Transpose<Derived>>;
    using Scalar = typename Derived::Scalar;
    static constexpr int Rows = Derived::Cols;
    static constexpr int Cols = Derived::Rows;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    constexpr Transpose(const Derived& xpr) : xpr_(xpr) { }
    constexpr Scalar operator()(int i, int j) const { return xpr_(j, i); }
    constexpr Scalar operator[](int i) const
        requires(Derived::Cols == 1 || Derived::Rows == 1) {
        fdapde_static_assert(
          Derived::Cols == 1 || Derived::Rows == 1, THIS_METHOD_IS_ONLY_FOR_CONSTEXPR_ROW_OR_COLUMN_VECTORS);
        return xpr_[i];
    }
    constexpr int rows() const { return xpr_.cols(); }
    constexpr int cols() const { return xpr_.rows(); }
   protected:
    internals::ref_select_t<const Derived> xpr_;
};

[[maybe_unused]] constexpr int Upper = 0;       // lower triangular view of matrix
[[maybe_unused]] constexpr int Lower = 1;       // upper triangular view of matrix
[[maybe_unused]] constexpr int UnitUpper = 2;   // lower triangular view of matrix with ones on the diagonal
[[maybe_unused]] constexpr int UnitLower = 3;   // upper triangular view of matrix with ones on the diagonal
  
template <typename Derived, int ViewMode>
struct TriangularView : public MatrixBase<Derived::Rows, Derived::Cols, TriangularView<Derived, ViewMode>> {
    fdapde_static_assert(
      Derived::Rows != 1 && Derived::Cols != 1 && Derived::Rows == Derived::Cols,
      TRIANGULAR_VIEW_DEFINED_ONLY_FOR_SQUARED_MATRICES);
    using Base = MatrixBase<Derived::Rows, Derived::Cols, TriangularView<Derived, ViewMode>>;
    using Scalar = typename Derived::Scalar;
    static constexpr int Rows = Derived::Cols;
    static constexpr int Cols = Derived::Rows;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = Derived::ReadOnly;

    constexpr TriangularView() = default;
    constexpr TriangularView(const Derived& xpr) : xpr_(xpr) { }
    constexpr int rows() const { return xpr_.rows(); }
    constexpr int cols() const { return xpr_.cols(); }
    constexpr Scalar operator()(int i, int j) const {
        if constexpr (ViewMode == Upper) return i > j ? 0 : xpr_(i, j);
        if constexpr (ViewMode == Lower) return i < j ? 0 : xpr_(i, j);
        if constexpr (ViewMode == UnitUpper) return i > j ? 0 : (i == j ? 1 : xpr_(i, j));
        if constexpr (ViewMode == UnitLower) return i < j ? 0 : (i == j ? 1 : xpr_(i, j));
    }
    // block assignment
    template <int Rows_, int Cols_, typename RhsType>
    constexpr TriangularView<Derived, ViewMode>& operator=(const MatrixBase<Rows_, Cols_, RhsType>& rhs) {
        fdapde_static_assert(Derived::ReadOnly == 0, BLOCK_ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION_IS_INVALID);
        fdapde_static_assert(
          ViewMode == Upper || ViewMode == Lower, TRIANGULAR_BLOCK_ASSIGNMENT_REQUIRES_EITHER_UPPER_OR_LOWER_VIEW);
        fdapde_static_assert(
          Derived::Rows == Rows_ && Derived::Cols_ == Cols &&
            std::is_convertible_v<typename RhsType::Scalar FDAPDE_COMMA Scalar>,
          INVALID_RHS_SIZE_OR_YOU_ARE_TRYING_TO_ASSIGN_A_BLOCK_WITH_NON_CONVERTIBLE_SCALAR_TYPE);
        int row = 0, col = 0;
        for (int i = 0; i < Rows_; ++i) {
            for (int j = 0; j < i; ++j) {
                if constexpr (ViewMode == Lower) { row = i; col = j; }
                if constexpr (ViewMode == Upper) { row = j; col = i; }
                xpr_(row, col) = rhs(row, col);
            }
        }
	// assign diagonal
        for (int i = 0; i < Rows_; ++i) { xpr_(i, i) = rhs(i, i); }
        return *this;
    }
   private:
    internals::ref_select_t<const Derived> xpr_;
};

template <typename Lhs, typename Rhs, typename BinaryOperation>
struct MatrixBinOp : public MatrixBase<Lhs::Rows, Lhs::Cols, MatrixBinOp<Lhs, Rhs, BinaryOperation>> {
    fdapde_static_assert(Lhs::Rows == Rhs::Rows && Lhs::Cols == Rhs::Cols, YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES);
    using Base = MatrixBase<Lhs::Rows, Lhs::Cols, MatrixBinOp<Lhs, Rhs, BinaryOperation>> ;
    using Scalar = decltype(std::declval<BinaryOperation>().operator()(
      std::declval<typename Lhs::Scalar>(), std::declval<typename Rhs::Scalar>()));
    static constexpr int Rows = Lhs::Rows;
    static constexpr int Cols = Lhs::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    constexpr MatrixBinOp(const Lhs& lhs, const Rhs& rhs, BinaryOperation op) : lhs_(lhs), rhs_(rhs), op_(op) { }
    constexpr Scalar operator()(int i, int j) const { return op_(lhs_(i, j), rhs_(i, j)); }
    constexpr Scalar operator[](int i) const
        requires((Lhs::Cols == 1 && Rhs::Cols == 1) || (Lhs::Rows == 1 && Rhs::Rows == 1)) {
        fdapde_static_assert(
          (Lhs::Cols == 1 && Rhs::Cols == 1) || (Lhs::Rows == 1 && Rhs::Rows == 1),
          THIS_METHOD_IS_ONLY_FOR_CONSTEXPR_ROW_OR_COLUMN_VECTORS);
        return op_(lhs_[i], rhs_[i]);
    }
    constexpr int rows() const { return lhs_.rows(); }
    constexpr int cols() const { return lhs_.cols(); }
   protected:
    internals::ref_select_t<const Lhs> lhs_;
    internals::ref_select_t<const Rhs> rhs_;
    BinaryOperation op_;
};
template <typename Lhs, typename Rhs>
constexpr MatrixBinOp<Lhs, Rhs, std::plus<>>
operator+(const MatrixBase<Lhs::Rows, Lhs::Cols, Lhs>& lhs, const MatrixBase<Rhs::Rows, Rhs::Cols, Rhs>& rhs) {
    return MatrixBinOp<Lhs, Rhs, std::plus<>> {lhs.derived(), rhs.derived(), std::plus<>()};
}
template <typename Lhs, typename Rhs>
constexpr MatrixBinOp<Lhs, Rhs, std::minus<>>
operator-(const MatrixBase<Lhs::Rows, Lhs::Cols, Lhs>& lhs, const MatrixBase<Rhs::Rows, Rhs::Cols, Rhs>& rhs) {
    return MatrixBinOp<Lhs, Rhs, std::minus<>> {lhs.derived(), rhs.derived(), std::minus<>()};
}

#ifdef __FDAPDE_HAS_EIGEN__

template <typename Lhs, typename Rhs>
constexpr MatrixBinOp<Lhs, internals::eigen_xpr_wrap<Rhs>, std::plus<>>
operator+(const MatrixBase<Lhs::Rows, Lhs::Cols, Lhs>& lhs, const Eigen::MatrixBase<Rhs>& rhs) {
    fdapde_static_assert(
      Rhs::RowsAtCompileTime == Lhs::Rows && Rhs::ColsAtCompileTime == Rhs::Cols,
      INVALID_MATRIX_DIMENSIONS_IN_BINARY_OPERATION);
    return MatrixBinOp<Lhs, internals::eigen_xpr_wrap<Rhs>, std::plus<>> {lhs.derived(), rhs.derived(), std::plus<>()};
}
template <typename Lhs, typename Rhs>
constexpr MatrixBinOp<internals::eigen_xpr_wrap<Lhs>, Rhs, std::plus<>>
operator+(const Eigen::MatrixBase<Lhs>& lhs, const MatrixBase<Rhs::Rows, Rhs::Cols, Rhs>& rhs) {
    fdapde_static_assert(
      Lhs::RowsAtCompileTime == Rhs::Rows && Lhs::ColsAtCompileTime == Rhs::Cols,
      INVALID_MATRIX_DIMENSIONS_IN_BINARY_OPERATION);
    return MatrixBinOp<internals::eigen_xpr_wrap<Lhs>, Rhs, std::plus<>> {lhs.derived(), rhs.derived(), std::plus<>()};
}
template <typename Lhs, typename Rhs>
constexpr MatrixBinOp<Lhs, internals::eigen_xpr_wrap<Rhs>, std::minus<>>
operator-(const MatrixBase<Lhs::Rows, Lhs::Cols, Lhs>& lhs, const Eigen::MatrixBase<Rhs>& rhs) {
    fdapde_static_assert(
      Rhs::RowsAtCompileTime == Lhs::Rows && Rhs::ColsAtCompileTime == Lhs::Cols,
      INVALID_MATRIX_DIMENSIONS_IN_BINARY_OPERATION);
    return MatrixBinOp<Lhs, internals::eigen_xpr_wrap<Rhs>, std::minus<>> {
      lhs.derived(), rhs.derived(), std::minus<>()};
}
template <typename Lhs, typename Rhs>
constexpr MatrixBinOp<internals::eigen_xpr_wrap<Lhs>, Rhs, std::minus<>>
operator-(const Eigen::MatrixBase<Lhs>& lhs, const MatrixBase<Rhs::Rows, Rhs::Cols, Rhs>& rhs) {
    fdapde_static_assert(
      Lhs::RowsAtCompileTime == Rhs::Rows && Lhs::ColsAtCompileTime == Rhs::Cols,
      INVALID_MATRIX_DIMENSIONS_IN_BINARY_OPERATION);
    return MatrixBinOp<internals::eigen_xpr_wrap<Lhs>, Rhs, std::minus<>> {
      lhs.derived(), rhs.derived(), std::minus<>()};
}
  
#endif

template <typename Lhs, typename Rhs, typename BinaryOperation>
struct MatrixCoeffWiseOp :
    public MatrixBase<
      std::conditional_t<std::is_arithmetic_v<Lhs>, Rhs, Lhs>::Rows,
      std::conditional_t<std::is_arithmetic_v<Lhs>, Rhs, Lhs>::Cols, MatrixCoeffWiseOp<Lhs, Rhs, BinaryOperation>> {
    fdapde_static_assert(
      (std::is_arithmetic_v<Lhs> || std::is_arithmetic_v<Rhs>) &&
        !(std::is_arithmetic_v<Lhs> && std::is_arithmetic_v<Rhs>),
      THIS_CLASS_MUST_HAVE_EXACTLY_ONE_BETWEEEN_LHS_AND_RHS_OF_ARITHMETIC_TYPE);
    using CoeffType_ = std::conditional_t<std::is_arithmetic_v<Lhs>, Lhs, Rhs>;
    using Lhs_ = std::decay_t<Lhs>;
    using Rhs_ = std::decay_t<Rhs>;
    using BinaryOperation_ = std::decay_t<BinaryOperation>;
    static constexpr bool is_coeff_lhs = std::is_arithmetic_v<Lhs>;
   public:
    using XprType = std::conditional_t<std::is_arithmetic_v<Lhs>, Rhs, Lhs>;
    using Base = MatrixCoeffWiseOp<Lhs, Rhs, BinaryOperation>;
    using Scalar = decltype(std::declval<BinaryOperation>().operator()(
      std::declval<typename XprType::Scalar>(), std::declval<CoeffType_>()));
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    constexpr MatrixCoeffWiseOp(const Lhs& lhs, const Rhs& rhs, BinaryOperation op) :
        lhs_(lhs), rhs_(rhs), op_(op) { }
    constexpr Scalar operator()(int i, int j) const {
        if constexpr ( is_coeff_lhs) { return op_(lhs_, rhs_(i, j)); }
        if constexpr (!is_coeff_lhs) { return op_(lhs_(i, j), rhs_); }
    }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(
          XprType::Rows == 1 || XprType::Cols == 1, THIS_METHOD_IS_ONLY_FOR_ROW_OR_COLUMN_VECTORS);
	if constexpr ( is_coeff_lhs) { return op_(lhs_, rhs_[i]); }
        if constexpr (!is_coeff_lhs) { return op_(lhs_[i], rhs_); }
    }
    constexpr int rows() const { return xpr().rows(); }
    constexpr int cols() const { return xpr().cols(); }
    constexpr const Lhs_& lhs() const { return lhs_; }
    constexpr const Rhs_& rhs() const { return rhs_; }
    constexpr const BinaryOperation_& functor() const { return op_; }
   protected:
    const XprType& xpr() const {
        if constexpr ( is_coeff_lhs) return rhs_;
        if constexpr (!is_coeff_lhs) return lhs_;
    }
    internals::ref_select_t<const Lhs> lhs_;
    internals::ref_select_t<const Rhs> rhs_;
    BinaryOperation op_;
};

template <typename XprType, typename Coeff>
constexpr MatrixCoeffWiseOp<XprType, Coeff, std::multiplies<>>
operator*(const MatrixBase<XprType::Rows, XprType::Cols, XprType>& lhs, Coeff rhs)
    requires(std::is_arithmetic_v<Coeff>) {
    return MatrixCoeffWiseOp<XprType, Coeff, std::multiplies<>> {lhs.derived(), rhs, std::multiplies<>()};
}
template <typename XprType, typename Coeff>
constexpr MatrixCoeffWiseOp<Coeff, XprType, std::multiplies<>>
operator*(Coeff lhs, const MatrixBase<XprType::Rows, XprType::Cols, XprType>& rhs)
    requires(std::is_arithmetic_v<Coeff>) {
    return MatrixCoeffWiseOp<Coeff, XprType, std::multiplies<>> {lhs, rhs.derived(), std::multiplies<>()};
}
template <typename XprType, typename Coeff>
constexpr MatrixCoeffWiseOp<XprType, Coeff, std::divides<>>
operator/(const MatrixBase<XprType::Rows, XprType::Cols, XprType>& lhs, Coeff rhs)
    requires(std::is_arithmetic_v<Coeff>) {
    return MatrixCoeffWiseOp<XprType, Coeff, std::divides<>> {lhs.derived(), rhs, std::divides<>()};
}

template <typename Lhs, typename Rhs>
struct MatrixProduct : public MatrixBase<Lhs::Rows, Rhs::Cols, MatrixProduct<Lhs, Rhs>> {
    fdapde_static_assert(Lhs::Cols == Rhs::Rows, INVALID_OPERAND_DIMENSIONS_FOR_MATRIX_MATRIX_PRODUCT);
    using Base = MatrixBase<Lhs::Rows, Rhs::Cols, MatrixProduct<Lhs, Rhs>>;
    using Scalar = decltype(std::declval<typename Lhs::Scalar>() * std::declval<typename Rhs::Scalar>());
    static constexpr int Rows = Lhs::Rows;
    static constexpr int Cols = Rhs::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    constexpr MatrixProduct(const Lhs& lhs, const Rhs& rhs) : lhs_(lhs), rhs_(rhs) { }
    constexpr Scalar operator()(int i, int j) const {
        Scalar tmp = 0;
        for (int k = 0; k < Lhs::Cols; ++k) tmp += lhs_(i, k) * rhs_(k, j);
        return tmp;
    }
    constexpr Scalar operator[](int i) const
        requires(Lhs::Rows == 1 || Rhs::Cols) {
        fdapde_static_assert(
          (Lhs::Rows == 1 && Lhs::Cols == Rhs::Rows) || (Rhs::Cols == 1 && Lhs::Cols == Rhs::Rows),
          THIS_METHOD_IS_ONLY_FOR_CONSTEXPR_ROW_OR_COLUMN_VECTORS);
        constexpr int size = Lhs::Rows == 1 ? Rows : Cols;
        Scalar tmp = 0;
        for (int k = 0; k < size; ++k) {
            if constexpr (Lhs::Rows == 1) tmp += lhs_[k] * rhs_(k, i);
            if constexpr (Rhs::Cols == 1) tmp += lhs_(i, k) * rhs_[k];
        }
	return tmp;
    }
    constexpr int rows() const { return lhs_.rows(); }
    constexpr int cols() const { return rhs_.cols(); }
   protected:
    internals::ref_select_t<const Lhs> lhs_;
    internals::ref_select_t<const Rhs> rhs_;
};
template <typename Lhs, typename Rhs>
constexpr MatrixProduct<Lhs, Rhs>
operator*(const MatrixBase<Lhs::Rows, Lhs::Cols, Lhs>& op1, const MatrixBase<Rhs::Rows, Rhs::Cols, Rhs>& op2) {
    return MatrixProduct<Lhs, Rhs> {op1.derived(), op2.derived()};
}

#ifdef __FDAPDE_HAS_EIGEN__
  
template <typename Lhs, typename Rhs>
constexpr MatrixProduct<Lhs, internals::eigen_xpr_wrap<Rhs>>
operator*(const MatrixBase<Lhs::Rows, Lhs::Cols, Lhs>& op1, const Eigen::MatrixBase<Rhs>& op2) {
    fdapde_static_assert(Lhs::Cols == Rhs::RowsAtCompileTime, INVALID_MATRIX_DIMENSIONS_IN_BINARY_OPERATION);
    return MatrixProduct<Lhs, internals::eigen_xpr_wrap<Rhs>> {op1.derived(), op2.derived()};
}
template <typename Lhs, typename Rhs>
constexpr MatrixProduct<internals::eigen_xpr_wrap<Lhs>, Rhs>
operator*(const Eigen::MatrixBase<Lhs>& op1, const MatrixBase<Rhs::Rows, Rhs::Cols, Rhs>& op2) {
    fdapde_static_assert(Lhs::ColsAtCompileTime == Rhs::Rows, INVALID_MATRIX_DIMENSIONS_IN_BINARY_OPERATION);
    return MatrixProduct<internals::eigen_xpr_wrap<Lhs>, Rhs> {op1.derived(), op2.derived()};
}

#endif

// kronecker tensor product between matrices
template <typename Lhs, typename Rhs>
struct MatrixKroneckerProduct :
    public MatrixBase<Lhs::Rows * Rhs::Rows, Lhs::Cols * Rhs::Cols, MatrixKroneckerProduct<Lhs, Rhs>> {
    using Base = MatrixBase<Lhs::Rows * Rhs::Rows, Lhs::Cols * Rhs::Cols, MatrixKroneckerProduct<Lhs, Rhs>>;
    using Scalar = decltype(std::declval<typename Lhs::Scalar>() * std::declval<typename Rhs::Scalar>());
    static constexpr int Rows = Lhs::Rows * Rhs::Rows;
    static constexpr int Cols = Lhs::Cols * Rhs::Cols;
    static constexpr int NestAsReaf = 0;
    static constexpr int ReadOnly = 1;

    constexpr MatrixKroneckerProduct(const Lhs& lhs, const Rhs& rhs) : lhs_(lhs), rhs_(rhs) { }
    constexpr Scalar operator()(int i, int j) const {
        // compute offsets in operand matrices
        int col_lhs = j / Rhs::Cols, row_lhs = i / Rhs::Rows;
        int col_rhs = j % Rhs::Cols, row_rhs = i % Rhs::Rows;
        return lhs_(row_lhs, col_lhs) * rhs_(row_rhs, col_rhs);
    }
    constexpr int rows() const { return lhs_.rows() * rhs_.rows(); }
    constexpr int cols() const { return lhs_.cols() * rhs_.cols(); }
   protected:
    internals::ref_select_t<const Lhs> lhs_;
    internals::ref_select_t<const Rhs> rhs_;
};
template <typename Lhs, typename Rhs>
constexpr MatrixKroneckerProduct<Lhs, Rhs>
kronecker(const MatrixBase<Lhs::Rows, Lhs::Cols, Lhs>& op1, const MatrixBase<Rhs::Rows, Rhs::Cols, Rhs>& op2) {
    return MatrixKroneckerProduct<Lhs, Rhs> {op1.derived(), op2.derived()};
}
  
template <int BlockRows_, int BlockCols_, typename Derived>
class MatrixBlock : public MatrixBase<BlockRows_, BlockCols_, MatrixBlock<BlockRows_, BlockCols_, Derived>> {
    fdapde_static_assert(
      BlockRows_ > 0 && BlockCols_ > 0 && BlockRows_ <= Derived::Rows && BlockCols_ <= Derived::Cols,
      INVALID_BLOCK_SIZES);
   public:
    using Base = MatrixBase<BlockRows_, BlockCols_, MatrixBlock<BlockRows_, BlockCols_, Derived>>;
    using Scalar = typename Derived::Scalar;
    static constexpr int Rows = BlockRows_;
    static constexpr int Cols = BlockCols_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = Derived::ReadOnly;

    constexpr MatrixBlock(Derived& xpr, int i) :
        start_row_(BlockRows_ == 1 ? i : 0), start_col_(BlockCols_ == 1 ? i : 0), xpr_(xpr) {
        fdapde_static_assert(BlockRows_ == 1 || BlockCols_ == 1, THIS_METHOD_IS_ONLY_FOR_ROW_AND_COLUMN_BLOCKS);
        fdapde_constexpr_assert(
          i >= 0 && ((BlockRows_ == 1 && i < xpr_.rows()) || (BlockCols_ == 1 && i < xpr_.cols())));
    }
    constexpr MatrixBlock(Derived& xpr, int start_row, int start_col) :
        start_row_(start_row), start_col_(start_col), xpr_(xpr) { }
    constexpr int rows() const { return Rows; }
    constexpr int cols() const { return Cols; }
    constexpr Scalar operator()(int i, int j) const { return xpr_(start_row_ + i, start_col_ + j); }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(
          BlockRows_ == 1 || BlockCols_ == 1, THIS_METHOD_IS_ONLY_FOR_CONSTEXPR_ROW_OR_COLUMN_BLOCKS);
        if constexpr (Rows == 1) return xpr_(start_row_, start_col_ + i);
        if constexpr (Cols == 1) return xpr_(start_row_ + i, start_col_);
    }
    constexpr Scalar& operator()(int i, int j) { return xpr_(start_row_ + i, start_col_ + j); }
    constexpr Scalar& operator[](int i) {
        fdapde_static_assert(
          BlockRows_ == 1 || BlockCols_ == 1, THIS_METHOD_IS_ONLY_FOR_CONSTEXPR_ROW_OR_COLUMN_BLOCKS);
        if constexpr (Rows == 1) return xpr_(start_row_, start_col_ + i);
        if constexpr (Cols == 1) return xpr_(start_row_ + i, start_col_);
    }
    // block assignment
    constexpr MatrixBlock& operator=(const MatrixBlock& other) {
        fdapde_static_assert(Derived::ReadOnly == 0, BLOCK_ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION_IS_INVALID);
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) { xpr_(start_row_ + i, start_col_ + j) = other(i, j); }
        }
        return *this;
    }
    constexpr MatrixBlock(const MatrixBlock& other) :
        start_row_(other.start_row_), start_col_(other.start_col_), xpr_(other.xpr_) { }
    template <int Rows_, int Cols_, typename RhsType>
    constexpr MatrixBlock<BlockRows_, BlockCols_, Derived>& operator=(const MatrixBase<Rows_, Cols_, RhsType>& rhs) {
        fdapde_static_assert(Derived::ReadOnly == 0, BLOCK_ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION_IS_INVALID);
        fdapde_static_assert(
          RhsType::Cols == Cols && RhsType::Rows == Rows &&
            std::is_convertible_v<typename RhsType::Scalar FDAPDE_COMMA Scalar>,
          INVALID_RHS_SIZE_OR_YOU_ARE_TRYING_TO_ASSIGN_A_BLOCK_WITH_NON_CONVERTIBLE_SCALAR_TYPE);
        for (int i = 0; i < Rows_; ++i) {
            for (int j = 0; j < Cols_; ++j) { xpr_(start_row_ + i, start_col_ + j) = rhs.derived()(i, j); }
        }
        return *this;
    }
   protected:
    int start_row_ = 0, start_col_ = 0;
    internals::ref_select_t<Derived> xpr_;
};

template <typename Derived>
struct DiagonalBlock : public MatrixBase<Derived::Rows, Derived::Cols, DiagonalBlock<Derived>> {
    fdapde_static_assert(Derived::Rows == Derived::Cols, DIAGONAL_BLOCK_DEFINED_ONLY_FOR_SQUARED_MATRICES);
    using Base = MatrixBase<Derived::Rows, Derived::Cols, DiagonalBlock<Derived>>;
    using Scalar = typename Derived::Scalar;
    static constexpr int Rows = Derived::Rows;
    static constexpr int Cols = Derived::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    constexpr DiagonalBlock() = default;
    constexpr DiagonalBlock(Derived& xpr) : xpr_(xpr) { }
    constexpr int rows() const { return xpr_.rows(); }
    constexpr int cols() const { return xpr_.cols(); }
    constexpr Scalar operator()(int i, int j) const { return i == j ? xpr_(i, i) : 0; }
    constexpr const Scalar& operator[](int i) const { return xpr_(i, i); }
    constexpr Scalar& operator[](int i) { return xpr_(i, i); }
    // assignment operator
    template <typename RhsType> constexpr DiagonalBlock<Derived>& operator=(const RhsType& rhs) {
        fdapde_static_assert(Derived::ReadOnly == 0, BLOCK_ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION_IS_INVALID);
        fdapde_static_assert(
          RhsType::Cols == 1 && RhsType::Rows == Rows &&
            std::is_convertible_v<typename RhsType::Scalar FDAPDE_COMMA Scalar>,
          VECTOR_REQUIRED_OR_YOU_ARE_TRYING_TO_ASSIGN_A_BLOCK_WITH_NON_CONVERTIBLE_SCALAR_TYPE);
        for (int i = 0; i < Rows; ++i) { xpr_(i, i) = rhs[i]; }
        return *this;
    }
   private:
    internals::ref_select_t<Derived> xpr_;
};

namespace internals {

// linear reduction loop on matrix expressions
template <typename XprType, typename Functor> struct linear_matrix_redux_op {
    using Scalar = typename XprType::Scalar;

    static constexpr Scalar run(const XprType& xpr, Scalar init, Functor f) {
        fdapde_constexpr_assert(xpr.size() > 0);
        Scalar res = init;
        int rows_ = xpr.rows(), cols_ = xpr.cols();
        for (int i = 0; i < rows_; ++i) {
            for (int j = 0; j < cols_; ++j) { res = f(res, xpr(i, j)); }
        }
        return res;
    }
};

}   // namespace internals

template <int Rows, int Cols, typename Derived> struct MatrixBase {
#ifdef __FDAPDE_HAS_EIGEN__ // compatibility with Eigen types
    static constexpr int RowsAtCompileTime = Rows;
    static constexpr int ColsAtCompileTime = Cols;
#endif
    constexpr int size() const { return derived().rows() * derived().cols(); }
    constexpr const Derived& derived() const { return static_cast<const Derived&>(*this); }
    constexpr Derived& derived() { return static_cast<Derived&>(*this); }
    // send matrix to out stream (this is not constexpr evaluable)
    friend std::ostream& operator<<(std::ostream& out, const MatrixBase& m) {
        for (int i = 0; i < m.derived().rows() - 1; ++i) {
            for (int j = 0; j < m.derived().cols(); ++j) { out << m.derived().operator()(i, j) << " "; }
            out << "\n";
        }
        // print last row without carriage return
        for (int j = 0; j < m.derived().cols(); ++j) {
            out << m.derived().operator()(m.derived().rows() - 1, j) << " ";
        }
        return out;
    }
    // frobenius norm (L^2 norm of a matrix)
    constexpr auto squared_norm() const {
        typename Derived::Scalar norm_ = 0;
        for (int i = 0; i < derived().rows(); ++i) {
            for (int j = 0; j < derived().cols(); ++j) { norm_ += fdapde::pow(derived().operator()(i, j), 2); }
        }
        return norm_;
    }
    constexpr auto norm() const { return fdapde::sqrt(squared_norm()); }
    // maximum norm (L^\infinity norm)
    constexpr auto inf_norm() const {
        using Scalar = typename Derived::Scalar;
        Scalar norm_ = std::numeric_limits<Scalar>::min();
        for (int i = 0; i < derived().rows(); ++i) {
            for (int j = 0; j < derived().cols(); ++j) {
                Scalar tmp = fdapde::abs(derived().operator()(i, j));
                if (tmp > norm_) norm_ = tmp;
            }
        }
        return norm_;
    }
    // redux operators
    template <typename Scalar_, typename Functor> constexpr auto redux(Scalar_ init, Functor&& f) const {
        using Scalar = typename Derived::Scalar;
        fdapde_constexpr_assert(derived().rows() > 0 && derived().cols() > 0);
        fdapde_static_assert(
          std::is_convertible_v<Scalar_ FDAPDE_COMMA Scalar>, INVALID_SCALAR_INIT_TYPE_IN_REDUX_OPERATION);
        // perform reduction loop
        return internals::linear_matrix_redux_op<Derived, Functor>::run(derived(), init, f);
    }
    constexpr auto sum() const {
        using Scalar = typename Derived::Scalar;
        if (Rows == 0 || Cols == 0) return Scalar(0);
        return redux(Scalar(0), [](Scalar tmp, Scalar x) { return tmp + x; });
    }
    constexpr auto prod() const {
        using Scalar = typename Derived::Scalar;
        if (Rows == 0 || Cols == 0) return Scalar(1);
        return redux(Scalar(1), [](Scalar tmp, Scalar x) { return tmp * x; });
    }
    constexpr auto mean() const { return derived().sum() / derived().size(); }
    constexpr auto max() const {
        using Scalar = typename Derived::Scalar;
	return redux(std::numeric_limits<Scalar>::min(), [](Scalar tmp, Scalar x) { return tmp > x ? tmp : x; });
    }
    constexpr auto min() const {
        using Scalar = typename Derived::Scalar;
        return redux(std::numeric_limits<Scalar>::max(), [](Scalar tmp, Scalar x) { return tmp < x ? tmp : x; });
    }
  
    // transpose
    constexpr Transpose<Derived> transpose() const { return Transpose<Derived>(derived()); }
    // block operations
    constexpr MatrixBlock<Rows, 1, Derived> col(int i) { return block<Rows, 1>(0, i); }
    constexpr MatrixBlock<Rows, 1, const Derived> col(int i) const {
        return MatrixBlock<Rows, 1, const Derived>(derived(), 0, i);
    }
    constexpr MatrixBlock<1, Cols, Derived> row(int i) { return block<1, Cols>(i, 0); }
    constexpr MatrixBlock<1, Cols, const Derived> row(int i) const {
        return MatrixBlock<1, Cols, const Derived>(derived(), i, 0);
    }
    template <int BlockRows, int BlockCols> constexpr MatrixBlock<BlockRows, BlockCols, Derived> block(int i, int j) {
        return MatrixBlock<BlockRows, BlockCols, Derived>(derived(), i, j);
    }
    template <int BlockRows> constexpr MatrixBlock<BlockRows, Cols, Derived> topRows() {
        return block<BlockRows, Cols>(0, 0);
    }
    template <int BlockRows> constexpr MatrixBlock<BlockRows, Cols, Derived> bottomRows() {
        return block<BlockRows, Cols>(Rows - BlockRows, 0);
    }
    template <int BlockCols> constexpr MatrixBlock<Rows, BlockCols, Derived> leftCols() {
        return block<Rows, BlockCols>(0, 0);
    }
    template <int BlockCols> constexpr MatrixBlock<Rows, BlockCols, Derived> rightCols() {
        return block<Rows, BlockCols>(0, Cols - BlockCols);
    }
    // dot product
    template <int RhsRows, int RhsCols, typename RhsDerived>
    constexpr auto dot(const MatrixBase<RhsRows, RhsCols, RhsDerived>& rhs) const {
        fdapde_static_assert(
          (RhsRows == 1 || RhsCols == 1) && ((Rows == 1 && (Cols == RhsRows || Cols == RhsCols)) ||
                                             (Cols == 1 && (Rows == RhsRows || Rows == RhsCols))),
          INVALID_OPERANDS_DIMENSIONS_FOR_DOT_PRODUCT);
        std::decay_t<typename Derived::Scalar> dot_ = 0;
        for (int i = 0; i < fdapde::max(Rows, Cols); ++i) {
            dot_ += derived().operator[](i) * rhs.derived().operator[](i);
        }
        return dot_;
    }
    // trace of matrix
    constexpr auto trace() const {
        fdapde_static_assert(Rows == Cols, CANNOT_COMPUTE_TRACE_OF_NON_SQUARE_MATRIX);
        typename Derived::Scalar trace_ = 0;
        for (int i = 0; i < Rows; ++i) trace_ += derived().operator()(i, i);
        return trace_;
    }
    // diagonal view of matrix expression
    constexpr DiagonalBlock<const Derived> diagonal() const { return DiagonalBlock<const Derived>(derived()); }
    constexpr DiagonalBlock<Derived> diagonal() { return DiagonalBlock<Derived>(derived()); }
    // triangular view of matrix expression
    template <int ViewMode> constexpr TriangularView<const Derived, ViewMode> triangular_view() const {
        return TriangularView<const Derived, ViewMode>(derived());
    }
    template <int ViewMode> constexpr TriangularView<Derived, ViewMode> triangular_view() {
        return TriangularView<Derived, ViewMode>(derived());
    }

    template <typename Dest> constexpr void copy_to(Dest& dest) const {
        fdapde_static_assert(
          std::is_invocable_v<Dest FDAPDE_COMMA int FDAPDE_COMMA int> ||
            internals::is_subscriptable<Dest FDAPDE_COMMA int>,
          DESTINATION_TYPE_MUST_EITHER_EXPOSE_A_MATRIX_LIKE_ACCESS_OPERATOR_OR_A_SUBSCRIPT_OPERATOR);
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                if constexpr (std::is_invocable_v<Dest FDAPDE_COMMA int FDAPDE_COMMA int>) {
                    dest(i, j) = derived().operator()(i, j);
                } else {
                    dest[i * derived().cols() + j] = derived().operator()(i, j);
                }
            }
        }
	return;
    }
    template <int OtherRows, int OtherCols, typename OtherDerived>
    constexpr Derived& operator+=(const MatrixBase<OtherRows, OtherCols, OtherDerived>& other) {
        fdapde_static_assert(Rows == OtherRows && Cols == OtherCols, YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES);
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) { derived().operator()(i, j) += other.derived()(i, j); }
        }
        return derived();
    }
    template <int OtherRows, int OtherCols, typename OtherDerived>
    constexpr Derived& operator-=(const MatrixBase<OtherRows, OtherCols, OtherDerived>& other) {
        fdapde_static_assert(Rows == OtherRows && Cols == OtherCols, YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES);
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) { derived().operator()(i, j) -= other.derived()(i, j); }
        }
        return derived();
    }
#ifdef __FDAPDE_HAS_EIGEN__
    // conversion to Eigen matrix
    auto as_eigen_matrix() const {
        Eigen::Matrix<typename Derived::Scalar, Rows, Cols> m;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) { m(i, j) = derived().operator()(i, j); }
        }
        return m;
    }
#endif
   protected:
    // trait to detect if Xpr is a compile-time vector
    template <typename Xpr> struct is_vector {
        static constexpr bool value = (Xpr::Cols == 1);
    };
    template <typename Xpr> using is_vector_v = is_vector<Xpr>::value;
};

// comparison operators
template <int Rows1, int Cols1, typename XprType1, int Rows2, int Cols2, typename XprType2>
constexpr bool
operator==(const MatrixBase<Rows1, Cols1, XprType1>& op1, const MatrixBase<Rows2, Cols2, XprType2>& op2) {
    fdapde_static_assert(Rows1 == Rows2 && Cols1 == Cols2, YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES);
    for (int i = 0; i < Rows1; ++i) {
        for (int j = 0; j < Cols1; ++j) {
            if (op1.derived()(i, j) != op2.derived()(i, j)) return false;
        }
    }
    return true;
}
template <int Rows1, int Cols1, typename XprType1, int Rows2, int Cols2, typename XprType2>
constexpr bool
operator!=(const MatrixBase<Rows1, Cols1, XprType1>& op1, const MatrixBase<Rows2, Cols2, XprType2>& op2) {
    fdapde_static_assert(Rows1 == Rows2 && Cols1 == Cols2, YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES);
    for (int i = 0; i < Rows1; ++i) {
        for (int j = 0; j < Cols1; ++j) {
            if (op1.derived()(i, j) == op2.derived()(i, j)) return false;
        }
    }
    return true;
}
template <int Rows1, int Cols1, typename XprType1, int Rows2, int Cols2, typename XprType2>
constexpr bool almost_equal(
  const MatrixBase<Rows1, Cols1, XprType1>& op1, const MatrixBase<Rows2, Cols2, XprType2>& op2, double epsilon = 1e-7) {
    fdapde_static_assert(Rows1 == Rows2 && Cols1 == Cols2, YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES);
    fdapde_static_assert(
      std::is_same_v<typename XprType1::Scalar FDAPDE_COMMA typename XprType2::Scalar>,
      YOU_MIXED_MATRICES_OF_DIFFERENT_SCALAR_TYPES);
    using Scalar_ = typename XprType1::Scalar;
    for (int i = 0; i < Rows1; ++i) {
        for (int j = 0; j < Cols1; ++j) {
            Scalar_ a = op1.derived()(i, j);
            Scalar_ b = op2.derived()(i, j);
            if (!(std::fabs(a - b) < epsilon ||
                  std::fabs(a - b) < ((std::fabs(a) < std::fabs(b) ? std::fabs(b) : std::fabs(a)) * epsilon))) {
                return false;
            }
        }
    }
    return true;
}

// maps an existing array of data to a cepxr::Matrix. This can be used also to integrate Eigen with cexpr linear algebra
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_ = ColMajor>
class Map : public MatrixBase<Rows_, Cols_, Map<Scalar_, Rows_, Cols_, StorageOrder_>> {
    fdapde_static_assert(Rows_ > 0 && Cols_ > 0, YOU_ARE_MAPPING_DATA_TO_AN_EMPTY_MATRIX);
   public:
    using XprType = Map<Scalar_, Rows_, Cols_, StorageOrder_>;
    using Base = MatrixBase<Rows_, Cols_, XprType>;
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int NestAsRef = 1;

    constexpr Map() : data_() { }
    constexpr Map(Scalar_* data) : data_(data) { }
    constexpr Map(Scalar_* data, int outer_stride, int inner_stride) :
        data_(data), outer_stride_(outer_stride), inner_stride_(inner_stride) { }
    // const access
    constexpr Scalar operator()(int i, int j) const {
        fdapde_assert(i < Rows && j < Cols);
        return data_[i * rowStride() + j * colStride()];
    }
    constexpr Scalar operator[](int i) const
        requires(Cols == 1 || Rows == 1) {
        fdapde_static_assert(Cols == 1 || Rows == 1, THIS_METHOD_IS_ONLY_FOR_CONSTEXPR_ROW_OR_COLUMN_VECTORS);
        return data_[i * innerStride()];
    }
    // non-const access
    constexpr Scalar& operator()(int i, int j) {
        fdapde_assert(i < Rows && j < Cols);
        return data_[i * rowStride() + j * colStride()];
    }
    constexpr Scalar& operator[](int i)
        requires(Cols == 1 || Rows == 1) {
        fdapde_static_assert(Cols == 1 || Rows == 1, THIS_METHOD_IS_ONLY_FOR_CONSTEXPR_ROW_OR_COLUMN_VECTORS);
        return data_[i * innerStride()];
    }
    constexpr int rows() const { return Rows; }
    constexpr int cols() const { return Cols; }
    constexpr int innerStride() const { return inner_stride_; }
    constexpr int outerStride() const {
        return outer_stride_ != 1 ? outer_stride_ : (StorageOrder_ == RowMajor ? Cols : Rows);
    }
    constexpr int rowStride() const { return StorageOrder_ == RowMajor ? outerStride() : innerStride(); }
    constexpr int colStride() const { return StorageOrder_ == RowMajor ? innerStride() : outerStride(); }
    constexpr const Scalar_* data() const { return data_; }
    constexpr Scalar_* data() { return data_; }
    // assignment operator
    template <int RhsRows_, int RhsCols_, typename RhsXprType>
    constexpr XprType& operator=(const MatrixBase<RhsRows_, RhsCols_, RhsXprType>& rhs) {
        fdapde_static_assert(
          Rows == RhsRows_ && Cols == RhsCols_ &&
            std::is_convertible_v<typename RhsXprType::Scalar FDAPDE_COMMA Scalar>,
          INVALID_MATRIX_ASSIGNMENT);
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) { operator()(i, j) = rhs.derived()(i, j); }
        }
        return *this;
    }
   private:
    Scalar_* data_ = nullptr;
    int outer_stride_ = 1;   // increment between two consecutive rows (RowMajor) or columns (ColMajor)
    int inner_stride_ = 1;   // increment between two consecutive entries within a row (RowMajor) or column (ColMajor)
};

}   // namespace fdapde

#endif   // _FDAPDE_MATRIX_BASE_H__
