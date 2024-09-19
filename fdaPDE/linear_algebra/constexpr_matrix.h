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

#ifndef __CONSTEXPR_MATRIX_H__
#define __CONSTEXPR_MATRIX_H__

#include <array>

namespace fdapde {
namespace cexpr {

  // visitors, rowwise, colwwise iterators, sum, max, min, ...
  
template <int Rows, int Cols, typename Derived> struct MatrixBase;

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
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(
          Derived::Cols == 1 || Derived::Rows == 1, THIS_METHOD_IS_ONLY_FOR_CONSTEXPR_ROW_OR_COLUMN_VECTORS);
        return xpr_[i];
    }
    constexpr int rows() const { return xpr_.cols(); }
    constexpr int cols() const { return xpr_.rows(); }
   protected:
    typename internals::ref_select<const Derived>::type xpr_;
};

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
                if constexpr (ViewMode == fdapde::Lower) { row = i; col = j; }
                if constexpr (ViewMode == fdapde::Upper) { row = j; col = i; }
                xpr_(row, col) = rhs(row, col);
            }
        }
	// assign diagonal
        for (int i = 0; i < Rows_; ++i) { xpr_(i, i) = rhs(i, i); }
        return *this;
    }
   private:
    typename internals::ref_select<const Derived>::type xpr_;
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
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(
          (Lhs::Cols == 1 && Rhs::Cols == 1) || (Lhs::Rows == 1 && Rhs::Rows == 1),
          THIS_METHOD_IS_ONLY_FOR_CONSTEXPR_ROW_OR_COLUMN_VECTORS);
        return op_(lhs_[i], rhs_[i]);
    }
    constexpr int rows() const { return lhs_.rows(); }
    constexpr int cols() const { return lhs_.cols(); }
   protected:
    typename internals::ref_select<const Lhs>::type lhs_;
    typename internals::ref_select<const Rhs>::type rhs_;
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

template <typename Lhs, typename Rhs, typename BinaryOperation>
struct MatrixCoeffWiseOp :
    public MatrixBase<
      std::conditional_t<std::is_arithmetic_v<Lhs>, Rhs, Lhs>::Rows,
      std::conditional_t<std::is_arithmetic_v<Lhs>, Rhs, Lhs>::Cols, MatrixCoeffWiseOp<Lhs, Rhs, BinaryOperation>> {
    using CoeffType_ = std::conditional_t<std::is_arithmetic_v<Lhs>, Lhs, Rhs>;
   public:
    using XprType = std::conditional_t<std::is_arithmetic_v<Lhs>, Rhs, Lhs>;
    using Base = MatrixCoeffWiseOp<Lhs, Rhs, BinaryOperation>;
    using Scalar = decltype(std::declval<BinaryOperation>().operator()(
      std::declval<typename XprType::Scalar>(), std::declval<CoeffType_>()));
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    constexpr MatrixCoeffWiseOp(const XprType& xpr, CoeffType_ coeff, BinaryOperation op) :
        xpr_(xpr), coeff_(coeff), op_(op) { }
    constexpr Scalar operator()(int i, int j) const { return op_(xpr_(i, j), coeff_); }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(
          XprType::Rows == 1 || XprType::Cols == 1, THIS_METHOD_IS_ONLY_FOR_CONSTEXPR_ROW_OR_COLUMN_VECTORS);
        return op_(xpr_[i], coeff_);
    }
    constexpr int rows() const { return xpr_.rows(); }
    constexpr int cols() const { return xpr_.cols(); }
   protected:
    typename internals::ref_select<const XprType>::type xpr_;
    CoeffType_ coeff_;
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
    return MatrixCoeffWiseOp<Coeff, XprType, std::multiplies<>> {rhs.derived(), lhs, std::multiplies<>()};
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
    constexpr Scalar operator[](int i) const {
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
    typename internals::ref_select<const Lhs>::type lhs_;
    typename internals::ref_select<const Rhs>::type rhs_;
};
template <typename Lhs, typename Rhs>
constexpr MatrixProduct<Lhs, Rhs>
operator*(const MatrixBase<Lhs::Rows, Lhs::Cols, Lhs>& op1, const MatrixBase<Rhs::Rows, Rhs::Cols, Rhs>& op2) {
    return MatrixProduct<Lhs, Rhs> {op1.derived(), op2.derived()};
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
    // block assignment
    constexpr MatrixBlock& operator=(const MatrixBlock& other) {
        fdapde_static_assert(Derived::ReadOnly == 0, BLOCK_ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION_IS_INVALID);
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) { xpr_(start_row_ + i, start_col_ + j) = other(i, j); }
        }
        return *this;
    };
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
    typename internals::ref_select<Derived>::type xpr_;
};

template <typename Scalar_, int Rows_, int Cols_, int NestAsRefBit_ = 1>
class Matrix : public MatrixBase<Rows_, Cols_, Matrix<Scalar_, Rows_, Cols_, NestAsRefBit_>> {
    fdapde_static_assert(Rows_ > 0 && Cols_ > 0, EMPTY_MATRIX_IS_ILL_FORMED);
   public:
    using Base = MatrixBase<Rows_, Cols_, Matrix<Scalar_, Rows_, Cols_, NestAsRefBit_>>;
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int NestAsRef = NestAsRefBit_;   // whether to store this node by ref or by copy in an expression
    static constexpr int ReadOnly = 0;

    constexpr Matrix() : data_() {};
    constexpr explicit Matrix(const std::array<Scalar, Rows * Cols>& data) : data_(data) { }
    template <typename Derived>
    constexpr Matrix(const MatrixBase<Rows_, Cols_, Derived>& xpr) : data_() {
        fdapde_static_assert(
          std::is_convertible_v<typename Derived::Scalar FDAPDE_COMMA Scalar>,
          INVALID_SCALAR_TYPES_CONVERSION_BETWEEN_MATRICES);
        fdapde_static_assert(
          Derived::Rows == Rows && Derived::Cols == Cols,
          YOU_ARE_TRYING_TO_CONSTRUCT_A_MATRIX_WITH_ANOTHER_MATRIX_OF_DIFFERENT_SIZE);
        for (int i = 0; i < rows(); ++i) {
            for (int j = 0; j < cols(); ++j) { data_[i * Cols + j] = xpr.derived().operator()(i, j); }
        }
    }
    template <typename Callable>
    constexpr explicit Matrix(Callable callable)
        requires(std::is_invocable_v<Callable>)
        : data_() {
        fdapde_static_assert(
          std::is_convertible_v<typename decltype(std::function {
            callable})::result_type FDAPDE_COMMA std::array<Scalar FDAPDE_COMMA Rows * Cols>>,
          CALLABLE_DOES_NOT_RETURN_SOMETHING_CONVERTIBLE_TO_AN_ARRAY_OF_SCALAR);
        data_ = callable();
    }
    constexpr explicit Matrix(const Scalar_ (&data)[Rows * Cols]) : data_() {
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) { data_[i * Cols + j] = data[i * Cols + j]; }
        }
    }
    constexpr explicit Matrix(Scalar x) : data_() {   // 1D point constructor
        fdapde_static_assert(Rows * Cols == 1, THIS_METHOD_IS_ONLY_FOR_MATRICES_WITH_ONE_ELEMENT);
	data_[0] = x;
    }
    constexpr explicit Matrix(Scalar x, Scalar y) : data_() {   // 2D point constructor
        fdapde_static_assert(Rows * Cols == 2, THIS_METHOD_IS_ONLY_FOR_MATRICES_WITH_TWO_ELEMENTS);
	data_ = {x, y};
    }
    constexpr explicit Matrix(Scalar x, Scalar y, Scalar z) : data_() {   // 3D point constructor
        fdapde_static_assert(Rows * Cols == 3, THIS_METHOD_IS_ONLY_FOR_MATRICES_WITH_THREE_ELEMENTS);
	data_ = {x, y, z};
    }
    // conversion from Eigen matrix
    template <typename Scalar__, int Rows__, int Cols__>
    explicit Matrix(const Eigen::Matrix<Scalar__, Rows__, Cols__>& other) {
        fdapde_static_assert(
          Rows__ != Dynamic && Cols__ != Dynamic && Rows__ == Rows && Cols__ == Cols &&
            std::is_convertible_v<Scalar_ FDAPDE_COMMA Scalar>,
          INVALID_CONVERSION_FROM_EIGEN_MATRIX_TO_CONSTEXPR_MATRIX);
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) { operator()(i, j) = other(i, j); }
        }
    }
    // static constructors
    static constexpr Matrix<Scalar, Rows, Cols> Constant(Scalar c) {
        Matrix<Scalar, Rows, Cols> m;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) { m(i, j) = c; }
        }
        return m;
    }
    static constexpr Matrix<Scalar, Rows, Cols> Zero() { return Constant(Scalar(0)); }
    static constexpr Matrix<Scalar, Rows, Cols> Ones() { return Constant(Scalar(1)); }
    // const access
    constexpr Scalar operator()(int i, int j) const { return data_.at(i * Cols + j); }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(Cols == 1 || Rows == 1, THIS_METHOD_IS_ONLY_FOR_CONSTEXPR_ROW_OR_COLUMN_VECTORS);
        return data_.at(i);
    }
    // non-const access
    constexpr Scalar& operator()(int i, int j) { return data_[i * Cols + j]; }
    constexpr Scalar& operator[](int i) {
        fdapde_static_assert(Cols == 1 || Rows == 1, THIS_METHOD_IS_ONLY_FOR_CONSTEXPR_ROW_OR_COLUMN_VECTORS);
        return data_[i];
    }
    constexpr int rows() const { return Rows; }
    constexpr int cols() const { return Cols; }
    constexpr const Scalar* data() const { return data_.data(); }
    Scalar* data() { return data_.data(); }
    // assignment operator
    template <int RhsRows_, int RhsCols_, typename RhsXprType>
    constexpr Matrix<Scalar_, Rows_, Cols_, NestAsRefBit_>&
    operator=(const MatrixBase<RhsRows_, RhsCols_, RhsXprType>& rhs) {
        fdapde_static_assert(
          Rows == RhsRows_ && Cols == RhsCols_ &&
            std::is_convertible_v<typename RhsXprType::Scalar FDAPDE_COMMA Scalar>,
          INVALID_RHS_DIMENSIONS_OR_YOU_ARE_TRYING_TO_ASSIGN_A_RHS_WITH_NON_CONVERTIBLE_SCALAR_TYPE);
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) { operator()(i, j) = rhs.derived()(i, j); }
        }
        return *this;
    }
    // assignment from std::array
    constexpr Matrix<Scalar_, Rows_, Cols_, NestAsRefBit_>& operator=(const std::array<Scalar_, Rows_ * Cols_>& rhs) {
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) { operator()(i, j) = rhs[i * Cols + j]; }
        }
        return *this;
    }
    // assignment from Eigen matrix
    template <typename Derived>
    Matrix<Scalar_, Rows_, Cols_, NestAsRefBit_>& operator=(const Eigen::MatrixBase<Derived>& rhs) {
        fdapde_static_assert(
          Derived::RowsAtCompileTime != Dynamic && Derived::ColsAtCompileTime != Dynamic &&
            std::is_convertible_v<typename Derived::Scalar FDAPDE_COMMA Scalar>,
          CANNOT_ASSIGN_FROM_EIGEN_HEAP_ALLOCATED_MATRIX_OR_INVALID_SCALAR_TYPE);
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) { operator()(i, j) = rhs.derived()(i, j); }
        }
        return *this;
    }
    constexpr void setConstant(Scalar c) {
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) { operator()(i, j) = c; }
        }
	return;
    }
    constexpr void setZero() { setConstant(Scalar(0)); }
    constexpr void setOnes() { setConstant(Scalar(1)); }
   private:
    std::array<Scalar, Rows * Cols> data_;
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
    typename internals::ref_select<Derived>::type xpr_;
};

template <int Rows, int Cols, typename Derived> struct MatrixBase {
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
            for (int j = 0; j < derived().cols(); ++j) { norm_ += std::pow(derived().operator()(i, j), 2); }
        }
        return norm_;
    }
    constexpr auto norm() const { return std::sqrt(squared_norm()); }
    // maximum norm (L^\infinity norm)
    constexpr auto inf_norm() const {
        using Scalar = typename Derived::Scalar;
        Scalar norm_ = std::numeric_limits<Scalar>::min();
        for (int i = 0; i < derived().rows(); ++i) {
            for (int j = 0; j < derived().cols(); ++j) {
                Scalar tmp = std::abs(derived().operator()(i, j));
                if (tmp > norm_) norm_ = tmp;
            }
        }
        return norm_;
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
    template <int BlockRows> constexpr MatrixBlock<BlockRows, Cols, Derived> topRows(int i) {
        return block<BlockRows, Cols>(i, 0);
    }
    template <int BlockRows> constexpr MatrixBlock<BlockRows, Cols, Derived> bottomRows(int i) {
        return block<BlockRows, Cols>(Rows - i - 1, 0);
    }
    template <int BlockCols> constexpr MatrixBlock<Rows, BlockCols, Derived> leftCols(int i) {
        return block<Rows, BlockCols>(0, i);
    }
    template <int BlockCols> constexpr MatrixBlock<Rows, BlockCols, Derived> rightCols(int i) {
        return block<Rows, BlockCols>(0, Cols - i - 1);
    }
    // dot product
    template <int RhsRows, typename RhsDerived>
    constexpr auto dot(const MatrixBase<RhsRows, 1, RhsDerived>& rhs) const {
        fdapde_static_assert(
          (Rows == 1 && Cols == RhsRows) || (Cols == 1 && Rows == RhsRows),
          INVALID_OPERANDS_DIMENSIONS_FOR_DOT_PRODUCT);
        typename Derived::Scalar dot_ = 0;
        for (int i = 0; i < Cols; ++i) { dot_ += derived().operator[](i) * rhs.derived().operator[](i); }
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
            fdapde::is_subscriptable<Dest FDAPDE_COMMA int>,
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

template <int Size_> struct PermutationMatrix : public MatrixBase<Size_, Size_, PermutationMatrix<Size_>> {
    using Base = MatrixBase<Size_, Size_, PermutationMatrix<Size_>>;
    using Scalar = int;
    using XprType = PermutationMatrix<Size_>;
    static constexpr int Rows = Size_;
    static constexpr int Cols = Size_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    constexpr PermutationMatrix() = default;
    constexpr explicit PermutationMatrix(const std::array<int, Size_>& permutation) : permutation_(permutation) { }
    constexpr int rows() const { return Size_; }
    constexpr int cols() const { return Size_; }
    // left multiplication by permutation matrix
    template <int RhsRows, int RhsCols, typename RhsType>
    constexpr Matrix<typename RhsType::Scalar, Rows, RhsCols>
    operator*(const MatrixBase<RhsRows, RhsCols, RhsType>& rhs) const {
        fdapde_static_assert(Cols == RhsRows, INVALID_OPERAND_DIMENSIONS_FOR_MATRIX_MATRIX_PRODUCT);
        using Scalar = typename RhsType::Scalar;
        Matrix<Scalar, Rows, RhsCols> permuted;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < RhsCols; ++j) { permuted(i, j) = rhs.derived().operator()(permutation_[i], j); }
        }
        return permuted;
    }
    // right multiplication by permutation matrix
    template <int RhsRows, int RhsCols, typename RhsType>
    constexpr friend Matrix<typename RhsType::Scalar, Rows, RhsCols>
    operator*(const MatrixBase<RhsRows, RhsCols, RhsType>& lhs, const PermutationMatrix<Size_>& rhs) {
        fdapde_static_assert(Cols == RhsRows, INVALID_OPERANDS_DIMENSION_FOR_MATRIX_MATRIX_PRODUCT);
        using Scalar = typename RhsType::Scalar;
        Matrix<Scalar, Rows, RhsCols> permuted;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < RhsCols; ++j) { permuted(j, i) = lhs.derived().operator()(j, rhs.permutation()[i]); }
        }
        return permuted;
    }
    constexpr const std::array<int, Size_>& permutation() const { return permutation_; }
   private:
    std::array<int, Size_> permutation_;
};

// alias export for constexpr-enabled vectors
template <typename Scalar_, int Rows_> using Vector = Matrix<Scalar_, Rows_, 1>;

template <typename Matrix, typename Rhs> constexpr auto backward_sub(const Matrix& A, const Rhs& b) {
    fdapde_static_assert(
      std::is_same_v<typename Matrix::Scalar FDAPDE_COMMA typename Rhs::Scalar>, OPERANDS_HAVE_DIFFERENT_SCALAR_TYPES);
    // check dimensions
    using Scalar = typename Matrix::Scalar;
    constexpr int rows = Matrix::Rows;
    Vector<Scalar, rows> res;
    int i = rows - 1;
    res[i] = b[i] / A(i, i);
    i--;
    for (; i >= 0; --i) {
        Scalar tmp = 0;
        for (int j = i + 1; j < rows; ++j) tmp += A(i, j) * res[j];
        res[i] = 1. / A(i, i) * (b[i] - tmp);
    }
    return res;
}

template <typename Matrix, typename Rhs> constexpr auto forward_sub(const Matrix& A, const Rhs& b) {
    fdapde_static_assert(
      std::is_same_v<typename Matrix::Scalar FDAPDE_COMMA typename Rhs::Scalar>, OPERANDS_HAVE_DIFFERENT_SCALAR_TYPES);
    // check dimensions
    using Scalar = typename Matrix::Scalar;
    constexpr int rows = Matrix::Rows;
    Vector<Scalar, rows> res;
    int i = 0;
    res[i] = b[i] / A(i, i);
    i++;
    for (; i < rows; ++i) {
        Scalar tmp = 0;
        for (int j = 0; j < i; ++j) tmp += A(i, j) * res[j];
        res[i] = 1. / A(i, i) * (b[i] - tmp);
    }
    return res;
}

// LU factorization of matrix with partial pivoting
template <typename MatrixType> class PartialPivLU {
    fdapde_static_assert(MatrixType::Rows == MatrixType::Cols, LU_FACTORIZATION_IS_ONLY_FOR_SQUARE_INVERTIBLE_MATRICES);
    static constexpr int Size = MatrixType::Rows;
    using Scalar = typename MatrixType::Scalar;
    MatrixType m_;
    PermutationMatrix<Size> P_;
   public:
    constexpr PartialPivLU() : m_(), P_() {};
    template <typename XprType> constexpr PartialPivLU(const MatrixBase<Size, Size, XprType>& m) : m_() { compute(m); }

    // computes the LU factorization of matrix m with partial (row) pivoting
    template <typename XprType> constexpr void compute(const MatrixBase<Size, Size, XprType>& m) {
        m_ = m;
        std::array<int, Size> P;
        for (int i = 0; i < Size; ++i) { P[i] = i; }
        int pivot_index = 0;
        int h, k;
        for (int i = 0; i < Size - 1; ++i) {
            // find pivotal element
            Scalar pivot = -std::numeric_limits<Scalar>::infinity();
            for (int j = i; j < Size; ++j) {
                Scalar abs_ = std::abs(m_(P[j], i));
                if (pivot < abs_) {
                    pivot = abs_;
                    pivot_index = j;
                }
            }
            // perform gaussian elimination step in place
            for (int j = i; j < Size; ++j) {
                if (P[j] != P[pivot_index]) {   // avoid to subtract row with itself
                    Scalar l = m_(P[j], i) / m_(P[pivot_index], i);
                    m_(P[j], i) = l;
                    for (int k = i + 1; k < Size; ++k) { m_(P[j], k) = m_(P[j], k) - l * m_(P[pivot_index], k); }
                }
            }
            // swap rows
            h = P[i], k = P[pivot_index];
            P[pivot_index] = h;
            P[i] = k;
        }
        P_ = PermutationMatrix<Size>(P);
	m_ = P_ * m_;
    }
    constexpr PermutationMatrix<Size> P() const { return P_; }
    // solve linear system Ax = b using A factorization PA = LU
    template <typename RhsType> constexpr Matrix<Scalar, Size, 1> solve(const RhsType& rhs) {
        fdapde_static_assert(
          rhs.rows() == Size && rhs.cols() == 1 && std::is_same_v<Scalar FDAPDE_COMMA typename RhsType::Scalar>,
          INVALID_RHS_VECTOR_OR_OPERANDS_HAVE_DIFFERENT_SCALAR_TYPES);
        Matrix<Scalar, Size, 1> x;
        // evaluate U^{-1} * (L^{-1} * (P * rhs))
        x = P_ * rhs;
        x = forward_sub(m_.template triangular_view<UnitLower>(), x);
        x = backward_sub(m_.template triangular_view<Upper>(), x);
        return x;
    }
};

// maps an existing array of data to a cepxr::Matrix. This can be used also to integrate Eigen with cexpr linear algebra
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_ = fdapde::ColMajor>
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
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(Cols == 1 || Rows == 1, THIS_METHOD_IS_ONLY_FOR_CONSTEXPR_ROW_OR_COLUMN_VECTORS);
        return data_[i * innerStride()];
    }
    // non-const access
    constexpr Scalar& operator()(int i, int j) {
        fdapde_assert(i < Rows && j < Cols);
        return data_[i * rowStride() + j * colStride()];
    }
    constexpr Scalar& operator[](int i) {
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

}   // namespace cexpr
}   // namespace fdapde

#endif   // _CONSTEXPR_MATRIX_H__
