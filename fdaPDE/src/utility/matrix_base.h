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

#include <iomanip>

#include "header_check.h"
// #include "square_matrix_base.h"

namespace fdapde {

// TODO: visitors, rowwise, colwwise, coeffwise iterators ... support Dynamic

// forward declaration
template <int Rows, int Cols, typename Derived> struct MatrixBase;
template <typename Derived> struct TransposeView;
template <typename Lhs, typename Rhs, typename BinaryOperation> struct MatrixBinOp;
template <typename Lhs, typename Rhs, typename BinaryOperation> struct MatrixCoeffWiseOp;
template <typename Lhs, typename Rhs> struct MatrixProduct;
template <typename Lhs, typename Rhs> struct MatrixKroneckerProduct;
template <int BlockRows, int BlockCols, typename Derived> class MatrixBlockView;

[[maybe_unused]] constexpr int RowMajor = 0;
[[maybe_unused]] constexpr int ColMajor = 1;

// matrix flags enumerator
enum class matrix_flags {
    none            = 0x0000,
    square          = 0x0001,
    symmetric       = 0x0002,
    skew_symmetric  = 0x0004
};

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

} // namespace internals

// is_view trait
namespace internals {

// default: not a view unless specialized
template <typename T>
struct is_view : std::false_type {};

// helper variable template
template <typename T>
static constexpr bool is_view_v = is_view<T>::value;

template <typename Derived>
struct is_view<TransposeView<Derived>> : std::true_type {};
template <int BlockRows, int BlockCols, typename Derived>
struct is_view<MatrixBlockView<BlockRows, BlockCols, Derived>> : std::true_type {};

}

// is_xpr_temp trait
namespace internals {

// default: not an xpr template unless specialized
template <typename T>
struct is_xpr_temp : std::false_type {};

// helper variable template
template <typename T>
static constexpr bool is_xpr_temp_v = is_xpr_temp<T>::value;

template <typename Lhs, typename Rhs, typename BinaryOperation>
struct is_xpr_temp<MatrixBinOp<Lhs, Rhs, BinaryOperation>> : std::true_type {};
template <typename Lhs, typename Rhs, typename BinaryOperation>
struct is_xpr_temp<MatrixCoeffWiseOp<Lhs, Rhs, BinaryOperation>> : std::true_type {};
template <typename Lhs, typename Rhs>
struct is_xpr_temp<MatrixProduct<Lhs, Rhs>> : std::true_type {};
template <typename Lhs, typename Rhs>
struct is_xpr_temp<MatrixKroneckerProduct<Lhs, Rhs>> : std::true_type {};

}

// is_symmetric trait
namespace internals {

// default: nothing is triangular
template <typename T>
struct is_symmetric : std::false_type {};

// helper variable templates
template <typename T>
static constexpr bool is_symmetric_v = is_symmetric<T>::value;

}

// transpose view
template <typename Derived>
struct TransposeView : public MatrixBase<Derived::Cols, Derived::Rows, TransposeView<Derived>> {
    using Base = MatrixBase<Derived::Cols, Derived::Rows, TransposeView<Derived>>;
    using Scalar = typename Derived::Scalar;
    static constexpr int Rows = Derived::Cols;
    static constexpr int Cols = Derived::Rows;
    static constexpr bool NestAsRef = false;
    static constexpr bool ReadOnly = true;
    static constexpr int XprBits = Derived::XprBits;

    constexpr TransposeView(const Derived& xpr) : xpr_(xpr) { }
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


// matrix binary operations
template <typename Lhs, typename Rhs, typename BinaryOperation>
struct MatrixBinOp : public MatrixBase<Lhs::Rows, Lhs::Cols, MatrixBinOp<Lhs, Rhs, BinaryOperation>> {
    fdapde_static_assert(Lhs::Rows == Rhs::Rows && Lhs::Cols == Rhs::Cols, YOU_MIXED_MATRICES_OF_DIFFERENT_SIZES);
    using Base = MatrixBase<Lhs::Rows, Lhs::Cols, MatrixBinOp<Lhs, Rhs, BinaryOperation>> ;
    using Scalar = decltype(std::declval<BinaryOperation>().operator()(
      std::declval<typename Lhs::Scalar>(), std::declval<typename Rhs::Scalar>()));
    static constexpr int Rows = Lhs::Rows;
    static constexpr int Cols = Lhs::Cols;
    static constexpr bool NestAsRef = false;
    static constexpr bool ReadOnly = true;
    static constexpr int XprBits = Rhs::XprBits & Lhs::XprBits;

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

// matrix coefficients-wise operations
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
    static constexpr bool NestAsRef = false;
    static constexpr bool ReadOnly = true;
    static constexpr int XprBits = XprType::XprBits;

    constexpr MatrixCoeffWiseOp(const Lhs& lhs, const Rhs& rhs, BinaryOperation op) :
        lhs_(lhs), rhs_(rhs), op_(op) { }
    constexpr Scalar operator()(int i, int j) const {
        if constexpr (is_coeff_lhs) { return op_(lhs_, rhs_(i, j)); }
        if constexpr (!is_coeff_lhs) { return op_(lhs_(i, j), rhs_); }
    }
    constexpr Scalar operator[](int i) const
    requires(XprType::Rows == 1 || XprType::Cols == 1) {
	    if constexpr (is_coeff_lhs) { return op_(lhs_, rhs_[i]); }
        if constexpr (!is_coeff_lhs) { return op_(lhs_[i], rhs_); }
    }
    constexpr int rows() const { return xpr().rows(); }
    constexpr int cols() const { return xpr().cols(); }
    constexpr const Lhs_& lhs() const { return lhs_; }
    constexpr const Rhs_& rhs() const { return rhs_; }
    constexpr const BinaryOperation_& functor() const { return op_; }
   protected:
    const XprType& xpr() const {
        if constexpr (is_coeff_lhs) return rhs_;
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

// matrix product
template <typename Lhs, typename Rhs>
struct MatrixProduct : public MatrixBase<Lhs::Rows, Rhs::Cols, MatrixProduct<Lhs, Rhs>> {
    fdapde_static_assert(Lhs::Cols == Rhs::Rows, INVALID_OPERAND_DIMENSIONS_FOR_MATRIX_MATRIX_PRODUCT);
    using Base = MatrixBase<Lhs::Rows, Rhs::Cols, MatrixProduct<Lhs, Rhs>>;
    using Scalar = decltype(std::declval<typename Lhs::Scalar>() * std::declval<typename Rhs::Scalar>());
    static constexpr int Rows = Lhs::Rows;
    static constexpr int Cols = Rhs::Cols;
    static constexpr bool NestAsRef = false;
    static constexpr bool ReadOnly = true;
    static constexpr int XprBits = (Lhs::Rows == Rhs::Cols) ? int(matrix_flags::square) : int(matrix_flags::none);

    constexpr MatrixProduct(const Lhs& lhs, const Rhs& rhs) : lhs_(lhs), rhs_(rhs) { }
    constexpr Scalar operator()(int i, int j) const {
        Scalar tmp = 0;
        for (int k = 0; k < Lhs::Cols; ++k) tmp += lhs_(i, k) * rhs_(k, j);
        return tmp;
    }
    constexpr Scalar operator[](int i) const
    requires(Lhs::Rows == 1 || Rhs::Cols == 1) {
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
    static constexpr bool NestAsReaf = false;
    static constexpr bool ReadOnly = true;

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
class MatrixBlockView : public MatrixBase<BlockRows_, BlockCols_, MatrixBlockView<BlockRows_, BlockCols_, Derived>> {
    fdapde_static_assert(
      BlockRows_ > 0 && BlockCols_ > 0 && BlockRows_ <= Derived::Rows && BlockCols_ <= Derived::Cols,
      INVALID_BLOCK_SIZES);
   public:
    using Base = MatrixBase<BlockRows_, BlockCols_, MatrixBlockView<BlockRows_, BlockCols_, Derived>>;
    using Scalar = typename Derived::Scalar;
    static constexpr int Rows = BlockRows_;
    static constexpr int Cols = BlockCols_;
    static constexpr bool NestAsRef = false;
    static constexpr bool ReadOnly = Derived::ReadOnly;
    static constexpr bool IsExpressionTemplate = false;
    static constexpr int XprBits = (BlockRows_ == BlockCols_) ? int(matrix_flags::square) : int(matrix_flags::none);


    constexpr MatrixBlockView(Derived& xpr, int i) :
        start_row_(BlockRows_ == 1 ? i : 0), start_col_(BlockCols_ == 1 ? i : 0), xpr_(xpr) {
        fdapde_static_assert(BlockRows_ == 1 || BlockCols_ == 1, THIS_METHOD_IS_ONLY_FOR_ROW_AND_COLUMN_BLOCKS);
        fdapde_constexpr_assert(
          i >= 0 && ((BlockRows_ == 1 && i < xpr_.rows()) || (BlockCols_ == 1 && i < xpr_.cols())));
    }
    constexpr MatrixBlockView(Derived& xpr, int start_row, int start_col) :
        start_row_(start_row), start_col_(start_col), xpr_(xpr) { }
    constexpr int rows() const { return Rows; }
    constexpr int cols() const { return Cols; }
    constexpr Scalar operator()(int i, int j) const { return xpr_(start_row_ + i, start_col_ + j); }
    constexpr Scalar operator[](int i) const
    requires(BlockRows_ == 1 || BlockCols_ == 1) {
        if constexpr (Rows == 1) return xpr_(start_row_, start_col_ + i);
        if constexpr (Cols == 1) return xpr_(start_row_ + i, start_col_);
    }
    constexpr Scalar& operator()(int i, int j) { return xpr_(start_row_ + i, start_col_ + j); }
    constexpr Scalar& operator[](int i)
    requires(BlockRows_ == 1 || BlockCols_ == 1) {
        if constexpr (Rows == 1) return xpr_(start_row_, start_col_ + i);
        if constexpr (Cols == 1) return xpr_(start_row_ + i, start_col_);
    }
    // block assignment
    constexpr MatrixBlockView& operator=(const MatrixBlockView& other) {
        fdapde_static_assert(Derived::ReadOnly == 0, BLOCK_ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION_IS_INVALID);
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) { xpr_(start_row_ + i, start_col_ + j) = other(i, j); }
        }
        return *this;
    }
    constexpr MatrixBlockView(const MatrixBlockView& other) :
        start_row_(other.start_row_), start_col_(other.start_col_), xpr_(other.xpr_) { }
    template <int Rows_, int Cols_, typename RhsType>
    constexpr MatrixBlockView<BlockRows_, BlockCols_, Derived>& operator=(const MatrixBase<Rows_, Cols_, RhsType>& rhs) {
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

} // namespace internals


// matrix base
template <int Rows, int Cols, typename Derived>
struct MatrixBase {
    // TODO: this does not work but shouldn't we have a check like this?
    // fdapde_static_assert(Derived::Cols == Cols && Derived::Rows == Rows, INVALID_DIMENSIONS);

    #ifdef __FDAPDE_HAS_EIGEN__ // compatibility with Eigen types
        static constexpr int RowsAtCompileTime = Rows;
        static constexpr int ColsAtCompileTime = Cols;
    #endif

    // access to derived
    constexpr const Derived& derived() const { return static_cast<const Derived&>(*this); }
    constexpr Derived& derived() { return static_cast<Derived&>(*this); }

    // dimensions
    [[nodiscard]] constexpr int size() const { return Rows * Cols; }
    [[nodiscard]] constexpr int rows() const { return Rows; }
    [[nodiscard]] constexpr int cols() const { return Cols; }

    // send matrix to ostream (this is not constexpr evaluable)
    friend std::ostream& operator<<(std::ostream& os, const MatrixBase& m) {
        std::cout << "[[ ";
        if constexpr (internals::is_view_v<Derived>) std::cout << "(View) ";
        if constexpr (internals::is_xpr_temp_v<Derived>) std::cout << "(Xpr template) ";
        if constexpr (internals::is_vector_like_v<Derived>) std::cout << "VectorLike ";
        if constexpr (internals::is_symmetric_v<Derived>) std::cout << "Symmetric ";
        // if constexpr (Derived::XprBits & int(matrix_flags::symmetric)) std::cout << "Symmetric ";
        if constexpr (Derived::XprBits & int(matrix_flags::skew_symmetric)) std::cout << "SkewSymmetric ";
        if constexpr (Derived::XprBits & int(matrix_flags::square)) std::cout << "Square ";
        std::cout << "Matrix ]]" << std::endl;

        const int rows = m.derived().rows();
        const int cols = m.derived().cols();

        // compute max width per column
        size_t width = 0;
        for (int j = 0; j < cols; ++j) {
            for (int i = 0; i < rows; ++i) {
                std::ostringstream ss;
                ss << m.derived()(i, j);
                width = std::max(width, ss.str().size());
            }
        }

        // print values with alignment
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                os << std::setw(int(width)) << m.derived()(i, j) << " ";
            }
            if (i != rows - 1) os << "\n";
        }

        return os;
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
    constexpr TransposeView<Derived> transpose() const { return TransposeView<Derived>(derived()); }

    // block operations
    constexpr MatrixBlockView<Rows, 1, Derived> col(int i) { return block<Rows, 1>(0, i); }
    constexpr MatrixBlockView<Rows, 1, const Derived> col(int i) const {
        return MatrixBlockView<Rows, 1, const Derived>(derived(), 0, i);
    }
    constexpr MatrixBlockView<1, Cols, Derived> row(int i) { return block<1, Cols>(i, 0); }
    constexpr MatrixBlockView<1, Cols, const Derived> row(int i) const {
        return MatrixBlockView<1, Cols, const Derived>(derived(), i, 0);
    }
    template <int BlockRows, int BlockCols> constexpr MatrixBlockView<BlockRows, BlockCols, Derived> block(int i, int j) {
        return MatrixBlockView<BlockRows, BlockCols, Derived>(derived(), i, j);
    }
    template <int BlockRows> constexpr MatrixBlockView<BlockRows, Cols, Derived> topRows() {
        return block<BlockRows, Cols>(0, 0);
    }
    template <int BlockRows> constexpr MatrixBlockView<BlockRows, Cols, Derived> bottomRows() {
        return block<BlockRows, Cols>(Rows - BlockRows, 0);
    }
    template <int BlockCols> constexpr MatrixBlockView<Rows, BlockCols, Derived> leftCols() {
        return block<Rows, BlockCols>(0, 0);
    }
    template <int BlockCols> constexpr MatrixBlockView<Rows, BlockCols, Derived> rightCols() {
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

    // copy
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
    }

    // arithmetic operators
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

}   // namespace fdapde

#endif   // _FDAPDE_MATRIX_BASE_H__
