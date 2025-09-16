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

#ifndef __FDAPDE_DIAGONAL_MATRIX_H__
#define __FDAPDE_DIAGONAL_MATRIX_H__

#include "../header_check.h"

namespace fdapde {

// diagonal matrix type system

// expression of the diagonal of a matrix
template <int Rows_, int Cols_, typename XprType>
struct Diagonal : public MatrixExpr<Rows_, Cols_, Diagonal<Rows_, Cols_, XprType>> {
    using Base = MatrixExpr<Rows_, Cols_, Diagonal<Rows_, Cols_, XprType>>;
    using XprTypeNested = internals::ref_select_t<XprType>;
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = XprType::ReadOnly;
    struct assignment_executor {
        template <typename SrcXprType>
        static constexpr void run(Diagonal<Rows_, 1, XprType>& dst, const SrcXprType& src) {
            fdapde_static_assert(XprType::ReadOnly == 0, ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION);
            for (int i = 0, size_ = dst.rows(); i < size_; ++i) { dst[i] = src[i]; }
        }
    };
  
    // constructor
    template <typename XprType_>
        requires(std::is_constructible_v<XprTypeNested, XprType_>)
    constexpr explicit Diagonal(XprType_&& xpr) : xpr_(std::forward<XprType_>(xpr)) {
        fdapde_static_assert(
          XprType::Rows == Dynamic || XprType::Cols == Dynamic || XprType::Rows == XprType::Cols,
          THIS_EXPRESSION_IS_FOR_SQUARE_MATRICES_ONLY);
        if constexpr (XprType::Rows == Dynamic || XprType::Cols == Dynamic) {
            fdapde_constexpr_assert(xpr_.rows() == xpr_.cols());
        }
    }
    // copy assignment
    constexpr Diagonal& operator=(const Diagonal& other) {
        fdapde_static_assert(XprType::ReadOnly == 0, ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION);
	assignment_executor::run(*this, other);
        return *this;
    }
    // inherit assignment from base
    using Base::operator=;
    // const access
    constexpr Scalar operator()(int i, int j) const {
        fdapde_constexpr_assert(i >= 0 && i < rows() && j >= 0 && j < cols());
        return xpr_(i, i);
    }
    constexpr Scalar operator[](int i) const {
        fdapde_constexpr_assert(i >= 0 && i < rows());
        return xpr_(i, i);
    }
    // non-const access
    constexpr Scalar& operator()(int i, int j) {
        fdapde_constexpr_assert(i >= 0 && i < rows() && j >= 0 && j < cols());
        return xpr_(i, i);
    }
    constexpr Scalar& operator[](int i) {
        fdapde_constexpr_assert(i >= 0 && i < rows());
        return xpr_(i, i);
    }
    // observers
    constexpr int rows() const { return Rows != Dynamic ? Rows : xpr_.rows(); }
    constexpr int cols() const { return 1; }
    constexpr int size() const { return rows(); }
   private:
    XprTypeNested xpr_;
};

template <typename Scalar, int DiagonalSize> class DiagonalMatrix;

template <int Rows_, int Cols_, typename XprType>
struct DiagonalMatrixExpr : public MatrixExpr<Rows_, Cols_, XprType> {
    using Base = MatrixExpr<Rows_, Cols_, XprType>;
    using Base::derived;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageSize = Rows_ == Dynamic ? Dynamic : Rows_;
    static constexpr int ReadOnly = std::is_const_v<XprType> ? 1 : 0;
    static constexpr int NestAsRef = 1;
    struct assignment_executor {
        template <typename SrcXprType> static constexpr void run(XprType& dst, const SrcXprType& src) {
            fdapde_static_assert(XprType::ReadOnly == 0, ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION);
            for (int i = 0, size_ = dst.rows(); i < size_; ++i) { dst.data()[i] = src(i, i); }
        }
    };

    DiagonalMatrixExpr() : size_((Rows_ == Dynamic || Cols_ == Dynamic) ? 0 : Rows_) { }
    DiagonalMatrixExpr(int size) : size_((Rows_ == Dynamic || Cols_ == Dynamic) ? size : Rows_) { }
    // inherit assignment from base
    using Base::operator=;
    // const access
    constexpr auto operator()(int i, int j) const {   // only const access allowed for (i, j) accessor
        fdapde_constexpr_assert(i >= 0 && i < rows() && j >= 0 && j < cols());
	using Scalar = XprType::Scalar;
        return i == j ? derived().data()[i] : Scalar(0);
    }
    constexpr auto operator[](int i) const {
        fdapde_constexpr_assert(i >= 0 && i < rows());
        return derived().data()[i];
    }
    // non-const access
    constexpr auto& operator[](int i) {
        fdapde_constexpr_assert(i >= 0 && i < rows());
        return derived().data()[i];
    }
    // converts the diagonal expression to a full dense matrix
    auto to_matrix() const { return Matrix<typename XprType::Scalar, Rows, Cols>(derived()); }
    // matrix inverse as 1/coeff
    auto inverse() const { return DiagonalMatrix<typename XprType::Scalar, Rows>(derived().cwise_inv()); }
    // linear system solver Ax = b
    template <typename RhsXprType> constexpr auto solve(const RhsXprType& b) const {
        using Scalar = typename XprType::Scalar;
        Vector<Scalar, Rows> x;
        if constexpr (Rows == Dynamic) { x.resize(derived().rows()); }
	for(int i = 0, n = derived().rows(); i < n; ++i) { x[i] = b[i] / derived().data()[i]; }
	return x;
    }
    double determinant() const { derived().diagonal().prod(); }
    // observers
    constexpr int rows() const { return size_; }
    constexpr int cols() const { return size_; }
   protected:
    int size_;
};

namespace internals {

// class wrapping an expression of the diagonal coefficients to a square matrix. internal usage only
template <int Rows_, int Cols_, typename DiagonalXprType>
struct diagonal_wrapper : DiagonalMatrixExpr<Rows_, Cols_, diagonal_wrapper<Rows_, Cols_, DiagonalXprType>> {
    using Base = DiagonalMatrixExpr<Rows_, Cols_, diagonal_wrapper<Rows_, Cols_, DiagonalXprType>>;
    using DiagonalXprTypeNested = internals::ref_select_t<const DiagonalXprType>;
    using Scalar = typename DiagonalXprType::Scalar;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename XprType>
        requires(std::is_constructible_v<DiagonalXprTypeNested, XprType>)
    constexpr diagonal_wrapper(XprType&& xpr) : Base(), xpr_(std::forward<XprType>(xpr)) { }
    constexpr Scalar operator()(int i, int j) const {
        fdapde_constexpr_assert(i >= 0 && i < Base::size_ && j >= 0 && j < Base::size_);
        return i == j ? xpr_[i] : Scalar(0);
    }
   private:
    DiagonalXprTypeNested xpr_;
};

}   // namespace internals

// diagonal matrix arithmetic (O(n) operations on the diagonal coefficients)
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator+(
  const DiagonalMatrixExpr<LhsXprType::Rows, LhsXprType::Cols, LhsXprType>& lhs,
  const DiagonalMatrixExpr<RhsXprType::Rows, RhsXprType::Cols, RhsXprType>& rhs) {
    return (lhs.diagonal() + rhs.diagonal()).as_diagonal();
}
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator-(
  const DiagonalMatrixExpr<LhsXprType::Rows, LhsXprType::Cols, LhsXprType>& lhs,
  const DiagonalMatrixExpr<RhsXprType::Rows, RhsXprType::Cols, RhsXprType>& rhs) {
    return (lhs.diagonal() - rhs.diagonal()).as_diagonal();
}
template <typename XprType, typename CoeffType>
    requires(std::is_arithmetic_v<CoeffType>)
constexpr auto operator*(const DiagonalMatrixExpr<XprType::Rows, XprType::Cols, XprType>& lhs, CoeffType rhs) {
    return (rhs * lhs.diagonal()).as_diagonal();
}
template <typename XprType, typename CoeffType>
    requires(std::is_arithmetic_v<CoeffType>)
constexpr auto operator*(CoeffType lhs, const DiagonalMatrixExpr<XprType::Rows, XprType::Cols, XprType>& rhs) {
    return rhs * lhs;
}
template <typename XprType, typename CoeffType>
    requires(std::is_arithmetic_v<CoeffType>)
constexpr auto operator/(const DiagonalMatrixExpr<XprType::Rows, XprType::Cols, XprType>& lhs, CoeffType rhs) {
    return (lhs.diagonal() / rhs).as_diagonal();
}

// specialized products
namespace internals {

// expressions of the (i,j)-th entry of the product between a diagonal and a dense matrix expression
template <typename LhsXprType, typename RhsXprType, int ProductMode> struct diagonal_matrix_product_executor {
    static constexpr auto run(int i, int j, const LhsXprType& lhs, const RhsXprType& rhs) {
        if constexpr (ProductMode == LhsMode) return lhs(i, i) * rhs(i, j);
        if constexpr (ProductMode == RhsMode) return lhs(i, j) * rhs(j, j);
    }
};

// expressions of the (i,j)-th entry of the product between diagonal expressions
template <typename LhsXprType, typename RhsXprType> struct diagonal_diagonal_product_executor {
    using LhsXprTypeClean = std::decay_t<LhsXprType>;
    using RhsXprTypeClean = std::decay_t<RhsXprType>;
    using Scalar =
      decltype(std::declval<typename LhsXprTypeClean::Scalar>() * std::declval<typename RhsXprTypeClean::Scalar>());

    static constexpr auto run(int i, int j, const LhsXprType& lhs, const RhsXprType& rhs) {
        return i == j ? lhs(i, i) * rhs(i, i) : Scalar(0);
    }
};  

}   // namespace internals
  
template <typename LhsXprType, typename RhsXprType, typename Executor> struct MatrixProductOp;
// diag(a_1, ..., a_n) * M
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator*(
  const DiagonalMatrixExpr<LhsXprType::Rows, LhsXprType::Cols, LhsXprType>& lhs,
  const MatrixExpr<RhsXprType::Rows, RhsXprType::Cols, RhsXprType>& rhs) {
    return MatrixProductOp<
      LhsXprType, RhsXprType, internals::diagonal_matrix_product_executor<LhsXprType, RhsXprType, LhsMode>> {
      lhs.derived(), rhs.derived()};
}
// M * diag(a_1, ..., a_n)
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator*(
  const MatrixExpr<LhsXprType::Rows, LhsXprType::Cols, LhsXprType>& lhs,
  const DiagonalMatrixExpr<RhsXprType::Rows, RhsXprType::Cols, RhsXprType>& rhs) {
    return MatrixProductOp<
      LhsXprType, RhsXprType, internals::diagonal_matrix_product_executor<LhsXprType, RhsXprType, RhsMode>> {
      lhs.derived(), rhs.derived()};
}
// diag(a_1, ..., a_n) * diag(b_1, ..., b_n)
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator*(
  const DiagonalMatrixExpr<LhsXprType::Rows, LhsXprType::Cols, LhsXprType>& lhs,
  const DiagonalMatrixExpr<RhsXprType::Rows, RhsXprType::Cols, RhsXprType>& rhs) {
    return MatrixProductOp<
             LhsXprType, RhsXprType, internals::diagonal_diagonal_product_executor<LhsXprType, RhsXprType>> {
      lhs.derived(), rhs.derived()}
      .diagonal()
      .as_diagonal();   // close wrt diagonal algebra
}

// owning storage diagonal matrix
template <typename Scalar_, int Rows_>
class DiagonalMatrix : public DiagonalMatrixExpr<Rows_, Rows_, DiagonalMatrix<Scalar_, Rows_>> {
   public:
    using Base = DiagonalMatrixExpr<Rows_, Rows_, DiagonalMatrix<Scalar_, Rows_>>;
    using Scalar = Scalar_;
    using StorageType = Vector<Scalar, Rows_>;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Rows_;
    static constexpr int StorageSize = Rows_ == Dynamic ? Dynamic : Rows_;
    static constexpr int NestAsRef = 1;
    static constexpr int ReadOnly = std::is_const_v<Scalar_> ? 1 : 0;

    constexpr DiagonalMatrix() : Base(), data_() { }
    constexpr explicit DiagonalMatrix(int size) : Base(size), data_() {
        fdapde_static_assert(
          Rows == Dynamic || Cols == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_DIAGONAL_MATRICES_ONLY);
        data_.resize(size);
    }
    template <int RhsRows_, int RhsCols_, typename RhsXprType_>
    constexpr DiagonalMatrix(const MatrixExpr<RhsRows_, RhsCols_, RhsXprType_>& rhs) : Base() {
        fdapde_constexpr_assert(Base::rows() == rhs.rows() && Base::cols() == rhs.cols());
        if constexpr (Rows == Dynamic || Cols == Dynamic) { resize(rhs.rows()); }
        *this = rhs;
    }
    template <typename DataT>
        requires(internals::is_vector_like_v<DataT> && !internals::is_matrix_like_v<DataT>)
    constexpr explicit DiagonalMatrix(DataT&& data) : Base(data.size()), data_() {
        if constexpr (Rows == Dynamic || Cols == Dynamic) { data_.resize(data.size()); }
	fdapde_constexpr_assert(data_.size() == data.size());
        for (int i = 0, n = data_.size(); i < n; ++i) { data_[i] = data[i]; }
    }
    template <std::size_t RhsSize> constexpr explicit DiagonalMatrix(const Scalar (&data)[RhsSize]) : Base() {
        fdapde_static_assert(Rows != Dynamic && Cols != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        fdapde_static_assert(StorageSize == RhsSize, INVALID_DATA_SIZE);
        for (int i = 0, n = data_.size(); i < n; ++i) { data_[i] = data[i]; }
    }
    // constructor for 1 x 1, 2 x 2, 3 x 3 static sized diagonals
    constexpr explicit DiagonalMatrix(Scalar x) : Base() {
        fdapde_static_assert(StorageSize == 1, THIS_METHOD_IS_FOR_1_X_1_DIAGONAL_MATRICES_ONLY);
        data_[0] = x;
    }
    constexpr DiagonalMatrix(Scalar x, Scalar y) : Base() {
        fdapde_static_assert(StorageSize == 2, THIS_METHOD_IS_FOR_2_X_2_DIAGONAL_MATRICES_ONLY);
        data_[0] = x;
	data_[1] = y;
    }
    constexpr DiagonalMatrix(Scalar x, Scalar y, Scalar z) : Base() {
        fdapde_static_assert(StorageSize == 3, THIS_METHOD_IS_FOR_3_X_3_DIAGONAL_MATRICES_ONLY);
        data_[0] = x;
	data_[1] = y;
	data_[2] = z;
    }
    constexpr const StorageType& diagonal() const { return data_; }
    // static named constructors
    static constexpr DiagonalMatrix Identity() { return StorageType::Ones().as_diagonal(); }
    static constexpr DiagonalMatrix Identity(int size) { return StorageType::Ones(size).as_diagonal(); }
    static constexpr DiagonalMatrix Zero() { return StorageType::Zero().as_diagonal(); }
    static constexpr DiagonalMatrix Zero(int size) { return StorageType::Zero(size).as_diagonal(); }
    // inherit assignment from Base
    using Base::operator=;
    // modifiers
    void resize(int size) {
        fdapde_static_assert(Rows == Dynamic || Cols == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_MATRICES_ONLY);
        const int size_ = StorageSize == Dynamic ? size : StorageSize;
        if (std::cmp_equal(size_, data_.size())) return;   // do not reallocate memory if sizes didn't changed
        // update and reallocate memory
        Base::rows_ = size_;
        Base::cols_ = size_;
        data_.resize(size);
        return;
    }
    // data pointers
    constexpr const Scalar* data() const { return data_.data(); }
    constexpr Scalar* data() { return data_.data(); }
   private:
    StorageType data_;
};

// diagonal view of an existing block of data
template <typename Scalar_, int Rows_>
class DiagonalMatrixView : public DiagonalMatrixExpr<Rows_, Rows_, DiagonalMatrixView<Scalar_, Rows_>> {
   public:
    using Base = DiagonalMatrixExpr<Rows_, Rows_, DiagonalMatrixView<Scalar_, Rows_>>;
    using Scalar = Scalar_;
    using StorageType = std::add_pointer_t<Scalar>;
    static constexpr int ReadOnly = std::is_const_v<Scalar_> ? 1 : 0;
    static constexpr int NestAsRef = 0;
  
    // constructors
    constexpr DiagonalMatrixView() : Base(), data_(nullptr) { }
    constexpr explicit DiagonalMatrixView(Scalar* data) : Base(), data_(data) {
        fdapde_static_assert(Rows_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_DIAGONAL_VIEWS_ONLY);
    }
    constexpr DiagonalMatrixView(Scalar* data, int size) : Base(size), data_(data) {
	fdapde_constexpr_assert(size > 0);
    }
    constexpr VectorView<Scalar, Rows_> diagonal() const { return VectorView<Scalar, Rows_>(data_); }
    // inherit assignment from Base
    using Base::operator=;
    // data pointers
    constexpr const StorageType data() const { return data_; }
    constexpr StorageType data() { return data_; }
   private:
    StorageType data_;
};

}   // namespace fdapde

#endif // __FDAPDE_DIAGONAL_MATRIX_H__
