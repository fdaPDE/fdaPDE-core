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

#ifndef __FDAPDE_LINALG_DIAGONAL_H__
#define __FDAPDE_LINALG_DIAGONAL_H__

#include "header_check.h"

namespace fdapde {

// diagonal matrix type system
template <typename XprType> struct DiagonalMatrixExpr;
  
namespace internals {

struct diagonal_assignment_executor {
    template <typename DstXprType, typename SrcXprType, typename AssignmentOp>
        requires(requires(AssignmentOp op, typename DstXprType::Scalar& l, const typename SrcXprType::Scalar& r) {
            { op(l, r) } -> std::same_as<void>;
        })
    static constexpr void run(DstXprType& dst, const SrcXprType& src, AssignmentOp&& op) {
        fdapde_static_assert(DstXprType::ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION);
        fdapde_static_assert(
          internals::is_dynamic_sized_v<DstXprType> || internals::is_dynamic_sized_v<SrcXprType> ||
            DstXprType::Rows == SrcXprType::Rows,   // diagonal expressions are vector-shaped
          INVALID_ASSIGNMENT__NOT_MATCHING_LHS_AND_RHS_STATIC_SIZES);
        if constexpr (internals::is_dynamic_sized_v<DstXprType> || internals::is_dynamic_sized_v<SrcXprType>) {
            fdapde_assert(dst.rows() == src.rows() && dst.cols() == src.cols());
        }
        int size_ = dst.rows();
        for (int i = 0; i < size_; ++i) { op(dst[i], src[i]); }
        return;
    }
};

}   // namespace internals

// expression of the diagonal of a matrix (this node acts as a vector expression)
template <typename XprType_> struct Diagonal : public MatrixExpr<Diagonal<XprType_>> {
   private:
    using Base = MatrixExpr<Diagonal<XprType_>>;
    using XprType = std::decay_t<XprType_>;
    using XprTypeNested = internals::ref_select_t<XprType_>;
   public:
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = 1;
    static constexpr int StorageOrder = XprType::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = XprType::ReadOnly;
    using assignment_executor = internals::diagonal_assignment_executor;

    // constructor
    template <typename XprType__>
        requires(std::is_constructible_v<XprTypeNested, XprType__>)
    constexpr explicit Diagonal(XprType__&& xpr) : xpr_(std::forward<XprType__>(xpr)) {
        fdapde_static_assert(
          internals::is_dynamic_sized_v<XprType> || XprType::Rows == XprType::Cols,
          DIAGONAL_BLOCKS_ARE_FOR_SQUARE_MATRICES_ONLY);
        if constexpr (internals::is_dynamic_sized_v<XprType>) { fdapde_assert(xpr_.rows() == xpr_.cols()); }
    }
    // copy-semantic
    constexpr Diagonal(const Diagonal& other) :
        xpr_(other.xpr_) { }   // don't copy as we implicitly point to other's diagonal
    constexpr Diagonal& operator=(const Diagonal& other) {
        fdapde_static_assert(XprType::ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION);
	assignment_executor::run(*this, other, [](auto& l, const auto& r) { l = r; });
        return *this;
    }
    // inherit assignment from base
    using Base::operator=;
    // const access
    constexpr Scalar operator()(int i, int j) const {
        fdapde_assert(i >= 0 && i < rows() && j >= 0 && j < cols());
        return xpr_(i, i);
    }
    constexpr Scalar operator[](int i) const {
        fdapde_assert(i >= 0 && i < rows());
        return xpr_(i, i);
    }
    // non-const access
    constexpr Scalar& operator()(int i, int j) {
        fdapde_assert(i >= 0 && i < rows() && j >= 0 && j < cols());
        return xpr_(i, i);
    }
    constexpr Scalar& operator[](int i) {
        fdapde_assert(i >= 0 && i < rows());
        return xpr_(i, i);
    }
    // observers
    constexpr int rows() const { return Rows != Dynamic ? Rows : xpr_.rows(); }
    constexpr int cols() const { return 1; }
    constexpr int size() const { return rows(); }
   private:
    XprTypeNested xpr_;
};

namespace internals {

// class wrapping an expression of the diagonal coefficients to a square matrix. internal usage only
template <int Rows_, int Cols_, typename DiagonalXprType_>
struct diagonal_wrapper : DiagonalMatrixExpr<diagonal_wrapper<Rows_, Cols_, DiagonalXprType_>> {
   private:
    using Base = DiagonalMatrixExpr<diagonal_wrapper<Rows_, Cols_, DiagonalXprType_>>;
    using XprType = std::decay_t<DiagonalXprType_>;
    using XprTypeNested = internals::ref_select_t<DiagonalXprType_>;
   public:
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = XprType::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename XprType__>
        requires(std::is_constructible_v<XprTypeNested, XprType__>)
    constexpr diagonal_wrapper(XprType__&& xpr) : Base(), xpr_(std::forward<XprType__>(xpr)) { }
    constexpr Scalar operator()(int i, int j) const {
        fdapde_assert(i >= 0 && i < this->size_ && j >= 0 && j < this->size_);
        return i == j ? xpr_[i] : Scalar(0);
    }
    const XprTypeNested& data() const { return xpr_; }
   private:
    XprTypeNested xpr_;
};

// helper cast function
template <typename XprType_> constexpr auto diagonal_cast(XprType_&& xpr) {
    using XprType = std::decay_t<XprType_>;
    constexpr int Rows = XprType::Rows, Cols = XprType::Cols;
    fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
    return diagonal_wrapper<Rows == 1 ? Cols : Rows, Cols == 1 ? Rows : Cols, XprType>(xpr);
}

}   // namespace internals

template <typename XprType_> struct DiagonalMatrixExpr : public MatrixExpr<XprType_> {
    using XprType = std::decay_t<XprType_>;
    // make derived() point to innermost type
    constexpr const XprType& derived() const { return static_cast<const XprType&>(*this); }
    constexpr XprType& derived() { return static_cast<XprType&>(*this); }
  
    constexpr DiagonalMatrixExpr() noexcept :
        size_((XprType::Rows == Dynamic || XprType::Cols == Dynamic) ? 0 : XprType::Rows) { }
    constexpr explicit DiagonalMatrixExpr(int size) noexcept :
        size_((XprType::Rows == Dynamic || XprType::Cols == Dynamic) ? size : XprType::Rows) { }
    // inherit assignment from base
    using MatrixExpr<XprType_>::operator=;
    // const access
    constexpr decltype(auto) operator()(int i, int j) const {   // only const access allowed for (i, j) accessor
        fdapde_assert(i >= 0 && i < rows() && j >= 0 && j < cols());
        return i == j ? derived().data()[i] : typename XprType::Scalar(0);
    }
    constexpr decltype(auto) operator[](int i) const {
        fdapde_assert(i >= 0 && i < rows());
        return derived().data()[i];
    }
    // non-const access
    constexpr decltype(auto) operator[](int i) {
        fdapde_assert(i >= 0 && i < rows());
        return derived().data()[i];
    }
    // converts to full dense matrix
    auto as_matrix() const { return Matrix<typename XprType::Scalar, XprType::Rows, XprType::Cols>(derived()); }
    // matrix inverse as 1/coeff
    auto inverse() const { return internals::diagonal_cast(derived().diagonal().cwise().inv()); }
    // linear system solver Ax = b
    template <typename RhsXprType> constexpr auto solve(const RhsXprType& b) const {
	constexpr int Rows = XprType::Rows;
        Vector<typename XprType::Scalar, Rows> x;
        if constexpr (Rows == Dynamic) { x.resize(derived().rows()); }
	for(int i = 0, n = derived().rows(); i < n; ++i) { x[i] = b[i] / derived().data()[i]; }
	return x;
    }
    double determinant() const { return derived().diagonal().prod(); }
    // observers
    constexpr int rows() const { return size_; }
    constexpr int cols() const { return size_; }
   protected:
    int size_;
};

// diagonal matrix arithmetic (O(n) operations on the diagonal coefficients)
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator+(const DiagonalMatrixExpr<LhsXprType>& lhs, const DiagonalMatrixExpr<RhsXprType>& rhs) {
    fdapde_static_assert(
      internals::same_static_shape_weak_v<LhsXprType FDAPDE_COMMA RhsXprType>,
      INVALID_OPERAND_DIMENSIONS_IN_BINARY_OPERATION);
    if constexpr (internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType>) {
        fdapde_assert(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols());
    }
    return internals::diagonal_cast(lhs.diagonal() + rhs.diagonal());
}
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator-(const DiagonalMatrixExpr<LhsXprType>& lhs, const DiagonalMatrixExpr<RhsXprType>& rhs) {
    fdapde_static_assert(
      internals::same_static_shape_weak_v<LhsXprType FDAPDE_COMMA RhsXprType>,
      INVALID_OPERAND_DIMENSIONS_IN_BINARY_OPERATION);
    if constexpr (internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType>) {
        fdapde_assert(lhs.rows() == rhs.rows() && lhs.cols() == rhs.cols());
    }
    return internals::diagonal_cast(lhs.diagonal() - rhs.diagonal());
}
template <typename XprType, typename CoeffType>
    requires(std::is_arithmetic_v<CoeffType>)
constexpr auto operator*(const DiagonalMatrixExpr<XprType>& lhs, CoeffType rhs) {
    return internals::diagonal_cast(rhs * lhs.diagonal());
}
template <typename XprType, typename CoeffType>
    requires(std::is_arithmetic_v<CoeffType>)
constexpr auto operator*(CoeffType lhs, const DiagonalMatrixExpr<XprType>& rhs) {
    return rhs * lhs;
}
template <typename XprType, typename CoeffType>
    requires(std::is_arithmetic_v<CoeffType>)
constexpr auto operator/(const DiagonalMatrixExpr<XprType>& lhs, CoeffType rhs) {
    return internals::diagonal_cast(lhs.diagonal() / rhs);
}

// specialized products
namespace internals {

// expressions of the (i,j)-th entry of the product between a diagonal and a dense matrix expression
template <int ProductMode> struct diagonal_matrix_product_executor {
    template <typename LhsXprType_, typename RhsXprType_>  
    static constexpr decltype(auto) run(int i, int j, const LhsXprType_& lhs, const RhsXprType_& rhs) {
        if constexpr (ProductMode == LhsMode) return lhs(i, i) * rhs(i, j);
        if constexpr (ProductMode == RhsMode) return lhs(i, j) * rhs(j, j);
    }
};
// expressions of the (i,j)-th entry of the product between diagonal expressions
struct diagonal_diagonal_product_executor {
    template <typename LhsXprType_, typename RhsXprType_>
    static constexpr decltype(auto) run(int i, int j, const LhsXprType_& lhs, const RhsXprType_& rhs) {
        using LhsXprType = std::decay_t<LhsXprType_>;
        using RhsXprType = std::decay_t<RhsXprType_>;
        using Scalar = promote_type_t<typename LhsXprType::Scalar, typename RhsXprType::Scalar>;
        return i == j ? lhs(i, i) * rhs(i, i) : Scalar(0);
    }
};

}   // namespace internals
  
template <typename LhsXprType, typename RhsXprType, typename Executor> struct MatrixMultiplicationOp;
// diag(a_1, ..., a_n) * M
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator*(const DiagonalMatrixExpr<LhsXprType>& lhs, const MatrixExpr<RhsXprType>& rhs) {
    return MatrixMultiplicationOp<LhsXprType, RhsXprType, internals::diagonal_matrix_product_executor<LhsMode>> {
      lhs.derived(), rhs.derived()};
}
// M * diag(a_1, ..., a_n)
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator*(const MatrixExpr<LhsXprType>& lhs, const DiagonalMatrixExpr<RhsXprType>& rhs) {
    return MatrixMultiplicationOp<LhsXprType, RhsXprType, internals::diagonal_matrix_product_executor<RhsMode>> {
      lhs.derived(), rhs.derived()};
}
// diag(a_1, ..., a_n) * diag(b_1, ..., b_n)
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator*(const DiagonalMatrixExpr<LhsXprType>& lhs, const DiagonalMatrixExpr<RhsXprType>& rhs) {
    return internals::diagonal_cast(
      MatrixMultiplicationOp<LhsXprType, RhsXprType, internals::diagonal_diagonal_product_executor>(
        lhs.derived(), rhs.derived())
        .diagonal());
}

// owning storage diagonal matrix
template <typename Scalar_, int Rows_>
class DiagonalMatrix : public DiagonalMatrixExpr<DiagonalMatrix<Scalar_, Rows_>> {
   private:
    using Base = DiagonalMatrixExpr<DiagonalMatrix<Scalar_, Rows_>>;
   public:
    using Scalar = Scalar_;
    using StorageType = Vector<Scalar, Rows_>;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Rows_;
    static constexpr int StorageOrder = StorageType::StorageOrder;
    static constexpr int NestAsRef = 1;
    static constexpr int ReadOnly = std::is_const_v<Scalar_>;
    using assignment_executor = internals::diagonal_assignment_executor;

    constexpr DiagonalMatrix() : Base(), data_() { }
    // copy semantic
    constexpr DiagonalMatrix(const DiagonalMatrix& rhs) : Base() { clone_(rhs); }
    constexpr DiagonalMatrix& operator=(const DiagonalMatrix& rhs) {
        clone_(rhs);
        return *this;
    }
    constexpr explicit DiagonalMatrix(int size) : Base(size), data_() {
        fdapde_static_assert(
          Rows == Dynamic || Cols == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_DIAGONAL_MATRICES_ONLY);
        data_.resize(size);
    }
    template <typename RhsXprType_> constexpr DiagonalMatrix(const MatrixExpr<RhsXprType_>& rhs) : Base() {
        fdapde_assert(this->rows() == rhs.rows() && this->cols() == rhs.cols());
	clone_(rhs.derived());
    }
    constexpr explicit DiagonalMatrix(const std::vector<Scalar>& vec) : Base(vec.size()), data_() {
        if constexpr (Rows == Dynamic || Cols == Dynamic) { data_.resize(vec.size()); }
	fdapde_assert(data_.size() == vec.size());
        for (int i = 0, n = data_.size(); i < n; ++i) { data_[i] = vec[i]; }
    }
    template <std::size_t RhsSize> constexpr explicit DiagonalMatrix(const Scalar (&data)[RhsSize]) : Base() {
        fdapde_static_assert(Rows != Dynamic && Cols != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        fdapde_static_assert(Rows == RhsSize, INVALID_DATA_SIZE);
        for (int i = 0, n = data_.size(); i < n; ++i) { data_[i] = data[i]; }
    }
    // constructor for 1 x 1, 2 x 2, 3 x 3 static sized diagonals
    constexpr explicit DiagonalMatrix(Scalar x) : Base() {
        fdapde_static_assert(Rows == 1, THIS_METHOD_IS_FOR_1_X_1_DIAGONAL_MATRICES_ONLY);
        data_[0] = x;
    }
    constexpr DiagonalMatrix(Scalar x, Scalar y) : Base() {
        fdapde_static_assert(Rows == 2, THIS_METHOD_IS_FOR_2_X_2_DIAGONAL_MATRICES_ONLY);
        data_[0] = x;
	data_[1] = y;
    }
    constexpr DiagonalMatrix(Scalar x, Scalar y, Scalar z) : Base() {
        fdapde_static_assert(Rows == 3, THIS_METHOD_IS_FOR_3_X_3_DIAGONAL_MATRICES_ONLY);
        data_[0] = x;
	data_[1] = y;
	data_[2] = z;
    }
    constexpr const StorageType& diagonal() const { return data_; }
    // modifiers
    void resize(int size) {
        fdapde_static_assert(Rows == Dynamic || Cols == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_MATRICES_ONLY);
        const int size_ = Rows == Dynamic ? size : Rows;
        if (std::cmp_equal(size_, data_.size())) return;   // do not reallocate memory if sizes didn't changed
        // update and reallocate memory
        this->rows_ = size_;
        this->cols_ = size_;
        data_.resize(size);
        return;
    }
    // data pointers
    constexpr const Scalar* data() const { return data_.data(); }
    constexpr Scalar* data() { return data_.data(); }
   private:
    template <typename RhsXprType> constexpr void clone_(const RhsXprType& rhs) {
        if constexpr (Rows == Dynamic || Cols == Dynamic) { resize(rhs.rows()); }
        assignment_executor::run(*this, rhs, [](auto&& l, const auto& r) { l = r; });
        return;
    }
    StorageType data_;
};

// diagonal view of an existing block of data
template <typename Scalar_, int Rows_>
class DiagonalMatrixView : public DiagonalMatrixExpr<DiagonalMatrixView<Scalar_, Rows_>> {
   private:
    using Base = DiagonalMatrixExpr<DiagonalMatrixView<Scalar_, Rows_>>;
   public:
    using Scalar = Scalar_;
    using StorageType = std::add_pointer_t<Scalar>;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Rows_;
    static constexpr int StorageOrder = StorageType::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = std::is_const_v<Scalar_>;
    using assignment_executor = internals::diagonal_assignment_executor;

    // constructors
    constexpr DiagonalMatrixView() : Base(), data_(nullptr) { }
    constexpr explicit DiagonalMatrixView(Scalar* data) : Base(), data_(data) {
        fdapde_static_assert(Rows_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_DIAGONAL_VIEWS_ONLY);
    }
    constexpr DiagonalMatrixView(Scalar* data, int size) : Base(size), data_(data) {
	fdapde_assert(size > 0);
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

// detection trait
template <typename XprType> struct is_diagonal_matrix {
    static constexpr bool value = std::is_base_of_v<DiagonalMatrixExpr<std::decay_t<XprType>>, XprType>;
};
template <typename XprType> static constexpr bool is_diagonal_matrix_v = is_diagonal_matrix<XprType>::value;
  
}   // namespace fdapde

#endif // __FDAPDE_LINALG_DIAGONAL_H__
