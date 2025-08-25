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

#ifndef __FDAPDE_MATRIX_H__
#define __FDAPDE_MATRIX_H__

#include "../header_check.h"

namespace fdapde {

// forward decl
template <typename Functor_, int Rows_, int Cols_> struct ProceduralMatrix;
// definition of procedrual matrices
template <int Rows, int Cols> using ZeroMatrix = ProceduralMatrix<decltype([](int i, int j) { return 0; }), Rows, Cols>;
template <int Rows, int Cols> using OnesMatrix = ProceduralMatrix<decltype([](int i, int j) { return 1; }), Rows, Cols>;
template <int Rows, int Cols>
using IdentityMatrix = ProceduralMatrix<decltype([](int i, int j) { return i == j ? 1 : 0; }), Rows, Cols>;
  
namespace internals {

template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_, typename MatrixType>
class matrix_impl : public MatrixBase<Rows_, Cols_, MatrixType> {
    fdapde_static_assert((Rows_ == Dynamic || Rows_ > 0) && (Cols_ == Dynamic || Cols_ > 0), INVALID_MATRIX_SIZES);
   public:
    using Base = MatrixBase<Rows_, Cols_, MatrixType>;
    using Base::derived;
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int NestAsRef = MatrixType::NestAsRef;
    static constexpr int ReadOnly = std::is_const_v<Scalar_> ? 1 : 0;
    struct assignment_executor {
        template <typename SrcXprType> static constexpr void run(MatrixType& dst, const SrcXprType& src) {
            fdapde_assert(dst.rows() == src.rows() && dst.cols() == src.cols());
            int rows_ = dst.rows();
            int cols_ = dst.cols();
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) { dst(i, j) = src(i, j); }
            }
            return;
        }
    };

    // constructors
    constexpr matrix_impl() :
        rows_(Rows == Dynamic ? 0 : Rows),
        cols_(Cols == Dynamic ? 0 : Cols),
        row_stride_(StorageOrder == RowMajor ? cols_ : 1),
        col_stride_(StorageOrder == RowMajor ? 1 : rows_) { }
    constexpr matrix_impl(int rows, int cols) :
        rows_(Rows == Dynamic ? rows : Rows),
        cols_(Cols == Dynamic ? cols : Cols),
	row_stride_(StorageOrder == RowMajor ? cols_ : 1),
	col_stride_(StorageOrder == RowMajor ? 1 : rows_) {
    }
    constexpr matrix_impl(int size) :
        rows_(Rows == 1 ? 1 : size),
        cols_(Cols == 1 ? 1 : size),
	row_stride_(StorageOrder == RowMajor ? cols_ : 1),
	col_stride_(StorageOrder == RowMajor ? 1 : rows_) {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
    }
    // copy assignment
    constexpr MatrixType& operator=(const MatrixType& other) {
        fdapde_static_assert(ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION);
        if constexpr (Rows == Dynamic || Cols == Dynamic) {
            fdapde_constexpr_assert(rows_ == other.rows() && cols_ == other.cols());
        }
        if (this == std::addressof(other)) { return derived(); }
	assignment_executor::run(*this, other);
        return derived();
    }
    // inherit assignment from base
    using Base::operator=;
    // access
    constexpr const Scalar& operator()(int i, int j) const {
        fdapde_constexpr_assert(i >= 0 && i < rows_ && j >= 0 && j < cols_);
        return derived().data()[i * row_stride_ + j * col_stride_];
    }
    constexpr const Scalar& operator[](int i) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        fdapde_constexpr_assert(i >= 0 && i < rows_ * cols_);
        return derived().data()[i];
    }
    constexpr Scalar& operator()(int i, int j) {
        fdapde_static_assert(ReadOnly == 0, ASSIGNMENT_TO_A_READ_ONLY_LOCATION_IS_INVALID);
        fdapde_assert(i >= 0 && i < rows_ && j >= 0 && j < cols_);
        return derived().data()[i * row_stride_ + j * col_stride_];
    }
    constexpr Scalar& operator[](const int i) {
        fdapde_static_assert(ReadOnly == 0, ASSIGNMENT_TO_A_READ_ONLY_LOCATION_IS_INVALID);
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        fdapde_assert(i >= 0 && i < rows_ * cols_);
        return derived().data()[i];
    }
    // observers
    constexpr int rows() const { return Rows != Dynamic ? Rows : rows_; }
    constexpr int cols() const { return Cols != Dynamic ? Cols : cols_; }
   protected:
    // sizes
    int rows_, cols_;
    int row_stride_, col_stride_;
};

}   // namespace internals

template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_ = RowMajor>
    requires(std::is_arithmetic_v<Scalar_>)
class Matrix :
    public internals::matrix_impl<Scalar_, Rows_, Cols_, StorageOrder_, Matrix<Scalar_, Rows_, Cols_, StorageOrder_>> {
   public:
    using Base =
      internals::matrix_impl<Scalar_, Rows_, Cols_, StorageOrder_, Matrix<Scalar_, Rows_, Cols_, StorageOrder_>>;
    using Scalar = Scalar_;
    static constexpr int StorageSize = (Rows_ == Dynamic || Cols_ == Dynamic) ? Dynamic : (Rows_ * Cols_);
    using StorageType =
      std::conditional_t<Rows_ == Dynamic || Cols_ == Dynamic, std::vector<Scalar>, std::array<Scalar, StorageSize>>;
    static constexpr int NestAsRef = 1;

    constexpr Matrix() : data_() { }
    constexpr Matrix(int size) : Base(size) {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        data_.resize(size);
    }
    constexpr Matrix(int rows, int cols) : Base(rows, cols) {
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) { data_.resize(rows * cols); }
    }
    constexpr Matrix(const Matrix& other) : Base() {
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) { resize(other.rows(), other.cols()); }
        using assignment = typename Base::assignment_executor;
        assignment::run(*this, other);
    }
    template <int RhsRows_, int RhsCols_, typename RhsXprType_>
    constexpr Matrix(const MatrixBase<RhsRows_, RhsCols_, RhsXprType_>& rhs) : Base() {
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) { resize(rhs.rows(), rhs.cols()); }
        using assignment = typename Base::assignment_executor;
        assignment::run(*this, rhs.derived());
    }
    template <typename DataT>
        requires(internals::is_vector_like_v<DataT> && !internals::is_matrix_like_v<DataT>)
    constexpr explicit Matrix(DataT&& data) {
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) { data_.resize(data.size()); }
        fdapde_constexpr_assert(data_.size() == data.size());
        for (int i = 0, n = Base::rows(); i < n; ++i) {
            for (int j = 0, m = Base::cols(); j < m; ++j) {
                Base::operator()(i, j) = data[i * Base::row_stride_ + j * Base::col_stride_];
            }
        }
    }
    template <std::size_t RhsSize> constexpr explicit Matrix(const Scalar (&data)[RhsSize]) : Base() {
        fdapde_static_assert(Rows_ != Dynamic && Cols_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        fdapde_static_assert(StorageSize == RhsSize, INVALID_DATA_SIZE);
        for (int i = 0; i < Rows_; ++i) {
            for (int j = 0; j < Cols_; ++j) {
                Base::operator()(i, j) = data[i * Base::row_stride_ + j * Base::col_stride_];
            }
        }
    }
    // constructor for 1 x 1, 2 x 1, 3 x 1 static sized vectors
    constexpr explicit Matrix(Scalar x) : Base() {
        fdapde_static_assert(Rows_ * Cols_ == 1, THIS_METHOD_IS_FOR_1D_VECTORS_ONLY);
        data_[0] = x;
    }
    constexpr Matrix(Scalar x, Scalar y) : Base() {
        fdapde_static_assert(Rows_ * Cols_ == 2, THIS_METHOD_IS_FOR_2D_VECTORS_ONLY);
        data_[0] = x;
	data_[1] = y;
    }
    constexpr Matrix(Scalar x, Scalar y, Scalar z) : Base() {
        fdapde_static_assert(Rows_ * Cols_ == 3, THIS_METHOD_IS_FOR_3D_VECTORS_ONLY);
        data_[0] = x;
	data_[1] = y;
	data_[2] = z;
    }
    // static named constructors
    static constexpr auto Zero() { return ZeroMatrix<Rows_, Cols_>(); }
    static constexpr auto Zero(int rows, int cols) { return ZeroMatrix<Dynamic, Dynamic>(rows, cols); }
    static constexpr auto Ones() { return OnesMatrix<Rows_, Cols_>(); }
    static constexpr auto Ones(int rows, int cols) { return OnesMatrix<Dynamic, Dynamic>(rows, cols); }
    static constexpr auto Identity() {
        fdapde_static_assert(Rows_ > 0 && Cols_ > 0 && Rows_ == Cols_, THIS_METHOD_IS_FOR_SQUARE_MATRICES_ONLY);
        return IdentityMatrix<Rows_, Cols_>();
    }
    static constexpr auto Identity(int rows, int cols) {
        fdapde_constexpr_assert(rows == cols);
        return IdentityMatrix<Dynamic, Dynamic>(rows, cols);
    }
    // inherit assignment from Base
    using Base::operator=;
    // modifiers
    void resize(int rows, int cols) {
        fdapde_static_assert(Rows_ == Dynamic || Cols_ == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_MATRICES_ONLY);
        const int rows_ = Rows_ == Dynamic ? rows : Rows_;
        const int cols_ = Cols_ == Dynamic ? cols : Cols_;
        if (rows_ == Base::rows_ && cols_ == Base::cols_) return;   // do not reallocate memory if sizes didn't changed
        // update and reallocate memory
        Base::rows_ = rows_;
        Base::cols_ = cols_;
        Base::row_stride_ = StorageOrder_ == RowMajor ? cols_ : 1;
        Base::col_stride_ = StorageOrder_ == RowMajor ? 1 : rows_;
        data_.resize(rows * cols);
	return;
    }
    void resize(int size) {
        fdapde_static_assert(
          (Rows_ == 1 && Cols_ == Dynamic) || (Cols_ == 1 && Rows_ == Dynamic),
          THIS_METHOD_IS_FOR_ROW_OR_COLUMN_DYNAMIC_VECTORS_ONLY);
        resize(Rows_ == Dynamic ? size : Rows_, Cols_ == Dynamic ? size : Cols_);
	return;
    }
    // data pointers
    constexpr const Scalar* data() const { return data_.data(); }
    constexpr Scalar* data() { return data_.data(); }
   private:
    StorageType data_;
};

// non-owning Matrix view of an existing block of data
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_ = RowMajor>
class MatrixView :
    public internals::matrix_impl<
      Scalar_, Rows_, Cols_, StorageOrder_, MatrixView<Scalar_, Rows_, Cols_, StorageOrder_>> {
   public:
    using Base =
      internals::matrix_impl<Scalar_, Rows_, Cols_, StorageOrder_, MatrixView<Scalar_, Rows_, Cols_, StorageOrder_>>;
    using Scalar = Scalar_;
    using StorageType = std::add_pointer_t<Scalar>;
    static constexpr int NestAsRef = 0;

    // constructors
    constexpr MatrixView() : Base(), data_(nullptr) { }
    constexpr explicit MatrixView(Scalar* data) : Base(), data_(data) {
        fdapde_static_assert(Rows_ != Dynamic && Cols_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
    }
    constexpr MatrixView(Scalar* data, int size) : Base(size), data_(data) {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
	fdapde_constexpr_assert(size > 0);
    }
    constexpr MatrixView(Scalar* data, int rows, int cols) : Base(rows, cols), data_(data) {
	fdapde_constexpr_assert(rows > 0 && cols > 0);
    }
    // inherit assignment from Base
    using Base::operator=;
    // data pointers
    constexpr const StorageType data() const { return data_; }
    constexpr StorageType data() { return data_; }
  protected:
    StorageType data_;
};

// vector aliases
template <typename Scalar, int Rows> using Vector = Matrix<Scalar, Rows, 1>;
template <typename Scalar, int Rows> using VectorView = MatrixView<Scalar, Rows, 1>;

}   // namespace fdapde

#endif   // _FDAPDE_MATRIX_H__
