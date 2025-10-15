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

#ifndef __FDAPDE_LINALG_MATRIX_H__
#define __FDAPDE_LINALG_MATRIX_H__

#include "header_check.h"

namespace fdapde {

// procedural matrices generates matrices whose entries exhibit a fixed pattern, without allocating memory
template <typename Functor_, int Rows_, int Cols_>
struct ProceduralMatrix : public MatrixExpr<Rows_, Cols_, ProceduralMatrix<Functor_, Rows_, Cols_>> {
    fdapde_static_assert(
      std::is_invocable_v<Functor_ FDAPDE_COMMA int FDAPDE_COMMA int>, FUNCTOR_NOT_CALLABLE_AT_INDEXES_PAIR);
    using Base = MatrixExpr<Rows_, Cols_, ProceduralMatrix<Functor_, Rows_, Cols_>>;
    using Scalar = typename decltype(std::function {std::declval<Functor_>()})::result_type;
    fdapde_static_assert(std::is_arithmetic_v<Scalar>, INVALID_FUNCTOR_RETURN_TYPE);
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    constexpr ProceduralMatrix() : rows_(Rows_ == Dynamic ? 0 : Rows), cols_(Cols_ == Dynamic ? 0 : Cols) { }
    constexpr explicit ProceduralMatrix(Functor_ f) :
        rows_(Rows_ == Dynamic ? 0 : Rows), cols_(Cols_ == Dynamic ? 0 : Cols), f_(f) { }
    constexpr ProceduralMatrix(int rows, int cols, Functor_ f) : rows_(rows), cols_(cols), f_(f) {
        fdapde_static_assert(Rows == Dynamic || Cols == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_MATRICES_ONLY);
        fdapde_assert(rows >= 0 && cols >= 0);
    }
    constexpr ProceduralMatrix(int rows, int cols) : ProceduralMatrix(rows, cols, Functor_()) { }

    constexpr Scalar operator()(int i, int j) const { return f_(i, j); }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return f_(i, 0);
    }
    constexpr int rows() const { return rows_; }
    constexpr int cols() const { return cols_; }
    constexpr void resize(int rows, int cols) {
        fdapde_static_assert(Rows == Dynamic || Cols == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_MATRICES_ONLY);
        rows_ = rows;
        cols_ = cols;
    }
   private:
    int rows_, cols_;
    Functor_ f_;
};

// definition of procedrual matrices
template <int Rows, int Cols> using ZeroMatrix = ProceduralMatrix<decltype([](int, int) { return 0; }), Rows, Cols>;
template <int Rows, int Cols> using OnesMatrix = ProceduralMatrix<decltype([](int, int) { return 1; }), Rows, Cols>;
template <int Rows, int Cols>
using IdentityMatrix = ProceduralMatrix<decltype([](int i, int j) { return i == j ? 1 : 0; }), Rows, Cols>;

namespace internals {

struct generic_assignment_executor {
    template <typename DstMatrixType, typename SrcXprType, typename AssignmentOp>
        requires(requires(AssignmentOp op, typename DstMatrixType::Scalar& l, const typename SrcXprType::Scalar& r) {
            { op(l, r) } -> std::same_as<void>;
        })
    static constexpr void run(DstMatrixType& dst, const SrcXprType& src, AssignmentOp&& op) {
        fdapde_static_assert(DstMatrixType::ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION);
        fdapde_static_assert(
          is_dynamic_sized_v<DstMatrixType> || is_dynamic_sized_v<SrcXprType> ||
            same_static_shape_v<DstMatrixType FDAPDE_COMMA SrcXprType>,
          INVALID_ASSIGNMENT__DIFFERENT_LHS_AND_RHS_STATIC_SIZES);
        if constexpr (internals::is_dynamic_sized_v<DstMatrixType> || internals::is_dynamic_sized_v<SrcXprType>) {
            fdapde_assert(dst.rows() == src.rows() && dst.cols() == src.cols());
        }
        int rows_ = dst.rows();
        int cols_ = dst.cols();
        // exploit cache-locality depending on StorageOrder of destination
        if constexpr (DstMatrixType::StorageOrder == RowMajor) {
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) { op(dst(i, j), src(i, j)); }
            }
        } else {   // ColMajor
            for (int j = 0; j < cols_; ++j) {
                for (int i = 0; i < rows_; ++i) { op(dst(i, j), src(i, j)); }
            }
        }
        return;
    }
};

}   // namespace internals

template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_, typename MatrixType>
class MatrixBase : public MatrixExpr<Rows_, Cols_, MatrixType> {
    fdapde_static_assert((Rows_ == Dynamic || Rows_ > 0) && (Cols_ == Dynamic || Cols_ > 0), INVALID_MATRIX_DIMENSIONS);
   public:
    using Base = MatrixExpr<Rows_, Cols_, MatrixType>;
    using Base::derived;
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int NestAsRef = MatrixType::NestAsRef;
    static constexpr int ReadOnly = std::is_const_v<Scalar_> ? 1 : 0;
    using assignment_executor = internals::generic_assignment_executor;

    // constructors
    constexpr MatrixBase() :
        rows_(Rows == Dynamic ? 0 : Rows),
        cols_(Cols == Dynamic ? 0 : Cols),
        row_stride_(StorageOrder == RowMajor ? cols_ : 1),
        col_stride_(StorageOrder == RowMajor ? 1 : rows_) { }
    constexpr MatrixBase(int rows, int cols) :
        rows_(Rows == Dynamic ? rows : Rows),
        cols_(Cols == Dynamic ? cols : Cols),
        row_stride_(StorageOrder == RowMajor ? cols_ : 1),
        col_stride_(StorageOrder == RowMajor ? 1 : rows_) {
        fdapde_static_assert(Rows_ != 1 && Cols_ != 1, THIS_METHOD_IS_FOR_PROPER_MATRICES);
    }
    constexpr MatrixBase(int size) :
        rows_(Rows == 1 ? 1 : size),
        cols_(Cols == 1 ? 1 : size),
        row_stride_(StorageOrder == RowMajor ? cols_ : 1),
        col_stride_(StorageOrder == RowMajor ? 1 : rows_) {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
    }
    // copy assignment
    constexpr MatrixType& operator=(const MatrixType& other) {
        fdapde_static_assert(ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION);
        if (this == std::addressof(other)) { return derived(); }
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) {
            if (rows_ != other.rows() || cols_ != other.cols()) { derived().resize(other.rows(), other.cols()); }
        }
        assignment_executor::run(*this, other, [](Scalar& l, const Scalar& r) { l = r; });
        return derived();
    }
    // inherit assignment from base
    using Base::operator=;
    // access
    constexpr const Scalar& operator()(int i, int j) const {
        fdapde_assert(i >= 0 && i < rows_ && j >= 0 && j < cols_);
        return derived().data()[i * row_stride_ + j * col_stride_];
    }
    constexpr const Scalar& operator[](int i) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        fdapde_assert(i >= 0 && i < rows_ * cols_);
        return derived().data()[i];
    }
    constexpr Scalar& operator()(int i, int j) {
        fdapde_static_assert(ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION);
        fdapde_assert(i >= 0 && i < rows_ && j >= 0 && j < cols_);
        return derived().data()[i * row_stride_ + j * col_stride_];
    }
    constexpr Scalar& operator[](const int i) {
        fdapde_static_assert(ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION);
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        fdapde_assert(i >= 0 && i < rows_ * cols_);
        return derived().data()[i];
    }
    // observers
    constexpr int rows() const { return Rows != Dynamic ? Rows : rows_; }
    constexpr int cols() const { return Cols != Dynamic ? Cols : cols_; }
   protected:
    int rows_, cols_;
    int row_stride_, col_stride_;
};

template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_ = RowMajor>
    requires(std::is_arithmetic_v<Scalar_>)
class Matrix : public MatrixBase<Scalar_, Rows_, Cols_, StorageOrder_, Matrix<Scalar_, Rows_, Cols_, StorageOrder_>> {
   public:
    using Base = MatrixBase<Scalar_, Rows_, Cols_, StorageOrder_, Matrix<Scalar_, Rows_, Cols_, StorageOrder_>>;
    using Scalar = Scalar_;
    static constexpr int StorageSize = (Rows_ == Dynamic || Cols_ == Dynamic) ? Dynamic : (Rows_ * Cols_);
    using StorageType = std::conditional_t<
      Rows_ == Dynamic || Cols_ == Dynamic, std::vector<Scalar>,
      std::array<Scalar, (StorageSize < 0) ? 0 : static_cast<std::size_t>(StorageSize)>>;   // avoid clang narrowing
    using iterator = typename StorageType::iterator;
    using const_iterator = typename StorageType::const_iterator;
    static constexpr int NestAsRef = 1;

    constexpr Matrix() : data_() { }
    constexpr Matrix(const Matrix& other) : Base() {
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) { resize(other.rows(), other.cols()); }
        using assignment = typename Base::assignment_executor;
        assignment::run(*this, other, [](Scalar& l, const Scalar& r) { l = r; });
    }
    constexpr Matrix& operator=(const Matrix& other) {
        Base::operator=(other);
        return *this;
    }
    template <int RhsRows_, int RhsCols_, typename RhsXprType_>   // construct from plain MatrixExpr
    constexpr Matrix(const MatrixExpr<RhsRows_, RhsCols_, RhsXprType_>& rhs) : Base(), data_() {
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) { resize(rhs.rows(), rhs.cols()); }
        using assignment = typename Base::assignment_executor;
        assignment::run(*this, rhs.derived(), [](Scalar& l, const Scalar& r) { l = r; });
    }
    // inherit assignment from base
    using Base::operator=;

    // Matrix API
    // value-initialized static-sized matrix, avoid vectors (Vector API only support 1D, 2D, 3D value intialization)
    constexpr explicit Matrix(Scalar v)
        requires(Rows_ > 1 && Cols_ > 1)
        : data_() {
        for (int i = 0; i < Rows_ * Cols_; ++i) { data_[i] = v; }
    }
    // zero-initialized dynamic-sized matrix. For static-sized matrices does nothing (exposed for API compatibility)
    constexpr Matrix(int rows, int cols)
        requires(Rows_ != 1 && Cols_ != 1)
        : Base(rows, cols) {
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) { data_.resize(rows * cols); }
    }
    // value-initialized dynamic-sized matrix, avoid vectors
    constexpr Matrix(int rows, int cols, Scalar v)
        requires(Rows_ == Dynamic && Cols_ == Dynamic)
        : Matrix(rows, cols) {
        for (int i = 0, n = rows * cols; i < n; ++i) { data_[i] = v; }
    }

    // Vector API
    // zero-initialized dynamic-sized vector
    constexpr explicit Matrix(int size)
        requires(Rows_ == Dynamic || Cols_ == Dynamic)
        : Base(size) {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) { data_.resize(size); }
    }
    // value-initialized dynamic-sized vector
    constexpr Matrix(int size, Scalar v)
        requires(Rows_ == Dynamic || Cols_ == Dynamic)
        : Matrix(size) {
        fdapde_static_assert(
          (Rows_ == 1 && Cols_ == Dynamic) || (Cols_ == 1 && Rows_ == Dynamic),
          THIS_METHOD_IS_FOR_ROW_OR_COLUMN_DYNAMIC_SIZED_VECTORS_ONLY);
        for (int i = 0; i < size; ++i) { data_[i] = v; }
    }
    // 1D static-sized vector
    constexpr explicit Matrix(Scalar x)
        requires(!internals::is_dynamic_sized_v<Base> && (Cols_ == 1 && Rows_ == 1))
        : Base() {
        fdapde_static_assert(Cols_ == 1 && Rows_ == 1, THIS_METHOD_IS_FOR_1D_VECTORS_ONLY);
        data_[0] = x;
    }
    // 2D static-sized vector
    constexpr Matrix(Scalar x, Scalar y)
        requires(!internals::is_dynamic_sized_v<Base> && (Cols_ == 1 || Rows_ == 1))
        : Base() {
        fdapde_static_assert(
          (Cols_ == 2 && Rows_ == 1) || (Cols_ == 1 && Rows_ == 2), THIS_METHOD_IS_FOR_2D_VECTORS_ONLY);
        data_[0] = x;
        data_[1] = y;
    }
    // 3D static-sized vector
    constexpr Matrix(Scalar x, Scalar y, Scalar z)
        requires(!internals::is_dynamic_sized_v<Base> && (Cols_ == 1 || Rows_ == 1))
        : Base() {
        fdapde_static_assert(
          (Cols_ == 3 && Rows_ == 1) || (Cols_ == 1 && Rows_ == 3), THIS_METHOD_IS_FOR_3D_VECTORS_ONLY);
        data_[0] = x;
        data_[1] = y;
        data_[2] = z;
    }

    // constructors taking external data
    template <std::size_t Size> constexpr explicit Matrix(const Scalar (&data)[Size]) : Base() {
        fdapde_static_assert(
          Rows_ != Dynamic && Cols_ != Dynamic && StorageSize == Size, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        for (int i = 0; i < Rows_; ++i) {
            for (int j = 0; j < Cols_; ++j) {
                Base::operator()(i, j) = data[i * Base::row_stride_ + j * Base::col_stride_];
            }
        }
    }

    // static named constructors
    static constexpr auto Zero() { return ZeroMatrix<Rows_, Cols_>(); }
    static constexpr auto Zero(int rows) {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return ZeroMatrix<Dynamic, Dynamic>(rows, 1);
    }
    static constexpr auto Zero(int rows, int cols) { return ZeroMatrix<Dynamic, Dynamic>(rows, cols); }
    static constexpr auto Ones() { return OnesMatrix<Rows_, Cols_>(); }
    static constexpr auto Ones(int rows) {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return OnesMatrix<Dynamic, Dynamic>(rows, 1);
    }
    static constexpr auto Ones(int rows, int cols) { return OnesMatrix<Dynamic, Dynamic>(rows, cols); }
    static constexpr auto Constant(Scalar value) { return value * Ones(); }
    static constexpr auto Constant(int rows, Scalar value) {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return value * Ones(rows);
    }
    static constexpr auto Constant(int rows, int cols, Scalar value) { return value * Ones(rows, cols); }
    static constexpr auto LinSpaced(int rows, Scalar a, Scalar b) {
        fdapde_static_assert(Cols_ == 1, THIS_METHOD_IS_FOR_COLUMN_VECTORS_ONLY);
        fdapde_assert(rows > 1);
        auto linspace = [a, h = (double)(b - a) / double(rows - 1)](int i, int) { return a + i * h; };
        return ProceduralMatrix<decltype(linspace), Dynamic, 1>(rows, 1, linspace);
    }
    static constexpr auto LinSpaced(Scalar a, Scalar b) {
        fdapde_static_assert(Cols_ == 1, THIS_METHOD_IS_FOR_COLUMN_VECTORS_ONLY);
        auto linspace = [a, h = (double)(b - a) / double(Rows_ - 1)](int i, int) { return a + i * h; };
        return ProceduralMatrix<decltype(linspace), Rows_, 1>(linspace);
    }

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
          THIS_METHOD_IS_FOR_DYNAMIC_SIZED_ROW_OR_COLUMN_VECTORS_ONLY);
        resize(Rows_ == Dynamic ? size : Rows_, Cols_ == Dynamic ? size : Cols_);
        return;
    }
  
    // data pointers
    constexpr const Scalar* data() const { return data_.data(); }
    constexpr Scalar* data() { return data_.data(); }

    // iterators
    constexpr iterator begin() { return data_.begin(); }
    constexpr const_iterator begin() const { return data_.begin(); }
    constexpr iterator end() { return data_.end(); }
    constexpr const_iterator end() const { return data_.end(); }
   private:
    StorageType data_;
};

// non-owning Matrix view of an existing block of data
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_ = RowMajor>
class MatrixView :
    public MatrixBase<Scalar_, Rows_, Cols_, StorageOrder_, MatrixView<Scalar_, Rows_, Cols_, StorageOrder_>> {
   public:
    using Base = MatrixBase<Scalar_, Rows_, Cols_, StorageOrder_, MatrixView<Scalar_, Rows_, Cols_, StorageOrder_>>;
    using Scalar = Scalar_;
    using StorageType = std::add_pointer_t<Scalar>;
    static constexpr int ReadOnly = std::is_const_v<Scalar_> ? 1 : 0;
    static constexpr int NestAsRef = 0;

    // constructors
    constexpr MatrixView() : Base(), data_(nullptr) { }
    constexpr explicit MatrixView(Scalar* data) : Base(), data_(data) {
        fdapde_static_assert(Rows_ != Dynamic && Cols_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
    }
    constexpr MatrixView(Scalar* data, int size) : Base(size), data_(data) {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        fdapde_assert(size > 0);
    }
    constexpr MatrixView(Scalar* data, int rows, int cols) : Base(rows, cols), data_(data) {
        fdapde_assert(rows > 0 && cols > 0);
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

#endif   // _FDAPDE_LINALG_MATRIX_H__
