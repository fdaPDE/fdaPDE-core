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

namespace fdapde::linalg {

// procedural matrices generates matrices whose entries exhibit a fixed pattern, without allocating memory
template <typename Functor_, int Rows_, int Cols_>
struct ProceduralMatrix : public MatrixExpr<ProceduralMatrix<Functor_, Rows_, Cols_>> {
    fdapde_static_assert(
      (Rows_ == Dynamic || Rows_ > 0) && (Cols_ == Dynamic || Cols_ > 0), INVALID_MATRIX_DIMENSIONS);
    fdapde_static_assert(
      std::is_invocable_v<Functor_ FDAPDE_COMMA int FDAPDE_COMMA int>, FUNCTOR_NOT_CALLABLE_AT_INDEXES_PAIR);
    using Scalar = typename decltype(std::function {std::declval<Functor_>()})::result_type;
    fdapde_static_assert(std::is_arithmetic_v<Scalar>, INVALID_FUNCTOR_RETURN_TYPE);
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = RowMajor;   // memoryless node, choose default
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    constexpr ProceduralMatrix() : rows_(Rows_ == Dynamic ? 0 : Rows), cols_(Cols_ == Dynamic ? 0 : Cols) { }
    constexpr explicit ProceduralMatrix(Functor_ f) :
        rows_(Rows_ == Dynamic ? 0 : Rows), cols_(Cols_ == Dynamic ? 0 : Cols), f_(f) { }
    constexpr ProceduralMatrix(int rows, int cols, Functor_ f) :
        rows_(Rows == Dynamic ? rows : Rows), cols_(Cols == Dynamic ? cols : Cols), f_(f) {
        const bool valid =
          rows >= 0 && cols >= 0 && (Rows == Dynamic || rows == Rows) && (Cols == Dynamic || cols == Cols);
        fdapde_assert(valid);
        if (!valid) {
            rows_ = Rows == Dynamic ? 0 : Rows;
            cols_ = Cols == Dynamic ? 0 : Cols;
        }
    }
    constexpr ProceduralMatrix(int rows, int cols) : ProceduralMatrix(rows, cols, Functor_()) { }
    constexpr explicit ProceduralMatrix(int size, Functor_ f) :
        rows_(Rows_ == Dynamic ? size : Rows), cols_(Cols_ == Dynamic ? size : Cols), f_(f) {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        const bool valid = size >= 0 && (Rows == Dynamic || Cols == Dynamic || size == Rows * Cols);
        fdapde_assert(valid);
        if (!valid) {
            rows_ = Rows == Dynamic ? 0 : Rows;
            cols_ = Cols == Dynamic ? 0 : Cols;
        }
    }
    constexpr explicit ProceduralMatrix(int size) : ProceduralMatrix(size, Functor_()) { }

    constexpr Scalar operator()(int i, int j) const { return f_(i, j); }
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return f_(Rows == 1 ? 0 : i, Cols == 1 ? i : 0);
    }
    constexpr int rows() const { return rows_; }
    constexpr int cols() const { return cols_; }
    constexpr void resize(int rows, int cols) {
        fdapde_static_assert(Rows == Dynamic || Cols == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_MATRICES_ONLY);
        const bool valid =
          rows >= 0 && cols >= 0 && (Rows == Dynamic || rows == Rows) && (Cols == Dynamic || cols == Cols);
        fdapde_assert(valid);
        if (!valid) return;
        rows_ = Rows == Dynamic ? rows : Rows;
        cols_ = Cols == Dynamic ? cols : Cols;
    }
   private:
    int rows_, cols_;
    Functor_ f_;
};
// definition of procedrual matrices
template <typename Scalar, int Rows, int Cols>
using ZeroMatrix = ProceduralMatrix<decltype([](int, int) { return Scalar(0); }), Rows, Cols>;
template <typename Scalar, int Rows, int Cols>
using OnesMatrix = ProceduralMatrix<decltype([](int, int) { return Scalar(1); }), Rows, Cols>;
template <typename Scalar, int Rows, int Cols>
using IdentityMatrix =
  ProceduralMatrix<decltype([](int i, int j) { return i == j ? Scalar(1) : Scalar(0); }), Rows, Cols>;

namespace internals {

struct generic_assignment_executor {
    template <typename DstMatrixType, typename SrcXprType, typename AssignmentOp>
    static constexpr void run(DstMatrixType& dst, const SrcXprType& src, AssignmentOp&& op) {
        fdapde_static_assert(DstMatrixType::ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION);
        if constexpr (!std::is_arithmetic_v<SrcXprType>) {
            fdapde_static_assert(
              (internals::is_dynamic_sized_v<DstMatrixType> || internals::is_dynamic_sized_v<SrcXprType> ||
               internals::same_static_shape_v<DstMatrixType FDAPDE_COMMA SrcXprType>),
              INVALID_ASSIGNMENT__DIFFERENT_LHS_AND_RHS_STATIC_SHAPES);
            if constexpr (internals::is_dynamic_sized_v<DstMatrixType> || internals::is_dynamic_sized_v<SrcXprType>) {
                if (dst.rows() != src.rows() || dst.cols() != src.cols()) {
                    fdapde_assert(dst.rows() == src.rows() && dst.cols() == src.cols());
                    return;
                }
            }
        }
        const int rows_ = dst.rows();
        const int cols_ = dst.cols();
        auto fetch = [](const SrcXprType& src, [[maybe_unused]] int i, [[maybe_unused]] int j) -> decltype(auto) {
            if constexpr (std::is_arithmetic_v<SrcXprType>) {
                return src;
            } else {
                return src(i, j);
            }
        };
        // exploit cache-locality depending on StorageOrder of destination
        if constexpr (DstMatrixType::StorageOrder == RowMajor) {
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) { op(dst(i, j), fetch(src, i, j)); }
            }
        } else {   // ColMajor
            for (int j = 0; j < cols_; ++j) {
                for (int i = 0; i < rows_; ++i) { op(dst(i, j), fetch(src, i, j)); }
            }
        }
        return;
    }
};
// assignment executor specialized for vector expressions
struct vector_assignment_executor {
    template <typename DstMatrixType, typename SrcXprType, typename AssignmentOp>
    static constexpr void run(DstMatrixType& dst, const SrcXprType& src, AssignmentOp&& op) {
        fdapde_static_assert(DstMatrixType::ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION);
        if constexpr (!std::is_arithmetic_v<SrcXprType>) {
            // NB: a row-shaped rhs can be assigned to a col-shaped lhs
            fdapde_static_assert(
              internals::is_vector_shaped_v<DstMatrixType>, INVALID_ASSIGNMENT__NOT_VECTOR_SHAPED_LVALUE);
            fdapde_static_assert(
              internals::is_dynamic_sized_v<SrcXprType> || internals::is_vector_shaped_v<SrcXprType>,
              INVALID_ASSIGNMENT__NOT_VECTOR_SHAPED_RVALUE);
            fdapde_static_assert(
              (internals::is_dynamic_sized_v<DstMatrixType> || internals::is_dynamic_sized_v<SrcXprType> ||
               internals::same_static_size_v<DstMatrixType FDAPDE_COMMA SrcXprType>),
              INVALID_ASSIGNMENT__DIFFERENT_LHS_AND_RHS_STATIC_SIZES);
            if constexpr (internals::is_dynamic_sized_v<DstMatrixType> || internals::is_dynamic_sized_v<SrcXprType>) {
                const bool compatible =
                  ((dst.rows() == 1 && src.rows() == 1) || (dst.cols() == 1 && src.cols() == 1) ||
                   (dst.rows() == 1 && src.cols() == 1) || (dst.cols() == 1 && src.rows() == 1)) &&
                  dst.size() == src.size();
                if (!compatible) {
                    fdapde_assert(compatible);
                    return;
                }
            }
        }
        const int size_ = dst.size();
        auto fetch = [](const SrcXprType& src, [[maybe_unused]] int i) -> decltype(auto) {
            if constexpr (std::is_arithmetic_v<SrcXprType>) {
                return src;
            } else {
                return src.rows() == 1 ? src(0, i) : src(i, 0);
            }
        };
        for (int i = 0; i < size_; ++i) { op(dst[i], fetch(src, i)); }
        return;
    }
};

}   // namespace internals

template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_, typename MatrixType>
class MatrixBase : public MatrixExpr<MatrixType> {
    fdapde_static_assert((Rows_ == Dynamic || Rows_ > 0) && (Cols_ == Dynamic || Cols_ > 0), INVALID_MATRIX_DIMENSIONS);
   protected:
    using Base = MatrixExpr<MatrixType>;
    using Base::derived;
   public:
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int NestAsRef = MatrixType::NestAsRef;
    static constexpr int ReadOnly = std::is_const_v<Scalar_>;
    using assignment_executor = std::conditional_t<
      (Rows_ == 1 || Cols_ == 1) && !(Rows_ == 1 && Cols_ == 1), internals::vector_assignment_executor,
      internals::generic_assignment_executor>;

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
        const bool valid =
          rows >= 0 && cols >= 0 && (Rows == Dynamic || rows == Rows) && (Cols == Dynamic || cols == Cols);
        fdapde_assert(valid);
        if (!valid) {
            rows_ = Rows == Dynamic ? 0 : Rows;
            cols_ = Cols == Dynamic ? 0 : Cols;
            row_stride_ = StorageOrder == RowMajor ? cols_ : 1;
            col_stride_ = StorageOrder == RowMajor ? 1 : rows_;
        }
    }
    constexpr MatrixBase(int size) :
        rows_(Rows == Dynamic ? size : Rows),
        cols_(Cols == Dynamic ? size : Cols),
        row_stride_(StorageOrder == RowMajor ? cols_ : 1),
        col_stride_(StorageOrder == RowMajor ? 1 : rows_) {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        const bool valid = size >= 0 && (Rows == Dynamic || Cols == Dynamic || size == Rows * Cols);
        fdapde_assert(valid);
        if (!valid) {
            rows_ = Rows == Dynamic ? 0 : Rows;
            cols_ = Cols == Dynamic ? 0 : Cols;
            row_stride_ = StorageOrder == RowMajor ? cols_ : 1;
            col_stride_ = StorageOrder == RowMajor ? 1 : rows_;
        }
    }
    // inherit assignment from base
    using Base::operator=;
    // access
    constexpr decltype(auto) operator()(int i, int j) const {
        fdapde_assert(i >= 0 && i < rows_ && j >= 0 && j < cols_);
        return derived().data()[i * row_stride_ + j * col_stride_];
    }
    constexpr decltype(auto) operator[](int i) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        fdapde_assert(i >= 0 && i < rows_ * cols_);
        return derived().data()[i];
    }
    constexpr decltype(auto) operator()(int i, int j) {
        fdapde_assert(i >= 0 && i < rows_ && j >= 0 && j < cols_);
        return derived().data()[i * row_stride_ + j * col_stride_];
    }
    constexpr decltype(auto) operator[](const int i) {
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
// requires(std::is_arithmetic_v<Scalar_>)
class Matrix : public MatrixBase<Scalar_, Rows_, Cols_, StorageOrder_, Matrix<Scalar_, Rows_, Cols_, StorageOrder_>> {
   private:
    using Base = MatrixBase<Scalar_, Rows_, Cols_, StorageOrder_, Matrix<Scalar_, Rows_, Cols_, StorageOrder_>>;
    static constexpr int StorageSize = (Rows_ == Dynamic || Cols_ == Dynamic) ? Dynamic : (Rows_ * Cols_);
    using StorageType = std::conditional_t<
      Rows_ == Dynamic || Cols_ == Dynamic, std::vector<Scalar_>,
      std::array<Scalar_, (StorageSize < 0) ? 0 : static_cast<std::size_t>(StorageSize)>>;   // avoid clang narrowing
   public:
    using Scalar = Scalar_;
    using iterator = typename StorageType::iterator;
    using const_iterator = typename StorageType::const_iterator;
    static constexpr int NestAsRef = 1;

    constexpr Matrix() : data_() { }
    // copy semantic
    constexpr Matrix(const Matrix& other) : Base() { clone_(other); }
    constexpr Matrix& operator=(const Matrix& other) & {
        clone_(other);
        return *this;
    }
    template <typename RhsXprType_>   // construct from plain MatrixExpr
    constexpr Matrix(const MatrixExpr<RhsXprType_>& rhs) : Base(), data_() {
        clone_(rhs.derived());
    }
    template <typename RhsXprType_>
    constexpr Matrix(const MatrixCoeffWiseExpr<RhsXprType_>& rhs) : Matrix(rhs.mwise()) { }

    // Matrix API
    // value-initialized static-sized matrix, avoid 1D vectors
    constexpr explicit Matrix(Scalar v)
        requires(Rows_ > 1 && Cols_ >= 1)
        : data_() {
        for (int i = 0; i < Rows_ * Cols_; ++i) { data_[i] = v; }
    }
    // zero-initialized dynamic-sized matrix. For static-sized matrices does nothing (exposed for API compatibility)
    constexpr Matrix(int rows, int cols)
        requires(Rows_ != 1 && Cols_ != 1)
        : Base(rows, cols), data_() {
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) { data_.resize(this->rows() * this->cols()); }
    }
    // value-initialized dynamic-sized matrix, avoid vectors
    constexpr Matrix(int rows, int cols, Scalar v)
        requires(Rows_ == Dynamic && Cols_ == Dynamic)
        : Matrix(rows, cols) {
        for (int i = 0, n = this->rows() * this->cols(); i < n; ++i) { data_[i] = v; }
    }

    // Vector API
    // zero-initialized dynamic-sized vector
    constexpr explicit Matrix(int size)
        requires(Rows_ == Dynamic || Cols_ == Dynamic)
        : Base(size) {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) { data_.resize(this->size()); }
    }
    // value-initialized dynamic-sized vector
    constexpr Matrix(int size, Scalar v)
        requires((Rows_ == Dynamic && Cols_ == 1) || (Rows_ == 1 && Cols_ == Dynamic))
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
    constexpr explicit Matrix(const std::vector<Scalar>& data) :
        Base(
          Rows_ == Dynamic ? static_cast<int>(data.size()) : Rows_,
          Cols_ == Dynamic ? static_cast<int>(data.size()) : Cols_),
        data_() {
        fdapde_static_assert(
          (Rows_ != Dynamic && Cols_ != Dynamic) || (Rows_ == 1 && Cols_ == Dynamic) ||
            (Cols_ == 1 && Rows_ == Dynamic),
          THIS_METHOD_IS_NOT_FOR_DYNAMIC_SIZED_MATRICES);
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) { data_.resize(data.size()); }
        const bool compatible = std::cmp_equal(data_.size() FDAPDE_COMMA data.size());
        fdapde_assert(compatible);
        if (!compatible) return;
        const int rows = this->rows();
        const int cols = this->cols();
        for (int i = 0, n = rows; i < n; ++i) {
            for (int j = 0, m = cols; j < m; ++j) {
                this->operator()(i, j) = data[i * cols + j];
            }
        }
    }
    template <std::size_t Size> constexpr explicit Matrix(const Scalar (&data)[Size]) : Base() {
        fdapde_static_assert(
          Rows_ != Dynamic && Cols_ != Dynamic && StorageSize == Size, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        for (int i = 0; i < Rows_; ++i) {
            for (int j = 0; j < Cols_; ++j) {
                this->operator()(i, j) = data[i * Cols_ + j];
            }
        }
    }
    constexpr Matrix& operator=(const std::initializer_list<Scalar>& data) & {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) { resize(static_cast<int>(data.size())); }
        const bool compatible = data_.size() == data.size();
        fdapde_assert(compatible);
        if (!compatible) return *this;
        int i = 0;
        for (Scalar v : data) { this->operator[](i++) = v; }
        return *this;
    }

    // static named constructors
    static constexpr auto Zero() { return ZeroMatrix<Scalar_, Rows_, Cols_>(); }
    static constexpr auto Zero(int size) {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return ZeroMatrix<Scalar_, Rows_ == 1 ? Rows_ : Dynamic, Cols_ == 1 ? Cols_ : Dynamic>(size);
    }
    static constexpr auto Zero(int rows, int cols) { return ZeroMatrix<Scalar_, Dynamic, Dynamic>(rows, cols); }
    static constexpr auto Ones() { return OnesMatrix<Scalar_, Rows_, Cols_>(); }
    static constexpr auto Ones(int size) {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return OnesMatrix<Scalar_, Rows_ == 1 ? Rows_ : Dynamic, Cols_ == 1 ? Cols_ : Dynamic>(size);
    }
    static constexpr auto Ones(int rows, int cols) { return OnesMatrix<Scalar_, Dynamic, Dynamic>(rows, cols); }
    static constexpr auto Constant(Scalar value) { return value * Ones(); }
    static constexpr auto Constant(int size, Scalar value) {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return value * Ones(size);
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
        const bool valid =
          rows >= 0 && cols >= 0 && (Rows_ == Dynamic || rows == Rows_) && (Cols_ == Dynamic || cols == Cols_);
        fdapde_assert(valid);
        if (!valid) return;
        const int rows_ = Rows_ == Dynamic ? rows : Rows_;
        const int cols_ = Cols_ == Dynamic ? cols : Cols_;
        if (rows_ == this->rows_ && cols_ == this->cols_) return;   // do not reallocate memory if sizes didn't changed
        // update and reallocate memory
        this->rows_ = rows_;
        this->cols_ = cols_;
        this->row_stride_ = StorageOrder_ == RowMajor ? cols_ : 1;
        this->col_stride_ = StorageOrder_ == RowMajor ? 1 : rows_;
        data_.resize(rows_ * cols_);
        return;
    }
    void resize(int size) {
        fdapde_static_assert(
          (Rows_ == 1 && Cols_ == Dynamic) || (Cols_ == 1 && Rows_ == Dynamic),
          THIS_METHOD_IS_FOR_DYNAMIC_SIZED_ROW_OR_COLUMN_VECTORS_ONLY);
        fdapde_assert(size >= 0);
        resize(Rows_ == Dynamic ? size : Rows_, Cols_ == Dynamic ? size : Cols_);
        return;
    }
    void set_zero() {
        for (int i = 0, n = data_.size(); i < n; ++i) { data_[i] = Scalar(0); }
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
    template <typename RhsXprType> constexpr void clone_(const RhsXprType& rhs) {
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) {
            if constexpr (Rows_ == 1 || Cols_ == 1) {
                resize(rhs.size());
            } else {
                resize(rhs.rows(), rhs.cols());
            }
        }
        using assignment_executor = typename Base::assignment_executor;
        assignment_executor::run(*this, rhs, [](auto&& l, const auto& r) { l = r; });
        return;
    }
    StorageType data_;
};

// non-owning Matrix view of an existing block of data
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_ = RowMajor>
class MatrixView :
    public MatrixBase<Scalar_, Rows_, Cols_, StorageOrder_, MatrixView<Scalar_, Rows_, Cols_, StorageOrder_>> {
   private:
    using Base = MatrixBase<Scalar_, Rows_, Cols_, StorageOrder_, MatrixView<Scalar_, Rows_, Cols_, StorageOrder_>>;
    using StorageType = std::add_pointer_t<Scalar_>;
   public:
    using Scalar = Scalar_;
    static constexpr int NestAsRef = 0;

    // constructors
    constexpr MatrixView(const MatrixView&) = default;
    constexpr MatrixView() requires(Rows_ == Dynamic || Cols_ == Dynamic) : Base(), data_(nullptr) { }
    constexpr MatrixView() requires(Rows_ != Dynamic && Cols_ != Dynamic) = delete;
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
    constexpr MatrixView& operator=(const MatrixView& other) & {
        static_cast<MatrixExpr<MatrixView>&>(*this).template operator=<MatrixView>(other);
        return *this;
    }
    constexpr MatrixView operator=(const MatrixView& other) && {
        static_cast<MatrixExpr<MatrixView>&>(*this).template operator=<MatrixView>(other);
        return *this;
    }
    // data pointers
    constexpr const Scalar* data() const { return data_; }
    constexpr StorageType data() { return data_; }
   private:
    StorageType data_;
};

// vector aliases
template <typename Scalar, int Rows> using Vector = Matrix<Scalar, Rows, 1>;
template <typename Scalar, int Rows> using VectorView = MatrixView<Scalar, Rows, 1>;

}   // namespace fdapde::linalg

#endif   // _FDAPDE_LINALG_MATRIX_H__
