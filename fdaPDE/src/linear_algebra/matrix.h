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

namespace internals {

/// @brief checks matrix shape
template <int Rows, int Cols> constexpr void validate_matrix_shape(int rows, int cols) {
    fdapde_assert(
      !(rows < 0 || cols < 0 || (Rows != Dynamic && rows != Rows) || (Cols != Dynamic && cols != Cols)),
      std::invalid_argument, "matrix dimensions do not match its static shape");
}

/// @brief checks matrix vector size
template <int Rows, int Cols> constexpr void validate_matrix_vector_size(int size) {
    const int static_size = Rows == 1 ? Cols : Rows;
    fdapde_assert(
      !(size < 0 || (Rows != Dynamic && Cols != Dynamic && size != static_size)), std::invalid_argument,
      "vector size does not match its static shape");
}

/// @brief computes the coefficient count after checking shape and overflow
constexpr int checked_matrix_size(int rows, int cols) {
    fdapde_assert(!(rows < 0 || cols < 0), std::invalid_argument, "matrix dimensions must be nonnegative");
    fdapde_strong_assert(
      !(rows != 0 && cols > std::numeric_limits<int>::max() / rows), std::length_error,
      "matrix size exceeds the supported range");
    return rows * cols;
}

/// @brief converts the input length after checking the supported range
constexpr int checked_matrix_data_size(std::size_t size) {
    fdapde_strong_assert(
      !(size > static_cast<std::size_t>(std::numeric_limits<int>::max())), std::length_error,
      "matrix input exceeds the supported range");
    return static_cast<int>(size);
}

/// @brief checks matrix index
constexpr void validate_matrix_index(int i, int j, int rows, int cols) {
    fdapde_assert(!(i < 0 || i >= rows || j < 0 || j >= cols), std::out_of_range, "matrix index out of range");
}

// named callables keep procedural-matrix alias specializations stable across compilers
/// @brief returns the same compile-time scalar at every matrix coordinate
template <typename Scalar, int Value> struct constant_matrix_functor {
    /// @brief returns the configured scalar independently of the coordinate
    constexpr Scalar operator()(int, int) const { return Scalar(Value); }
};

/// @brief generates one on the main diagonal and zero elsewhere
template <typename Scalar> struct identity_matrix_functor {
    /// @brief returns one on the diagonal and zero off the diagonal
    constexpr Scalar operator()(int i, int j) const { return i == j ? Scalar(1) : Scalar(0); }
};

}   // namespace internals

/// @brief views externally owned dense storage
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_> class MatrixView;

namespace internals {

// cRTP writable constraints can form before MatrixView's inherited ReadOnly member is visible
/// @brief identifies a view whose scalar type permits writes
template <typename Scalar, int Rows, int Cols, int StorageOrder>
struct is_mutable_matrix_view<MatrixView<Scalar, Rows, Cols, StorageOrder>> :
    std::bool_constant<!std::is_const_v<Scalar>> { };

}   // namespace internals

// procedural matrices generates matrices whose entries exhibit a fixed pattern, without allocating memory
/// @brief evaluates coefficients from a callable without storing a dense array
template <typename Functor_, int Rows_, int Cols_>
struct ProceduralMatrix : public MatrixExpr<ProceduralMatrix<Functor_, Rows_, Cols_>> {
    fdapde_static_assert((Rows_ == Dynamic || Rows_ > 0) && (Cols_ == Dynamic || Cols_ > 0), INVALID_MATRIX_DIMENSIONS);
    fdapde_static_assert(
      Rows_ == Dynamic || Cols_ == Dynamic ||
        static_cast<std::uint64_t>(Rows_) * static_cast<std::uint64_t>(Cols_) <=
          static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
      MATRIX_SIZE_EXCEEDS_SUPPORTED_RANGE);
    fdapde_static_assert(
      std::is_invocable_v<Functor_ FDAPDE_COMMA int FDAPDE_COMMA int>, FUNCTOR_NOT_CALLABLE_AT_INDEXES_PAIR);
    using Scalar = typename decltype(std::function {std::declval<Functor_>()})::result_type;
    fdapde_static_assert(std::is_arithmetic_v<Scalar>, INVALID_FUNCTOR_RETURN_TYPE);
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = RowMajor;   // memoryless node, choose default
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    /// @brief initializes static dimensions and leaves dynamic axes empty with a default coefficient functor
    constexpr ProceduralMatrix() : rows_(Rows_ == Dynamic ? 0 : Rows), cols_(Cols_ == Dynamic ? 0 : Cols) { }
    /// @brief stores a coefficient functor using static dimensions or empty dynamic axes
    constexpr explicit ProceduralMatrix(Functor_ f) :
        rows_(Rows_ == Dynamic ? 0 : Rows), cols_(Cols_ == Dynamic ? 0 : Cols), f_(f) { }
    /// @brief validates the matrix dimensions and stores the coefficient functor
    constexpr ProceduralMatrix(int rows, int cols, Functor_ f) :
        rows_(Rows == Dynamic ? rows : Rows), cols_(Cols == Dynamic ? cols : Cols), f_(f) {
        internals::validate_matrix_shape<Rows, Cols>(rows, cols);
        (void)internals::checked_matrix_size(rows_, cols_);
    }
    /// @brief validates the matrix dimensions and default-constructs the coefficient functor
    constexpr ProceduralMatrix(int rows, int cols) : ProceduralMatrix(rows, cols, Functor_()) { }
    /// @brief validates the vector length and stores the coefficient functor
    constexpr explicit ProceduralMatrix(int size, Functor_ f) :
        rows_(Rows_ == Dynamic ? size : Rows), cols_(Cols_ == Dynamic ? size : Cols), f_(f) {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        internals::validate_matrix_vector_size<Rows, Cols>(size);
        (void)internals::checked_matrix_size(rows_, cols_);
    }
    /// @brief validates the vector length and default-constructs the coefficient functor
    constexpr explicit ProceduralMatrix(int size) : ProceduralMatrix(size, Functor_()) { }

    /// @brief validates the coordinate and evaluates the stored coefficient functor
    constexpr Scalar operator()(int i, int j) const {
        internals::validate_matrix_index(i, j, rows_, cols_);
        return f_(i, j);
    }
    /// @brief accesses the requested vector coefficient
    constexpr Scalar operator[](int i) const {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        fdapde_assert(!(i < 0 || i >= rows_ * cols_), std::out_of_range, "matrix index out of range");
        return f_(Rows == 1 ? 0 : i, Cols == 1 ? i : 0);
    }
    /// @brief returns the row count
    constexpr int rows() const { return rows_; }
    /// @brief returns the column count
    constexpr int cols() const { return cols_; }
    /// @brief updates the procedural shape without allocating coefficient storage
    constexpr void resize(int rows, int cols) {
        fdapde_static_assert(Rows == Dynamic || Cols == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_MATRICES_ONLY);
        internals::validate_matrix_shape<Rows, Cols>(rows, cols);
        (void)internals::checked_matrix_size(rows, cols);
        rows_ = Rows == Dynamic ? rows : Rows;
        cols_ = Cols == Dynamic ? cols : Cols;
    }
   private:
    int rows_, cols_;
    Functor_ f_;
};
// definition of procedrual matrices
template <typename Scalar, int Rows, int Cols>
using ZeroMatrix = ProceduralMatrix<internals::constant_matrix_functor<Scalar, 0>, Rows, Cols>;
template <typename Scalar, int Rows, int Cols>
using OnesMatrix = ProceduralMatrix<internals::constant_matrix_functor<Scalar, 1>, Rows, Cols>;
template <typename Scalar, int Rows, int Cols>
using IdentityMatrix = ProceduralMatrix<internals::identity_matrix_functor<Scalar>, Rows, Cols>;

namespace internals {

/// @brief assigns matrix coefficients using loops matched to destination storage order
struct generic_assignment_executor {
    /// @brief applies an assignment functor in destination storage order after checking operand shapes
    template <typename DstMatrixType, typename SrcXprType, typename AssignmentOp>
    static constexpr void run(DstMatrixType& dst, const SrcXprType& src, AssignmentOp&& op) {
        fdapde_static_assert(DstMatrixType::ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION);
        if constexpr (!std::is_arithmetic_v<SrcXprType>) {
            fdapde_static_assert(
              (internals::is_dynamic_sized_v<DstMatrixType> || internals::is_dynamic_sized_v<SrcXprType> ||
               internals::same_static_shape_v<DstMatrixType FDAPDE_COMMA SrcXprType>),
              INVALID_ASSIGNMENT__DIFFERENT_LHS_AND_RHS_STATIC_SIZES);
            if constexpr (internals::is_dynamic_sized_v<DstMatrixType> || internals::is_dynamic_sized_v<SrcXprType>) {
                fdapde_assert(
                  !(dst.rows() != src.rows() || dst.cols() != src.cols()), std::invalid_argument,
                  "matrix assignment requires matching shapes");
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
/// @brief assigns vector coefficients independently of row or column orientation
struct vector_assignment_executor {
    /// @brief applies an assignment functor by vector index, allowing row-to-column assignment
    template <typename DstMatrixType, typename SrcXprType, typename AssignmentOp>
    static constexpr void run(DstMatrixType& dst, const SrcXprType& src, AssignmentOp&& op) {
        fdapde_static_assert(DstMatrixType::ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION);
        if constexpr (!std::is_arithmetic_v<SrcXprType>) {
            // vector assignment preserves coefficient order when row and column orientations differ
            fdapde_static_assert(
              internals::is_vector_shaped_v<DstMatrixType>, INVALID_ASSIGNMENT__NOT_VECTOR_SHAPED_LVALUE);
            fdapde_static_assert(
              internals::is_dynamic_sized_v<SrcXprType> || std::decay_t<SrcXprType>::Rows == 1 ||
                std::decay_t<SrcXprType>::Cols == 1,
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
                fdapde_assert(!(!compatible), std::invalid_argument, "vector assignment requires matching sizes");
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

/// @brief provides shared shape and access operations for dense storage
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_, typename MatrixType>
class MatrixBase : public MatrixExpr<MatrixType> {
    fdapde_static_assert((Rows_ == Dynamic || Rows_ > 0) && (Cols_ == Dynamic || Cols_ > 0), INVALID_MATRIX_DIMENSIONS);
    fdapde_static_assert(
      Rows_ == Dynamic || Cols_ == Dynamic ||
        static_cast<std::uint64_t>(Rows_) * static_cast<std::uint64_t>(Cols_) <=
          static_cast<std::uint64_t>(std::numeric_limits<int>::max()),
      MATRIX_SIZE_EXCEEDS_SUPPORTED_RANGE);
    fdapde_static_assert(StorageOrder_ == RowMajor || StorageOrder_ == ColMajor, INVALID_STORAGE_ORDER);
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
    /// @brief initializes dimensions and strides, using zero for each dynamic axis
    constexpr MatrixBase() :
        rows_(Rows == Dynamic ? 0 : Rows),
        cols_(Cols == Dynamic ? 0 : Cols),
        row_stride_(StorageOrder == RowMajor ? cols_ : 1),
        col_stride_(StorageOrder == RowMajor ? 1 : rows_) { }
    /// @brief validates matrix dimensions and derives contiguous-storage strides
    constexpr MatrixBase(int rows, int cols) :
        rows_(Rows == Dynamic ? rows : Rows),
        cols_(Cols == Dynamic ? cols : Cols),
        row_stride_(StorageOrder == RowMajor ? cols_ : 1),
        col_stride_(StorageOrder == RowMajor ? 1 : rows_) {
        internals::validate_matrix_shape<Rows, Cols>(rows, cols);
        (void)internals::checked_matrix_size(rows_, cols_);
    }
    /// @brief validates a vector length and derives strides for its row or column orientation
    constexpr MatrixBase(int size) :
        rows_(Rows == Dynamic ? size : Rows),
        cols_(Cols == Dynamic ? size : Cols),
        row_stride_(StorageOrder == RowMajor ? cols_ : 1),
        col_stride_(StorageOrder == RowMajor ? 1 : rows_) {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        internals::validate_matrix_vector_size<Rows, Cols>(size);
        (void)internals::checked_matrix_size(rows_, cols_);
    }
    // copy assignment
    /// @brief copies a same-type matrix, resizing dynamic dimensions when necessary
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
    /// @brief reads a coefficient through the const storage pointer and layout strides
    constexpr decltype(auto) operator()(int i, int j) const {
        internals::validate_matrix_index(i, j, rows_, cols_);
        return derived().data()[i * row_stride_ + j * col_stride_];
    }
    /// @brief accesses the requested vector coefficient
    constexpr decltype(auto) operator[](int i) const
        requires(Rows == 1 || Cols == 1)
    {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        fdapde_assert(!(i < 0 || i >= rows_ * cols_), std::out_of_range, "matrix index out of range");
        return derived().data()[i];
    }
    /// @brief accesses a coefficient through the storage pointer and layout strides
    constexpr decltype(auto) operator()(int i, int j) {
        internals::validate_matrix_index(i, j, rows_, cols_);
        return derived().data()[i * row_stride_ + j * col_stride_];
    }
    /// @brief accesses the requested vector coefficient
    constexpr decltype(auto) operator[](const int i)
        requires(Rows == 1 || Cols == 1)
    {
        fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        fdapde_assert(!(i < 0 || i >= rows_ * cols_), std::out_of_range, "matrix index out of range");
        return derived().data()[i];
    }
    // observers
    /// @brief returns the row count
    constexpr int rows() const { return Rows != Dynamic ? Rows : rows_; }
    /// @brief returns the column count
    constexpr int cols() const { return Cols != Dynamic ? Cols : cols_; }
   protected:
    int rows_, cols_;
    int row_stride_, col_stride_;
};

/// @brief owns a dense matrix with static or runtime dimensions
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_ = RowMajor>
class Matrix : public MatrixBase<Scalar_, Rows_, Cols_, StorageOrder_, Matrix<Scalar_, Rows_, Cols_, StorageOrder_>> {
   private:
    using Base = MatrixBase<Scalar_, Rows_, Cols_, StorageOrder_, Matrix<Scalar_, Rows_, Cols_, StorageOrder_>>;
    static constexpr bool HasSupportedStaticStorage = Rows_ > 0 && Cols_ > 0 &&
                                                      static_cast<std::uint64_t>(Rows_) *
                                                          static_cast<std::uint64_t>(Cols_) <=
                                                        static_cast<std::uint64_t>(std::numeric_limits<int>::max());
    static constexpr std::size_t StorageSize =
      HasSupportedStaticStorage ? static_cast<std::size_t>(Rows_) * static_cast<std::size_t>(Cols_) : 0;
    using StorageType =
      std::conditional_t<Rows_ == Dynamic || Cols_ == Dynamic, std::vector<Scalar_>, std::array<Scalar_, StorageSize>>;
   public:
    using Scalar = Scalar_;
    using iterator = typename StorageType::iterator;
    using const_iterator = typename StorageType::const_iterator;
    static constexpr int NestAsRef = 1;

    /// @brief value-initializes fixed storage and leaves dynamic storage empty
    constexpr Matrix() : data_() { }
    // copy semantic
    /// @brief copies the source shape and coefficients into independent storage
    constexpr Matrix(const Matrix& other) : Base() { clone_(other); }
    /// @brief copies shape and coefficients into this owner's independent storage
    constexpr Matrix& operator=(const Matrix& other) & {
        clone_(other);
        return *this;
    }
    /// @brief evaluates a matrix expression into independent storage
    template <typename RhsXprType_>   // construct from plain MatrixExpr
    constexpr Matrix(const MatrixExpr<RhsXprType_>& rhs) : Base(), data_() {
        clone_(rhs.derived());
    }
    /// @brief evaluates a coefficient-wise expression into independent matrix storage
    template <typename RhsXprType_>
    constexpr Matrix(const MatrixCoeffWiseExpr<RhsXprType_>& rhs) : Matrix(rhs.mwise()) { }

    // matrix API
    // value-initialized static-sized matrix, avoid 1D vectors
    /// @brief fills a fixed-size matrix with the supplied scalar
    constexpr explicit Matrix(Scalar v)
        requires(Rows_ > 1 && Cols_ >= 1)
        : data_() {
        for (int i = 0; i < Rows_ * Cols_; ++i) { data_[i] = v; }
    }
    /// @brief initializes coefficients to zero after validating the requested dimensions
    constexpr Matrix(int rows, int cols)
        requires(Rows_ != 1 && Cols_ != 1)
        : Base(rows, cols), data_() {
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) {
            data_.resize(static_cast<std::size_t>(internals::checked_matrix_size(this->rows(), this->cols())));
        }
    }
    // value-initialized dynamic-sized matrix, avoid vectors
    /// @brief allocates a dynamic matrix and fills its coefficients with the supplied scalar
    constexpr Matrix(int rows, int cols, Scalar v)
        requires(Rows_ == Dynamic && Cols_ == Dynamic)
        : Matrix(rows, cols) {
        for (int i = 0, n = this->size(); i < n; ++i) { data_[i] = v; }
    }

    // vector API
    // zero-initialized dynamic-sized vector
    /// @brief allocates a dynamic vector with value-initialized coefficients
    constexpr explicit Matrix(int size)
        requires(Rows_ == Dynamic || Cols_ == Dynamic)
        : Base(size) {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) {
            data_.resize(static_cast<std::size_t>(internals::checked_matrix_size(this->rows(), this->cols())));
        }
    }
    // value-initialized dynamic-sized vector
    /// @brief allocates a dynamic vector and fills it with the supplied scalar
    constexpr Matrix(int size, Scalar v)
        requires((Rows_ == Dynamic && Cols_ == 1) || (Rows_ == 1 && Cols_ == Dynamic))
        : Matrix(size) {
        fdapde_static_assert(
          (Rows_ == 1 && Cols_ == Dynamic) || (Cols_ == 1 && Rows_ == Dynamic),
          THIS_METHOD_IS_FOR_ROW_OR_COLUMN_DYNAMIC_SIZED_VECTORS_ONLY);
        for (int i = 0; i < size; ++i) { data_[i] = v; }
    }
    // 1D static-sized vector
    /// @brief initializes the sole coefficient of a one-dimensional vector
    constexpr explicit Matrix(Scalar x)
        requires(!internals::is_dynamic_sized_v<Base> && (Cols_ == 1 && Rows_ == 1))
        : Base() {
        fdapde_static_assert(Cols_ == 1 && Rows_ == 1, THIS_METHOD_IS_FOR_1D_VECTORS_ONLY);
        data_[0] = x;
    }
    // 2D static-sized vector
    /// @brief initializes a fixed two-dimensional vector from its coordinates
    constexpr Matrix(Scalar x, Scalar y)
        requires(!internals::is_dynamic_sized_v<Base> && (Cols_ == 1 || Rows_ == 1))
        : Base() {
        fdapde_static_assert(
          (Cols_ == 2 && Rows_ == 1) || (Cols_ == 1 && Rows_ == 2), THIS_METHOD_IS_FOR_2D_VECTORS_ONLY);
        data_[0] = x;
        data_[1] = y;
    }
    // 3D static-sized vector
    /// @brief initializes a fixed three-dimensional vector from its coordinates
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
    /// @brief copies row-ordered input, inferring the length only for dynamic vectors
    constexpr explicit Matrix(const std::vector<Scalar>& data) :
        Base(
          Rows_ == Dynamic ? internals::checked_matrix_data_size(data.size()) : Rows_,
          Cols_ == Dynamic ? internals::checked_matrix_data_size(data.size()) : Cols_),
        data_() {
        fdapde_static_assert(
          (Rows_ != Dynamic && Cols_ != Dynamic) || (Rows_ == 1 && Cols_ == Dynamic) ||
            (Cols_ == 1 && Rows_ == Dynamic),
          THIS_METHOD_IS_NOT_FOR_DYNAMIC_SIZED_MATRICES);
        const int expected_size = internals::checked_matrix_size(this->rows(), this->cols());
        fdapde_assert(
          !(!std::cmp_equal(expected_size FDAPDE_COMMA data.size())), std::invalid_argument,
          "matrix input size does not match its shape");
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) { data_.resize(data.size()); }
        const int rows = this->rows();
        const int cols = this->cols();
        for (int i = 0, n = rows; i < n; ++i) {
            for (int j = 0, m = cols; j < m; ++j) { this->operator()(i, j) = data[i * cols + j]; }
        }
    }
    /// @brief copies a fixed-size C array into matrix coordinates in logical row order
    template <std::size_t Size> constexpr explicit Matrix(const Scalar (&data)[Size]) : Base() {
        fdapde_static_assert(
          Rows_ != Dynamic && Cols_ != Dynamic && StorageSize == Size, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        for (int i = 0; i < Rows_; ++i) {
            for (int j = 0; j < Cols_; ++j) { this->operator()(i, j) = data[i * Cols_ + j]; }
        }
    }
    /// @brief copies initializer values into a vector, resizing only a dynamic length
    constexpr Matrix& operator=(const std::initializer_list<Scalar>& data) & {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        const int size = internals::checked_matrix_data_size(data.size());
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) {
            resize(size);
        } else
            fdapde_assert(
              !(this->size() != size), std::invalid_argument, "vector input size does not match its static size");
        int i = 0;
        for (Scalar v : data) { this->operator[](i++) = v; }
        return *this;
    }

    // static named constructors
    /// @brief returns an expression with every coefficient equal to zero
    static constexpr auto Zero() { return ZeroMatrix<Scalar_, Rows_, Cols_>(); }
    /// @brief returns an expression with every coefficient equal to zero
    static constexpr auto Zero(int size) {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return ZeroMatrix < Scalar_, Rows_ == 1 ? Rows_ : Dynamic, Cols_ == 1 ? Cols_ : Dynamic > (size);
    }
    /// @brief returns an expression with every coefficient equal to zero
    static constexpr auto Zero(int rows, int cols) { return ZeroMatrix<Scalar_, Dynamic, Dynamic>(rows, cols); }
    /// @brief returns an expression with every coefficient equal to one
    static constexpr auto Ones() { return OnesMatrix<Scalar_, Rows_, Cols_>(); }
    /// @brief returns an expression with every coefficient equal to one
    static constexpr auto Ones(int size) {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return OnesMatrix < Scalar_, Rows_ == 1 ? Rows_ : Dynamic, Cols_ == 1 ? Cols_ : Dynamic > (size);
    }
    /// @brief returns an expression with every coefficient equal to one
    static constexpr auto Ones(int rows, int cols) { return OnesMatrix<Scalar_, Dynamic, Dynamic>(rows, cols); }
    /// @brief returns an expression with a constant coefficient value
    static constexpr auto Constant(Scalar value) { return value * Ones(); }
    /// @brief returns an expression with a constant coefficient value
    static constexpr auto Constant(int size, Scalar value) {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        return value * Ones(size);
    }
    /// @brief returns an expression with a constant coefficient value
    static constexpr auto Constant(int rows, int cols, Scalar value) { return value * Ones(rows, cols); }
    /// @brief returns a vector of evenly spaced coefficients
    static constexpr auto LinSpaced(int rows, Scalar a, Scalar b) {
        fdapde_static_assert(Cols_ == 1, THIS_METHOD_IS_FOR_COLUMN_VECTORS_ONLY);
        fdapde_assert(!(rows < 2), std::invalid_argument, "LinSpaced requires at least two points");
        auto linspace = [a, h = (double)(b - a) / double(rows - 1)](int i, int) { return a + i * h; };
        return ProceduralMatrix<decltype(linspace), Dynamic, 1>(rows, 1, linspace);
    }
    /// @brief returns a vector of evenly spaced coefficients
    static constexpr auto LinSpaced(Scalar a, Scalar b) {
        fdapde_static_assert(Cols_ == 1, THIS_METHOD_IS_FOR_COLUMN_VECTORS_ONLY);
        fdapde_static_assert(Rows_ != Dynamic && Rows_ > 1, LINSPACED_REQUIRES_AT_LEAST_TWO_STATIC_POINTS);
        auto linspace = [a, h = (double)(b - a) / double(Rows_ - 1)](int i, int) { return a + i * h; };
        return ProceduralMatrix<decltype(linspace), Rows_, 1>(linspace);
    }

    // modifiers
    /// @brief updates dimensions and strides while retaining the common prefix of physical storage
    void resize(int rows, int cols) {
        fdapde_static_assert(Rows_ == Dynamic || Cols_ == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_MATRICES_ONLY);
        internals::validate_matrix_shape<Rows_, Cols_>(rows, cols);
        const int rows_ = Rows_ == Dynamic ? rows : Rows_;
        const int cols_ = Cols_ == Dynamic ? cols : Cols_;
        if (rows_ == this->rows_ && cols_ == this->cols_) return;   // do not reallocate memory if sizes didn't changed
        const int size_ = internals::checked_matrix_size(rows_, cols_);
        data_.resize(static_cast<std::size_t>(size_));
        // update strides only after the new storage is valid
        this->rows_ = rows_;
        this->cols_ = cols_;
        this->row_stride_ = StorageOrder_ == RowMajor ? cols_ : 1;
        this->col_stride_ = StorageOrder_ == RowMajor ? 1 : rows_;
        return;
    }
    /// @brief changes a dynamic vector length while retaining its existing coefficient prefix
    void resize(int size) {
        fdapde_static_assert(
          (Rows_ == 1 && Cols_ == Dynamic) || (Cols_ == 1 && Rows_ == Dynamic),
          THIS_METHOD_IS_FOR_DYNAMIC_SIZED_ROW_OR_COLUMN_VECTORS_ONLY);
        resize(Rows_ == Dynamic ? size : Rows_, Cols_ == Dynamic ? size : Cols_);
        return;
    }
    /// @brief sets every coefficient to zero
    void set_zero() {
        for (int i = 0, n = data_.size(); i < n; ++i) { data_[i] = Scalar(0); }
        return;
    }
    // data pointers
    /// @brief returns the underlying storage pointer
    constexpr const Scalar* data() const { return data_.data(); }
    /// @brief returns the underlying storage pointer
    constexpr Scalar* data() { return data_.data(); }
    // iterators
    /// @brief returns an iterator to the first coefficient
    constexpr iterator begin() { return data_.begin(); }
    /// @brief returns an iterator to the first coefficient
    constexpr const_iterator begin() const { return data_.begin(); }
    /// @brief returns the past-the-end iterator
    constexpr iterator end() { return data_.end(); }
    /// @brief returns the past-the-end iterator
    constexpr const_iterator end() const { return data_.end(); }
   private:
    /// @brief resizes dynamic storage if needed and copies source coefficients in logical coordinates
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
/// @brief views externally owned dense storage
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_ = RowMajor>
class MatrixView :
    public MatrixBase<Scalar_, Rows_, Cols_, StorageOrder_, MatrixView<Scalar_, Rows_, Cols_, StorageOrder_>> {
   private:
    using Base = MatrixBase<Scalar_, Rows_, Cols_, StorageOrder_, MatrixView<Scalar_, Rows_, Cols_, StorageOrder_>>;
    using XprBase = MatrixExpr<MatrixView<Scalar_, Rows_, Cols_, StorageOrder_>>;
    using StorageType = std::add_pointer_t<Scalar_>;
   public:
    using Scalar = Scalar_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = Base::ReadOnly;

    // constructors
    /// @brief copies the pointer and dimensions while sharing the same external storage
    constexpr MatrixView(const MatrixView&) = default;
    /// @brief creates an empty dynamic view without binding external storage
    constexpr MatrixView()
        requires(Rows_ == Dynamic || Cols_ == Dynamic)
        : Base(), data_(nullptr) { }
    /// @brief rejects a fixed-size view without an explicit storage pointer
    constexpr MatrixView()
        requires(Rows_ != Dynamic && Cols_ != Dynamic)
    = delete;
    /// @brief binds external storage using the compile-time matrix dimensions
    constexpr explicit MatrixView(Scalar* data) : Base(), data_(data) {
        fdapde_static_assert(Rows_ != Dynamic && Cols_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
    }
    /// @brief binds external storage as a vector of the requested positive length
    constexpr MatrixView(Scalar* data, int size) : Base(size), data_(data) {
        fdapde_static_assert(Rows_ == 1 || Cols_ == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
        fdapde_assert(!(size <= 0), std::invalid_argument, "matrix view size must be positive");
    }
    /// @brief binds external storage with positive dimensions and the selected storage order
    constexpr MatrixView(Scalar* data, int rows, int cols) : Base(rows, cols), data_(data) {
        fdapde_assert(!(rows <= 0 || cols <= 0), std::invalid_argument, "matrix view dimensions must be positive");
    }
    // inherit expression assignment without the owner-style MatrixBase copy assignment
    using XprBase::operator=;
    /// @brief copies logical coefficients into the bound storage without rebinding the view
    constexpr MatrixView& operator=(const MatrixView& other) &
        requires(ReadOnly == 0)
    {
        static_cast<XprBase&>(*this).template operator= <MatrixView>(other);
        return *this;
    }
    /// @brief copies into a temporary view's bound storage and returns the view by value
    constexpr MatrixView operator=(const MatrixView& other) &&
      requires(ReadOnly == 0) {
          static_cast<XprBase&>(*this).template operator= <MatrixView>(other);
          return *this;
      }
      // data pointers
      /// @brief returns the underlying storage pointer
      constexpr const Scalar* data() const {
        return data_;
    }
    /// @brief returns the underlying storage pointer
    constexpr StorageType data() { return data_; }
   private:
    StorageType data_;
};

// vector aliases
template <typename Scalar, int Rows> using Vector = Matrix<Scalar, Rows, 1>;
template <typename Scalar, int Rows> using VectorView = MatrixView<Scalar, Rows, 1>;

}   // namespace fdapde

#endif   // _FDAPDE_LINALG_MATRIX_H__
