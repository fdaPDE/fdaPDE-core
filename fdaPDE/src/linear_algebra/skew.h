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

#ifndef __FDAPDE_LINALG_SKEW_H__
#define __FDAPDE_LINALG_SKEW_H__

#include "header_check.h"

namespace fdapde {

/// @brief provides expression operations that preserve skew symmetry
template <typename XprType> struct SkewSymmetricMatrixExpr;
/// @brief views external strict-upper storage as a skew-symmetric matrix
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_> class SkewSymmetricMatrixView;

namespace internals {

/// @brief identifies a view whose scalar type permits writes
template <typename Scalar, int Rows, int Cols, int StorageOrder>
struct is_mutable_matrix_view<SkewSymmetricMatrixView<Scalar, Rows, Cols, StorageOrder>> :
    std::bool_constant<!std::is_const_v<Scalar>> { };

}   // namespace internals

namespace internals {

/// @brief infers the skew-symmetric dimension from packed storage size
constexpr int compute_skew_symmetric_shape(int storage_size) {
    if (storage_size < 0) return -1;
    int size = 0;
    while (static_cast<long long>(size) * (size - 1) / 2 < storage_size) ++size;
    return static_cast<long long>(size) * (size - 1) / 2 == storage_size ? size : -1;
}

/// @brief validates the skew-symmetric matrix dimension
constexpr int checked_skew_symmetric_shape(std::size_t size) {
    const int input_size = checked_matrix_data_size(size);
    const int shape = compute_skew_symmetric_shape(input_size);
    fdapde_assert(!(shape < 0), std::invalid_argument, "packed skew-symmetric input has an invalid length");
    return shape;
}

/// @brief computes the packed skew-symmetric size with overflow checks
constexpr int checked_skew_symmetric_storage_size(int rows) {
    (void)checked_matrix_size(rows, rows);
    return static_cast<int>(static_cast<long long>(rows) * (rows - 1) / 2);
}

/// @brief assigns the independent strict-upper coefficients of a skew-symmetric matrix
struct skew_symmetric_assignment_executor {
    /// @brief applies the assignment functor only to strict-upper entries, preserving the implicit diagonal and
    /// reflection
    template <typename DstXprType, typename SrcXprType, typename AssignmentOp>
    static constexpr void run(DstXprType& dst, const SrcXprType& src, AssignmentOp&& op) {
        fdapde_static_assert(DstXprType::ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION);
        if constexpr (!std::is_arithmetic_v<SrcXprType>) {
            fdapde_static_assert(
              internals::is_dynamic_sized_v<DstXprType> || internals::is_dynamic_sized_v<SrcXprType> ||
                internals::same_static_shape_v<DstXprType FDAPDE_COMMA SrcXprType>,
              INVALID_ASSIGNMENT__DIFFERENT_LHS_AND_RHS_STATIC_SHAPES);
            fdapde_assert(
              !(dst.rows() != src.rows() || dst.cols() != src.cols()), std::invalid_argument,
              "skew-symmetric assignment requires matching dimensions");
        }
        auto fetch = [](const SrcXprType& source, [[maybe_unused]] int i, [[maybe_unused]] int j) -> decltype(auto) {
            if constexpr (std::is_arithmetic_v<SrcXprType>) {
                return source;
            } else {
                return source(i, j);
            }
        };
        for (int i = 0, size = dst.rows(); i < size; ++i) {
            for (int j = i + 1; j < size; ++j) {
                auto& target = dst.data()[i * (2 * size - i - 1) / 2 + (j - i - 1)];
                op(target, fetch(src, i, j));
            }
        }
    }
};

/// @brief reflects one selected triangle with a sign change and supplies a zero diagonal
template <int ViewMode_, typename SkewXprType_>
class skew_symmetric_wrapper : public SkewSymmetricMatrixExpr<skew_symmetric_wrapper<ViewMode_, SkewXprType_>> {
   private:
    fdapde_static_assert(ViewMode_ == Lower || ViewMode_ == Upper, VIEW_MODE_MUST_BE_EITHER_LOWER_OR_UPPER);
    using Base = SkewSymmetricMatrixExpr<skew_symmetric_wrapper<ViewMode_, SkewXprType_>>;
    using XprType = std::remove_reference_t<SkewXprType_>;
    using XprTypeClean = std::remove_cv_t<XprType>;
    using XprTypeNested = internals::ref_select_t<SkewXprType_>;
    static constexpr int ViewMode = ViewMode_;
   public:
    using Scalar = typename XprTypeClean::Scalar;
    static constexpr int Rows = XprTypeClean::Rows;
    static constexpr int Cols = XprTypeClean::Cols;
    static constexpr int StorageOrder = XprTypeClean::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    /// @brief copies the skew-symmetric adaptor while retaining its nested expression
    constexpr skew_symmetric_wrapper(const skew_symmetric_wrapper&) = default;
    /// @brief borrows one triangle, negates its reflection and supplies a zero diagonal
    template <typename XprType__>
        requires(!std::same_as<std::remove_cvref_t<XprType__>, skew_symmetric_wrapper> &&
                 internals::safely_nestable<XprTypeNested, XprType__>)
    constexpr explicit skew_symmetric_wrapper(XprType__&& xpr) :
        Base(), xpr_(std::forward<XprType__>(xpr)), size_(xpr_.rows() == xpr_.cols() ? xpr_.rows() : 0) {
        fdapde_static_assert(
          Rows == Dynamic || Cols == Dynamic || Rows == Cols, THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
        fdapde_assert(
          !(xpr_.rows() < 0 || xpr_.rows() != xpr_.cols()), std::invalid_argument,
          "skew-symmetric view requires square dimensions");
    }
    /// @brief reflects the selected triangle with a sign change and returns zero on the diagonal
    constexpr Scalar operator()(int i, int j) const {
        internals::validate_matrix_index(i, j, size_, size_);
        if (i == j) return Scalar(0);
        if constexpr (ViewMode == Upper) { return i < j ? xpr_(i, j) : -xpr_(j, i); }
        if constexpr (ViewMode == Lower) { return i > j ? xpr_(i, j) : -xpr_(j, i); }
    }
    /// @brief returns the row count
    constexpr int rows() const { return size_; }
    /// @brief returns the column count
    constexpr int cols() const { return size_; }
   private:
    XprTypeNested xpr_;
    int size_;
};

/// @brief adapts an expression to skew-symmetric matrix operations
template <int ViewMode_, typename XprType_> constexpr auto skew_symmetric_cast(XprType_&& xpr) {
    return skew_symmetric_wrapper<ViewMode_, XprType_>(std::forward<XprType_>(xpr));
}

}   // namespace internals

/// @brief provides expression operations that preserve skew symmetry
template <typename XprType_> struct SkewSymmetricMatrixExpr : public MatrixExpr<XprType_> {
    using XprType = std::decay_t<XprType_>;
    using Base = MatrixExpr<XprType_>;
    using Base::derived;
    using Base::operator=;
    using Base::operator*=;

    /// @brief rejects matrix compound multiplication because a general product need not be skew-symmetric
    template <internals::matrix_expression RhsXprType> constexpr void operator*=(const RhsXprType&) = delete;
};

/// @brief maps square coordinates to packed strict-upper storage and reflection signs
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_, typename SkewMatrixType>
class SkewSymmetricMatrixBase : public SkewSymmetricMatrixExpr<SkewMatrixType> {
   protected:
    using Base = SkewSymmetricMatrixExpr<SkewMatrixType>;
    using Base::derived;
   public:
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int ReadOnly = std::is_const_v<Scalar_>;
    using Base::operator=;

    /// @brief reads signed reflected values and prevents nonzero writes to the implicit diagonal
    template <typename Scalar__>
        requires(std::is_same_v<std::remove_cv_t<Scalar>, std::remove_cv_t<Scalar__>>)
    class skew_symmetric_proxy {
       private:
        using Value = std::remove_const_t<Scalar__>;
       public:
        /// @brief records the packed upper-triangle index, reflection sign and structural diagonal flag
        constexpr skew_symmetric_proxy(Scalar__* data, int i, int j, int size) :
            data_(data),
            index_(compute_linear_index_(i < j ? i : j, i < j ? j : i, size)),
            sign_flip_(i > j),
            diagonal_(i == j) { }
        /// @brief stores the sign-adjusted coefficient, rejecting nonzero writes to the implicit diagonal
        template <typename T>
            requires(std::is_convertible_v<T, Value> && !std::is_const_v<Scalar__>)
        constexpr skew_symmetric_proxy& operator=(T value) {
            const Value converted = static_cast<Value>(value);
            if (diagonal_) {
                fdapde_assert(
                  !(converted != Value(0)), std::invalid_argument, "skew-symmetric diagonal coefficients must be zero");
                return *this;
            }
            data_[index_] = sign_flip_ ? -converted : converted;
            return *this;
        }
        /// @brief reads zero on the diagonal and applies the reflected sign elsewhere
        constexpr operator Value() const {
            if (diagonal_) return Value(0);
            const Value value = data_[index_];
            return value == Value(0) || !sign_flip_ ? value : -value;
        }
       private:
        /// @brief maps a matrix coordinate to packed storage
        static constexpr int compute_linear_index_(int i, int j, int size) {
            return i * (2 * size - i - 1) / 2 + (j - i - 1);
        }
        Scalar__* data_;
        int index_;
        bool sign_flip_;
        bool diagonal_;
    };
    using reference = skew_symmetric_proxy<Scalar>;
    using const_reference = skew_symmetric_proxy<const std::remove_const_t<Scalar>>;

    /// @brief initializes fixed square dimensions or an empty dynamic shape
    constexpr SkewSymmetricMatrixBase() : rows_(default_shape_()), cols_(default_shape_()) { }
    /// @brief validates square dimensions and their agreement with static extents
    constexpr SkewSymmetricMatrixBase(int rows, int cols) : rows_(rows), cols_(cols) {
        internals::validate_matrix_shape<Rows, Cols>(rows, cols);
        (void)internals::checked_matrix_size(rows, cols);
        fdapde_assert(!(rows != cols), std::invalid_argument, "skew-symmetric matrix requires square dimensions");
    }
    /// @brief returns a read-only proxy applying the reflected sign and implicit zero diagonal
    constexpr const_reference operator()(int i, int j) const {
        internals::validate_matrix_index(i, j, rows_, cols_);
        return const_reference(derived().data(), i, j, rows_);
    }
    /// @brief returns a writable proxy preserving reflected signs and the zero diagonal
    constexpr reference operator()(int i, int j)
        requires(ReadOnly == 0)
    {
        internals::validate_matrix_index(i, j, rows_, cols_);
        return reference(derived().data(), i, j, rows_);
    }
    /// @brief returns the row count
    constexpr int rows() const { return rows_; }
    /// @brief returns the column count
    constexpr int cols() const { return cols_; }
    /// @brief returns the number of stored coefficients
    constexpr int storage_size() const { return rows_ * (rows_ - 1) / 2; }
   protected:
    /// @brief returns the default matrix dimensions
    static constexpr int default_shape_() {
        if constexpr (Rows != Dynamic) return Rows;
        if constexpr (Cols != Dynamic) return Cols;
        return 0;
    }
    int rows_, cols_;
};

// skew-symmetric matrices form a vector space. Products generally do not preserve the structure
/// @brief adds equally shaped operands while preserving skew symmetry
template <typename LhsXprType, typename RhsXprType>
constexpr auto
operator+(const SkewSymmetricMatrixExpr<LhsXprType>& lhs, const SkewSymmetricMatrixExpr<RhsXprType>& rhs) {
    return internals::skew_symmetric_cast<Upper>(
      MatrixBinOp<LhsXprType, RhsXprType, std::plus<>>(lhs.derived(), rhs.derived(), std::plus<>()));
}
/// @brief subtracts equally shaped operands while preserving skew symmetry
template <typename LhsXprType, typename RhsXprType>
constexpr auto
operator-(const SkewSymmetricMatrixExpr<LhsXprType>& lhs, const SkewSymmetricMatrixExpr<RhsXprType>& rhs) {
    return internals::skew_symmetric_cast<Upper>(
      MatrixBinOp<LhsXprType, RhsXprType, std::minus<>>(lhs.derived(), rhs.derived(), std::minus<>()));
}
/// @brief scales by the right scalar while preserving skew symmetry
template <typename XprType, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator*(const SkewSymmetricMatrixExpr<XprType>& lhs, ScalarType rhs) {
    return internals::skew_symmetric_cast<Upper>(MatrixScalarMultiplicationOp<XprType, ScalarType>(lhs.derived(), rhs));
}
/// @brief scales by the left scalar while preserving skew symmetry
template <typename XprType, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator*(ScalarType lhs, const SkewSymmetricMatrixExpr<XprType>& rhs) {
    return rhs * lhs;
}
/// @brief divides each coefficient by the scalar while preserving skew symmetry
template <typename XprType, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator/(const SkewSymmetricMatrixExpr<XprType>& lhs, ScalarType rhs) {
    return internals::skew_symmetric_cast<Upper>(static_cast<const MatrixExpr<XprType>&>(lhs) / rhs);
}

/// @brief owns a skew-symmetric matrix
template <typename Scalar_, int Rows_, int Cols_ = Rows_, int StorageOrder_ = RowMajor>
class SkewSymmetricMatrix :
    public SkewSymmetricMatrixBase<
      Scalar_, Rows_, Cols_, StorageOrder_, SkewSymmetricMatrix<Scalar_, Rows_, Cols_, StorageOrder_>> {
   private:
    using Base = SkewSymmetricMatrixBase<
      Scalar_, Rows_, Cols_, StorageOrder_, SkewSymmetricMatrix<Scalar_, Rows_, Cols_, StorageOrder_>>;
    static constexpr int StaticShape = Rows_ != Dynamic ? Rows_ : Cols_;
    static constexpr bool HasSupportedStaticStorage =
      StaticShape == Dynamic || static_cast<std::uint64_t>(StaticShape) * static_cast<std::uint64_t>(StaticShape) <=
                                  static_cast<std::uint64_t>(std::numeric_limits<int>::max());
   public:
    fdapde_static_assert((Rows_ == Dynamic || Rows_ > 0) && (Cols_ == Dynamic || Cols_ > 0), INVALID_MATRIX_DIMENSIONS);
    fdapde_static_assert(
      Rows_ == Dynamic || Cols_ == Dynamic || Rows_ == Cols_, THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
    fdapde_static_assert(StorageOrder_ == RowMajor, PACKED_COL_MAJOR_STRUCTURED_STORAGE_IS_NOT_SUPPORTED);
    fdapde_static_assert(HasSupportedStaticStorage, MATRIX_SIZE_EXCEEDS_SUPPORTED_RANGE);
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int StorageSize =
      StaticShape == Dynamic || !HasSupportedStaticStorage ? Dynamic : StaticShape * (StaticShape - 1) / 2;
    static constexpr int NestAsRef = 1;
    static constexpr int ReadOnly = std::is_const_v<Scalar_>;
    using assignment_executor = internals::skew_symmetric_assignment_executor;
    using StorageType = std::conditional_t < StorageSize == Dynamic, std::vector<Scalar_>,
          std::array<Scalar_, StorageSize<0 ? 0 : static_cast<std::size_t>(StorageSize)>>;

    /// @brief value-initializes fixed packed storage and leaves dynamic storage empty
    constexpr SkewSymmetricMatrix() : Base(), data_() { }
    /// @brief copies the shape and strict upper triangle into independent storage
    constexpr SkewSymmetricMatrix(const SkewSymmetricMatrix& rhs) : Base(rhs.rows(), rhs.cols()), data_(rhs.data_) { }
    /// @brief copies the source shape and independent packed upper coefficients
    constexpr SkewSymmetricMatrix& operator=(const SkewSymmetricMatrix& rhs) & {
        if (this == std::addressof(rhs)) return *this;
        data_ = rhs.data_;
        if constexpr (Rows == Dynamic || Cols == Dynamic) {
            this->rows_ = rhs.rows();
            this->cols_ = rhs.cols();
        }
        return *this;
    }
    /// @brief rejects assignment to a temporary skew-symmetric owner
    constexpr void operator=(const SkewSymmetricMatrix&) && = delete;
    using Base::operator=;

    /// @brief allocates a square matrix of the requested dynamic dimension
    constexpr explicit SkewSymmetricMatrix(int size) : SkewSymmetricMatrix(size, size) { }
    /// @brief validates square dimensions and allocates the strict upper triangle
    constexpr SkewSymmetricMatrix(int rows, int cols) : Base(rows, cols), data_() {
        fdapde_static_assert(Rows == Dynamic || Cols == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_MATRICES_ONLY);
        const int storage_size = internals::checked_skew_symmetric_storage_size(this->rows_);
        if constexpr (StorageSize == Dynamic) { data_.resize(static_cast<std::size_t>(storage_size)); }
    }
    /// @brief evaluates the strict upper triangle of a skew-symmetric expression into owned storage
    template <typename RhsXprType_>
    constexpr SkewSymmetricMatrix(const SkewSymmetricMatrixExpr<RhsXprType_>& rhs) :
        Base(rhs.rows(), rhs.cols()), data_() {
        if constexpr (StorageSize == Dynamic) {
            data_.resize(static_cast<std::size_t>(internals::checked_skew_symmetric_storage_size(this->rows_)));
        }
        assignment_executor::run(*this, rhs.derived(), [](auto& l, const auto& r) { l = r; });
    }
    /// @brief snapshots a skew expression before assigning its independent coefficients
    template <typename RhsXprType_>
    constexpr SkewSymmetricMatrix& operator=(const SkewSymmetricMatrixExpr<RhsXprType_>& rhs) & {
        static_cast<MatrixExpr<SkewSymmetricMatrix>&>(*this).template operator= <RhsXprType_>(rhs);
        return *this;
    }
    /// @brief copies packed strict-upper coefficients and infers the square dimension from their count
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr explicit SkewSymmetricMatrix(const std::vector<Scalar__>& data) :
        Base(
          internals::checked_skew_symmetric_shape(data.size()), internals::checked_skew_symmetric_shape(data.size())),
        data_() {
        const int input_size = internals::checked_matrix_data_size(data.size());
        if constexpr (StorageSize == Dynamic) { data_.resize(data.size()); }
        fdapde_assert(
          !(!std::cmp_equal(data_.size(), input_size)), std::invalid_argument,
          "packed skew-symmetric input does not match its static size");
        for (int i = 0; i < input_size; ++i) data_[static_cast<std::size_t>(i)] = data[static_cast<std::size_t>(i)];
    }
    /// @brief copies a C array matching the fixed strict-upper storage size
    template <typename Scalar__, std::size_t Size>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr explicit SkewSymmetricMatrix(const Scalar__ (&data)[Size]) : Base(), data_() {
        fdapde_static_assert(
          Rows_ != Dynamic && Cols_ != Dynamic && StorageSize == static_cast<int>(Size),
          THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        for (int i = 0; i < StorageSize; ++i) data_[static_cast<std::size_t>(i)] = data[static_cast<std::size_t>(i)];
    }
    /// @brief resizes the owned storage to the requested dimensions
    void resize(int size) { resize(size, size); }
    /// @brief resizes the owned storage to the requested dimensions
    void resize(int rows, int cols) {
        fdapde_static_assert(Rows == Dynamic || Cols == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_MATRICES_ONLY);
        internals::validate_matrix_shape<Rows, Cols>(rows, cols);
        const int effective_rows = Rows == Dynamic ? rows : Rows;
        const int effective_cols = Cols == Dynamic ? cols : Cols;
        fdapde_assert(
          !(effective_rows != effective_cols), std::invalid_argument,
          "skew-symmetric matrix resize requires square dimensions");
        const int storage_size = internals::checked_skew_symmetric_storage_size(effective_rows);
        if (
          this->rows_ == effective_rows && this->cols_ == effective_cols &&
          std::cmp_equal(data_.size(), storage_size)) {
            return;
        }
        if constexpr (StorageSize == Dynamic) { data_.resize(static_cast<std::size_t>(storage_size)); }
        this->rows_ = effective_rows;
        this->cols_ = effective_cols;
    }
    /// @brief returns the underlying storage pointer
    constexpr const std::remove_const_t<Scalar_>* data() const { return data_.data(); }
    /// @brief returns the underlying storage pointer
    constexpr Scalar_* data() { return data_.data(); }
   private:
    StorageType data_;
};

/// @brief views external strict-upper storage as a skew-symmetric matrix
template <typename Scalar_, int Rows_, int Cols_ = Rows_, int StorageOrder_ = RowMajor>
class SkewSymmetricMatrixView :
    public SkewSymmetricMatrixBase<
      Scalar_, Rows_, Cols_, StorageOrder_, SkewSymmetricMatrixView<Scalar_, Rows_, Cols_, StorageOrder_>> {
   private:
    using Base = SkewSymmetricMatrixBase<
      Scalar_, Rows_, Cols_, StorageOrder_, SkewSymmetricMatrixView<Scalar_, Rows_, Cols_, StorageOrder_>>;
    static constexpr int StaticShape = Rows_ != Dynamic ? Rows_ : Cols_;
    static constexpr bool HasSupportedStaticStorage =
      StaticShape == Dynamic || static_cast<std::uint64_t>(StaticShape) * static_cast<std::uint64_t>(StaticShape) <=
                                  static_cast<std::uint64_t>(std::numeric_limits<int>::max());
   public:
    fdapde_static_assert((Rows_ == Dynamic || Rows_ > 0) && (Cols_ == Dynamic || Cols_ > 0), INVALID_MATRIX_DIMENSIONS);
    fdapde_static_assert(
      Rows_ == Dynamic || Cols_ == Dynamic || Rows_ == Cols_, THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
    fdapde_static_assert(StorageOrder_ == RowMajor, PACKED_COL_MAJOR_STRUCTURED_STORAGE_IS_NOT_SUPPORTED);
    fdapde_static_assert(HasSupportedStaticStorage, MATRIX_SIZE_EXCEEDS_SUPPORTED_RANGE);
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int StorageSize =
      StaticShape == Dynamic || !HasSupportedStaticStorage ? Dynamic : StaticShape * (StaticShape - 1) / 2;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = std::is_const_v<Scalar_>;
    using assignment_executor = internals::skew_symmetric_assignment_executor;

    /// @brief copies the binding to external packed strict-upper storage
    constexpr SkewSymmetricMatrixView(const SkewSymmetricMatrixView&) = default;
    /// @brief creates an empty dynamic skew-symmetric view without a buffer
    constexpr SkewSymmetricMatrixView()
        requires(Rows_ == Dynamic && Cols_ == Dynamic)
        : Base(), data_(nullptr) { }
    /// @brief rejects default construction when either dimension is fixed
    constexpr SkewSymmetricMatrixView()
        requires(Rows_ != Dynamic || Cols_ != Dynamic)
    = delete;
    /// @brief binds fixed dimensions to external strict-upper storage, permitting null only when no coefficients are
    /// stored
    template <typename Scalar__>
        requires(std::is_convertible_v<Scalar__*, Scalar_*>)
    constexpr explicit SkewSymmetricMatrixView(Scalar__* data) : Base(), data_(data) {
        fdapde_static_assert(Rows != Dynamic && Cols != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        fdapde_assert(
          !(StorageSize > 0 && data == nullptr), std::invalid_argument,
          "nonempty skew-symmetric view requires storage");
    }
    /// @brief binds external strict-upper storage using a single square dimension
    template <typename Scalar__>
        requires(std::is_convertible_v<Scalar__*, Scalar_*>)
    constexpr SkewSymmetricMatrixView(Scalar__* data, int size) : SkewSymmetricMatrixView(data, size, size) { }
    /// @brief validates square dimensions and binds sufficient external strict-upper storage
    template <typename Scalar__>
        requires(std::is_convertible_v<Scalar__*, Scalar_*>)
    constexpr SkewSymmetricMatrixView(Scalar__* data, int rows, int cols) : Base(rows, cols), data_(data) {
        fdapde_assert(
          !(this->storage_size() > 0 && data == nullptr), std::invalid_argument,
          "nonempty skew-symmetric view requires storage");
    }
    using Base::operator=;
    /// @brief copies a skew source snapshot into bound storage without rebinding the view
    constexpr SkewSymmetricMatrixView& operator=(const SkewSymmetricMatrixView& other) &
        requires(ReadOnly == 0)
    {
        static_cast<Base&>(*this).template operator= <SkewSymmetricMatrixView>(other);
        return *this;
    }
    /// @brief copies into a temporary skew view and returns its binding by value
    constexpr SkewSymmetricMatrixView operator=(const SkewSymmetricMatrixView& other) &&
      requires(ReadOnly == 0) {
          static_cast<Base&>(*this).template operator= <SkewSymmetricMatrixView>(other);
          return *this;
      }
      /// @brief returns the underlying storage pointer
      constexpr const std::remove_const_t<Scalar_>* data() const {
        return data_;
    }
    /// @brief returns the underlying storage pointer
    constexpr Scalar_* data()
        requires(!std::is_const_v<Scalar_>)
    {
        return data_;
    }
   private:
    Scalar_* data_;
};

/// @brief identifies skew symmetric matrix expressions after removing cv and reference qualifiers
template <typename XprType> struct is_skew_symmetric_matrix {
    using Type = std::remove_cvref_t<XprType>;
    static constexpr bool value = std::is_base_of_v<SkewSymmetricMatrixExpr<Type>, Type>;
};
template <typename XprType> static constexpr bool is_skew_symmetric_matrix_v = is_skew_symmetric_matrix<XprType>::value;

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_SKEW_H__
