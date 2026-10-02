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

#ifndef __FDAPDE_LINALG_SYMMETRIC_H__
#define __FDAPDE_LINALG_SYMMETRIC_H__

#include "header_check.h"

namespace fdapde {

// symmetric matrix type system
/// @brief provides symmetric matrix expressions and eigendecomposition access
template <typename XprType> struct SymmetricMatrixExpr;
/// @brief views packed lower-triangular storage as a symmetric matrix
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_> class SymmetricMatrixView;

namespace internals {

/// @brief identifies a view whose scalar type permits writes
template <typename Scalar, int Rows, int Cols, int StorageOrder>
struct is_mutable_matrix_view<SymmetricMatrixView<Scalar, Rows, Cols, StorageOrder>> :
    std::bool_constant<!std::is_const_v<Scalar>> { };

}   // namespace internals

// forward decls
/// @brief computes the eigendecomposition of a symmetric matrix
template <typename XprType> class EVD;

namespace internals {

// class wrapping a generic expression to the expression of a symmetric matrix. internal usage only
/// @brief mirrors one selected triangle of a square expression
template <int ViewMode_, typename SymmetricXprType_>
struct symmetric_wrapper : public SymmetricMatrixExpr<symmetric_wrapper<ViewMode_, SymmetricXprType_>> {
   private:
    fdapde_static_assert(ViewMode_ == Lower || ViewMode_ == Upper, VIEW_MODE_MUST_BE_EITHER_LOWER_OR_UPPER);
    using Base = SymmetricMatrixExpr<symmetric_wrapper<ViewMode_, SymmetricXprType_>>;
    using XprType = std::remove_reference_t<SymmetricXprType_>;
    using XprTypeClean = std::remove_cv_t<XprType>;
    using XprTypeNested = internals::ref_select_t<SymmetricXprType_>;
    static constexpr int ViewMode = ViewMode_;
   public:
    using Scalar = typename XprTypeClean::Scalar;
    static constexpr int Rows = XprTypeClean::Rows;
    static constexpr int Cols = XprTypeClean::Cols;
    static constexpr int StorageOrder = XprTypeClean::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    /// @brief copies the symmetric adaptor while retaining its nested expression
    constexpr symmetric_wrapper(const symmetric_wrapper&) = default;
    /// @brief borrows one triangle of a nonempty square expression and mirrors it across the diagonal
    template <typename XprType__>
        requires(!std::same_as<std::remove_cvref_t<XprType__>, symmetric_wrapper> &&
                 internals::safely_nestable<XprTypeNested, XprType__>)
    constexpr symmetric_wrapper(XprType__&& xpr) :
        Base(), xpr_(std::forward<XprType__>(xpr)), size_(xpr_.rows() == xpr_.cols() ? xpr_.rows() : 0) {
        fdapde_static_assert(
          Rows == Dynamic || Cols == Dynamic || Rows == Cols, THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
        fdapde_assert(
          !(xpr_.rows() <= 0 || xpr_.rows() != xpr_.cols()), std::invalid_argument,
          "symmetric view requires positive square dimensions");
    }
    /// @brief reflects the selected source triangle across the main diagonal
    constexpr Scalar operator()(int i, int j) const {
        internals::validate_matrix_index(i, j, size_, size_);
        if constexpr (ViewMode == Upper) { return i > j ? xpr_(j, i) : xpr_(i, j); }
        if constexpr (ViewMode == Lower) { return i < j ? xpr_(j, i) : xpr_(i, j); }
    }
    /// @brief returns the row count
    constexpr int rows() const { return size_; }
    /// @brief returns the column count
    constexpr int cols() const { return size_; }
    /// @brief returns the stored representation
    constexpr const XprTypeNested& rep() const { return xpr_; }
   private:
    XprTypeNested xpr_;
    int size_;
};

// helper cast function
/// @brief adapts an expression to symmetric matrix operations
template <int ViewMode_, typename XprType_> constexpr auto symmetric_cast(XprType_&& xpr) {
    return symmetric_wrapper<ViewMode_, XprType_>(std::forward<XprType_>(xpr));
}

}   // namespace internals

/// @brief provides symmetric matrix expressions and eigendecomposition access
template <typename XprType_> struct SymmetricMatrixExpr : public MatrixExpr<XprType_> {
    using XprType = std::decay_t<XprType_>;
    using Base = MatrixExpr<XprType_>;
    using Base::derived;
    // inherit assignment from base
    using Base::operator=;

    /// @brief computes the symmetric eigendecomposition
    auto evd() const { return EVD<XprType>(derived()); }

    // internal triangular matrix representation
    /// @brief returns the stored representation
    constexpr decltype(auto) rep() const { return derived().rep(); }
    /// @brief returns the stored representation
    constexpr decltype(auto) rep() { return derived().rep(); }
};

// base class for symmetric matrices
/// @brief maps symmetric coordinates to a shared packed lower triangle
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_, typename SymmetricMatrixType>
class SymmetricMatrixBase : public SymmetricMatrixExpr<SymmetricMatrixType> {
   private:
    using Base = SymmetricMatrixExpr<SymmetricMatrixType>;
    using Base::derived;
   public:
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int ReadOnly = std::is_const_v<Scalar_>;
    using Base::operator=;

    /// @brief reads or updates the shared storage entry for a pair of mirrored coordinates
    template <typename Scalar__>
        requires(std::is_same_v<std::remove_cv_t<Scalar>, std::remove_cv_t<Scalar__>>)
    struct symmetric_proxy {
        using Scalar = Scalar__;

        /// @brief maps either mirrored coordinate to the same packed lower-triangular coefficient
        constexpr symmetric_proxy(Scalar__* data, int i, int j, int size) :
            data_(data), index_(compute_linear_index_(i < j ? j : i, i < j ? i : j, size)) { }
        /// @brief writes the packed coefficient shared by the two reflected coordinates
        template <typename T>
            requires(std::is_convertible_v<T, Scalar> && !std::is_const_v<Scalar__>)
        constexpr symmetric_proxy& operator=(T value) {
            data_[index_] = value;
            return *this;
        }
        /// @brief reads the packed coefficient shared by reflected coordinates
        constexpr operator Scalar() const { return data_[index_]; }
       private:
        /// @brief maps a matrix coordinate to packed storage
        constexpr int compute_linear_index_(int i, int j, [[maybe_unused]] int size) const {
            if constexpr (StorageOrder == RowMajor) { return i * (i + 1) / 2 + j; }
            if constexpr (StorageOrder == ColMajor) { return j * (2 * size - j + 1) / 2 + (i - j); }
        }
        Scalar* data_;
        int index_;
    };
    using reference = symmetric_proxy<Scalar>;
    using const_reference = symmetric_proxy<const Scalar>;

    /// @brief initializes the shared interface for packed symmetric coefficient access
    constexpr SymmetricMatrixBase() = default;
    // access
    /// @brief returns a read-only proxy to the packed coefficient shared by reflected coordinates
    constexpr auto operator()(int i, int j) const {
        internals::validate_matrix_index(i, j, derived().rows(), derived().cols());
        return const_reference(derived().data(), i, j, derived().rows());
    }
    /// @brief returns a writable proxy to the packed coefficient shared by reflected coordinates
    constexpr auto operator()(int i, int j)
        requires(ReadOnly == 0)
    {
        internals::validate_matrix_index(i, j, derived().rows(), derived().cols());
        return reference(derived().data(), i, j, derived().rows());
    }
};

// symmetric matrices vector-space structure (additive group)
/// @brief adds equally shaped operands while preserving symmetry
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator+(const SymmetricMatrixExpr<LhsXprType>& lhs, const SymmetricMatrixExpr<RhsXprType>& rhs) {
    return internals::symmetric_cast<Lower>(lhs.rep() + rhs.rep());
}
/// @brief subtracts equally shaped operands while preserving symmetry
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator-(const SymmetricMatrixExpr<LhsXprType>& lhs, const SymmetricMatrixExpr<RhsXprType>& rhs) {
    return internals::symmetric_cast<Lower>(lhs.rep() - rhs.rep());
}
/// @brief scales by the right scalar while preserving symmetry
template <typename XprType, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator*(const SymmetricMatrixExpr<XprType>& lhs, ScalarType rhs) {
    return internals::symmetric_cast<Lower>(lhs.rep() * rhs);
}
/// @brief scales by the left scalar while preserving symmetry
template <typename XprType, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator*(ScalarType lhs, const SymmetricMatrixExpr<XprType>& rhs) {
    return internals::symmetric_cast<Lower>(lhs * rhs.rep());
}
/// @brief divides each coefficient by the scalar while preserving symmetry
template <typename XprType, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator/(const SymmetricMatrixExpr<XprType>& lhs, ScalarType rhs) {
    return internals::symmetric_cast<Lower>(lhs.rep() / rhs);
}
// any other operation doesn't preserve symmetry. A raw MatrixExpr is returned

// owning storage symmetric matrix
/// @brief owns a symmetric matrix
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_ = RowMajor>
class SymmetricMatrix :
    public SymmetricMatrixBase<
      Scalar_, Rows_, Cols_, StorageOrder_, SymmetricMatrix<Scalar_, Rows_, Cols_, StorageOrder_>> {
   private:
    using Base =
      SymmetricMatrixBase<Scalar_, Rows_, Cols_, StorageOrder_, SymmetricMatrix<Scalar_, Rows_, Cols_, StorageOrder_>>;
    using StorageType = TriangularMatrix<Scalar_, Rows_, Cols_, Lower, StorageOrder_>;
   public:
    fdapde_static_assert((Rows_ == Dynamic || Rows_ > 0) && (Cols_ == Dynamic || Cols_ > 0), INVALID_MATRIX_DIMENSIONS);
    fdapde_static_assert(
      Rows_ == Dynamic || Cols_ == Dynamic || Rows_ == Cols_, THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
    fdapde_static_assert(StorageOrder_ == RowMajor, PACKED_COL_MAJOR_STRUCTURED_STORAGE_IS_NOT_SUPPORTED);
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int NestAsRef = 1;
    static constexpr int ViewMode = StorageType::ViewMode;
    static constexpr int StorageOrder = StorageType::StorageOrder;
    static constexpr int ReadOnly = std::is_const_v<Scalar_>;
    using assignment_executor = typename StorageType::assignment_executor;

    /// @brief value-initializes fixed packed storage and leaves dynamic storage empty
    constexpr SymmetricMatrix() : Base(), data_() { }
    // copy semantic
    /// @brief copies the packed lower triangle into independent storage
    constexpr SymmetricMatrix(const SymmetricMatrix& rhs) : Base(), data_(rhs.rep()) { }
    /// @brief copies the source triangle and shape into independent packed storage
    constexpr SymmetricMatrix& operator=(const SymmetricMatrix& rhs) & {
        data_ = rhs.rep();
        return *this;
    }
    /// @brief rejects assignment to a temporary symmetric owner
    constexpr void operator=(const SymmetricMatrix&) && = delete;
    using Base::operator=;

    /// @brief allocates packed storage for the requested dynamic dimensions
    constexpr explicit SymmetricMatrix(int rows, int cols) : Base(), data_(rows, cols) {
        fdapde_static_assert(Rows == Dynamic || Cols == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_MATRICES_ONLY);
    }
    /// @brief evaluates the lower triangle of a symmetric expression into owned storage
    template <typename RhsXprType_>
    constexpr SymmetricMatrix(const SymmetricMatrixExpr<RhsXprType_>& rhs) : Base(), data_(rhs) { }
    /// @brief evaluates a symmetric expression into this owner's packed triangle
    template <typename RhsXprType_>
    constexpr SymmetricMatrix& operator=(const SymmetricMatrixExpr<RhsXprType_>& rhs) & {
        data_ = rhs;
        return *this;
    }
    /// @brief copies packed lower-triangular coefficients and infers dynamic dimensions from their count
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr explicit SymmetricMatrix(const std::vector<Scalar__>& data) : Base(), data_(data) { }
    /// @brief copies a C array of packed lower-triangular coefficients into fixed storage
    template <typename Scalar__, std::size_t Size>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr explicit SymmetricMatrix(const Scalar__ (&data)[Size]) : Base(), data_(data) {
        fdapde_static_assert(Rows_ != Dynamic && Cols_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
    }
    // modifiers
    /// @brief resizes the owned storage to the requested dimensions
    void resize(int rows, int cols) {
        fdapde_static_assert(Rows == Dynamic || Cols == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_MATRICES_ONLY);
        data_.resize(rows, cols);
    }
    // observers
    /// @brief returns the row count
    constexpr int rows() const { return data_.rows(); }
    /// @brief returns the column count
    constexpr int cols() const { return data_.cols(); }
    /// @brief returns the stored representation
    constexpr const StorageType& rep() const { return data_; }
    /// @brief returns the stored representation
    constexpr StorageType& rep() { return data_; }
    // data pointers
    /// @brief returns the underlying storage pointer
    constexpr const Scalar* data() const { return data_.data(); }
    /// @brief returns the underlying storage pointer
    constexpr Scalar* data() { return data_.data(); }
   private:
    StorageType data_;
};

// symmetric view of an existing block of data
/// @brief views packed lower-triangular storage as a symmetric matrix
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_ = RowMajor>
class SymmetricMatrixView :
    public SymmetricMatrixBase<
      Scalar_, Rows_, Cols_, StorageOrder_, SymmetricMatrixView<Scalar_, Rows_, Cols_, StorageOrder_>> {
    using Base = SymmetricMatrixBase<
      Scalar_, Rows_, Cols_, StorageOrder_, SymmetricMatrixView<Scalar_, Rows_, Cols_, StorageOrder_>>;
    using StorageType = TriangularMatrixView<Scalar_, Rows_, Cols_, Lower, StorageOrder_>;
   public:
    fdapde_static_assert((Rows_ == Dynamic || Rows_ > 0) && (Cols_ == Dynamic || Cols_ > 0), INVALID_MATRIX_DIMENSIONS);
    fdapde_static_assert(
      Rows_ == Dynamic || Cols_ == Dynamic || Rows_ == Cols_, THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
    fdapde_static_assert(StorageOrder_ == RowMajor, PACKED_COL_MAJOR_STRUCTURED_STORAGE_IS_NOT_SUPPORTED);
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int NestAsRef = 0;
    static constexpr int ViewMode = StorageType::ViewMode;
    static constexpr int StorageOrder = StorageType::StorageOrder;
    static constexpr int ReadOnly = std::is_const_v<Scalar_>;
    using assignment_executor = typename StorageType::assignment_executor;

    // constructors
    /// @brief copies the packed-storage binding without copying coefficients
    constexpr SymmetricMatrixView(const SymmetricMatrixView&) = default;
    /// @brief creates an empty dynamic symmetric view without external storage
    constexpr SymmetricMatrixView()
        requires(Rows_ == Dynamic && Cols_ == Dynamic)
        : Base(), data_() { }
    /// @brief rejects default construction when either dimension is fixed
    constexpr SymmetricMatrixView()
        requires(Rows_ != Dynamic || Cols_ != Dynamic)
    = delete;
    /// @brief binds fixed dimensions to external packed lower-triangular storage
    template <typename Scalar__>
        requires(std::is_convertible_v<Scalar__*, Scalar_*>)
    constexpr explicit SymmetricMatrixView(Scalar__* data) : Base(), data_(data) {
        fdapde_static_assert(Rows != Dynamic && Cols != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
    }
    /// @brief binds explicit dimensions to external packed lower-triangular storage
    template <typename Scalar__>
        requires(std::is_convertible_v<Scalar__*, Scalar_*>)
    constexpr SymmetricMatrixView(Scalar__* data, int rows, int cols) : Base(), data_(data, rows, cols) { }
    using Base::operator=;
    /// @brief copies a symmetric source snapshot into bound storage without rebinding the view
    constexpr SymmetricMatrixView& operator=(const SymmetricMatrixView& other) &
        requires(ReadOnly == 0)
    {
        static_cast<Base&>(*this).template operator= <SymmetricMatrixView>(other);
        return *this;
    }
    /// @brief copies into a temporary symmetric view and returns its binding by value
    constexpr SymmetricMatrixView operator=(const SymmetricMatrixView& other) &&
      requires(ReadOnly == 0) {
          static_cast<Base&>(*this).template operator= <SymmetricMatrixView>(other);
          return *this;
      }
      // observers
      /// @brief returns the row count
      constexpr int rows() const {
        return data_.rows();
    }
    /// @brief returns the column count
    constexpr int cols() const { return data_.cols(); }
    /// @brief returns the stored representation
    constexpr const StorageType& rep() const { return data_; }
    /// @brief returns the stored representation
    constexpr StorageType& rep() { return data_; }
    // data pointers
    /// @brief returns the underlying storage pointer
    constexpr const std::remove_const_t<Scalar_>* data() const { return data_.data(); }
    /// @brief returns the underlying storage pointer
    constexpr Scalar_* data()
        requires(!std::is_const_v<Scalar_>)
    {
        return data_.data();
    }
   private:
    StorageType data_;
};

// detection trait
/// @brief identifies symmetric matrix expressions after removing cv and reference qualifiers
template <typename XprType> struct is_symmetric_matrix {
    using Type = std::remove_cvref_t<XprType>;
    static constexpr bool value = std::is_base_of_v<SymmetricMatrixExpr<Type>, Type>;
};
template <typename XprType> static constexpr bool is_symmetric_matrix_v = is_symmetric_matrix<XprType>::value;

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_SYMMETRIC_H__
