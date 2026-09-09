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

#ifndef __FDAPDE_LINALG_ORTHOGONAL_H__
#define __FDAPDE_LINALG_ORTHOGONAL_H__

#include "header_check.h"

namespace fdapde {

// orthogonal matrix type system (the general orthogonal group O(n))

namespace internals {

/// @brief represents orthogonalize t
struct orthogonalize_t { };

/// @brief infers a square dimension from the coefficient count
constexpr int checked_orthogonal_square_shape(std::size_t size) {
    const int input_size = checked_matrix_data_size(size);
    fdapde_assert(!(input_size <= 0), std::invalid_argument, "orthogonal matrix input must be a positive square");
    int shape = 1;
    while (static_cast<std::uint64_t>(shape) * static_cast<std::uint64_t>(shape) <
           static_cast<std::uint64_t>(input_size)) {
        ++shape;
    }
    fdapde_assert(
      !(static_cast<std::uint64_t>(shape) * static_cast<std::uint64_t>(shape) !=
        static_cast<std::uint64_t>(input_size)),
      std::invalid_argument, "orthogonal matrix input must be a positive square");
    return shape;
}

/// @brief reports is orthogonal
template <typename XprType> constexpr bool is_orthogonal(const XprType& xpr) {
    if (xpr.rows() <= 0 || xpr.rows() != xpr.cols()) return false;
    using Scalar = std::remove_cv_t<typename std::remove_cvref_t<XprType>::Scalar>;
    const Scalar tolerance = Scalar(64) * std::numeric_limits<Scalar>::epsilon() * Scalar(xpr.rows());
    for (int i = 0; i < xpr.rows(); ++i) {
        for (int j = 0; j < xpr.rows(); ++j) {
            Scalar dot = 0;
            for (int k = 0; k < xpr.cols(); ++k) {
                dot += static_cast<Scalar>(xpr(i, k)) * static_cast<Scalar>(xpr(j, k));
            }
            const Scalar expected = i == j ? Scalar(1) : Scalar(0);
            if (!(std::abs(dot - expected) <= tolerance)) return false;
        }
    }
    return true;
}

/// @brief adapts an expression to orthogonal matrix operations
template <typename XprType> constexpr auto orthogonal_cast(XprType&& xpr);

}   // namespace internals

[[maybe_unused]] inline constexpr internals::orthogonalize_t orthogonalize {};

/// @brief represents orthogonal matrix expr
template <typename XprType_> struct OrthogonalMatrixExpr : public MatrixExpr<XprType_> {
    using XprType = std::remove_cvref_t<XprType_>;

    /// @brief returns the inverse matrix expression
    constexpr auto inverse() const& { return internals::orthogonal_cast(this->derived().transpose()); }
    /// @brief returns the inverse matrix expression
    constexpr auto inverse() const&&
        requires(XprType::NestAsRef == 0)
    {
        return internals::orthogonal_cast(this->derived().transpose());
    }
    /// @brief returns the inverse matrix expression
    constexpr void inverse() const&&
        requires(XprType::NestAsRef != 0)
    = delete;

    /// @brief solves the factored linear system for the supplied right-hand side
    template <typename RhsXprType> constexpr auto solve(const RhsXprType& rhs) const {
        using RhsType = std::remove_cvref_t<RhsXprType>;
        using Scalar = promote_type_t<typename XprType::Scalar, typename RhsType::Scalar>;
        return Matrix<Scalar, XprType::Cols, RhsType::Cols>(inverse() * rhs);
    }
};

/// @brief detects is orthogonal matrix
template <typename XprType> struct is_orthogonal_matrix {
    using Type = std::remove_cvref_t<XprType>;
    static constexpr bool value = std::is_base_of_v<OrthogonalMatrixExpr<Type>, Type>;
};
template <typename XprType> inline constexpr bool is_orthogonal_matrix_v = is_orthogonal_matrix<XprType>::value;

/// @brief represents orthogonal matrix base
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_, typename OrthogonalMatrixType_>
struct OrthogonalMatrixBase : public OrthogonalMatrixExpr<OrthogonalMatrixType_> {
   private:
    using Base = OrthogonalMatrixExpr<OrthogonalMatrixType_>;
    using Base::derived;
   public:
    using Scalar = std::remove_cv_t<Scalar_>;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;
    using assignment_executor = internals::generic_assignment_executor;

    /// @brief constructs orthogonal matrix base from the supplied state
    constexpr OrthogonalMatrixBase()
        requires(Rows_ == Dynamic && Cols_ == Dynamic)
        : rows_(0), cols_(0) { }
    /// @brief constructs orthogonal matrix base from the supplied state
    constexpr OrthogonalMatrixBase()
        requires(Rows_ != Dynamic || Cols_ != Dynamic)
    = delete;
    /// @brief constructs orthogonal matrix base from the supplied state
    constexpr OrthogonalMatrixBase(int rows, int cols) :
        rows_(Rows_ == Dynamic ? rows : Rows_), cols_(Cols_ == Dynamic ? cols : Cols_) {
        internals::validate_matrix_shape<Rows_, Cols_>(rows, cols);
        (void)internals::checked_matrix_size(rows_, cols_);
        fdapde_assert(
          !(rows_ <= 0 || rows_ != cols_), std::invalid_argument,
          "orthogonal matrices require positive square dimensions");
    }

    /// @brief accesses or evaluates the requested coefficient
    constexpr Scalar operator()(int i, int j) const {
        internals::validate_matrix_index(i, j, rows_, cols_);
        return static_cast<Scalar>(derived().data()(i, j));
    }
    /// @brief returns the row count
    constexpr int rows() const { return rows_; }
    /// @brief returns the column count
    constexpr int cols() const { return cols_; }
   protected:
    /// @brief orthogonalizes the matrix columns and reports success
    template <typename MatrixType> bool orthogonalize_(MatrixType& matrix) {
        Matrix<Scalar, Rows, Cols, StorageOrder> q;
        if constexpr (Rows == Dynamic || Cols == Dynamic) { q.resize(matrix.rows(), matrix.cols()); }
        const Scalar scale = matrix.norm();
        const Scalar epsilon = internals::scale_safe_sqrt(std::numeric_limits<Scalar>::epsilon());

        int rank = 0;
        for (int i = 0; i < matrix.cols(); ++i) {
            Vector<Scalar, Rows> vector(matrix.col(i));
            for (int pass = 0; pass < 2; ++pass) {
                for (int j = 0; j < rank; ++j) { vector -= q.col(j).dot(vector) * q.col(j); }
            }
            const Scalar vector_norm = vector.norm();
            const Scalar threshold =
              epsilon * fdapde::max(scale, fdapde::max(static_cast<Scalar>(matrix.col(i).norm()), Scalar(1)));
            if (!(vector_norm > threshold)) break;
            q.col(rank++) = vector / vector_norm;
        }
        if (rank != matrix.rows() || !internals::is_orthogonal(q)) return false;
        matrix = q;
        return true;
    }

    int rows_, cols_;
};

/// @brief stores an orthogonal matrix
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_ = RowMajor>
class OrthogonalMatrix :
    public OrthogonalMatrixBase<
      Scalar_, Rows_, Cols_, StorageOrder_, OrthogonalMatrix<Scalar_, Rows_, Cols_, StorageOrder_>> {
   private:
    using Base = OrthogonalMatrixBase<
      Scalar_, Rows_, Cols_, StorageOrder_, OrthogonalMatrix<Scalar_, Rows_, Cols_, StorageOrder_>>;
    using StorageType = Matrix<Scalar_, Rows_, Cols_, StorageOrder_>;
   public:
    fdapde_static_assert((Rows_ == Dynamic || Rows_ > 0) && (Cols_ == Dynamic || Cols_ > 0), INVALID_MATRIX_DIMENSIONS);
    fdapde_static_assert(
      Rows_ == Dynamic || Cols_ == Dynamic || Rows_ == Cols_, THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
    fdapde_static_assert(std::is_floating_point_v<Scalar_>, ORTHOGONAL_MATRICES_REQUIRE_FLOATING_POINT_SCALARS);
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int ReadOnly = 1;
    static constexpr int NestAsRef = 1;
    using assignment_executor = typename StorageType::assignment_executor;

    /// @brief constructs orthogonal matrix from the supplied state
    constexpr OrthogonalMatrix() = delete;
    /// @brief constructs orthogonal matrix from the supplied state
    constexpr OrthogonalMatrix(int, int) = delete;
    /// @brief constructs orthogonal matrix from the supplied state
    constexpr OrthogonalMatrix(const OrthogonalMatrix& other) : Base(other.rows(), other.cols()), data_(other.data_) { }
    /// @brief assigns the supplied coefficients
    constexpr OrthogonalMatrix& operator=(const OrthogonalMatrix& other) & {
        if (this == std::addressof(other)) return *this;
        data_ = other.data_;
        this->rows_ = other.rows();
        this->cols_ = other.cols();
        return *this;
    }
    /// @brief assigns the supplied coefficients
    constexpr void operator=(const OrthogonalMatrix&) && = delete;

    /// @brief constructs orthogonal matrix from the supplied state
    template <typename RhsXprType>
    constexpr OrthogonalMatrix(const MatrixExpr<RhsXprType>& rhs, internals::unchecked_t) :
        Base(rhs.rows(), rhs.cols()), data_(rhs) { }
    /// @brief constructs orthogonal matrix from the supplied state
    template <typename RhsXprType>
    constexpr OrthogonalMatrix(const OrthogonalMatrixExpr<RhsXprType>& rhs) :
        OrthogonalMatrix(rhs.derived(), unchecked) { }

    /// @brief constructs orthogonal matrix from the supplied state
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr OrthogonalMatrix(const std::vector<Scalar__>& data, internals::orthogonalize_t) :
        Base(
          internals::checked_orthogonal_square_shape(data.size()),
          internals::checked_orthogonal_square_shape(data.size())),
        data_() {
        load_data_(data);
        fdapde_assert(
          !(!this->orthogonalize_(data_)), std::invalid_argument,
          "orthogonalization requires a finite full-rank square matrix");
    }
    /// @brief constructs orthogonal matrix from the supplied state
    template <typename Scalar__, std::size_t Size>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr OrthogonalMatrix(const Scalar__ (&data)[Size], internals::orthogonalize_t) :
        Base(internals::checked_orthogonal_square_shape(Size), internals::checked_orthogonal_square_shape(Size)),
        data_() {
        fdapde_static_assert(
          Rows_ != Dynamic && Cols_ != Dynamic &&
            static_cast<std::size_t>(Rows_) * static_cast<std::size_t>(Cols_) == Size,
          THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        load_data_(data);
        fdapde_assert(
          !(!this->orthogonalize_(data_)), std::invalid_argument,
          "orthogonalization requires a finite full-rank square matrix");
    }

    /// @brief constructs orthogonal matrix from the supplied state
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr OrthogonalMatrix(const std::vector<Scalar__>& data, internals::checked_t) :
        Base(
          internals::checked_orthogonal_square_shape(data.size()),
          internals::checked_orthogonal_square_shape(data.size())),
        data_() {
        load_data_(data);
        fdapde_assert(
          !(!internals::is_orthogonal(data_)), std::invalid_argument, "checked orthogonal input is not orthogonal");
    }
    /// @brief constructs orthogonal matrix from the supplied state
    template <typename Scalar__, std::size_t Size>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr OrthogonalMatrix(const Scalar__ (&data)[Size], internals::checked_t) :
        Base(internals::checked_orthogonal_square_shape(Size), internals::checked_orthogonal_square_shape(Size)),
        data_() {
        fdapde_static_assert(
          Rows_ != Dynamic && Cols_ != Dynamic &&
            static_cast<std::size_t>(Rows_) * static_cast<std::size_t>(Cols_) == Size,
          THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        load_data_(data);
        fdapde_assert(
          !(!internals::is_orthogonal(data_)), std::invalid_argument, "checked orthogonal input is not orthogonal");
    }

    /// @brief constructs orthogonal matrix from the supplied state
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr OrthogonalMatrix(const std::vector<Scalar__>& data, internals::unchecked_t) :
        Base(
          internals::checked_orthogonal_square_shape(data.size()),
          internals::checked_orthogonal_square_shape(data.size())),
        data_() {
        load_data_(data);
    }
    /// @brief constructs orthogonal matrix from the supplied state
    template <typename Scalar__, std::size_t Size>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr OrthogonalMatrix(const Scalar__ (&data)[Size], internals::unchecked_t) :
        Base(internals::checked_orthogonal_square_shape(Size), internals::checked_orthogonal_square_shape(Size)),
        data_() {
        fdapde_static_assert(
          Rows_ != Dynamic && Cols_ != Dynamic &&
            static_cast<std::size_t>(Rows_) * static_cast<std::size_t>(Cols_) == Size,
          THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        load_data_(data);
    }

    /// @brief returns the underlying storage pointer
    constexpr const StorageType& data() const { return data_; }
   private:
    /// @brief copies the supplied coefficients into internal storage
    template <typename Data> constexpr void load_data_(const Data& data) {
        const int input_size = internals::checked_matrix_data_size(std::size(data));
        const int expected_size = internals::checked_matrix_size(this->rows_, this->cols_);
        fdapde_assert(
          !(input_size != expected_size), std::invalid_argument, "orthogonal input does not match its matrix shape");
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) { data_.resize(this->rows_, this->cols_); }
        for (int i = 0; i < this->rows_; ++i) {
            for (int j = 0; j < this->cols_; ++j) {
                data_(i, j) = static_cast<Scalar>(data[static_cast<std::size_t>(i * this->cols_ + j)]);
            }
        }
    }

    StorageType data_;
};

namespace internals {

/// @brief represents orthogonal wrapper
template <typename OrthogonalXprType_>
class orthogonal_wrapper : public OrthogonalMatrixExpr<orthogonal_wrapper<OrthogonalXprType_>> {
   private:
    using XprType = std::remove_cvref_t<OrthogonalXprType_>;
    using XprTypeNested = internals::ref_select_t<const XprType>;
   public:
    using Scalar = std::remove_cv_t<typename XprType::Scalar>;
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    static constexpr int StorageOrder = XprType::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    /// @brief constructs orthogonal wrapper from the supplied state
    constexpr orthogonal_wrapper(const orthogonal_wrapper&) = default;
    /// @brief constructs orthogonal wrapper from the supplied state
    template <typename XprType__>
        requires(
          !std::same_as<std::remove_cvref_t<XprType__>, orthogonal_wrapper> &&
          internals::safely_nestable<XprTypeNested, XprType__>)
    constexpr explicit orthogonal_wrapper(XprType__&& xpr) : xpr_(std::forward<XprType__>(xpr)) { }
    /// @brief accesses or evaluates the requested coefficient
    constexpr Scalar operator()(int i, int j) const {
        internals::validate_matrix_index(i, j, rows(), cols());
        return static_cast<Scalar>(xpr_(i, j));
    }
    /// @brief returns the row count
    constexpr int rows() const { return xpr_.rows(); }
    /// @brief returns the column count
    constexpr int cols() const { return xpr_.cols(); }
   private:
    XprTypeNested xpr_;
};

/// @brief adapts an expression to orthogonal matrix operations
template <typename XprType> constexpr auto orthogonal_cast(XprType&& xpr) {
    return orthogonal_wrapper<XprType>(std::forward<XprType>(xpr));
}

}   // namespace internals

/// @brief implements the operator* expression operation
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator*(const OrthogonalMatrixExpr<LhsXprType>& lhs, const OrthogonalMatrixExpr<RhsXprType>& rhs) {
    return internals::orthogonal_cast(
      MatrixMultiplicationOp<LhsXprType, RhsXprType, internals::generic_matrix_product_executor> {
        lhs.derived(), rhs.derived()});
}

/// @brief implements the operator* expression operation
template <internals::matrix_expression Lhs, internals::matrix_expression Rhs>
    requires(is_orthogonal_matrix_v<Lhs> && is_orthogonal_matrix_v<Rhs> &&
             (internals::is_owning_rvalue_expression_v<Lhs &&> || internals::is_owning_rvalue_expression_v<Rhs &&>))
constexpr void operator*(Lhs&&, Rhs&&) = delete;

/// @brief represents orthogonal matrix view
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_ = RowMajor>
class OrthogonalMatrixView :
    public OrthogonalMatrixBase<
      Scalar_, Rows_, Cols_, StorageOrder_, OrthogonalMatrixView<Scalar_, Rows_, Cols_, StorageOrder_>> {
   private:
    using Base = OrthogonalMatrixBase<
      Scalar_, Rows_, Cols_, StorageOrder_, OrthogonalMatrixView<Scalar_, Rows_, Cols_, StorageOrder_>>;
    using StorageType = MatrixView<Scalar_, Rows_, Cols_, StorageOrder_>;
   public:
    fdapde_static_assert((Rows_ == Dynamic || Rows_ > 0) && (Cols_ == Dynamic || Cols_ > 0), INVALID_MATRIX_DIMENSIONS);
    fdapde_static_assert(
      Rows_ == Dynamic || Cols_ == Dynamic || Rows_ == Cols_, THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
    fdapde_static_assert(
      std::is_floating_point_v<std::remove_const_t<Scalar_>>, ORTHOGONAL_MATRICES_REQUIRE_FLOATING_POINT_SCALARS);
    using Scalar = std::remove_const_t<Scalar_>;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;
    using assignment_executor = typename StorageType::assignment_executor;

    /// @brief constructs orthogonal matrix view from the supplied state
    constexpr OrthogonalMatrixView(const OrthogonalMatrixView&) = default;
    /// @brief constructs orthogonal matrix view from the supplied state
    constexpr OrthogonalMatrixView()
        requires(Rows_ == Dynamic && Cols_ == Dynamic)
        : Base(), data_() { }
    /// @brief constructs orthogonal matrix view from the supplied state
    constexpr OrthogonalMatrixView()
        requires(Rows_ != Dynamic || Cols_ != Dynamic)
    = delete;

    /// @brief constructs orthogonal matrix view from the supplied state
    template <typename Scalar__>
        requires(std::is_convertible_v<Scalar__*, Scalar_*>)
    constexpr OrthogonalMatrixView(Scalar__* data, internals::checked_t) : Base(Rows_, Cols_), data_(data) {
        fdapde_static_assert(Rows_ != Dynamic && Cols_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_VIEWS_ONLY);
        validate_storage_(data);
        fdapde_assert(
          !(!internals::is_orthogonal(data_)), std::invalid_argument, "checked orthogonal view is not orthogonal");
    }
    /// @brief constructs orthogonal matrix view from the supplied state
    template <typename Scalar__>
        requires(std::is_convertible_v<Scalar__*, Scalar_*>)
    constexpr OrthogonalMatrixView(Scalar__* data, int rows, int cols, internals::checked_t) :
        Base(rows, cols), data_(data, rows, cols) {
        validate_storage_(data);
        fdapde_assert(
          !(!internals::is_orthogonal(data_)), std::invalid_argument, "checked orthogonal view is not orthogonal");
    }
    /// @brief constructs orthogonal matrix view from the supplied state
    template <typename Scalar__>
        requires(std::is_convertible_v<Scalar__*, Scalar_*>)
    constexpr OrthogonalMatrixView(Scalar__* data, internals::unchecked_t) : Base(Rows_, Cols_), data_(data) {
        fdapde_static_assert(Rows_ != Dynamic && Cols_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_VIEWS_ONLY);
        validate_storage_(data);
    }
    /// @brief constructs orthogonal matrix view from the supplied state
    template <typename Scalar__>
        requires(std::is_convertible_v<Scalar__*, Scalar_*>)
    constexpr OrthogonalMatrixView(Scalar__* data, int rows, int cols, internals::unchecked_t) :
        Base(rows, cols), data_(data, rows, cols) {
        validate_storage_(data);
    }

    /// @brief assigns the supplied coefficients
    constexpr OrthogonalMatrixView& operator=(const OrthogonalMatrixView& other) &
        requires(!std::is_const_v<Scalar_>)
    {
        data_ = other.data_;
        return *this;
    }
    /// @brief assigns the supplied coefficients
    constexpr OrthogonalMatrixView operator=(const OrthogonalMatrixView& other) &&
      requires(!std::is_const_v<Scalar_>) {
          data_ = other.data_;
          return *this;
      }
      /// @brief assigns the supplied coefficients
      constexpr OrthogonalMatrixView& operator=(const OrthogonalMatrixView&) &
          requires(std::is_const_v<Scalar_>)
      = delete;
    /// @brief assigns the supplied coefficients
    constexpr OrthogonalMatrixView operator=(const OrthogonalMatrixView&) &&
      requires(std::is_const_v<Scalar_>) = delete;

    /// @brief assigns the supplied coefficients
    template <typename RhsXprType>
    constexpr OrthogonalMatrixView& operator=(const OrthogonalMatrixExpr<RhsXprType>& rhs) &
        requires(!std::is_const_v<Scalar_>)
    {
        data_ = rhs.derived();
        return *this;
    }
    /// @brief assigns the supplied coefficients
    template <typename RhsXprType>
      constexpr OrthogonalMatrixView operator=(const OrthogonalMatrixExpr<RhsXprType>& rhs) &&
      requires(!std::is_const_v<Scalar_>) {
          data_ = rhs.derived();
          return *this;
      }

      /// @brief returns the underlying storage pointer
      constexpr const StorageType& data() const {
        return data_;
    }
   private:
    /// @brief checks storage
    template <typename Scalar__> constexpr void validate_storage_(Scalar__* data) const {
        fdapde_assert(!(data == nullptr), std::invalid_argument, "nonempty orthogonal view requires storage");
    }

    StorageType data_;
};

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_ORTHOGONAL_H__
