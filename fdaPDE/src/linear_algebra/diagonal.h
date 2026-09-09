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
/// @brief represents diagonal matrix expr
template <typename XprType> struct DiagonalMatrixExpr;
/// @brief represents diagonal matrix view
template <typename Scalar_, int Rows_> class DiagonalMatrixView;

namespace internals {

// two-level CRTP constraints form before DiagonalMatrixView publishes ReadOnly
/// @brief detects is mutable matrix view
template <typename Scalar, int Rows>
struct is_mutable_matrix_view<DiagonalMatrixView<Scalar, Rows>> : std::bool_constant<!std::is_const_v<Scalar>> { };

}   // namespace internals

namespace internals {

/// @brief represents diagonal assignment executor
struct diagonal_assignment_executor {
    /// @brief executes the coefficient operation over the supplied expressions
    template <typename DstXprType, typename SrcXprType, typename AssignmentOp>
    static constexpr void run(DstXprType& dst, const SrcXprType& src, AssignmentOp&& op) {
        fdapde_static_assert(DstXprType::ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION);
        if constexpr (!std::is_arithmetic_v<SrcXprType>) {
            fdapde_static_assert(
              internals::is_dynamic_sized_v<DstXprType> || internals::is_dynamic_sized_v<SrcXprType> ||
                DstXprType::Rows == SrcXprType::Rows,   // diagonal expressions are vector-shaped
              INVALID_ASSIGNMENT__NOT_MATCHING_LHS_AND_RHS_STATIC_SIZES);
            if constexpr (internals::is_dynamic_sized_v<DstXprType> || internals::is_dynamic_sized_v<SrcXprType>) {
                fdapde_assert(
                  !(dst.rows() != src.rows() || dst.cols() != src.cols()), std::invalid_argument,
                  "diagonal assignment requires matching dimensions");
            }
        }
        const int size = dst.rows();
        auto fetch = [](const SrcXprType& src, [[maybe_unused]] int i) -> decltype(auto) {
            if constexpr (std::is_arithmetic_v<SrcXprType>) {
                return src;
            } else if constexpr (internals::is_vector_shaped_v<SrcXprType>) {
                return src[i];
            } else {
                return src(i, i);
            }
        };
        for (int i = 0; i < size; ++i) { op(dst[i], fetch(src, i)); }
        return;
    }
};

}   // namespace internals

// expression of the diagonal of a matrix (this node acts as a vector expression)
/// @brief represents diagonal
template <typename XprType_> struct Diagonal : public MatrixExpr<Diagonal<XprType_>> {
   private:
    using Base = MatrixExpr<Diagonal<XprType_>>;
    using XprType = std::remove_reference_t<XprType_>;
    using XprTypeClean = std::remove_cv_t<XprType>;
    using XprTypeNested = internals::ref_select_t<XprType_>;
   public:
    using Scalar = typename XprTypeClean::Scalar;
    static constexpr int Rows =
      XprTypeClean::Rows == Dynamic || XprTypeClean::Cols == Dynamic ? Dynamic : XprTypeClean::Rows;
    static constexpr int Cols = 1;
    static constexpr int StorageOrder = XprTypeClean::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = std::is_const_v<XprType> || XprTypeClean::ReadOnly;
    using assignment_executor = internals::diagonal_assignment_executor;

    // constructor
    /// @brief constructs diagonal from the supplied state
    constexpr Diagonal(const Diagonal&) = default;
    /// @brief constructs diagonal from the supplied state
    template <typename XprType__>
        requires(!std::same_as<std::remove_cvref_t<XprType__>, Diagonal> &&
                 internals::safely_nestable<XprTypeNested, XprType__>)
    constexpr explicit Diagonal(XprType__&& xpr) : xpr_(std::forward<XprType__>(xpr)), size_(xpr_.rows()) {
        fdapde_static_assert(
          internals::is_dynamic_sized_v<XprTypeClean> || XprTypeClean::Rows == XprTypeClean::Cols,
          DIAGONAL_BLOCKS_ARE_FOR_SQUARE_MATRICES_ONLY);
        fdapde_assert(!(xpr_.rows() != xpr_.cols()), std::invalid_argument, "diagonal requires a square matrix");
    }
    // copy-semantic
    /// @brief assigns the supplied coefficients
    constexpr Diagonal& operator=(const Diagonal& other) &
        requires(ReadOnly == 0)
    {
        static_cast<Base&>(*this).template operator= <Diagonal>(other);
        return *this;
    }
    /// @brief assigns the supplied coefficients
    constexpr Diagonal operator=(const Diagonal& other) &&
      requires(ReadOnly == 0) {
          static_cast<Base&>(*this).template operator= <Diagonal>(other);
          return *this;
      }
      // inherit assignment from base
      using Base::operator=;
    // const access
    /// @brief accesses or evaluates the requested coefficient
    constexpr Scalar operator()(int i, int j) const {
        fdapde_assert(
          !(i < 0 || i >= rows() || j < 0 || j >= cols()), std::out_of_range, "diagonal index out of range");
        return xpr_(i, i);
    }
    /// @brief accesses the requested vector coefficient
    constexpr Scalar operator[](int i) const {
        fdapde_assert(!(i < 0 || i >= rows()), std::out_of_range, "diagonal index out of range");
        return xpr_(i, i);
    }
    // non-const access
    /// @brief accesses or evaluates the requested coefficient
    constexpr decltype(auto) operator()(int i, int j)
        requires(ReadOnly == 0)
    {
        fdapde_assert(
          !(i < 0 || i >= rows() || j < 0 || j >= cols()), std::out_of_range, "diagonal index out of range");
        return xpr_(i, i);
    }
    /// @brief accesses the requested vector coefficient
    constexpr decltype(auto) operator[](int i)
        requires(ReadOnly == 0)
    {
        fdapde_assert(!(i < 0 || i >= rows()), std::out_of_range, "diagonal index out of range");
        return xpr_(i, i);
    }
    // observers
    /// @brief returns the row count
    constexpr int rows() const { return size_; }
    /// @brief returns the column count
    constexpr int cols() const { return 1; }
    /// @brief returns the coefficient count
    constexpr int size() const { return rows(); }
   private:
    XprTypeNested xpr_;
    int size_;
};

namespace internals {

// class wrapping an expression of the diagonal coefficients to a square matrix. internal usage only
/// @brief represents diagonal wrapper
template <int Rows_, int Cols_, typename DiagonalXprType_>
struct diagonal_wrapper : DiagonalMatrixExpr<diagonal_wrapper<Rows_, Cols_, DiagonalXprType_>> {
   private:
    using Base = DiagonalMatrixExpr<diagonal_wrapper<Rows_, Cols_, DiagonalXprType_>>;
    using XprType = std::remove_reference_t<DiagonalXprType_>;
    using XprTypeClean = std::remove_cv_t<XprType>;
    using XprTypeNested = internals::ref_select_t<DiagonalXprType_>;
   public:
    using Scalar = typename XprTypeClean::Scalar;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = XprTypeClean::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    /// @brief constructs diagonal wrapper from the supplied state
    constexpr diagonal_wrapper(const diagonal_wrapper&) = default;
    /// @brief constructs diagonal wrapper from the supplied state
    template <typename XprType__>
        requires(!std::same_as<std::remove_cvref_t<XprType__>, diagonal_wrapper> &&
                 internals::safely_nestable<XprTypeNested, XprType__>)
    constexpr diagonal_wrapper(XprType__&& xpr) :
        Base(XprTypeClean::Rows == 1 ? xpr.cols() : xpr.rows()), xpr_(std::forward<XprType__>(xpr)) { }
    /// @brief accesses or evaluates the requested coefficient
    constexpr Scalar operator()(int i, int j) const {
        fdapde_assert(
          !(i < 0 || i >= this->size_ || j < 0 || j >= this->size_), std::out_of_range,
          "diagonal matrix index out of range");
        return i == j ? xpr_[i] : Scalar(0);
    }
    /// @brief returns the main diagonal representation
    constexpr const XprTypeNested& diagonal() const { return xpr_; }
    /// @brief returns the underlying storage pointer
    constexpr const XprTypeNested& data() const { return xpr_; }
   private:
    XprTypeNested xpr_;
};

// helper cast function
/// @brief adapts an expression to diagonal matrix operations
template <typename XprType_> constexpr auto diagonal_cast(XprType_&& xpr) {
    using XprType = std::decay_t<XprType_>;
    constexpr int Rows = XprType::Rows, Cols = XprType::Cols;
    fdapde_static_assert(Rows == 1 || Cols == 1, THIS_METHOD_IS_FOR_ROW_OR_COLUMN_VECTORS_ONLY);
    return diagonal_wrapper < Rows == 1 ? Cols : Rows, Cols == 1 ? Rows : Cols,
           XprType_ > (std::forward<XprType_>(xpr));
}

}   // namespace internals

/// @brief represents diagonal matrix expr
template <typename XprType_> struct DiagonalMatrixExpr : public MatrixExpr<XprType_> {
    using XprType = std::decay_t<XprType_>;
    using Base = MatrixExpr<XprType_>;
    using Base::derived;

    /// @brief constructs diagonal matrix expr from the supplied state
    constexpr DiagonalMatrixExpr() noexcept :
        size_((XprType::Rows == Dynamic || XprType::Cols == Dynamic) ? 0 : XprType::Rows) { }
    /// @brief constructs diagonal matrix expr from the supplied state
    constexpr explicit DiagonalMatrixExpr(int size) noexcept :
        size_((XprType::Rows == Dynamic || XprType::Cols == Dynamic) ? size : XprType::Rows) { }
    // inherit assignment from base
    using Base::operator=;
    // const access
    /// @brief accesses or evaluates the requested coefficient
    constexpr decltype(auto) operator()(int i, int j) const {   // only const access allowed for (i, j) accessor
        fdapde_assert(
          !(i < 0 || i >= rows() || j < 0 || j >= cols()), std::out_of_range, "diagonal matrix index out of range");
        return i == j ? derived().data()[i] : typename XprType::Scalar(0);
    }
    /// @brief accesses the requested vector coefficient
    constexpr decltype(auto) operator[](int i) const {
        fdapde_assert(!(i < 0 || i >= rows()), std::out_of_range, "diagonal matrix index out of range");
        return derived().data()[i];
    }
    // non-const access
    /// @brief accesses the requested vector coefficient
    constexpr decltype(auto) operator[](int i)
        requires(XprType::ReadOnly == 0)
    {
        fdapde_assert(!(i < 0 || i >= rows()), std::out_of_range, "diagonal matrix index out of range");
        return derived().data()[i];
    }
    // converts to full dense matrix
    /// @brief materializes the expression as a dense matrix
    auto as_matrix() const {
        using Scalar = std::remove_cv_t<typename XprType::Scalar>;
        return Matrix<Scalar, XprType::Rows, XprType::Cols>(derived());
    }
    // matrix inverse as 1/coeff
    /// @brief returns the inverse matrix expression
    auto inverse() const& { return internals::diagonal_cast(derived().diagonal().cwise().inv()); }
    /// @brief returns the inverse matrix expression
    auto inverse() const&&
        requires(XprType::NestAsRef == 0)
    {
        return internals::diagonal_cast(derived().diagonal().cwise().inv());
    }
    /// @brief returns the inverse matrix expression
    void inverse() const&&
        requires(XprType::NestAsRef != 0)
    = delete;
    // linear system solver Ax = b
    /// @brief solves the factored linear system for the supplied right-hand side
    template <typename RhsXprType> constexpr auto solve(const RhsXprType& b) const {
        constexpr int Rows = XprType::Rows;
        using Scalar = std::remove_cv_t<typename XprType::Scalar>;
        Vector<Scalar, Rows> x;
        if constexpr (Rows == Dynamic) { x.resize(derived().rows()); }
        fdapde_assert(
          !(b.size() != derived().rows()), std::invalid_argument, "diagonal solve requires a matching right-hand side");
        for (int i = 0, n = derived().rows(); i < n; ++i) { x[i] = b[i] / derived().data()[i]; }
        return x;
    }
    /// @brief returns the matrix determinant
    constexpr auto determinant() const { return derived().diagonal().prod(); }
    // observers
    /// @brief returns the row count
    constexpr int rows() const { return size_; }
    /// @brief returns the column count
    constexpr int cols() const { return size_; }
   protected:
    int size_;
};

// diagonal matrix arithmetic (O(n) operations on the diagonal coefficients)
/// @brief implements the operator+ expression operation
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator+(const DiagonalMatrixExpr<LhsXprType>& lhs, const DiagonalMatrixExpr<RhsXprType>& rhs) {
    fdapde_static_assert(
      internals::same_static_shape_weak_v<LhsXprType FDAPDE_COMMA RhsXprType>,
      INVALID_OPERAND_DIMENSIONS_IN_BINARY_OPERATION);
    if constexpr (internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType>) {
        fdapde_assert(
          !(lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()), std::invalid_argument,
          "diagonal addition requires matching dimensions");
    }
    return internals::diagonal_cast(lhs.diagonal() + rhs.diagonal());
}
/// @brief implements the operator- expression operation
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator-(const DiagonalMatrixExpr<LhsXprType>& lhs, const DiagonalMatrixExpr<RhsXprType>& rhs) {
    fdapde_static_assert(
      internals::same_static_shape_weak_v<LhsXprType FDAPDE_COMMA RhsXprType>,
      INVALID_OPERAND_DIMENSIONS_IN_BINARY_OPERATION);
    if constexpr (internals::is_dynamic_sized_v<LhsXprType> || internals::is_dynamic_sized_v<RhsXprType>) {
        fdapde_assert(
          !(lhs.rows() != rhs.rows() || lhs.cols() != rhs.cols()), std::invalid_argument,
          "diagonal subtraction requires matching dimensions");
    }
    return internals::diagonal_cast(lhs.diagonal() - rhs.diagonal());
}
/// @brief implements the operator* expression operation
template <typename XprType, typename CoeffType>
    requires(std::is_arithmetic_v<CoeffType>)
constexpr auto operator*(const DiagonalMatrixExpr<XprType>& lhs, CoeffType rhs) {
    return internals::diagonal_cast(rhs * lhs.diagonal());
}
/// @brief implements the operator* expression operation
template <typename XprType, typename CoeffType>
    requires(std::is_arithmetic_v<CoeffType>)
constexpr auto operator*(CoeffType lhs, const DiagonalMatrixExpr<XprType>& rhs) {
    return rhs * lhs;
}
/// @brief implements the operator/ expression operation
template <typename XprType, typename CoeffType>
    requires(std::is_arithmetic_v<CoeffType>)
constexpr auto operator/(const DiagonalMatrixExpr<XprType>& lhs, CoeffType rhs) {
    return internals::diagonal_cast(lhs.diagonal() / rhs);
}

// specialized products
namespace internals {

// expressions of the (i,j)-th entry of the product between a diagonal and a dense matrix expression
/// @brief represents diagonal matrix product executor
template <int ProductMode> struct diagonal_matrix_product_executor {
    /// @brief executes the coefficient operation over the supplied expressions
    template <typename LhsXprType_, typename RhsXprType_>
    static constexpr decltype(auto) run(int i, int j, const LhsXprType_& lhs, const RhsXprType_& rhs) {
        if constexpr (ProductMode == LhsMode) return lhs(i, i) * rhs(i, j);
        if constexpr (ProductMode == RhsMode) return lhs(i, j) * rhs(j, j);
    }
};
// expressions of the (i,j)-th entry of the product between diagonal expressions
/// @brief represents diagonal diagonal product executor
struct diagonal_diagonal_product_executor {
    /// @brief executes the coefficient operation over the supplied expressions
    template <typename LhsXprType_, typename RhsXprType_>
    static constexpr decltype(auto) run(int i, int j, const LhsXprType_& lhs, const RhsXprType_& rhs) {
        using LhsXprType = std::decay_t<LhsXprType_>;
        using RhsXprType = std::decay_t<RhsXprType_>;
        using Scalar = promote_type_t<typename LhsXprType::Scalar, typename RhsXprType::Scalar>;
        return i == j ? lhs(i, i) * rhs(i, i) : Scalar(0);
    }
};

}   // namespace internals

/// @brief represents matrix multiplication op
template <typename LhsXprType, typename RhsXprType, typename Executor> struct MatrixMultiplicationOp;
// diag(a_1, ..., a_n) * M
/// @brief implements the operator* expression operation
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator*(const DiagonalMatrixExpr<LhsXprType>& lhs, const MatrixExpr<RhsXprType>& rhs) {
    return MatrixMultiplicationOp<LhsXprType, RhsXprType, internals::diagonal_matrix_product_executor<LhsMode>> {
      lhs.derived(), rhs.derived()};
}
// m * diag(a_1, ..., a_n)
/// @brief implements the operator* expression operation
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator*(const MatrixExpr<LhsXprType>& lhs, const DiagonalMatrixExpr<RhsXprType>& rhs) {
    return MatrixMultiplicationOp<LhsXprType, RhsXprType, internals::diagonal_matrix_product_executor<RhsMode>> {
      lhs.derived(), rhs.derived()};
}
// diag(a_1, ..., a_n) * diag(b_1, ..., b_n)
/// @brief implements the operator* expression operation
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator*(const DiagonalMatrixExpr<LhsXprType>& lhs, const DiagonalMatrixExpr<RhsXprType>& rhs) {
    return internals::diagonal_cast(
      MatrixMultiplicationOp<LhsXprType, RhsXprType, internals::diagonal_diagonal_product_executor>(
        lhs.derived(), rhs.derived())
        .diagonal());
}

// owning storage diagonal matrix
/// @brief owns a diagonal matrix
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

    /// @brief constructs diagonal matrix from the supplied state
    constexpr DiagonalMatrix() : Base(), data_() { }
    // copy semantic
    /// @brief constructs diagonal matrix from the supplied state
    constexpr DiagonalMatrix(const DiagonalMatrix& rhs) : Base() { clone_(rhs); }
    /// @brief assigns the supplied coefficients
    constexpr DiagonalMatrix& operator=(const DiagonalMatrix& rhs) & {
        clone_(rhs);
        return *this;
    }
    /// @brief constructs diagonal matrix from the supplied state
    template <typename Size>
        requires(Rows == Dynamic && std::same_as<Size, int>)
    constexpr explicit DiagonalMatrix(Size size) : Base(size), data_() {
        fdapde_assert(!(size < 0), std::invalid_argument, "diagonal matrix size must be nonnegative");
        data_.resize(size);
    }
    /// @brief constructs diagonal matrix from the supplied state
    template <typename RhsXprType_>
    constexpr DiagonalMatrix(const DiagonalMatrixExpr<RhsXprType_>& rhs) : Base(rhs.rows()), data_() {
        clone_(rhs.derived());
    }
    /// @brief constructs diagonal matrix from the supplied state
    constexpr explicit DiagonalMatrix(const std::vector<Scalar>& vec) :
        Base(internals::checked_matrix_data_size(vec.size())), data_() {
        const int input_size = internals::checked_matrix_data_size(vec.size());
        if constexpr (Rows == Dynamic || Cols == Dynamic) { data_.resize(input_size); }
        fdapde_assert(
          !(data_.size() != input_size), std::invalid_argument,
          "diagonal matrix input size does not match its static size");
        for (int i = 0, n = data_.size(); i < n; ++i) { data_[i] = vec[i]; }
    }
    /// @brief constructs diagonal matrix from the supplied state
    template <std::size_t RhsSize> constexpr explicit DiagonalMatrix(const Scalar (&data)[RhsSize]) : Base() {
        fdapde_static_assert(Rows != Dynamic && Cols != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        fdapde_static_assert(Rows == RhsSize, INVALID_DATA_SIZE);
        for (int i = 0, n = data_.size(); i < n; ++i) { data_[i] = data[i]; }
    }
    // constructor for 1 x 1, 2 x 2, 3 x 3 static sized diagonals
    /// @brief constructs diagonal matrix from the supplied state
    constexpr explicit DiagonalMatrix(Scalar x)
        requires(Rows == 1)
        : Base() {
        data_[0] = x;
    }
    /// @brief constructs diagonal matrix from the supplied state
    constexpr DiagonalMatrix(Scalar x, Scalar y) : Base() {
        fdapde_static_assert(Rows == 2, THIS_METHOD_IS_FOR_2_X_2_DIAGONAL_MATRICES_ONLY);
        data_[0] = x;
        data_[1] = y;
    }
    /// @brief constructs diagonal matrix from the supplied state
    constexpr DiagonalMatrix(Scalar x, Scalar y, Scalar z) : Base() {
        fdapde_static_assert(Rows == 3, THIS_METHOD_IS_FOR_3_X_3_DIAGONAL_MATRICES_ONLY);
        data_[0] = x;
        data_[1] = y;
        data_[2] = z;
    }
    /// @brief returns the main diagonal representation
    constexpr const StorageType& diagonal() const { return data_; }
    // modifiers
    /// @brief resizes the owned storage to the requested dimensions
    void resize(int size) {
        fdapde_static_assert(Rows == Dynamic || Cols == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_MATRICES_ONLY);
        fdapde_assert(!(size < 0), std::invalid_argument, "diagonal matrix size must be nonnegative");
        const int size_ = Rows == Dynamic ? size : Rows;
        if (std::cmp_equal(size_, data_.size())) return;   // do not reallocate memory if sizes didn't changed
        data_.resize(size);
        this->size_ = size_;
        return;
    }
    // data pointers
    /// @brief returns the underlying storage pointer
    constexpr const Scalar* data() const { return data_.data(); }
    /// @brief returns the underlying storage pointer
    constexpr Scalar* data() { return data_.data(); }
   private:
    /// @brief returns an owning copy of the coefficients
    template <typename RhsXprType> constexpr void clone_(const RhsXprType& rhs) {
        if constexpr (Rows == Dynamic || Cols == Dynamic) { resize(rhs.rows()); }
        assignment_executor::run(*this, rhs, [](auto&& l, const auto& r) { l = r; });
        return;
    }
    StorageType data_;
};

// diagonal view of an existing block of data
/// @brief represents diagonal matrix view
template <typename Scalar_, int Rows_>
class DiagonalMatrixView : public DiagonalMatrixExpr<DiagonalMatrixView<Scalar_, Rows_>> {
   private:
    using Base = DiagonalMatrixExpr<DiagonalMatrixView<Scalar_, Rows_>>;
   public:
    using Scalar = Scalar_;
    using StorageType = std::add_pointer_t<Scalar>;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Rows_;
    static constexpr int StorageOrder = RowMajor;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = std::is_const_v<Scalar_>;
    using assignment_executor = internals::diagonal_assignment_executor;

    // constructors
    /// @brief constructs diagonal matrix view from the supplied state
    constexpr DiagonalMatrixView(const DiagonalMatrixView&) = default;
    /// @brief constructs diagonal matrix view from the supplied state
    constexpr DiagonalMatrixView()
        requires(Rows_ == Dynamic)
        : Base(), data_(nullptr) { }
    /// @brief constructs diagonal matrix view from the supplied state
    constexpr DiagonalMatrixView()
        requires(Rows_ != Dynamic)
    = delete;
    /// @brief constructs diagonal matrix view from the supplied state
    constexpr explicit DiagonalMatrixView(Scalar* data) : Base(), data_(data) {
        fdapde_static_assert(Rows_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_DIAGONAL_VIEWS_ONLY);
        fdapde_assert(!(data == nullptr), std::invalid_argument, "nonempty diagonal view requires storage");
    }
    /// @brief constructs diagonal matrix view from the supplied state
    constexpr DiagonalMatrixView(Scalar* data, int size)
        requires(Rows_ == Dynamic)
        : Base(size), data_(data) {
        fdapde_assert(
          !(size < 0 || (size > 0 && data == nullptr)), std::invalid_argument,
          "diagonal view requires a nonnegative size and valid storage");
    }
    /// @brief returns the main diagonal representation
    constexpr VectorView<const std::remove_const_t<Scalar>, Rows_> diagonal() const {
        if constexpr (Rows_ == Dynamic) {
            if (this->rows() == 0) { return VectorView<const std::remove_const_t<Scalar>, Rows_>(); }
            return VectorView<const std::remove_const_t<Scalar>, Rows_>(data_, this->rows());
        } else {
            return VectorView<const std::remove_const_t<Scalar>, Rows_>(data_);
        }
    }
    /// @brief returns the main diagonal representation
    constexpr VectorView<Scalar, Rows_> diagonal()
        requires(!std::is_const_v<Scalar>)
    {
        if constexpr (Rows_ == Dynamic) {
            if (this->rows() == 0) { return VectorView<Scalar, Rows_>(); }
            return VectorView<Scalar, Rows_>(data_, this->rows());
        } else {
            return VectorView<Scalar, Rows_>(data_);
        }
    }
    // inherit assignment from Base
    using Base::operator=;
    /// @brief assigns the supplied coefficients
    constexpr DiagonalMatrixView& operator=(const DiagonalMatrixView& other) &
        requires(ReadOnly == 0)
    {
        static_cast<Base&>(*this).template operator= <DiagonalMatrixView>(other);
        return *this;
    }
    /// @brief assigns the supplied coefficients
    constexpr DiagonalMatrixView operator=(const DiagonalMatrixView& other) &&
      requires(ReadOnly == 0) {
          static_cast<Base&>(*this).template operator= <DiagonalMatrixView>(other);
          return *this;
      }
      // data pointers
      /// @brief returns the underlying storage pointer
      constexpr const std::remove_const_t<Scalar>* data() const {
        return data_;
    }
    /// @brief returns the underlying storage pointer
    constexpr StorageType data() { return data_; }
   private:
    StorageType data_;
};

// detection trait
/// @brief detects is diagonal matrix
template <typename XprType> struct is_diagonal_matrix {
    using Type = std::remove_cvref_t<XprType>;
    static constexpr bool value = std::is_base_of_v<DiagonalMatrixExpr<Type>, Type>;
};
template <typename XprType> static constexpr bool is_diagonal_matrix_v = is_diagonal_matrix<XprType>::value;

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_DIAGONAL_H__
