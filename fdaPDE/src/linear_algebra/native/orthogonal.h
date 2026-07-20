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

namespace fdapde::linalg {

// orthogonal matrix type system (implementation of the general orthogonal Lie-group O(n))

namespace internals {

struct orthogonalize_t { };   // tag used to activate input orthogonalization

constexpr int compute_square_shape(std::size_t size) {
    int n = 0;
    while (static_cast<std::size_t>(n) * static_cast<std::size_t>(n) < size) ++n;
    return static_cast<std::size_t>(n) * static_cast<std::size_t>(n) == size ? n : -1;
}

template <typename XprType> auto orthogonal_cast(XprType&& xpr);

template <typename XprType> constexpr bool is_orthogonal(const XprType& xpr) {
    if (xpr.rows() <= 0 || xpr.rows() != xpr.cols()) return false;
    using Scalar = std::remove_cv_t<typename std::remove_cvref_t<XprType>::Scalar>;
    const Scalar tolerance = Scalar(64) * std::numeric_limits<Scalar>::epsilon() * Scalar(xpr.rows());
    for (int i = 0; i < xpr.rows(); ++i) {
        for (int j = 0; j < xpr.rows(); ++j) {
            Scalar dot = 0;
            for (int k = 0; k < xpr.cols(); ++k) dot += xpr(i, k) * xpr(j, k);
            const Scalar expected = i == j ? Scalar(1) : Scalar(0);
            if (!(std::abs(dot - expected) <= tolerance)) return false;
        }
    }
    return true;
}

}   // namespace internals

[[maybe_unused]] inline constexpr internals::orthogonalize_t orthogonalize {};

template <typename XprType_> struct OrthogonalMatrixExpr : public MatrixExpr<XprType_> {
    using XprType = std::decay_t<XprType_>;

    constexpr auto inverse() const & { return internals::orthogonal_cast(this->derived().transpose()); }
    constexpr auto inverse() const && requires(XprType::NestAsRef == 0) {
        return internals::orthogonal_cast(this->derived().transpose());
    }
    constexpr void inverse() const && requires(XprType::NestAsRef != 0) = delete;
    template <typename RhsXprType> constexpr auto solve(const RhsXprType& b) const {
        using RhsType = std::decay_t<RhsXprType>;
        using Scalar = promote_type_t<typename XprType::Scalar, typename RhsType::Scalar>;
        return Matrix<Scalar, XprType::Cols, RhsType::Cols>(inverse() * b);
    }
};

template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_, typename OrthogonalMatrixType_>
struct OrthogonalMatrixBase : public OrthogonalMatrixExpr<OrthogonalMatrixType_> {
   private:
    using Base = OrthogonalMatrixExpr<OrthogonalMatrixType_>;
    using Base::derived;
   public:
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;
    using assignment_executor = internals::generic_assignment_executor;

    OrthogonalMatrixBase() = delete;
    OrthogonalMatrixBase(int rows, int cols) :
        rows_(Rows_ == Dynamic ? rows : Rows_), cols_(Cols_ == Dynamic ? cols : Cols_) {
        const bool valid = rows > 0 && cols > 0 && rows == cols && (Rows == Dynamic || rows == Rows) &&
                           (Cols == Dynamic || cols == Cols);
        if (!valid) {
            fdapde_assert(valid);
            rows_ = 0;
            cols_ = 0;
        }
    }
    // only read access allowed (write access could break orthogonality invariant)
    constexpr Scalar operator()(int i, int j) const {
        fdapde_assert(i >= 0 && i < rows_ && j >= 0 && j < cols_);
        return derived().data()(i, j);
    }
    // observers
    constexpr int rows() const { return rows_; }
    constexpr int cols() const { return cols_; }
   protected:
    // orthogonalize (modified gram–schmidt), with normalization
    template <typename MatrixType_> bool orthogonalize_(MatrixType_& m) {
        Matrix<Scalar, Rows, Cols> Q;
        if constexpr (Rows == Dynamic || Cols == Dynamic) { Q.resize(m.rows(), m.cols()); }
        const double tol = m.norm();
        const double eps = fdapde::sqrt(std::numeric_limits<Scalar>::epsilon());

        // Modified Gram–Schmidt
        int rank = 0;
        for (int i = 0, n = m.rows(); i < n; ++i) {
            Vector<Scalar, Rows> v(m.col(i));
            for (int pass = 0; pass < 2; ++pass) {
                for (int j = 0; j < rank; ++j) { v -= (Q.col(j).dot(v)) * Q.col(j); }
            }

            double v_norm = v.norm();
            if (!(v_norm > eps * fdapde::max(tol, fdapde::max(m.col(i).norm(), 1.0)))) {
                break;   // dependent column (don’t store a zero column in Q)
            }
            Q.col(rank++) = v / v_norm;
        }
        if (rank != m.rows() || !internals::is_orthogonal(Q)) return false;
        m = Q;
        return true;
    }

    int rows_, cols_;
};

// A matrix with enforced orthogonality check (i.e., M * M^\top = I)
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_ = RowMajor>
struct OrthogonalMatrix :
    public OrthogonalMatrixBase<
      Scalar_, Rows_, Cols_, StorageOrder_, OrthogonalMatrix<Scalar_, Rows_, Cols_, StorageOrder_>> {
    fdapde_static_assert(
      Rows_ == Dynamic || Cols_ == Dynamic || Rows_ == Cols_, THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
    fdapde_static_assert(std::is_floating_point_v<Scalar_>, ORTHOGONAL_MATRICES_REQUIRE_FLOATING_POINT_SCALARS);
   private:
    using Base = OrthogonalMatrixBase<
      Scalar_, Rows_, Cols_, StorageOrder_, OrthogonalMatrix<Scalar_, Rows_, Cols_, StorageOrder_>>;
    using StorageType = Matrix<Scalar_, Rows_, Cols_, StorageOrder_>;
   public:
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int ReadOnly = 1;
    static constexpr int NestAsRef = 1;
    using assignment_executor = typename StorageType::assignment_executor;

    // empty orthogonal matrices are ill-formed by definition
    constexpr OrthogonalMatrix() = delete;
    constexpr OrthogonalMatrix(int rows, int cols) = delete;
    // copy semantic
    constexpr OrthogonalMatrix(const OrthogonalMatrix& other) : Base(other.rows(), other.cols()) { clone_(other); }
    constexpr OrthogonalMatrix& operator=(const OrthogonalMatrix& rhs) & {
        if (this == std::addressof(rhs)) return *this;
        clone_(rhs);
        return *this;
    }
    template <typename RhsXprType_>
    constexpr OrthogonalMatrix(const MatrixExpr<RhsXprType_>& rhs, internals::unchecked_t) :
        Base(rhs.rows(), rhs.cols()), m_() {
        fdapde_static_assert(
          Rows == Dynamic || Cols == Dynamic ||
            (Rows == RhsXprType_::Rows && Cols == RhsXprType_::Cols),
          INVALID_ORTHOGONAL_MATRIX_SHAPE);
        if (!load_expression_(rhs.derived())) set_identity_(fallback_shape_());
    }
    template <typename RhsXprType_>
    constexpr OrthogonalMatrix(const OrthogonalMatrixExpr<RhsXprType_>& rhs) :
        OrthogonalMatrix(rhs.derived(), unchecked) { }
    // constructors taking external data
    // orthogonalize input
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr OrthogonalMatrix(const std::vector<Scalar__>& data, internals::orthogonalize_t) :
        Base(internals::compute_square_shape(data.size()), internals::compute_square_shape(data.size())), m_() {
        if (load_data_(data) && !this->orthogonalize_(m_)) {
            fdapde_assert(false);
            set_identity_(this->rows_);
        }
    }
    template <typename Scalar__, std::size_t Size>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr OrthogonalMatrix(const Scalar__ (&data)[Size], internals::orthogonalize_t) :
        Base(internals::compute_square_shape(Size), internals::compute_square_shape(Size)), m_() {
        fdapde_static_assert(
          Rows_ != Dynamic && Cols_ != Dynamic && Rows_ * Cols_ == Size, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        if (load_data_(data) && !this->orthogonalize_(m_)) {
            fdapde_assert(false);
            set_identity_(this->rows_);
        }
    }
    // assume input already orthogonal, abort if assumption failed
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr OrthogonalMatrix(const std::vector<Scalar__>& data, internals::checked_t) :
        Base(internals::compute_square_shape(data.size()), internals::compute_square_shape(data.size())), m_() {
        if (load_data_(data) && !internals::is_orthogonal(m_)) {
            fdapde_assert(false);
            set_identity_(this->rows_);
        }
    }
    template <typename Scalar__, std::size_t Size>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr OrthogonalMatrix(const Scalar__ (&data)[Size], internals::checked_t) :
        Base(internals::compute_square_shape(Size), internals::compute_square_shape(Size)), m_() {
        fdapde_static_assert(
          Rows_ != Dynamic && Cols_ != Dynamic && Rows_ * Cols_ == Size, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        if (load_data_(data) && !internals::is_orthogonal(m_)) {
            fdapde_assert(false);
            set_identity_(this->rows_);
        }
    }
    // assume input already orthogonal, trusts the caller
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr OrthogonalMatrix(const std::vector<Scalar__>& data, internals::unchecked_t) :
        Base(internals::compute_square_shape(data.size()), internals::compute_square_shape(data.size())), m_() {
        load_data_(data);
    }
    template <typename Scalar__, std::size_t Size>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr OrthogonalMatrix(const Scalar__ (&data)[Size], internals::unchecked_t) :
        Base(internals::compute_square_shape(Size), internals::compute_square_shape(Size)), m_() {
        fdapde_static_assert(
          Rows_ != Dynamic && Cols_ != Dynamic && Rows_ * Cols_ == Size, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        load_data_(data);
    }
    // read-only storage access preserves the invariant
    constexpr const StorageType& data() const { return m_; }
   private:
    static constexpr int fallback_shape_() {
        if constexpr (Rows_ != Dynamic) return Rows_;
        if constexpr (Cols_ != Dynamic) return Cols_;
        return 1;
    }
    constexpr void set_identity_(int size) {
        const int n = size > 0 ? size : fallback_shape_();
        this->rows_ = n;
        this->cols_ = n;
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) m_.resize(n, n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) m_(i, j) = i == j ? Scalar(1) : Scalar(0);
        }
    }
    template <typename Data> constexpr bool load_data_(const Data& data) {
        const std::size_t size = std::size(data);
        const bool valid = this->rows_ > 0 && this->rows_ == this->cols_ &&
                           size == static_cast<std::size_t>(this->rows_ * this->cols_);
        if (!valid) {
            fdapde_assert(valid);
            set_identity_(fallback_shape_());
            return false;
        }
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) m_.resize(this->rows_, this->cols_);
        for (int i = 0; i < this->rows_; ++i) {
            for (int j = 0; j < this->cols_; ++j) m_(i, j) = data[i * this->cols_ + j];
        }
        return true;
    }
    template <typename RhsXprType> constexpr bool load_expression_(const RhsXprType& rhs) {
        const bool valid = this->rows_ > 0 && rhs.rows() == this->rows_ && rhs.cols() == this->cols_;
        if (!valid) {
            fdapde_assert(valid);
            return false;
        }
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) m_.resize(this->rows_, this->cols_);
        assignment_executor::run(m_, rhs, [](auto& l, const auto& r) { l = r; });
        return true;
    }
    template <typename RhsXprType> constexpr void clone_(const RhsXprType& rhs) {
        this->rows_ = rhs.rows();
        this->cols_ = rhs.cols();
        if (!load_expression_(rhs)) set_identity_(fallback_shape_());
    }
    StorageType m_;
};

namespace internals {

// class used by the product operation to achieve closure wrt group operation. internal usage only
template <typename OrthogonalXprType_>
struct orthogonal_wrapper : OrthogonalMatrixExpr<orthogonal_wrapper<OrthogonalXprType_>> {
   private:
    using Base = OrthogonalMatrixExpr<orthogonal_wrapper<OrthogonalXprType_>>;
    using XprType = std::decay_t<OrthogonalXprType_>;
    using XprTypeNested = fdapde::internals::ref_select_t<const XprType>;
   public:
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    static constexpr int StorageOrder = XprType::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    constexpr orthogonal_wrapper(const orthogonal_wrapper&) = default;
    template <typename XprType__>
        requires(
          !std::same_as<std::remove_cvref_t<XprType__>, orthogonal_wrapper> &&
          internals::safely_nestable<XprTypeNested, XprType__>)
    constexpr orthogonal_wrapper(XprType__&& xpr) : xpr_(std::forward<XprType__>(xpr)) { }
    constexpr Scalar operator()(int i, int j) const {
        fdapde_assert(i >= 0 && i < xpr_.rows() && j >= 0 && j < xpr_.cols());
        return xpr_(i, j);
    }
    constexpr int rows() const { return xpr_.rows(); }
    constexpr int cols() const { return xpr_.cols(); }
   private:
    XprTypeNested xpr_;
};

// helper cast function
template <typename XprType_> auto orthogonal_cast(XprType_&& xpr) {
    return orthogonal_wrapper<XprType_>(std::forward<XprType_>(xpr));
}

}   // namespace internals

// orthogonal group operation
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator*(const OrthogonalMatrixExpr<LhsXprType>& lhs, const OrthogonalMatrixExpr<RhsXprType>& rhs) {
    return internals::orthogonal_cast(
      MatrixMultiplicationOp<LhsXprType, RhsXprType, internals::generic_matrix_product_executor> {
        lhs.derived(), rhs.derived()});
}
// any other operation doesn't preserve orthogonality. A raw MatrixExpr is returned

// orthogonal view of an existing block of data
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_ = RowMajor>
class OrthogonalMatrixView :
    public OrthogonalMatrixBase<
      Scalar_, Rows_, Cols_, StorageOrder_, OrthogonalMatrixView<Scalar_, Rows_, Cols_, StorageOrder_>> {
    fdapde_static_assert(
      Rows_ == Dynamic || Cols_ == Dynamic || Rows_ == Cols_, THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
    fdapde_static_assert(
      std::is_floating_point_v<std::remove_const_t<Scalar_>>, ORTHOGONAL_MATRICES_REQUIRE_FLOATING_POINT_SCALARS);
   private:
    using Base = OrthogonalMatrixBase<
      Scalar_, Rows_, Cols_, StorageOrder_, OrthogonalMatrixView<Scalar_, Rows_, Cols_, StorageOrder_>>;
    using StorageType = MatrixView<Scalar_, Rows_, Cols_, StorageOrder_>;
   public:
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;
    using assignment_executor = typename StorageType::assignment_executor;

    constexpr OrthogonalMatrixView() = delete;
    constexpr OrthogonalMatrixView(const OrthogonalMatrixView&) = default;
    template <typename Scalar__>
        requires(std::is_convertible_v<Scalar__*, Scalar_*>)
    constexpr OrthogonalMatrixView(Scalar__* data, internals::checked_t) : Base(Rows_, Cols_), data_(data) {
        fdapde_static_assert(Rows_ != Dynamic && Cols_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_VIEWS_ONLY);
        validate_(data != nullptr && internals::is_orthogonal(data_));
    }
    template <typename Scalar__>
        requires(std::is_convertible_v<Scalar__*, Scalar_*>)
    constexpr OrthogonalMatrixView(Scalar__* data, int rows, int cols, internals::checked_t) :
        Base(rows, cols), data_(data, rows, cols) {
        validate_(data != nullptr && internals::is_orthogonal(data_));
    }
    template <typename Scalar__>
        requires(std::is_convertible_v<Scalar__*, Scalar_*>)
    constexpr OrthogonalMatrixView(Scalar__* data, internals::unchecked_t) : Base(Rows_, Cols_), data_(data) {
        fdapde_static_assert(Rows_ != Dynamic && Cols_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_VIEWS_ONLY);
        validate_(data != nullptr);
    }
    template <typename Scalar__>
        requires(std::is_convertible_v<Scalar__*, Scalar_*>)
    constexpr OrthogonalMatrixView(Scalar__* data, int rows, int cols, internals::unchecked_t) :
        Base(rows, cols), data_(data, rows, cols) {
        validate_(data != nullptr && this->rows_ == rows && this->cols_ == cols);
    }
    constexpr OrthogonalMatrixView& operator=(const OrthogonalMatrixView& other) &
        requires(!std::is_const_v<Scalar_>) {
        data_ = other.data_;
        return *this;
    }
    constexpr const StorageType& data() const { return data_; }
   private:
    constexpr void validate_(bool valid) {
        if (!valid) {
            fdapde_assert(valid);
            this->rows_ = 0;
            this->cols_ = 0;
        }
    }
    StorageType data_;
};

// detection trait
template <typename XprType> struct is_orthogonal_matrix {
    using Type = std::remove_cvref_t<XprType>;
    static constexpr bool value = std::is_base_of_v<OrthogonalMatrixExpr<Type>, Type>;
};
template <typename XprType> static constexpr bool is_orthogonal_matrix_v = is_orthogonal_matrix<XprType>::value;

}   // namespace fdapde::linalg

#endif   // _FDAPDE_LINALG_ORTHOGONAL_H__
