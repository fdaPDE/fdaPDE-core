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

// orthogonal matrix type system (implementation of the general orthogonal Lie-group O(n))

namespace internals {

struct orthogonalize_t { };   // tag used to activate input orthogonalization

}   // namespace internals

[[maybe_unused]] inline constexpr internals::orthogonalize_t orthogonalize {};

template <typename XprType_> struct OrthogonalMatrixExpr : public MatrixExpr<XprType_> {
    // make derived() point to innermost type
    constexpr const XprType_& derived() const { return static_cast<const XprType_&>(*this); }
    constexpr XprType_& derived() { return static_cast<XprType_&>(*this); }

    constexpr auto inverse() const { return derived().transpose().as_orthogonal(); }   // M^{-1} = M^\top
    template <typename RhsXprType> constexpr auto solve(const RhsXprType& b) const {
        return Vector<typename XprType_::Scalar, XprType_::Rows>(inverse() * b);
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
    static constexpr int ReadOnly = std::is_const_v<Scalar_>;
    using assignment_executor = internals::generic_assignment_executor;

    OrthogonalMatrixBase() = delete;
    OrthogonalMatrixBase(int rows, int cols) :
        rows_(Rows_ == Dynamic ? rows : Rows_), cols_(Cols_ == Dynamic ? cols : Cols_) {
        fdapde_assert(rows > 0 && cols > 0 && rows == cols);
    }
    // copy assignment
    constexpr OrthogonalMatrixType_& operator=(const OrthogonalMatrixType_& other) {
        fdapde_static_assert(ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION);
        if (this == std::addressof(other)) { return derived(); }
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) {
            if (rows_ != other.rows() || cols_ != other.cols()) { derived().resize(other.rows(), other.cols()); }
        }
        assignment_executor::run(*this, other, [](auto&& l, const auto& r) { l = r; });
        return derived();
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
    template <typename MatrixType_> void orthogonalize_(MatrixType_& m) {
        Matrix<Scalar, Rows, Cols> Q;
        if constexpr (Rows == Dynamic || Cols == Dynamic) { Q.resize(m.rows(), m.cols()); }
        const double tol = m.norm();
        const double eps = fdapde::sqrt(std::numeric_limits<Scalar>::epsilon());

        // Modified Gram–Schmidt
        int rank = 0;
        for (int i = 0, n = m.rows(); i < n; ++i) {
            Vector<Scalar, Rows> v(derived().col(i));
            for (int j = 0; j < rank; ++j) { v -= (Q.col(j).dot(v)) * Q.col(j); }   // Q.col(j) is unit-norm

            double v_norm = v.norm();
            if (v_norm <= eps * fdapde::max(tol, fdapde::max(m.col(i).norm(), 1.0))) {
                break;   // dependent column (don’t store a zero column in Q)
            }
            Q.col(rank++) = v / v_norm;
        }
        fdapde_assert(rank == derived().rows());   // not full rank, unable to orthogonalize to square matrix
        m = Q;
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
   private:
    using Base = OrthogonalMatrixBase<
      Scalar_, Rows_, Cols_, StorageOrder_, OrthogonalMatrix<Scalar_, Rows_, Cols_, StorageOrder_>>;
    using StorageType = Matrix<Scalar_, Rows_, Cols_>;
   public:
    using Scalar = Scalar_;
    static constexpr int NestAsRef = 1;
    using assignment_executor = typename StorageType::assignment_executor;

    // empty orthogonal matrices are ill-formed by definition
    constexpr OrthogonalMatrix() = delete;
    constexpr OrthogonalMatrix(int rows, int cols) = delete;
    // copy semantic
    constexpr OrthogonalMatrix(const OrthogonalMatrix& other) : Base(other.rows(), other.cols()) { clone_(other); }
    constexpr OrthogonalMatrix& operator=(const OrthogonalMatrix& rhs) {
        this->rows_ = rhs.rows_;
        this->cols_ = rhs.cols_;
        clone_(rhs);
        return *this;
    }
    template <typename RhsXprType_>
    constexpr OrthogonalMatrix(const MatrixExpr<RhsXprType_>& rhs, internals::unchecked_t) :
        Base(rhs.rows(), rhs.cols()) {
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) { m_.resize(rhs.rows(), rhs.cols()); }
        assignment_executor::run(m_, rhs.derived(), [](auto&& l, const auto& r) { l = r; });
    }
    template <typename RhsXprType_>
    constexpr OrthogonalMatrix(const OrthogonalMatrixExpr<RhsXprType_>& rhs) : OrthogonalMatrix(rhs, unchecked) { }
    // constructors taking external data
    // orthogonalize input
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr OrthogonalMatrix(const std::vector<Scalar__>& data, internals::orthogonalize_t) :
        Base(fdapde::sqrt(data.size()), fdapde::sqrt(data.size())), m_(data) {
        this->orthogonalize_(m_);
    }
    template <typename Scalar__, std::size_t Size>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr OrthogonalMatrix(const Scalar__ (&data)[Size], internals::orthogonalize_t) :
        Base(fdapde::sqrt(data.size()), fdapde::sqrt(data.size())), m_(data) {
        fdapde_static_assert(
          Rows_ != Dynamic && Cols_ != Dynamic && Rows_ * Cols_ == Size, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        this->orthogonalize_(m_);
    }
    // assume input already orthogonal, abort if assumption failed
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr OrthogonalMatrix(const std::vector<Scalar__>& data, internals::checked_t) :
        Base(fdapde::sqrt(data.size()), fdapde::sqrt(data.size())), m_(data) {
        const int n = m_.rows();
        auto I = IdentityMatrix<Dynamic, Dynamic>(n, n);
        fdapde_assert(almost_equal(m_ * m_.transpose() FDAPDE_COMMA I FDAPDE_COMMA 1e-14));
    }
    template <typename Scalar__, std::size_t Size>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr OrthogonalMatrix(const Scalar__ (&data)[Size], internals::checked_t) :
        Base(fdapde::sqrt(data.size()), fdapde::sqrt(data.size())), m_(data) {
        fdapde_static_assert(
          Rows_ != Dynamic && Cols_ != Dynamic && Rows_ * Cols_ == Size, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        const int n = m_.rows();
        auto I = IdentityMatrix<Dynamic, Dynamic>(n, n);
        fdapde_assert(almost_equal(m_ * m_.transpose() FDAPDE_COMMA I FDAPDE_COMMA 1e-14));
    }
    // assume input already orthogonal, trusts the caller
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr OrthogonalMatrix(const std::vector<Scalar__>& data, internals::unchecked_t) :
        Base(fdapde::sqrt(data.size()), fdapde::sqrt(data.size())), m_(data) { }
    template <typename Scalar__, std::size_t Size>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr OrthogonalMatrix(const Scalar__ (&data)[Size], internals::unchecked_t) :
        Base(fdapde::sqrt(data.size()), fdapde::sqrt(data.size())), m_(data) {
        fdapde_static_assert(
          Rows_ != Dynamic && Cols_ != Dynamic && Rows_ * Cols_ == Size, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
    }
    // data pointers
    constexpr const StorageType& data() const { return m_; }
    constexpr StorageType& data() { return m_; }
   private:
    template <typename RhsXprType> constexpr void clone_(const RhsXprType& rhs) {
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) { m_.resize(rhs.rows(), rhs.cols()); }
        assignment_executor::run(m_, rhs, [](auto&& l, const auto& r) { l = r; });
        return;
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
    using XprTypeNested = internals::ref_select_t<const XprType>;
   public:
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    static constexpr int StorageOrder = XprType::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename XprType__>
        requires(std::is_constructible_v<XprTypeNested, XprType__>)
    constexpr orthogonal_wrapper(XprType__&& xpr) : xpr_(std::forward<XprType__>(xpr)) { }
    constexpr Scalar operator()(int i, int j) const {
        fdapde_assert(i >= 0 && i < xpr_.rows() && j >= 0 && j < xpr_.cols());
        return xpr_(i, j);
    }
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
    return internals::orthogonal_cast(lhs * rhs);
}
// any other operation doesn't preserve orthogonality. A raw MatrixExpr is returned

// orthogonal view of an existing block of data
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_ = RowMajor>
class OrthogonalMatrixView : public OrthogonalMatrixExpr<OrthogonalMatrixView<Scalar_, Rows_, Cols_, StorageOrder_>> {
   private:
    using Base = OrthogonalMatrixExpr<OrthogonalMatrixView<Scalar_, Rows_, Cols_, StorageOrder_>>;
    using StorageType = MatrixView<Scalar_, Rows_, Rows_, StorageOrder_>;
   public:
    using Scalar = Scalar_;
    static constexpr int NestAsRef = 0;

    // constructors
    constexpr OrthogonalMatrixView() : Base(), data_() { }
    // assume input already orthogonal, abort if assumption failed
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr OrthogonalMatrixView(Scalar__* data, internals::checked_t) : Base(), data_(data) {
        fdapde_static_assert(Rows_ != Dynamic && Cols_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_VIEWS_ONLY);
        const int size = data_.rows();
        auto I = IdentityMatrix<Dynamic, Dynamic>(size, size);
        fdapde_assert(almost_equal(data_ * data_.transpose() FDAPDE_COMMA I FDAPDE_COMMA 1e-14));
    }
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr OrthogonalMatrixView(Scalar__* data, int rows, int cols, internals::checked_t) :
        Base(rows, cols), data_(data) {
        fdapde_assert(rows > 0 && cols > 0 && rows == cols);
        auto I = IdentityMatrix<Dynamic, Dynamic>(rows, cols);
        fdapde_assert(almost_equal(data_ * data_.transpose() FDAPDE_COMMA I FDAPDE_COMMA 1e-14));
    }
    // assume input already orthogonal, trusts the caller
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr OrthogonalMatrixView(Scalar__* data, internals::unchecked_t) : Base(), data_(data) {
        fdapde_static_assert(Rows_ != Dynamic && Cols_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_VIEWS_ONLY);
    }
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr OrthogonalMatrixView(Scalar__* data, int rows, int cols, internals::unchecked_t) :
        Base(rows, cols), data_(data) {
        fdapde_assert(rows > 0 && cols > 0 && rows == cols);
    }
    // data pointers
    constexpr const StorageType& data() const { return data_; }
    constexpr StorageType& data() { return data_; }
   private:
    StorageType data_;
};

// detection trait
template <typename XprType> struct is_orthogonal_matrix {
    static constexpr bool value = std::is_base_of_v<OrthogonalMatrixExpr<std::decay_t<XprType>>, XprType>;
};
template <typename XprType> static constexpr bool is_orthogonal_matrix_v = is_orthogonal_matrix<XprType>::value;

}   // namespace fdapde

#endif   // _FDAPDE_LINALG_ORTHOGONAL_H__
