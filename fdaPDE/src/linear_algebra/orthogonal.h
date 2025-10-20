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

template <typename XprType_> struct OrthogonalMatrixExpr : public MatrixExpr<XprType_> {
    using Base = MatrixExpr<XprType_>;
    using Base::derived;

    constexpr auto inverse() const { return derived().transpose().as_orthogonal(); }
    template <typename RhsXprType> constexpr auto solve(const RhsXprType& b) const {
        return Vector<typename XprType_::Scalar, XprType_::Rows>(inverse() * b);
    }
};

template <typename Scalar_, int Size_, int StorageOrder_, typename OrthogonalMatrixType_>
struct OrthogonalMatrixBase : public OrthogonalMatrixExpr<OrthogonalMatrixType_> {
    using Base = OrthogonalMatrixExpr<OrthogonalMatrixType_>;
    using Base::derived;
    using Scalar = Scalar_;
    static constexpr int Rows = Size_;
    static constexpr int Cols = Size_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = std::is_const_v<Scalar_> ? 1 : 0;
    struct assignment_executor {
        template <typename SrcXprType> static constexpr void run(OrthogonalMatrixType_& dst, const SrcXprType& src) {
            fdapde_assert(dst.rows() == src.rows() && dst.cols() == src.cols());
            int rows_ = dst.rows();
            int cols_ = dst.cols();
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) { dst.data()(i, j) = src(i, j); }
            }
            return;
        }
    };
  
    OrthogonalMatrixBase() = delete;
    OrthogonalMatrixBase(int size) : size_(Size_ == Dynamic ? size : Size_) { }
      // copy assignment
    constexpr OrthogonalMatrixType_& operator=(const OrthogonalMatrixType_& other) {
        fdapde_static_assert(ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION);
        if constexpr (Size_ == Dynamic) { fdapde_assert(size_ == other.rows() && size_ == other.cols()); }
        if (this == std::addressof(other)) { return derived(); }
        assignment_executor::run(*this, other);
        return derived();
    }
    // only read access allowed (write access could break orthogonality invariant)
    constexpr const Scalar& operator()(int i, int j) const {
        fdapde_assert(i >= 0 && i < size_ && j >= 0 && j < size_);
        return derived().data()(i, j);
    }
    // observers
    constexpr int rows() const { return derived().data().rows(); }
    constexpr int cols() const { return derived().data().cols(); }
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

    int size_;
};
  
// A matrix with enforced orthogonality check (i.e., M * M^\top = I)
template <typename Scalar_, int Size_, int StorageOrder_ = RowMajor>
struct OrthogonalMatrix :
    public OrthogonalMatrixBase<Scalar_, Size_, StorageOrder_, OrthogonalMatrix<Scalar_, Size_, StorageOrder_>> {
    using Base = OrthogonalMatrixBase<Scalar_, Size_, StorageOrder_, OrthogonalMatrix<Scalar_, Size_, StorageOrder_>>;
    using Scalar = Scalar_;
    static constexpr int StorageSize = (Size_ == Dynamic) ? Dynamic : (Size_ * Size_);
    using StorageType = Matrix<Scalar, Size_, Size_>;
    static constexpr int Rows = Size_;
    static constexpr int Cols = Size_;
    static constexpr int NestAsRef = 1;
  

    // empty orthogonal matrices are ill-formed by definition
    constexpr OrthogonalMatrix() = delete;
    constexpr OrthogonalMatrix(int rows, int cols) = delete;
    constexpr OrthogonalMatrix(const OrthogonalMatrix& other) : Base(other.rows()) {
        if constexpr (Size_ == Dynamic) { m_.resize(other.rows(), other.cols()); }
        using assignment = typename Base::assignment_executor;
        assignment::run(*this, other);
    }
    template <typename RhsXprType_>
    constexpr OrthogonalMatrix(const OrthogonalMatrixExpr<RhsXprType_>& rhs) : Base(rhs.rows()) {
        if constexpr (Size_ == Dynamic) { m_.resize(rhs.rows(), rhs.cols()); }
        using assignment = typename Base::assignment_executor;
        assignment::run(*this, rhs.derived());
    }
    constexpr OrthogonalMatrix(const std::vector<Scalar>& data, bool orthogonalize = false) :
        Base(fdapde::sqrt(static_cast<double>(data.size()))) {
        m_ = StorageType(data);
        if (orthogonalize) {
            Base::orthogonalize_(m_);
        } else {
            fdapde_assert(
              almost_equal(m_ * m_.transpose() FDAPDE_COMMA Identity(m_.rows()) FDAPDE_COMMA 1e-14));
        }
    }
    template <std::size_t RhsSize>
    constexpr OrthogonalMatrix(const Scalar (&data)[RhsSize], bool orthogonalize = false) :
        Base(fdapde::sqrt(static_cast<double>(RhsSize))) {
        fdapde_static_assert(Size_ != Dynamic && StorageSize == RhsSize, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        m_ = StorageType(data);
        if (orthogonalize) {
            Base::orthogonalize_(m_);
        } else {
            fdapde_assert(
              almost_equal(m_ * m_.transpose() FDAPDE_COMMA Identity(m_.rows()) FDAPDE_COMMA 1e-14));
        }
    }
    // named constructors
    static constexpr auto Identity() { return IdentityMatrix<Rows, Cols>(); }
    static constexpr auto Identity(int size) { return IdentityMatrix<Dynamic, Dynamic>(size, size); }
    // data pointers
    constexpr const StorageType& data() const { return m_; }
    constexpr StorageType& data() { return m_; }
   private:
    StorageType m_;
};

namespace internals {

// class used by the product operation to achieve closure wrt group operation. internal usage only
template <typename OrthogonalXprType>
struct orthogonal_wrapper : OrthogonalMatrixExpr<orthogonal_wrapper<OrthogonalXprType>> {
    using Base = OrthogonalMatrixExpr<orthogonal_wrapper<OrthogonalXprType>>;
    using OrthogonalXprTypeNested = internals::ref_select_t<const OrthogonalXprType>;
    using Scalar = typename OrthogonalXprType::Scalar;
  
    template <typename XprType>
        requires(std::is_constructible_v<OrthogonalXprTypeNested, XprType>)
    constexpr orthogonal_wrapper(XprType&& xpr) : Base(), xpr_(std::forward<XprType>(xpr)) { }
    constexpr Scalar operator()(int i, int j) const {
        fdapde_assert(i >= 0 && i < xpr_.rows() && j >= 0 && j < xpr_.cols());
        return xpr_(i, j);
    }
   private:
    OrthogonalXprTypeNested xpr_;
};

}   // namespace internals

// orthogonal group operation
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator*(const OrthogonalMatrixExpr<LhsXprType>& lhs, const OrthogonalMatrixExpr<RhsXprType>& rhs) {
    return internals::orthogonal_wrapper<decltype(lhs * rhs)>(lhs * rhs);
}
// any other operation doesn't preserve orthogonality. A raw MatrixExpr is returned

// orthogonal view of an existing block of data
template <typename Scalar_, int Rows_, int StorageOrder_ = RowMajor>
class OrthogonalMatrixView : public OrthogonalMatrixExpr<OrthogonalMatrixView<Scalar_, Rows_>> {
   public:
    using Base = OrthogonalMatrixExpr<OrthogonalMatrixView<Scalar_, Rows_>>;
    using Scalar = Scalar_;
    using StorageType = MatrixView<Scalar_, Rows_, Rows_, StorageOrder_>;
    static constexpr int ReadOnly = std::is_const_v<Scalar_> ? 1 : 0;
    static constexpr int NestAsRef = 0;
  
    // constructors
    constexpr OrthogonalMatrixView() : Base(), m_() { }
    constexpr explicit OrthogonalMatrixView(Scalar* data) : Base(), m_(data) {
        fdapde_static_assert(Rows_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_DIAGONAL_VIEWS_ONLY);
        fdapde_assert(almost_equal(m_ * m_.transpose() FDAPDE_COMMA Identity(m_.rows()) FDAPDE_COMMA 1e-14));
    }
    constexpr OrthogonalMatrixView(Scalar* data, int size) : Base(size), m_(data) {
        fdapde_assert(size > 0);
        fdapde_assert(almost_equal(m_ * m_.transpose() FDAPDE_COMMA Identity(m_.rows()) FDAPDE_COMMA 1e-14));
    }
    // data pointers
    constexpr const StorageType& data() const { return m_; }
    constexpr StorageType& data() { return m_; }
   private:
    StorageType m_;
};

}   // namespace fdapde

#endif   // _FDAPDE_LINALG_ORTHOGONAL_H__
