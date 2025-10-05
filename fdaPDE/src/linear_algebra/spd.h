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

#ifndef __FDAPDE_LINALG_SPD_H__
#define __FDAPDE_LINALG_SPD_H__

#include "header_check.h"

namespace fdapde {

// we let the Symmetric Positive Definite TS specialize the Symmetric TS. Methods which apply to Symmetric TS also apply
// to the more constrained SPD TS
template <int Rows_, int Cols_, typename XprType_>
struct SPDMatrixExpr : public SymmetricMatrixExpr<Rows_, Cols_, XprType_> {
    fdapde_static_assert(Rows_ == Cols_, THIS_TYPE_SYSTEM_IS_FOR_SQUARE_MATRICES_ONLY);
    using Base = SymmetricMatrixExpr<Rows_, Cols_, XprType_>;
    using Base::derived;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
};

template <typename Scalar_, int Size_, typename SPDMatrixType_>
struct SPDMatrixBase : public SPDMatrixExpr<Size_, Size_, SPDMatrixType_> {
    using Base = SPDMatrixExpr<Size_, Size_, SPDMatrixType_>;
    using Base::derived;
    using Scalar = Scalar_;
    static constexpr int Rows = Size_;
    static constexpr int Cols = Size_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = std::is_const_v<Scalar_> ? 1 : 0;
    struct assignment_executor {
        template <typename SrcXprType> static constexpr void run(SPDMatrixType_& dst, const SrcXprType& src) {
            fdapde_assert(dst.rows() == src.rows() && dst.cols() == src.cols());
            int rows_ = dst.rows();
            int cols_ = dst.cols();
            for (int i = 0; i < rows_; ++i) {
                for (int j = 0; j < cols_; ++j) { dst.data()(i, j) = src(i, j); }
            }
            return;
        }
    };

    constexpr SPDMatrixBase() = delete;
    constexpr SPDMatrixBase(int size) : size_(Size_ == Dynamic ? size_ : Size_) { }
    // copy assignment
    constexpr SPDMatrixType_& operator=(const SPDMatrixType_& other) {
        fdapde_static_assert(ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION);
        if constexpr (Size_ == Dynamic) { fdapde_assert(size_ == other.rows() && size_ == other.cols()); }
        if (this == std::addressof(other)) { return derived(); }
        assignment_executor::run(*this, other);
        return derived();
    }
    // only read access allowed (write access could break SPD invariant)
    constexpr const Scalar& operator()(int i, int j) const {
        fdapde_assert(i >= 0 && i < size_ && j >= 0 && j < size_);
        return derived().data()(i, j);
    }
    // observers
    constexpr int rows() const { return derived().data().rows(); }
    constexpr int cols() const { return derived().data().cols(); }
    constexpr const EVD<Scalar, Size_>& evd() const {
        if (!evd_) {
            evd_.emplace();
            evd_->compute(derived());
        }
        return *evd_;
    }
   protected:
    // compute EVD and guarantees positive definiteness invariant
    void assert_spd_() {
        // check symmetry
        fdapde_assert(almost_equal(m FDAPDE_COMMA m.transpose() FDAPDE_COMMA 1e-14));
        // check positive definiteness
        bool is_positive_definite =
          std::all_of(evd().eigenvalues().begin(), evd().eigenvalues().end(), [](double e) { return e > 0; });
        fdapde_assert(is_positive_definite);
        return;
    }
    mutable std::optional<EVD<Scalar, Size_>> evd_;
    int size_;
};

// any arithmetic operation doesn't preserve in general the spd invariant. Specifically, the TS guarantees
//  - A SymmetricMatrixExpr is for all linear operations
//  - A MatrixExpr is for the matrix product

namespace internals {

// tag types to enable/disable costly O(n^2) SPD check
struct spd_checked_t { };
struct spd_unchecked_t { };

}   // namespace internals

inline constexpr internals::spd_checked_t   spd_checked   {};
inline constexpr internals::spd_unchecked_t spd_unchecked {};
  
// A symmetric matrix with enforced positive definiteness check
template <typename Scalar_, int Size_>
struct SPDMatrix : public SPDMatrixBase<Scalar_, Size_, SPDMatrix<Scalar_, Size_>> {
    using Base = SPDMatrixBase<Scalar_, Size_, SPDMatrix<Scalar_, Size_>>;
    using Scalar = Scalar_;
    using StorageType = SymmetricMatrix<Scalar_, Size_>;
    static constexpr int StorageSize = StorageType::StorageSize;
    static constexpr int Rows = Size_;
    static constexpr int Cols = Size_;
    static constexpr int NestAsRef = 1;
    static constexpr int ReadOnly = std::is_const_v<Scalar_> ? 1 : 0;
    friend Base;

    // empty orthogonal matrices are ill-formed by definition
    constexpr SPDMatrix() = delete;
    constexpr SPDMatrix(int rows, int cols) = delete;
    constexpr SPDMatrix(const SPDMatrix& other) : Base(other.rows()) {
        if constexpr (Size_ == Dynamic) { m_.resize(other.rows(), other.cols()); }
        using assignment = typename Base::assignment_executor;
        assignment::run(*this, other);
    }
    template <int RhsRows_, int RhsCols_, typename RhsXprType_>
    constexpr SPDMatrix(const SPDMatrixExpr<RhsRows_, RhsCols_, RhsXprType_>& rhs) : Base(RhsRows_) {
        if constexpr (Size_ == Dynamic) { m_.resize(rhs.rows(), rhs.cols()); }
        using assignment = typename Base::assignment_executor;
        assignment::run(*this, rhs.derived());
    }
    template <typename DataT>
        requires(internals::is_vector_like_v<DataT> && !internals::is_matrix_like_v<DataT>)
    constexpr explicit SPDMatrix(DataT&& data, internals::spd_unchecked_t) :
        Base(fdapde::sqrt(static_cast<double>(data.size()))) {
        m_ = StorageType(data);
    }
    template <typename DataT>
        requires(internals::is_vector_like_v<DataT> && !internals::is_matrix_like_v<DataT>)
    constexpr explicit SPDMatrix(DataT&& data, internals::spd_checked_t) : SPDMatrix(data, spd_unchecked) {
        Base::assert_spd_();
    }
    template <std::size_t RhsSize>
    constexpr explicit SPDMatrix(const Scalar (&data)[RhsSize], internals::spd_unchecked_t) :
        Base(fdapde::sqrt(static_cast<double>(RhsSize))) {
        fdapde_static_assert(Size_ != Dynamic && StorageSize == RhsSize, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        m_ = StorageType(data);
    }
    template <std::size_t RhsSize>
    constexpr explicit SPDMatrix(const Scalar (&data)[RhsSize], internals::spd_checked_t) :
        SPDMatrix(data, spd_unchecked) {
        Base::assert_spd_();
    }
    // data pointers
    constexpr const StorageType& data() const { return m_; }
    constexpr StorageType& data() { return m_; }
   protected:
    StorageType m_;
};

// spd view of an existing block of data
template <typename Scalar_, int Rows_>
class SPDMatrixView : public SPDMatrixExpr<Rows_, Rows_, SPDMatrixView<Scalar_, Rows_>> {
   public:
    using Base = SPDMatrixExpr<Rows_, Rows_, SPDMatrixView<Scalar_, Rows_>>;
    using Scalar = Scalar_;
    using StorageType = SymmetricMatrixView<Scalar_, Rows_>;
    static constexpr int ReadOnly = std::is_const_v<Scalar_> ? 1 : 0;
    static constexpr int NestAsRef = 0;
  
    // constructors
    constexpr SPDMatrixView() = delete;
    constexpr SPDMatrixView(Scalar* data, internals::spd_unchecked_t) : Base(), m_(data) {
        fdapde_static_assert(Rows_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_VIEWS_ONLY);
    }
    constexpr SPDMatrixView(Scalar* data, internals::spd_checked_t) : SPDMatrixView(data, spd_unchecked) {
        Base::assert_spd_();
    }
    constexpr SPDMatrixView(Scalar* data, int size, internals::spd_unchecked_t) : Base(size), m_(data) {
        fdapde_assert(size > 0);
    }
    constexpr SPDMatrixView(Scalar* data, int size, internals::spd_checked_t) :
        SPDMatrixView(data, size, spd_unchecked) {
        Base::assert_spd_();
    }
    // data pointers
    constexpr const StorageType& data() const { return m_; }
    constexpr StorageType& data() { return m_; }
   private:
    StorageType m_;
};

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_SPD_H__
