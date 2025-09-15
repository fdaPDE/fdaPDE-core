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

#ifndef __FDAPDE_SYMMETRIC_MATRIX_H__
#define __FDAPDE_SYMMETRIC_MATRIX_H__

#include "../header_check.h"

namespace fdapde {

// forward decl
template <typename Scalar, int Size> class EVD;

template <int Rows_, int Cols_, typename XprType_>
struct SymmetricMatrixExpr : public MatrixExpr<Rows_, Cols_, XprType_> {
    using Base = MatrixExpr<Rows_, Cols_, XprType_>;
    using Base::derived;

    // compute EVD of expression
    auto evd() const {
        using Scalar = typename XprType_::Scalar;
        EVD<Scalar, Rows_> evd(derived());
        return evd;
    }
};

namespace internals {

// class wrapping a generic expression to the expression of a symmetric matrix. internal usage only
template <int Rows_, int Cols_, int ViewMode_, typename SymmetricXprType>
struct symmetric_wrapper :
    public SymmetricMatrixExpr<Rows_, Cols_, symmetric_wrapper<Rows_, Cols_, ViewMode_, SymmetricXprType>> {
    fdapde_static_assert(Rows_ == Cols_, THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
    fdapde_static_assert(ViewMode_ == Lower || ViewMode_ == Upper, VIEW_MODE_MUST_BE_EITHER_LOWER_OR_UPPER);
    using Base = SymmetricMatrixExpr<Rows_, Cols_, symmetric_wrapper<Rows_, Cols_, ViewMode_, SymmetricXprType>>;
    using SymmetricXprTypeNested = internals::ref_select_t<const SymmetricXprType>;
    using Scalar = typename SymmetricXprType::Scalar;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int ViewMode = ViewMode_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename XprType>
        requires(std::is_constructible_v<SymmetricXprTypeNested, XprType>)
    constexpr symmetric_wrapper(XprType&& xpr) : Base(), xpr_(std::forward<XprType>(xpr)) { }
    constexpr Scalar operator()(int i, int j) const {
        fdapde_constexpr_assert(i >= 0 && i < xpr_.rows() && j >= 0 && j < xpr_.cols());
        if constexpr (ViewMode == Upper) return i > j ? xpr_(j, i) : xpr_(i, j);
        if constexpr (ViewMode == Lower) return i < j ? xpr_(i, j) : xpr_(j, i);
    }
   private:
    SymmetricXprTypeNested xpr_;
};

}   // namespace internals

template <typename XprType, typename CoeffType> struct MatrixCoeffWiseOp;

// symmetric matrices linear vector-space structure (additive group)
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator+(
  const SymmetricMatrixExpr<LhsXprType::Rows, LhsXprType::Cols, LhsXprType>& lhs,
  const SymmetricMatrixExpr<RhsXprType::Rows, RhsXprType::Cols, RhsXprType>& rhs) {
    return (lhs + rhs).template as_symmetric<Lower>();
}
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator-(
  const SymmetricMatrixExpr<LhsXprType::Rows, LhsXprType::Cols, LhsXprType>& lhs,
  const SymmetricMatrixExpr<RhsXprType::Rows, RhsXprType::Cols, RhsXprType>& rhs) {
    return (lhs - rhs).template as_symmetric<Lower>();
}
template <typename XprType, typename CoeffType>
    requires(std::is_arithmetic_v<CoeffType>)
constexpr auto operator*(const SymmetricMatrixExpr<XprType::Rows, XprType::Cols, XprType>& lhs, CoeffType rhs) {
    using Scalar = typename std::decay_t<XprType>::Scalar;
    auto op_ = [rhs](const Scalar& x) { return x * rhs; };
    return MatrixCoeffWiseOp<XprType, decltype(op_)>(lhs.derived(), op_).template as_symmetric<Lower>();
}
template <typename XprType, typename CoeffType>
    requires(std::is_arithmetic_v<CoeffType>)
constexpr auto operator*(CoeffType lhs, const SymmetricMatrixExpr<XprType::Rows, XprType::Cols, XprType>& rhs) {
    return rhs * lhs;
}
template <typename XprType, typename CoeffType>
    requires(std::is_arithmetic_v<CoeffType>)
constexpr auto operator/(const SymmetricMatrixExpr<XprType::Rows, XprType::Cols, XprType>& lhs, CoeffType rhs) {
    return (lhs * (CoeffType(1) / rhs)).template as_symmetric<Lower>();
}
// any other operation doesn't preserve symmetry. A raw MatrixExpr is returned
  
template <typename Scalar_, int Size_, typename SymmetricMatrixType>
class SymmetricMatrixBase : public SymmetricMatrixExpr<Size_, Size_, SymmetricMatrixType> {
   public:
    using Base = SymmetricMatrixExpr<Size_, Size_, SymmetricMatrixType>;
    using Base::derived;
    using Scalar = Scalar_;
    static constexpr int Rows = Size_;
    static constexpr int Cols = Size_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = std::is_const_v<Scalar_> ? 1 : 0;
    struct assignment_executor {
        template <typename SrcXprType> static constexpr void run(SymmetricMatrixType& dst, const SrcXprType& src) {
            fdapde_static_assert(SymmetricMatrixType::ReadOnly == 0, ASSIGNMENT_TO_A_READ_ONLY_EXPRESSION);
            int row = 0, col = 0;
            for (int i = 0, n = dst.rows(); i < n; ++i) {
                for (int j = 0; j <= i; ++j) { dst(i, j) = src(i, j); }
            }
        }
    };
    // struct to proxy the behaviour of a mirrored reference
    template <typename StorageType_> struct symmetric_proxy {
        using StorageType = std::add_pointer_t<StorageType_>;
        using Scalar = typename std::decay_t<StorageType_>::Scalar;

        constexpr symmetric_proxy(StorageType_& m, int i, int j) :
            row_(i < j ? j : i), col_(i < j ? i : j), m_(std::addressof(m)) { }
        // lvalue reference behaviour
        template <typename Scalar__>
            requires(std::is_convertible_v<Scalar__, Scalar>)
        constexpr symmetric_proxy& operator=(Scalar__ value) {
            m_->operator()(row_, col_) = value;
            return *this;
        }
        // implicit conversion to Scalar
        constexpr operator Scalar() { return m_->operator()(row_, col_); }
       private:
        int row_, col_;
        StorageType m_;
    };

    constexpr SymmetricMatrixBase() : Base(), size_((Size_ == Dynamic) ? 0 : Size_) { }
    constexpr explicit SymmetricMatrixBase(int size) : Base(), size_(size) { }
    // access
    constexpr auto operator()(int i, int j) const {
        fdapde_constexpr_assert(i >= 0 && i < size_ && j >= 0 && j < size_);
        return symmetric_proxy<std::add_const_t<typename SymmetricMatrixType::StorageType>>(derived().data(), i, j);
    }
    constexpr auto operator()(int i, int j) {
        fdapde_static_assert(ReadOnly == 0, WRITE_ACCESS_TO_READ_ONLY_LOCATION);
        fdapde_constexpr_assert(i >= 0 && i < size_ && j >= 0 && j < size_);
        return symmetric_proxy<typename SymmetricMatrixType::StorageType>(derived().data(), i, j);
    }
    // observers
    constexpr int rows() const { return size_; }
    constexpr int cols() const { return size_; }
   protected:
    int size_;
};

template <typename Scalar_, int Size_>
struct SymmetricMatrix : public SymmetricMatrixBase<Scalar_, Size_, SymmetricMatrix<Scalar_, Size_>> {
    using Base = SymmetricMatrixBase<Scalar_, Size_, SymmetricMatrix<Scalar_, Size_>>;
    using Scalar = Scalar_;
    using StorageType = TriangularMatrix<Scalar, Size_, Lower>;
    static constexpr int StorageSize = StorageType::StorageSize;
    static constexpr int Rows = Size_;
    static constexpr int Cols = Size_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = std::is_const_v<Scalar_> ? 1 : 0;

    constexpr SymmetricMatrix() : Base(), data_() { }
    constexpr explicit SymmetricMatrix(int size) : Base(size), data_(size) { }
    template <int Rows_, int Cols_, int ViewMode_, typename TriangularXprType_>
    constexpr SymmetricMatrix(const TriangularMatrixExpr<Rows_, Cols_, ViewMode_, TriangularXprType_>& rhs) :
        data_(rhs) { }
    template <typename DataT>
        requires(internals::is_vector_like_v<DataT> && !internals::is_matrix_like_v<DataT>)
    constexpr explicit SymmetricMatrix(DataT&& data) : Base(data.size()), data_(data) { }
    template <std::size_t RhsSize>
    constexpr explicit SymmetricMatrix(const Scalar (&data)[RhsSize]) : Base(RhsSize), data_(data) {
        fdapde_static_assert(Rows != Dynamic && Cols != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        fdapde_static_assert(StorageSize == RhsSize, INVALID_DATA_SIZE);
    }
    // static named constructors
    static constexpr SymmetricMatrix Ones() { return StorageType::Ones(); }
    static constexpr SymmetricMatrix Ones(int size) { return StorageType::Ones(size); }
    static constexpr SymmetricMatrix Zero() { return StorageType::Zero(); }
    static constexpr SymmetricMatrix Zero(int size) { return StorageType::Zero(size); }
    static constexpr SymmetricMatrix Identity() { return StorageType::Identity(); }
    static constexpr SymmetricMatrix Identity(int size) { return StorageType::Identity(size); }
    // modifiers
    void resize(int size) {
        fdapde_static_assert(Rows == Dynamic || Cols == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_MATRICES_ONLY);
	data_.resize(size);
	Base::size_ = size;
        return;
    }
    // data pointers
    constexpr const StorageType& data() const { return data_; }
    constexpr StorageType& data() { return data_; }
   private:
    StorageType data_;
};

// symmetric view of an existing block of data
template <typename Scalar_, int Rows_>
class SymmetricMatrixView :
    public SymmetricMatrixBase<Scalar_, Rows_, SymmetricMatrixView<Scalar_, Rows_>> {
   public:
    using Base = SymmetricMatrixBase<Scalar_, Rows_, SymmetricMatrixView<Scalar_, Rows_>>;
    using Scalar = Scalar_;
    using StorageType = TriangularMatrixView<Scalar_, Rows_, Lower>;
    static constexpr int ReadOnly = std::is_const_v<Scalar_> ? 1 : 0;
    static constexpr int NestAsRef = 1;
  
    // constructors
    constexpr SymmetricMatrixView() : Base(), data_(nullptr) { }
    constexpr explicit SymmetricMatrixView(Scalar* data) : Base(), data_(data) {
        fdapde_static_assert(Rows_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
    }
    constexpr SymmetricMatrixView(Scalar* data, int size) : Base(size), data_(data) {
	fdapde_constexpr_assert(size > 0);
    }
    // data pointers
    constexpr const StorageType& data() const { return data_; }
    constexpr StorageType& data() { return data_; }
   private:
    StorageType data_;
};

}   // namespace fdapde

#endif // __FDAPDE_SYMMETRIC_MATRIX_H__
