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

template <typename XprType> class EVD;
template <typename Scalar_, int Rows_, int Cols_, typename Metric_, int StorageOrder_ = RowMajor> struct SPDMatrix;
struct log_euclidean { };
  
// symmetric matrix type system
template <typename XprType> struct SymmetricMatrixExpr;
  
namespace internals {

// class wrapping a generic expression to the expression of a symmetric matrix. internal usage only
template <int ViewMode_, typename SymmetricXprType_>
struct symmetric_wrapper : public SymmetricMatrixExpr<symmetric_wrapper<ViewMode_, SymmetricXprType_>> {
   private:
    fdapde_static_assert(ViewMode_ == Lower || ViewMode_ == Upper, VIEW_MODE_MUST_BE_EITHER_LOWER_OR_UPPER);
    using Base = SymmetricMatrixExpr<symmetric_wrapper<ViewMode_, SymmetricXprType_>>;
    using XprType = std::decay_t<SymmetricXprType_>;
    using XprTypeNested = internals::ref_select_t<SymmetricXprType_>;
    static constexpr int ViewMode = ViewMode_;
   public:
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    static constexpr int StorageOrder = XprType::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename XprType__>
        requires(std::is_constructible_v<XprTypeNested, XprType__>)
    constexpr symmetric_wrapper(XprType__&& xpr) : Base(), xpr_(std::forward<XprType__>(xpr)) { }
    constexpr Scalar operator()(int i, int j) const {
        fdapde_assert(i >= 0 && i < xpr_.rows() && j >= 0 && j < xpr_.cols());
        if constexpr (ViewMode == Upper) { return i > j ? xpr_(j, i) : xpr_(i, j); }
        if constexpr (ViewMode == Lower) { return i < j ? xpr_(j, i) : xpr_(i, j); }
    }
    constexpr const XprTypeNested& rep() const { return xpr_; }
   private:
    XprTypeNested xpr_;
};

// helper cast function
template <int ViewMode_, typename XprType_> auto symmetric_cast(XprType_&& xpr) {
    return symmetric_wrapper<ViewMode_, XprType_>(xpr);
}

}   // namespace internals

template <typename XprType_> struct SymmetricMatrixExpr : public MatrixExpr<XprType_> {
    using XprType = std::decay_t<XprType_>;
    // make derived() point to innermost type
    constexpr const XprType& derived() const { return static_cast<const XprType&>(*this); }
    constexpr XprType& derived() { return static_cast<XprType&>(*this); }
    // inherit assignment from base
    using MatrixExpr<XprType_>::operator=;

    auto evd() const { return EVD<XprType>(derived()); }
    // compute matrix exponential, returns SPD matrix bound to MetricType
    template <typename MetricType> constexpr auto exp() const {
        using Scalar = typename XprType::Scalar;
        constexpr int Rows = XprType::Rows;
        constexpr int Cols = XprType::Cols;
        // symmetric matrices are the tangent space to the SPD cone under the log-euclidean metric. avoid to compute a
        // matrix exp (from symm to spd) followed by a matrix log (from spd back to symm)
        if constexpr (std::is_same_v<std::decay_t<MetricType>, log_euclidean>) {
            return SPDMatrix<Scalar, Rows, Cols, log_euclidean>(derived());
        } else {
            // generic fallback
            EVD<XprType> evd_(derived());
            Vector<Scalar, Rows> exp_eigval = evd_.eigenvalues().cwise().exp();   // extract eigenvalues' exponential
            return SPDMatrix<double, Rows, Cols, MetricType>(
              evd_.eigenvectors() * exp_eigval.as_diagonal() * evd_.eigenvectors().transpose(), fdapde::unchecked);
        }
    }

    // internal triangular matrix representation
    constexpr decltype(auto) rep() const { return derived().rep(); }
    constexpr decltype(auto) rep() { return derived().rep(); }
};

// base class for symmetric matrices
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

    template <typename Scalar__>
        requires(std::is_same_v<Scalar, std::decay_t<Scalar__>>)
    struct symmetric_proxy {
        using Scalar = Scalar__;

        constexpr symmetric_proxy(Scalar__* data, int i, int j, int size) :
            data_(data), index_(compute_linear_index_(i < j ? j : i, i < j ? i : j, size)) { }
        template <typename T>
            requires(std::is_convertible_v<T, Scalar>)
        constexpr symmetric_proxy& operator=(T value) {
            data_[index_] = value;
            return *this;
        }
        constexpr operator Scalar() { return data_[index_]; }
        constexpr operator Scalar() const { return data_[index_]; }
       private:
        constexpr int compute_linear_index_(int i, int j, [[maybe_unused]] int size) const {
            if constexpr (StorageOrder == RowMajor) { return i * (i + 1) / 2 + j; }
            if constexpr (StorageOrder == ColMajor) { return j * (2 * size - j + 1) / 2 + (i - j); }
        }
        Scalar* data_;
        int index_;
    };
    using reference = symmetric_proxy<Scalar>;
    using const_reference = symmetric_proxy<const Scalar>;

    constexpr SymmetricMatrixBase() = default;
    // access
    constexpr auto operator()(int i, int j) const {
        fdapde_assert(i >= 0 && i < derived().rows() && j >= 0 && j < derived().cols());
        return const_reference(derived().data(), i, j, derived().rows());
    }
    constexpr auto operator()(int i, int j) {
        fdapde_static_assert(ReadOnly == 0, WRITE_ACCESS_TO_READ_ONLY_LOCATION);
        fdapde_assert(i >= 0 && i < derived().rows() && j >= 0 && j < derived().cols());
        return reference(derived().data(), i, j, derived().rows());
    }
};

// symmetric matrices vector-space structure (additive group)
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator+(const SymmetricMatrixExpr<LhsXprType>& lhs, const SymmetricMatrixExpr<RhsXprType>& rhs) {
    return internals::symmetric_cast<Lower>(lhs.rep() + rhs.rep());
}
template <typename LhsXprType, typename RhsXprType>
constexpr auto operator-(const SymmetricMatrixExpr<LhsXprType>& lhs, const SymmetricMatrixExpr<RhsXprType>& rhs) {
    return internals::symmetric_cast<Lower>(lhs.rep() - rhs.rep());
}
template <typename XprType, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator*(const SymmetricMatrixExpr<XprType>& lhs, ScalarType rhs) {
    return internals::symmetric_cast<Lower>(lhs.rep() * rhs);
}
template <typename XprType, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator*(ScalarType lhs, const SymmetricMatrixExpr<XprType>& rhs) {
    return internals::symmetric_cast<Lower>(lhs * rhs.rep());
}
template <typename XprType, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator/(const SymmetricMatrixExpr<XprType>& lhs, ScalarType rhs) {
    return internals::symmetric_cast<Lower>(lhs.rep() / rhs);
}
// any other operation doesn't preserve symmetry. A raw MatrixExpr is returned

// owning storage symmetric matrix
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_ = RowMajor>
struct SymmetricMatrix :
    public SymmetricMatrixBase<
      Scalar_, Rows_, Cols_, StorageOrder_, SymmetricMatrix<Scalar_, Rows_, Cols_, StorageOrder_>> {
    fdapde_static_assert(
      Rows_ == Dynamic || Cols_ == Dynamic || Rows_ == Cols_, THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
   private:
    using Base =
      SymmetricMatrixBase<Scalar_, Rows_, Cols_, StorageOrder_, SymmetricMatrix<Scalar_, Rows_, Cols_, StorageOrder_>>;
    using StorageType = TriangularMatrix<Scalar_, Rows_, Cols_, Lower>;
   public:
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int NestAsRef = 0;
    static constexpr int ViewMode = StorageType::ViewMode;
    static constexpr int StorageOrder = StorageType::StorageOrder;
    static constexpr int ReadOnly = std::is_const_v<Scalar_>;
    using assignment_executor = typename StorageType::assignment_executor;

    constexpr SymmetricMatrix() : Base() { }
    // copy semantic
    constexpr SymmetricMatrix(const SymmetricMatrix& rhs) : Base() { data_ = rhs.rep(); }
    constexpr SymmetricMatrix& operator=(const SymmetricMatrix& rhs) {
        this->rows_ = rhs.rows();
        this->cols_ = rhs.cols();
        data_ = rhs.rep();
        return *this;
    }

    constexpr explicit SymmetricMatrix(int rows, int cols) : Base(), data_(rows, cols) {
        fdapde_static_assert(Rows == Dynamic || Cols == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_MATRICES_ONLY);
        fdapde_assert(rows > 0 && cols > 0 && rows == cols);
    }
    template <typename RhsXprType_>
    constexpr SymmetricMatrix(const SymmetricMatrixExpr<RhsXprType_>& rhs) : data_(rhs) { }
    template <typename RhsXprType_> constexpr SymmetricMatrix& operator=(const SymmetricMatrixExpr<RhsXprType_>& rhs) {
        data_ = rhs;
        return *this;
    }
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr explicit SymmetricMatrix(const std::vector<Scalar__>& data) : data_(data) { }
    template <typename Scalar__, std::size_t Size>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr explicit SymmetricMatrix(const Scalar__ (&data)[Size]) : data_(data) {
        fdapde_static_assert(Rows_ != Dynamic && Cols_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
    }
    // modifiers
    void resize(int rows, int cols) {
        fdapde_static_assert(Rows == Dynamic || Cols == Dynamic, THIS_METHOD_IS_FOR_DYNAMIC_SIZED_MATRICES_ONLY);
        fdapde_assert(rows == cols);
        data_.resize(rows, cols);
        return;
    }
    // observers
    constexpr int rows() const { return data_.rows(); }
    constexpr int cols() const { return data_.cols(); }
    constexpr const StorageType& rep() const { return data_; }
    constexpr StorageType& rep() { return data_; }
    // data pointers
    constexpr const Scalar* data() const { return data_.data(); }
    constexpr Scalar* data() { return data_.data(); }
   private:
    StorageType data_;
};

// symmetric view of an existing block of data
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_>
class SymmetricMatrixView :
    public SymmetricMatrixBase<
      Scalar_, Rows_, Cols_, StorageOrder_, SymmetricMatrixView<Scalar_, Rows_, Cols_, StorageOrder_>> {
    fdapde_static_assert(
      Rows_ == Dynamic || Cols_ == Dynamic || Rows_ == Cols_, THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
    using Base = SymmetricMatrixBase<
      Scalar_, Rows_, Cols_, StorageOrder_, SymmetricMatrixView<Scalar_, Rows_, Cols_, StorageOrder_>>;
    using StorageType = TriangularMatrixView<Scalar_, Rows_, Cols_, Lower, StorageOrder_>;
   public:
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int NestAsRef = 1;
    static constexpr int StorageOrder = StorageType::StorageOrder;
    static constexpr int ReadOnly = std::is_const_v<Scalar_>;
    using assignment_executor = typename StorageType::assignment_executor;

    // constructors
    constexpr SymmetricMatrixView() : Base(), data_() { }
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr explicit SymmetricMatrixView(Scalar__* data) : Base(), data_(data) {
        fdapde_static_assert(Rows != Dynamic && Cols != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
    }
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr SymmetricMatrixView(Scalar__* data, int rows, int cols) : Base(rows, cols), data_(data) {
        fdapde_assert(rows > 0 && cols > 0 && rows == cols);
    }
    // observers
    constexpr int rows() const { return data_.rows(); }
    constexpr int cols() const { return data_.cols(); }
    constexpr const StorageType& rep() const { return data_; }
    constexpr StorageType& rep() { return data_; }
    // data pointers
    constexpr const Scalar* data() const { return data_.data(); }
    constexpr Scalar* data() { return data_.data(); }
   private:
    StorageType data_;
};

// detection trait
template <typename XprType> struct is_symmetric_matrix {
    static constexpr bool value = std::is_base_of_v<SymmetricMatrixExpr<std::decay_t<XprType>>, XprType>;
};
template <typename XprType> static constexpr bool is_symmetric_matrix_v = is_symmetric_matrix<XprType>::value;
  
}   // namespace fdapde

#endif // __FDAPDE_LINALG_SYMMETRIC_H__

