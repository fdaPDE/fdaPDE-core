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

template <typename XprType_, typename MetricType_> struct SPDMatrixExpr;

namespace internals {

// class wrapping a generic expression to the expression of an SPD matrix. internal usage only
template <typename SPDXprType_, typename MetricType_> struct spd_wrapper;

// log-euclidean specialization. It is assumed that the expression SPDXprType_ already lies in the tangent space
template <typename SPDXprType_>
struct spd_wrapper<SPDXprType_, log_euclidean> :
    public SPDMatrixExpr<spd_wrapper<SPDXprType_, log_euclidean>, log_euclidean> {
   private:
    using Base = SPDMatrixExpr<spd_wrapper<SPDXprType_, log_euclidean>, log_euclidean>;
    using XprType = std::decay_t<SPDXprType_>;
    using XprTypeNested = internals::ref_select_t<SPDXprType_>;
   public:
    using Scalar = typename XprType::Scalar;
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    static constexpr int StorageOrder = XprType::StorageOrder;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;

    template <typename XprType__>
        requires(std::is_constructible_v<XprTypeNested, XprType__>)
    constexpr spd_wrapper(XprType__&& xpr) : Base(), xpr_(std::forward<XprType__>(xpr)) { }
    constexpr Scalar operator()(int i, int j) const {
        fdapde_assert(i >= 0 && i < xpr_.rows() && j >= 0 && j < xpr_.cols());
	return xpr_(i, j);
    }
    constexpr auto log() const { return xpr_; }
   private:
    XprTypeNested xpr_;
};

// helper cast function
template <typename XprType_, typename MetricType_> auto spd_cast(XprType_&& xpr) {
    return spd_wrapper<XprType_, MetricType_>(xpr);
}

}   // namespace internals

// SPD, log-euclidean TS
template <typename XprType_> struct SPDMatrixExpr<XprType_, log_euclidean> : public SymmetricMatrixExpr<XprType_> {
    using XprType = std::decay_t<XprType_>;
    // make derived() point to innermost type
    constexpr const XprType& derived() const { return static_cast<const XprType&>(*this); }
    constexpr XprType& derived() { return static_cast<XprType&>(*this); }


    // ostream. output the matrix exponential to let the user see the matrix in the SPD domain
    friend std::ostream& operator<<(std::ostream& os, const SPDMatrixExpr& m) {
        // evaluate exponential once
        SPDMatrix<typename SPDMatrixExpr::Scalar, SPDMatrixExpr::Rows, SPDMatrixExpr::Cols, log_euclidean> spd(m);
        os << spd;
        return os;
    }

    constexpr auto log() const { return derived().log(); }

    // disable direct access for SPD expressions
    constexpr auto operator()(int i, int j) const = delete;
    constexpr auto operator()(int i, int j) = delete;
};

template <typename Scalar_, int Rows_, int Cols_, typename MetricType_, int StorageOrder_, typename SPDMatrixType_>
struct SPDMatrixBase : public SPDMatrixExpr<SPDMatrixType_, MetricType_> {
   private:
    using Base = SPDMatrixExpr<SPDMatrixType_, MetricType_>;
    using Base::derived;
   public:
    using Scalar = Scalar_;
    using MetricType = std::decay_t<MetricType_>;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int ReadOnly = std::is_const_v<Scalar_>;
    using assignment_executor = internals::triangular_assignment_executor;   // ------------- da checkkare

    constexpr SPDMatrixBase() = default;
    // copy assignment
    constexpr SPDMatrixType_& operator=(const SPDMatrixType_& other) {   // ----------- da checkkare
        fdapde_static_assert(ReadOnly == 0, ASSIGNMENT_TO_READ_ONLY_LOCATION);
        if (this == std::addressof(other)) { return derived(); }
        if constexpr (Rows_ == Dynamic || Cols_ == Dynamic) {
            if (derived().rows() != other.rows() || derived().cols() != other.cols()) {
                derived().storage().resize(other.rows(), other.cols());
            }
        }
        assignment_executor::run(*this, other, [](auto&& l, const auto& r) { l = r; });
        return derived();
    }
    // only read access allowed (write access could break SPD invariant)
    constexpr auto operator()(int i, int j) const {
        fdapde_assert(i >= 0 && i < derived().rows() && j >= 0 && j < derived().cols());
        return derived().storage()(i, j);
    }
    // computes matrix logarithm
    constexpr SymmetricMatrix<Scalar, Rows, Cols> log() const {
        // extract eigenvalues' logarithm
        Vector<Scalar, Rows> log_eigval = derived().evd().eigenvalues().cwise().log();
        return (derived().evd().eigenvectors() * log_eigval.as_diagonal() * derived().evd().eigenvectors().transpose())
          .template as_symmetric<Lower>();
    }
};

template <typename Scalar_, int Rows_, int Cols_, typename Metric_, int StorageOrder_> struct SPDMatrix;

template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_>
struct SPDMatrix<Scalar_, Rows_, Cols_, log_euclidean, StorageOrder_> :
    public SPDMatrixBase<
      Scalar_, Rows_, Cols_, log_euclidean, StorageOrder_,
      SPDMatrix<Scalar_, Rows_, Cols_, log_euclidean, StorageOrder_>> {
    fdapde_static_assert(
      Rows_ == Dynamic || Cols_ == Dynamic || Rows_ == Cols_, THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
   private:
    using Base = SPDMatrixBase<
      Scalar_, Rows_, Cols_, log_euclidean, StorageOrder_,
      SPDMatrix<Scalar_, Rows_, Cols_, log_euclidean, StorageOrder_>>;
    using StorageType = SymmetricMatrix<Scalar_, Rows_, Cols_, StorageOrder_>;
   public:
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int NestAsRef = 1;
    static constexpr int ViewMode = StorageType::ViewMode;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int ReadOnly = std::is_const_v<Scalar_>;
    using assignment_executor = typename StorageType::assignment_executor;

    // empty spd matrices are ill-formed by definition
    constexpr SPDMatrix() = delete;
    constexpr SPDMatrix(int rows, int cols) = delete;
    // copy semantic
    constexpr SPDMatrix(const SPDMatrix& other) : Base(), log_data_(other.log_data_), data_(other.data_) { }
    constexpr SPDMatrix& operator=(const SPDMatrix& rhs) {
        log_data_ = rhs.log_data_;
        data_ = rhs.data_;
        return *this;
    }
    // assume MatrixExpr encoding an SPD matrix
    template <typename RhsXprType_>
    constexpr SPDMatrix(const MatrixExpr<RhsXprType_>& rhs, internals::unchecked_t) :
        Base(), data_(rhs.template as_symmetric<Lower>()) {
        EVD<StorageType> evd(data_);
        log_(evd, log_data_);
    }
    template <typename RhsXprType_>
    constexpr SPDMatrix(const MatrixExpr<RhsXprType_>& rhs, internals::checked_t) :
        Base(), data_(rhs.template as_symmetric<Lower>()) {
        // assert spd property
        fdapde_assert(almost_equal(data_ FDAPDE_COMMA data_.transpose() FDAPDE_COMMA 1e-14));
        EVD<StorageType> evd(data_);
        fdapde_assert(
          std::all_of(evd.eigenvalues().begin() FDAPDE_COMMA evd.eigenvalues().end() FDAPDE_COMMA [](double e) {
              return e > 0;
          }));
        log_(evd, log_data_);
    }

    // symmetric input, already in log-euclidean domain
    template <typename RhsXprType_>
    constexpr SPDMatrix(const SymmetricMatrixExpr<RhsXprType_>& rhs) : Base(), log_data_(rhs) {
        EVD<StorageType> evd(rhs);
        exp_(evd, data_);
    }
    template <typename RhsXprType_>
    constexpr SPDMatrix(const SPDMatrixExpr<RhsXprType_, log_euclidean>& rhs) : Base(), log_data_(rhs) {
        EVD<StorageType> evd(rhs);      
        exp_(evd, data_);
    }
    template <typename RhsXprType_> constexpr SPDMatrix& operator=(SPDMatrixExpr<RhsXprType_, log_euclidean>& rhs) {
        log_data_ = rhs;
        EVD<StorageType> evd(rhs);
        exp_(evd, data_);
        return *this;
    }
    // constructors taking external data
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr SPDMatrix(const std::vector<Scalar__>& data, internals::checked_t) : Base(), data_(data) {
        // assert spd property
        fdapde_assert(almost_equal(data_ FDAPDE_COMMA data_.transpose() FDAPDE_COMMA 1e-14));
        EVD<StorageType> evd(data_);
        fdapde_assert(
          std::all_of(evd.eigenvalues().begin() FDAPDE_COMMA evd.eigenvalues().end() FDAPDE_COMMA [](double e) {
              return e > 0;
          }));
        log_(evd, log_data_);
    }
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr SPDMatrix(const std::vector<Scalar__>& data, internals::unchecked_t) : Base(), data_(data) {
        EVD<StorageType> evd(data_);
        log_(evd, log_data_);
    }
    template <typename Scalar__, std::size_t Size>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr SPDMatrix(const Scalar__ (&data)[Size], internals::unchecked_t) : Base(), data_(data) {
        fdapde_static_assert(Rows_ != Dynamic && Cols_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        EVD<StorageType> evd(data_);
        log_(evd, log_data_);
    }
    template <typename Scalar__, std::size_t Size>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr SPDMatrix(const Scalar__ (&data)[Size], internals::checked_t) : Base(), data_(data) {
        fdapde_static_assert(Rows_ != Dynamic && Cols_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        // assert spd property
        fdapde_assert(almost_equal(data_ FDAPDE_COMMA data_.transpose() FDAPDE_COMMA 1e-14));
        EVD<StorageType> evd(data_);
        fdapde_assert(
          std::all_of(evd.eigenvalues().begin() FDAPDE_COMMA evd.eigenvalues().end() FDAPDE_COMMA [](double e) {
              return e > 0;
          }));
        log_(evd, log_data_);
    }
    // observers
    constexpr int rows() const { return log_data_.rows(); }
    constexpr int cols() const { return log_data_.cols(); }
    // ostream. output the matrix exponential to let the user see the matrix in the SPD domain
    friend std::ostream& operator<<(std::ostream& os, const SPDMatrix& m) {
        os << m.data_;
        return os;
    }

    constexpr const StorageType& storage() const { return data_; }
    constexpr StorageType& storage() { return data_; }

    // here we need a log() to access to its log-representation
    constexpr const StorageType& log() const { return log_data_; }

    constexpr Matrix<Scalar, Rows, Cols, StorageOrder> as_matrix() const { return data_; }

    // data pointers
  // constexpr const StorageType& data() const { return data_; } // should return pointer to Scalar
  //   constexpr StorageType& data() { return data_; }
   protected:
    // compute log of XprType_ and store it in dst
    template <typename XprType_, typename DstType_> constexpr void log_(const EVD<XprType_>& evd, DstType_& dst) const {
        Vector<Scalar, Rows> lev = evd.eigenvalues().cwise().log();
        dst = (evd.eigenvectors() * lev.as_diagonal() * evd.eigenvectors().transpose()).template as_symmetric<Lower>();
    }
    // compute exp of XprType_ and store it in dst
    template <typename XprType_, typename DstType_> constexpr void exp_(const EVD<XprType_>& evd, DstType_& dst) const {
        Vector<Scalar, Rows> eev = evd.eigenvalues().cwise().exp();
        dst = (evd.eigenvectors() * eev.as_diagonal() * evd.eigenvectors().transpose()).template as_symmetric<Lower>();
    }

    StorageType data_;       // matrix in the SPD domain
    StorageType log_data_;   // matrix in the log domain
};

// log-euclidean arithmetic
template <typename LhsXprType, typename RhsXprType>
constexpr auto
operator+(const SPDMatrixExpr<LhsXprType, log_euclidean>& lhs, const SPDMatrixExpr<RhsXprType, log_euclidean>& rhs) {
    return internals::spd_cast<log_euclidean>(lhs.log() + rhs.log());
}
template <typename LhsXprType, typename RhsXprType>
constexpr auto
operator-(const SPDMatrixExpr<LhsXprType, log_euclidean>& lhs, const SPDMatrixExpr<RhsXprType, log_euclidean>& rhs) {
    return internals::spd_cast<log_euclidean>(lhs.log() - rhs.log());
}
template <typename XprType, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator*(const SPDMatrixExpr<XprType, log_euclidean>& lhs, ScalarType rhs) {
    return internals::spd_cast<log_euclidean>(lhs.log() * rhs);
}
template <typename XprType, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator*(ScalarType lhs, const SPDMatrixExpr<XprType, log_euclidean>& rhs) {
    return internals::spd_cast<log_euclidean>(lhs * rhs.log());
}
template <typename XprType, typename ScalarType>
    requires(std::is_arithmetic_v<ScalarType>)
constexpr auto operator/(const SPDMatrixExpr<XprType, log_euclidean>& lhs, ScalarType rhs) {
    return internals::spd_cast<log_euclidean>(lhs.log() / rhs);
}
  
  
template <typename Scalar_, int Rows_, int Cols_, typename Metric_, int StorageOrder_> class SPDMatrixView;

// spd view of an existing block of data
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_>
class SPDMatrixView<Scalar_, Rows_, Cols_, log_euclidean, StorageOrder_> :
    public SPDMatrixBase<
      Scalar_, Rows_, Cols_, log_euclidean, StorageOrder_,
      SPDMatrixView<Scalar_, Rows_, Cols_, log_euclidean, StorageOrder_>> {
    fdapde_static_assert(
      Rows_ == Dynamic || Cols_ == Dynamic || Rows_ == Cols_, THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
    using Base = SPDMatrixBase<
      Scalar_, Rows_, Cols_, log_euclidean, StorageOrder_,
      SPDMatrixView<Scalar_, Rows_, Cols_, log_euclidean, StorageOrder_>>;
    using StorageType = SymmetricMatrixView<Scalar_, Rows_, Cols_, StorageOrder_>;
   public:
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = std::is_const_v<Scalar_>;
    using assignment_executor = typename StorageType::assignment_executor;

    // constructors
    constexpr SPDMatrixView() = delete;
    // constexpr SPDMatrixView(Scalar* data, internals::spd_unchecked_t) : Base(), m_(data) {
    //     fdapde_static_assert(Rows_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_VIEWS_ONLY);
    // }
    // constexpr SPDMatrixView(Scalar* data, internals::checked_t) : SPDMatrixView(data, unchecked) {
    //     this->assert_spd_();
    // }
    // constexpr SPDMatrixView(Scalar* data, int size, internals::unchecked_t) : Base(size), m_(data) {
    //     fdapde_assert(size > 0);
    // }
    // constexpr SPDMatrixView(Scalar* data, int size, internals::checked_t) :
    //     SPDMatrixView(data, size, spd_unchecked) {
    //     this->assert_spd_();
    // }
    // data pointers
    constexpr const StorageType& data() const { return m_; }
    constexpr StorageType& data() { return m_; }
   private:
    StorageType m_;
};

// detection trait
template <typename XprType> struct is_spd_matrix {
    static constexpr bool value = std::is_base_of_v<SPDMatrixExpr<std::decay_t<XprType>, log_euclidean>, XprType>;
};
template <typename XprType> static constexpr bool is_spd_matrix_v = is_spd_matrix<XprType>::value;
  
}   // namespace fdapde

#endif   // __FDAPDE_LINALG_SPD_H__
