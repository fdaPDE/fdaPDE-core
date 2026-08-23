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

// Symmetric Positive Definite TS
template <typename XprType_, typename MetricType_> class SPDMatrixExpr;
namespace internals {
  
// spd matrix implementations (to be specialized wrt MetricType_)
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_, typename MetricType_, typename SPDMatrixType_>
class spd_matrix_base;
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_, typename MetricType_> class spd_matrix_impl;
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_, typename MetricType_> class spd_matrix_view_impl;
// class wrapping a generic expression to the expression of an SPD matrix. internal usage only
template <typename SPDXprType_, typename MetricType_> class spd_wrapper;
  
}   // namespace internals

// SPD, log-euclidean TS
struct log_euclidean { };   // log-euclidean sub-TS tag
  
template <typename XprType_> class SPDMatrixExpr<XprType_, log_euclidean> : public SymmetricMatrixExpr<XprType_> {
   public:
    using XprType = std::decay_t<XprType_>;
    // make derived() point to innermost type
    constexpr const XprType& derived() const { return static_cast<const XprType&>(*this); }
    constexpr XprType& derived() { return static_cast<XprType&>(*this); }

    constexpr const auto& log() const { return derived().log(); }
};
    
namespace internals {

// spd_wrapper log-euclidean specialization. It is assumed that the SPDXprType_ already lies in the tangent space (i.e.,
// is a symmetric expression)
template <typename SPDXprType_>
class spd_wrapper<SPDXprType_, log_euclidean> :
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
        if (!exp_.has_value()) {
            // compute and cache the exponential at first access (one EVD cost)
            EVD<SymmetricMatrix<Scalar, Rows, Cols>> evd(xpr_);
            Vector<Scalar, Rows> eev = evd.eigenvalues().cwise().exp();
            exp_ =
              (evd.eigenvectors() * eev.as_diagonal() * evd.eigenvectors().transpose()).template as_symmetric<Lower>();
        }
	return exp_->operator()(i, j);
    }
    constexpr const XprTypeNested& log() const { return xpr_; }
   private:
    XprTypeNested xpr_;
    mutable std::optional<SymmetricMatrix<Scalar, Rows, Cols>> exp_;
};

// helper cast function
template <typename MetricType_, typename XprType_> auto spd_cast(XprType_&& xpr) {
    return spd_wrapper<XprType_, MetricType_>(std::forward<XprType_>(xpr));
}

template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_, typename SPDMatrixType_>
class spd_matrix_base<Scalar_, Rows_, Cols_, StorageOrder_, log_euclidean, SPDMatrixType_> :
    public SPDMatrixExpr<SPDMatrixType_, log_euclidean> {
   private:
    using Base = SPDMatrixExpr<SPDMatrixType_, log_euclidean>;
    using Base::derived;
   public:
    using Scalar = Scalar_;
    using MetricType = log_euclidean;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int ReadOnly = std::is_const_v<Scalar_>;
    using assignment_executor = internals::triangular_assignment_executor;

    constexpr spd_matrix_base() = default;
    // only read access allowed (write access could break SPD invariant)
    constexpr auto operator()(int i, int j) const {
        fdapde_assert(i >= 0 && i < derived().rows() && j >= 0 && j < derived().cols());
        return derived().rep()(i, j);
    }
    constexpr auto operator()(int i, int j) = delete;
    // computes matrix logarithm
    constexpr SymmetricMatrix<Scalar, Rows, Cols> log() const {
        const auto decomposition = derived().evd();
        Vector<Scalar, Rows> log_eigval = decomposition.eigenvalues().cwise().log();
        return (decomposition.eigenvectors() * log_eigval.as_diagonal() * decomposition.eigenvectors().transpose())
          .template as_symmetric<Lower>();
    }
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
};

template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_>
class spd_matrix_impl<Scalar_, Rows_, Cols_, StorageOrder_, log_euclidean> :
    public spd_matrix_base<
      Scalar_, Rows_, Cols_, StorageOrder_, log_euclidean,
      spd_matrix_impl<Scalar_, Rows_, Cols_, StorageOrder_, log_euclidean>> {
    fdapde_static_assert(
      Rows_ == Dynamic || Cols_ == Dynamic || Rows_ == Cols_, THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
   private:
    using Base = spd_matrix_base<
      Scalar_, Rows_, Cols_, StorageOrder_, log_euclidean,
      spd_matrix_impl<Scalar_, Rows_, Cols_, StorageOrder_, log_euclidean>>;
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
    constexpr spd_matrix_impl() = delete;
    constexpr spd_matrix_impl(int rows, int cols) = delete;
    // copy semantic
    constexpr spd_matrix_impl(const spd_matrix_impl& other) : Base(), log_data_(other.log_data_), data_(other.data_) { }
    constexpr spd_matrix_impl& operator=(const spd_matrix_impl& rhs) {
        log_data_ = rhs.log_data_;
        data_ = rhs.data_;
        return *this;
    }
    // initialize from SPD expression
    template <typename RhsXprType_>
    constexpr spd_matrix_impl(const SPDMatrixExpr<RhsXprType_, log_euclidean>& rhs) : Base(), log_data_(rhs) {
        EVD<StorageType> evd(rhs);
        this->exp_(evd, data_);
    }
    template <typename RhsXprType_>
    constexpr spd_matrix_impl& operator=(SPDMatrixExpr<RhsXprType_, log_euclidean>& rhs) {
        log_data_ = rhs;
        EVD<StorageType> evd(rhs);
        this->exp_(evd, data_);
        return *this;
    }
    // initializes from symmetric expression, assume already in log-euclidean domain
    template <typename RhsXprType_>
    constexpr spd_matrix_impl(const SymmetricMatrixExpr<RhsXprType_>& rhs) : Base(), log_data_(rhs) {
        EVD<StorageType> evd(rhs);
        this->exp_(evd, data_);
    }
    // constructors taking external data
    template <typename RhsXprType_>
    constexpr spd_matrix_impl(const MatrixExpr<RhsXprType_>& rhs, internals::unchecked_t) :
        Base(), data_(rhs.template as_symmetric<Lower>()) {
        EVD<StorageType> evd(data_);
        this->log_(evd, log_data_);
    }
    template <typename RhsXprType_>
    constexpr spd_matrix_impl(const MatrixExpr<RhsXprType_>& rhs, internals::checked_t) :
        Base(), data_(rhs.template as_symmetric<Lower>()) {
        // assert spd property
        fdapde_assert(almost_equal(data_ FDAPDE_COMMA data_.transpose() FDAPDE_COMMA 1e-14));
        EVD<StorageType> evd(data_);
        fdapde_assert(
          std::all_of(evd.eigenvalues().begin() FDAPDE_COMMA evd.eigenvalues().end()
                        FDAPDE_COMMA [](double e) { return e > 0; }));
        this->log_(evd, log_data_);
    }
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr spd_matrix_impl(const std::vector<Scalar__>& data, internals::checked_t) : Base(), data_(data) {
        // assert spd property
        fdapde_assert(almost_equal(data_ FDAPDE_COMMA data_.transpose() FDAPDE_COMMA 1e-14));
        EVD<StorageType> evd(data_);
        fdapde_assert(
          std::all_of(evd.eigenvalues().begin() FDAPDE_COMMA evd.eigenvalues().end()
                        FDAPDE_COMMA [](double e) { return e > 0; }));
        this->log_(evd, log_data_);
    }
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr spd_matrix_impl(const std::vector<Scalar__>& data, internals::unchecked_t) : Base(), data_(data) {
        EVD<StorageType> evd(data_);
        this->log_(evd, log_data_);
    }
    template <typename Scalar__, std::size_t Size>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr spd_matrix_impl(const Scalar__ (&data)[Size], internals::unchecked_t) : Base(), data_(data) {
        fdapde_static_assert(Rows_ != Dynamic && Cols_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        EVD<StorageType> evd(data_);
        this->log_(evd, log_data_);
    }
    template <typename Scalar__, std::size_t Size>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr spd_matrix_impl(const Scalar__ (&data)[Size], internals::checked_t) : Base(), data_(data) {
        fdapde_static_assert(Rows_ != Dynamic && Cols_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_MATRICES_ONLY);
        // assert spd property
        fdapde_assert(almost_equal(data_ FDAPDE_COMMA data_.transpose() FDAPDE_COMMA 1e-14));
        EVD<StorageType> evd(data_);
        fdapde_assert(
          std::all_of(evd.eigenvalues().begin() FDAPDE_COMMA evd.eigenvalues().end()
                        FDAPDE_COMMA [](double e) { return e > 0; }));
        this->log_(evd, log_data_);
    }
    // observers
    constexpr int rows() const { return log_data_.rows(); }
    constexpr int cols() const { return log_data_.cols(); }
    constexpr const StorageType& log() const { return log_data_; }
    constexpr const StorageType& rep() const { return data_; }
    constexpr StorageType& rep() { return data_; }
    // ostream
    friend std::ostream& operator<<(std::ostream& os, const spd_matrix_impl& m) {
        os << m.data_;
        return os;
    }
    // data pointers
    constexpr const Scalar* data() const { return data_.data(); }
    constexpr Scalar* data() { return data_.data(); }
   protected:
    StorageType data_;       // matrix in the SPD domain
    StorageType log_data_;   // matrix in the log domain
};

// log-euclidean SPD view of an existing block of data
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_>
class spd_matrix_view_impl<Scalar_, Rows_, Cols_, StorageOrder_, log_euclidean> :
    public spd_matrix_base<
      Scalar_, Rows_, Cols_, StorageOrder_, log_euclidean,
      spd_matrix_view_impl<Scalar_, Rows_, Cols_, StorageOrder_, log_euclidean>> {
    fdapde_static_assert(
      Rows_ == Dynamic || Cols_ == Dynamic || Rows_ == Cols_, THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
    using Base = spd_matrix_base<
      Scalar_, Rows_, Cols_, StorageOrder_, log_euclidean,
      spd_matrix_view_impl<Scalar_, Rows_, Cols_, StorageOrder_, log_euclidean>>;
    using StorageType = SymmetricMatrixView<Scalar_, Rows_, Cols_, StorageOrder_>;
   public:
    using Scalar = Scalar_;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = std::is_const_v<Scalar_>;
    using assignment_executor = typename StorageType::assignment_executor;

    // constructors
    constexpr spd_matrix_view_impl() = delete;
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr spd_matrix_view_impl(Scalar__* data, internals::checked_t) : Base(), data_(data) {
        fdapde_static_assert(Rows_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_VIEWS_ONLY);
        // assert spd property
        fdapde_assert(almost_equal(data_ FDAPDE_COMMA data_.transpose() FDAPDE_COMMA 1e-14));
        EVD<StorageType> evd(data_);
        fdapde_assert(
          std::all_of(evd.eigenvalues().begin() FDAPDE_COMMA evd.eigenvalues().end()
                        FDAPDE_COMMA [](double e) { return e > 0; }));
        this->log_(evd, log_data_);
    }
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr spd_matrix_view_impl(Scalar__* data, int rows, int cols, internals::checked_t) :
        Base(rows, cols), data_(data) {
        // assert spd property
        fdapde_assert(almost_equal(data_ FDAPDE_COMMA data_.transpose() FDAPDE_COMMA 1e-14));
        EVD<StorageType> evd(data_);
        fdapde_assert(
          std::all_of(evd.eigenvalues().begin() FDAPDE_COMMA evd.eigenvalues().end()
                        FDAPDE_COMMA [](double e) { return e > 0; }));
        this->log_(evd, log_data_);
    }
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr spd_matrix_view_impl(Scalar__* data, internals::unchecked_t) : Base(), data_(data) {
        fdapde_static_assert(Rows_ != Dynamic, THIS_METHOD_IS_FOR_STATIC_SIZED_VIEWS_ONLY);
        EVD<StorageType> evd(data_);
        this->log_(evd, log_data_);
    }
    template <typename Scalar__>
        requires(std::is_constructible_v<Scalar_, Scalar__>)
    constexpr spd_matrix_view_impl(Scalar__* data, int rows, int cols, internals::unchecked_t) :
        Base(rows, cols), data_(data) {
        EVD<StorageType> evd(data_);
        this->log_(evd, log_data_);
    }
    // data pointers
    constexpr const StorageType* data() const { return data_.data(); }
    constexpr StorageType* data() { return data_.data(); }
   private:
    StorageType data_;                               // matrix in the spd domain
    SymmetricMatrix<Scalar, Rows, Cols> log_data_;   // matrix in the log domain
};

}   // namespace internals

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

// public alias types
template <typename Scalar_, int Rows_, int Cols_, typename MetricType_, int StorageOrder_ = RowMajor>
using SPDMatrix = internals::spd_matrix_impl<Scalar_, Rows_, Cols_, StorageOrder_, MetricType_>;
template <typename Scalar_, int Rows_, int Cols_, typename MetricType_, int StorageOrder_ = RowMajor>
using SPDMatrixView = internals::spd_matrix_view_impl<Scalar_, Rows_, Cols_, StorageOrder_, MetricType_>;

// detection trait
template <typename XprType> struct is_spd_matrix {
    static constexpr bool value = std::is_base_of_v<SPDMatrixExpr<std::decay_t<XprType>, log_euclidean>, XprType>;
};
template <typename XprType> static constexpr bool is_spd_matrix_v = is_spd_matrix<XprType>::value;
  
}   // namespace fdapde

#endif   // __FDAPDE_LINALG_SPD_H__
