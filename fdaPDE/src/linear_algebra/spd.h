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

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <type_traits>

#include "header_check.h"
#include "spd_cache.h"

namespace fdapde {

/// @brief borrows verified SPD coefficients and their selected cache
template <typename Scalar, int Rows, int Cols, typename Policy = Cache::None, int StorageOrder = RowMajor>
class SPDMatrixView;
/// @brief owns a compact row-major collection of matrices
template <typename MatrixType, int StorageOrder = RowMajor> class MatrixBatch;
/// @brief identifies a deferred geometric operation evaluated globally at materialization
template <typename Derived> struct GeometryExpr;

namespace internals {
template <typename Scalar, int Rows, int Cols, typename Policy, int StorageOrder> class spd_matrix_impl;
/// @brief rejects arbitrary expressions as evidence of already verified SPD storage
template <typename T> struct is_verified_spd : std::false_type { };
/// @brief recognizes the native checked owner without accepting arbitrary SPD expression subclasses
template <typename S, int R, int C, typename P, int O>
struct is_verified_spd<spd_matrix_impl<S, R, C, P, O>> : std::true_type { };
/// @brief recognizes views whose bindings are created exclusively from checked native storage
template <typename S, int R, int C, typename P, int O>
struct is_verified_spd<SPDMatrixView<S, R, C, P, O>> : std::true_type { };
}   // namespace internals

template <typename T>
concept SPDLike = internals::is_verified_spd<std::remove_cvref_t<T>>::value;

/// @brief identifies symmetric expressions with a positive-definite mathematical contract
template <typename XprType_> struct SPDMatrixExpr : public SymmetricMatrixExpr<XprType_> {
    using XprType = std::decay_t<XprType_>;
    using SymmetricMatrixExpr<XprType_>::derived;
    /// @brief multiplies retained eigenvalues when present, otherwise uses native determinant factorization
    auto determinant() const {
        if constexpr (SPDLike<XprType>) {
            if constexpr (internals::spd_cache_has_v<typename XprType::CachePolicy, Cache::Spectral>) {
                auto values = derived().cache().eigenvalues();
                std::remove_cv_t<typename XprType::Scalar> result = 1;
                for (int i = 0; i < derived().rows(); ++i) result *= values[i];
                return result;
            } else
                return MatrixExpr<XprType>::determinant();
        } else
            return MatrixExpr<XprType>::determinant();
    }
};

/// @brief returns a validated SPD exponential from a finite symmetric expression
template <typename Policy_ = Cache::None, typename XprType_>
auto matrix_exp(const SymmetricMatrixExpr<XprType_>& matrix);
/// @brief returns a validated SPD principal square root
template <typename Policy_ = Cache::None, typename XprType_> auto matrix_sqrt(const SPDMatrixExpr<XprType_>& matrix);
/// @brief returns a validated SPD inverse principal square root
template <typename Policy_ = Cache::None, typename XprType_>
auto matrix_inverse_sqrt(const SPDMatrixExpr<XprType_>& matrix);

namespace internals {

/// @brief rejects nonfinite eigenvalues and spectra with min <= 64 * dimension * epsilon * max
template <typename VectorType_> void validate_positive_spectrum(const VectorType_& eigenvalues, int dimension) {
    using Scalar = std::remove_cv_t<typename VectorType_::Scalar>;
    Scalar minimum = std::numeric_limits<Scalar>::max();
    Scalar maximum = std::numeric_limits<Scalar>::lowest();
    for (int i = 0; i < dimension; ++i) {
        const Scalar eigenvalue = static_cast<Scalar>(eigenvalues[i]);
        fdapde_strong_assert(std::isfinite(eigenvalue), std::domain_error, "SPDMatrix: eigendecomposition failed");
        minimum = std::min(minimum, eigenvalue);
        maximum = std::max(maximum, eigenvalue);
    }
    const Scalar threshold = Scalar(64) * Scalar(dimension) * std::numeric_limits<Scalar>::epsilon() * maximum;
    fdapde_strong_assert(
      maximum > Scalar(0) && minimum > threshold, std::domain_error,
      "SPDMatrix: matrix is not numerically positive definite");
}

/// @brief owns checked SPD coefficients in packed symmetric storage without a metric representation
/// @details coefficient access is read-only; replacement validates a candidate before changing the owner
template <typename Scalar_, int Rows_, int Cols_, typename Policy_, int StorageOrder_>
class spd_matrix_impl : public SPDMatrixExpr<spd_matrix_impl<Scalar_, Rows_, Cols_, Policy_, StorageOrder_>> {
    fdapde_static_assert(
      (Rows_ == Dynamic && Cols_ == Dynamic) || (Rows_ > 0 && Cols_ > 0 && Rows_ == Cols_),
      SPD_MATRICES_MUST_BE_SQUARE_AND_EITHER_FULLY_STATIC_OR_FULLY_DYNAMIC);
    fdapde_static_assert(
      std::is_floating_point_v<Scalar_> && !std::is_const_v<Scalar_> && !std::is_volatile_v<Scalar_>,
      SPD_MATRICES_REQUIRE_AN_UNQUALIFIED_FLOATING_POINT_SCALAR);
    fdapde_static_assert(StorageOrder_ == RowMajor, PACKED_COL_MAJOR_STRUCTURED_STORAGE_IS_NOT_SUPPORTED);
    fdapde_static_assert(
      Rows_ == Dynamic || std::int64_t(Rows_) * std::int64_t(Rows_) <= std::numeric_limits<int>::max(),
      SPD_DENSE_WORKSPACE_SIZE_EXCEEDS_SUPPORTED_RANGE);

    using Base = SPDMatrixExpr<spd_matrix_impl<Scalar_, Rows_, Cols_, Policy_, StorageOrder_>>;
    using StorageType = SymmetricMatrix<Scalar_, Rows_, Cols_, StorageOrder_>;
   public:
    using Scalar = Scalar_;
    using CachePolicy = Policy_;
    using View = SPDMatrixView<Scalar, Rows_, Cols_, CachePolicy, StorageOrder_>;
    using ConstView = SPDMatrixView<const Scalar, Rows_, Cols_, CachePolicy, StorageOrder_>;
    using CacheSlot = spd_cache_slot<Scalar, Rows_, CachePolicy>;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = StorageType::StorageOrder;
    static constexpr int NestAsRef = 1;
    static constexpr int ReadOnly = 1;
    using assignment_executor = deleted_assignment_executor;

    /// @brief forbids an uninitialized SPD owner
    spd_matrix_impl() = delete;
    /// @brief forbids allocation without coefficients whose positive definiteness can be checked
    spd_matrix_impl(int, int) = delete;

    /// @brief copies verified coefficients and independently owns all selected cache quantities
    spd_matrix_impl(const spd_matrix_impl& other) : Base(), data_(other.data_) {
        if constexpr (CachePolicy::Flags != 0) cache_ = std::make_unique<CacheOwner>(*other.cache_);
    }
    /// @brief copies the complete candidate before committing coefficients and cache
    spd_matrix_impl& operator=(const spd_matrix_impl& other) & {
        if (this != &other) {
            spd_matrix_impl candidate(other);
            commit_(candidate);
        }
        return *this;
    }
    /// @brief preserves valid independent ownership when constructing from an expiring value
    spd_matrix_impl(spd_matrix_impl&& other) : spd_matrix_impl(static_cast<const spd_matrix_impl&>(other)) { }
    /// @brief preserves both owners through candidate-based replacement from an expiring value
    spd_matrix_impl& operator=(spd_matrix_impl&& other) & {
        return operator=(static_cast<const spd_matrix_impl&>(other));
    }
    /// @brief forbids copy assignment to a temporary owner
    void operator=(const spd_matrix_impl&) && = delete;

    /// @brief copies square finite input after checking shape, symmetry and numerical positive definiteness
    /// @details verified native sources reuse compatible intermediates; other expressions are fully validated
    template <typename RhsXprType_>
        requires(!requires(const RhsXprType_& expression) { expression.eval_matrix(); })
    explicit spd_matrix_impl(const MatrixExpr<RhsXprType_>& rhs) : Base(), data_(make_storage_(rhs.derived())) {
        if constexpr (SPDLike<RhsXprType_> && std::same_as<Scalar, std::remove_cv_t<typename RhsXprType_::Scalar>>) {
            prepare_from_verified_(rhs.derived());
        } else {
            validate_spd_(rhs.derived(), data_);
        }
    }
    /// @brief evaluates a geometric operation once using this owner's destination policy
    template <typename Rhs>
        requires requires(const Rhs& expression) { expression.eval_matrix(); }
    explicit spd_matrix_impl(const MatrixExpr<Rhs>& rhs) :
        spd_matrix_impl(rhs.derived().template eval<CachePolicy>()) { }
    /// @brief validates replacement coefficients and cache before changing this owner
    template <typename Rhs> spd_matrix_impl& assign(const MatrixExpr<Rhs>& rhs) & {
        spd_matrix_impl candidate(rhs);
        commit_(candidate);
        return *this;
    }
    /// @brief assigns a matrix expression through independent SPD validation
    template <typename Rhs> spd_matrix_impl& operator=(const MatrixExpr<Rhs>& rhs) & { return assign(rhs); }
    /// @brief constructs an identity with known coefficients and cache without spectral validation
    static spd_matrix_impl Identity(int order = Rows_) {
        validate_identity_order_(order);
        StorageType storage;
        if constexpr (Rows_ == Dynamic) storage.resize(order, order);
        for (int i = 0; i < order; ++i)
            for (int j = 0; j <= i; ++j) storage(i, j) = i == j ? Scalar(1) : Scalar(0);
        spd_matrix_impl result(storage, trusted_t {});
        if constexpr (CachePolicy::Flags != 0) {
            result.cache_ = std::make_unique<CacheOwner>(order);
            result.cache_->slot.set_identity();
        }
        return result;
    }
    /// @brief borrows this owner's ready cache while the owner and its current value remain alive
    const CacheSlot& cache() const&
        requires(CachePolicy::Flags != 0)
    {
        return cache_->slot;
    }
    /// @brief prevents a cache reference from escaping a temporary owner
    void cache() const&&
        requires(CachePolicy::Flags != 0)
    = delete;
    /// @brief borrows mutable verified value access without exposing writable coefficients
    View view() & { return View(*this); }
    /// @brief borrows read-only coefficients and cache
    ConstView view() const& { return ConstView(*this); }
    /// @brief prevents a view from borrowing a temporary owner
    void view() const&& = delete;

    /// @brief returns the square matrix dimension
    constexpr int rows() const { return data_.rows(); }
    /// @brief returns the square matrix dimension
    constexpr int cols() const { return data_.cols(); }
    /// @brief returns a coefficient by value, reflecting the stored lower triangle above the diagonal
    constexpr Scalar operator()(int i, int j) const { return static_cast<Scalar>(data_(i, j)); }
    /// @brief borrows the read-only packed symmetric representation while this owner remains alive
    constexpr const StorageType& rep() const& { return data_; }
    /// @brief prevents borrowing storage from a temporary owner
    constexpr void rep() const&& = delete;
    /// @brief borrows read-only lower-triangular coefficients packed row by row
    constexpr const Scalar* data() const& { return data_.data(); }
    /// @brief prevents a data pointer from escaping a temporary owner
    constexpr void data() const&& = delete;

    /// @brief writes the full symmetric matrix through its stored representation
    friend std::ostream& operator<<(std::ostream& os, const spd_matrix_impl& matrix) {
        os << matrix.data_;
        return os;
    }
   private:
    template <typename, int, int, typename, int> friend class fdapde::SPDMatrixView;
    template <typename, int> friend class fdapde::MatrixBatch;
    template <typename Policy, typename XprType_>
    friend auto fdapde::matrix_exp(const SymmetricMatrixExpr<XprType_>& matrix);
    template <typename Policy, typename XprType_>
    friend auto fdapde::matrix_sqrt(const SPDMatrixExpr<XprType_>& matrix);
    template <typename Policy, typename XprType_>
    friend auto fdapde::matrix_inverse_sqrt(const SPDMatrixExpr<XprType_>& matrix);

    /// @brief permits storage adoption only after a spectral primitive has established the SPD postconditions
    struct trusted_t { };

    /// @brief copies validated symmetric storage without repeating validation or creating a spectral cache
    spd_matrix_impl(const StorageType& storage, trusted_t) : Base(), data_(storage) { }

    /// @brief certifies the rounded finite reconstruction before invoking the private trusted constructor
    /// @details only spectral friends may call this after reconstruct_symmetric has checked every coefficient
    static spd_matrix_impl from_spectral_(const StorageType& storage) {
        // positive transformed eigenvalues alone do not certify the rounded reconstructed matrix
        validate_shape_(storage);
        spd_matrix_impl result(storage, trusted_t {});
        result.validate_spectrum_(storage);
        return result;
    }

    /// @brief rejects empty, nonsquare or incompatible input and dense workspaces beyond the int index range
    template <typename RhsXprType_> static void validate_shape_(const RhsXprType_& rhs) {
        fdapde_strong_assert(
          rhs.rows() > 0 && rhs.rows() == rhs.cols(), std::invalid_argument,
          "SPDMatrix: expected a nonempty square matrix");
        if constexpr (Rows != Dynamic) {
            fdapde_strong_assert(
              rhs.rows() == Rows, std::invalid_argument, "SPDMatrix: incompatible matrix dimensions");
        }
        const std::int64_t dimension = rhs.rows();
        const std::int64_t dense_size = dimension * dimension;
        fdapde_strong_assert(
          dense_size <= std::numeric_limits<int>::max(), std::length_error,
          "SPDMatrix: dense workspace size exceeds supported range");
    }

    /// @brief checks dimensions and copies the lower triangle into an independent candidate
    template <typename RhsXprType_> static StorageType make_storage_(const RhsXprType_& rhs) {
        validate_shape_(rhs);
        StorageType storage;
        if constexpr (Rows == Dynamic) { storage.resize(rhs.rows(), rhs.cols()); }
        for (int i = 0; i < rhs.rows(); ++i) {
            for (int j = 0; j <= i; ++j) { storage(i, j) = static_cast<Scalar>(rhs(i, j)); }
        }
        return storage;
    }

    /// @brief validates finite symmetric input and the candidate spectrum before publishing an SPD owner
    template <typename RhsXprType_> void validate_spd_(const RhsXprType_& rhs, const StorageType& storage) {
        Scalar scale = Scalar(0);
        for (int i = 0; i < rhs.rows(); ++i) {
            for (int j = 0; j < rhs.cols(); ++j) {
                const Scalar value = static_cast<Scalar>(rhs(i, j));
                fdapde_strong_assert(
                  std::isfinite(value), std::invalid_argument, "SPDMatrix: coefficients must be finite");
                scale = std::max(scale, std::abs(value));
            }
        }

        const Scalar tolerance = Scalar(32) * Scalar(rhs.rows()) * std::numeric_limits<Scalar>::epsilon() * scale;
        for (int i = 0; i < rhs.rows(); ++i) {
            for (int j = 0; j < i; ++j) {
                const Scalar difference = std::abs(static_cast<Scalar>(rhs(i, j)) - static_cast<Scalar>(rhs(j, i)));
                fdapde_strong_assert(
                  std::isfinite(difference) && difference <= tolerance, std::invalid_argument,
                  "SPDMatrix: matrix must be symmetric");
            }
        }

        validate_spectrum_(storage);
    }

    /// @brief rejects eigensolver failure or numerical loss of positive definiteness in finite symmetric storage
    void validate_spectrum_(const StorageType& storage) {
        const EVD<StorageType> evd(storage);
        fdapde_strong_assert(evd.computed(), std::domain_error, "SPDMatrix: eigendecomposition failed");
        validate_positive_spectrum(evd.eigenvalues(), storage.rows());
        if constexpr (CachePolicy::Flags != 0) {
            cache_ = std::make_unique<CacheOwner>(storage.rows());
            cache_->slot.prepare(evd);
        }
    }
    /// @brief validates the explicit order used by the identity factory before allocating storage
    static void validate_identity_order_(int order) {
        fdapde_strong_assert(
          order > 0 && (Rows_ == Dynamic || order == Rows_), std::invalid_argument,
          "SPDMatrix: incompatible identity order");
        fdapde_strong_assert(
          std::int64_t(order) * order <= std::numeric_limits<int>::max(), std::length_error,
          "SPDMatrix: dense workspace size exceeds supported range");
    }
    /// @brief reuses retained common quantities and computes missing quantities from one coherent spectrum
    template <SPDLike Rhs> void prepare_from_verified_(const Rhs& rhs) {
        if constexpr (CachePolicy::Flags != 0) {
            cache_ = std::make_unique<CacheOwner>(rows());
            using SourcePolicy = typename Rhs::CachePolicy;
            if constexpr ((SourcePolicy::Flags & CachePolicy::Flags) == CachePolicy::Flags) {
                cache_->slot.copy_common(rhs.cache());
            } else if constexpr (spd_cache_has_v<SourcePolicy, Cache::Spectral>) {
                cache_->slot.prepare(rhs.cache());
                cache_->slot.copy_common(rhs.cache());
            } else {
                const EVD<StorageType> evd(data_);
                fdapde_strong_assert(evd.computed(), std::domain_error, "SPDMatrix: eigendecomposition failed");
                cache_->slot.prepare(evd);
                if constexpr (SourcePolicy::Flags != 0) cache_->slot.template copy_common<false>(rhs.cache());
            }
        }
    }
    /// @brief copies prepared numeric storage before publishing the replacement cache with a nonthrowing swap
    void commit_(spd_matrix_impl& candidate) {
        data_ = candidate.data_;
        if constexpr (CachePolicy::Flags != 0) {
            if (cache_ && cache_->slot.rows() == candidate.rows())
                cache_->slot.copy_from(candidate.cache_->slot);
            else
                cache_.swap(candidate.cache_);
        }
    }
    using CacheOwner = owning_spd_cache<Scalar, Rows_, CachePolicy>;
    using CacheStorage = std::conditional_t<CachePolicy::Flags == 0, empty_spd_cache<>, std::unique_ptr<CacheOwner>>;
    StorageType data_;
    [[no_unique_address]] CacheStorage cache_;
};

/// @brief checks nonempty square shape, workspace size and finite coefficients of a symmetric expression
template <typename XprType_> void validate_finite_symmetric(const XprType_& matrix) {
    using Scalar = std::remove_cv_t<typename XprType_::Scalar>;
    fdapde_strong_assert(
      matrix.rows() > 0 && matrix.rows() == matrix.cols(), std::invalid_argument,
      "SPD spectral operation: expected a nonempty square matrix");
    const std::int64_t dimension = matrix.rows();
    fdapde_strong_assert(
      dimension * dimension <= std::numeric_limits<int>::max(), std::length_error,
      "SPD spectral operation: dense workspace size exceeds supported range");
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            fdapde_strong_assert(
              std::isfinite(static_cast<Scalar>(matrix(i, j))), std::invalid_argument,
              "SPD spectral operation: coefficients must be finite");
        }
    }
}

/// @brief reports eigensolver failure before any spectral result is consumed
template <typename XprType_> void require_computed(const EVD<XprType_>& evd) {
    fdapde_strong_assert(evd.computed(), std::domain_error, "SPD spectral operation: eigendecomposition failed");
}

/// @brief materializes transformed eigenvalues and rejects nonfinite scalar results
template <typename XprType_, typename UnaryOp_>
auto transform_eigenvalues(const XprType_& evd, int dimension, UnaryOp_&& op) {
    using XprType = std::decay_t<XprType_>;
    using Scalar = std::remove_cv_t<typename XprType::Scalar>;
    constexpr int Rows = XprType::Rows;
    Vector<Scalar, Rows> values;
    if constexpr (Rows == Dynamic) { values.resize(dimension); }
    for (int i = 0; i < dimension; ++i) {
        values[i] = op(evd.eigenvalues()[i]);
        fdapde_strong_assert(std::isfinite(values[i]), std::domain_error, "SPD spectral operation: nonfinite result");
    }
    return values;
}

/// @brief forms Q * diag(values) * Q transpose in owned lower-triangular storage
template <typename XprType_, typename VectorType_>
auto reconstruct_symmetric(const XprType_& evd, int dimension, const VectorType_& values) {
    using XprType = std::decay_t<XprType_>;
    using Scalar = std::remove_cv_t<typename XprType::Scalar>;
    constexpr int Rows = XprType::Rows;
    constexpr int Cols = XprType::Cols;
    SymmetricMatrix<Scalar, Rows, Cols> result;
    if constexpr (Rows == Dynamic) { result.resize(dimension, dimension); }
    const auto eigenvectors = evd.eigenvectors();
    for (int i = 0; i < dimension; ++i) {
        for (int j = 0; j <= i; ++j) {
            Scalar value = Scalar(0);
            for (int k = 0; k < dimension; ++k) { value += eigenvectors(i, k) * values[k] * eigenvectors(j, k); }
            fdapde_strong_assert(std::isfinite(value), std::domain_error, "SPD spectral operation: nonfinite result");
            result(i, j) = value;
        }
    }
    return result;
}

/// @brief evaluates the logarithmic divided difference using its repeated-eigenvalue limit and log1p near equality
template <typename Scalar_> Scalar_ log_divided_difference(Scalar_ x, Scalar_ y) {
    if (x == y) return Scalar_(1) / x;
    const Scalar_ delta = x - y;
    if (std::abs(delta) <= Scalar_(0.5) * std::min(x, y)) {
        return Scalar_(0.5) * (std::log1p(delta / y) / delta + std::log1p(-delta / x) / -delta);
    }
    return (std::log(x) - std::log(y)) / delta;
}

/// @brief evaluates the exponential divided difference using its repeated-eigenvalue limit and expm1 near equality
template <typename Scalar_> Scalar_ exp_divided_difference(Scalar_ x, Scalar_ y) {
    if (x == y) return std::exp(x);
    if (x > y) return std::exp(x) * (-std::expm1(y - x)) / (x - y);
    return std::exp(y) * (-std::expm1(x - y)) / (y - x);
}

/// @brief applies spectral divided differences to a symmetric direction in the eigenvector basis
template <typename XprType_, typename DirectionXprType_, typename DividedDifference_>
auto frechet_symmetric(
  const XprType_& evd, int dimension, const DirectionXprType_& direction, DividedDifference_&& divided_difference) {
    using XprType = std::decay_t<XprType_>;
    using Scalar = std::remove_cv_t<typename XprType::Scalar>;
    constexpr int Rows = XprType::Rows;
    constexpr int Cols = XprType::Cols;
    fdapde_strong_assert(
      direction.rows() == dimension && direction.cols() == dimension, std::invalid_argument,
      "SPD spectral operation: incompatible direction dimensions");
    validate_finite_symmetric(direction);

    // preserve a known static loop bound when GCC inlines packed coefficient access
    constexpr int StaticOrder = Rows != Dynamic ? Rows : std::decay_t<DirectionXprType_>::Rows;
    fdapde_strong_assert(
      StaticOrder == Dynamic || dimension == StaticOrder, std::invalid_argument,
      "SPD spectral operation: incompatible static dimensions");
    const int extent = StaticOrder == Dynamic ? dimension : StaticOrder;

    Matrix<Scalar, Rows, Cols> hq;
    Matrix<Scalar, Rows, Cols> coefficients;
    Matrix<Scalar, Rows, Cols> q_coefficients;
    if constexpr (Rows == Dynamic) {
        hq.resize(dimension, dimension);
        coefficients.resize(dimension, dimension);
        q_coefficients.resize(dimension, dimension);
    }

    // rotate the direction into the eigenvector basis before applying the divided differences
    const auto eigenvectors = evd.eigenvectors();
    for (int i = 0; i < extent; ++i) {
        for (int j = 0; j < extent; ++j) {
            Scalar value = Scalar(0);
            for (int k = 0; k < extent; ++k) value += static_cast<Scalar>(direction(i, k)) * eigenvectors(k, j);
            hq(i, j) = value;
        }
    }
    for (int i = 0; i < extent; ++i) {
        for (int j = 0; j < extent; ++j) {
            Scalar value = Scalar(0);
            for (int k = 0; k < extent; ++k) { value += eigenvectors(k, i) * hq(k, j); }
            if constexpr (std::is_invocable_v<DividedDifference_, Scalar, Scalar, int, int>) {
                coefficients(i, j) = divided_difference(evd.eigenvalues()[i], evd.eigenvalues()[j], i, j) * value;
            } else {
                coefficients(i, j) = divided_difference(evd.eigenvalues()[i], evd.eigenvalues()[j]) * value;
            }
            fdapde_strong_assert(
              std::isfinite(coefficients(i, j)), std::domain_error,
              "SPD spectral operation: nonfinite Frechet derivative");
        }
    }
    for (int i = 0; i < extent; ++i) {
        for (int j = 0; j < extent; ++j) {
            Scalar value = Scalar(0);
            for (int k = 0; k < extent; ++k) { value += eigenvectors(i, k) * coefficients(k, j); }
            q_coefficients(i, j) = value;
        }
    }

    SymmetricMatrix<Scalar, Rows, Cols> result;
    if constexpr (Rows == Dynamic) { result.resize(dimension, dimension); }
    for (int i = 0; i < extent; ++i) {
        for (int j = 0; j <= i; ++j) {
            Scalar value = Scalar(0);
            for (int k = 0; k < extent; ++k) { value += q_coefficients(i, k) * eigenvectors(j, k); }
            fdapde_strong_assert(
              std::isfinite(value), std::domain_error, "SPD spectral operation: nonfinite Frechet derivative");
            result(i, j) = value;
        }
    }
    return result;
}

/// @brief passes ready native spectral factors or one validated local decomposition to an operation
template <typename Xpr, typename Operation> auto with_spd_spectral(const Xpr& value, Operation operation) {
    if constexpr (SPDLike<Xpr>) {
        if constexpr (spd_cache_has_v<typename Xpr::CachePolicy, Cache::Spectral>)
            return operation(value.cache());
        else {
            const EVD<Xpr> evd(value);
            require_computed(evd);
            return operation(evd);
        }
    } else {
        validate_finite_symmetric(value);
        const EVD<Xpr> evd(value);
        require_computed(evd);
        validate_positive_spectrum(evd.eigenvalues(), value.rows());
        return operation(evd);
    }
}

}   // namespace internals

/// @brief provides checked SPD ownership for unqualified floating scalars and square fixed or fully dynamic shapes
/// @details only row-major packed storage is supported; public construction always validates the input
template <typename Scalar_, int Rows_, int Cols_, typename Policy_ = Cache::None, int StorageOrder_ = RowMajor>
using SPDMatrix = internals::spd_matrix_impl<Scalar_, Rows_, Cols_, Policy_, StorageOrder_>;

/// @brief borrows a verified packed SPD value, with assignment transferring values rather than bindings
/// @details the owner must outlive the view; owner shape changes and batch replacement invalidate all views
template <typename Scalar_, int Rows_, int Cols_, typename Policy_, int StorageOrder_>
class SPDMatrixView : public SPDMatrixExpr<SPDMatrixView<Scalar_, Rows_, Cols_, Policy_, StorageOrder_>> {
    using ValueScalar = std::remove_const_t<Scalar_>;
    using Owner = SPDMatrix<ValueScalar, Rows_, Cols_, Policy_, StorageOrder_>;
    using Slot = typename Owner::CacheSlot;
    using CachePointer = std::conditional_t<
      Policy_::Flags == 0, internals::empty_spd_cache<1>,
      std::conditional_t<std::is_const_v<Scalar_>, const Slot*, Slot*>>;
   public:
    using Scalar = ValueScalar;
    using CachePolicy = Policy_;
    using View = SPDMatrixView<ValueScalar, Rows_, Cols_, Policy_, StorageOrder_>;
    using ConstView = SPDMatrixView<const ValueScalar, Rows_, Cols_, Policy_, StorageOrder_>;
    static constexpr int Rows = Rows_;
    static constexpr int Cols = Cols_;
    static constexpr int StorageOrder = StorageOrder_;
    static constexpr int NestAsRef = 0;
    static constexpr int ReadOnly = 1;
    using assignment_executor = internals::deleted_assignment_executor;

    /// @brief copies a view binding without copying or validating its matrix
    SPDMatrixView(const SPDMatrixView&) = default;
    /// @brief rejects an unbound SPD view
    SPDMatrixView() = delete;
    /// @brief binds mutable verified storage owned by a persistent SPD matrix
    explicit SPDMatrixView(Owner& owner) : data_(owner.data_.data(), owner.rows(), owner.cols()) {
        if constexpr (CachePolicy::Flags != 0) cache_ = &owner.cache_->slot;
    }
    /// @brief binds read-only verified storage owned by a persistent SPD matrix
    explicit SPDMatrixView(const Owner& owner)
        requires(std::is_const_v<Scalar_>)
        : data_(owner.data(), owner.rows(), owner.cols()) {
        if constexpr (CachePolicy::Flags != 0) cache_ = &owner.cache_->slot;
    }
    /// @brief prevents borrowing coefficients from a temporary owner
    SPDMatrixView(Owner&&) = delete;
    /// @brief prevents borrowing coefficients from a const temporary owner
    SPDMatrixView(const Owner&&) = delete;
    /// @brief converts mutable value access to a read-only binding
    SPDMatrixView(const View& other)
        requires(std::is_const_v<Scalar_>)
        : data_(other.data(), other.rows(), other.cols()) {
        if constexpr (CachePolicy::Flags != 0) cache_ = other.cache_;
    }
    /// @brief returns the verified matrix order
    int rows() const { return data_.rows(); }
    /// @brief returns the verified matrix order
    int cols() const { return data_.cols(); }
    /// @brief reads a symmetric coefficient without exposing mutable packed storage
    Scalar operator()(int i, int j) const { return data_(i, j); }
    /// @brief borrows read-only lower-triangular coefficients
    const Scalar* data() const { return data_.data(); }
    /// @brief returns a read-only symmetric view suitable for ordinary matrix expressions
    auto rep() const { return SymmetricMatrixView<const Scalar, Rows, Cols, StorageOrder>(data(), rows(), cols()); }
    /// @brief returns the owner's ready selected cache without extending its lifetime
    const Slot& cache() const
        requires(CachePolicy::Flags != 0)
    {
        return *cache_;
    }
    /// @brief validates a complete replacement before copying it into this binding
    template <typename Rhs>
    SPDMatrixView& assign(const MatrixExpr<Rhs>& rhs)
        requires(!std::is_const_v<Scalar_>)
    {
        fdapde_strong_assert(
          rhs.derived().rows() == rows() && rhs.derived().cols() == cols(), std::invalid_argument,
          "SPDMatrixView: assignment cannot change the bound shape");
        const Owner candidate(rhs);
        commit_(candidate);
        return *this;
    }
    /// @brief transfers a source view's verified value while preserving this view's binding
    SPDMatrixView& operator=(const SPDMatrixView& other)
        requires(!std::is_const_v<Scalar_>)
    {
        return assign(other);
    }
    /// @brief forbids value assignment through a read-only view
    SPDMatrixView& operator=(const SPDMatrixView&)
        requires(std::is_const_v<Scalar_>)
    = delete;
    /// @brief assigns a matrix expression through independent validation
    template <typename Rhs>
    SPDMatrixView& operator=(const MatrixExpr<Rhs>& rhs)
        requires(!std::is_const_v<Scalar_>)
    {
        return assign(rhs);
    }
   private:
    template <typename, int> friend class MatrixBatch;
    template <typename, int, int, typename, int> friend class SPDMatrixView;
    /// @brief binds a batch-owned verified coefficient row and its ready cache slot
    SPDMatrixView(Scalar_* data, int n, CachePointer cache) : data_(data, n, n), cache_(cache) { }
    /// @brief commits already validated coefficients and intermediates with no allocating operations
    void commit_(const Owner& candidate)
        requires(!std::is_const_v<Scalar_>)
    {
        std::copy_n(candidate.data(), std::size_t(rows()) * (rows() + 1) / 2, data_.data());
        if constexpr (CachePolicy::Flags != 0) cache_->copy_from(candidate.cache());
    }
    SymmetricMatrixView<Scalar_, Rows_, Cols_, StorageOrder_> data_;
    [[no_unique_address]] CachePointer cache_;
};

/// @brief returns the owned symmetric logarithm of a numerically positive-definite expression
template <typename XprType_> auto matrix_log(const SPDMatrixExpr<XprType_>& matrix) {
    const XprType_& value = matrix.derived();
    if constexpr (SPDLike<XprType_>) {
        if constexpr (internals::spd_cache_has_v<typename XprType_::CachePolicy, Cache::Log>) {
            return SymmetricMatrix<typename XprType_::Scalar, XprType_::Rows, XprType_::Cols>(
              value.cache().template matrix<Cache::Log>());
        } else
            return internals::with_spd_spectral(value, [&](const auto& evd) {
                const auto eigenvalues =
                  internals::transform_eigenvalues(evd, value.rows(), [](auto x) { return std::log(x); });
                return internals::reconstruct_symmetric(evd, value.rows(), eigenvalues);
            });
    } else
        return internals::with_spd_spectral(value, [&](const auto& evd) {
            const auto eigenvalues =
              internals::transform_eigenvalues(evd, value.rows(), [](auto x) { return std::log(x); });
            return internals::reconstruct_symmetric(evd, value.rows(), eigenvalues);
        });
}

/// @brief returns the checked SPD exponential of a finite symmetric expression
/// @details rejects overflow, underflow to a singular spectrum and numerically ill-conditioned SPD results
template <typename Policy_, typename XprType_> auto matrix_exp(const SymmetricMatrixExpr<XprType_>& matrix) {
    using XprType = std::decay_t<XprType_>;
    using Scalar = std::remove_cv_t<typename XprType::Scalar>;
    const XprType_& value = matrix.derived();
    internals::validate_finite_symmetric(value);
    const EVD<XprType_> evd(value);
    internals::require_computed(evd);
    const auto eigenvalues = internals::transform_eigenvalues(evd, value.rows(), [](auto x) { return std::exp(x); });
    internals::validate_positive_spectrum(eigenvalues, value.rows());
    return SPDMatrix<Scalar, XprType::Rows, XprType::Cols, Policy_>::from_spectral_(
      internals::reconstruct_symmetric(evd, value.rows(), eigenvalues));
}

/// @brief returns the checked SPD principal square root, preserving the input scalar and static shape
template <typename Policy_, typename XprType_> auto matrix_sqrt(const SPDMatrixExpr<XprType_>& matrix) {
    using XprType = std::decay_t<XprType_>;
    using Scalar = std::remove_cv_t<typename XprType::Scalar>;
    const XprType_& value = matrix.derived();
    using Result = SPDMatrix<Scalar, XprType::Rows, XprType::Cols, Policy_>;
    if constexpr (SPDLike<XprType>) {
        if constexpr (internals::spd_cache_has_v<typename XprType::CachePolicy, Cache::Sqrt>) {
            return Result(value.cache().template matrix<Cache::Sqrt>());
        } else
            return internals::with_spd_spectral(value, [&](const auto& evd) {
                const auto eigenvalues =
                  internals::transform_eigenvalues(evd, value.rows(), [](auto x) { return std::sqrt(x); });
                internals::validate_positive_spectrum(eigenvalues, value.rows());
                return Result::from_spectral_(internals::reconstruct_symmetric(evd, value.rows(), eigenvalues));
            });
    } else
        return internals::with_spd_spectral(value, [&](const auto& evd) {
            const auto eigenvalues =
              internals::transform_eigenvalues(evd, value.rows(), [](auto x) { return std::sqrt(x); });
            internals::validate_positive_spectrum(eigenvalues, value.rows());
            return Result::from_spectral_(internals::reconstruct_symmetric(evd, value.rows(), eigenvalues));
        });
}

/// @brief returns the checked SPD inverse principal square root of a numerically positive-definite expression
template <typename Policy_, typename XprType_> auto matrix_inverse_sqrt(const SPDMatrixExpr<XprType_>& matrix) {
    using XprType = std::decay_t<XprType_>;
    using Scalar = std::remove_cv_t<typename XprType::Scalar>;
    const XprType_& value = matrix.derived();
    using Result = SPDMatrix<Scalar, XprType::Rows, XprType::Cols, Policy_>;
    if constexpr (SPDLike<XprType>) {
        if constexpr (internals::spd_cache_has_v<typename XprType::CachePolicy, Cache::InverseSqrt>) {
            return Result(value.cache().template matrix<Cache::InverseSqrt>());
        } else
            return internals::with_spd_spectral(value, [&](const auto& evd) {
                const auto eigenvalues =
                  internals::transform_eigenvalues(evd, value.rows(), [](auto x) { return Scalar(1) / std::sqrt(x); });
                internals::validate_positive_spectrum(eigenvalues, value.rows());
                return Result::from_spectral_(internals::reconstruct_symmetric(evd, value.rows(), eigenvalues));
            });
    } else
        return internals::with_spd_spectral(value, [&](const auto& evd) {
            const auto eigenvalues =
              internals::transform_eigenvalues(evd, value.rows(), [](auto x) { return Scalar(1) / std::sqrt(x); });
            internals::validate_positive_spectrum(eigenvalues, value.rows());
            return Result::from_spectral_(internals::reconstruct_symmetric(evd, value.rows(), eigenvalues));
        });
}

/// @brief returns the owned symmetric Frechet derivative of log at an SPD point along a symmetric direction
template <typename XprType_, typename DirectionXprType_>
auto matrix_log_frechet(
  const SPDMatrixExpr<XprType_>& matrix, const SymmetricMatrixExpr<DirectionXprType_>& direction) {
    const XprType_& value = matrix.derived();
    return internals::with_spd_spectral(value, [&](const auto& evd) {
        if constexpr (SPDLike<XprType_>) {
            using Policy = typename XprType_::CachePolicy;
            if constexpr (
              internals::spd_cache_has_v<Policy, Cache::Spectral> &&
              internals::spd_cache_has_v<Policy, Cache::LogDividedDifferences>) {
                const auto differences = value.cache().log_divided_differences();
                return internals::frechet_symmetric(
                  evd, value.rows(), direction.derived(), [&](auto, auto, int i, int j) { return differences(i, j); });
            } else
                return internals::frechet_symmetric(evd, value.rows(), direction.derived(), [](auto x, auto y) {
                    return internals::log_divided_difference(x, y);
                });
        } else
            return internals::frechet_symmetric(evd, value.rows(), direction.derived(), [](auto x, auto y) {
                return internals::log_divided_difference(x, y);
            });
    });
}

/// @brief returns the owned symmetric Frechet derivative of exp at a symmetric point along a symmetric direction
template <typename XprType_, typename DirectionXprType_>
auto matrix_exp_frechet(
  const SymmetricMatrixExpr<XprType_>& matrix, const SymmetricMatrixExpr<DirectionXprType_>& direction) {
    const XprType_& value = matrix.derived();
    internals::validate_finite_symmetric(value);
    const EVD<XprType_> evd(value);
    internals::require_computed(evd);
    return internals::frechet_symmetric(
      evd, value.rows(), direction.derived(), [](auto x, auto y) { return internals::exp_divided_difference(x, y); });
}

/// @brief detects the SPD expression contract after removing reference and cv qualifiers
template <typename XprType_> struct is_spd_matrix {
    using XprType = std::remove_cvref_t<XprType_>;
    static constexpr bool value = std::is_base_of_v<SPDMatrixExpr<XprType>, XprType>;
};
template <typename XprType> inline constexpr bool is_spd_matrix_v = is_spd_matrix<XprType>::value;

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_SPD_H__
