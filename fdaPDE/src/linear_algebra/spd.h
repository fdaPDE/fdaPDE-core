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

namespace fdapde {

/// @brief identifies symmetric expressions with a positive-definite mathematical contract
template <typename XprType_> struct SPDMatrixExpr : public SymmetricMatrixExpr<XprType_> {
    using XprType = std::decay_t<XprType_>;
    using SymmetricMatrixExpr<XprType_>::derived;
};

/// @brief returns a validated SPD exponential from a finite symmetric expression
template <typename XprType_> auto matrix_exp(const SymmetricMatrixExpr<XprType_>& matrix);
/// @brief returns a validated SPD principal square root
template <typename XprType_> auto matrix_sqrt(const SPDMatrixExpr<XprType_>& matrix);
/// @brief returns a validated SPD inverse principal square root
template <typename XprType_> auto matrix_inverse_sqrt(const SPDMatrixExpr<XprType_>& matrix);

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
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_>
class spd_matrix_impl : public SPDMatrixExpr<spd_matrix_impl<Scalar_, Rows_, Cols_, StorageOrder_>> {
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

    using Base = SPDMatrixExpr<spd_matrix_impl<Scalar_, Rows_, Cols_, StorageOrder_>>;
    using StorageType = SymmetricMatrix<Scalar_, Rows_, Cols_, StorageOrder_>;
   public:
    using Scalar = Scalar_;
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

    /// @brief copies validated coefficients and dimensions
    spd_matrix_impl(const spd_matrix_impl&) = default;
    /// @brief replaces this owner with a copy of another validated SPD matrix
    spd_matrix_impl& operator=(const spd_matrix_impl&) & = default;
    /// @brief forbids copy assignment to a temporary owner
    void operator=(const spd_matrix_impl&) && = delete;

    /// @brief copies square finite input after checking shape, symmetry and numerical positive definiteness
    /// @details symmetry uses 32 * dimension * epsilon * max_abs_coefficient; the lower triangle is stored
    template <typename RhsXprType_>
    explicit spd_matrix_impl(const MatrixExpr<RhsXprType_>& rhs) : Base(), data_(make_storage_(rhs.derived())) {
        validate_spd_(rhs.derived(), data_);
    }

    /// @brief validates replacement coefficients before committing them, preserving the owner on validation failure
    template <typename RhsXprType_> spd_matrix_impl& assign(const MatrixExpr<RhsXprType_>& rhs) & {
        // validate independent storage before replacing the current coefficients or dimensions
        StorageType candidate = make_storage_(rhs.derived());
        validate_spd_(rhs.derived(), candidate);
        data_ = candidate;
        return *this;
    }

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
    template <typename XprType_> friend auto fdapde::matrix_exp(const SymmetricMatrixExpr<XprType_>& matrix);
    template <typename XprType_> friend auto fdapde::matrix_sqrt(const SPDMatrixExpr<XprType_>& matrix);
    template <typename XprType_> friend auto fdapde::matrix_inverse_sqrt(const SPDMatrixExpr<XprType_>& matrix);

    /// @brief permits storage adoption only after a spectral primitive has established the SPD postconditions
    struct trusted_t { };

    /// @brief copies validated symmetric storage without repeating validation or creating a spectral cache
    spd_matrix_impl(const StorageType& storage, trusted_t) : Base(), data_(storage) { }

    /// @brief certifies the rounded finite reconstruction before invoking the private trusted constructor
    /// @details only spectral friends may call this after reconstruct_symmetric has checked every coefficient
    static spd_matrix_impl from_spectral_(const StorageType& storage) {
        // positive transformed eigenvalues alone do not certify the rounded reconstructed matrix
        validate_shape_(storage);
        validate_spectrum_(storage);
        return spd_matrix_impl(storage, trusted_t {});
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
    template <typename RhsXprType_> static void validate_spd_(const RhsXprType_& rhs, const StorageType& storage) {
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
    static void validate_spectrum_(const StorageType& storage) {
        const EVD<StorageType> evd(storage);
        fdapde_strong_assert(evd.computed(), std::domain_error, "SPDMatrix: eigendecomposition failed");
        validate_positive_spectrum(evd.eigenvalues(), storage.rows());
    }

    StorageType data_;
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
auto transform_eigenvalues(const EVD<XprType_>& evd, int dimension, UnaryOp_&& op) {
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
auto reconstruct_symmetric(const EVD<XprType_>& evd, int dimension, const VectorType_& values) {
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
  const EVD<XprType_>& evd, int dimension, const DirectionXprType_& direction,
  DividedDifference_&& divided_difference) {
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
            coefficients(i, j) = divided_difference(evd.eigenvalues()[i], evd.eigenvalues()[j]) * value;
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

}   // namespace internals

/// @brief provides checked SPD ownership for unqualified floating scalars and square fixed or fully dynamic shapes
/// @details only row-major packed storage is supported; public construction always validates the input
template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_ = RowMajor>
using SPDMatrix = internals::spd_matrix_impl<Scalar_, Rows_, Cols_, StorageOrder_>;

/// @brief returns the owned symmetric logarithm of a numerically positive-definite expression
template <typename XprType_> auto matrix_log(const SPDMatrixExpr<XprType_>& matrix) {
    const XprType_& value = matrix.derived();
    internals::validate_finite_symmetric(value);
    const EVD<XprType_> evd(value);
    internals::require_computed(evd);
    internals::validate_positive_spectrum(evd.eigenvalues(), value.rows());
    const auto eigenvalues = internals::transform_eigenvalues(evd, value.rows(), [](auto x) { return std::log(x); });
    return internals::reconstruct_symmetric(evd, value.rows(), eigenvalues);
}

/// @brief returns the checked SPD exponential of a finite symmetric expression
/// @details rejects overflow, underflow to a singular spectrum and numerically ill-conditioned SPD results
template <typename XprType_> auto matrix_exp(const SymmetricMatrixExpr<XprType_>& matrix) {
    using XprType = std::decay_t<XprType_>;
    using Scalar = std::remove_cv_t<typename XprType::Scalar>;
    const XprType_& value = matrix.derived();
    internals::validate_finite_symmetric(value);
    const EVD<XprType_> evd(value);
    internals::require_computed(evd);
    const auto eigenvalues = internals::transform_eigenvalues(evd, value.rows(), [](auto x) { return std::exp(x); });
    internals::validate_positive_spectrum(eigenvalues, value.rows());
    return SPDMatrix<Scalar, XprType::Rows, XprType::Cols>::from_spectral_(
      internals::reconstruct_symmetric(evd, value.rows(), eigenvalues));
}

/// @brief returns the checked SPD principal square root, preserving the input scalar and static shape
template <typename XprType_> auto matrix_sqrt(const SPDMatrixExpr<XprType_>& matrix) {
    using XprType = std::decay_t<XprType_>;
    using Scalar = std::remove_cv_t<typename XprType::Scalar>;
    const XprType_& value = matrix.derived();
    internals::validate_finite_symmetric(value);
    const EVD<XprType_> evd(value);
    internals::require_computed(evd);
    internals::validate_positive_spectrum(evd.eigenvalues(), value.rows());
    const auto eigenvalues = internals::transform_eigenvalues(evd, value.rows(), [](auto x) { return std::sqrt(x); });
    return SPDMatrix<Scalar, XprType::Rows, XprType::Cols>::from_spectral_(
      internals::reconstruct_symmetric(evd, value.rows(), eigenvalues));
}

/// @brief returns the checked SPD inverse principal square root of a numerically positive-definite expression
template <typename XprType_> auto matrix_inverse_sqrt(const SPDMatrixExpr<XprType_>& matrix) {
    using XprType = std::decay_t<XprType_>;
    using Scalar = std::remove_cv_t<typename XprType::Scalar>;
    const XprType_& value = matrix.derived();
    internals::validate_finite_symmetric(value);
    const EVD<XprType_> evd(value);
    internals::require_computed(evd);
    internals::validate_positive_spectrum(evd.eigenvalues(), value.rows());
    const auto eigenvalues =
      internals::transform_eigenvalues(evd, value.rows(), [](auto x) { return Scalar(1) / std::sqrt(x); });
    internals::validate_positive_spectrum(eigenvalues, value.rows());
    return SPDMatrix<Scalar, XprType::Rows, XprType::Cols>::from_spectral_(
      internals::reconstruct_symmetric(evd, value.rows(), eigenvalues));
}

/// @brief returns the owned symmetric Frechet derivative of log at an SPD point along a symmetric direction
template <typename XprType_, typename DirectionXprType_>
auto matrix_log_frechet(
  const SPDMatrixExpr<XprType_>& matrix, const SymmetricMatrixExpr<DirectionXprType_>& direction) {
    const XprType_& value = matrix.derived();
    internals::validate_finite_symmetric(value);
    const EVD<XprType_> evd(value);
    internals::require_computed(evd);
    internals::validate_positive_spectrum(evd.eigenvalues(), value.rows());
    return internals::frechet_symmetric(
      evd, value.rows(), direction.derived(), [](auto x, auto y) { return internals::log_divided_difference(x, y); });
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
