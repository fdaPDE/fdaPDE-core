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

#ifndef __FDAPDE_LINALG_NATIVE_SPD_H__
#define __FDAPDE_LINALG_NATIVE_SPD_H__

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <ostream>
#include <stdexcept>
#include <type_traits>

#include "header_check.h"

namespace fdapde::linalg {

template <typename XprType_> struct SPDMatrixExpr : public SymmetricMatrixExpr<XprType_> {
    using XprType = std::decay_t<XprType_>;
    using SymmetricMatrixExpr<XprType_>::derived;
};

namespace internals {

template <typename VectorType_> void validate_positive_spectrum(const VectorType_& eigenvalues, int dimension) {
    using Scalar = std::remove_cv_t<typename VectorType_::Scalar>;
    Scalar minimum = std::numeric_limits<Scalar>::max();
    Scalar maximum = std::numeric_limits<Scalar>::lowest();
    for (int i = 0; i < dimension; ++i) {
        const Scalar eigenvalue = static_cast<Scalar>(eigenvalues[i]);
        if (!std::isfinite(eigenvalue)) { throw std::domain_error("SPDMatrix: eigendecomposition failed"); }
        minimum = std::min(minimum, eigenvalue);
        maximum = std::max(maximum, eigenvalue);
    }
    const Scalar threshold = Scalar(64) * Scalar(dimension) * std::numeric_limits<Scalar>::epsilon() * maximum;
    if (!(maximum > Scalar(0)) || minimum <= threshold) {
        throw std::domain_error("SPDMatrix: matrix is not numerically positive definite");
    }
}

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

    spd_matrix_impl() = delete;
    spd_matrix_impl(int, int) = delete;

    spd_matrix_impl(const spd_matrix_impl&) = default;
    spd_matrix_impl& operator=(const spd_matrix_impl&) & = default;
    void operator=(const spd_matrix_impl&) && = delete;

    template <typename RhsXprType_>
    explicit spd_matrix_impl(const MatrixExpr<RhsXprType_>& rhs, checked_t) :
        Base(), data_(make_storage_(rhs.derived())) {
        validate_spd_(rhs.derived(), data_);
    }

    template <typename RhsXprType_>
    explicit spd_matrix_impl(const MatrixExpr<RhsXprType_>& rhs, unchecked_t) :
        Base(), data_(make_storage_(rhs.derived())) { }

    template <typename RhsXprType_> spd_matrix_impl& assign(const MatrixExpr<RhsXprType_>& rhs, checked_t) & {
        StorageType candidate = make_storage_(rhs.derived());
        validate_spd_(rhs.derived(), candidate);
        data_ = candidate;
        return *this;
    }

    constexpr int rows() const { return data_.rows(); }
    constexpr int cols() const { return data_.cols(); }
    constexpr Scalar operator()(int i, int j) const { return static_cast<Scalar>(data_(i, j)); }
    constexpr const StorageType& rep() const& { return data_; }
    constexpr void rep() const&& = delete;
    constexpr const Scalar* data() const& { return data_.data(); }
    constexpr void data() const&& = delete;

    friend std::ostream& operator<<(std::ostream& os, const spd_matrix_impl& matrix) {
        os << matrix.data_;
        return os;
    }
   private:
    template <typename RhsXprType_> static void validate_shape_(const RhsXprType_& rhs) {
        if (rhs.rows() <= 0 || rhs.rows() != rhs.cols()) {
            throw std::invalid_argument("SPDMatrix: expected a nonempty square matrix");
        }
        if constexpr (Rows != Dynamic) {
            if (rhs.rows() != Rows) { throw std::invalid_argument("SPDMatrix: incompatible matrix dimensions"); }
        }
        const std::int64_t dimension = rhs.rows();
        const std::int64_t dense_size = dimension * dimension;
        if (dense_size > std::numeric_limits<int>::max()) {
            throw std::length_error("SPDMatrix: dense workspace size exceeds supported range");
        }
    }

    template <typename RhsXprType_> static StorageType make_storage_(const RhsXprType_& rhs) {
        validate_shape_(rhs);
        StorageType storage;
        if constexpr (Rows == Dynamic) { storage.resize(rhs.rows(), rhs.cols()); }
        for (int i = 0; i < rhs.rows(); ++i) {
            for (int j = 0; j <= i; ++j) { storage(i, j) = static_cast<Scalar>(rhs(i, j)); }
        }
        return storage;
    }

    template <typename RhsXprType_> static void validate_spd_(const RhsXprType_& rhs, const StorageType& storage) {
        Scalar scale = Scalar(0);
        for (int i = 0; i < rhs.rows(); ++i) {
            for (int j = 0; j < rhs.cols(); ++j) {
                const Scalar value = static_cast<Scalar>(rhs(i, j));
                if (!std::isfinite(value)) { throw std::invalid_argument("SPDMatrix: coefficients must be finite"); }
                scale = std::max(scale, std::abs(value));
            }
        }

        const Scalar tolerance = Scalar(32) * Scalar(rhs.rows()) * std::numeric_limits<Scalar>::epsilon() * scale;
        for (int i = 0; i < rhs.rows(); ++i) {
            for (int j = 0; j < i; ++j) {
                const Scalar difference = std::abs(static_cast<Scalar>(rhs(i, j)) - static_cast<Scalar>(rhs(j, i)));
                if (!std::isfinite(difference) || difference > tolerance) {
                    throw std::invalid_argument("SPDMatrix: matrix must be symmetric");
                }
            }
        }

        const EVD<StorageType> evd(storage);
        if (!evd.computed()) { throw std::domain_error("SPDMatrix: eigendecomposition failed"); }
        validate_positive_spectrum(evd.eigenvalues(), rhs.rows());
    }

    StorageType data_;
};

template <typename XprType_> void validate_finite_symmetric(const XprType_& matrix) {
    using Scalar = std::remove_cv_t<typename XprType_::Scalar>;
    if (matrix.rows() <= 0 || matrix.rows() != matrix.cols()) {
        throw std::invalid_argument("SPD spectral operation: expected a nonempty square matrix");
    }
    const std::int64_t dimension = matrix.rows();
    if (dimension * dimension > std::numeric_limits<int>::max()) {
        throw std::length_error("SPD spectral operation: dense workspace size exceeds supported range");
    }
    for (int i = 0; i < matrix.rows(); ++i) {
        for (int j = 0; j < matrix.cols(); ++j) {
            if (!std::isfinite(static_cast<Scalar>(matrix(i, j)))) {
                throw std::invalid_argument("SPD spectral operation: coefficients must be finite");
            }
        }
    }
}

template <typename XprType_> void require_computed(const EVD<XprType_>& evd) {
    if (!evd.computed()) { throw std::domain_error("SPD spectral operation: eigendecomposition failed"); }
}

template <typename XprType_, typename UnaryOp_>
auto transform_eigenvalues(const EVD<XprType_>& evd, int dimension, UnaryOp_&& op) {
    using XprType = std::decay_t<XprType_>;
    using Scalar = std::remove_cv_t<typename XprType::Scalar>;
    constexpr int Rows = XprType::Rows;
    Vector<Scalar, Rows> values;
    if constexpr (Rows == Dynamic) { values.resize(dimension); }
    for (int i = 0; i < dimension; ++i) {
        values[i] = op(evd.eigenvalues()[i]);
        if (!std::isfinite(values[i])) { throw std::domain_error("SPD spectral operation: nonfinite result"); }
    }
    return values;
}

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
            if (!std::isfinite(value)) { throw std::domain_error("SPD spectral operation: nonfinite result"); }
            result(i, j) = value;
        }
    }
    return result;
}

template <typename Scalar_> Scalar_ log_divided_difference(Scalar_ x, Scalar_ y) {
    if (x == y) return Scalar_(1) / x;
    const Scalar_ delta = x - y;
    if (std::abs(delta) <= Scalar_(0.5) * std::min(x, y)) {
        return Scalar_(0.5) * (std::log1p(delta / y) / delta + std::log1p(-delta / x) / -delta);
    }
    return (std::log(x) - std::log(y)) / delta;
}

template <typename Scalar_> Scalar_ exp_divided_difference(Scalar_ x, Scalar_ y) {
    if (x == y) return std::exp(x);
    if (x > y) return std::exp(x) * (-std::expm1(y - x)) / (x - y);
    return std::exp(y) * (-std::expm1(x - y)) / (y - x);
}

template <typename XprType_, typename DirectionXprType_, typename DividedDifference_>
auto frechet_symmetric(
  const EVD<XprType_>& evd, int dimension, const DirectionXprType_& direction,
  DividedDifference_&& divided_difference) {
    using XprType = std::decay_t<XprType_>;
    using Scalar = std::remove_cv_t<typename XprType::Scalar>;
    constexpr int Rows = XprType::Rows;
    constexpr int Cols = XprType::Cols;
    if (direction.rows() != dimension || direction.cols() != dimension) {
        throw std::invalid_argument("SPD spectral operation: incompatible direction dimensions");
    }
    validate_finite_symmetric(direction);

    Matrix<Scalar, Rows, Cols> hq;
    Matrix<Scalar, Rows, Cols> coefficients;
    Matrix<Scalar, Rows, Cols> q_coefficients;
    if constexpr (Rows == Dynamic) {
        hq.resize(dimension, dimension);
        coefficients.resize(dimension, dimension);
        q_coefficients.resize(dimension, dimension);
    }

    const auto eigenvectors = evd.eigenvectors();
    for (int i = 0; i < dimension; ++i) {
        for (int j = 0; j < dimension; ++j) {
            Scalar value = Scalar(0);
            for (int k = 0; k < dimension; ++k) { value += static_cast<Scalar>(direction(i, k)) * eigenvectors(k, j); }
            hq(i, j) = value;
        }
    }
    for (int i = 0; i < dimension; ++i) {
        for (int j = 0; j < dimension; ++j) {
            Scalar value = Scalar(0);
            for (int k = 0; k < dimension; ++k) { value += eigenvectors(k, i) * hq(k, j); }
            coefficients(i, j) = divided_difference(evd.eigenvalues()[i], evd.eigenvalues()[j]) * value;
            if (!std::isfinite(coefficients(i, j))) {
                throw std::domain_error("SPD spectral operation: nonfinite Frechet derivative");
            }
        }
    }
    for (int i = 0; i < dimension; ++i) {
        for (int j = 0; j < dimension; ++j) {
            Scalar value = Scalar(0);
            for (int k = 0; k < dimension; ++k) { value += eigenvectors(i, k) * coefficients(k, j); }
            q_coefficients(i, j) = value;
        }
    }

    SymmetricMatrix<Scalar, Rows, Cols> result;
    if constexpr (Rows == Dynamic) { result.resize(dimension, dimension); }
    for (int i = 0; i < dimension; ++i) {
        for (int j = 0; j <= i; ++j) {
            Scalar value = Scalar(0);
            for (int k = 0; k < dimension; ++k) { value += q_coefficients(i, k) * eigenvectors(j, k); }
            if (!std::isfinite(value)) {
                throw std::domain_error("SPD spectral operation: nonfinite Frechet derivative");
            }
            result(i, j) = value;
        }
    }
    return result;
}

}   // namespace internals

template <typename Scalar_, int Rows_, int Cols_, int StorageOrder_ = RowMajor>
using SPDMatrix = internals::spd_matrix_impl<Scalar_, Rows_, Cols_, StorageOrder_>;

template <typename XprType_> auto matrix_log(const SPDMatrixExpr<XprType_>& matrix) {
    const XprType_& value = matrix.derived();
    internals::validate_finite_symmetric(value);
    const EVD<XprType_> evd(value);
    internals::require_computed(evd);
    internals::validate_positive_spectrum(evd.eigenvalues(), value.rows());
    const auto eigenvalues = internals::transform_eigenvalues(evd, value.rows(), [](auto x) { return std::log(x); });
    return internals::reconstruct_symmetric(evd, value.rows(), eigenvalues);
}

template <typename XprType_> auto matrix_exp(const SymmetricMatrixExpr<XprType_>& matrix) {
    using XprType = std::decay_t<XprType_>;
    using Scalar = std::remove_cv_t<typename XprType::Scalar>;
    const XprType_& value = matrix.derived();
    internals::validate_finite_symmetric(value);
    const EVD<XprType_> evd(value);
    internals::require_computed(evd);
    const auto eigenvalues = internals::transform_eigenvalues(evd, value.rows(), [](auto x) { return std::exp(x); });
    internals::validate_positive_spectrum(eigenvalues, value.rows());
    return SPDMatrix<Scalar, XprType::Rows, XprType::Cols>(
      internals::reconstruct_symmetric(evd, value.rows(), eigenvalues), checked);
}

template <typename XprType_> auto matrix_sqrt(const SPDMatrixExpr<XprType_>& matrix) {
    using XprType = std::decay_t<XprType_>;
    using Scalar = std::remove_cv_t<typename XprType::Scalar>;
    const XprType_& value = matrix.derived();
    internals::validate_finite_symmetric(value);
    const EVD<XprType_> evd(value);
    internals::require_computed(evd);
    internals::validate_positive_spectrum(evd.eigenvalues(), value.rows());
    const auto eigenvalues = internals::transform_eigenvalues(evd, value.rows(), [](auto x) { return std::sqrt(x); });
    return SPDMatrix<Scalar, XprType::Rows, XprType::Cols>(
      internals::reconstruct_symmetric(evd, value.rows(), eigenvalues), checked);
}

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
    return SPDMatrix<Scalar, XprType::Rows, XprType::Cols>(
      internals::reconstruct_symmetric(evd, value.rows(), eigenvalues), checked);
}

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

template <typename XprType_> struct is_spd_matrix {
    using XprType = std::remove_cvref_t<XprType_>;
    static constexpr bool value = std::is_base_of_v<SPDMatrixExpr<XprType>, XprType>;
};
template <typename XprType> inline constexpr bool is_spd_matrix_v = is_spd_matrix<XprType>::value;

}   // namespace fdapde::linalg

#endif   // __FDAPDE_LINALG_NATIVE_SPD_H__
