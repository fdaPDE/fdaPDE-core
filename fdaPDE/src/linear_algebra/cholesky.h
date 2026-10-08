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

#ifndef __FDAPDE_LINALG_CHOLESKY_H__
#define __FDAPDE_LINALG_CHOLESKY_H__

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>

#include "header_check.h"

namespace fdapde {
namespace internals {

/// @brief permits Cholesky cache reuse only from certified native SPD owners and views
template <typename T> struct is_cholesky_cache_source : std::false_type { };

/// @brief fills a lower factor using extended accumulation and rejects nonpositive or nonfinite pivots
template <typename Coefficients, typename Factor>
void compute_cholesky_factor(const Coefficients& coefficients, Factor& factor) {
    using Scalar = std::remove_cv_t<typename Factor::Scalar>;
    for (int i = 0; i < coefficients.rows(); ++i) {
        for (int j = 0; j <= i; ++j) {
            long double value = static_cast<Scalar>(coefficients(i, j));
            for (int k = 0; k < j; ++k) value -= static_cast<long double>(factor(i, k)) * factor(j, k);
            if (i == j) {
                fdapde_strong_assert(
                  value > 0 && std::isfinite(value), std::domain_error, "Cholesky requires positive finite pivots");
                factor(i, j) = static_cast<Scalar>(std::sqrt(value));
            } else {
                factor(i, j) = static_cast<Scalar>(value / factor(j, j));
            }
            fdapde_strong_assert(
              std::isfinite(factor(i, j)) && (i != j || factor(i, j) > Scalar(0)), std::domain_error,
              "Cholesky factor is not representable");
        }
    }
}

}   // namespace internals

/// @brief factors symmetric positive-definite matrices as L times L transpose with an owning lower factor
template <typename XprType_> class Cholesky {
    using XprType = std::decay_t<XprType_>;
    fdapde_static_assert(
      XprType::Rows == Dynamic || XprType::Cols == Dynamic || XprType::Rows == XprType::Cols,
      THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
   public:
    using Scalar = std::remove_cv_t<typename XprType::Scalar>;
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    static constexpr int Order = Rows == Dynamic ? Cols : Rows;
    fdapde_static_assert(std::is_floating_point_v<Scalar>, CHOLESKY_REQUIRES_FLOATING_POINT_SCALARS);
    fdapde_static_assert(
      Order == Dynamic || std::int64_t(Order) * Order <= std::numeric_limits<int>::max(),
      CHOLESKY_WORKSPACE_SIZE_EXCEEDS_SUPPORTED_RANGE);

    /// @brief creates an uncomputed decomposition with no available factor
    Cholesky() = default;
    /// @brief factors a finite symmetric positive-definite matrix or copies its retained native factor
    template <typename MatrixType> explicit Cholesky(const MatrixExpr<MatrixType>& matrix) { compute(matrix); }

    /// @brief replaces the factor after validating the input and hides previous factors if computation fails
    template <typename MatrixType> void compute(const MatrixExpr<MatrixType>& matrix) {
        computed_ = false;
        fdapde_static_assert(
          MatrixType::Rows == Dynamic || Rows == Dynamic || MatrixType::Rows == Rows,
          INVALID_CHOLESKY_MATRIX_STATIC_SHAPE);
        fdapde_static_assert(
          MatrixType::Cols == Dynamic || Cols == Dynamic || MatrixType::Cols == Cols,
          INVALID_CHOLESKY_MATRIX_STATIC_SHAPE);
        const int n = matrix.rows();
        fdapde_strong_assert(
          n > 0 && n == matrix.cols() && (Rows == Dynamic || n == Rows) && (Cols == Dynamic || n == Cols),
          std::invalid_argument, "Cholesky requires a nonempty square matrix matching its static shape");
        fdapde_strong_assert(
          std::int64_t(n) * n <= std::numeric_limits<int>::max(), std::length_error,
          "Cholesky workspace exceeds supported range");

        if constexpr (
          internals::is_cholesky_cache_source<std::remove_cvref_t<MatrixType>>::value &&
          std::same_as<Scalar, std::remove_cv_t<typename MatrixType::Scalar>>) {
            const LowerTriangularMatrix<Scalar, Order> candidate(matrix.derived().cache().cholesky());
            L_.emplace(candidate);
        } else {
            internals::validate_symmetric_coefficients(matrix.derived());
            LowerTriangularMatrix<Scalar, Order> candidate;
            if constexpr (Order == Dynamic) candidate.resize(n);
            internals::compute_cholesky_factor(matrix.derived(), candidate);
            L_.emplace(candidate);
        }
        computed_ = true;
    }

    /// @brief borrows the independent lower factor until recomputation or destruction
    const LowerTriangularMatrix<Scalar, Order>& L() const& {
        fdapde_strong_assert(computed_, std::logic_error, "Cholesky has not been computed");
        return *L_;
    }
    /// @brief prevents a factor reference from escaping a temporary decomposition
    void L() const&& = delete;
    /// @brief reports whether a complete factor is available
    bool computed() const { return computed_; }

    /// @brief solves one or more finite right-hand sides by forward and backward substitution
    template <typename Rhs> auto solve(const MatrixExpr<Rhs>& rhs) const {
        const auto& factor = L();
        fdapde_strong_assert(
          rhs.rows() == factor.rows() && rhs.cols() > 0, std::invalid_argument,
          "Cholesky solve requires matching nonempty right-hand sides");
        Matrix<Scalar, Order, Rhs::Cols> result;
        if constexpr (Order == Dynamic || Rhs::Cols == Dynamic) result.resize(rhs.rows(), rhs.cols());
        for (int col = 0; col < rhs.cols(); ++col) {
            for (int i = 0; i < rhs.rows(); ++i) {
                Scalar value = static_cast<Scalar>(rhs.derived()(i, col));
                fdapde_strong_assert(
                  std::isfinite(value), std::invalid_argument, "Cholesky solve requires finite input");
                for (int k = 0; k < i; ++k) value -= factor(i, k) * result(k, col);
                result(i, col) = value / factor(i, i);
            }
            for (int i = rhs.rows() - 1; i >= 0; --i) {
                Scalar value = result(i, col);
                for (int k = i + 1; k < rhs.rows(); ++k) value -= factor(k, i) * result(k, col);
                result(i, col) = value / factor(i, i);
                fdapde_strong_assert(
                  std::isfinite(result(i, col)), std::domain_error, "Cholesky solution is not representable");
            }
        }
        return result;
    }
   private:
    std::optional<LowerTriangularMatrix<Scalar, Order>> L_;
    bool computed_ = false;
};

template <typename XprType> Cholesky(const MatrixExpr<XprType>&) -> Cholesky<XprType>;

}   // namespace fdapde
#endif   // __FDAPDE_LINALG_CHOLESKY_H__
