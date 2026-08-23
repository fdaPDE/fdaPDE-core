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

#ifndef __FDAPDE_LINALG_PRECONDITIONERS_H__
#define __FDAPDE_LINALG_PRECONDITIONERS_H__

#include <cmath>
#include <stdexcept>
#include <type_traits>

#include "header_check.h"

namespace fdapde {

template <typename XprType_> class IdentityPreconditioner {
    using XprType = std::decay_t<XprType_>;
    fdapde_static_assert(
      XprType::Rows == Dynamic || XprType::Cols == Dynamic || XprType::Rows == XprType::Cols,
      THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
   public:
    using Scalar = std::remove_cv_t<typename XprType::Scalar>;
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;

    constexpr IdentityPreconditioner() = default;
    template <typename MatrixType> constexpr explicit IdentityPreconditioner(const MatrixExpr<MatrixType>& matrix) {
        compute(matrix);
    }

    template <typename MatrixType> constexpr void compute(const MatrixExpr<MatrixType>& matrix) {
        fdapde_static_assert(
          MatrixType::Rows == Dynamic || Rows == Dynamic || MatrixType::Rows == Rows,
          INVALID_PRECONDITIONER_MATRIX_STATIC_SHAPE);
        fdapde_static_assert(
          MatrixType::Cols == Dynamic || Cols == Dynamic || MatrixType::Cols == Cols,
          INVALID_PRECONDITIONER_MATRIX_STATIC_SHAPE);
        reset_();
        const int n = matrix.rows();
        if (n <= 0 || n != matrix.cols() || (Rows != Dynamic && n != Rows) || (Cols != Dynamic && n != Cols)) {
            throw std::invalid_argument(
              "IdentityPreconditioner requires a nonempty square matrix matching its static shape");
        }
        size_ = n;
        valid_ = true;
    }

    template <typename RhsType> constexpr auto solve(const MatrixExpr<RhsType>& rhs) const {
        fdapde_static_assert(
          RhsType::Rows == Dynamic || Rows == Dynamic || RhsType::Rows == Rows,
          INVALID_PRECONDITIONER_RHS_STATIC_SHAPE);
        if (!valid_) { throw std::domain_error("IdentityPreconditioner solve requires a valid preconditioner"); }
        if (rhs.rows() != size_ || rhs.cols() <= 0) {
            throw std::invalid_argument("IdentityPreconditioner solve requires a matching nonempty right-hand side");
        }
        return Matrix<Scalar, RhsType::Rows, RhsType::Cols>(rhs);
    }

    constexpr bool valid() const { return valid_; }
   private:
    constexpr void reset_() {
        size_ = 0;
        valid_ = false;
    }

    int size_ = 0;
    bool valid_ = false;
};

template <typename XprType_> class DiagonalPreconditioner {
    using XprType = std::decay_t<XprType_>;
    fdapde_static_assert(
      XprType::Rows == Dynamic || XprType::Cols == Dynamic || XprType::Rows == XprType::Cols,
      THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
   public:
    using Scalar = std::remove_cv_t<typename XprType::Scalar>;
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    fdapde_static_assert(std::is_floating_point_v<Scalar>, PRECONDITIONERS_REQUIRE_FLOATING_POINT_SCALARS);

    constexpr DiagonalPreconditioner() = default;
    template <typename MatrixType> constexpr explicit DiagonalPreconditioner(const MatrixExpr<MatrixType>& matrix) {
        compute(matrix);
    }

    template <typename MatrixType> constexpr void compute(const MatrixExpr<MatrixType>& matrix) {
        fdapde_static_assert(
          MatrixType::Rows == Dynamic || Rows == Dynamic || MatrixType::Rows == Rows,
          INVALID_PRECONDITIONER_MATRIX_STATIC_SHAPE);
        fdapde_static_assert(
          MatrixType::Cols == Dynamic || Cols == Dynamic || MatrixType::Cols == Cols,
          INVALID_PRECONDITIONER_MATRIX_STATIC_SHAPE);
        reset_();
        const int n = matrix.rows();
        if (n <= 0 || n != matrix.cols() || (Rows != Dynamic && n != Rows) || (Cols != Dynamic && n != Cols)) {
            throw std::invalid_argument(
              "DiagonalPreconditioner requires a nonempty square matrix matching its static shape");
        }
        if constexpr (Rows == Dynamic) inverse_.resize(n);
        for (int i = 0; i < n; ++i) {
            const Scalar diagonal = static_cast<Scalar>(matrix.derived()(i, i));
            if (!std::isfinite(diagonal)) {
                reset_();
                throw std::invalid_argument("DiagonalPreconditioner requires finite diagonal coefficients");
            }
            if (diagonal == Scalar(0)) {
                reset_();
                throw std::domain_error("DiagonalPreconditioner requires nonzero diagonal coefficients");
            }
            const Scalar inverse = Scalar(1) / diagonal;
            if (!std::isfinite(inverse)) {
                reset_();
                throw std::domain_error("DiagonalPreconditioner diagonal reciprocal is not finite");
            }
            inverse_[i] = inverse;
        }
        size_ = n;
        valid_ = true;
    }

    template <typename RhsType> constexpr auto solve(const MatrixExpr<RhsType>& rhs) const {
        fdapde_static_assert(
          RhsType::Rows == Dynamic || Rows == Dynamic || RhsType::Rows == Rows,
          INVALID_PRECONDITIONER_RHS_STATIC_SHAPE);
        if (!valid_) { throw std::domain_error("DiagonalPreconditioner solve requires a valid preconditioner"); }
        if (rhs.rows() != size_ || rhs.cols() <= 0) {
            throw std::invalid_argument("DiagonalPreconditioner solve requires a matching nonempty right-hand side");
        }
        Matrix<Scalar, RhsType::Rows, RhsType::Cols> result(rhs);
        for (int row = 0; row < result.rows(); ++row) {
            for (int col = 0; col < result.cols(); ++col) result(row, col) *= inverse_[row];
        }
        return result;
    }

    constexpr bool valid() const { return valid_; }
   private:
    constexpr void reset_() {
        if constexpr (Rows == Dynamic) {
            inverse_.resize(0);
        } else {
            inverse_.set_zero();
        }
        size_ = 0;
        valid_ = false;
    }

    Vector<Scalar, Rows> inverse_;
    int size_ = 0;
    bool valid_ = false;
};

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_PRECONDITIONERS_H__
