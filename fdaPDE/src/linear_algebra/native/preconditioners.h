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

#ifndef __FDAPDE_LINALG_NATIVE_PRECONDITIONERS_H__
#define __FDAPDE_LINALG_NATIVE_PRECONDITIONERS_H__

#include <cmath>
#include <type_traits>

#include "header_check.h"

namespace fdapde::linalg {

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
        const bool valid = matrix.rows() > 0 && matrix.rows() == matrix.cols() &&
                           (Rows == Dynamic || matrix.rows() == Rows) && (Cols == Dynamic || matrix.cols() == Cols);
        if (!valid) fdapde_assert(valid);
        size_ = valid ? matrix.rows() : 0;
        valid_ = valid;
    }

    template <typename RhsType> constexpr auto solve(const MatrixExpr<RhsType>& rhs) const {
        fdapde_static_assert(
          RhsType::Rows == Dynamic || Rows == Dynamic || RhsType::Rows == Rows,
          INVALID_PRECONDITIONER_RHS_STATIC_SHAPE);
        Matrix<Scalar, RhsType::Rows, RhsType::Cols> result(rhs);
        const bool valid = valid_ && rhs.rows() == size_ && rhs.cols() > 0;
        if (!valid) {
            fdapde_assert(valid);
            result.set_zero();
        }
        return result;
    }

    constexpr bool valid() const { return valid_; }
   private:
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
        const bool shape_valid = matrix.rows() > 0 && matrix.rows() == matrix.cols() &&
                                 (Rows == Dynamic || matrix.rows() == Rows) &&
                                 (Cols == Dynamic || matrix.cols() == Cols);
        if constexpr (Rows == Dynamic) inverse_.resize(shape_valid ? matrix.rows() : 0);
        if (!shape_valid) {
            fdapde_assert(shape_valid);
            inverse_.set_zero();
            size_ = 0;
            valid_ = false;
            return;
        }

        size_ = matrix.rows();
        valid_ = true;
        for (int i = 0; i < size_; ++i) {
            const Scalar diagonal = static_cast<Scalar>(matrix.derived()(i, i));
            bool coefficient_valid = diagonal != Scalar(0) && std::isfinite(diagonal);
            const Scalar inverse = coefficient_valid ? Scalar(1) / diagonal : Scalar(0);
            coefficient_valid = coefficient_valid && std::isfinite(inverse);
            if (!coefficient_valid) fdapde_assert(coefficient_valid);
            inverse_[i] = coefficient_valid ? inverse : Scalar(0);
            valid_ = valid_ && coefficient_valid;
        }
    }

    template <typename RhsType> constexpr auto solve(const MatrixExpr<RhsType>& rhs) const {
        fdapde_static_assert(
          RhsType::Rows == Dynamic || Rows == Dynamic || RhsType::Rows == Rows,
          INVALID_PRECONDITIONER_RHS_STATIC_SHAPE);
        Matrix<Scalar, RhsType::Rows, RhsType::Cols> result(rhs);
        const bool valid = valid_ && rhs.rows() == size_ && rhs.cols() > 0;
        if (!valid) {
            fdapde_assert(valid);
            result.set_zero();
            return result;
        }
        for (int i = 0; i < result.rows(); ++i) {
            for (int j = 0; j < result.cols(); ++j) result(i, j) *= inverse_[i];
        }
        return result;
    }

    constexpr bool valid() const { return valid_; }
   private:
    Vector<Scalar, Rows> inverse_;
    int size_ = 0;
    bool valid_ = false;
};

}   // namespace fdapde::linalg

#endif   // __FDAPDE_LINALG_NATIVE_PRECONDITIONERS_H__
