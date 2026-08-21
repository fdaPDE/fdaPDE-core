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

#ifndef __FDAPDE_LINALG_PARTIAL_PIV_LU_H__
#define __FDAPDE_LINALG_PARTIAL_PIV_LU_H__

#include "header_check.h"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace fdapde {

// Doolittle LU factorization with partial row pivoting.
template <typename XprType_> class PartialPivLU {
    using XprType = std::decay_t<XprType_>;
    fdapde_static_assert(
      XprType::Rows == Dynamic || XprType::Cols == Dynamic || XprType::Rows == XprType::Cols,
      THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
   public:
    using Scalar = std::remove_cv_t<typename XprType::Scalar>;
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    fdapde_static_assert(std::is_floating_point_v<Scalar>, LU_DECOMPOSITION_REQUIRES_FLOATING_POINT_SCALARS);

    constexpr PartialPivLU() = default;
    template <typename MatrixType> constexpr explicit PartialPivLU(const MatrixExpr<MatrixType>& matrix) {
        compute(matrix);
    }

    template <typename MatrixType> constexpr void compute(const MatrixExpr<MatrixType>& matrix) {
        fdapde_static_assert(
          MatrixType::Rows == Dynamic || Rows == Dynamic || MatrixType::Rows == Rows, INVALID_LU_MATRIX_STATIC_SHAPE);
        fdapde_static_assert(
          MatrixType::Cols == Dynamic || Cols == Dynamic || MatrixType::Cols == Cols, INVALID_LU_MATRIX_STATIC_SHAPE);
        const int n = matrix.rows();
        const bool shape_valid =
          n > 0 && n == matrix.cols() && (Rows == Dynamic || n == Rows) && (Cols == Dynamic || n == Cols);
        if (!shape_valid) {
            reset_();
            throw std::invalid_argument("PartialPivLU requires a nonempty square matrix matching its static shape");
        }

        determinant_ = Scalar(0);
        determinant_computed_ = false;
        Matrix<Scalar, Rows, Cols> normalized(matrix);
        Scalar scale = Scalar(0);
        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
                const Scalar value = normalized(row, col);
                if (!is_finite_(value)) {
                    reset_();
                    throw std::invalid_argument("PartialPivLU requires finite matrix coefficients");
                }
                scale = fdapde::max(scale, fdapde::abs(value));
            }
        }

        determinant_ = algebraic_determinant_(matrix.derived());
        determinant_computed_ = true;
        scale_ = scale == Scalar(0) ? Scalar(1) : scale;
        normalized /= scale_;
        Matrix<Scalar, Rows, Cols> lu(normalized);
        const Scalar threshold = std::numeric_limits<Scalar>::epsilon() * static_cast<Scalar>(n);

        Vector<int, Rows> permutation;
        if constexpr (Rows == Dynamic) permutation.resize(n);
        for (int i = 0; i < n; ++i) permutation[i] = i;

        info_ = 0;
        rank_ = 0;
        for (int k = 0; k < n; ++k) {
            int pivot_index = k;
            Scalar max_value = Scalar(0);
            for (int row = k; row < n; ++row) {
                const Scalar candidate = fdapde::abs(lu(row, k));
                if (candidate > max_value) {
                    max_value = candidate;
                    pivot_index = row;
                }
            }

            if (!(max_value > threshold)) {
                if (info_ == 0) info_ = k + 1;
            } else {
                ++rank_;
            }
            if (max_value == Scalar(0)) continue;

            if (pivot_index != k) {
                for (int col = 0; col < n; ++col) std::swap(lu(k, col), lu(pivot_index, col));
                std::swap(permutation[k], permutation[pivot_index]);
            }
            for (int row = k + 1; row < n; ++row) {
                const Scalar multiplier = lu(row, k) / lu(k, k);
                if (!is_finite_(multiplier)) {
                    reset_factors_();
                    return;
                }
                lu(row, k) = multiplier;
                for (int col = k + 1; col < n; ++col) {
                    lu(row, col) -= multiplier * lu(k, col);
                    if (!is_finite_(lu(row, col))) {
                        reset_factors_();
                        return;
                    }
                }
            }
        }

        if (info_ != 0) rank_ = numerical_rank_(normalized, threshold);
        L_ = lu.template triangular_block<Lower>();
        for (int i = 0; i < n; ++i) L_(i, i) = Scalar(1);
        normalized_U_ = lu.template triangular_block<Upper>();
        U_ = normalized_U_;
        U_ *= scale_;
        P_ = PermutationMatrix<Rows, Cols>(permutation);
    }

    constexpr const PermutationMatrix<Rows, Cols>& P() const & { return P_; }
    constexpr void P() const && = delete;
    constexpr const LowerTriangularMatrix<Scalar, Rows, Cols>& L() const & { return L_; }
    constexpr void L() const && = delete;
    // U is reported in the input scale. It can overflow even when the normalized solve remains usable.
    constexpr const UpperTriangularMatrix<Scalar, Rows, Cols>& U() const & { return U_; }
    constexpr void U() const && = delete;
    // -1: factors unavailable, 0: success, >0: first unusable pivot (one based).
    constexpr int info() const { return info_; }
    constexpr int rank() const { return rank_; }

    constexpr Scalar determinant() const {
        if (!determinant_computed_) { throw std::domain_error("PartialPivLU determinant is unavailable"); }
        return determinant_;
    }

    template <typename RhsType> constexpr auto solve(const MatrixExpr<RhsType>& rhs) const {
        fdapde_static_assert(
          RhsType::Rows == Dynamic || Rows == Dynamic || RhsType::Rows == Rows, INVALID_LU_RHS_STATIC_SHAPE);
        if (rhs.rows() != L_.rows() || rhs.cols() <= 0) {
            throw std::invalid_argument("PartialPivLU solve requires a matching nonempty right-hand side");
        }
        if (info_ != 0) { throw std::domain_error("PartialPivLU solve requires a nonsingular factorization"); }

        Matrix<Scalar, RhsType::Rows, RhsType::Cols> solution(rhs);
        for (int row = 0; row < solution.rows(); ++row) {
            for (int col = 0; col < solution.cols(); ++col) solution(row, col) = Scalar(0);
        }
        Matrix<Scalar, RhsType::Rows, RhsType::Cols> permuted(P_ * rhs);
        permuted /= scale_;
        Matrix<Scalar, RhsType::Rows, RhsType::Cols> forward(permuted);
        const int n = L_.rows();
        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < rhs.cols(); ++col) {
                for (int k = 0; k < row; ++k) forward(row, col) -= L_(row, k) * forward(k, col);
            }
        }
        for (int row = n - 1; row >= 0; --row) {
            for (int col = 0; col < rhs.cols(); ++col) {
                Scalar value = forward(row, col);
                for (int k = row + 1; k < n; ++k) value -= normalized_U_(row, k) * solution(k, col);
                solution(row, col) = value / normalized_U_(row, row);
            }
        }
        return solution;
    }
   private:
    struct ScaledValue {
        Scalar significand = Scalar(0);
        long long exponent = 0;
    };

    static constexpr bool is_finite_(Scalar value) {
        const Scalar infinity = std::numeric_limits<Scalar>::infinity();
        return value == value && value != infinity && value != -infinity;
    }

    static constexpr ScaledValue normalize_(Scalar significand, long long exponent) {
        if (significand == Scalar(0)) return {};
        int shift = 0;
        const Scalar normalized = std::frexp(significand, &shift);
        return {normalized, exponent + shift};
    }

    static constexpr ScaledValue scaled_value_(Scalar value) { return normalize_(value, 0); }

    static constexpr bool abs_greater_(const ScaledValue& lhs, const ScaledValue& rhs) {
        if (lhs.significand == Scalar(0)) return false;
        if (rhs.significand == Scalar(0)) return true;
        if (lhs.exponent != rhs.exponent) return lhs.exponent > rhs.exponent;
        return fdapde::abs(lhs.significand) > fdapde::abs(rhs.significand);
    }

    static constexpr ScaledValue multiply_(const ScaledValue& lhs, const ScaledValue& rhs) {
        if (lhs.significand == Scalar(0) || rhs.significand == Scalar(0)) return {};
        return normalize_(lhs.significand * rhs.significand, lhs.exponent + rhs.exponent);
    }

    static constexpr ScaledValue divide_(const ScaledValue& lhs, const ScaledValue& rhs) {
        return normalize_(lhs.significand / rhs.significand, lhs.exponent - rhs.exponent);
    }

    static constexpr ScaledValue subtract_(const ScaledValue& lhs, const ScaledValue& rhs) {
        if (rhs.significand == Scalar(0)) return lhs;
        if (lhs.significand == Scalar(0)) return {-rhs.significand, rhs.exponent};
        constexpr long long insignificant = -std::numeric_limits<Scalar>::digits - 2;
        if (lhs.exponent >= rhs.exponent) {
            const long long shift = rhs.exponent - lhs.exponent;
            if (shift < insignificant) return lhs;
            return normalize_(lhs.significand - std::ldexp(rhs.significand, static_cast<int>(shift)), lhs.exponent);
        }
        const long long shift = lhs.exponent - rhs.exponent;
        if (shift < insignificant) return {-rhs.significand, rhs.exponent};
        return normalize_(std::ldexp(lhs.significand, static_cast<int>(shift)) - rhs.significand, rhs.exponent);
    }

    static constexpr Scalar materialize_(const ScaledValue& value) {
        if (value.significand == Scalar(0)) return Scalar(0);
        if (value.exponent > std::numeric_limits<int>::max()) {
            return std::copysign(std::numeric_limits<Scalar>::infinity(), value.significand);
        }
        if (value.exponent < std::numeric_limits<int>::min()) {
            return std::copysign(Scalar(0), value.significand);
        }
        return std::ldexp(value.significand, static_cast<int>(value.exponent));
    }

    template <typename MatrixType> static constexpr Scalar constexpr_determinant_(const MatrixType& matrix) {
        Matrix<Scalar, Rows, Cols> work(matrix);
        Scalar determinant = Scalar(1);
        const int n = matrix.rows();
        for (int col = 0; col < n; ++col) {
            int pivot_row = col;
            int pivot_col = col;
            for (int row = col; row < n; ++row) {
                for (int candidate_col = col; candidate_col < n; ++candidate_col) {
                    if (fdapde::abs(work(row, candidate_col)) > fdapde::abs(work(pivot_row, pivot_col))) {
                        pivot_row = row;
                        pivot_col = candidate_col;
                    }
                }
            }
            if (work(pivot_row, pivot_col) == Scalar(0)) return Scalar(0);
            if (pivot_row != col) {
                for (int j = 0; j < n; ++j) std::swap(work(col, j), work(pivot_row, j));
                determinant = -determinant;
            }
            if (pivot_col != col) {
                for (int row = 0; row < n; ++row) std::swap(work(row, col), work(row, pivot_col));
                determinant = -determinant;
            }
            const Scalar pivot = work(col, col);
            determinant *= pivot;
            for (int row = col + 1; row < n; ++row) {
                const Scalar multiplier = work(row, col) / pivot;
                for (int j = col + 1; j < n; ++j) work(row, j) -= multiplier * work(col, j);
            }
        }
        return determinant;
    }

    // Exponent-tracked complete-pivot elimination avoids intermediate range loss for runtime determinants.
    template <typename MatrixType> static constexpr Scalar algebraic_determinant_(const MatrixType& matrix) {
        if (std::is_constant_evaluated()) return constexpr_determinant_(matrix);

        const int n = matrix.rows();
        const std::size_t size = static_cast<std::size_t>(n);
        std::vector<ScaledValue> work(size * size);
        const auto index = [size](int row, int col) {
            return static_cast<std::size_t>(row) * size + static_cast<std::size_t>(col);
        };
        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) work[index(row, col)] = scaled_value_(matrix(row, col));
        }

        ScaledValue determinant = scaled_value_(Scalar(1));
        for (int col = 0; col < n; ++col) {
            int pivot_row = col;
            int pivot_col = col;
            for (int row = col; row < n; ++row) {
                for (int candidate_col = col; candidate_col < n; ++candidate_col) {
                    if (abs_greater_(work[index(row, candidate_col)], work[index(pivot_row, pivot_col)])) {
                        pivot_row = row;
                        pivot_col = candidate_col;
                    }
                }
            }
            if (work[index(pivot_row, pivot_col)].significand == Scalar(0)) return Scalar(0);
            if (pivot_row != col) {
                for (int j = 0; j < n; ++j) std::swap(work[index(col, j)], work[index(pivot_row, j)]);
                determinant.significand = -determinant.significand;
            }
            if (pivot_col != col) {
                for (int row = 0; row < n; ++row) std::swap(work[index(row, col)], work[index(row, pivot_col)]);
                determinant.significand = -determinant.significand;
            }
            const ScaledValue pivot = work[index(col, col)];
            determinant = multiply_(determinant, pivot);
            for (int row = col + 1; row < n; ++row) {
                const ScaledValue multiplier = divide_(work[index(row, col)], pivot);
                for (int j = col + 1; j < n; ++j) {
                    work[index(row, j)] = subtract_(work[index(row, j)], multiply_(multiplier, work[index(col, j)]));
                }
            }
        }
        return materialize_(determinant);
    }

    template <typename MatrixType> static constexpr int numerical_rank_(const MatrixType& matrix, Scalar threshold) {
        Matrix<Scalar, Dynamic, Dynamic> echelon(matrix);
        int pivot_row = 0;
        for (int col = 0; col < echelon.cols() && pivot_row < echelon.rows(); ++col) {
            int pivot = pivot_row;
            Scalar max_value = Scalar(0);
            for (int row = pivot_row; row < echelon.rows(); ++row) {
                const Scalar candidate = fdapde::abs(echelon(row, col));
                if (candidate > max_value) {
                    max_value = candidate;
                    pivot = row;
                }
            }
            if (!(max_value > threshold)) continue;
            if (pivot != pivot_row) {
                for (int j = 0; j < echelon.cols(); ++j) std::swap(echelon(pivot_row, j), echelon(pivot, j));
            }
            for (int row = pivot_row + 1; row < echelon.rows(); ++row) {
                const Scalar multiplier = echelon(row, col) / echelon(pivot_row, col);
                for (int j = col; j < echelon.cols(); ++j) echelon(row, j) -= multiplier * echelon(pivot_row, j);
            }
            ++pivot_row;
        }
        return pivot_row;
    }

    constexpr void reset_factors_() {
        L_ = LowerTriangularMatrix<Scalar, Rows, Cols>();
        U_ = UpperTriangularMatrix<Scalar, Rows, Cols>();
        normalized_U_ = UpperTriangularMatrix<Scalar, Rows, Cols>();
        P_ = PermutationMatrix<Rows, Cols>();
        scale_ = Scalar(1);
        info_ = -1;
        rank_ = 0;
    }

    constexpr void reset_() {
        reset_factors_();
        determinant_ = Scalar(0);
        determinant_computed_ = false;
    }

    LowerTriangularMatrix<Scalar, Rows, Cols> L_;
    UpperTriangularMatrix<Scalar, Rows, Cols> U_;
    UpperTriangularMatrix<Scalar, Rows, Cols> normalized_U_;
    PermutationMatrix<Rows, Cols> P_;
    Scalar scale_ = Scalar(1);
    Scalar determinant_ = Scalar(0);
    bool determinant_computed_ = false;
    int info_ = -1;
    int rank_ = 0;
};

template <typename XprType> PartialPivLU(const MatrixExpr<XprType>&) -> PartialPivLU<XprType>;

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_PARTIAL_PIV_LU_H__
