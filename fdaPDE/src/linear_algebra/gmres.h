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

#ifndef __FDAPDE_LINALG_GMRES_H__
#define __FDAPDE_LINALG_GMRES_H__

#include <cmath>
#include <concepts>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

#include "header_check.h"

namespace fdapde {

/// @brief solves dense systems by left-preconditioned restarted GMRES with an owned operator
template <typename XprType_, typename Preconditioner_> class GMRES {
    using XprType = std::decay_t<XprType_>;
    fdapde_static_assert(
      XprType::Rows == Dynamic || XprType::Cols == Dynamic || XprType::Rows == XprType::Cols,
      THIS_CLASS_IS_FOR_SQUARE_MATRICES_ONLY);
   public:
    using Scalar = std::remove_cv_t<typename XprType::Scalar>;
    static constexpr int Rows = XprType::Rows;
    static constexpr int Cols = XprType::Cols;
    fdapde_static_assert(std::is_floating_point_v<Scalar>, GMRES_REQUIRES_FLOATING_POINT_SCALARS);

    /// @brief constructs solver settings with positive iteration limits and finite positive tolerance
    template <typename Preconditioner>
        requires(std::is_constructible_v<Preconditioner_, Preconditioner>)
    constexpr GMRES(Preconditioner&& preconditioner, int max_iterations, int restart, Scalar tolerance) :
        max_iterations_(max_iterations),
        restart_(restart),
        tolerance_(tolerance),
        preconditioner_(std::forward<Preconditioner>(preconditioner)) {
        fdapde_strong_assert(
          max_iterations_ > 0 && restart_ > 0 && tolerance_ > Scalar(0) && std::isfinite(tolerance_),
          std::invalid_argument, "GMRES requires positive iteration limits and a finite positive tolerance");
    }

    /// @brief uses defaults of 500 iterations, restart 50 and relative tolerance 1e-6
    template <typename Preconditioner>
        requires(std::is_constructible_v<Preconditioner_, Preconditioner>)
    constexpr explicit GMRES(Preconditioner&& preconditioner) :
        GMRES(std::forward<Preconditioner>(preconditioner), 500, 50, Scalar(1e-6)) { }

    /// @brief copies the matrix and initializes preconditioning with explicit solver settings
    template <typename MatrixType, typename Preconditioner>
        requires(
          std::is_same_v<XprType, std::decay_t<MatrixType>> && std::is_constructible_v<Preconditioner_, Preconditioner>)
    constexpr GMRES(
      const MatrixExpr<MatrixType>& matrix, Preconditioner&& preconditioner, int max_iterations, int restart,
      Scalar tolerance) :
        GMRES(std::forward<Preconditioner>(preconditioner), max_iterations, restart, tolerance) {
        compute(matrix);
    }

    /// @brief copies the matrix and initializes preconditioning with default settings
    template <typename MatrixType, typename Preconditioner>
        requires(
          std::is_same_v<XprType, std::decay_t<MatrixType>> && std::is_constructible_v<Preconditioner_, Preconditioner>)
    constexpr GMRES(const MatrixExpr<MatrixType>& matrix, Preconditioner&& preconditioner) :
        GMRES(matrix, std::forward<Preconditioner>(preconditioner), 500, 50, Scalar(1e-6)) { }

    /// @brief replaces the owned matrix and workspaces, invalidating the solver on failure
    template <typename MatrixType>
        requires(std::is_same_v<XprType, std::decay_t<MatrixType>>)
    constexpr void compute(const MatrixExpr<MatrixType>& matrix) {
        initialized_ = false;
        reset_observers_();
        const int n = matrix.rows();
        fdapde_strong_assert(
          n > 0 && n == matrix.cols() && (Rows == Dynamic || n == Rows) && (Cols == Dynamic || n == Cols),
          std::invalid_argument, "GMRES requires a nonempty square matrix matching its static shape");
        for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
                fdapde_strong_assert(
                  std::isfinite(static_cast<Scalar>(matrix.derived()(row, col))), std::invalid_argument,
                  "GMRES requires finite matrix coefficients");
            }
        }

        matrix_ = matrix;
        preconditioner_.compute(matrix_);
        fdapde_strong_assert(preconditioner_valid_(), std::domain_error, "GMRES requires a valid preconditioner");

        krylov_dimension_ = fdapde::min(restart_, n);
        H_.resize(krylov_dimension_ + 1, krylov_dimension_);
        V_.resize(n, krylov_dimension_ + 1);
        c_.resize(krylov_dimension_);
        s_.resize(krylov_dimension_);
        g_.resize(krylov_dimension_ + 1);
        initialized_ = true;
    }

    /// @brief returns an owning iterate from a finite initial guess and resets solve diagnostics
    template <typename RhsType, typename InitialType>
    constexpr auto solve(const MatrixExpr<RhsType>& rhs, const MatrixExpr<InitialType>& initial) {
        fdapde_static_assert(
          RhsType::Cols == 1 && InitialType::Cols == 1, GMRES_REQUIRES_COLUMN_VECTOR_RIGHT_HAND_SIDE_AND_INITIAL_GUESS);
        fdapde_static_assert(
          RhsType::Rows == Dynamic || Rows == Dynamic || RhsType::Rows == Rows, INVALID_GMRES_RHS_STATIC_SHAPE);
        fdapde_static_assert(
          InitialType::Rows == Dynamic || Rows == Dynamic || InitialType::Rows == Rows,
          INVALID_GMRES_INITIAL_GUESS_STATIC_SHAPE);

        reset_observers_();
        fdapde_strong_assert(initialized_, std::domain_error, "GMRES solve requires a successfully computed solver");
        fdapde_strong_assert(
          rhs.cols() == 1 && initial.cols() == 1 && rhs.rows() == matrix_.rows() && initial.rows() == matrix_.rows(),
          std::invalid_argument, "GMRES requires matching column-vector right-hand side and initial guess");
        for (int row = 0; row < matrix_.rows(); ++row) {
            fdapde_strong_assert(
              std::isfinite(static_cast<Scalar>(rhs.derived()(row, 0))) &&
                std::isfinite(static_cast<Scalar>(initial.derived()(row, 0))),
              std::invalid_argument, "GMRES requires finite right-hand side and initial-guess coefficients");
        }

        Vector<Scalar, Rows> solution(initial);
        const Vector<Scalar, Rows> preconditioned_rhs(preconditioner_.solve(rhs));
        const NormComponents rhs_norm = norm_components_(preconditioned_rhs);
        if (!rhs_norm.finite) return solution;

        while (last_iterations_ < max_iterations_) {
            Vector<Scalar, Rows> residual = preconditioned_residual_(rhs, solution);
            NormComponents residual_norm = norm_components_(residual);
            last_residual_ = materialized_norm_(residual_norm);
            if (!residual_norm.finite) return solution;
            if (within_tolerance_(residual_norm, rhs_norm)) {
                converged_ = true;
                return solution;
            }
            const Scalar residual_scale = residual_norm.scale;
            const Scalar beta = residual_norm.magnitude;

            // normalize the residual before Arnoldi so its norm need not fit in Scalar
            H_.set_zero();
            V_.set_zero();
            c_.set_zero();
            s_.set_zero();
            g_.set_zero();
            for (int row = 0; row < matrix_.rows(); ++row) V_(row, 0) = (residual[row] / residual_scale) / beta;
            g_[0] = beta;

            int used = 0;
            bool solution_updated = false;
            bool breakdown = false;
            for (int j = 0; j < krylov_dimension_ && last_iterations_ < max_iterations_; ++j) {
                Vector<Scalar, Rows> basis_vector;
                if constexpr (Rows == Dynamic) basis_vector.resize(matrix_.rows());
                for (int row = 0; row < matrix_.rows(); ++row) basis_vector[row] = V_(row, j);

                Vector<Scalar, Rows> product(matrix_ * basis_vector);
                Vector<Scalar, Rows> arnoldi(preconditioner_.solve(product));
                const Scalar arnoldi_scale = arnoldi.norm();
                // modified Gram-Schmidt builds the next Hessenberg column
                for (int i = 0; i <= j; ++i) {
                    Scalar projection = Scalar(0);
                    for (int row = 0; row < matrix_.rows(); ++row) projection += V_(row, i) * arnoldi[row];
                    H_(i, j) = projection;
                    for (int row = 0; row < matrix_.rows(); ++row) arnoldi[row] -= projection * V_(row, i);
                }
                H_(j + 1, j) = arnoldi.norm();
                const Scalar breakdown_threshold =
                  std::numeric_limits<Scalar>::epsilon() * static_cast<Scalar>(matrix_.rows()) * arnoldi_scale;
                const bool happy_breakdown = H_(j + 1, j) <= breakdown_threshold;
                if (!happy_breakdown) {
                    for (int row = 0; row < matrix_.rows(); ++row) V_(row, j + 1) = arnoldi[row] / H_(j + 1, j);
                }

                // apply prior Givens rotations before eliminating the new subdiagonal
                for (int i = 0; i < j; ++i) {
                    const Scalar upper = c_[i] * H_(i, j) + s_[i] * H_(i + 1, j);
                    H_(i + 1, j) = -s_[i] * H_(i, j) + c_[i] * H_(i + 1, j);
                    H_(i, j) = upper;
                }

                const Scalar diagonal = H_(j, j);
                const Scalar subdiagonal = H_(j + 1, j);
                const Scalar rotation_norm = std::hypot(diagonal, subdiagonal);
                if (!(rotation_norm > Scalar(0)) || !std::isfinite(rotation_norm)) {
                    breakdown = true;
                    break;
                }
                c_[j] = diagonal / rotation_norm;
                s_[j] = subdiagonal / rotation_norm;
                H_(j, j) = rotation_norm;
                H_(j + 1, j) = Scalar(0);
                const Scalar previous = g_[j];
                g_[j] = c_[j] * previous;
                g_[j + 1] = -s_[j] * previous;
                used = j + 1;
                ++last_iterations_;

                const Scalar estimated_magnitude = fdapde::abs(g_[j + 1]);
                const NormComponents estimated_norm {
                  estimated_magnitude == Scalar(0) ? Scalar(0) : residual_scale, estimated_magnitude, true};
                if (within_tolerance_(estimated_norm, rhs_norm) || happy_breakdown) {
                    solution_updated = update_solution_(solution, used, residual_scale);
                    if (!solution_updated) {
                        breakdown = true;
                        break;
                    }
                    // confirm estimated convergence with a freshly computed residual
                    residual = preconditioned_residual_(rhs, solution);
                    residual_norm = norm_components_(residual);
                    last_residual_ = materialized_norm_(residual_norm);
                    if (within_tolerance_(residual_norm, rhs_norm)) {
                        converged_ = true;
                        return solution;
                    }
                    breakdown = happy_breakdown;
                    break;
                }
            }

            if (!solution_updated && used > 0) {
                if (!update_solution_(solution, used, residual_scale)) breakdown = true;
            }
            if (breakdown) {
                const Vector<Scalar, Rows> residual = preconditioned_residual_(rhs, solution);
                last_residual_ = materialized_norm_(norm_components_(residual));
                return solution;
            }
        }

        const Vector<Scalar, Rows> residual = preconditioned_residual_(rhs, solution);
        const NormComponents residual_norm = norm_components_(residual);
        last_residual_ = materialized_norm_(residual_norm);
        converged_ = within_tolerance_(residual_norm, rhs_norm);
        return solution;
    }

    /// @brief solves from a zero initial guess and returns the last valid iterate
    template <typename RhsType> constexpr auto solve(const MatrixExpr<RhsType>& rhs) {
        Vector<Scalar, Rows> initial;
        if constexpr (Rows == Dynamic) initial.resize(rhs.rows());
        initial.set_zero();
        return solve(rhs, initial);
    }

    /// @brief reports whether the last solve satisfied the preconditioned residual tolerance
    constexpr bool converged() const { return converged_; }
    /// @brief returns the number of completed Arnoldi steps in the last solve
    constexpr int iterations() const { return last_iterations_; }
    /// @brief returns the absolute norm of P^-1 (b - A x), or infinity if unavailable or unrepresentable
    constexpr Scalar residual() const { return last_residual_; }
    /// @brief reports whether compute completed successfully
    constexpr bool initialized() const { return initialized_; }
   private:
    /// @brief stores a norm as a scale and normalized magnitude to avoid overflow
    struct NormComponents {
        Scalar scale;
        Scalar magnitude;
        bool finite;
    };

    /// @brief separates a finite vector norm into its maximum coefficient scale and normalized magnitude
    template <typename VectorType> static constexpr NormComponents norm_components_(const VectorType& vector) {
        Scalar scale = Scalar(0);
        for (int row = 0; row < vector.rows(); ++row) {
            const Scalar value = static_cast<Scalar>(vector[row]);
            if (!std::isfinite(value)) return {Scalar(0), Scalar(0), false};
            scale = fdapde::max(scale, fdapde::abs(value));
        }
        if (scale == Scalar(0)) return {Scalar(0), Scalar(0), true};
        Scalar squared_magnitude = Scalar(0);
        for (int row = 0; row < vector.rows(); ++row) {
            const Scalar normalized = static_cast<Scalar>(vector[row]) / scale;
            squared_magnitude += normalized * normalized;
        }
        return {scale, std::sqrt(squared_magnitude), true};
    }

    /// @brief materializes a scaled norm or returns infinity when it exceeds the scalar range
    static constexpr Scalar materialized_norm_(const NormComponents& norm) {
        if (!norm.finite) return std::numeric_limits<Scalar>::infinity();
        if (norm.scale == Scalar(0) || norm.magnitude == Scalar(0)) return Scalar(0);
        if (norm.scale > std::numeric_limits<Scalar>::max() / norm.magnitude)
            return std::numeric_limits<Scalar>::infinity();
        return norm.scale * norm.magnitude;
    }

    /// @brief compares scaled norms relatively, using absolute tolerance for a zero reference
    constexpr bool within_tolerance_(const NormComponents& residual, const NormComponents& reference) const {
        if (!residual.finite || !reference.finite) return false;
        if (residual.scale == Scalar(0) || residual.magnitude == Scalar(0)) return true;
        if (reference.scale == Scalar(0) || reference.magnitude == Scalar(0))
            return residual.scale <= tolerance_ / residual.magnitude;
        if (residual.scale <= reference.scale) {
            return residual.scale / reference.scale <= tolerance_ * (reference.magnitude / residual.magnitude);
        }
        return residual.magnitude / reference.magnitude <= tolerance_ * (reference.scale / residual.scale);
    }

    /// @brief consults the optional validity observer of a supplied preconditioner
    constexpr bool preconditioner_valid_() const {
        if constexpr (requires(const Preconditioner_& preconditioner) {
                          { preconditioner.valid() } -> std::convertible_to<bool>;
                      }) {
            return static_cast<bool>(preconditioner_.valid());
        }
        return true;
    }

    /// @brief recomputes the left-preconditioned residual using the owned operator
    template <typename RhsType>
    constexpr Vector<Scalar, Rows>
    preconditioned_residual_(const MatrixExpr<RhsType>& rhs, const Vector<Scalar, Rows>& solution) const {
        Vector<Scalar, Rows> raw(rhs - matrix_ * solution);
        return Vector<Scalar, Rows>(preconditioner_.solve(raw));
    }

    /// @brief applies a finite triangular-solve correction atomically to the iterate
    constexpr bool update_solution_(Vector<Scalar, Rows>& solution, int used, Scalar correction_scale) {
        Scalar scale = Scalar(0);
        for (int i = 0; i < used; ++i) {
            for (int j = i; j < used; ++j) scale = fdapde::max(scale, fdapde::abs(H_(i, j)));
        }
        if (!(scale > Scalar(0))) return false;
        const Scalar threshold = std::numeric_limits<Scalar>::epsilon() * static_cast<Scalar>(used) * scale;
        std::vector<Scalar> coefficients(static_cast<std::size_t>(used), Scalar(0));
        for (int i = used - 1; i >= 0; --i) {
            if (!(fdapde::abs(H_(i, i)) > threshold)) return false;
            Scalar value = g_[i];
            for (int j = i + 1; j < used; ++j) value -= H_(i, j) * coefficients[static_cast<std::size_t>(j)];
            coefficients[static_cast<std::size_t>(i)] = value / H_(i, i);
        }
        Vector<Scalar, Rows> updated(solution);
        for (int row = 0; row < matrix_.rows(); ++row) {
            Scalar normalized_correction = Scalar(0);
            for (int j = 0; j < used; ++j)
                normalized_correction += V_(row, j) * coefficients[static_cast<std::size_t>(j)];
            const Scalar correction = normalized_correction * correction_scale;
            if (!std::isfinite(correction)) return false;
            updated[row] += correction;
            if (!std::isfinite(updated[row])) return false;
        }
        solution = updated;
        return true;
    }

    /// @brief clears convergence and iteration data before compute or solve
    constexpr void reset_observers_() {
        converged_ = false;
        last_iterations_ = 0;
        last_residual_ = std::numeric_limits<Scalar>::infinity();
    }

    int max_iterations_;
    int restart_;
    Scalar tolerance_;
    Preconditioner_ preconditioner_;
    Matrix<Scalar, Rows, Cols> matrix_;
    Matrix<Scalar, Dynamic, Dynamic> H_;
    Matrix<Scalar, Dynamic, Dynamic, ColMajor> V_;
    Vector<Scalar, Dynamic> c_;
    Vector<Scalar, Dynamic> s_;
    Vector<Scalar, Dynamic> g_;
    int krylov_dimension_ = 0;
    bool initialized_ = false;
    bool converged_ = false;
    int last_iterations_ = 0;
    Scalar last_residual_ = std::numeric_limits<Scalar>::infinity();
};

template <typename XprType, typename Preconditioner>
GMRES(const MatrixExpr<XprType>&, Preconditioner&&, int, int, std::remove_cv_t<typename XprType::Scalar>)
  -> GMRES<XprType, std::decay_t<Preconditioner>>;
template <typename XprType, typename Preconditioner>
GMRES(const MatrixExpr<XprType>&, Preconditioner&&) -> GMRES<XprType, std::decay_t<Preconditioner>>;

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_GMRES_H__
