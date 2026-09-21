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

#ifndef __FDAPDE_LINALG_WOODBURY_H__
#define __FDAPDE_LINALG_WOODBURY_H__

#include <cmath>
#include <concepts>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

#include "header_check.h"

namespace fdapde {
namespace internals {

template <typename Solver>
concept woodbury_solver = requires(std::remove_cvref_t<Solver>& solver, const Vector<double, Dynamic>& rhs) {
    requires matrix_expression<decltype(solver.solve(rhs))>;
};

}   // namespace internals

/// @brief solves low-rank updates A + U C V by owning a base solver and caching A-inverse U
template <typename Solver_>
    requires(internals::woodbury_solver<std::remove_cvref_t<Solver_>>)
class Woodbury {
   public:
    using Scalar = double;
    using MatrixType = Matrix<Scalar, Dynamic, Dynamic>;
    using SolverType = std::remove_cvref_t<Solver_>;
    using DenseSolver = PartialPivLU<MatrixType>;

    /// @brief constructs an unavailable decomposition that must be assigned before solving
    Woodbury() = default;

    /// @brief owns the base solver and materializes U, C-inverse and V before factoring the correction
    template <
      internals::matrix_expression UType, internals::matrix_expression CInvType, internals::matrix_expression VType>
        requires(
          std::convertible_to<typename UType::Scalar, Scalar> &&
          std::convertible_to<typename CInvType::Scalar, Scalar> && std::convertible_to<typename VType::Scalar, Scalar>)
    Woodbury(
      SolverType solver, const MatrixExpr<UType>& u, const MatrixExpr<CInvType>& inverse_c,
      const MatrixExpr<VType>& v) {
        initialize_(std::move(solver), u.derived(), inverse_c.derived(), v.derived());
    }

    /// @brief copies the cached update and base solver according to the solver value semantics
    Woodbury(const Woodbury&) = default;
    /// @brief replaces the decomposition with a copy of the cached update and solver
    Woodbury& operator=(const Woodbury&) = default;
    /// @brief moves the cached decomposition and marks the source unavailable
    Woodbury(Woodbury&& other) : state_(std::move(other.state_)) { other.state_.reset(); }
    /// @brief transfers the cached decomposition and clears the source, tolerating self-move
    Woodbury& operator=(Woodbury&& other) {
        if (this != &other) {
            state_ = std::move(other.state_);
            other.state_.reset();
        }
        return *this;
    }

    /// @brief solves one or more right-hand sides into owning double storage with the input shape and layout
    template <internals::matrix_expression RhsType>
        requires(std::convertible_to<typename RhsType::Scalar, Scalar>)
    Matrix<Scalar, RhsType::Rows, RhsType::Cols, RhsType::StorageOrder> solve(const MatrixExpr<RhsType>& rhs) {
        fdapde_strong_assert(
          state_.has_value(), std::domain_error, "Woodbury solve requires an initialized decomposition");
        const RhsType& rhs_derived = rhs.derived();
        fdapde_strong_assert(
          !(rhs_derived.rows() != state_->dimension || rhs_derived.cols() <= 0), std::invalid_argument,
          "Woodbury requires a matching nonempty right-hand side");
        fdapde_strong_assert(
          all_finite_(rhs_derived), std::invalid_argument, "Woodbury requires finite right-hand-side coefficients");

        const MatrixType rhs_owned(rhs_derived);
        MatrixType y = base_solve_(state_->solver, rhs_owned, state_->dimension, rhs_owned.cols());
        const MatrixType projected(state_->v * y);
        fdapde_strong_assert(
          all_finite_(projected), std::domain_error, "Woodbury projection produced nonfinite values");
        const MatrixType t(state_->inverse_g.solve(projected));
        fdapde_strong_assert(
          !(t.rows() != state_->rank || t.cols() != rhs_owned.cols() || !all_finite_(t)), std::domain_error,
          "Woodbury correction solve produced an invalid result");
        const MatrixType correction(state_->inverse_a_u * t);
        const MatrixType result(y - correction);
        fdapde_strong_assert(all_finite_(result), std::domain_error, "Woodbury solve produced nonfinite values");
        return Matrix<Scalar, RhsType::Rows, RhsType::Cols, RhsType::StorageOrder>(result);
    }
   private:
    /// @brief stores the base solver, cached update and factorization of C-inverse plus V A-inverse U
    struct State {
        SolverType solver;
        MatrixType inverse_a_u;
        MatrixType v;
        DenseSolver inverse_g;
        int dimension;
        int rank;

        /// @brief takes ownership of a fully validated decomposition
        State(
          SolverType solver_, MatrixType inverse_a_u_, MatrixType v_, DenseSolver inverse_g_, int dimension_,
          int rank_) :
            solver(std::move(solver_)),
            inverse_a_u(std::move(inverse_a_u_)),
            v(std::move(v_)),
            inverse_g(std::move(inverse_g_)),
            dimension(dimension_),
            rank(rank_) { }
    };

    /// @brief checks that every coefficient is finite after conversion to the working scalar
    template <typename XprType> static bool all_finite_(const XprType& xpr) {
        for (int row = 0; row < xpr.rows(); ++row) {
            for (int col = 0; col < xpr.cols(); ++col) {
                if (!std::isfinite(static_cast<Scalar>(xpr(row, col)))) return false;
            }
        }
        return true;
    }

    /// @brief applies the base solver column by column and rejects invalid backend results
    static MatrixType base_solve_(SolverType& solver, const MatrixType& rhs, int expected_rows, int expected_cols) {
        MatrixType result(expected_rows, expected_cols);
        for (int col = 0; col < expected_cols; ++col) {
            Vector<Scalar, Dynamic> rhs_column(expected_rows);
            for (int row = 0; row < expected_rows; ++row) rhs_column[row] = rhs(row, col);
            auto solution = solver.solve(rhs_column);
            fdapde_strong_assert(
              !(solution.rows() != expected_rows || solution.cols() != 1 || !all_finite_(solution)), std::domain_error,
              "Woodbury base solver produced an invalid result");
            for (int row = 0; row < expected_rows; ++row) result(row, col) = static_cast<Scalar>(solution(row, 0));
        }
        return result;
    }

    /// @brief validates the update and publishes its cache after the correction factorization succeeds
    template <typename UType, typename CInvType, typename VType>
    void initialize_(SolverType solver, const UType& u, const CInvType& inverse_c, const VType& v) {
        const int dimension = u.rows();
        const int rank = u.cols();
        fdapde_strong_assert(
          !(dimension <= 0 || rank <= 0 || inverse_c.rows() != rank || inverse_c.cols() != rank || v.rows() != rank ||
            v.cols() != dimension),
          std::invalid_argument, "Woodbury requires U(n,q), inverse_c(q,q), and V(q,n) with n,q positive");
        fdapde_strong_assert(
          !(!all_finite_(u) || !all_finite_(inverse_c) || !all_finite_(v)), std::invalid_argument,
          "Woodbury requires finite update coefficients");

        const MatrixType u_owned(u);
        const MatrixType inverse_c_owned(inverse_c);
        MatrixType v_owned(v);
        MatrixType inverse_a_u = base_solve_(solver, u_owned, dimension, rank);
        const MatrixType g(inverse_c_owned + v_owned * inverse_a_u);
        fdapde_strong_assert(all_finite_(g), std::domain_error, "Woodbury correction matrix contains nonfinite values");
        DenseSolver inverse_g(g);
        fdapde_strong_assert(
          !(inverse_g.info() != 0), std::domain_error, "Woodbury requires a nonsingular correction matrix");

        state_.emplace(
          std::move(solver), std::move(inverse_a_u), std::move(v_owned), std::move(inverse_g), dimension, rank);
    }

    std::optional<State> state_;
};

template <typename Solver, typename UType, typename CInvType, typename VType>
Woodbury(Solver, const MatrixExpr<UType>&, const MatrixExpr<CInvType>&, const MatrixExpr<VType>&)
  -> Woodbury<std::remove_cvref_t<Solver>>;

}   // namespace fdapde

#endif   // __FDAPDE_LINALG_WOODBURY_H__
